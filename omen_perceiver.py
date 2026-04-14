"""
omen_perceiver.py — Ієрархічне кодування: Token → Concept
==========================================================
Компоненти:

  RMSNorm            : Стабілізація без зміщення  (LLaMA-стиль)
  RotaryEmbedding    : RoPE позиційне кодування  (без обмежень на довжину)
  SwiGLUFFN          : Ефективна FFN з активацією SwiGLU
  LlamaAttention     : MHA + RoPE + опціональний FlashAttention
  LlamaDecoderBlock  : Блок трансформера (норма → attn → норма → FFN)
  PerceiverResampler : Token (B,T,d_tok) → Concept (B,n,d_lat)
  l_scale_penalty    : MDL-регуляризатор Σ||z||² для рівнів

Математика:
  L_scale = λ_tok·(1/T)·Σ_t ||z_t||² + λ_conc·(1/|C|)·Σ_c ||c||²

  Це не дає моделі "розмазати" інформацію по всьому вектору —
  вона змушена кластеризувати знання в концепти.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ПРИМІТИВИ (LLaMA-стиль)
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm — без зміщення, без centering."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Не потребує фіксованого seq_len — обчислення "on-the-fly".
    """
    def __init__(self, d_head: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self.d_head = d_head
        # OPT-ROPE: кеш cos/sin per (T, device) — ініціалізуємо в __init__
        self._cs_cache: dict = {}

    def _get_cos_sin(self, T: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        # OPT-ROPE: cos/sin залежать тільки від T (inv_freq — константа).
        # Кешуємо на першому виклику для кожного T; всі наступні — dict lookup.
        key = (T, str(device))
        cached = self._cs_cache.get(key)
        if cached is not None:
            return cached
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)           # (T, d/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (T, d)
        result = (emb.cos()[None, None], emb.sin()[None, None])  # (1,1,T,d)
        self._cs_cache[key] = result
        return result

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        cos, sin = self._get_cos_sin(T, q.device)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN: FFN(x) = (xW₁ ⊙ SiLU(xV)) · W₂
    Ефективніше за GELU при рівній кількості параметрів.
    """
    def __init__(self, d: int, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or int(d * 8 / 3)   # LLaMA-стиль: 8/3 замість 4
        self.gate  = nn.Linear(d, d_ff, bias=False)
        self.up    = nn.Linear(d, d_ff, bias=False)
        self.down  = nn.Linear(d_ff, d, bias=False)
        # OPT-4: inplace=True — dropout не виділяє новий тензор, пише поверх
        self.drop  = nn.Dropout(dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LlamaAttention
# ══════════════════════════════════════════════════════════════════════════════

class LlamaAttention(nn.Module):
    """
    Multi-Head Attention з RoPE.
    Використовує F.scaled_dot_product_attention (FlashAttention-сумісний).
    """
    def __init__(self, d: int, n_heads: int,
                 dropout: float = 0.0,
                 causal: bool = True,
                 cross_attn: bool = False,
                 kv_dim: Optional[int] = None):
        super().__init__()
        assert d % n_heads == 0, f"d={d} не ділиться на n_heads={n_heads}"
        self.h    = n_heads
        self.dh   = d // n_heads
        self.causal = causal
        self.cross_attn = cross_attn

        # Для cross-attention K/V беруться зі стороннього контексту
        kv_d = kv_dim if kv_dim else d
        self.q_proj  = nn.Linear(d,   d,   bias=False)
        self.k_proj  = nn.Linear(kv_d, d,  bias=False)
        self.v_proj  = nn.Linear(kv_d, d,  bias=False)
        self.o_proj  = nn.Linear(d,   d,   bias=False)
        self.drop    = dropout
        self.rope    = RotaryEmbedding(self.dh) if not cross_attn else None

    def forward(self,
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)

        ctx = context if (self.cross_attn and context is not None) else x
        S   = ctx.shape[1]
        k = self.k_proj(ctx).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.v_proj(ctx).view(B, S, self.h, self.dh).transpose(1, 2)

        # RoPE тільки для self-attention
        if self.rope is not None:
            q, k = self.rope(q, k)

        if need_weights:
            scale = self.dh ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, T, S)

            if self.causal and not self.cross_attn:
                causal = torch.triu(
                    torch.ones(T, S, dtype=torch.bool, device=x.device),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

            if attn_mask is not None:
                mask = attn_mask
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, float("-inf"))
                else:
                    scores = scores + mask

            weights = scores.softmax(dim=-1)
            attn = F.dropout(weights, p=self.drop, training=self.training)
            out = torch.matmul(attn, v)
            out = self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))
            return out, weights

        # F.scaled_dot_product_attention: FlashAttention якщо доступна
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.drop if self.training else 0.0,
            is_causal=(self.causal and not self.cross_attn and attn_mask is None),
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LlamaDecoderBlock  (token-level, Fine)
# ══════════════════════════════════════════════════════════════════════════════

class LlamaDecoderBlock(nn.Module):
    """
    Стандартний блок LLaMA:
      x ← x + Attention(RMSNorm(x))
      x ← x + SwiGLUFFN(RMSNorm(x))
    Використовується на токен-рівні (d_tok, n_heads_tok).
    """
    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn  = LlamaAttention(d, n_heads, dropout=dropout, causal=True)
        self.norm2 = RMSNorm(d)
        self.ffn   = SwiGLUFFN(d, dropout=dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        if need_weights:
            attn_out, attn_weights = self.attn(self.norm1(x), need_weights=True)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PerceiverResampler  (Token → Concept, Fine → Coarse)
# ══════════════════════════════════════════════════════════════════════════════

class PerceiverResampler(nn.Module):
    """
    Стискає довільну послідовність токенів у фіксовану кількість латентів.

    Архітектура:
      1. Проекція: d_tok → d_latent (якщо різні)
      2. Cross-attention: latent queries ← tokens  (T→n_latents)
      3. Self-attention серед латентів
      4. Повтор 2–3 по n_layers_lat разів
      5. Pool: mean(latents) → (B, d_latent)

    Вхід:  (B, T, d_tok)
    Вихід: latents (B, n_latents, d_latent),  pooled (B, d_latent)
    """

    def __init__(self,
                 d_tok: int,
                 d_latent: int,
                 n_latents: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float = 0.1):
        super().__init__()

        # 0. Проекція токен-рівня → концепт-рівень
        self.tok_proj = (nn.Linear(d_tok, d_latent, bias=False)
                         if d_tok != d_latent else nn.Identity())

        # 1. Навчальні латентні запити (фіксований розмір)
        self.latents = nn.Parameter(
            torch.randn(1, n_latents, d_latent) * d_latent ** -0.5)

        # 2+3. Пари (cross-attn, self-attn) × n_layers
        self.cross_norms = nn.ModuleList([RMSNorm(d_latent) for _ in range(n_layers)])
        self.cross_attns = nn.ModuleList([
            LlamaAttention(d_latent, n_heads, dropout=dropout,
                           causal=False, cross_attn=True, kv_dim=d_latent)
            for _ in range(n_layers)
        ])
        self.self_norms  = nn.ModuleList([RMSNorm(d_latent) for _ in range(n_layers)])
        self.self_attns  = nn.ModuleList([
            LlamaAttention(d_latent, n_heads, dropout=dropout, causal=False)
            for _ in range(n_layers)
        ])
        self.ffn_norms   = nn.ModuleList([RMSNorm(d_latent) for _ in range(n_layers)])
        self.ffns        = nn.ModuleList([
            SwiGLUFFN(d_latent, dropout=dropout) for _ in range(n_layers)
        ])

        self.out_norm = RMSNorm(d_latent)
        self.n_layers = n_layers
        self.d_latent = d_latent

    def forward(self, tokens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens : (B, T, d_tok)
        Returns: latents (B, n_latents, d_latent),  pooled (B, d_latent)
        """
        B = tokens.shape[0]
        ctx = self.tok_proj(tokens)                          # (B, T, d_lat)
        lat = self.latents.expand(B, -1, -1)                # (B, n, d_lat)

        for i in range(self.n_layers):
            # Cross-attention: latents ← context
            lat = lat + self.cross_attns[i](
                self.cross_norms[i](lat), context=ctx)
            # Self-attention серед латентів
            lat = lat + self.self_attns[i](self.self_norms[i](lat))
            # FFN
            lat = lat + self.ffns[i](self.ffn_norms[i](lat))

        lat = self.out_norm(lat)
        pooled = lat.mean(1)                                 # (B, d_latent)
        return lat, pooled


# ══════════════════════════════════════════════════════════════════════════════
# 5.  L_scale PENALTY  (MDL-регуляризатор рівнів)
# ══════════════════════════════════════════════════════════════════════════════

def l_scale_penalty(z_tok: torch.Tensor,
                    z_conc: torch.Tensor,
                    lambda_tok: float,
                    lambda_conc: float) -> torch.Tensor:
    """
    L_scale = λ_tok·(1/T)·Σ_t ||z_t||² + λ_conc·(1/|C|)·Σ_c ||c||²

    Примушує модель кластеризувати знання в концепти, а не
    "розмазувати" інформацію по всьому гігантському вектору.

    z_tok  : (B, T, d_tok)   — токен-рівень
    z_conc : (B, n, d_lat)   — концепт-рівень (latents або pooled)
    """
    tok_penalty  = lambda_tok  * z_tok.pow(2).mean()
    conc_penalty = lambda_conc * z_conc.pow(2).mean()
    return tok_penalty + conc_penalty


# ══════════════════════════════════════════════════════════════════════════════
# 6.  INLINE ТЕСТИ
# ══════════════════════════════════════════════════════════════════════════════

def _run_perceiver_tests() -> None:
    sep = lambda s: print(f"\n{'─'*60}\n  {s}\n{'─'*60}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[omen_perceiver] device={device}")

    # T1: RMSNorm
    sep("T1 · RMSNorm")
    norm = RMSNorm(64).to(device)
    x = torch.randn(4, 16, 64, device=device)
    y = norm(x)
    assert y.shape == x.shape
    rms = y.pow(2).mean(-1).sqrt()
    assert rms.mean().item() < 1.5, f"RMS={rms.mean():.3f} занадто велике"
    print(f"  shape {tuple(y.shape)}  rms≈{rms.mean():.3f}  [PASS]")

    # T2: RotaryEmbedding
    sep("T2 · RotaryEmbedding")
    rope = RotaryEmbedding(32).to(device)
    q = torch.randn(2, 4, 8, 32, device=device)  # (B,h,T,dh)
    k = torch.randn(2, 4, 8, 32, device=device)
    q_r, k_r = rope(q, k)
    assert q_r.shape == q.shape
    # Норми мають зберігатися (обертання ортогональне)
    assert torch.allclose(q_r.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    print(f"  norm preserved: max_err={( q_r.norm(dim=-1) - q.norm(dim=-1)).abs().max():.2e}  [PASS]")

    # T3: SwiGLUFFN
    sep("T3 · SwiGLUFFN")
    ffn = SwiGLUFFN(64, dropout=0.0).to(device)
    x = torch.randn(2, 16, 64, device=device)
    y = ffn(x)
    assert y.shape == x.shape
    print(f"  shape {tuple(y.shape)}  mean={y.mean():.4f}  [PASS]")

    # T4: LlamaDecoderBlock
    sep("T4 · LlamaDecoderBlock")
    blk = LlamaDecoderBlock(64, 4, dropout=0.0).to(device)
    x = torch.randn(2, 32, 64, device=device)
    y = blk(x)
    assert y.shape == x.shape
    print(f"  shape {tuple(y.shape)}  [PASS]")

    # T5: PerceiverResampler — форма
    sep("T5 · PerceiverResampler — shape")
    pr = PerceiverResampler(
        d_tok=128, d_latent=64, n_latents=8, n_heads=4, n_layers=2
    ).to(device)
    tokens = torch.randn(3, 64, 128, device=device)
    latents, pooled = pr(tokens)
    assert latents.shape == (3, 8, 64), f"latents {latents.shape}"
    assert pooled.shape  == (3, 64),    f"pooled {pooled.shape}"
    print(f"  latents {tuple(latents.shape)}  pooled {tuple(pooled.shape)}  [PASS]")

    # T6: PerceiverResampler — стиснення T→n_latents
    sep("T6 · PerceiverResampler — compression ratio")
    T, n = 256, 8
    pr2 = PerceiverResampler(
        d_tok=64, d_latent=32, n_latents=n, n_heads=4, n_layers=1
    ).to(device)
    tok_in = torch.randn(2, T, 64, device=device)
    lat_out, _ = pr2(tok_in)
    ratio = T / n
    print(f"  T={T}→n={n}  ratio={ratio:.0f}×  lat_shape={tuple(lat_out.shape)}  [PASS]")

    # T7: l_scale_penalty
    sep("T7 · l_scale_penalty")
    z_t = torch.randn(2, 16, 128, device=device)
    z_c = torch.randn(2, 8, 32,  device=device)
    pen = l_scale_penalty(z_t, z_c, 1e-4, 1e-3)
    assert pen.item() > 0
    print(f"  L_scale={pen.item():.6f}  [PASS]")

    # T8: Backward через PerceiverResampler
    sep("T8 · Backward через PerceiverResampler")
    pr3 = PerceiverResampler(
        d_tok=64, d_latent=32, n_latents=8, n_heads=4, n_layers=1
    ).to(device)
    tok = torch.randn(2, 32, 64, device=device)
    _, pooled = pr3(tok)
    loss = pooled.pow(2).mean()
    loss.backward()
    grad_sum = sum(p.grad.norm().item() for p in pr3.parameters() if p.grad is not None)
    assert grad_sum > 0, "FAIL: немає градієнту"
    print(f"  grad_sum={grad_sum:.4f}  [PASS]")

    print("\n  ✅  omen_perceiver: всі тести пройдено\n")


if __name__ == "__main__":
    _run_perceiver_tests()

"""
omen_scale.py — OMEN-Scale: Повна архітектура наступного покоління
====================================================================
Три рівні + Async M-Core + ∂-Prolog:

  Level 1 (Token/Fine):   LlamaDecoderBlock stack,  d_tok, V≥50k
  Level 2 (Concept/Coarse): PerceiverResampler,     d_latent
  Level 3 (Symbolic):     DifferentiableProver (∂-Prolog), Γ

Фундаментальна формула:

  OMEN-Scale ≡ min_{θ,Γ} { Complexity(θ) + Complexity(Γ)
                           + E_World[Surprise(Data | θ, Γ)] }

  J(θ,Γ,M) = Perplexity(θ)
            + β·E_{(c,g)~Tasks}[L_proof(π,Γ,c,g)]   ← символьне узагальнення
            + γ·E_{z~Q_θ}[||z - Read(M,z) - Sim(z)||²] ← консистентність світу
            - α·I(Z;Mem)                              ← стиснення в пам'яті
            + L_scale                                 ← MDL рівнів
            + λ_rule·Complexity(Γ)                    ← MDL правил

Зовнішні залежності: omen_v2, omen_perceiver, omen_prolog, omen_scale_config
"""

from __future__ import annotations
import math, time, random, warnings
from collections import defaultdict, deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Локальні модулі ──────────────────────────────────────────────────────────
from omen_scale_config import OMENScaleConfig
from omen_perceiver    import (PerceiverResampler, LlamaDecoderBlock,
                                RMSNorm, SwiGLUFFN, LlamaAttention,
                                l_scale_penalty)
from omen_prolog       import DifferentiableProver, HornAtom

# NET: Neural Epistemic Tokenizer (замінює GPT-2 BPE)
from omen_net_tokenizer import NeuralEpistemicTokenizer

# EMC: Efficient Meta-Controller (адаптивний контролер міркування)
from omen_emc import EfficientMetaController

# OSF: OMEN Synthesis Framework (ієрархічна нейро-символьна генерація)
from omen_osf import OSFSynthesizer, OSFConfig

# Запозичуємо стабільні компоненти v2
from omen_v2 import (
    WorldRNN,
    EpistemicGapDetector,
    CuriosityModule,
    OMENv2Config,
    make_counting, make_python, make_rule_transfer, collate,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ASYNC TENSOR PRODUCT MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class AsyncTensorProductMemory(nn.Module):
    """
    Голографічна пам'ять M ∈ R^{H × d × d} з асинхронними записами.

    Замість щоразового оновлення M після кожного кроку:
      · Записи буферизуються у _buf_*
      · Кожні update_steps кроків викликається flush() —
        одне батчеве оновлення замість N окремих

    Це прибирає torch.einsum з критичного шляху кожного батчу.

    Запис : M_h ← M_h + λ·(k ⊗ v)   [@ flush, без backprop]
    Читання: v = Σ_h M_h · k          [O(H·d²), диф-бельне]
    """

    def __init__(self, cfg: OMENScaleConfig):
        super().__init__()
        d, H = cfg.d_latent, cfg.mem_heads
        self.register_buffer("memory", torch.zeros(H, d, d))
        self.key_proj = nn.Linear(d, d * H, bias=False)
        self.val_proj = nn.Linear(d, d * H, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.d, self.H = d, H
        self.write_tau    = cfg.mem_write_tau
        self.update_steps = cfg.mem_update_steps
        self.cache:  deque = deque(maxlen=cfg.mem_cache_size)
        self.n_writes = 0
        self._step    = 0

        # Асинхронний буфер (CPU-тензори для економії VRAM)
        self._buf_s: List[torch.Tensor] = []
        self._buf_v: List[torch.Tensor] = []
        self._buf_c: List[torch.Tensor] = []

        # Прапорець: чи є відкладений flush після останнього backward
        self._pending_flush: bool = False

    # ── Читання (диференційоване, щоразу) ─────────────────────────────────────
    def read(self, z_query: torch.Tensor) -> torch.Tensor:
        # FIX: self.memory оновлюється в flush() через `self.memory += delta`
        # (in-place операція), що інкрементує version-counter НАВІТЬ під @no_grad.
        # Autograd зберігає посилання на self.memory під час forward і очікує
        # version=N під час backward, але знаходить version=N+1 → RuntimeError.
        #
        # Рішення: .detach() ізолює self.memory від autograd-графу.
        # Градієнти все одно течуть через key_proj → z (диференційовані),
        # а self.memory — register_buffer з requires_grad=False, тому
        # PyTorch ніколи і не намагався обчислити d/d(memory).
        k = self.key_proj(z_query).view(-1, self.H, self.d)
        v = torch.einsum('bhd,hde->bhe', k, self.memory.detach())
        return self.out_proj(v.mean(1))                            # (B, d)

    # ── Буферизація запису ─────────────────────────────────────────────────────
    def schedule_write(self,
                       z_state:    torch.Tensor,
                       z_value:    torch.Tensor,
                       confidence: torch.Tensor) -> None:
        """
        Записуємо аргументи у буфер; M НЕ чіпаємо під час forward.
        Flush відкладається до AFTER backward() — інакше inplace
        модифікація memory ламає autograd (version mismatch).
        """
        self._buf_s.append(z_state.detach().cpu())
        self._buf_v.append(z_value.detach().cpu())
        self._buf_c.append(confidence.detach().cpu())
        self._step += 1
        # НЕ викликаємо flush() тут — він буде викликаний ззовні
        # після optimizer.step() (де немає autograd graph)
        self._pending_flush = (self._step % self.update_steps == 0)

    # ── Батчеве оновлення пам'яті ─────────────────────────────────────────────
    @torch.no_grad()
    def flush(self) -> None:
        """
        Застосовує всі буферизовані записи одним вектором.
        Викликати ТІЛЬКИ після optimizer.step() (коли autograd graph знищено).

        Використовуємо `.copy_()` щоб уникнути += (який створює новий тензор
        і може викликати проблеми з версіонуванням у деяких випадках).
        """
        if not self._buf_s:
            return
        dev = self.memory.device
        # FIX-AMP: schedule_write() зберігає тензори у dtype forward-пасу.
        # Під torch.autocast(fp16) вони стають FP16, але key_proj / val_proj
        # є nn.Linear поза autocast → weights FP32 → «Half != float» RuntimeError.
        # Рішення: кастуємо до dtype вагів (FP32 або BF16 при BF16 training).
        # `.to(device, dtype=)` — один kernel-call замість двох окремих викликів.
        w_dtype = self.key_proj.weight.dtype
        z_s = torch.cat(self._buf_s, 0).to(dev, dtype=w_dtype)
        z_v = torch.cat(self._buf_v, 0).to(dev, dtype=w_dtype)
        lam = torch.cat(self._buf_c, 0).to(dev, dtype=w_dtype)
        lam = (1.0 - lam).clamp(0, 1)
        mask = lam > self.write_tau
        if mask.any():
            z_s_m  = z_s[mask]; z_v_m = z_v[mask]; lam_m = lam[mask]
            k = self.key_proj(z_s_m).view(-1, self.H, self.d)
            v = self.val_proj(z_v_m).view(-1, self.H, self.d)
            delta = torch.einsum('bhd,bhe,b->hde', k, v, lam_m)
            new_mem = self.memory + delta / (mask.sum().float() + 1e-6)
            self.memory.copy_(new_mem)
            for i in range(z_s_m.size(0)):
                self.cache.append((z_s_m[i], z_v_m[i]))
            self.n_writes += mask.sum().item()
        self._buf_s.clear(); self._buf_v.clear(); self._buf_c.clear()
        self._pending_flush = False

    def maybe_flush(self) -> None:
        """
        Безпечний flush — викликається ПІСЛЯ optimizer.step().
        Перевіряє _pending_flush і тільки тоді оновлює memory.
        """
        if self._pending_flush:
            self.flush()

    # ── Episodic recall (k-NN) ─────────────────────────────────────────────────
    @torch.no_grad()
    def episodic_recall(self, z_query: torch.Tensor, k: int = 4) -> torch.Tensor:
        if len(self.cache) == 0:
            return torch.zeros_like(z_query)
        # FIX-AMP: кеш зберігається через flush() (вже FP32 після виправлення),
        # але z_query під autocast може бути FP16.  Кастуємо кеш до dtype запиту.
        cache_keys = torch.stack([c[0] for c in self.cache], 0).to(
            z_query.device, dtype=z_query.dtype)
        cache_vals = torch.stack([c[1] for c in self.cache], 0).to(
            z_query.device, dtype=z_query.dtype)
        sims = F.cosine_similarity(
            z_query.unsqueeze(1), cache_keys.unsqueeze(0), dim=-1)
        topk = sims.topk(min(k, len(self.cache)), dim=1).indices
        return cache_vals[topk].mean(1)

    def memory_footprint_bytes(self) -> int:
        return self.memory.numel() * self.memory.element_size()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TOKEN-LEVEL ENCODER  (Fine, LlamaDecoderBlock stack)
# ══════════════════════════════════════════════════════════════════════════════

class TokenEncoder(nn.Module):
    """
    Рівень 1 (Fine): LlamaDecoderBlock stack.
    Не приймає рішень — лише ПРЕДСТАВЛЯЄ дані.
    Рішення приймаються на концепт-рівні (d_latent).
    """

    def __init__(self, cfg: OMENScaleConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_tok)
        # Scaled init: ембеддинги масштабуємо на 1/√d_tok
        nn.init.normal_(self.embed.weight, std=cfg.d_tok ** -0.5)
        self.blocks = nn.ModuleList([
            LlamaDecoderBlock(cfg.d_tok, cfg.n_heads_tok, cfg.dropout)
            for _ in range(cfg.n_layers_tok)
        ])
        self.norm = RMSNorm(cfg.d_tok)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) → hidden: (B, T, d_tok)"""
        x = self.drop(self.embed(tokens))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                                        # (B, T, d_tok)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TOKEN-LEVEL DECODER  (з cross-attention до z_final)
# ══════════════════════════════════════════════════════════════════════════════

class TokenDecoder(nn.Module):
    """
    Рівень 1 Decoder: LlamaDecoderBlock stack + cross-attention до концепт-z.
    """

    def __init__(self, cfg: OMENScaleConfig):
        super().__init__()
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_tok)
        nn.init.normal_(self.embed.weight, std=cfg.d_tok ** -0.5)
        self.z_proj = nn.Linear(cfg.d_latent, cfg.d_tok, bias=False)

        self.blocks  = nn.ModuleList([
            LlamaDecoderBlock(cfg.d_tok, cfg.n_heads_tok, cfg.dropout)
            for _ in range(cfg.n_layers_tok)
        ])
        # Cross-attention: tokens ← концепт-z
        self.cross_norm = RMSNorm(cfg.d_tok)
        self.cross_attn = LlamaAttention(
            cfg.d_tok, cfg.n_heads_tok, dropout=cfg.dropout,
            causal=False, cross_attn=True, kv_dim=cfg.d_tok)

        self.out_norm = RMSNorm(cfg.d_tok)
        self.lm_head  = nn.Linear(cfg.d_tok, cfg.vocab_size, bias=False)
        self.drop     = nn.Dropout(cfg.dropout)

    def forward(self, tokens: torch.Tensor,
                z_final: torch.Tensor) -> torch.Tensor:
        """
        tokens  : (B, T)
        z_final : (B, d_latent) — концепт-рівень
        Returns: logits (B, T, vocab_size)
        """
        x  = self.drop(self.embed(tokens))
        z_ctx = self.z_proj(z_final).unsqueeze(1)              # (B, 1, d_tok)

        # Inject концепт у декодер через cross-attention
        x = x + self.cross_attn(self.cross_norm(x), context=z_ctx)

        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.out_norm(x))                  # (B, T, V)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  OMEN-SCALE LOSS:  J(θ, Γ, M)
# ══════════════════════════════════════════════════════════════════════════════

class OMENScaleLoss(nn.Module):
    """
    Повний Епістемічний Функціонал Якості (оновлена версія):

      J(θ,Γ,M) = Perplexity(θ)                               ← ймовірнісна модель
              + β·L_proof(π,Γ)                               ← символьне узагальнення
              + γ·||z - Read(M,z) - Sim(z)||²                ← консистентність світу
              - α·I(Z;Γ)                                     ← взаємна інф. (семантичний feedback)
              + λ_tok·||z_tok||² + λ_conc·||z_con||²         ← L_scale MDL рівнів
              + λ_rule·Σ_{R∈Γ}(Complexity(R) − η·Utility(R)) ← MDL правил з корисністю
              + η·L_recall                                   ← пам'ять точність
              + δ·E_{R~Abduction}[max(0,τ−U(R))]            ← VeM штраф

    Порівняно з v1:
      · λ_rule·Complexity(Γ) → λ_rule·Σ(Complexity−η·Utility): корисні правила не штрафуються
      · Додано -α·I(Z;Γ) через L_semantic (NET semantic feedback)
      · Додано δ·VeM_penalty: скеровує AbductionHead до корисних правил
    """

    def __init__(self, cfg: OMENScaleConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self,
                logits:       torch.Tensor,
                targets:      torch.Tensor,
                z:            torch.Tensor,        # (B, d_latent)
                z_tok:        torch.Tensor,        # (B, T, d_tok)
                z_latents:    torch.Tensor,        # (B, n, d_latent)
                z_sim:        torch.Tensor,        # (B, d_latent)
                v_mem:        torch.Tensor,        # (B, d_latent)
                sym_loss:     torch.Tensor,
                ltm_penalty:  float,
                curiosity_l:  torch.Tensor,
                world_rnn:    WorldRNN,
                net_loss:     torch.Tensor,        # L_NET від NeuralEpistemicTokenizer
                vem_penalty:  Optional[torch.Tensor] = None,  # δ·E[max(0,τ−U(R))]
                meta_loss:    Optional[torch.Tensor] = None,  # ω_meta·L_AC (EMC)
                traj_reward:  Optional[float]       = None,   # Σ_t r_t траєкторії
                ) -> Dict:
        cfg = self.cfg

        # ── 1. Перплексія (next-token prediction, зсунуте) ───────────────────
        # FIX: logits[t] = P(next | tgt[0..t]) — прогнозує НАСТУПНИЙ токен.
        # Попередня версія CE(logits, targets) порівнювала logits[t] з tgt[t]
        # (поточним), що давало артефактно низький PPL (~1.4) бо токен tgt[t]
        # вже присутній у контексті. Правильно: logits[:-1] → targets[1:].
        L_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            targets[:, 1:].reshape(-1),
            ignore_index=0,
        )

        # ── 2. WorldRNN Training: huber(z_sim, z.detach()) ────────────────────
        # FIX (критичне): попередній варіант
        #   z_target = (z_sim + v_mem).detach()
        #   L_world  = huber(z, z_target)   ← grad → z, НЕ WorldRNN
        # WorldRNN.parameters() ніколи не отримували градієнту → random init назавжди.
        # Безглузді z_sim → z тягнувся до сміттєвих цілей → CE 0.33→3.1 (деградація).
        #
        # РІШЕННЯ: перевертаємо напрямок.
        #   L_world = huber(z_sim, z.detach())
        # Градієнт тепер іде: L_world → z_sim → WorldRNN.parameters()
        # WorldRNN навчається: "якщо концепт z, то симулюй z_sim ≈ z".
        # z більше НЕ тягнеться до помилкових цілей WorldRNN.
        L_world_raw  = F.huber_loss(z_sim, z.detach(), delta=1.0)
        L_world      = torch.log1p(L_world_raw)

        # ── 3. WorldRNN Complexity (скінченні різниці) ────────────────────────
        with torch.no_grad():
            eps  = 1e-2
            dz   = torch.randn_like(z) * eps
            dummy = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            zn1, _ = world_rnn(z.detach(), dummy)
            zn2, _ = world_rnn((z + dz).detach(), dummy)
            L_complex = ((zn1 - zn2) / eps).pow(2).mean().clamp(max=5.0)

        # ── 4. Memory Recall ──────────────────────────────────────────────────
        v_norm    = v_mem.detach().norm(dim=-1, keepdim=True).clamp(min=1e-4)
        L_recall  = F.mse_loss(z, (v_mem / v_norm).detach())

        # ── 5. Novelty Bonus: -α·I(Z;M) ──────────────────────────────────────
        I_zm = v_mem.norm(dim=-1).mean()

        # ── 6. L_scale: MDL для рівнів ────────────────────────────────────────
        L_scale = l_scale_penalty(z_tok, z_latents, cfg.lambda_tok, cfg.lambda_conc)

        # ── 7. Symbolic Generalization ────────────────────────────────────────
        L_sym   = sym_loss

        # ── 8. VeM Penalty: δ·E[max(0, τ − U(R))] ───────────────────────────
        if vem_penalty is not None and torch.is_tensor(vem_penalty):
            L_vem = vem_penalty.clamp(max=5.0)
        else:
            L_vem = torch.zeros(1, device=z.device).squeeze()

        # ── 9. EMC Meta-Loss: ω_meta·L_AC ────────────────────────────────────
        if meta_loss is not None and torch.is_tensor(meta_loss):
            L_meta = meta_loss.clamp(-5.0, 5.0)   # обрізаємо вибухи Actor-Critic
        else:
            L_meta = torch.zeros(1, device=z.device).squeeze()
        omega_meta = getattr(cfg, 'omega_meta', 0.05)

        # ── 10. J_OMEN+EMC: 7-ма компонента — траєкторна мета-винагорода ─────
        if traj_reward is not None and traj_reward != 0.0:
            L_traj = -torch.tensor(traj_reward, dtype=torch.float32, device=z.device).clamp(-5.0, 5.0)
        else:
            L_traj = torch.zeros(1, device=z.device).squeeze()

        # ── ltm_penalty: клемпуємо щоб уникнути вибуху зі зростанням LTM ─────
        # Без кліпу: при 1024 правилах × complexity≈5 → ltm_pen≈25 → домінує loss.
        ltm_penalty_clamped = min(float(ltm_penalty), 2.0)

        # ── Sym loss: обрізаємо нестабільний symbolic consistency ──────────────
        L_sym_clamped = L_sym.clamp(max=5.0) if torch.is_tensor(L_sym) else torch.tensor(
            min(float(L_sym), 5.0), device=z.device)

        # ── Збираємо J(θ,Γ,M) + ω_meta·L_AC + ω_meta·L_traj ─────────────────
        # Curiosity: обрізаємо щоб уникнути домінування при великих gap_norm
        curiosity_clamped = curiosity_l.clamp(max=5.0) if torch.is_tensor(curiosity_l) \
                            else min(float(curiosity_l), 5.0)
        total = (
            L_ce
          + cfg.gamma  * L_world
          + cfg.delta  * L_complex
          + cfg.eta    * L_recall
          - cfg.alpha  * I_zm
          + L_scale
          + cfg.beta   * L_sym_clamped
          + ltm_penalty_clamped
          + 0.1 * curiosity_clamped
          + cfg.eta_tok * net_loss          # ← L_NET (включає L_semantic)
          + getattr(cfg, 'delta_vem', 1e-3) * L_vem  # ← δ·VeM
          + omega_meta  * L_meta            # ← ω_meta·L_AC (EMC Actor-Critic)
          + omega_meta  * L_traj            # ← ω_meta·(-Σ_t r_t) (7-ма компонента)
        )

        # Фінальна NaN-guard: якщо total=NaN → повертаємо тільки L_ce
        if torch.isnan(total) or torch.isinf(total):
            total = L_ce

        # net_loss може бути dict або scalar — нормалізуємо
        net_loss_scalar = (net_loss["net_total"].item()
                           if isinstance(net_loss, dict)
                           else (net_loss.item() if torch.is_tensor(net_loss)
                                 else float(net_loss)))

        return {
            "total":      total,
            "ce":         L_ce.item(),
            "world":      L_world.item(),
            "complex":    L_complex.item(),
            "recall":     L_recall.item(),
            "novelty":    I_zm.item(),
            "l_scale":    L_scale.item(),
            "sym_ground": L_sym_clamped.item() if torch.is_tensor(L_sym_clamped) else float(L_sym_clamped),
            "ltm_pen":    ltm_penalty_clamped,
            "ltm_pen_raw": float(ltm_penalty),   # для діагностики нескліпованого значення
            "curiosity":  curiosity_l.item(),
            "net_loss":   net_loss_scalar,
            "vem_pen":    L_vem.item() if torch.is_tensor(L_vem) else float(L_vem),
            "meta_loss":  L_meta.item() if torch.is_tensor(L_meta) else float(L_meta),
            "traj_reward": traj_reward if traj_reward is not None else 0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  OMEN-SCALE — ПОВНА МОДЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════

class OMENScale(nn.Module):
    """
    Повний OMEN-Scale:

      [1] TokenEncoder (Fine)          — представляє токени
        ↓ PerceiverResampler           — T tokens → n_latents concepts
      [2] WorldRNN + M-Core (Coarse)   — моделює світ
        ↓ EpistemicGap + Curiosity     — виявляє незнання
      [3] DifferentiableProver (Sym)   — логічний висновок
        ↓ z_final = z + z_sym + v_mem
      [4] TokenDecoder                 — генерує текст

    Два цикли навчання:
      · Швидкий (GPU):  оновлення θ через L_ce + L_world + L_scale
      · Повільний (CPU): оновлення Γ через абдукцію (без градієнтів)
    """

    def __init__(self, cfg: OMENScaleConfig):
        super().__init__()
        self.cfg = cfg

        # ─── NET: Neural Epistemic Tokenizer ──────────────────────────────────
        # Якщо net_enabled=True  → NET повністю замінює tok_encoder + tok_decoder.
        #   tok_encoder і tok_decoder НЕ ініціалізуються — нульові мертві параметри.
        # Якщо net_enabled=False → класичний режим (TokenEncoder / TokenDecoder).
        #   Раніше обидва блоки завжди ініціалізувались → ~5.26M мертвих параметрів.
        self.net_enabled = cfg.net_enabled
        if cfg.net_enabled:
            self.net         = NeuralEpistemicTokenizer(cfg)
            self.tok_encoder = None   # явно None — не є nn.Module
            self.tok_decoder = None
        else:
            self.net         = None
            self.tok_encoder = TokenEncoder(cfg)
            self.tok_decoder = TokenDecoder(cfg)

        # ─── Perceiver: Token → Concept ───────────────────────────────────────
        self.perceiver = PerceiverResampler(
            d_tok=cfg.d_tok, d_latent=cfg.d_latent,
            n_latents=cfg.n_latents, n_heads=cfg.n_heads_lat,
            n_layers=cfg.n_layers_lat, dropout=cfg.dropout,
        )

        # ─── Рівень 2: Concept ────────────────────────────────────────────────
        v2cfg = _make_v2_compat(cfg)
        self.world_rnn = WorldRNN(v2cfg)
        self.memory    = AsyncTensorProductMemory(cfg)

        self.epistemic = EpistemicGapDetector(v2cfg)
        self.curiosity = CuriosityModule(v2cfg)

        # ─── Рівень 3: Symbolic (∂-Prolog) ────────────────────────────────────
        self.prover = DifferentiableProver(
            d_latent   = cfg.d_latent,
            sym_vocab  = cfg.sym_vocab,
            max_rules  = cfg.ltm_max_rules,
            max_depth  = cfg.max_proof_depth,
            n_cands    = cfg.n_proof_cands,
            alpha      = cfg.alpha,
            vem_tau    = getattr(cfg, 'vem_tau', 0.3),
            eta_utility = getattr(cfg, 'eta_utility', 0.1),
            consolidate_every = getattr(cfg, 'rule_consolidate_every', 100),
        )

        # ─── KB ↔ NET інтеграція: NET реєструє концепти прямо в Prolog-KB ──────
        # Раніше KB була відключена від NET → абдукція не бачила токен-концептів.
        if cfg.net_enabled:
            self.net.attach_kb(self.prover.kb)

        # ─── EMC: Efficient Meta-Controller ───────────────────────────────────
        # π_meta(a|s) вирішує КОЛИ зупинитись і ЯКУ дію виконати.
        # emc_enabled=True → повністю замінює fixed-depth prover.forward().
        # emc_enabled=False → стара поведінка (prover.forward() напряму).
        self.emc_enabled = getattr(cfg, 'emc_enabled', False)
        if self.emc_enabled:
            self.emc = EfficientMetaController(cfg)

        # ─── Loss ─────────────────────────────────────────────────────────────
        self.loss_fn = OMENScaleLoss(cfg)

        # ─── OSF: OMEN Synthesis Framework ────────────────────────────────────
        # OSF замінює / доповнює TokenDecoder ієрархічною генерацією.
        # osf_enabled=True → OSFSynthesizer використовується у forward().
        # osf_enabled=False → класичний TokenDecoder (зворотна сумісність).
        self.osf_enabled = getattr(cfg, 'osf_enabled', False)
        if self.osf_enabled:
            osf_cfg = OSFConfig(
                osf_enabled     = True,
                d_intent        = getattr(cfg, 'osf_d_intent', 64),
                n_goals         = getattr(cfg, 'osf_n_goals', 32),
                d_plan          = getattr(cfg, 'osf_d_plan', 64),
                n_operators     = getattr(cfg, 'osf_n_operators', 32),
                template_len    = getattr(cfg, 'osf_template_len', 8),
                max_plan_depth  = getattr(cfg, 'osf_max_plan_depth', 4),
                beam_width      = getattr(cfg, 'osf_beam_width', 2),
                lambda_plan     = getattr(cfg, 'osf_lambda_plan', 0.10),
                lambda_sim      = getattr(cfg, 'osf_lambda_sim', 0.05),
                lambda_refl     = getattr(cfg, 'osf_lambda_refl', 0.05),
                lambda_meta     = getattr(cfg, 'osf_lambda_meta', 0.05),
                lambda_intent   = getattr(cfg, 'osf_lambda_intent', 0.01),
                use_simulation  = getattr(cfg, 'osf_use_simulation', True),
                use_reflection  = getattr(cfg, 'osf_use_reflection', True),
                use_meta        = getattr(cfg, 'osf_use_meta', True),
                meta_beta       = getattr(cfg, 'osf_meta_beta', 0.1),
                gumbel_tau      = getattr(cfg, 'osf_gumbel_tau', 1.0),
                dropout         = cfg.dropout,
            )
            self.osf = OSFSynthesizer(
                cfg        = osf_cfg,
                d_latent   = cfg.d_latent,
                d_tok      = cfg.d_tok,
                vocab_size = cfg.vocab_size,
                n_heads    = getattr(cfg, 'n_heads_tok', 4),
            )
            self._osf_lambda_total = getattr(cfg, 'osf_lambda_total', 0.3)
            # Running CE estimate: ковзне середнє CE попередніх батчів.
            # EMA decay=0.9 → ~10-батчевий горизонт. Передається до OSFSynthesizer
            # замість хардкоду 5.0, щоб мета-контролер отримував реальну якість.
            self._osf_running_ce: float = 5.0

        # ─── torch.compile (опціонально) ──────────────────────────────────────
        if cfg.compile_model and not cfg.net_enabled:
            self.tok_encoder = torch.compile(self.tok_encoder)
            self.tok_decoder = torch.compile(self.tok_decoder)

        # ─── OPT-SEM: кеш semantic_feedback_pairs ─────────────────────────────
        # semantic_feedback_pairs() — O(n_rules) Python-цикл, кликається кожен батч.
        # Пари правил змінюються лише при додаванні/видаленні правил у KB.
        # Кешуємо результат; інвалідуємо тільки при зміні кількості правил.
        self._sem_pairs_cache: list = []
        self._sem_pairs_n_rules: int = -1


    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, src: torch.Tensor,
                tgt: torch.Tensor) -> Dict:
        """
        src: (B, T)  — вхідна послідовність
        tgt: (B, T)  — цільова послідовність

        Два режими:
          NET (net_enabled=True):  src → ByteContextEncoder → EpistemicQuantizer
                                       → Perceiver → ... → ByteDecoder
          Classic (net_enabled=False): src → TokenEncoder → Perceiver → ... → TokenDecoder
        """
        # ══ Рівень 1: Token → Concept  ═══════════════════════════════════════
        net_info   = {}
        vq_indices = None
        if self.net_enabled:
            # ── NET шлях: контекстне кодування + семантичне квантування ────────
            h_tok, vq_indices, net_info = self.net.encode(src)        # (B,T,d_tok)
        else:
            # ── Класичний шлях: просте Embedding + LlamaDecoderBlock ─────────
            h_tok = self.tok_encoder(src)                               # (B,T,d_tok)

        latents, z = self.perceiver(h_tok)                             # (B,n,d_lat), (B,d_lat)

        # ── Level 2: M-Core читання ───────────────────────────────────────────
        v_mem = self.memory.read(z)                                    # (B, d_lat)

        # ── Level 2: WorldRNN симуляція ───────────────────────────────────────
        # FIX: z.detach() — WorldRNN отримує тільки «знімок» концепту z,
        # без backprop крізь z. Це ізолює WorldRNN-градієнт від решти граду:
        # градієнт L_world тече → z_sim → WorldRNN.params, а не → z → perceiver.
        sim_tgt  = tgt[:, -8:] if tgt.size(1) > 8 else tgt
        z_sim_traj = self.world_rnn.simulate_sequence(z.detach(), sim_tgt)
        z_sim    = z_sim_traj[:, -1]                                   # (B, d_lat)

        # ── Epistemic Gap ─────────────────────────────────────────────────────
        E, gap_norm, hot_dims = self.epistemic.compute(
            z, self.world_rnn, z_sim)

        # ── Curiosity (якщо gap великий) ──────────────────────────────────────
        z_enr, cf_loss = self.curiosity(
            z, E, hot_dims, gap_norm, self.memory, self.world_rnn)

        # ── Level 3: ∂-Prolog (через EMC або напряму) ─────────────────────────
        world_err  = F.mse_loss(z_sim, z.detach()).detach()

        if self.emc_enabled and self.training:
            # ── EMC: Адаптивний контролер міркування ───────────────────────────
            # π_meta визначає: Stop | RecallMCore | ForwardChainStep | Abduce
            # Повертає (z_sym, sym_loss, v_mem_emc, meta_loss, traj_stats)
            z_sym, sym_loss, v_mem_emc, meta_loss, traj_stats = self.emc.run_episode(
                z_enr, gap_norm, self.prover, self.memory,
                world_err, device=z_enr.device,
            )
            # Якщо EMC виконав RecallMCore → використовуємо збагачений v_mem
            if v_mem_emc.norm() > 1e-6:
                v_mem = v_mem_emc
        else:
            # ── Класичний шлях (eval або emc_enabled=False) ────────────────────
            z_sym, sym_loss = self.prover(z_enr, world_err)
            meta_loss = torch.zeros(1, device=z_enr.device).squeeze()
            traj_stats = None

        # ── Semantic Feedback Pairs для NET (I(Z;Γ) апроксимація) ────────────
        # S-Core виявляє логічно пов'язані токени → NET наближає їх вектори.
        # Перетворюємо HornClause-пари на (token_idx_1, token_idx_2, score).
        # OPT-SEM: кешуємо результат, інвалідуємо лише при зміні кількості правил.
        _n_rules_now = len(self.prover)
        if _n_rules_now != self._sem_pairs_n_rules:
            self._sem_pairs_cache = self.prover.semantic_feedback_pairs(max_pairs=32)
            self._sem_pairs_n_rules = _n_rules_now
        sem_pairs_raw = self._sem_pairs_cache
        sem_pairs_net: list = []
        if self.net_enabled and sem_pairs_raw:
            V_cur = self.net.quantizer.current_size.item()
            for (r1, r2, score) in sem_pairs_raw:
                # Використовуємо pred як проксі для token_idx (обидва < V_cur)
                i1 = r1.head.pred % max(V_cur, 1)
                i2 = r2.head.pred % max(V_cur, 1)
                if i1 != i2:
                    sem_pairs_net.append((i1, i2, score))

        # ── VeM Penalty: δ·E[max(0, τ − U(R))] ─────────────────────────────
        vem_pen = self.prover.vem_loss(
            z_enr,
            delta=getattr(self.cfg, 'delta_vem', 1e-3)
        )

        # ── Об'єднуємо рівні ─────────────────────────────────────────────────
        z_final = z_enr + 0.1 * z_sym + 0.1 * v_mem                   # (B, d_lat)

        # ── M-Core: буферизований запис ───────────────────────────────────────
        conf = (1.0 - gap_norm.clamp(0, 1))
        self.memory.schedule_write(z.detach(), z_sim.detach(), conf.detach())

        # ══ Рівень 1: Decode  ════════════════════════════════════════════════
        # Ініціалізуємо osf_out заздалегідь — завжди визначена змінна незалежно
        # від того, який шлях декодування обрано (NET / OSF / classic).
        osf_out: Dict = {}

        if self.net_enabled:
            # NET декодер: реконструює оригінальні токени з h_tok + z_final
            logits, l_rec = self.net.decode(tgt, z_final, h_tok)      # (B,T,V), scalar
            # L_NET з семантичним feedback: -λ·I(Z;Γ) через sem_pairs_net
            net_loss_dict = self.net.compute_loss(
                net_info, l_rec,
                sem_pairs=sem_pairs_net if sem_pairs_net else None
            )
            net_loss = net_loss_dict["net_total"]
        elif self.osf_enabled:
            # ── OSF: ієрархічна генерація H1→H2→H3→H4 ─────────────────────────
            # OSF замінює TokenDecoder через нейро-символьне планування.
            # Повертає ті самі logits (B, T, V) + словник OSF losів.
            logits, osf_out = self.osf(
                h_tok      = h_tok,
                z_final    = z_final,
                tgt        = tgt,
                world_rnn  = self.world_rnn,
                gap_norm   = float(gap_norm.mean().item()),
                ce_loss    = self._osf_running_ce,  # EMA CE попереднього батчу
                n_rules    = len(self.prover),
                n_writes   = self.memory.n_writes,
            )
            net_loss      = torch.tensor(0.0, device=src.device)
            net_loss_dict = {}
        else:
            logits   = self.tok_decoder(tgt, z_final)                  # (B, T, V)
            net_loss = torch.tensor(0.0, device=src.device)
            net_loss_dict = {}
            sem_pairs_net = []

        # ── Loss J(θ,Γ,M) + η_tok·L_NET + δ·VeM + ω_meta·L_AC ──────────────
        ltm_pen = self.prover.rule_regularizer(
            self.cfg.lam_sym,
            eta_utility=getattr(self.cfg, 'eta_utility', 0.1)
        )
        # 7-ма компонента J_OMEN+EMC: ω_meta·E_τ[Σ_t r_t]
        traj_reward = traj_stats.trajectory_reward if traj_stats is not None else None
        out = self.loss_fn(
            logits, tgt, z_final,
            h_tok, latents,
            z_sim, v_mem, sym_loss, ltm_pen, cf_loss,
            self.world_rnn,
            net_loss,
            vem_penalty=vem_pen,
            meta_loss=meta_loss,
            traj_reward=traj_reward,
        )

        # ── OSF: додаємо J_OSF до загального лосу ─────────────────────────────
        # osf_out завжди визначений (ініціалізований вище як {}), тому
        # перевірка 'osf_out' in dir() не потрібна — замінено на пряму перевірку ключа.
        if self.osf_enabled and not self.net_enabled and "j_osf" in osf_out:
            j_osf = osf_out["j_osf"]
            if torch.is_tensor(j_osf):
                out["total"] = out["total"] + self._osf_lambda_total * j_osf
            # Додаємо OSF-статистику до out
            for k, v in osf_out.items():
                if k != "j_osf":
                    out[k] = v

        # ── OSF: оновлюємо running CE estimate (EMA decay=0.9) ────────────────
        # Виконується ПІСЛЯ loss_fn, щоб CE поточного батчу врахувалась
        # у мета-контролері наступного батчу.
        if self.osf_enabled:
            _ce_cur = out.get("ce", None)
            if _ce_cur is not None:
                try:
                    _ce_f = float(_ce_cur)
                    if not (math.isnan(_ce_f) or math.isinf(_ce_f)):
                        self._osf_running_ce = (
                            0.9 * self._osf_running_ce + 0.1 * _ce_f
                        )
                except (TypeError, ValueError):
                    pass

        out["logits"]    = logits
        out["z"]         = z_final
        out["gap_norm"]  = gap_norm.mean().item()
        out["n_rules"]   = len(self.prover)
        out["n_writes"]  = self.memory.n_writes
        out["pend_writes"] = len(self.memory._buf_s)
        out["unknown_ex"]= self.curiosity.unknown_flag_count

        # EMC-специфічна статистика (доступна лише при emc_enabled=True у training)
        if traj_stats is not None:
            out["emc_steps"]    = traj_stats.n_steps
            out["emc_stop"]     = traj_stats.stop_reason
            out["emc_proved"]   = int(traj_stats.goal_proved)
            out["emc_traj_r"]   = traj_stats.trajectory_reward
            out["emc_mdl"]      = traj_stats.proof_mdl
            out["emc_gap_fin"]  = traj_stats.final_gap
            # Частоти дій: [stop, recall, fc, abduce]
            hist = traj_stats.action_histogram
            if hist:
                total_a = max(sum(hist), 1)
                out["emc_a_stop"]   = hist[0] / total_a
                out["emc_a_recall"] = hist[1] / total_a
                out["emc_a_fc"]     = hist[2] / total_a
                out["emc_a_abduce"] = hist[3] / total_a

        # NET-специфічна статистика
        if self.net_enabled:
            out["net_vocab"]    = net_loss_dict.get("net_vocab_size", 0)
            out["net_entropy"]  = net_loss_dict.get("net_entropy", 0.0)
            out["net_mean_sim"] = net_loss_dict.get("net_mean_sim",  0.0)
            out["net_semantic"] = net_loss_dict.get("net_semantic",  0.0)
        # VeM / Epistemic Rule Tracker статистика
        out["vem_pen"]      = out.get("vem_pen", 0.0)
        # PERF FIX: використовуємо кешовані O(1) лічильники замість
        # O(n_records) sum comprehension по _records на кожному батчі.
        out["n_proposed"]   = self.prover.kb.n_proposed
        out["n_verified"]   = self.prover.kb.n_verified
        return out

    # ── Генерація ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor,
                 max_new: int = 32,
                 temperature: float = 0.8,
                 dynamic_reasoning: bool = True) -> torch.Tensor:
        """
        Генерує max_new токенів після prompt.

        dynamic_reasoning=True (за замовчуванням):
          На КОЖНОМУ кроці генерації перекодуємо поточний контекст ctx_t
          і запускаємо повний «повільний контур»:

            ctx_t ──[NET/TokenEnc]──> h_ctx
            h_ctx ──[Perceiver]─────> z_ctx       (концепт поточного контексту)
            z_ctx ──[∂-Prolog]──────> z_sym_step  (verified правила з LTM)
            z_ctx ──[M-Core read]───> v_mem_step  (episodic recall)
            z_final = z_ctx + 0.1·z_sym_step + 0.1·v_mem_step

          Це відповідає «Сценарію Б» (живе міркування під час генерації):
          S-Core динамічно зміщує розподіл ймовірностей декодера,
          спираючись на актуальний символьний стан знань.

        dynamic_reasoning=False:
          Класичний режим: z_final обчислюється ОДИН РАЗ по prompt
          і залишається фіксованим протягом всієї генерації.
          Швидший, але без динамічного «розуміння».
        """
        self.eval()

        # ── Ініціальний стан (для dynamic=False або як база) ─────────────────
        if self.net_enabled:
            h_tok, _, _ = self.net.encode(prompt)
        else:
            h_tok = self.tok_encoder(prompt)
        _, z  = self.perceiver(h_tok)
        v_mem = self.memory.read(z)

        # Початкове символьне представлення: через EMC (якщо ввімкнено) або прямо
        if self.emc_enabled:
            # Ініціальний Epistemic Gap для Bellman stopping.
            # FIX Bug-EGD: раніше тут створювався новий EpistemicGapDetector(v2cfg)
            # на кожен виклик generate() — нові (не навчені) ваги + зайва пам'ять.
            # self.epistemic вже існує і compute() є pure-functional (закрита формула),
            # тому використовуємо self.epistemic замість fresh instance.
            sim_tgt = prompt[:, -8:] if prompt.size(1) > 8 else prompt
            z_sim_traj = self.world_rnn.simulate_sequence(z.detach(), sim_tgt)
            z_sim0     = z_sim_traj[:, -1]
            _, gap_norm0, _ = self.epistemic.compute(z, self.world_rnn, z_sim0)
            z_sym, v_mem_emc = self.emc.run_episode_eval(
                z, gap_norm0, self.prover, self.memory, device=z.device)
            if v_mem_emc.norm() > 1e-6:
                v_mem = v_mem_emc
        else:
            z_sym, _ = self.prover(z, torch.tensor(0.0, device=z.device))

        z_final = z + 0.1 * z_sym + 0.1 * v_mem

        generated = prompt.clone()
        for _ in range(max_new):
            ctx = generated[:, -self.cfg.seq_len:]

            if dynamic_reasoning:
                # ── reasoning_step: перекодуємо + оновлюємо ∂-Prolog / M-Core
                if self.net_enabled:
                    h_ctx, _, _ = self.net.encode(ctx)
                else:
                    h_ctx = self.tok_encoder(ctx)
                _, z_ctx = self.perceiver(h_ctx)

                if self.emc_enabled:
                    # ── EMC адаптивне міркування під час inference ────────────
                    # π_meta* = argmax_π E[R_task − λ_time·T − λ_MDL·MDL(proof)]
                    # Bellman зупинка замість фіксованого max_depth.
                    # FIX Bug-EGD: self.epistemic замість fresh EpistemicGapDetector
                    z_sim_step = self.world_rnn.simulate_sequence(
                        z_ctx.detach(), ctx[:, -8:] if ctx.size(1) > 8 else ctx
                    )[:, -1]
                    _, gap_step, _ = self.epistemic.compute(z_ctx, self.world_rnn, z_sim_step)
                    z_sym_step, v_mem_step = self.emc.run_episode_eval(
                        z_ctx, gap_step, self.prover, self.memory,
                        device=z_ctx.device)
                else:
                    # ── Класичний шлях: прямий prover ────────────────────────
                    z_sym_step, _ = self.prover(
                        z_ctx, torch.tensor(0.0, device=z_ctx.device))
                    v_mem_step = self.memory.read(z_ctx)

                if self.emc_enabled and v_mem_step.norm() > 1e-6:
                    v_mem_use = v_mem_step
                else:
                    v_mem_use = self.memory.read(z_ctx)

                z_final      = z_ctx + 0.1 * z_sym_step + 0.1 * v_mem_use
                h_for_decode = h_ctx
            else:
                if self.net_enabled:
                    h_for_decode, _, _ = self.net.encode(ctx)
                elif self.osf_enabled:
                    # OSF потребує h_tok так само як NET → кодуємо через tok_encoder
                    h_for_decode = self.tok_encoder(ctx)   # (B, T, d_tok)
                else:
                    h_for_decode = None   # tok_decoder не потребує h_tok

            if self.net_enabled:
                logits, _ = self.net.decode(ctx, z_final, h_for_decode)
            elif self.osf_enabled:
                # ── OSF inference: ієрархічний декодер (без лосів) ───────────
                # Кроки: z_final → IntentEncoder → SymbolicPlanner → HierarchicalDecoder
                # Використовуємо eval-режим (self.training=False → Gumbel=sharpened softmax,
                # planner=greedy argmax, meta_ctrl не активується).
                _intent_state = self.osf.intent_encoder(z_final)
                _plan         = self.osf.planner(_intent_state)
                logits, _     = self.osf.hier_decoder(
                    h_tok    = h_for_decode,
                    z_intent = _intent_state.z_intent,
                    plan     = _plan,
                )
            else:
                logits = self.tok_decoder(ctx, z_final)

            probs     = F.softmax(logits[:, -1] / temperature, -1)
            generated = torch.cat([generated, torch.multinomial(probs, 1)], 1)

        return generated

    def memory_report(self) -> str:
        d  = self.cfg.d_latent
        H  = self.cfg.mem_heads
        mb = self.memory.memory_footprint_bytes()
        cb = len(self.memory.cache) * d * 4 * 2
        pm = sum(p.numel() * p.element_size()
                 for p in self.parameters()) / 1024 / 1024
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode = "NET (байт-рівень)" if self.net_enabled else "Classic (BPE)"

        # OPT-REPORT: використовуємо O(1) кешовані лічильники замість
        # O(n_records) sum comprehension по _records.values().
        n_prop = self.prover.kb.n_proposed
        n_ver  = self.prover.kb.n_verified
        # n_contradicted: окремий O(n) обхід лише у звіті (не на кожному батчі)
        n_cont = sum(1 for r in self.prover.kb._records.values()
                     if r.status.value == "contradicted")
        avg_util = (sum(r.utility() for r in self.prover.kb._records.values())
                    / max(len(self.prover.kb._records), 1))

        base = (
            f"  Режим токенайзера: {mode}\n"
            f"  Параметри      : {n_params:,}\n"
            f"  Розмір         : {pm:.2f} MB\n"
            f"  M-Core tensor  : {mb/1024:.1f} KB  (H={H}, d={d})\n"
            f"  M-Core cache   : {cb/1024:.1f} KB  ({len(self.memory.cache)} ep.)\n"
            f"  M-Core writes  : {self.memory.n_writes}\n"
            f"  ∂-Prolog rules : {len(self.prover)}\n"
            f"  KB facts       : {self.prover.kb.n_facts()}\n"
            f"  KB↔NET linked  : {self.net_enabled and self.net.quantizer.kb is not None}\n"
            f"  UNKNOWN flags  : {self.curiosity.unknown_flag_count}\n"
            f"  ── Epistemic Rule Tracker ──\n"
            f"  Rules proposed   : {n_prop}\n"
            f"  Rules verified   : {n_ver}\n"
            f"  Rules contrad.   : {n_cont}\n"
            f"  Avg Utility(R)   : {avg_util:.4f}\n"
            f"  VeM tau          : {getattr(self.cfg, 'vem_tau', 0.3)}\n"
            f"  VeM buffer       : {len(self.prover.vem._train_embs)}\n"
            f"  ── EMC Meta-Controller ──\n"
            f"  EMC enabled      : {self.emc_enabled}\n"
        )
        if self.emc_enabled:
            base += (
                f"  EMC max_steps    : {getattr(self.cfg, 'emc_max_steps', 5)}\n"
                f"  EMC lambda_time  : {getattr(self.cfg, 'emc_lambda_time', 0.05)}\n"
                f"  EMC lambda_mdl   : {getattr(self.cfg, 'emc_lambda_mdl', 0.01)}\n"
                f"  EMC use_gae      : {getattr(self.cfg, 'emc_use_gae', True)}\n"
                f"  omega_meta       : {getattr(self.cfg, 'omega_meta', 0.05)}\n"
            )
        if self.net_enabled:
            base += self.net.tokenizer_report()
        if self.osf_enabled:
            base += self.osf.memory_report()
        return base



# ══════════════════════════════════════════════════════════════════════════════
# 6.  ДОПОМІЖНЕ: OMENv2Config-compat wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _make_v2_compat(cfg: OMENScaleConfig) -> OMENv2Config:
    """Будує мінімальний OMENv2Config для компонентів WorldRNN / EGD / Curiosity."""
    return OMENv2Config(
        vocab_size        = cfg.vocab_size,
        d_model           = cfg.d_latent,
        d_latent          = cfg.d_latent,
        n_heads           = max(1, cfg.d_latent // 16),
        n_layers          = 1,
        seq_len           = cfg.seq_len,
        world_rnn_hidden  = cfg.world_rnn_hidden,
        dropout           = cfg.dropout,
        mem_heads         = cfg.mem_heads,
        mem_cache_size    = cfg.mem_cache_size,
        mem_write_tau     = cfg.mem_write_tau,
        epistemic_tau     = cfg.epistemic_tau,
        n_counterfactual  = cfg.n_counterfactual,
        sym_vocab         = cfg.sym_vocab,
        sym_embed_dim     = cfg.sym_embed_dim,
        sym_gnn_layers    = cfg.sym_gnn_layers,
        sym_max_facts     = cfg.sym_max_facts,
        abduct_candidates = cfg.abduct_candidates,
        ltm_max_rules     = cfg.ltm_max_rules,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ТРЕНУВАЛЬНИЙ ЦИКЛ
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch_scale(model: OMENScale, dataset, optimizer,
                      batch_size: int = 8, max_batches: int = 8) -> Dict:
    model.train()
    random.shuffle(dataset)
    agg   = defaultdict(float)
    n_b   = 0
    t0    = time.perf_counter()
    tot_tok = 0

    for start in range(0, len(dataset) - batch_size, batch_size):
        batch  = dataset[start: start + batch_size]
        src, tgt = collate(batch)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        optimizer.zero_grad()
        out = model(src, tgt)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # FIX Bug-Flush: maybe_flush() ПІСЛЯ optimizer.step() — autograd graph вже знищено,
        # safe для inplace оновлення memory buffer (уникаємо version mismatch).
        # joint_train() правильно викликає maybe_flush() після кожного step;
        # train_epoch_scale() раніше цього не робив → пам'ять оновлювалась лише в кінці епохи.
        model.memory.maybe_flush()

        for k, v in out.items():
            if k not in ("logits", "z", "emc_stop"):
                try:
                    agg[k] += float(v) if torch.is_tensor(v) else float(v)
                except (TypeError, ValueError):
                    pass  # пропускаємо нечислові значення (наприклад, рядки)

        tot_tok += tgt.numel()
        n_b += 1
        if n_b >= max_batches:
            break

    # Примусовий flush залишкового буфера пам'яті
    model.memory.flush()

    elapsed = (time.perf_counter() - t0) * 1000
    avg = {k: v / n_b for k, v in agg.items()}
    avg["ppl"] = math.exp(min(avg.get("ce", 10), 10))
    avg["tps"] = tot_tok / (elapsed / 1000)
    avg["ms"]  = elapsed
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# 8.  INLINE ТЕСТИ
# ══════════════════════════════════════════════════════════════════════════════

def run_tests_scale(cfg: OMENScaleConfig) -> None:
    sep = lambda s: print(f"\n{'═'*70}\n  {s}\n{'═'*70}")

    sep("TEST S0 · Ініціалізація та footprint")
    model = OMENScale(cfg).to(DEVICE)
    print(model.memory_report())
    print("  [PASS]")

    B, T = 4, cfg.seq_len - 1
    src = torch.randint(1, min(cfg.vocab_size, 200), (B, T)).to(DEVICE)
    tgt = torch.randint(1, min(cfg.vocab_size, 200), (B, T)).to(DEVICE)

    sep("TEST S1 · Forward pass — форми всіх виходів")
    t0  = time.perf_counter()
    out = model(src, tgt)
    ms  = (time.perf_counter() - t0) * 1000
    for k in ("total", "ce", "world", "l_scale", "sym_ground", "gap_norm"):
        assert k in out, f"FAIL: відсутній ключ {k}"
    assert out["logits"].shape == (B, T, cfg.vocab_size), \
        f"logits {out['logits'].shape}"
    assert out["z"].shape == (B, cfg.d_latent), f"z {out['z'].shape}"
    print(f"  Forward {ms:.0f} ms  CE={out['ce']:.3f}  L_scale={out['l_scale']:.5f}")
    print(f"  gap_norm={out['gap_norm']:.4f}  rules={out['n_rules']}  [PASS]")

    sep("TEST S2 · Backward — grad flow по всіх рівнях")
    model.train()
    out2 = model(src, tgt)
    out2["total"].backward()
    model.memory.flush()

    # NET: градієнти у net.byte_encoder; Classic: у tok_encoder
    if cfg.net_enabled:
        enc_g = sum(p.grad.norm().item() for n, p in model.named_parameters()
                    if "net.byte_encoder" in n and p.grad is not None)
        enc_label = "ByteContextEncoder (NET)"
    else:
        enc_g = sum(p.grad.norm().item() for n, p in model.named_parameters()
                    if "tok_encoder" in n and p.grad is not None)
        enc_label = "TokenEncoder (Classic)"

    perceiver_g= sum(p.grad.norm().item() for n,p in model.named_parameters()
                     if "perceiver" in n and p.grad is not None)
    prover_g   = sum(p.grad.norm().item() for n,p in model.named_parameters()
                     if "prover" in n and p.grad is not None)
    wrnn_g     = sum(p.grad.norm().item() for n,p in model.named_parameters()
                     if "world_rnn" in n and p.grad is not None)

    print(f"  {enc_label} grad : {enc_g:.4f}")
    print(f"  Perceiver grad      : {perceiver_g:.4f}")
    print(f"  ∂-Prolog grad       : {prover_g:.4f}")
    print(f"  WorldRNN grad       : {wrnn_g:.4f}")
    assert enc_g       > 0, f"FAIL: {enc_label} без граду"
    assert perceiver_g > 0, "FAIL: Perceiver без граду"
    model.zero_grad()
    print("  [PASS]")

    sep("TEST S3 · Async M-Core — flush та запис")
    model.eval()
    n_before = model.memory.n_writes
    z_test   = torch.randn(B, cfg.d_latent, device=DEVICE)
    conf_test = torch.ones(B, device=DEVICE) * 0.1   # низька впевненість → пишемо
    for _ in range(cfg.mem_update_steps + 1):
        model.memory.schedule_write(z_test, z_test, conf_test)
    # Явний flush після schedule_write (в реальному тренуванні — після optimizer.step)
    model.memory.flush()
    n_after = model.memory.n_writes
    print(f"  Writes before={n_before}  after={n_after}  (delta={n_after-n_before})")
    assert n_after > n_before, "FAIL: flush не спрацював"
    print("  [PASS]")

    sep("TEST S4 · ∂-Prolog — Forward Chaining + Abduce")
    prover = model.prover
    from omen_prolog import HornAtom, HornClause, Var, Const

    # Правило: p50(X, Const(22)) :- p50(X, Const(22)) → p51(X, Const(0))
    # Var("X") — іменована змінна (буде звязана при уніфікації)
    X = Var("X_fc_test")

    f_atom = HornAtom(pred=50, args=(Const(11), Const(22)))
    prover.kb.add_fact(f_atom)

    rule_body = (HornAtom(pred=50, args=(X, Const(22))),)
    rule_head = HornAtom(pred=51, args=(X, Const(0)))
    prover.kb.add_rule(HornClause(head=rule_head, body=rule_body))

    derived = prover.kb.forward_chain(max_depth=3)
    derived_preds = {f.pred for f in derived}
    print(f"  KB facts={prover.kb.n_facts()}  rules={len(prover.kb)}")
    print(f"  Derived predicates: {derived_preds}")
    assert 51 in derived_preds, f"FAIL: правило не застосувалось, derived={derived_preds}"

    z_abd = torch.randn(1, cfg.d_latent, device=DEVICE)
    n_add  = prover.abduce_and_learn(z_abd, error=2.0)
    print(f"  Abduce додав: {n_add} правил  [PASS]")

    sep("TEST S5 · L_scale penalty — MDL ефект")
    from omen_perceiver import l_scale_penalty
    # Великий вектор → великий штраф (заохочує стиснення)
    z_big   = torch.randn(B, T, cfg.d_tok, device=DEVICE) * 10
    z_small = torch.randn(B, T, cfg.d_tok, device=DEVICE) * 0.1
    z_conc  = torch.randn(B, cfg.n_latents, cfg.d_latent, device=DEVICE)
    pen_big   = l_scale_penalty(z_big,   z_conc, cfg.lambda_tok, cfg.lambda_conc)
    pen_small = l_scale_penalty(z_small, z_conc, cfg.lambda_tok, cfg.lambda_conc)
    assert pen_big > pen_small, "FAIL: L_scale не реагує на норму"
    print(f"  L_scale(big)={pen_big.item():.4f}  L_scale(small)={pen_small.item():.4f}  [PASS]")

    sep("TEST S6 · Мінімальне навчання 15 ітерацій")
    model.train()
    ds  = make_counting(64, cfg.seq_len)
    opt = AdamW(model.parameters(), lr=3e-4)
    hist_ce = []
    for step in range(15):
        batch = random.sample(ds, 4)
        s, t  = collate(batch)
        s, t  = s.to(DEVICE), t.to(DEVICE)
        opt.zero_grad()
        o = model(s, t)
        o["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist_ce.append(o["ce"])
    model.memory.flush()
    first5 = sum(hist_ce[:5])  / 5
    last5  = sum(hist_ce[-5:]) / 5
    print(f"  CE (перші 5): {first5:.3f}  (останні 5): {last5:.3f}")
    assert last5 < first5, "FAIL: CE не знижується"
    print("  [PASS]")

    sep("TEST S7 · Генерація токенів (dynamic_reasoning=True/False)")
    model.eval()
    prompt = torch.randint(10, 100, (1, 8), device=DEVICE)

    # dynamic_reasoning=True — S-Core + M-Core оновлюються на кожному кроці
    with torch.no_grad():
        gen_dyn = model.generate(prompt, max_new=12, dynamic_reasoning=True)
    assert gen_dyn.shape[1] == 20, f"FAIL: gen_dyn shape {gen_dyn.shape}"
    print(f"  Prompt          : {prompt[0].tolist()}")
    print(f"  Output (dynamic): {gen_dyn[0, 8:].tolist()}")

    # dynamic_reasoning=False — класичний режим (z_final фіксований)
    with torch.no_grad():
        gen_static = model.generate(prompt, max_new=12, dynamic_reasoning=False)
    assert gen_static.shape[1] == 20, f"FAIL: gen_static shape {gen_static.shape}"
    print(f"  Output (static) : {gen_static[0, 8:].tolist()}")
    print("  [PASS]")

    print(f"\n{'═'*70}")
    print("  ✅  Всі тести OMEN-Scale пройдено успішно")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_scale(cfg: OMENScaleConfig, epochs: int = 4) -> None:
    print("╔" + "═"*74 + "╗")
    print("║   OMEN-Scale — BENCHMARK" + " "*49 + "║")
    print("╚" + "═"*74 + "╝\n")

    datasets = {
        "Count":        make_counting(128, cfg.seq_len),
        "Python":       make_python(128, cfg.seq_len),
        "RuleTransfer": make_rule_transfer(128, cfg.seq_len),
    }

    fmt = "{:>7}" * 9
    hdr = fmt.format("Ep", "CE↓", "World", "LScale", "SymGr",
                     "Gap", "Rules", "PPL↓", "ms")

    for ds_name, ds in datasets.items():
        print(f"\n  ── {ds_name} ──")
        print(hdr)
        print("-" * 63)
        model = OMENScale(cfg).to(DEVICE)
        opt   = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = CosineAnnealingLR(opt, T_max=epochs)
        best_ppl = float("inf")

        for ep in range(1, epochs + 1):
            avg = train_epoch_scale(model, ds, opt, batch_size=8, max_batches=6)
            sched.step()
            best_ppl = min(best_ppl, avg["ppl"])
            print(fmt.format(
                ep,
                f"{avg.get('ce',0):.3f}",
                f"{avg.get('world',0):.3f}",
                f"{avg.get('l_scale',0):.4f}",
                f"{avg.get('sym_ground',0):.3f}",
                f"{avg.get('gap_norm',0):.3f}",
                f"{int(avg.get('n_rules',0))}",
                f"{avg.get('ppl',0):.1f}",
                f"{avg.get('ms',0):.0f}",
            ))

        print("-" * 63)
        print(f"  Best PPL: {best_ppl:.2f}")
        print(model.memory_report())


# ══════════════════════════════════════════════════════════════════════════════
# 10.  ABLATION: OMEN-Scale vs OMENv2 vs CE-only
# ══════════════════════════════════════════════════════════════════════════════

def ablation_compare(cfg: OMENScaleConfig) -> None:
    """Порівнює OMEN-Scale з базовими варіантами на RuleTransfer."""
    from omen_v2 import OMENv2
    print(f"\n{'═'*70}")
    print("  ABLATION: OMEN-Scale vs OMENv2 (full) vs CE-only")
    print(f"{'═'*70}")

    ds = make_rule_transfer(128, cfg.seq_len)

    # OMEN-Scale
    model_sc = OMENScale(cfg).to(DEVICE)
    opt_sc   = AdamW(model_sc.parameters(), lr=1e-3)
    ppl_sc   = []
    for _ in range(4):
        avg = train_epoch_scale(model_sc, ds, opt_sc, batch_size=8, max_batches=6)
        ppl_sc.append(round(avg["ppl"], 1))

    # OMENv2
    v2cfg  = _make_v2_compat(cfg)
    model_v2 = OMENv2(v2cfg).to(DEVICE)
    opt_v2   = AdamW(model_v2.parameters(), lr=1e-3)
    from omen_v2 import train_epoch
    ppl_v2 = []
    for _ in range(4):
        avg = train_epoch(model_v2, ds, opt_v2, batch_size=8, max_batches=6)
        model_v2.memory.write(*avg.get("write_args", (
            torch.zeros(1, v2cfg.d_latent, device=DEVICE),
            torch.zeros(1, v2cfg.d_latent, device=DEVICE),
            torch.ones(1, device=DEVICE),
        )))
        ppl_v2.append(round(avg["ppl"], 1))

    fmt_row = lambda name, ppls: (
        f"  {name:<28} " + " → ".join(map(str, ppls))
    )
    print(fmt_row("OMEN-Scale (3 рівні)", ppl_sc))
    print(fmt_row("OMENv2 (v2 full)", ppl_v2))
    print(f"\n  OMEN-Scale min PPL : {min(ppl_sc):.1f}")
    print(f"  OMENv2     min PPL : {min(ppl_v2):.1f}")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42); random.seed(42)

    cfg = OMENScaleConfig.demo()   # тести на demo-конфігу
    print(f"OMEN-Scale demo config:")
    print(f"  vocab={cfg.vocab_size}  d_tok={cfg.d_tok}  d_latent={cfg.d_latent}")
    print(f"  seq_len={cfg.seq_len}  n_latents={cfg.n_latents}")
    print(f"  mem_update_steps={cfg.mem_update_steps}  max_proof_depth={cfg.max_proof_depth}")
    print(f"  device={DEVICE}\n")

    run_tests_scale(cfg)
    benchmark_scale(cfg, epochs=4)
    ablation_compare(cfg)
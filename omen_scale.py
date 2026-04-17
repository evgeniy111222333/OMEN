"""
omen_scale.py — OMEN-Scale: canonical OMEN runtime stack
========================================================
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

Зовнішні залежності: omen_world_model, omen_data, omen_perceiver, omen_prolog, omen_scale_config
"""

from __future__ import annotations
import math, time, random, warnings
from collections import defaultdict, deque
from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Локальні модулі ──────────────────────────────────────────────────────────
from omen_scale_config import OMENScaleConfig
from omen_canonical import (
    CANONICAL_OMEN_SPEC,
    CanonicalArchitectureSpec,
    inject_canonical_metadata,
)
from omen_perceiver    import (PerceiverResampler, LlamaDecoderBlock,
                                RMSNorm, SwiGLUFFN, LlamaAttention,
                                l_scale_penalty)
from omen_prolog       import (
    Const,
    DifferentiableProver,
    HornAtom,
    HornClause,
    SymbolicTaskContext,
    EpistemicStatus,
    Var,
    SEQ_ACTUAL_NEXT_PRED,
    SEQ_AST_SUPPORT_PRED,
    SEQ_EDGE_PRED,
    SEQ_GAP_DIM_PRED,
    SEQ_LAST_TOKEN_PRED,
    SEQ_PREDICT_NEXT_PRED,
    SEQ_SALIENCY_SUPPORT_PRED,
    SEQ_DECODER_GUESS_PRED,
    SEQ_DECODER_MISS_PRED,
    SEQ_DECODER_SURPRISE_PRED,
)
from omen_ast_multilang import MultiLangASTParser

# NET: Neural Epistemic Tokenizer (замінює GPT-2 BPE)
from omen_net_tokenizer import NeuralEpistemicTokenizer
from omen_saliency import SaliencyTraceModule
from omen_symbolic.integration import SymbolicStateIntegrator
from omen_symbolic.memory_index import SymbolicMemoryIndex
from omen_symbolic.world_graph import CanonicalWorldState, WorldGraphBatch, WorldGraphEncoder
from omen_symbolic.execution_trace import (
    TRACE_ASSIGN_EVENT_PRED,
    TRACE_BINOP_EVENT_PRED,
    TRACE_ERROR_EVENT_PRED,
    TRACE_PARAM_BIND_PRED,
    TRACE_RETURN_EVENT_PRED,
    TRACE_STATE_VALUE_PRED,
    build_symbolic_trace_bundle,
)
from omen_symbolic.universal_bits import (
    gaussian_kl_bits,
    gaussian_nll_bits,
    gaussian_tensor_bits,
)

# EMC: Efficient Meta-Controller (адаптивний контролер міркування)
from omen_emc import EfficientMetaController

# OSF: OMEN Synthesis Framework (ієрархічна нейро-символьна генерація)
from omen_osf import OSFSynthesizer, OSFConfig

# Канонічні world/model компоненти та датасет-утиліти
from omen_world_model import (
    WorldRNN,
    EpistemicGapDetector,
    CuriosityModule,
    OMENCoreConfig,
)
from omen_data import make_counting, make_python, make_rule_transfer, collate


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
        self.decay        = float(getattr(cfg, "mem_decay", 1.0))
        self.update_steps = cfg.mem_update_steps
        self.cache:  deque = deque(maxlen=cfg.mem_cache_size)
        self.symbolic_index = SymbolicMemoryIndex(
            max_entries=int(getattr(cfg, "mem_symbolic_cache_size", cfg.mem_cache_size))
        )
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
            base_mem = self.memory * self.decay
            new_mem = base_mem + delta / (mask.sum().float() + 1e-6)
            self.memory.data.copy_(new_mem)
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

    def write_symbolic_atoms(
        self,
        facts: List[object],
        embeddings: torch.Tensor,
    ) -> int:
        if not facts or embeddings.numel() == 0:
            return 0
        embs = embeddings.detach()
        if embs.dim() == 1:
            embs = embs.unsqueeze(0)
        # Exact symbolic recall and associative M-Core recall now share
        # the same long-term write path instead of two unrelated stores.
        added = self.symbolic_index.write(facts, embs)
        write_conf = torch.zeros(embs.size(0), device=embs.device, dtype=embs.dtype)
        self.schedule_write(embs, embs, write_conf)
        return added

    @torch.no_grad()
    def recall_symbolic_atoms(
        self,
        z_query: torch.Tensor,
        top_k: int = 8,
        min_sim: float = 0.2,
        predicate_hints: Optional[List[int]] = None,
        anchor_values: Optional[List[int]] = None,
    ) -> List[object]:
        return self.symbolic_index.recall(
            z_query,
            top_k=top_k,
            min_sim=min_sim,
            predicate_hints=predicate_hints,
            anchor_values=anchor_values,
            structured_limit=max(2, top_k // 2),
        )


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

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        """tokens: (B, T) → hidden: (B, T, d_tok)"""
        x = self.drop(self.embed(tokens))
        attn_maps: List[torch.Tensor] = []
        for blk in self.blocks:
            if return_attn:
                x, attn_weights = blk(x, need_weights=True)
                attn_maps.append(attn_weights)
            else:
                x = blk(x)
        x = self.norm(x)
        if return_attn:
            return x, torch.stack(attn_maps, dim=1)
        return x                                        # (B, T, d_tok)


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

    @staticmethod
    def _gaussian_kl(
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: Optional[torch.Tensor] = None,
        logvar_p: Optional[torch.Tensor] = None,
        free_bits: float = 0.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        return gaussian_kl_bits(
            mu_q,
            logvar_q,
            mu_p=mu_p,
            logvar_p=logvar_p,
            free_bits=free_bits,
            reduction=reduction,
        )

    @staticmethod
    def _gaussian_nll(
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        to_bits: bool = False,
        reduction: str = "mean",
    ) -> torch.Tensor:
        nll_bits = gaussian_nll_bits(x, mu, logvar, reduction=reduction)
        if to_bits:
            return nll_bits
        return nll_bits * math.log(2.0)

    def forward(self,
                logits:       torch.Tensor,
                targets:      torch.Tensor,
                z:            torch.Tensor,        # (B, d_latent)
                mu:           torch.Tensor,        # (B, d_latent)
                logvar:       torch.Tensor,        # (B, d_latent)
                z_tok:        torch.Tensor,        # (B, T, d_tok)
                z_latents:    torch.Tensor,        # (B, n, d_latent)
                z_sim:        torch.Tensor,        # (B, d_latent)
                z_world_targets: torch.Tensor,     # (B, T_w, d_latent)
                v_mem:        torch.Tensor,        # (B, d_latent)
                z_sym:        torch.Tensor,        # (B, d_latent)
                sym_loss:     torch.Tensor,
                ltm_penalty:  float,
                curiosity_l:  torch.Tensor,
                world_rnn:    WorldRNN,
                net_loss:     torch.Tensor,        # L_NET від NeuralEpistemicTokenizer
                priors:       Optional[Dict[str, torch.Tensor]] = None,
                model_bits:   Optional[Dict[str, torch.Tensor]] = None,
                vem_penalty:  Optional[torch.Tensor] = None,  # δ·E[max(0,τ−U(R))]
                meta_loss:    Optional[torch.Tensor] = None,  # ω_meta·L_AC (EMC)
                traj_reward:  Optional[float]       = None,   # Σ_t r_t траєкторії
                reasoning_cost: Optional[float]    = None,   # Cost(Reasoning)
                program_anchor: Optional[torch.Tensor] = None,
                program_decoder_ce: Optional[torch.Tensor] = None,
                seen_tokens:  Optional[int]        = None,
                train_step:   int                  = 0,
                ) -> Dict:
        cfg = self.cfg

        # ── 1. Перплексія (next-token prediction, зсунуте) ───────────────────
        # FIX: logits[t] = P(next | tgt[0..t]) — прогнозує НАСТУПНИЙ токен.
        # Попередня версія CE(logits, targets) порівнювала logits[t] з tgt[t]
        # (поточним), що давало артефактно низький PPL (~1.4) бо токен tgt[t]
        # вже присутній у контексті. Правильно: logits[:-1] → targets[1:].
        valid_tokens = max(int(targets[:, 1:].ne(0).sum().item()), 1)
        token_nll_nats = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            targets[:, 1:].reshape(-1),
            ignore_index=0,
        )
        token_nll = token_nll_nats / math.log(2.0)
        token_bits_total = token_nll * float(valid_tokens)

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
        world_pred = z_sim if z_sim.dim() == z_world_targets.dim() else z_sim.unsqueeze(1)
        L_world_raw  = F.huber_loss(world_pred, z_world_targets.detach(), delta=1.0)
        L_world      = torch.log1p(L_world_raw)
        world_obs_logvar = (
            priors["world_obs_logvar"]
            if priors is not None and "world_obs_logvar" in priors
            else torch.zeros(1, 1, 1, device=z.device, dtype=z.dtype)
        )
        world_nll = self._gaussian_nll(
            z_world_targets.detach(),
            world_pred,
            world_obs_logvar,
            to_bits=True,
            reduction="sum",
        )

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
        free_bits = float(getattr(cfg, "vfe_free_bits", 0.0))
        mem_prior_mu = (
            priors["mem_mu"]
            if priors is not None and "mem_mu" in priors
            else v_mem.detach()
        )
        mem_prior_logvar = (
            priors["mem_logvar"]
            if priors is not None and "mem_logvar" in priors
            else None
        )
        sym_prior_mu = (
            priors["sym_mu"]
            if priors is not None and "sym_mu" in priors
            else z_sym.detach()
        )
        sym_prior_logvar = (
            priors["sym_logvar"]
            if priors is not None and "sym_logvar" in priors
            else None
        )
        L_kl = self._gaussian_kl(mu, logvar, free_bits=free_bits, reduction="sum")
        L_mem_kl = self._gaussian_kl(
            mu,
            logvar,
            mu_p=mem_prior_mu,
            logvar_p=mem_prior_logvar,
            free_bits=free_bits,
            reduction="sum",
        )
        L_sym_kl = self._gaussian_kl(
            mu,
            logvar,
            mu_p=sym_prior_mu,
            logvar_p=sym_prior_logvar,
            free_bits=free_bits,
            reduction="sum",
        )

        # Replace the old ||v_mem|| proxy with an explicit read-likelihood term:
        #   -log P(Read(M, z) | z)
        mem_read_mu = (
            priors["mem_read_mu"]
            if priors is not None and "mem_read_mu" in priors
            else z.detach()
        )
        mem_read_logvar = (
            priors["mem_read_logvar"]
            if priors is not None and "mem_read_logvar" in priors
            else torch.zeros_like(mem_read_mu)
        )
        memory_read_nll = self._gaussian_nll(
            v_mem.detach(),
            mem_read_mu,
            mem_read_logvar,
            to_bits=True,
            reduction="sum",
        )
        memory_read_alpha = float(getattr(cfg, "alpha", 0.1))
        vfe_beta = float(getattr(cfg, "vfe_beta_kl", 1.0))
        latent_scale_bits = (
            l_scale_penalty(
                z_tok,
                z_latents,
                float(getattr(cfg, "lambda_tok", 1.0)),
                float(getattr(cfg, "lambda_conc", 1.0)),
            )
            * float(valid_tokens)
        )

        # ── 6. L_scale: MDL для рівнів у єдиній валюті bits/token ─────────────
        if model_bits is None:
            neural_model_bits = torch.zeros((), device=z.device, dtype=z.dtype)
            vocab_model_bits = torch.zeros((), device=z.device, dtype=z.dtype)
        else:
            neural_model_bits = model_bits.get(
                "neural", torch.zeros((), device=z.device, dtype=z.dtype)
            )
            vocab_model_bits = model_bits.get(
                "vocab", torch.zeros((), device=z.device, dtype=z.dtype)
            )
        L_scale = neural_model_bits + vocab_model_bits
        rule_bits_raw = torch.as_tensor(
            float(ltm_penalty),
            dtype=z.dtype,
            device=z.device,
        )
        rule_bits = float(getattr(cfg, "lambda_rule", 1e-4)) * rule_bits_raw

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
        use_aux_schedule = bool(getattr(cfg, "use_aux_loss_schedule", False))
        aux_phase = 1.0
        if use_aux_schedule:
            aux_warmup = max(int(getattr(cfg, 'loss_aux_warmup', 500)), 1)
            aux_phase = min(max(float(train_step) / aux_warmup, 0.0), 1.0)
            world_w = cfg.gamma * (0.35 + 0.65 * aux_phase)
            sym_w = cfg.beta * (0.25 + 0.75 * aux_phase)
            recall_w = cfg.eta * (0.50 + 0.50 * aux_phase)
            curiosity_w = 0.05 + 0.05 * aux_phase
            net_w = cfg.eta_tok * (0.40 + 0.60 * aux_phase)
            vem_w = getattr(cfg, 'delta_vem', 1e-3) * (0.35 + 0.65 * aux_phase)
            meta_w = omega_meta * (0.25 + 0.75 * aux_phase)
        else:
            world_w = cfg.gamma
            sym_w = cfg.beta
            recall_w = cfg.eta
            curiosity_w = 0.10
            net_w = cfg.eta_tok
            vem_w = getattr(cfg, 'delta_vem', 1e-3)
            meta_w = omega_meta

        # ── 10. J_OMEN+EMC: 7-ма компонента — траєкторна мета-винагорода ─────
        program_enabled = bool(getattr(cfg, "program_anchor_enabled", True))
        program_anchor_w = (
            float(getattr(cfg, "program_anchor_weight", 0.10))
            if program_enabled else 0.0
        )
        program_decoder_w = (
            float(getattr(cfg, "program_decoder_weight", 0.05))
            if program_enabled else 0.0
        )
        program_anchor_t = (
            program_anchor.clamp(max=5.0)
            if program_anchor is not None and torch.is_tensor(program_anchor)
            else torch.zeros((), device=z.device, dtype=z.dtype)
        )
        program_decoder_nats = (
            program_decoder_ce.clamp(max=10.0)
            if program_decoder_ce is not None and torch.is_tensor(program_decoder_ce)
            else torch.zeros((), device=z.device, dtype=z.dtype)
        )
        program_decoder_bits = program_decoder_nats / math.log(2.0)

        if traj_reward is not None and traj_reward != 0.0:
            L_traj = -torch.tensor(traj_reward, dtype=torch.float32, device=z.device).clamp(-5.0, 5.0)
        else:
            L_traj = torch.zeros(1, device=z.device).squeeze()
        L_reason = torch.as_tensor(
            0.0 if reasoning_cost is None else reasoning_cost,
            dtype=z.dtype,
            device=z.device,
        ).clamp(0.0, 10.0)

        # ── ltm_penalty: клемпуємо щоб уникнути вибуху зі зростанням LTM ─────
        # Без кліпу: при 1024 правилах × complexity≈5 → ltm_pen≈25 → домінує loss.
        # Rule bits are already amortized above; no extra clipping needed here.

        # ── Sym loss: обрізаємо нестабільний symbolic consistency ──────────────
        L_sym_clamped = L_sym.clamp(max=5.0) if torch.is_tensor(L_sym) else torch.tensor(
            min(float(L_sym), 5.0), device=z.device)

        # ── Assemble explicit MDL bits + auxiliary training signals ───────────
        # Everything below `total_bits` is in the same currency: bits.
        # Auxiliary terms stay outside MDL and are tracked separately.
        curiosity_clamped = curiosity_l.clamp(max=5.0) if torch.is_tensor(curiosity_l) \
                            else min(float(curiosity_l), 5.0)
        observation_bits = token_bits_total + world_nll
        local_complexity_bits = (
            vfe_beta * (L_kl + L_mem_kl + L_sym_kl)
            + memory_read_alpha * memory_read_nll
            + latent_scale_bits
        )
        mdl_seen_tokens = max(int(seen_tokens or valid_tokens), valid_tokens)
        global_model_bits = L_scale + rule_bits
        complexity_bits = local_complexity_bits + global_model_bits
        total_bits = observation_bits + complexity_bits
        bits_per_token = (
            (observation_bits + local_complexity_bits) / float(valid_tokens)
            + global_model_bits / float(mdl_seen_tokens)
        )
        auxiliary_energy = (
            world_w * L_world
            + sym_w * L_sym_clamped
            + recall_w * L_recall
            + cfg.delta * L_complex
            + curiosity_w * curiosity_clamped
            + net_w * net_loss
            + vem_w * L_vem
            + meta_w * L_meta
            + meta_w * L_traj
            + meta_w * L_reason
            + program_anchor_w * program_anchor_t
            + program_decoder_w * program_decoder_bits
        )
        free_energy = bits_per_token
        total = bits_per_token + auxiliary_energy

        # Фінальна NaN-guard: якщо total=NaN → повертаємо тільки L_ce
        if torch.isnan(total) or torch.isinf(total):
            total = token_nll

        # net_loss може бути dict або scalar — нормалізуємо
        net_loss_scalar = (net_loss["net_total"].item()
                           if isinstance(net_loss, dict)
                           else (net_loss.item() if torch.is_tensor(net_loss)
                                 else float(net_loss)))

        return {
            "total":      total,
            "ce":         token_nll_nats.item(),
            "ce_bits":    token_nll.item(),
            "token_bits": token_bits_total.item(),
            "world":      L_world.item(),
            "world_nll":  (world_nll / float(valid_tokens)).item(),
            "world_bits": world_nll.item(),
            "complex":    L_complex.item(),
            "kl":         (L_kl / float(valid_tokens)).item(),
            "mem_kl":     (L_mem_kl / float(valid_tokens)).item(),
            "sym_kl":     (L_sym_kl / float(valid_tokens)).item(),
            "vfe_beta_kl": vfe_beta,
            "recall":     L_recall.item(),
            "novelty":    (memory_read_alpha * memory_read_nll / float(valid_tokens)).item(),
            "memory_read_nll": (memory_read_alpha * memory_read_nll / float(valid_tokens)).item(),
            "memory_read_nll_raw": (memory_read_nll / float(valid_tokens)).item(),
            "memory_read_alpha": memory_read_alpha,
            "memory_bits": (memory_read_alpha * memory_read_nll).item(),
            "l_scale":    (L_scale / float(mdl_seen_tokens)).item(),
            "l_scale_latent": (latent_scale_bits / float(valid_tokens)).item(),
            "model_bits": neural_model_bits.item(),
            "model_bits_bpt": (neural_model_bits / float(mdl_seen_tokens)).item(),
            "vocab_bits": vocab_model_bits.item(),
            "vocab_bits_bpt": (vocab_model_bits / float(mdl_seen_tokens)).item(),
            "rule_bits":  rule_bits.item(),
            "rule_bits_raw": rule_bits_raw.item(),
            "rule_lambda": float(getattr(cfg, "lambda_rule", 1e-4)),
            "rule_bits_bpt": (rule_bits / float(mdl_seen_tokens)).item(),
            "mdl_token_budget": float(valid_tokens),
            "mdl_seen_tokens": float(mdl_seen_tokens),
            "sym_ground": L_sym_clamped.item() if torch.is_tensor(L_sym_clamped) else float(L_sym_clamped),
            "free_energy": free_energy.item(),
            "total_bits": total_bits.item(),
            "bits_per_token": bits_per_token.item(),
            "fe_obs":     (observation_bits / float(valid_tokens)).item(),
            "fe_complex": (
                (local_complexity_bits / float(valid_tokens))
                + (global_model_bits / float(mdl_seen_tokens))
            ).item(),
            "fe_aux":     auxiliary_energy.item(),
            "fe_reasoning": L_reason.item(),
            "ltm_pen":    (rule_bits / float(mdl_seen_tokens)).item(),
            "ltm_pen_raw": float(ltm_penalty),   # для діагностики нескліпованого значення
            "curiosity":  curiosity_l.item(),
            "net_loss":   net_loss_scalar,
            "vem_pen":    L_vem.item() if torch.is_tensor(L_vem) else float(L_vem),
            "meta_loss":  L_meta.item() if torch.is_tensor(L_meta) else float(L_meta),
            "program_anchor": program_anchor_t.item(),
            "program_anchor_w": program_anchor_w,
            "program_decoder_ce": program_decoder_nats.item(),
            "program_decoder_bits": program_decoder_bits.item(),
            "program_decoder_w": program_decoder_w,
            "traj_reward": traj_reward if traj_reward is not None else 0.0,
            "reasoning_cost": L_reason.item(),
            "aux_phase":  aux_phase,
            "world_w":    world_w,
            "sym_w":      sym_w,
            "net_w":      net_w,
            "meta_w":     meta_w,
        }



# ══════════════════════════════════════════════════════════════════════════════
# 4b.  SYMBOLIC FACT CACHE  (офлайн-кеш символьних фактів)
# ══════════════════════════════════════════════════════════════════════════════

class SymbolicFactCache:
    """
    LRU-кеш для попередньо витягнутих символьних фактів.

    Відповідає пункту 1 «Ідеальної реалізації»:
      «Парсинг виконується один раз при підготовці даних, не замедляє навчання.»

    Ключ: SHA-1 хеш від байтів вхідної послідовності.
    Значення: Tuple[List[HornAtom], List[HornClause], object, Optional[str]]
      (facts + rule templates + optional execution-trace bundle + detected AST language)

    Розмір кешу обмежений max_entries (LRU eviction через OrderedDict).
    Thread-safe для читання (запис відбувається лише в одному потоці під GIL).
    """

    def __init__(self, max_entries: int = 4096):
        import collections
        self._cache: "collections.OrderedDict" = collections.OrderedDict()
        self._max = max_entries
        self._hits = 0
        self._misses = 0

    def _key(self, src_row: torch.Tensor) -> str:
        import hashlib
        raw = bytes(int(v) % 256 for v in src_row.tolist()).rstrip(b"\x00")
        return hashlib.sha1(raw).hexdigest()

    @staticmethod
    def _normalize_entry(entry):
        if entry is None:
            return None
        if len(entry) == 2:
            facts, rules = entry
            return facts, rules, None, None
        if len(entry) == 3:
            facts, rules, trace_bundle = entry
            return facts, rules, trace_bundle, None
        return entry

    def get(self, src_row: torch.Tensor):
        """Повертає (facts, rules, trace_bundle, detected_lang) або None при cache miss."""
        k = self._key(src_row)
        if k in self._cache:
            self._cache.move_to_end(k)
            self._hits += 1
            return self._normalize_entry(self._cache[k])
        self._misses += 1
        return None

    def put(self, src_row: torch.Tensor, facts, rules, trace_bundle=None, detected_lang: Optional[str] = None) -> None:
        k = self._key(src_row)
        self._cache[k] = (facts, rules, trace_bundle, detected_lang)
        self._cache.move_to_end(k)
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4c.  SYMBOLIC QUERY GENERATOR  (пункт 3 ідеальної реалізації)
# ══════════════════════════════════════════════════════════════════════════════

class SymbolicQueryGenerator(nn.Module):
    """
    Генератор логічних запитів до S-Core з прихованого стану декодера.

    Відповідає пункту 3 «Ідеальної реалізації»:
      «Декодер формує логічний запит до S-Core.
       S-Core пробує довести ціль; результат доведення модифікує
       логіти декодера — підвищує ймовірності відповідних токенів.»

    Архітектура:
      h_decoder (B, T, d_tok)
        ↓  [last token hidden]
      h_last (B, d_tok)
        ↓  query_proj → (B, d_latent)  [нейронне кодування запиту]
        ↓  pred_head  → (B, sym_vocab) [розподіл по предикатах]
        ↓  arg_head   → (B, 2)         [два аргументи]
      HornAtom(pred=argmax(pred_logits), args=(arg0, arg1))
        ↓  prover.prove(goal) → z_proof (B, d_latent)
        ↓  logit_bias_proj → (B, vocab_size) [зміщення логітів]
      logits_final = logits + α_sym · logit_bias

    Навчання:
      · Якщо доведення успішне → reward через VeM.reinforce_recent_rules
      · Якщо ні → абдукція при великому world_error
    """

    def __init__(self, d_tok: int, d_latent: int, sym_vocab: int,
                 vocab_size: int, alpha_sym: float = 0.1,
                 gumbel_tau: float = 0.85,
                 entropy_beta: float = 1e-3,
                 hard_mask_threshold: float = 0.75):
        super().__init__()
        self.d_latent  = d_latent
        self.sym_vocab = sym_vocab
        self.alpha_sym = alpha_sym
        self.gumbel_tau = gumbel_tau
        self.entropy_beta = entropy_beta
        self.hard_mask_threshold = hard_mask_threshold

        pred_candidates: List[int] = []
        seen_preds: Set[int] = set()
        for pred_id in (
            SEQ_PREDICT_NEXT_PRED,
            SEQ_ACTUAL_NEXT_PRED,
            SEQ_EDGE_PRED,
            SEQ_LAST_TOKEN_PRED,
            SEQ_AST_SUPPORT_PRED,
            SEQ_SALIENCY_SUPPORT_PRED,
            SEQ_GAP_DIM_PRED,
            *range(sym_vocab),
        ):
            pred_int = int(pred_id)
            if pred_int in seen_preds:
                continue
            seen_preds.add(pred_int)
            pred_candidates.append(pred_int)
        self.pred_candidates: Tuple[int, ...] = tuple(pred_candidates)
        self.pred_to_index: Dict[int, int] = {
            pred_id: idx for idx, pred_id in enumerate(self.pred_candidates)
        }
        self.default_query_preds: Tuple[int, ...] = tuple(
            pred_id for pred_id in (
                SEQ_PREDICT_NEXT_PRED,
                SEQ_ACTUAL_NEXT_PRED,
                SEQ_EDGE_PRED,
                SEQ_DECODER_GUESS_PRED,
                SEQ_DECODER_MISS_PRED,
                SEQ_AST_SUPPORT_PRED,
                SEQ_SALIENCY_SUPPORT_PRED,
            )
            if pred_id in self.pred_to_index
        )

        # h_decoder → нейронний запит
        self.query_proj = nn.Sequential(
            nn.Linear(d_tok, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        # Вибір предиката
        self.pred_head = nn.Linear(d_latent, len(self.pred_candidates))
        # Вибір аргументів (2 слоти)
        self.arg_head  = nn.Linear(d_latent, 2 * sym_vocab)
        self.sym_query_emb = nn.Embedding(len(self.pred_candidates), d_latent)
        self.arg_query_emb = nn.Embedding(sym_vocab, d_latent)

        # z_proof → зміщення логітів декодера
        self.logit_bias_proj = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, vocab_size),
        )
        self.query_bias_proj = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, vocab_size),
        )
        # Ворота: скільки довіряти символьному результату
        self.proof_gate = nn.Sequential(
            nn.Linear(d_latent * 2, 1),
            nn.Sigmoid(),
        )
        self.last_query_info: Dict[str, object] = {}

    def _build_query_state(
        self,
        h_last: Optional[torch.Tensor],
        symbolic_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        if h_last is not None:
            parts.append(self.query_proj(h_last))
        if symbolic_state is not None:
            if symbolic_state.dim() == 1:
                symbolic_state = symbolic_state.unsqueeze(0)
            parts.append(self.context_proj(symbolic_state))
        if not parts:
            raise ValueError("SymbolicQueryGenerator needs h_last or symbolic_state")
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total / float(len(parts))

    def _mask_candidate_preds(
        self,
        pred_logits: torch.Tensor,
        candidate_preds: Optional[Tuple[int, ...]],
    ) -> torch.Tensor:
        candidate_preds = self._normalize_candidate_preds(candidate_preds)
        if not candidate_preds:
            return pred_logits
        allowed = {
            self.pred_to_index[int(pred)]
            for pred in candidate_preds
            if int(pred) in self.pred_to_index
        }
        if not allowed:
            return pred_logits
        masked = torch.full_like(pred_logits, -1e4)
        for idx in allowed:
            masked[:, idx] = pred_logits[:, idx]
        return masked

    def _normalize_candidate_preds(
        self,
        candidate_preds: Optional[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        if candidate_preds:
            normalized = tuple(
                int(pred) for pred in candidate_preds
                if int(pred) in self.pred_to_index
            )
            if normalized:
                return normalized
        return self.default_query_preds or self.pred_candidates

    def generate_query(self,
                       h_last: Optional[torch.Tensor],
                       sym_vocab: int,
                       context_anchor: Optional[int] = None,
                       symbolic_state: Optional[torch.Tensor] = None,
                       candidate_preds: Optional[Tuple[int, ...]] = None) -> "HornAtom":
        """
        h_last: (1, d_tok) — прихований стан останнього токена
        Повертає HornAtom — логічний запит до S-Core.

        Пункт 6 ідеальної реалізації:
          «generate_query повинен бути обучаємим і вариативним.
           Він повинен повертати Horn-атом з предикатом, вибраним з sym_vocab,
           а не фіксований SEQ_PREDICT_NEXT_PRED.»

        Використовуємо нейронні передбачення pred_id, arg0 замість хардкоду.
        arg1 залишається Var("NEXT") — ми завжди запитуємо «що наступне?»
        """
        z_q = self._build_query_state(h_last, symbolic_state)
        pred_logits = self._mask_candidate_preds(
            self.pred_head(z_q),
            candidate_preds,
        )
        pred_idx = int(pred_logits.argmax(-1).item()) % max(len(self.pred_candidates), 1)
        pred_id = self.pred_candidates[pred_idx]

        arg_logits = self.arg_head(z_q).view(1, 2, sym_vocab)  # (1,2,sv)
        arg0 = (
            int(context_anchor)
            if context_anchor is not None
            else int(arg_logits[:, 0].argmax(-1).item()) % max(sym_vocab, 1)
        )
        # arg1 завжди Var("NEXT"): ми шукаємо значення наступного токена
        # pred_id вибирається нейронно — це є ключова відмінність від v1
        return HornAtom(pred_id, (arg0, Var("NEXT")))

    def forward(self,
                logits:   torch.Tensor,
                h_tok:    torch.Tensor,
                z_sym:    torch.Tensor,
                prover:   "DifferentiableProver") -> torch.Tensor:
        """
        Модифікує логіти декодера символьним результатом доведення.

        logits  : (B, T, vocab_size) — оригінальні логіти
        h_tok   : (B, T, d_tok)      — приховані стани декодера
        z_sym   : (B, d_latent)      — символьний вектор від prover
        prover  : DifferentiableProver

        Returns: logits + α_sym · gate · logit_bias  (той самий shape)
        """
        B, T, V = logits.shape
        # Беремо прихований стан останнього токена
        h_last = h_tok[:, -1, :]                     # (B, d_tok)
        task_context = getattr(prover, "task_context", None)
        context_anchor = None
        gold_next = None
        candidate_preds: Tuple[int, ...] = self.default_query_preds
        context_state: Optional[torch.Tensor] = z_sym
        if task_context is not None:
            context_anchor = int(task_context.metadata.get("last_src", 0))
            gold_next = int(task_context.metadata.get("last_tgt", -1))
            observed_facts = tuple(task_context.observed_facts)
            if observed_facts:
                pred_hints = {int(fact.pred) for fact in observed_facts if fact.arity() >= 2}
                if task_context.goal is not None and task_context.goal.arity() >= 2:
                    pred_hints.add(int(task_context.goal.pred))
                pred_hints.add(SEQ_PREDICT_NEXT_PRED)
                candidate_preds = tuple(sorted(pred_hints))
                context_state = prover.ground(
                    task_context.observed_facts,
                    logits.device,
                ).expand(B, -1)
        elif getattr(prover, "last_goal", None) is not None and getattr(prover.last_goal, "args", ()):
            anchor_term = prover.last_goal.args[0]
            if isinstance(anchor_term, Const):
                context_anchor = int(anchor_term.val)
            elif isinstance(anchor_term, int):
                context_anchor = int(anchor_term)
        z_q = self._build_query_state(h_last, context_state)
        raw_pred_logits = self.pred_head(z_q)
        pred_logits = self._mask_candidate_preds(
            raw_pred_logits,
            candidate_preds,
        )
        arg_logits = self.arg_head(z_q).view(B, 2, self.sym_vocab)
        if self.training:
            pred_probs = F.gumbel_softmax(
                pred_logits, tau=self.gumbel_tau, hard=False, dim=-1
            )
            arg_probs = F.gumbel_softmax(
                arg_logits, tau=self.gumbel_tau, hard=False, dim=-1
            )
        else:
            pred_probs = F.one_hot(
                pred_logits.argmax(dim=-1),
                num_classes=len(self.pred_candidates),
            ).to(dtype=logits.dtype)
            arg_probs = F.one_hot(
                arg_logits.argmax(dim=-1),
                num_classes=self.sym_vocab,
            ).to(dtype=logits.dtype)
        pred_ctx = pred_probs @ self.sym_query_emb.weight
        arg0_ctx = arg_probs[:, 0, :] @ self.arg_query_emb.weight
        arg1_ctx = arg_probs[:, 1, :] @ self.arg_query_emb.weight
        query_ctx = pred_ctx + 0.5 * (arg0_ctx + arg1_ctx)

        # Обчислюємо зміщення логітів на основі z_sym
        query_goal = self.generate_query(
            h_last[:1],
            self.sym_vocab,
            context_anchor=context_anchor,
            symbolic_state=None if context_state is None else context_state[:1],
            candidate_preds=candidate_preds,
        )
        z_query_proof, answer_ids, proof_support = prover.answer_query(
            query_goal,
            logits.device,
        )
        used_fallback = False
        if proof_support.item() <= 0.0 and context_anchor is not None:
            fallback_goal = HornAtom(SEQ_PREDICT_NEXT_PRED, (context_anchor, Var("NEXT")))
            fallback_state, fallback_answers, fallback_support = prover.answer_query(
                fallback_goal,
                logits.device,
            )
            if fallback_support.item() > proof_support.item():
                query_goal = fallback_goal
                z_query_proof = fallback_state
                answer_ids = fallback_answers
                proof_support = fallback_support
                used_fallback = True
        z_query_proof = z_query_proof.expand(B, -1)
        proof_support_exp = proof_support.to(logits.dtype).view(1, 1).expand(B, 1)
        valid_answer_ids = [token_id for token_id in answer_ids if 0 <= token_id < V]

        pred_entropy = -(
            pred_probs[:1] * torch.log(pred_probs[:1].clamp_min(1e-8))
        ).sum(dim=-1).mean()
        query_hit = 1.0 if gold_next is not None and gold_next in valid_answer_ids else 0.0
        query_reward = 0.5 * float(proof_support.item()) + 0.5 * query_hit
        answer_bias = torch.zeros(B, V, device=logits.device, dtype=logits.dtype)
        if valid_answer_ids and (not self.training or query_hit > 0.0):
            answer_boost = (0.5 + 0.5 * float(proof_support.item())) / max(self.alpha_sym, 1e-3)
            bias_step = answer_boost / float(len(valid_answer_ids))
            for token_id in valid_answer_ids:
                answer_bias[:, token_id] += bias_step

        logit_bias = (
            self.logit_bias_proj(z_sym + z_query_proof)
            + self.query_bias_proj(torch.cat([z_q, query_ctx + z_query_proof], dim=-1))
            + answer_bias
        )      # (B, V)

        # Адаптивна ворота: наскільки довіряємо символьному результату
        gate = self.proof_gate(
            torch.cat([z_q + query_ctx, z_sym + z_query_proof], dim=-1)   # (B, 2*d_lat)
        ) * (0.25 + 0.75 * proof_support_exp)                              # (B, 1)

        query_aux_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        target_pred_idx = self.pred_to_index.get(int(query_goal.pred))
        if target_pred_idx is not None:
            target_pred = torch.tensor([target_pred_idx], device=logits.device)
            reward_scale = 0.25 + 0.75 * query_reward
            target_allowed = (
                not candidate_preds
                or int(query_goal.pred) in set(candidate_preds)
            )
            pred_supervision_logits = pred_logits[:1] if target_allowed else raw_pred_logits[:1]
            query_aux_loss = reward_scale * F.cross_entropy(pred_supervision_logits, target_pred)
            if context_anchor is not None and 0 <= context_anchor < self.sym_vocab:
                arg_target = torch.tensor([int(context_anchor)], device=logits.device)
                query_aux_loss = query_aux_loss + 0.25 * reward_scale * F.cross_entropy(
                    arg_logits[:1, 0, :],
                    arg_target,
                )
            query_aux_loss = query_aux_loss - self.entropy_beta * pred_entropy

        self.last_query_info = {
            "pred": int(query_goal.pred),
            "candidate_count": float(len(candidate_preds)),
            "support": float(proof_support.item()),
            "n_answers": float(len(valid_answer_ids)),
            "answer_ids": tuple(int(token_id) for token_id in valid_answer_ids[:4]),
            "fallback": 1.0 if used_fallback else 0.0,
            "hit": query_hit,
            "reward": query_reward,
            "entropy": float(pred_entropy.detach().item()),
            "aux_loss": float(query_aux_loss.detach().item()),
            "aux_loss_tensor": query_aux_loss,
        }

        # This query reasons about the current next-token step only.
        logits_out = logits.clone()
        logits_out[:, -1, :] = logits_out[:, -1, :] + self.alpha_sym * gate * logit_bias
        if (
            valid_answer_ids
            and float(proof_support.item()) >= float(self.hard_mask_threshold)
            and (not self.training or query_hit > 0.0)
        ):
            veto_mask = torch.full_like(logits_out[:, -1, :], -1e4)
            veto_mask[:, valid_answer_ids] = 0.0
            logits_out[:, -1, :] = logits_out[:, -1, :] + veto_mask
            self.last_query_info["hard_mask"] = 1.0
        else:
            self.last_query_info["hard_mask"] = 0.0
        return logits_out


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
        self.posterior_mu = nn.Linear(cfg.d_latent, cfg.d_latent)
        self.posterior_logvar = nn.Linear(cfg.d_latent, cfg.d_latent)
        nn.init.zeros_(self.posterior_mu.weight)
        nn.init.zeros_(self.posterior_mu.bias)
        nn.init.zeros_(self.posterior_logvar.weight)
        nn.init.constant_(self.posterior_logvar.bias, -4.0)
        self.memory_prior_mu = nn.Linear(cfg.d_latent, cfg.d_latent)
        self.memory_prior_logvar = nn.Linear(cfg.d_latent, cfg.d_latent)
        self.memory_read_mu = nn.Linear(cfg.d_latent, cfg.d_latent)
        self.memory_read_logvar = nn.Linear(cfg.d_latent, cfg.d_latent)
        self.symbolic_prior_mu = nn.Linear(cfg.d_latent, cfg.d_latent)
        self.symbolic_prior_logvar = nn.Linear(cfg.d_latent, cfg.d_latent)
        for layer in (
            self.memory_prior_mu,
            self.memory_prior_logvar,
            self.memory_read_mu,
            self.memory_read_logvar,
            self.symbolic_prior_mu,
            self.symbolic_prior_logvar,
        ):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.memory_prior_mu.bias)
        nn.init.constant_(self.memory_prior_logvar.bias, -2.0)
        nn.init.zeros_(self.memory_read_mu.bias)
        nn.init.zeros_(self.memory_read_logvar.bias)
        nn.init.zeros_(self.symbolic_prior_mu.bias)
        nn.init.constant_(self.symbolic_prior_logvar.bias, -2.0)
        self.token_code_mu = nn.Parameter(torch.zeros(cfg.d_tok))
        self.token_code_logvar = nn.Parameter(torch.zeros(cfg.d_tok))
        self.concept_code_mu = nn.Parameter(torch.zeros(cfg.d_latent))
        self.concept_code_logvar = nn.Parameter(torch.zeros(cfg.d_latent))
        self.world_obs_logvar = nn.Parameter(torch.zeros(()))

        # ─── Рівень 2: Concept ────────────────────────────────────────────────
        core_cfg = _make_core_compat(cfg)
        self.world_rnn = WorldRNN(core_cfg)
        self.memory    = AsyncTensorProductMemory(cfg)

        self.epistemic = EpistemicGapDetector(core_cfg)
        self.curiosity = CuriosityModule(core_cfg)
        self.world_graph_enabled = bool(getattr(cfg, "world_graph_enabled", True))
        self.world_graph = WorldGraphEncoder(
            d_latent=cfg.d_latent,
            pred_buckets=int(getattr(cfg, "world_graph_pred_buckets", 4096)),
            term_buckets=int(getattr(cfg, "world_graph_term_buckets", 8192)),
            max_nodes=int(getattr(cfg, "world_graph_max_nodes", 128)),
            max_edges=int(getattr(cfg, "world_graph_max_edges", 512)),
            n_layers=int(getattr(cfg, "world_graph_layers", 2)),
            max_transitions=int(getattr(cfg, "world_graph_max_transitions", 16)),
        )
        self.world_graph_gate = nn.Sequential(
            nn.Linear(cfg.d_latent * 2, cfg.d_latent),
            nn.GELU(),
            nn.Linear(cfg.d_latent, cfg.d_latent),
            nn.Sigmoid(),
        )
        self.world_target_proj = nn.Sequential(
            nn.Linear(cfg.d_tok, cfg.d_latent),
            nn.GELU(),
            nn.Linear(cfg.d_latent, cfg.d_latent),
        )
        self.world_state_prior = nn.Parameter(torch.zeros(cfg.d_latent))
        self.mem_query_proj = nn.Linear(cfg.d_latent, cfg.d_latent, bias=False)
        self.state_integrator = SymbolicStateIntegrator(cfg.d_latent)
        self.symbolic_token_head = nn.Sequential(
            nn.Linear(cfg.d_latent * 2, cfg.d_latent),
            nn.GELU(),
            nn.Linear(cfg.d_latent, cfg.vocab_size),
        )
        self.decoder_surprise_head = nn.Sequential(
            nn.Linear(cfg.d_tok + cfg.d_latent, cfg.d_tok),
            nn.GELU(),
            nn.Linear(cfg.d_tok, cfg.vocab_size),
        )
        self._decoder_surprise_enabled = bool(
            getattr(cfg, "sym_decoder_surprise_enabled", True)
        )

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
        self._install_symbolic_bootstrap_rules()

        # ─── КРИТИЧНО: WorldRNN → ∂-Prolog ін'єкція ──────────────────────────
        # set_world_rnn() увімкнює:
        #   · Дедукція: _mental_simulate_rule() використовує WorldRNN для
        #     латентного Prediction Error ДО реального застосування правила
        #   · Абдукція: _pred_error_for_rule() отримує WorldRNN-компоненту
        #     в MDL PredError(R, Trace) — 30% ваги latent-space consistency
        # БЕЗ цього виклику self._world_rnn = None у всіх методах прувера
        # → Дедукція та Абдукція працювали суто символьно, без latent сигналу.
        self.prover.set_world_rnn(self.world_rnn)
        self.prover.set_allow_latent_goal_fallback(False)
        self.prover.configure_hypothesis_cycle(
            enabled=getattr(cfg, "continuous_cycle_enabled", True),
            eval_enabled=getattr(cfg, "continuous_cycle_eval_enabled", True),
            eval_learning_enabled=getattr(cfg, "continuous_cycle_eval_learning_enabled", True),
            max_contextual=getattr(cfg, "continuous_cycle_contextual", 4),
            max_neural=getattr(cfg, "continuous_cycle_neural", 4),
            accept_threshold=getattr(cfg, "continuous_cycle_accept_threshold", 0.55),
            verify_threshold=getattr(cfg, "continuous_cycle_verify_threshold", 0.75),
            contradict_threshold=getattr(cfg, "continuous_cycle_contradict_threshold", 0.15),
            symbolic_weight=getattr(cfg, "continuous_cycle_symbolic_weight", 0.45),
            world_weight=getattr(cfg, "continuous_cycle_world_weight", 0.25),
            token_weight=getattr(cfg, "continuous_cycle_token_weight", 0.30),
            soft_symbolic_weight=getattr(cfg, "continuous_cycle_soft_symbolic_weight", 0.45),
            policy_weight=getattr(cfg, "continuous_cycle_policy_weight", 0.25),
            policy_baseline_momentum=getattr(cfg, "continuous_cycle_policy_baseline_momentum", 0.90),
            candidate_tau=getattr(cfg, "continuous_cycle_candidate_tau", 0.70),
            repair_enabled=getattr(cfg, "continuous_cycle_repair_enabled", True),
            repair_threshold=getattr(cfg, "continuous_cycle_repair_threshold", 0.35),
            max_repairs=getattr(cfg, "continuous_cycle_max_repairs", 2),
        )
        self.prover.configure_graph_reasoning(
            enabled=getattr(cfg, "sym_graph_reasoning_enabled", True),
            top_k_facts=getattr(cfg, "sym_graph_reasoning_top_k_facts", 12),
            max_fact_subset=getattr(cfg, "sym_graph_reasoning_max_fact_subset", 96),
            attention_threshold=getattr(cfg, "sym_graph_reasoning_attention_threshold", 0.02),
            tau=getattr(cfg, "sym_graph_reasoning_tau", 0.5),
            full_scan_cutoff=getattr(cfg, "sym_graph_reasoning_full_scan_cutoff", 64),
        )

        # ─── KB ↔ NET інтеграція: NET реєструє концепти прямо в Prolog-KB ──────
        # Раніше KB була відключена від NET → абдукція не бачила токен-концептів.
        self.prover.configure_creative_cycle(
            enabled=getattr(cfg, "creative_cycle_enabled", True),
            cycle_every=getattr(cfg, "creative_cycle_every", 4),
            max_selected_rules=getattr(cfg, "creative_max_selected_rules", 2),
            analogy_dim=getattr(cfg, "ame_embedding_dim", 16),
            tau_analogy=getattr(cfg, "ame_tau_analogy", 0.82),
            analogy_hidden_dim=getattr(cfg, "ame_hidden_dim", 64),
            analogy_gnn_layers=getattr(cfg, "ame_gnn_layers", 2),
            analogy_spec_ratio=getattr(cfg, "ame_spec_ratio", 0.5),
            analogy_temperature=getattr(cfg, "ame_temperature", 0.07),
            analogy_contrastive_steps=getattr(cfg, "ame_contrastive_steps", 2),
            analogy_contrastive_lr=getattr(cfg, "ame_contrastive_lr", 3e-3),
            analogy_dropout=getattr(cfg, "ame_dropout", 0.10),
            cwe_max_rule_mods=getattr(cfg, "cwe_max_rule_mods", 2),
            cwe_surprise_lambda=getattr(cfg, "cwe_surprise_lambda", 0.5),
            cwe_max_candidates=getattr(cfg, "cwe_max_candidates", 8),
            cwe_max_transforms_per_rule=getattr(cfg, "cwe_max_transforms_per_rule", 4),
            aee_population=getattr(cfg, "aee_population", 16),
            aee_generations=getattr(cfg, "aee_generations", 2),
            aee_gamma=getattr(cfg, "aee_gamma", 0.25),
            aee_mutation_rate=getattr(cfg, "aee_mutation_rate", 0.35),
            aee_crossover_rate=getattr(cfg, "aee_crossover_rate", 0.5),
            aee_ltm_seed_ratio=getattr(cfg, "aee_ltm_seed_ratio", 0.35),
            aee_gene_pool_size=getattr(cfg, "aee_gene_pool_size", 32),
            oee_gap_threshold=getattr(cfg, "oee_gap_threshold", 0.45),
            oee_contradiction_threshold=getattr(cfg, "oee_contradiction_threshold", 1),
            oee_d_latent=getattr(cfg, "oee_d_latent", 32),
            oee_consistency_lambda=getattr(cfg, "oee_consistency_lambda", 0.1),
            oee_online_lr=getattr(cfg, "oee_online_lr", 1e-3),
            oee_forward_chain_depth=getattr(cfg, "oee_forward_chain_depth", 2),
            oee_max_interaction_preds=getattr(cfg, "oee_max_interaction_preds", 3),
            oee_max_hypotheses=getattr(cfg, "oee_max_hypotheses", 8),
            ice_state_history=getattr(cfg, "ice_state_history", 128),
            ice_goal_threshold=getattr(cfg, "ice_goal_threshold", 0.35),
        )
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
        self.saliency_enabled = getattr(cfg, 'saliency_enabled', False)
        if self.saliency_enabled:
            self.saliency = SaliencyTraceModule(cfg.d_tok, cfg.d_latent, cfg)

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
        self._ctx_max_facts = int(getattr(cfg, "symbolic_context_max_facts", 96))
        self._ctx_ast_max_facts = int(getattr(cfg, "symbolic_ast_max_facts", 48))
        self.ast_parser = MultiLangASTParser()
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("_seen_tokens", torch.zeros((), dtype=torch.long), persistent=False)

        # ── Ідеальна реалізація: пункти 1 та 3 ───────────────────────────────
        # SymbolicFactCache: офлайн-кеш фактів щоб уникнути повторного AST-парсингу
        self._fact_cache = SymbolicFactCache(
            max_entries=int(getattr(cfg, "fact_cache_size", 4096))
        )
        # SymbolicQueryGenerator: генерує Horn-цілі з прихованого стану декодера
        # і коригує логіти декодера відповідно до результату доведення (пункт 3)
        sym_qg_enabled = getattr(cfg, "sym_query_gen_enabled", True)
        if sym_qg_enabled:
            self.sym_query_gen = SymbolicQueryGenerator(
                d_tok      = cfg.d_tok,
                d_latent   = cfg.d_latent,
                sym_vocab  = cfg.sym_vocab,
                vocab_size = cfg.vocab_size,
                alpha_sym  = float(getattr(cfg, "sym_query_alpha", 0.05)),
                gumbel_tau = float(getattr(cfg, "sym_query_gumbel_tau", 0.85)),
                entropy_beta = float(getattr(cfg, "sym_query_entropy_beta", 1e-3)),
                hard_mask_threshold = float(getattr(cfg, "sym_query_hard_mask_threshold", 0.75)),
            )
        else:
            self.sym_query_gen = None
        self._sym_qg_enabled = sym_qg_enabled
        self._last_ce_utility: float = 0.0
        self.ce_reinforce_enabled: bool = bool(getattr(cfg, "ce_reinforce_enabled", False))
        self.ce_reinforce_eval_enabled: bool = bool(
            getattr(cfg, "ce_reinforce_eval_enabled", True)
        )
        self.ce_reinforce_fallback_only: bool = bool(getattr(cfg, "ce_reinforce_fallback_only", True))
        self.ce_reinforce_retro_every: int = int(getattr(cfg, "ce_reinforce_retro_every", 0))
        # Running CE EMA для reward feedback у S-Core (пункт 1/5 ідеальної реалізації)
        # _prev_ce: EMA попереднього CE — використовується для визначення
        #   чи правила, що були абдуковані, допомогли покращити мовну модель.
        self._prev_ce: float = float("inf")
        self._ce_ema: float = float("inf")   # окремий smooth EMA для VeM retro
        self.last_generate_info: Dict[str, float] = {}

    @staticmethod
    def canonical_architecture() -> CanonicalArchitectureSpec:
        return CANONICAL_OMEN_SPEC

    # ── CE-Driven Abduce→Deduce→Induce: замикання зворотного зв'язку ─────────
    def _install_symbolic_bootstrap_rules(self) -> None:
        x_var = Var("BOOT_X")
        y_var = Var("BOOT_Y")
        bootstrap_rules = [
            HornClause(
                head=HornAtom(SEQ_PREDICT_NEXT_PRED, (x_var, y_var)),
                body=(HornAtom(SEQ_EDGE_PRED, (x_var, y_var)),),
            ),
            HornClause(
                head=HornAtom(SEQ_ACTUAL_NEXT_PRED, (x_var, y_var)),
                body=(HornAtom(SEQ_EDGE_PRED, (x_var, y_var)),),
            ),
        ]
        for rule in bootstrap_rules:
            try:
                self.prover.kb.add_rule(rule, status=EpistemicStatus.verified)
            except Exception:
                continue

    def _per_rule_induction(
        self,
        rules: list,
        target_facts,
        all_facts,
        global_ce_utility: float,
        device: torch.device,
    ) -> None:
        """
        Локальна (per-rule) Індукція (концепція, розділ 3):
          «Перевіряємо конкретне правило на конкретному прикладі:
           якщо передбачення не збіглося → правило відкидається або модифікується»

        Замість глобального CE-сигналу для всіх правил → перевіряємо кожне
        правило окремо: чи факти, виведені ЦИМ правилом, потрапили у target_facts.

        Алгоритм:
          1. Для кожного правила → forward_chain_step_local → derived_facts
          2. hits = |derived_facts ∩ target_facts|
          3. rule_utility = (0.7 * hits_score + 0.3 * global_ce_utility)
             де hits_score = 1.0 якщо hits > 0, інакше 0.0
          4. Якщо rule_utility < 0.2 → mark_contradicted

        Це відрізняється від глобального _ce_reinforce:
          СТАРО: utility(ALL rules) = f(CE_global)
          НОВО:  utility(R_i) = f(CE_global, hits(R_i, target_facts))
        """
        if not rules or not target_facts:
            return

        from omen_prolog import (
            freshen_vars, find_all_substitutions, unify, _atoms_conflict
        )

        for rule in rules:
            if not rule.body:
                # Правило-факт: перевіряємо голову проти target_facts
                hits = sum(
                    1 for tf in target_facts
                    if unify(rule.head, tf) is not None
                )
                hits_score = 1.0 if hits > 0 else 0.0
                rule_utility = 0.7 * hits_score + 0.3 * global_ce_utility
            else:
                # Правило з тілом: перевіряємо виведені факти проти target_facts
                try:
                    fresh = freshen_vars(rule)
                    base_facts = all_facts or self.prover.kb.facts
                    derived_by_rule = set()
                    for sigma in find_all_substitutions(
                        fresh.body, base_facts, max_solutions=8
                    ):
                        derived = sigma.apply_atom(fresh.head)
                        if derived.is_ground():
                            derived_by_rule.add(derived)

                    # Скільки виведених фактів потрапили у target_facts
                    hits = sum(
                        1 for df in derived_by_rule
                        for tf in target_facts
                        if unify(df, tf) is not None
                    )
                    hits_score = min(1.0, float(hits) / max(len(derived_by_rule), 1))
                    # Якщо правило нічого не вивело взагалі — нейтральний сигнал
                    if not derived_by_rule:
                        hits_score = 0.3  # не корисне, але не шкідливе
                    rule_utility = 0.7 * hits_score + 0.3 * global_ce_utility
                except Exception:
                    rule_utility = global_ce_utility

            # Оновлюємо VeM та статус правила на основі ЛОКАЛЬНОЇ перевірки
            self.prover.vem.record_outcome(
                rule, utility_target=rule_utility, device=device
            )
            if hasattr(self.prover, "_record_rule_utility"):
                self.prover._record_rule_utility(rule, rule_utility)

            # Епістемічний статус: верифікується тільки якщо правило РЕАЛЬНО
            # допомогло (висока локальна utility), а не через глобальний CE
            if rule_utility >= 0.85:
                self.prover.kb.mark_rule_verified(rule)
            elif rule_utility <= 0.15:
                self.prover.kb.mark_rule_contradicted(rule)

    def _ce_reinforce(self, cur_ce: float, device: torch.device) -> None:
        """
        Замикає Abduce→Deduce→Induce цикл.

        Інтегрує два рівні зворотного зв'язку:
          1. ГЛОБАЛЬНИЙ (через CE delta) — для загального напрямку навчання
          2. ЛОКАЛЬНИЙ (через per-rule target_facts verification) — для точної
             верифікації кожного правила окремо (Fix 3: реальна Індукція)

        Концепція, розділ 3:
          «Локальна перевірка конкретного правила на конкретному прикладі:
           якщо передбачення не збіглося → правило відкидається, а не підганяється»
        """
        if math.isnan(cur_ce) or math.isinf(cur_ce):
            return
        if not self.ce_reinforce_enabled:
            self._last_ce_utility = 0.0
            if self._prev_ce < float("inf"):
                self._prev_ce = 0.9 * self._prev_ce + 0.1 * cur_ce
                self._ce_ema = 0.95 * self._ce_ema + 0.05 * cur_ce
            else:
                self._prev_ce = cur_ce
                self._ce_ema = cur_ce
            return

        # ── Глобальний сигнал (CE delta) ──────────────────────────────────
        if self._prev_ce < float("inf"):
            ce_delta = self._prev_ce - cur_ce   # > 0 = покращення
            global_utility = float(torch.sigmoid(torch.tensor(ce_delta * 3.0)).item())
            if ce_delta > 0.05:
                global_utility = max(global_utility, 0.9)
            elif ce_delta < -0.05:
                global_utility = min(global_utility, 0.1)
        else:
            global_utility = 0.5

        self._last_ce_utility = global_utility

        # ── Локальна per-rule Індукція (FIX 3) ────────────────────────────
        # Перевіряємо кожне нещодавно абдуковане та використане правило ЛОКАЛЬНО
        # проти target_facts прувера (не через глобальний CE)
        target_facts = self.prover._task_target_facts()
        all_facts = self.prover.last_all_facts

        recent_rules = list(self.prover.last_abduced_rules)
        used_rules   = list(self.prover.last_used_rules)
        if not target_facts and self.ce_reinforce_fallback_only:
            recent_rules = []
            used_rules = []
            self.prover.last_abduced_rules = []
            if hasattr(self.prover, "last_used_rules"):
                self.prover.last_used_rules = []
            if hasattr(self.prover, "_last_used_rule_hashes"):
                self.prover._last_used_rule_hashes.clear()

        if target_facts:
            # Є конкретні target_facts → per-rule verification
            if recent_rules:
                self._per_rule_induction(
                    recent_rules, target_facts, all_facts, global_utility, device
                )
            if used_rules:
                self._per_rule_induction(
                    used_rules, target_facts, all_facts, global_utility, device
                )
            # Очищаємо після per-rule перевірки
            self.prover.last_abduced_rules = []
            if hasattr(self.prover, "last_used_rules"):
                self.prover.last_used_rules = []
            if hasattr(self.prover, "_last_used_rule_hashes"):
                self.prover._last_used_rule_hashes.clear()
        else:
            # Немає конкретних target_facts → fallback до глобального CE сигналу
            # (зберігаємо стару поведінку для backward compatibility)
            self.prover.reinforce_recent_rules(
                utility_target=global_utility, device=device
            )
            if hasattr(self.prover, "reinforce_used_rules"):
                self.prover.reinforce_used_rules(
                    utility_target=global_utility, device=device
                )

        # ── VeM CE-based retrospective (рідко) ────────────────────────────
        step = int(self._train_step.item())
        if (
            self.ce_reinforce_retro_every > 0
            and step > 0
            and step % self.ce_reinforce_retro_every == 0
            and self._ce_ema < float("inf")
        ):
            self.prover.vem_retrospective_update(
                ce_utility=global_utility,
                device=device,
            )

        # ── Оновлюємо EMA CE ───────────────────────────────────────────────
        alpha = 0.1
        if self._prev_ce < float("inf"):
            self._prev_ce = (1.0 - alpha) * self._prev_ce + alpha * cur_ce
            self._ce_ema  = (1.0 - 0.05) * self._ce_ema  + 0.05  * cur_ce
        else:
            self._prev_ce = cur_ce
            self._ce_ema  = cur_ce

    def _world_teacher_forcing_ratio(self) -> float:
        start = float(getattr(self.cfg, 'world_teacher_forcing_start', 0.35))
        end = float(getattr(self.cfg, 'world_teacher_forcing_end', 0.05))
        steps = max(int(getattr(self.cfg, 'world_teacher_forcing_steps', 2000)), 1)
        progress = min(float(self._train_step.item()) / steps, 1.0)
        return start + (end - start) * progress

    def _retrieve_memory(self, z_query: torch.Tensor) -> torch.Tensor:
        q_mem = self.mem_query_proj(z_query)
        v_holo = self.memory.read(q_mem)
        v_epi = self.memory.episodic_recall(q_mem)
        sims = torch.stack([
            F.cosine_similarity(q_mem, v_holo, dim=-1),
            F.cosine_similarity(q_mem, v_epi, dim=-1),
        ], dim=-1)
        weights = F.softmax(sims, dim=-1)
        return weights[:, :1] * v_holo + weights[:, 1:] * v_epi

    def _build_world_graph_batch(
        self,
        src: torch.Tensor,
        saliency_out: Optional[object] = None,
    ) -> WorldGraphBatch:
        if not self.world_graph_enabled:
            zeros = torch.zeros(src.size(0), self.cfg.d_latent, device=src.device)
            return WorldGraphBatch(graphs=tuple(), pooled_states=zeros, metadata={"enabled": 0.0})

        graphs = []
        pooled_states = []
        total_nodes = 0.0
        total_edges = 0.0
        trace_graphs = 0.0
        max_pairs = max(int(getattr(self.cfg, "world_graph_max_nodes", 128)) // 2, 8)
        for batch_idx in range(src.size(0)):
            src_row = src[batch_idx].detach().cpu()
            pair_cap = min(max(src_row.numel() - 1, 0), max_pairs)
            start_idx = max(src_row.numel() - pair_cap - 1, 0)
            graph_facts: List[HornAtom] = []
            for pos in range(start_idx, max(src_row.numel() - 1, 0)):
                graph_facts.append(
                    HornAtom(
                        SEQ_EDGE_PRED,
                        (int(src_row[pos].item()), int(src_row[pos + 1].item())),
                    )
                )
            if src_row.numel() > 0:
                graph_facts.append(
                    HornAtom(
                        SEQ_LAST_TOKEN_PRED,
                        (int(src_row[-1].item()), max(int(src_row.numel()) - 1, 0)),
                    )
                )

            ast_facts = self._ast_facts_from_bytes(src_row)
            if ast_facts:
                graph_facts.extend(ast_facts)
            trace_bundle = self._ast_trace_from_bytes(src_row)
            saliency_facts: List[HornAtom] = []
            if saliency_out is not None:
                saliency_facts.extend(list(saliency_out.sal_semantic_facts[batch_idx]))
                saliency_facts.extend(list(saliency_out.sal_expected_facts[batch_idx]))
            graph_facts = self._dedupe_facts(
                graph_facts,
                limit=int(getattr(self.cfg, "world_graph_max_nodes", 128)),
            )
            saliency_facts = self._dedupe_facts(
                saliency_facts,
                limit=max(int(getattr(self.cfg, "world_graph_max_nodes", 128)) // 2, 8),
            )
            graph = self.world_graph(
                facts=graph_facts,
                trace_bundle=trace_bundle,
                saliency_facts=saliency_facts,
                device=src.device,
            )
            graphs.append(graph)
            pooled_states.append(graph.pooled_state)
            total_nodes += float(len(graph.node_keys))
            total_edges += float(len(graph.edges))
            trace_graphs += float(graph.transition_targets is not None and graph.transition_targets.size(0) > 0)

        metadata = {
            "enabled": 1.0,
            "mean_nodes": total_nodes / max(len(graphs), 1),
            "mean_edges": total_edges / max(len(graphs), 1),
            "trace_graphs": trace_graphs,
            "trace_supervised_steps": 0.0,
        }
        if pooled_states:
            pooled = torch.stack(pooled_states, dim=0)
        else:
            pooled = torch.zeros(src.size(0), self.cfg.d_latent, device=src.device)
        return WorldGraphBatch(graphs=tuple(graphs), pooled_states=pooled, metadata=metadata)

    def _ground_world_state(
        self,
        z_neural: torch.Tensor,
        world_graph_batch: Optional[WorldGraphBatch],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            world_graph_batch is None
            or world_graph_batch.pooled_states.numel() == 0
            or not self.world_graph_enabled
        ):
            zeros = torch.zeros_like(z_neural)
            ones = torch.zeros_like(z_neural)
            return z_neural, zeros, ones
        z_graph = world_graph_batch.pooled_states.to(device=z_neural.device, dtype=z_neural.dtype)
        gate = self.world_graph_gate(torch.cat([z_neural, z_graph], dim=-1))
        mix = float(getattr(self.cfg, "world_graph_state_mix", 0.35))
        z_grounded = z_neural + mix * gate * (z_graph - z_neural)
        return z_grounded, z_graph, gate

    @staticmethod
    def _fit_graph_sequence(
        seq: Optional[torch.Tensor],
        steps: int,
        *,
        pad_with_first: bool = True,
    ) -> Optional[torch.Tensor]:
        if seq is None or seq.numel() == 0 or steps <= 0:
            return None
        if seq.size(0) >= steps:
            return seq[-steps:]
        pad_steps = steps - seq.size(0)
        pad_ref = seq[:1] if pad_with_first else seq[-1:]
        pad = pad_ref.expand(pad_steps, -1)
        return torch.cat([pad, seq], dim=0)

    def _prior_world_targets(
        self,
        batch_size: int,
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prior = self.world_state_prior.to(device=device, dtype=dtype).view(1, 1, -1)
        prior = prior.expand(batch_size, steps, -1).clone()
        return prior, prior.clone()

    def _hidden_world_targets(
        self,
        h_tok: torch.Tensor,
        actions: torch.Tensor,
        max_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        steps = min(max_steps, h_tok.size(1) - 1, actions.size(1))
        h_slice = h_tok[:, -(steps + 1):, :]
        teacher_states = self.world_target_proj(h_slice[:, :-1]).detach()
        world_targets = self.world_target_proj(h_slice[:, 1:]).detach()
        return teacher_states, world_targets, steps

    def _execution_world_targets(
        self,
        world_graph_batch: Optional[WorldGraphBatch],
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        pad_with_first: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        stats = {
            "execution_steps": 0.0,
            "hidden_fallback_steps": 0.0,
            "trace_supervised_steps": 0.0,
            "trace_samples": 0.0,
        }
        if (
            steps <= 0
            or world_graph_batch is None
            or not world_graph_batch.graphs
            or world_graph_batch.pooled_states.numel() == 0
            or not self.world_graph_enabled
        ):
            return None, None, stats

        pooled_states = world_graph_batch.pooled_states.detach().to(device=device, dtype=dtype)
        teacher_states = pooled_states.unsqueeze(1).expand(-1, steps, -1).clone()
        world_targets = teacher_states.clone()
        stats["execution_steps"] = float(teacher_states.size(0) * steps)

        for batch_idx, graph in enumerate(world_graph_batch.graphs):
            if graph.transition_targets is None or graph.transition_states is None:
                continue
            trace_teacher = self._fit_graph_sequence(
                graph.transition_states.detach().to(device=device, dtype=dtype),
                steps,
                pad_with_first=pad_with_first,
            )
            trace_targets = self._fit_graph_sequence(
                graph.transition_targets.detach().to(device=device, dtype=dtype),
                steps,
                pad_with_first=pad_with_first,
            )
            if trace_teacher is None or trace_targets is None:
                continue
            teacher_states[batch_idx] = trace_teacher
            world_targets[batch_idx] = trace_targets
            stats["trace_samples"] += 1.0
            stats["trace_supervised_steps"] += float(
                min(steps, graph.transition_targets.size(0), graph.transition_states.size(0))
            )
        return teacher_states, world_targets, stats

    def _compose_canonical_world_state(
        self,
        *,
        z_neural: torch.Tensor,
        z_graph_grounded: torch.Tensor,
        z_graph_readout: torch.Tensor,
        z_graph_anchor: torch.Tensor,
        z_grounded: torch.Tensor,
        z_graph: torch.Tensor,
        z_program: torch.Tensor,
        z_symbolic: torch.Tensor,
        v_mem: torch.Tensor,
        has_program_state: bool,
        world_graph_batch: Optional[WorldGraphBatch],
        task_context: SymbolicTaskContext,
    ) -> CanonicalWorldState:
        if world_graph_batch is None or not world_graph_batch.graphs:
            fallback_graph = self.world_graph(
                facts=tuple(task_context.observed_facts),
                trace_bundle=task_context.execution_trace,
                device=z_grounded.device,
            )
            graphs = tuple(fallback_graph for _ in range(z_grounded.size(0)))
        else:
            graphs = tuple(world_graph_batch.graphs)
        metadata = {
            "canonical_stack": 1.0,
            "literal_graph_z": 1.0,
            "graph_state_norm": float(z_graph.norm(dim=-1).mean().item()) if z_graph.numel() > 0 else 0.0,
            "graph_readout_norm": float(z_graph_readout.norm(dim=-1).mean().item()) if z_graph_readout.numel() > 0 else 0.0,
            "graph_anchor": float(z_graph_anchor.mean().item()) if z_graph_anchor.numel() > 0 else 0.0,
            "grounded_state_norm": float(z_grounded.norm(dim=-1).mean().item()) if z_grounded.numel() > 0 else 0.0,
            "program_state_norm": float(z_program.norm(dim=-1).mean().item()) if z_program.numel() > 0 else 0.0,
        }
        return CanonicalWorldState(
            graphs=graphs,
            neural_state=z_neural.detach(),
            graph_grounded_state=z_graph_grounded.detach(),
            graph_projection=z_graph.detach(),
            graph_readout_state=z_graph_readout.detach(),
            grounded_state=z_grounded.detach(),
            symbolic_state=z_symbolic.detach(),
            memory_state=v_mem.detach(),
            program_state=z_program.detach() if has_program_state else None,
            symbolic_facts=tuple(sorted(list(task_context.observed_facts), key=self._fact_sort_key)),
            target_facts=tuple(sorted(list(task_context.target_facts), key=self._fact_sort_key)),
            metadata=metadata,
        )

    def _graph_centered_decoder_state(
        self,
        z_query: torch.Tensor,
        world_graph_batch: Optional[WorldGraphBatch],
        *,
        program_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if world_graph_batch is None or not world_graph_batch.graphs:
            zeros = torch.zeros_like(z_query)
            anchors = torch.zeros(
                z_query.size(0),
                device=z_query.device,
                dtype=z_query.dtype,
            )
            return z_query, zeros, anchors
        graph_mix = float(getattr(self.cfg, "world_graph_decoder_mix", 0.55))
        return self.state_integrator.graph_centered(
            z_query,
            world_graph_batch.graphs,
            program_state=program_state,
            graph_mix=graph_mix,
        )

    def _world_rollout_from_hidden(
        self,
        h_tok: torch.Tensor,
        actions: torch.Tensor,
        world_graph_batch: Optional[WorldGraphBatch] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_steps = max(int(getattr(self.cfg, 'world_rollout_steps', 8)), 1)
        if h_tok.size(1) > 1 and actions.size(1) > 0:
            steps = min(max_steps, h_tok.size(1) - 1, actions.size(1))
            execution_driven = bool(getattr(self.cfg, "world_graph_execution_driven", True))
            pad_with_first = bool(getattr(self.cfg, "world_graph_trace_pad_with_first", True))
            rollout_stats = {
                "execution_steps": 0.0,
                "hidden_fallback_steps": 0.0,
                "trace_supervised_steps": 0.0,
                "trace_samples": 0.0,
                "hidden_teacher_applied": 0.0,
                "neutral_prior_applied": 0.0,
                "neutral_prior_steps": 0.0,
            }
            teacher_states = None
            world_targets = None

            if execution_driven:
                execution_teacher, execution_targets, execution_stats = self._execution_world_targets(
                    world_graph_batch,
                    steps,
                    device=h_tok.device,
                    dtype=self.world_state_prior.dtype,
                    pad_with_first=pad_with_first,
                )
                rollout_stats.update(execution_stats)
                if execution_teacher is not None and execution_targets is not None:
                    teacher_states = execution_teacher
                    world_targets = execution_targets
                    rollout_stats["hidden_fallback_steps"] = 0.0
                    rollout_stats["hidden_teacher_applied"] = 0.0
                else:
                    teacher_states, world_targets = self._prior_world_targets(
                        h_tok.size(0),
                        steps,
                        device=h_tok.device,
                        dtype=self.world_state_prior.dtype,
                    )
                    rollout_stats["execution_steps"] = float(h_tok.size(0) * steps)
                    rollout_stats["hidden_fallback_steps"] = 0.0
                    rollout_stats["hidden_teacher_applied"] = 0.0
                    rollout_stats["neutral_prior_applied"] = 1.0
                    rollout_stats["neutral_prior_steps"] = float(h_tok.size(0) * steps)
            else:
                hidden_teacher, hidden_targets, _ = self._hidden_world_targets(
                    h_tok,
                    actions,
                    max_steps,
                )
                rollout_stats["hidden_fallback_steps"] = float(hidden_teacher.size(0) * steps)
                rollout_stats["hidden_teacher_applied"] = 1.0
                teacher_states = hidden_teacher
                world_targets = hidden_targets
            if (
                not execution_driven
                and world_graph_batch is not None
                and world_graph_batch.graphs
            ):
                pooled_states = world_graph_batch.pooled_states.detach().to(
                    device=teacher_states.device,
                    dtype=teacher_states.dtype,
                )
                pooled_states = pooled_states.unsqueeze(1).expand(-1, steps, -1)
                pooled_mix = float(getattr(self.cfg, "world_graph_pooled_mix", 0.15))
                trace_mix = float(getattr(self.cfg, "world_graph_trace_mix", 1.0))
                teacher_states = torch.lerp(hidden_teacher, pooled_states, pooled_mix)
                world_targets = torch.lerp(hidden_targets, pooled_states, pooled_mix)
                for batch_idx, graph in enumerate(world_graph_batch.graphs):
                    if graph.transition_targets is None or graph.transition_states is None:
                        continue
                    trace_teacher = self._fit_graph_sequence(
                        graph.transition_states.detach().to(
                            device=teacher_states.device,
                            dtype=teacher_states.dtype,
                        ),
                        steps,
                        pad_with_first=pad_with_first,
                    )
                    trace_targets = self._fit_graph_sequence(
                        graph.transition_targets.detach().to(
                            device=world_targets.device,
                            dtype=world_targets.dtype,
                        ),
                        steps,
                        pad_with_first=pad_with_first,
                    )
                    if trace_teacher is None or trace_targets is None:
                        continue
                    sample_steps = min(
                        steps,
                        graph.transition_targets.size(0),
                        graph.transition_states.size(0),
                    )
                    teacher_states[batch_idx, -sample_steps:] = torch.lerp(
                        teacher_states[batch_idx, -sample_steps:],
                        trace_teacher[-sample_steps:],
                        trace_mix,
                    )
                    world_targets[batch_idx, -sample_steps:] = torch.lerp(
                        world_targets[batch_idx, -sample_steps:],
                        trace_targets[-sample_steps:],
                        trace_mix,
                    )
                    rollout_stats["trace_supervised_steps"] += float(sample_steps)
                    rollout_stats["trace_samples"] += 1.0
                    rollout_stats["execution_steps"] += float(sample_steps)
                    rollout_stats["hidden_fallback_steps"] = max(
                        0.0,
                        rollout_stats["hidden_fallback_steps"] - float(sample_steps),
                    )
            if world_graph_batch is not None:
                world_graph_batch.metadata["trace_supervised_steps"] = rollout_stats["trace_supervised_steps"]
                world_graph_batch.metadata["execution_supervised_steps"] = rollout_stats["execution_steps"]
                world_graph_batch.metadata["hidden_fallback_steps"] = rollout_stats["hidden_fallback_steps"]
                world_graph_batch.metadata["trace_primary_samples"] = rollout_stats["trace_samples"]
                world_graph_batch.metadata["hidden_teacher_applied"] = rollout_stats["hidden_teacher_applied"]
                world_graph_batch.metadata["neutral_prior_applied"] = rollout_stats["neutral_prior_applied"]
                world_graph_batch.metadata["neutral_prior_steps"] = rollout_stats["neutral_prior_steps"]
            action_seq = actions[:, -steps:]
            z_sim_traj = self.world_rnn.simulate_sequence(
                teacher_states[:, 0],
                action_seq,
                teacher_forcing_ratio=teacher_forcing_ratio,
                teacher_states=teacher_states,
            )
            return z_sim_traj, world_targets

        execution_driven = bool(getattr(self.cfg, "world_graph_execution_driven", True))
        fallback_targets = None
        if execution_driven:
            execution_teacher, execution_targets, rollout_stats = self._execution_world_targets(
                world_graph_batch,
                1,
                device=h_tok.device,
                dtype=self.world_state_prior.dtype,
                pad_with_first=bool(getattr(self.cfg, "world_graph_trace_pad_with_first", True)),
            )
            if execution_teacher is not None and execution_targets is not None:
                fallback_state = execution_teacher
                fallback_targets = execution_targets
                if world_graph_batch is not None:
                    world_graph_batch.metadata["trace_supervised_steps"] = rollout_stats["trace_supervised_steps"]
                    world_graph_batch.metadata["execution_supervised_steps"] = rollout_stats["execution_steps"]
                    world_graph_batch.metadata["hidden_fallback_steps"] = 0.0
                    world_graph_batch.metadata["trace_primary_samples"] = rollout_stats["trace_samples"]
                    world_graph_batch.metadata["hidden_teacher_applied"] = 0.0
                    world_graph_batch.metadata["neutral_prior_applied"] = 0.0
                    world_graph_batch.metadata["neutral_prior_steps"] = 0.0
            else:
                fallback_state, fallback_targets = self._prior_world_targets(
                    h_tok.size(0),
                    1,
                    device=h_tok.device,
                    dtype=self.world_state_prior.dtype,
                )
                if world_graph_batch is not None:
                    world_graph_batch.metadata["trace_supervised_steps"] = 0.0
                    world_graph_batch.metadata["execution_supervised_steps"] = float(fallback_state.size(0))
                    world_graph_batch.metadata["hidden_fallback_steps"] = 0.0
                    world_graph_batch.metadata["trace_primary_samples"] = 0.0
                    world_graph_batch.metadata["hidden_teacher_applied"] = 0.0
                    world_graph_batch.metadata["neutral_prior_applied"] = 1.0
                    world_graph_batch.metadata["neutral_prior_steps"] = float(fallback_state.size(0))
        elif world_graph_batch is not None and world_graph_batch.graphs:
            fallback_state = self.world_target_proj(h_tok[:, -1:, :]).detach()
            pooled_states = world_graph_batch.pooled_states.detach().to(
                device=fallback_state.device,
                dtype=fallback_state.dtype,
            ).unsqueeze(1)
            fallback_state = torch.lerp(
                fallback_state,
                pooled_states,
                float(getattr(self.cfg, "world_graph_pooled_mix", 0.15)),
            )
            world_graph_batch.metadata["trace_supervised_steps"] = 0.0
            world_graph_batch.metadata["execution_supervised_steps"] = float(fallback_state.size(0))
            world_graph_batch.metadata["hidden_fallback_steps"] = 0.0
            world_graph_batch.metadata["trace_primary_samples"] = 0.0
            world_graph_batch.metadata["hidden_teacher_applied"] = 1.0
            world_graph_batch.metadata["neutral_prior_applied"] = 0.0
            world_graph_batch.metadata["neutral_prior_steps"] = 0.0
        elif world_graph_batch is not None:
            fallback_state = self.world_target_proj(h_tok[:, -1:, :]).detach()
            world_graph_batch.metadata["trace_supervised_steps"] = 0.0
            world_graph_batch.metadata["execution_supervised_steps"] = 0.0
            world_graph_batch.metadata["hidden_fallback_steps"] = float(fallback_state.size(0))
            world_graph_batch.metadata["trace_primary_samples"] = 0.0
            world_graph_batch.metadata["hidden_teacher_applied"] = 1.0
            world_graph_batch.metadata["neutral_prior_applied"] = 0.0
            world_graph_batch.metadata["neutral_prior_steps"] = 0.0
        else:
            fallback_state = self.world_target_proj(h_tok[:, -1:, :]).detach()
        fallback_actions = actions[:, -1:] if actions.size(1) > 0 else torch.zeros(
            h_tok.size(0), 1, dtype=torch.long, device=h_tok.device
        )
        z_sim_traj = self.world_rnn.simulate_sequence(
            fallback_state[:, 0],
            fallback_actions,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        if execution_driven and fallback_targets is not None:
            return z_sim_traj, fallback_targets.to(device=z_sim_traj.device, dtype=z_sim_traj.dtype)
        return z_sim_traj, z_sim_traj.detach()

    def _eval_world_self_update_active(self) -> bool:
        if self.training:
            return False
        if not bool(getattr(self.cfg, "eval_world_self_update_enabled", True)):
            return False
        if not bool(getattr(self.cfg, "continuous_cycle_eval_learning_enabled", True)):
            return False
        if not torch.is_grad_enabled():
            return False
        if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
            return False
        return True

    def _generation_online_learning_active(self) -> bool:
        if self.training:
            return False
        if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
            return False
        if not torch.is_grad_enabled():
            return False
        return bool(getattr(self.cfg, "continuous_cycle_eval_learning_enabled", True))

    def _maybe_eval_world_self_update(
        self,
        *,
        world_loss: torch.Tensor,
        program_anchor_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        stats = {
            "applied": 0.0,
            "loss": 0.0,
            "grad_norm": 0.0,
            "effective_lr": 0.0,
            "parameter_tensors": 0.0,
            "parameter_elements": 0.0,
            "program_weight": 0.0,
        }
        if not self._eval_world_self_update_active():
            return stats
        if not torch.is_tensor(world_loss) or not world_loss.requires_grad:
            return stats
        params = [param for param in self.world_rnn.parameters() if param.requires_grad]
        if not params:
            return stats

        update_loss = world_loss
        program_weight = float(getattr(self.cfg, "eval_world_self_update_program_weight", 0.0))
        if (
            program_weight > 0.0
            and torch.is_tensor(program_anchor_loss)
            and program_anchor_loss.requires_grad
        ):
            update_loss = update_loss + program_weight * program_anchor_loss
            stats["program_weight"] = program_weight
        if not bool(torch.isfinite(update_loss).all().item()):
            return stats

        grads = torch.autograd.grad(
            update_loss,
            params,
            allow_unused=True,
            retain_graph=True,
        )
        grad_sq = 0.0
        valid_pairs = []
        param_elements = 0.0
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            if not bool(torch.isfinite(grad).all().item()):
                continue
            valid_pairs.append((param, grad))
            grad_sq += float(grad.detach().pow(2).sum().item())
            param_elements += float(param.numel())
        if not valid_pairs:
            return stats

        grad_norm = grad_sq ** 0.5
        clip = float(getattr(self.cfg, "eval_world_self_update_clip", 1.0))
        lr = float(getattr(self.cfg, "eval_world_self_update_lr", 1e-3))
        scale = 1.0
        if clip > 0.0 and grad_norm > clip:
            scale = clip / max(grad_norm, 1e-12)
        effective_lr = lr * scale

        with torch.no_grad():
            for param, grad in valid_pairs:
                param.add_(grad, alpha=-effective_lr)

        stats.update(
            {
                "applied": 1.0,
                "loss": float(update_loss.detach().item()),
                "grad_norm": float(grad_norm),
                "effective_lr": float(effective_lr),
                "parameter_tensors": float(len(valid_pairs)),
                "parameter_elements": float(param_elements),
            }
        )
        return stats

    def _combine_levels(
        self,
        z_neural: torch.Tensor,
        z_symbolic: torch.Tensor,
        v_mem: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_integrator.post_symbolic(z_neural, z_symbolic, v_mem)

    def _pre_symbolic_state(
        self,
        z_neural: torch.Tensor,
        v_mem: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_integrator.pre_symbolic(z_neural, v_mem)

    def _model_description_bits(self) -> Dict[str, torch.Tensor]:
        sigma = float(getattr(self.cfg, "mdl_param_sigma", 0.05))
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        neural_bits = torch.zeros((), device=device, dtype=dtype)
        vocab_bits = torch.zeros((), device=device, dtype=dtype)
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "codebook.weight" in name:
                continue
            neural_bits = neural_bits + gaussian_tensor_bits(param, sigma=sigma)
        if self.net_enabled and hasattr(self, "net") and hasattr(self.net, "quantizer"):
            vocab_bits = self.net.quantizer.vocab_description_bits().to(
                device=device,
                dtype=dtype,
            )
        return {
            "neural": neural_bits,
            "vocab": vocab_bits,
        }

    def _recall_symbolic_memory_facts(
        self,
        z_query: torch.Tensor,
        hint_facts: Optional[List[HornAtom]] = None,
        goal: Optional[HornAtom] = None,
    ) -> List[HornAtom]:
        top_k = int(getattr(self.cfg, "mem_symbolic_recall_topk", 8))
        min_sim = float(getattr(self.cfg, "mem_symbolic_min_sim", 0.2))
        predicate_hints: Set[int] = set()
        anchor_values: Set[int] = set()
        for fact in hint_facts or []:
            predicate_hints.add(int(fact.pred))
            anchor_values.update(self._const_values_from_atom(fact))
        if goal is not None:
            predicate_hints.add(int(goal.pred))
            anchor_values.update(self._const_values_from_atom(goal))
        recalled = self.memory.recall_symbolic_atoms(
            z_query[:1],
            top_k=top_k,
            min_sim=min_sim,
            predicate_hints=sorted(predicate_hints),
            anchor_values=sorted(anchor_values),
        )
        return [fact for fact in recalled if isinstance(fact, HornAtom)]

    def _write_symbolic_memory_facts(
        self,
        facts: FrozenSet[HornAtom],
        confidence: float,
    ) -> int:
        if not facts or confidence <= float(getattr(self.cfg, "mem_write_tau", 0.3)):
            return 0
        fact_list = list(facts)[: int(getattr(self.cfg, "sym_max_facts", 64))]
        fact_embs = self.prover.term_emb(fact_list, next(self.parameters()).device)
        return self.memory.write_symbolic_atoms(fact_list, fact_embs)

    def _sample_variational_latent(
        self,
        z_det: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build q(z|o) with a conservative residual posterior."""
        if not bool(getattr(self.cfg, "vfe_enabled", True)):
            zeros = torch.zeros_like(z_det)
            return z_det, z_det, zeros
        mu = z_det + self.posterior_mu(z_det)
        logvar = self.posterior_logvar(z_det).clamp(-6.0, 2.0)
        if self.training:
            eps = torch.randn_like(mu)
            z_post = mu + eps * (0.5 * logvar).exp()
        else:
            z_post = mu
        return z_post, mu, logvar

    @staticmethod
    def _conditional_gaussian_prior(
        anchor: torch.Tensor,
        mu_layer: nn.Linear,
        logvar_layer: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base = anchor.detach()
        mu = base + mu_layer(base)
        logvar = logvar_layer(base).clamp(-6.0, 2.0)
        return mu, logvar

    @staticmethod
    def _decoder_query_from_context(task_context: SymbolicTaskContext) -> HornAtom:
        last_src = int(task_context.metadata.get("last_src", 0))
        return HornAtom(SEQ_PREDICT_NEXT_PRED, (last_src, Var("NEXT")))

    def _symbolic_token_logits(
        self,
        z_symbolic: torch.Tensor,
        task_context: SymbolicTaskContext,
    ) -> torch.Tensor:
        query = self._decoder_query_from_context(task_context)
        query_emb = self.prover.ground(frozenset({query}), z_symbolic.device).expand(z_symbolic.size(0), -1)
        return self.symbolic_token_head(torch.cat([z_symbolic, query_emb], dim=-1))

    @staticmethod
    def _safe_cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = 0,
    ) -> torch.Tensor:
        flat_targets = targets.reshape(-1)
        flat_logits = logits.reshape(flat_targets.size(0), -1)
        valid = flat_targets.ne(ignore_index)
        if not bool(valid.any()):
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(flat_logits[valid], flat_targets[valid], reduction="mean")

    def _decoder_surprise_signal(
        self,
        h_tok: Optional[torch.Tensor],
        z_enriched: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Optional[Dict[str, object]]:
        """Estimate a local next-token miss signal before symbolic reasoning."""
        if (
            not self._decoder_surprise_enabled
            or h_tok is None
            or h_tok.numel() == 0
            or tgt.numel() == 0
        ):
            return None
        probe_input = torch.cat([h_tok[:, -1, :], z_enriched], dim=-1)
        probe_logits = self.decoder_surprise_head(probe_input)
        targets = tgt[:, -1].long()
        probe_ce = F.cross_entropy(probe_logits, targets, reduction="none")
        probe_probs = F.softmax(probe_logits, dim=-1)
        pred_tokens = probe_logits.argmax(dim=-1)
        gold_probs = probe_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        misses = pred_tokens.ne(targets).to(probe_logits.dtype)
        surprise = 0.5 * misses + 0.5 * (1.0 - gold_probs)
        return {
            "pred_tokens": pred_tokens.detach(),
            "targets": targets.detach(),
            "misses": misses.detach(),
            "surprise": surprise.detach(),
            "probe_ce": probe_ce.detach(),
            "loss_tensor": probe_ce.mean(),
            "loss": float(probe_ce.mean().detach().item()),
        }

    @staticmethod
    def _dedupe_facts(facts: List[HornAtom], limit: int) -> List[HornAtom]:
        unique: List[HornAtom] = []
        seen: Set[HornAtom] = set()
        for fact in facts:
            if fact in seen:
                continue
            unique.append(fact)
            seen.add(fact)
            if len(unique) >= limit:
                break
        return unique

    @staticmethod
    def _decode_source_bytes(src_row: torch.Tensor) -> str:
        raw = bytes(int(v) % 256 for v in src_row.tolist()).rstrip(b"\x00")
        return raw.decode("utf-8", errors="ignore")

    @staticmethod
    def _prioritize_trace_targets(
        facts: FrozenSet[HornAtom],
        limit: int,
    ) -> List[HornAtom]:
        priority = {
            TRACE_RETURN_EVENT_PRED: 6,
            TRACE_ERROR_EVENT_PRED: 6,
            TRACE_BINOP_EVENT_PRED: 5,
            TRACE_ASSIGN_EVENT_PRED: 4,
            TRACE_PARAM_BIND_PRED: 3,
            TRACE_STATE_VALUE_PRED: 2,
        }

        def fact_key(atom: HornAtom) -> Tuple[int, int, Tuple[int, ...]]:
            values = OMENScale._const_values_from_atom(atom)
            return (
                -priority.get(int(atom.pred), 0),
                int(atom.pred),
                tuple(values),
            )

        ranked = sorted(list(facts), key=fact_key)
        return ranked[: max(int(limit), 0)]

    @staticmethod
    def _fact_sort_key(atom: HornAtom) -> Tuple[int, Tuple[int, ...], int]:
        return (
            int(atom.pred),
            tuple(OMENScale._const_values_from_atom(atom)),
            len(atom.args),
        )

    def _program_anchor_facts(
        self,
        task_context: SymbolicTaskContext,
    ) -> FrozenSet[HornAtom]:
        if not bool(getattr(self.cfg, "program_anchor_enabled", True)):
            return frozenset()
        limit = max(int(getattr(self.cfg, "program_anchor_max_facts", 24)), 0)
        if limit <= 0:
            return frozenset()
        facts: List[HornAtom] = []
        if task_context.goal is not None and task_context.goal.is_ground():
            facts.append(task_context.goal)
        facts.extend(sorted(list(task_context.target_facts), key=self._fact_sort_key))
        trace_bundle = task_context.execution_trace
        if trace_bundle is not None:
            facts.extend(self._prioritize_trace_targets(trace_bundle.target_facts, limit=8))
            facts.extend(
                sorted(list(trace_bundle.observed_facts), key=self._fact_sort_key)[:8]
            )
        return frozenset(self._dedupe_facts(facts, limit=limit))

    @staticmethod
    def _first_const(atom: HornAtom) -> int:
        for arg in atom.args:
            if isinstance(arg, Const):
                return int(arg.val)
        return int(atom.pred)

    @staticmethod
    def _const_values_from_atom(atom: HornAtom) -> List[int]:
        values: List[int] = [int(atom.pred)]

        def visit(term) -> None:
            if isinstance(term, Const):
                values.append(int(term.val))
                return
            func = getattr(term, "func", None)
            if func is not None:
                values.append(int(func))
            for subterm in getattr(term, "subterms", ()) or ():
                visit(subterm)

        for arg in atom.args:
            visit(arg)
        return values

    @staticmethod
    def _queryable_candidate_preds(
        facts: List[HornAtom],
        goal: Optional[HornAtom] = None,
    ) -> Tuple[int, ...]:
        preds: Set[int] = {SEQ_PREDICT_NEXT_PRED}
        for fact in facts:
            if fact.arity() >= 2:
                preds.add(int(fact.pred))
        if goal is not None and goal.arity() >= 2:
            preds.add(int(goal.pred))
        return tuple(sorted(preds))

    def _seed_symbolic_memory_facts(
        self,
        tokens: torch.Tensor,
        decoder_signal: Optional[Dict[str, object]] = None,
    ) -> List[HornAtom]:
        row = tokens[0].detach().cpu()
        if row.numel() == 0:
            return []
        seed: List[HornAtom] = []
        edge_cap = min(max(int(row.numel()) - 1, 0), 6)
        start_idx = max(int(row.numel()) - edge_cap - 1, 0)
        for idx in range(start_idx, max(int(row.numel()) - 1, 0)):
            seed.append(HornAtom(
                SEQ_EDGE_PRED,
                (int(row[idx].item()), int(row[idx + 1].item())),
            ))
        last_token = int(row[-1].item())
        seed.append(HornAtom(SEQ_LAST_TOKEN_PRED, (last_token, max(int(row.numel()) - 1, 0))))
        if decoder_signal is not None and torch.is_tensor(decoder_signal.get("pred_tokens")):
            pred_tokens = decoder_signal["pred_tokens"]
            if pred_tokens.numel() > 0:
                decoder_pred = int(pred_tokens[0].item())
                seed.append(HornAtom(SEQ_DECODER_GUESS_PRED, (last_token, decoder_pred)))
        seed.extend(self._ast_facts_from_bytes(row)[:8])
        return self._dedupe_facts(seed, limit=24)

    def _ast_facts_from_bytes(self, src_row: torch.Tensor) -> List[HornAtom]:
        """
        Витягує Horn-факти з байтового рядка через MultiLangASTParser.

        Ідеальна реалізація (пункт 1):
          · Перевіряємо SymbolicFactCache — якщо hit, повертаємо без парсингу.
          · Автодетект мови (detect_lang) замість фіксованого 'python'.
          · Зберігаємо результат у кеш для наступних батчів.

        Returns:
            List[HornAtom] — знайдені факти (ліміт _ctx_ast_max_facts)
        """
        cached = self._fact_cache.get(src_row)
        if cached is not None:
            facts, _rules, _trace, _lang = cached
            return facts

        try:
            code = self._decode_source_bytes(src_row)
        except Exception:
            return []
        if not code.strip():
            return []
        detected_lang = "python"
        try:
            detected_lang = self.ast_parser.detect_lang(code)
            # Автодетект мови замість фіксованого 'python'
            facts = self.ast_parser.parse_autodetect(code, source_id=0)
        except Exception:
            return []
        try:
            trace_bundle = build_symbolic_trace_bundle(
                code,
                lang_hint=detected_lang,
                max_steps=int(getattr(self.cfg, "sym_trace_max_steps", 24)),
                max_counterexamples=int(getattr(self.cfg, "sym_trace_max_counterexamples", 4)),
            )
        except Exception:
            trace_bundle = None
        facts = self._dedupe_facts(facts, self._ctx_ast_max_facts)
        # Витягуємо правила-шаблони (пункт 6 ідеальної реалізації)
        try:
            rule_templates = self.ast_parser.extract_rule_templates(facts, max_rules=16)
        except Exception:
            rule_templates = []
        # Зберігаємо в кеш
        self._fact_cache.put(src_row, facts, rule_templates, trace_bundle, detected_lang)
        return facts

    def _ast_rules_from_bytes(self, src_row: torch.Tensor):
        """
        Повертає правила-шаблони для src_row (з кешу або парсинг).
        Відповідає пункту 6 ідеальної реалізації:
          «AST-факти як джерело правил, а не тільки фактів.»
        """
        # Спочатку перевіряємо кеш (попередньо заповнений _ast_facts_from_bytes)
        cached = self._fact_cache.get(src_row)
        if cached is not None:
            _facts, rules, _trace, _lang = cached
            return rules
        # Примусово парсимо, щоб заповнити кеш
        self._ast_facts_from_bytes(src_row)
        cached = self._fact_cache.get(src_row)
        if cached is not None:
            return cached[1]
        return []

    def _ast_trace_from_bytes(self, src_row: torch.Tensor):
        cached = self._fact_cache.get(src_row)
        if cached is not None:
            _facts, _rules, trace_bundle, _lang = cached
            return trace_bundle
        self._ast_facts_from_bytes(src_row)
        cached = self._fact_cache.get(src_row)
        if cached is not None:
            return cached[2]
        return None

    def _ast_lang_from_bytes(self, src_row: torch.Tensor) -> str:
        cached = self._fact_cache.get(src_row)
        if cached is not None:
            _facts, _rules, _trace, detected_lang = cached
            if isinstance(detected_lang, str) and detected_lang:
                return detected_lang
        try:
            code = self._decode_source_bytes(src_row)
        except Exception:
            return "python"
        if not code.strip():
            return "python"
        try:
            detected_lang = self.ast_parser.detect_lang(code)
        except Exception:
            detected_lang = "python"
        self._ast_facts_from_bytes(src_row)
        return detected_lang

    def _build_counterfactual_actions(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        n_cf = max(int(getattr(self.cfg, "n_counterfactual", 0)), 0)
        if n_cf <= 0:
            return torch.zeros(src.size(0), 0, dtype=torch.long, device=src.device)
        choices: List[torch.Tensor] = [tgt[:, -1], src[:, -1]]
        if src.size(1) > 1:
            choices.append(src[:, -2])
        if tgt.size(1) > 1:
            choices.append(tgt[:, -2])
        cf = torch.stack(choices[:max(1, min(len(choices), n_cf))], dim=1)
        if cf.size(1) < n_cf:
            repeats = math.ceil(n_cf / max(cf.size(1), 1))
            cf = cf.repeat(1, repeats)
        return cf[:, :n_cf]

    def _build_symbolic_task_context(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        gap_norm: torch.Tensor,
        hot_dims: torch.Tensor,
        saliency_out: Optional[object],
        h_tok: Optional[torch.Tensor] = None,
        decoder_signal: Optional[Dict[str, object]] = None,
        memory_facts: Optional[List[HornAtom]] = None,
    ) -> SymbolicTaskContext:
        """
        Будує SymbolicTaskContext для S-Core.

        Ідеальна реалізація (пункти 1, 3, 6):
          1. Факти беруться з SymbolicFactCache (без повторного парсингу).
          6. AST правила-шаблони завантажуються в прувер KB (статус: verified).
          3. Ціль формується динамічно через SymbolicQueryGenerator
             (якщо доступний h_tok), а не як останній факт AST.
        """
        src_row = src[0].detach().cpu()
        tgt_row = tgt[0].detach().cpu()
        observed: List[HornAtom] = list(memory_facts or [])

        pair_cap = min(src_row.numel(), max(self._ctx_max_facts // 2, 8))
        start_idx = max(src_row.numel() - pair_cap, 0)
        for idx in range(start_idx, src_row.numel()):
            observed.append(HornAtom(
                SEQ_EDGE_PRED,
                (int(src_row[idx].item()), int(tgt_row[idx].item())),
            ))

        last_src = int(src_row[-1].item())
        last_tgt = int(tgt_row[-1].item())
        observed.append(HornAtom(SEQ_LAST_TOKEN_PRED, (last_src, src_row.numel() - 1)))

        decoder_pred = last_tgt
        decoder_miss = 0.0
        decoder_surprise = 0.0
        decoder_probe_ce = 0.0
        if decoder_signal is not None and torch.is_tensor(decoder_signal.get("pred_tokens")):
            pred_tokens = decoder_signal["pred_tokens"]
            misses = decoder_signal["misses"]
            surprises = decoder_signal["surprise"]
            probe_ce = decoder_signal["probe_ce"]
            if pred_tokens.numel() > 0:
                decoder_pred = int(pred_tokens[0].item())
                decoder_miss = float(misses[0].item())
                decoder_surprise = float(surprises[0].item())
                decoder_probe_ce = float(probe_ce[0].item())
                surprise_bucket = int(round(max(0.0, min(1.0, decoder_surprise)) * 100.0))
                observed.append(HornAtom(SEQ_DECODER_GUESS_PRED, (last_src, decoder_pred)))
                observed.append(HornAtom(SEQ_DECODER_SURPRISE_PRED, (last_src, surprise_bucket)))
                if decoder_miss > 0.0:
                    observed.append(HornAtom(SEQ_DECODER_MISS_PRED, (decoder_pred, last_tgt)))

        hot_idx = hot_dims[0].nonzero(as_tuple=True)[0].tolist()[:8]
        for dim_idx in hot_idx:
            observed.append(HornAtom(SEQ_GAP_DIM_PRED, (int(dim_idx), 1)))

        next_goal  = HornAtom(SEQ_PREDICT_NEXT_PRED, (last_src, last_tgt))
        actual_next = HornAtom(SEQ_ACTUAL_NEXT_PRED, (last_src, last_tgt))
        goal = HornAtom(SEQ_PREDICT_NEXT_PRED, (last_src, Var("NEXT")))
        provenance = "token"

        # ── Пункт 1: факти з кешу (або on-the-fly при miss) ──────────────────
        ast_facts = self._ast_facts_from_bytes(src_row)
        trace_bundle = self._ast_trace_from_bytes(src_row)
        ast_lang = self._ast_lang_from_bytes(src_row)

        # ── Пункт 6: завантажуємо AST правила-шаблони в KB (verified) ─────────
        # Це відповідає: «AST-факти як джерело правил, а не тільки фактів.»
        ast_rules = self._ast_rules_from_bytes(src_row)
        if ast_rules:
            from omen_prolog import EpistemicStatus
            for rule in ast_rules:
                # Перевіряємо що правило ще не в KB
                try:
                    self.prover.kb.add_rule(rule, status=EpistemicStatus.verified)
                except Exception:
                    pass

        saliency_semantic = list(saliency_out.sal_semantic_facts[0]) if saliency_out is not None else []
        saliency_expected = list(saliency_out.sal_expected_facts[0]) if saliency_out is not None else []

        # ── Базова ціль і дискретний контекст ─────────────────────────────────
        if ast_facts:
            observed.extend(ast_facts)
            observed.append(HornAtom(SEQ_AST_SUPPORT_PRED, (ast_facts[-1].pred, self._first_const(ast_facts[-1]))))
            provenance = "ast"
        if trace_bundle is not None:
            observed.extend(list(trace_bundle.observed_facts)[: max(self._ctx_ast_max_facts // 2, 12)])
            provenance = "ast_trace" if provenance.startswith("ast") else "trace"
        elif saliency_expected:
            goal = saliency_expected[0]
            observed.extend(saliency_expected[1:])
            observed.append(HornAtom(SEQ_SALIENCY_SUPPORT_PRED, (goal.pred, self._first_const(goal))))
            provenance = "saliency"

        observed.extend(saliency_semantic)
        if provenance not in ("ast", "ast_dynamic"):
            observed.extend(ast_facts[:4])
        observed = self._dedupe_facts(observed, self._ctx_max_facts)
        if (
            provenance in ("token", "ast")
            and self._sym_qg_enabled
            and self.sym_query_gen is not None
            and observed
        ):
            try:
                device = next(self.parameters()).device
                h_last = h_tok[0:1, -1, :] if h_tok is not None and h_tok.numel() > 0 else None
                symbolic_state = self.prover.ground(frozenset(observed), device)
                candidate_preds = self._queryable_candidate_preds(observed, goal=goal)
                goal = self.sym_query_gen.generate_query(
                    h_last,
                    self.cfg.sym_vocab,
                    context_anchor=last_src,
                    symbolic_state=symbolic_state,
                    candidate_preds=candidate_preds,
                )
                provenance = "ast_dynamic" if provenance == "ast" else "token_dynamic"
            except Exception:
                pass

        trace_targets = (
            self._prioritize_trace_targets(trace_bundle.target_facts, limit=8)
            if trace_bundle is not None else []
        )
        target_facts = self._dedupe_facts(
            [next_goal, actual_next] + ([goal] if goal.is_ground() else []) + trace_targets,
            limit=16,
        )
        gap_value = float(gap_norm.mean().item())
        sal_consistency = (
            float(getattr(saliency_out, "sal_consistency", 1.0))
            if saliency_out is not None else 1.0
        )
        trigger_abduction = (
            decoder_miss > 0.0
            or decoder_surprise >= float(getattr(self.cfg, "sym_decoder_surprise_threshold", 0.35))
            or gap_value > float(getattr(self.cfg, "epistemic_tau", 0.3))
            or sal_consistency < float(getattr(self.cfg, "saliency_consistency_threshold", 0.55))
        )
        return SymbolicTaskContext(
            observed_facts=frozenset(observed),
            goal=goal,
            target_facts=frozenset(target_facts),
            execution_trace=trace_bundle,
            provenance=provenance,
            trigger_abduction=trigger_abduction,
            hot_dims=tuple(int(i) for i in hot_idx),
            metadata={
                "gap_norm": gap_value,
                "saliency_consistency": sal_consistency,
                "last_src": float(last_src),
                "last_tgt": float(last_tgt),
                "decoder_pred": float(decoder_pred),
                "decoder_target": float(last_tgt),
                "decoder_miss": decoder_miss,
                "decoder_surprise": decoder_surprise,
                "decoder_probe_ce": decoder_probe_ce,
                "memory_facts": float(len(memory_facts or [])),
                "trace_steps": float(len(trace_bundle.transitions) if trace_bundle is not None else 0),
                "trace_counterexamples": float(len(trace_bundle.counterexamples) if trace_bundle is not None else 0),
                "ast_lang": ast_lang,
            },
        )

    def _build_generation_task_context(
        self,
        prompt: torch.Tensor,
        h_tok: Optional[torch.Tensor] = None,
        memory_facts: Optional[List[HornAtom]] = None,
    ) -> SymbolicTaskContext:
        prompt_row = prompt[0].detach().cpu()
        observed: List[HornAtom] = list(memory_facts or [])
        if prompt_row.numel() > 1:
            pair_cap = min(prompt_row.numel() - 1, max(self._ctx_max_facts // 2, 8))
            start_idx = max(prompt_row.numel() - pair_cap - 1, 0)
            for idx in range(start_idx, prompt_row.numel() - 1):
                observed.append(HornAtom(
                    SEQ_EDGE_PRED,
                    (int(prompt_row[idx].item()), int(prompt_row[idx + 1].item())),
                ))

        last_token = int(prompt_row[-1].item()) if prompt_row.numel() > 0 else 0
        last_pos = max(int(prompt_row.numel()) - 1, 0)
        observed.append(HornAtom(SEQ_LAST_TOKEN_PRED, (last_token, last_pos)))

        ast_facts = self._ast_facts_from_bytes(prompt_row)
        trace_bundle = self._ast_trace_from_bytes(prompt_row)
        ast_lang = self._ast_lang_from_bytes(prompt_row)
        provenance = "token"
        if ast_facts:
            observed.extend(ast_facts)
            observed.append(HornAtom(SEQ_AST_SUPPORT_PRED, (ast_facts[-1].pred, self._first_const(ast_facts[-1]))))
            provenance = "ast"
        if trace_bundle is not None:
            observed.extend(list(trace_bundle.observed_facts)[: max(self._ctx_ast_max_facts // 2, 12)])
            provenance = "ast_trace" if provenance.startswith("ast") else "trace"
        goal = HornAtom(SEQ_PREDICT_NEXT_PRED, (last_token, Var("NEXT")))
        observed = self._dedupe_facts(observed, self._ctx_max_facts)
        if self._sym_qg_enabled and self.sym_query_gen is not None and observed:
            try:
                device = next(self.parameters()).device
                h_last = h_tok[0:1, -1, :] if h_tok is not None and h_tok.numel() > 0 else None
                symbolic_state = self.prover.ground(frozenset(observed), device)
                candidate_preds = self._queryable_candidate_preds(observed, goal=goal)
                goal = self.sym_query_gen.generate_query(
                    h_last,
                    self.cfg.sym_vocab,
                    context_anchor=last_token,
                    symbolic_state=symbolic_state,
                    candidate_preds=candidate_preds,
                )
                provenance = "ast_dynamic" if provenance == "ast" else "token_dynamic"
            except Exception:
                pass
        return SymbolicTaskContext(
            observed_facts=frozenset(observed),
            goal=goal,
            target_facts=frozenset({goal} | (trace_bundle.target_facts if trace_bundle is not None else frozenset())),
            execution_trace=trace_bundle,
            provenance=provenance,
            trigger_abduction=False,
            hot_dims=tuple(),
            metadata={
                "mode": "generate",
                "last_src": float(last_token),
                "memory_facts": float(len(memory_facts or [])),
                "trace_steps": float(len(trace_bundle.transitions) if trace_bundle is not None else 0),
                "trace_counterexamples": float(len(trace_bundle.counterexamples) if trace_bundle is not None else 0),
                "ast_lang": ast_lang,
            },
        )


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
        if self.training:
            self._train_step.add_(1)
            self._seen_tokens.add_(int(max(tgt[:, 1:].ne(0).sum().item(), 1)))
        # ══ Рівень 1: Token → Concept  ═══════════════════════════════════════
        net_info   = {}
        vq_indices = None
        attn_maps = None
        saliency_hidden = None
        if self.net_enabled:
            # ── NET шлях: контекстне кодування + семантичне квантування ────────
            h_tok, vq_indices, net_info = self.net.encode(
                src, return_attn=self.saliency_enabled)        # (B,T,d_tok)
            attn_maps = net_info.get("attention_maps")
            saliency_hidden = net_info.get("h_ctx", h_tok)
        else:
            # ── Класичний шлях: просте Embedding + LlamaDecoderBlock ─────────
            if self.saliency_enabled:
                h_tok, attn_maps = self.tok_encoder(src, return_attn=True)
            else:
                h_tok = self.tok_encoder(src)                               # (B,T,d_tok)
            saliency_hidden = h_tok

        latents, z_base = self.perceiver(h_tok)                        # (B,n,d_lat), (B,d_lat)
        z, z_mu, z_logvar = self._sample_variational_latent(z_base)    # q(z|o)
        z_neural = z

        saliency_out = None
        if self.saliency_enabled:
            saliency_out = self.saliency(
                attn_maps=attn_maps,
                token_hidden=saliency_hidden,
                z_neural=z,
                prover=self.prover,
                train_step=int(self._train_step.item()),
            )


        world_graph_batch = self._build_world_graph_batch(src, saliency_out=saliency_out)
        z, z_graph, z_graph_gate = self._ground_world_state(z, world_graph_batch)

        # ── Level 2: M-Core читання ───────────────────────────────────────────
        v_mem = self._retrieve_memory(z)                               # (B, d_lat)

        # ── Level 2: WorldRNN симуляція ───────────────────────────────────────
        # FIX: z.detach() — WorldRNN отримує тільки «знімок» концепту z,
        # без backprop крізь z. Це ізолює WorldRNN-градієнт від решти граду:
        # градієнт L_world тече → z_sim → WorldRNN.params, а не → z → perceiver.
        teacher_forcing_ratio = self._world_teacher_forcing_ratio() if self.training else 0.0
        z_sim_traj, world_targets = self._world_rollout_from_hidden(
            h_tok,
            src,
            world_graph_batch=world_graph_batch,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        z_sim    = z_sim_traj[:, -1]                                   # (B, d_lat)

        # ── Epistemic Gap ─────────────────────────────────────────────────────
        E, gap_norm, hot_dims = self.epistemic.compute(
            z, self.world_rnn, z_sim)

        # ── Curiosity (якщо gap великий) ──────────────────────────────────────
        cf_actions = self._build_counterfactual_actions(src, tgt)
        z_enr, cf_loss = self.curiosity(
            z, E, hot_dims, gap_norm, self.memory, self.world_rnn,
            counterfactual_actions=cf_actions,
        )
        v_mem = self._retrieve_memory(z_enr)
        z_symbolic_in = self._pre_symbolic_state(z_enr, v_mem)
        decoder_signal = self._decoder_surprise_signal(h_tok, z_symbolic_in, tgt)
        seed_memory_facts = self._seed_symbolic_memory_facts(src, decoder_signal=decoder_signal)
        recalled_memory_facts = self._recall_symbolic_memory_facts(
            z_symbolic_in,
            hint_facts=seed_memory_facts,
        )

        # ── Пункт 1+3+6: передаємо h_tok для динамічної цілі та кешованих фактів
        task_context = self._build_symbolic_task_context(
            src, tgt, gap_norm, hot_dims, saliency_out,
            h_tok=h_tok,  # <-- тепер передається для SymbolicQueryGenerator
            decoder_signal=decoder_signal,
            memory_facts=recalled_memory_facts,
        )
        if world_graph_batch.graphs:
            task_context.metadata.update({
                "world_graph_nodes": float(world_graph_batch.metadata.get("mean_nodes", 0.0)),
                "world_graph_edges": float(world_graph_batch.metadata.get("mean_edges", 0.0)),
                "world_graph_trace_steps": float(world_graph_batch.metadata.get("trace_supervised_steps", 0.0)),
                "world_graph_execution_steps": float(world_graph_batch.metadata.get("execution_supervised_steps", 0.0)),
                "world_graph_hidden_fallback_steps": float(world_graph_batch.metadata.get("hidden_fallback_steps", 0.0)),
                "world_graph_neutral_prior_steps": float(world_graph_batch.metadata.get("neutral_prior_steps", 0.0)),
            })
        program_target_facts = self._program_anchor_facts(task_context)
        self.prover.set_task_context(task_context)

        # ── Пункт 2: load_observed_facts() — завантажуємо факти у WM, не в KB ──
        # (materialize_task_context_facts вже викликається в prover.forward(),
        #  але load_observed_facts() дає явний контроль до prover.forward())
        if task_context.observed_facts:
            self.prover.load_observed_facts(
                task_context.observed_facts,
                limit=int(getattr(self.cfg, "symbolic_context_max_facts", 96)),
            )

        # ── Level 3: ∂-Prolog (через EMC або напряму) ─────────────────────────
        world_loss = F.mse_loss(z_sim_traj, world_targets)
        world_err  = world_loss.detach()

        if self.emc_enabled and self.training:
            # ── EMC: Адаптивний контролер міркування ───────────────────────────
            # π_meta визначає: Stop | RecallMCore | ForwardChainStep | Abduce
            # Повертає (z_sym, sym_loss, v_mem_emc, meta_loss, traj_stats)
            z_sym, sym_loss, v_mem_emc, meta_loss, traj_stats = self.emc.run_episode(
                z_symbolic_in, gap_norm, hot_dims, self.prover, self.memory,
                world_err, device=z_enr.device,
            )
            # Якщо EMC виконав RecallMCore → використовуємо збагачений v_mem
            if v_mem_emc.norm() > 1e-6:
                v_mem = v_mem_emc
                z_symbolic_in = self._pre_symbolic_state(z_enr, v_mem)
        else:
            # ── Класичний шлях (eval або emc_enabled=False) ────────────────────
            z_sym, sym_loss = self.prover(z_symbolic_in, world_err)
            meta_loss = torch.zeros(1, device=z_enr.device).squeeze()
            traj_stats = None
        # ── Semantic Feedback Pairs для NET (I(Z;Γ) апроксимація) ────────────
        # FIX: прибрано дублювання — раніше sem_pairs_net обчислювався двічі:
        #   1) через _sem_pairs_cache (рядки 1267-1284)
        #   2) через semantic_feedback_pairs() (рядок 1287, перезаписував #1)
        # Тепер — єдина точка обчислення з кешем.
        _n_rules_now = len(self.prover)
        if _n_rules_now != self._sem_pairs_n_rules:
            # Кеш застарів — оновлюємо
            self._sem_pairs_cache = (
                self.prover.semantic_feedback_pairs(max_pairs=64)
                if self.net_enabled else []
            )
            self._sem_pairs_n_rules = _n_rules_now
        sem_pairs_net: list = self._sem_pairs_cache

        # ── VeM Penalty: δ·E[max(0, τ − U(R))] ─────────────────────────────
        vem_pen = self.prover.vem_loss(
            z_enr,
            delta=getattr(self.cfg, 'delta_vem', 1e-3)
        )

        # ── Об'єднуємо рівні ─────────────────────────────────────────────────
        z_fused = self._combine_levels(z_symbolic_in, z_sym, v_mem)  # (B, d_lat)

        # ── M-Core: буферизований запис ───────────────────────────────────────
        z_program = torch.zeros_like(z_fused)
        if program_target_facts:
            z_program = self.prover.ground(program_target_facts, z_fused.device).expand(
                z_fused.size(0), -1
            )
        z_final, z_graph_readout, z_graph_anchor = self._graph_centered_decoder_state(
            z_fused,
            world_graph_batch,
            program_state=z_program if program_target_facts else None,
        )
        canonical_world_state = self._compose_canonical_world_state(
            z_neural=z_neural,
            z_graph_grounded=z,
            z_graph_readout=z_graph_readout,
            z_graph_anchor=z_graph_anchor,
            z_grounded=z_final,
            z_graph=z_graph,
            z_program=z_program,
            z_symbolic=z_sym,
            v_mem=v_mem,
            has_program_state=bool(program_target_facts),
            world_graph_batch=world_graph_batch,
            task_context=task_context,
        )
        program_anchor_loss = torch.zeros((), device=src.device)
        program_decoder_ce = torch.zeros((), device=src.device)
        if program_target_facts:
            program_anchor_loss = 0.5 * (
                F.mse_loss(z_final, z_program.detach())
                + F.mse_loss(z_final.detach(), z_program)
            )
            program_decoder_logits = self._symbolic_token_logits(z_final, task_context)
            program_decoder_ce = self._safe_cross_entropy(
                program_decoder_logits,
                tgt[:, -1].long(),
                ignore_index=0,
            )

        sym_stats = getattr(self.prover, "last_forward_info", {})
        target_coverage = float(sym_stats.get("target_coverage", 1.0))
        goal_proved = float(sym_stats.get("goal_proved", 1.0))
        surprise = 0.5 * gap_norm.clamp(0, 1) + 0.5 * (1.0 - target_coverage * goal_proved)
        conf = (1.0 - surprise).clamp(0.0, 1.0)
        self.memory.schedule_write(z_final.detach(), world_targets[:, -1].detach(), conf.detach())
        sym_mem_written = self._write_symbolic_memory_facts(
            getattr(self.prover, "last_all_facts", frozenset()),
            float(conf.mean().item()),
        )

        # ══ Рівень 1: Decode  ════════════════════════════════════════════════
        # Ініціалізуємо osf_out заздалегідь — завжди визначена змінна незалежно
        # від того, який шлях декодування обрано (NET / OSF / classic).
        osf_out: Dict = {}

        if self.net_enabled:
            # NET декодер: реконструює оригінальні токени з h_tok + z_final
            net_logits, l_rec = self.net.decode(tgt, z_final, h_tok)  # (B,T,V), scalar
            logits = net_logits
            # L_NET з семантичним feedback: -λ·I(Z;Γ) через sem_pairs_net
            net_loss_dict = self.net.compute_loss(
                net_info, l_rec,
                sem_pairs=sem_pairs_net if sem_pairs_net else None
            )
            net_loss = net_loss_dict.get("net_aux_tensor", net_loss_dict["net_total"])
            if self.osf_enabled:
                osf_logits, osf_out = self.osf(
                    h_tok      = h_tok,
                    z_final    = z_final,
                    tgt        = tgt,
                    world_rnn  = self.world_rnn,
                    gap_norm   = float(gap_norm.mean().item()),
                    ce_loss    = self._osf_running_ce,
                    n_rules    = len(self.prover),
                    n_writes   = self.memory.n_writes,
                    prover     = self.prover,
                    symbolic_goal = getattr(self.prover, "last_goal", None) or task_context.goal,
                    symbolic_facts = getattr(self.prover, "last_context_facts", frozenset()) or task_context.observed_facts,
                )
                logits = 0.8 * net_logits + 0.2 * osf_logits
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
                prover     = self.prover,
                symbolic_goal = getattr(self.prover, "last_goal", None) or task_context.goal,
                symbolic_facts = getattr(self.prover, "last_context_facts", frozenset()) or task_context.observed_facts,
            )
            net_loss      = torch.tensor(0.0, device=src.device)
            net_loss_dict = {}
        else:
            logits   = self.tok_decoder(tgt, z_final)                  # (B, T, V)
            net_loss = torch.tensor(0.0, device=src.device)
            net_loss_dict = {}
            sem_pairs_net = []

        if self._sym_qg_enabled and self.sym_query_gen is not None:
            logits = self.sym_query_gen(
                logits=logits,
                h_tok=h_tok,
                z_sym=z_sym,
                prover=self.prover,
            )

        # ── Loss J(θ,Γ,M) + η_tok·L_NET + δ·VeM + ω_meta·L_AC ──────────────
        rule_bits_raw = self.prover.kb.complexity_penalty()
        rule_utility_adjusted = self.prover.kb.utility_adjusted_penalty(
            getattr(self.cfg, 'eta_utility', 0.1)
        )
        rule_utility_credit = max(float(rule_bits_raw) - float(rule_utility_adjusted), 0.0)
        # 7-ма компонента J_OMEN+EMC: ω_meta·E_τ[Σ_t r_t]
        traj_reward = traj_stats.trajectory_reward if traj_stats is not None else None
        mem_prior_mu, mem_prior_logvar = self._conditional_gaussian_prior(
            v_mem,
            self.memory_prior_mu,
            self.memory_prior_logvar,
        )
        sym_prior_mu, sym_prior_logvar = self._conditional_gaussian_prior(
            z_sym,
            self.symbolic_prior_mu,
            self.symbolic_prior_logvar,
        )
        mem_read_mu = z + self.memory_read_mu(z)
        mem_read_logvar = self.memory_read_logvar(z).clamp(-6.0, 2.0)
        reasoning_cost = 0.0
        if traj_stats is not None:
            reasoning_cost = (
                float(getattr(self.cfg, "emc_lambda_mdl", 0.01)) * float(traj_stats.proof_mdl)
                + float(getattr(self.cfg, "emc_lambda_time", 0.05)) * float(traj_stats.n_steps)
            )
        priors = {
            "mem_mu": mem_prior_mu,
            "mem_logvar": mem_prior_logvar,
            "mem_read_mu": mem_read_mu,
            "mem_read_logvar": mem_read_logvar,
            "sym_mu": sym_prior_mu,
            "sym_logvar": sym_prior_logvar,
            "tok_code_mu": self.token_code_mu.view(1, 1, -1),
            "tok_code_logvar": self.token_code_logvar.view(1, 1, -1).clamp(-6.0, 2.0),
            "conc_code_mu": self.concept_code_mu.view(1, 1, -1),
            "conc_code_logvar": self.concept_code_logvar.view(1, 1, -1).clamp(-6.0, 2.0),
            "world_obs_logvar": self.world_obs_logvar.view(1, 1, 1).clamp(-6.0, 2.0),
        }
        model_bits = self._model_description_bits()
        out = self.loss_fn(
            logits, tgt, z_final, z_mu, z_logvar,
            h_tok, latents,
            z_sim_traj, world_targets, v_mem, z_sym, sym_loss, rule_bits_raw, cf_loss,
            self.world_rnn,
            net_loss,
            priors=priors,
            model_bits=model_bits,
            vem_penalty=vem_pen,
            meta_loss=meta_loss,
            program_anchor=program_anchor_loss,
            program_decoder_ce=program_decoder_ce,
            traj_reward=traj_reward,
            reasoning_cost=reasoning_cost,
            seen_tokens=int(max(int(self._seen_tokens.item()), 1)),
            train_step=int(self._train_step.item()),
        )
        eval_world_update_stats = self._maybe_eval_world_self_update(
            world_loss=world_loss,
            program_anchor_loss=program_anchor_loss,
        )
        query_stats = getattr(self.sym_query_gen, "last_query_info", {}) if self.sym_query_gen is not None else {}
        query_aux_tensor = query_stats.get("aux_loss_tensor")
        if self.training and torch.is_tensor(query_aux_tensor):
            out["total"] = out["total"] + float(getattr(self.cfg, "sym_query_lambda", 0.05)) * query_aux_tensor
        decoder_probe_tensor = None if decoder_signal is None else decoder_signal.get("loss_tensor")
        if self.training and torch.is_tensor(decoder_probe_tensor):
            out["total"] = out["total"] + float(
                getattr(self.cfg, "sym_decoder_surprise_lambda", 0.05)
            ) * decoder_probe_tensor
        out["rule_utility_credit"] = float(rule_utility_credit)
        out["rule_utility_adjusted_bits"] = float(rule_utility_adjusted)

        # ── OSF: додаємо J_OSF до загального лосу ─────────────────────────────
        # osf_out завжди визначений (ініціалізований вище як {}), тому
        # перевірка 'osf_out' in dir() не потрібна — замінено на пряму перевірку ключа.
        if saliency_out is not None:
            out["total"] = out["total"] + saliency_out.sal_total
            out["sal_total"] = float(saliency_out.sal_total.item())
            out["sal_role"] = float(saliency_out.sal_role.item())
            out["sal_struct"] = float(saliency_out.sal_struct.item())
            out["sal_cons_loss"] = float(saliency_out.sal_consistency_loss.item())
            out["sal_rule_pen"] = float(saliency_out.sal_rule_penalty.item())
            out["sal_consistency"] = saliency_out.sal_consistency
            out["sal_tau"] = saliency_out.sal_tau
            out["sal_observed"] = saliency_out.sal_observed
            out["sal_expected"] = saliency_out.sal_expected
            out["sal_edges"] = saliency_out.sal_edges
            out["sal_abduced"] = saliency_out.sal_abduced
            out["sal_role_schema"] = "|".join(saliency_out.sal_role_names)
            out["sal_named_role_preds"] = float(len(getattr(self.saliency, "named_role_predicates", {})))
            out["sal_named_role_rel_preds"] = float(
                len(getattr(self.saliency, "named_role_relation_predicates", {}))
            )

        if self.osf_enabled and "j_osf" in osf_out:
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

        sym_stats = getattr(self.prover, "last_forward_info", {})
        out["sym_goal_proved"] = float(sym_stats.get("goal_proved", 0.0))
        out["sym_target_coverage"] = float(sym_stats.get("target_coverage", 0.0))
        out["sym_target_hits"] = float(sym_stats.get("target_hits", 0.0))
        out["sym_target_total"] = float(sym_stats.get("target_total", 0.0))
        out["sym_unresolved_targets"] = float(sym_stats.get("unresolved_targets", 0.0))
        out["sym_abduced_rules"] = float(sym_stats.get("abduced_rules", 0.0))
        out["sym_abduction_utility"] = float(sym_stats.get("abduction_utility", 0.0))
        out["sym_induction_checked"] = float(sym_stats.get("induction_checked", 0.0))
        out["sym_induction_verified"] = float(sym_stats.get("induction_verified", 0.0))
        out["sym_induction_contradicted"] = float(sym_stats.get("induction_contradicted", 0.0))
        out["sym_induction_retained"] = float(sym_stats.get("induction_retained", 0.0))
        out["sym_induction_repaired"] = float(sym_stats.get("induction_repaired", 0.0))
        out["sym_induction_matches"] = float(sym_stats.get("induction_matches", 0.0))
        out["sym_induction_score"] = float(sym_stats.get("induction_score", 0.0))
        out["sym_cycle_active"] = float(sym_stats.get("cycle_active", 0.0))
        out["sym_cycle_eval_active"] = float(sym_stats.get("cycle_eval_active", 0.0))
        out["sym_cycle_learning_active"] = float(sym_stats.get("cycle_learning_active", 0.0))
        out["sym_cycle_candidate_budget"] = float(sym_stats.get("cycle_candidate_budget", 0.0))
        out["sym_cycle_trace_candidates"] = float(sym_stats.get("cycle_trace_candidates", 0.0))
        out["sym_cycle_contextual_candidates"] = float(sym_stats.get("cycle_contextual_candidates", 0.0))
        out["sym_cycle_neural_candidates"] = float(sym_stats.get("cycle_neural_candidates", 0.0))
        out["sym_cycle_checked"] = float(sym_stats.get("cycle_checked", 0.0))
        out["sym_cycle_accepted"] = float(sym_stats.get("cycle_accepted", 0.0))
        out["sym_cycle_added"] = float(sym_stats.get("cycle_added", 0.0))
        out["sym_cycle_verified"] = float(sym_stats.get("cycle_verified", 0.0))
        out["sym_cycle_contradicted"] = float(sym_stats.get("cycle_contradicted", 0.0))
        out["sym_cycle_retained"] = float(sym_stats.get("cycle_retained", 0.0))
        out["sym_cycle_repaired"] = float(sym_stats.get("cycle_repaired", 0.0))
        out["sym_cycle_error"] = float(sym_stats.get("cycle_error", 0.0))
        out["sym_cycle_symbolic_error"] = float(sym_stats.get("cycle_symbolic_error", 0.0))
        out["sym_cycle_soft_symbolic_error"] = float(sym_stats.get("cycle_soft_symbolic_error", 0.0))
        out["sym_cycle_relaxed_body_error"] = float(sym_stats.get("cycle_relaxed_body_error", 0.0))
        out["sym_cycle_relaxed_head_error"] = float(sym_stats.get("cycle_relaxed_head_error", 0.0))
        out["sym_cycle_trace_error"] = float(sym_stats.get("cycle_trace_error", 0.0))
        out["sym_cycle_counterexample_error"] = float(sym_stats.get("cycle_counterexample_error", 0.0))
        out["sym_cycle_world_error"] = float(sym_stats.get("cycle_world_error", 0.0))
        out["sym_cycle_token_error"] = float(sym_stats.get("cycle_token_error", 0.0))
        out["sym_cycle_graph_energy"] = float(sym_stats.get("cycle_graph_energy", 0.0))
        out["sym_cycle_policy_loss"] = float(sym_stats.get("cycle_policy_loss", 0.0))
        out["sym_cycle_loss"] = float(sym_stats.get("cycle_loss", 0.0))
        out["sym_graph_reasoning_calls"] = float(sym_stats.get("graph_reasoning_calls", 0.0))
        out["sym_graph_reasoning_guided_calls"] = float(sym_stats.get("graph_reasoning_guided_calls", 0.0))
        out["sym_graph_reasoning_fallbacks"] = float(sym_stats.get("graph_reasoning_fallbacks", 0.0))
        out["sym_graph_reasoning_mean_subset"] = float(sym_stats.get("graph_reasoning_mean_subset", 0.0))
        out["sym_graph_reasoning_mean_full_facts"] = float(sym_stats.get("graph_reasoning_mean_full_facts", 0.0))
        out["sym_graph_reasoning_mean_solutions"] = float(sym_stats.get("graph_reasoning_mean_solutions", 0.0))
        out["sym_trace_steps"] = float(sym_stats.get("trace_steps", 0.0))
        out["sym_trace_counterexamples"] = float(sym_stats.get("trace_counterexamples", 0.0))
        out["sym_used_rules"] = float(sym_stats.get("used_rules", 0.0))
        out["sym_cycle_mode"] = sym_stats.get("cycle_mode", "off")
        out["sym_provenance"] = sym_stats.get("provenance", task_context.provenance)
        out["creative_abduction_candidates"] = float(sym_stats.get("creative_abduction_candidates", 0.0))
        out["creative_analogy_candidates"] = float(sym_stats.get("creative_analogy_candidates", 0.0))
        out["creative_metaphor_candidates"] = float(sym_stats.get("creative_metaphor_candidates", 0.0))
        out["creative_counterfactual_candidates"] = float(sym_stats.get("creative_counterfactual_candidates", 0.0))
        out["creative_ame_total_candidates"] = float(sym_stats.get("creative_ame_total_candidates", 0.0))
        out["creative_ontology_candidates"] = float(sym_stats.get("creative_ontology_candidates", 0.0))
        out["creative_selected_rules"] = float(sym_stats.get("creative_selected_rules", 0.0))
        out["creative_intrinsic_value"] = float(sym_stats.get("creative_intrinsic_value", 0.0))
        out["creative_analogy_projector_loss"] = float(sym_stats.get("creative_analogy_projector_loss", 0.0))
        out["creative_analogy_embedding_source"] = float(
            sym_stats.get("creative_analogy_embedding_source", 0.0)
        )
        out["creative_oee_model_initialized"] = float(
            sym_stats.get("creative_oee_model_initialized", 0.0)
        )
        out["creative_oee_feedback_buffer_size"] = float(
            sym_stats.get("creative_oee_feedback_buffer_size", 0.0)
        )
        out["creative_oee_online_train_applied"] = float(
            sym_stats.get("creative_oee_online_train_applied", 0.0)
        )
        out["creative_oee_online_train_loss"] = float(
            sym_stats.get("creative_oee_online_train_loss", 0.0)
        )
        out["creative_oee_online_train_steps"] = float(
            sym_stats.get("creative_oee_online_train_steps", 0.0)
        )
        out["sym_query_pred"] = float(query_stats.get("pred", -1.0))
        out["sym_query_support"] = float(query_stats.get("support", 0.0))
        out["sym_query_answers"] = float(query_stats.get("n_answers", 0.0))
        out["sym_query_candidates"] = float(query_stats.get("candidate_count", 0.0))
        out["sym_query_fallback"] = float(query_stats.get("fallback", 0.0))
        out["sym_query_hit"] = float(query_stats.get("hit", 0.0))
        out["sym_query_reward"] = float(query_stats.get("reward", 0.0))
        out["sym_query_entropy"] = float(query_stats.get("entropy", 0.0))
        out["sym_query_loss"] = float(query_stats.get("aux_loss", 0.0))
        out["sym_query_hard_mask"] = float(query_stats.get("hard_mask", 0.0))
        out["sym_decoder_pred"] = float(task_context.metadata.get("decoder_pred", -1.0))
        out["sym_decoder_target"] = float(task_context.metadata.get("decoder_target", -1.0))
        out["sym_decoder_miss"] = float(task_context.metadata.get("decoder_miss", 0.0))
        out["sym_decoder_surprise"] = float(task_context.metadata.get("decoder_surprise", 0.0))
        out["sym_decoder_probe_ce"] = float(task_context.metadata.get("decoder_probe_ce", 0.0))
        out["sym_trigger_abduction"] = float(task_context.trigger_abduction)
        out["sym_mem_recalled"] = float(len(recalled_memory_facts))
        out["sym_mem_written"] = float(sym_mem_written)
        ast_lang = str(task_context.metadata.get("ast_lang", ""))
        out["sym_ast_lang_python"] = 1.0 if ast_lang == "python" else 0.0
        out["sym_ast_lang_javascript"] = 1.0 if ast_lang == "javascript" else 0.0
        out["sym_ast_lang_rust"] = 1.0 if ast_lang == "rust" else 0.0
        out["sym_ast_lang_other"] = 1.0 if ast_lang not in ("", "python", "javascript", "rust") else 0.0
        if net_loss_dict:
            out["net_aux"] = float(net_loss_dict.get("net_aux", 0.0))
            out["net_vocab_pen"] = float(net_loss_dict.get("net_vocab_pen", 0.0))

        out["logits"]    = logits
        out["z"]         = canonical_world_state
        out["z_dense"]   = z_final
        out["z_program"] = z_program
        out["z_graph"]   = z_graph
        out["z_graph_readout"] = z_graph_readout
        out["z_graph_struct"] = canonical_world_state.z
        out["world_state"] = canonical_world_state
        out["z_world"] = canonical_world_state.dense_state
        out["program_target_facts"] = float(len(program_target_facts))
        out["world_graph_nodes"] = float(world_graph_batch.metadata.get("mean_nodes", 0.0))
        out["world_graph_edges"] = float(world_graph_batch.metadata.get("mean_edges", 0.0))
        out["world_graph_trace_steps"] = float(world_graph_batch.metadata.get("trace_supervised_steps", 0.0))
        out["world_graph_execution_steps"] = float(world_graph_batch.metadata.get("execution_supervised_steps", 0.0))
        out["world_graph_hidden_fallback_steps"] = float(world_graph_batch.metadata.get("hidden_fallback_steps", 0.0))
        out["world_graph_trace_samples"] = float(world_graph_batch.metadata.get("trace_primary_samples", 0.0))
        out["world_graph_hidden_teacher_applied"] = float(world_graph_batch.metadata.get("hidden_teacher_applied", 0.0))
        out["world_graph_neutral_prior_applied"] = float(world_graph_batch.metadata.get("neutral_prior_applied", 0.0))
        out["world_graph_neutral_prior_steps"] = float(world_graph_batch.metadata.get("neutral_prior_steps", 0.0))
        out["eval_world_self_update_applied"] = float(eval_world_update_stats.get("applied", 0.0))
        out["eval_world_self_update_loss"] = float(eval_world_update_stats.get("loss", 0.0))
        out["eval_world_self_update_grad_norm"] = float(eval_world_update_stats.get("grad_norm", 0.0))
        out["eval_world_self_update_lr"] = float(eval_world_update_stats.get("effective_lr", 0.0))
        out["eval_world_self_update_params"] = float(eval_world_update_stats.get("parameter_tensors", 0.0))
        out["eval_world_self_update_param_elems"] = float(
            eval_world_update_stats.get("parameter_elements", 0.0)
        )
        out["eval_world_self_update_program_weight"] = float(
            eval_world_update_stats.get("program_weight", 0.0)
        )
        out["world_graph_gate"] = float(z_graph_gate.mean().item())
        out["z_graph_primary"] = 1.0
        out["z_graph_anchor"] = float(z_graph_anchor.mean().item()) if z_graph_anchor.numel() > 0 else 0.0
        out["z_graph_batch"] = float(canonical_world_state.batch_size)
        out["gap_norm"]  = gap_norm.mean().item()
        out["n_rules"]   = len(self.prover)
        out["n_writes"]  = self.memory.n_writes
        out["pend_writes"] = len(self.memory._buf_s)
        out["unknown_ex"]= self.curiosity.unknown_flag_count
        out["teacher_forcing"] = teacher_forcing_ratio

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

        # ── CE-driven Induce: ЗАМИКАЄМО Abduce→Deduce→Induce цикл ───────────
        # Це ключовий пункт 1 ідеальної реалізації:
        #   «reinforce_recent_rules ПОВИНЕН викликатись після L_ce,
        #    з utility_target пропорційним покращенню перплексії»
        if self.ce_reinforce_enabled and (
            self.training or self.ce_reinforce_eval_enabled
        ):
            self._ce_reinforce(out.get("ce", float("inf")), src.device)
        out["ce_reinforce_utility"] = getattr(self, "_last_ce_utility", 0.0)
        inject_canonical_metadata(out)

        self.prover.clear_task_context()
        return out

    def _generate_symbolic(
        self,
        prompt: torch.Tensor,
        max_new: int,
        temperature: float,
        dynamic_reasoning: bool,
    ) -> torch.Tensor:
        self.eval()
        adaptive_generation = self._generation_online_learning_active()
        generate_info: Dict[str, float] = {
            "adaptive_learning_active": 1.0 if adaptive_generation else 0.0,
            "eval_world_self_update_applied": 0.0,
            "eval_world_self_update_loss": 0.0,
            "eval_world_self_update_grad_norm": 0.0,
            "eval_world_self_update_steps": 0.0,
            "generated_tokens": 0.0,
        }
        self.last_generate_info = dict(generate_info)
        if self.net_enabled:
            h_tok, _, _ = self.net.encode(prompt)
        else:
            h_tok = self.tok_encoder(prompt)
        _, z_det = self.perceiver(h_tok)
        z, _, _ = self._sample_variational_latent(z_det)
        init_world_graph = self._build_world_graph_batch(prompt, saliency_out=None)
        z, _, _ = self._ground_world_state(z, init_world_graph)
        v_mem = self._retrieve_memory(z)
        z_symbolic_in = self._pre_symbolic_state(z, v_mem)
        seed_memory_facts = self._seed_symbolic_memory_facts(prompt)
        recalled_memory_facts = self._recall_symbolic_memory_facts(
            z_symbolic_in,
            hint_facts=seed_memory_facts,
        )
        init_context = self._build_generation_task_context(
            prompt,
            h_tok=h_tok,
            memory_facts=recalled_memory_facts,
        )
        self.prover.set_task_context(init_context)
        if init_context.observed_facts:
            self.prover.load_observed_facts(
                init_context.observed_facts,
                limit=int(getattr(self.cfg, "symbolic_context_max_facts", 96)),
            )

        init_z_sim_traj = None
        init_world_targets = None
        if self.emc_enabled or adaptive_generation:
            init_z_sim_traj, init_world_targets = self._world_rollout_from_hidden(
                h_tok,
                prompt,
                world_graph_batch=init_world_graph,
                teacher_forcing_ratio=0.0,
            )
        if adaptive_generation and init_z_sim_traj is not None and init_world_targets is not None:
            init_update = self._maybe_eval_world_self_update(
                world_loss=F.mse_loss(init_z_sim_traj, init_world_targets)
            )
            generate_info["eval_world_self_update_applied"] += float(init_update.get("applied", 0.0))
            generate_info["eval_world_self_update_loss"] += float(init_update.get("loss", 0.0))
            generate_info["eval_world_self_update_grad_norm"] += float(init_update.get("grad_norm", 0.0))
            generate_info["eval_world_self_update_steps"] += float(init_update.get("applied", 0.0))

        if self.emc_enabled:
            assert init_z_sim_traj is not None
            z_sim0 = init_z_sim_traj[:, -1]
            _, gap_norm0, _ = self.epistemic.compute(z, self.world_rnn, z_sim0)
            z_sym, v_mem_emc = self.emc.run_episode_eval(
                z_symbolic_in, gap_norm0, None, self.prover, self.memory, device=z.device
            )
            if v_mem_emc.norm() > 1e-6:
                v_mem = v_mem_emc
                z_symbolic_in = self._pre_symbolic_state(z, v_mem)
        else:
            z_sym, _ = self.prover(z_symbolic_in, torch.tensor(0.0, device=z.device))

        z_final = self._combine_levels(z_symbolic_in, z_sym, v_mem)
        z_final, _, _ = self._graph_centered_decoder_state(
            z_final,
            init_world_graph,
        )
        generated = prompt.clone()

        for _ in range(max_new):
            ctx = generated[:, -self.cfg.seq_len:]
            if self.net_enabled:
                h_ctx, _, _ = self.net.encode(ctx)
            else:
                h_ctx = self.tok_encoder(ctx)
            _, z_ctx_det = self.perceiver(h_ctx)
            z_ctx, _, _ = self._sample_variational_latent(z_ctx_det)
            step_world_graph = self._build_world_graph_batch(ctx, saliency_out=None)
            z_ctx, _, _ = self._ground_world_state(z_ctx, step_world_graph)
            v_mem_ctx = self._retrieve_memory(z_ctx)
            z_symbolic_ctx = self._pre_symbolic_state(z_ctx, v_mem_ctx)
            step_seed_facts = self._seed_symbolic_memory_facts(ctx)
            recalled_step = self._recall_symbolic_memory_facts(
                z_symbolic_ctx,
                hint_facts=step_seed_facts,
            )
            step_context = self._build_generation_task_context(
                ctx,
                h_tok=h_ctx,
                memory_facts=recalled_step,
            )
            self.prover.set_task_context(step_context)
            if step_context.observed_facts:
                self.prover.load_observed_facts(
                    step_context.observed_facts,
                    limit=int(getattr(self.cfg, "symbolic_context_max_facts", 96)),
                )

            step_z_sim_traj = None
            step_world_targets = None
            if self.emc_enabled or adaptive_generation:
                step_z_sim_traj, step_world_targets = self._world_rollout_from_hidden(
                    h_ctx,
                    ctx,
                    world_graph_batch=step_world_graph,
                    teacher_forcing_ratio=0.0,
                )
            if adaptive_generation and step_z_sim_traj is not None and step_world_targets is not None:
                step_update = self._maybe_eval_world_self_update(
                    world_loss=F.mse_loss(step_z_sim_traj, step_world_targets)
                )
                generate_info["eval_world_self_update_applied"] += float(step_update.get("applied", 0.0))
                generate_info["eval_world_self_update_loss"] += float(step_update.get("loss", 0.0))
                generate_info["eval_world_self_update_grad_norm"] += float(step_update.get("grad_norm", 0.0))
                generate_info["eval_world_self_update_steps"] += float(step_update.get("applied", 0.0))

            if dynamic_reasoning:
                if self.emc_enabled:
                    assert step_z_sim_traj is not None
                    z_sim_step = step_z_sim_traj[:, -1]
                    _, gap_step, _ = self.epistemic.compute(z_ctx, self.world_rnn, z_sim_step)
                    z_sym_step, v_mem_step = self.emc.run_episode_eval(
                        z_symbolic_ctx,
                        gap_step,
                        None,
                        self.prover,
                        self.memory,
                        device=z_ctx.device,
                    )
                else:
                    z_sym_step, _ = self.prover(
                        z_symbolic_ctx, torch.tensor(0.0, device=z_ctx.device)
                    )
                    v_mem_step = v_mem_ctx

                if self.emc_enabled and v_mem_step.norm() > 1e-6:
                    v_mem_use = v_mem_step
                else:
                    v_mem_use = v_mem_ctx
                z_symbolic_ctx = self._pre_symbolic_state(z_ctx, v_mem_use)
                z_final = self._combine_levels(z_symbolic_ctx, z_sym_step, v_mem_use)
                z_final, _, _ = self._graph_centered_decoder_state(
                    z_final,
                    step_world_graph,
                )
                z_sym_for_bias = z_sym_step
            else:
                z_sym_for_bias = z_sym

            h_for_decode = h_ctx if (self.net_enabled or self.osf_enabled) else None
            symbolic_goal = getattr(self.prover, "last_goal", None) or step_context.goal
            symbolic_facts = getattr(self.prover, "last_context_facts", frozenset()) or step_context.observed_facts
            if self.net_enabled:
                net_logits, _ = self.net.decode(ctx, z_final, h_for_decode)
                logits = net_logits
                if self.osf_enabled:
                    _intent_state = self.osf.intent_encoder(z_final)
                    _plan = self.osf.planner(
                        _intent_state,
                        symbolic_goal=symbolic_goal,
                        symbolic_facts=symbolic_facts,
                        prover=self.prover,
                    )
                    osf_logits, _ = self.osf.hier_decoder(
                        h_tok=h_for_decode,
                        z_intent=_intent_state.z_intent,
                        plan=_plan,
                    )
                    logits = 0.8 * net_logits + 0.2 * osf_logits
            elif self.osf_enabled:
                _intent_state = self.osf.intent_encoder(z_final)
                _plan = self.osf.planner(
                    _intent_state,
                    symbolic_goal=symbolic_goal,
                    symbolic_facts=symbolic_facts,
                    prover=self.prover,
                )
                logits, _ = self.osf.hier_decoder(
                    h_tok=h_for_decode,
                    z_intent=_intent_state.z_intent,
                    plan=_plan,
                )
            else:
                logits = self.tok_decoder(ctx, z_final)

            if self._sym_qg_enabled and self.sym_query_gen is not None and h_for_decode is not None:
                logits = self.sym_query_gen(
                    logits=logits,
                    h_tok=h_for_decode,
                    z_sym=z_sym_for_bias,
                    prover=self.prover,
                )

            probs = F.softmax(logits[:, -1] / temperature, -1)
            generated = torch.cat([generated, torch.multinomial(probs, 1)], 1)
            generate_info["generated_tokens"] += 1.0

        if generate_info["eval_world_self_update_steps"] > 0.0:
            scale = generate_info["eval_world_self_update_steps"]
            generate_info["eval_world_self_update_loss"] /= scale
            generate_info["eval_world_self_update_grad_norm"] /= scale
        self.last_generate_info = generate_info
        self.prover.clear_task_context()
        return generated

    # ── Генерація ─────────────────────────────────────────────────────────────
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
        ctx = nullcontext() if self._generation_online_learning_active() else torch.no_grad()
        with ctx:
            return self._generate_symbolic(
                prompt=prompt,
                max_new=max_new,
                temperature=temperature,
                dynamic_reasoning=dynamic_reasoning,
            )

        # ── Ініціальний стан (для dynamic=False або як база) ─────────────────
        if self.net_enabled:
            h_tok, _, _ = self.net.encode(prompt)
        else:
            h_tok = self.tok_encoder(prompt)
        _, z  = self.perceiver(h_tok)
        init_world_graph = self._build_world_graph_batch(prompt, saliency_out=None)
        z, _, _ = self._ground_world_state(z, init_world_graph)
        v_mem = self._retrieve_memory(z)

        # Початкове символьне представлення: через EMC (якщо ввімкнено) або прямо
        if self.emc_enabled:
            # Ініціальний Epistemic Gap для Bellman stopping.
            # FIX Bug-EGD: раніше тут створювався новий EpistemicGapDetector(v2cfg)
            # на кожен виклик generate() — нові (не навчені) ваги + зайва пам'ять.
            # self.epistemic вже існує і compute() є pure-functional (закрита формула),
            # тому використовуємо self.epistemic замість fresh instance.
            z_sim_traj, _ = self._world_rollout_from_hidden(
                h_tok,
                prompt,
                world_graph_batch=init_world_graph,
                teacher_forcing_ratio=0.0,
            )
            z_sim0     = z_sim_traj[:, -1]
            _, gap_norm0, _ = self.epistemic.compute(z, self.world_rnn, z_sim0)
            z_sym, v_mem_emc = self.emc.run_episode_eval(
                z, gap_norm0, None, self.prover, self.memory, device=z.device)
            if v_mem_emc.norm() > 1e-6:
                v_mem = v_mem_emc
        else:
            z_sym, _ = self.prover(z, torch.tensor(0.0, device=z.device))

        z_final = self._combine_levels(z, z_sym, v_mem)

        generated = prompt.clone()
        for _ in range(max_new):
            ctx = generated[:, -self.cfg.seq_len:]
            step_context = self._build_generation_task_context(ctx)
            self.prover.set_task_context(step_context)

            if dynamic_reasoning:
                # ── reasoning_step: перекодуємо + оновлюємо ∂-Prolog / M-Core
                if self.net_enabled:
                    h_ctx, _, _ = self.net.encode(ctx)
                else:
                    h_ctx = self.tok_encoder(ctx)
                _, z_ctx = self.perceiver(h_ctx)
                step_world_graph = self._build_world_graph_batch(ctx, saliency_out=None)
                z_ctx, _, _ = self._ground_world_state(z_ctx, step_world_graph)

                if self.emc_enabled:
                    # ── EMC адаптивне міркування під час inference ────────────
                    # π_meta* = argmax_π E[R_task − λ_time·T − λ_MDL·MDL(proof)]
                    # Bellman зупинка замість фіксованого max_depth.
                    # FIX Bug-EGD: self.epistemic замість fresh EpistemicGapDetector
                    z_sim_step = self._world_rollout_from_hidden(
                        h_ctx,
                        ctx,
                        world_graph_batch=step_world_graph,
                        teacher_forcing_ratio=0.0,
                    )[0][:, -1]
                    _, gap_step, _ = self.epistemic.compute(z_ctx, self.world_rnn, z_sim_step)
                    z_sym_step, v_mem_step = self.emc.run_episode_eval(
                        z_ctx, gap_step, None, self.prover, self.memory,
                        device=z_ctx.device)
                else:
                    # ── Класичний шлях: прямий prover ────────────────────────
                    z_sym_step, _ = self.prover(
                        z_ctx, torch.tensor(0.0, device=z_ctx.device))
                    v_mem_step = self._retrieve_memory(z_ctx)

                if self.emc_enabled and v_mem_step.norm() > 1e-6:
                    v_mem_use = v_mem_step
                else:
                    v_mem_use = self._retrieve_memory(z_ctx)

                z_final      = self._combine_levels(z_ctx, z_sym_step, v_mem_use)
                h_for_decode = h_ctx
            else:
                if self.net_enabled:
                    h_for_decode, _, _ = self.net.encode(ctx)
                elif self.osf_enabled:
                    # OSF потребує h_tok так само як NET → кодуємо через tok_encoder
                    h_for_decode = self.tok_encoder(ctx)   # (B, T, d_tok)
                else:
                    h_for_decode = None   # tok_decoder не потребує h_tok

            symbolic_goal = getattr(self.prover, "last_goal", None) or step_context.goal
            symbolic_facts = getattr(self.prover, "last_context_facts", frozenset()) or step_context.observed_facts
            if self.net_enabled:
                net_logits, _ = self.net.decode(ctx, z_final, h_for_decode)
                logits = net_logits
                if self.osf_enabled:
                    _intent_state = self.osf.intent_encoder(z_final)
                    _plan         = self.osf.planner(
                        _intent_state,
                        symbolic_goal=symbolic_goal,
                        symbolic_facts=symbolic_facts,
                        prover=self.prover,
                    )
                    osf_logits, _ = self.osf.hier_decoder(
                        h_tok    = h_for_decode,
                        z_intent = _intent_state.z_intent,
                        plan     = _plan,
                    )
                    logits = 0.8 * net_logits + 0.2 * osf_logits
            elif self.osf_enabled:
                # ── OSF inference: ієрархічний декодер (без лосів) ───────────
                # Кроки: z_final → IntentEncoder → SymbolicPlanner → HierarchicalDecoder
                # Використовуємо eval-режим (self.training=False → Gumbel=sharpened softmax,
                # planner=greedy argmax, meta_ctrl не активується).
                _intent_state = self.osf.intent_encoder(z_final)
                _plan         = self.osf.planner(
                    _intent_state,
                    symbolic_goal=symbolic_goal,
                    symbolic_facts=symbolic_facts,
                    prover=self.prover,
                )
                logits, _     = self.osf.hier_decoder(
                    h_tok    = h_for_decode,
                    z_intent = _intent_state.z_intent,
                    plan     = _plan,
                )
            else:
                logits = self.tok_decoder(ctx, z_final)

            if self._sym_qg_enabled and self.sym_query_gen is not None and h_for_decode is not None:
                z_sym_for_bias = z_sym_step if dynamic_reasoning else z_sym
                logits = self.sym_query_gen(
                    logits=logits,
                    h_tok=h_for_decode,
                    z_sym=z_sym_for_bias,
                    prover=self.prover,
                )

            probs     = F.softmax(logits[:, -1] / temperature, -1)
            generated = torch.cat([generated, torch.multinomial(probs, 1)], 1)

        self.prover.clear_task_context()
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

def _make_core_compat(cfg: OMENScaleConfig) -> OMENCoreConfig:
    """Будує мінімальний OMENv2Config для компонентів WorldRNN / EGD / Curiosity."""
    return OMENCoreConfig(
        vocab_size        = cfg.vocab_size,
        d_latent          = cfg.d_latent,
        world_rnn_hidden  = cfg.world_rnn_hidden,
        epistemic_tau     = cfg.epistemic_tau,
        epistemic_exact_grad = getattr(cfg, "epistemic_exact_grad", False),
        n_counterfactual  = cfg.n_counterfactual,
    )


def _make_v2_compat(cfg: OMENScaleConfig):
    """Legacy-only bridge for ablations against the historical OMENv2 model."""
    from omen_v2 import OMENv2Config

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
        epistemic_exact_grad = getattr(cfg, "epistemic_exact_grad", False),
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
    for k in (
        "sym_query_pred",
        "sym_query_support",
        "sym_query_answers",
        "sym_query_candidates",
        "sym_query_fallback",
        "sym_query_hit",
        "sym_query_reward",
        "sym_query_entropy",
        "sym_query_loss",
        "sym_decoder_pred",
        "sym_decoder_target",
        "sym_decoder_miss",
        "sym_decoder_surprise",
        "sym_decoder_probe_ce",
        "sym_trigger_abduction",
        "sym_induction_checked",
        "sym_induction_verified",
        "sym_induction_contradicted",
        "sym_induction_repaired",
        "sym_cycle_checked",
        "sym_cycle_accepted",
        "sym_cycle_added",
        "sym_cycle_repaired",
        "sym_cycle_error",
        "sym_cycle_relaxed_body_error",
        "sym_cycle_relaxed_head_error",
        "sym_cycle_trace_error",
        "sym_cycle_counterexample_error",
        "sym_cycle_world_error",
        "sym_cycle_token_error",
        "sym_cycle_loss",
        "sym_trace_steps",
        "sym_trace_counterexamples",
    ):
        assert k in out, f"FAIL: РІС–РґСЃСѓС‚РЅС–Р№ symbolic query key {k}"
    for k in ("total", "total_bits", "bits_per_token", "mdl_seen_tokens", "ce", "world", "world_nll", "kl", "mem_kl", "sym_kl", "memory_read_nll", "reasoning_cost", "free_energy", "fe_obs", "fe_complex", "l_scale", "sym_ground", "gap_norm"):
        assert k in out, f"FAIL: відсутній ключ {k}"
    assert out["logits"].shape == (B, T, cfg.vocab_size), \
        f"logits {out['logits'].shape}"
    assert isinstance(out["z"], CanonicalWorldState), f"z type {type(out['z'])}"
    assert out["z_dense"].shape == (B, cfg.d_latent), f"z_dense {out['z_dense'].shape}"
    assert out["z"] is out["world_state"], "FAIL: canonical world state alias broken"
    assert torch.allclose(out["z_world"], out["z_dense"]), "FAIL: dense world-state view drifted"
    if cfg.osf_enabled:
        for k in ("osf_l_plan", "osf_plan_depth"):
            assert k in out, f"FAIL: відсутній OSF ключ {k}"
    if cfg.net_enabled and cfg.osf_enabled:
        assert "osf_l_refl" in out, "FAIL: NET+OSF не інтегровані у forward()"
    if getattr(cfg, "saliency_enabled", False):
        for k in ("sal_total", "sal_role", "sal_struct", "sal_consistency", "sal_edges"):
            assert k in out, f"FAIL: відсутній Saliency ключ {k}"
    print(
        f"  Forward {ms:.0f} ms  CE={out['ce']:.3f}  "
        f"FE={out['free_energy']:.3f}  obs={out['fe_obs']:.3f}  "
        f"memNLL={out['memory_read_nll']:.3f}  reason={out['reasoning_cost']:.3f}"
    )
    assert 0.0 <= out["sym_query_support"] <= 1.0, f"bad query support {out['sym_query_support']}"
    assert out["sym_cycle_checked"] >= 1.0, "FAIL: continuous hypothesis cycle did not run"
    assert 0.0 <= out["sym_cycle_relaxed_body_error"] <= 1.0, "FAIL: bad relaxed body error"
    assert 0.0 <= out["sym_cycle_relaxed_head_error"] <= 1.0, "FAIL: bad relaxed head error"
    assert 0.0 <= out["sym_cycle_trace_error"] <= 1.0, "FAIL: bad trace error"
    assert 0.0 <= out["sym_cycle_counterexample_error"] <= 1.0, "FAIL: bad counterexample error"
    assert out["sym_graph_reasoning_calls"] >= 0.0, "FAIL: bad graph reasoning telemetry"
    assert out["sym_graph_reasoning_mean_subset"] >= 0.0, "FAIL: bad graph reasoning subset telemetry"
    if not getattr(cfg, "ce_reinforce_enabled", False):
        assert out["ce_reinforce_utility"] == 0.0, "FAIL: CE reinforce leaked into disabled path"
    print(
        f"  gap_norm={out['gap_norm']:.4f}  rules={out['n_rules']}  "
        f"q_sup={out['sym_query_support']:.3f}  q_loss={out['sym_query_loss']:.4f}  "
        f"miss={out['sym_decoder_miss']:.1f}  surprise={out['sym_decoder_surprise']:.3f}  "
        f"cycle={out['sym_cycle_checked']:.0f}/{out['sym_cycle_added']:.0f}  "
        f"repair={out['sym_cycle_repaired']:.0f}  "
        f"ind_v={out['sym_induction_verified']:.1f}  [PASS]"
    )

    sep("TEST S1b · Trace-aware symbolic task context")
    code = "def add(a, b):\n    return a + b\n"
    code_ids = torch.tensor([[ord(ch) for ch in code]], device=DEVICE, dtype=torch.long)
    gap_stub = torch.zeros(1, device=DEVICE)
    hot_stub = torch.zeros(1, cfg.d_latent, device=DEVICE, dtype=torch.bool)
    trace_ctx = model._build_symbolic_task_context(
        code_ids,
        code_ids,
        gap_stub,
        hot_stub,
        saliency_out=None,
        h_tok=None,
        decoder_signal=None,
        memory_facts=None,
    )
    assert trace_ctx.execution_trace is not None, "FAIL: execution trace not attached to task context"
    assert len(trace_ctx.execution_trace.transitions) >= 1, "FAIL: missing primary trace transitions"
    assert len(trace_ctx.execution_trace.counterexamples) >= 1, "FAIL: missing counterexample traces"
    assert len(trace_ctx.target_facts) >= 1, "FAIL: trace targets missing from task context"
    print(
        f"  trace_steps={len(trace_ctx.execution_trace.transitions)}  "
        f"counterexamples={len(trace_ctx.execution_trace.counterexamples)}  "
        f"targets={len(trace_ctx.target_facts)}  [PASS]"
    )

    sep("TEST S1c · Rich code trace semantics")
    rich_code = (
        "def total(xs):\n"
        "    acc = 0\n"
        "    for x in xs:\n"
        "        acc = acc + x\n"
        "    return acc\n\n"
        "nums = [1, 2, 3]\n"
        "first = nums[0]\n"
        "lookup = {'a': 7}\n"
        "value = lookup['a']\n"
        "result = total(nums)\n"
    )
    rich_ids = torch.tensor([[ord(ch) for ch in rich_code]], device=DEVICE, dtype=torch.long)
    rich_ctx = model._build_symbolic_task_context(
        rich_ids,
        rich_ids,
        gap_stub,
        hot_stub,
        saliency_out=None,
        h_tok=None,
        decoder_signal=None,
        memory_facts=None,
    )
    assert rich_ctx.execution_trace is not None, "FAIL: rich trace missing from task context"
    rich_targets = list(rich_ctx.target_facts)
    assert len(rich_ctx.execution_trace.transitions) >= 4, "FAIL: rich trace too short"
    assert any(fact.pred == TRACE_RETURN_EVENT_PRED for fact in rich_targets), "FAIL: rich trace missing return targets"
    assert any(fact.pred == TRACE_BINOP_EVENT_PRED for fact in rich_targets), "FAIL: rich trace missing operator targets"
    print(
        f"  rich_steps={len(rich_ctx.execution_trace.transitions)}  "
        f"rich_counter={len(rich_ctx.execution_trace.counterexamples)}  "
        f"rich_targets={len(rich_targets)}  [PASS]"
    )

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
    abductor_g = sum(p.grad.norm().item() for n,p in model.named_parameters()
                     if "prover.abductor" in n and p.grad is not None)
    graph_unif_g = sum(p.grad.norm().item() for n,p in model.named_parameters()
                       if "prover.graph_unif" in n and p.grad is not None)
    wrnn_g     = sum(p.grad.norm().item() for n,p in model.named_parameters()
                     if "world_rnn" in n and p.grad is not None)
    sal_g = 0.0
    if getattr(cfg, "saliency_enabled", False):
        sal_g = sum(
            p.grad.norm().item()
            for n, p in model.named_parameters()
            if "saliency.role_classifier" in n and p.grad is not None
        )

    print(f"  {enc_label} grad : {enc_g:.4f}")
    print(f"  Perceiver grad      : {perceiver_g:.4f}")
    print(f"  ∂-Prolog grad       : {prover_g:.4f}")
    print(f"  AbductionHead grad  : {abductor_g:.4f}")
    print(f"  GraphUnif grad      : {graph_unif_g:.4f}")
    print(f"  WorldRNN grad       : {wrnn_g:.4f}")
    if getattr(cfg, "saliency_enabled", False):
        print(f"  Saliency grad       : {sal_g:.4f}")
    assert enc_g       > 0, f"FAIL: {enc_label} без граду"
    assert perceiver_g > 0, "FAIL: Perceiver без граду"
    assert abductor_g  > 0, "FAIL: AbductionHead without gradient"
    assert graph_unif_g > 0, "FAIL: GraphMatchingUnifier without gradient"
    if getattr(cfg, "saliency_enabled", False):
        assert sal_g > 0, "FAIL: Saliency role classifier без граду"
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

    sep("TEST S3b · Symbolic memory shares exact and associative recall")
    from omen_prolog import HornAtom, Const
    sym_facts = [
        HornAtom(pred=77, args=(Const(1), Const(2))),
        HornAtom(pred=78, args=(Const(2), Const(3))),
    ]
    sym_embs = model.prover.term_emb(sym_facts, DEVICE)
    n_sym = model.memory.write_symbolic_atoms(sym_facts, sym_embs)
    model.memory.flush()
    recalled_sym = model.memory.recall_symbolic_atoms(sym_embs[:1], top_k=2, min_sim=0.0)
    recalled_vec = model.memory.episodic_recall(sym_embs[:1], k=1)
    recalled_struct = model.memory.recall_symbolic_atoms(
        sym_embs[:1],
        top_k=2,
        min_sim=0.0,
        predicate_hints=[78],
        anchor_values=[3],
    )
    assert n_sym == len(sym_facts), f"FAIL: wrote {n_sym} symbolic atoms"
    assert recalled_sym and recalled_sym[0] == sym_facts[0], f"FAIL: exact recall {recalled_sym}"
    assert recalled_vec.norm().item() > 0.0, "FAIL: associative symbolic recall is empty"
    assert sym_facts[1] in recalled_struct, f"FAIL: structured symbolic recall missed {sym_facts[1]}"
    graph_goal = model.sym_query_gen.generate_query(
        h_last=None,
        sym_vocab=cfg.sym_vocab,
        context_anchor=1,
        symbolic_state=model.prover.ground(frozenset(sym_facts), DEVICE),
        candidate_preds=(77, 78, SEQ_PREDICT_NEXT_PRED),
    )
    assert graph_goal.pred in {77, 78, SEQ_PREDICT_NEXT_PRED}, f"FAIL: graph-conditioned query {graph_goal}"
    unary_facts = [HornAtom(pred=5, args=(Const(9),))]
    unary_candidates = model._queryable_candidate_preds(unary_facts)
    assert unary_candidates == (SEQ_PREDICT_NEXT_PRED,), f"FAIL: unary facts should not become query candidates: {unary_candidates}"
    print(
        f"  exact={len(recalled_sym)}  structured={len(recalled_struct)}  "
        f"assoc_norm={recalled_vec.norm().item():.4f}  [PASS]"
    )

    sep("TEST S3c · Symbolic hard mask stays safe during training")
    from omen_prolog import SymbolicTaskContext, Var as QueryVar
    train_ctx = SymbolicTaskContext(
        observed_facts=frozenset(sym_facts),
        goal=HornAtom(SEQ_PREDICT_NEXT_PRED, (Const(1), QueryVar("NEXT"))),
        metadata={"last_src": 1.0, "last_tgt": 9.0},
    )

    class _FakeProver:
        def __init__(self, ctx, d_latent):
            self.task_context = ctx
            self.last_goal = ctx.goal
            self._d_latent = d_latent

        def ground(self, facts, device):
            return torch.zeros(1, self._d_latent, device=device)

        def answer_query(self, goal, device):
            return (
                torch.zeros(1, self._d_latent, device=device),
                (7,),
                torch.tensor(1.0, device=device),
            )

    fake_prover = _FakeProver(train_ctx, cfg.d_latent)
    fake_logits = torch.zeros(1, 4, cfg.vocab_size, device=DEVICE)
    fake_h = torch.randn(1, 4, cfg.d_tok, device=DEVICE)
    fake_z = torch.randn(1, cfg.d_latent, device=DEVICE)
    model.sym_query_gen.train()
    masked_train = model.sym_query_gen(fake_logits.clone(), fake_h, fake_z, fake_prover)
    assert model.sym_query_gen.last_query_info["hard_mask"] == 0.0, "FAIL: train hard mask should not veto mismatched gold"
    model.sym_query_gen.eval()
    masked_eval = model.sym_query_gen(fake_logits.clone(), fake_h, fake_z, fake_prover)
    assert model.sym_query_gen.last_query_info["hard_mask"] == 1.0, "FAIL: eval hard mask should activate on confident proof"
    assert masked_train[0, -1, 9].item() > -1e3, "FAIL: training veto masked out gold token"
    assert masked_eval[0, -1, 9].item() < -1e3, "FAIL: eval veto did not constrain logits"
    assert masked_eval[0, -1, 7].item() > masked_train[0, -1, 7].item(), "FAIL: train mode still over-trusts mismatched symbolic answer"
    model.sym_query_gen.train()
    print("  training_safe=1  eval_veto=1  [PASS]")

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
    n_add, _, _, _ = prover.abduce_and_learn(z_abd, error=2.0, force=True)
    print(f"  Abduce додав: {n_add} правил  [PASS]")

    sep("TEST S5 · Fixed-bit MDL penalties")
    from omen_prolog import Compound
    small_weights = torch.full((8,), 0.05, device=DEVICE)
    big_weights = torch.full((8,), 3.0, device=DEVICE)
    bits_small = gaussian_tensor_bits(small_weights, sigma=0.1)
    bits_big = gaussian_tensor_bits(big_weights, sigma=0.1)
    assert bits_big > bits_small, "FAIL: parameter bit-cost does not grow with weight magnitude"

    short_rule = HornClause(
        head=HornAtom(pred=1, args=(Var("X"),)),
        body=(HornAtom(pred=2, args=(Var("X"),)),),
    )
    long_rule = HornClause(
        head=HornAtom(pred=1, args=(Var("X"), Compound(3, (Const(4), Var("Y"))))),
        body=(
            HornAtom(pred=2, args=(Var("X"),)),
            HornAtom(pred=5, args=(Compound(6, (Var("Y"), Const(7))),)),
        ),
    )
    short_bits = short_rule.description_length_bits()
    long_bits = long_rule.description_length_bits()
    assert long_bits > short_bits, "FAIL: rule bit-cost does not grow with structural complexity"
    mdl_bits = model._model_description_bits()
    if cfg.net_enabled:
        vocab_bits = model.net.quantizer.vocab_description_bits()
        assert torch.allclose(
            mdl_bits["vocab"],
            vocab_bits.to(device=mdl_bits["vocab"].device, dtype=mdl_bits["vocab"].dtype),
            atol=1e-4,
            rtol=1e-4,
        ), "FAIL: vocab bits in model MDL should match quantizer fixed-code bits"
    print(
        f"  param_bits: small={bits_small.item():.2f} big={bits_big.item():.2f}  "
        f"rule_bits: short={short_bits:.2f} long={long_bits:.2f}  [PASS]"
    )

    sep("TEST S6 · Мінімальне навчання 15 ітерацій")
    model.train()
    ds  = make_counting(64, cfg.seq_len)
    opt = AdamW(model.parameters(), lr=3e-4)
    hist_ce = []
    seen_start = out["mdl_seen_tokens"]
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
    post_train = model(src, tgt)
    assert post_train["mdl_seen_tokens"] > seen_start, "FAIL: seen-token amortization did not advance"
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

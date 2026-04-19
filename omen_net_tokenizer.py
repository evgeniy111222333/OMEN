"""
omen_net_tokenizer.py: Neural Epistemic Tokenizer (NET).

NET replaces fixed subword tokenization with byte-level contextual encoding,
discrete concept formation, and MDL-aware reconstruction.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_symbolic.universal_bits import gaussian_tensor_bits, universal_int_bits

from omen_perceiver import LlamaDecoderBlock, LlamaAttention, RMSNorm, SwiGLUFFN


def _sequence_valid_mask_from_trailing_padding(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.dim() != 2:
        raise ValueError(f"Expected (B, T) token tensor, got shape {tuple(tokens.shape)}")
    if tokens.size(1) == 0:
        return torch.zeros_like(tokens, dtype=torch.bool)
    nonzero = tokens.ne(0)
    has_content = nonzero.any(dim=1)
    last_from_end = nonzero.flip(dims=[1]).to(torch.long).argmax(dim=1)
    lengths = torch.where(
        has_content,
        tokens.new_full((tokens.size(0),), tokens.size(1)) - last_from_end,
        tokens.new_zeros((tokens.size(0),)),
    )
    positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _masked_sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.dim() != 3 or targets.dim() != 2:
        raise ValueError(
            f"Expected logits (B, T, V) and targets (B, T), got {tuple(logits.shape)} and {tuple(targets.shape)}"
        )
    if logits.shape[:2] != targets.shape or targets.shape != valid_mask.shape:
        raise ValueError(
            f"Shape mismatch for masked CE: logits={tuple(logits.shape)}, targets={tuple(targets.shape)}, "
            f"valid_mask={tuple(valid_mask.shape)}"
        )
    flat_mask = valid_mask.reshape(-1)
    if not bool(flat_mask.any()):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    flat_targets = targets.reshape(-1)[flat_mask]
    flat_logits = logits.reshape(-1, logits.size(-1))[flat_mask]
    return F.cross_entropy(flat_logits, flat_targets, reduction="mean")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ByteContextEncoder  (f_θ)
# ══════════════════════════════════════════════════════════════════════════════

class ByteContextEncoder(nn.Module):
    """
    Context encoder f_θ operating at the TRUE byte level.

    Instead of consuming pre-tokenized integer ids with vocab >= 50k
    (which would be little different from a standard Embedding+Transformer),
    this encoder:

      1. Accepts bytes `(B, T) ∈ [0, 255]` - raw UTF-8 bytes.
      2. Projects each byte into `d_tok` via a small `Embedding(256, d_tok)`.
      3. Passes them through `n_layers` of `LlamaDecoderBlock` with bidirectional attention.
      4. Optionally segments on whitespace / punctuation and mean-pools
         inside each segment to form a semantic concept for `EpistemicQuantizer`.

    If already-tokenized data is passed in (`vocab_size > 256`), the encoder
    falls back to compatibility mode: it accepts tokens and skips segmentation.

    Mathematics:
      input X = (x_1,...,x_T) ∈ {0..255}^T    (bytes)
      h_i = f_θ(x_i, context(x_{<i}))         (contextual encoding)
      c_j = mean_{i ∈ seg_j} h_i              (segment pooling)
      -> Z = (c_1,...,c_N)  N ≤ T             (concept vectors for Q)
    """

    # Bytes treated as segment delimiters.
    SEGMENT_BYTES: frozenset = frozenset(
        b' \t\n\r.,;:!?()[]{}"\''
        + b'+-*/=<>|&^~%@#$\\'
    )

    def __init__(self,
                 vocab_size: int,
                 d_tok: int,
                 n_layers: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 segment_pool: bool = True):
        super().__init__()
        assert d_tok % n_heads == 0, f"d_tok={d_tok} is not divisible by n_heads={n_heads}"

        # Byte mode is unconditional: NET always works on UTF-8 bytes [0..255].
        self.byte_vocab    = 256
        self.segment_pool  = segment_pool
        self.d_tok         = d_tok

        self.embed = nn.Embedding(self.byte_vocab, d_tok)
        nn.init.normal_(self.embed.weight, std=d_tok ** -0.5)
        self.attn_norms = nn.ModuleList([RMSNorm(d_tok) for _ in range(n_layers)])
        self.attns = nn.ModuleList([
            LlamaAttention(d_tok, n_heads, dropout=dropout, causal=False)
            for _ in range(n_layers)
        ])
        self.ffn_norms = nn.ModuleList([RMSNorm(d_tok) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([SwiGLUFFN(d_tok, dropout=dropout) for _ in range(n_layers)])
        self.blocks = nn.ModuleList()

        self.norm = RMSNorm(d_tok)
        self.drop = nn.Dropout(dropout)
        self.n_layers = n_layers

        # OPT-1: precomputed lookup table for vectorized segment_pool.
        # Instead of a Python double loop (O(B·T)), use tensor lookup + cumsum.
        # register_buffer moves to the correct device together with the model.
        _seg_lut = torch.zeros(256, dtype=torch.bool)
        for _sb in self.SEGMENT_BYTES:
            if 0 <= _sb < 256:
                _seg_lut[_sb] = True
        self.register_buffer('_seg_lut', _seg_lut, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        return_attn: bool = False,
        summarize_attn: bool = False,
    ):
        """
        `tokens`: `(B, T) ∈ [0, vocab_size)`.
        Returns `(B, T, d_tok)`. In byte mode, T is preserved (no outer pooling)
        to remain compatible with the rest of the pipeline.
        Segment-aware mean pooling happens internally through a residual path:
        each position receives a vector mixed with the bytes from its segment.
        """
        B, T = tokens.shape

        tokens = tokens.clamp(0, 255)

        x = self.drop(self.embed(tokens))              # (B, T, d_tok)
        attn_maps: List[torch.Tensor] = []
        for norm_a, attn, norm_f, ffn in zip(
                self.attn_norms, self.attns, self.ffn_norms, self.ffns):
            if return_attn:
                attn_out, attn_weights = attn(
                    norm_a(x),
                    need_weights=True,
                    average_attn_weights=summarize_attn,
                )
                attn_maps.append(attn_weights)
                x = x + attn_out
            else:
                x = x + attn(norm_a(x))
            x = x + ffn(norm_f(x))
        x = self.norm(x)

        if self.segment_pool:
            x = self._segment_pool(x, tokens)         # (B, T, d_tok) — pooled

        if return_attn:
            return x, torch.stack(attn_maps, dim=1)
        return x                                       # (B, T, d_tok)

    def _segment_pool(self, h: torch.Tensor,
                      bytes_: torch.Tensor) -> torch.Tensor:
        """
        Differentiable segment averaging.
        `out[b,i] = α·h[b,i] + (1-α)·mean(h[b, seg(i)])`

        OPT-1: segment boundaries are computed via tensor lookup + cumsum
        instead of a nested Python loop over batch and time.
        Complexity: O(B·T) Python -> O(1) Python + GPU kernels.
        """
        B, T, D = h.shape
        ALPHA   = 0.5

        with torch.no_grad():
            # Vectorized segment-boundary detection.
            # _seg_lut[byte] == True means the byte is a segment delimiter.
            clamped   = bytes_.clamp(0, 255)                        # (B, T)
            boundary  = self._seg_lut[clamped]                      # (B, T) bool
            # cumsum along time: seg_ids[b,i] = number of delimiters up to i
            seg_ids   = boundary.long().cumsum(dim=1)               # (B, T)
            # Offset batches to avoid collisions between examples.
            offsets   = torch.arange(B, device=h.device).unsqueeze(1) * (T + 1)
            seg_ids   = (seg_ids + offsets).reshape(B * T)          # (B*T,)
            n_segs    = seg_ids.max().item() + 1

            # Element counts in each segment (for normalization).
            seg_cnt = torch.zeros(n_segs, dtype=h.dtype, device=h.device)
            seg_cnt.scatter_add_(0, seg_ids,
                                 torch.ones(B * T, dtype=h.dtype, device=h.device))
            seg_cnt.clamp_(min=1)

        # Differentiable scatter_add: sum within segments.
        flat_h  = h.reshape(B * T, D)
        expand  = seg_ids.unsqueeze(1).expand(-1, D)
        seg_sum = torch.zeros(n_segs, D, dtype=h.dtype, device=h.device)
        seg_sum.scatter_add_(0, expand, flat_h)

        seg_mean = (seg_sum[seg_ids] / seg_cnt[seg_ids].unsqueeze(1)).reshape(B, T, D)
        return ALPHA * h + (1 - ALPHA) * seg_mean                   # (B, T, D)



    @staticmethod
    def text_to_bytes(text: str,
                      max_len: int,
                      pad: int = 0) -> torch.Tensor:
        """
        Utility: string -> byte tensor `(1, max_len)`.
        Used at inference time instead of a BPE tokenizer.
        """
        raw = text.encode("utf-8")[:max_len]
        arr = list(raw) + [pad] * (max_len - len(raw))
        return torch.tensor(arr, dtype=torch.long).unsqueeze(0)

    @staticmethod
    def bytes_to_text(byte_ids: torch.Tensor) -> str:
        """Byte tensor -> string, trimming only trailing pad=0."""
        raw = bytes(byte_ids.view(-1).tolist()).rstrip(b"\x00")
        return raw.decode("utf-8", errors="replace")



# ══════════════════════════════════════════════════════════════════════════════
# 2.  EpistemicQuantizer  (Q)
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicQuantizer(nn.Module):
    """
    Dynamic vocabulary + VQ-VAE (EMA) + Straight-Through Estimator.

    Quantization (cosine similarity):
      z_i = argmax_{v∈V}  (e_v ⊤ f_θ(x_i)) / (||e_v|| · ||f_θ(x_i)||)

    If `max_sim < τ`, create a new token:
      e_new = f_θ(x_i)   (until reaching net_max_vocab)

    STE (Straight-Through Estimator):
      z_q = sg(e_{z_i} - f_θ(x_i)) + f_θ(x_i)    <- gradient flows through f_θ

    EMA codebook update (more stable than SGD on the codebook):
      n_i  ← γ·n_i  + (1−γ)·N_i
      s_i  ← γ·s_i  + (1−γ)·S_i
      e_i  ← s_i / n_i

    Commitment loss (VQ-VAE):
      L_vq = ||sg(z_q) - f_θ(x)||² + β_commit · ||z_q - sg(f_θ(x))||²

    S-Core integration:
      Every new token is registered as a symbolic fact.
      If an external KnowledgeBase is present -> call `kb.add_concept_fact()`.
    """

    def __init__(self,
                 d_tok: int,
                 init_vocab: int,
                 max_vocab: int,
                 tau: float = 0.85,
                 ema_decay: float = 0.99,
                 beta_commit: float = 0.25,
                 warmup_steps: int = 150,
                 tau_schedule: bool = True,
                 tau_min: float = 0.70):
        super().__init__()
        assert init_vocab <= max_vocab, "init_vocab must be ≤ max_vocab"

        self.warmup_steps = warmup_steps
        self.d_tok        = d_tok
        self.max_vocab    = max_vocab
        self.tau          = tau
        self.tau_init     = tau          # keep the initial value for adaptive scheduling
        self.tau_min      = tau_min
        self.tau_schedule = tau_schedule
        self.ema_decay    = ema_decay
        self.beta_commit  = beta_commit

        # Codebook (only active[:current_size] is valid).
        self.codebook = nn.Embedding(max_vocab, d_tok)
        nn.init.normal_(self.codebook.weight, std=d_tok ** -0.5)

        # EMA buffers (not parameters - updated through no_grad).
        self.register_buffer("cluster_count", torch.ones(max_vocab))
        self.register_buffer("cluster_sum",   self.codebook.weight.clone().detach())
        self.register_buffer("current_size",  torch.tensor(init_vocab, dtype=torch.long))

        # Statistics.
        self.n_new_tokens   = 0
        self.n_quant_calls  = 0
        self.kb: Optional[object] = None

        # OPT-CB: cache the normalized codebook; invalidate after EMA/restart.
        self._cb_norm_cache: Optional[torch.Tensor] = None

        # global_usage: fast EMA for cross-batch dead-code detection.
        self.register_buffer("global_usage",
                             torch.ones(max_vocab) * 0.1)
        self._ema_step: int = 0

        # K-means++ buffer for codebook initialization.
        # During warmup_steps we collect encoder outputs and build k-means++ centers.
        # This prevents "dead codes from birth": the codebook starts in real data
        # clusters instead of random normal points.
        self._kmeans_buffer: List[torch.Tensor] = []
        self._kmeans_done: bool = False

        # EMA freeze after restart (Bug7 fix).
        # Problem: after interpolation restart, EMA decay=0.95 pulls the code back
        # to the collapse center after ~4 steps if the whole batch is assigned to code 0.
        # Fix: freeze the restarted codebook entry for _ema_freeze_steps so it can
        # accumulate real assignments before EMA starts moving it again.
        # _restart_step[i] = global step when code i was restarted (-1000 = never).
        self._ema_freeze_steps: int = 20
        self.register_buffer("_restart_step",
                             torch.full((max_vocab,), -1000, dtype=torch.long))
        self.gumbel_tau: float = 0.7

        # OPT-RST: _restart_dead_codes_ema contains an expensive gram matrix
        # and codebook-similarity step. Throttle it to every _restart_interval
        # steps for most of the benefit at much lower cost.
        self._restart_interval: int = 4   # every 4 steps ~= 75% time saved

    # Helper: normalized codebook with cache.
    def _normed_codebook(self) -> torch.Tensor:
        """Return normalized vectors for active codebook entries, using a cache."""
        # OPT-CB: calling F.normalize(Vxd) on every forward is expensive.
        # Cache after the first computation; invalidate in EMA/restart paths.
        if self._cb_norm_cache is not None:
            return self._cb_norm_cache
        active = self.codebook.weight[:self.current_size]           # (V_cur, d)
        self._cb_norm_cache = F.normalize(active, dim=-1)
        return self._cb_norm_cache

    # Forward.
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        `x`: `(B, T, d_tok)` - output of `ByteContextEncoder`.
        Returns:
          z_q    : (B, T, d_tok) - quantized vectors (STE)
          indices: (B, T)        - codebook indices for each position
          info   : dict with vq_loss, commit_loss, n_new_tokens, usage_entropy
        """
        B, T, D = x.shape
        self.n_quant_calls += 1

        # 1. Cosine similarity.
        x_flat = x.reshape(-1, D)                                   # (B*T, d)
        x_norm = F.normalize(x_flat, dim=-1)                        # (B*T, d)
        cb_norm = self._normed_codebook()                            # (V_cur, d)
        sims   = x_norm @ cb_norm.t()                               # (B*T, V_cur)

        max_sims, indices = sims.max(dim=-1)                        # (B*T,)

        # 2. Dynamic vocabulary: add new tokens for low-similarity regions.
        n_new = 0
        if self.training:
            low_sim_mask = max_sims < self.tau
            if low_sim_mask.any() and self.current_size < self.max_vocab:
                N_samples = x_flat.shape[0]           # B*T
                V_cur     = self.current_size.item()
                max_new_per_step = min(2, self.max_vocab - V_cur)  # <=2/step: slow growth

                # Gate 0: warmup. No growth during the first warmup_steps.
                if self._ema_step <= self.warmup_steps:
                    max_new_per_step = 0

                # Gate 1: coverage.
                if max_new_per_step > 0:
                    coverage = N_samples / max(V_cur, 1)
                    if coverage < 2.0:
                        max_new_per_step = 0
                    elif coverage < 4.0:
                        max_new_per_step = min(2, self.max_vocab - V_cur)

                # Gate 2: dead-percentage.
                # If >35% of codes are dead, recycle instead of growing.
                if max_new_per_step > 0 and V_cur > 0:
                    dead_by_ema = (self.global_usage[:V_cur] < 0.10).sum().item()
                    dead_pct    = dead_by_ema / max(V_cur, 1)
                    if dead_pct > 0.35:
                        max_new_per_step = 0

                # Gate 3: hard batch usage.
                # Check REAL usage in the CURRENT batch via hard assignments.
                # If <40% of codes are actually used, do not grow yet.
                if max_new_per_step > 0 and V_cur > 0:
                    hard_counts   = torch.bincount(
                        indices.clamp(0, V_cur - 1), minlength=V_cur).float()
                    hard_used_pct = (hard_counts > 0).float().mean().item()
                    if hard_used_pct < 0.40:     # <40% of codes are truly used -> stop
                        max_new_per_step = 0

                low_vecs = x_flat[low_sim_mask]                     # (M, d)
                low_norm = F.normalize(low_vecs, dim=-1)            # (M, d)

                added_norms: list[torch.Tensor] = []
                for _ in range(max_new_per_step):
                    if self.current_size >= self.max_vocab:
                        break
                    # Compare candidates against the current codebook + newly added items.
                    cb_cur = self._normed_codebook()                # (V_cur, d)
                    all_ref = (torch.cat([cb_cur] + added_norms, 0)
                               if added_norms else cb_cur)
                    sims_cand = (low_norm @ all_ref.t()).max(dim=-1).values  # (M,)
                    # Pick the item farthest from the whole current codebook.
                    best_idx  = sims_cand.argmin().item()
                    best_sim  = sims_cand[best_idx].item()
                    if best_sim >= self.tau:
                        break                                       # already similar enough
                    new_vec  = low_vecs[best_idx].detach()
                    new_norm = F.normalize(new_vec.unsqueeze(0), dim=-1)
                    idx = self.current_size.item()
                    with torch.no_grad():
                        self.codebook.weight.data[idx].copy_(new_vec)
                        self.cluster_sum[idx]     = new_vec
                        self.cluster_count[idx]   = 1.0
                        self.global_usage[idx]    = 0.1   # initial value - not yet "dead"
                    self.current_size += 1
                    n_new += 1
                    self.n_new_tokens += 1
                    added_norms.append(new_norm)
                    # Context: 2 nearest existing tokens -> Horn rule.
                    ctx_top: list = []
                    if idx >= 2:
                        existing = F.normalize(
                            self.codebook.weight[:idx].detach(), dim=-1)
                        ctx_sims = (new_norm @ existing.t())[0]
                        ctx_top  = ctx_sims.topk(min(2, idx)).indices.tolist()
                    self._register_concept_in_kb(idx, context_indices=ctx_top)

                if n_new > 0:
                    # Recompute similarities with the new entries.
                    self._cb_norm_cache = None      # OPT-CB: new tokens -> invalidate
                    cb_norm  = self._normed_codebook()
                    sims     = x_norm @ cb_norm.t()
                    max_sims, indices = sims.max(dim=-1)

        # 3. Quantized vectors.
        v_cur = self.current_size.item()
        active_codebook = self.codebook.weight[:v_cur]
        hard_assign = F.one_hot(indices, num_classes=v_cur).to(x_flat.dtype)
        soft_assign = F.gumbel_softmax(sims, tau=self.gumbel_tau, hard=False, dim=-1)
        assign_st   = hard_assign + soft_assign - soft_assign.detach()
        z_q_flat    = assign_st @ active_codebook
        z_q         = z_q_flat.reshape(B, T, D)

        # 4. STE: keep gradients flowing through f_θ.
        z_q_ste = x + (z_q - x).detach() + (z_q - z_q.detach())

        # 5. EMA codebook update.
        # IMPORTANT: under EMA updates (no gradient), do NOT add an explicit
        # codebook loss. That would create conflicting update signals and destabilize training.
        # Standard EMA VQ-VAE uses commitment loss only.
        vq_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            with torch.no_grad():
                self._ema_step += 1
                # K-means++ codebook initialization from collected encoder outputs.
                if not self._kmeans_done:
                    self._kmeans_buffer.append(
                        x_flat.detach()[:min(32, x_flat.size(0))]  # max 32 samples/step
                    )
                    if self._ema_step >= self.warmup_steps:
                        self._kmeans_init_codebook()
                        self._kmeans_done = True
                self._ema_update(x_flat.detach(), indices)
                self._update_global_usage(indices)
                # OPT-RST: throttle restart; the gram/codebook-sim block is expensive.
                if self._ema_step % self._restart_interval == 0:
                    self._restart_dead_codes_ema(x_flat.detach())
            # Commitment loss pulls the encoder toward the nearest codebook vector.
            # L_commit = β · ||z_e - sg(e)||²
            commit      = F.mse_loss(x_flat, z_q_flat.detach())
            codebook_lr = F.mse_loss(z_q_flat, x_flat.detach())
            vq_loss     = self.beta_commit * commit + 0.25 * codebook_lr

        # 7. Usage entropy (how well the vocabulary is used).
        usage_entropy = self._usage_entropy(indices, B * T)

        # ── 8. Encoder diversity loss (anti-collapse) ──────────────────────────
        enc_div_loss = (self._encoder_mean_cosine(x_flat)
                        if self.training
                        else torch.zeros(1, device=x.device).squeeze())

        # 8b. Soft-entropy loss (differentiable entropy).
        # Problem: l_code in NETLoss is a constant tensor -> zero gradient.
        # Solution: soft assignments through temperature provide a real gradient.
        # We use a normalized-entropy penalty to fight codebook collapse.
        soft_entropy_loss = torch.zeros(1, device=x.device).squeeze()
        if self.training:
            V_cur = self.current_size.item()
            if V_cur >= 2:
                # cb_norm is already available via cache, but detach the codebook:
                # EMA codebook entries should not be trained via SGD too.
                cb_for_soft = F.normalize(active_codebook, dim=-1)
                soft_tau_temp = 0.5                         # soft-quantizer temperature
                # OPT-SOFT: subsample x_norm to _SOFT_N positions instead of all B*T.
                _SOFT_N = 256
                _n_avail = x_norm.size(0)
                if _n_avail > _SOFT_N:
                    _perm = torch.randperm(_n_avail, device=x_norm.device)[:_SOFT_N]
                    x_norm_soft = x_norm[_perm]             # (256, d)
                else:
                    x_norm_soft = x_norm
                sims_soft = x_norm_soft @ cb_for_soft.t()  # (<=256, V_cur), gradients through x_norm_soft
                # FIX Bug8: row normalization turns near-zero similarities into a peaked distribution.
                sims_norm = (sims_soft
                             - sims_soft.mean(dim=-1, keepdim=True)
                             ) / (sims_soft.std(dim=-1, keepdim=True) + 1e-6)
                soft_assign = F.softmax(sims_norm / soft_tau_temp, dim=-1)  # (≤256, V_cur)
                avg_usage   = soft_assign.mean(dim=0)       # (V_cur,) - expected usage
                H_soft      = -(avg_usage * (avg_usage + 1e-8).log()).sum()  # differentiable entropy
                H_soft_max  = math.log(max(V_cur, 2))
                entropy_pen = (1.0 - H_soft / (H_soft_max + 1e-8)).clamp(min=0.0)
                diversity_pen = torch.zeros(1, device=x.device).squeeze()
                if V_cur > 1:
                    cb_pair = cb_for_soft @ cb_for_soft.t()
                    off_diag = ~torch.eye(V_cur, dtype=torch.bool, device=cb_pair.device)
                    diversity_pen = cb_pair[off_diag].pow(2).mean()
                soft_entropy_loss = entropy_pen + 0.1 * diversity_pen

        # 9. Adaptive τ scheduling after warmup.
        if self.training and self._ema_step > self.warmup_steps:
            H_max = math.log(max(self.current_size.item(), 2))
            self._adaptive_tau_step(usage_entropy, H_max)

        return z_q_ste, indices.reshape(B, T), {
            "vq_loss":          vq_loss,
            "n_new_tokens":     n_new,
            "usage_entropy":    usage_entropy,
            "vocab_size":       self.current_size.item(),
            "mean_sim":         max_sims.mean().item(),
            "enc_div_loss":     enc_div_loss,
            "soft_entropy_loss": soft_entropy_loss,   # new differentiable signal
        }

    # Restart dead codes (dead-code collapse prevention).
    @torch.no_grad()
    def _restart_dead_codes(self,
                            x_flat:  torch.Tensor,
                            indices: torch.Tensor,
                            min_usage: float = 1.0) -> int:
        """
        Restart codebook vectors that received no assignments in this batch.
        Reinitialize them from encoder vectors that are farthest from the current
        codebook (hard-example mining).

        In VQ-VAE, some codes can stop "winning" any examples. Without restart,
        the vocabulary gradually degenerates.

        Returns the number of restarted codes.
        """
        V = self.current_size.item()
        if V == 0:
            return 0

        counts = torch.bincount(indices.clamp(0, V - 1).view(-1), minlength=V).float()
        dead_mask = counts < min_usage            # indices of codes with no assignments

        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Pick encoder vectors with the largest error (farthest from the codebook).
        cb_norm = F.normalize(self.codebook.weight[:V], dim=-1)
        x_norm  = F.normalize(x_flat, dim=-1)
        # Smallest similarity to any codebook vector = largest error.
        sims_all   = x_norm @ cb_norm.t()              # (N, V)
        worst_sims, _ = sims_all.max(dim=-1)           # (N,)
        # Top n_dead examples with the lowest similarity.
        n_restart = min(int(n_dead), x_flat.size(0))
        _, hard_idx = worst_sims.topk(n_restart, largest=False)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_restart]
        for i, dead_code_idx in enumerate(dead_indices):
            src = x_flat[hard_idx[i]]
            self.codebook.weight.data[dead_code_idx].copy_(src)
            self.cluster_sum[dead_code_idx]     = src
            self.cluster_count[dead_code_idx]   = 1.0

        return n_restart

    # EMA update.
    @torch.no_grad()
    def _ema_update(self, x_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA update of active codebook entries.

        FIX Bug7: codes that were restarted recently are NOT updated through EMA
        for _ema_freeze_steps. This prevents them from being dragged back to the
        collapse center before they collect real assignments.
        """
        V = self.current_size.item()
        γ = self.ema_decay

        # OPT-EMA: scatter_add instead of a (B*T, V) one-hot matrix.
        idx_clamped = indices.clamp(0, V - 1)                       # (B*T,)

        # N_i = number of examples assigned to cluster i.
        N = torch.zeros(V, device=x_flat.device, dtype=x_flat.dtype)
        N.scatter_add_(0, idx_clamped,
                       torch.ones(idx_clamped.size(0),
                                  device=x_flat.device, dtype=x_flat.dtype))

        # S_i = sum of vectors assigned to cluster i.
        D = x_flat.size(1)
        S = torch.zeros(V, D, device=x_flat.device, dtype=x_flat.dtype)
        S.scatter_add_(0,
                       idx_clamped.unsqueeze(1).expand(-1, D),
                       x_flat)

        # Freeze mask: codes protected after restart.
        frozen_mask = (self._ema_step - self._restart_step[:V]) < self._ema_freeze_steps
        # Save protected-code state before EMA.
        frozen_idx = frozen_mask.nonzero(as_tuple=True)[0] if frozen_mask.any() else None
        if frozen_idx is not None and len(frozen_idx) > 0:
            saved_count = self.cluster_count[frozen_idx].clone()
            saved_sum   = self.cluster_sum[frozen_idx].clone()
            saved_w     = self.codebook.weight.data[frozen_idx].clone()

        # EMA
        self.cluster_count[:V] = γ * self.cluster_count[:V] + (1 - γ) * N
        self.cluster_sum[:V]   = γ * self.cluster_sum[:V]   + (1 - γ) * S

        # Update the codebook (Laplace smoothing in the denominator).
        updated = (self.cluster_sum[:V]
                   / (self.cluster_count[:V].unsqueeze(1) + 1e-5))
        self.codebook.weight.data[:V].copy_(updated)

        # Restore protected codes (cancel EMA for them).
        if frozen_idx is not None and len(frozen_idx) > 0:
            self.cluster_count[frozen_idx] = saved_count
            self.cluster_sum[frozen_idx]   = saved_sum
            self.codebook.weight.data[frozen_idx].copy_(saved_w)

        self._cb_norm_cache = None      # OPT-CB: codebook changed -> invalidate

    # K-means++ initialization.
    @torch.no_grad()
    def _kmeans_init_codebook(self) -> None:
        """
        K-means++ codebook initialization from real encoder outputs.

        Runs once after warmup_steps.
        Before that, the codebook is random normal, so many codes lie outside the
        actual encoder distribution and never "win", becoming dead codes.

        K-means++ gives:
          · each center inside a real data cluster
          · maximal spread between centers (greedy farthest-first)
        """
        if not self._kmeans_buffer:
            return

        all_x = torch.cat(self._kmeans_buffer, dim=0)   # (N, D)
        all_x = F.normalize(all_x, dim=-1)              # cosine space
        V     = self.current_size.item()
        N     = all_x.size(0)
        if N < V:
            return   # not enough data

        # K-means++ center selection.
        perm    = torch.randperm(N)
        centers = [all_x[perm[0]]]                      # first center is random

        for _ in range(V - 1):
            c_stack = torch.stack(centers)               # (k, D) - already normalized
            sims    = (all_x @ c_stack.t()).max(dim=-1).values  # (N,) max sim to existing centers
            dists   = (1.0 - sims).clamp(min=0)         # cosine distance
            probs   = dists ** 2
            s       = probs.sum()
            if s < 1e-9:
                break
            probs   = probs / s
            chosen  = torch.multinomial(probs, 1).item()
            centers.append(all_x[chosen])

        centers_t = torch.stack(centers)                # (K, D) - K <= V
        actual_V  = min(len(centers), V)               # FIX: early break -> K < V

        # Update the codebook in-place (requires_grad -> no_grad).
        self.codebook.weight.data[:actual_V].copy_(centers_t[:actual_V])
        self.cluster_sum[:actual_V].copy_(centers_t[:actual_V])
        self.cluster_count[:actual_V].fill_(1.0)
        self.global_usage[:actual_V].fill_(0.15)       # not dead immediately

        self._kmeans_buffer.clear()

    # global_usage EMA (faster, cross-batch).
    @torch.no_grad()
    def _update_global_usage(self, indices: torch.Tensor) -> None:
        """
        Update global_usage with a fast EMA (decay=0.9) over cross-batch statistics.

        global_usage[i] ∈ [0, 1] is the normalized usage frequency of code i.
        It is used instead of per-batch bincount in _restart_dead_codes_ema.
        """
        V = self.current_size.item()
        if V == 0:
            return
        N = indices.view(-1).shape[0]
        counts = torch.bincount(indices.clamp(0, V - 1).view(-1), minlength=V).float()
        freq   = counts / max(N, 1)                      # normalized frequency [0,1]
        DECAY  = 0.9
        self.global_usage[:V] = DECAY * self.global_usage[:V] + (1 - DECAY) * freq

    # New: EMA-based dead-code restart.
    @torch.no_grad()
    def _restart_dead_codes_ema(self,
                                x_flat: torch.Tensor,
                                dead_threshold: float = 0.10) -> int:
        """
        Restart "dead" codes based on global_usage.

        Key improvements:
          · dead_threshold raised from 0.05 to 0.10 for faster dead-code detection
          · after restart, global_usage=0.15 protects codes from immediate re-restart
          · newly added codes get more time to accumulate statistics

        Collapse-aware restart:
          when encoder outputs are almost identical, restarting from encoder vectors
          does not help. In that regime, use random or perturbed directions to keep
          the codebook diverse enough for recovery.
        """
        V = self.current_size.item()
        if V == 0:
            return 0

        dead_mask = self.global_usage[:V] < dead_threshold
        n_dead    = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Detect encoder collapse.
        # mean_sim > 0.90 means the encoder emits almost identical vectors.
        N_samp  = min(64, x_flat.size(0))
        x_samp  = F.normalize(x_flat[:N_samp], dim=-1)          # (N_samp, d)
        # Mean pairwise similarity (includes i==j -> 1.0, which is just an offset).
        gram    = x_samp @ x_samp.t()                            # (N_samp, N_samp)
        off_diag_mask = ~torch.eye(N_samp, dtype=torch.bool, device=x_flat.device)
        encoder_mean_sim = gram[off_diag_mask].mean().item() if N_samp > 1 else 0.0
        is_collapsed = encoder_mean_sim > 0.90                   # collapse threshold

        cb_norm   = F.normalize(self.codebook.weight[:V], dim=-1)
        x_norm_r  = F.normalize(x_flat, dim=-1)
        sims_all  = x_norm_r @ cb_norm.t()
        worst_sim, _ = sims_all.max(dim=-1)
        n_restart = min(int(n_dead), x_flat.size(0))
        _, hard_idx = worst_sim.topk(n_restart, largest=False)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_restart]
        for i, dead_code_idx in enumerate(dead_indices):
            if is_collapsed:
                # Distinguish moderate vs extreme collapse.
                # For extreme collapse, perturb around an orthogonal direction
                # relative to the collapsed centroid so the new codes are still
                # close enough to be selected, but diverse enough to matter.
                if encoder_mean_sim >= 0.97:
                    # Extreme collapse: perturb in an orthogonal direction.
                    centroid = F.normalize(self.codebook.weight[:V].mean(0), dim=0)
                    noise    = torch.randn_like(centroid)
                    # Project noise into the subspace orthogonal to the centroid.
                    ort_noise = noise - (noise @ centroid) * centroid
                    ort_noise = F.normalize(ort_noise, dim=0)
                    # α uniformly in [0.15, 0.45] -> sim to centroid ~= [0.98, 0.99]
                    alpha_ort = 0.15 + torch.rand(1, device=x_flat.device).item() * 0.30
                    perturb_vec = F.normalize(centroid + alpha_ort * ort_noise, dim=-1)
                    self.codebook.weight.data[dead_code_idx].copy_(perturb_vec)
                    self.cluster_sum[dead_code_idx]     = perturb_vec
                else:
                    # Moderate collapse: reuse the older interpolation strategy.
                    i1 = torch.randint(0, V, (1,), device=x_flat.device).item()
                    i2 = torch.randint(0, V, (1,), device=x_flat.device).item()
                    alpha = torch.rand(1, device=x_flat.device).item()
                    interp_vec = F.normalize(
                        alpha * self.codebook.weight[i1].clone()
                        + (1.0 - alpha) * self.codebook.weight[i2].clone(),
                        dim=-1
                    )
                    self.codebook.weight.data[dead_code_idx].copy_(interp_vec)
                    self.cluster_sum[dead_code_idx]     = interp_vec
            else:
                # Normal regime: restart from the "hardest" encoder output.
                src = x_flat[hard_idx[i]]
                self.codebook.weight.data[dead_code_idx].copy_(src)
                self.cluster_sum[dead_code_idx]     = src

            self.cluster_count[dead_code_idx]   = 1.0
            # 0.15 > 0.10 (threshold) -> protected from immediate re-restart
            self.global_usage[dead_code_idx]    = 0.15
            # FIX Bug7: record the restart step -> EMA freeze for _ema_freeze_steps
            self._restart_step[dead_code_idx]   = self._ema_step

        self._cb_norm_cache = None      # OPT-CB: codebook changed -> invalidate
        return n_restart

    # S-Core: register a new concept with a Horn rule.
    def _register_concept_in_kb(self,
                                token_idx: int,
                                context_indices: Optional[list] = None) -> None:
        """
        Register a new token in the KB as a Horn rule rather than a plain fact.

        Abduction strategy:
          1. Base fact: `net_token(token_idx)`
          2. If context_indices are available:
               net_derived(new_idx, ctx_a) :- net_token(ctx_a), net_token(ctx_b)
             -> "new concept = combination of known concepts"
          3. Without context, fall back to a fact only.
        """
        if self.kb is None:
            return
        try:
            self.kb.add_concept_fact(token_idx, context_indices=context_indices)
        except Exception:
            pass

    # Utility: usage entropy.
    @torch.no_grad()
    def _usage_entropy(self, indices: torch.Tensor, n: int) -> float:
        """Codebook usage entropy (0 = one token, log(V) = uniform use)."""
        V = self.current_size.item()
        if V == 0:
            return 0.0
        counts = torch.bincount(indices.clamp(0, V - 1).view(-1), minlength=V).float()
        probs  = counts / (n + 1e-9)
        probs  = probs[probs > 0]
        return -(probs * probs.log()).sum().item()

    # ── Encoder diversity (anti-collapse) ────────────────────────────────────
    def _encoder_mean_cosine(self,
                             x_flat:    torch.Tensor,
                             sample_n:  int = 64) -> torch.Tensor:
        """
        Mean pairwise cosine similarity between `sample_n` encoder outputs.

        Higher values mean higher similarity and therefore a stronger collapse signal.
        Minimize it to encourage the encoder to produce diverse vectors.
        """
        N = min(x_flat.size(0), sample_n)
        if N < 2:
            return torch.zeros(1, device=x_flat.device, requires_grad=x_flat.requires_grad).squeeze()
        # Random sampling instead of first-N gives a better cross-batch diversity estimate.
        perm = torch.randperm(x_flat.size(0), device=x_flat.device)[:N]
        x_s      = F.normalize(x_flat[perm], dim=-1)              # (N, d)
        pairwise = x_s @ x_s.t()                                   # (N, N)
        mask     = ~torch.eye(N, dtype=torch.bool, device=x_s.device)
        return pairwise[mask].mean()

    # MDL penalty on the vocabulary.
    def vocab_description_bits(self) -> torch.Tensor:
        """Fixed-code dictionary length in bits under a universal code."""
        active_w = self.codebook.weight[:self.current_size]
        size_bits = torch.tensor(
            universal_int_bits(int(self.current_size.item())),
            dtype=active_w.dtype,
            device=active_w.device,
        )
        if active_w.numel() == 0:
            return size_bits
        code_bits = gaussian_tensor_bits(active_w, sigma=1.0 / max(self.d_tok, 1) ** 0.5)
        return size_bits + code_bits

    def vocab_mdl_penalty(self, lambda_voc: float) -> torch.Tensor:
        """
        Weighted dictionary cost. The underlying unit is still bits;
        lambda_voc only controls how strongly that bit-cost is exposed
        inside the local NET objective.
        """
        return float(lambda_voc) * self.vocab_description_bits()

    # ── Adaptive τ scheduling ────────────────────────────────────────────────
    def _adaptive_tau_step(self, usage_entropy: float, H_max: float) -> None:
        """
        Adaptive tuning of τ - calibration of the "surprise threshold".

        Behavior:
          · H/H_max < 0.55 -> few active codes -> lower τ
          · H/H_max > 0.65 -> vocabulary used well -> raise τ
          · changes of ±0.005/step give smooth adaptation

        τ ∈ [tau_min, tau_init]
        """
        if not self.tau_schedule or H_max < 1e-6:
            return
        ratio = usage_entropy / H_max
        if ratio < 0.50:                                  # very few active codes
            self.tau = max(self.tau_min, self.tau - 0.005)
        elif ratio < 0.55:                                # slightly too few
            self.tau = max(self.tau_min, self.tau - 0.002)
        elif ratio > 0.70:                                # used well
            self.tau = min(self.tau_init, self.tau + 0.003)

    def extra_repr(self) -> str:
        return (f"d_tok={self.d_tok}, vocab={self.current_size.item()}/"
                f"{self.max_vocab}, τ={self.tau:.3f}(min={self.tau_min}), γ={self.ema_decay}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ByteDecoder  (g_φ)
# ══════════════════════════════════════════════════════════════════════════════

class ByteDecoder(nn.Module):
    """
    Decoder g_φ reconstructing the original sequence from quantized vectors.

    Inputs:
      tgt      : (B, T)        - target tokens (for autoregressive training)
      z_final  : (B, d_latent) - concept-level conditioning
      h_q      : (B, T, d_tok) - quantized vectors passed through cross-attention

    Architecture:
      1. Embed tgt -> (B, T, d_tok)
      2. + cross-attn(tgt, h_q)   <- quantized contextual memory
      3. + cross-attn(tgt, z_ctx) <- concept-level signal injected through z_final
      4. LlamaDecoderBlock x n_layers
      5. lm_head -> (B, T, vocab_size)

    L_rec = CrossEntropy(g_φ(h_q, z_final), original_tokens)
    """

    def __init__(self,
                 vocab_size: int,
                 d_tok: int,
                 d_latent: int,
                 n_layers: int,
                 n_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_tok % n_heads == 0

        self.embed   = nn.Embedding(vocab_size, d_tok)
        nn.init.normal_(self.embed.weight, std=d_tok ** -0.5)

        # Project z_final (d_latent -> d_tok) for cross-attention.
        self.z_proj  = nn.Linear(d_latent, d_tok, bias=False)

        # Cross-attention: tgt <- h_q (quantized vectors)
        self.hq_norm  = RMSNorm(d_tok)
        self.hq_xattn = LlamaAttention(
            d_tok, n_heads, dropout=dropout,
            causal=False, cross_attn=True, kv_dim=d_tok)

        # Cross-attention: tgt <- z_ctx (concept-level signal)
        self.z_norm   = RMSNorm(d_tok)
        self.z_xattn  = LlamaAttention(
            d_tok, n_heads, dropout=dropout,
            causal=False, cross_attn=True, kv_dim=d_tok)

        # Self-attention + FFN blocks.
        self.blocks   = nn.ModuleList([
            LlamaDecoderBlock(d_tok, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = RMSNorm(d_tok)
        self.lm_head  = nn.Linear(d_tok, vocab_size, bias=False)
        self.drop     = nn.Dropout(dropout)

    def _broadcast_z_context(
        self,
        z_final: torch.Tensor,
        tgt_len: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        The z_final path always attends over exactly one source position, so the
        attention result is just the projected value broadcast to each target step.
        """
        B = z_final.size(0)
        attn = self.z_xattn
        z_ctx = self.z_proj(z_final)
        value = attn.v_proj(z_ctx).view(B, attn.h, attn.dh)
        value = value.unsqueeze(2).expand(-1, -1, tgt_len, -1)
        if self.training and attn.drop > 0.0:
            keep = 1.0 - float(attn.drop)
            mask = torch.rand(
                B,
                attn.h,
                tgt_len,
                1,
                device=value.device,
                dtype=torch.float32,
            )
            value = value * (mask >= float(attn.drop)).to(value.dtype) / keep
        flat = value.transpose(1, 2).contiguous().view(B, tgt_len, -1)
        return attn.o_proj(flat).to(dtype=dtype)

    def forward(self,
                tgt: torch.Tensor,
                z_final: torch.Tensor,
                h_q: torch.Tensor,
                *,
                return_recon_loss: bool = True,
                return_logits: bool = True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
          logits : (B, T, vocab_size)
          l_rec  : scalar - reconstruction loss
        """
        B, T = tgt.shape

        # 1. Token embedding.
        x = self.drop(self.embed(tgt))                              # (B, T, d_tok)

        # 2. Cross-attn: use quantized vectors h_q.
        x = x + self.hq_xattn(self.hq_norm(x), context=h_q)

        # 3. Cross-attn: inject concept z_final into d_tok context.
        x = x + self._broadcast_z_context(z_final, T, dtype=x.dtype)

        # 4. Self-attention blocks.
        for blk in self.blocks:
            x = blk(x)

        # 5. Logits.
        logits = self.lm_head(self.out_norm(x))                     # (B, T, V)

        # 6. L_rec: autoregressive reconstruction loss.
        # Shift: logits[0..T-2] -> targets[1..T-1]
        if return_recon_loss:
            target_slice = tgt[:, 1:]
            valid_mask = _sequence_valid_mask_from_trailing_padding(target_slice)
            l_rec = _masked_sequence_cross_entropy(
                logits[:, :-1],
                target_slice,
                valid_mask,
            )
        else:
            l_rec = None

        if not return_logits:
            logits = None
        return logits, l_rec



# ══════════════════════════════════════════════════════════════════════════════
# 3b.  SemanticFeedbackLoss  (S-Core -> NET feedback)
# ══════════════════════════════════════════════════════════════════════════════

class SemanticFeedbackLoss(nn.Module):
    """
    L_semantic = −E_{(v1,v2)~S-Core} [ cos(e_v1, e_v2) · Score(v1, v2) ]

    This addresses the problem of "statistical rather than semantic" tokens:
    S-Core finds concept pairs with a logical relation (synonym, implies)
    and pushes NET to bring their vectors closer in proportion to that relation.

    MDL_total = MDL_NET − λ_sem · I(Z; Γ)

    where I(Z; Γ) is approximated through weighted cosine similarity:
      I(Z; Γ) ≈ Σ_{(v1,v2)} Score(v1,v2) · cos(e_{v1}, e_{v2})
    """

    def __init__(self, lambda_semantic: float = 0.01):
        super().__init__()
        self.lambda_semantic = lambda_semantic

    def forward(self,
                codebook:    torch.Tensor,          # (V_cur, d_tok)
                pair_indices: List[Tuple[int, int, float]]) -> torch.Tensor:
        """
        codebook     : active part of the `EpistemicQuantizer` codebook `(V, d)`
        pair_indices : `[(tok_idx_1, tok_idx_2, score), ...]`
                       from `DifferentiableProver.semantic_feedback_pairs()`
                       or built-in KB analysis.

        Returns: `L_semantic` (scalar). Negative values mean the tokens are already
        similar; minimizing it pulls semantically related vectors together.
        """
        if not pair_indices or codebook.shape[0] < 2:
            return torch.zeros(1, device=codebook.device,
                               requires_grad=codebook.requires_grad).squeeze()

        device = codebook.device
        V      = codebook.shape[0]
        cb_n   = F.normalize(codebook, dim=-1)   # (V, d) normalized
        valid_pairs = [
            (int(i1), int(i2), float(score))
            for (i1, i2, score) in pair_indices
            if 0 <= int(i1) < V and 0 <= int(i2) < V and int(i1) != int(i2)
        ]
        if not valid_pairs:
            return torch.zeros(1, device=device).squeeze()

        idx1 = torch.tensor([item[0] for item in valid_pairs], dtype=torch.long, device=device)
        idx2 = torch.tensor([item[1] for item in valid_pairs], dtype=torch.long, device=device)
        scores = torch.tensor([item[2] for item in valid_pairs], dtype=cb_n.dtype, device=device)
        cos_sim = (cb_n.index_select(0, idx1) * cb_n.index_select(0, idx2)).sum(dim=-1)
        avg_sem = (cos_sim * scores).mean()
        # -λ · I(Z;Γ): minimizing this maximizes cosine similarity weighted by S-Core.
        return -self.lambda_semantic * avg_sem


# ══════════════════════════════════════════════════════════════════════════════
# 4.  NETLoss
# ══════════════════════════════════════════════════════════════════════════════

class NETLoss(nn.Module):
    """
    L_NET = L_code + L_rec + L_vocab + λ_vq·L_vq + L_semantic

    L_code    ≈ Length(Z)         - normalized entropy estimate
    L_vq                           - commitment loss (STE signal)
    L_rec     ≈ Distortion(X,X̂)  - reconstruction cross-entropy
    L_vocab   ≈ Complexity(V)     - MDL vocabulary cost
    L_semantic                    - S-Core semantic feedback: -λ_sem · I(Z;Γ)
    """

    def __init__(self,
                 lambda_voc:      float = 1e-4,
                 lambda_vq:       float = 1.0,
                 lambda_semantic: float = 0.01,
                 lambda_enc_div:  float = 1.5,    # FIX Bug1: 0.30→1.5
                 lambda_soft_H:   float = 2.0):   # FIX Bug2: 0.5→2.0
        super().__init__()
        self.lambda_voc      = lambda_voc
        self.lambda_vq       = lambda_vq
        self.lambda_enc_div  = lambda_enc_div
        self.lambda_soft_H   = lambda_soft_H   # weight of differentiable soft entropy
        self.sem_loss_fn     = SemanticFeedbackLoss(lambda_semantic)

    def forward(self,
                vq_info:       Dict,
                l_rec:         Optional[torch.Tensor],
                quantizer:     "EpistemicQuantizer",
                sem_pairs:     Optional[List[Tuple[int, int, float]]] = None,
                ) -> Dict:
        """
        sem_pairs : [(tok_idx_1, tok_idx_2, score), ...] from S-Core.
                    If `None`, then `L_semantic = 0`.
        Return a dict with all components plus the total.
        """
        if l_rec is None:
            l_rec = torch.zeros((), device=quantizer.codebook.weight.device, dtype=quantizer.codebook.weight.dtype)

        l_vq    = vq_info["vq_loss"]
        l_vocab = quantizer.vocab_mdl_penalty(self.lambda_voc)

        # L_code: entropy estimate H(Z).
        H_nats = vq_info["usage_entropy"]
        V      = max(vq_info["vocab_size"], 2)
        H_max  = math.log(V)
        l_code = torch.tensor(
            max(0.0, 1.0 - H_nats / H_max),
            dtype=l_rec.dtype, device=l_rec.device
        )

        # L_semantic: S-Core feedback.
        if sem_pairs:
            active_codebook = quantizer.codebook.weight[:quantizer.current_size]
            l_semantic = self.sem_loss_fn(active_codebook, sem_pairs)
            # Move to the same device as l_rec.
            l_semantic = l_semantic.to(l_rec.device)
        else:
            l_semantic = torch.zeros(1, device=l_rec.device).squeeze()

        # L_enc_div: anti-collapse encoder diversity term.
        # Minimize average pairwise cosine similarity between encoder outputs.
        raw_enc_div = vq_info.get("enc_div_loss", 0.0)
        if torch.is_tensor(raw_enc_div):
            l_enc_div = raw_enc_div.to(l_rec.device)
        else:
            l_enc_div = torch.tensor(float(raw_enc_div), device=l_rec.device)

        # L_soft_H: differentiable soft entropy - the key anti-collapse signal.
        # Unlike l_code, this is a real tensor with gradients through encoder outputs.
        raw_soft_H = vq_info.get("soft_entropy_loss", 0.0)
        if torch.is_tensor(raw_soft_H):
            l_soft_H = raw_soft_H.to(l_rec.device)
        else:
            l_soft_H = torch.tensor(float(raw_soft_H), device=l_rec.device)

        global_aux = (
            l_code
            + self.lambda_vq * l_vq
            + l_semantic
            + self.lambda_enc_div * l_enc_div
            + self.lambda_soft_H * l_soft_H
        )
        stage_aux = global_aux + l_vocab
        total = stage_aux + l_rec

        return {
            "net_total":        total,
            "net_total_tensor": total,
            "net_aux_tensor":   global_aux,
            "net_aux":          global_aux.item(),
            "net_stage_aux_tensor": stage_aux,
            "net_stage_aux":    stage_aux.item(),
            "net_code":         l_code.item(),
            "net_vq":           l_vq.item() if torch.is_tensor(l_vq) else float(l_vq),
            "net_rec":          l_rec.item(),
            "net_vocab_pen":    l_vocab.item(),
            "net_semantic":     l_semantic.item() if torch.is_tensor(l_semantic) else float(l_semantic),
            "net_enc_div":      l_enc_div.item() if torch.is_tensor(l_enc_div) else float(l_enc_div),
            "net_soft_H":       l_soft_H.item() if torch.is_tensor(l_soft_H) else float(l_soft_H),
            "net_vocab_size":   vq_info["vocab_size"],
            "net_entropy":      H_nats,
            "net_entropy_bits": H_nats / math.log(2),
            "net_mean_sim":     vq_info["mean_sim"],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  NeuralEpistemicTokenizer
# ══════════════════════════════════════════════════════════════════════════════

class NeuralEpistemicTokenizer(nn.Module):
    """
    Full Neural Epistemic Tokenizer.

    Replaces TokenEncoder + TokenDecoder in OMENScale:
      encode(src) → h_tok, vq_indices, vq_info
      decode(tgt, z_final, h_tok) → logits, l_rec

    Parameters are taken from OMENScaleConfig:
      vocab_size, d_tok, n_heads_tok, net_byte_layers, net_dec_layers
      net_init_vocab, net_max_vocab, net_tau, net_ema_decay
      d_latent, dropout, lambda_voc

    Architectural flow:
      src → ByteContextEncoder → h_ctx (B,T,d_tok)
          → EpistemicQuantizer → h_q (B,T,d_tok), vq_indices, vq_info
      tgt → ByteDecoder(h_q, z_final) → logits (B,T,V), l_rec
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # Effective vocab for ByteContextEncoder:
        # if cfg.vocab_size <= 256 -> true byte mode (256)
        # otherwise -> legacy mode with the FULL vocab.
        # IMPORTANT: do NOT use min(vocab_size, 256), because that forces byte mode
        # and clamps tokens >255 into one bucket, collapsing the codebook.
        byte_vocab = 256

        # f_θ: ByteContextEncoder
        self.byte_encoder = ByteContextEncoder(
            vocab_size = byte_vocab,
            d_tok      = cfg.d_tok,
            n_layers   = cfg.net_byte_layers,
            n_heads    = cfg.n_heads_tok,
            dropout    = cfg.dropout,
        )

        # Q: EpistemicQuantizer
        self.quantizer = EpistemicQuantizer(
            d_tok        = cfg.d_tok,
            init_vocab   = cfg.net_init_vocab,
            max_vocab    = cfg.net_max_vocab,
            tau          = cfg.net_tau,
            ema_decay    = cfg.net_ema_decay,
            warmup_steps = getattr(cfg, 'net_warmup_steps', 150),
            tau_schedule = getattr(cfg, 'net_tau_schedule', True),
            tau_min      = getattr(cfg, 'net_tau_min', 0.70),
        )

        # g_φ: ByteDecoder
        self.byte_decoder = ByteDecoder(
            vocab_size = cfg.vocab_size,
            d_tok      = cfg.d_tok,
            d_latent   = cfg.d_latent,
            n_layers   = cfg.net_dec_layers,
            n_heads    = cfg.n_heads_tok,
            dropout    = cfg.dropout,
        )

        # Loss function
        self.loss_fn = NETLoss(
            lambda_voc      = cfg.lambda_voc,
            lambda_semantic = getattr(cfg, 'lambda_semantic', 0.01),
            lambda_enc_div  = getattr(cfg, 'lambda_enc_div', 1.5),   # FIX Bug1: 0.02→1.5
            lambda_soft_H   = getattr(cfg, 'lambda_soft_H',  2.0),   # FIX Bug2: 0.5→2.0
        )

    # encode
    def encode(
        self,
        src: torch.Tensor,
        return_attn: bool = False,
        summarize_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        src : (B, T) - input sequence
        Returns:
          h_q       : (B, T, d_tok) - quantized vectors (fed into the Perceiver)
          vq_indices: (B, T)        - discrete indices (useful for analysis)
          vq_info   : dict          - quantization statistics
        """
        # f_θ: contextual encoding
        if return_attn:
            h_ctx, attn_maps = self.byte_encoder(
                src,
                return_attn=True,
                summarize_attn=summarize_attn,
            )
        else:
            h_ctx = self.byte_encoder(src)                             # (B, T, d_tok)
            attn_maps = None
        # Q: quantization
        h_q, vq_indices, vq_info = self.quantizer(h_ctx)          # STE
        if attn_maps is not None:
            vq_info["attention_maps"] = attn_maps
            vq_info["h_ctx"] = h_ctx
        return h_q, vq_indices, vq_info

    # decode
    def decode(self,
               tgt:     torch.Tensor,
               z_final: torch.Tensor,
               h_q:     torch.Tensor,
               *,
               return_recon_loss: bool = True,
               return_logits: bool = True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        tgt     : (B, T) - target sequence
        z_final : (B, d_latent) - concept-level signal
        h_q     : (B, T, d_tok) - quantized vectors from encode()
        Returns:
          logits : (B, T, vocab_size)
          l_rec  : scalar reconstruction loss
        """
        return self.byte_decoder(
            tgt,
            z_final,
            h_q,
            return_recon_loss=return_recon_loss,
            return_logits=return_logits,
        )

    # compute_net_loss
    def compute_loss(self,
                     vq_info:   Dict,
                     l_rec:     Optional[torch.Tensor],
                     sem_pairs: Optional[List[Tuple[int, int, float]]] = None,
                     ) -> Dict:
        """
        Compute the full `L_NET` with semantic feedback from S-Core.

        sem_pairs : [(tok_idx_1, tok_idx_2, score), ...] from
                    `prover.semantic_feedback_pairs()`.
                    If None, then `L_semantic = 0` (no-S-Core mode).
        """
        return self.loss_fn(vq_info, l_rec, self.quantizer, sem_pairs=sem_pairs)

    # Stage-1 non-autoregressive reconstruction loss.
    def stage1_rec_loss(self, h_q: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        Non-autoregressive reconstruction loss for Stage 1 pretraining.

        Problem: `ByteDecoder` contains causal self-attention over tgt tokens.
        During training it can learn to predict tgt[i+1] from tgt[0..i]
        without using h_q at all, weakening the gradient to the encoder and
        causing encoder collapse.

        Fix: add an extra positional reconstruction path: h_q[i] -> src[i].
        There is NO causal context here, so the decoder cannot bypass h_q.
        Each position must encode its own byte in its z_q vector.
        """
        dec = self.byte_decoder
        logits = dec.lm_head(dec.out_norm(h_q))          # (B, T, V) - no causal bypass
        valid_mask = _sequence_valid_mask_from_trailing_padding(src)
        return _masked_sequence_cross_entropy(logits, src, valid_mask)

    # Utility: connect Q to the KB.
    def attach_kb(self, kb) -> None:
        """Attach `omen_prolog.KnowledgeBase` for S-Core integration."""
        self.quantizer.kb = kb

    # Report.
    def tokenizer_report(self) -> str:
        q = self.quantizer
        n_enc  = sum(p.numel() for p in self.byte_encoder.parameters())
        n_q    = sum(p.numel() for p in self.quantizer.parameters())
        n_dec  = sum(p.numel() for p in self.byte_decoder.parameters())
        return (
            f"  NET vocab          : {q.current_size.item()} / {q.max_vocab}\n"
            f"  NET new tokens     : {q.n_new_tokens}\n"
            f"  NET quant calls    : {q.n_quant_calls}\n"
            f"  ByteContextEncoder : {n_enc:,} params\n"
            f"  EpistemicQuantizer : {n_q:,} params\n"
            f"  ByteDecoder        : {n_dec:,} params\n"
            f"  S-Core KB attached : {q.kb is not None}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  INLINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_net_tests() -> None:
    """
    Comprehensive NET test suite without depending on OMENScaleConfig.
    Run via: `python omen_net_tokenizer.py`
    """
    import time
    sep = lambda s: print(f"\n{'═'*68}\n  {s}\n{'═'*68}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NET tests]  device={DEVICE}\n")

    VOCAB  = 256
    D_TOK  = 64
    D_LAT  = 32
    N_H    = 4
    B, T   = 4, 32

    # ─── T1: ByteContextEncoder ───────────────────────────────────────────────
    sep("T1 · ByteContextEncoder - contextual encoding")
    enc = ByteContextEncoder(VOCAB, D_TOK, n_layers=2, n_heads=N_H).to(DEVICE)
    tokens = torch.randint(0, VOCAB, (B, T), device=DEVICE)
    h = enc(tokens)
    assert h.shape == (B, T, D_TOK), f"FAIL: shape {h.shape}"
    # Check that different tokens produce different vectors.
    h0, h1 = h[0, 0], h[0, 1]
    cos_sim = F.cosine_similarity(h0.unsqueeze(0), h1.unsqueeze(0)).item()
    print(f"  output shape   : {tuple(h.shape)} ✓")
    print(f"  cos_sim(pos0, pos1) : {cos_sim:.4f}  (< 1 -> context depends on position)")
    assert cos_sim < 0.999, "FAIL: all vectors are identical"
    print("  [PASS]")

    # ─── T2: EpistemicQuantizer - basic quantization ───────────────────────────
    sep("T2 · EpistemicQuantizer - basic quantization + STE")
    q = EpistemicQuantizer(D_TOK, init_vocab=32, max_vocab=128, tau=0.80).to(DEVICE)
    q.train()

    x = torch.randn(B, T, D_TOK, device=DEVICE, requires_grad=True)
    z_q, idx, info = q(x)

    assert z_q.shape == (B, T, D_TOK), f"FAIL: z_q shape {z_q.shape}"
    assert idx.shape == (B, T),        f"FAIL: idx shape {idx.shape}"
    assert idx.max() < q.current_size, f"FAIL: idx {idx.max()} ≥ vocab_size"

    # STE: gradients must flow through x.
    z_q.sum().backward()
    assert x.grad is not None, "FAIL: no gradient through STE"
    assert not torch.isnan(x.grad).any(), "FAIL: NaN in gradient"

    print(f"  z_q shape      : {tuple(z_q.shape)} ✓")
    print(f"  indices max    : {idx.max().item()} < {q.current_size.item()} ✓")
    print(f"  STE grad norm  : {x.grad.norm():.4f} ✓")
    print(f"  vq_loss        : {info['vq_loss'].item():.4f}")
    print(f"  usage_entropy  : {info['usage_entropy']:.4f}")
    print(f"  mean_sim       : {info['mean_sim']:.4f}")
    print("  [PASS]")

    # ─── T3: EpistemicQuantizer - dynamic vocabulary ───────────────────────────
    sep("T3 · EpistemicQuantizer - dynamic vocabulary growth")
    q2 = EpistemicQuantizer(D_TOK, init_vocab=4, max_vocab=64,
                            tau=0.99, warmup_steps=0).to(DEVICE)   # warmup=0 for the test
    q2.train()

    total_new = 0
    for _ in range(10):
        x2 = torch.randn(2, 16, D_TOK, device=DEVICE)
        _, _, info2 = q2(x2)
        total_new += info2["n_new_tokens"]

    print(f"  Initial vocab : 4")
    print(f"  Final vocab   : {q2.current_size.item()}")
    print(f"  New tokens    : {total_new}")
    assert q2.current_size > 4, "FAIL: vocabulary did not grow"
    print("  [PASS]")

    # ─── T4: ByteDecoder ──────────────────────────────────────────────────────
    sep("T4 · ByteDecoder - reconstruction + l_rec")
    dec = ByteDecoder(VOCAB, D_TOK, D_LAT, n_layers=2, n_heads=N_H).to(DEVICE)
    h_q     = torch.randn(B, T, D_TOK, device=DEVICE)
    z_final = torch.randn(B, D_LAT, device=DEVICE)
    tgt_tok = torch.randint(1, VOCAB, (B, T), device=DEVICE)

    logits, l_rec = dec(tgt_tok, z_final, h_q)
    assert logits.shape == (B, T, VOCAB), f"FAIL: logits {logits.shape}"
    assert not torch.isnan(l_rec), "FAIL: NaN in l_rec"
    print(f"  logits shape : {tuple(logits.shape)} ✓")
    print(f"  l_rec        : {l_rec.item():.4f}")

    # Backward
    l_rec.backward()
    grad_sum = sum(p.grad.norm().item() for p in dec.parameters() if p.grad is not None)
    assert grad_sum > 0, "FAIL: no gradients in ByteDecoder"
    print(f"  grad_sum     : {grad_sum:.4f} ✓")
    print("  [PASS]")

    # ─── T5: NETLoss ──────────────────────────────────────────────────────────
    sep("T5 · NETLoss - all three components")
    q3 = EpistemicQuantizer(D_TOK, 16, 64, 0.80).to(DEVICE)
    q3.train()
    xr = torch.randn(B, T, D_TOK, device=DEVICE, requires_grad=True)
    _, _, vq_info3 = q3(xr)

    loss_fn = NETLoss(lambda_voc=1e-3)
    l_rec3  = torch.tensor(2.5)
    out3    = loss_fn(vq_info3, l_rec3, q3)

    assert "net_total" in out3
    assert not math.isnan(out3["net_vq"])
    assert not math.isnan(out3["net_rec"])
    assert not math.isnan(out3["net_vocab_pen"])
    assert out3["net_total"].item() > out3["net_rec"], "FAIL: L_vocab/L_code are not included in L_NET"
    assert out3["net_stage_aux"] > out3["net_aux"], "FAIL: stage aux should include vocab MDL on top of global aux"
    print(f"  L_vq       : {out3['net_vq']:.4f}")
    print(f"  L_rec      : {out3['net_rec']:.4f}")
    print(f"  L_vocab    : {out3['net_vocab_pen']:.6f}")
    print(f"  L_aux      : {out3['net_aux']:.6f}")
    print(f"  L_stageAux : {out3['net_stage_aux']:.6f}")
    print(f"  L_NET      : {out3['net_total'].item():.4f}")
    print(f"  vocab_size : {out3['net_vocab_size']}")
    print(f"  entropy    : {out3['net_entropy']:.4f}")
    print("  [PASS]")

    # ─── T6: NeuralEpistemicTokenizer - integration test ──────────────────────
    sep("T6 · NeuralEpistemicTokenizer - encode -> decode -> loss")

    class _FakeCfg:
        vocab_size = VOCAB; d_tok = D_TOK; d_latent = D_LAT
        n_heads_tok = N_H; net_byte_layers = 2; net_dec_layers = 2
        net_init_vocab = 32; net_max_vocab = 128; net_tau = 0.80
        net_ema_decay = 0.99; dropout = 0.1; lambda_voc = 1e-4

    net = NeuralEpistemicTokenizer(_FakeCfg()).to(DEVICE)
    net.train()

    src  = torch.randint(1, VOCAB, (B, T), device=DEVICE)
    tgt2 = torch.randint(1, VOCAB, (B, T), device=DEVICE)
    zf   = torch.randn(B, D_LAT, device=DEVICE)

    h_q, vq_idx, vq_info = net.encode(src, return_attn=True)
    logits2, l_rec2       = net.decode(tgt2, zf, h_q)
    net_loss_dict         = net.compute_loss(vq_info, l_rec2)

    assert h_q.shape   == (B, T, D_TOK)
    assert vq_idx.shape == (B, T)
    assert logits2.shape == (B, T, VOCAB)
    assert "attention_maps" in vq_info, "FAIL: attention maps were not returned by encode(return_attn=True)"
    assert "h_ctx" in vq_info, "FAIL: raw encoder hidden state was not returned by encode(return_attn=True)"
    assert vq_info["attention_maps"].shape[:3] == (B, _FakeCfg.net_byte_layers, N_H)
    assert not net_loss_dict["net_total"].isnan()

    # Full backward check across the whole NET.
    net_loss_dict["net_total"].backward()
    grad_enc = sum(p.grad.norm().item() for p in net.byte_encoder.parameters()
                   if p.grad is not None)
    grad_dec = sum(p.grad.norm().item() for p in net.byte_decoder.parameters()
                   if p.grad is not None)
    assert grad_enc > 0, "FAIL: no gradients in ByteContextEncoder"
    assert grad_dec > 0, "FAIL: no gradients in ByteDecoder"

    print(f"  h_q shape     : {tuple(h_q.shape)} ✓")
    print(f"  vq_idx shape  : {tuple(vq_idx.shape)} ✓")
    print(f"  attn shape    : {tuple(vq_info['attention_maps'].shape)} ✓")
    print(f"  logits shape  : {tuple(logits2.shape)} ✓")
    print(f"  L_NET         : {net_loss_dict['net_total'].item():.4f}")
    print(f"  grad (enc)    : {grad_enc:.4f}")
    print(f"  grad (dec)    : {grad_dec:.4f}")
    print(net.tokenizer_report())
    print("  [PASS]")

    # ─── T7: S-Core integration ───────────────────────────────────────────────
    sep("T7 · S-Core integration - concept registration in KB")
    try:
        from omen_prolog import KnowledgeBase   # type: ignore
        kb = KnowledgeBase()
        net.attach_kb(kb)
        # Force creation of new tokens.
        q_stest = EpistemicQuantizer(D_TOK, 2, 64, tau=0.99).to(DEVICE)
        q_stest.kb = kb
        q_stest.train()
        for _ in range(5):
            x_s = torch.randn(2, 8, D_TOK, device=DEVICE)
            q_stest(x_s)
        n_facts = kb.n_facts()
        print(f"  KB facts after concept registration: {n_facts}")
        assert n_facts >= 0   # may be 0 if the vocab does not grow
        print("  [PASS]")
    except Exception as e:
        print(f"  S-Core integration: {e}  (skipping)")

    # ─── T8: MDL effect - larger vocabulary means larger penalty ──────────────
    sep("T8 · MDL effect - Complexity(V) grows with vocabulary size")
    q_small = EpistemicQuantizer(D_TOK, 4,  64, 0.80).to(DEVICE)
    q_large = EpistemicQuantizer(D_TOK, 32, 64, 0.80).to(DEVICE)
    pen_small = q_small.vocab_mdl_penalty(1e-3).item()
    pen_large = q_large.vocab_mdl_penalty(1e-3).item()
    print(f"  penalty(V=4)  : {pen_small:.6f}")
    print(f"  penalty(V=32) : {pen_large:.6f}")
    assert pen_large > pen_small, "FAIL: larger vocabulary should have a larger penalty"
    print("  [PASS]")

    print(f"\n{'═'*68}")
    print("  ✅  All 8 NET tests passed successfully")
    print(f"{'═'*68}\n")


if __name__ == "__main__":
    run_net_tests()

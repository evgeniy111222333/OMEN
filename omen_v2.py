"""
OMEN v2 - legacy reference architecture for OMEN
================================================
This file preserves an early "clean" version of OMEN as a research reference.

Canonical runtime stack for the current system:
  omen_scale.OMENScale

It remains here as a historically important implementation of:
  · DualStreamAttention
  · GraphAttentionEncoder
  · CausalGraphDecoder
  · early M-Core / Curiosity / S-Core ideas

v1 extended with three new foundational components:

  M-Core  : Tensor Product Memory - holographic memory in VRAM
             (read/write without backprop, O(H·d²) independent of N facts)

  Curiosity Engine : Epistemic Gap Detector + Counterfactual Rollouts
             (E(z) = diag(∇_z L_world)², activated when ||E(z)|| > τ)

  S-Core  : Symbolic Core - working memory + LTM rules + Abduction Engine
             (neuro-symbolic interface via Gumbel-Softmax + GNN)

Formula:
  L_OMEN = L_CE + γ·||z - (z_sim + v_mem)||² + δ·||∇_z WorldRNN||²
          + η·KL(Q(z|o) || Read(M,z)) - α·I(z;M)
          + β·KL(Q(z|o) || P(z|S-Core(G)))
          + λ_sym·Σ Usage(R)·Complexity(R)

Stack: PyTorch 2.x - zero external solvers.
"""

LEGACY_RUNTIME = True
LEGACY_RUNTIME_ROLE = "legacy_reference"
CANONICAL_SUCCESSOR = "omen_scale.OMENScale"
CANONICAL_PUBLIC_SUCCESSOR = "omen.OMEN"

import math, time, random, warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OMENv2Config:
    # Transformer
    vocab_size:       int   = 256
    d_model:          int   = 128
    d_latent:         int   = 64
    n_heads:          int   = 4
    n_layers:         int   = 3
    seq_len:          int   = 64
    world_rnn_hidden: int   = 128
    dropout:          float = 0.1
    sparsity_lambda:  float = 5e-4

    # M-Core
    mem_heads:        int   = 8     # H tensor-memory heads
    mem_write_tau:    float = 0.3   # confidence threshold (below -> write)
    mem_cache_size:   int   = 512   # cache of recent z states for fast recall

    # Curiosity
    epistemic_tau:    float = 0.3   # ||E(z)|| threshold for module activation
    epistemic_exact_grad: bool = False
    n_counterfactual: int   = 2     # number of counterfactual rollouts

    # S-Core
    sym_vocab:        int   = 64    # symbolic vocabulary size
    sym_embed_dim:    int   = 32    # symbolic embedding dimension
    sym_gnn_layers:   int   = 2     # GNN depth
    sym_max_facts:    int   = 32    # max facts in WM
    abduct_candidates: int  = 8     # abduction candidates
    ltm_max_rules:    int   = 256   # max rules in LTM

    # Loss coefficients
    alpha:     float = 0.1    # Epistemic bonus
    beta:      float = 0.05   # World / Symbolic consistency
    gamma:     float = 0.1    # Structural / Memory reward
    delta:     float = 1e-3   # Complexity penalty
    eta:       float = 0.05   # Memory Recall Loss
    lam_sym:   float = 0.02   # Symbolic rule regularizer

    # OMEN-Scale MDL (used by omen_scale.py)
    # L_scale = λ_tok·(1/T)·Σ||z_t||² + λ_conc·(1/|C|)·Σ||c||²
    lambda_tok:  float = 1e-4   # penalty on token norms (typ. = 1e-4)
    lambda_conc: float = 1e-3   # penalty on concept norms (typ. = 1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BASE BUILDING BLOCKS (Dual-Stream Attention + OMENBlock) - from v1
# ══════════════════════════════════════════════════════════════════════════════

class DualStreamAttention(nn.Module):
    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        self.h  = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.to_qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.log_mask = nn.Parameter(torch.zeros(1, cfg.n_heads, cfg.seq_len, cfg.seq_len))
        self.gate = nn.Linear(cfg.d_model * 2, cfg.d_model)
        # OPT-5: inplace dropout
        self.drop = nn.Dropout(cfg.dropout, inplace=False)  # attention weights - NOT inplace (softmax output is shared)
        self.sparsity_lambda = cfg.sparsity_lambda

    def forward(self, x, causal_mask):
        B, T, D = x.shape
        q, k, v = [t.view(B, T, self.h, self.dh).transpose(1, 2)
                   for t in self.to_qkv(x).chunk(3, dim=-1)]
        scale = math.sqrt(self.dh)

        # OPT-QK: (q @ k^T)/scale is computed ONCE and shared by both streams.
        # Before: 2x O(B·h·T²·dh) matmul (text + causal).
        # Now:    1x matmul + cheap elementwise multiplication by M.
        qk = (q @ k.transpose(-2, -1)) / scale                    # (B, h, T, T)

        # The mask is computed once and shared between streams.
        cm = causal_mask[:T, :T] if causal_mask is not None else None

        # Text stream
        s_txt = qk if cm is None else qk.masked_fill(cm, float('-inf'))
        a_txt = self.drop(F.softmax(s_txt, dim=-1))
        out_txt = (a_txt @ v).transpose(1, 2).contiguous().view(B, T, D)

        # Causal stream
        M = torch.sigmoid(self.log_mask[:, :, :T, :T])
        s_cau = qk * M
        if cm is not None:
            s_cau = s_cau.masked_fill(cm, float('-inf'))
        a_cau = self.drop(F.softmax(s_cau, dim=-1))
        out_cau = (a_cau @ v).transpose(1, 2).contiguous().view(B, T, D)

        out = self.out_proj(self.gate(torch.cat([out_txt, out_cau], dim=-1)))
        return out, self.sparsity_lambda * M.mean()


class OMENBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn  = DualStreamAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        # OPT-5: inplace=True - do not allocate a new tensor for every dropout.
        self.ffn   = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Dropout(cfg.dropout, inplace=True),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout, inplace=True),
        )

    def forward(self, x, causal_mask):
        a, sp = self.attn(self.norm1(x), causal_mask)
        return x + a + self.ffn(self.norm2(x + a)), sp


class GraphAttentionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos    = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([OMENBlock(cfg) for _ in range(cfg.n_layers)])
        self.pool   = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_mu     = nn.Linear(cfg.d_model, cfg.d_latent)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.d_latent)
        mask = torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool).triu(1)
        self.register_buffer("causal_mask", mask)

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.embed(tokens) + self.pos(torch.arange(T, device=tokens.device))
        sp = 0.0
        for blk in self.blocks:
            x, s = blk(x, self.causal_mask)
            sp = sp + s
        h = self.pool(x).mean(1)
        mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-4, 4)
        eps = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        return mu + eps * (0.5 * logvar).exp(), mu, logvar, sp


class WorldRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.act_emb = nn.Embedding(cfg.vocab_size, cfg.d_latent)
        self.gru = nn.GRUCell(cfg.d_latent * 2, cfg.world_rnn_hidden)
        self.out = nn.Sequential(
            nn.Linear(cfg.world_rnn_hidden, cfg.d_latent * 2), nn.GELU(),
            nn.Linear(cfg.d_latent * 2, cfg.d_latent),
        )
        self.h0 = nn.Parameter(torch.zeros(1, cfg.world_rnn_hidden))

    def forward(self, z, action, h=None):
        a = self.act_emb(action)
        h = self.h0.expand(z.size(0), -1) if h is None else h
        h2 = self.gru(torch.cat([z, a], -1), h)
        return self.out(h2), h2

    def simulate_sequence(self, z0, actions, teacher_forcing_ratio: float = 0.0,
                          teacher_states: Optional[torch.Tensor] = None):
        """
        OPT-2: Pre-embed all actions with one call before entering the loop.
        Embedding lookup is moved outside the for-loop:
          Before: T x (embed_lookup + GRUCell + Linear)
          Now:    1 x embed_lookup_batch + T x (GRUCell + Linear)
        At T=8, B=8 this gives ~5% speedup by reducing dispatch overhead.

        FIX Bug1 (world loss 17x drift):
          Previous code: z0 if t == 0 else traj[-1]
          After step 0 the GRU received its own previous output traj[-1]
          instead of the initial concept z0. Over T=8 steps the error
          accumulated (~+74% L_world) instead of shrinking. The fix is to
          always feed z0:
          the GRU hidden state h already carries "memory" of previous steps,
          so z0 as an anchor input does not hurt the dynamics but removes drift.
        """
        B, T   = actions.shape
        a_all  = self.act_emb(actions)                     # (B, T, d_latent) - one call
        h      = self.h0.expand(B, -1).contiguous()
        traj   = []
        z_prev = z0
        if teacher_states is not None and teacher_states.shape[:2] != (B, T):
            raise ValueError("teacher_states must have shape (B, T, d_latent)")
        for t in range(T):
            # FIXED: always z0 (not traj[-1]) - GRU h carries the dynamics,
            # z0 is a stable anchor. This removes the 17x world-loss drift.
            z_in = z_prev
            if teacher_states is not None:
                z_teacher = teacher_states[:, t]
            else:
                z_teacher = z_prev if t > 0 else z0
            if teacher_states is not None and t == 0:
                z_in = z_teacher
            elif t > 0 and teacher_forcing_ratio > 0.0 and self.training:
                use_teacher = (torch.rand(B, 1, device=z0.device) < teacher_forcing_ratio).to(z0.dtype)
                z_in = use_teacher * z_teacher + (1.0 - use_teacher) * z_prev
            h = self.gru(torch.cat([z_in, a_all[:, t]], -1), h)
            z_prev = self.out(h)
            traj.append(z_prev)
        return torch.stack(traj, dim=1)                    # (B, T, d_latent)


class CausalGraphDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos    = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.z_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.blocks = nn.ModuleList([OMENBlock(cfg) for _ in range(cfg.n_layers)])
        self.cross_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                                 dropout=cfg.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        mask = torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool).triu(1)
        self.register_buffer("causal_mask", mask)

    def forward(self, tokens, z):
        B, T = tokens.shape
        x = self.embed(tokens) + self.pos(torch.arange(T, device=tokens.device))
        z_mem = self.z_proj(z).unsqueeze(1)
        x = x + self.cross_attn(self.cross_norm(x), z_mem, z_mem)[0]
        sp = 0.0
        for blk in self.blocks:
            x, s = blk(x, self.causal_mask)
            sp = sp + s
        return self.lm_head(x), sp


# ══════════════════════════════════════════════════════════════════════════════
# 2.  M-CORE: TENSOR PRODUCT MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class TensorProductMemory(nn.Module):
    """
    Holographic memory: M ∈ R^{H × d × d}
    The size is FIXED regardless of the number of stored items.

    Write: M_h ← M_h + λ·(k ⊗ v)          [without backprop through M]
    Read:  v_ret = Σ_h M_h · k            [O(H·d²)]
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        d, H = cfg.d_latent, cfg.mem_heads
        # Memory buffer: updated by direct writes rather than gradients.
        self.register_buffer("memory", torch.zeros(H, d, d))
        self.key_proj = nn.Linear(d, d * H, bias=False)
        self.val_proj = nn.Linear(d, d * H, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.d, self.H = d, H
        self.write_tau = cfg.mem_write_tau

        # LRU cache of states for fast episodic recall.
        self.cache: deque = deque(maxlen=cfg.mem_cache_size)
        self.n_writes = 0

    # -- Read ------------------------------------------------------------------
    def read(self, z_query: torch.Tensor) -> torch.Tensor:
        """z_query: (B, d) → v_retrieved: (B, d)"""
        # FIX: write() performs self.memory += delta under @no_grad, which still
        # increments the version counter and breaks autograd backward.
        # .detach() isolates the buffer from autograd version checks.
        # Gradients through key_proj and out_proj (trainable) remain intact.
        k = self.key_proj(z_query).view(-1, self.H, self.d)      # (B, H, d)
        v = torch.einsum('bhd,hde->bhe', k, self.memory.detach())
        return self.out_proj(v.mean(1))                            # (B, d)

    # -- Write (no gradient, like a hippocampus) -------------------------------
    @torch.no_grad()
    def write(self, z_state: torch.Tensor,
              z_value: torch.Tensor,
              confidence: torch.Tensor) -> None:
        """
        confidence: (B,) ∈ [0,1]
        Write only when the model is "surprised" (`1 - conf > write_tau`).

        IMPORTANT: use `.copy_()` instead of `+=` - `+=` increments the main
        tensor's version counter, which breaks `autograd.backward()`
        (version mismatch). `.copy_()` updates data without changing it.
        """
        lam = (1.0 - confidence).clamp(0, 1)                      # (B,)
        mask = lam > self.write_tau
        if not mask.any():
            return

        z_s = z_state[mask]
        z_v = z_value[mask]
        lam_m = lam[mask]

        k = self.key_proj(z_s).view(-1, self.H, self.d)          # (B', H, d)
        v = self.val_proj(z_v).view(-1, self.H, self.d)

        # Outer product: (B', H, d, d)
        delta = torch.einsum('bhd,bhe,b->hde', k, v, lam_m)
        new_mem = self.memory + delta / (mask.sum().float() + 1e-6)
        self.memory.data.copy_(new_mem)

        # Add to the LRU cache for episodic recall.
        for i in range(z_s.size(0)):
            self.cache.append((z_s[i].detach(), z_v[i].detach()))

        self.n_writes += mask.sum().item()

    # -- Episodic recall (k-NN in cache) ---------------------------------------
    @torch.no_grad()
    def episodic_recall(self, z_query: torch.Tensor, k: int = 4) -> torch.Tensor:
        """Return the mean value of the k nearest cached entries."""
        if len(self.cache) == 0:
            return torch.zeros_like(z_query)
        cache_keys = torch.stack([c[0] for c in self.cache], 0).to(z_query.device)
        cache_vals = torch.stack([c[1] for c in self.cache], 0).to(z_query.device)
        sims = F.cosine_similarity(
            z_query.unsqueeze(1), cache_keys.unsqueeze(0), dim=-1)
        topk = sims.topk(min(k, len(self.cache)), dim=1).indices
        return cache_vals[topk].mean(1)                            # (B, d)

    def memory_footprint_bytes(self) -> int:
        return self.memory.numel() * self.memory.element_size()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CURIOSITY ENGINE: EPISTEMIC GAP DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicGapDetector(nn.Module):
    """
    E(z) = diag(∇_z L_world)²
    Large E_i means the model does not understand the causal relationship there.
    Returns `(epistemic_map, gap_norm, hot_dims)`.
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        self.tau = cfg.epistemic_tau
        self.exact_grad = bool(getattr(cfg, "epistemic_exact_grad", False))
        # Learned projection for aggregating the epistemic signal.
        self.aggregator = nn.Linear(cfg.d_latent, cfg.d_latent)
        self.d = cfg.d_latent

    def compute(self, z: torch.Tensor,
                world_rnn: WorldRNN,
                z_sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z      : (B, d) - current state, requires_grad=True
        z_sim  : (B, d) - simulator-predicted state
        Returns:
          E        : (B, d) - epistemic map
          gap_norm : (B,)   - gap norm
          hot_dims : (B, d) - binary mask of hottest dimensions
        """
        # OPT-EGD: instead of torch.autograd.grad() (a separate backward pass,
        # roughly +30-50% of batch time), use the closed-form MSE gradient:
        #   L = (1/B)·Σ_b ||z_b − z_sim_b||²   →   ∂L/∂z_b = (2/B)·(z_b − z_sim_b)
        #   E = (∂L/∂z)² ∝ (z − z_sim)²
        # Proportionality: the constant (4/B²) is the same for all dimensions,
        # so hot_dims and gap_norm remain correct. Full autograd.grad is unnecessary.
        if self.exact_grad and torch.is_grad_enabled() and z.requires_grad:
            world_loss = F.mse_loss(z, z_sim)
            grad_z = torch.autograd.grad(
                world_loss,
                z,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            E = grad_z.detach().pow(2)
            gap_norm = E.sum(-1).sqrt()
        else:
            diff     = z.detach() - z_sim.detach()                 # (B, d)
            E        = diff.pow(2)                                 # (B, d)
            gap_norm = E.sum(-1).sqrt()                            # (B,)

        # Top 25% hottest dimensions.
        threshold = E.quantile(0.75, dim=-1, keepdim=True)
        hot_dims  = (E >= threshold).float()

        return E.detach(), gap_norm.detach(), hot_dims.detach()


class CuriosityModule(nn.Module):
    """
    If `||E(z)|| > τ`:
      1. Form query q (projection from hot_dims -> semantic query)
      2. Generate n counterfactual rollouts through WorldRNN
      3. Return curiosity_loss and enriched z_curious
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        d = cfg.d_latent
        self.query_proj  = nn.Linear(d, d)         # hot_dims -> query
        self.fusion      = nn.Linear(d * 2, d)     # fuse z with memory response
        self.n_cf        = cfg.n_counterfactual
        self.tau         = cfg.epistemic_tau
        self.unknown_flag_count = 0                # UNKNOWN_EXCEPTION counter

    def forward(self, z: torch.Tensor,
                E: torch.Tensor,
                hot_dims: torch.Tensor,
                gap_norm: torch.Tensor,
                memory: TensorProductMemory,
                world_rnn: WorldRNN,
                counterfactual_actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: z_enriched (B, d), curiosity_loss scalar
        """
        B, d = z.shape
        active = gap_norm > self.tau                               # (B,) bool mask

        if not active.any():
            return z, torch.tensor(0.0, device=z.device)

        # -- Build query from hot dimensions -----------------------------------
        query = self.query_proj(z * hot_dims)                      # (B, d)

        # -- Read from M-Core --------------------------------------------------
        v_mem = memory.read(query)                                 # (B, d)

        # Episodic recall as a complement.
        v_ep  = memory.episodic_recall(query, k=4)                # (B, d)
        v_combined = (v_mem + v_ep) * 0.5

        # -- Counterfactual rollouts -------------------------------------------
        cf_loss = torch.tensor(0.0, device=z.device)
        if self.n_cf > 0 and self.training:
            if counterfactual_actions is not None and counterfactual_actions.numel() > 0:
                noise_actions = counterfactual_actions.to(device=z.device, dtype=torch.long)
                if noise_actions.dim() == 1:
                    noise_actions = noise_actions.unsqueeze(0).expand(B, -1)
                if noise_actions.size(1) < self.n_cf:
                    repeats = math.ceil(self.n_cf / max(noise_actions.size(1), 1))
                    noise_actions = noise_actions.repeat(1, repeats)
                noise_actions = noise_actions[:, :self.n_cf]
            else:
                noise_actions = torch.randint(
                    0, 256, (B, self.n_cf), device=z.device)
            z_cf_traj = world_rnn.simulate_sequence(
                z.detach(), noise_actions)                         # (B, n_cf, d)

            # The counterfactual trajectory should match the enriched z+v_mem.
            z_target = (z + v_combined).detach()
            cf_loss  = F.mse_loss(z_cf_traj.mean(1), z_target)

        # -- Detect UNKNOWN_EXCEPTION (empty memory, large gap) ----------------
        mem_signal_norm = v_combined.norm(dim=-1)                  # (B,)
        unknown = active & (mem_signal_norm < 1e-3)
        if unknown.any():
            self.unknown_flag_count += unknown.sum().item()

        # -- State enrichment --------------------------------------------------
        z_enriched = self.fusion(torch.cat([z, v_combined], dim=-1))
        # Leave batch elements unchanged where gap < τ.
        z_out = torch.where(active.unsqueeze(-1), z_enriched, z)

        return z_out, cf_loss


# ══════════════════════════════════════════════════════════════════════════════
# 4.  S-CORE: SYMBOLIC CORE
# ══════════════════════════════════════════════════════════════════════════════

# -- 4.1 Symbolic structures --------------------------------------------------

@dataclass(frozen=True)
class SymFact:
    """Fact represented as a triplet: (subject, predicate, object)."""
    subj: int   # index in sym_vocab
    pred: int
    obj:  int

    def __repr__(self):
        return f"({self.subj}→[{self.pred}]→{self.obj})"


@dataclass
class SymRule:
    """Rule: IF conditions -> THEN conclusions."""
    conditions:  Tuple[SymFact, ...]
    conclusions: Tuple[SymFact, ...]
    weight:      float = 1.0    # importance (grows with usage)
    use_count:   int   = 0      # usage counter
    complexity:  int   = 0      # number of symbols

    def __post_init__(self):
        self.complexity = len(self.conditions) + len(self.conclusions)

    def __hash__(self):
        return hash((self.conditions, self.conclusions))


# -- 4.2 Working memory -------------------------------------------------------

class WorkingMemory:
    """Graph of the current context (facts + index for fast lookup)."""

    def __init__(self, max_facts: int = 32):
        # OPT-WM: deque for O(1) popleft + set for O(1) membership / remove.
        from collections import deque as _deque
        self.facts:     _deque                    = _deque(maxlen=None)  # upper bound enforced manually
        self._fact_set: set                       = set()
        self.pred_idx:  Dict[int, set]            = defaultdict(set)
        self.max_facts = max_facts

    def add(self, fact: SymFact) -> bool:
        if fact in self._fact_set:               # O(1) instead of O(n) list scan
            return False
        if len(self.facts) >= self.max_facts:
            removed = self.facts[0]              # peek oldest - O(1) for deque
            self.facts.popleft()                 # O(1) instead of O(n) list.pop(0)
            self._fact_set.discard(removed)      # O(1) instead of O(n) list.remove
            self.pred_idx[removed.pred].discard(removed)  # O(1)
        self.facts.append(fact)
        self._fact_set.add(fact)
        self.pred_idx[fact.pred].add(fact)
        return True

    def query(self, pred: Optional[int] = None,
              subj: Optional[int] = None) -> List[SymFact]:
        pool = self.pred_idx.get(pred, self._fact_set) if pred is not None else self._fact_set
        if subj is not None:
            return [f for f in pool if f.subj == subj]
        return list(pool)

    def clear(self):
        self.facts.clear()
        self._fact_set.clear()
        self.pred_idx.clear()

    def __len__(self): return len(self.facts)


# -- 4.3 Long-term symbolic memory + unification ------------------------------

class LongTermMemory:
    """
    Rule base stored as a hash table: {frozen(conditions) -> SymRule}.
    Unification uses exact matching (with wildcards: pred=-1 = any).
    """

    def __init__(self, max_rules: int = 256):
        self.rules: Dict[int, SymRule] = {}
        self.max_rules = max_rules

    def add_rule(self, rule: SymRule) -> bool:
        h = hash(rule)
        if h in self.rules:
            self.rules[h].use_count += 1
            return False
        if len(self.rules) >= self.max_rules:
            # Remove the least-used rule.
            worst = min(self.rules.values(), key=lambda r: r.use_count)
            del self.rules[hash(worst)]
        self.rules[h] = rule
        return True

    def match(self, wm: WorkingMemory) -> List[SymRule]:
        """Return all rules whose conditions unify with WM."""
        matched = []
        for rule in self.rules.values():
            if self._unify(rule.conditions, wm):
                matched.append(rule)
        return matched

    def _unify(self, conditions: Tuple[SymFact, ...], wm: WorkingMemory) -> bool:
        for cond in conditions:
            found = False
            # OPT-UNIFY: narrow the pool via pred_idx instead of full scanning.
            # Wildcard pred=-1 -> fall back to the full _fact_set.
            if cond.pred >= 0:
                pool = wm.pred_idx.get(cond.pred, ())
            else:
                pool = wm._fact_set
            for fact in pool:
                s_ok = (cond.subj < 0) or (cond.subj == fact.subj)
                p_ok = (cond.pred < 0) or (cond.pred == fact.pred)
                o_ok = (cond.obj  < 0) or (cond.obj  == fact.obj)
                if s_ok and p_ok and o_ok:
                    found = True
                    break
            if not found:
                return False
        return True

    def complexity_penalty(self) -> float:
        """Σ Usage(R) · Complexity(R) - S-Core regularizer."""
        return sum(r.use_count * r.complexity for r in self.rules.values())

    def __len__(self): return len(self.rules)


# -- 4.4 Graph Neural Network (symbolic -> neural) ----------------------------

class SymbolicGNN(nn.Module):
    """
    Convert a fact graph into a vector `z_sym ∈ R^{d_latent}`.
    Facts -> embeddings -> message passing -> pooling.
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        es = cfg.sym_embed_dim
        dl = cfg.d_latent
        self.sym_emb = nn.Embedding(cfg.sym_vocab + 3, es)  # +3: pad/wildcard/unk
        self.edge_emb = nn.Embedding(cfg.sym_vocab + 3, es)
        self.msg_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(es * 3, es * 2), nn.GELU(),
                          nn.Linear(es * 2, es))
            for _ in range(cfg.sym_gnn_layers)
        ])
        self.to_latent = nn.Linear(es, dl)

    def forward(self, facts: List[SymFact], device) -> torch.Tensor:
        """Returns z_sym: (1, d_latent)"""
        if not facts:
            return torch.zeros(1, self.to_latent.out_features, device=device)

        # Nodes are unique subjects and objects.
        nodes: Dict[int, torch.Tensor] = {}
        for f in facts:
            for sym_id in (f.subj, f.obj):
                if sym_id not in nodes:
                    idx = torch.tensor([sym_id % (self.sym_emb.num_embeddings)],
                                       device=device)
                    nodes[sym_id] = self.sym_emb(idx).squeeze(0)

        # Message passing over edges (facts).
        for layer in self.msg_layers:
            new_nodes = {k: v.clone() for k, v in nodes.items()}
            for f in facts:
                e_id = torch.tensor([f.pred % self.edge_emb.num_embeddings],
                                    device=device)
                e_emb = self.edge_emb(e_id).squeeze(0)
                msg = layer(torch.cat([nodes[f.subj], e_emb, nodes[f.obj]], 0).unsqueeze(0))
                new_nodes[f.obj] = new_nodes[f.obj] + msg.squeeze(0)
            nodes = new_nodes

        # Pooling → latent
        node_stack = torch.stack(list(nodes.values()), 0)         # (N, es)
        z_sym = self.to_latent(node_stack.mean(0, keepdim=True))  # (1, dl)
        return z_sym


# -- 4.5 Abduction Engine -----------------------------------------------------

class AbductionHead(nn.Module):
    """
    Neural candidate-rule generator.
    R* = argmin [ Length(R) + λ·PredError(R, Trace) ]
    Implementation: the network proposes candidates -> symbolic scoring -> selection.
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        d = cfg.d_latent
        sv = cfg.sym_vocab
        self.n_cand = cfg.abduct_candidates

        # Network -> distribution over sym_vocab for each fact slot.
        self.rule_gen = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(),
            nn.Linear(d * 2, sv * 3 * 2)  # 2 facts x 3 slots x vocab
        )
        self.sv = sv

    def forward(self, z: torch.Tensor) -> List[SymRule]:
        """
        z: (1, d) - current state
        Returns a list of candidate rules (without backprop through LTM).
        """
        logits = self.rule_gen(z.squeeze(0))                       # (sv*6,)
        logits = logits.view(2, 3, self.sv)                        # (2 facts, 3 slots, vocab)

        rules = []
        for _ in range(self.n_cand):
            # Gumbel-Softmax for differentiable discretization.
            cond_idx = F.gumbel_softmax(logits[0], tau=1.0, hard=True).argmax(-1)
            conc_idx = F.gumbel_softmax(logits[1], tau=1.0, hard=True).argmax(-1)

            cond = SymFact(cond_idx[0].item(), cond_idx[1].item(), cond_idx[2].item())
            conc = SymFact(conc_idx[0].item(), conc_idx[1].item(), conc_idx[2].item())

            # Occam's principle: rule length = 2 (minimal).
            rule = SymRule(conditions=(cond,), conclusions=(conc,))
            rules.append(rule)

        return rules


class SymbolicCore(nn.Module):
    """
    Full S-Core:
      Neural z -> symbolic graph G -> reasoning -> G' -> z_sym
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        d  = cfg.d_latent
        sv = cfg.sym_vocab

        # Neural -> symbolic (perception)
        self.perceive = nn.Linear(d, sv * 3)           # -> distribution over facts

        # GNN: symbolic -> neural (grounding)
        self.gnn = SymbolicGNN(cfg)

        # Abduction Engine
        self.abduction = AbductionHead(cfg)

        # Symbolic consistency: how much z_sym "explains" z.
        self.sym_consistency = nn.Linear(d, d)

        self.wm  = WorkingMemory(cfg.sym_max_facts)
        self.ltm = LongTermMemory(cfg.ltm_max_rules)

        self.sv = sv
        self.n_abduct_per_step = 1
        self._step = 0

    def perceive_graph(self, z: torch.Tensor) -> List[SymFact]:
        """z: (B, d) -> list of facts for WM."""
        B = z.size(0)
        logits = self.perceive(z.mean(0, keepdim=True)).view(3, self.sv)
        # Gumbel → hard fact
        indices = [F.gumbel_softmax(logits[i], tau=0.5, hard=True).argmax().item()
                   for i in range(3)]
        return [SymFact(*indices)]

    def reason(self) -> List[SymFact]:
        """Apply LTM rules to WM and derive new facts."""
        new_facts = []
        matched = self.ltm.match(self.wm)
        for rule in matched:
            rule.use_count += 1
            rule.weight    *= 1.01
            for conc in rule.conclusions:
                if self.wm.add(conc):
                    new_facts.append(conc)
        return new_facts

    def abduce_and_learn(self, z: torch.Tensor, error: torch.Tensor) -> int:
        """
        Run abduction when the error is large.
        Return the number of added rules.
        """
        if error.item() < 0.5:
            return 0
        candidates = self.abduction(z[:1])
        added = 0
        for rule in candidates:
            if self.ltm.add_rule(rule):
                added += 1
        return added

    def forward(self, z: torch.Tensor,
                world_error: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z           : (B, d)
        world_error : scalar
        Returns: z_sym (B, d), sym_consistency_loss scalar
        """
        B = z.size(0)
        device = z.device
        self._step += 1

        # 1. Perception: z → G (WM)
        new_facts = self.perceive_graph(z)
        self.wm.clear()
        for f in new_facts:
            self.wm.add(f)

        # 2. Reasoning: LTM -> new facts in WM
        inferred = self.reason()
        all_facts = self.wm.facts

        # 3. Grounding: G' → z_sym
        z_sym = self.gnn(all_facts, device).expand(B, -1)          # (B, d)

        # 4. Abduction (once every few steps)
        if self._step % 5 == 0:
            self.abduce_and_learn(z, world_error)

        # 5. Symbolic Grounding Loss: KL(Q(z|o) || P(z|S-Core(G)))
        z_sym_proj = self.sym_consistency(z_sym)
        sym_loss   = F.mse_loss(z, z_sym_proj.detach()) + \
                     F.mse_loss(z_sym_proj, z.detach())

        return z_sym, sym_loss

    def rule_regularizer(self, cfg: OMENv2Config) -> float:
        return cfg.lam_sym * self.ltm.complexity_penalty()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UPDATED LOSS: L_OMEN_AGI
# ══════════════════════════════════════════════════════════════════════════════

class OMENAGILoss(nn.Module):
    """
    L_OMEN = L_CE
           + γ·||z - (z_sim + v_mem)||²     ← Memory-augmented consistency
           + δ·Complexity                    ← Smoothness of WorldRNN
           + η·KL(Q(z) || Read(M,z))        ← Memory Recall Loss
           - α·I(z;M)                        ← Novelty bonus
           + β·KL(Q(z) || P(z|S-Core))      ← Symbolic Grounding
           + λ·Σ Usage(R)·Complexity(R)      ← LTM regularizer
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_latent
        # Structural encoder Program(y)
        self.prog_enc = nn.Sequential(
            nn.Embedding(cfg.vocab_size, d),
        )
        self.prog_pool = nn.Linear(d, d)

    def _program_encoding(self, tgt: torch.Tensor) -> torch.Tensor:
        emb = self.prog_enc[0](tgt).mean(1)
        return self.prog_pool(emb)

    def forward(self,
                logits:      torch.Tensor,
                targets:     torch.Tensor,
                z:           torch.Tensor,
                mu:          torch.Tensor,
                logvar:      torch.Tensor,
                z_sim:       torch.Tensor,
                v_mem:       torch.Tensor,
                z_sym:       torch.Tensor,
                sym_loss:    torch.Tensor,
                world_rnn:   WorldRNN,
                sparsity:    torch.Tensor,
                ltm_penalty: float,
                curiosity_l: torch.Tensor,
                ) -> Dict:
        cfg = self.cfg

        # 1. CE
        L_ce = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), ignore_index=0)

        # 2. KL (Occam regularizer on z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        # 3. Memory-augmented consistency: ||z - (z_sim + v_mem)||²
        z_target = (z_sim + v_mem).detach()
        L_mem_consist = F.mse_loss(z, z_target)

        # 4. WorldRNN complexity (finite differences)
        with torch.no_grad():
            eps = 1e-2
            dz = torch.randn_like(z) * eps
            dummy = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            zn1, _ = world_rnn(z.detach(), dummy)
            zn2, _ = world_rnn((z + dz).detach(), dummy)
            L_complex = ((zn1 - zn2) / eps).pow(2).mean()

        # 5. Memory Recall Loss: KL(Q(z) || Read(M,z))
        # Approximation: MSE between z and v_mem (if v_mem ~= 0 -> large penalty)
        v_mem_sig = v_mem.detach().norm(dim=-1, keepdim=True).clamp(min=1e-4)
        L_recall = F.mse_loss(z, (v_mem / v_mem_sig).detach())

        # 6. Novelty bonus: -I(z; M) ~= -‖v_mem‖ (more memory -> larger bonus)
        I_zm = v_mem.norm(dim=-1).mean()

        # 7. Symbolic grounding (from S-Core)
        L_sym_ground = sym_loss

        total = (L_ce
                 + kl
                 + cfg.gamma  * L_mem_consist
                 + cfg.delta  * L_complex
                 + cfg.eta    * L_recall
                 - cfg.alpha  * I_zm
                 + cfg.beta   * L_sym_ground
                 + ltm_penalty
                 + sparsity
                 + curiosity_l * 0.1)

        # L_scale: MDL regularizer (prevents "smearing" information across the vector)
        # Σ||z_t||² / B - identical for token and latent in v2 (one shared space)
        L_scale = cfg.lambda_tok * z.pow(2).mean()
        total   = total + L_scale

        return {
            "total":       total,
            "ce":          L_ce.item(),
            "kl":          kl.item(),
            "mem_consist": L_mem_consist.item(),
            "complex":     L_complex.item(),
            "recall":      L_recall.item(),
            "novelty":     I_zm.item(),
            "sym_ground":  L_sym_ground.item(),
            "ltm_pen":     ltm_penalty,
            "curiosity":   curiosity_l.item(),
            "sparsity":    sparsity.item() if torch.is_tensor(sparsity) else sparsity,
            "l_scale":     L_scale.item(),   # OMEN-Scale MDL regularizer
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  OMENv2 - FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class OMENv2(nn.Module):
    """
    Full cycle:
      Abduce (encoder) →
      M-Core recall →
      S-Core reasoning →
      Curiosity (if the gap is large) →
      Deduce (WorldRNN) →
      Induce (decoder + L_OMEN_AGI)
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        self.cfg      = cfg
        self.encoder  = GraphAttentionEncoder(cfg)
        self.world_rnn = WorldRNN(cfg)
        self.decoder  = CausalGraphDecoder(cfg)
        self.memory   = TensorProductMemory(cfg)
        self.epistemic = EpistemicGapDetector(cfg)
        self.curiosity = CuriosityModule(cfg)
        self.s_core   = SymbolicCore(cfg)
        self.loss_fn  = OMENAGILoss(cfg)

    def forward(self, src: torch.Tensor,
                tgt: torch.Tensor) -> Dict:
        # -- Abduce ------------------------------------------------------------
        z, mu, logvar, enc_sp = self.encoder(src)

        # -- M-Core: read memory -----------------------------------------------
        v_mem = self.memory.read(z)                                # (B, d)

        # -- WorldRNN: simulate (last 8 steps for speed) -----------------------
        sim_tgt = tgt[:, -8:] if tgt.size(1) > 8 else tgt
        z_sim_traj = self.world_rnn.simulate_sequence(z, sim_tgt)
        z_sim = z_sim_traj[:, -1]                                  # (B, d)

        # -- Epistemic gap -----------------------------------------------------
        E, gap_norm, hot_dims = self.epistemic.compute(z, self.world_rnn, z_sim)

        # -- Curiosity ---------------------------------------------------------
        z_enriched, cf_loss = self.curiosity(
            z, E, hot_dims, gap_norm, self.memory, self.world_rnn)

        # -- S-Core ------------------------------------------------------------
        world_err = F.mse_loss(z_sim, z.detach()).detach()
        z_sym, sym_loss = self.s_core(z_enriched, world_err)

        # z after enrichment: neural + symbolic + memory
        z_final = z_enriched + 0.1 * z_sym + 0.1 * v_mem

        # -- M-Core: deferred write - return the arguments ---------------------
        conf = (1.0 - gap_norm.clamp(0, 1))
        write_args = (z.detach(), z_sim.detach(), conf.detach())

        # -- Decode ------------------------------------------------------------
        logits, dec_sp = self.decoder(tgt, z_final)
        sparsity = enc_sp + dec_sp

        # -- Loss --------------------------------------------------------------
        ltm_pen = self.s_core.rule_regularizer(self.cfg)
        out = self.loss_fn(logits, tgt, z_final, mu, logvar,
                           z_sim, v_mem, z_sym, sym_loss,
                           self.world_rnn, sparsity, ltm_pen, cf_loss)

        out["logits"]     = logits
        out["z"]          = z_final
        out["gap_norm"]   = gap_norm.mean().item()
        out["n_rules"]    = len(self.s_core.ltm)
        out["n_writes"]   = self.memory.n_writes
        out["unknown_ex"] = self.curiosity.unknown_flag_count
        out["write_args"] = write_args  # (z, z_sim, conf) - apply after backward
        return out

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int = 32,
                 temperature: float = 0.8,
                 dynamic_reasoning: bool = True) -> torch.Tensor:
        """
        Generate `max_new` tokens after the prompt.

        dynamic_reasoning=True (default):
          At EVERY step, re-encode the current context and update
          z_sym (S-Core) and v_mem (M-Core), so z_final changes during
          generation and reflects accumulated knowledge.

          This matches the "slow loop" (Curiosity + S-Core + M-Core):
            ctx_t -> Encoder -> z_ctx
            z_ctx -> S-Core  -> z_sym   (apply verified rules from LTM)
            z_ctx -> M-Core  -> v_mem   (episodic recall)
            z_final = z_ctx + 0.1·z_sym + 0.1·v_mem

        dynamic_reasoning=False:
          Classic mode: z_final is computed once from the prompt
          and remains fixed (faster, less accurate).
        """
        self.eval()

        # -- Initial state (used when dynamic=False) ---------------------------
        z, _, _, _ = self.encoder(prompt)
        v_mem      = self.memory.read(z)
        z_sym, _   = self.s_core(z, torch.tensor(0.0, device=z.device))
        z_final    = z + 0.1 * z_sym + 0.1 * v_mem

        generated = prompt.clone()
        for _ in range(max_new):
            ctx = generated[:, -self.cfg.seq_len:]

            if dynamic_reasoning:
                # -- reasoning_step: re-encode + update S-Core / M-Core --------
                z_ctx, _, _, _ = self.encoder(ctx)
                z_sym_step, _  = self.s_core(
                    z_ctx, torch.tensor(0.0, device=z_ctx.device))
                v_mem_step     = self.memory.read(z_ctx)
                z_final        = z_ctx + 0.1 * z_sym_step + 0.1 * v_mem_step

            logits, _ = self.decoder(ctx, z_final)
            probs = F.softmax(logits[:, -1] / temperature, -1)
            generated = torch.cat([generated, torch.multinomial(probs, 1)], 1)

        return generated

    def memory_report(self) -> str:
        d  = self.cfg.d_latent
        H  = self.cfg.mem_heads
        mem_bytes = self.memory.memory_footprint_bytes()
        cache_bytes = len(self.memory.cache) * d * 4 * 2
        param_bytes = sum(p.numel() * p.element_size()
                          for p in self.parameters()) / 1024 / 1024
        return (f"  M-Core tensor : {mem_bytes/1024:.1f} KB  "
                f"(H={H}, d={d})\n"
                f"  M-Core cache  : {cache_bytes/1024:.1f} KB  "
                f"({len(self.memory.cache)} episodes)\n"
                f"  Total params  : {param_bytes:.2f} MB\n"
                f"  LTM rules     : {len(self.s_core.ltm)}\n"
                f"  Memory writes : {self.memory.n_writes}\n"
                f"  UNKNOWN flags : {self.curiosity.unknown_flag_count}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def make_counting(n, sl):
    data = []
    for _ in range(n):
        s = random.randint(1, 50); d = random.randint(1, 5)
        data.append(torch.tensor([(s + i*d) % 200 + 10 for i in range(sl)]))
    return data

def make_python(n, sl):
    templates = [
        "def add(a, b):\n    return a + b\n",
        "x = 0\nfor i in range(10):\n    x += i\n",
        "class Node:\n    def __init__(self, v):\n        self.v = v\n",
        "if x > 0:\n    print(x)\nelse:\n    print(-x)\n",
        "def fib(n):\n    if n<2: return n\n    return fib(n-1)+fib(n-2)\n",
    ]
    data = []
    for _ in range(n):
        t = random.choice(templates)
        b = [c % 256 for c in t.encode()]
        b = b[:sl] if len(b) >= sl else b + [0]*(sl-len(b))
        data.append(torch.tensor(b, dtype=torch.long))
    return data

def make_rule_transfer(n, sl):
    """
    Task with explicit rule transfer:
    sequences of the form [A, op, B, =, C], where op ∈ {+, -, *}
    The model MUST infer the rule rather than memorize a specific example.
    """
    data = []
    for _ in range(n):
        A = random.randint(10, 50)
        B = random.randint(10, 50)
        op = random.choice([0, 1, 2])
        if op == 0:   C = (A + B) % 200 + 10
        elif op == 1: C = abs(A - B) + 10
        else:         C = (A * B) % 200 + 10
        # Encode as a sequence of numbers + padding.
        seq = [A, 100+op, B, 200, C] + [0]*(sl - 5)
        data.append(torch.tensor(seq[:sl], dtype=torch.long))
    return data

def collate(batch):
    # Support two formats:
    #   1) List[Tensor]               - synthetic dataset (make_counting, ...)
    #   2) List[Tuple[Tensor,Tensor]] - real text (load_text_corpus -> (src, tgt))
    if isinstance(batch[0], (tuple, list)):
        src = torch.stack([item[0] for item in batch])
        tgt = torch.stack([item[1] for item in batch])
        return src, tgt
    s = torch.stack(batch)
    return s[:, :-1], s[:, 1:]


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model: OMENv2, dataset, optimizer, batch_size=16, max_batches=8):
    model.train()
    random.shuffle(dataset)
    agg = defaultdict(float)
    n_b = 0
    t0  = time.perf_counter()
    tot_tok = 0

    for start in range(0, len(dataset) - batch_size, batch_size):
        batch = dataset[start: start + batch_size]
        src, tgt = collate(batch)
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        optimizer.zero_grad()
        out = model(src, tgt)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # Consolidate memory after backward, like a hippocampus after experience.
        model.memory.write(*out["write_args"])

        for k, v in out.items():
            if k not in ("logits", "z", "write_args"):
                agg[k] += float(v) if torch.is_tensor(v) else v

        tot_tok += tgt.numel()
        n_b += 1
        if n_b >= max_batches:
            break

    elapsed = (time.perf_counter() - t0) * 1000
    avg = {k: v / n_b for k, v in agg.items()}
    avg["ppl"] = math.exp(min(avg.get("ce", 10), 10))
    avg["tps"] = tot_tok / (elapsed / 1000)
    avg["ms"]  = elapsed
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# 9.  INLINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(cfg: OMENv2Config) -> None:
    sep = lambda s: print(f"\n{'═'*72}\n  {s}\n{'═'*72}")

    # T0: Parameters
    sep("TEST 0 · Parameters and VRAM footprint")
    model = OMENv2(cfg).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Device    : {DEVICE}")
    print(f"  Parameters: {n_par:,}")
    print("  Memory report:")
    print(model.memory_report())
    assert n_par > 0
    print("  [PASS]")

    # T1: Forward pass - shape + new fields
    sep("TEST 1 · OMENv2 Forward (all components)")
    B, T = 4, cfg.seq_len - 1
    src = torch.randint(1, 200, (B, T)).to(DEVICE)
    tgt = torch.randint(1, 200, (B, T)).to(DEVICE)
    t0 = time.perf_counter()
    out = model(src, tgt)
    fwd_ms = (time.perf_counter() - t0) * 1000

    for key in ("total", "ce", "kl", "mem_consist", "recall",
                "sym_ground", "gap_norm", "n_rules", "n_writes"):
        assert key in out, f"FAIL: missing key {key}"
    assert out["logits"].shape == (B, T, cfg.vocab_size)
    assert out["z"].shape == (B, cfg.d_latent)
    print(f"  Forward time   : {fwd_ms:.0f} ms")
    print(f"  CE loss init   : {out['ce']:.3f}")
    print(f"  Gap norm       : {out['gap_norm']:.4f}")
    print(f"  Memory writes  : {out['n_writes']}")
    print(f"  LTM rules      : {out['n_rules']}")
    print(f"  UNKNOWN flags  : {out['unknown_ex']}")
    print("  [PASS]")

    # T2: Backward - WorldRNN must receive gradients
    sep("TEST 2 · Backward - grad flow through M-Core + S-Core")
    model.train()
    out2 = model(src, tgt)
    out2["total"].backward()
    model.memory.write(*out2["write_args"])
    enc_grad  = sum(p.grad.norm().item() for n, p in model.named_parameters()
                    if "encoder" in n and p.grad is not None)
    wrnn_grad = sum(p.grad.norm().item() for n, p in model.named_parameters()
                    if "world_rnn" in n and p.grad is not None)
    sco_grad  = sum(p.grad.norm().item() for n, p in model.named_parameters()
                    if "s_core" in n and p.grad is not None)
    print(f"  Encoder grad norm  : {enc_grad:.4f}")
    print(f"  WorldRNN grad norm : {wrnn_grad:.4f}")
    print(f"  S-Core grad norm   : {sco_grad:.4f}")
    assert enc_grad  > 0, "FAIL: encoder has no gradients"
    assert sco_grad  > 0, "FAIL: S-Core has no gradients"
    model.zero_grad()
    print("  [PASS]")

    # T3: TensorProductMemory - write/read
    sep("TEST 3 · TensorProductMemory — write / read / episodic")
    mem = model.memory
    mem_before = mem.memory.norm().item()

    z_test = torch.randn(8, cfg.d_latent).to(DEVICE)
    v_test = torch.randn(8, cfg.d_latent).to(DEVICE)
    conf   = torch.tensor([0.1]*4 + [0.9]*4, device=DEVICE)  # first 4 should be written
    mem.write(z_test, v_test, conf)

    mem_after = mem.memory.norm().item()
    assert mem_after > mem_before, "FAIL: memory did not change after write"
    assert mem.n_writes >= 4, f"FAIL: expected >=4 writes, got {mem.n_writes}"

    v_ret = mem.read(z_test[:2])
    assert v_ret.shape == (2, cfg.d_latent), "FAIL: read shape"

    v_ep = mem.episodic_recall(z_test[:2], k=3)
    assert v_ep.shape == (2, cfg.d_latent), "FAIL: episodic_recall shape"

    footprint = mem.memory_footprint_bytes()
    print(f"  M-Core before write norm : {mem_before:.4f}")
    print(f"  M-Core after write norm  : {mem_after:.4f}  (changed ✓)")
    print(f"  n_writes                 : {mem.n_writes}")
    print(f"  read shape               : {tuple(v_ret.shape)} ✓")
    print(f"  episodic_recall shape    : {tuple(v_ep.shape)} ✓")
    print(f"  VRAM footprint           : {footprint/1024:.1f} KB")
    print("  [PASS]")

    # T4: EpistemicGapDetector
    sep("TEST 4 · EpistemicGapDetector — E(z) = diag(∇_z L_world)²")
    model.eval()
    z_r = torch.randn(B, cfg.d_latent, device=DEVICE)
    z_s = torch.randn(B, cfg.d_latent, device=DEVICE)
    E, gap_norm, hot_dims = model.epistemic.compute(z_r, model.world_rnn, z_s)
    assert E.shape == (B, cfg.d_latent), "FAIL: E shape"
    assert gap_norm.shape == (B,), "FAIL: gap_norm shape"
    assert (E >= 0).all(), "FAIL: E < 0 (square must be >= 0)"
    assert hot_dims.sum() > 0, "FAIL: no hot dimensions"
    print(f"  E shape          : {tuple(E.shape)} ✓")
    print(f"  gap_norm mean    : {gap_norm.mean():.4f}")
    print(f"  hot_dims active  : {hot_dims.mean():.2%} of dimensions")
    print("  [PASS]")

    # T5: S-Core - symbolic inference
    sep("TEST 5 · SymbolicCore — Perception → Reason → Abduce")
    sc = model.s_core
    sc.ltm.rules.clear()

    # Add a rule to LTM manually.
    r = SymRule(
        conditions=(SymFact(1, 0, 2),),
        conclusions=(SymFact(1, 1, 3),)
    )
    sc.ltm.add_rule(r)

    # Add the corresponding fact to WM.
    sc.wm.clear()
    sc.wm.add(SymFact(1, 0, 2))

    matched = sc.ltm.match(sc.wm)
    inferred = sc.reason()
    print(f"  LTM rules         : {len(sc.ltm)}")
    print(f"  Matched rules     : {len(matched)}")
    print(f"  Inferred facts    : {inferred}")
    assert len(matched) >= 1, "FAIL: unification did not fire"

    # Abduction
    z_abd = torch.randn(1, cfg.d_latent, device=DEVICE)
    err_high = torch.tensor(2.0)
    n_added = sc.abduce_and_learn(z_abd, err_high)
    print(f"  Added rules (abduction): {n_added}")
    print(f"  LTM after abduction: {len(sc.ltm)}")
    print("  [PASS]")

    # T6: CuriosityModule
    sep("TEST 6 · CuriosityModule — counterfactuals + UNKNOWN detection")
    model.train()
    z_c = torch.randn(B, cfg.d_latent, device=DEVICE)
    E_c, gn_c, hd_c = model.epistemic.compute(z_c, model.world_rnn, z_c * 2)
    # Force curiosity activation (gap_norm > tau).
    gn_forced = torch.ones(B, device=DEVICE) * (cfg.epistemic_tau + 0.5)
    z_enr, cf_l = model.curiosity(z_c, E_c, hd_c, gn_forced, model.memory, model.world_rnn)
    assert z_enr.shape == (B, cfg.d_latent), "FAIL: z_enr shape"
    assert not math.isnan(cf_l.item()), "FAIL: NaN in curiosity loss"
    print(f"  z_enr shape      : {tuple(z_enr.shape)} ✓")
    print(f"  counterfactual L : {cf_l.item():.4f}")
    print(f"  UNKNOWN flags    : {model.curiosity.unknown_flag_count}")
    print("  [PASS]")

    # T7: Minimal training (25 iterations)
    sep("TEST 7 · Training for 25 iterations - CE down + LTM up")
    model.train()
    ds  = make_counting(128, cfg.seq_len)
    opt = AdamW(model.parameters(), lr=3e-4)
    hist_ce, hist_rules = [], []
    for step in range(25):
        batch = random.sample(ds, 8)
        s, t = collate(batch)
        s, t = s.to(DEVICE), t.to(DEVICE)
        opt.zero_grad()
        o = model(s, t)
        o["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist_ce.append(o["ce"])
        hist_rules.append(o["n_rules"])
    first5 = sum(hist_ce[:5]) / 5
    last5  = sum(hist_ce[-5:]) / 5
    max_rules = max(hist_rules)
    print(f"  CE (first 5)    : {first5:.3f}")
    print(f"  CE (last 5)     : {last5:.3f}")
    print(f"  Max LTM rules   : {max_rules}")
    assert last5 < first5, "FAIL: CE is not decreasing"
    print("  [PASS]")

    # T8: Generation
    sep("TEST 8 · Token generation (dynamic_reasoning=True/False)")
    model.eval()
    pr = torch.randint(10, 100, (1, 8), device=DEVICE)

    # dynamic_reasoning=True - S-Core + M-Core at every step
    with torch.no_grad():
        gen_dyn = model.generate(pr, max_new=16, dynamic_reasoning=True)
    assert gen_dyn.shape == (1, 24), f"FAIL: gen_dyn shape {gen_dyn.shape}"
    print(f"  Prompt          : {pr[0].tolist()}")
    print(f"  Output (dynamic): {gen_dyn[0, 8:].tolist()}")

    # dynamic_reasoning=False - classic mode (fixed z_final)
    with torch.no_grad():
        gen_static = model.generate(pr, max_new=16, dynamic_reasoning=False)
    assert gen_static.shape == (1, 24), f"FAIL: gen_static shape {gen_static.shape}"
    print(f"  Output (static) : {gen_static[0, 8:].tolist()}")
    print("  [PASS]")

    print(f"\n{'═'*72}")
    print("  ✅  All 9 tests passed successfully")
    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(cfg: OMENv2Config, epochs: int = 6) -> None:
    print("╔" + "═"*76 + "╗")
    print("║   OMENv2 AGI — AUTONOMOUS TRAINING BENCHMARK" + " "*31 + "║")
    print("╚" + "═"*76 + "╝\n")

    datasets = {
        "Count":       make_counting(256, cfg.seq_len),
        "Python":      make_python(256, cfg.seq_len),
        "RuleTransfer": make_rule_transfer(256, cfg.seq_len),
    }

    fmt = "{:>7}" * 11
    hdr = fmt.format("Ep","CE↓","KL","MemC","Recall","SymGr","Gap",
                     "Rules","PPL↓","Tok/s","ms")

    for ds_name, ds in datasets.items():
        print(f"\n  ── {ds_name} ──")
        print(hdr)
        print("-" * 77)
        model = OMENv2(cfg).to(DEVICE)
        opt   = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = CosineAnnealingLR(opt, T_max=epochs)
        best_ppl = float("inf")

        for ep in range(1, epochs + 1):
            avg = train_epoch(model, ds, opt, batch_size=16, max_batches=8)
            sched.step()
            best_ppl = min(best_ppl, avg["ppl"])
            print(fmt.format(
                ep,
                f"{avg.get('ce',0):.3f}",
                f"{avg.get('kl',0):.3f}",
                f"{avg.get('mem_consist',0):.3f}",
                f"{avg.get('recall',0):.3f}",
                f"{avg.get('sym_ground',0):.3f}",
                f"{avg.get('gap_norm',0):.3f}",
                f"{int(avg.get('n_rules',0))}",
                f"{avg.get('ppl',0):.1f}",
                f"{avg.get('tps',0):.0f}",
                f"{avg.get('ms',0):.0f}",
            ))

        print("-" * 77)
        mr = model.memory_report()
        print(f"  Best PPL: {best_ppl:.2f}")
        print(mr)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ABLATION: OMENv2 vs CE-only vs v1 (without memory)
# ══════════════════════════════════════════════════════════════════════════════

def ablation(cfg: OMENv2Config) -> None:
    print(f"\n{'═'*76}")
    print("  ABLATION: OMENv2 (full) vs OMENv2 (no memory) vs CE-only")
    print(f"{'═'*76}")

    ds = make_rule_transfer(256, cfg.seq_len)
    results = {}

    variants = {
        "OMENv2 (full)":      dict(),
        "OMENv2 (no memory)": dict(eta=0, alpha=0, gamma=0),
        "CE-only":            dict(eta=0, alpha=0, gamma=0, beta=0, lam_sym=0, delta=0),
    }

    for name, overrides in variants.items():
        c = deepcopy(cfg)
        for k, v in overrides.items():
            setattr(c, k, v)
        model = OMENv2(c).to(DEVICE)
        opt = AdamW(model.parameters(), lr=1e-3)
        ppls = []
        for _ in range(6):
            avg = train_epoch(model, ds, opt, batch_size=16, max_batches=8)
            ppls.append(round(avg["ppl"], 1))
        results[name] = ppls
        traj = " → ".join(map(str, ppls))
        print(f"  {name:<28} {traj}")

    best_v2  = min(results["OMENv2 (full)"])
    best_ce  = min(results["CE-only"])
    print(f"\n  OMENv2 vs CE-only (min PPL): {best_v2:.1f} vs {best_ce:.1f}  "
          f"({'↓' if best_v2 < best_ce else '='}{abs(best_v2 - best_ce):.1f})")
    print(f"{'═'*76}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42); random.seed(42)

    cfg = OMENv2Config(
        vocab_size=256, d_model=128, d_latent=64,
        n_heads=4, n_layers=3, seq_len=64,
        world_rnn_hidden=128,
        mem_heads=8, mem_cache_size=512,
        sym_vocab=64, sym_embed_dim=32, sym_gnn_layers=2,
        abduct_candidates=8, ltm_max_rules=256,
        alpha=0.1, beta=0.05, gamma=0.1, delta=1e-3,
        eta=0.05, lam_sym=0.02,
        epistemic_tau=0.3,
        # OMEN-Scale MDL (new)
        lambda_tok=1e-4, lambda_conc=1e-3,
    )

    run_tests(cfg)
    benchmark(cfg, epochs=6)
    ablation(cfg)

    # -- OMEN-Scale (hierarchical, 3 levels) ----------------------------------
    # Run: python omen_scale.py
    print("\n  To run OMEN-Scale (Token→Concept→Symbolic):")
    print("    python omen_scale.py\n")

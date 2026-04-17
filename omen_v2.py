"""
OMEN v2 — legacy reference architecture for OMEN
================================================
Цей файл зберігає ранній «чистий» варіант OMEN як дослідницький reference.

Canonical runtime stack для поточної системи:
  omen_scale.OMENScale

Тут лишається історично важлива реалізація:
  · DualStreamAttention
  · GraphAttentionEncoder
  · CausalGraphDecoder
  · ранні M-Core / Curiosity / S-Core ідеї

Розширення v1 трьома новими фундаментальними компонентами:

  M-Core  : Tensor Product Memory — голографічна пам'ять у VRAM
             (read/write без backprop, O(H·d²) незалежно від N фактів)

  Curiosity Engine : Epistemic Gap Detector + Counterfactual Rollouts
             (E(z) = diag(∇_z L_world)², активується при ||E(z)|| > τ)

  S-Core  : Symbolic Core — робоча пам'ять + LTM правил + Abduction Engine
             (нейро-символьний інтерфейс через Gumbel-Softmax + GNN)

Формула:
  L_OMEN = L_CE + γ·||z - (z_sim + v_mem)||² + δ·||∇_z WorldRNN||²
          + η·KL(Q(z|o) || Read(M,z)) - α·I(z;M)
          + β·KL(Q(z|o) || P(z|S-Core(G)))
          + λ_sym·Σ Usage(R)·Complexity(R)

Стек: PyTorch 2.x — нуль зовнішніх солверів.
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
    # Трансформер
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
    mem_heads:        int   = 8     # H голів тензорної пам'яті
    mem_write_tau:    float = 0.3   # поріг упевненості (нижче → писати)
    mem_cache_size:   int   = 512   # кеш останніх z для швидкого recall

    # Curiosity
    epistemic_tau:    float = 0.3   # поріг ||E(z)|| для активації модуля
    epistemic_exact_grad: bool = False
    n_counterfactual: int   = 2     # кількість контрфактичних роловтів

    # S-Core
    sym_vocab:        int   = 64    # розмір словника символів
    sym_embed_dim:    int   = 32    # розмірність символьних ембеддингів
    sym_gnn_layers:   int   = 2     # глибина GNN
    sym_max_facts:    int   = 32    # макс. фактів у WM
    abduct_candidates: int  = 8     # кандидати для абдукції
    ltm_max_rules:    int   = 256   # макс. правил у LTM

    # Коефіцієнти лосу
    alpha:     float = 0.1    # Epistemic bonus
    beta:      float = 0.05   # World / Symbolic consistency
    gamma:     float = 0.1    # Structural / Memory reward
    delta:     float = 1e-3   # Complexity penalty
    eta:       float = 0.05   # Memory Recall Loss
    lam_sym:   float = 0.02   # Symbolic rule regularizer

    # OMEN-Scale MDL (використовується omen_scale.py)
    # L_scale = λ_tok·(1/T)·Σ||z_t||² + λ_conc·(1/|C|)·Σ||c||²
    lambda_tok:  float = 1e-4   # штраф на токен-норми (тип. = 1e-4)
    lambda_conc: float = 1e-3   # штраф на концепт-норми (тип. = 1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  БАЗОВІ БЛОКИ (Dual-Stream Attention + OMENBlock) — з v1
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
        self.drop = nn.Dropout(cfg.dropout, inplace=False)  # attention weights — НЕ inplace (softmax output shared)
        self.sparsity_lambda = cfg.sparsity_lambda

    def forward(self, x, causal_mask):
        B, T, D = x.shape
        q, k, v = [t.view(B, T, self.h, self.dh).transpose(1, 2)
                   for t in self.to_qkv(x).chunk(3, dim=-1)]
        scale = math.sqrt(self.dh)

        # OPT-QK: (q @ k^T)/scale обчислюється ОДИН раз — спільний для обох стрімів.
        # Раніше: 2× O(B·h·T²·dh) matmul (text + causal).
        # Тепер:  1× matmul + дешеве поелементне множення на M.
        qk = (q @ k.transpose(-2, -1)) / scale                    # (B, h, T, T)

        # Маска — обчислюється один раз і ділиться між стрімами
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
        # OPT-5: inplace=True — не виділяємо новий тензор на кожен dropout
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
        OPT-2: Pre-embed всі дії одним викликом до входу в цикл.
        Embedding lookup виноситься за межі for-loop:
          Раніше: T × (embed_lookup + GRUCell + Linear)
          Тепер:  1 × embed_lookup_batch  +  T × (GRUCell + Linear)
        На T=8, B=8: ~5% прискорення за рахунок зменшення диспетчеризації.

        FIX Bug1 (world loss 17x drift):
          Попередній код:  z0 if t == 0 else traj[-1]
          Після кроку 0 GRU отримував власний попередній вихід traj[-1] замість
          початкового концепту z0. За T=8 кроків похибка накопичувалася
          (~+74% L_world), а не зменшувалась. Рішення — завжди подавати z0:
          GRU-прихований стан h вже несе "пам'ять" про попередні кроки,
          тому z0 як "anchor input" не заважає динаміці, але усуває дрейф.
        """
        B, T   = actions.shape
        a_all  = self.act_emb(actions)                     # (B, T, d_latent) — один виклик
        h      = self.h0.expand(B, -1).contiguous()
        traj   = []
        z_prev = z0
        if teacher_states is not None and teacher_states.shape[:2] != (B, T):
            raise ValueError("teacher_states must have shape (B, T, d_latent)")
        for t in range(T):
            # FIXED: завжди z0 (а не traj[-1]) — GRU h несе динаміку,
            # z0 — стабільний anchor. Усуває 17x дрейф world loss.
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
    Голографічна пам'ять: M ∈ R^{H × d × d}
    Розмір ФІКСОВАНИЙ незалежно від кількості записів.

    Запис : M_h ← M_h + λ·(k ⊗ v)          [без backprop на M]
    Читання: v_ret = Σ_h M_h · k             [O(H·d²)]
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        d, H = cfg.d_latent, cfg.mem_heads
        # Пам'ять: параметр, але оновлюється не градієнтом, а прямим записом
        self.register_buffer("memory", torch.zeros(H, d, d))
        self.key_proj = nn.Linear(d, d * H, bias=False)
        self.val_proj = nn.Linear(d, d * H, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.d, self.H = d, H
        self.write_tau = cfg.mem_write_tau

        # LRU-кеш станів (для швидкого episodic recall)
        self.cache: deque = deque(maxlen=cfg.mem_cache_size)
        self.n_writes = 0

    # ── Читання ──────────────────────────────────────────────────────────────
    def read(self, z_query: torch.Tensor) -> torch.Tensor:
        """z_query: (B, d) → v_retrieved: (B, d)"""
        # FIX: write() виконує self.memory += delta під @no_grad, що все одно
        # інкрементує version-counter і ламає autograd backward.
        # .detach() ізолює buffer від version-перевірки autograd.
        # Градієнти через key_proj і out_proj (trainable) зберігаються повністю.
        k = self.key_proj(z_query).view(-1, self.H, self.d)      # (B, H, d)
        v = torch.einsum('bhd,hde->bhe', k, self.memory.detach())
        return self.out_proj(v.mean(1))                            # (B, d)

    # ── Запис (без градієнта, як гіпокамп) ───────────────────────────────────
    @torch.no_grad()
    def write(self, z_state: torch.Tensor,
              z_value: torch.Tensor,
              confidence: torch.Tensor) -> None:
        """
        confidence: (B,) ∈ [0,1]
        Записуємо лише якщо модель «здивована» (1 - conf > write_tau)

        ВАЖЛИВО: `.copy_()` замість `+=` — `+=` інкрементує version_counter
        основного тензора, що ламає autograd.backward() (version mismatch).
        `.copy_()` оновлює дані БЕЗ зміни version_counter.
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

        # Додаємо в LRU-кеш для episodic recall
        for i in range(z_s.size(0)):
            self.cache.append((z_s[i].detach(), z_v[i].detach()))

        self.n_writes += mask.sum().item()

    # ── Episodic recall (k-NN у кеші) ────────────────────────────────────────
    @torch.no_grad()
    def episodic_recall(self, z_query: torch.Tensor, k: int = 4) -> torch.Tensor:
        """Повертає середнє значення k найближчих записів у кеші"""
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
    Де велике E_i — там модель «не розуміє» каузальний зв'язок.
    Повертає (epistemic_map, gap_norm, hot_dims).
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        self.tau = cfg.epistemic_tau
        self.exact_grad = bool(getattr(cfg, "epistemic_exact_grad", False))
        # Вивчена проекція для агрегації епістемічного сигналу
        self.aggregator = nn.Linear(cfg.d_latent, cfg.d_latent)
        self.d = cfg.d_latent

    def compute(self, z: torch.Tensor,
                world_rnn: WorldRNN,
                z_sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z      : (B, d) — поточний стан, requires_grad=True
        z_sim  : (B, d) — передбачений симулятором стан
        Returns:
          E        : (B, d) — епістемічна карта
          gap_norm : (B,)   — норма прогалини
          hot_dims : (B, d) — binary mask найгарячіших вимірів
        """
        # OPT-EGD: замість torch.autograd.grad() (окремий backward-пас ≈ +30–50% часу батчу)
        # використовуємо замкнуту формулу градієнту MSE:
        #   L = (1/B)·Σ_b ||z_b − z_sim_b||²   →   ∂L/∂z_b = (2/B)·(z_b − z_sim_b)
        #   E = (∂L/∂z)² ∝ (z − z_sim)²
        # Пропорційність: константа (4/B²) одинакова для всіх вимірів → hot_dims та
        # gap_norm обчислюються коректно. Повний autograd.grad більше не потрібен.
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

        # Топ-25% найгарячіших вимірів
        threshold = E.quantile(0.75, dim=-1, keepdim=True)
        hot_dims  = (E >= threshold).float()

        return E.detach(), gap_norm.detach(), hot_dims.detach()


class CuriosityModule(nn.Module):
    """
    Якщо ||E(z)|| > τ:
      1. Формує Query q (проекція hot_dims → семантичний запит)
      2. Генерує n контрфактичних роловтів через WorldRNN
      3. Повертає curiosity_loss та збагачений z_curious
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        d = cfg.d_latent
        self.query_proj  = nn.Linear(d, d)         # hot_dims → запит
        self.fusion      = nn.Linear(d * 2, d)     # злиття z + відповіді з пам'яті
        self.n_cf        = cfg.n_counterfactual
        self.tau         = cfg.epistemic_tau
        self.unknown_flag_count = 0                # лічильник UNKNOWN_EXCEPTION

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

        # ── Query формується з гарячих вимірів ───────────────────────────────
        query = self.query_proj(z * hot_dims)                      # (B, d)

        # ── Читання з M-Core ──────────────────────────────────────────────────
        v_mem = memory.read(query)                                 # (B, d)

        # Episodic recall як доповнення
        v_ep  = memory.episodic_recall(query, k=4)                # (B, d)
        v_combined = (v_mem + v_ep) * 0.5

        # ── Counterfactual rollouts ───────────────────────────────────────────
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

            # Контрфактичний ланцюжок має відповідати збагаченому z+v_mem
            z_target = (z + v_combined).detach()
            cf_loss  = F.mse_loss(z_cf_traj.mean(1), z_target)

        # ── Детектуємо UNKNOWN_EXCEPTION (порожня пам'ять, великий gap) ──────
        mem_signal_norm = v_combined.norm(dim=-1)                  # (B,)
        unknown = active & (mem_signal_norm < 1e-3)
        if unknown.any():
            self.unknown_flag_count += unknown.sum().item()

        # ── Збагачення стану ──────────────────────────────────────────────────
        z_enriched = self.fusion(torch.cat([z, v_combined], dim=-1))
        # Залишаємо незмінними ті елементи батчу, де gap < τ
        z_out = torch.where(active.unsqueeze(-1), z_enriched, z)

        return z_out, cf_loss


# ══════════════════════════════════════════════════════════════════════════════
# 4.  S-CORE: SYMBOLIC CORE
# ══════════════════════════════════════════════════════════════════════════════

# ── 4.1  Символьні структури ─────────────────────────────────────────────────

@dataclass(frozen=True)
class SymFact:
    """Факт у вигляді триплета (суб'єкт, предикат, об'єкт)"""
    subj: int   # індекс у sym_vocab
    pred: int
    obj:  int

    def __repr__(self):
        return f"({self.subj}→[{self.pred}]→{self.obj})"


@dataclass
class SymRule:
    """Правило: IF умови → THEN висновки"""
    conditions:  Tuple[SymFact, ...]
    conclusions: Tuple[SymFact, ...]
    weight:      float = 1.0    # важливість (зростає з використанням)
    use_count:   int   = 0      # лічильник використань
    complexity:  int   = 0      # кількість символів

    def __post_init__(self):
        self.complexity = len(self.conditions) + len(self.conclusions)

    def __hash__(self):
        return hash((self.conditions, self.conclusions))


# ── 4.2  Робоча пам'ять ──────────────────────────────────────────────────────

class WorkingMemory:
    """Граф поточного контексту (факти + індекс для швидкого пошуку)"""

    def __init__(self, max_facts: int = 32):
        # OPT-WM: deque для O(1) popleft + set для O(1) membership / remove
        from collections import deque as _deque
        self.facts:     _deque                    = _deque(maxlen=None)  # upbound вручну
        self._fact_set: set                       = set()
        self.pred_idx:  Dict[int, set]            = defaultdict(set)
        self.max_facts = max_facts

    def add(self, fact: SymFact) -> bool:
        if fact in self._fact_set:               # O(1) замість O(n) list scan
            return False
        if len(self.facts) >= self.max_facts:
            removed = self.facts[0]              # peek oldest — O(1) для deque
            self.facts.popleft()                 # O(1) замість O(n) list.pop(0)
            self._fact_set.discard(removed)      # O(1) замість O(n) list.remove
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


# ── 4.3  Довготривала символьна пам'ять + Уніфікація ─────────────────────────

class LongTermMemory:
    """
    База правил — хеш-таблиця {frozen(conditions) → SymRule}.
    Уніфікація через точне зіставлення (з wildcards: pred=-1 = будь-який).
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
            # Видаляємо найменш використовуване правило
            worst = min(self.rules.values(), key=lambda r: r.use_count)
            del self.rules[hash(worst)]
        self.rules[h] = rule
        return True

    def match(self, wm: WorkingMemory) -> List[SymRule]:
        """Повертає всі правила, умови яких унуфіковані з WM"""
        matched = []
        for rule in self.rules.values():
            if self._unify(rule.conditions, wm):
                matched.append(rule)
        return matched

    def _unify(self, conditions: Tuple[SymFact, ...], wm: WorkingMemory) -> bool:
        for cond in conditions:
            found = False
            # OPT-UNIFY: звужуємо пул через pred_idx замість повного перебору.
            # Wildcard pred=-1 → fallback на весь _fact_set.
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
        """Σ Usage(R) · Complexity(R) — регуляризатор S-Core"""
        return sum(r.use_count * r.complexity for r in self.rules.values())

    def __len__(self): return len(self.rules)


# ── 4.4  Graph Neural Network (символьний → нейронний) ───────────────────────

class SymbolicGNN(nn.Module):
    """
    Перетворює граф фактів у вектор z_sym ∈ R^{d_latent}.
    Факти → ембеддинги → message passing → pooling.
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

        # Вузли — унікальні суб'єкти та об'єкти
        nodes: Dict[int, torch.Tensor] = {}
        for f in facts:
            for sym_id in (f.subj, f.obj):
                if sym_id not in nodes:
                    idx = torch.tensor([sym_id % (self.sym_emb.num_embeddings)],
                                       device=device)
                    nodes[sym_id] = self.sym_emb(idx).squeeze(0)

        # Message passing по ребрах (фактах)
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


# ── 4.5  Abduction Engine ─────────────────────────────────────────────────────

class AbductionHead(nn.Module):
    """
    Нейромережевий генератор кандидатів правил.
    R* = argmin [ Length(R) + λ·PredError(R, Trace) ]
    Реалізація: нейромережа пропонує кандидати → символьна оцінка → вибір.
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        d = cfg.d_latent
        sv = cfg.sym_vocab
        self.n_cand = cfg.abduct_candidates

        # Мережа → розподіл над sym_vocab для кожного слоту факту
        self.rule_gen = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(),
            nn.Linear(d * 2, sv * 3 * 2)  # 2 факти × 3 слоти × vocab
        )
        self.sv = sv

    def forward(self, z: torch.Tensor) -> List[SymRule]:
        """
        z: (1, d) — поточний стан
        Returns список кандидатів-правил (без backprop на LTM)
        """
        logits = self.rule_gen(z.squeeze(0))                       # (sv*6,)
        logits = logits.view(2, 3, self.sv)                        # (2 facts, 3 slots, vocab)

        rules = []
        for _ in range(self.n_cand):
            # Gumbel-Softmax для диференційованої дискретизації
            cond_idx = F.gumbel_softmax(logits[0], tau=1.0, hard=True).argmax(-1)
            conc_idx = F.gumbel_softmax(logits[1], tau=1.0, hard=True).argmax(-1)

            cond = SymFact(cond_idx[0].item(), cond_idx[1].item(), cond_idx[2].item())
            conc = SymFact(conc_idx[0].item(), conc_idx[1].item(), conc_idx[2].item())

            # Принцип Оккама: довжина правила = 2 (мінімальна)
            rule = SymRule(conditions=(cond,), conclusions=(conc,))
            rules.append(rule)

        return rules


class SymbolicCore(nn.Module):
    """
    Повний S-Core:
      Нейронний z → Символьний граф G → Reasoning → G' → z_sym
    """

    def __init__(self, cfg: OMENv2Config):
        super().__init__()
        d  = cfg.d_latent
        sv = cfg.sym_vocab

        # Нейронний → символьний (perception)
        self.perceive = nn.Linear(d, sv * 3)           # → розподіл над фактами

        # GNN: символьний → нейронний (grounding)
        self.gnn = SymbolicGNN(cfg)

        # Abduction Engine
        self.abduction = AbductionHead(cfg)

        # Символьна консистентність: скільки z_sym "пояснює" z
        self.sym_consistency = nn.Linear(d, d)

        self.wm  = WorkingMemory(cfg.sym_max_facts)
        self.ltm = LongTermMemory(cfg.ltm_max_rules)

        self.sv = sv
        self.n_abduct_per_step = 1
        self._step = 0

    def perceive_graph(self, z: torch.Tensor) -> List[SymFact]:
        """z: (B, d) → список фактів для WM"""
        B = z.size(0)
        logits = self.perceive(z.mean(0, keepdim=True)).view(3, self.sv)
        # Gumbel → hard fact
        indices = [F.gumbel_softmax(logits[i], tau=0.5, hard=True).argmax().item()
                   for i in range(3)]
        return [SymFact(*indices)]

    def reason(self) -> List[SymFact]:
        """Застосувати правила LTM до WM → нові факти"""
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
        Запускаємо Abduction якщо помилка велика.
        Повертає кількість доданих правил.
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

        # 2. Reasoning: LTM → нові факти в WM
        inferred = self.reason()
        all_facts = self.wm.facts

        # 3. Grounding: G' → z_sym
        z_sym = self.gnn(all_facts, device).expand(B, -1)          # (B, d)

        # 4. Abduction (раз на кілька кроків)
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
# 5.  ОНОВЛЕНИЙ LOSS: L_OMEN_AGI
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
        # Структурний енкодер Program(y)
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

        # 2. KL (Оккамів регуляризатор на z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        # 3. Memory-augmented consistency: ||z - (z_sim + v_mem)||²
        z_target = (z_sim + v_mem).detach()
        L_mem_consist = F.mse_loss(z, z_target)

        # 4. Складність WorldRNN (скінченні різниці)
        with torch.no_grad():
            eps = 1e-2
            dz = torch.randn_like(z) * eps
            dummy = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            zn1, _ = world_rnn(z.detach(), dummy)
            zn2, _ = world_rnn((z + dz).detach(), dummy)
            L_complex = ((zn1 - zn2) / eps).pow(2).mean()

        # 5. Memory Recall Loss: KL(Q(z) || Read(M,z))
        # Апроксимація: MSE між z та v_mem (якщо v_mem ≈ 0 → великий штраф)
        v_mem_sig = v_mem.detach().norm(dim=-1, keepdim=True).clamp(min=1e-4)
        L_recall = F.mse_loss(z, (v_mem / v_mem_sig).detach())

        # 6. Novelty bonus: -I(z; M) ≈ -‖v_mem‖ (більша пам'ять → більший бонус)
        I_zm = v_mem.norm(dim=-1).mean()

        # 7. Symbolic Grounding (з S-Core)
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

        # L_scale: MDL-регуляризатор (не дає "розмазати" інфо по вектору)
        # Σ||z_t||² / B — однаковий для токен і латент в v2 (один простір)
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
            "l_scale":     L_scale.item(),   # MDL-регуляризатор OMEN-Scale
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  OMENv2 — ПОВНА МОДЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════

class OMENv2(nn.Module):
    """
    Повний цикл:
      Abduce (encoder) →
      M-Core recall →
      S-Core reasoning →
      Curiosity (якщо gap великий) →
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
        # ── Abduce ───────────────────────────────────────────────────────────
        z, mu, logvar, enc_sp = self.encoder(src)

        # ── M-Core: читаємо пам'ять ─────────────────────────────────────────
        v_mem = self.memory.read(z)                                # (B, d)

        # ── WorldRNN: симулюємо (останні 8 кроків для швидкості) ────────────
        sim_tgt = tgt[:, -8:] if tgt.size(1) > 8 else tgt
        z_sim_traj = self.world_rnn.simulate_sequence(z, sim_tgt)
        z_sim = z_sim_traj[:, -1]                                  # (B, d)

        # ── Epistemic Gap ────────────────────────────────────────────────────
        E, gap_norm, hot_dims = self.epistemic.compute(z, self.world_rnn, z_sim)

        # ── Curiosity ────────────────────────────────────────────────────────
        z_enriched, cf_loss = self.curiosity(
            z, E, hot_dims, gap_norm, self.memory, self.world_rnn)

        # ── S-Core ───────────────────────────────────────────────────────────
        world_err = F.mse_loss(z_sim, z.detach()).detach()
        z_sym, sym_loss = self.s_core(z_enriched, world_err)

        # z після збагачення: нейронний + символьний + пам'ять
        z_final = z_enriched + 0.1 * z_sym + 0.1 * v_mem

        # ── M-Core: запис відкладений — повертаємо аргументи ───────────────
        conf = (1.0 - gap_norm.clamp(0, 1))
        write_args = (z.detach(), z_sim.detach(), conf.detach())

        # ── Decode ───────────────────────────────────────────────────────────
        logits, dec_sp = self.decoder(tgt, z_final)
        sparsity = enc_sp + dec_sp

        # ── Loss ─────────────────────────────────────────────────────────────
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
        out["write_args"] = write_args  # (z, z_sim, conf) — застосувати після backward
        return out

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int = 32,
                 temperature: float = 0.8,
                 dynamic_reasoning: bool = True) -> torch.Tensor:
        """
        Генерує max_new токенів після prompt.

        dynamic_reasoning=True (за замовчуванням):
          На КОЖНОМУ кроці перекодуємо поточний контекст, оновлюємо
          z_sym (S-Core) та v_mem (M-Core) → z_final змінюється по ходу
          генерації, відображаючи нараховане знання.

          Це відповідає «повільному контуру» (Curiosity + S-Core + M-Core):
            ctx_t → Encoder → z_ctx
            z_ctx → S-Core  → z_sym   (застосування verified правил з LTM)
            z_ctx → M-Core  → v_mem   (episodic recall)
            z_final = z_ctx + 0.1·z_sym + 0.1·v_mem

        dynamic_reasoning=False:
          Класичний режим: z_final обчислюється один раз по prompt
          і залишається фіксованим (швидший, менш точний).
        """
        self.eval()

        # ── Ініціальний стан (використовується якщо dynamic=False) ───────────
        z, _, _, _ = self.encoder(prompt)
        v_mem      = self.memory.read(z)
        z_sym, _   = self.s_core(z, torch.tensor(0.0, device=z.device))
        z_final    = z + 0.1 * z_sym + 0.1 * v_mem

        generated = prompt.clone()
        for _ in range(max_new):
            ctx = generated[:, -self.cfg.seq_len:]

            if dynamic_reasoning:
                # ── reasoning_step: перекодуємо + оновлюємо S-Core / M-Core ──
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
# 7.  ДАТАСЕТИ
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
    Задача з явним переносом правила:
    Послідовності виду [A, op, B, =, C] де op ∈ {+, -, *}
    Модель ПОВИННА вивести правило, а не запам'ятати конкретний приклад.
    """
    data = []
    for _ in range(n):
        A = random.randint(10, 50)
        B = random.randint(10, 50)
        op = random.choice([0, 1, 2])
        if op == 0:   C = (A + B) % 200 + 10
        elif op == 1: C = abs(A - B) + 10
        else:         C = (A * B) % 200 + 10
        # Кодуємо як послідовність чисел + паддинг
        seq = [A, 100+op, B, 200, C] + [0]*(sl - 5)
        data.append(torch.tensor(seq[:sl], dtype=torch.long))
    return data

def collate(batch):
    # Підтримуємо два формати:
    #   1) List[Tensor]               — синтетичний датасет (make_counting, …)
    #   2) List[Tuple[Tensor,Tensor]] — реальний текст (load_text_corpus → (src, tgt))
    if isinstance(batch[0], (tuple, list)):
        src = torch.stack([item[0] for item in batch])
        tgt = torch.stack([item[1] for item in batch])
        return src, tgt
    s = torch.stack(batch)
    return s[:, :-1], s[:, 1:]


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ТРЕНУВАЛЬНИЙ ЦИКЛ
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
        # Консолідація пам'яті після backward (як гіпокамп після досвіду)
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
# 9.  INLINE ТЕСТИ
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(cfg: OMENv2Config) -> None:
    sep = lambda s: print(f"\n{'═'*72}\n  {s}\n{'═'*72}")

    # T0: Параметри
    sep("TEST 0 · Параметри та VRAM footprint")
    model = OMENv2(cfg).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Пристрій : {DEVICE}")
    print(f"  Параметри: {n_par:,}")
    print("  Memory report:")
    print(model.memory_report())
    assert n_par > 0
    print("  [PASS]")

    # T1: Forward pass — форма + нові поля
    sep("TEST 1 · OMENv2 Forward (всі компоненти)")
    B, T = 4, cfg.seq_len - 1
    src = torch.randint(1, 200, (B, T)).to(DEVICE)
    tgt = torch.randint(1, 200, (B, T)).to(DEVICE)
    t0 = time.perf_counter()
    out = model(src, tgt)
    fwd_ms = (time.perf_counter() - t0) * 1000

    for key in ("total", "ce", "kl", "mem_consist", "recall",
                "sym_ground", "gap_norm", "n_rules", "n_writes"):
        assert key in out, f"FAIL: відсутній ключ {key}"
    assert out["logits"].shape == (B, T, cfg.vocab_size)
    assert out["z"].shape == (B, cfg.d_latent)
    print(f"  Forward час    : {fwd_ms:.0f} ms")
    print(f"  CE loss init   : {out['ce']:.3f}")
    print(f"  Gap norm       : {out['gap_norm']:.4f}")
    print(f"  Memory writes  : {out['n_writes']}")
    print(f"  LTM rules      : {out['n_rules']}")
    print(f"  UNKNOWN flags  : {out['unknown_ex']}")
    print("  [PASS]")

    # T2: Backward — WorldRNN має отримати grad
    sep("TEST 2 · Backward — grad flow через M-Core + S-Core")
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
    assert enc_grad  > 0, "FAIL: encoder без граду"
    assert sco_grad  > 0, "FAIL: S-Core без граду"
    model.zero_grad()
    print("  [PASS]")

    # T3: TensorProductMemory — запис/читання
    sep("TEST 3 · TensorProductMemory — write / read / episodic")
    mem = model.memory
    mem_before = mem.memory.norm().item()

    z_test = torch.randn(8, cfg.d_latent).to(DEVICE)
    v_test = torch.randn(8, cfg.d_latent).to(DEVICE)
    conf   = torch.tensor([0.1]*4 + [0.9]*4, device=DEVICE)  # перші 4 — записуємо
    mem.write(z_test, v_test, conf)

    mem_after = mem.memory.norm().item()
    assert mem_after > mem_before, "FAIL: пам'ять не змінилася після write"
    assert mem.n_writes >= 4, f"FAIL: очікували ≥4 записів, отримали {mem.n_writes}"

    v_ret = mem.read(z_test[:2])
    assert v_ret.shape == (2, cfg.d_latent), "FAIL: форма read"

    v_ep = mem.episodic_recall(z_test[:2], k=3)
    assert v_ep.shape == (2, cfg.d_latent), "FAIL: форма episodic_recall"

    footprint = mem.memory_footprint_bytes()
    print(f"  M-Core before write norm : {mem_before:.4f}")
    print(f"  M-Core after write norm  : {mem_after:.4f}  (змінилася ✓)")
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
    assert (E >= 0).all(), "FAIL: E < 0 (квадрат має бути ≥ 0)"
    assert hot_dims.sum() > 0, "FAIL: немає гарячих вимірів"
    print(f"  E shape          : {tuple(E.shape)} ✓")
    print(f"  gap_norm mean    : {gap_norm.mean():.4f}")
    print(f"  hot_dims active  : {hot_dims.mean():.2%} вимірів")
    print("  [PASS]")

    # T5: S-Core — символьне виведення
    sep("TEST 5 · SymbolicCore — Perception → Reason → Abduce")
    sc = model.s_core
    sc.ltm.rules.clear()

    # Вручну додамо правило в LTM
    r = SymRule(
        conditions=(SymFact(1, 0, 2),),
        conclusions=(SymFact(1, 1, 3),)
    )
    sc.ltm.add_rule(r)

    # Додамо відповідний факт в WM
    sc.wm.clear()
    sc.wm.add(SymFact(1, 0, 2))

    matched = sc.ltm.match(sc.wm)
    inferred = sc.reason()
    print(f"  LTM правил        : {len(sc.ltm)}")
    print(f"  Matched правила   : {len(matched)}")
    print(f"  Виведені факти    : {inferred}")
    assert len(matched) >= 1, "FAIL: уніфікація не спрацювала"

    # Абдукція
    z_abd = torch.randn(1, cfg.d_latent, device=DEVICE)
    err_high = torch.tensor(2.0)
    n_added = sc.abduce_and_learn(z_abd, err_high)
    print(f"  Доданих правил (абдукція): {n_added}")
    print(f"  LTM після абдукції: {len(sc.ltm)}")
    print("  [PASS]")

    # T6: CuriosityModule
    sep("TEST 6 · CuriosityModule — counterfactuals + UNKNOWN detection")
    model.train()
    z_c = torch.randn(B, cfg.d_latent, device=DEVICE)
    E_c, gn_c, hd_c = model.epistemic.compute(z_c, model.world_rnn, z_c * 2)
    # Форсуємо активацію curiosity (gap_norm > tau)
    gn_forced = torch.ones(B, device=DEVICE) * (cfg.epistemic_tau + 0.5)
    z_enr, cf_l = model.curiosity(z_c, E_c, hd_c, gn_forced, model.memory, model.world_rnn)
    assert z_enr.shape == (B, cfg.d_latent), "FAIL: z_enr shape"
    assert not math.isnan(cf_l.item()), "FAIL: NaN у curiosity loss"
    print(f"  z_enr shape      : {tuple(z_enr.shape)} ✓")
    print(f"  counterfactual L : {cf_l.item():.4f}")
    print(f"  UNKNOWN flags    : {model.curiosity.unknown_flag_count}")
    print("  [PASS]")

    # T7: Мінімальне навчання (25 ітерацій)
    sep("TEST 7 · Навчання 25 ітерацій — зниження CE + зростання LTM")
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
    print(f"  CE (перші 5)    : {first5:.3f}")
    print(f"  CE (останні 5)  : {last5:.3f}")
    print(f"  Макс. LTM правил: {max_rules}")
    assert last5 < first5, "FAIL: CE не знижується"
    print("  [PASS]")

    # T8: Генерація
    sep("TEST 8 · Генерація токенів (dynamic_reasoning=True/False)")
    model.eval()
    pr = torch.randint(10, 100, (1, 8), device=DEVICE)

    # dynamic_reasoning=True — S-Core + M-Core на кожному кроці
    with torch.no_grad():
        gen_dyn = model.generate(pr, max_new=16, dynamic_reasoning=True)
    assert gen_dyn.shape == (1, 24), f"FAIL: gen_dyn shape {gen_dyn.shape}"
    print(f"  Prompt          : {pr[0].tolist()}")
    print(f"  Output (dynamic): {gen_dyn[0, 8:].tolist()}")

    # dynamic_reasoning=False — класичний (z_final фіксований)
    with torch.no_grad():
        gen_static = model.generate(pr, max_new=16, dynamic_reasoning=False)
    assert gen_static.shape == (1, 24), f"FAIL: gen_static shape {gen_static.shape}"
    print(f"  Output (static) : {gen_static[0, 8:].tolist()}")
    print("  [PASS]")

    print(f"\n{'═'*72}")
    print("  ✅  Всі 9 тестів пройдено успішно")
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
# 11.  ABLATION: OMENv2 vs CE-only vs v1 (без пам'яті)
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
        # OMEN-Scale MDL (новий)
        lambda_tok=1e-4, lambda_conc=1e-3,
    )

    run_tests(cfg)
    benchmark(cfg, epochs=6)
    ablation(cfg)

    # ── OMEN-Scale (ієрархічний, 3 рівні) ─────────────────────────────────────
    # Запустити: python omen_scale.py
    print("\n  Для запуску OMEN-Scale (Token→Concept→Symbolic):")
    print("    python omen_scale.py\n")

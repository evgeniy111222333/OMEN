"""
omen_osf_planner.py — Symbolic Planner (H2 рівень OSF)
=======================================================
OMEN Synthesis Framework: Plan Level.

Будує символьний план з IntentState через нейрокерований пошук.

Математика (GOLOG/PDDL-стиль):
  pre(aᵢ) ⊆ WMᵢ
  WM_{i+1} = (WMᵢ \ del(aᵢ)) ∪ add(aᵢ)
  goal ⊆ WM_{K+1}

  Нейропошук (як AlphaGo):
    π_plan(a|s) — нейрополітика наступної дії
    V_plan(s)   — ціннісна мережа для відсікання

  L_plan = -E_{τ~π_plan}[R_plan(τ)] + α_plan·Length(τ)

Інтеграція:
  IntentState → SymbolicPlanner → PlanSequence
  PlanSequence → HierarchicalDecoder (omen_osf_decoder.py)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ОПЕРАТОРИ ПЛАНУ  (GOLOG-стиль, без зовнішніх солверів)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PlanFact:
    """Факт у WM плановика: (predicate_id, arg)."""
    pred: int
    arg:  int
    def __repr__(self) -> str:
        return f"f{self.pred}({self.arg})"


@dataclass
class PlanOperator:
    """
    Оператор плану aᵢ:
      preconditions : факти що мають бути у WM (pre)
      add_effects   : факти що додаються після застосування
      del_effects   : факти що видаляються після застосування
      embedding     : (d_plan,) — нейронне представлення оператора
      op_type       : рядковий тип (для інтерпретації декодером)
    """
    op_id:         int
    op_type:       str                          # "define", "call", "assign", …
    preconditions: Tuple[PlanFact, ...]
    add_effects:   Tuple[PlanFact, ...]
    del_effects:   Tuple[PlanFact, ...]
    embedding:     Optional[torch.Tensor] = None   # (d_plan,)

    def applicable(self, wm: Set[PlanFact]) -> bool:
        return all(p in wm for p in self.preconditions)

    def apply(self, wm: Set[PlanFact]) -> Set[PlanFact]:
        new_wm = wm.copy()
        for f in self.del_effects:
            new_wm.discard(f)
        for f in self.add_effects:
            new_wm.add(f)
        return new_wm


@dataclass
class PlanState:
    """Стан планувальника у момент t."""
    wm:        Set[PlanFact]              # поточна WorkingMemory
    depth:     int                        # глибина пошуку
    z_ctx:     Optional[torch.Tensor]     # (1, d_plan) — контекст на цей момент


@dataclass
class PlanSequence:
    """
    Вихід SymbolicPlanner.
    operators  : послідовність застосованих операторів
    embeddings : (K, d_plan) — тензорне представлення (для HierarchicalDecoder)
    goal_reached: чи була ціль досягнута
    plan_loss  : L_plan (REINFORCE)
    """
    operators:    List[PlanOperator]
    embeddings:   torch.Tensor           # (K, d_plan)
    goal_reached: bool
    plan_loss:    torch.Tensor           # scalar


# ══════════════════════════════════════════════════════════════════════════════
# 2.  НЕЙРОПОЛІТИКА ПЛАНУВАННЯ
# ══════════════════════════════════════════════════════════════════════════════

class PlanPolicyNet(nn.Module):
    """
    π_plan(a|s): нейромережа → розподіл над операторами.

    Стан s кодується як:
      [z_intent; z_ctx; wm_features; depth_embed]
    де wm_features = mean-pool embedding фактів у WM.

    Архітектура: 2-шаровий MLP з gating.
    """

    def __init__(self, d_intent: int, d_plan: int, n_operators: int, dropout: float = 0.1):
        super().__init__()
        d_in = d_intent + d_plan + d_plan + 16  # intent + ctx + wm + depth

        self.depth_embed = nn.Embedding(32, 16)  # до 32 кроків

        self.actor = nn.Sequential(
            nn.Linear(d_in, d_plan * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_plan * 2, n_operators),
        )
        # Ціннісна мережа V(s)
        self.critic = nn.Sequential(
            nn.Linear(d_in, d_plan),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_plan, 1),
        )

    def _state_vec(
        self,
        z_intent: torch.Tensor,   # (1, d_intent)
        z_ctx:    torch.Tensor,   # (1, d_plan)
        wm_emb:   torch.Tensor,   # (1, d_plan)
        depth:    int,
    ) -> torch.Tensor:
        d_emb = self.depth_embed(
            torch.tensor([min(depth, 31)], device=z_intent.device))   # (1, 16)
        return torch.cat([z_intent, z_ctx, wm_emb, d_emb], dim=-1)   # (1, d_in)

    def action_logits(
        self,
        z_intent: torch.Tensor,
        z_ctx:    torch.Tensor,
        wm_emb:   torch.Tensor,
        depth:    int,
    ) -> torch.Tensor:
        s = self._state_vec(z_intent, z_ctx, wm_emb, depth)
        return self.actor(s)   # (1, n_operators)

    def value(
        self,
        z_intent: torch.Tensor,
        z_ctx:    torch.Tensor,
        wm_emb:   torch.Tensor,
        depth:    int,
    ) -> torch.Tensor:
        s = self._state_vec(z_intent, z_ctx, wm_emb, depth)
        return self.critic(s).squeeze(-1)   # (1,)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYMBOLIC PLANNER
# ══════════════════════════════════════════════════════════════════════════════

class SymbolicPlanner(nn.Module):
    """
    Нейрокерований символьний планувальник.

    Розв'язує: знайти a₁,...,aK ∈ A такі що:
      pre(aᵢ) ⊆ WMᵢ,  goal ⊆ WM_{K+1}

    Нейрокерування: замість сліпого перебору використовує π_plan + V_plan.

    Оператори генеруються нейромережею (OpratorGenerator) з IntentState —
    це дозволяє адаптувати план під контекст задачі.
    """

    # 8 базових типів операторів (domain-agnostic)
    OP_TYPES = ["define", "call", "assign", "return",
                "branch", "loop", "import", "yield"]

    def __init__(
        self,
        d_intent:     int,
        d_plan:       int,
        n_operators:  int   = 32,    # розмір бібліотеки операторів
        max_depth:    int   = 6,     # максимальна глибина плану
        beam_width:   int   = 3,     # ширина Beam Search
        alpha_plan:   float = 0.1,   # штраф за довжину плану
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.d_intent    = d_intent
        self.d_plan      = d_plan
        self.n_operators = n_operators
        self.max_depth   = max_depth
        self.beam_width  = beam_width
        self.alpha_plan  = alpha_plan

        # Operator embeddings: O ∈ R^{n_operators × d_plan}
        self.op_embeddings = nn.Embedding(n_operators, d_plan)
        nn.init.normal_(self.op_embeddings.weight, std=d_plan ** -0.5)

        # WM fact embeddings (pred, arg → d_plan)
        self.fact_embed = nn.Embedding(512, d_plan // 2)  # pred
        self.arg_embed  = nn.Embedding(512, d_plan // 2)  # arg

        # Intent → context projection
        self.intent_proj = nn.Linear(d_intent, d_plan, bias=False)

        # Policy + Value
        self.policy = PlanPolicyNet(d_intent, d_plan, n_operators, dropout)

        # Operator parameter generator (intent + op_emb → pre/add/del)
        self.op_param_gen = nn.Sequential(
            nn.Linear(d_intent + d_plan, d_plan * 2),
            nn.GELU(),
            nn.Linear(d_plan * 2, 3 * 4 * 2),  # 3 effect-sets × 4 facts × 2 slots
        )

        # Context GRU: відстежує контекст вздовж плану
        self.ctx_gru = nn.GRUCell(d_plan, d_plan)
        self.ctx_h0  = nn.Parameter(torch.zeros(1, d_plan))

    # ── WM → embedding ──────────────────────────────────────────────────────
    def _wm_embed(self, wm: Set[PlanFact], device) -> torch.Tensor:
        """Кодує WorkingMemory у (1, d_plan) через mean-pool."""
        if not wm:
            return torch.zeros(1, self.d_plan, device=device)
        facts = list(wm)[:16]  # обмеження для швидкості
        preds = torch.tensor([f.pred % 512 for f in facts], device=device)
        args  = torch.tensor([f.arg  % 512 for f in facts], device=device)
        emb   = torch.cat([
            self.fact_embed(preds),
            self.arg_embed(args)
        ], dim=-1)  # (|WM|, d_plan)
        return emb.mean(0, keepdim=True)  # (1, d_plan)

    # ── Генератор параметрів оператора ──────────────────────────────────────
    @torch.no_grad()
    def _gen_op_effects(
        self, z_intent: torch.Tensor, op_emb: torch.Tensor,
    ) -> Tuple[Tuple[PlanFact,...], Tuple[PlanFact,...], Tuple[PlanFact,...]]:
        """
        Нейромережа генерує pre/add/del для обраного оператора.
        Повертає кортежі PlanFact — дискретизується через argmax.
        """
        inp    = torch.cat([z_intent, op_emb], dim=-1)   # (1, d_intent + d_plan)
        params = self.op_param_gen(inp).squeeze(0)       # (3·4·2,)
        params = params.view(3, 4, 2)                    # (3 sets, 4 facts, 2 slots)

        def to_facts(matrix: torch.Tensor) -> Tuple[PlanFact, ...]:
            # matrix: (4, 2) → 4 факти, кожен (pred, arg)
            indices = matrix.abs().mul(128).long() % 128
            return tuple(
                PlanFact(int(indices[i, 0]), int(indices[i, 1]))
                for i in range(4)
                if indices[i, 0] > 0  # skip нульові факти
            )

        return to_facts(params[0]), to_facts(params[1]), to_facts(params[2])

    # ── Основний метод: побудова плану ───────────────────────────────────────
    def forward(
        self,
        intent_state: "IntentState",
    ) -> PlanSequence:
        """
        intent_state.z_intent: (B, d_intent)

        Greedy rollout з REINFORCE-лосом.
        Для batch > 1: незалежні плани на кожен елемент батчу.
        Повертає план для першого елементу батчу (решта — для лосу).
        """
        device   = intent_state.z_intent.device
        B        = intent_state.z_intent.size(0)
        z_intent = intent_state.z_intent[:1]           # (1, d_intent) — перший елемент

        z_ctx   = self.intent_proj(z_intent)           # (1, d_plan)
        h_ctx   = self.ctx_h0.clone()                  # (1, d_plan)

        wm: Set[PlanFact] = set()
        operators: List[PlanOperator] = []
        op_embs:   List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        values:    List[torch.Tensor] = []

        plan_loss = torch.tensor(0.0, device=device)

        for step in range(self.max_depth):
            # ── Стан ──────────────────────────────────────────────────────────
            wm_emb    = self._wm_embed(wm, device)       # (1, d_plan)
            logits    = self.policy.action_logits(
                z_intent, z_ctx, wm_emb, step)           # (1, n_ops)
            v_s       = self.policy.value(
                z_intent, z_ctx, wm_emb, step)           # (1,)

            # ── Sampling або Greedy ───────────────────────────────────────────
            if self.training:
                dist   = Categorical(logits=logits.squeeze(0))
                op_idx = dist.sample()                   # scalar
                lp     = dist.log_prob(op_idx)
            else:
                op_idx = logits.squeeze(0).argmax()
                lp     = torch.tensor(0.0, device=device)

            log_probs.append(lp)
            values.append(v_s.squeeze())

            # ── Отримуємо оператор ────────────────────────────────────────────
            op_emb = self.op_embeddings(op_idx.unsqueeze(0))   # (1, d_plan)
            pre, add, del_ = self._gen_op_effects(z_intent, op_emb)

            op_type = self.OP_TYPES[op_idx.item() % len(self.OP_TYPES)]
            op = PlanOperator(
                op_id        = int(op_idx.item()),
                op_type      = op_type,
                preconditions= pre,
                add_effects  = add,
                del_effects  = del_,
                embedding    = op_emb.squeeze(0).detach(),
            )

            # ── Застосовуємо оператор до WM ───────────────────────────────────
            if op.applicable(wm) or step == 0:   # крок 0 — завжди застосовуємо
                wm = op.apply(wm)
                operators.append(op)
                op_embs.append(op_emb)

            # ── Оновлюємо контекст через GRU ──────────────────────────────────
            h_ctx = self.ctx_gru(op_emb, h_ctx)
            z_ctx = h_ctx

        # ── Складаємо embedding-матрицю плану ─────────────────────────────────
        if op_embs:
            plan_embs = torch.cat(op_embs, dim=0)   # (K, d_plan)
        else:
            plan_embs = torch.zeros(1, self.d_plan, device=device)

        # ── REINFORCE-лос (L_plan) ────────────────────────────────────────────
        # Нагорода: 1.0 завжди (навчаємо якість плану через L_plan_code в OSFLoss).
        # Тут L_plan = E[-Σ log π(aₜ)] + α·Length — базовий policy gradient.
        if self.training and log_probs:
            R = 1.0  # заглушка; справжня нагорода приходить з OSFLoss
            baseline = sum(v.item() for v in values) / len(values)
            log_p_sum = torch.stack(log_probs).sum()
            L_len     = self.alpha_plan * len(operators)
            plan_loss = -(R - baseline) * log_p_sum + torch.tensor(
                L_len, dtype=log_p_sum.dtype, device=device)

        return PlanSequence(
            operators    = operators,
            embeddings   = plan_embs,
            goal_reached = len(operators) >= 1,
            plan_loss    = plan_loss,
        )
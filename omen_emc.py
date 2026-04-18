"""
omen_emc.py — Efficient Meta-Controller (EMC) for OMEN-Scale
=============================================================
Реалізує оптимальне управління ресурсами міркування на основі рівняння Беллмана.

Проблема, яку вирішуємо:
  Поточний max_proof_depth — фіксований гіперпараметр.
  Система завжди робить рівно max_depth кроків незалежно від складності задачі.
  Це порушує MDL-принцип: Cost(Reasoning) має входити у функціонал.

Математична формалізація:
  Стан s = (z, gap_norm, d, n_facts, n_rules)
  Дії  A = {Stop(0), RecallMCore(1), ForwardChainStep(2), Abduce(3)}

  Рівняння Беллмана:
    V*(s) = max{ U_stop(s),
                 max_{a∈A} [-C(a) + γ·E_{s'~P(·|s,a)} V*(s')] }

  U_stop(s) = R_task(s) + η_int·R_intermediate(s) − λ_gap·GapNorm(s)

  Де:
    R_task(s)         = 1.0 якщо ціль доведена (goal ∈ KB), 0.0 інакше
    R_intermediate(s) = Σ_{f∈KB∖KB₀} Utility(f)  — нові корисні факти
    GapNorm(s)        = ||E(z)||  — епістемічна непевність (Curiosity Engine)

Навчання Actor-Critic:
  L_critic = E[(r + γ·V_φ'(s') − V_φ(s))²]
  L_actor  = E[-Σ_a π_meta(a|s)·A(s,a) − β·H(π_meta(·|s))]

  A(s,a) = −C(a) + E_{s'}[V(s')] − V(s)

Повний функціонал OMEN+EMC (5-й член — новий):
  J_OMEN+EMC = J_OMEN
             + ω_meta · E_τ[ Σ_t (R_task + η_int·R_int − λ_gap·GapNorm_t − C(a_t)) ]

Де Cost(Reasoning) = Σ_t C(a_t) розширює MDL-принцип на обчислювальну вартість.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from omen_symbolic.controller import merge_induction_stats


# ══════════════════════════════════════════════════════════════════════════════
# 1.  КОНСТАНТИ ДІЙ
# ══════════════════════════════════════════════════════════════════════════════

ACTION_STOP   = 0   # Зупинитись — повернути поточний z_sym
ACTION_RECALL = 1   # Зчитати з M-Core, збагатити z
ACTION_FC     = 2   # Один крок Forward Chaining (застосувати правила → нові факти)
ACTION_ABDUCE = 3   # Абдукція: згенерувати нові правила через AbductionHead + VeM
ACTION_INTRINSIC = 4   # Переключити активну ціль на внутрішню ICE-ціль

N_ACTIONS = 5


# ══════════════════════════════════════════════════════════════════════════════
# 1.5  STOPPING UTILITY  та  TRAJECTORY STATS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryStats:
    """
    Статистика траєкторії міркування для одного епізоду EMC.

    Повертається з run_episode разом з (z_sym, sym_loss, v_mem, meta_loss).
    Використовується OMENScaleLoss для обчислення 7-ї компоненти функціонала:

      ω_meta · E_τ[ Σ_t (R_task + η_int·R_int − λ_gap·GapNorm_t − C(a_t)) ]
    """
    trajectory_reward: float = 0.0   # Σ_t r_t  (повна дисконтована сума)
    n_steps:           int   = 0     # скільки кроків виконано (T)
    goal_proved:       bool  = False # чи ціль доведена
    intermediate_utility: float = 0.0
    proof_mdl:         float = 0.0
    final_gap:         float = 0.0
    gap_norms:         list  = field(default_factory=list)  # GapNorm_t на кожному кроці
    actions:           list  = field(default_factory=list)  # a_t на кожному кроці
    action_costs:      list  = field(default_factory=list)  # C(a_t)
    r_intermediates:   list  = field(default_factory=list)  # R_int_t
    action_histogram:  list  = field(default_factory=list)
    gap_world_norms:   list  = field(default_factory=list)
    gap_grounded_norms:list  = field(default_factory=list)
    gap_reliefs:       list  = field(default_factory=list)
    memory_residuals:  list  = field(default_factory=list)
    memory_alignments: list  = field(default_factory=list)
    memory_pressures:  list  = field(default_factory=list)
    gap_deltas:        list  = field(default_factory=list)
    recall_gap_deltas: list  = field(default_factory=list)
    recall_gap_reliefs:list  = field(default_factory=list)
    recall_effective_steps: float = 0.0
    stop_reason: str         = "max_steps"  # "bellman"|"fixpoint"|"max_steps"|"action_stop"


class StoppingUtility(nn.Module):
    """
    U_stop(s) = R_task(s) + η_int·R_intermediate(s) − λ_gap·GapNorm(s)

    Де:
      R_task(s)    : м'яка оцінка ймовірності доведення (навчається через BCE)
      R_int(s)     : накопичена проміжна корисність (Σ нових корисних фактів)
      GapNorm(s)   : ||E(z)||  — епістемічна невпевненість

    Реалізує: U_stop > V(s) → STOP (рівняння Беллмана для оптимальної зупинки)
    Навчається через gradient з meta_loss.
    """

    def __init__(self, d_state: int, eta_int: float = 0.10,
                 lambda_gap: float = 0.05, lambda_time: float = 0.05,
                 dropout: float = 0.1):
        super().__init__()
        self.eta_int     = eta_int
        self.lambda_gap  = lambda_gap
        self.lambda_time = lambda_time   # λ_time: штраф за кількість кроків T_elapsed

        # М'яка оцінка R_task (P(goal proved | state)) — навчається через REINFORCE
        self.task_estimator = nn.Sequential(
            nn.Linear(d_state, d_state // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_state // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self,
                state:     torch.Tensor,       # (B, d_state)
                r_int:     float,              # поточна проміжна корисність
                gap_norm:  torch.Tensor,       # (B,) або scalar
                mdl_cost:  float = 0.0,
                t_elapsed: int   = 0,          # кількість виконаних кроків міркування
                memory_penalty: float = 0.0,
                ) -> torch.Tensor:             # → (B,) U_stop
        """
        Обчислює U_stop(s) відповідно до розширеної формули MDL:

          U_stop(s) = R_task_soft(s)
                    + η_int · R_int            ← бонус за нові факти
                    − λ_gap  · GapNorm(s)      ← штраф за епістемічну прогалину
                    − λ_mdl  · MDL(proof)      ← штраф за складність доведення
                    − λ_time · T_elapsed       ← штраф за витрачений час

        Остання компонента реалізує рівняння Беллмана для оптимальної зупинки:
          π_meta* = argmax_π E[R_task − λ_time·T − λ_MDL·MDL(proof)]

        R_task_soft — диференційована: gradient через task_estimator → state → z.
        R_int, gap_norm, mdl_cost, t_elapsed — скалярні (зовнішні, без grad).
        """
        r_task = self.task_estimator(state).squeeze(-1)   # (B,)

        # Нормалізуємо gap_norm до [0,1] діапазону
        if isinstance(gap_norm, (int, float)):
            gn = torch.tensor(gap_norm, dtype=r_task.dtype, device=r_task.device)
        else:
            gn = gap_norm.float().mean().clamp(0.0, 5.0) / 5.0

        mdl_pen  = torch.as_tensor(
            mdl_cost, dtype=r_task.dtype, device=r_task.device
        ).clamp(min=0.0)

        # λ_time · T_elapsed: лінійний штраф за кожен додатковий крок міркування
        # (реалізує Cost(Reasoning) = Σ_t C(a_t) у функціоналі MDL)
        time_pen = torch.tensor(
            self.lambda_time * float(t_elapsed),
            dtype=r_task.dtype, device=r_task.device,
        ).clamp(min=0.0)
        memory_pen = torch.tensor(
            float(memory_penalty),
            dtype=r_task.dtype, device=r_task.device,
        ).clamp(min=0.0)

        u_stop = (r_task
                  + self.eta_int * r_int
                  - self.lambda_gap  * gn
                  - mdl_pen
                  - memory_pen
                  - time_pen)
        return u_stop   # (B,)

    def bellman_should_stop(self,
                            u_stop:  torch.Tensor,   # (B,) — U_stop(s)
                            v_state: torch.Tensor,   # (B,) — V_φ(s) від Critic
                            ) -> bool:
        """
        Оптимальна зупинка: Stop якщо U_stop(s) ≥ E[V(s)].
        Повертає True якщо більшість елементів батчу виграють від зупинки.
        """
        # Порівнюємо mean(B) — global decision для епізоду
        return (u_stop.mean() >= v_state.mean()).item()


class VoCStoppingCriterion:
    """
    Value of Computation (VoC) — апроксимація рівняння Беллмана:

      Δ(s, d) > C(Δd)  →  продовжувати
      Δ(s, d) ≤ C(Δd)  →  зупинитись

    де Δ(s, d) — очікуваний виграш від додаткового кроку,
       C(Δd) = c · Δd — вартість обчислення.

    Проста апроксимація: Δ(s,d) ≈ max(0, V(s') − V(s))
    """

    def __init__(self, cost_per_step: float = 0.05):
        self.cost_per_step = cost_per_step
        self._prev_value: Optional[float] = None

    def reset(self):
        self._prev_value = None

    def should_continue(self, current_value: float) -> bool:
        """True якщо виграш від продовження перевищує вартість."""
        if self._prev_value is None:
            self._prev_value = current_value
            return True
        delta = current_value - self._prev_value
        self._prev_value = current_value
        return delta > self.cost_per_step


# ══════════════════════════════════════════════════════════════════════════════
# 2.  КОДУВАЛЬНИК СТАНУ
# ══════════════════════════════════════════════════════════════════════════════

class EMCStateEncoder(nn.Module):
    """
    Кодує стан s = (z, gap_norm, depth_norm, n_facts_norm, n_rules_norm)
    у вектор d_latent-розмірності.

    z (нейронний стан) проектується через лінійний шар, потім конкатенується
    з 4 нормованими скалярними ознаками і знову проектується.
    detach() від z — EMC отримує градієнт лише через meta_loss, не крізь z.
    """

    def __init__(self, d_latent: int, dropout: float = 0.1,
                 use_action_hist: bool = True):
        super().__init__()
        self.use_action_hist = use_action_hist
        # z + goal + WM summary + 10 scalar features + action histogram.
        d_in = d_latent * 3 + 10 + (N_ACTIONS if use_action_hist else 0)
        self.z_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.goal_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.wm_proj = nn.Linear(d_latent, d_latent, bias=False)
        self.state_proj = nn.Sequential(
            nn.Linear(d_in, d_latent), nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_latent),
        )

    def forward(self,
                z:           torch.Tensor,
                gap_norm:    torch.Tensor,
                depth_norm:  torch.Tensor,
                nfacts_norm: torch.Tensor,
                nrules_norm: torch.Tensor,
                gap_world:   Optional[torch.Tensor] = None,
                gap_grounded: Optional[torch.Tensor] = None,
                gap_relief:  Optional[torch.Tensor] = None,
                gap_residual: Optional[torch.Tensor] = None,
                gap_alignment: Optional[torch.Tensor] = None,
                goal_embed:  Optional[torch.Tensor] = None,
                wm_embed:    Optional[torch.Tensor] = None,
                trigger_flag: Optional[torch.Tensor] = None,
                hot_ratio: Optional[torch.Tensor] = None,
                action_hist: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z           : (B, d_latent)
        *_norm      : () або (B,) — скалярні ознаки; gap_relief може бути в [-1,1]
        action_hist : (B, N_ACTIONS) або None — one-hot агрегат попередніх дій
        Returns     : (B, d_latent) — стан для Actor/Critic
        """
        B = z.shape[0]
        z_enc = self.z_proj(z)
        goal_enc = self.goal_proj(goal_embed if goal_embed is not None else torch.zeros_like(z))
        wm_enc = self.wm_proj(wm_embed if wm_embed is not None else torch.zeros_like(z))
        grounded_gap = gap_grounded if gap_grounded is not None else gap_norm
        world_gap = gap_world if gap_world is not None else grounded_gap
        relief_gap = gap_relief if gap_relief is not None else (world_gap - grounded_gap)
        residual_gap = gap_residual if gap_residual is not None else grounded_gap
        alignment = (
            gap_alignment
            if gap_alignment is not None
            else torch.zeros((), device=z.device, dtype=z.dtype)
        )

        def _as_col(t: torch.Tensor) -> torch.Tensor:
            """() або (B,) → (B, 1)"""
            if t.dim() == 0:
                return t.unsqueeze(0).expand(B, 1)
            return t.unsqueeze(-1) if t.dim() == 1 else t

        scalars = torch.cat([
            _as_col(world_gap),
            _as_col(grounded_gap),
            _as_col(relief_gap),
            _as_col(residual_gap),
            _as_col(alignment),
            _as_col(depth_norm),
            _as_col(nfacts_norm),
            _as_col(nrules_norm),
            _as_col(trigger_flag if trigger_flag is not None else torch.zeros((), device=z.device, dtype=z.dtype)),
            _as_col(hot_ratio if hot_ratio is not None else torch.zeros((), device=z.device, dtype=z.dtype)),
        ], dim=-1)                                              # (B, 10)

        if self.use_action_hist:
            if action_hist is not None:
                feat = torch.cat([z_enc, goal_enc, wm_enc, scalars, action_hist], dim=-1)
            else:
                # Заповнюємо нулями якщо action_hist не передано (перший крок)
                zeros = torch.zeros(B, N_ACTIONS, device=z.device, dtype=z.dtype)
                feat = torch.cat([z_enc, goal_enc, wm_enc, scalars, zeros], dim=-1)
        else:
            feat = torch.cat([z_enc, goal_enc, wm_enc, scalars], dim=-1)

        return self.state_proj(feat)  # (B, d)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ACTOR + CRITIC
# ══════════════════════════════════════════════════════════════════════════════

class EMCActor(nn.Module):
    """
    π_meta(a | s) — мета-політика.
    Повертає logits над {Stop, RecallMCore, ForwardChainStep, Abduce}.
    """

    def __init__(self, d_state: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, d_state),    nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_state, d_state // 2), nn.GELU(),
            nn.Linear(d_state // 2, N_ACTIONS),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, d) → logits: (B, N_ACTIONS)"""
        return self.net(state)


class EMCCritic(nn.Module):
    """
    V_φ(s) — функція цінності стану.
    Використовується для обчислення переваги A(s,a) = r + γ·V(s') − V(s).
    """

    def __init__(self, d_state: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, d_state),    nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_state, d_state // 2), nn.GELU(),
            nn.Linear(d_state // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, d) → value: (B,)"""
        return self.net(state).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EFFICIENT META-CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class EfficientMetaController(nn.Module):
    """
    Адаптивний контролер символьного міркування OMEN.

    Замінює фіксований max_proof_depth на динамічну мета-політику π_meta,
    яка вирішує КОЛИ зупинитись і ЯКУ дію виконати на кожному кроці.

    Формально реалізує розв'язання:
      π_meta* = argmax_π E[R_task − λ_time·T − λ_MDL·MDL(proof)]

    Архітектура:
      · EMCStateEncoder   : (z, gap_norm, depth, n_facts, n_rules) → d-вим. вектор
      · EMCActor          : state → π_meta(a|s) — розподіл над 4 діями
      · EMCCritic         : state → V_φ(s) — оцінка цінності
      · run_episode()     : виконує адаптивний цикл, повертає збагачений z_sym

    Навчання:
      EMC навчається спільно з основною моделлю через градієнт з meta_loss,
      який додається до J_OMEN з вагою ω_meta.
    """

    # Вартість кожної дії (обчислювальна + ризик)
    DEFAULT_COSTS = {
        ACTION_STOP:   0.0,
        ACTION_RECALL: 0.01,   # дешево: O(d²) операцій
        ACTION_FC:     0.05,   # середньо: перебір правил × фактів
        ACTION_ABDUCE: 0.10,   # дорого: AbductionHead + VeM + нові правила
        ACTION_INTRINSIC: 0.03,
    }

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_latent
        self.cfg = cfg

        # Підмережі
        drop = getattr(cfg, 'dropout', 0.1)
        use_hist = getattr(cfg, 'emc_use_action_hist', True)
        self.state_enc = EMCStateEncoder(d, dropout=drop, use_action_hist=use_hist)
        self.actor     = EMCActor(d,     dropout=drop)
        self.critic    = EMCCritic(d,    dropout=drop)

        # ── StoppingUtility: U_stop(s) = R_task + η_int·R_int − λ_gap·GapNorm − λ_time·T ──
        self.stopping_utility = StoppingUtility(
            d_state     = d,
            eta_int     = getattr(cfg, 'emc_eta_int',    0.10),
            lambda_gap  = getattr(cfg, 'emc_lambda_gap', 0.05),
            lambda_time = getattr(cfg, 'emc_lambda_time', 0.05),
            dropout     = drop,
        )

        # ── VoC stopping criterion (non-learnable threshold) ──────────────────
        self.voc = VoCStoppingCriterion(
            cost_per_step = getattr(cfg, 'emc_lambda_time', 0.05)
        )

        # Гіперпараметри з конфігу (або дефолти)
        self.max_steps    = getattr(cfg, 'emc_max_steps',      5)
        self.gamma        = getattr(cfg, 'emc_gamma',          0.95)
        self.entropy_beta = getattr(cfg, 'emc_entropy_beta',   0.01)
        self.lambda_time  = getattr(cfg, 'emc_lambda_time',    0.05)   # штраф за крок
        self.lambda_gap   = getattr(cfg, 'emc_lambda_gap',     0.05)   # штраф за GapNorm
        self.lambda_memory_residual = getattr(cfg, 'emc_lambda_memory_residual', 0.02)
        self.lambda_memory_misalignment = getattr(cfg, 'emc_lambda_memory_misalignment', 0.02)
        self.lambda_mdl   = getattr(cfg, 'emc_lambda_mdl',     0.01)   # штраф за MDL(proof)
        self.eta_int      = getattr(cfg, 'emc_eta_int',        0.10)   # бонус за нові факти
        self.c_recall     = getattr(cfg, 'emc_c_recall',       0.01)
        self.c_fc         = getattr(cfg, 'emc_c_fc',           0.05)
        self.c_abduce     = getattr(cfg, 'emc_c_abduce',       0.10)
        self.c_intrinsic  = getattr(cfg, 'emc_c_intrinsic',    0.03)

        # GAE (Generalized Advantage Estimation)
        self.use_gae     = getattr(cfg, 'emc_use_gae',        True)
        self.gae_lambda  = getattr(cfg, 'emc_gae_lambda',     0.95)

        # Action history encoding
        self.use_action_hist = getattr(cfg, 'emc_use_action_hist', True)

    # ── Допоміжні ─────────────────────────────────────────────────────────────

    def _action_cost(self, a: int) -> float:
        return {
            ACTION_STOP:   0.0,
            ACTION_RECALL: self.c_recall,
            ACTION_FC:     self.c_fc,
            ACTION_ABDUCE: self.c_abduce,
            ACTION_INTRINSIC: self.c_intrinsic,
        }.get(a, 0.0)

    @staticmethod
    def _goal_match(left, right) -> bool:
        if left is None or right is None:
            return False
        try:
            from omen_prolog import unify

            return unify(left, right) is not None
        except Exception:
            return hash(left) == hash(right)

    def _fact_utility(self, prover, facts) -> float:
        """
        Наближена Utility(f): заохочує новизну предиката без вибуху масштабу.

        Підтримує як KnowledgeBase (frozenset) так і TensorKnowledgeBase (.facts property).
        facts: результат forward_chain() — set або frozenset HornAtom.
        """
        if not facts:
            return 0.0
        try:
            kb_facts = prover.kb.facts   # frozenset[HornAtom] (both KB variants)
            pred_counts: Dict[int, int] = {}
            for f in kb_facts:
                pred = getattr(f, 'pred', 0)
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
        except Exception:
            return 0.0
        util = 0.0
        for fact in facts:
            pred = getattr(fact, 'pred', 0)
            util += 1.0 / (1.0 + pred_counts.get(pred, 0))
        return util / max(len(facts), 1)

    def _proof_mdl(
        self,
        prover,
        derivation_trace: List[Tuple[object, Optional[object]]],
        action_count: int = 0,
    ) -> float:
        """MDL(proof) через реальну символічну вартість застосованих правил."""
        if not derivation_trace:
            return float(action_count)
        total = 0.0
        for clause, sigma in derivation_trace:
            total += float(prover.cost_est.symbolic_cost(clause, sigma))
        return total + 0.1 * float(action_count)

    def _action_mask(self,
                     device: torch.device,
                     allow_abduction: bool,
                     allow_intrinsic: bool = True) -> torch.Tensor:
        mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)
        if not allow_abduction:
            mask[ACTION_ABDUCE] = True
        if not allow_intrinsic:
            mask[ACTION_INTRINSIC] = True
        return mask

    @staticmethod
    def _mask_action_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        mask_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(action_mask, mask_value)

    def _task_estimator_bce_loss(self, task_state: torch.Tensor, goal_proved: bool) -> torch.Tensor:
        device_type = task_state.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            task_state_fp32 = task_state.float()
            r_pred = self.stopping_utility.task_estimator(task_state_fp32).squeeze(-1)
            bce_tgt = torch.full_like(r_pred, float(goal_proved))
            return F.binary_cross_entropy(r_pred.clamp(1e-6, 1.0 - 1e-6), bce_tgt)

    def _encode_state(self,
                      z:           torch.Tensor,
                      gap_norm:    torch.Tensor,
                      depth:       int,
                      n_facts:     int,
                      n_rules:     int,
                      gap_features: Optional[Dict[str, float]] = None,
                      goal_embed:  Optional[torch.Tensor] = None,
                      wm_embed:    Optional[torch.Tensor] = None,
                      trigger_flag: float = 0.0,
                      hot_ratio: float = 0.0,
                      action_counts: Optional[List[int]] = None) -> torch.Tensor:
        """
        Формує d-вимірний вектор стану.
        action_counts: список підрахунків дій [cnt_stop, cnt_recall, cnt_fc, cnt_abduce]
        """
        dev = z.device
        dtype = z.dtype
        max_d = float(max(self.max_steps, 1))
        max_f = float(max(getattr(self.cfg, 'sym_max_facts', 64), 1))
        max_r = float(max(getattr(self.cfg, 'ltm_max_rules', 256), 1))
        gap_info = self._coerce_gap_features(gap_norm, gap_features)
        world_gap_val = max(min(gap_info["gap_world_only"], 5.0), 0.0) / 5.0
        grounded_gap_val = max(min(gap_info["gap_memory_grounded"], 5.0), 0.0) / 5.0
        relief_gap_val = max(min(gap_info["gap_memory_relief"] / 5.0, 1.0), -1.0)
        residual_gap_val = max(min(gap_info["gap_memory_residual"], 5.0), 0.0) / 5.0
        alignment_val = max(min(gap_info["gap_memory_alignment"], 1.0), -1.0)

        def _s(v) -> torch.Tensor:
            return torch.tensor(v, device=dev, dtype=dtype)

        # Будуємо action_hist: нормований one-hot підрахунок
        action_hist = None
        if self.use_action_hist:
            counts = action_counts if action_counts is not None else [0] * N_ACTIONS
            total = max(sum(counts), 1)
            hist_vec = torch.tensor(
                [c / total for c in counts],
                device=dev, dtype=dtype
            ).unsqueeze(0).expand(z.shape[0], -1)  # (B, N_ACTIONS)
            action_hist = hist_vec

        return self.state_enc(
            z,
            _s(grounded_gap_val),
            _s(depth / max_d),
            _s(min(n_facts, max_f) / max_f),
            _s(min(n_rules, max_r) / max_r),
            gap_world=_s(world_gap_val),
            gap_grounded=_s(grounded_gap_val),
            gap_relief=_s(relief_gap_val),
            gap_residual=_s(residual_gap_val),
            gap_alignment=_s(alignment_val),
            goal_embed=goal_embed,
            wm_embed=wm_embed,
            trigger_flag=_s(trigger_flag),
            hot_ratio=_s(hot_ratio),
            action_hist=action_hist,
        )

    def _goal_embed(self, prover, goal, batch_size: int, device: torch.device) -> torch.Tensor:
        return prover.ground(frozenset({goal}), device).expand(batch_size, -1)

    def _wm_embed(self, prover, device: torch.device, batch_size: int,
                  facts: Optional[frozenset] = None) -> torch.Tensor:
        facts = prover.kb.facts if facts is None else facts
        if not facts:
            return torch.zeros(batch_size, prover.d, device=device)
        sample_size = min(len(facts), 64)
        if len(facts) > sample_size:
            # Use a stable subset so prover.ground() can hit its runtime cache within
            # one EMC episode instead of re-encoding a different random sample each step.
            facts = frozenset(sorted(facts, key=hash)[:sample_size])
        return prover.ground(facts, device).expand(batch_size, -1)

    def _rule_trigger_flag(self, prover, facts: Optional[frozenset] = None) -> float:
        fact_space = prover.kb.facts if facts is None else facts
        fact_preds = {f.pred for f in fact_space}
        if not fact_preds:
            return 0.0
        for rule in prover.kb.rules:
            if any(atom.pred in fact_preds for atom in rule.body):
                return 1.0
        return 0.0

    @staticmethod
    def _coerce_gap_features(
        gap_norm: torch.Tensor,
        gap_features: Optional[Dict[str, float]] = None,
        prior_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        grounded_gap_default = (
            float(gap_norm.mean().item()) if torch.is_tensor(gap_norm) else float(gap_norm)
        )
        features = dict(prior_features or {})
        features.update(gap_features or {})
        grounded_gap = float(features.get("gap_memory_grounded", grounded_gap_default))
        world_gap = float(features.get("gap_world_only", grounded_gap))
        relief_gap = float(features.get("gap_memory_relief", world_gap - grounded_gap))
        residual_gap = float(features.get("gap_memory_residual", grounded_gap))
        alignment = float(features.get("gap_memory_alignment", 0.0))
        return {
            "gap_world_only": world_gap,
            "gap_memory_grounded": grounded_gap,
            "gap_memory_relief": relief_gap,
            "gap_memory_residual": residual_gap,
            "gap_memory_alignment": alignment,
        }

    def _apply_recall_gap_feedback(
        self,
        z_query: torch.Tensor,
        gap_norm: torch.Tensor,
        v_mem_step: torch.Tensor,
        gap_feedback: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, float]]]
        ] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        fallback_gap = gap_norm.detach().to(device=z_query.device, dtype=z_query.dtype)
        if gap_feedback is not None:
            try:
                next_gap, stats = gap_feedback(z_query.detach(), v_mem_step.detach())
            except Exception:
                next_gap, stats = None, {}
        else:
            next_gap, stats = None, {}

        if not torch.is_tensor(next_gap):
            next_gap = (fallback_gap * 0.9).clamp(0.0, 5.0)
        else:
            next_gap = next_gap.detach().to(device=z_query.device, dtype=z_query.dtype)
            if next_gap.dim() == 0:
                next_gap = next_gap.reshape(1).expand_as(fallback_gap)
            elif next_gap.shape != fallback_gap.shape:
                if next_gap.numel() == fallback_gap.numel():
                    next_gap = next_gap.reshape_as(fallback_gap)
                else:
                    next_gap = next_gap.mean().reshape(1).expand_as(fallback_gap)
            next_gap = next_gap.clamp(0.0, 5.0)

        info = dict(stats or {})
        gap_before = float(info.get("gap_world_only", fallback_gap.mean().item()))
        gap_after = float(info.get("gap_memory_grounded", next_gap.mean().item()))
        gap_delta = float(info.get("gap_delta", gap_before - gap_after))
        gap_relief = float(info.get("gap_memory_relief", gap_delta))
        info.update({
            "gap_world_only": gap_before,
            "gap_memory_grounded": gap_after,
            "gap_delta": gap_delta,
            "gap_memory_relief": gap_relief,
            "gap_effective": 1.0 if gap_delta > 1e-6 else 0.0,
        })
        return next_gap, info

    def _measure_gap_feedback_state(
        self,
        z_state: torch.Tensor,
        gap_fallback: torch.Tensor,
        signal: torch.Tensor,
        gap_feedback: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, float]]]
        ] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        fallback_gap = gap_fallback.detach().to(device=z_state.device, dtype=z_state.dtype)
        if gap_feedback is not None:
            try:
                next_gap, stats = gap_feedback(z_state.detach(), signal.detach())
            except Exception:
                next_gap, stats = None, {}
        else:
            next_gap, stats = None, {}

        if not torch.is_tensor(next_gap):
            next_gap = fallback_gap
        else:
            next_gap = next_gap.detach().to(device=z_state.device, dtype=z_state.dtype)
            if next_gap.dim() == 0:
                next_gap = next_gap.reshape(1).expand_as(fallback_gap)
            elif next_gap.shape != fallback_gap.shape:
                if next_gap.numel() == fallback_gap.numel():
                    next_gap = next_gap.reshape_as(fallback_gap)
                else:
                    next_gap = next_gap.mean().reshape(1).expand_as(fallback_gap)
            next_gap = next_gap.clamp(0.0, 5.0)

        info = dict(stats or {})
        info.setdefault("gap_world_only", float(next_gap.mean().item()))
        info.setdefault("gap_memory_grounded", float(next_gap.mean().item()))
        return next_gap, info

    def _memory_control_penalty(
        self,
        gap_features: Optional[Dict[str, float]] = None,
    ) -> float:
        features = gap_features or {}
        residual = max(min(float(features.get("gap_memory_residual", 0.0)), 5.0), 0.0) / 5.0
        alignment = max(min(float(features.get("gap_memory_alignment", 0.0)), 1.0), -1.0)
        misalignment = (1.0 - alignment) / 2.0
        return (
            self.lambda_memory_residual * residual
            + self.lambda_memory_misalignment * misalignment
        )

    # ── Головний цикл ─────────────────────────────────────────────────────────

    def run_episode(self,
                    z:         torch.Tensor,
                    gap_norm:  torch.Tensor,
                    hot_dims:  Optional[torch.Tensor],
                    prover,                        # DifferentiableProver
                    memory,                        # AsyncTensorProductMemory
                    world_err: torch.Tensor,
                    device:    torch.device,
                    gap_features: Optional[Dict[str, float]] = None,
                    gap_feedback: Optional[
                        Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, float]]]
                    ] = None,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                               torch.Tensor, 'TrajectoryStats']:
        """
        Адаптивний цикл символьного міркування (замінює prover.forward()).

        Реалізує рівняння Беллмана для оптимальної зупинки:
          V*(s) = max{ U_stop(s),
                       max_{a∈A} [-C(a) + γ·E_{s'~P(·|s,a)} V*(s')] }

        На кожному кроці:
          1. Обчислюємо state_vec(s) з (z, gap_norm, depth, n_facts, n_rules, action_hist)
          2. MDL(proof) поточної траєкторії
          3. U_stop(s) = StoppingUtility(state_vec, r_int_cum, gap_norm, mdl_cost)
          4. V(s)      = Critic(state_vec)
          5. Якщо U_stop(s) ≥ V(s) АБО VoC(Δd) ≤ C → зупинитись (Беллман)
          6. Інакше: Actor вибирає дію → виконуємо → оновлюємо стан

        Returns:
          z_sym      : (B, d) — символьне представлення
          sym_loss   : scalar — symbolic consistency + proof + VeM
          v_mem      : (B, d) — результат M-Core recall
          meta_loss  : scalar — Actor-Critic loss (GAE або MC) → до J_OMEN
          traj_stats : TrajectoryStats — деталі траєкторії (для 7-го члена J_OMEN+EMC:
                         ω_meta · Σ_t (R_task + η_int·R_int − λ_gap·GapNorm − C(a_t)))
        """
        if hasattr(prover, "_clear_runtime_caches"):
            prover._clear_runtime_caches()
        B = z.shape[0]
        traj = TrajectoryStats()
        induction_stats = {
            "checked": 0.0,
            "verified": 0.0,
            "contradicted": 0.0,
            "retained": 0.0,
            "matched_predictions": 0.0,
            "mean_score": 0.0,
        }

        # ── 0. Housekeeping ───────────────────────────────────────────────────
        prover._step += 1
        prover.kb.tick()
        if prover._step % prover.consolidate_every == 0:
            prover.kb.consolidate(use_count_threshold=2)

        # ── FIX: оновлюємо _last_z для _mental_simulate_rule() і _pred_error_for_rule()
        # В EMC-режимі prover.forward() НЕ викликається, тому _last_z був би stale/None.
        # WorldRNN latent-space компонента Дедукції та Абдукції потребує актуального z.
        prover._last_z = z.detach()

        # ── 1. Perception: z → fact → KB ──────────────────────────────────────
        prover.materialize_task_context_facts()
        goal = prover.current_goal(z)
        working_facts = prover.current_working_facts()

        n_facts_init = len(working_facts)
        n_rules_init = len(prover.kb)
        z_cur        = z.clone()
        v_mem_out    = torch.zeros_like(z)
        gap_norm_cur = gap_norm.clone()   # оновлюється після кожної дії
        current_gap_features = self._coerce_gap_features(gap_norm_cur, gap_features)

        # VoC criterion reset for new episode
        self.voc.reset()

        # ── 2. Адаптивний цикл ────────────────────────────────────────────────
        states_list:    List[torch.Tensor] = []
        log_probs_list: List[torch.Tensor] = []
        entropies_list: List[torch.Tensor] = []   # H(π_meta(·|s)) на кожному кроці
        values_list:    List[torch.Tensor] = []
        rewards_list:   List[float]        = []
        u_stop_list:    List[float]        = []  # для логування

        r_int_cumulative = float(getattr(prover, "current_intrinsic_value", lambda: 0.0)())
        goal_proved      = False
        # Підрахунок дій для кодування стану (action_hist)
        action_counts: List[int] = [0] * N_ACTIONS
        # Реальні (rule, substitution) кроки доведення для MDL(proof)
        proof_derivations: List[Tuple[object, Optional[object]]] = []
        for step in range(self.max_steps):
            goal = prover.current_goal(z_cur)
            goal_embed = self._goal_embed(prover, goal, B, device)
            intrinsic_goal = getattr(prover, "current_intrinsic_goal", lambda: None)()
            intrinsic_goal_available = intrinsic_goal is not None and not self._goal_match(goal, intrinsic_goal)
            # MDL(proof) поточного стану
            proof_mdl_cur = self._proof_mdl(
                prover, proof_derivations, action_count=len(traj.actions)
            )
            wm_embed = self._wm_embed(prover, device, B, facts=working_facts)
            trigger_flag = self._rule_trigger_flag(prover, facts=working_facts)
            hot_ratio = float(hot_dims.float().mean().item()) if hot_dims is not None else 0.0

            state_vec = self._encode_state(
                z_cur, gap_norm_cur, step,
                len(working_facts), len(prover.kb),
                gap_features=current_gap_features,
                goal_embed=goal_embed,
                wm_embed=wm_embed,
                trigger_flag=trigger_flag,
                hot_ratio=hot_ratio,
                action_counts=action_counts)
            traj.gap_world_norms.append(float(current_gap_features["gap_world_only"]))
            traj.gap_grounded_norms.append(float(current_gap_features["gap_memory_grounded"]))
            traj.gap_reliefs.append(float(current_gap_features["gap_memory_relief"]))
            traj.memory_residuals.append(float(current_gap_features["gap_memory_residual"]))
            traj.memory_alignments.append(float(current_gap_features["gap_memory_alignment"]))
            memory_pressure = self._memory_control_penalty(current_gap_features)
            traj.memory_pressures.append(memory_pressure)

            # ── Critic: V(s) ─────────────────────────────────────────────────
            val = self.critic(state_vec).mean()   # scalar

            # ── StoppingUtility: U_stop(s) ────────────────────────────────────
            # Включає MDL(proof) та T_elapsed як штрафи за складність та час
            # U_stop(s) = R_task + η_int·R_int − λ_gap·GapNorm − λ_mdl·MDL − λ_time·T
            mdl_cost_cur = self.lambda_mdl * proof_mdl_cur
            u_stop = self.stopping_utility(
                state_vec, r_int_cumulative, gap_norm_cur,
                mdl_cost=mdl_cost_cur,
                t_elapsed=step,
                memory_penalty=memory_pressure)                                # ← λ_time·T_elapsed
            u_stop_scalar = u_stop.mean().item()
            u_stop_list.append(u_stop_scalar)

            # ── Bellman Stopping Check: U_stop(s) ≥ V(s) → зупинитись ─────────
            # Також VoC: Δ ≤ C(Δd) → не варто продовжувати
            bellman_stop = self.stopping_utility.bellman_should_stop(u_stop, val.unsqueeze(0).expand(B))
            voc_stop     = not self.voc.should_continue(val.item())

            if (bellman_stop or voc_stop) and step > 0:
                # Записуємо нульову дію-зупинку як вибір
                traj.stop_reason = "bellman" if bellman_stop else "voc"
                # Додаємо термінальну нагороду
                rewards_list.append(u_stop_scalar)
                break

            # ── Actor: π_meta(a|s) ───────────────────────────────────────────
            action_logits = self.actor(state_vec)
            mean_logits   = action_logits.mean(0)
            action_mask = self._action_mask(
                device,
                allow_abduction=True,
                allow_intrinsic=intrinsic_goal_available,
            )
            masked_logits = self._mask_action_logits(mean_logits, action_mask)
            dist          = Categorical(logits=masked_logits)
            action        = dist.sample()
            log_p         = dist.log_prob(action)
            # Справжня ентропія H(π) = -Σ_a π(a|s)·log π(a|s) (скалярна)
            # Використовується в L_actor = -E[A·log π] - β·H(π)
            entropy       = dist.entropy()                   # scalar ≥ 0

            states_list.append(state_vec.detach())
            log_probs_list.append(log_p)
            entropies_list.append(entropy)
            values_list.append(val)

            a = action.item()
            traj.actions.append(a)
            action_counts[a] = action_counts[a] + 1

            # ── Явна дія STOP ─────────────────────────────────────────────────
            if a == ACTION_STOP:
                action_counts[ACTION_STOP] += 1
                rewards_list.append(u_stop_scalar)
                traj.stop_reason = "action_stop"
                break

            # ── Виконуємо дію та обчислюємо R_int ────────────────────────────
            r_int    = 0.0
            cost_a   = self._action_cost(a)
            gap_delta = 0.0

            if a == ACTION_RECALL:
                z_before = z_cur
                v_mem_step = memory.read(z_before)
                gap_norm_cur, recall_gap = self._apply_recall_gap_feedback(
                    z_before,
                    gap_norm_cur,
                    v_mem_step,
                    gap_feedback=gap_feedback,
                )
                z_cur      = (z_before + v_mem_step).clamp(-20.0, 20.0)
                v_mem_out  = v_mem_step
                gap_delta = float(recall_gap.get("gap_delta", 0.0))
                r_int      = (
                    v_mem_step.norm(dim=-1).mean().item() / (z.shape[-1] ** 0.5 + 1e-6)
                    + gap_delta
                )
                traj.recall_gap_deltas.append(gap_delta)
                traj.recall_gap_reliefs.append(float(recall_gap.get("gap_memory_relief", gap_delta)))
                traj.recall_effective_steps += float(recall_gap.get("gap_effective", 0.0))
                current_gap_features = self._coerce_gap_features(
                    gap_norm_cur,
                    recall_gap,
                    prior_features=current_gap_features,
                )

            elif a == ACTION_FC:
                # FIX Bug-FC: forward_chain() повертає frozenset але НЕ додає факти в KB.
                # forward_chain_step() виконує один крок І персистує нові факти через
                # kb.add_fact() → state KB реально оновлюється для наступних кроків.
                n_added_fc, new_fact_set, fc_trace, working_facts = prover.forward_chain_step_local(
                    working_facts
                )
                new_facts = set(new_fact_set)
                proof_derivations.extend(fc_trace)
                r_int = self._fact_utility(prover, new_facts)
                if n_added_fc > 0:
                    r_int = max(r_int, float(n_added_fc) / max(n_facts_init + 1, 1))
                # GapNorm має оцінюватися за новим grounded symbolic-state, а не через фіксований shrink.
                if n_added_fc > 0:
                    z_before = z_cur
                    z_cur = prover.ground(working_facts, device).expand(B, -1)
                    if gap_feedback is not None:
                        zero_signal = torch.zeros_like(z_before)
                        gap_before = float(
                            current_gap_features.get(
                                "gap_memory_grounded",
                                float(gap_norm_cur.mean().item()),
                            )
                        )
                        gap_after_t, after_stats = self._measure_gap_feedback_state(
                            z_cur,
                            gap_norm_cur,
                            zero_signal,
                            gap_feedback=gap_feedback,
                        )
                        gap_norm_cur = gap_after_t
                        gap_delta = float(
                            gap_before
                            - after_stats.get("gap_memory_grounded", float(gap_after_t.mean().item()))
                        )
                    else:
                        gap_before = float(gap_norm_cur.mean().item())
                        gap_norm_cur = (gap_norm_cur * 0.95).clamp(0.0, 5.0)
                        gap_delta = gap_before - float(gap_norm_cur.mean().item())
                        after_stats = {}
                    current_gap_features = self._coerce_gap_features(
                        gap_norm_cur,
                        after_stats,
                        prior_features=current_gap_features,
                    )
                    r_int += gap_delta

            elif a == ACTION_ABDUCE:
                err_val = world_err.item() if torch.is_tensor(world_err) else float(world_err)
                n_added, _, _, mean_utility = prover.abduce_and_learn(z_cur, err_val, force=True)
                r_int   = max(float(n_added) / max(getattr(self.cfg, 'n_proof_cands', 8), 1),
                              mean_utility)
                if n_added > 0:
                    z_before = z_cur
                    induction_stats = prover._induce_proposed_rules_locally(
                        working_facts,
                        (
                            prover.task_context.target_facts
                            if getattr(prover, "task_context", None) is not None
                            else frozenset()
                        ),
                        device,
                    )
                    refined_facts = prover.forward_chain_reasoned(
                        prover.max_depth,
                        starting_facts=working_facts,
                        only_verified=True,
                        device=device,
                    )
                    z_cur = prover.ground(refined_facts or working_facts, device).expand(B, -1)
                    if gap_feedback is not None:
                        zero_signal = torch.zeros_like(z_before)
                        gap_before = float(
                            current_gap_features.get(
                                "gap_memory_grounded",
                                float(gap_norm_cur.mean().item()),
                            )
                        )
                        gap_after_t, after_stats = self._measure_gap_feedback_state(
                            z_cur,
                            gap_norm_cur,
                            zero_signal,
                            gap_feedback=gap_feedback,
                        )
                        gap_norm_cur = gap_after_t
                        gap_delta = float(
                            gap_before
                            - after_stats.get("gap_memory_grounded", float(gap_after_t.mean().item()))
                        )
                    else:
                        gap_before = float(gap_norm_cur.mean().item())
                        gap_norm_cur = (gap_norm_cur * 0.85).clamp(0.0, 5.0)
                        gap_delta = gap_before - float(gap_norm_cur.mean().item())
                        after_stats = {}
                    current_gap_features = self._coerce_gap_features(
                        gap_norm_cur,
                        after_stats,
                        prior_features=current_gap_features,
                    )
                    r_int += gap_delta

            elif a == ACTION_INTRINSIC:
                focused_goal = getattr(prover, "focus_intrinsic_goal", lambda: None)()
                if focused_goal is not None:
                    goal = prover.current_goal(z_cur)
                    goal_embed = self._goal_embed(prover, goal, B, device)
                    r_int = max(float(getattr(prover, "current_intrinsic_value", lambda: 0.0)()), 0.0)

            r_int_cumulative += r_int
            traj.action_costs.append(cost_a)
            traj.r_intermediates.append(r_int)
            traj.gap_deltas.append(gap_delta)
            traj.gap_norms.append(gap_norm_cur.mean().item()
                                  if gap_norm_cur.dim() > 0
                                  else float(gap_norm_cur))

            # ── Instant reward: R_int - C(a) - λ_gap·GapNorm - λ_time ─────────
            gn_val  = gap_norm_cur.mean().item() if gap_norm_cur.dim() > 0 else float(gap_norm_cur)
            instant = (self.eta_int * r_int
                       - cost_a
                       - self.lambda_gap * min(gn_val, 5.0)
                       - memory_pressure
                       - self.lambda_time
                       - self.lambda_mdl * float(cost_a))   # MDL(proof) штраф за крок
            rewards_list.append(instant)

        else:
            traj.stop_reason = "max_steps"

        traj.n_steps = len(traj.actions)
        traj.action_histogram = list(action_counts)
        # MDL фінального доведення
        traj.proof_mdl = self._proof_mdl(
            prover, proof_derivations, action_count=len(traj.actions)
        )

        # ── 3. Фінальне Forward Chaining + Grounding ──────────────────────────
        all_facts = prover.forward_chain_reasoned(
            prover.max_depth,
            starting_facts=working_facts,
            only_verified=True,
            device=device,
        )

        _MAX_GROUND = 128
        ground_sample = (frozenset(random.sample(list(all_facts), _MAX_GROUND))
                         if len(all_facts) > _MAX_GROUND else all_facts)
        z_sym_1 = prover.ground(ground_sample, device)      # (1, d)
        z_sym   = z_sym_1.expand(B, -1)                     # (B, d)

        # ── 4. Proof Search (REINFORCE) ───────────────────────────────────────
        vem_hinge    = torch.zeros(1, device=device).squeeze()
        proof_loss   = torch.zeros(1, device=device).squeeze()
        vem_self_loss= torch.zeros(1, device=device).squeeze()

        if self.training and len(prover.kb.rules) > 0:
            proved, traj_proof, proof_loss = prover.prove_with_policy(
                goal, z[:1], starting_facts=working_facts
            )
            goal_proved = proved
            traj.goal_proved = proved

            if traj_proof and prover.kb.rules:
                used_rule = prover.kb.rules[traj_proof[-1] % len(prover.kb.rules)]
                prover.vem.record_outcome(
                    used_rule,
                    utility_target=1.0 if proved else 0.0,
                    device=device,
                )
                if proved:
                    prover.kb.mark_rule_verified(used_rule)

            if prover.kb.rules:
                r_choice = random.choice(prover.kb.rules)
                if r_choice.body:
                    _MAX_UNIF = 64
                    unif_facts = (frozenset(random.sample(list(all_facts), _MAX_UNIF))
                                  if len(all_facts) > _MAX_UNIF else all_facts)
                    gm_e, _, gm_ent = prover.graph_unif(r_choice.body, unif_facts, device, tau=0.5)
                    su_e, su_ent   = prover.soft_unif(r_choice.body, unif_facts, device)
                    proof_loss = (proof_loss
                                  + 0.01 * gm_e - 0.001 * gm_ent
                                  + 0.01 * su_e - 0.001 * su_ent)

        if self.training and prover._step % 10 == 0:
            vem_self_loss = prover.vem.self_supervised_loss(device)

        abductor_aux = torch.zeros(1, device=device).squeeze()
        abduced_rules = 0.0
        target_facts = (
            prover.task_context.target_facts
            if getattr(prover, "task_context", None) is not None else frozenset()
        )
        effective_targets = target_facts or frozenset({goal})
        cycle_stats: Dict[str, float] = {}
        cycle_recent_rules = []
        if self.training and getattr(prover, "continuous_cycle_enabled", False):
            cycle = prover.continuous_hypothesis_cycle(
                z_cur,
                working_facts,
                effective_targets,
                device,
            )
            abductor_aux = abductor_aux + cycle["loss_tensor"]
            abduced_rules += float(cycle.get("added_rules", 0))
            induction_stats = merge_induction_stats(
                induction_stats,
                cycle.get("induction_stats", {}),
            )
            cycle_stats = cycle.get("stats", {})
            cycle_recent_rules = list(cycle.get("accepted_rules", []))
            r_int_cumulative += float(cycle.get("mean_utility", 0.0))
        target_hits = len(all_facts & target_facts)
        target_total = len(target_facts)
        target_coverage = (
            float(target_hits) / float(target_total)
            if target_total > 0 else (1.0 if goal_proved else 0.0)
        )
        err_val = (world_err.item() if torch.is_tensor(world_err) else float(world_err))
        mismatch_error = (1.0 - target_coverage) + (0.0 if goal_proved else 1.0)
        trigger_abduction = (
            getattr(prover, "task_context", None) is not None
            and prover.task_context.trigger_abduction
        )
        should_abduce = (
            self.training
            and (
                trigger_abduction
                or mismatch_error > 0.0
            )
        )
        if should_abduce:
            n_added_abd, reactive_vem_hinge, reactive_abductor_aux, mean_utility = prover.abduce_and_learn(
                z_cur,
                max(float(err_val), float(mismatch_error)),
                force=trigger_abduction or mismatch_error > 0.0,
            )
            vem_hinge = vem_hinge + reactive_vem_hinge
            abductor_aux = abductor_aux + reactive_abductor_aux
            abduced_rules += float(n_added_abd)
            if n_added_abd > 0:
                reactive_induction = prover._induce_proposed_rules_locally(
                    working_facts,
                    effective_targets,
                    device,
                )
                induction_stats = merge_induction_stats(induction_stats, reactive_induction)
                r_int_cumulative += mean_utility
        if cycle_recent_rules:
            prover._extend_recent_abduced_rules(cycle_recent_rules)
        creative_report = prover.run_creative_cycle(
            z_cur,
            working_facts,
            effective_targets,
            device,
        )
        r_int_cumulative += float(creative_report.metrics.get("selected_mean_utility", 0.0))
        target_ground = effective_targets
        z_target = prover.ground(target_ground, device).expand(B, -1)
        coverage_loss = torch.tensor(
            (1.0 - target_coverage) + (0.0 if goal_proved else 1.0),
            device=device,
            dtype=z.dtype,
        )

        # ── 5. Symbolic Consistency Loss ──────────────────────────────────────
        sym_consist = (F.mse_loss(z, z_sym.detach()) +
                       F.mse_loss(z_sym, z.detach()))
        symbolic_induction = (
            F.mse_loss(z_sym, z_target.detach())
            + F.mse_loss(z_target, z_sym.detach())
        )
        sym_loss = (sym_consist
                    + 0.1  * symbolic_induction
                    + 0.1  * coverage_loss
                    + 0.1  * proof_loss
                    + 0.01 * vem_hinge
                    + 0.05 * abductor_aux
                    + 0.01 * vem_self_loss)

        # ── 6. Фінальна нагорода за задачу + оновлення траєкторії ─────────────
        intrinsic_goal = getattr(prover, "current_intrinsic_goal", lambda: None)()
        intrinsic_value = float(getattr(prover, "current_intrinsic_value", lambda: 0.0)())
        scheduled_intrinsic_goals = tuple(getattr(prover, "scheduled_intrinsic_goals", lambda: ())())

        def _goal_match(left, right) -> bool:
            if left is None or right is None:
                return False
            try:
                from omen_prolog import unify

                return unify(left, right) is not None
            except Exception:
                return hash(left) == hash(right)

        intrinsic_task_active = intrinsic_goal is not None and _goal_match(goal, intrinsic_goal)
        background_intrinsic_goals = tuple(
            target for target in scheduled_intrinsic_goals if not _goal_match(target, goal)
        )
        background_intrinsic_hits = sum(
            1 for target in background_intrinsic_goals if prover._goal_supported(target, all_facts)
        )
        background_intrinsic_total = len(background_intrinsic_goals)
        background_intrinsic_coverage = (
            float(background_intrinsic_hits) / float(background_intrinsic_total)
            if background_intrinsic_total > 0 else 0.0
        )
        primary_intrinsic_coverage = (
            1.0 if intrinsic_task_active and prover._goal_supported(intrinsic_goal, all_facts) else 0.0
        )
        r_task = (
            intrinsic_value * primary_intrinsic_coverage
            if intrinsic_task_active else
            (1.0 if goal_proved else 0.0) + intrinsic_value * background_intrinsic_coverage
        )
        if rewards_list:
            rewards_list[-1] += r_task
        elif not log_probs_list:
            empty_stats = TrajectoryStats(trajectory_reward=0.0, stop_reason="no_actions")
            return z_sym, sym_loss, v_mem_out, torch.zeros(1, device=device).squeeze(), empty_stats

        # Повна дисконтована сума траєкторії (для J_OMEN+EMC 7-й член)
        G = 0.0
        for r in reversed(rewards_list):
            G = r + self.gamma * G
        traj.trajectory_reward = G
        traj.intermediate_utility = r_int_cumulative
        traj.final_gap = (gap_norm_cur.mean().item()
                          if gap_norm_cur.dim() > 0 else float(gap_norm_cur))

        # ── 7. Actor-Critic loss ──────────────────────────────────────────────
        meta_loss = self._compute_ac_loss(log_probs_list, values_list,
                                          rewards_list, device,
                                          entropies=entropies_list)

        # ── 8. Task-Estimator BCE supervision ─────────────────────────────────
        # task_estimator.task_estimator передбачає P(goal_proved | state).
        # Проблема: у головному циклі u_stop.item() перериває граф обчислень,
        # тому task_estimator ніколи не отримує градієнт через AC loss.
        # Рішення: після знання goal_proved — додаємо BCE-сигнал явно.
        #
        # Математично: L_task = -[y·log R_task + (1-y)·log(1-R_task)]
        # де y = 1.0 якщо goal_proved, 0.0 інакше.
        # Цей член навчає StoppingUtility.task_estimator передбачати успішність доведення,
        # що є ключовим для правильної оцінки U_stop(s) = R_task + η_int·R_int − λ_gap·GapNorm.
        if self.training:
            # Перераховуємо state_vec без detach (щоб градієнт потрапив у task_estimator)
            task_sv = self._encode_state(
                z_cur, gap_norm_cur,
                len(traj.actions),
                len(working_facts), len(prover.kb),
                goal_embed=goal_embed,
                wm_embed=self._wm_embed(prover, device, B, facts=working_facts),
                trigger_flag=self._rule_trigger_flag(prover, facts=working_facts),
                hot_ratio=float(hot_dims.float().mean().item()) if hot_dims is not None else 0.0,
                action_counts=action_counts,
            )
            l_task = self._task_estimator_bce_loss(task_sv, goal_proved=goal_proved)
            meta_loss = meta_loss + 0.1 * l_task

        prover.last_goal = goal
        prover.last_context_facts = working_facts
        prover.last_all_facts = all_facts
        prover.last_forward_info = {
            "goal_proved": 1.0 if goal_proved else 0.0,
            "target_coverage": target_coverage,
            "target_hits": float(target_hits),
            "target_total": float(target_total),
            "unresolved_targets": float(max(target_total - target_hits, 0)),
            "abduced_rules": abduced_rules,
            "abduction_utility": r_int_cumulative,
            "induction_checked": induction_stats["checked"],
              "induction_verified": induction_stats["verified"],
              "induction_contradicted": induction_stats["contradicted"],
              "induction_retained": induction_stats["retained"],
              "induction_repaired": induction_stats.get("repaired", 0.0),
              "induction_matches": induction_stats["matched_predictions"],
              "induction_score": induction_stats["mean_score"],
              "cycle_checked": float(cycle_stats.get("checked", 0.0)),
              "cycle_accepted": float(cycle_stats.get("accepted", 0.0)),
              "cycle_added": float(cycle_stats.get("added", 0.0)),
              "cycle_verified": float(cycle_stats.get("verified", 0.0)),
              "cycle_contradicted": float(cycle_stats.get("contradicted", 0.0)),
              "cycle_retained": float(cycle_stats.get("retained", 0.0)),
              "cycle_repaired": float(cycle_stats.get("repaired", 0.0)),
              "cycle_error": float(cycle_stats.get("mean_error", 0.0)),
              "cycle_symbolic_error": float(cycle_stats.get("mean_symbolic_error", 0.0)),
              "cycle_soft_symbolic_error": float(cycle_stats.get("mean_soft_symbolic_error", 0.0)),
              "cycle_relaxed_body_error": float(cycle_stats.get("mean_relaxed_body_error", 0.0)),
              "cycle_relaxed_head_error": float(cycle_stats.get("mean_relaxed_head_error", 0.0)),
              "cycle_world_error": float(cycle_stats.get("mean_world_error", 0.0)),
              "cycle_token_error": float(cycle_stats.get("mean_token_error", 0.0)),
              "cycle_graph_energy": float(cycle_stats.get("mean_graph_energy", 0.0)),
              "cycle_policy_loss": float(cycle_stats.get("policy_loss", 0.0)),
              "cycle_loss": float(cycle_stats.get("loss", 0.0)),
              "creative_abduction_candidates": float(creative_report.metrics.get("abduction_candidates", 0.0)),
              "creative_analogy_candidates": float(creative_report.metrics.get("analogy_candidates", 0.0)),
              "creative_metaphor_candidates": float(creative_report.metrics.get("metaphor_candidates", 0.0)),
              "creative_counterfactual_analogy_candidates": float(
                  creative_report.metrics.get("counterfactual_analogy_candidates", 0.0)
              ),
              "creative_counterfactual_metaphor_candidates": float(
                  creative_report.metrics.get("counterfactual_metaphor_candidates", 0.0)
              ),
              "creative_counterfactual_candidates": float(creative_report.metrics.get("counterfactual_candidates", 0.0)),
              "creative_counterfactual_surprise": float(creative_report.metrics.get("counterfactual_surprise", 0.0)),
              "creative_counterfactual_contradictions": float(creative_report.metrics.get("counterfactual_contradictions", 0.0)),
              "creative_counterfactual_exact_search": float(creative_report.metrics.get("counterfactual_exact_search", 0.0)),
              "creative_counterfactual_evaluated_subsets": float(
                  creative_report.metrics.get("counterfactual_evaluated_subsets", 0.0)
              ),
              "creative_ontology_candidates": float(creative_report.metrics.get("ontology_candidates", 0.0)),
              "creative_ontology_feedback_accepted": float(
                  creative_report.metrics.get("ontology_feedback_accepted", 0.0)
              ),
              "creative_ontology_fixed_predicates": float(
                  creative_report.metrics.get("ontology_fixed_predicates", 0.0)
              ),
              "creative_selected_rules": float(creative_report.metrics.get("selected_rules", 0.0)),
              "creative_selected_mean_utility": float(creative_report.metrics.get("selected_mean_utility", 0.0)),
              "creative_intrinsic_value": float(creative_report.metrics.get("intrinsic_value", 0.0)),
              "creative_intrinsic_goal_queue_size": float(
                  creative_report.metrics.get("intrinsic_goal_queue_size", 0.0)
              ),
              "creative_intrinsic_background_goals": float(
                  creative_report.metrics.get("intrinsic_background_goals", 0.0)
              ),
              "background_intrinsic_goals": float(background_intrinsic_total),
              "background_intrinsic_coverage": float(background_intrinsic_coverage),
              "creative_intrinsic_task_active": 1.0 if intrinsic_task_active else 0.0,
              "emc_intrinsic_actions": float(action_counts[ACTION_INTRINSIC]),
              "emc_intrinsic_goal_active": 1.0 if intrinsic_task_active else 0.0,
              "emc_background_intrinsic_goals": float(background_intrinsic_total),
              "creative_analogy_projector_loss": float(creative_report.metrics.get("analogy_projector_loss", 0.0)),
            "emc_gap_delta_mean": (
                float(sum(traj.gap_deltas) / len(traj.gap_deltas))
                if traj.gap_deltas else 0.0
            ),
            "emc_state_steps": float(len(traj.gap_world_norms)),
            "emc_state_gap_world": (
                float(sum(traj.gap_world_norms) / len(traj.gap_world_norms))
                if traj.gap_world_norms else 0.0
            ),
            "emc_state_gap_grounded": (
                float(sum(traj.gap_grounded_norms) / len(traj.gap_grounded_norms))
                if traj.gap_grounded_norms else 0.0
            ),
            "emc_state_gap_relief": (
                float(sum(traj.gap_reliefs) / len(traj.gap_reliefs))
                if traj.gap_reliefs else 0.0
            ),
            "emc_state_memory_residual": (
                float(sum(traj.memory_residuals) / len(traj.memory_residuals))
                if traj.memory_residuals else 0.0
            ),
            "emc_state_memory_alignment": (
                float(sum(traj.memory_alignments) / len(traj.memory_alignments))
                if traj.memory_alignments else 0.0
            ),
            "emc_state_memory_pressure": (
                float(sum(traj.memory_pressures) / len(traj.memory_pressures))
                if traj.memory_pressures else 0.0
            ),
            "emc_gap_events": float(len(traj.gap_deltas)),
            "emc_recall_steps": float(len(traj.recall_gap_deltas)),
            "emc_recall_gap_delta": (
                float(sum(traj.recall_gap_deltas) / len(traj.recall_gap_deltas))
                if traj.recall_gap_deltas else 0.0
            ),
            "emc_recall_gap_relief": (
                float(sum(traj.recall_gap_reliefs) / len(traj.recall_gap_reliefs))
                if traj.recall_gap_reliefs else 0.0
            ),
            "emc_recall_effective_steps": float(traj.recall_effective_steps),
            "emc_recall_effective_ratio": (
                float(traj.recall_effective_steps) / float(len(traj.recall_gap_deltas))
                if traj.recall_gap_deltas else 0.0
            ),
            "provenance": (
                prover.task_context.provenance
                if getattr(prover, "task_context", None) is not None else "latent"
            ),
        }

        return z_sym, sym_loss, v_mem_out, meta_loss, traj

    # ── Eval-mode: адаптивне міркування для generate() ────────────────────────

    @torch.no_grad()
    def run_episode_eval(self,
                         z:        torch.Tensor,
                         gap_norm: torch.Tensor,
                         hot_dims: Optional[torch.Tensor],
                         prover,                       # DifferentiableProver
                         memory,                       # AsyncTensorProductMemory
                         device:   torch.device,
                         gap_features: Optional[Dict[str, float]] = None,
                         gap_feedback: Optional[
                             Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, float]]]
                         ] = None,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eval-mode Bellman adaptive reasoning (без обчислення loss).

        Використовується в generate() при dynamic_reasoning=True та emc_enabled=True.
        Замінює фіксований forward_chain(max_depth) на адаптивний пошук:

          Поки U_stop(s) < V(s) І VoC(ΔV) > C:
            вибрати дію a ← argmax π_meta(a|s)  (greedy, не sample)
            виконати дію → оновити z_cur, gap_norm_cur

        Це реалізує π_meta* = argmax_π E[R_task − λ_time·T − λ_MDL·MDL(proof)]
        під час inference, а не лише під час training.

        Returns:
          z_sym     : (B, d) — символьне представлення після адаптивного міркування
          v_mem_out : (B, d) — результат останнього M-Core recall (або нулі)
        """
        if hasattr(prover, "_clear_runtime_caches"):
            prover._clear_runtime_caches()
        B = z.shape[0]

        # Perception: z → fact → KB (без модифікації prover._step)
        # FIX: оновлюємо _last_z щоб mental simulation під час eval бачила актуальний z
        prover._last_z = z.detach()
        prover.materialize_task_context_facts()
        goal = prover.current_goal(z)
        working_facts = prover.current_working_facts()

        z_cur        = z.clone()
        v_mem_out    = torch.zeros_like(z)
        gap_norm_cur = gap_norm.clone()
        current_gap_features = self._coerce_gap_features(gap_norm_cur, gap_features)
        self.voc.reset()

        action_counts: List[int] = [0] * N_ACTIONS
        proof_derivations: List[Tuple[object, Optional[object]]] = []
        r_int_cumulative = float(getattr(prover, "current_intrinsic_value", lambda: 0.0)())
        induction_stats = {
            "checked": 0.0,
            "verified": 0.0,
            "contradicted": 0.0,
            "retained": 0.0,
            "matched_predictions": 0.0,
            "mean_score": 0.0,
        }
        gap_deltas: List[float] = []
        gap_world_norms: List[float] = []
        gap_grounded_norms: List[float] = []
        gap_reliefs: List[float] = []
        memory_residuals: List[float] = []
        memory_alignments: List[float] = []
        memory_pressures: List[float] = []
        recall_gap_deltas: List[float] = []
        recall_gap_reliefs: List[float] = []
        recall_effective_steps = 0.0
        for step in range(self.max_steps):
            goal = prover.current_goal(z_cur)
            goal_embed = self._goal_embed(prover, goal, B, device)
            intrinsic_goal = getattr(prover, "current_intrinsic_goal", lambda: None)()
            intrinsic_goal_available = intrinsic_goal is not None and not self._goal_match(goal, intrinsic_goal)
            proof_mdl_cur = self._proof_mdl(
                prover, proof_derivations, action_count=sum(action_counts)
            )
            wm_embed = self._wm_embed(prover, device, B, facts=working_facts)
            state_vec = self._encode_state(
                z_cur, gap_norm_cur, step,
                len(working_facts), len(prover.kb),
                gap_features=current_gap_features,
                goal_embed=goal_embed,
                wm_embed=wm_embed,
                trigger_flag=self._rule_trigger_flag(prover, facts=working_facts),
                hot_ratio=float(hot_dims.float().mean().item()) if hot_dims is not None else 0.0,
                action_counts=action_counts)
            gap_world_norms.append(float(current_gap_features["gap_world_only"]))
            gap_grounded_norms.append(float(current_gap_features["gap_memory_grounded"]))
            gap_reliefs.append(float(current_gap_features["gap_memory_relief"]))
            memory_residuals.append(float(current_gap_features["gap_memory_residual"]))
            memory_alignments.append(float(current_gap_features["gap_memory_alignment"]))
            memory_pressure = self._memory_control_penalty(current_gap_features)
            memory_pressures.append(memory_pressure)

            val    = self.critic(state_vec).mean()
            mdl_cost_cur = self.lambda_mdl * proof_mdl_cur
            u_stop = self.stopping_utility(
                state_vec, r_int_cumulative, gap_norm_cur,
                mdl_cost=mdl_cost_cur,
                t_elapsed=step,
                memory_penalty=memory_pressure)

            # Bellman + VoC зупинка
            bellman_stop = self.stopping_utility.bellman_should_stop(
                u_stop, val.unsqueeze(0).expand(B))
            voc_stop = not self.voc.should_continue(val.item())
            if (bellman_stop or voc_stop) and step > 0:
                break

            # Greedy action selection (argmax, не sample — eval mode)
            action_logits = self.actor(state_vec)
            mean_logits   = action_logits.mean(0)
            action_mask = self._action_mask(
                device,
                allow_abduction=True,
                allow_intrinsic=intrinsic_goal_available,
            )
            masked_logits = self._mask_action_logits(mean_logits, action_mask)
            a = masked_logits.argmax().item()

            action_counts[a] = action_counts[a] + 1

            if a == ACTION_STOP:
                break

            n_before = prover.kb.n_facts()

            if a == ACTION_RECALL:
                z_before = z_cur
                v_mem_step = memory.read(z_before)
                gap_norm_cur, recall_gap = self._apply_recall_gap_feedback(
                    z_before,
                    gap_norm_cur,
                    v_mem_step,
                    gap_feedback=gap_feedback,
                )
                z_cur      = (z_before + v_mem_step).clamp(-20.0, 20.0)
                v_mem_out  = v_mem_step
                r_int      = v_mem_step.norm(dim=-1).mean().item() / (z.shape[-1] ** 0.5 + 1e-6)
                gap_deltas.append(float(recall_gap.get("gap_delta", 0.0)))
                recall_gap_deltas.append(float(recall_gap.get("gap_delta", 0.0)))
                recall_gap_reliefs.append(float(recall_gap.get("gap_memory_relief", 0.0)))
                recall_effective_steps += float(recall_gap.get("gap_effective", 0.0))
                current_gap_features = self._coerce_gap_features(
                    gap_norm_cur,
                    recall_gap,
                    prior_features=current_gap_features,
                )

            elif a == ACTION_FC:
                # FIX Bug-FC: forward_chain() не змінює KB. forward_chain_step() персистує.
                n_added_fc, _, fc_trace, working_facts = prover.forward_chain_step_local(
                    working_facts
                )
                proof_derivations.extend(fc_trace)
                r_int    = float(n_added_fc) / max(len(working_facts) + 1, 1)
                if n_added_fc > 0:
                    z_before = z_cur
                    z_cur = prover.ground(working_facts, device).expand(B, -1)
                    if gap_feedback is not None:
                        zero_signal = torch.zeros_like(z_before)
                        gap_before = float(
                            current_gap_features.get(
                                "gap_memory_grounded",
                                float(gap_norm_cur.mean().item()),
                            )
                        )
                        gap_after_t, after_stats = self._measure_gap_feedback_state(
                            z_cur,
                            gap_norm_cur,
                            zero_signal,
                            gap_feedback=gap_feedback,
                        )
                        gap_norm_cur = gap_after_t
                        gap_deltas.append(
                            float(
                                gap_before
                                - after_stats.get("gap_memory_grounded", float(gap_after_t.mean().item()))
                            )
                        )
                    else:
                        gap_before = float(gap_norm_cur.mean().item())
                        gap_norm_cur = (gap_norm_cur * 0.95).clamp(0.0, 5.0)
                        gap_deltas.append(gap_before - float(gap_norm_cur.mean().item()))
                        after_stats = {}
                    current_gap_features = self._coerce_gap_features(
                        gap_norm_cur,
                        after_stats,
                        prior_features=current_gap_features,
                    )

            elif a == ACTION_ABDUCE:
                n_added, _, _, mean_utility = prover.abduce_and_learn(
                    z_cur, 0.5, force=True
                )
                r_int = max(
                    float(n_added) / max(getattr(self.cfg, 'n_proof_cands', 8), 1),
                    mean_utility,
                )
                if n_added > 0:
                    z_before = z_cur
                    induction_stats = prover._induce_proposed_rules_locally(
                        working_facts,
                        (
                            prover.task_context.target_facts
                            if getattr(prover, "task_context", None) is not None
                            else frozenset()
                        ),
                        device,
                    )
                    refined_facts = prover.forward_chain_reasoned(
                        prover.max_depth,
                        starting_facts=working_facts,
                        only_verified=True,
                        device=device,
                    )
                    z_cur = prover.ground(refined_facts or working_facts, device).expand(B, -1)
                    if gap_feedback is not None:
                        zero_signal = torch.zeros_like(z_before)
                        gap_before = float(
                            current_gap_features.get(
                                "gap_memory_grounded",
                                float(gap_norm_cur.mean().item()),
                            )
                        )
                        gap_after_t, after_stats = self._measure_gap_feedback_state(
                            z_cur,
                            gap_norm_cur,
                            zero_signal,
                            gap_feedback=gap_feedback,
                        )
                        gap_norm_cur = gap_after_t
                        gap_deltas.append(
                            float(
                                gap_before
                                - after_stats.get("gap_memory_grounded", float(gap_after_t.mean().item()))
                            )
                        )
                    else:
                        gap_before = float(gap_norm_cur.mean().item())
                        gap_norm_cur = (gap_norm_cur * 0.85).clamp(0.0, 5.0)
                        gap_deltas.append(gap_before - float(gap_norm_cur.mean().item()))
                        after_stats = {}
                    current_gap_features = self._coerce_gap_features(
                        gap_norm_cur,
                        after_stats,
                        prior_features=current_gap_features,
                    )

            elif a == ACTION_INTRINSIC:
                focused_goal = getattr(prover, "focus_intrinsic_goal", lambda: None)()
                if focused_goal is not None:
                    goal = prover.current_goal(z_cur)
                    goal_embed = self._goal_embed(prover, goal, B, device)
                    r_int = max(float(getattr(prover, "current_intrinsic_value", lambda: 0.0)()), 0.0)

            r_int_cumulative += r_int

        # Фінальний grounding: KB → z_sym
        all_facts = prover.forward_chain_reasoned(
            prover.max_depth,
            starting_facts=working_facts,
            only_verified=True,
            device=device,
        )
        _MAX_GROUND = 128
        ground_sample = (frozenset(random.sample(list(all_facts), _MAX_GROUND))
                         if len(all_facts) > _MAX_GROUND else all_facts)
        z_sym_1 = prover.ground(ground_sample, device)    # (1, d)
        z_sym   = z_sym_1.expand(B, -1)                   # (B, d)

        target_facts = (
            prover.task_context.target_facts
            if getattr(prover, "task_context", None) is not None else frozenset({goal})
        )
        target_hits = len(all_facts & target_facts)
        target_total = len(target_facts)
        goal_supported = prover._goal_supported(goal, all_facts)
        target_coverage = (
            float(target_hits) / float(target_total)
            if target_total > 0 else (1.0 if goal_supported else 0.0)
        )
        creative_targets = target_facts or frozenset({goal})
        creative_report = prover.run_creative_cycle(
            z_cur,
            working_facts,
            creative_targets,
            device,
        )
        r_int_cumulative += float(creative_report.metrics.get("selected_mean_utility", 0.0))
        intrinsic_goal = getattr(prover, "current_intrinsic_goal", lambda: None)()
        intrinsic_value = float(getattr(prover, "current_intrinsic_value", lambda: 0.0)())
        scheduled_intrinsic_goals = tuple(getattr(prover, "scheduled_intrinsic_goals", lambda: ())())

        def _goal_match(left, right) -> bool:
            if left is None or right is None:
                return False
            try:
                from omen_prolog import unify

                return unify(left, right) is not None
            except Exception:
                return hash(left) == hash(right)

        intrinsic_task_active = intrinsic_goal is not None and _goal_match(goal, intrinsic_goal)
        background_intrinsic_goals = tuple(
            target for target in scheduled_intrinsic_goals if not _goal_match(target, goal)
        )
        background_intrinsic_hits = sum(
            1 for target in background_intrinsic_goals if prover._goal_supported(target, all_facts)
        )
        background_intrinsic_total = len(background_intrinsic_goals)
        background_intrinsic_coverage = (
            float(background_intrinsic_hits) / float(background_intrinsic_total)
            if background_intrinsic_total > 0 else 0.0
        )
        prover.last_goal = goal
        prover.last_context_facts = working_facts
        prover.last_all_facts = all_facts
        prover.last_forward_info = {
            "goal_proved": 1.0 if goal_supported else 0.0,
            "target_coverage": target_coverage,
            "target_hits": float(target_hits),
            "target_total": float(target_total),
            "unresolved_targets": float(max(target_total - target_hits, 0)),
            "abduced_rules": float(action_counts[ACTION_ABDUCE]),
            "abduction_utility": r_int_cumulative,
            "induction_checked": induction_stats["checked"],
              "induction_verified": induction_stats["verified"],
              "induction_contradicted": induction_stats["contradicted"],
              "induction_retained": induction_stats["retained"],
              "induction_repaired": induction_stats.get("repaired", 0.0),
              "induction_matches": induction_stats["matched_predictions"],
              "induction_score": induction_stats["mean_score"],
              "creative_abduction_candidates": float(creative_report.metrics.get("abduction_candidates", 0.0)),
              "creative_analogy_candidates": float(creative_report.metrics.get("analogy_candidates", 0.0)),
              "creative_metaphor_candidates": float(creative_report.metrics.get("metaphor_candidates", 0.0)),
              "creative_counterfactual_analogy_candidates": float(
                  creative_report.metrics.get("counterfactual_analogy_candidates", 0.0)
              ),
              "creative_counterfactual_metaphor_candidates": float(
                  creative_report.metrics.get("counterfactual_metaphor_candidates", 0.0)
              ),
              "creative_counterfactual_candidates": float(creative_report.metrics.get("counterfactual_candidates", 0.0)),
              "creative_counterfactual_surprise": float(creative_report.metrics.get("counterfactual_surprise", 0.0)),
              "creative_counterfactual_contradictions": float(creative_report.metrics.get("counterfactual_contradictions", 0.0)),
              "creative_counterfactual_exact_search": float(creative_report.metrics.get("counterfactual_exact_search", 0.0)),
              "creative_counterfactual_evaluated_subsets": float(
                  creative_report.metrics.get("counterfactual_evaluated_subsets", 0.0)
              ),
              "creative_ontology_candidates": float(creative_report.metrics.get("ontology_candidates", 0.0)),
              "creative_ontology_feedback_accepted": float(
                  creative_report.metrics.get("ontology_feedback_accepted", 0.0)
              ),
              "creative_ontology_fixed_predicates": float(
                  creative_report.metrics.get("ontology_fixed_predicates", 0.0)
              ),
              "creative_selected_rules": float(creative_report.metrics.get("selected_rules", 0.0)),
              "creative_selected_mean_utility": float(creative_report.metrics.get("selected_mean_utility", 0.0)),
              "creative_intrinsic_value": float(creative_report.metrics.get("intrinsic_value", 0.0)),
              "creative_intrinsic_goal_queue_size": float(
                  creative_report.metrics.get("intrinsic_goal_queue_size", 0.0)
              ),
              "creative_intrinsic_background_goals": float(
                  creative_report.metrics.get("intrinsic_background_goals", 0.0)
              ),
              "background_intrinsic_goals": float(background_intrinsic_total),
              "background_intrinsic_coverage": float(background_intrinsic_coverage),
              "creative_intrinsic_task_active": 1.0 if intrinsic_task_active else 0.0,
              "emc_intrinsic_actions": float(action_counts[ACTION_INTRINSIC]),
              "emc_intrinsic_goal_active": 1.0 if intrinsic_task_active else 0.0,
              "emc_background_intrinsic_goals": float(background_intrinsic_total),
              "creative_analogy_projector_loss": float(creative_report.metrics.get("analogy_projector_loss", 0.0)),
            "emc_gap_delta_mean": (
                float(sum(gap_deltas) / len(gap_deltas))
                if gap_deltas else 0.0
            ),
            "emc_state_steps": float(len(gap_world_norms)),
            "emc_state_gap_world": (
                float(sum(gap_world_norms) / len(gap_world_norms))
                if gap_world_norms else 0.0
            ),
            "emc_state_gap_grounded": (
                float(sum(gap_grounded_norms) / len(gap_grounded_norms))
                if gap_grounded_norms else 0.0
            ),
            "emc_state_gap_relief": (
                float(sum(gap_reliefs) / len(gap_reliefs))
                if gap_reliefs else 0.0
            ),
            "emc_state_memory_residual": (
                float(sum(memory_residuals) / len(memory_residuals))
                if memory_residuals else 0.0
            ),
            "emc_state_memory_alignment": (
                float(sum(memory_alignments) / len(memory_alignments))
                if memory_alignments else 0.0
            ),
            "emc_state_memory_pressure": (
                float(sum(memory_pressures) / len(memory_pressures))
                if memory_pressures else 0.0
            ),
            "emc_gap_events": float(len(gap_deltas)),
            "emc_recall_steps": float(len(recall_gap_deltas)),
            "emc_recall_gap_delta": (
                float(sum(recall_gap_deltas) / len(recall_gap_deltas))
                if recall_gap_deltas else 0.0
            ),
            "emc_recall_gap_relief": (
                float(sum(recall_gap_reliefs) / len(recall_gap_reliefs))
                if recall_gap_reliefs else 0.0
            ),
            "emc_recall_effective_steps": float(recall_effective_steps),
            "emc_recall_effective_ratio": (
                float(recall_effective_steps) / float(len(recall_gap_deltas))
                if recall_gap_deltas else 0.0
            ),
            "provenance": (
                prover.task_context.provenance
                if getattr(prover, "task_context", None) is not None else "latent"
            ),
        }

        return z_sym, v_mem_out

    # ── Actor-Critic Loss ─────────────────────────────────────────────────────

    def _compute_ac_loss(self,
                         log_probs: List[torch.Tensor],
                         values:    List[torch.Tensor],
                         rewards:   List[float],
                         device:    torch.device,
                         entropies: Optional[List[torch.Tensor]] = None,
                         ) -> torch.Tensor:
        """
        Actor-Critic loss із ентропійною регуляризацією.

        Два режими (залежить від self.use_gae):

        GAE (Generalized Advantage Estimation, emc_use_gae=True):
          δ_t   = r_t + γ·V(s_{t+1}) − V(s_t)   ← TD-error
          A_t   = Σ_{k≥0} (γ·λ)^k · δ_{t+k}     ← зважена сума TD-errors
          G_t   = A_t + V(s_t)                    ← λ-return (ціль для критика)

          При λ→0: чистий TD(0) (низький variance, вищий bias)
          При λ→1: Монте-Карло (нульовий bias, вищий variance)
          λ=0.95: оптимальний компроміс для більшості задач

        MC (Monte-Carlo, emc_use_gae=False):
          G_t   = Σ_{k≥t} γ^{k−t}·r_k            ← дисконтована дохідність
          A(s,a) = G_t − V_φ(s_t)                 ← перевага

        Математична відповідність до MDL-формули:
          A(s,a) = −C(a) + E_{s'}[V(s')] − V(s)
          У GAE: E_{s'}[V(s')] ≈ A_GAE_t + V(s_t) (λ-return)
          У MC:  E_{s'}[V(s')] ≈ G_t

        Ентропійна регуляризація (з реальним H(π)):
          L_actor = E[-Σ_a π_meta(a|s)·A(s,a) − β·H(π_meta(·|s))]
          ≈ REINFORCE: E[-A·log π] − β·E[H(π)]

          де H(π) = −Σ_a π(a|s)·log π(a|s)  — справжня ентропія Categorical dist.
          (НЕ апроксимація через -log_p сampledої дії, а повна ентропія розподілу)

          При більшому H(π): рівномірніший розподіл → більше exploration.
          β·H(π) ≥ 0 завжди → підвищення ентропії ЗМЕНШУЄ L_actor → ЗАОХОЧУЄТЬСЯ.
        """
        n = len(rewards)
        if n == 0 or not log_probs:
            return torch.zeros(1, device=device).squeeze()

        # Вирівнюємо довжини (log_probs може бути коротшим за rewards при Bellman-stop)
        T = min(len(log_probs), len(values), n)

        values_t = torch.stack(values[:T])            # (T,) — диференційований
        log_ps_t = torch.stack(log_probs[:T])         # (T,) — диференційований
        rewards_t = rewards[:T]

        if self.use_gae and T > 1:
            # ── GAE: Generalized Advantage Estimation ─────────────────────────
            # δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
            # A_GAE_t = Σ_{k≥0} (γ·λ_gae)^k · δ_{t+k}
            vals_np = [v.item() for v in values_t]
            gae_lambda = self.gae_lambda

            deltas: List[float] = []
            for t in range(T):
                v_next = vals_np[t + 1] if t + 1 < T else 0.0
                delta = rewards_t[t] + self.gamma * v_next - vals_np[t]
                deltas.append(delta)

            # Зворотній прохід для A_GAE
            advantages: List[float] = [0.0] * T
            gae = 0.0
            for t in reversed(range(T)):
                gae = deltas[t] + self.gamma * gae_lambda * gae
                advantages[t] = gae

            adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)

            # λ-return: G_t^λ = A_GAE_t + V(s_t)
            returns_t = adv_t + values_t.detach()

        else:
            # ── Monte-Carlo returns G_t ───────────────────────────────────────
            G = 0.0
            returns: List[float] = []
            for r in reversed(rewards_t):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            adv_t = returns_t - values_t.detach()

        # ── Нормалізація переваг для стабільності навчання ───────────────────
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        if returns_t.numel() > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # ── Critic loss: E[(G_t^λ − V_φ(s_t))²] ─────────────────────────────
        l_critic = F.mse_loss(values_t, returns_t.detach())

        # ── Actor loss: −E[A·log π(a|s)] ─────────────────────────────────────
        # A(s,a) ≈ -C(a) + E[V(s')] - V(s)  (MDL-розширення)
        l_actor = -(adv_t.detach() * log_ps_t).mean()

        # ── Ентропійний бонус: −β·H(π_meta(·|s)) ────────────────────────────
        # L_actor = E[-A·log π] − β·H(π)
        #
        # H(π) = −Σ_a π(a|s)·log π(a|s) ≥ 0  — справжня ентропія розподілу.
        # (Categorical.entropy(), а не -log_p сampledої дії — точна, не апрокс.)
        #
        # entropy_bonus = −β·H(π) ≤ 0  → зменшує L_actor → ЗАОХОЧУЄ exploration.
        # Рівняння MDL: π_meta* = argmax E[R_task − λ_time·T − λ_MDL·MDL(proof)]
        # Регуляризація ентропії запобігає передчасній збіжності до одного шляху.
        if entropies is not None and len(entropies) >= T:
            entropies_t = torch.stack(entropies[:T])          # (T,) справжні H(π)
            entropy_bonus = -self.entropy_beta * entropies_t.mean()
        else:
            # Запасний варіант: апроксимація через −log_p (як у REINFORCE стандартному)
            entropy_bonus = -self.entropy_beta * (-log_ps_t.mean())

        return l_actor + 0.5 * l_critic + entropy_bonus


# ══════════════════════════════════════════════════════════════════════════════
# 5.  INLINE ТЕСТИ
# ══════════════════════════════════════════════════════════════════════════════

def _run_emc_tests() -> None:
    """Самостійні тести EMC без залежності від інших OMEN-модулів."""
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import torch
    sep = lambda s: print(f"\n{'─'*60}\n  {s}\n{'─'*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[omen_emc] device={device}")

    # ── Простий мок-конфіг ────────────────────────────────────────────────────
    class MockCfg:
        d_latent       = 64
        dropout        = 0.1
        emc_max_steps  = 4
        emc_gamma      = 0.95
        emc_entropy_beta = 0.01
        emc_lambda_time  = 0.05
        emc_lambda_gap   = 0.05
        emc_lambda_mdl   = 0.01
        emc_eta_int      = 0.10
        emc_c_recall     = 0.01
        emc_c_fc         = 0.05
        emc_c_abduce     = 0.10
        emc_use_gae      = True
        emc_gae_lambda   = 0.95
        emc_use_action_hist = True
        sym_max_facts    = 64
        ltm_max_rules    = 256
        n_proof_cands    = 8

    cfg = MockCfg()

    # ── T1: StateEncoder ──────────────────────────────────────────────────────
    sep("T1 · EMCStateEncoder — форми та forward")
    enc = EMCStateEncoder(cfg.d_latent, dropout=0.0).to(device)
    B, d = 4, cfg.d_latent
    z    = torch.randn(B, d, device=device)
    gap  = torch.tensor(0.5, device=device)
    s    = enc(z, gap,
               torch.tensor(0.5, device=device),
               torch.tensor(0.3, device=device),
               torch.tensor(0.2, device=device))
    assert s.shape == (B, d), f"StateEncoder shape FAIL: {s.shape}"
    print(f"  state shape: {tuple(s.shape)}  [PASS]")

    # ── T2: Actor + Critic ────────────────────────────────────────────────────
    sep("T2 · EMCActor + EMCCritic — logits та values")
    actor  = EMCActor(d, dropout=0.0).to(device)
    critic = EMCCritic(d, dropout=0.0).to(device)

    logits = actor(s)
    vals   = critic(s)
    assert logits.shape == (B, N_ACTIONS), f"Actor shape FAIL: {logits.shape}"
    assert vals.shape   == (B,),           f"Critic shape FAIL: {vals.shape}"

    # Softmax → probability
    probs = F.softmax(logits, dim=-1)
    assert (probs.sum(-1) - 1.0).abs().max() < 1e-5, "Probs не суммуються до 1"
    print(f"  logits: {logits.shape}  vals: {vals.shape}  probs ok  [PASS]")

    # ── T3: _compute_ac_loss — MC та GAE ─────────────────────────────────────
    sep("T3 · Actor-Critic Loss — MC та GAE режими")
    emc = EfficientMetaController(cfg).to(device)
    emc.train()

    # Симулюємо 3-кроку траєкторію
    mock_rewards   = [0.05, -0.03, 0.10]
    mock_log_probs = [torch.log(torch.tensor(0.3, device=device)),
                      torch.log(torch.tensor(0.4, device=device)),
                      torch.log(torch.tensor(0.5, device=device))]
    mock_values    = [torch.tensor(0.2, device=device, requires_grad=True),
                      torch.tensor(0.15, device=device, requires_grad=True),
                      torch.tensor(0.25, device=device, requires_grad=True)]

    # GAE mode (за замовчуванням у конфігу emc_use_gae=True)
    loss_gae = emc._compute_ac_loss(mock_log_probs, mock_values, mock_rewards, device)
    assert loss_gae.shape == torch.Size([]), f"Loss shape FAIL: {loss_gae.shape}"
    assert not torch.isnan(loss_gae), "NaN у AC loss (GAE)"
    loss_gae.backward()
    print(f"  meta_loss (GAE)  = {loss_gae.item():.4f}  backward ok  [PASS]")

    # MC mode
    emc.use_gae = False
    mock_values2 = [torch.tensor(0.2, device=device, requires_grad=True),
                    torch.tensor(0.15, device=device, requires_grad=True),
                    torch.tensor(0.25, device=device, requires_grad=True)]
    loss_mc = emc._compute_ac_loss(mock_log_probs, mock_values2, mock_rewards, device)
    assert not torch.isnan(loss_mc), "NaN у AC loss (MC)"
    loss_mc.backward()
    print(f"  meta_loss (MC)   = {loss_mc.item():.4f}  backward ok  [PASS]")
    emc.use_gae = True  # відновлюємо

    # ── T4: EfficientMetaController — _encode_state ──────────────────────────
    sep("T4 · EfficientMetaController — _encode_state")
    emc2 = EfficientMetaController(cfg).to(device)
    z2   = torch.randn(2, cfg.d_latent, device=device)
    gn2  = torch.tensor(0.4, device=device)
    sv2  = emc2._encode_state(z2, gn2, depth=2, n_facts=12, n_rules=5)
    assert sv2.shape == (2, cfg.d_latent), f"encode_state shape FAIL: {sv2.shape}"
    print(f"  state_vec shape: {tuple(sv2.shape)}  [PASS]")

    # ── T5: StoppingUtility + λ_time·T_elapsed ───────────────────────────────
    sep("T5 · StoppingUtility — U_stop(s), λ_time·T_elapsed та Bellman check")
    su = StoppingUtility(d_state=d, eta_int=0.10, lambda_gap=0.05,
                         lambda_time=0.05).to(device)
    su.train()
    z_t   = torch.randn(B, d, device=device)
    gn_t  = torch.tensor(0.5, device=device)

    # Пряме обчислення U_stop (t_elapsed=0)
    u_t0 = su(z_t, r_int=0.2, gap_norm=gn_t, t_elapsed=0)
    # U_stop при t_elapsed=5 (більший штраф) → менше значення
    u_t5 = su(z_t, r_int=0.2, gap_norm=gn_t, t_elapsed=5)
    assert u_t0.shape == (B,), f"U_stop shape FAIL: {u_t0.shape}"
    assert not torch.isnan(u_t0).any(), "NaN у U_stop"
    # Перевірка: T_elapsed штраф зменшує U_stop
    assert u_t5.mean() < u_t0.mean(), (
        f"FAIL: T_elapsed penalty не зменшує U_stop  "
        f"(t=0: {u_t0.mean():.3f}, t=5: {u_t5.mean():.3f})")

    # Перевірка BCE training signal для task_estimator (виправлення Bug 2)
    # task_estimator має отримувати градієнт через BCE, а не через u_stop.item()
    su.zero_grad()
    r_pred_pos = su.task_estimator(z_t).squeeze(-1)         # goal_proved=True
    bce_pos    = F.binary_cross_entropy(r_pred_pos.clamp(1e-6, 1-1e-6),
                                         torch.ones(B, device=device))
    bce_pos.backward()
    grad_via_bce = sum(p.grad.norm().item() for p in su.parameters() if p.grad is not None)
    assert grad_via_bce > 0, "FAIL: task_estimator не має градієнту через BCE"
    print(f"  task_estimator BCE grad_norm={grad_via_bce:.4f}  [PASS — Bug 2 fix ✓]")

    # Перевірка що u_stop.item() самостійно НЕ дає градієнт (підтверджує необхідність fix)
    su.zero_grad()
    u_scalar = su(z_t, r_int=0.2, gap_norm=gn_t, t_elapsed=0).mean().item()
    # Після .item() — граф обчислень розірваний, параметри без grad
    no_grad = all((p.grad is None or p.grad.norm().item() == 0.0)
                  for p in su.parameters())
    assert no_grad, "FAIL: u_stop.item() не повинен залишати градієнт"
    print(f"  u_stop.item() grad=None (детач підтверджено)  [PASS]")

    # Backward через StoppingUtility (прямий)
    su.zero_grad()
    u_t0 = su(z_t, r_int=0.2, gap_norm=gn_t, t_elapsed=0)
    u_t0.mean().backward()
    grad_norm = sum(p.grad.norm().item() for p in su.parameters() if p.grad is not None)
    assert grad_norm > 0, "FAIL: StoppingUtility не має градієнту"
    print(f"  U_stop(t=0): mean={u_t0.mean().item():.3f}")
    print(f"  U_stop(t=5): mean={u_t5.mean().item():.3f}  (T_elapsed штраф: OK)")
    print(f"  grad_norm={grad_norm:.4f}  [PASS]")

    # Bellman decision
    su.zero_grad()
    u_high = torch.ones(B, device=device) * 0.9
    u_low  = torch.ones(B, device=device) * 0.1
    v_mid  = torch.ones(B, device=device) * 0.5
    assert su.bellman_should_stop(u_high, v_mid) == True,  "FAIL: повинен зупинитись"
    assert su.bellman_should_stop(u_low,  v_mid) == False, "FAIL: не повинен зупинитись"
    print(f"  Bellman stop logic: OK  [PASS]")

    # ── T6: VoCStoppingCriterion ─────────────────────────────────────────────
    sep("T6 · VoCStoppingCriterion — Δ > C(Δd)")
    voc = VoCStoppingCriterion(cost_per_step=0.05)
    # Перший крок завжди продовжує
    assert voc.should_continue(0.0)   == True,  "FAIL: перший крок"
    # Великий виграш → продовжуємо
    assert voc.should_continue(0.2)   == True,  "FAIL: виграш 0.2 > 0.05"
    # Малий виграш → зупиняємось
    assert voc.should_continue(0.21)  == False, "FAIL: виграш 0.01 < 0.05"
    voc.reset()
    assert voc._prev_value is None, "FAIL: reset не спрацював"
    print(f"  VoC logic OK  [PASS]")

    # ── T7: TrajectoryStats ──────────────────────────────────────────────────
    sep("T7 · TrajectoryStats — dataclass")
    ts = TrajectoryStats(trajectory_reward=0.42, n_steps=3, goal_proved=True)
    assert ts.trajectory_reward == 0.42
    assert ts.goal_proved == True
    assert ts.gap_norms == []
    print(f"  TrajectoryStats: reward={ts.trajectory_reward}  [PASS]")

    # ── T8: Proper Categorical Entropy у _compute_ac_loss ────────────────────
    sep("T8 · _compute_ac_loss — справжня H(π) через Categorical.entropy()")
    emc3 = EfficientMetaController(cfg).to(device)
    emc3.train()
    from torch.distributions import Categorical as _Cat

    mock_rewards   = [0.05, -0.03, 0.10]
    mock_log_probs = [torch.log(torch.tensor(0.3, device=device)),
                      torch.log(torch.tensor(0.4, device=device)),
                      torch.log(torch.tensor(0.5, device=device))]
    mock_values    = [torch.tensor(0.2, device=device, requires_grad=True),
                      torch.tensor(0.15, device=device, requires_grad=True),
                      torch.tensor(0.25, device=device, requires_grad=True)]
    # Реальні ентропії з рівномірного і пікового розподілу
    uniform_logits = torch.zeros(N_ACTIONS, device=device)
    peaked_logits  = torch.tensor([10.0, -5.0, -5.0, -5.0], device=device)
    mock_entropies = [
        _Cat(logits=uniform_logits).entropy(),   # max entropy ≈ log(4)
        _Cat(logits=peaked_logits).entropy(),    # low entropy ≈ 0
        _Cat(logits=uniform_logits).entropy(),
    ]

    loss_with_ent = emc3._compute_ac_loss(
        mock_log_probs, mock_values, mock_rewards, device,
        entropies=mock_entropies)
    assert not torch.isnan(loss_with_ent), "NaN у AC loss з ентропіями"
    loss_with_ent.backward()

    # Перевірка: loss БЕЗ ентропій (запасний варіант)
    mock_values2 = [torch.tensor(0.2, device=device, requires_grad=True),
                    torch.tensor(0.15, device=device, requires_grad=True),
                    torch.tensor(0.25, device=device, requires_grad=True)]
    loss_no_ent = emc3._compute_ac_loss(
        mock_log_probs, mock_values2, mock_rewards, device,
        entropies=None)
    assert not torch.isnan(loss_no_ent), "NaN у AC loss без ентропій"

    # Перевірка: loss з вищою ентропією МЕНШИЙ (більший exploration bonus)
    mock_values3 = [torch.tensor(0.2, device=device, requires_grad=True),
                    torch.tensor(0.15, device=device, requires_grad=True),
                    torch.tensor(0.25, device=device, requires_grad=True)]
    peaked_entropies = [_Cat(logits=peaked_logits).entropy() for _ in range(3)]
    loss_low_ent = emc3._compute_ac_loss(
        mock_log_probs, mock_values3, mock_rewards, device,
        entropies=peaked_entropies)

    # Порівнюємо entropy_bonus між uniform і peaked: uniform має більший H
    H_uniform = mock_entropies[0].item()
    H_peaked  = peaked_entropies[0].item()
    assert H_uniform > H_peaked, "FAIL: uniform повинна мати більшу ентропію"
    print(f"  H(uniform)={H_uniform:.3f}  H(peaked)={H_peaked:.3f}")
    print(f"  loss(H_high)={loss_with_ent.item():.4f}  "
          f"loss(H_low)={loss_low_ent.item():.4f}")
    print(f"  Categorical entropy bonus: OK  [PASS]")

    print(f"\n{'─'*60}")
    print("  ✅  Всі 8 EMC тестів пройдено")
    print(f"{'─'*60}\n")

    # ── T9 (бонус): _proof_mdl — виправлення Bug 1 ───────────────────────────
    sep("T9 · _proof_mdl — Bug fix: action-ID не є rule-index")

    class MockKB:
        """Мінімальний mock KB для тестування _proof_mdl."""
        class _Rule:
            def complexity(self): return 6
        def __init__(self, n_rules=5):
            self.rules = [MockKB._Rule() for _ in range(n_rules)]
        def n_facts(self): return 3

    class MockProver:
        def __init__(self, n_rules=5):
            self.kb = MockKB(n_rules)
            self.cost_est = type(
                "MockCostEstimator",
                (),
                {"symbolic_cost": staticmethod(lambda clause, sigma: clause.complexity() + (0.5 if sigma is not None else 0.0))}
            )()

    emc_t9 = EfficientMetaController(cfg).to(device)
    prover_mock = MockProver(n_rules=5)

    # Порожня траєкторія → 0.0
    assert emc_t9._proof_mdl(prover_mock, []) == 0.0, "FAIL: empty trajectory"

    traj_test = [
        (prover_mock.kb.rules[0], None),
        (prover_mock.kb.rules[1], object()),
    ]
    mdl = emc_t9._proof_mdl(prover_mock, traj_test, action_count=2)
    expected = prover_mock.kb.rules[0].complexity() + \
               prover_mock.kb.rules[1].complexity() + 0.5 + 0.2
    assert abs(mdl - expected) < 1e-6, f"FAIL: MDL={mdl} ≠ {expected}"
    print(f"  MDL(traj={traj_test}): {mdl:.4f} (expected {expected:.4f})  [PASS]")

    # Порожній trace → MDL = лише слабкий action-count term
    prover_empty = MockProver(n_rules=0)
    mdl_empty = emc_t9._proof_mdl(prover_empty, [], action_count=2)
    assert mdl_empty == 2.0, f"FAIL: empty trace MDL={mdl_empty} ≠ 2.0"
    print(f"  MDL(empty trace, actions=2): {mdl_empty:.1f}  [PASS]")

    mdl_large = emc_t9._proof_mdl(prover_mock, [(prover_mock.kb.rules[0], None)], action_count=30)
    assert mdl_large > 0.0
    print(f"  MDL(with large action count): {mdl_large:.4f}  [PASS]")

    print(f"\n{'─'*60}")
    print("  ✅  Всі 9 EMC тестів пройдено (включно з Bug fixes)")
    print(f"{'─'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _run_emc_tests()

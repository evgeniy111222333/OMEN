"""
omen_osf_meta.py — Synthesis Meta-Controller (OSF)
===================================================
OMEN Synthesis Framework: мета-контролер стратегії генерації.

Вибирає стратегію σ ∈ {Fast, Careful, Exploratory} на основі задачі.

Математика (REINFORCE):
  π_meta(σ|s_task) — policy над стратегіями
  L_meta = E_{σ~π_meta}[−R(σ) + β·Cost(σ)]

  R(σ)    — якість результату (якість генерації)
  Cost(σ) — обчислювальна вартість (час, ресурси)
  β       — баланс якість/вартість

Три стратегії:
  Fast        : без планування, прямий decode (σ=0)
    Cost=0.1  → мало часу, нижча якість
  Careful     : 1 Planning + Simulation + 1 Reflection (σ=1)
    Cost=0.5  → середній баланс (за замовчуванням)
  Exploratory : Multi-plan beam + повна рефлексія + 3 спроби (σ=2)
    Cost=1.0  → багато часу, найвища якість

Стан s_task включає:
  - task_difficulty: оцінка складності (з gap_norm та план_depth)
  - quality_estimate: поточна якість (з CE loss)
  - resources_used: скільки вже витрачено

Інтеграція:
  OSFSynthesizer.forward() запитує SynthesisMetaController.select_strategy()
  Стратегія визначає кількість ітерацій рефлексії та глибину планування.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ══════════════════════════════════════════════════════════════════════════════
# 1.  СТРАТЕГІЇ
# ══════════════════════════════════════════════════════════════════════════════

STRATEGY_FAST        = 0
STRATEGY_CAREFUL     = 1
STRATEGY_EXPLORATORY = 2
N_STRATEGIES         = 3

STRATEGY_NAMES = {
    STRATEGY_FAST:        "Fast",
    STRATEGY_CAREFUL:     "Careful",
    STRATEGY_EXPLORATORY: "Exploratory",
}

# Вартість кожної стратегії (REINFORCE Cost term)
STRATEGY_COSTS = torch.tensor([0.1, 0.5, 1.0])


@dataclass
class StrategyConfig:
    """Конфігурація параметрів для обраної стратегії."""
    strategy_id:    int
    plan_depth:     int     # max_depth для SymbolicPlanner
    n_reflections:  int     # кількість ітерацій ReflectionModule
    sim_steps:      int     # кроків симуляції WorldSimulator
    confidence_tau: float   # поріг впевненості для прийняття результату


STRATEGY_CONFIGS: Dict[int, StrategyConfig] = {
    STRATEGY_FAST: StrategyConfig(
        strategy_id=0, plan_depth=1, n_reflections=0,
        sim_steps=2, confidence_tau=0.3),
    STRATEGY_CAREFUL: StrategyConfig(
        strategy_id=1, plan_depth=4, n_reflections=1,
        sim_steps=4, confidence_tau=0.5),
    STRATEGY_EXPLORATORY: StrategyConfig(
        strategy_id=2, plan_depth=8, n_reflections=3,
        sim_steps=8, confidence_tau=0.7),
}


@dataclass
class MetaTrajectory:
    """Статистика траєкторії мета-контролера."""
    strategy_id:    int
    strategy_name:  str
    quality_reward: float
    cost_penalty:   float
    net_reward:     float
    meta_loss:      torch.Tensor


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TASK STATE ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class TaskStateEncoder(nn.Module):
    """
    Кодує стан задачі у векторне представлення для π_meta.

    Стан s_task = (gap_norm, ce_loss, plan_depth, n_rules, n_writes)
    """

    def __init__(self, d_state: int = 32, dropout: float = 0.1):
        super().__init__()
        # Скалярні ознаки → d_state
        self.encoder = nn.Sequential(
            nn.Linear(5, d_state),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_state, d_state),
        )
        self.norm = nn.LayerNorm(d_state)

    def forward(
        self,
        gap_norm:  float,
        ce_loss:   float,
        plan_depth: int,
        n_rules:   int,
        n_writes:  int,
    ) -> torch.Tensor:
        """Returns: (1, d_state)"""
        device = next(self.parameters()).device
        feats  = torch.tensor(
            [gap_norm, ce_loss,
             float(plan_depth) / 10.0,
             float(n_rules) / 100.0,
             float(n_writes) / 100.0],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)  # (1, 5)
        return self.norm(self.encoder(feats))   # (1, d_state)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYNTHESIS META-CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class SynthesisMetaController(nn.Module):
    """
    Мета-контролер стратегії генерації OSF.

    Вибирає σ ∈ {Fast, Careful, Exploratory} через Actor-Critic.

    Actor:  π_meta(σ|s_task) = softmax(W_actor · s_task)
    Critic: V_meta(s_task)   = W_critic · s_task  (baseline для REINFORCE)

    Навчання (REINFORCE):
      A(s, σ) = R(σ) − β·Cost(σ) − V(s)   (advantage)
      L_actor = -E[log π_meta(σ|s) · A(s, σ)] − entropy_beta·H(π_meta)
      L_critic = E[(R(σ) − β·Cost(σ) − V(s))²]
      L_meta   = L_actor + 0.5·L_critic
    """

    def __init__(
        self,
        d_state:      int   = 32,
        beta:         float = 0.1,     # баланс якість/вартість
        entropy_beta: float = 0.01,    # exploration bonus
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.beta         = beta
        self.entropy_beta = entropy_beta

        # Task state encoder
        self.state_enc = TaskStateEncoder(d_state, dropout)

        # Actor π_meta(σ|s)
        self.actor = nn.Linear(d_state, N_STRATEGIES)

        # Critic V_meta(s)
        self.critic = nn.Linear(d_state, 1)

        # Лічильники для статистики
        self._strategy_counts = [0] * N_STRATEGIES
        self._total_reward    = 0.0
        self._n_episodes      = 0

    def select_strategy(
        self,
        gap_norm:  float,
        ce_loss:   float,
        plan_depth: int   = 0,
        n_rules:   int   = 0,
        n_writes:  int   = 0,
    ) -> Tuple[StrategyConfig, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вибирає стратегію σ.

        Returns:
          cfg      : StrategyConfig для обраної стратегії
          log_prob : log π_meta(σ|s) — для REINFORCE backward
          value    : V_meta(s) — baseline
        """
        s = self.state_enc(gap_norm, ce_loss, plan_depth, n_rules, n_writes)  # (1, d)

        logits = self.actor(s)   # (1, N_strategies)
        value  = self.critic(s)  # (1, 1)
        dist   = Categorical(logits=logits.squeeze(0))
        entropy = dist.entropy()

        if self.training:
            sigma_id = dist.sample().item()
            log_prob = dist.log_prob(
                torch.tensor(sigma_id, device=logits.device))
        else:
            sigma_id = logits.squeeze(0).argmax().item()
            log_prob = torch.tensor(0.0, device=logits.device)
            entropy = torch.tensor(0.0, device=logits.device)

        self._strategy_counts[sigma_id] += 1

        cfg = STRATEGY_CONFIGS[sigma_id]
        return cfg, log_prob, value.squeeze(), entropy

    def compute_meta_loss(
        self,
        log_prob:       torch.Tensor,  # log π(σ|s)
        value:          torch.Tensor,  # V(s)
        quality_reward: float,         # R(σ) — якість (1/CE-normalized)
        strategy_id:    int,
    ) -> MetaTrajectory:
        """
        Обчислює L_meta = L_actor + 0.5·L_critic.

        quality_reward: ∈ [0, 1] — нормована якість (вища = краще).
        """
        device = log_prob.device

        cost    = STRATEGY_COSTS[strategy_id].to(device)
        R_net   = quality_reward - self.beta * cost.item()   # R(σ) − β·Cost(σ)

        R_tensor = torch.tensor(R_net, dtype=torch.float32, device=device)

        # Advantage: A(s,σ) = R_net − V(s)
        advantage = (R_tensor - value.detach()).clamp(-5.0, 5.0)

        # Actor loss: −log π(σ|s) · A(s,σ)
        L_actor  = -(log_prob * advantage)

        # Entropy bonus (exploration): -H(π)
        # Обчислюємо через log_prob + Σπ·log π approximation
        L_entropy = self.entropy_beta * log_prob  # soft approximation

        # Critic loss: (R_net − V(s))²
        L_critic = (R_tensor - value).pow(2) * 0.5

        L_meta = (L_actor + L_entropy + L_critic).clamp(-10.0, 10.0)

        # Статистика
        self._total_reward  += R_net
        self._n_episodes    += 1

        return MetaTrajectory(
            strategy_id    = strategy_id,
            strategy_name  = STRATEGY_NAMES[strategy_id],
            quality_reward = quality_reward,
            cost_penalty   = cost.item() * self.beta,
            net_reward     = R_net,
            meta_loss      = L_meta,
        )

    def compute_meta_loss(
        self,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        quality_reward: float,
        strategy_id: int,
        entropy: torch.Tensor | None = None,
    ) -> MetaTrajectory:
        """
        Перевизначена версія з реальною ентропією policy distribution.
        """
        device = log_prob.device
        cost = STRATEGY_COSTS[strategy_id].to(device)
        r_net = quality_reward - self.beta * cost.item()
        r_tensor = torch.tensor(r_net, dtype=torch.float32, device=device)
        advantage = (r_tensor - value.detach()).clamp(-5.0, 5.0)
        l_actor = -(log_prob * advantage)
        if entropy is not None and torch.is_tensor(entropy):
            l_entropy = -self.entropy_beta * entropy
        else:
            l_entropy = torch.zeros(1, device=device).squeeze()
        l_critic = (r_tensor - value).pow(2) * 0.5
        l_meta = (l_actor + l_entropy + l_critic).clamp(-10.0, 10.0)
        self._total_reward += r_net
        self._n_episodes += 1
        return MetaTrajectory(
            strategy_id=strategy_id,
            strategy_name=STRATEGY_NAMES[strategy_id],
            quality_reward=quality_reward,
            cost_penalty=cost.item() * self.beta,
            net_reward=r_net,
            meta_loss=l_meta,
        )

    def strategy_stats(self) -> Dict:
        total = max(sum(self._strategy_counts), 1)
        return {
            f"meta_freq_{STRATEGY_NAMES[i]}": c / total
            for i, c in enumerate(self._strategy_counts)
        } | {
            "meta_avg_reward": self._total_reward / max(self._n_episodes, 1),
        }

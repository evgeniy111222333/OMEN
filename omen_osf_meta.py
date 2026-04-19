"""
omen_osf_meta.py — Synthesis Meta-Controller (OSF)
==================================================
OMEN Synthesis Framework: meta-controller for generation strategy.

Chooses a strategy σ ∈ {Fast, Careful, Exploratory} from task state.

REINFORCE formulation:
  π_meta(σ|s_task) — policy over strategies
  L_meta = E_{σ~π_meta}[−R(σ) + β·Cost(σ)]

  R(σ)    — output quality (generation quality)
  Cost(σ) — computational cost (time, resources)
  β       — quality/cost tradeoff

Three strategies:
  Fast        : no planning, direct decode (σ=0)
    Cost=0.1  -> little time, lower quality
  Careful     : 1 Planning + Simulation + 1 Reflection (σ=1)
    Cost=0.5  -> balanced default
  Exploratory : multi-plan beam + full reflection + 3 attempts (σ=2)
    Cost=1.0  -> high time cost, highest quality

State `s_task` includes:
  - task_difficulty: complexity estimate (from gap_norm and plan_depth)
  - quality_estimate: current quality (from CE loss)
  - resources_used: how much has already been spent

Integration:
  OSFSynthesizer.forward() calls `SynthesisMetaController.select_strategy()`
  The strategy controls reflection count and planning depth.
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
# 1. STRATEGIES
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

# Cost of each strategy (REINFORCE cost term).
STRATEGY_COSTS = torch.tensor([0.1, 0.5, 1.0])


@dataclass
class StrategyConfig:
    """Configuration parameters for a selected strategy."""
    strategy_id:    int
    plan_depth:     int     # max_depth for SymbolicPlanner
    n_reflections:  int     # number of ReflectionModule iterations
    sim_steps:      int     # number of WorldSimulator simulation steps
    confidence_tau: float   # confidence threshold for accepting the result


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
    """Meta-controller trajectory statistics."""
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
    Encode the task state into a vector representation for π_meta.

    State s_task = (gap_norm, ce_loss, plan_depth, n_rules, n_writes)
    """

    def __init__(self, d_state: int = 32, dropout: float = 0.1):
        super().__init__()
        # Scalar features -> d_state.
        self.encoder = nn.Sequential(
            nn.Linear(8, d_state),
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
        goal_entropy: float = 0.0,
        goal_confidence: float = 0.0,
        latent_norm: float = 0.0,
    ) -> torch.Tensor:
        """Returns: (1, d_state)"""
        device = next(self.parameters()).device
        feats  = torch.tensor(
            [gap_norm, ce_loss,
             float(plan_depth) / 10.0,
             float(n_rules) / 100.0,
             float(n_writes) / 100.0,
             goal_entropy,
             goal_confidence,
             latent_norm],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)  # (1, 8)
        return self.norm(self.encoder(feats))   # (1, d_state)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYNTHESIS META-CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class SynthesisMetaController(nn.Module):
    """
    Meta-controller for OSF generation strategy.

    Chooses σ ∈ {Fast, Careful, Exploratory} through Actor-Critic.

    Actor:  π_meta(σ|s_task) = softmax(W_actor · s_task)
    Critic: V_meta(s_task)   = W_critic · s_task as the REINFORCE baseline

    Training (REINFORCE):
      A(s, σ) = R(σ) − β·Cost(σ) − V(s)   (advantage)
      L_actor = -E[log π_meta(σ|s) · A(s, σ)] − entropy_beta·H(π_meta)
      L_critic = E[(R(σ) − β·Cost(σ) − V(s))²]
      L_meta   = L_actor + 0.5·L_critic
    """

    def __init__(
        self,
        d_state:      int   = 32,
        beta:         float = 0.1,     # quality/cost balance
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

        # Counters for statistics.
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
        goal_entropy: float = 0.0,
        goal_confidence: float = 0.0,
        latent_norm: float = 0.0,
    ) -> Tuple[StrategyConfig, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select a strategy σ.

        Returns:
          cfg      : StrategyConfig for the selected strategy
          log_prob : log π_meta(σ|s) for REINFORCE backward
          value    : V_meta(s), the baseline
        """
        s = self.state_enc(
            gap_norm,
            ce_loss,
            plan_depth,
            n_rules,
            n_writes,
            goal_entropy=goal_entropy,
            goal_confidence=goal_confidence,
            latent_norm=latent_norm,
        )  # (1, d)

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
        log_prob: torch.Tensor,
        value: torch.Tensor,
        quality_reward: float,
        strategy_id: int,
        entropy: torch.Tensor | None = None,
    ) -> MetaTrajectory:
        """Version that uses the true entropy of the policy distribution."""
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


def run_tests_osf_meta() -> None:
    torch.manual_seed(0)

    meta = SynthesisMetaController(d_state=24, beta=0.2, entropy_beta=0.01, dropout=0.0)
    meta.eval()

    s_a = meta.state_enc(
        gap_norm=0.4,
        ce_loss=1.0,
        plan_depth=2,
        n_rules=5,
        n_writes=3,
        goal_entropy=0.1,
        goal_confidence=0.9,
        latent_norm=0.2,
    )
    s_b = meta.state_enc(
        gap_norm=0.4,
        ce_loss=1.0,
        plan_depth=2,
        n_rules=5,
        n_writes=3,
        goal_entropy=1.5,
        goal_confidence=0.2,
        latent_norm=1.1,
    )
    assert s_a.shape == (1, 24)
    assert s_b.shape == (1, 24)
    assert not torch.allclose(s_a, s_b), "TaskStateEncoder must react to intent-aware features"

    cfg, log_prob, value, entropy = meta.select_strategy(
        gap_norm=0.7,
        ce_loss=0.8,
        plan_depth=4,
        n_rules=9,
        n_writes=2,
        goal_entropy=0.6,
        goal_confidence=0.7,
        latent_norm=0.5,
    )
    assert cfg.strategy_id in STRATEGY_CONFIGS
    assert log_prob.ndim == 0 and value.ndim == 0 and entropy.ndim == 0

    meta.train()
    cfg_t, log_prob_t, value_t, entropy_t = meta.select_strategy(
        gap_norm=0.5,
        ce_loss=1.2,
        plan_depth=3,
        n_rules=4,
        n_writes=1,
        goal_entropy=0.3,
        goal_confidence=0.8,
        latent_norm=0.4,
    )
    traj = meta.compute_meta_loss(
        log_prob_t,
        value_t,
        quality_reward=0.75,
        strategy_id=cfg_t.strategy_id,
        entropy=entropy_t,
    )
    assert torch.isfinite(traj.meta_loss)
    stats = meta.strategy_stats()
    assert "meta_avg_reward" in stats
    assert any(key.startswith("meta_freq_") for key in stats)
    print("OSF meta tests passed.")


if __name__ == "__main__":
    run_tests_osf_meta()

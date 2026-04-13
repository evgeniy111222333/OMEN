"""
omen_osf_intent.py — Intent Encoder (H1 рівень OSF)
=====================================================
OMEN Synthesis Framework: перший рівень ієрархічної генерації.

Intent Level (H1): перетворює z_final (Perceiver output) на символьну ціль.

Математична модель:
  g = IntentEncoder(z) ∈ R^{d_intent}
  P(H1|context) = Gumbel-Softmax(W·z) над n_goals цілей

  Символьні цілі G = {g₁, ..., g_K} — domain-agnostic оператори:
    "реалізувати функцію X", "згенерувати текст стилю Y", ...

Інтеграція:
  OMENScale.forward() → Perceiver → z_final → IntentEncoder → IntentState
  IntentState → SymbolicPlanner (omen_osf_planner.py)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  СТРУКТУРИ ДАНИХ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentState:
    """
    Вихід IntentEncoder — H1 рівень OSF.

    Fields:
      goal_vector  : (B, d_intent) — зважена сума goal embeddings
      goal_probs   : (B, n_goals)  — soft розподіл цілей
      goal_entropy : (B,)          — ентропія H(π_goal) = невизначеність
      z_intent     : (B, d_intent) — уточненийIntent-вектор (z + goal context)
    """
    goal_vector:  torch.Tensor
    goal_probs:   torch.Tensor
    goal_entropy: torch.Tensor
    z_intent:     torch.Tensor

    def to(self, device) -> "IntentState":
        return IntentState(
            goal_vector  = self.goal_vector.to(device),
            goal_probs   = self.goal_probs.to(device),
            goal_entropy = self.goal_entropy.to(device),
            z_intent     = self.z_intent.to(device),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  INTENT ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class IntentEncoder(nn.Module):
    """
    z_final (B, d_latent) → IntentState

    Архітектура:
      z_final → MLP → goal_logits (B, n_goals)
                    → Gumbel-Softmax → goal_soft (B, n_goals)
                    → @ G           → goal_vector (B, d_intent)
      z_intent = LayerNorm( MLP([z_final; goal_vector]) )

    Gumbel-Softmax (training): диференційована дискретизація.
    Argmax (inference): детерміноване призначення цілі.

    Regularization:
      L_intent = −H(goal_probs) / log(n_goals)
      Штрафує collapse до однієї цілі (підтримує різноманіття).
    """

    def __init__(
        self,
        d_latent:   int,
        d_intent:   int,
        n_goals:    int,
        dropout:    float = 0.1,
        gumbel_tau: float = 1.0,
    ):
        super().__init__()
        self.d_intent   = d_intent
        self.n_goals    = n_goals
        self.gumbel_tau = gumbel_tau

        # z → goal distribution
        self.goal_classifier = nn.Sequential(
            nn.Linear(d_latent, d_intent * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_intent * 2, n_goals),
        )

        # Learnable goal embeddings G ∈ R^{n_goals × d_intent}
        self.goal_embeddings = nn.Embedding(n_goals, d_intent)
        nn.init.normal_(self.goal_embeddings.weight, std=d_intent ** -0.5)

        # Intent refinement: [z; goal_vec] → z_intent
        self.intent_refine = nn.Sequential(
            nn.Linear(d_latent + d_intent, d_intent * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_intent * 2, d_intent),
        )
        self.norm = nn.LayerNorm(d_intent)

        # Проекція z_final → d_intent (для skip-connection)
        self.skip_proj = nn.Linear(d_latent, d_intent, bias=False)

    def forward(self, z_final: torch.Tensor) -> IntentState:
        """
        z_final : (B, d_latent)
        Returns  : IntentState
        """
        # ── Goal distribution ───────────────────────────────────────────────
        goal_logits = self.goal_classifier(z_final)              # (B, n_goals)

        if self.training:
            # Differentiable discrete: Gumbel-Softmax
            goal_soft = F.gumbel_softmax(
                goal_logits, tau=self.gumbel_tau, hard=False)    # (B, n_goals)
        else:
            # Inference: sharpened softmax
            goal_soft = F.softmax(goal_logits / 0.5, dim=-1)

        # Entropy H(π_goal) — міра невизначеності про ціль
        goal_probs   = F.softmax(goal_logits, dim=-1)            # (B, n_goals)
        goal_entropy = -(goal_probs * (goal_probs + 1e-10).log()).sum(-1)  # (B,)

        # ── Goal vector: soft mixture of embeddings ──────────────────────────
        G           = self.goal_embeddings.weight                 # (n_goals, d_intent)
        goal_vector = goal_soft @ G                               # (B, d_intent)

        # ── Refined intent: z + goal context ────────────────────────────────
        z_concat = torch.cat([z_final, goal_vector], dim=-1)     # (B, d_lat + d_intent)
        z_intent = self.intent_refine(z_concat)                   # (B, d_intent)
        # Skip connection: проектований z_final + рафінований intent
        z_intent = self.norm(z_intent + self.skip_proj(z_final))  # (B, d_intent)

        return IntentState(
            goal_vector  = goal_vector,
            goal_probs   = goal_probs,
            goal_entropy = goal_entropy,
            z_intent     = z_intent,
        )

    def intent_loss(self, state: IntentState) -> torch.Tensor:
        """
        L_intent = −mean_entropy / log(n_goals)
        Мінімізуємо: заохочуємо РІЗНОМАНІТНІСТЬ цілей (anti-collapse).
        """
        mean_H   = state.goal_entropy.mean()
        max_H    = math.log(self.n_goals + 1e-10)
        # Повертаємо негативну нормовану ентропію (менше = гірше)
        return -(mean_H / max_H).clamp(-1.0, 1.0)
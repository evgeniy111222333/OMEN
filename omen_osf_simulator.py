"""
omen_osf_simulator.py — World Simulator + Reflection Module (OSF)
==================================================================
OMEN Synthesis Framework: виконання в уяві + самовиправлення.

WorldSimulator:
  Виконує «код» (як послідовність latent-дій) через WorldRNN.
  Порівнює отриману трасу з очікуваною (з плану або прикладів).

  L_sim = E[Σ_t ||State_real(t) − Simulate(code, State(t−1))||²]

ReflectionModule:
  Якщо симуляція виявляє невідповідність → локалізує помилку
  і генерує мінімальне виправлення.

  Δ* = argmin_Δ [ Size(Δ)  s.t. Verify(Apply(code, Δ)) = ∅ ]

  Пошук Δ спрямовується нейромережею → soft minimum description length.

Інтеграція:
  OSFSynthesizer.forward() викликає WorldSimulator і (якщо L_sim > τ) ReflectionModule.
  Результати входять до J_OSF як L_sim і L_refl.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_osf_planner import PlanSequence


# ══════════════════════════════════════════════════════════════════════════════
# 1.  СТРУКТУРИ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    """Результат WorldSimulator."""
    trace:          torch.Tensor   # (B, T_sim, d_latent) — симульована траса
    expected_trace: torch.Tensor   # (B, T_sim, d_latent) — очікувана траса
    mismatch_mask:  torch.Tensor   # (B, T_sim) bool — де є невідповідність
    l_sim:          torch.Tensor   # scalar


@dataclass
class PatchResult:
    """Результат ReflectionModule."""
    patch_emb:  torch.Tensor   # (B, d_latent) — latent представлення патчу
    patch_score: torch.Tensor  # (B,) — ймовірність успіху патчу
    l_refl:     torch.Tensor   # scalar — Cost(Δ)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  WORLD SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class WorldSimulator(nn.Module):
    """
    Симулює виконання «коду» (latent action sequence) через WorldRNN.

    Очікувана траса будується з PlanSequence:
      кожен PlanOperator → action_id → WorldRNN крок

    Симульована траса:
      z0, a₁, a₂, ..., a_T → (z₁_sim, z₂_sim, ..., z_T_sim)

    L_sim = (1/T) Σ_t ||z_t_sim − z_t_expected||² (Huber loss для стабільності)

    mismatch_tau: поріг для визначення «помилки» у трасі.
    """

    def __init__(
        self,
        d_latent:      int,
        n_action_vocab: int   = 256,
        mismatch_tau:  float = 0.5,
    ):
        super().__init__()
        self.d_latent      = d_latent
        self.n_action_vocab = n_action_vocab
        self.mismatch_tau  = mismatch_tau

        # Проекція план-embedding → action_id (для WorldRNN)
        # plan_emb (d_plan) → action embedding space
        self.plan_to_action = nn.Linear(d_latent, n_action_vocab, bias=False)

        # Очікувана траса: з plan → target sequence у latent space
        self.plan_to_target = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )
        self.target_norm = nn.LayerNorm(d_latent)

    def forward(
        self,
        z0:          torch.Tensor,    # (B, d_latent) — початковий стан
        plan:        PlanSequence,    # план (K операторів)
        world_rnn:   nn.Module,       # WorldRNN з omen_v2.py
    ) -> SimResult:
        """
        z0       : (B, d_latent)
        plan     : PlanSequence з K операторами
        world_rnn: навчений симулятор світу (вже є в OMENScale)

        Returns: SimResult
        """
        device   = z0.device
        B        = z0.size(0)
        plan_emb = plan.embeddings.to(device)                  # (K, d_plan)
        K        = plan_emb.size(0)

        # ── Проекція plan → d_latent (якщо d_plan ≠ d_latent) ────────────────
        # Якщо розмірності різні — лінійна проекція. Якщо однакові — skip.
        if plan_emb.size(-1) != self.d_latent:
            # Паддинг або проекція
            pad = self.d_latent - plan_emb.size(-1)
            if pad > 0:
                plan_lat = F.pad(plan_emb, (0, pad))          # (K, d_latent)
            else:
                plan_lat = plan_emb[:, :self.d_latent]
        else:
            plan_lat = plan_emb

        # ── Очікувана траса: P_target_k = LayerNorm(MLP(plan_lat_k)) ─────────
        plan_lat_flat  = plan_lat.view(K, -1)                 # (K, d_latent)
        targets_flat   = self.plan_to_target(plan_lat_flat)   # (K, d_latent)
        targets_flat   = self.target_norm(targets_flat)

        # Розширюємо на batch
        expected = targets_flat.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, K, d_lat)

        # ── Симульована траса через WorldRNN ──────────────────────────────────
        # Перетворюємо plan_emb у action_ids через argmax logits.
        # CRITICAL FIX: n_action_vocab (WorldSimulator) може відрізнятись від
        # world_rnn.act_emb.num_embeddings (WorldRNN ініціалізований з vocab_size
        # основної моделі). Без clamp → IndexError при action_id ≥ act_emb.num_embeddings.
        # Рішення: clamp через % до фактичного розміру act_emb.
        action_logits    = self.plan_to_action(plan_lat)            # (K, n_action_vocab)
        act_vocab_actual = world_rnn.act_emb.num_embeddings         # справжній розмір
        action_ids       = action_logits.argmax(-1) % act_vocab_actual  # (K,) safe
        action_seq       = action_ids.unsqueeze(0).expand(B, -1)    # (B, K)

        # simulate_sequence(z0, actions) → (B, K, d_latent)
        simulated = world_rnn.simulate_sequence(z0.detach(), action_seq.detach())

        # ── L_sim: Huber loss між traced та expected ──────────────────────────
        l_sim = F.huber_loss(simulated, expected.detach(), delta=1.0)

        # ── Mismatch mask: де похибка > mismatch_tau ─────────────────────────
        diff          = (simulated - expected.detach()).pow(2).sum(-1)  # (B, K)
        mismatch_mask = diff > (self.mismatch_tau ** 2)                # (B, K) bool

        return SimResult(
            trace          = simulated,
            expected_trace = expected,
            mismatch_mask  = mismatch_mask,
            l_sim          = l_sim,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  REFLECTION MODULE
# ══════════════════════════════════════════════════════════════════════════════

class ReflectionModule(nn.Module):
    """
    Виявляє та виправляє невідповідності у згенерованому виводі.

    Задача мінімальної правки:
      Δ* = argmin_Δ [ Size(Δ)  s.t. Verify(Apply(code, Δ)) = ∅ ]

    Реалізація:
      1. Ідентифікуємо hot spots з mismatch_mask
      2. PatchNet генерує патч Δ у latent space
      3. PatchScore оцінює ймовірність успіху патчу
      4. L_refl = MDL(Δ) = ||patch_emb||₁ (мінімальне розширення)
         плюс штраф якщо patch_score < verify_tau

    Verify наближається через: patch_score > verify_tau
    """

    def __init__(
        self,
        d_latent:    int,
        verify_tau:  float = 0.5,
        lambda_mdl:  float = 0.01,
    ):
        super().__init__()
        self.verify_tau  = verify_tau
        self.lambda_mdl  = lambda_mdl

        # PatchNet: виявляє патч Δ з hot spots
        self.patch_net = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )

        # PatchScorer: оцінює ймовірність успіху (0=fail, 1=success)
        self.patch_scorer = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, 1),
            nn.Sigmoid(),
        )

        # Verify: чи всі mismatch виправлені
        self.verify_net = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_latent)

    def forward(
        self,
        z:          torch.Tensor,    # (B, d_latent) — поточний latent
        sim_result: SimResult,
    ) -> PatchResult:
        """
        z          : (B, d_latent)
        sim_result : SimResult від WorldSimulator

        Returns: PatchResult
        """
        device = z.device
        B      = z.size(0)

        # ── Виявляємо гарячі позиції ─────────────────────────────────────────
        # Зважуємо симульовану трасу за mismatch_mask
        mask_f  = sim_result.mismatch_mask.float()              # (B, K)
        trace   = sim_result.trace                               # (B, K, d_lat)

        # Aggregated mismatch signal
        if mask_f.sum() > 0:
            weights    = mask_f / (mask_f.sum(dim=1, keepdim=True) + 1e-8)
            hot_signal = (trace * weights.unsqueeze(-1)).sum(1)  # (B, d_lat)
        else:
            hot_signal = trace.mean(1)                           # (B, d_lat)

        # ── PatchNet: Δ = f(z, hot_signal) ───────────────────────────────────
        patch_in  = torch.cat([z, hot_signal], dim=-1)          # (B, 2·d_lat)
        patch_emb = self.patch_net(patch_in)                     # (B, d_lat)
        patch_emb = self.norm(patch_emb)

        # ── PatchScorer: P(success|Δ) ────────────────────────────────────────
        score_in    = torch.cat([patch_emb, z], dim=-1)         # (B, 2·d_lat)
        patch_score = self.patch_scorer(score_in).squeeze(-1)   # (B,)

        # ── L_refl = MDL(Δ) + штраф за низький score ─────────────────────────
        #
        # MDL(Δ): наближаємо через L1-норму patch (менший патч = простіша правка)
        mdl_patch = patch_emb.abs().mean()                       # scalar

        # Верифікація: якщо patch_score < verify_tau → штраф
        verify_penalty = F.relu(
            torch.tensor(self.verify_tau, device=device) - patch_score
        ).mean()

        # L_refl = λ_mdl·MDL(Δ) + verify_penalty
        l_refl = self.lambda_mdl * mdl_patch + verify_penalty

        return PatchResult(
            patch_emb   = patch_emb,
            patch_score = patch_score,
            l_refl      = l_refl,
        )

    def apply_patch(
        self,
        z:          torch.Tensor,   # (B, d_latent)
        patch:      PatchResult,
    ) -> torch.Tensor:
        """
        Застосовує патч до latent стану.
        Тільки для позицій з patch_score > verify_tau.
        Повертає покращений z.
        """
        accept = (patch.patch_score > self.verify_tau).float()  # (B,)
        z_patched = z + accept.unsqueeze(-1) * patch.patch_emb
        return z_patched
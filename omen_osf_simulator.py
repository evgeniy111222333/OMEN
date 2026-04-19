"""
omen_osf_simulator.py — World Simulator + Reflection Module (OSF)
=================================================================
OMEN Synthesis Framework: imagined execution plus self-repair.

WorldSimulator:
  Executes "code" (as a sequence of latent actions) through WorldRNN.
  Compares the resulting trace with the expected one (from a plan or examples).

  L_sim = E[Σ_t ||State_real(t) − Simulate(code, State(t−1))||²]

ReflectionModule:
  If simulation detects a mismatch, it localizes the error
  and generates a minimal fix.

  Δ* = argmin_Δ [ Size(Δ)  s.t. Verify(Apply(code, Δ)) = ∅ ]

  Search for Δ is guided by a neural network toward a soft minimum description length.

Integration:
  OSFSynthesizer.forward() calls WorldSimulator and, if `L_sim > τ`, ReflectionModule.
  Their outputs contribute to J_OSF as L_sim and L_refl.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_osf_planner import PlanFact, PlanSequence


# ══════════════════════════════════════════════════════════════════════════════
# 1. STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    """Result produced by WorldSimulator."""
    trace:          torch.Tensor   # (B, T_sim, d_latent) — simulated trace
    expected_trace: torch.Tensor   # (B, T_sim, d_latent) — expected trace
    mismatch_mask:  torch.Tensor   # (B, T_sim) bool — locations with mismatch
    l_sim:          torch.Tensor   # scalar


@dataclass
class PatchResult:
    """Result produced by ReflectionModule."""
    patch_emb:  torch.Tensor   # (B, d_latent) — latent patch representation
    patch_score: torch.Tensor  # (B,) — probability of patch success
    l_refl:     torch.Tensor   # scalar — Cost(Δ)


@dataclass
class SymbolicVerifyResult:
    """Result of symbolic plan verification."""
    mismatch_mask: torch.Tensor   # (B, T_plan) bool
    goal_progress: torch.Tensor   # (B,)
    progress_trace: torch.Tensor  # (B, T_plan)
    l_verify:      torch.Tensor   # scalar


# ══════════════════════════════════════════════════════════════════════════════
# 2.  WORLD SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class WorldSimulator(nn.Module):
    """
    Simulate execution of "code" (a latent action sequence) through WorldRNN.

    The expected trace is built from PlanSequence:
      each PlanOperator -> action_id -> one WorldRNN step

    Simulated trace:
      z0, a₁, a₂, ..., a_T -> (z₁_sim, z₂_sim, ..., z_T_sim)

    L_sim = (1/T) Σ_t ||z_t_sim − z_t_expected||², using Huber loss for stability.

    `mismatch_tau` is the threshold used to define a trace error.
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

        # Project plan embeddings -> action_id for WorldRNN.
        # plan_emb (d_plan) → action embedding space
        self.plan_to_action = nn.Linear(d_latent, n_action_vocab, bias=False)

        # Expected trace: map the plan into a latent-space target sequence.
        self.plan_to_target = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )
        self.target_norm = nn.LayerNorm(d_latent)

    def forward(
        self,
        z0:          torch.Tensor,    # (B, d_latent) — initial state
        plan:        PlanSequence,    # plan with K operators
        world_rnn:   nn.Module,       # WorldRNN from omen_v2.py
    ) -> SimResult:
        """
        z0       : (B, d_latent)
        plan     : PlanSequence with K operators
        world_rnn: trained world simulator already present in OMENScale

        Returns: SimResult
        """
        device   = z0.device
        B        = z0.size(0)
        plan_emb = plan.embeddings.to(device)                  # (K, d_plan)
        K        = plan_emb.size(0)

        # ── Project plan -> d_latent when d_plan ≠ d_latent ─────────────────
        # If dimensions differ, use a linear projection; otherwise skip it.
        if plan_emb.size(-1) != self.d_latent:
            # Padding or projection.
            pad = self.d_latent - plan_emb.size(-1)
            if pad > 0:
                plan_lat = F.pad(plan_emb, (0, pad))          # (K, d_latent)
            else:
                plan_lat = plan_emb[:, :self.d_latent]
        else:
            plan_lat = plan_emb

        # ── Expected trace: P_target_k = LayerNorm(MLP(plan_lat_k)) ──────────
        plan_lat_flat  = plan_lat.view(K, -1)                 # (K, d_latent)
        targets_flat   = self.plan_to_target(plan_lat_flat)   # (K, d_latent)
        targets_flat   = self.target_norm(targets_flat)

        # Broadcast across the batch.
        expected = targets_flat.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, K, d_lat)

        # ── Simulated trace through WorldRNN ─────────────────────────────────
        # Convert plan_emb to action_ids through argmax logits.
        # CRITICAL FIX: n_action_vocab (WorldSimulator) may differ from
        # world_rnn.act_emb.num_embeddings (WorldRNN initialized from the base
        # model vocabulary size). Without clamping, action_id ≥ act_emb.num_embeddings
        # would raise IndexError. The fix uses modulo against the real act_emb size.
        action_logits    = self.plan_to_action(plan_lat)            # (K, n_action_vocab)
        act_vocab_actual = world_rnn.act_emb.num_embeddings         # actual size
        action_ids       = action_logits.argmax(-1) % act_vocab_actual  # (K,) safe
        action_seq       = action_ids.unsqueeze(0).expand(B, -1)    # (B, K)

        # simulate_sequence(z0, actions) → (B, K, d_latent)
        simulated = world_rnn.simulate_sequence(z0, action_seq)

        # ── L_sim: Huber loss between traced and expected ────────────────────
        l_sim = F.huber_loss(simulated, expected.detach(), delta=1.0)

        # ── Mismatch mask: where error > mismatch_tau ────────────────────────
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
    Detect and repair mismatches in generated output.

    Minimal-edit objective:
      Δ* = argmin_Δ [ Size(Δ)  s.t. Verify(Apply(code, Δ)) = ∅ ]

    Implementation:
      1. Identify hot spots from mismatch_mask
      2. PatchNet generates patch Δ in latent space
      3. PatchScore estimates the probability of patch success
      4. L_refl = MDL(Δ) = ||patch_emb||₁ as the minimal extension
         plus a penalty when patch_score < verify_tau

    Verification is approximated by: patch_score > verify_tau
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

        # PatchNet detects patch Δ from hot spots.
        self.patch_net = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )

        # PatchScorer estimates success probability (0=fail, 1=success).
        self.patch_scorer = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, 1),
            nn.Sigmoid(),
        )

        # Verify: whether all mismatches are repaired.
        self.verify_net = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_latent)

    def forward(
        self,
        z:          torch.Tensor,    # (B, d_latent) — current latent state
        sim_result: SimResult,
    ) -> PatchResult:
        """
        z          : (B, d_latent)
        sim_result : SimResult from WorldSimulator

        Returns: PatchResult
        """
        device = z.device
        B      = z.size(0)

        # ── Detect hot positions ─────────────────────────────────────────────
        # Weight the simulated trace by mismatch_mask.
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

        # ── L_refl = MDL(Δ) + penalty for low score ──────────────────────────
        #
        # Approximate MDL(Δ) with patch L1 norm (smaller patch = simpler fix).
        mdl_patch = patch_emb.abs().mean()                       # scalar

        # Verification: if patch_score < verify_tau, apply a penalty.
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
        Apply a patch to the latent state.
        Only positions with patch_score > verify_tau are updated.
        Returns the improved z.
        """
        accept = (patch.patch_score > self.verify_tau).float()  # (B,)
        z_patched = z + accept.unsqueeze(-1) * patch.patch_emb
        return z_patched


class SymbolicPlanVerifier(nn.Module):
    """
    Lightweight symbolic verifier for PlanSequence.

    Checks whether each operator is applicable to the current WM and measures
    real progress toward goal_facts without involving WorldRNN.
    """

    def __init__(self, invalid_penalty: float = 1.0):
        super().__init__()
        self.invalid_penalty = invalid_penalty

    @staticmethod
    def _goal_progress(wm: Set[PlanFact], goal_facts: Tuple[PlanFact, ...]) -> float:
        if not goal_facts:
            return 0.0
        hit = sum(1 for fact in goal_facts if fact in wm)
        return hit / float(len(goal_facts))

    @staticmethod
    def _initial_wm(plan: PlanSequence) -> Set[PlanFact]:
        goal_id = plan.goal_facts[0].arg if plan.goal_facts else 0
        return {PlanFact(300, goal_id)}

    def _verify_single(
        self,
        plan: PlanSequence,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = max(len(plan.operators), 1)
        mismatch = torch.zeros(steps, dtype=torch.bool, device=device)
        progress = torch.zeros(steps, dtype=torch.float32, device=device)
        wm = self._initial_wm(plan)

        if not plan.operators:
            progress[0] = self._goal_progress(wm, plan.goal_facts)
            return mismatch, progress, progress[-1]

        for idx, op in enumerate(plan.operators[:steps]):
            applicable = op.applicable(wm)
            mismatch[idx] = not applicable
            if applicable:
                wm = op.apply(wm)
            progress[idx] = self._goal_progress(wm, plan.goal_facts)

        if len(plan.operators) < steps:
            progress[len(plan.operators):] = progress[len(plan.operators) - 1]

        return mismatch, progress, progress[-1]

    def forward(
        self,
        plan: PlanSequence,
        batch_size: int,
        device: torch.device,
    ) -> SymbolicVerifyResult:
        mismatch_1, progress_1, goal_progress_1 = self._verify_single(plan, device)
        mismatch_mask = mismatch_1.unsqueeze(0).expand(batch_size, -1).contiguous()
        progress_trace = progress_1.unsqueeze(0).expand(batch_size, -1).contiguous()
        goal_progress = goal_progress_1.expand(batch_size)
        invalid_rate = mismatch_mask.float().mean()
        goal_gap = 1.0 - goal_progress.mean()
        l_verify = self.invalid_penalty * invalid_rate + goal_gap
        return SymbolicVerifyResult(
            mismatch_mask=mismatch_mask,
            goal_progress=goal_progress,
            progress_trace=progress_trace,
            l_verify=l_verify,
        )

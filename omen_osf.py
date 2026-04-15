"""
omen_osf.py — OMEN Synthesis Framework: повний синтезатор
=========================================================
Інтегрує всі компоненти OSF та замінює TokenDecoder в OMENScale.

Ієрархічна генерація (4 рівні):
  H1: IntentEncoder     z_final → symbolic goal
  H2: SymbolicPlanner   goal    → operator sequence
  H3: TemplateGenerator ops     → expression templates
  H4: HierarchicalDecoder templ → token logits

Цикл рефлексії:
  WorldSimulator  → trace ≈ plan
  ReflectionModule → мінімальний патч Δ*

Мета-контролер:
  SynthesisMetaController → σ ∈ {Fast, Careful, Exploratory}

Повний функціонал J_OSF:
  J_OSF = E[−logP(code*|spec)]                 ← L_ce (accuracy)
        + λ_plan · E[L_plan(plan, code)]        ← план-код консистентність
        + λ_sim  · E[||Sim(code)−ExpTrace||²]  ← точність симуляції
        + λ_refl · E[min_Δ Cost(Δ)|Verify]     ← здатність до самовиправлення
        + λ_meta · E[−R(σ) + β·Cost(σ)]        ← оптимізація стратегії

Інтеграція з OMENScale:
  OMENScale.osf_synthesizer = OSFSynthesizer(cfg)
  logits, osf_info = model.osf_synthesizer(h_tok, z_final, tgt, plan_data)
  total_loss += cfg.lambda_osf * osf_info["j_osf"]
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_osf_intent    import IntentEncoder, IntentState
from omen_osf_planner   import SymbolicPlanner, PlanSequence
from omen_osf_decoder   import HierarchicalDecoder
from omen_osf_simulator import SymbolicPlanVerifier, WorldSimulator, ReflectionModule
from omen_osf_meta      import (
    SynthesisMetaController, StrategyConfig,
    STRATEGY_CONFIGS, STRATEGY_FAST, STRATEGY_CAREFUL, STRATEGY_EXPLORATORY,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OSFConfig:
    """
    Конфігурація OMEN Synthesis Framework.
    Зазвичай вбудовується в OMENScaleConfig.
    """
    # Увімкнення
    osf_enabled:    bool  = True

    # Архітектура
    d_intent:       int   = 64     # розмірність Intent-простору
    n_goals:        int   = 32     # кількість абстрактних цілей
    d_plan:         int   = 64     # розмірність Plan-простору
    n_operators:    int   = 32     # розмір бібліотеки план-операторів
    template_len:   int   = 8      # довжина шаблону одного оператора

    # Планування
    max_plan_depth: int   = 4      # глибина плану
    beam_width:     int   = 2      # ширина beam search
    alpha_plan:     float = 0.1    # штраф за довжину плану

    # Симуляція/рефлексія
    mismatch_tau:   float = 0.5    # поріг невідповідності
    verify_tau:     float = 0.5    # поріг верифікації патчу
    lambda_mdl_refl: float = 0.01  # MDL для патчу

    # Мета-контролер
    meta_beta:      float = 0.1    # баланс якість/вартість σ
    meta_entropy_beta: float = 0.01  # exploration bonus

    # Ваги J_OSF
    lambda_plan:    float = 0.1    # план-код консистентність
    lambda_sim:     float = 0.05   # симуляція
    lambda_refl:    float = 0.05   # рефлексія
    lambda_meta:    float = 0.05   # мета-стратегія
    lambda_intent:  float = 0.01   # Intent anti-collapse

    # Увімкнення підкомпонентів
    use_simulation:  bool  = True
    use_reflection:  bool  = True
    use_meta:        bool  = True

    # Gumbel tau
    gumbel_tau:     float = 1.0

    dropout:        float = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# 2.  OSF SYNTHESIZER
# ══════════════════════════════════════════════════════════════════════════════

class OSFSynthesizer(nn.Module):
    """
    Повний OMEN Synthesis Framework.

    Замінює TokenDecoder в OMENScale коли osf_enabled=True.
    Повертає ті самі logits (B, T, vocab_size) + словник OSF losів.

    Виклик з OMENScale.forward():
      logits, osf_out = self.osf_synthesizer(
          h_tok    = h_tok,          # (B, T, d_tok) — TokenEncoder output
          z_final  = z_final,        # (B, d_latent) — Perceiver output
          tgt      = tgt,            # (B, T) — target tokens
          world_rnn = self.world_rnn,
          gap_norm  = gap_norm_mean,  # float
          ce_loss   = current_ce,     # float
          n_rules   = n_rules,        # int
          n_writes  = n_writes,       # int
      )
      j_osf = osf_out["j_osf"]
    """

    def __init__(
        self,
        cfg:        OSFConfig,
        d_latent:   int,
        d_tok:      int,
        vocab_size: int,
        n_heads:    int = 4,
    ):
        super().__init__()
        self.cfg        = cfg
        self.d_latent   = d_latent
        self.d_tok      = d_tok
        self.vocab_size = vocab_size

        # ── H1: Intent Encoder ───────────────────────────────────────────────
        self.intent_encoder = IntentEncoder(
            d_latent   = d_latent,
            d_intent   = cfg.d_intent,
            n_goals    = cfg.n_goals,
            dropout    = cfg.dropout,
            gumbel_tau = cfg.gumbel_tau,
        )

        # ── H2: Symbolic Planner ─────────────────────────────────────────────
        self.planner = SymbolicPlanner(
            d_intent    = cfg.d_intent,
            d_plan      = cfg.d_plan,
            n_operators = cfg.n_operators,
            max_depth   = cfg.max_plan_depth,
            beam_width  = cfg.beam_width,
            alpha_plan  = cfg.alpha_plan,
            dropout     = cfg.dropout,
        )

        # ── H3+H4: Hierarchical Decoder ──────────────────────────────────────
        self.hier_decoder = HierarchicalDecoder(
            d_tok         = d_tok,
            d_latent      = d_latent,
            d_plan        = cfg.d_plan,
            d_intent      = cfg.d_intent,
            vocab_size    = vocab_size,
            n_heads       = n_heads,
            template_len  = cfg.template_len,
            dropout       = cfg.dropout,
            lambda_struct = 0.1,
        )

        # ── Симуляція та рефлексія ────────────────────────────────────────────
        self.symbolic_verifier = SymbolicPlanVerifier()
        if cfg.use_simulation:
            self.simulator = WorldSimulator(
                d_latent        = d_latent,
                n_action_vocab  = 256,
                mismatch_tau    = cfg.mismatch_tau,
            )

        if cfg.use_reflection:
            self.reflection = ReflectionModule(
                d_latent   = d_latent,
                verify_tau = cfg.verify_tau,
                lambda_mdl = cfg.lambda_mdl_refl,
            )

        # ── Мета-контролер стратегії ──────────────────────────────────────────
        if cfg.use_meta:
            self.meta_ctrl = SynthesisMetaController(
                d_state      = 32,
                beta         = cfg.meta_beta,
                entropy_beta = cfg.meta_entropy_beta,
                dropout      = cfg.dropout,
            )

        # ── Plan-Code Consistency: порівнює план з кодом (L_plan) ────────────
        # L_plan = MSE(z_plan_summary, z_code_summary)
        self.plan_code_align = nn.Sequential(
            nn.Linear(cfg.d_plan, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        self.code_summarizer = nn.Linear(d_tok, d_latent, bias=False)

    # ── Допоміжний: якість → нормований reward ──────────────────────────────
    @staticmethod
    def _quality_reward(ce_loss: float) -> float:
        """Нормований reward з CE: 1 при CE=0, ~0 при CE=5."""
        return max(0.0, 1.0 - ce_loss / 5.0)

    def _plan_with_strategy(
        self,
        intent_state: IntentState,
        strategy_id: int,
        plan_depth_use: int,
        symbolic_goal=None,
        symbolic_facts=None,
        prover=None,
    ) -> PlanSequence:
        orig_depth = self.planner.max_depth
        try:
            if strategy_id == STRATEGY_FAST:
                self.planner.max_depth = 1
            else:
                self.planner.max_depth = min(plan_depth_use, orig_depth)
            return self.planner(
                intent_state,
                symbolic_goal=symbolic_goal,
                symbolic_facts=symbolic_facts,
                prover=prover,
            )
        finally:
            self.planner.max_depth = orig_depth

    @staticmethod
    def _truncate_plan(plan: PlanSequence, sim_steps: int) -> PlanSequence:
        if sim_steps <= 0 or plan.embeddings.size(0) <= sim_steps:
            return plan
        return PlanSequence(
            operators=plan.operators[:sim_steps],
            embeddings=plan.embeddings[:sim_steps],
            goal_reached=plan.goal_reached,
            goal_progress=plan.goal_progress,
            goal_facts=plan.goal_facts,
            plan_loss=plan.plan_loss,
        )

    def _repair_plan_symbolically(
        self,
        intent_state: IntentState,
        current_plan: PlanSequence,
        strategy_id: int,
        plan_depth_use: int,
        batch_size: int,
        device: torch.device,
        symbolic_goal=None,
        symbolic_facts=None,
        prover=None,
    ) -> Tuple[PlanSequence, object, int]:
        best_plan = current_plan
        best_verify = self.symbolic_verifier(current_plan, batch_size=batch_size, device=device)
        seen = {(strategy_id, plan_depth_use)}
        tried = 0
        candidate_specs = [
            (strategy_id, min(plan_depth_use + 1, self.cfg.max_plan_depth + 2)),
            (STRATEGY_CAREFUL, max(plan_depth_use, 2)),
            (STRATEGY_EXPLORATORY, min(plan_depth_use + 1, self.cfg.max_plan_depth + 2)),
        ]
        for cand_strategy, cand_depth in candidate_specs:
            spec = (cand_strategy, cand_depth)
            if spec in seen:
                continue
            seen.add(spec)
            candidate_plan = self._plan_with_strategy(
                intent_state,
                cand_strategy,
                cand_depth,
                symbolic_goal=symbolic_goal,
                symbolic_facts=symbolic_facts,
                prover=prover,
            )
            candidate_verify = self.symbolic_verifier(candidate_plan, batch_size=batch_size, device=device)
            tried += 1
            cand_score = (
                float(candidate_verify.goal_progress.mean().item()),
                -float(candidate_verify.l_verify.item()),
                -int(candidate_verify.mismatch_mask.sum().item()),
            )
            best_score = (
                float(best_verify.goal_progress.mean().item()),
                -float(best_verify.l_verify.item()),
                -int(best_verify.mismatch_mask.sum().item()),
            )
            if cand_score > best_score:
                best_plan = candidate_plan
                best_verify = candidate_verify
        return best_plan, best_verify, tried

    # ── Основний forward ────────────────────────────────────────────────────
    def forward(
        self,
        h_tok:     torch.Tensor,           # (B, T, d_tok)
        z_final:   torch.Tensor,           # (B, d_latent)
        tgt:       torch.Tensor,           # (B, T) токени
        world_rnn: nn.Module,              # WorldRNN з omen_v2
        gap_norm:  float = 0.0,
        ce_loss:   float = 5.0,
        n_rules:   int   = 0,
        n_writes:  int   = 0,
        prover=None,
        symbolic_goal=None,
        symbolic_facts=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
          logits  : (B, T, vocab_size)
          osf_out : dict з лос-компонентами та статистикою
        """
        device = z_final.device
        B, T   = tgt.shape

        # ── Вибір стратегії σ ─────────────────────────────────────────────────
        intent_state = self.intent_encoder(z_final)              # IntentState
        goal_entropy = float(intent_state.goal_entropy.mean().item())
        goal_confidence = float(intent_state.goal_probs.max(dim=-1).values.mean().item())
        latent_norm = float(z_final.norm(dim=-1).mean().item() / math.sqrt(max(self.d_latent, 1)))

        cfg_used   = self.cfg
        log_prob_meta  = torch.tensor(0.0, device=device)
        value_meta     = torch.tensor(0.0, device=device)
        entropy_meta   = torch.tensor(0.0, device=device)
        strategy_id    = STRATEGY_CAREFUL
        strategy_tau   = 0.5

        if cfg_used.use_meta:
            strat_cfg, log_prob_meta, value_meta, entropy_meta = self.meta_ctrl.select_strategy(
                gap_norm=gap_norm, ce_loss=ce_loss,
                plan_depth=cfg_used.max_plan_depth,
                n_rules=n_rules, n_writes=n_writes,
                goal_entropy=goal_entropy,
                goal_confidence=goal_confidence,
                latent_norm=latent_norm,
            )
            strategy_id    = strat_cfg.strategy_id
            plan_depth_use = strat_cfg.plan_depth
            n_reflections  = strat_cfg.n_reflections
            sim_steps      = strat_cfg.sim_steps
            strategy_tau   = strat_cfg.confidence_tau
        else:
            # Inference або без meta: Careful за замовчуванням
            plan_depth_use = cfg_used.max_plan_depth
            n_reflections  = 1 if cfg_used.use_reflection else 0
            sim_steps      = 4

        # ── H1: Intent ────────────────────────────────────────────────────────
        intent_state = self.intent_encoder(z_final)              # IntentState
        goal_entropy = float(intent_state.goal_entropy.mean().item())
        goal_confidence = float(intent_state.goal_probs.max(dim=-1).values.mean().item())
        latent_norm = float(z_final.norm(dim=-1).mean().item() / math.sqrt(max(self.d_latent, 1)))

        # ── H2: Plan ─────────────────────────────────────────────────────────
        plan = self._plan_with_strategy(
            intent_state, strategy_id, plan_depth_use,
            symbolic_goal=symbolic_goal,
            symbolic_facts=symbolic_facts,
            prover=prover,
        )

        # ── Симуляція + symbolic verification ───────────────────────────────
        l_sim       = torch.tensor(0.0, device=device)
        l_verify    = torch.tensor(0.0, device=device)
        z_reflected = z_final
        l_refl      = torch.tensor(0.0, device=device)
        refl_iters  = 0
        mismatch_before = 0
        mismatch_after  = 0
        symbolic_mismatch_before = 0
        symbolic_mismatch_after  = 0
        symbolic_goal_progress   = float(plan.goal_progress)

        sim_plan = self._truncate_plan(plan, sim_steps)
        verify_result = self.symbolic_verifier(plan, batch_size=B, device=device)
        l_verify = verify_result.l_verify
        symbolic_goal_progress = float(verify_result.goal_progress.mean().item())
        symbolic_mismatch_before = int(verify_result.mismatch_mask.sum().item())
        symbolic_mismatch_after = symbolic_mismatch_before
        mismatch_before = symbolic_mismatch_before
        mismatch_after = symbolic_mismatch_after
        symbolic_repairs = 0

        symbolic_failed = (
            symbolic_mismatch_before > 0
            or symbolic_goal_progress + 1e-6 < max(float(plan.goal_progress), strategy_tau)
        )
        if cfg_used.use_reflection and symbolic_failed:
            repaired_plan, repaired_verify, symbolic_repairs = self._repair_plan_symbolically(
                intent_state,
                plan,
                strategy_id,
                plan_depth_use,
                batch_size=B,
                device=device,
                symbolic_goal=symbolic_goal,
                symbolic_facts=symbolic_facts,
                prover=prover,
            )
            if symbolic_repairs > 0:
                refl_iters += symbolic_repairs
                l_refl = l_refl + self.cfg.lambda_mdl_refl * torch.tensor(
                    float(symbolic_repairs),
                    device=device,
                )
            repaired_score = (
                float(repaired_verify.goal_progress.mean().item()),
                -float(repaired_verify.l_verify.item()),
                -int(repaired_verify.mismatch_mask.sum().item()),
            )
            current_score = (
                symbolic_goal_progress,
                -float(l_verify.item()),
                -symbolic_mismatch_before,
            )
            if repaired_score > current_score:
                plan = repaired_plan
                verify_result = repaired_verify
                l_verify = verify_result.l_verify
                symbolic_goal_progress = float(verify_result.goal_progress.mean().item())
                symbolic_mismatch_after = int(verify_result.mismatch_mask.sum().item())
                mismatch_after = symbolic_mismatch_after
                sim_plan = self._truncate_plan(plan, sim_steps)

        if cfg_used.use_simulation and strategy_id != STRATEGY_FAST:
            sim_result = self.simulator(z_reflected, sim_plan, world_rnn)
            l_sim = sim_result.l_sim
            mismatch_before = max(mismatch_before, int(sim_result.mismatch_mask.sum().item()))
            mismatch_after = max(mismatch_after, int(sim_result.mismatch_mask.sum().item()))

            # ── Рефлексія з повторним simulation+verification ───────────────
            needs_latent_reflection = (
                cfg_used.use_reflection
                and (
                    sim_result.mismatch_mask.any()
                    or symbolic_mismatch_after > 0
                    or symbolic_goal_progress + 1e-6 < strategy_tau
                )
            )
            if needs_latent_reflection:
                best_sim = sim_result
                best_verify = verify_result
                best_z = z_reflected
                best_plan = plan
                best_sim_plan = sim_plan
                for _ in range(min(n_reflections, 3)):
                    patch = self.reflection(best_z, best_sim)
                    candidate_z = self.reflection.apply_patch(best_z, patch)
                    candidate_intent = self.intent_encoder(candidate_z)
                    candidate_plan = self._plan_with_strategy(
                        candidate_intent, strategy_id, plan_depth_use,
                        symbolic_goal=symbolic_goal,
                        symbolic_facts=symbolic_facts,
                        prover=prover,
                    )
                    candidate_verify = self.symbolic_verifier(
                        candidate_plan, batch_size=B, device=device
                    )
                    if (
                        int(candidate_verify.mismatch_mask.sum().item()) > 0
                        or float(candidate_verify.goal_progress.mean().item()) + 1e-6 < strategy_tau
                    ):
                        candidate_plan, candidate_verify, repair_tries = self._repair_plan_symbolically(
                            candidate_intent,
                            candidate_plan,
                            strategy_id,
                            plan_depth_use,
                            batch_size=B,
                            device=device,
                            symbolic_goal=symbolic_goal,
                            symbolic_facts=symbolic_facts,
                            prover=prover,
                        )
                        if repair_tries > 0:
                            symbolic_repairs += repair_tries
                            refl_iters += repair_tries
                            l_refl = l_refl + self.cfg.lambda_mdl_refl * torch.tensor(
                                float(repair_tries),
                                device=device,
                            )
                    candidate_sim_plan = self._truncate_plan(candidate_plan, sim_steps)
                    candidate_sim = self.simulator(candidate_z, candidate_sim_plan, world_rnn)
                    refl_iters += 1
                    l_refl = l_refl + patch.l_refl
                    cand_score = (
                        float(candidate_verify.goal_progress.mean().item()),
                        -int(candidate_verify.mismatch_mask.sum().item()),
                        -(float(candidate_verify.l_verify.item()) + 0.25 * float(candidate_sim.l_sim.item())),
                    )
                    best_score = (
                        float(best_verify.goal_progress.mean().item()),
                        -int(best_verify.mismatch_mask.sum().item()),
                        -(float(best_verify.l_verify.item()) + 0.25 * float(best_sim.l_sim.item())),
                    )
                    if cand_score > best_score:
                        best_z = candidate_z
                        best_plan = candidate_plan
                        best_sim_plan = candidate_sim_plan
                        best_sim = candidate_sim
                        best_verify = candidate_verify
                        symbolic_mismatch_after = int(candidate_verify.mismatch_mask.sum().item())
                        mismatch_after = max(
                            int(candidate_sim.mismatch_mask.sum().item()),
                            symbolic_mismatch_after,
                        )
                        symbolic_goal_progress = float(candidate_verify.goal_progress.mean().item())
                    if (
                        symbolic_mismatch_after == 0
                        and mismatch_after == 0
                        and symbolic_goal_progress + 1e-6 >= strategy_tau
                    ):
                        break
                z_reflected = best_z
                plan = best_plan
                sim_plan = best_sim_plan
                sim_result = best_sim
                verify_result = best_verify
                l_sim = sim_result.l_sim
                l_verify = verify_result.l_verify
                mismatch_after = max(mismatch_after, symbolic_mismatch_after)

        # ── H3+H4: Hierarchical Decoder ──────────────────────────────────────
        decode_intent_state = (
            self.intent_encoder(z_reflected)
            if refl_iters > 0 and cfg_used.use_simulation and strategy_id != STRATEGY_FAST
            else intent_state
        )
        decode_goal_confidence = float(
            decode_intent_state.goal_probs.max(dim=-1).values.mean().item()
        )


        logits, struct_loss = self.hier_decoder(
            h_tok      = h_tok,
            z_intent   = decode_intent_state.z_intent,
            plan       = plan,
            tgt_tokens = tgt,
        )

        # ── L_plan: план-код консистентність ─────────────────────────────────
        # Summary плану: mean(plan.embeddings)
        plan_emb_lat = self.plan_code_align(
            plan.embeddings.to(device).mean(0, keepdim=True))       # (1, d_latent)
        plan_emb_lat = plan_emb_lat.expand(B, -1)                   # (B, d_latent)

        # Summary коду (через h_tok): mean по часовій осі
        code_summary = self.code_summarizer(h_tok.mean(1))          # (B, d_latent)

        goal_penalty = F.relu(
            torch.tensor(strategy_tau - symbolic_goal_progress, device=device)
        )
        l_plan = F.mse_loss(code_summary, plan_emb_lat.detach()) + goal_penalty

        # ── Intent anti-collapse ──────────────────────────────────────────────
        l_intent = self.intent_encoder.intent_loss(decode_intent_state)

        # ── L_meta: REINFORCE для стратегії ──────────────────────────────────
        l_meta = torch.tensor(0.0, device=device)
        if cfg_used.use_meta and self.training:
            quality   = self._quality_reward(ce_loss)
            meta_traj = self.meta_ctrl.compute_meta_loss(
                log_prob_meta, value_meta, quality, strategy_id, entropy=entropy_meta)
            l_meta    = meta_traj.meta_loss

        # ── J_OSF = L_plan + λ_sim·L_sim + λ_refl·L_refl + λ_meta·L_meta ───
        # plan.plan_loss — REINFORCE-лос планувальника (policy gradient).
        # Обрізаємо до [-1, 1] для стабільності: великі PG-грани розбалансовують
        # спільний функціонал J_OSF, тому tight clamp тут критичний.
        plan_rl_loss = plan.plan_loss.clamp(-1.0, 1.0) if self.training else torch.tensor(0.0, device=device)
        sim_mix = l_verify + 0.25 * l_sim

        j_osf = (
            plan_rl_loss                            # REINFORCE для планувальника
          + cfg_used.lambda_plan   * l_plan
          + cfg_used.lambda_sim    * sim_mix
          + cfg_used.lambda_refl   * l_refl
          + cfg_used.lambda_meta   * l_meta
          + cfg_used.lambda_intent * l_intent
          + struct_loss                         # вже зважений у HierarchicalDecoder
        )

        # Clamp для стабільності
        j_osf = j_osf.clamp(-10.0, 10.0) if torch.is_tensor(j_osf) else j_osf

        # ── Статистика ────────────────────────────────────────────────────────
        osf_out: Dict = {
            "j_osf":           j_osf,
            "osf_l_plan":      l_plan.item(),
            "osf_l_sim":       l_sim.item()   if torch.is_tensor(l_sim)   else float(l_sim),
            "osf_l_verify":    l_verify.item() if torch.is_tensor(l_verify) else float(l_verify),
            "osf_l_sim_mix":   sim_mix.item() if torch.is_tensor(sim_mix) else float(sim_mix),
            "osf_l_refl":      l_refl.item()  if torch.is_tensor(l_refl)  else float(l_refl),
            "osf_l_meta":      l_meta.item()  if torch.is_tensor(l_meta)  else float(l_meta),
            "osf_l_intent":    l_intent.item() if torch.is_tensor(l_intent) else float(l_intent),
            "osf_struct":      struct_loss.item(),
            "osf_strategy":    strategy_id,
            "osf_confidence_tau": strategy_tau,
            "osf_plan_depth":  len(plan.operators),
            "osf_sim_steps":   sim_plan.embeddings.size(0),
            "osf_reflections": refl_iters,
            "osf_symbolic_repairs": symbolic_repairs,
            "osf_mismatch_before": mismatch_before,
            "osf_mismatch_after": mismatch_after,
            "osf_symbolic_mismatch_before": symbolic_mismatch_before,
            "osf_symbolic_mismatch_after": symbolic_mismatch_after,
            "osf_goal_progress": plan.goal_progress,
            "osf_symbolic_goal_progress": symbolic_goal_progress,
            "osf_goal_entropy": decode_intent_state.goal_entropy.mean().item(),
            "osf_goal_confidence": decode_goal_confidence,
            "osf_plan_rl":     plan_rl_loss.item() if self.training else 0.0,
            "osf_plan_loss":   plan.plan_loss.item(),
        }

        if cfg_used.use_meta and hasattr(self, "meta_ctrl"):
            osf_out.update(self.meta_ctrl.strategy_stats())

        return logits, osf_out

    def memory_report(self) -> str:
        """Звіт про параметри OSF."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cfg      = self.cfg
        return (
            f"  ── OSF Synthesis Framework ──\n"
            f"  OSF enabled         : {cfg.osf_enabled}\n"
            f"  OSF params          : {n_params:,}\n"
            f"  d_intent            : {cfg.d_intent}\n"
            f"  n_goals             : {cfg.n_goals}\n"
            f"  d_plan              : {cfg.d_plan}\n"
            f"  n_operators         : {cfg.n_operators}\n"
            f"  template_len        : {cfg.template_len}\n"
            f"  max_plan_depth      : {cfg.max_plan_depth}\n"
            f"  use_simulation      : {cfg.use_simulation}\n"
            f"  use_reflection      : {cfg.use_reflection}\n"
            f"  use_meta            : {cfg.use_meta}\n"
            f"  λ_plan/sim/refl     : {cfg.lambda_plan}/{cfg.lambda_sim}/{cfg.lambda_refl}\n"
        )
class _DummyWorldRNN(nn.Module):
    def __init__(self, d_latent: int, vocab_size: int = 256):
        super().__init__()
        self.act_emb = nn.Embedding(vocab_size, d_latent)
        nn.init.normal_(self.act_emb.weight, std=0.2)

    def simulate_sequence(self, z0: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        act = self.act_emb(action_seq)
        return z0.unsqueeze(1) + 0.1 * act.cumsum(dim=1)


def run_tests_osf() -> None:
    torch.manual_seed(0)

    B, T = 2, 12
    d_latent = 16
    d_tok = 16
    vocab_size = 64
    h_tok = torch.randn(B, T, d_tok)
    z_final = torch.randn(B, d_latent)
    tgt = torch.randint(0, vocab_size, (B, T))
    world_rnn = _DummyWorldRNN(d_latent, vocab_size=256)

    cfg_plain = OSFConfig(
        osf_enabled=True,
        d_intent=16,
        n_goals=8,
        d_plan=16,
        n_operators=8,
        template_len=4,
        max_plan_depth=5,
        beam_width=2,
        use_simulation=True,
        use_reflection=True,
        use_meta=False,
        dropout=0.0,
    )
    osf_plain = OSFSynthesizer(cfg_plain, d_latent=d_latent, d_tok=d_tok, vocab_size=vocab_size, n_heads=2)
    osf_plain.eval()
    logits_plain, out_plain = osf_plain(
        h_tok=h_tok,
        z_final=z_final,
        tgt=tgt,
        world_rnn=world_rnn,
        gap_norm=0.3,
        ce_loss=1.1,
        n_rules=4,
        n_writes=2,
    )
    assert logits_plain.shape == (B, T, vocab_size)
    assert torch.isfinite(out_plain["j_osf"])
    assert out_plain["osf_sim_steps"] <= 4
    assert out_plain["osf_mismatch_after"] <= out_plain["osf_mismatch_before"]
    assert 0.0 <= out_plain["osf_goal_progress"] <= 1.0
    assert out_plain["osf_l_verify"] >= 0.0
    assert out_plain["osf_symbolic_mismatch_after"] <= out_plain["osf_symbolic_mismatch_before"]
    assert 0.0 <= out_plain["osf_symbolic_goal_progress"] <= 1.0

    cfg_meta = OSFConfig(
        osf_enabled=True,
        d_intent=16,
        n_goals=8,
        d_plan=16,
        n_operators=8,
        template_len=4,
        max_plan_depth=6,
        beam_width=2,
        use_simulation=True,
        use_reflection=True,
        use_meta=True,
        dropout=0.0,
    )
    osf_meta = OSFSynthesizer(cfg_meta, d_latent=d_latent, d_tok=d_tok, vocab_size=vocab_size, n_heads=2)
    osf_meta.eval()
    logits_meta, out_meta = osf_meta(
        h_tok=h_tok,
        z_final=z_final,
        tgt=tgt,
        world_rnn=world_rnn,
        gap_norm=0.6,
        ce_loss=0.9,
        n_rules=7,
        n_writes=5,
    )
    assert logits_meta.shape == (B, T, vocab_size)
    assert out_meta["osf_strategy"] in (STRATEGY_FAST, STRATEGY_CAREFUL, STRATEGY_EXPLORATORY)
    assert out_meta["osf_sim_steps"] <= STRATEGY_CONFIGS[out_meta["osf_strategy"]].sim_steps
    assert 0.0 <= out_meta["osf_goal_confidence"] <= 1.0
    assert out_meta["osf_l_verify"] >= 0.0
    assert "meta_avg_reward" in out_meta
    print("OSF tests passed.")


if __name__ == "__main__":
    run_tests_osf()

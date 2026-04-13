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
from omen_osf_simulator import WorldSimulator, ReflectionModule
from omen_osf_meta      import (
    SynthesisMetaController, StrategyConfig,
    STRATEGY_FAST, STRATEGY_CAREFUL, STRATEGY_EXPLORATORY,
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
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
          logits  : (B, T, vocab_size)
          osf_out : dict з лос-компонентами та статистикою
        """
        device = z_final.device
        B, T   = tgt.shape

        # ── Вибір стратегії σ ─────────────────────────────────────────────────
        cfg_used   = self.cfg
        log_prob_meta  = torch.tensor(0.0, device=device)
        value_meta     = torch.tensor(0.0, device=device)
        strategy_id    = STRATEGY_CAREFUL

        if cfg_used.use_meta and self.training:
            strat_cfg, log_prob_meta, value_meta = self.meta_ctrl.select_strategy(
                gap_norm=gap_norm, ce_loss=ce_loss,
                plan_depth=cfg_used.max_plan_depth,
                n_rules=n_rules, n_writes=n_writes,
            )
            strategy_id    = strat_cfg.strategy_id
            plan_depth_use = strat_cfg.plan_depth
            n_reflections  = strat_cfg.n_reflections
            sim_steps      = strat_cfg.sim_steps
        else:
            # Inference або без meta: Careful за замовчуванням
            plan_depth_use = cfg_used.max_plan_depth
            n_reflections  = 1 if cfg_used.use_reflection else 0
            sim_steps      = 4

        # ── H1: Intent ────────────────────────────────────────────────────────
        intent_state = self.intent_encoder(z_final)              # IntentState

        # ── H2: Plan (якщо стратегія не Fast) ────────────────────────────────
        if strategy_id == STRATEGY_FAST:
            # Fast: мінімальний план (1 крок)
            _orig_depth           = self.planner.max_depth
            self.planner.max_depth = 1
            plan = self.planner(intent_state)
            self.planner.max_depth = _orig_depth
        else:
            # Careful / Exploratory
            _orig_depth           = self.planner.max_depth
            self.planner.max_depth = min(plan_depth_use, self.planner.max_depth)
            plan = self.planner(intent_state)
            self.planner.max_depth = _orig_depth

        # ── Симуляція (якщо не Fast) ─────────────────────────────────────────
        l_sim       = torch.tensor(0.0, device=device)
        z_reflected = z_final

        if cfg_used.use_simulation and strategy_id != STRATEGY_FAST:
            sim_result = self.simulator(z_final, plan, world_rnn)
            l_sim      = sim_result.l_sim

            # ── Рефлексія (якщо є невідповідності) ───────────────────────────
            if cfg_used.use_reflection and sim_result.mismatch_mask.any():
                for _ in range(min(n_reflections, 3)):
                    patch      = self.reflection(z_reflected, sim_result)
                    z_reflected = self.reflection.apply_patch(z_reflected, patch)
                    l_refl_i   = patch.l_refl
                    # Оновлюємо симуляцію з виправленим z
                    if sim_result.mismatch_mask.any() and self.training:
                        sim_result = self.simulator(z_reflected, plan, world_rnn)
                l_refl = patch.l_refl if cfg_used.use_reflection else torch.tensor(0.0, device=device)
            else:
                l_refl = torch.tensor(0.0, device=device)
        else:
            l_refl = torch.tensor(0.0, device=device)

        # ── H3+H4: Hierarchical Decoder ──────────────────────────────────────
        logits, struct_loss = self.hier_decoder(
            h_tok      = h_tok,
            z_intent   = intent_state.z_intent,
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

        l_plan = F.mse_loss(code_summary, plan_emb_lat.detach())    # scalar

        # ── Intent anti-collapse ──────────────────────────────────────────────
        l_intent = self.intent_encoder.intent_loss(intent_state)

        # ── L_meta: REINFORCE для стратегії ──────────────────────────────────
        l_meta = torch.tensor(0.0, device=device)
        if cfg_used.use_meta and self.training:
            quality   = self._quality_reward(ce_loss)
            meta_traj = self.meta_ctrl.compute_meta_loss(
                log_prob_meta, value_meta, quality, strategy_id)
            l_meta    = meta_traj.meta_loss

        # ── J_OSF = L_plan + λ_sim·L_sim + λ_refl·L_refl + λ_meta·L_meta ───
        # plan.plan_loss — REINFORCE-лос планувальника (policy gradient).
        # Обрізаємо до [-1, 1] для стабільності: великі PG-грани розбалансовують
        # спільний функціонал J_OSF, тому tight clamp тут критичний.
        plan_rl_loss = plan.plan_loss.clamp(-1.0, 1.0) if self.training else torch.tensor(0.0, device=device)

        j_osf = (
            plan_rl_loss                            # REINFORCE для планувальника
          + cfg_used.lambda_plan   * l_plan
          + cfg_used.lambda_sim    * l_sim
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
            "osf_l_refl":      l_refl.item()  if torch.is_tensor(l_refl)  else float(l_refl),
            "osf_l_meta":      l_meta.item()  if torch.is_tensor(l_meta)  else float(l_meta),
            "osf_l_intent":    l_intent.item() if torch.is_tensor(l_intent) else float(l_intent),
            "osf_struct":      struct_loss.item(),
            "osf_strategy":    strategy_id,
            "osf_plan_depth":  len(plan.operators),
            "osf_goal_entropy": intent_state.goal_entropy.mean().item(),
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
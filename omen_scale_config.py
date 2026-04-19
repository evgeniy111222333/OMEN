"""
omen_scale_config.py: configuration surface for the canonical OMEN runtime.

The config spans byte/token perception, graph-native world modeling, symbolic
reasoning, memory, saliency, NET, EMC, creative cycle modules, and evaluation
protocol controls.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class OMENScaleConfig:
    allow_noncanonical_ablation: bool = False

    # ─── Level 1: Token-level (Fine) ──────────────────────────────────────────
    vocab_size:      int   = 256      # NET operates on raw UTF-8 bytes [0..255]
    d_tok:           int   = 1_024    # Token-level dimensionality (>=1024)
    n_heads_tok:     int   = 16       # Token-level attention heads
    n_layers_tok:    int   = 12       # Token-level transformer layers
    seq_len:         int   = 4_096    # Context length (>=4096)

    # ─── Level 2: Concept-level (Coarse) - Perceiver Resampler ───────────────
    d_latent:        int   = 256      # Concept space (used by WorldRNN and M-Core)
    n_latents:       int   = 64       # Number of Perceiver latent queries
    n_heads_lat:     int   = 8        # Perceiver cross-attention heads
    n_layers_lat:    int   = 2        # Perceiver self-attention layers

    # ─── WorldRNN (concept level) ─────────────────────────────────────────────
    world_rnn_hidden: int  = 512
    world_rollout_steps: int = 8
    world_teacher_forcing_start: float = 0.35
    world_teacher_forcing_end:   float = 0.05
    world_teacher_forcing_steps: int   = 2_000
    world_graph_enabled: bool = True
    world_graph_pred_buckets: int = 4_096
    world_graph_term_buckets: int = 8_192
    world_graph_layers: int = 2
    world_graph_max_nodes: int = 128
    world_graph_max_edges: int = 512
    world_graph_max_transitions: int = 16
    world_graph_state_mix: float = 0.35
    world_graph_teacher_mix: float = 0.65
    world_graph_pooled_mix: float = 0.15
    world_graph_trace_mix: float = 1.0
    world_graph_execution_driven: bool = True
    world_graph_hidden_mix: float = 0.15
    world_graph_trace_pad_with_first: bool = True
    world_state_anchor_mix: float = 1.0
    world_graph_context_limit: int = 32
    world_graph_transition_mix: float = 0.2

    # ─── M-Core (async updates) ───────────────────────────────────────────────
    mem_heads:        int   = 16
    mem_cache_size:   int   = 2_048
    mem_symbolic_cache_size: int = 2_048
    mem_write_tau:    float = 0.3
    mem_update_steps: int   = 8      # update every N steps (not every step)
    mem_decay:        float = 0.995
    mem_symbolic_recall_topk: int = 8
    mem_symbolic_min_sim: float = 0.20

    # ─── ∂-Prolog (Symbolic) ──────────────────────────────────────────────────
    sym_vocab:        int   = 128     # symbolic vocabulary size
    sym_embed_dim:    int   = 64      # symbolic embedding dimensionality
    max_proof_depth:  int   = 5       # proof search depth
    n_proof_cands:    int   = 16      # abduction candidate count
    ltm_max_rules:    int   = 1_024   # max rules in KnowledgeBase
    sym_max_facts:    int   = 64      # max facts in WorkingMemory
    abduct_candidates: int  = 8       # abduction engine candidates
    sym_gnn_layers:   int   = 2       # compatibility with OMENv2
    continuous_cycle_enabled: bool = True
    continuous_cycle_eval_enabled: bool = True
    continuous_cycle_trace_candidates: int = 4
    continuous_cycle_contextual: int = 4
    continuous_cycle_neural: int = 4
    continuous_cycle_accept_threshold: float = 0.55
    continuous_cycle_verify_threshold: float = 0.75
    continuous_cycle_contradict_threshold: float = 0.15
    continuous_cycle_symbolic_weight: float = 0.30
    continuous_cycle_world_weight: float = 0.55
    continuous_cycle_token_weight: float = 0.15
    continuous_cycle_world_reject_threshold: float = 0.75
    continuous_cycle_soft_symbolic_weight: float = 0.45
    continuous_cycle_policy_weight: float = 0.25
    continuous_cycle_policy_baseline_momentum: float = 0.90
    continuous_cycle_candidate_tau: float = 0.70
    continuous_cycle_repair_enabled: bool = True
    continuous_cycle_repair_threshold: float = 0.35
    continuous_cycle_max_repairs: int = 2
    continuous_cycle_eval_learning_enabled: bool = True
    eval_world_self_update_enabled: bool = True
    eval_world_self_update_lr: float = 1e-3
    eval_world_self_update_clip: float = 1.0
    eval_world_self_update_program_weight: float = 0.05
    world_rule_symbolic_weight: float = 0.25
    world_rule_world_weight: float = 0.75
    world_abduction_symbolic_weight: float = 0.20
    world_abduction_trace_weight: float = 0.15
    world_abduction_world_weight: float = 0.65
    world_causal_weight: float = 0.35
    creative_cycle_enabled: bool = True
    creative_cycle_every: int = 4
    creative_max_selected_rules: int = 2
    ame_embedding_dim: int = 16
    ame_tau_analogy: float = 0.82
    ame_hidden_dim: int = 64
    ame_gnn_layers: int = 2
    ame_spec_ratio: float = 0.5
    ame_temperature: float = 0.07
    ame_contrastive_steps: int = 2
    ame_contrastive_lr: float = 3e-3
    ame_dropout: float = 0.10
    cwe_max_rule_mods: int = 2
    cwe_surprise_lambda: float = 0.5
    cwe_max_candidates: int = 8
    cwe_max_transforms_per_rule: int = 4
    aee_population: int = 16
    aee_generations: int = 2
    aee_gamma: float = 0.25
    aee_mutation_rate: float = 0.35
    aee_crossover_rate: float = 0.5
    aee_ltm_seed_ratio: float = 0.35
    aee_gene_pool_size: int = 32
    oee_gap_threshold: float = 0.45
    oee_contradiction_threshold: int = 1
    oee_d_latent: int = 32
    oee_consistency_lambda: float = 0.1
    oee_online_lr: float = 1e-3
    oee_forward_chain_depth: int = 2
    oee_max_interaction_preds: int = 3
    oee_max_hypotheses: int = 8
    creative_train_fast_cwe_max_rule_mods: int = 1
    creative_train_fast_cwe_max_candidates: int = 2
    creative_train_fast_cwe_max_transforms_per_rule: int = 1
    creative_train_fast_oee_max_candidates: int = 2
    creative_train_fast_oee_max_targets: int = 2
    creative_train_fast_oee_max_paradox_facts: int = 2
    creative_train_fast_oee_max_hypotheses: int = 4
    creative_train_fast_oee_max_scored_hypotheses: int = 32
    creative_train_fast_oee_max_open_body_literals: int = 1
    creative_train_fast_oee_max_open_patterns: int = 2
    creative_train_fast_oee_max_open_head_patterns: int = 2
    creative_train_fast_oee_bundle_beam_width: int = 2
    creative_train_fast_oee_max_bundle_rules: int = 2
    creative_train_fast_oee_bundle_seed_k: int = 4
    ice_state_history: int = 128
    ice_goal_threshold: float = 0.35
    ce_reinforce_enabled: bool = True
    ce_reinforce_eval_enabled: bool = True
    ce_reinforce_fallback_only: bool = False
    ce_reinforce_retro_every: int = 0

    # ─── Epistemic / Curiosity ────────────────────────────────────────────────
    epistemic_tau:    float = 0.3
    epistemic_exact_grad: bool = False
    epistemic_memory_mix: float = 1.0
    n_counterfactual: int   = 2
    symbolic_context_max_facts: int = 96
    symbolic_ast_max_facts: int = 48
    sym_trace_max_steps: int = 24
    sym_trace_max_counterexamples: int = 4
    sym_graph_reasoning_enabled: bool = True
    sym_graph_reasoning_top_k_facts: int = 12
    sym_graph_reasoning_max_fact_subset: int = 96
    sym_graph_reasoning_attention_threshold: float = 0.02
    sym_graph_reasoning_tau: float = 0.5
    sym_graph_reasoning_full_scan_cutoff: int = 64
    sym_query_gen_enabled: bool = True
    sym_query_alpha: float = 0.05
    sym_query_lambda: float = 0.05
    sym_query_entropy_beta: float = 1e-3
    sym_query_gumbel_tau: float = 0.85
    sym_query_hard_mask_threshold: float = 0.75
    sym_decoder_surprise_enabled: bool = True
    sym_decoder_surprise_lambda: float = 0.05
    sym_decoder_surprise_threshold: float = 0.35
    sym_cycle_loss_weight: float = 0.10
    sym_abduction_loss_weight: float = 0.05
    program_anchor_enabled: bool = True
    program_anchor_weight: float = 0.10
    program_decoder_weight: float = 0.05
    program_anchor_max_facts: int = 24
    vfe_enabled: bool = True
    vfe_beta_kl: float = 1.0
    vfe_free_bits: float = 0.0
    use_aux_loss_schedule: bool = False

    # ─── MDL / VFE Coefficients ───────────────────────────────────────────────
    # FreeEnergy ~= Surprise(Data | z, G) + DescriptionLength(theta, rules)
    #              + alpha * [-log P(Read(M, z) | z)]
    lambda_tok:  float = 1.0    # kept for backward compatibility; no longer used as a unit-conversion hack
    lambda_conc: float = 1.0    # kept for backward compatibility; no longer used as a unit-conversion hack
    lambda_rule: float = 1e-4   # Symbolic rule complexity weight
    alpha:       float = 0.1    # Memory-read likelihood weight
    beta:        float = 0.05   # Symbolic generalization
    gamma:       float = 0.1    # World consistency
    delta:       float = 1e-3   # WorldRNN complexity
    eta:         float = 0.05   # Memory recall
    lam_sym:     float = 0.005   # LTM regularizer (v2 compatibility)
    mdl_param_sigma: float = 0.05
    mdl_token_budget: int = 262144  # legacy field; MDL is now normalized by actual valid tokens

    # ─── Neural Epistemic Tokenizer (NET) ─────────────────────────────────────
    # Replaces GPT-2 BPE (vocab=50257) with a neuro-symbolic compressor.
    # MDL optimization: L_NET = L_vq + L_rec + lambda_voc * sum ||e_v||^2
    # Full objective: J_total = J_OMEN + eta_tok * L_NET
    #
    net_enabled:      bool  = True       # enables NET instead of BPE
    net_byte_layers:  int   = 2          # ByteContextEncoder (f_theta) layers
    net_dec_layers:   int   = 2          # ByteDecoder (g_phi) layers
    net_init_vocab:   int   = 512        # initial NET vocabulary size
    net_max_vocab:    int   = 8_192      # maximum NET vocabulary size
    net_tau:          float = 0.85       # cosine-similarity threshold (new token)
    net_ema_decay:    float = 0.95       # EMA for codebook updates (0.95 > 0.99 means faster adaptation)
    net_warmup_steps: int   = 150        # vocabulary-growth freeze steps at startup
                                         # (encoder is unstable -> do not add tokens yet)
    eta_tok:          float = 0.1        # weight of L_NET inside the global J
    lambda_voc:       float = 1e-4       # MDL vocabulary regularizer

    # ─── Adaptive τ scheduling ────────────────────────────────────────────────
    # tau controls the novelty threshold: if cos_sim < tau -> create a new concept.
    # Adaptive mode lowers tau when H/H_max < 0.55 (too few active codes),
    # and raises it when H/H_max > 0.65 (the vocabulary is being used well).
    # Result: the system balances stability and vocabulary growth on its own.
    net_tau_schedule: bool  = True       # enables adaptive tau after warmup
    net_tau_min:      float = 0.70       # lower tau bound (by default net_tau is the upper bound)

    # ─── Training ─────────────────────────────────────────────────────────────
    dropout:          float = 0.1
    sparsity_lambda:  float = 5e-4
    compile_model:    bool  = False   # torch.compile (enable on A100/H100)
    use_flash_attn:   bool  = True    # FlashAttention when available

    # ─── Verification Module (VeM) ────────────────────────────────────────────
    # Filters AbductionHead candidates before adding them to LTM.
    # U(R) = E[Success(R) − α·Cost(R)]
    # Candidates = {R ~ AbductionHead(z) | U(R) > vem_tau}
    # delta * E_{R~Abduction}[max(0, tau - U(R))] <- penalty for generating poor candidates
    vem_tau:          float = 0.3    # utility threshold (U(R) > vem_tau -> accepted)
    delta_vem:        float = 1e-3   # weight of the VeM penalty inside the global J

    # ─── Epistemic Rule Tracker ───────────────────────────────────────────────
    # Each rule: proposed -> verified / contradicted
    # L_rule = Σ_{R∈LTM} (Complexity(R) − η·Utility(R))
    eta_utility:      float = 0.1    # reward for useful rules (-eta * Utility)
    rule_consolidate_every: int = 100  # steps between Rule Consolidation

    # ─── Semantic Feedback Loop ───────────────────────────────────────────────
    # L_semantic = −E_{(v1,v2)~S-Core}[cos(e_v1, e_v2)·Score(v1, v2)]
    # MDL_total  = MDL_NET − λ_sem·I(Z;Γ)
    lambda_semantic:  float = 0.01   # weight of L_semantic
    lambda_enc_div:   float = 1.5    # Encoder diversity anti-collapse
                                     # FIX Bug1: 0.30->1.5 (at 0.30 enc_div grad was ~87x weaker than l_rec)
    lambda_soft_H:    float = 2.0    # Differentiable soft entropy (key anti-collapse signal)
                                     # FIX Bug2: 0.5->2.0 (at 0.5 soft_H grad was ~4600x weaker than l_rec)
                                     # Soft assignments through temperature=0.5 keep the H gradient nonzero under collapse.

    # --- Saliency Trace language mode -----------------------------------------
    saliency_enabled: bool  = True
    saliency_tau:     float = 0.20
    saliency_top_k:   int   = 4
    saliency_max_facts: int = 512
    saliency_role_slots: int = 6   # prefix of a fixed role ontology; no anonymous role_n fallback
    saliency_beta_struct: float = 0.05
    saliency_gamma_role:  float = 0.05
    saliency_delta_cons:  float = 0.05
    saliency_eta_rule:    float = 1e-4
    saliency_abduce_every: int  = 5
    saliency_consistency_threshold: float = 0.55

    # ─── Efficient Meta-Controller (EMC) ─────────────────────────────────────
    # EMC replaces fixed max_proof_depth with an adaptive meta-policy pi_meta.
    #
    # Bellman equation:
    #   V*(s) = max{ U_stop(s), max_{a∈A} [-C(a) + γ·E V*(s')] }
    #   U_stop(s) = R_task(s) + η_int·R_int(s) − λ_gap·GapNorm(s)
    #
    # Actor-Critic training:
    #   L_meta = L_actor + 0.5 * L_critic   (added to J_OMEN with weight omega_meta)
    #
    # J_OMEN+EMC = J_OMEN
    #            + ω_meta · E_τ[Σ_t (R_task + η_int·R_int − λ_gap·GapNorm − C(a))]
    emc_enabled:       bool  = True    # True -> EMC controls the prover; False -> legacy behavior
    emc_max_steps:     int   = 5       # maximum EMC steps per forward pass
    emc_gamma:         float = 0.95    # future-reward discount factor
    emc_entropy_beta:  float = 0.01    # beta: entropy weight (exploration bonus)
    emc_lambda_time:   float = 0.05    # penalty for each extra reasoning step
    emc_lambda_gap:    float = 0.05    # GapNorm penalty (ignorance)
    emc_lambda_memory_residual: float = 0.02   # memory-residual penalty in meta-control
    emc_lambda_memory_misalignment: float = 0.02  # memory-misalignment penalty in meta-control
    emc_eta_int:       float = 0.10    # bonus for new facts/rules (R_int)
    emc_c_recall:      float = 0.01    # RecallMCore action cost
    emc_c_fc:          float = 0.05    # ForwardChainStep action cost
    emc_c_abduce:      float = 0.10    # Abduce action cost
    emc_c_intrinsic:   float = 0.03    # FocusIntrinsicGoal action cost
    omega_meta:        float = 0.05    # omega_meta: meta_loss weight in the global J
    loss_aux_warmup:   int   = 500     # auxiliary-loss warmup so CE does not dominate forever

    # ─── EMC extension: GAE + MDL(proof) + History ────────────────────────────
    # GAE (Generalized Advantage Estimation):
    #   A_GAE_t = Σ_{k≥0} (γλ)^k · δ_{t+k}
    #   where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  (TD error)
    #   lambda->0: pure TD(0); lambda->1: Monte Carlo. Bias/variance trade-off.
    emc_use_gae:       bool  = True    # True -> GAE instead of Monte Carlo returns
    emc_gae_lambda:    float = 0.95    # lambda in GAE (0=TD, 1=MC)
    emc_train_fast_maintenance_every: int = 4  # cadence for heavy symbolic maintenance in train_fast
    emc_train_fast_cycle_trace_candidates: int = 2
    emc_train_fast_cycle_contextual: int = 2
    emc_train_fast_cycle_neural: int = 2
    emc_train_fast_cycle_max_repairs: int = 1

    # MDL(proof) component:
    #   U_stop(s) -= λ_mdl · MDL(proof)
    #   MDL(proof) = Σ_{R ∈ used_rules} Complexity(R) + depth·c_per_step
    #   Encourages compact proofs (MDL principle extended to reasoning)
    emc_lambda_mdl:    float = 0.01    # proof-complexity penalty

    # History encoding:
    #   State extended to: s_t = (z, gap_norm, depth, n_facts, n_rules, action_hist)
    #   action_hist is a one-hot aggregate of past actions (helps avoid cycles)
    emc_use_action_hist: bool = True   # True -> encode action history into the state

    # ─── Compatibility with OMENv2 ────────────────────────────────────────────
    # (fields expected by OMENAGILoss/WorldRNN)
    n_heads:         int   = 16
    n_layers:        int   = 12
    d_model:         int   = 1_024
    world_rnn_hidden_v2: int = 512    # alias

    # ════════════════════════════════════════════════════════════════════════
    # ─── OMEN Synthesis Framework (OSF) ──────────────────────────────────────
    # OSF replaces TokenDecoder with hierarchical neuro-symbolic generation.
    # Four levels: Intent -> Plan -> Expression -> Tokens.
    #
    # J_OSF = λ_plan·L_plan + λ_sim·L_sim + λ_refl·L_refl + λ_meta·L_meta
    # Full objective: J_total = J_OMEN + lambda_osf * J_OSF
    #
    osf_enabled:      bool  = True      # True -> OSF instead of TokenDecoder
    osf_d_intent:     int   = 64        # Intent-space size (H1)
    osf_n_goals:      int   = 32        # number of abstract goals
    osf_d_plan:       int   = 64        # Plan-space size (H2)
    osf_n_operators:  int   = 32        # library of plan operators
    osf_template_len: int   = 8         # operator template length (H3)
    osf_max_plan_depth: int = 4         # maximum plan depth
    osf_beam_width:   int   = 2         # beam-search width
    osf_lambda_plan:  float = 0.10      # weight of L_plan
    osf_lambda_sim:   float = 0.05      # weight of L_sim
    osf_lambda_refl:  float = 0.05      # weight of L_refl
    osf_lambda_meta:  float = 0.05      # weight of L_meta (strategy)
    osf_lambda_intent: float = 0.01     # anti-collapse intent
    osf_lambda_total: float = 0.3       # lambda_osf: weight of J_OSF in J_total
    osf_use_simulation: bool = True     # WorldSimulator
    osf_use_reflection: bool = True     # ReflectionModule
    osf_use_meta:     bool  = True      # SynthesisMetaController
    osf_meta_beta:    float = 0.1       # quality/cost balance for sigma
    osf_gumbel_tau:   float = 1.0       # Gumbel-Softmax temperature

    @classmethod
    def demo(cls) -> "OMENScaleConfig":
        """Config for testing on any hardware (CPU/GPU)."""
        return cls(
            # IMPORTANT: vocab_size=256 -> true byte mode -> bidirectional attention
            # + segment pooling -> diverse vectors -> no encoder collapse.
            # vocab_size=4096 (legacy) produced MeanSim=0.9935 -> 83% dead codes.
            vocab_size=256,     d_tok=256,    n_heads_tok=4,  n_layers_tok=2,
            seq_len=128,        d_latent=64,  n_latents=16,   n_heads_lat=4,
            n_layers_lat=1,     world_rnn_hidden=128, world_rollout_steps=8,
            world_graph_max_nodes=64, world_graph_max_edges=192, world_graph_max_transitions=8,
            mem_heads=4,        mem_cache_size=256,  mem_update_steps=4,
            sym_vocab=64,       sym_embed_dim=32,    max_proof_depth=3,
            n_proof_cands=8,    ltm_max_rules=256,   sym_max_facts=32,
            abduct_candidates=8, n_heads=4, n_layers=2, d_model=256,
            continuous_cycle_trace_candidates=2,
            # NET: small vocabulary for fast CPU/GPU testing
            net_enabled=True,   net_byte_layers=1,   net_dec_layers=1,
            net_init_vocab=32,  net_max_vocab=512,   net_tau=0.85,
            net_ema_decay=0.95, eta_tok=0.1,         lambda_voc=1e-4,
            net_warmup_steps=80,
            lambda_enc_div=1.5,   # FIX Bug1: 0.30→1.5
            lambda_soft_H=2.0,    # FIX Bug2: 0.5→2.0
            net_tau_schedule=True, net_tau_min=0.70,
            vem_tau=0.3,        delta_vem=1e-3,      eta_utility=0.1,
            lambda_semantic=0.01, rule_consolidate_every=50,
            # EMC: adaptive reasoning controller
            emc_enabled=True,   emc_max_steps=3,     emc_gamma=0.95,
            emc_entropy_beta=0.01, emc_lambda_time=0.05, emc_lambda_gap=0.05,
            emc_eta_int=0.1,    emc_c_recall=0.01,   emc_c_fc=0.05,
            emc_c_abduce=0.10,  omega_meta=0.05,
            # EMC extension
            emc_use_gae=True,   emc_gae_lambda=0.95, emc_lambda_mdl=0.01,
            emc_use_action_hist=True,
            # OSF: Synthesis Framework (small demo setup)
            osf_enabled=True,   osf_d_intent=32,     osf_n_goals=16,
            osf_d_plan=32,      osf_n_operators=16,  osf_template_len=4,
            osf_max_plan_depth=3, osf_beam_width=2,
            osf_lambda_plan=0.10, osf_lambda_sim=0.05, osf_lambda_refl=0.05,
            osf_lambda_meta=0.05, osf_lambda_total=0.3,
            osf_use_simulation=True, osf_use_reflection=True, osf_use_meta=True,
        )

    @classmethod
    def strong(cls) -> "OMENScaleConfig":
        """
        ~80-120M-parameter configuration for a single modern GPU (RTX 3080/4080/A100).

        Key differences from demo:
          - vocab_size=256 (byte mode) -> bidirectional attention + segment pooling
          - d_tok=512, n_layers=4, seq_len=512 -> much larger capacity
          - net_max_vocab=4096 -> room for a richer symbolic vocabulary
          - net_tau_schedule=True -> adaptive threshold (self-balancing)
          - lambda_enc_div=1.5 -> active anti-collapse encoder signal (FIX Bug1: 0.3->1.5)

        Expected results relative to demo:
          - Used/Vocab > 60% (instead of ~18% in demo)
          - MeanSim < 0.70 (instead of 0.99)
          - Entropy > 6 bits (instead of ~4)
          - PPL < 2.5 on Python code (instead of ~3.6)
        """
        return cls(
            vocab_size=256,      d_tok=512,     n_heads_tok=8,  n_layers_tok=4,
            seq_len=512,         d_latent=128,  n_latents=32,   n_heads_lat=8,
            n_layers_lat=2,      world_rnn_hidden=256,
            mem_heads=8,         mem_cache_size=512,  mem_update_steps=4,
            sym_vocab=128,       sym_embed_dim=64,    max_proof_depth=4,
            n_proof_cands=16,    ltm_max_rules=512,   sym_max_facts=64,
            abduct_candidates=8, n_heads=8,    n_layers=4,     d_model=512,
            # NET: full byte-level setup, large vocabulary
            net_enabled=True,    net_byte_layers=2,   net_dec_layers=2,
            net_init_vocab=64,   net_max_vocab=4096,  net_tau=0.85,
            net_ema_decay=0.95,  eta_tok=0.1,         lambda_voc=1e-4,
            net_warmup_steps=300,
            lambda_enc_div=1.5,           # FIX Bug1: 0.30→1.5
            lambda_soft_H=2.0,            # FIX Bug2: 0.5→2.0
            net_tau_schedule=True, net_tau_min=0.70,
            vem_tau=0.3,         delta_vem=1e-3,      eta_utility=0.1,
            lambda_semantic=0.01, rule_consolidate_every=100,
            dropout=0.1, sparsity_lambda=5e-4,
            # EMC
            emc_enabled=True,   emc_max_steps=5,     emc_gamma=0.95,
            emc_entropy_beta=0.01, emc_lambda_time=0.05, emc_lambda_gap=0.05,
            emc_eta_int=0.1,    emc_c_recall=0.01,   emc_c_fc=0.05,
            emc_c_abduce=0.10,  omega_meta=0.05,
            # EMC extension
            emc_use_gae=True,   emc_gae_lambda=0.95, emc_lambda_mdl=0.01,
            emc_use_action_hist=True,
            # OSF
            osf_enabled=True,   osf_d_intent=64,     osf_n_goals=32,
            osf_d_plan=64,      osf_n_operators=32,  osf_template_len=8,
            osf_max_plan_depth=4, osf_beam_width=2,
            osf_lambda_plan=0.10, osf_lambda_sim=0.05, osf_lambda_refl=0.05,
            osf_lambda_meta=0.05, osf_lambda_total=0.3,
        )

    @classmethod
    def mid(cls) -> "OMENScaleConfig":
        """~1B-parameter configuration for a single A100."""
        return cls(
            vocab_size=256,    d_tok=1_024, n_heads_tok=16, n_layers_tok=16,
            seq_len=2_048,     d_latent=256, n_latents=64,  n_heads_lat=8,
            n_layers_lat=2,    world_rnn_hidden=512,
            mem_heads=16,      mem_cache_size=1_024, mem_update_steps=8,
            n_heads=16, n_layers=16, d_model=1_024,
            # NET: full-size setup for serious training
            net_enabled=True,  net_byte_layers=2,    net_dec_layers=2,
            net_init_vocab=512, net_max_vocab=8_192, net_tau=0.85,
            net_ema_decay=0.95, eta_tok=0.1,          lambda_voc=1e-4,
            net_warmup_steps=500,
            lambda_enc_div=1.5,           # FIX Bug1: 0.30→1.5
            lambda_soft_H=2.0,            # FIX Bug2: 0.5→2.0
            net_tau_schedule=True, net_tau_min=0.70,
        )

    @classmethod
    def full(cls) -> "OMENScaleConfig":
        """Full scale (>=4x A100 80 GB, FSDP)."""
        return cls()

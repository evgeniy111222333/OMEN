"""
omen_scale_config.py — Конфігурація OMEN-Scale
================================================
Три рівні абстракції:
  Fine     : d_tok  = 1024, V = 50k, seq_len = 4096
  Coarse   : d_latent = 256, n_latents = 64  (Perceiver Resampler)
  Symbolic : ∂-Prolog KnowledgeBase (розмір не обмежений)

MDL-принцип:
  min_{θ,Γ} { Complexity(θ) + Complexity(Γ) + E_World[Surprise(Data|θ,Γ)] }
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class OMENScaleConfig:
    # ─── Рівень 1: Token-level (Fine) ─────────────────────────────────────────
    vocab_size:      int   = 256      # NET працює на сирих байтах UTF-8 [0..255]
    d_tok:           int   = 1_024    # Розмірність токен-рівня  (≥1024)
    n_heads_tok:     int   = 16       # Головки уваги токен-рівня
    n_layers_tok:    int   = 12       # Шарів трансформера токен-рівня
    seq_len:         int   = 4_096    # Контекст  (≥4096)

    # ─── Рівень 2: Concept-level (Coarse) — Perceiver Resampler ───────────────
    d_latent:        int   = 256      # Концепт-простір (живе WorldRNN та M-Core)
    n_latents:       int   = 64       # Кількість Perceiver latent queries
    n_heads_lat:     int   = 8        # Головки Perceiver cross-attention
    n_layers_lat:    int   = 2        # Шарів Perceiver self-attention

    # ─── WorldRNN (концепт-рівень) ─────────────────────────────────────────────
    world_rnn_hidden: int  = 512
    world_teacher_forcing_start: float = 0.35
    world_teacher_forcing_end:   float = 0.05
    world_teacher_forcing_steps: int   = 2_000

    # ─── M-Core (async updates) ───────────────────────────────────────────────
    mem_heads:        int   = 16
    mem_cache_size:   int   = 2_048
    mem_write_tau:    float = 0.3
    mem_update_steps: int   = 8      # оновлення кожні N кроків (не на кожному)
    mem_decay:        float = 0.995

    # ─── ∂-Prolog (Symbolic) ──────────────────────────────────────────────────
    sym_vocab:        int   = 128     # розмір символьного словника
    sym_embed_dim:    int   = 64      # розмірність символьних ембеддингів
    max_proof_depth:  int   = 5       # глибина пошуку доведення
    n_proof_cands:    int   = 16      # кандидатів для абдукції
    ltm_max_rules:    int   = 1_024   # макс. правил у KnowledgeBase
    sym_max_facts:    int   = 64      # макс. фактів у WorkingMemory
    abduct_candidates: int  = 8       # кандидати абдуктивного движка
    sym_gnn_layers:   int   = 2       # сумісність з OMENv2

    # ─── Epistemic / Curiosity ────────────────────────────────────────────────
    epistemic_tau:    float = 0.3
    epistemic_exact_grad: bool = False
    n_counterfactual: int   = 2
    symbolic_context_max_facts: int = 96
    symbolic_ast_max_facts: int = 48

    # ─── MDL Loss Coefficients ─────────────────────────────────────────────────
    #   J(θ,Γ,M) = Perplexity + β·L_proof + γ·L_world - α·I(Z;M)
    #             + λ_tok·||z_t||² + λ_conc·||c||² + λ_rule·Σlen(R)
    lambda_tok:  float = 1e-4   # L_scale: стискаємо токен-простір
    lambda_conc: float = 1e-3   # L_scale: стискаємо концепт-простір
    lambda_rule: float = 1e-4   # Complexity(Γ) = λ2·Σ_R len(R)
    alpha:       float = 0.1    # Novelty bonus  −α·I(Z;M)
    beta:        float = 0.05   # Symbolic generalization
    gamma:       float = 0.1    # World consistency
    delta:       float = 1e-3   # WorldRNN complexity
    eta:         float = 0.05   # Memory recall
    lam_sym:     float = 0.005   # LTM regularizer (сумісність v2)

    # ─── Neural Epistemic Tokenizer (NET) ─────────────────────────────────────
    # Замінює GPT-2 BPE (vocab=50257) нейро-символьним компресором.
    # MDL оптимізація:  L_NET = L_vq + L_rec + λ_voc·Σ||e_v||²
    # Повний функціонал: J_total = J_OMEN + η_tok · L_NET
    #
    net_enabled:      bool  = True       # вмикає NET замість BPE
    net_byte_layers:  int   = 2          # шарів ByteContextEncoder (f_θ)
    net_dec_layers:   int   = 2          # шарів ByteDecoder (g_φ)
    net_init_vocab:   int   = 512        # початковий розмір NET-словника
    net_max_vocab:    int   = 8_192      # максимальний розмір NET-словника
    net_tau:          float = 0.85       # поріг cos-подібності (новий токен)
    net_ema_decay:    float = 0.95       # EMA для оновлення кодбуку (0.95 > 0.99: швидша адаптація)
    net_warmup_steps: int   = 150        # кроків заморожування росту словника на старті
                                         # (encoder нестабільний → не додаємо токени)
    eta_tok:          float = 0.1        # вага L_NET у загальному J
    lambda_voc:       float = 1e-4       # MDL регуляризатор словника

    # ─── Adaptive τ scheduling ────────────────────────────────────────────────
    # τ керує порогом «новизни»: якщо cos_sim < τ → новий концепт.
    # Адаптивний режим: τ знижується коли H/H_max < 0.55 (мало активних кодів),
    # підвищується коли H/H_max > 0.65 (словник добре використовується).
    # Результат: система самостійно балансує між стабільністю і ростом словника.
    net_tau_schedule: bool  = True       # вмикає адаптивний τ після warmup
    net_tau_min:      float = 0.70       # нижня межа τ (за замовчуванням net_tau = верхня)

    # ─── Навчання ─────────────────────────────────────────────────────────────
    dropout:          float = 0.1
    sparsity_lambda:  float = 5e-4
    compile_model:    bool  = False   # torch.compile (вмикати на A100/H100)
    use_flash_attn:   bool  = True    # FlashAttention якщо доступна

    # ─── Verification Module (VeM) ────────────────────────────────────────────
    # Фільтрує кандидати AbductionHead до додавання в LTM.
    # U(R) = E[Success(R) − α·Cost(R)]
    # Candidates = {R ~ AbductionHead(z) | U(R) > vem_tau}
    # δ·E_{R~Abduction}[max(0, τ − U(R))] ← штраф за генерацію поганих кандидатів
    vem_tau:          float = 0.3    # поріг корисності (U(R) > vem_tau → приймається)
    delta_vem:        float = 1e-3   # вага VeM-штрафу у загальному J

    # ─── Epistemic Rule Tracker ───────────────────────────────────────────────
    # Кожне правило: proposed → verified / contradicted
    # L_rule = Σ_{R∈LTM} (Complexity(R) − η·Utility(R))
    eta_utility:      float = 0.1    # винагорода за корисні правила (−η·Utility)
    rule_consolidate_every: int = 100  # кроків між Rule Consolidation

    # ─── Semantic Feedback Loop ───────────────────────────────────────────────
    # L_semantic = −E_{(v1,v2)~S-Core}[cos(e_v1, e_v2)·Score(v1, v2)]
    # MDL_total  = MDL_NET − λ_sem·I(Z;Γ)
    lambda_semantic:  float = 0.01   # вага L_semantic
    lambda_enc_div:   float = 1.5    # Encoder diversity anti-collapse
                                     # FIX Bug1: 0.30→1.5 (при 0.30 enc_div grad ~87x слабший за l_rec)
    lambda_soft_H:    float = 2.0    # Диференційована soft-entropy (anti-collapse ключовий сигнал)
                                     # FIX Bug2: 0.5→2.0 (при 0.5 soft_H grad ~4600x слабший за l_rec)
                                     # Soft assignments через temperature=0.5 → H градієнт ненульовий при collapse.

    # --- Saliency Trace language mode -----------------------------------------
    saliency_enabled: bool  = True
    saliency_tau:     float = 0.20
    saliency_top_k:   int   = 4
    saliency_max_facts: int = 512
    saliency_beta_struct: float = 0.05
    saliency_gamma_role:  float = 0.05
    saliency_delta_cons:  float = 0.05
    saliency_eta_rule:    float = 1e-4
    saliency_abduce_every: int  = 5
    saliency_consistency_threshold: float = 0.55

    # ─── Efficient Meta-Controller (EMC) ─────────────────────────────────────
    # EMC замінює фіксований max_proof_depth на адаптивну мета-політику π_meta.
    #
    # Рівняння Беллмана:
    #   V*(s) = max{ U_stop(s), max_{a∈A} [-C(a) + γ·E V*(s')] }
    #   U_stop(s) = R_task(s) + η_int·R_int(s) − λ_gap·GapNorm(s)
    #
    # Навчання Actor-Critic:
    #   L_meta = L_actor + 0.5·L_critic   (додається до J_OMEN з вагою ω_meta)
    #
    # J_OMEN+EMC = J_OMEN
    #            + ω_meta · E_τ[Σ_t (R_task + η_int·R_int − λ_gap·GapNorm − C(a))]
    emc_enabled:       bool  = True    # True → EMC керує прологом; False → стара поведінка
    emc_max_steps:     int   = 5       # максимальна кількість кроків EMC за forward
    emc_gamma:         float = 0.95    # коефіцієнт дисконтування майбутньої нагороди
    emc_entropy_beta:  float = 0.01    # β: вага ентропії (exploration bonus)
    emc_lambda_time:   float = 0.05    # штраф за кожен зайвий крок міркування
    emc_lambda_gap:    float = 0.05    # штраф за GapNorm (незнання)
    emc_eta_int:       float = 0.10    # бонус за нові факти/правила (R_int)
    emc_c_recall:      float = 0.01    # вартість дії RecallMCore
    emc_c_fc:          float = 0.05    # вартість дії ForwardChainStep
    emc_c_abduce:      float = 0.10    # вартість дії Abduce
    omega_meta:        float = 0.05    # ω_meta: вага meta_loss у загальному J
    loss_aux_warmup:   int   = 500     # розігрів auxiliary loss-ів, щоб CE не домінував вічно

    # ─── EMC розширення: GAE + MDL(proof) + History ───────────────────────────
    # GAE (Generalized Advantage Estimation):
    #   A_GAE_t = Σ_{k≥0} (γλ)^k · δ_{t+k}
    #   де δ_t = r_t + γ·V(s_{t+1}) − V(s_t)  (TD error)
    #   При λ→0: чистий TD(0); при λ→1: Monte-Carlo. Компроміс bias/variance.
    emc_use_gae:       bool  = True    # True → GAE замість Монте-Карло returns
    emc_gae_lambda:    float = 0.95    # λ у GAE (0=TD, 1=MC)

    # MDL(proof) компонент:
    #   U_stop(s) -= λ_mdl · MDL(proof)
    #   MDL(proof) = Σ_{R ∈ used_rules} Complexity(R) + depth·c_per_step
    #   Заохочує стислі доведення (MDL-принцип розширений на reasoning)
    emc_lambda_mdl:    float = 0.01    # штраф за складність доведення

    # History encoding:
    #   Стан розширено: s_t = (z, gap_norm, depth, n_facts, n_rules, action_hist)
    #   action_hist — one-hot агрегат попередніх дій (допомагає уникнути cycles)
    emc_use_action_hist: bool = True   # True → кодуємо історію дій у стані

    # ─── Сумісність з OMENv2 ──────────────────────────────────────────────────
    # (поля, що очікує OMENAGILoss/WorldRNN)
    n_heads:         int   = 16
    n_layers:        int   = 12
    d_model:         int   = 1_024
    world_rnn_hidden_v2: int = 512    # alias

    # ════════════════════════════════════════════════════════════════════════
    # ─── OMEN Synthesis Framework (OSF) ──────────────────────────────────────
    # OSF замінює TokenDecoder ієрархічною нейро-символьною генерацією.
    # Чотири рівні: Intent → Plan → Expression → Tokens.
    #
    # J_OSF = λ_plan·L_plan + λ_sim·L_sim + λ_refl·L_refl + λ_meta·L_meta
    # Повний функціонал: J_total = J_OMEN + λ_osf · J_OSF
    #
    osf_enabled:      bool  = True      # True → OSF замість TokenDecoder
    osf_d_intent:     int   = 64        # розмір Intent-простору (H1)
    osf_n_goals:      int   = 32        # кількість абстрактних цілей
    osf_d_plan:       int   = 64        # розмір Plan-простору (H2)
    osf_n_operators:  int   = 32        # бібліотека план-операторів
    osf_template_len: int   = 8         # довжина шаблону оператора (H3)
    osf_max_plan_depth: int = 4         # максимальна глибина плану
    osf_beam_width:   int   = 2         # ширина beam search
    osf_lambda_plan:  float = 0.10      # вага L_plan
    osf_lambda_sim:   float = 0.05      # вага L_sim
    osf_lambda_refl:  float = 0.05      # вага L_refl
    osf_lambda_meta:  float = 0.05      # вага L_meta (стратегія)
    osf_lambda_intent: float = 0.01     # anti-collapse intent
    osf_lambda_total: float = 0.3       # λ_osf: вага J_OSF у J_total
    osf_use_simulation: bool = True     # WorldSimulator
    osf_use_reflection: bool = True     # ReflectionModule
    osf_use_meta:     bool  = True      # SynthesisMetaController
    osf_meta_beta:    float = 0.1       # баланс якість/вартість σ
    osf_gumbel_tau:   float = 1.0       # Gumbel-Softmax temperature

    @classmethod
    def demo(cls) -> "OMENScaleConfig":
        """Конфіг для тестування на будь-якому залізі (CPU/GPU)"""
        return cls(
            # ВАЖЛИВО: vocab_size=256 → true byte mode → bidirectional attention
            # + segment pooling → різноманітні вектори → немає encoder collapse!
            # vocab_size=4096 (legacy) давало MeanSim=0.9935 → 83% dead codes.
            vocab_size=256,     d_tok=256,    n_heads_tok=4,  n_layers_tok=2,
            seq_len=128,        d_latent=64,  n_latents=16,   n_heads_lat=4,
            n_layers_lat=1,     world_rnn_hidden=128,
            mem_heads=4,        mem_cache_size=256,  mem_update_steps=4,
            sym_vocab=64,       sym_embed_dim=32,    max_proof_depth=3,
            n_proof_cands=8,    ltm_max_rules=256,   sym_max_facts=32,
            abduct_candidates=8, n_heads=4, n_layers=2, d_model=256,
            # NET: малий словник для швидкого тестування на CPU/GPU
            net_enabled=True,   net_byte_layers=1,   net_dec_layers=1,
            net_init_vocab=32,  net_max_vocab=512,   net_tau=0.85,
            net_ema_decay=0.95, eta_tok=0.1,         lambda_voc=1e-4,
            net_warmup_steps=80,
            lambda_enc_div=1.5,   # FIX Bug1: 0.30→1.5
            lambda_soft_H=2.0,    # FIX Bug2: 0.5→2.0
            net_tau_schedule=True, net_tau_min=0.70,
            vem_tau=0.3,        delta_vem=1e-3,      eta_utility=0.1,
            lambda_semantic=0.01, rule_consolidate_every=50,
            # EMC: адаптивний контролер міркування
            emc_enabled=True,   emc_max_steps=3,     emc_gamma=0.95,
            emc_entropy_beta=0.01, emc_lambda_time=0.05, emc_lambda_gap=0.05,
            emc_eta_int=0.1,    emc_c_recall=0.01,   emc_c_fc=0.05,
            emc_c_abduce=0.10,  omega_meta=0.05,
            # EMC розширення
            emc_use_gae=True,   emc_gae_lambda=0.95, emc_lambda_mdl=0.01,
            emc_use_action_hist=True,
            # OSF: Synthesis Framework (малий для demo)
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
        ~80–120M-параметрова конфігурація для одного сучасного GPU (RTX 3080/4080/A100).

        Ключові відмінності від demo:
          · vocab_size=256 (байт-режим) — bidirectional attention + segment pooling
          · d_tok=512, n_layers=4, seq_len=512 — значно більша ємність
          · net_max_vocab=4096 — простір для багатого символьного словника
          · net_tau_schedule=True — адаптивний поріг (самобалансування)
          · lambda_enc_div=1.5 — активний анти-колапс encoder (FIX Bug1: 0.3→1.5)

        Очікувані результати (порівняно з demo):
          · Used/Vocab > 60% (замість ~18% у demo)
          · MeanSim < 0.70 (замість 0.99)
          · Entropy > 6 bits (замість ~4)
          · PPL < 2.5 на Python code (замість ~3.6)
        """
        return cls(
            vocab_size=256,      d_tok=512,     n_heads_tok=8,  n_layers_tok=4,
            seq_len=512,         d_latent=128,  n_latents=32,   n_heads_lat=8,
            n_layers_lat=2,      world_rnn_hidden=256,
            mem_heads=8,         mem_cache_size=512,  mem_update_steps=4,
            sym_vocab=128,       sym_embed_dim=64,    max_proof_depth=4,
            n_proof_cands=16,    ltm_max_rules=512,   sym_max_facts=64,
            abduct_candidates=8, n_heads=8,    n_layers=4,     d_model=512,
            # NET: повний байт-рівень, великий словник
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
            # EMC розширення
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
        """~1B-параметрова конфігурація для одного A100"""
        return cls(
            vocab_size=256,    d_tok=1_024, n_heads_tok=16, n_layers_tok=16,
            seq_len=2_048,     d_latent=256, n_latents=64,  n_heads_lat=8,
            n_layers_lat=2,    world_rnn_hidden=512,
            mem_heads=16,      mem_cache_size=1_024, mem_update_steps=8,
            n_heads=16, n_layers=16, d_model=1_024,
            # NET: повний розмір для серйозного тренування
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
        """Повний масштаб (≥4×A100 80 GB, FSDP)"""
        return cls()

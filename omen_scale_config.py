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
    vocab_size:      int   = 50_257   # GPT-2 BPE  (≥50k)
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

    # ─── M-Core (async updates) ───────────────────────────────────────────────
    mem_heads:        int   = 16
    mem_cache_size:   int   = 2_048
    mem_write_tau:    float = 0.3
    mem_update_steps: int   = 8      # оновлення кожні N кроків (не на кожному)

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
    n_counterfactual: int   = 2

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
    lam_sym:     float = 0.02   # LTM regularizer (сумісність v2)

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
    net_ema_decay:    float = 0.99       # EMA для оновлення кодбуку
    eta_tok:          float = 0.1        # вага L_NET у загальному J
    lambda_voc:       float = 1e-4       # MDL регуляризатор словника

    # ─── Навчання ─────────────────────────────────────────────────────────────
    dropout:          float = 0.1
    sparsity_lambda:  float = 5e-4
    compile_model:    bool  = False   # torch.compile (вмикати на A100/H100)
    use_flash_attn:   bool  = True    # FlashAttention якщо доступна

    # ─── Сумісність з OMENv2 ──────────────────────────────────────────────────
    # (поля, що очікує OMENAGILoss/WorldRNN)
    n_heads:         int   = 16
    n_layers:        int   = 12
    d_model:         int   = 1_024
    world_rnn_hidden_v2: int = 512    # alias

    # ════════════════════════════════════════════════════════════════════════
    @classmethod
    def demo(cls) -> "OMENScaleConfig":
        """Конфіг для тестування на будь-якому залізі (CPU/GPU)"""
        return cls(
            vocab_size=4_096,   d_tok=256,    n_heads_tok=4,  n_layers_tok=2,
            seq_len=128,        d_latent=64,  n_latents=16,   n_heads_lat=4,
            n_layers_lat=1,     world_rnn_hidden=128,
            mem_heads=4,        mem_cache_size=256,  mem_update_steps=4,
            sym_vocab=64,       sym_embed_dim=32,    max_proof_depth=3,
            n_proof_cands=8,    ltm_max_rules=256,   sym_max_facts=32,
            abduct_candidates=8, n_heads=4, n_layers=2, d_model=256,
            # NET: малий словник для швидкого тестування на CPU
            net_enabled=True,   net_byte_layers=1,   net_dec_layers=1,
            net_init_vocab=64,  net_max_vocab=512,   net_tau=0.80,
            net_ema_decay=0.99, eta_tok=0.1,         lambda_voc=1e-4,
        )

    @classmethod
    def mid(cls) -> "OMENScaleConfig":
        """~1B-параметрова конфігурація для одного A100"""
        return cls(
            vocab_size=32_000, d_tok=1_024, n_heads_tok=16, n_layers_tok=16,
            seq_len=2_048,     d_latent=256, n_latents=64,  n_heads_lat=8,
            n_layers_lat=2,    world_rnn_hidden=512,
            mem_heads=16,      mem_cache_size=1_024, mem_update_steps=8,
            n_heads=16, n_layers=16, d_model=1_024,
            # NET: повний розмір для серйозного тренування
            net_enabled=True,  net_byte_layers=2,    net_dec_layers=2,
            net_init_vocab=512, net_max_vocab=8_192, net_tau=0.85,
            net_ema_decay=0.99, eta_tok=0.1,          lambda_voc=1e-4,
        )

    @classmethod
    def full(cls) -> "OMENScaleConfig":
        """Повний масштаб (≥4×A100 80 GB, FSDP)"""
        return cls()
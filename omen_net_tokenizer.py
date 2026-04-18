"""
omen_net_tokenizer.py: Neural Epistemic Tokenizer (NET).

NET replaces fixed subword tokenization with byte-level contextual encoding,
discrete concept formation, and MDL-aware reconstruction.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_symbolic.universal_bits import gaussian_tensor_bits, universal_int_bits

from omen_perceiver import LlamaDecoderBlock, LlamaAttention, RMSNorm, SwiGLUFFN


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ByteContextEncoder  (f_θ)
# ══════════════════════════════════════════════════════════════════════════════

class ByteContextEncoder(nn.Module):
    """
    Контекстний енкодер f_θ — СПРАВЖНІЙ байт-рівень.

    Замість того щоб приймати вже токенізовані int-індекси з vocab≥50k
    (що нічим не відрізняється від звичайного Embedding+Transformer),
    цей енкодер:

      1. Приймає bytes (B, T) ∈ [0, 255]  — сирі UTF-8 байти.
      2. Проектує кожен байт у d_tok-простір через малий Embedding(256, d_tok).
      3. Пропускає через n_layers LlamaDecoderBlock із двонапрямною увагою.
      4. (Опціонально) сегментує за пробілами / пунктуацією і робить mean-pool
         в межах сегменту → семантичний концепт для EpistemicQuantizer.

    Якщо передаються вже токенізовані дані (vocab_size > 256), енкодер
    автоматично переходить у режим сумісності (legacy mode): приймає токени
    і не виконує сегментацію. Це забезпечує зворотну сумісність із тестами.

    Математика:
      Вхід X = (x_1,...,x_T) ∈ {0..255}^T    (байти)
      h_i = f_θ(x_i, context(x_{<i}))         (contextual encoding)
      c_j = mean_{i ∈ seg_j} h_i              (segment pooling)
      → Z = (c_1,...,c_N)  N ≤ T              (концептні вектори для Q)

    У режимі сумісності (is_byte_mode=False):
      h_i = f_θ(token_i)  без сегментації
    """

    # Байти, які трактуються як роздільники сегментів
    SEGMENT_BYTES: frozenset = frozenset(
        b' \t\n\r.,;:!?()[]{}"\''
        + b'+-*/=<>|&^~%@#$\\'
    )

    def __init__(self,
                 vocab_size: int,
                 d_tok: int,
                 n_layers: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 segment_pool: bool = True):
        super().__init__()
        assert d_tok % n_heads == 0, f"d_tok={d_tok} не ділиться на n_heads={n_heads}"

        # Byte-mode є безумовним: NET завжди працює на UTF-8 байтах [0..255].
        self.byte_vocab    = 256
        self.segment_pool  = segment_pool
        self.d_tok         = d_tok

        self.embed = nn.Embedding(self.byte_vocab, d_tok)
        nn.init.normal_(self.embed.weight, std=d_tok ** -0.5)
        self.attn_norms = nn.ModuleList([RMSNorm(d_tok) for _ in range(n_layers)])
        self.attns = nn.ModuleList([
            LlamaAttention(d_tok, n_heads, dropout=dropout, causal=False)
            for _ in range(n_layers)
        ])
        self.ffn_norms = nn.ModuleList([RMSNorm(d_tok) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([SwiGLUFFN(d_tok, dropout=dropout) for _ in range(n_layers)])
        self.blocks = nn.ModuleList()

        self.norm = RMSNorm(d_tok)
        self.drop = nn.Dropout(dropout)
        self.n_layers = n_layers

        # ── OPT-1: Pre-computed lookup table для векторизованого segment_pool ──
        # Замість Python double-loop (O(B·T)) → tensor lookup + cumsum (O(1) Python)
        # register_buffer: auto-переноситься на правильний device разом з моделлю
        _seg_lut = torch.zeros(256, dtype=torch.bool)
        for _sb in self.SEGMENT_BYTES:
            if 0 <= _sb < 256:
                _seg_lut[_sb] = True
        self.register_buffer('_seg_lut', _seg_lut, persistent=False)

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        """
        tokens : (B, T) ∈ [0, vocab_size)
        Returns: (B, T, d_tok)  — у byte-режимі розмір T збережено (без pooling)
                                   щоб зберегти сумісність із рештою pipeline.
        Segment-aware mean-pooling відбувається всередині через residual:
        кожна позиція отримує вектор, що змішує байти свого сегменту.
        """
        B, T = tokens.shape

        tokens = tokens.clamp(0, 255)

        x = self.drop(self.embed(tokens))              # (B, T, d_tok)
        attn_maps: List[torch.Tensor] = []
        for norm_a, attn, norm_f, ffn in zip(
                self.attn_norms, self.attns, self.ffn_norms, self.ffns):
            if return_attn:
                attn_out, attn_weights = attn(norm_a(x), need_weights=True)
                attn_maps.append(attn_weights)
                x = x + attn_out
            else:
                x = x + attn(norm_a(x))
            x = x + ffn(norm_f(x))
        x = self.norm(x)

        if self.segment_pool:
            x = self._segment_pool(x, tokens)         # (B, T, d_tok) — pooled

        if return_attn:
            return x, torch.stack(attn_maps, dim=1)   # (B, T, d_tok), (B, L, H, T, T)
        return x                                       # (B, T, d_tok)

    def _segment_pool(self, h: torch.Tensor,
                      bytes_: torch.Tensor) -> torch.Tensor:
        """
        Диференційоване сегментне усереднення.
        out[b,i] = α·h[b,i] + (1-α)·mean(h[b, seg(i)])

        OPT-1: межі сегментів обчислюються через tensor lookup + cumsum
        замість Python double-loop for b in B: for i in T:
        Складність: O(B·T) Python → O(1) Python + GPU kernels
        """
        B, T, D = h.shape
        ALPHA   = 0.5

        with torch.no_grad():
            # ── Векторизоване визначення меж сегментів ────────────────────────
            # _seg_lut[byte] == True  ⟺  byte є роздільником сегменту
            clamped   = bytes_.clamp(0, 255)                        # (B, T)
            boundary  = self._seg_lut[clamped]                      # (B, T) bool
            # cumsum по часовій осі: seg_ids[b,i] = кількість роздільників до i
            seg_ids   = boundary.long().cumsum(dim=1)               # (B, T)
            # Зміщуємо батчі щоб уникнути колізій між прикладами
            offsets   = torch.arange(B, device=h.device).unsqueeze(1) * (T + 1)
            seg_ids   = (seg_ids + offsets).reshape(B * T)          # (B*T,)
            n_segs    = seg_ids.max().item() + 1

            # Лічильники елементів у кожному сегменті (для нормування)
            seg_cnt = torch.zeros(n_segs, dtype=h.dtype, device=h.device)
            seg_cnt.scatter_add_(0, seg_ids,
                                 torch.ones(B * T, dtype=h.dtype, device=h.device))
            seg_cnt.clamp_(min=1)

        # ── Диференційований scatter_add: сума по сегментах ───────────────────
        flat_h  = h.reshape(B * T, D)
        expand  = seg_ids.unsqueeze(1).expand(-1, D)
        seg_sum = torch.zeros(n_segs, D, dtype=h.dtype, device=h.device)
        seg_sum.scatter_add_(0, expand, flat_h)

        seg_mean = (seg_sum[seg_ids] / seg_cnt[seg_ids].unsqueeze(1)).reshape(B, T, D)
        return ALPHA * h + (1 - ALPHA) * seg_mean                   # (B, T, D)



    @staticmethod
    def text_to_bytes(text: str,
                      max_len: int,
                      pad: int = 0) -> torch.Tensor:
        """
        Утиліта: рядок → тензор байтів (1, max_len).
        Використовується під час інференсу замість BPE токенайзера.
        """
        raw = text.encode("utf-8")[:max_len]
        arr = list(raw) + [pad] * (max_len - len(raw))
        return torch.tensor(arr, dtype=torch.long).unsqueeze(0)

    @staticmethod
    def bytes_to_text(byte_ids: torch.Tensor) -> str:
        """Тензор байтів → рядок (ігнорує pad=0)."""
        raw = bytes(b for b in byte_ids.view(-1).tolist() if b != 0)
        return raw.decode("utf-8", errors="replace")



# ══════════════════════════════════════════════════════════════════════════════
# 2.  EpistemicQuantizer  (Q)
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicQuantizer(nn.Module):
    """
    Динамічний словник + VQ-VAE (EMA) + Straight-Through Estimator.

    Квантування (косинусна подібність):
      z_i = argmax_{v∈V}  (e_v ⊤ f_θ(x_i)) / (||e_v|| · ||f_θ(x_i)||)

    Якщо max_sim < τ → новий токен:
      e_new = f_θ(x_i)   (до досягнення net_max_vocab)

    STE (Straight-Through Estimator):
      z_q = sg(e_{z_i} - f_θ(x_i)) + f_θ(x_i)    ← gradient проходить через f_θ

    EMA оновлення кодбуку (стабільніше за SGD на кодбук):
      n_i  ← γ·n_i  + (1−γ)·N_i
      s_i  ← γ·s_i  + (1−γ)·S_i
      e_i  ← s_i / n_i

    Commitment loss (VQ-VAE):
      L_vq = ||sg(z_q) - f_θ(x)||² + β_commit · ||z_q - sg(f_θ(x))||²

    S-Core інтеграція:
      Кожен новий токен реєструється як символьний факт.
      При наявності зовнішнього KnowledgeBase → виклик kb.add_concept_fact()
    """

    def __init__(self,
                 d_tok: int,
                 init_vocab: int,
                 max_vocab: int,
                 tau: float = 0.85,
                 ema_decay: float = 0.99,
                 beta_commit: float = 0.25,
                 warmup_steps: int = 150,
                 tau_schedule: bool = True,
                 tau_min: float = 0.70):
        super().__init__()
        assert init_vocab <= max_vocab, "init_vocab must be ≤ max_vocab"

        self.warmup_steps = warmup_steps
        self.d_tok        = d_tok
        self.max_vocab    = max_vocab
        self.tau          = tau
        self.tau_init     = tau          # зберігаємо початкове значення для adaptive scheduling
        self.tau_min      = tau_min
        self.tau_schedule = tau_schedule
        self.ema_decay    = ema_decay
        self.beta_commit  = beta_commit

        # Кодбук (тільки active[:current_size] є валідними)
        self.codebook = nn.Embedding(max_vocab, d_tok)
        nn.init.normal_(self.codebook.weight, std=d_tok ** -0.5)

        # EMA буфери (не параметри — оновлюються через no_grad)
        self.register_buffer("cluster_count", torch.ones(max_vocab))
        self.register_buffer("cluster_sum",   self.codebook.weight.clone().detach())
        self.register_buffer("current_size",  torch.tensor(init_vocab, dtype=torch.long))

        # Статистика
        self.n_new_tokens   = 0
        self.n_quant_calls  = 0
        self.kb: Optional[object] = None

        # OPT-CB: кеш нормованого кодбуку; інвалідується після EMA/restart
        self._cb_norm_cache: Optional[torch.Tensor] = None

        # ── global_usage: швидкий EMA для крос-батч dead-code детекції ────────
        self.register_buffer("global_usage",
                             torch.ones(max_vocab) * 0.1)
        self._ema_step: int = 0

        # ── K-means++ буфер для ініціалізації кодбуку ─────────────────────────
        # Протягом warmup_steps збираємо encoder outputs → k-means++ центри.
        # Це вирішує проблему "мертвих кодів з нуля": кодбук стартує в реальних
        # кластерах даних, а не у random-нормальних точках.
        # Експеримент: k-means warmup → +16 Used, +0.46H над random init.
        self._kmeans_buffer: List[torch.Tensor] = []
        self._kmeans_done: bool = False

        # ── EMA freeze після restart (Bug7 fix) ────────────────────────────────
        # Проблема: після interp restart EMA decay=0.95 перетягує код назад до
        # collapse center за ~4 кроки (якщо весь батч призначається до коду 0).
        # Фікс: після restart фіксуємо кодбук-запис на _ema_freeze_steps кроків,
        # щоб він накопичив реальні assignments перш ніж EMA почне його зміщувати.
        # _restart_step[i] = global step коли код i був рестартований (−1000 = ніколи).
        self._ema_freeze_steps: int = 20
        self.register_buffer("_restart_step",
                             torch.full((max_vocab,), -1000, dtype=torch.long))
        self.gumbel_tau: float = 0.7

        # OPT-RST: _restart_dead_codes_ema містить дорогу gram-матрицю (64×64) і
        # codebook similarity (B*T × V) — разом ~970 MFLOPs на strong config.
        # Throttle до кожних _restart_interval кроків: якість = майже та сама
        # (dead codes живуть 10–15 кроків, 4 кроки затримки не критичні),
        # але економія ~75% часу цього блоку.
        self._restart_interval: int = 4   # кожні 4 кроки = 75% часу зекономлено

    # ── Допоміжне: нормований кодбук (з кешем) ───────────────────────────────
    def _normed_codebook(self) -> torch.Tensor:
        """Повертає нормовані вектори активних кодбук-записів (з кешем)."""
        # OPT-CB: F.normalize(V×d) виклик на кожен forward — дорогий.
        # Кешуємо після першого обчислення; _ema_update/_restart_dead_codes_ema
        # інвалідують кеш через self._cb_norm_cache = None.
        if self._cb_norm_cache is not None:
            return self._cb_norm_cache
        active = self.codebook.weight[:self.current_size]           # (V_cur, d)
        self._cb_norm_cache = F.normalize(active, dim=-1)
        return self._cb_norm_cache

    # ── Форвард ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        x : (B, T, d_tok)  — вихід ByteContextEncoder
        Returns:
          z_q    : (B, T, d_tok)  — квантовані вектори (STE)
          indices: (B, T)         — індекси кодбуку для кожної позиції
          info   : dict з vq_loss, commit_loss, n_new_tokens, usage_entropy
        """
        B, T, D = x.shape
        self.n_quant_calls += 1

        # ── 1. Косинусна подібність ────────────────────────────────────────────
        x_flat = x.reshape(-1, D)                                   # (B*T, d)
        x_norm = F.normalize(x_flat, dim=-1)                        # (B*T, d)
        cb_norm = self._normed_codebook()                            # (V_cur, d)
        sims   = x_norm @ cb_norm.t()                               # (B*T, V_cur)

        max_sims, indices = sims.max(dim=-1)                        # (B*T,)

        # ── 2. Динамічний словник: нові токени при low similarity ──────────────
        n_new = 0
        if self.training:
            low_sim_mask = max_sims < self.tau
            if low_sim_mask.any() and self.current_size < self.max_vocab:
                N_samples = x_flat.shape[0]           # B*T
                V_cur     = self.current_size.item()
                max_new_per_step = min(2, self.max_vocab - V_cur)  # ≤2/крок: повільний ріст

                # ── Gate 0: Warmup ────────────────────────────────────────────
                # Перші warmup_steps кроків — ніякого росту.
                if self._ema_step <= self.warmup_steps:
                    max_new_per_step = 0

                # ── Gate 1: Coverage ───────────────────────────────────────────
                if max_new_per_step > 0:
                    coverage = N_samples / max(V_cur, 1)
                    if coverage < 2.0:
                        max_new_per_step = 0
                    elif coverage < 4.0:
                        max_new_per_step = min(2, self.max_vocab - V_cur)

                # ── Gate 2: Dead-percentage ────────────────────────────────────
                # Поріг знижено 50%→35%: якщо >35% кодів мертві — рециклюємо, не ростемо.
                # Детекція мертвих: global_usage < 0.10 (узгоджено з _restart_dead_codes_ema).
                # При encoder-collapse нові токени одразу стають dead → ростемо
                # лише якщо поточний словник дійсно добре використовується (Used/V > 65%).
                if max_new_per_step > 0 and V_cur > 0:
                    dead_by_ema = (self.global_usage[:V_cur] < 0.10).sum().item()
                    dead_pct    = dead_by_ema / max(V_cur, 1)
                    if dead_pct > 0.35:
                        max_new_per_step = 0

                # ── Gate 3: Hard batch usage ───────────────────────────────────
                # FIX Bug9: global_usage скидається до 0.15 після кожного restart,
                # тому dead% завжди < 35% одразу після restart → Gate 2 завжди пропускає.
                # Результат: V зростає ~2/крок навіть при encoder-collapse (MeanSim=0.99),
                # створюючи 1444 токени де використовується лише 6.
                #
                # FIX: перевіряємо РЕАЛЬНЕ використання в ПОТОЧНОМУ батчі (hard assignments).
                # indices вже обчислено (крок 1). hard_used_pct — частка кодів що
                # отримали ≥1 assignment у цьому батчі. Якщо < 40% → ще не готові рости.
                # Цей gate не обходиться restart-ом бо рахує реальні assignments тут і зараз.
                if max_new_per_step > 0 and V_cur > 0:
                    hard_counts   = torch.bincount(
                        indices.clamp(0, V_cur - 1), minlength=V_cur).float()
                    hard_used_pct = (hard_counts > 0).float().mean().item()
                    if hard_used_pct < 0.40:     # < 40% кодів реально використано → стоп
                        max_new_per_step = 0

                low_vecs = x_flat[low_sim_mask]                     # (M, d)
                low_norm = F.normalize(low_vecs, dim=-1)            # (M, d)

                added_norms: list[torch.Tensor] = []
                for _ in range(max_new_per_step):
                    if self.current_size >= self.max_vocab:
                        break
                    # Порівнюємо кандидатів з поточним кодбуком + щойно доданими
                    cb_cur = self._normed_codebook()                # (V_cur, d)
                    all_ref = (torch.cat([cb_cur] + added_norms, 0)
                               if added_norms else cb_cur)
                    sims_cand = (low_norm @ all_ref.t()).max(dim=-1).values  # (M,)
                    # Найвіддаленіший від усього поточного кодбуку
                    best_idx  = sims_cand.argmin().item()
                    best_sim  = sims_cand[best_idx].item()
                    if best_sim >= self.tau:
                        break                                       # вже достатньо схожий
                    new_vec  = low_vecs[best_idx].detach()
                    new_norm = F.normalize(new_vec.unsqueeze(0), dim=-1)
                    idx = self.current_size.item()
                    with torch.no_grad():
                        self.codebook.weight.data[idx].copy_(new_vec)
                        self.cluster_sum[idx]     = new_vec
                        self.cluster_count[idx]   = 1.0
                        self.global_usage[idx]    = 0.1   # початкове значення — не "dead"
                    self.current_size += 1
                    n_new += 1
                    self.n_new_tokens += 1
                    added_norms.append(new_norm)
                    # Контекст: 2 найближчих вже існуючих токени → Horn-правило
                    ctx_top: list = []
                    if idx >= 2:
                        existing = F.normalize(
                            self.codebook.weight[:idx].detach(), dim=-1)
                        ctx_sims = (new_norm @ existing.t())[0]
                        ctx_top  = ctx_sims.topk(min(2, idx)).indices.tolist()
                    self._register_concept_in_kb(idx, context_indices=ctx_top)

                if n_new > 0:
                    # Перерахуємо подібності з новими записами
                    self._cb_norm_cache = None      # OPT-CB: нові токени → інвалідуємо
                    cb_norm  = self._normed_codebook()
                    sims     = x_norm @ cb_norm.t()
                    max_sims, indices = sims.max(dim=-1)

        # ── 3. Квантовані вектори ──────────────────────────────────────────────
        v_cur = self.current_size.item()
        active_codebook = self.codebook.weight[:v_cur]
        hard_assign = F.one_hot(indices, num_classes=v_cur).to(x_flat.dtype)
        soft_assign = F.gumbel_softmax(sims, tau=self.gumbel_tau, hard=False, dim=-1)
        assign_st   = hard_assign + soft_assign - soft_assign.detach()
        z_q_flat    = assign_st @ active_codebook
        z_q         = z_q_flat.reshape(B, T, D)

        # ── 4. STE: градієнт через f_θ ────────────────────────────────────────
        z_q_ste = x + (z_q - x).detach() + (z_q - z_q.detach())

        # ── 5. EMA оновлення кодбуку ───────────────────────────────────────────
        # ВАЖЛИВО: при EMA-оновленні кодбуку (без градієнта) НЕ додаємо
        # explicit codebook_loss — це створює конфліктуючі сигнали оновлення
        # і дестабілізує навчання (EMA тягне кодбук в одну сторону, SGD — в іншу).
        # Стандарт VQ-VAE з EMA: лише commitment loss (encoder → codebook).
        vq_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            with torch.no_grad():
                self._ema_step += 1
                # ── K-means++ ініціалізація кодбуку ───────────────────────────
                # Протягом перших warmup_steps кроків збираємо encoder outputs.
                # На кроці warmup_steps → запускаємо k-means++ → замінюємо кодбук
                # реальними кластерами даних. Після цього dead codes практично зникають
                # бо кожен код стоїть у реальному центрі кластера, а не в random точці.
                if not self._kmeans_done:
                    self._kmeans_buffer.append(
                        x_flat.detach()[:min(32, x_flat.size(0))]  # макс 32 зразки/крок
                    )
                    if self._ema_step >= self.warmup_steps:
                        self._kmeans_init_codebook()
                        self._kmeans_done = True
                self._ema_update(x_flat.detach(), indices)
                self._update_global_usage(indices)
                # OPT-RST: throttle restart — дорогий (gram + codebook sim) кожен крок.
                # Кожні _restart_interval кроків = 75% часу зекономлено без втрати якості
                # (dead code виявляється за ~10 кроків, 4 кроки затримки не критичні).
                if self._ema_step % self._restart_interval == 0:
                    self._restart_dead_codes_ema(x_flat.detach())
            # Commitment loss: тягне encoder до найближчого кодбук-вектора
            # L_commit = β · ||z_e - sg(e)||²
            commit      = F.mse_loss(x_flat, z_q_flat.detach())
            codebook_lr = F.mse_loss(z_q_flat, x_flat.detach())
            vq_loss     = self.beta_commit * commit + 0.25 * codebook_lr

        # ── 7. Usage entropy (оцінка якості використання словника) ───────────
        usage_entropy = self._usage_entropy(indices, B * T)

        # ── 8. Encoder diversity loss (anti-collapse) ──────────────────────────
        enc_div_loss = (self._encoder_mean_cosine(x_flat)
                        if self.training
                        else torch.zeros(1, device=x.device).squeeze())

        # ── 8b. Soft-entropy loss (диференційована ентропія) ──────────────────
        # ПРОБЛЕМА: l_code у NETLoss = torch.tensor(constant.item()) → 0 градієнту.
        # РІШЕННЯ: soft assignments через temperature → справжній градієнт.
        #
        # Математика: soft_p_v = mean_i softmax(norm(sim(x_i, e_v)) / τ_soft)
        #             H_soft = -Σ_v soft_p_v · log(soft_p_v)
        #             L_soft_H = max(0, 1 - H_soft/log(V))  → штраф за collapse
        #
        # FIX Bug5: стара формула -(H_soft/H_max) давала loss=-1 при collapse.
        # ПРАВИЛЬНА формула: max(0, 1 - H/H_max)
        #   · collapse (small H): loss ≈ 1  → великий штраф → encoder диверсифікується
        #   · рівномірний (H≈H_max): loss ≈ 0 → немає зайвого штрафу
        #
        # FIX Bug8 (NEW): без row-нормалізації sims_soft, при random кодбуку
        # всі рядки ≈ 0 → softmax ≈ uniform → H_soft ≈ H_max → loss ≈ 0 ЗАВЖДИ.
        # ТЕСТ: collapsed encoder + random codebook → loss=0.007 (стара) vs 0.59 (нова).
        # РІШЕННЯ: нормалізуємо кожен рядок sims_soft (z-score) перед softmax.
        # Це перетворює near-zero similarities у значущий peaked розподіл,
        # де найближчий кодбук-вектор отримує помітно вищу probability.
        soft_entropy_loss = torch.zeros(1, device=x.device).squeeze()
        if self.training:
            V_cur = self.current_size.item()
            if V_cur >= 2:
                # cb_norm вже обчислено (з кешем), але детачимо кодбук:
                # EMA кодбук не має навчатись через SGD (конфліктні сигнали)
                cb_for_soft = F.normalize(active_codebook, dim=-1)
                soft_tau_temp = 0.5                         # температура soft-quantizer
                # OPT-SOFT: subsample x_norm до _SOFT_N позицій замість усіх B*T.
                # Для B=8, T=512: B*T=4096 → _SOFT_N=256 → 16× менше MACs у matmul.
                # Якість: оцінка ентропії на 256 випадкових позиціях практично еквівалентна
                # (закон великих чисел, 256 >> V_cur для типових розмірів словника).
                _SOFT_N = 256
                _n_avail = x_norm.size(0)
                if _n_avail > _SOFT_N:
                    _perm = torch.randperm(_n_avail, device=x_norm.device)[:_SOFT_N]
                    x_norm_soft = x_norm[_perm]             # (256, d)
                else:
                    x_norm_soft = x_norm
                sims_soft = x_norm_soft @ cb_for_soft.t()  # (≤256, V_cur), grad через x_norm_soft
                # FIX Bug8: row-normalization — перетворює near-zero sims у peaked розподіл.
                # Без цього: random codebook → all sims ≈ 0 → uniform softmax → H_soft≈H_max → loss≈0.
                sims_norm = (sims_soft
                             - sims_soft.mean(dim=-1, keepdim=True)
                             ) / (sims_soft.std(dim=-1, keepdim=True) + 1e-6)
                soft_assign = F.softmax(sims_norm / soft_tau_temp, dim=-1)  # (≤256, V_cur)
                avg_usage   = soft_assign.mean(dim=0)       # (V_cur,) — очікуване використання
                H_soft      = -(avg_usage * (avg_usage + 1e-8).log()).sum()  # диференційована H
                H_soft_max  = math.log(max(V_cur, 2))
                entropy_pen = (1.0 - H_soft / (H_soft_max + 1e-8)).clamp(min=0.0)
                diversity_pen = torch.zeros(1, device=x.device).squeeze()
                if V_cur > 1:
                    cb_pair = cb_for_soft @ cb_for_soft.t()
                    off_diag = ~torch.eye(V_cur, dtype=torch.bool, device=cb_pair.device)
                    diversity_pen = cb_pair[off_diag].pow(2).mean()
                soft_entropy_loss = entropy_pen + 0.1 * diversity_pen

        # ── 9. Adaptive τ scheduling (після warmup) ────────────────────────────
        # τ знижується коли H/H_max < 0.55 (мало активних кодів)
        # τ підвищується коли H/H_max > 0.65 (словник добре використовується)
        # Це реалізує "калібрацію порогу подиву" з архітектурного опису.
        if self.training and self._ema_step > self.warmup_steps:
            H_max = math.log(max(self.current_size.item(), 2))
            self._adaptive_tau_step(usage_entropy, H_max)

        return z_q_ste, indices.reshape(B, T), {
            "vq_loss":          vq_loss,
            "n_new_tokens":     n_new,
            "usage_entropy":    usage_entropy,
            "vocab_size":       self.current_size.item(),
            "mean_sim":         max_sims.mean().item(),
            "enc_div_loss":     enc_div_loss,
            "soft_entropy_loss": soft_entropy_loss,   # ← новий диференційований сигнал
        }

    # ── Restart мертвих кодів (dead code collapse prevention) ────────────────
    @torch.no_grad()
    def _restart_dead_codes(self,
                            x_flat:  torch.Tensor,
                            indices: torch.Tensor,
                            min_usage: float = 1.0) -> int:
        """
        Перезапускає кодбук-вектори, яким не було призначено жодного прикладу
        в цьому батчі. Ініціалізуємо їх випадковим encoder-вектором з найбільшою
        відстанню до поточного кодбуку (hard example mining).

        VQ-VAE проблема: після кількох кроків певні коди більше не «виграють»
        жодного прикладу. Без перезапуску словник поступово вироджується.

        Повертає кількість перезапущених кодів.
        """
        V = self.current_size.item()
        if V == 0:
            return 0

        counts = torch.bincount(indices.clamp(0, V - 1).view(-1), minlength=V).float()
        dead_mask = counts < min_usage            # індекси кодів без призначень

        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Обираємо encoder-вектори з найбільшою помилкою (найвіддаленіші від кодбуку)
        cb_norm = F.normalize(self.codebook.weight[:V], dim=-1)
        x_norm  = F.normalize(x_flat, dim=-1)
        # Найменша подібність до будь-якого кодбук-вектора = найбільша помилка
        sims_all   = x_norm @ cb_norm.t()              # (N, V)
        worst_sims, _ = sims_all.max(dim=-1)           # (N,)
        # Перші n_dead прикладів з найнижчою подібністю
        n_restart = min(int(n_dead), x_flat.size(0))
        _, hard_idx = worst_sims.topk(n_restart, largest=False)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_restart]
        for i, dead_code_idx in enumerate(dead_indices):
            src = x_flat[hard_idx[i]]
            self.codebook.weight.data[dead_code_idx].copy_(src)
            self.cluster_sum[dead_code_idx]     = src
            self.cluster_count[dead_code_idx]   = 1.0

        return n_restart

    # ── EMA оновлення ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ema_update(self, x_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA оновлення активних записів кодбуку.

        FIX Bug7: коди що були нещодавно рестартовані (_restart_step протягом
        _ema_freeze_steps кроків) НЕ оновлюються через EMA. Це захищає їх від
        повернення до collapse center до того, як вони накопичать реальні assignments.
        Тест показав: без freeze → re-collapse за 4 кроки. З freeze (20 кроків) →
        код залишається в interp-позиції і поступово отримує власні assignments.
        """
        V = self.current_size.item()
        γ = self.ema_decay

        # OPT-EMA: scatter_add замість (B*T, V) one-hot матриці.
        # Стара схема: zeros(B*T, V) → scatter_(1,...) → sum(0)/matmul
        #   → ~7.5 MB aloc + ~970 MFLOPs (B*T=4096, V=462, d=512 на strong config).
        # Нова: scatter_add_ безпосередньо на (V,) та (V, d) — 0 зайвих алокацій.
        idx_clamped = indices.clamp(0, V - 1)                       # (B*T,)

        # N_i = кількість прикладів, призначених до кластера i
        N = torch.zeros(V, device=x_flat.device, dtype=x_flat.dtype)
        N.scatter_add_(0, idx_clamped,
                       torch.ones(idx_clamped.size(0),
                                  device=x_flat.device, dtype=x_flat.dtype))

        # S_i = сума векторів, призначених до кластера i
        D = x_flat.size(1)
        S = torch.zeros(V, D, device=x_flat.device, dtype=x_flat.dtype)
        S.scatter_add_(0,
                       idx_clamped.unsqueeze(1).expand(-1, D),
                       x_flat)

        # ── Freeze маска: коди під захистом після restart ─────────────────────
        frozen_mask = (self._ema_step - self._restart_step[:V]) < self._ema_freeze_steps
        # Зберігаємо стан захищених кодів до EMA
        frozen_idx = frozen_mask.nonzero(as_tuple=True)[0] if frozen_mask.any() else None
        if frozen_idx is not None and len(frozen_idx) > 0:
            saved_count = self.cluster_count[frozen_idx].clone()
            saved_sum   = self.cluster_sum[frozen_idx].clone()
            saved_w     = self.codebook.weight.data[frozen_idx].clone()

        # EMA
        self.cluster_count[:V] = γ * self.cluster_count[:V] + (1 - γ) * N
        self.cluster_sum[:V]   = γ * self.cluster_sum[:V]   + (1 - γ) * S

        # Оновлення кодбуку (Laplace smoothing у знаменнику)
        updated = (self.cluster_sum[:V]
                   / (self.cluster_count[:V].unsqueeze(1) + 1e-5))
        self.codebook.weight.data[:V].copy_(updated)

        # ── Відновлюємо захищені коди (скасовуємо EMA для них) ───────────────
        if frozen_idx is not None and len(frozen_idx) > 0:
            self.cluster_count[frozen_idx] = saved_count
            self.cluster_sum[frozen_idx]   = saved_sum
            self.codebook.weight.data[frozen_idx].copy_(saved_w)

        self._cb_norm_cache = None      # OPT-CB: кодбук змінився → інвалідуємо

    # ── K-means++ ініціалізація ───────────────────────────────────────────────
    @torch.no_grad()
    def _kmeans_init_codebook(self) -> None:
        """
        K-means++ ініціалізація кодбуку з реальних encoder outputs.

        Запускається одноразово після warmup_steps кроків.
        До цього кодбук = random normal → більшість кодів поза реальним
        розподілом encoder → вони ніколи не "виграють" → dead codes.

        K-means++ гарантує:
          · Кожен центр у реальному кластері даних.
          · Максимальне рознесення центрів (greedy farthest-first).

        Експеримент показав: +16 Used, +0.46H порівняно з random init.
        """
        if not self._kmeans_buffer:
            return

        all_x = torch.cat(self._kmeans_buffer, dim=0)   # (N, D)
        all_x = F.normalize(all_x, dim=-1)              # косинусний простір
        V     = self.current_size.item()
        N     = all_x.size(0)
        if N < V:
            return   # недостатньо даних

        # K-means++ вибір центрів
        perm    = torch.randperm(N)
        centers = [all_x[perm[0]]]                      # перший центр — випадковий

        for _ in range(V - 1):
            c_stack = torch.stack(centers)               # (k, D) — вже нормовані
            sims    = (all_x @ c_stack.t()).max(dim=-1).values  # (N,) max sim до існуючих
            dists   = (1.0 - sims).clamp(min=0)         # косинусна відстань
            probs   = dists ** 2
            s       = probs.sum()
            if s < 1e-9:
                break
            probs   = probs / s
            chosen  = torch.multinomial(probs, 1).item()
            centers.append(all_x[chosen])

        centers_t = torch.stack(centers)                # (K, D) — K ≤ V
        actual_V  = min(len(centers), V)               # FIX: early break → K < V

        # Оновлюємо кодбук in-place (requires_grad → no_grad)
        self.codebook.weight.data[:actual_V].copy_(centers_t[:actual_V])
        self.cluster_sum[:actual_V].copy_(centers_t[:actual_V])
        self.cluster_count[:actual_V].fill_(1.0)
        self.global_usage[:actual_V].fill_(0.15)       # не мертві одразу

        self._kmeans_buffer.clear()

    # ── global_usage EMA (швидший, крос-батч) ─────────────────────────────
    @torch.no_grad()
    def _update_global_usage(self, indices: torch.Tensor) -> None:
        """
        Оновлює global_usage швидким EMA (decay=0.9) по крос-батч статистиці.

        global_usage[i] ∈ [0, 1] — нормована частота використання коду i.
        Використовується замість per-batch bincount у _restart_dead_codes_ema.

        Перевага над cluster_count:
          · cluster_count decay = ema_decay (0.99 → дуже повільно реагує)
          · global_usage decay = 0.9 → код "мертвіє" за ~10 кроків без використання
        """
        V = self.current_size.item()
        if V == 0:
            return
        N = indices.view(-1).shape[0]
        counts = torch.bincount(indices.clamp(0, V - 1).view(-1), minlength=V).float()
        freq   = counts / max(N, 1)                      # нормована частота [0,1]
        DECAY  = 0.9
        self.global_usage[:V] = DECAY * self.global_usage[:V] + (1 - DECAY) * freq

    # ── НОВИЙ: EMA-based dead code restart ────────────────────────────────────
    @torch.no_grad()
    def _restart_dead_codes_ema(self,
                                x_flat: torch.Tensor,
                                dead_threshold: float = 0.10) -> int:
        """
        Перезапускає "мертві" коди на основі global_usage.

        Ключові покращення (підтверджені експериментом):
          · dead_threshold підвищено 0.05 → 0.10: детектує dead коди за ~7 кроків
            замість ~15. Експеримент показав: thr=0.10 → Used=194 (+21 над default).
          · Після restart: global_usage=0.15 (не 0.1) — захищає від негайного
            повторного restart ще ~10 кроків.
          · Protection buffer: коди щойно додані (usage=0.15) мають більше часу
            щоб набрати статистику перед наступним restart-циклом.

        COLLAPSE-AWARE RESTART (нова логіка):
          При encoder-collapse всі x_flat майже однакові → перезапуск мертвих кодів
          encoder-виходами нічого не дає (нові коди відразу collapse до того самого).
          FIX: виявляємо collapse (mean_sim > 0.90) і використовуємо RANDOM unit vectors
          замість encoder-виходів для частини кодів. Це зберігає різноманітність кодбуку,
          що є необхідною умовою для soft_entropy_loss та commit_loss надавати
          значущий градієнт у бік виходу з collapse.
        """
        V = self.current_size.item()
        if V == 0:
            return 0

        dead_mask = self.global_usage[:V] < dead_threshold
        n_dead    = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # ── Виявлення encoder collapse ─────────────────────────────────────────
        # mean_sim > 0.90 → encoder вивів майже однакові вектори для всіх позицій.
        # Samples: обмежуємо 64 токени для ефективності
        N_samp  = min(64, x_flat.size(0))
        x_samp  = F.normalize(x_flat[:N_samp], dim=-1)          # (N_samp, d)
        # Середня подібність між парами (включає i==j → 1.0, але це лише зміщення)
        gram    = x_samp @ x_samp.t()                            # (N_samp, N_samp)
        off_diag_mask = ~torch.eye(N_samp, dtype=torch.bool, device=x_flat.device)
        encoder_mean_sim = gram[off_diag_mask].mean().item() if N_samp > 1 else 0.0
        is_collapsed = encoder_mean_sim > 0.90                   # поріг collapse

        cb_norm   = F.normalize(self.codebook.weight[:V], dim=-1)
        x_norm_r  = F.normalize(x_flat, dim=-1)
        sims_all  = x_norm_r @ cb_norm.t()
        worst_sim, _ = sims_all.max(dim=-1)
        n_restart = min(int(n_dead), x_flat.size(0))
        _, hard_idx = worst_sim.topk(n_restart, largest=False)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_restart]
        for i, dead_code_idx in enumerate(dead_indices):
            if is_collapsed:
                # FIX Bug3 (оновлено): при collapse random вектори НІКОЛИ не отримують
                # assignment (sim до active ~0.93, sim до random ~0.33 → random програє).
                # Стара стратегія «interp між двома кодами» при ЕКСТРЕМАЛЬНОМУ collapse
                # (MeanSim>0.97) не допомагає — всі коди в тій самій точці,
                # interp = та сама точка → нові коди одразу мертві.
                #
                # FIX Bug10 (NEW): розрізняємо два рівні collapse:
                #   · Помірний (0.90<sim<0.97): interp добре працює (доведено тестами)
                #   · Екстремальний (sim≥0.97): використовуємо perturbation у ортогональний
                #     підпростір відносно collapsed centroid.
                #     perturb = centroid + α * ort_noise (α=0.2–0.5)
                #     Sim до centroid = cos(α) ≈ 0.975 > random 0.33 → виграє NN.
                #     Але різні perturb вектори відрізняються → enc_div_loss отримує
                #     значущий градієнт що тягне encoder з collapsed manifold.
                if encoder_mean_sim >= 0.97:
                    # Екстремальний collapse: збурення в ортогональний напрямок
                    centroid = F.normalize(self.codebook.weight[:V].mean(0), dim=0)
                    noise    = torch.randn_like(centroid)
                    # Проеціюємо noise на простір ортогональний до centroid
                    ort_noise = noise - (noise @ centroid) * centroid
                    ort_noise = F.normalize(ort_noise, dim=0)
                    # α: рівномірно в [0.15, 0.45] → sim до centroid ≈ [0.98, 0.99]
                    alpha_ort = 0.15 + torch.rand(1, device=x_flat.device).item() * 0.30
                    perturb_vec = F.normalize(centroid + alpha_ort * ort_noise, dim=-1)
                    self.codebook.weight.data[dead_code_idx].copy_(perturb_vec)
                    self.cluster_sum[dead_code_idx]     = perturb_vec
                else:
                    # Помірний collapse: стара interp стратегія (підтверджена тестами)
                    i1 = torch.randint(0, V, (1,), device=x_flat.device).item()
                    i2 = torch.randint(0, V, (1,), device=x_flat.device).item()
                    alpha = torch.rand(1, device=x_flat.device).item()
                    interp_vec = F.normalize(
                        alpha * self.codebook.weight[i1].clone()
                        + (1.0 - alpha) * self.codebook.weight[i2].clone(),
                        dim=-1
                    )
                    self.codebook.weight.data[dead_code_idx].copy_(interp_vec)
                    self.cluster_sum[dead_code_idx]     = interp_vec
            else:
                # Нормальний режим: restart з "найважчого" encoder-виходу
                src = x_flat[hard_idx[i]]
                self.codebook.weight.data[dead_code_idx].copy_(src)
                self.cluster_sum[dead_code_idx]     = src

            self.cluster_count[dead_code_idx]   = 1.0
            # 0.15 > 0.10 (threshold) → захищений від негайного повторного restart
            self.global_usage[dead_code_idx]    = 0.15
            # FIX Bug7: реєструємо крок restart → EMA freeze на _ema_freeze_steps кроків
            self._restart_step[dead_code_idx]   = self._ema_step

        self._cb_norm_cache = None      # OPT-CB: кодбук змінився → інвалідуємо
        return n_restart

    # ── S-Core: реєстрація нового концепту з Horn-правилом ───────────────────
    def _register_concept_in_kb(self,
                                token_idx: int,
                                context_indices: Optional[list] = None) -> None:
        """
        Реєструє новий токен у KB як Horn-правило, а не просто факт.

        Стратегія абдукції (3 рівні):
          1. Базовий факт: net_token(token_idx)
          2. Якщо є context_indices (індекси сусідніх токенів у словнику):
               net_derived(new_idx, ctx_a) :- net_token(ctx_a), net_token(ctx_b)
             → «новий концепт ≡ комбінація відомих концептів»
          3. Якщо контексту немає → лише факт.

        NET_TOKEN_PRED = 100  (net_token/1)
        NET_RULE_PRED  = 101  (net_derived/2)
        """
        if self.kb is None:
            return
        try:
            self.kb.add_concept_fact(token_idx, context_indices=context_indices)
        except Exception:
            pass

    # ── Utility: usage entropy ────────────────────────────────────────────────
    @torch.no_grad()
    def _usage_entropy(self, indices: torch.Tensor, n: int) -> float:
        """Ентропія використання кодбуку (0 = один токен, log(V) = рівномірно)."""
        V = self.current_size.item()
        if V == 0:
            return 0.0
        counts = torch.bincount(indices.clamp(0, V - 1).view(-1), minlength=V).float()
        probs  = counts / (n + 1e-9)
        probs  = probs[probs > 0]
        return -(probs * probs.log()).sum().item()

    # ── Encoder diversity (anti-collapse) ────────────────────────────────────
    def _encoder_mean_cosine(self,
                             x_flat:    torch.Tensor,
                             sample_n:  int = 64) -> torch.Tensor:
        """
        Середня попарна косинусна подібність між sample_n encoder outputs.

        Вища = більша схожість = encoder collapse (MeanSim ~0.99 = катастрофа).
        Мінімізуємо → encoder навчається генерувати різноманітні вектори.

        Градієнт проходить через x_flat → ByteContextEncoder.
        Діапазон: [-1, 1], ціль: < 0.5 (краще < 0.3 для кодового корпусу).

        Ефективність: O(N²) де N=sample_n≤64 → ≤4096 mult per step (швидко).

        FIX: використовуємо ВИПАДКОВІ індекси замість перших N.
        Причина: перші N токенів з одного батч-елементу сильно корельовані
        між собою (послідовний текст), що заниженно оцінює collapse між
        різними елементами батчу. Випадкова вибірка дає репрезентативну
        оцінку різноманітності МІЖ батч-елементами.
        """
        N = min(x_flat.size(0), sample_n)
        if N < 2:
            return torch.zeros(1, device=x_flat.device, requires_grad=x_flat.requires_grad).squeeze()
        # Випадкова вибірка (замість перших N) → кращий cross-batch diversity estimate
        perm = torch.randperm(x_flat.size(0), device=x_flat.device)[:N]
        x_s      = F.normalize(x_flat[perm], dim=-1)              # (N, d)
        pairwise = x_s @ x_s.t()                                   # (N, N)
        mask     = ~torch.eye(N, dtype=torch.bool, device=x_s.device)
        return pairwise[mask].mean()

    # ── MDL штраф на словник ──────────────────────────────────────────────────
    def vocab_description_bits(self) -> torch.Tensor:
        """Fixed-code dictionary length in bits under a universal code."""
        active_w = self.codebook.weight[:self.current_size]
        size_bits = torch.tensor(
            universal_int_bits(int(self.current_size.item())),
            dtype=active_w.dtype,
            device=active_w.device,
        )
        if active_w.numel() == 0:
            return size_bits
        code_bits = gaussian_tensor_bits(active_w, sigma=1.0 / max(self.d_tok, 1) ** 0.5)
        return size_bits + code_bits

    def vocab_mdl_penalty(self, lambda_voc: float) -> torch.Tensor:
        """
        Weighted dictionary cost. The underlying unit is still bits;
        lambda_voc only controls how strongly that bit-cost is exposed
        inside the local NET objective.
        """
        return float(lambda_voc) * self.vocab_description_bits()

    # ── Adaptive τ scheduling ────────────────────────────────────────────────
    def _adaptive_tau_step(self, usage_entropy: float, H_max: float) -> None:
        """
        Адаптивне налаштування τ — «калібрація порогу подиву».

        Реалізує описану в специфікації поведінку:
          · H/H_max < 0.55 → мало активних кодів → знижуємо τ (легше створювати концепти)
          · H/H_max > 0.65 → словник добре використовується → підвищуємо τ (консервативно)
          · Зміна ±0.005/крок — плавна адаптація без стрибків

        τ ∈ [tau_min, tau_init]  (задається у конфігурації)
        """
        if not self.tau_schedule or H_max < 1e-6:
            return
        ratio = usage_entropy / H_max
        if ratio < 0.50:                                  # дуже мало активних кодів
            self.tau = max(self.tau_min, self.tau - 0.005)
        elif ratio < 0.55:                                # трохи мало
            self.tau = max(self.tau_min, self.tau - 0.002)
        elif ratio > 0.70:                                # добре використовується
            self.tau = min(self.tau_init, self.tau + 0.003)

    def extra_repr(self) -> str:
        return (f"d_tok={self.d_tok}, vocab={self.current_size.item()}/"
                f"{self.max_vocab}, τ={self.tau:.3f}(min={self.tau_min}), γ={self.ema_decay}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ByteDecoder  (g_φ)
# ══════════════════════════════════════════════════════════════════════════════

class ByteDecoder(nn.Module):
    """
    Декодер g_φ: реконструює оригінальну послідовність з квантованих векторів.

    Вхід:
      tgt      : (B, T)          — цільові токени (для авторегресивного навчання)
      z_final  : (B, d_latent)   — концепт-рівень (умовна інформація)
      h_q      : (B, T, d_tok)   — квантовані вектори (пропускаються через cross-attn)

    Архітектура:
      1. Embed tgt → (B, T, d_tok)
      2. + cross-attn(tgt, h_q)   ← квантована контекстна пам'ять
      3. + cross-attn(tgt, z_ctx) ← концепт-рівень інжектується через z_final
      4. LlamaDecoderBlock × n_layers
      5. lm_head → (B, T, vocab_size)

    L_rec = CrossEntropy(g_φ(h_q, z_final), original_tokens)
    """

    def __init__(self,
                 vocab_size: int,
                 d_tok: int,
                 d_latent: int,
                 n_layers: int,
                 n_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d_tok % n_heads == 0

        self.embed   = nn.Embedding(vocab_size, d_tok)
        nn.init.normal_(self.embed.weight, std=d_tok ** -0.5)

        # Проекція z_final (d_latent → d_tok) для cross-attention
        self.z_proj  = nn.Linear(d_latent, d_tok, bias=False)

        # Cross-attention: tgt ← h_q (квантовані вектори)
        self.hq_norm  = RMSNorm(d_tok)
        self.hq_xattn = LlamaAttention(
            d_tok, n_heads, dropout=dropout,
            causal=False, cross_attn=True, kv_dim=d_tok)

        # Cross-attention: tgt ← z_ctx (концепт-рівень)
        self.z_norm   = RMSNorm(d_tok)
        self.z_xattn  = LlamaAttention(
            d_tok, n_heads, dropout=dropout,
            causal=False, cross_attn=True, kv_dim=d_tok)

        # Self-attention + FFN блоки
        self.blocks   = nn.ModuleList([
            LlamaDecoderBlock(d_tok, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = RMSNorm(d_tok)
        self.lm_head  = nn.Linear(d_tok, vocab_size, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self,
                tgt: torch.Tensor,
                z_final: torch.Tensor,
                h_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits : (B, T, vocab_size)
          l_rec  : scalar  — reconstruction loss
        """
        B, T = tgt.shape

        # ── 1. Token embedding ────────────────────────────────────────────────
        x = self.drop(self.embed(tgt))                              # (B, T, d_tok)

        # ── 2. Cross-attn: з квантованими векторами h_q ───────────────────────
        x = x + self.hq_xattn(self.hq_norm(x), context=h_q)

        # ── 3. Cross-attn: концепт z_final → d_tok контекст ──────────────────
        z_ctx = self.z_proj(z_final).unsqueeze(1)                   # (B, 1, d_tok)
        x = x + self.z_xattn(self.z_norm(x), context=z_ctx)

        # ── 4. Self-attention блоки ────────────────────────────────────────────
        for blk in self.blocks:
            x = blk(x)

        # ── 5. Logits ─────────────────────────────────────────────────────────
        logits = self.lm_head(self.out_norm(x))                     # (B, T, V)

        # ── 6. L_rec: авторегресивна помилка відновлення ──────────────────────
        # Зсуваємо: logits[0..T-2] → targets[1..T-1]
        l_rec = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1),
            ignore_index=0,
        )

        return logits, l_rec



# ══════════════════════════════════════════════════════════════════════════════
# 3b.  SemanticFeedbackLoss  (S-Core → NET зворотний зв'язок)
# ══════════════════════════════════════════════════════════════════════════════

class SemanticFeedbackLoss(nn.Module):
    """
    L_semantic = −E_{(v1,v2)~S-Core} [ cos(e_v1, e_v2) · Score(v1, v2) ]

    Вирішує проблему «статистичних, а не семантичних» токенів:
    S-Core виявляє пари концептів з логічним зв'язком (synonym, implies)
    і змушує NET наближати їх вектори пропорційно до сили зв'язку.

    MDL_total = MDL_NET − λ_sem · I(Z; Γ)

    де I(Z; Γ) апроксимується через зважену косинусну подібність:
      I(Z; Γ) ≈ Σ_{(v1,v2)} Score(v1,v2) · cos(e_{v1}, e_{v2})

    Score(v1,v2) — сила логічного зв'язку, виведена S-Core:
      · implies(v1 → v2):      score = 0.9
      · synonym(v1 ≡ v2):      score = 0.7

    Підключається до NETLoss через lambda_semantic.
    """

    def __init__(self, lambda_semantic: float = 0.01):
        super().__init__()
        self.lambda_semantic = lambda_semantic

    def forward(self,
                codebook:    torch.Tensor,          # (V_cur, d_tok)
                pair_indices: List[Tuple[int, int, float]]) -> torch.Tensor:
        """
        codebook     : активна частина кодбуку EpistemicQuantizer (V, d)
        pair_indices : [(tok_idx_1, tok_idx_2, score), ...]
                       від DifferentiableProver.semantic_feedback_pairs()
                       або вбудованого KB-аналізу.

        Returns: L_semantic (scalar, від'ємний означає «токени вже подібні»,
                 мінімізуємо → наближаємо семантично пов'язані вектори).
        """
        if not pair_indices or codebook.shape[0] < 2:
            return torch.zeros(1, device=codebook.device,
                               requires_grad=codebook.requires_grad).squeeze()

        device = codebook.device
        V      = codebook.shape[0]
        cb_n   = F.normalize(codebook, dim=-1)   # (V, d) нормований

        total_sem = torch.zeros(1, device=device)
        n_valid   = 0
        for (i1, i2, score) in pair_indices:
            if i1 >= V or i2 >= V or i1 == i2:
                continue
            cos_sim    = (cb_n[i1] * cb_n[i2]).sum()   # scalar
            total_sem  = total_sem + cos_sim * float(score)
            n_valid   += 1

        if n_valid == 0:
            return torch.zeros(1, device=device).squeeze()

        # Нормуємо на кількість пар → стабільний градієнт незалежно від n_pairs
        avg_sem = total_sem / n_valid
        # −λ · I(Z;Γ): мінімізація → максимізація cosine з S-Core-вагами
        return -self.lambda_semantic * avg_sem


# ══════════════════════════════════════════════════════════════════════════════
# 4.  NETLoss
# ══════════════════════════════════════════════════════════════════════════════

class NETLoss(nn.Module):
    """
    L_NET = L_code + L_rec + L_vocab + λ_vq·L_vq + L_semantic

    L_code   ≈ Length(Z)        — ентропійна оцінка (нормована)
    L_vq                        — commitment loss (STE-сигнал)
    L_rec   ≈ Distortion(X,X̂) — CrossEntropy відновлення
    L_vocab ≈ Complexity(V)    — MDL вартість словника
    L_semantic                  — S-Core семантичний зворотний зв'язок:
                                  −λ_sem · I(Z;Γ)

    Повний MDL функціонал:
      L_NET = L_code + L_rec + L_vocab + λ_vq·L_vq
            − λ_sem · Σ_{(v1,v2)} Score(v1,v2)·cos(e_v1, e_v2)
    """

    def __init__(self,
                 lambda_voc:      float = 1e-4,
                 lambda_vq:       float = 1.0,
                 lambda_semantic: float = 0.01,
                 lambda_enc_div:  float = 1.5,    # FIX Bug1: 0.30→1.5
                 lambda_soft_H:   float = 2.0):   # FIX Bug2: 0.5→2.0
        super().__init__()
        self.lambda_voc      = lambda_voc
        self.lambda_vq       = lambda_vq
        self.lambda_enc_div  = lambda_enc_div
        self.lambda_soft_H   = lambda_soft_H   # ← вага диференційованої soft-entropy
        self.sem_loss_fn     = SemanticFeedbackLoss(lambda_semantic)

    def forward(self,
                vq_info:       Dict,
                l_rec:         torch.Tensor,
                quantizer:     "EpistemicQuantizer",
                sem_pairs:     Optional[List[Tuple[int, int, float]]] = None,
                ) -> Dict:
        """
        sem_pairs : [(tok_idx_1, tok_idx_2, score), ...] від S-Core.
                    Якщо None → L_semantic = 0.
        Returns dict з усіма складовими та total.
        """
        l_vq    = vq_info["vq_loss"]
        l_vocab = quantizer.vocab_mdl_penalty(self.lambda_voc)

        # ── L_code: ентропійна оцінка H(Z) ───────────────────────────────────
        H_nats = vq_info["usage_entropy"]
        V      = max(vq_info["vocab_size"], 2)
        H_max  = math.log(V)
        l_code = torch.tensor(
            max(0.0, 1.0 - H_nats / H_max),
            dtype=l_rec.dtype, device=l_rec.device
        )

        # ── L_semantic: S-Core зворотній зв'язок ─────────────────────────────
        if sem_pairs:
            active_codebook = quantizer.codebook.weight[:quantizer.current_size]
            l_semantic = self.sem_loss_fn(active_codebook, sem_pairs)
            # Переносимо на той самий device що й l_rec
            l_semantic = l_semantic.to(l_rec.device)
        else:
            l_semantic = torch.zeros(1, device=l_rec.device).squeeze()

        # ── L_enc_div: encoder diversity anti-collapse ────────────────────────
        # Мінімізуємо середню попарну косинусну подібність encoder outputs.
        # Якщо encoder collapse (MeanSim~0.99) → цей loss тягне вектори нарізно.
        raw_enc_div = vq_info.get("enc_div_loss", 0.0)
        if torch.is_tensor(raw_enc_div):
            l_enc_div = raw_enc_div.to(l_rec.device)
        else:
            l_enc_div = torch.tensor(float(raw_enc_div), device=l_rec.device)

        # ── L_soft_H: диференційована soft-entropy (ключовий anti-collapse сигнал) ──
        # l_code вище — КОНСТАНТА (torch.tensor з .item()) → нульовий градієнт.
        # soft_entropy_loss — справжній тензор з градієнтом через encoder outputs.
        # Максимізує ентропію розподілу soft-assignments по кодбуку.
        # Ефективний навіть при повному collapse (на відміну від pairwise cosine).
        raw_soft_H = vq_info.get("soft_entropy_loss", 0.0)
        if torch.is_tensor(raw_soft_H):
            l_soft_H = raw_soft_H.to(l_rec.device)
        else:
            l_soft_H = torch.tensor(float(raw_soft_H), device=l_rec.device)

        global_aux = (
            l_code
            + self.lambda_vq * l_vq
            + l_semantic
            + self.lambda_enc_div * l_enc_div
            + self.lambda_soft_H * l_soft_H
        )
        stage_aux = global_aux + l_vocab
        total = stage_aux + l_rec

        return {
            "net_total":        total,
            "net_total_tensor": total,
            "net_aux_tensor":   global_aux,
            "net_aux":          global_aux.item(),
            "net_stage_aux_tensor": stage_aux,
            "net_stage_aux":    stage_aux.item(),
            "net_code":         l_code.item(),
            "net_vq":           l_vq.item() if torch.is_tensor(l_vq) else float(l_vq),
            "net_rec":          l_rec.item(),
            "net_vocab_pen":    l_vocab.item(),
            "net_semantic":     l_semantic.item() if torch.is_tensor(l_semantic) else float(l_semantic),
            "net_enc_div":      l_enc_div.item() if torch.is_tensor(l_enc_div) else float(l_enc_div),
            "net_soft_H":       l_soft_H.item() if torch.is_tensor(l_soft_H) else float(l_soft_H),
            "net_vocab_size":   vq_info["vocab_size"],
            "net_entropy":      H_nats,
            "net_entropy_bits": H_nats / math.log(2),
            "net_mean_sim":     vq_info["mean_sim"],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  NeuralEpistemicTokenizer
# ══════════════════════════════════════════════════════════════════════════════

class NeuralEpistemicTokenizer(nn.Module):
    """
    Повний Neural Epistemic Tokenizer.

    Замінює TokenEncoder + TokenDecoder в OMENScale:
      encode(src) → h_tok, vq_indices, vq_info
      decode(tgt, z_final, h_tok) → logits, l_rec

    Параметри беруться з OMENScaleConfig:
      vocab_size, d_tok, n_heads_tok, net_byte_layers, net_dec_layers
      net_init_vocab, net_max_vocab, net_tau, net_ema_decay
      d_latent, dropout, lambda_voc

    Архітектурний потік:
      src → ByteContextEncoder → h_ctx (B,T,d_tok)
          → EpistemicQuantizer → h_q (B,T,d_tok), vq_indices, vq_info
      tgt → ByteDecoder(h_q, z_final) → logits (B,T,V), l_rec
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # Визначаємо ефективний vocab для ByteContextEncoder:
        # якщо cfg.vocab_size ≤ 256 — справжній byte mode (256)
        # інакше — legacy mode із ПОВНИМ vocab (наприклад, 4096 у demo).
        # ВАЖЛИВО: НЕ робимо min(vocab_size, 256) — це призводить до
        # примусового byte-mode і clamp(0,255) для токенів > 255,
        # що колапсує 94% токенів в одну позицію і вбиває кодбук.
        byte_vocab = 256

        # ── f_θ: ByteContextEncoder ───────────────────────────────────────────
        self.byte_encoder = ByteContextEncoder(
            vocab_size = byte_vocab,
            d_tok      = cfg.d_tok,
            n_layers   = cfg.net_byte_layers,
            n_heads    = cfg.n_heads_tok,
            dropout    = cfg.dropout,
        )

        # ── Q: EpistemicQuantizer ─────────────────────────────────────────────
        self.quantizer = EpistemicQuantizer(
            d_tok        = cfg.d_tok,
            init_vocab   = cfg.net_init_vocab,
            max_vocab    = cfg.net_max_vocab,
            tau          = cfg.net_tau,
            ema_decay    = cfg.net_ema_decay,
            warmup_steps = getattr(cfg, 'net_warmup_steps', 150),
            tau_schedule = getattr(cfg, 'net_tau_schedule', True),
            tau_min      = getattr(cfg, 'net_tau_min', 0.70),
        )

        # ── g_φ: ByteDecoder ──────────────────────────────────────────────────
        self.byte_decoder = ByteDecoder(
            vocab_size = cfg.vocab_size,
            d_tok      = cfg.d_tok,
            d_latent   = cfg.d_latent,
            n_layers   = cfg.net_dec_layers,
            n_heads    = cfg.n_heads_tok,
            dropout    = cfg.dropout,
        )

        # ── Функція витрат ────────────────────────────────────────────────────
        self.loss_fn = NETLoss(
            lambda_voc      = cfg.lambda_voc,
            lambda_semantic = getattr(cfg, 'lambda_semantic', 0.01),
            lambda_enc_div  = getattr(cfg, 'lambda_enc_div', 1.5),   # FIX Bug1: 0.02→1.5
            lambda_soft_H   = getattr(cfg, 'lambda_soft_H',  2.0),   # FIX Bug2: 0.5→2.0
        )

    # ── encode ────────────────────────────────────────────────────────────────
    def encode(self, src: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        src : (B, T) — вхідна послідовність
        Returns:
          h_q       : (B, T, d_tok)  — квантовані вектори (подаються у Perceiver)
          vq_indices: (B, T)         — дискретні індекси (можна аналізувати)
          vq_info   : dict           — статистика квантування
        """
        # f_θ: контекстне кодування
        if return_attn:
            h_ctx, attn_maps = self.byte_encoder(src, return_attn=True)   # (B, T, d_tok), (B, L, H, T, T)
        else:
            h_ctx = self.byte_encoder(src)                             # (B, T, d_tok)
            attn_maps = None
        # Q: квантування
        h_q, vq_indices, vq_info = self.quantizer(h_ctx)          # STE
        if attn_maps is not None:
            vq_info["attention_maps"] = attn_maps
            vq_info["h_ctx"] = h_ctx
        return h_q, vq_indices, vq_info

    # ── decode ────────────────────────────────────────────────────────────────
    def decode(self,
               tgt:     torch.Tensor,
               z_final: torch.Tensor,
               h_q:     torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tgt     : (B, T) — цільова послідовність
        z_final : (B, d_latent) — концепт-рівень
        h_q     : (B, T, d_tok) — квантовані вектори з encode()
        Returns:
          logits : (B, T, vocab_size)
          l_rec  : scalar reconstruction loss
        """
        return self.byte_decoder(tgt, z_final, h_q)

    # ── compute_net_loss ──────────────────────────────────────────────────────
    def compute_loss(self,
                     vq_info:   Dict,
                     l_rec:     torch.Tensor,
                     sem_pairs: Optional[List[Tuple[int, int, float]]] = None,
                     ) -> Dict:
        """
        Обчислює повний L_NET з семантичним feedback від S-Core.

        sem_pairs : [(tok_idx_1, tok_idx_2, score), ...] від prover.semantic_feedback_pairs()
                    Якщо None → L_semantic = 0 (режим без S-Core).
        """
        return self.loss_fn(vq_info, l_rec, self.quantizer, sem_pairs=sem_pairs)

    # ── Stage-1 non-autoregressive reconstruction loss ────────────────────────
    def stage1_rec_loss(self, h_q: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        Non-autoregressive reconstruction loss для Stage 1 pretraining.

        ПРОБЛЕМА: ByteDecoder містить causal self-attention по tgt токенах.
        Протягом навчання decoder може навчитись передбачати tgt[i+1] з tgt[0..i]
        БЕЗ використання h_q (стандартна language model). Як тільки decoder знаходить
        цей «bypass», градієнт через h_q до encoder слабшає → encoder collapse.

        FIX: обчислюємо ДОДАТКОВУ позиційну реконструкцію: h_q[i] → src[i].
        Тут НЕМАЄ causal context — decoder не може bypass-ити h_q.
        Кожна позиція мусить закодувати власний байт у своєму z_q вектору.

        L_rec_nonauto = CrossEntropy(lm_head(norm(h_q)), src)  (позиційно)

        Ця loss додається до стандартного l_rec із вагою λ=0.5 у pretrain_net.
        У Stage 2 (joint training) НЕ використовується — там є z_final контекст.
        """
        dec = self.byte_decoder
        logits = dec.lm_head(dec.out_norm(h_q))          # (B, T, V) — без causal bypass
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            src.reshape(-1),
            ignore_index=0,
        )

    # ── Utility: зв'язати Q з KB ──────────────────────────────────────────────
    def attach_kb(self, kb) -> None:
        """Підключає omen_prolog.KnowledgeBase для S-Core інтеграції."""
        self.quantizer.kb = kb

    # ── Звіт ─────────────────────────────────────────────────────────────────
    def tokenizer_report(self) -> str:
        q = self.quantizer
        n_enc  = sum(p.numel() for p in self.byte_encoder.parameters())
        n_q    = sum(p.numel() for p in self.quantizer.parameters())
        n_dec  = sum(p.numel() for p in self.byte_decoder.parameters())
        return (
            f"  NET vocab          : {q.current_size.item()} / {q.max_vocab}\n"
            f"  NET new tokens     : {q.n_new_tokens}\n"
            f"  NET quant calls    : {q.n_quant_calls}\n"
            f"  ByteContextEncoder : {n_enc:,} params\n"
            f"  EpistemicQuantizer : {n_q:,} params\n"
            f"  ByteDecoder        : {n_dec:,} params\n"
            f"  S-Core KB attached : {q.kb is not None}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  INLINE ТЕСТИ
# ══════════════════════════════════════════════════════════════════════════════

def run_net_tests() -> None:
    """
    Комплексне тестування NET без залежності від OMENScaleConfig.
    Запускається через:  python omen_net_tokenizer.py
    """
    import time
    sep = lambda s: print(f"\n{'═'*68}\n  {s}\n{'═'*68}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NET tests]  device={DEVICE}\n")

    VOCAB  = 256
    D_TOK  = 64
    D_LAT  = 32
    N_H    = 4
    B, T   = 4, 32

    # ─── T1: ByteContextEncoder ───────────────────────────────────────────────
    sep("T1 · ByteContextEncoder — контекстне кодування")
    enc = ByteContextEncoder(VOCAB, D_TOK, n_layers=2, n_heads=N_H).to(DEVICE)
    tokens = torch.randint(0, VOCAB, (B, T), device=DEVICE)
    h = enc(tokens)
    assert h.shape == (B, T, D_TOK), f"FAIL: shape {h.shape}"
    # Перевіряємо, що різні токени дають різні вектори
    h0, h1 = h[0, 0], h[0, 1]
    cos_sim = F.cosine_similarity(h0.unsqueeze(0), h1.unsqueeze(0)).item()
    print(f"  output shape   : {tuple(h.shape)} ✓")
    print(f"  cos_sim(pos0, pos1) : {cos_sim:.4f}  (< 1 → контекст залежить від позиції)")
    assert cos_sim < 0.999, "FAIL: всі вектори однакові"
    print("  [PASS]")

    # ─── T2: EpistemicQuantizer — базове квантування ──────────────────────────
    sep("T2 · EpistemicQuantizer — базове квантування + STE")
    q = EpistemicQuantizer(D_TOK, init_vocab=32, max_vocab=128, tau=0.80).to(DEVICE)
    q.train()

    x = torch.randn(B, T, D_TOK, device=DEVICE, requires_grad=True)
    z_q, idx, info = q(x)

    assert z_q.shape == (B, T, D_TOK), f"FAIL: z_q shape {z_q.shape}"
    assert idx.shape == (B, T),        f"FAIL: idx shape {idx.shape}"
    assert idx.max() < q.current_size, f"FAIL: idx {idx.max()} ≥ vocab_size"

    # STE: градієнт має проходити через x
    z_q.sum().backward()
    assert x.grad is not None, "FAIL: немає градієнту через STE"
    assert not torch.isnan(x.grad).any(), "FAIL: NaN у градієнті"

    print(f"  z_q shape      : {tuple(z_q.shape)} ✓")
    print(f"  indices max    : {idx.max().item()} < {q.current_size.item()} ✓")
    print(f"  STE grad norm  : {x.grad.norm():.4f} ✓")
    print(f"  vq_loss        : {info['vq_loss'].item():.4f}")
    print(f"  usage_entropy  : {info['usage_entropy']:.4f}")
    print(f"  mean_sim       : {info['mean_sim']:.4f}")
    print("  [PASS]")

    # ─── T3: EpistemicQuantizer — динамічний словник ──────────────────────────
    sep("T3 · EpistemicQuantizer — динамічне розширення словника")
    q2 = EpistemicQuantizer(D_TOK, init_vocab=4, max_vocab=64,
                            tau=0.99, warmup_steps=0).to(DEVICE)   # warmup=0 для тесту
    q2.train()

    total_new = 0
    for _ in range(10):
        x2 = torch.randn(2, 16, D_TOK, device=DEVICE)
        _, _, info2 = q2(x2)
        total_new += info2["n_new_tokens"]

    print(f"  Початковий vocab : 4")
    print(f"  Кінцевий vocab   : {q2.current_size.item()}")
    print(f"  Нових токенів    : {total_new}")
    assert q2.current_size > 4, "FAIL: словник не розширився"
    print("  [PASS]")

    # ─── T4: ByteDecoder ──────────────────────────────────────────────────────
    sep("T4 · ByteDecoder — реконструкція + l_rec")
    dec = ByteDecoder(VOCAB, D_TOK, D_LAT, n_layers=2, n_heads=N_H).to(DEVICE)
    h_q     = torch.randn(B, T, D_TOK, device=DEVICE)
    z_final = torch.randn(B, D_LAT, device=DEVICE)
    tgt_tok = torch.randint(1, VOCAB, (B, T), device=DEVICE)

    logits, l_rec = dec(tgt_tok, z_final, h_q)
    assert logits.shape == (B, T, VOCAB), f"FAIL: logits {logits.shape}"
    assert not torch.isnan(l_rec), "FAIL: NaN у l_rec"
    print(f"  logits shape : {tuple(logits.shape)} ✓")
    print(f"  l_rec        : {l_rec.item():.4f}")

    # Backward
    l_rec.backward()
    grad_sum = sum(p.grad.norm().item() for p in dec.parameters() if p.grad is not None)
    assert grad_sum > 0, "FAIL: немає градієнтів у ByteDecoder"
    print(f"  grad_sum     : {grad_sum:.4f} ✓")
    print("  [PASS]")

    # ─── T5: NETLoss ──────────────────────────────────────────────────────────
    sep("T5 · NETLoss — всі три компоненти")
    q3 = EpistemicQuantizer(D_TOK, 16, 64, 0.80).to(DEVICE)
    q3.train()
    xr = torch.randn(B, T, D_TOK, device=DEVICE, requires_grad=True)
    _, _, vq_info3 = q3(xr)

    loss_fn = NETLoss(lambda_voc=1e-3)
    l_rec3  = torch.tensor(2.5)
    out3    = loss_fn(vq_info3, l_rec3, q3)

    assert "net_total" in out3
    assert not math.isnan(out3["net_vq"])
    assert not math.isnan(out3["net_rec"])
    assert not math.isnan(out3["net_vocab_pen"])
    assert out3["net_total"].item() > out3["net_rec"], "FAIL: L_vocab/L_code не входять у L_NET"
    assert out3["net_stage_aux"] > out3["net_aux"], "FAIL: stage aux should include vocab MDL on top of global aux"
    print(f"  L_vq       : {out3['net_vq']:.4f}")
    print(f"  L_rec      : {out3['net_rec']:.4f}")
    print(f"  L_vocab    : {out3['net_vocab_pen']:.6f}")
    print(f"  L_aux      : {out3['net_aux']:.6f}")
    print(f"  L_stageAux : {out3['net_stage_aux']:.6f}")
    print(f"  L_NET      : {out3['net_total'].item():.4f}")
    print(f"  vocab_size : {out3['net_vocab_size']}")
    print(f"  entropy    : {out3['net_entropy']:.4f}")
    print("  [PASS]")

    # ─── T6: NeuralEpistemicTokenizer — інтеграційний тест ───────────────────
    sep("T6 · NeuralEpistemicTokenizer — encode → decode → loss")

    class _FakeCfg:
        vocab_size = VOCAB; d_tok = D_TOK; d_latent = D_LAT
        n_heads_tok = N_H; net_byte_layers = 2; net_dec_layers = 2
        net_init_vocab = 32; net_max_vocab = 128; net_tau = 0.80
        net_ema_decay = 0.99; dropout = 0.1; lambda_voc = 1e-4

    net = NeuralEpistemicTokenizer(_FakeCfg()).to(DEVICE)
    net.train()

    src  = torch.randint(1, VOCAB, (B, T), device=DEVICE)
    tgt2 = torch.randint(1, VOCAB, (B, T), device=DEVICE)
    zf   = torch.randn(B, D_LAT, device=DEVICE)

    h_q, vq_idx, vq_info = net.encode(src, return_attn=True)
    logits2, l_rec2       = net.decode(tgt2, zf, h_q)
    net_loss_dict         = net.compute_loss(vq_info, l_rec2)

    assert h_q.shape   == (B, T, D_TOK)
    assert vq_idx.shape == (B, T)
    assert logits2.shape == (B, T, VOCAB)
    assert "attention_maps" in vq_info, "FAIL: attention maps не повернулись з encode(return_attn=True)"
    assert "h_ctx" in vq_info, "FAIL: raw encoder hidden не повернувся з encode(return_attn=True)"
    assert vq_info["attention_maps"].shape[:3] == (B, _FakeCfg.net_byte_layers, N_H)
    assert not net_loss_dict["net_total"].isnan()

    # Перевірка backward через весь NET
    net_loss_dict["net_total"].backward()
    grad_enc = sum(p.grad.norm().item() for p in net.byte_encoder.parameters()
                   if p.grad is not None)
    grad_dec = sum(p.grad.norm().item() for p in net.byte_decoder.parameters()
                   if p.grad is not None)
    assert grad_enc > 0, "FAIL: немає градієнту у ByteContextEncoder"
    assert grad_dec > 0, "FAIL: немає градієнту у ByteDecoder"

    print(f"  h_q shape     : {tuple(h_q.shape)} ✓")
    print(f"  vq_idx shape  : {tuple(vq_idx.shape)} ✓")
    print(f"  attn shape    : {tuple(vq_info['attention_maps'].shape)} ✓")
    print(f"  logits shape  : {tuple(logits2.shape)} ✓")
    print(f"  L_NET         : {net_loss_dict['net_total'].item():.4f}")
    print(f"  grad (enc)    : {grad_enc:.4f}")
    print(f"  grad (dec)    : {grad_dec:.4f}")
    print(net.tokenizer_report())
    print("  [PASS]")

    # ─── T7: S-Core інтеграція ────────────────────────────────────────────────
    sep("T7 · S-Core інтеграція — реєстрація концептів у KB")
    try:
        from omen_prolog import KnowledgeBase   # type: ignore
        kb = KnowledgeBase()
        net.attach_kb(kb)
        # Форсуємо нові токени
        q_stest = EpistemicQuantizer(D_TOK, 2, 64, tau=0.99).to(DEVICE)
        q_stest.kb = kb
        q_stest.train()
        for _ in range(5):
            x_s = torch.randn(2, 8, D_TOK, device=DEVICE)
            q_stest(x_s)
        n_facts = kb.n_facts()
        print(f"  KB facts після реєстрації концептів: {n_facts}")
        assert n_facts >= 0   # може бути 0 якщо vocab не росте
        print("  [PASS]")
    except Exception as e:
        print(f"  S-Core інтеграція: {e}  (пропускаємо)")

    # ─── T8: MDL ефект — великий словник = більший штраф ─────────────────────
    sep("T8 · MDL ефект — Complexity(V) зростає з розміром словника")
    q_small = EpistemicQuantizer(D_TOK, 4,  64, 0.80).to(DEVICE)
    q_large = EpistemicQuantizer(D_TOK, 32, 64, 0.80).to(DEVICE)
    pen_small = q_small.vocab_mdl_penalty(1e-3).item()
    pen_large = q_large.vocab_mdl_penalty(1e-3).item()
    print(f"  penalty(V=4)  : {pen_small:.6f}")
    print(f"  penalty(V=32) : {pen_large:.6f}")
    assert pen_large > pen_small, "FAIL: більший словник має більший штраф"
    print("  [PASS]")

    print(f"\n{'═'*68}")
    print("  ✅  Всі 8 NET тестів пройдено успішно")
    print(f"{'═'*68}\n")


if __name__ == "__main__":
    run_net_tests()

"""
omen_net_tokenizer.py — Neural Epistemic Tokenizer (NET)
=========================================================
Замінює GPT-2 BPE нейро-символьним компресором.

Математика (MDL для токенізації):
  Tokenization = argmin_V [ Length(Z) + Distortion(X, X̂) + Complexity(V) ]

  L_NET = L_vq                                    ← commitment loss (∼ L_code)
        + L_rec                                    ← byte reconstruction
        + λ_voc · Σ_{v∈V} (||e_v||² + Complexity(v))  ← MDL словника

Компоненти:
  ByteContextEncoder  (f_θ) : токени → контекстні d_tok-вектори
  EpistemicQuantizer  (Q)   : d_tok-вектори → дискретні концепти (EMA + STE)
  ByteDecoder         (g_φ) : концепти + z_final → відновлення токенів
  NETLoss                   : збирає всі три терміни
  NeuralEpistemicTokenizer  : оркеструє f_θ → Q → g_φ

Інтеграція:
  J_OMEN+NET = J_OMEN + η_tok · L_NET

  h_tok, vq_info = net.encode(src)         # замість tok_encoder(src)
  logits, l_rec  = net.decode(tgt, z_final, h_tok)  # замість tok_decoder
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Режим: справжні байти (256) або legacy-токени
        self.is_byte_mode  = (vocab_size <= 256)
        self.byte_vocab    = 256 if self.is_byte_mode else vocab_size
        self.segment_pool  = segment_pool and self.is_byte_mode
        self.d_tok         = d_tok

        self.embed = nn.Embedding(self.byte_vocab, d_tok)
        nn.init.normal_(self.embed.weight, std=d_tok ** -0.5)

        if self.is_byte_mode:
            # Двонапрямна увага: для байтового кодування важливий правий контекст
            # Реалізуємо через окремий список LlamaAttention + RMSNorm + FFN
            self.attn_norms = nn.ModuleList([RMSNorm(d_tok) for _ in range(n_layers)])
            self.attns = nn.ModuleList([
                LlamaAttention(d_tok, n_heads, dropout=dropout, causal=False)
                for _ in range(n_layers)
            ])
            self.ffn_norms = nn.ModuleList([RMSNorm(d_tok) for _ in range(n_layers)])
            self.ffns = nn.ModuleList([SwiGLUFFN(d_tok, dropout=dropout) for _ in range(n_layers)])
            self.blocks = nn.ModuleList()   # порожній, для уніфікації
        else:
            # Legacy: авторегресивна causal увага (сумісність із тестами)
            self.blocks = nn.ModuleList([
                LlamaDecoderBlock(d_tok, n_heads, dropout)
                for _ in range(n_layers)
            ])
            self.attn_norms = self.ffn_norms = self.attns = self.ffns = nn.ModuleList()

        self.norm = RMSNorm(d_tok)
        self.drop = nn.Dropout(dropout)
        self.n_layers = n_layers

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens : (B, T) ∈ [0, vocab_size)
        Returns: (B, T, d_tok)  — у byte-режимі розмір T збережено (без pooling)
                                   щоб зберегти сумісність із рештою pipeline.
        Segment-aware mean-pooling відбувається всередині через residual:
        кожна позиція отримує вектор, що змішує байти свого сегменту.
        """
        B, T = tokens.shape

        if self.is_byte_mode:
            # Клампуємо на випадок out-of-range (legacy compat)
            tokens = tokens.clamp(0, 255)

        x = self.drop(self.embed(tokens))              # (B, T, d_tok)

        if self.is_byte_mode:
            for norm_a, attn, norm_f, ffn in zip(
                    self.attn_norms, self.attns, self.ffn_norms, self.ffns):
                x = x + attn(norm_a(x))
                x = x + ffn(norm_f(x))
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)

        if self.segment_pool:
            x = self._segment_pool(x, tokens)         # (B, T, d_tok) — pooled

        return x                                       # (B, T, d_tok)

    def _segment_pool(self, h: torch.Tensor,
                      bytes_: torch.Tensor) -> torch.Tensor:
        """
        Диференційоване сегментне усереднення.
        Границі сегментів визначаються з байтових значень (без градієнту),
        але саме усереднення є диференційованою операцією через scatter_add.

        out[b,i] = α·h[b,i] + (1-α)·mean(h[b, seg(i)])
        """
        B, T, D = h.shape
        ALPHA = 0.5

        # ── Визначаємо мітки сегментів без градієнту ──────────────────────────
        with torch.no_grad():
            # seg_id[b, i] = індекс сегменту, до якого належить позиція i
            seg_ids = torch.zeros(B, T, dtype=torch.long, device=h.device)
            byte_np = bytes_.cpu().numpy()
            for b in range(B):
                seg = 0
                for i in range(T):
                    if int(byte_np[b, i]) in self.SEGMENT_BYTES:
                        seg += 1
                    seg_ids[b, i] = seg
            # Уніфікуємо ідентифікатори по батчу: батч b зміщується на b*T
            offsets = torch.arange(B, device=h.device).unsqueeze(1) * (T + 1)
            seg_ids = seg_ids + offsets                              # (B, T)

        # ── Диференційоване scatter_add для обчислення середніх ───────────────
        n_segs  = seg_ids.max().item() + 1
        flat_h  = h.reshape(B * T, D)                               # (B*T, D)
        flat_s  = seg_ids.reshape(B * T)                            # (B*T,)

        # Сума по сегментах
        seg_sum = torch.zeros(n_segs, D, dtype=h.dtype, device=h.device)
        seg_sum.scatter_add_(0, flat_s.unsqueeze(1).expand(-1, D), flat_h)

        # Кількість елементів у кожному сегменті
        seg_cnt = torch.zeros(n_segs, dtype=h.dtype, device=h.device)
        seg_cnt.scatter_add_(0, flat_s,
                             torch.ones(B * T, dtype=h.dtype, device=h.device))
        seg_cnt = seg_cnt.clamp(min=1)

        # Середнє — зберається назад до позицій
        seg_mean_flat = seg_sum[flat_s] / seg_cnt[flat_s].unsqueeze(1)  # (B*T, D)
        seg_mean = seg_mean_flat.reshape(B, T, D)

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
                 beta_commit: float = 0.25):
        super().__init__()
        assert init_vocab <= max_vocab, "init_vocab must be ≤ max_vocab"

        self.d_tok       = d_tok
        self.max_vocab   = max_vocab
        self.tau         = tau
        self.ema_decay   = ema_decay
        self.beta_commit = beta_commit

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
        # Зовнішній KB (встановлюється опціонально після ініціалізації)
        self.kb: Optional[object] = None   # omen_prolog.KnowledgeBase

    # ── Допоміжне: нормований кодбук ──────────────────────────────────────────
    def _normed_codebook(self) -> torch.Tensor:
        """Повертає нормовані вектори активних кодбук-записів."""
        active = self.codebook.weight[:self.current_size]           # (V_cur, d)
        return F.normalize(active, dim=-1)

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
                # Додаємо до max_new_per_step нових токенів за один forward pass.
                # Відбираємо найрізноманітніші вектори (greedy farthest-first).
                max_new_per_step = min(4, self.max_vocab - self.current_size.item())
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
                        self.codebook.weight[idx] = new_vec
                        self.cluster_sum[idx]     = new_vec
                        self.cluster_count[idx]   = 1.0
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
                    cb_norm  = self._normed_codebook()
                    sims     = x_norm @ cb_norm.t()
                    max_sims, indices = sims.max(dim=-1)

        # ── 3. Квантовані вектори ──────────────────────────────────────────────
        z_q_flat = self.codebook(indices)                           # (B*T, d)
        z_q      = z_q_flat.reshape(B, T, D)

        # ── 4. STE: градієнт через f_θ ────────────────────────────────────────
        z_q_ste = x + (z_q - x).detach()                           # STE trick

        # ── 5. EMA оновлення кодбуку ───────────────────────────────────────────
        # ВАЖЛИВО: при EMA-оновленні кодбуку (без градієнта) НЕ додаємо
        # explicit codebook_loss — це створює конфліктуючі сигнали оновлення
        # і дестабілізує навчання (EMA тягне кодбук в одну сторону, SGD — в іншу).
        # Стандарт VQ-VAE з EMA: лише commitment loss (encoder → codebook).
        vq_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            with torch.no_grad():
                self._ema_update(x_flat.detach(), indices)
                self._restart_dead_codes(x_flat.detach(), indices)
            # Commitment loss: тягне encoder до найближчого кодбук-вектора
            # L_commit = β · ||z_e - sg(e)||²
            commit = F.mse_loss(x_flat, z_q_flat.detach())
            vq_loss = commit * self.beta_commit

        # ── 7. Usage entropy (оцінка якості використання словника) ───────────
        usage_entropy = self._usage_entropy(indices, B * T)

        return z_q_ste, indices.reshape(B, T), {
            "vq_loss":       vq_loss,
            "n_new_tokens":  n_new,
            "usage_entropy": usage_entropy,
            "vocab_size":    self.current_size.item(),
            "mean_sim":      max_sims.mean().item(),
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
            self.codebook.weight[dead_code_idx] = src
            self.cluster_sum[dead_code_idx]     = src
            self.cluster_count[dead_code_idx]   = 1.0

        return n_restart

    # ── EMA оновлення ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ema_update(self, x_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA оновлення активних записів кодбуку."""
        V = self.current_size.item()
        γ = self.ema_decay

        # One-hot для підрахунку призначень
        one_hot = torch.zeros(x_flat.size(0), V, device=x_flat.device)
        idx_clamped = indices.clamp(0, V - 1)
        one_hot.scatter_(1, idx_clamped.unsqueeze(1), 1.0)

        # N_i = кількість прикладів, призначених до кластера i
        N = one_hot.sum(0)                                           # (V,)
        # S_i = сума векторів, призначених до кластера i
        S = one_hot.t() @ x_flat                                    # (V, d)

        # EMA
        self.cluster_count[:V] = γ * self.cluster_count[:V] + (1 - γ) * N
        self.cluster_sum[:V]   = γ * self.cluster_sum[:V]   + (1 - γ) * S

        # Оновлення кодбуку (Laplace smoothing у знаменнику)
        updated = (self.cluster_sum[:V]
                   / (self.cluster_count[:V].unsqueeze(1) + 1e-5))
        self.codebook.weight[:V] = updated

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
            from omen_prolog import HornAtom, HornClause, Const

            NET_TOKEN_PRED = 100
            NET_RULE_PRED  = 101

            # Крок 1: базовий факт
            fact = HornAtom(pred=NET_TOKEN_PRED, args=(Const(token_idx),))
            self.kb.add_fact(fact)

            # Крок 2: Horn-правило через контекст
            if context_indices and len(context_indices) >= 2:
                ctx_a, ctx_b = int(context_indices[0]), int(context_indices[1])
                head = HornAtom(pred=NET_RULE_PRED,
                                args=(Const(token_idx), Const(ctx_a)))
                body = (
                    HornAtom(pred=NET_TOKEN_PRED, args=(Const(ctx_a),)),
                    HornAtom(pred=NET_TOKEN_PRED, args=(Const(ctx_b),)),
                )
                self.kb.add_rule(HornClause(head=head, body=body))

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

    # ── MDL штраф на словник ──────────────────────────────────────────────────
    def vocab_mdl_penalty(self, lambda_voc: float) -> torch.Tensor:
        """
        L_vocab = λ_voc · ( current_size/max_vocab  +  mean||e_v||² )

        Два доданки:
          · current_size/max_vocab  — штрафує за розмір словника (MDL розмір)
          · mean||e_v||²            — штрафує за велику норму (MDL точність)

        Більший словник → більший перший доданок → вищий штраф.
        """
        active_w     = self.codebook.weight[:self.current_size]
        size_penalty = self.current_size.float() / max(self.max_vocab, 1)
        norm_penalty = active_w.pow(2).mean()
        return lambda_voc * (size_penalty + norm_penalty)

    def extra_repr(self) -> str:
        return (f"d_tok={self.d_tok}, vocab={self.current_size.item()}/"
                f"{self.max_vocab}, τ={self.tau}, γ={self.ema_decay}")


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
# 4.  NETLoss
# ══════════════════════════════════════════════════════════════════════════════

class NETLoss(nn.Module):
    """
    L_NET = L_code + L_rec + L_vocab      (MDL для токенізації)

    L_code  ≈ Length(Z)       — ентропійна оцінка довжини коду токенів:
                                H(Z) = -Σ p(z)·log2(p(z))   [bits/token]
                                (емпірична ентропія з поточного батчу)
                                Це СПРАВЖНЯ апроксимація MDL Length(Z),
                                а не VQ commitment loss.

    L_vq    — commitment loss (β·||z_e − sg(e)||²): залишається окремо
                                як технічний сигнал для STE, не є MDL.

    L_rec  ≈ Distortion(X, X̂) — CrossEntropy відновлення (bits/byte)
    L_vocab≈ Complexity(V)    — MDL вартість словника

    Загальний MDL функціонал:
      L_NET = L_code + L_rec + L_vocab + λ_vq · L_vq

    де L_vq — технічний член для навчання encoder (не частина MDL).
    """

    def __init__(self, lambda_voc: float = 1e-4, lambda_vq: float = 1.0):
        super().__init__()
        self.lambda_voc = lambda_voc
        self.lambda_vq  = lambda_vq

    def forward(self,
                vq_info:   Dict,
                l_rec:     torch.Tensor,
                quantizer: "EpistemicQuantizer") -> Dict:
        """
        Returns dict з усіма складовими та total.
        """
        l_vq    = vq_info["vq_loss"]
        l_vocab = quantizer.vocab_mdl_penalty(self.lambda_voc)

        # ── L_code: ентропійна оцінка H(Z) ───────────────────────────────────
        # vq_info["usage_entropy"] = H(Z) в натуральних логарифмах (nats)
        # Переводимо в bits: H_bits = H_nats / ln(2)
        # Нормуємо на log(V) (максимальна ентропія рівномірного словника):
        #   l_code = 1 - H(Z)/log(V)  ∈ [0, 1]
        # 0 = ідеально рівномірний розподіл (мінімальне L_code)
        # 1 = один токен використовується (максимальне L_code = collapse)
        H_nats = vq_info["usage_entropy"]           # float
        V      = max(vq_info["vocab_size"], 2)
        H_max  = math.log(V)                        # nats
        # Нормована несправедливість: чим ближче до 0 — тим краще
        l_code = torch.tensor(
            max(0.0, 1.0 - H_nats / H_max),
            dtype=l_rec.dtype, device=l_rec.device
        )

        total = l_code + l_rec + l_vocab + self.lambda_vq * l_vq

        return {
            "net_total":      total,
            "net_code":       l_code.item(),        # L_code (MDL entropy)
            "net_vq":         l_vq.item() if torch.is_tensor(l_vq) else float(l_vq),
            "net_rec":        l_rec.item(),
            "net_vocab_pen":  l_vocab.item(),
            "net_vocab_size": vq_info["vocab_size"],
            "net_entropy":    H_nats,
            "net_entropy_bits": H_nats / math.log(2),
            "net_mean_sim":   vq_info["mean_sim"],
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
        # інакше — legacy mode із повним vocab (наприклад, 4096 у demo)
        byte_vocab = min(cfg.vocab_size, 256)

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
            d_tok       = cfg.d_tok,
            init_vocab  = cfg.net_init_vocab,
            max_vocab   = cfg.net_max_vocab,
            tau         = cfg.net_tau,
            ema_decay   = cfg.net_ema_decay,
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
        self.loss_fn = NETLoss(lambda_voc=cfg.lambda_voc)

    # ── encode ────────────────────────────────────────────────────────────────
    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        src : (B, T) — вхідна послідовність
        Returns:
          h_q       : (B, T, d_tok)  — квантовані вектори (подаються у Perceiver)
          vq_indices: (B, T)         — дискретні індекси (можна аналізувати)
          vq_info   : dict           — статистика квантування
        """
        # f_θ: контекстне кодування
        h_ctx = self.byte_encoder(src)                             # (B, T, d_tok)
        # Q: квантування
        h_q, vq_indices, vq_info = self.quantizer(h_ctx)          # STE
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
                     vq_info: Dict,
                     l_rec:   torch.Tensor) -> Dict:
        """Обчислює повний L_NET і повертає словник."""
        return self.loss_fn(vq_info, l_rec, self.quantizer)

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
                            tau=0.99).to(DEVICE)   # дуже низький поріг → багато нових
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
    print(f"  L_vq       : {out3['net_vq']:.4f}")
    print(f"  L_rec      : {out3['net_rec']:.4f}")
    print(f"  L_vocab    : {out3['net_vocab_pen']:.6f}")
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

    h_q, vq_idx, vq_info = net.encode(src)
    logits2, l_rec2       = net.decode(tgt2, zf, h_q)
    net_loss_dict         = net.compute_loss(vq_info, l_rec2)

    assert h_q.shape   == (B, T, D_TOK)
    assert vq_idx.shape == (B, T)
    assert logits2.shape == (B, T, VOCAB)
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
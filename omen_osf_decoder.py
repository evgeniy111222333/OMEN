"""
omen_osf_decoder.py — Hierarchical Decoder (H3 + H4 рівні OSF)
===============================================================
OMEN Synthesis Framework: Expression Level + Token Level.

Замінює простий TokenDecoder — генерує токени через деревоподібну структуру.

Математика:
  P(T|plan) = Π_{n∈nodes(T)} P(production_n | context(n), plan)

  де production_n — граматичне правило розкриття вузла n.

Два під-рівні:
  H3 (Expression Level):
    PlanSequence → template для кожного оператора
    template_i ∈ R^{template_len × d_tok}

  H4 (Token Level):
    template → конкретні токени (лм-голова)
    Кожен токен кондиціонується на: [h_tok; z_intent; op_emb; template]

Структурний агент вирішує порядок розкриття вузлів (Structural Agent).
На кожному кроці може запитати S-Core (via Prolog KB) допустимі типи.

Інтеграція:
  PlanSequence + h_tok → HierarchicalDecoder → logits (B, T, V)
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_osf_intent  import IntentState
from omen_osf_planner import PlanSequence


# ══════════════════════════════════════════════════════════════════════════════
# 1.  STRUCTURAL AGENT
# ══════════════════════════════════════════════════════════════════════════════

class StructuralAgent(nn.Module):
    """
    Вирішує ПОРЯДОК розкриття вузлів синтаксичного дерева.

    Стан: (op_type_emb, context_h, depth)
    Дії: {top_down, bottom_up, left_to_right, right_to_left}

    Повертає expansion order (ваги для зважування шаблонів).
    """

    N_ORDERS = 4

    def __init__(self, d_plan: int, d_tok: int, dropout: float = 0.1):
        super().__init__()
        self.order_policy = nn.Sequential(
            nn.Linear(d_plan + d_tok, d_tok),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_tok, self.N_ORDERS),
        )

    def forward(
        self,
        op_emb:  torch.Tensor,   # (B, d_plan)
        ctx_h:   torch.Tensor,   # (B, d_tok)
    ) -> torch.Tensor:
        """Returns expansion weights (B, N_ORDERS) — soft ordering."""
        inp    = torch.cat([op_emb, ctx_h], dim=-1)   # (B, d_plan+d_tok)
        logits = self.order_policy(inp)                # (B, N_ORDERS)
        return F.softmax(logits, dim=-1)               # (B, N_ORDERS)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TEMPLATE GENERATOR (H3: Plan → Expression templates)
# ══════════════════════════════════════════════════════════════════════════════

class TemplateGenerator(nn.Module):
    """
    H3: PlanOperator → expression template.

    Кожен оператор розкривається у «шаблон» — послідовність d_tok-векторів
    довжиною template_len. Далі HierarchicalDecoder використовує ці шаблони
    як додаткові ключі/значення для cross-attention у TokenDecoder.

    op_emb (B, d_plan) + z_intent (B, d_intent) → template (B, template_len, d_tok)
    """

    def __init__(
        self,
        d_plan:       int,
        d_intent:     int,
        d_tok:        int,
        template_len: int = 8,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.template_len = template_len

        # Проекція оператора у template-простір
        d_in = d_plan + d_intent
        self.template_proj = nn.Sequential(
            nn.Linear(d_in, d_tok * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_tok * 2, d_tok * template_len),
        )
        self.norm = nn.LayerNorm(d_tok)

        # Позиційні embedding для template
        self.pos_emb = nn.Embedding(template_len, d_tok)

    def forward(
        self,
        op_embs:  torch.Tensor,   # (B, K, d_plan)
        z_intent: torch.Tensor,   # (B, d_intent)
    ) -> torch.Tensor:
        """
        op_embs  : (B, K, d_plan)   — K операторів плану
        z_intent : (B, d_intent)
        Returns  : (B, K·template_len, d_tok) — розгорнуті шаблони
        """
        B, K, d_plan = op_embs.shape
        d_tok        = self.pos_emb.embedding_dim

        # Розширюємо z_intent на K операторів
        z_exp = z_intent.unsqueeze(1).expand(-1, K, -1)   # (B, K, d_intent)

        inp      = torch.cat([op_embs, z_exp], dim=-1)    # (B, K, d_plan+d_intent)
        inp_flat = inp.view(B * K, -1)                    # (B·K, d_in)

        tmpl_flat = self.template_proj(inp_flat)          # (B·K, d_tok·T_len)
        tmpl = tmpl_flat.view(B * K, self.template_len, d_tok)

        # Позиційні ембеддинги
        pos  = self.pos_emb(torch.arange(self.template_len, device=op_embs.device))
        tmpl = self.norm(tmpl + pos)                      # (B·K, T_len, d_tok)

        # Reshape → (B, K·T_len, d_tok)
        tmpl = tmpl.view(B, K * self.template_len, d_tok)
        return tmpl


# ══════════════════════════════════════════════════════════════════════════════
# 3.  HIERARCHICAL DECODER
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalDecoder(nn.Module):
    """
    H3 + H4: PlanSequence → logits (B, T, vocab_size)

    Архітектура:
      1. TemplateGenerator: plan ops → templates (B, K·T_len, d_tok)
      2. StructuralAgent: визначає expansion order (ваги)
      3. Cross-attention: h_tok ← templates (вводимо план у декодер)
      4. Plan-guided projection: z_intent + template → додатковий контекст
      5. LM Head: → logits (B, T, V)

    Виходи:
      logits: (B, T, vocab_size)  — ті самі що TokenDecoder (dropout-in)
      struct_loss: scalar         — structural consistency loss

    Математика P(T|plan) факторизується по вузлах:
      ≈ CrossEntropy(logits, targets)  + λ_struct·L_struct
    де L_struct = ||h_tok − proj(template)||² (консистентність структури)
    """

    def __init__(
        self,
        d_tok:        int,
        d_latent:     int,
        d_plan:       int,
        d_intent:     int,
        vocab_size:   int,
        n_heads:      int    = 4,
        template_len: int    = 8,
        dropout:      float  = 0.1,
        lambda_struct: float = 0.1,
    ):
        super().__init__()
        self.lambda_struct = lambda_struct
        self.d_tok         = d_tok
        self.template_len  = template_len

        # Проекція plan embeddings → d_tok
        self.plan_proj  = nn.Linear(d_plan, d_tok, bias=False)

        # Проекція z_intent → d_tok
        self.intent_proj = nn.Linear(d_intent, d_tok, bias=False)

        # Template Generator (H3)
        self.template_gen = TemplateGenerator(
            d_plan=d_plan, d_intent=d_intent,
            d_tok=d_tok, template_len=template_len, dropout=dropout)

        # Structural Agent
        self.struct_agent = StructuralAgent(d_plan, d_tok, dropout)

        # Cross-attention: h_tok ← templates
        self.cross_attn = nn.MultiheadAttention(
            d_tok, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_tok)

        # Fusion gate: combines h_tok with plan-context
        self.gate = nn.Linear(d_tok * 2, d_tok)

        # Structural consistency: h_tok → template space (для L_struct)
        self.struct_proj = nn.Linear(d_tok, d_tok)
        self.struct_norm = nn.LayerNorm(d_tok)

        # LM Head
        self.out_norm = nn.LayerNorm(d_tok)
        self.lm_head  = nn.Linear(d_tok, vocab_size, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        h_tok:       torch.Tensor,       # (B, T, d_tok)  — TokenEncoder output
        z_intent:    torch.Tensor,       # (B, d_intent)
        plan:        PlanSequence,       # план від SymbolicPlanner
        tgt_tokens:  Optional[torch.Tensor] = None,  # (B, T) — for causal mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits      : (B, T, vocab_size)
          struct_loss : scalar
        """
        B, T, D = h_tok.shape
        device  = h_tok.device

        # ── H3: генеруємо шаблони ─────────────────────────────────────────────
        plan_embs = plan.embeddings.to(device)                   # (K, d_plan)
        K = plan_embs.size(0)

        # Розширюємо на batch
        plan_embs_b = plan_embs.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_plan)

        # Templates: (B, K·T_len, d_tok)
        templates = self.template_gen(plan_embs_b, z_intent)

        # ── Structural Agent ──────────────────────────────────────────────────
        # Ваги для зважування template позицій (soft expansion order).
        # BUG FIX: StructuralAgent ініціалізовано з Linear(d_plan + d_tok, ...),
        # тому op_emb MАЄ бути (B, d_plan) — до plan_proj. Раніше передавався
        # plan_proj(mean) → (B, d_tok), що давало Linear((d_tok+d_tok)=64, ...)
        # але очікувалось (d_plan+d_tok)=48 → RuntimeError.
        op_mean_raw = plan_embs_b.mean(1)                            # (B, d_plan) — raw
        op_mean     = self.plan_proj(op_mean_raw)                    # (B, d_tok)  — for intent
        ctx_mean    = h_tok.mean(1)                                  # (B, d_tok)
        order_w     = self.struct_agent(op_mean_raw, ctx_mean)       # (B, N_ORDERS) ✓
        # Застосовуємо order weights до template (soft permutation)
        # Упрощена версія: scale templates за entropy-зваженим scalar
        order_scale = order_w.mean(-1, keepdim=True).unsqueeze(-1)  # (B,1,1)
        templates_w = templates * (1.0 + order_scale)               # (B, K·T_len, d_tok)

        # ── H4: Cross-attention: h_tok ← templates ───────────────────────────
        h_norm    = self.cross_norm(h_tok)
        h_crossed, _ = self.cross_attn(h_norm, templates_w, templates_w)  # (B,T,d_tok)

        # Intent context: broadcast z_intent → (B, 1, d_tok) + add to h
        intent_ctx = self.intent_proj(z_intent).unsqueeze(1)    # (B, 1, d_tok)
        h_with_intent = h_tok + h_crossed + intent_ctx          # (B, T, d_tok)

        # Gating: combine original h with plan-informed h
        h_fused = self.gate(
            torch.cat([h_tok, h_with_intent], dim=-1))          # (B, T, d_tok)

        # ── Structural Consistency Loss L_struct ──────────────────────────────
        # Мета: h_tok ≈ proj(template_mean) — декодер «розуміє» шаблони
        tmpl_mean  = templates_w.mean(1)                         # (B, d_tok)
        struct_pred = self.struct_proj(h_fused.mean(1))          # (B, d_tok)
        struct_pred = self.struct_norm(struct_pred)
        struct_loss = F.mse_loss(
            struct_pred,
            tmpl_mean.detach())                                  # scalar

        # ── LM Head → logits ─────────────────────────────────────────────────
        h_out  = self.out_norm(h_fused)
        logits = self.lm_head(self.drop(h_out))                  # (B, T, V)

        return logits, struct_loss * self.lambda_struct
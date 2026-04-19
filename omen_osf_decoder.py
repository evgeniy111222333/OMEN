"""
omen_osf_decoder.py — Hierarchical Decoder (OSF H3 + H4 levels)
================================================================
OMEN Synthesis Framework: Expression Level + Token Level.

Replaces the simple TokenDecoder and generates tokens through a tree-like structure.

Mathematics:
  P(T|plan) = Π_{n∈nodes(T)} P(production_n | context(n), plan)

  where `production_n` is the grammar rule that expands node `n`.

Two sub-levels:
  H3 (Expression Level):
    PlanSequence -> template for each operator
    template_i ∈ R^{template_len × d_tok}

  H4 (Token Level):
    template -> concrete tokens (LM head)
    Each token is conditioned on: [h_tok; z_intent; op_emb; template]

The structural agent decides the node expansion order.
At each step it may query S-Core (via Prolog KB) for valid types.

Integration:
  PlanSequence + h_tok -> HierarchicalDecoder -> logits (B, T, V)
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
    Decide the ORDER in which syntax-tree nodes are expanded.

    State: (op_type_emb, context_h, depth)
    Actions: {top_down, bottom_up, left_to_right, right_to_left}

    Returns an expansion order as weights used to mix templates.
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
    H3: PlanOperator -> expression template.

    Each operator expands into a template, i.e. a sequence of d_tok vectors
    with length `template_len`. `HierarchicalDecoder` then uses these templates
    as extra keys/values for cross-attention inside the TokenDecoder.

    op_emb (B, d_plan) + z_intent (B, d_intent) -> template (B, template_len, d_tok)
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

        # Project the operator into template space.
        d_in = d_plan + d_intent
        self.template_proj = nn.Sequential(
            nn.Linear(d_in, d_tok * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_tok * 2, d_tok * template_len),
        )
        self.norm = nn.LayerNorm(d_tok)

        # Positional embeddings for the template.
        self.pos_emb = nn.Embedding(template_len, d_tok)

    def forward(
        self,
        op_embs:  torch.Tensor,   # (B, K, d_plan)
        z_intent: torch.Tensor,   # (B, d_intent)
    ) -> torch.Tensor:
        """
        op_embs  : (B, K, d_plan)   — K plan operators
        z_intent : (B, d_intent)
        Returns  : (B, K·template_len, d_tok) — flattened templates
        """
        B, K, d_plan = op_embs.shape
        d_tok        = self.pos_emb.embedding_dim

        # Broadcast z_intent across K operators.
        z_exp = z_intent.unsqueeze(1).expand(-1, K, -1)   # (B, K, d_intent)

        inp      = torch.cat([op_embs, z_exp], dim=-1)    # (B, K, d_plan+d_intent)
        inp_flat = inp.view(B * K, -1)                    # (B·K, d_in)

        tmpl_flat = self.template_proj(inp_flat)          # (B·K, d_tok·T_len)
        tmpl = tmpl_flat.view(B * K, self.template_len, d_tok)

        # Positional embeddings.
        pos  = self.pos_emb(torch.arange(self.template_len, device=op_embs.device))
        tmpl = self.norm(tmpl + pos)                      # (B·K, T_len, d_tok)

        # Reshape -> (B, K·T_len, d_tok)
        tmpl = tmpl.view(B, K * self.template_len, d_tok)
        return tmpl


# ══════════════════════════════════════════════════════════════════════════════
# 3.  HIERARCHICAL DECODER
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalDecoder(nn.Module):
    """
    H3 + H4: PlanSequence → logits (B, T, vocab_size)

    Architecture:
      1. TemplateGenerator: plan ops -> templates (B, K·T_len, d_tok)
      2. StructuralAgent: determines expansion order (weights)
      3. Cross-attention: h_tok <- templates (inject the plan into the decoder)
      4. Plan-guided projection: z_intent + template -> extra context
      5. LM Head -> logits (B, T, V)

    Outputs:
      logits: (B, T, vocab_size)  — same shape as TokenDecoder
      struct_loss: scalar         — structural consistency loss

    P(T|plan) is factorized over nodes:
      ≈ CrossEntropy(logits, targets)  + λ_struct·L_struct
    where L_struct = ||h_tok − proj(template)||² enforces structural consistency
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

        # Project plan embeddings -> d_tok.
        self.plan_proj  = nn.Linear(d_plan, d_tok, bias=False)

        # Project z_intent -> d_tok.
        self.intent_proj = nn.Linear(d_intent, d_tok, bias=False)

        # Template Generator (H3)
        self.template_gen = TemplateGenerator(
            d_plan=d_plan, d_intent=d_intent,
            d_tok=d_tok, template_len=template_len, dropout=dropout)

        # Structural Agent
        self.struct_agent = StructuralAgent(d_plan, d_tok, dropout)
        self.op_type_to_idx = {
            "define": 0, "call": 1, "assign": 2, "return": 3,
            "branch": 4, "loop": 5, "import": 6, "yield": 7,
        }
        self.n_productions = 4
        self.production_selector = nn.Sequential(
            nn.Linear(d_plan + d_intent + d_tok, d_tok * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_tok * 2, self.n_productions),
        )
        self.production_bank = nn.Parameter(
            torch.randn(len(self.op_type_to_idx), self.n_productions, template_len, d_tok) * (d_tok ** -0.5)
        )
        self.order_bank = nn.Parameter(
            torch.randn(StructuralAgent.N_ORDERS, template_len, d_tok) * (d_tok ** -0.5)
        )

        # Cross-attention: h_tok ← templates
        self.cross_attn = nn.MultiheadAttention(
            d_tok, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_tok)

        # Fusion gate: combines h_tok with plan-context
        self.gate = nn.Linear(d_tok * 2, d_tok)

        # Structural consistency: h_tok -> template space for L_struct.
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
        plan:        PlanSequence,       # plan produced by SymbolicPlanner
        tgt_tokens:  Optional[torch.Tensor] = None,  # (B, T) — for causal mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits      : (B, T, vocab_size)
          struct_loss : scalar
        """
        B, T, D = h_tok.shape
        device  = h_tok.device

        # ── H3: generate templates ───────────────────────────────────────────
        plan_embs = plan.embeddings.to(device)                   # (K, d_plan)
        K = plan_embs.size(0)

        # Broadcast across the batch.
        plan_embs_b = plan_embs.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_plan)

        # Templates: base continuous templates + grammar-aligned discrete productions.
        base_templates = self.template_gen(plan_embs_b, z_intent).view(
            B, K, self.template_len, self.d_tok
        )
        op_type_ids = torch.tensor(
            [self.op_type_to_idx.get(op.op_type, 0) for op in plan.operators],
            device=device,
            dtype=torch.long,
        )
        if op_type_ids.numel() == 0:
            op_type_ids = torch.zeros(K, device=device, dtype=torch.long)

        ctx_mean = h_tok.mean(1)                                  # (B, d_tok)
        ctx_exp = ctx_mean.unsqueeze(1).expand(-1, K, -1)
        z_exp = z_intent.unsqueeze(1).expand(-1, K, -1)
        prod_inp = torch.cat([plan_embs_b, z_exp, ctx_exp], dim=-1)   # (B,K,*)
        prod_logits = self.production_selector(prod_inp.reshape(B * K, -1)).view(
            B, K, self.n_productions
        )
        if self.training:
            prod_w = F.gumbel_softmax(prod_logits, tau=1.0, hard=False, dim=-1)
        else:
            prod_w = F.softmax(prod_logits / 0.35, dim=-1)

        bank = self.production_bank[op_type_ids].unsqueeze(0).expand(B, -1, -1, -1, -1)
        discrete_templates = torch.einsum("bkp,bkptd->bktd", prod_w, bank)

        # ── Structural Agent ─────────────────────────────────────────────────
        # Weights used to mix template positions (soft expansion order).
        # BUG FIX: StructuralAgent is initialized with Linear(d_plan + d_tok, ...),
        # so `op_emb` MUST stay in shape (B, d_plan) before plan_proj.
        # Previously, `plan_proj(mean)` with shape (B, d_tok) was passed in,
        # which built Linear((d_tok+d_tok)=64, ...) while the module expected
        # (d_plan+d_tok)=48, causing a RuntimeError.
        op_mean_raw = plan_embs_b.mean(1)                            # (B, d_plan) — raw
        order_w     = self.struct_agent(op_mean_raw, ctx_mean)       # (B, N_ORDERS)
        order_bias  = torch.einsum("bo,otd->btd", order_w, self.order_bank).unsqueeze(1)
        templates_w = (base_templates + discrete_templates + order_bias).reshape(
            B, K * self.template_len, self.d_tok
        )

        # ── H4: Cross-attention: h_tok <- templates ──────────────────────────
        h_norm    = self.cross_norm(h_tok)
        h_crossed, _ = self.cross_attn(h_norm, templates_w, templates_w)  # (B,T,d_tok)

        # Intent context: broadcast z_intent -> (B, 1, d_tok) and add it to h.
        intent_ctx = self.intent_proj(z_intent).unsqueeze(1)    # (B, 1, d_tok)
        h_with_intent = h_tok + h_crossed + intent_ctx          # (B, T, d_tok)

        # Gating: combine original h with plan-informed h
        gate = torch.sigmoid(self.gate(torch.cat([h_tok, h_with_intent], dim=-1)))
        h_fused = gate * h_with_intent + (1.0 - gate) * h_tok

        # ── Structural Consistency Loss L_struct ─────────────────────────────
        # Goal: h_tok ≈ proj(template_mean), so the decoder "understands" templates.
        tmpl_mean  = templates_w.mean(1)                         # (B, d_tok)
        struct_pred = self.struct_proj(h_fused.mean(1))          # (B, d_tok)
        struct_pred = self.struct_norm(struct_pred)
        prod_entropy = -(prod_w * (prod_w.clamp_min(1e-8)).log()).sum(-1).mean()
        struct_loss = (
            F.mse_loss(struct_pred, tmpl_mean.detach())
            + 0.05 * prod_entropy
        )

        # ── LM Head → logits ─────────────────────────────────────────────────
        h_out  = self.out_norm(h_fused)
        logits = self.lm_head(self.drop(h_out))                  # (B, T, V)

        return logits, struct_loss * self.lambda_struct

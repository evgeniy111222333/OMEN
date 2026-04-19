"""
omen_osf_planner.py — Symbolic Planner (OSF H2 level)
=====================================================
OMEN Synthesis Framework: Plan Level.

Builds a symbolic plan from `IntentState` through neural-guided search.

Mathematics (GOLOG/PDDL style):
  pre(aᵢ) ⊆ WMᵢ
  WM_{i+1} = (WMᵢ \ del(aᵢ)) ∪ add(aᵢ)
  goal ⊆ WM_{K+1}

  Neural search (similar to AlphaGo):
    π_plan(a|s) — neural policy for the next action
    V_plan(s)   — value network for pruning

  L_plan = -E_{τ~π_plan}[R_plan(τ)] + α_plan·Length(τ)

Integration:
  IntentState -> SymbolicPlanner -> PlanSequence
  PlanSequence -> HierarchicalDecoder (omen_osf_decoder.py)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLAN OPERATORS (GOLOG-style, without external solvers)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PlanFact:
    """Fact in the planner WM: `(predicate_id, arg)`."""
    pred: int
    arg:  int
    def __repr__(self) -> str:
        return f"f{self.pred}({self.arg})"


@dataclass
class PlanOperator:
    """
    Plan operator aᵢ:
      preconditions : facts that must be present in WM (pre)
      add_effects   : facts added after application
      del_effects   : facts removed after application
      embedding     : (d_plan,) — neural representation of the operator
      op_type       : string type used by the decoder for interpretation
    """
    op_id:         int
    op_type:       str                          # "define", "call", "assign", …
    preconditions: Tuple[PlanFact, ...]
    add_effects:   Tuple[PlanFact, ...]
    del_effects:   Tuple[PlanFact, ...]
    embedding:     Optional[torch.Tensor] = None   # (d_plan,)

    def applicable(self, wm: Set[PlanFact]) -> bool:
        return all(p in wm for p in self.preconditions)

    def apply(self, wm: Set[PlanFact]) -> Set[PlanFact]:
        new_wm = wm.copy()
        for f in self.del_effects:
            new_wm.discard(f)
        for f in self.add_effects:
            new_wm.add(f)
        return new_wm


@dataclass
class PlanState:
    """Planner state at time step `t`."""
    wm:        Set[PlanFact]              # current WorkingMemory
    depth:     int                        # search depth
    z_ctx:     Optional[torch.Tensor]     # (1, d_plan) — context at this step


@dataclass
class PlanSequence:
    """
    Output of `SymbolicPlanner`.
    operators   : sequence of applied operators
    embeddings  : (K, d_plan) — tensor representation for HierarchicalDecoder
    goal_reached: whether the goal was reached
    plan_loss   : L_plan (REINFORCE)
    """
    operators:    List[PlanOperator]
    embeddings:   torch.Tensor           # (K, d_plan)
    goal_reached: bool
    goal_progress: float
    goal_facts:    Tuple[PlanFact, ...]
    plan_loss:    torch.Tensor           # scalar


# ══════════════════════════════════════════════════════════════════════════════
# 2. NEURAL PLANNING POLICY
# ══════════════════════════════════════════════════════════════════════════════

class PlanPolicyNet(nn.Module):
    """
    π_plan(a|s): neural network -> distribution over operators.

    State `s` is encoded as:
      [z_intent; z_ctx; wm_features; depth_embed]
    where `wm_features` is the mean-pooled embedding of facts in WM.

    Architecture: a 2-layer MLP with gating.
    """

    def __init__(self, d_intent: int, d_plan: int, n_operators: int, dropout: float = 0.1):
        super().__init__()
        d_in = d_intent + d_plan + d_plan + 16  # intent + ctx + wm + depth

        self.depth_embed = nn.Embedding(32, 16)  # up to 32 steps

        self.register_buffer("_depth_ids", torch.arange(32, dtype=torch.long), persistent=False)

        self.actor = nn.Sequential(
            nn.Linear(d_in, d_plan * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_plan * 2, n_operators),
        )
        # Value network V(s).
        self.critic = nn.Sequential(
            nn.Linear(d_in, d_plan),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_plan, 1),
        )

    def _state_vec(
        self,
        z_intent: torch.Tensor,   # (1, d_intent)
        z_ctx:    torch.Tensor,   # (1, d_plan)
        wm_emb:   torch.Tensor,   # (1, d_plan)
        depth:    int,
    ) -> torch.Tensor:
        depth_idx = self._depth_ids[min(depth, 31)].to(device=z_intent.device).view(1)
        d_emb = self.depth_embed(depth_idx)   # (1, 16)
        return torch.cat([z_intent, z_ctx, wm_emb, d_emb], dim=-1)   # (1, d_in)

    def evaluate(
        self,
        z_intent: torch.Tensor,
        z_ctx:    torch.Tensor,
        wm_emb:   torch.Tensor,
        depth:    int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._state_vec(z_intent, z_ctx, wm_emb, depth)
        return self.actor(s), self.critic(s).squeeze(-1)

    def action_logits(
        self,
        z_intent: torch.Tensor,
        z_ctx:    torch.Tensor,
        wm_emb:   torch.Tensor,
        depth:    int,
    ) -> torch.Tensor:
        logits, _ = self.evaluate(z_intent, z_ctx, wm_emb, depth)
        return logits   # (1, n_operators)

    def value(
        self,
        z_intent: torch.Tensor,
        z_ctx:    torch.Tensor,
        wm_emb:   torch.Tensor,
        depth:    int,
    ) -> torch.Tensor:
        _, value = self.evaluate(z_intent, z_ctx, wm_emb, depth)
        return value   # (1,)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SYMBOLIC PLANNER
# ══════════════════════════════════════════════════════════════════════════════

class SymbolicPlanner(nn.Module):
    """
    Neural-guided symbolic planner.

    Solves: find a₁,...,aK ∈ A such that
      pre(aᵢ) ⊆ WMᵢ,  goal ⊆ WM_{K+1}

    Neural guidance: uses π_plan + V_plan instead of blind enumeration.

    Operators are generated by a neural network (OperatorGenerator) from IntentState,
    which lets the planner adapt to task context.
    """

    # Eight base operator types (domain-agnostic).
    OP_TYPES = ["define", "call", "assign", "return",
                "branch", "loop", "import", "yield"]

    def __init__(
        self,
        d_intent:     int,
        d_plan:       int,
        n_operators:  int   = 32,    # size of the operator library
        max_depth:    int   = 6,     # maximum plan depth
        beam_width:   int   = 3,     # beam-search width
        alpha_plan:   float = 0.1,   # penalty for plan length
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.d_intent    = d_intent
        self.d_plan      = d_plan
        self.n_operators = n_operators
        self.max_depth   = max_depth
        self.beam_width  = beam_width
        self.alpha_plan  = alpha_plan

        # Operator embeddings: O ∈ R^{n_operators × d_plan}
        self.op_embeddings = nn.Embedding(n_operators, d_plan)
        nn.init.normal_(self.op_embeddings.weight, std=d_plan ** -0.5)

        # WM fact embeddings (pred, arg → d_plan)
        self.fact_embed = nn.Embedding(512, d_plan // 2)  # pred
        self.arg_embed  = nn.Embedding(512, d_plan // 2)  # arg

        # Intent → context projection
        self.intent_proj = nn.Linear(d_intent, d_plan, bias=False)

        # Policy + Value
        self.policy = PlanPolicyNet(d_intent, d_plan, n_operators, dropout)

        # Operator parameter generator (intent + op_emb → pre/add/del)
        self.op_param_gen = nn.Sequential(
            nn.Linear(d_intent + d_plan, d_plan * 2),
            nn.GELU(),
            nn.Linear(d_plan * 2, 3 * 4 * 2),  # 3 effect-sets × 4 facts × 2 slots
        )

        # Context GRU tracks context along the plan.
        self.ctx_gru = nn.GRUCell(d_plan, d_plan)
        self.ctx_h0  = nn.Parameter(torch.zeros(1, d_plan))

    # ── WM → embedding ──────────────────────────────────────────────────────
    def _wm_embed(self, wm: Set[PlanFact], device) -> torch.Tensor:
        """Encode WorkingMemory into `(1, d_plan)` through mean pooling."""
        if not wm:
            return torch.zeros(1, self.d_plan, device=device)
        facts = list(wm)[:16]  # cap for speed
        preds = torch.tensor([f.pred % 512 for f in facts], device=device)
        args  = torch.tensor([f.arg  % 512 for f in facts], device=device)
        emb   = torch.cat([
            self.fact_embed(preds),
            self.arg_embed(args)
        ], dim=-1)  # (|WM|, d_plan)
        return emb.mean(0, keepdim=True)  # (1, d_plan)

    def _goal_facts(self, intent_state: "IntentState") -> Tuple[int, Tuple[PlanFact, ...]]:
        """Derive a compact symbolic goal from the intent distribution."""
        goal_id = int(intent_state.goal_probs[:1].mean(0).argmax().item())
        declared = PlanFact(301, goal_id)
        composed = PlanFact(302, goal_id)
        output = PlanFact(306, goal_id)
        satisfied = PlanFact(303, goal_id)

        mode = goal_id % 3
        if mode == 0:
            return goal_id, (declared, satisfied)
        if mode == 1:
            return goal_id, (composed, satisfied)
        return goal_id, (output, satisfied)

    @staticmethod
    def _horn_atom_to_plan_facts(atom) -> Tuple[PlanFact, ...]:
        facts: List[PlanFact] = []
        for arg in getattr(atom, "args", ()):
            val = getattr(arg, "val", None)
            if val is None:
                continue
            facts.append(PlanFact(int(atom.pred) % 512, int(val) % 512))
        if not facts:
            facts.append(PlanFact(int(atom.pred) % 512, 0))
        return tuple(dict.fromkeys(facts))

    def _symbolic_seed(
        self,
        symbolic_goal,
        symbolic_facts: Optional[FrozenSet],
    ) -> Tuple[Optional[int], Tuple[PlanFact, ...], Set[PlanFact]]:
        goal_facts = self._horn_atom_to_plan_facts(symbolic_goal) if symbolic_goal is not None else tuple()
        wm_seed: Set[PlanFact] = set()
        if symbolic_facts:
            for atom in list(symbolic_facts)[:32]:
                wm_seed.update(self._horn_atom_to_plan_facts(atom))
        goal_id = goal_facts[0].arg if goal_facts else None
        return goal_id, goal_facts, wm_seed

    def _operator_library(
        self,
        z_intent: torch.Tensor,
        goal_id: int,
        device: torch.device,
    ) -> List[PlanOperator]:
        """
        Build an operator library with real pre/add/del semantics for search.
        The neural network still sets embeddings and side effects, but core transitions are no longer blind.
        """
        start = PlanFact(300, goal_id)
        declared = PlanFact(301, goal_id)
        composed = PlanFact(302, goal_id)
        satisfied = PlanFact(303, goal_id)
        control = PlanFact(305, goal_id)
        output = PlanFact(306, goal_id)

        lib: List[PlanOperator] = []
        op_ids = torch.arange(self.n_operators, device=device)
        op_embs = self.op_embeddings(op_ids)
        _, aux_add_batch, aux_del_batch = self._gen_op_effects_batch(z_intent, op_embs)
        goal_mode = goal_id % 3
        for op_id, op_emb, aux_add, aux_del in zip(
            range(self.n_operators),
            op_embs.unbind(0),
            aux_add_batch,
            aux_del_batch,
        ):
            op_type = self.OP_TYPES[op_id % len(self.OP_TYPES)]

            if op_type == "define":
                pre = (start,)
                add = (declared,)
            elif op_type in ("assign", "call"):
                pre = (declared,)
                add = (composed,)
            elif op_type in ("branch", "loop"):
                pre = (declared,)
                add = (control,)
            elif op_type == "yield":
                pre = (control,)
                add = (output,)
            else:  # import / return
                if op_type == "import":
                    pre = (start,)
                    add = (declared,)
                elif goal_mode == 0:
                    pre = (declared,)
                    add = (satisfied,)
                elif goal_mode == 1:
                    pre = (composed,)
                    add = (satisfied,)
                else:
                    pre = (output,)
                    add = (satisfied,)

            add_all = tuple(dict.fromkeys(add + aux_add[:2]))
            del_all = tuple(dict.fromkeys(aux_del[:1]))
            lib.append(PlanOperator(
                op_id=op_id,
                op_type=op_type,
                preconditions=pre,
                add_effects=add_all,
                del_effects=del_all,
                embedding=op_emb,
            ))
        return lib

    def _symbolic_operator_library(
        self,
        prover,
        symbolic_facts: Optional[FrozenSet],
        device: torch.device,
    ) -> List[PlanOperator]:
        if prover is None or not getattr(prover.kb, "rules", None):
            return []
        current_preds = {fact.pred for fact in (symbolic_facts or frozenset())}
        lib: List[PlanOperator] = []
        for idx, rule in enumerate(prover.kb.rules):
            if not rule.body:
                continue
            if current_preds and not any(atom.pred in current_preds for atom in rule.body):
                continue
            pre: List[PlanFact] = []
            for atom in rule.body[:4]:
                pre.extend(self._horn_atom_to_plan_facts(atom)[:1])
            add = list(self._horn_atom_to_plan_facts(rule.head))
            if not pre or not add:
                continue
            emb = self.op_embeddings(torch.tensor([idx % self.n_operators], device=device)).squeeze(0)
            lib.append(PlanOperator(
                op_id=idx % self.n_operators,
                op_type="ltm_rule",
                preconditions=tuple(dict.fromkeys(pre)),
                add_effects=tuple(dict.fromkeys(add)),
                del_effects=tuple(),
                embedding=emb,
            ))
            if len(lib) >= self.n_operators:
                break
        return lib

    @staticmethod
    def _goal_progress(wm: Set[PlanFact], goal_facts: Tuple[PlanFact, ...]) -> float:
        if not goal_facts:
            return 0.0
        hit = sum(1 for fact in goal_facts if fact in wm)
        return hit / float(len(goal_facts))

    # ── Operator parameter generator ────────────────────────────────────────
    @torch.no_grad()
    def _gen_op_effects(
        self, z_intent: torch.Tensor, op_emb: torch.Tensor,
    ) -> Tuple[Tuple[PlanFact,...], Tuple[PlanFact,...], Tuple[PlanFact,...]]:
        """
        The neural network generates pre/add/del for the selected operator.
        Returns `PlanFact` tuples discretized through argmax.
        """
        inp    = torch.cat([z_intent, op_emb], dim=-1)   # (1, d_intent + d_plan)
        params = self.op_param_gen(inp).squeeze(0)       # (3·4·2,)
        params = params.view(3, 4, 2)                    # (3 sets, 4 facts, 2 slots)

        def to_facts(matrix: torch.Tensor) -> Tuple[PlanFact, ...]:
            # matrix: (4, 2) -> 4 facts, each in (pred, arg) form
            indices = matrix.abs().mul(128).long() % 128
            return tuple(
                PlanFact(int(indices[i, 0]), int(indices[i, 1]))
                for i in range(4)
                if indices[i, 0] > 0  # skip zero facts
            )

        return to_facts(params[0]), to_facts(params[1]), to_facts(params[2])

    @staticmethod
    def _decode_plan_facts(indices: torch.Tensor) -> Tuple[PlanFact, ...]:
        facts: List[PlanFact] = []
        for pred_idx, arg_idx in indices.tolist():
            if pred_idx > 0:
                facts.append(PlanFact(int(pred_idx), int(arg_idx)))
        return tuple(dict.fromkeys(facts))

    @torch.no_grad()
    def _gen_op_effects_batch(
        self,
        z_intent: torch.Tensor,
        op_embs: torch.Tensor,
    ) -> Tuple[List[Tuple[PlanFact, ...]], List[Tuple[PlanFact, ...]], List[Tuple[PlanFact, ...]]]:
        if op_embs.dim() == 1:
            op_embs = op_embs.unsqueeze(0)
        z_batch = z_intent.expand(op_embs.size(0), -1)
        inp = torch.cat([z_batch, op_embs], dim=-1)
        params = self.op_param_gen(inp).view(-1, 3, 4, 2)
        indices = params.abs().mul(128).long() % 128
        pre_list: List[Tuple[PlanFact, ...]] = []
        add_list: List[Tuple[PlanFact, ...]] = []
        del_list: List[Tuple[PlanFact, ...]] = []
        for op_idx in range(indices.size(0)):
            pre_list.append(self._decode_plan_facts(indices[op_idx, 0]))
            add_list.append(self._decode_plan_facts(indices[op_idx, 1]))
            del_list.append(self._decode_plan_facts(indices[op_idx, 2]))
        return pre_list, add_list, del_list

    # ── Main method: build a plan ───────────────────────────────────────────
    def forward(
        self,
        intent_state: "IntentState",
        symbolic_goal=None,
        symbolic_facts: Optional[FrozenSet] = None,
        prover=None,
    ) -> PlanSequence:
        """
        intent_state.z_intent: (B, d_intent)

        Greedy rollout with REINFORCE loss.
        For batch > 1, builds independent plans for each batch element.
        Returns the plan for the first batch element; the rest only affect the loss.
        """
        return self._forward_beam(
            intent_state,
            symbolic_goal=symbolic_goal,
            symbolic_facts=symbolic_facts,
            prover=prover,
        )

    def _forward_beam(
        self,
        intent_state: "IntentState",
        symbolic_goal=None,
        symbolic_facts: Optional[FrozenSet] = None,
        prover=None,
    ) -> PlanSequence:
        """
        Real search-based planner on top of Working Memory.
        Beam search moves over symbolic states, while policy/value only guide expansion.
        """
        device = intent_state.z_intent.device
        z_intent = intent_state.z_intent[:1]
        z_ctx = self.intent_proj(z_intent)
        sym_goal_id, symbolic_goal_facts, symbolic_wm = self._symbolic_seed(
            symbolic_goal, symbolic_facts
        )
        goal_id, default_goal_facts = self._goal_facts(intent_state)
        goal_facts = symbolic_goal_facts or default_goal_facts
        goal_id = sym_goal_id if sym_goal_id is not None else goal_id
        symbolic_ops = self._symbolic_operator_library(prover, symbolic_facts, device)
        default_ops = self._operator_library(z_intent, goal_id, device)
        operator_lib = symbolic_ops + default_ops[:max(self.n_operators - len(symbolic_ops), 0)]
        start_fact = PlanFact(300, goal_id)
        start_wm = symbolic_wm or {start_fact}

        beam = [{
            "wm": start_wm,
            "operators": [],
            "op_embs": [],
            "log_probs": [],
            "values": [],
            "ctx": z_ctx,
            "score": 0.0,
            "goal_progress": 0.0,
        }]
        plan_loss = torch.tensor(0.0, device=device)

        for step in range(self.max_depth):
            expanded = []
            for state in beam:
                wm = state["wm"]
                progress = self._goal_progress(wm, goal_facts)
                if progress >= 1.0:
                    expanded.append({**state, "score": state["score"] + 2.0})
                    continue

                wm_emb = self._wm_embed(wm, device)
                logits, value = self.policy.evaluate(z_intent, state["ctx"], wm_emb, step)
                logits = logits.squeeze(0)
                value = value.squeeze()

                applicable = [i for i, op in enumerate(operator_lib) if op.applicable(wm)]
                if not applicable:
                    expanded.append({**state, "score": state["score"] - 1.0})
                    continue

                applicable_idx = torch.tensor(applicable, device=device, dtype=torch.long)
                masked_logits = logits.index_select(0, applicable_idx)
                masked_log_probs = F.log_softmax(masked_logits, dim=-1)
                topk = min(self.beam_width, len(applicable))
                top_idx = masked_logits.topk(topk).indices
                value_score = float(value.detach().item())
                next_depth_penalty = self.alpha_plan * (len(state["operators"]) + 1)

                for local_idx in top_idx.tolist():
                    op = operator_lib[applicable[local_idx]]
                    op_emb = op.embedding.unsqueeze(0)
                    new_wm = op.apply(wm)
                    new_progress = self._goal_progress(new_wm, goal_facts)
                    log_prob = masked_log_probs[local_idx]
                    new_ctx = self.ctx_gru(op_emb, state["ctx"])
                    heuristic = (
                        2.5 * new_progress
                        + 0.2 * value_score
                        - next_depth_penalty
                    )
                    expanded.append({
                        "wm": new_wm,
                        "operators": state["operators"] + [op],
                        "op_embs": state["op_embs"] + [op_emb],
                        "log_probs": state["log_probs"] + [log_prob],
                        "values": state["values"] + [value],
                        "ctx": new_ctx,
                        "score": state["score"] + heuristic,
                        "goal_progress": new_progress,
                    })

            if not expanded:
                break

            expanded.sort(key=lambda s: (s["goal_progress"], s["score"]), reverse=True)
            beam = expanded[:self.beam_width]
            if beam[0]["goal_progress"] >= 1.0:
                break

        best = max(beam, key=lambda s: (s["goal_progress"], s["score"]))
        operators = best["operators"]
        op_embs = best["op_embs"]
        log_probs = best["log_probs"]
        values = best["values"]
        goal_progress = float(best["goal_progress"])
        goal_reached = goal_progress >= 1.0

        if op_embs:
            plan_embs = torch.cat(op_embs, dim=0)
        else:
            plan_embs = torch.zeros(1, self.d_plan, device=device)

        if self.training and log_probs:
            reward = goal_progress + (0.25 if goal_reached else 0.0)
            baseline = sum(v.item() for v in values) / len(values)
            log_p_sum = torch.stack(log_probs).sum()
            L_len = self.alpha_plan * len(operators)
            plan_loss = -(reward - baseline) * log_p_sum + torch.tensor(
                L_len, dtype=log_p_sum.dtype, device=device
            )

        return PlanSequence(
            operators=operators,
            embeddings=plan_embs,
            goal_reached=goal_reached,
            goal_progress=goal_progress,
            goal_facts=goal_facts,
            plan_loss=plan_loss,
        )

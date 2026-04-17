"""
ontology_engine.py — Ontology Expansion Engine (OEE)

Повна реалізація концепції з нейромережевим спрямованим пошуком:

  · PredicateLatentModel — нейромережа, що відображає стан KB → h ∈ ℝ^d_latent.
    Входи: агрегат ембеддингів існуючих предикатів (з AME HypergraphGNN) + gap_norm.
    Виходи: h (латентний опис нового предиката), arity_logits, interaction_scores,
            confidence, context_input (для online training).
    Cross-attention: h запитує ембеддинги існуючих предикатів для визначення
    партнерів взаємодії P_new.

  · RuleHypothesisSampler — генерує та ранжує кандидатні правила за:
      argmax_{Γ_new} [Consistency(Γ ∪ Γ_new, Data) − λ · Complexity(Γ_new)]
    де Consistency = |FC_sandbox(Γ ∪ {r}) ∩ Data| − |FC(Γ) ∩ Data|,
        Complexity  = MDL_bits(r).
    Шаблони: unary_link, transitive, bridge_goal, fact_coverage.

  · Online learning: VeM зворотний зв'язок оновлює PredicateLatentModel через
    контрастивний лос на буфері context-input векторів.
    Прийняті правила → позитивні пари (h кластеризуються).
    Відхилені → негативні пари (відштовхуються від прийнятих).

Математична модель:
  P_new* = argmax_P [max_{Γ_new} (Consistency(Γ ∪ Γ_new, Data) − λ · Complexity(Γ_new))]
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omen_symbolic.creative_types import RuleCandidate


# ─── Латентна модель нового предиката ────────────────────────────────────────

class PredicateLatentModel(nn.Module):
    """
    Нейромережа для спрямованого пошуку нового предиката P_new.

    Архітектура:
      context_encoder : (embed_dim + 1) → d_latent   (pooled embeds + gap_norm)
      h_decoder       : d_latent → d_latent           (латентний вектор h P_new)
      arity_head      : d_latent → max_arity          (передбачення арності)
      attn_query/key  : cross-attention h → pred_embeds (interaction scores)
      confidence_head : d_latent → 1                  (впевненість моделі)

    Навчання:
      Онлайн контрастивний лос на буфері (context_input, accepted).
      Прийняті VeM правила → позитивні пари (h мають кластеризуватись).
      Відхилені → негативні пари (h відштовхуються від прийнятих).
    """

    def __init__(
        self,
        embed_dim: int,
        d_latent: int = 32,
        max_arity: int = 3,
        state_dim: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.d_latent = int(d_latent)
        self.max_arity = int(max_arity)
        self.state_dim = int(max(state_dim, 0))

        # Encoder: pooled pred embeddings + gap_norm + summary(z) → latent context
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim + 1 + self.state_dim, d_latent * 2),
            nn.GELU(),
            nn.LayerNorm(d_latent * 2),
            nn.Linear(d_latent * 2, d_latent),
        )
        # h decoder: context → latent representation of P_new
        self.h_decoder = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
            nn.LayerNorm(d_latent),
        )
        # Arity predictor
        self.arity_head = nn.Linear(d_latent, max_arity)
        # Cross-attention: query from h, keys from existing pred embeddings
        self.attn_query = nn.Linear(d_latent, d_latent, bias=False)
        self.attn_key   = nn.Linear(embed_dim, d_latent, bias=False)
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(d_latent, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def encode_context_input(
        self,
        pred_embeddings: torch.Tensor,
        gap_norm: float,
        unexplained_mask: Optional[torch.Tensor] = None,
        state_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Будує вхідний контекст-вектор (embed_dim + 1 + state_dim,).
        Повертає сирий вектор — зберігається для online training.
        """
        if pred_embeddings.size(0) == 0:
            pooled = torch.zeros(self.embed_dim, dtype=torch.float32)
        elif unexplained_mask is not None and bool(unexplained_mask.any()):
            weights = unexplained_mask.float() * 2.0 + 1.0
            weights = weights / weights.sum()
            pooled = (pred_embeddings * weights.unsqueeze(-1)).sum(dim=0)
        else:
            pooled = pred_embeddings.mean(dim=0)
        gap_t = torch.tensor([float(gap_norm)], dtype=torch.float32)
        if self.state_dim > 0:
            if state_z is None or state_z.numel() == 0:
                state_summary = torch.zeros(self.state_dim, dtype=torch.float32)
            else:
                pooled_z = state_z.detach().mean(dim=0).flatten().float().cpu()
                stats = torch.tensor(
                    [
                        float(pooled_z.mean().item()),
                        float(pooled_z.std(unbiased=False).item() if pooled_z.numel() > 1 else 0.0),
                        float(pooled_z.norm().item() / max(pooled_z.numel(), 1) ** 0.5),
                        float(pooled_z.abs().max().item()),
                    ],
                    dtype=torch.float32,
                )
                state_summary = stats[: self.state_dim]
                if state_summary.numel() < self.state_dim:
                    state_summary = F.pad(state_summary, (0, self.state_dim - state_summary.numel()))
        else:
            state_summary = torch.zeros(0, dtype=torch.float32)
        return torch.cat([pooled, gap_t, state_summary], dim=-1)

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        gap_norm: float,
        unexplained_mask: Optional[torch.Tensor] = None,
        state_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          h                  : (d_latent,)    — latent representation of P_new
          arity_logits       : (max_arity,)   — raw logits for arity [1,2,3]
          interaction_scores : (n_pred,)      — softmax attention weights
          confidence         : (1,)           — model confidence [0,1]
          context_input      : (embed_dim+1+state_dim,) — raw input (for online training buffer)
        """
        context_input = self.encode_context_input(
            pred_embeddings,
            gap_norm,
            unexplained_mask,
            state_z=state_z,
        )
        context = self.context_encoder(context_input)      # (d_latent,)
        h = self.h_decoder(context)                        # (d_latent,)
        arity_logits = self.arity_head(h)                  # (max_arity,)
        confidence = self.confidence_head(h)               # (1,)

        # Cross-attention: h queries existing predicate embeddings
        if pred_embeddings.size(0) > 0:
            q = self.attn_query(h).unsqueeze(0)            # (1, d_latent)
            k = self.attn_key(pred_embeddings)             # (n_pred, d_latent)
            scale = math.sqrt(float(self.d_latent))
            scores = (q @ k.T).squeeze(0) / scale         # (n_pred,)
            interaction_scores = F.softmax(scores, dim=-1)
        else:
            interaction_scores = torch.zeros(0, dtype=torch.float32)

        return h, arity_logits, interaction_scores, confidence, context_input


# ─── Гіпотеза правила ────────────────────────────────────────────────────────

@dataclass
class _RuleHypothesis:
    """Одна гіпотеза правила з оцінкою Consistency − λ · Complexity."""
    clause: Any
    consistency: float       # |FC_sandbox ∩ Data| − |FC_baseline ∩ Data|
    complexity: float        # MDL_bits(rule)
    score: float             # consistency − lambda * complexity
    pred_id: int
    interaction_preds: Tuple[int, ...]
    template: str            # назва шаблону для діагностики
    coverage_gain: float = 0.0
    novelty_gain: float = 0.0
    bundle_score: float = 0.0
    bundle_size: int = 1
    bundle_templates: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class PredicateVocabEntry:
    pred_id: int
    arity: int
    status: str = "proposed"
    confidence: float = 0.0
    gap_before: float = 0.0
    gap_after: float = 0.0
    supporting_rules: Tuple[int, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Генератор та ранжировщик гіпотез ────────────────────────────────────────

class RuleHypothesisSampler:
    """
    Генерує та ранжує кандидатні правила для нового предиката P_new.

    Реалізує оптимізацію:
      argmax_{Γ_new} [Consistency(Γ ∪ Γ_new, Data) − λ · Complexity(Γ_new)]

    Шаблони:
      1. unary_link    : P_new(X,...) :- P_interact(X,...)
      2. transitive    : P_new(X,Z)   :- P_a(X,Y), P_b(Y,Z)
      3. bridge_goal   : P_goal(...)  :- P_new(...)
      4. fact_coverage : P_new(X,...) :- P_unexplained(X,...)
    """

    def __init__(
        self,
        consistency_lambda: float = 0.1,
        max_interaction_preds: int = 3,
        max_hypotheses: int = 8,
        forward_chain_depth: int = 2,
        max_open_body_literals: int = 2,
        max_open_patterns: int = 4,
        max_open_head_patterns: int = 3,
        bundle_beam_width: int = 4,
        max_bundle_rules: int = 3,
        bundle_seed_k: int = 12,
    ) -> None:
        self.consistency_lambda = float(max(consistency_lambda, 1e-6))
        self.max_interaction_preds = max(int(max_interaction_preds), 1)
        self.max_hypotheses = max(int(max_hypotheses), 1)
        self.forward_chain_depth = max(int(forward_chain_depth), 1)
        self.max_open_body_literals = max(int(max_open_body_literals), 1)
        self.max_open_patterns = max(int(max_open_patterns), 4)
        self.max_open_head_patterns = max(int(max_open_head_patterns), 4)
        self.bundle_beam_width = max(int(bundle_beam_width), 2)
        self.max_bundle_rules = max(int(max_bundle_rules), 2)
        self.bundle_seed_k = max(int(bundle_seed_k), self.bundle_beam_width)
        self._last_beam_states: int = 0
        self._last_beam_best_size: int = 1
        self._last_beam_best_score: float = 0.0

    @staticmethod
    def _make_vars(n: int) -> Tuple[Any, ...]:
        from omen_prolog import Var
        return tuple(Var(f"V{i}") for i in range(n))

    @staticmethod
    def _mdl_bits(rule: Any) -> float:
        try:
            return float(rule.description_length_bits())
        except Exception:
            return float(8 + 4 * len(tuple(getattr(rule, "body", ()))))

    @staticmethod
    def _make_rule(head: Any, body: Tuple[Any, ...]) -> Any:
        from omen_prolog import HornClause
        return HornClause(head=head, body=body, weight=1.0, use_count=0)

    @staticmethod
    def _make_atom(pred: int, args: Tuple[Any, ...]) -> Any:
        from omen_prolog import HornAtom
        return HornAtom(pred=int(pred), args=args)

    def _build_sandbox(self, kb: Any) -> Any:
        try:
            sb = type(kb)(max_rules=max(int(getattr(kb, "max_rules", 512)), 32))
        except TypeError:
            sb = type(kb)(max_rules=max(int(getattr(kb, "max_rules", 512)), 32), device=None)
        for fact in getattr(kb, "facts", frozenset()):
            sb.add_fact(fact)
        for rule in getattr(kb, "rules", ()):
            sb.add_rule(rule)
        return sb

    @staticmethod
    def _is_safe_rule(rule: Any) -> bool:
        head_vars = set(getattr(rule.head, "vars", lambda: frozenset())())
        if not head_vars:
            return True
        body_vars: set[str] = set()
        for atom in getattr(rule, "body", ()):
            body_vars |= set(getattr(atom, "vars", lambda: frozenset())())
        return head_vars.issubset(body_vars)

    @staticmethod
    def _supports_target(target: Any, facts: FrozenSet[Any]) -> bool:
        from omen_prolog import unify

        for fact in facts:
            try:
                if unify(target, fact) is not None:
                    return True
            except Exception:
                continue
        return False

    @classmethod
    def _coverage_score(cls, facts: FrozenSet[Any], targets: Sequence[Any]) -> float:
        if not targets:
            return 0.0
        return float(sum(1.0 for target in targets if cls._supports_target(target, facts)))

    def _consistency_score(
        self,
        kb: Any,
        new_rules: Sequence[Any],
        baseline: FrozenSet[Any],
        starting_facts: FrozenSet[Any],
        target_facts: Optional[Sequence[Any]] = None,
    ) -> Tuple[float, float, float]:
        """
        Consistency(Γ ∪ {r}, Data) − Consistency(Γ, Data):
        target coverage gain plus a small novelty bonus.
        """
        try:
            sb = self._build_sandbox(kb)
            for rule in new_rules:
                sb.add_rule(rule)
            derived = sb.forward_chain(
                self.forward_chain_depth,
                starting_facts=starting_facts,
                only_verified=False,
            )
            novelty_gain = float(len(derived - baseline))
            targets = tuple(target_facts or ())
            baseline_coverage = self._coverage_score(baseline, targets)
            derived_coverage = self._coverage_score(derived, targets)
            coverage_gain = float(derived_coverage - baseline_coverage)
            if targets:
                consistency = 3.0 * coverage_gain + 0.10 * novelty_gain
            else:
                consistency = novelty_gain
            return float(consistency), coverage_gain, novelty_gain
        except Exception:
            return 0.0, 0.0, 0.0

    def _add_hypothesis(
        self,
        hypotheses: List[_RuleHypothesis],
        seen: set,
        rule: Any,
        template: str,
        interaction_preds: Tuple[int, ...],
        kb: Any,
        baseline: FrozenSet[Any],
        starting_facts: FrozenSet[Any],
        target_facts: Optional[Sequence[Any]] = None,
    ) -> None:
        if not self._is_safe_rule(rule):
            return
        h = hash(rule)
        if h in seen:
            return
        seen.add(h)
        consistency, coverage_gain, novelty_gain = self._consistency_score(
            kb,
            [rule],
            baseline,
            starting_facts,
            target_facts=target_facts,
        )
        complexity = self._mdl_bits(rule)
        score = consistency - self.consistency_lambda * complexity
        hypotheses.append(_RuleHypothesis(
            clause=rule,
            consistency=consistency,
            complexity=complexity,
            score=score,
            pred_id=int(getattr(rule.head, "pred", 0)),
            interaction_preds=interaction_preds,
            template=template,
            coverage_gain=coverage_gain,
            novelty_gain=novelty_gain,
        ))

    @staticmethod
    def _canonicalize_assignments(assignments: Sequence[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], ...]:
        mapping: Dict[int, int] = {}
        next_id = 0
        canonical: List[Tuple[int, ...]] = []
        for atom_args in assignments:
            mapped_args: List[int] = []
            for slot in atom_args:
                if slot not in mapping:
                    mapping[slot] = next_id
                    next_id += 1
                mapped_args.append(mapping[slot])
            canonical.append(tuple(mapped_args))
        return tuple(canonical)

    @classmethod
    def _connected_assignments(cls, assignments: Sequence[Tuple[int, ...]]) -> bool:
        if len(assignments) <= 1:
            return True
        var_sets = [set(atom_args) for atom_args in assignments]
        agenda = [0]
        seen = {0}
        while agenda:
            idx = agenda.pop()
            for other_idx, other in enumerate(var_sets):
                if other_idx in seen:
                    continue
                if var_sets[idx] & other:
                    seen.add(other_idx)
                    agenda.append(other_idx)
        return len(seen) == len(var_sets)

    def _enumerate_body_assignments(
        self,
        body_arities: Sequence[int],
        max_vars: int,
    ) -> List[Tuple[Tuple[int, ...], ...]]:
        if not body_arities:
            return []
        total_slots = sum(int(arity) for arity in body_arities)
        max_vars = max(int(max_vars), 1)
        candidates: Dict[Tuple[Tuple[int, ...], ...], Tuple[int, int]] = {}
        for raw_slots in product(range(max_vars), repeat=total_slots):
            cursor = 0
            atom_args: List[Tuple[int, ...]] = []
            for arity in body_arities:
                atom_args.append(tuple(raw_slots[cursor : cursor + int(arity)]))
                cursor += int(arity)
            if not self._connected_assignments(atom_args):
                continue
            canonical = self._canonicalize_assignments(atom_args)
            used_vars = len({slot for atom in canonical for slot in atom})
            shared_slots = sum(len(atom) for atom in canonical) - used_vars
            rank = (shared_slots, -used_vars)
            current = candidates.get(canonical)
            if current is None or rank > current:
                candidates[canonical] = rank
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        return [pattern for pattern, _ in ranked[: self.max_open_patterns]]

    def _enumerate_head_assignments(
        self,
        head_arity: int,
        body_assignments: Sequence[Tuple[int, ...]],
    ) -> List[Tuple[int, ...]]:
        if head_arity <= 0:
            return [tuple()]
        degrees: Dict[int, int] = defaultdict(int)
        for atom_args in body_assignments:
            for slot in atom_args:
                degrees[int(slot)] += 1
        available = tuple(sorted(degrees))
        if not available:
            return []
        ranked: Dict[Tuple[int, ...], Tuple[int, int]] = {}
        for head_args in product(available, repeat=head_arity):
            distinct = len(set(head_args))
            support = sum(degrees[int(slot)] for slot in head_args)
            rank = (support, distinct)
            current = ranked.get(tuple(head_args))
            if current is None or rank > current:
                ranked[tuple(head_args)] = rank
        ordered = sorted(ranked.items(), key=lambda item: item[1], reverse=True)
        return [pattern for pattern, _ in ordered[: self.max_open_head_patterns]]

    def _open_second_order_search(
        self,
        hypotheses: List[_RuleHypothesis],
        seen: set,
        pred_id: int,
        arity: int,
        top_preds: Sequence[int],
        pred_arity_map: Dict[int, int],
        explanation_targets: Sequence[Any],
        kb: Any,
        baseline: FrozenSet[Any],
        starting_facts: FrozenSet[Any],
    ) -> None:
        if not top_preds:
            return

        vars_ = self._make_vars(8)
        body_limit = min(self.max_open_body_literals, max(len(top_preds), 1))
        open_pred_pool = [int(pred) for pred in top_preds[: min(self.max_interaction_preds, 2)]]

        def _rule_from_patterns(
            head_pred: int,
            head_assignment: Tuple[int, ...],
            body_preds: Sequence[int],
            body_assignments: Sequence[Tuple[int, ...]],
        ) -> Any:
            head = self._make_atom(int(head_pred), tuple(vars_[int(idx)] for idx in head_assignment))
            body = tuple(
                self._make_atom(int(body_pred), tuple(vars_[int(idx)] for idx in assignment))
                for body_pred, assignment in zip(body_preds, body_assignments)
            )
            return self._make_rule(head, body)

        for body_size in range(1, body_limit + 1):
            for body_preds in combinations_with_replacement(open_pred_pool, body_size):
                body_arities = [int(pred_arity_map.get(int(pred), arity)) for pred in body_preds]
                max_vars = min(max(max(body_arities), arity) + 1, len(vars_))
                body_patterns = self._enumerate_body_assignments(body_arities, max_vars=max_vars)
                for body_pattern in body_patterns:
                    for head_pattern in self._enumerate_head_assignments(arity, body_pattern):
                        try:
                            rule = _rule_from_patterns(pred_id, head_pattern, body_preds, body_pattern)
                            self._add_hypothesis(
                                hypotheses,
                                seen,
                                rule,
                                f"open_invented_{body_size}",
                                tuple(int(pred) for pred in body_preds),
                                kb,
                                baseline,
                                starting_facts,
                                target_facts=explanation_targets,
                            )
                        except Exception:
                            continue

        for target in explanation_targets:
            target_pred = int(getattr(target, "pred", -1))
            target_arity = len(tuple(getattr(target, "args", ())))
            for extra_size in range(0, min(self.max_open_body_literals - 1, len(open_pred_pool)) + 1):
                for extra_preds in combinations_with_replacement(open_pred_pool, extra_size):
                    body_preds = (int(pred_id),) + tuple(int(pred) for pred in extra_preds)
                    body_arities = [int(arity)] + [int(pred_arity_map.get(int(pred), target_arity or arity)) for pred in extra_preds]
                    max_vars = min(max([target_arity, arity] + body_arities) + 1, len(vars_))
                    body_patterns = self._enumerate_body_assignments(body_arities, max_vars=max_vars)
                    for body_pattern in body_patterns:
                        for head_pattern in self._enumerate_head_assignments(target_arity, body_pattern):
                            try:
                                bridge_rule = _rule_from_patterns(target_pred, head_pattern, body_preds, body_pattern)
                                self._add_hypothesis(
                                    hypotheses,
                                    seen,
                                    bridge_rule,
                                    f"open_bridge_{len(body_preds)}",
                                    tuple(int(pred) for pred in body_preds[1:]),
                                    kb,
                                    baseline,
                                    starting_facts,
                                    target_facts=explanation_targets,
                                )
                            except Exception:
                                continue

        for alias_pred in open_pred_pool:
            alias_arity = int(pred_arity_map.get(int(alias_pred), arity))
            for extra_size in range(0, min(self.max_open_body_literals - 1, len(open_pred_pool)) + 1):
                for extra_preds in combinations_with_replacement(open_pred_pool, extra_size):
                    body_preds = (int(pred_id),) + tuple(int(pred) for pred in extra_preds)
                    body_arities = [int(arity)] + [int(pred_arity_map.get(int(pred), alias_arity or arity)) for pred in extra_preds]
                    max_vars = min(max([alias_arity, arity] + body_arities) + 1, len(vars_))
                    body_patterns = self._enumerate_body_assignments(body_arities, max_vars=max_vars)
                    for body_pattern in body_patterns:
                        for head_pattern in self._enumerate_head_assignments(alias_arity, body_pattern):
                            try:
                                alias_rule = _rule_from_patterns(int(alias_pred), head_pattern, body_preds, body_pattern)
                                self._add_hypothesis(
                                    hypotheses,
                                    seen,
                                    alias_rule,
                                    f"open_alias_{len(body_preds)}",
                                    tuple(int(pred) for pred in body_preds[1:]),
                                    kb,
                                    baseline,
                                    starting_facts,
                                    target_facts=explanation_targets,
                                )
                            except Exception:
                                continue

    def _attach_bundle_scores(
        self,
        hypotheses: List[_RuleHypothesis],
        kb: Any,
        baseline: FrozenSet[Any],
        starting_facts: FrozenSet[Any],
        target_facts: Sequence[Any],
    ) -> None:
        if len(hypotheses) < 2:
            return

        core_templates = {"unary_link", "transitive", "pair_factor", "fact_coverage"}
        bridge_templates = {"bridge_goal", "bridge_target", "goal_factorization", "target_factorization"}
        alias_templates = {"inverse_alias"}

        def _is_core(template: str) -> bool:
            return template in core_templates or template.startswith("open_invented_")

        def _is_bridge(template: str) -> bool:
            return template in bridge_templates or template.startswith("open_bridge_")

        def _is_alias(template: str) -> bool:
            return template in alias_templates or template.startswith("open_alias_")

        core = [hyp for hyp in hypotheses if _is_core(hyp.template)]
        bridges = [hyp for hyp in hypotheses if _is_bridge(hyp.template)]
        aliases = [hyp for hyp in hypotheses if _is_alias(hyp.template)]
        seen_bundles: set[Tuple[int, ...]] = set()

        def _update_bundle(hypothesis: _RuleHypothesis, score: float, templates: Tuple[str, ...], size: int) -> None:
            has_bundle = hypothesis.bundle_size > 1 or bool(hypothesis.bundle_templates)
            if has_bundle and score <= hypothesis.bundle_score:
                return
            hypothesis.bundle_score = float(score)
            hypothesis.bundle_templates = templates
            hypothesis.bundle_size = int(size)

        for base in core[: self.max_hypotheses]:
            for bridge in bridges[: self.max_hypotheses]:
                pair_rules = [base.clause, bridge.clause]
                pair_key = tuple(sorted(hash(rule) for rule in pair_rules))
                if pair_key in seen_bundles:
                    continue
                seen_bundles.add(pair_key)
                consistency, _, _ = self._consistency_score(
                    kb,
                    pair_rules,
                    baseline,
                    starting_facts,
                    target_facts=target_facts,
                )
                pair_score = consistency - self.consistency_lambda * (base.complexity + bridge.complexity)
                pair_share = pair_score / 2.0
                pair_templates = tuple(sorted({base.template, bridge.template}))
                _update_bundle(base, pair_share, pair_templates, 2)
                _update_bundle(bridge, pair_share, pair_templates, 2)

                for alias in aliases[: self.max_hypotheses]:
                    triple_rules = [base.clause, bridge.clause, alias.clause]
                    triple_key = tuple(sorted(hash(rule) for rule in triple_rules))
                    if triple_key in seen_bundles:
                        continue
                    seen_bundles.add(triple_key)
                    tri_consistency, _, _ = self._consistency_score(
                        kb,
                        triple_rules,
                        baseline,
                        starting_facts,
                        target_facts=target_facts,
                    )
                    tri_score = tri_consistency - self.consistency_lambda * (
                        base.complexity + bridge.complexity + alias.complexity
                    )
                    tri_share = tri_score / 3.0
                    tri_templates = tuple(sorted({base.template, bridge.template, alias.template}))
                    _update_bundle(base, tri_share, tri_templates, 3)
                    _update_bundle(bridge, tri_share, tri_templates, 3)
                    _update_bundle(alias, tri_share, tri_templates, 3)

    def _beam_bundle_search(
        self,
        hypotheses: List[_RuleHypothesis],
        kb: Any,
        baseline: FrozenSet[Any],
        starting_facts: FrozenSet[Any],
        target_facts: Sequence[Any],
    ) -> None:
        if len(hypotheses) < 2:
            self._last_beam_states = 0
            self._last_beam_best_size = 1
            self._last_beam_best_score = max((hyp.score for hyp in hypotheses), default=0.0)
            return

        ranked_hypotheses = sorted(
            enumerate(hypotheses),
            key=lambda item: (max(item[1].score, item[1].bundle_score), item[1].coverage_gain, item[1].novelty_gain),
            reverse=True,
        )
        seed_items = ranked_hypotheses[: min(len(ranked_hypotheses), self.bundle_seed_k)]
        score_cache: Dict[Tuple[int, ...], Tuple[float, Tuple[str, ...]]] = {}
        best_for_hyp: Dict[int, Tuple[float, Tuple[str, ...], int]] = {}
        beam: List[Tuple[Tuple[int, ...], float, Tuple[str, ...]]] = []
        visited_states: set[Tuple[int, ...]] = set()
        evaluated_states = 0

        def _state_score(state: Tuple[int, ...]) -> Tuple[float, Tuple[str, ...]]:
            cached = score_cache.get(state)
            if cached is not None:
                return cached
            rules = [hypotheses[idx].clause for idx in state]
            consistency, _, _ = self._consistency_score(
                kb,
                rules,
                baseline,
                starting_facts,
                target_facts=target_facts,
            )
            complexity = sum(hypotheses[idx].complexity for idx in state)
            templates = tuple(sorted({hypotheses[idx].template for idx in state}))
            score = float(consistency - self.consistency_lambda * complexity)
            score_cache[state] = (score, templates)
            return score, templates

        def _record_state(state: Tuple[int, ...], score: float, templates: Tuple[str, ...]) -> None:
            for idx in state:
                current = best_for_hyp.get(idx)
                if current is None or score > current[0]:
                    best_for_hyp[idx] = (float(score), templates, len(state))

        for idx, hyp in seed_items:
            state = (int(idx),)
            score, templates = _state_score(state)
            beam.append((state, score, templates))
            visited_states.add(state)
            evaluated_states += 1
            _record_state(state, score, templates)

        best_state_size = max((len(state) for state, _, _ in beam), default=1)
        best_state_score = max((score for _, score, _ in beam), default=0.0)

        for _depth in range(2, self.max_bundle_rules + 1):
            expanded: List[Tuple[Tuple[int, ...], float, Tuple[str, ...]]] = []
            for state, _score, _templates in beam[: self.bundle_beam_width]:
                used = set(state)
                last_idx = state[-1]
                for next_idx, _hyp in seed_items:
                    next_idx = int(next_idx)
                    if next_idx in used or next_idx <= last_idx:
                        continue
                    new_state = tuple(sorted(state + (next_idx,)))
                    if new_state in visited_states:
                        continue
                    visited_states.add(new_state)
                    score, templates = _state_score(new_state)
                    expanded.append((new_state, score, templates))
                    evaluated_states += 1
                    _record_state(new_state, score, templates)
            if not expanded:
                break
            expanded.sort(key=lambda item: item[1], reverse=True)
            beam = expanded[: self.bundle_beam_width]
            best_state_size = max(best_state_size, max((len(state) for state, _, _ in beam), default=1))
            best_state_score = max(best_state_score, max((score for _, score, _ in beam), default=best_state_score))

        for idx, hyp in enumerate(hypotheses):
            state_info = best_for_hyp.get(idx)
            if state_info is None:
                continue
            bundle_score, templates, size = state_info
            has_bundle = hyp.bundle_size > 1 or bool(hyp.bundle_templates)
            if has_bundle and bundle_score <= hyp.bundle_score:
                continue
            hyp.bundle_score = float(bundle_score)
            hyp.bundle_templates = tuple(templates)
            hyp.bundle_size = int(size)

        self._last_beam_states = int(evaluated_states)
        self._last_beam_best_size = int(best_state_size)
        self._last_beam_best_score = float(best_state_score)

    def sample(
        self,
        pred_id: int,
        arity: int,
        interaction_preds: Sequence[int],
        interaction_scores: Sequence[float],
        current_facts: Sequence[Any],
        kb: Any,
        goal: Optional[Any],
        target_facts: Optional[Sequence[Any]] = None,
    ) -> List[_RuleHypothesis]:
        """
        Генерує гіпотези за всіма шаблонами і ранжує за
        argmax [Consistency − λ · Complexity].
        """
        if not current_facts:
            return []

        starting_facts = frozenset(current_facts)
        try:
            baseline: FrozenSet[Any] = kb.forward_chain(
                self.forward_chain_depth,
                starting_facts=starting_facts,
                only_verified=False,
            )
        except Exception:
            baseline = frozenset()

        max_fact_arity = max((len(tuple(getattr(fact, "args", ()))) for fact in current_facts), default=0)
        goal_arity = len(tuple(getattr(goal, "args", ()))) if goal is not None else 0
        max_target_arity = max(
            (len(tuple(getattr(target, "args", ()))) for target in (target_facts or ())),
            default=0,
        )
        vars_ = self._make_vars(max(arity, max_fact_arity, goal_arity, max_target_arity, 3) + 1)
        hypotheses: List[_RuleHypothesis] = []
        seen: set = set()
        explanation_targets: List[Any] = []
        seen_targets: set[int] = set()
        if goal is not None:
            explanation_targets.append(goal)
            seen_targets.add(hash(goal))
        for target in target_facts or ():
            target_hash = hash(target)
            if target_hash in seen_targets:
                continue
            explanation_targets.append(target)
            seen_targets.add(target_hash)

        # Індекс фактів по предикату
        ip_facts_map: Dict[int, List[Any]] = defaultdict(list)
        for f in current_facts:
            ip_facts_map[int(getattr(f, "pred", -1))].append(f)

        pred_arity_map: Dict[int, int] = {
            pred: len(tuple(getattr(facts[0], "args", ())))
            for pred, facts in ip_facts_map.items()
            if facts
        }

        # ── Шаблон 1: unary_link ─────────────────────────────────────────────
        # P_new(X,...) :- P_interact(X,...) — новий предикат як структурний псевдонім
        for i, (ip, _) in enumerate(zip(interaction_preds, interaction_scores)):
            if i >= self.max_interaction_preds:
                break
            ip_facts = ip_facts_map.get(ip, [])
            ip_arity = len(tuple(getattr(ip_facts[0], "args", ()))) if ip_facts else arity
            body_vars = vars_[:ip_arity]
            head_vars = vars_[:min(arity, ip_arity)]
            try:
                rule = self._make_rule(
                    self._make_atom(pred_id, tuple(head_vars)),
                    (self._make_atom(ip, tuple(body_vars)),),
                )
                self._add_hypothesis(
                    hypotheses,
                    seen,
                    rule,
                    "unary_link",
                    (ip,),
                    kb,
                    baseline,
                    starting_facts,
                    target_facts=explanation_targets,
                )
            except Exception:
                continue

        # ── Шаблон 2: transitive ─────────────────────────────────────────────
        # P_new(X,Z) :- P_a(X,Y), P_b(Y,Z) — транзитивна ланка
        if arity >= 2 and len(interaction_preds) >= 2 and len(vars_) >= 3:
            x, y, z = vars_[0], vars_[1], vars_[2]
            p_a, p_b = interaction_preds[0], interaction_preds[1]
            try:
                rule = self._make_rule(
                    self._make_atom(pred_id, (x, z)),
                    (self._make_atom(p_a, (x, y)), self._make_atom(p_b, (y, z))),
                )
                self._add_hypothesis(
                    hypotheses,
                    seen,
                    rule,
                    "transitive",
                    (p_a, p_b),
                    kb,
                    baseline,
                    starting_facts,
                    target_facts=explanation_targets,
                )
            except Exception:
                pass

        # ── Шаблон 2b: pair_factor ───────────────────────────────────────────
        # P_new(...) :- P_a(...), P_b(...) — загальніша second-order факторизація
        top_preds = list(interaction_preds[: self.max_interaction_preds])
        for idx, p_a in enumerate(top_preds):
            for p_b in top_preds[idx + 1 :]:
                a_arity = int(pred_arity_map.get(int(p_a), arity))
                b_arity = int(pred_arity_map.get(int(p_b), arity))
                left_vars = vars_[:a_arity]
                right_start = max(a_arity - 1, 0)
                right_vars = vars_[right_start : right_start + b_arity]
                merged_head_vars = tuple(list(left_vars) + [v for v in right_vars if v not in left_vars])
                if len(merged_head_vars) < arity:
                    continue
                head_vars = merged_head_vars[:arity]
                try:
                    rule = self._make_rule(
                        self._make_atom(pred_id, tuple(head_vars)),
                        (
                            self._make_atom(int(p_a), tuple(left_vars)),
                            self._make_atom(int(p_b), tuple(right_vars)),
                        ),
                    )
                    self._add_hypothesis(
                        hypotheses,
                        seen,
                        rule,
                        "pair_factor",
                        (int(p_a), int(p_b)),
                        kb,
                        baseline,
                        starting_facts,
                        target_facts=explanation_targets,
                    )
                except Exception:
                    continue

        # ── Шаблон 3: bridge_goal ────────────────────────────────────────────
        # P_goal(...) :- P_new(...) — bridge від нового предиката до цільового
        if goal is not None:
            try:
                goal_arity = len(tuple(getattr(goal, "args", ())))
                if goal_arity <= arity:
                    goal_vars = vars_[:goal_arity]
                    bridge_vars = vars_[:arity]
                    rule = self._make_rule(
                        self._make_atom(int(goal.pred), tuple(goal_vars)),
                        (self._make_atom(pred_id, tuple(bridge_vars)),),
                    )
                    self._add_hypothesis(
                        hypotheses,
                        seen,
                        rule,
                        "bridge_goal",
                        (),
                        kb,
                        baseline,
                        starting_facts,
                        target_facts=explanation_targets,
                    )
            except Exception:
                pass

        # ── Шаблон 3b: goal_factorization ────────────────────────────────────
        # P_goal(...) :- P_new(...), P_interact(...)
        if goal is not None:
            goal_arity = len(tuple(getattr(goal, "args", ())))
            for p_a in top_preds[: self.max_interaction_preds]:
                a_arity = int(pred_arity_map.get(int(p_a), goal_arity or arity))
                if goal_arity > max(arity, a_arity):
                    continue
                goal_vars = vars_[:goal_arity]
                new_vars = vars_[:arity]
                aux_vars = vars_[:a_arity]
                try:
                    rule = self._make_rule(
                        self._make_atom(int(goal.pred), tuple(goal_vars)),
                        (
                            self._make_atom(pred_id, tuple(new_vars)),
                            self._make_atom(int(p_a), tuple(aux_vars)),
                        ),
                    )
                    self._add_hypothesis(
                        hypotheses,
                        seen,
                        rule,
                        "goal_factorization",
                        (int(p_a),),
                        kb,
                        baseline,
                        starting_facts,
                        target_facts=explanation_targets,
                    )
                except Exception:
                    continue

        # ── Шаблон 3c: inverse_alias ──────────────────────────────────────────
        # P_interact(...) :- P_new(...) — існуючий предикат як прояв нового концепту
        for p_a in top_preds[: self.max_interaction_preds]:
            a_arity = int(pred_arity_map.get(int(p_a), arity))
            if a_arity > arity:
                continue
            head_vars = vars_[:a_arity]
            body_vars = vars_[:arity]
            try:
                rule = self._make_rule(
                    self._make_atom(int(p_a), tuple(head_vars)),
                    (self._make_atom(pred_id, tuple(body_vars)),),
                )
                self._add_hypothesis(
                    hypotheses,
                    seen,
                    rule,
                    "inverse_alias",
                    (int(p_a),),
                    kb,
                    baseline,
                    starting_facts,
                    target_facts=explanation_targets,
                )
            except Exception:
                continue

        # ── Шаблон 3d/3e: bridge_target / target_factorization ───────────────────────────────
        for target in explanation_targets:
            target_pred = int(getattr(target, "pred", -1))
            target_arity = len(tuple(getattr(target, "args", ())))
            if target_arity <= arity:
                try:
                    rule = self._make_rule(
                        self._make_atom(target_pred, tuple(vars_[:target_arity])),
                        (self._make_atom(pred_id, tuple(vars_[:arity])),),
                    )
                    self._add_hypothesis(
                        hypotheses,
                        seen,
                        rule,
                        "bridge_target",
                        (),
                        kb,
                        baseline,
                        starting_facts,
                        target_facts=explanation_targets,
                    )
                except Exception:
                    pass
            for p_a in top_preds[: self.max_interaction_preds]:
                a_arity = int(pred_arity_map.get(int(p_a), target_arity or arity))
                if target_arity > max(arity, a_arity):
                    continue
                try:
                    rule = self._make_rule(
                        self._make_atom(target_pred, tuple(vars_[:target_arity])),
                        (
                            self._make_atom(pred_id, tuple(vars_[:arity])),
                            self._make_atom(int(p_a), tuple(vars_[:a_arity])),
                        ),
                    )
                    self._add_hypothesis(
                        hypotheses,
                        seen,
                        rule,
                        "target_factorization",
                        (int(p_a),),
                        kb,
                        baseline,
                        starting_facts,
                        target_facts=explanation_targets,
                    )
                except Exception:
                    continue

        # ── Шаблон 4: fact_coverage ──────────────────────────────────────────
        # P_new(X,...) :- P_unexplained(X,...) — пояснення непоясненого факту
        unexplained = [f for f in current_facts if f not in baseline]
        if unexplained:
            uf = unexplained[0]
            f_arity = len(tuple(getattr(uf, "args", ())))
            body_vars = vars_[:f_arity]
            head_vars = vars_[:min(arity, f_arity)]
            try:
                rule = self._make_rule(
                    self._make_atom(pred_id, tuple(head_vars)),
                    (self._make_atom(int(uf.pred), tuple(body_vars)),),
                )
                self._add_hypothesis(
                    hypotheses,
                    seen,
                    rule,
                    "fact_coverage",
                    (int(uf.pred),),
                    kb,
                    baseline,
                    starting_facts,
                    target_facts=explanation_targets,
                )
            except Exception:
                pass

        # Відкритий second-order search: не спирається на фіксований список шаблонів,
        # а перебирає вільні head/body-композиції з P_new та існуючих предикатів.
        self._open_second_order_search(
            hypotheses,
            seen,
            pred_id=pred_id,
            arity=arity,
            top_preds=top_preds,
            pred_arity_map=pred_arity_map,
            explanation_targets=explanation_targets,
            kb=kb,
            baseline=baseline,
            starting_facts=starting_facts,
        )

        self._attach_bundle_scores(
            hypotheses,
            kb,
            baseline,
            starting_facts,
            explanation_targets,
        )
        self._beam_bundle_search(
            hypotheses,
            kb,
            baseline,
            starting_facts,
            explanation_targets,
        )

        # Ранжуємо: argmax [Consistency − λ · Complexity]
        hypotheses.sort(
            key=lambda hyp: (max(hyp.score, hyp.bundle_score), hyp.coverage_gain, hyp.novelty_gain),
            reverse=True,
        )
        if len(hypotheses) <= self.max_hypotheses:
            return hypotheses

        selected: List[_RuleHypothesis] = []
        selected_hashes: set[int] = set()
        seen_templates: set[str] = set()
        for hyp in hypotheses:
            if hyp.template in seen_templates:
                continue
            selected.append(hyp)
            selected_hashes.add(hash(hyp.clause))
            seen_templates.add(hyp.template)
            if len(selected) >= self.max_hypotheses:
                return selected
        for hyp in hypotheses:
            hyp_hash = hash(hyp.clause)
            if hyp_hash in selected_hashes:
                continue
            selected.append(hyp)
            selected_hashes.add(hyp_hash)
            if len(selected) >= self.max_hypotheses:
                break
        return selected[: self.max_hypotheses]


# ─── Буфер зворотного зв'язку ────────────────────────────────────────────────

@dataclass
class _FeedbackRecord:
    """Запис для online learning: збережений вхід context_encoder + рішення VeM."""
    context_input: torch.Tensor   # (embed_dim + 1,)
    gap_norm: float
    accepted: bool                # True = VeM прийняв, False = відхилив


# ─── Головний клас OEE ───────────────────────────────────────────────────────

class OntologyExpansionEngine:
    """
    Повна нейро-символьна реалізація OEE.

    Параметри
    ----------
    gap_threshold            : поріг gap_norm для тригера
    contradiction_threshold  : поріг кількості суперечностей (з CWE)
    max_new_preds            : максимум нових предикатів за виклик
    predicate_start          : початковий ID (900_000 щоб не перетинатись з KB)
    d_latent                 : розмірність h ∈ ℝ^d_latent
    consistency_lambda       : λ у Consistency − λ·Complexity
    online_lr                : lr для онлайн навчання
    feedback_buffer_size     : розмір FIFO-буфера зворотного зв'язку
    train_every_n_calls      : частота кроків online training
    forward_chain_depth      : глибина sandbox forward-chain для Consistency
    max_interaction_preds    : top-k предикатів для взаємодії (cross-attention)
    max_hypotheses           : максимум гіпотез на один новий предикат
    """

    def __init__(
        self,
        gap_threshold: float = 0.45,
        contradiction_threshold: int = 1,
        max_new_preds: int = 1,
        predicate_start: int = 900_000,
        d_latent: int = 32,
        consistency_lambda: float = 0.1,
        online_lr: float = 1e-3,
        feedback_buffer_size: int = 64,
        train_every_n_calls: int = 4,
        forward_chain_depth: int = 2,
        max_interaction_preds: int = 3,
        max_hypotheses: int = 8,
        oee_bundle_beam_width: int = 4,
        oee_max_bundle_rules: int = 3,
        oee_bundle_seed_k: int = 12,
    ) -> None:
        self.gap_threshold = float(max(gap_threshold, 0.0))
        self.contradiction_threshold = max(int(contradiction_threshold), 0)
        self.max_new_preds = max(int(max_new_preds), 1)
        self.d_latent = max(int(d_latent), 8)
        self.consistency_lambda = float(max(consistency_lambda, 1e-6))
        self.online_lr = float(max(online_lr, 1e-6))
        self.feedback_buffer_size = max(int(feedback_buffer_size), 4)
        self.train_every_n_calls = max(int(train_every_n_calls), 1)

        self._next_predicate_id: int = int(predicate_start)
        self._cluster_pressure: Dict[Tuple[int, ...], float] = defaultdict(float)
        self._cluster_counts: Dict[Tuple[int, ...], float] = defaultdict(float)
        self._last_cluster_key: Tuple[int, ...] = tuple()

        # Нейромережева компонента (ліниво ініціалізується при першому виклику)
        self._latent_model: Optional[PredicateLatentModel] = None
        self._latent_model_embed_dim: int = 0
        self._optimizer: Optional[torch.optim.Adam] = None

        # Генератор гіпотез правил
        self._sampler = RuleHypothesisSampler(
            consistency_lambda=consistency_lambda,
            max_interaction_preds=max_interaction_preds,
            max_hypotheses=max_hypotheses,
            forward_chain_depth=forward_chain_depth,
            bundle_beam_width=oee_bundle_beam_width,
            max_bundle_rules=oee_max_bundle_rules,
            bundle_seed_k=oee_bundle_seed_k,
        )

        # Буфер для online learning
        self._feedback_buffer: Deque[_FeedbackRecord] = deque(maxlen=feedback_buffer_size)
        self._call_count: int = 0
        self._online_train_steps: int = 0
        self._last_online_train_applied: float = 0.0
        self._last_online_train_loss: float = 0.0
        self._last_online_train_buffer_size: float = 0.0

        # Стан останнього виклику (для record_feedback)
        self._last_context_input: Optional[torch.Tensor] = None
        self._last_gap_norm: float = 0.0
        self._last_pred_ids: List[int] = []
        self._last_h: Optional[torch.Tensor] = None

        # Статистика прийнятих предикатів
        self._accepted_pred_ids: List[int] = []
        self._predicate_vocab: Dict[int, PredicateVocabEntry] = {}

    # ─── Ініціалізація моделі ─────────────────────────────────────────────────

    def _ensure_model(self, embed_dim: int) -> PredicateLatentModel:
        """Ліниво будує або перебудовує модель при зміні embed_dim."""
        if self._latent_model is None or self._latent_model_embed_dim != embed_dim:
            self._latent_model = PredicateLatentModel(
                embed_dim=embed_dim,
                d_latent=self.d_latent,
            )
            self._latent_model_embed_dim = embed_dim
            self._optimizer = torch.optim.Adam(
                self._latent_model.parameters(),
                lr=self.online_lr,
                weight_decay=1e-5,
            )
            self._latent_model.eval()
        return self._latent_model

    def _register_predicate(
        self,
        pred_id: int,
        arity: int,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pred_id = int(pred_id)
        arity = max(int(arity), 1)
        current = self._predicate_vocab.get(pred_id)
        merged_meta = dict(current.metadata) if current is not None else {}
        merged_meta.update(dict(metadata or {}))
        self._predicate_vocab[pred_id] = PredicateVocabEntry(
            pred_id=pred_id,
            arity=arity,
            status=current.status if current is not None else "proposed",
            confidence=float(max(confidence, current.confidence if current is not None else 0.0)),
            gap_before=float(current.gap_before if current is not None else 0.0),
            gap_after=float(current.gap_after if current is not None else 0.0),
            supporting_rules=tuple(current.supporting_rules) if current is not None else tuple(),
            metadata=merged_meta,
        )

    def fixed_entries(self, pred_ids: Optional[Sequence[int]] = None) -> List[PredicateVocabEntry]:
        if pred_ids is None:
            entries = self._predicate_vocab.values()
        else:
            unique_ids = {int(pred_id) for pred_id in pred_ids}
            entries = [self._predicate_vocab[pred_id] for pred_id in unique_ids if pred_id in self._predicate_vocab]
        return [entry for entry in entries if entry.status == "fixed"]

    def predicate_vocab(self) -> Dict[int, PredicateVocabEntry]:
        return dict(self._predicate_vocab)

    # ─── Спостереження кластерного тиску ─────────────────────────────────────

    def observe_gap_cluster(
        self,
        gap_norm: float,
        hot_dims: Sequence[int],
        z: Optional[torch.Tensor] = None,
    ) -> None:
        """Накопичує gap_norm по кластерах латентного простору стану z."""
        if hot_dims:
            cluster = tuple(sorted(int(dim) for dim in hot_dims[:4]))
        elif z is not None and z.numel() > 0:
            pooled = z.detach().mean(dim=0).abs().flatten().cpu()
            top_k = min(4, int(pooled.numel()))
            if top_k > 0:
                cluster = tuple(sorted(int(idx) for idx in pooled.topk(top_k).indices.tolist()))
            else:
                cluster = tuple()
        else:
            cluster = tuple()
        self._last_cluster_key = cluster
        self._cluster_pressure[cluster] += float(gap_norm)
        self._cluster_counts[cluster] += 1.0

    # ─── Зворотний зв'язок від VeM ────────────────────────────────────────────

    def record_feedback(
        self,
        accepted: bool,
        *,
        accepted_pred_ids: Optional[Sequence[int]] = None,
        gap_before: Optional[float] = None,
        gap_after: Optional[float] = None,
        supporting_rules: Optional[Sequence[Any]] = None,
    ) -> None:
        """
        Реєструє результат VeM для останнього виклику generate_candidates.

        Зберігає context_input у буфер для online learning.
        Градієнти течуть через context_encoder → h_decoder під час тренування,
        оскільки ми ре-форвардимо збережені вхідні вектори через поточні параметри.
        """
        if self._last_context_input is not None:
            self._feedback_buffer.append(_FeedbackRecord(
                context_input=self._last_context_input.detach().clone(),
                gap_norm=self._last_gap_norm,
                accepted=accepted,
            ))
        pred_ids = list(accepted_pred_ids) if accepted_pred_ids is not None else list(self._last_pred_ids)
        gap_before_f = float(self._last_gap_norm if gap_before is None else gap_before)
        gap_after_f = float(gap_before_f if gap_after is None else gap_after)
        rule_hashes = tuple(hash(rule) for rule in (supporting_rules or ()))
        for pred_id in pred_ids:
            pred_id = int(pred_id)
            entry = self._predicate_vocab.get(pred_id)
            if entry is None:
                continue
            entry_meta = dict(entry.metadata)
            entry_meta["feedback_accepted"] = 1.0 if accepted else 0.0
            entry_meta["feedback_count"] = float(entry_meta.get("feedback_count", 0.0) + 1.0)
            if accepted and gap_after_f + 1e-6 < gap_before_f:
                status = "fixed"
                if pred_id not in self._accepted_pred_ids:
                    self._accepted_pred_ids.append(pred_id)
            elif accepted:
                status = entry.status if entry.status == "fixed" else "proposed"
            else:
                status = "rejected"
            self._predicate_vocab[pred_id] = PredicateVocabEntry(
                pred_id=entry.pred_id,
                arity=entry.arity,
                status=status,
                confidence=entry.confidence,
                gap_before=gap_before_f,
                gap_after=gap_after_f,
                supporting_rules=rule_hashes or entry.supporting_rules,
                metadata=entry_meta,
            )

    # ─── Online learning ──────────────────────────────────────────────────────

    def _online_train_step(self, model: PredicateLatentModel) -> float:
        """
        Один крок онлайн навчання на буфері зворотного зв'язку.

        Контрастивний лос на h-просторі:
          · Прийняті правила: h_acc_i · h_acc_j > 0.5  (кластеризація)
          · Прийняті vs відхилені: h_acc · h_rej < -0.2 (відштовхування)

        Градієнти течуть через context_encoder + h_decoder, оскільки
        ми ре-форвардимо збережені context_input вектори через поточну модель.
        """
        self._last_online_train_applied = 0.0
        self._last_online_train_loss = 0.0
        self._last_online_train_buffer_size = float(len(self._feedback_buffer))
        if len(self._feedback_buffer) < 4 or self._optimizer is None:
            return 0.0

        records = list(self._feedback_buffer)
        accepted_recs = [r for r in records if r.accepted]
        rejected_recs = [r for r in records if not r.accepted]

        if not accepted_recs:
            return 0.0

        model.train()
        loss_val = 0.0
        try:
            acc_inp = torch.stack([r.context_input for r in accepted_recs])  # (n_acc, D+1)
            acc_ctx = model.context_encoder(acc_inp)    # (n_acc, d_latent)
            acc_h   = model.h_decoder(acc_ctx)          # (n_acc, d_latent)
            acc_n   = F.normalize(acc_h, dim=-1)

            loss = torch.zeros(1, requires_grad=True).squeeze()

            # Позитивні пари: прийняті h мають кластеризуватись (sim > 0.5)
            if len(accepted_recs) >= 2:
                sim = acc_n @ acc_n.T                   # (n_acc, n_acc)
                eye = torch.eye(len(accepted_recs), dtype=torch.bool)
                off = sim.masked_fill(eye, 1.0)
                loss = loss + (0.5 - off).clamp(min=0.0).mean()

            # Негативні пари: прийняті vs відхилені (sim < -0.2)
            if rejected_recs:
                n_neg = min(len(rejected_recs), len(accepted_recs))
                rej_inp = torch.stack([r.context_input for r in rejected_recs[:n_neg]])
                rej_ctx = model.context_encoder(rej_inp)
                rej_h   = model.h_decoder(rej_ctx)
                rej_n   = F.normalize(rej_h, dim=-1)
                neg_sim = (acc_n[:n_neg] * rej_n).sum(dim=-1)
                loss = loss + (neg_sim + 0.2).clamp(min=0.0).mean()

            if loss.requires_grad:
                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self._optimizer.step()
                loss_val = float(loss.item())
                self._online_train_steps += 1
                self._last_online_train_applied = 1.0
                self._last_online_train_loss = loss_val

        except Exception:
            pass
        finally:
            model.eval()

        return loss_val

    # ─── Генерація нових ID предикатів ───────────────────────────────────────

    def _new_predicate_id(self) -> int:
        pred = self._next_predicate_id
        self._next_predicate_id += 1
        return pred

    # ─── Публічний API ───────────────────────────────────────────────────────

    def generate_candidates(
        self,
        current_facts: Sequence[Any],
        goal: Optional[Any],
        gap_norm: float,
        contradiction_count: int,
        max_candidates: int = 4,
        paradox_facts: Optional[Sequence[Any]] = None,
        target_facts: Optional[Sequence[Any]] = None,
        state_z: Optional[torch.Tensor] = None,
        # Нові параметри для нейромережевого пошуку (з AME / CreativeCycleCoordinator)
        predicate_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        kb: Optional[Any] = None,
        existing_rules: Optional[Sequence[Any]] = None,
    ) -> List[RuleCandidate]:
        """
        Генерує кандидатні правила з новим предикатом.

        Тригер: gap_norm ≥ gap_threshold АБО contradiction_count ≥ threshold.

        Режими:
          Neural  (повний): predicate_embeddings + kb → PredicateLatentModel → argmax
          Heuristic (fallback): evристика якщо ембеддинги недоступні
        """
        self._call_count += 1
        self._last_online_train_applied = 0.0
        self._last_online_train_loss = 0.0
        self._last_online_train_buffer_size = float(len(self._feedback_buffer))

        cluster_sum = float(self._cluster_pressure.get(self._last_cluster_key, 0.0))
        cluster_count = float(self._cluster_counts.get(self._last_cluster_key, 0.0))
        active_cluster_pressure = cluster_sum / max(cluster_count, 1.0)
        if (
            gap_norm < self.gap_threshold
            and contradiction_count < self.contradiction_threshold
            and active_cluster_pressure < self.gap_threshold
        ):
            return []

        # Online training кожен train_every_n_calls виклик
        if (
            self._latent_model is not None
            and self._call_count % self.train_every_n_calls == 0
            and len(self._feedback_buffer) >= 4
        ):
            self._online_train_step(self._latent_model)

        fact_pool: List[Any] = list(current_facts)
        for fact in paradox_facts or ():
            if all(hash(fact) != hash(existing) for existing in fact_pool):
                fact_pool.append(fact)
        explanation_targets: List[Any] = []
        seen_targets: set[int] = set()
        if goal is not None:
            explanation_targets.append(goal)
            seen_targets.add(hash(goal))
        for fact in target_facts or ():
            fact_hash = hash(fact)
            if fact_hash in seen_targets:
                continue
            explanation_targets.append(fact)
            seen_targets.add(fact_hash)
        for fact in paradox_facts or ():
            fact_hash = hash(fact)
            if fact_hash in seen_targets:
                continue
            explanation_targets.append(fact)
            seen_targets.add(fact_hash)

        if predicate_embeddings and kb is not None:
            return self._neural_generate(
                current_facts=current_facts,
                context_facts=fact_pool,
                goal=goal,
                target_facts=explanation_targets,
                gap_norm=gap_norm,
                contradiction_count=contradiction_count,
                max_candidates=max_candidates,
                predicate_embeddings=predicate_embeddings,
                kb=kb,
                existing_rules=list(existing_rules or []),
                state_z=state_z,
            )

        return self._heuristic_generate(
            current_facts=current_facts,
            context_facts=fact_pool,
            goal=goal,
            target_facts=explanation_targets,
            gap_norm=gap_norm,
            contradiction_count=contradiction_count,
            max_candidates=max_candidates,
        )

    # ─── Нейромережевий режим ─────────────────────────────────────────────────

    def _neural_generate(
        self,
        current_facts: Sequence[Any],
        context_facts: Sequence[Any],
        goal: Optional[Any],
        target_facts: Sequence[Any],
        gap_norm: float,
        contradiction_count: int,
        max_candidates: int,
        predicate_embeddings: Dict[int, torch.Tensor],
        kb: Any,
        existing_rules: List[Any],
        state_z: Optional[torch.Tensor],
    ) -> List[RuleCandidate]:
        """
        Повний нейро-символьний пошук P_new:

        1. Матриця ембеддингів з AME predicate_embeddings.
        2. PredicateLatentModel → h, arity_logits, interaction_scores.
           h відображає "ідеальну роль" нового предиката в KB.
        3. Top-k взаємодіючих предикатів через cross-attention.
        4. RuleHypothesisSampler → шаблони + оцінка argmax [C − λ·MDL].
        5. RuleCandidate з повними метаданими → AEE для еволюційного відбору.
        """
        pred_ids = list(predicate_embeddings.keys())
        if not pred_ids:
            return self._heuristic_generate(
                current_facts,
                context_facts,
                goal,
                target_facts,
                gap_norm,
                contradiction_count,
                max_candidates,
            )

        embed_dim = next(iter(predicate_embeddings.values())).shape[-1]
        model = self._ensure_model(embed_dim)

        emb_tensor = torch.stack(
            [predicate_embeddings[p].detach() for p in pred_ids], dim=0
        )  # (n_pred, embed_dim)

        # Фокусуємо модель на предикатах з поточного непоясненого контексту, а
        # кластерний тиск z лишається окремим тригером для запуску OEE.
        focus_preds: set[int] = {int(getattr(fact, "pred", -1)) for fact in context_facts}
        if goal is not None:
            focus_preds.add(int(getattr(goal, "pred", -1)))
        for fact in target_facts:
            focus_preds.add(int(getattr(fact, "pred", -1)))
        unexplained_mask: Optional[torch.Tensor] = (
            torch.tensor([p in focus_preds for p in pred_ids], dtype=torch.bool)
            if focus_preds else None
        )

        # Forward pass: PredicateLatentModel
        model.eval()
        with torch.no_grad():
            h, arity_logits, interaction_scores, confidence, context_input = model(
                emb_tensor, gap_norm, unexplained_mask, state_z=state_z
            )

        # Зберігаємо стан для record_feedback
        self._last_context_input = context_input.detach()
        self._last_gap_norm = gap_norm
        self._last_h = h.detach()

        # Арність: 1, 2 або 3
        arity = int(arity_logits.argmax().item()) + 1

        # Top-k взаємодіючих предикатів (cross-attention output)
        n_top = min(self._sampler.max_interaction_preds, len(pred_ids))
        if interaction_scores.numel() >= n_top:
            top_idx = interaction_scores.topk(n_top).indices.tolist()
        else:
            top_idx = list(range(len(pred_ids)))
        top_preds = [pred_ids[i] for i in top_idx]
        top_scores = [float(interaction_scores[i].item()) for i in top_idx]

        all_candidates: List[RuleCandidate] = []
        new_pred_ids: List[int] = []
        cluster_sum = float(self._cluster_pressure.get(self._last_cluster_key, 0.0))
        cluster_count = float(self._cluster_counts.get(self._last_cluster_key, 0.0))
        cluster_pressure = cluster_sum / max(cluster_count, 1.0)
        n_to_gen = min(self.max_new_preds, max(max_candidates // 2, 1))

        for _ in range(n_to_gen):
            pred_id = self._new_predicate_id()
            new_pred_ids.append(pred_id)
            self._register_predicate(
                pred_id,
                arity=arity,
                confidence=float(confidence.squeeze().item()),
                metadata={
                    "latent_h_norm": float(h.norm().item()),
                    "interaction_preds": tuple(int(pred) for pred in top_preds),
                    "created_call": float(self._call_count),
                },
            )

            # argmax [Consistency − λ · Complexity] по шаблонах
            hypotheses = self._sampler.sample(
                pred_id=pred_id,
                arity=arity,
                interaction_preds=top_preds,
                interaction_scores=top_scores,
                current_facts=current_facts,
                kb=kb,
                goal=goal,
                target_facts=target_facts,
            )

            for hyp in hypotheses:
                # Загальна оцінка кандидата: gap-сигнал + argmax-оцінка гіпотези
                effective_hypothesis_score = max(hyp.score, hyp.bundle_score)
                candidate_score = float(gap_norm + contradiction_count) + max(effective_hypothesis_score, 0.0)
                all_candidates.append(RuleCandidate(
                    clause=hyp.clause,
                    source="ontology",
                    score=candidate_score,
                    metadata={
                        "invented_predicate": float(pred_id),
                        "latent_h_norm": float(h.norm().item()),
                        "model_confidence": float(confidence.squeeze().item()),
                        "cluster_pressure": float(cluster_pressure),
                        "state_z_guided": 1.0 if state_z is not None and state_z.numel() > 0 else 0.0,
                        "consistency": float(hyp.consistency),
                        "coverage_gain": float(hyp.coverage_gain),
                        "novelty_gain": float(hyp.novelty_gain),
                        "complexity": float(hyp.complexity),
                        "hypothesis_score": float(hyp.score),
                        "effective_hypothesis_score": float(effective_hypothesis_score),
                        "bundle_score": float(hyp.bundle_score),
                        "bundle_size": float(hyp.bundle_size),
                        "bundle_templates": ",".join(hyp.bundle_templates),
                        "arity": float(arity),
                        "interaction_preds": str(list(hyp.interaction_preds)),
                        "template": hyp.template,
                        "neural_guided": 1.0,
                    },
                ))

        self._last_pred_ids = new_pred_ids

        # Фінальний відбір: argmax hypothesis_score (Consistency − λ·Complexity)
        all_candidates.sort(
            key=lambda c: float(c.metadata.get("effective_hypothesis_score", 0.0)),
            reverse=True,
        )
        return all_candidates[:max_candidates]

    # ─── Евристичний fallback ─────────────────────────────────────────────────

    def _heuristic_generate(
        self,
        current_facts: Sequence[Any],
        context_facts: Sequence[Any],
        goal: Optional[Any],
        target_facts: Sequence[Any],
        gap_norm: float,
        contradiction_count: int,
        max_candidates: int,
    ) -> List[RuleCandidate]:
        """
        Евристичний генератор (збережено для зворотної сумісності).
        Використовується коли AME ембеддинги недоступні.
        """
        from omen_prolog import HornClause, Var

        def _abstract(atom: Any) -> Any:
            seen: Dict[str, Any] = {}
            abs_args = []
            for idx, arg in enumerate(getattr(atom, "args", ())):
                key = repr(arg)
                if key not in seen:
                    seen[key] = Var(f"X{idx}")
                abs_args.append(seen[key])
            return type(atom)(pred=int(atom.pred), args=tuple(abs_args))

        self._last_context_input = None
        self._last_pred_ids = []
        candidates: List[RuleCandidate] = []

        for anchor in list(current_facts)[: self.max_new_preds]:
            pred_id = self._new_predicate_id()
            anchor_arity = len(tuple(getattr(anchor, "args", ())))
            self._register_predicate(
                pred_id,
                arity=max(anchor_arity, 1),
                confidence=0.5,
                metadata={
                    "created_call": float(self._call_count),
                    "heuristic_anchor_pred": float(int(getattr(anchor, "pred", 0))),
                },
            )
            abstract_anchor = _abstract(anchor)
            invented_head = type(anchor)(pred=pred_id, args=tuple(abstract_anchor.args))
            primary_rule = HornClause(
                head=invented_head,
                body=(abstract_anchor,),
                weight=1.0,
                use_count=0,
            )
            candidates.append(RuleCandidate(
                clause=primary_rule,
                source="ontology",
                score=float(gap_norm) + float(contradiction_count),
                metadata={
                    "invented_predicate": float(pred_id),
                    "neural_guided": 0.0,
                    "template": "heuristic_anchor",
                },
            ))
            self._last_pred_ids.append(pred_id)

            bridge_targets: List[Any] = []
            seen_bridge_targets: set[int] = set()
            if goal is not None:
                bridge_targets.append(goal)
                seen_bridge_targets.add(hash(goal))
            for target in target_facts:
                target_hash = hash(target)
                if target_hash in seen_bridge_targets:
                    continue
                bridge_targets.append(target)
                seen_bridge_targets.add(target_hash)

            for target in bridge_targets:
                target_arity = len(tuple(getattr(target, "args", ())))
                if target_arity > len(invented_head.args):
                    continue
                bridge_head = type(target)(
                    pred=int(getattr(target, "pred", -1)),
                    args=tuple(invented_head.args[:target_arity]),
                )
                bridge_rule = HornClause(
                    head=bridge_head,
                    body=(invented_head,),
                    weight=1.0,
                    use_count=0,
                )
                template = "heuristic_bridge" if goal is not None and hash(target) == hash(goal) else "heuristic_target_bridge"
                candidates.append(RuleCandidate(
                    clause=bridge_rule,
                    source="ontology",
                    score=0.75 * float(gap_norm) + float(contradiction_count),
                    metadata={
                        "invented_predicate": float(pred_id),
                        "neural_guided": 0.0,
                        "template": template,
                    },
                ))

        return candidates[:max_candidates]

    # ─── Статистика ───────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, float]:
        buf = list(self._feedback_buffer)
        accepted_n = sum(1 for r in buf if r.accepted)
        vocab = list(self._predicate_vocab.values())
        fixed_n = sum(1 for entry in vocab if entry.status == "fixed")
        proposed_n = sum(1 for entry in vocab if entry.status == "proposed")
        rejected_n = sum(1 for entry in vocab if entry.status == "rejected")
        gap_reductions = [
            max(float(entry.gap_before) - float(entry.gap_after), 0.0)
            for entry in vocab
            if entry.status == "fixed"
        ]
        return {
            "oee_next_pred_id": float(self._next_predicate_id),
            "oee_accepted_preds": float(len(self._accepted_pred_ids)),
            "oee_feedback_buffer_size": float(len(buf)),
            "oee_feedback_accepted_ratio": float(accepted_n / max(len(buf), 1)),
            "oee_cluster_pressure_n": float(len(self._cluster_pressure)),
            "oee_active_cluster_pressure": float(
                self._cluster_pressure.get(self._last_cluster_key, 0.0)
                / max(self._cluster_counts.get(self._last_cluster_key, 0.0), 1.0)
            ),
            "oee_model_initialized": float(self._latent_model is not None),
            "oee_call_count": float(self._call_count),
            "oee_online_train_applied": float(self._last_online_train_applied),
            "oee_online_train_loss": float(self._last_online_train_loss),
            "oee_online_train_steps": float(self._online_train_steps),
            "oee_online_train_buffer_size": float(self._last_online_train_buffer_size),
            "oee_predicate_vocab_size": float(len(vocab)),
            "oee_fixed_preds": float(fixed_n),
            "oee_proposed_preds": float(proposed_n),
            "oee_rejected_preds": float(rejected_n),
            "oee_mean_gap_reduction": float(sum(gap_reductions) / max(len(gap_reductions), 1)),
            "oee_bundle_beam_width": float(self._sampler.bundle_beam_width),
            "oee_max_bundle_rules": float(self._sampler.max_bundle_rules),
            "oee_bundle_seed_k": float(self._sampler.bundle_seed_k),
            "oee_beam_states": float(self._sampler._last_beam_states),
            "oee_beam_best_bundle_size": float(self._sampler._last_beam_best_size),
            "oee_beam_best_bundle_score": float(self._sampler._last_beam_best_score),
        }

    # ─── Публічні властивості ─────────────────────────────────────────────────

    @property
    def last_h(self) -> Optional[torch.Tensor]:
        """Останній латентний вектор h (для діагностики / ICE)."""
        return self._last_h

    @property
    def last_gap_norm(self) -> float:
        return self._last_gap_norm

    @property
    def last_pred_ids(self) -> List[int]:
        return list(self._last_pred_ids)

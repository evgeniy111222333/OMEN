from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import math
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from omen_symbolic.aesthetic_engine import AestheticEvolutionEngine
from omen_symbolic.analogy_engine import AnalogyMetaphorEngine
from omen_symbolic.counterfactual_engine import CounterfactualWorldEngine
from omen_symbolic.creative_types import CreativeCycleReport, OntologyPredicateState, RuleCandidate
from omen_symbolic.intrinsic_engine import IntrinsicCuriosityEngine
from omen_symbolic.ontology_engine import OntologyExpansionEngine


class CreativeCycleCoordinator:
    def __init__(
        self,
        *,
        enabled: bool = True,
        cycle_every: int = 4,
        max_selected_rules: int = 2,
        # AME / HypergraphGNN params
        analogy_dim: int = 32,
        tau_analogy: float = 0.82,
        tau_metaphor: Optional[float] = None,
        analogy_hidden_dim: int = 64,
        analogy_gnn_layers: int = 2,
        analogy_spec_ratio: float = 0.5,
        analogy_temperature: float = 0.07,
        analogy_contrastive_steps: int = 4,
        analogy_contrastive_lr: float = 3e-3,
        analogy_dropout: float = 0.10,
        # CWE params
        cwe_max_rule_mods: int = 2,
        cwe_surprise_lambda: float = 0.5,
        cwe_max_candidates: int = 8,
        cwe_max_transforms_per_rule: int = 4,
        # AEE params
        aee_population: int = 16,
        aee_generations: int = 2,
        aee_gamma: float = 0.25,
        aee_mutation_rate: float = 0.35,
        aee_crossover_rate: float = 0.5,
        aee_ltm_seed_ratio: float = 0.35,
        aee_gene_pool_size: int = 32,
        # OEE params
        oee_gap_threshold: float = 0.45,
        oee_contradiction_threshold: int = 1,
        oee_d_latent: int = 32,
        oee_consistency_lambda: float = 0.1,
        oee_online_lr: float = 1e-3,
        oee_forward_chain_depth: int = 2,
        oee_max_interaction_preds: int = 3,
        oee_max_hypotheses: int = 8,
        oee_bundle_beam_width: int = 4,
        oee_max_bundle_rules: int = 3,
        oee_bundle_seed_k: int = 12,
        train_fast_cwe_max_rule_mods: int = 1,
        train_fast_cwe_max_candidates: int = 2,
        train_fast_cwe_max_transforms_per_rule: int = 1,
        train_fast_oee_max_candidates: int = 2,
        train_fast_oee_max_targets: int = 2,
        train_fast_oee_max_paradox_facts: int = 2,
        train_fast_oee_max_hypotheses: int = 4,
        train_fast_oee_max_scored_hypotheses: int = 32,
        train_fast_oee_max_open_body_literals: int = 1,
        train_fast_oee_max_open_patterns: int = 2,
        train_fast_oee_max_open_head_patterns: int = 2,
        train_fast_oee_bundle_beam_width: int = 2,
        train_fast_oee_max_bundle_rules: int = 2,
        train_fast_oee_bundle_seed_k: int = 4,
        # ICE params
        ice_state_history: int = 128,
        ice_goal_threshold: float = 0.35,
    ):
        self.enabled = bool(enabled)
        self.cycle_every = max(int(cycle_every), 1)
        self.max_selected_rules = max(int(max_selected_rules), 1)
        self.train_fast_cwe_max_rule_mods = max(int(train_fast_cwe_max_rule_mods), 1)
        self.train_fast_cwe_max_candidates = max(int(train_fast_cwe_max_candidates), 1)
        self.train_fast_cwe_max_transforms_per_rule = max(int(train_fast_cwe_max_transforms_per_rule), 1)
        self.train_fast_oee_max_candidates = max(int(train_fast_oee_max_candidates), 1)
        self.train_fast_oee_max_targets = max(int(train_fast_oee_max_targets), 1)
        self.train_fast_oee_max_paradox_facts = max(int(train_fast_oee_max_paradox_facts), 0)
        self.train_fast_oee_max_hypotheses = max(int(train_fast_oee_max_hypotheses), 1)
        self.train_fast_oee_max_scored_hypotheses = max(
            int(train_fast_oee_max_scored_hypotheses),
            self.train_fast_oee_max_hypotheses,
        )
        self.train_fast_oee_max_open_body_literals = max(int(train_fast_oee_max_open_body_literals), 1)
        self.train_fast_oee_max_open_patterns = max(int(train_fast_oee_max_open_patterns), 1)
        self.train_fast_oee_max_open_head_patterns = max(int(train_fast_oee_max_open_head_patterns), 1)
        self.train_fast_oee_bundle_beam_width = max(int(train_fast_oee_bundle_beam_width), 2)
        self.train_fast_oee_max_bundle_rules = max(int(train_fast_oee_max_bundle_rules), 2)
        self.train_fast_oee_bundle_seed_k = max(
            int(train_fast_oee_bundle_seed_k),
            self.train_fast_oee_bundle_beam_width,
        )
        self.analogy_engine = AnalogyMetaphorEngine(
            embedding_dim=analogy_dim,
            tau_analogy=tau_analogy,
            tau_metaphor=tau_metaphor,
            hidden_dim=analogy_hidden_dim,
            n_gnn_layers=analogy_gnn_layers,
            spec_ratio=analogy_spec_ratio,
            temperature=analogy_temperature,
            contrastive_steps=analogy_contrastive_steps,
            contrastive_lr=analogy_contrastive_lr,
            dropout=analogy_dropout,
        )
        self.counterfactual_engine = CounterfactualWorldEngine(
            max_rule_mods=cwe_max_rule_mods,
            surprise_lambda=cwe_surprise_lambda,
            max_candidates=cwe_max_candidates,
            max_transforms_per_rule=cwe_max_transforms_per_rule,
        )
        self.aesthetic_engine = AestheticEvolutionEngine(
            population_size=aee_population,
            generations=aee_generations,
            gamma=aee_gamma,
            mutation_rate=aee_mutation_rate,
            crossover_rate=aee_crossover_rate,
            max_selected=max_selected_rules,
            ltm_seed_ratio=aee_ltm_seed_ratio,
            gene_pool_size=aee_gene_pool_size,
        )
        self.ontology_engine = OntologyExpansionEngine(
            gap_threshold=oee_gap_threshold,
            contradiction_threshold=oee_contradiction_threshold,
            d_latent=oee_d_latent,
            consistency_lambda=oee_consistency_lambda,
            online_lr=oee_online_lr,
            forward_chain_depth=oee_forward_chain_depth,
            max_interaction_preds=oee_max_interaction_preds,
            max_hypotheses=oee_max_hypotheses,
            max_scored_hypotheses=max(max(oee_max_hypotheses * 8, 64), oee_max_hypotheses),
            oee_bundle_beam_width=oee_bundle_beam_width,
            oee_max_bundle_rules=oee_max_bundle_rules,
            oee_bundle_seed_k=oee_bundle_seed_k,
        )
        self.intrinsic_engine = IntrinsicCuriosityEngine(
            state_history=ice_state_history,
            goal_threshold=ice_goal_threshold,
        )
        self.last_report = CreativeCycleReport()
        self.rule_origins: Dict[str, str] = {}
        self._runtime_derived_facts_cache: Optional[Dict[Tuple[int, Tuple[int, ...]], Any]] = None
        self._runtime_task_gap_cache: Optional[Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], float]] = None

    @staticmethod
    def _cpu_report(report: CreativeCycleReport) -> CreativeCycleReport:
        return replace(
            report,
            predicate_embeddings={
                int(pred_id): emb.detach().cpu().clone()
                for pred_id, emb in dict(report.predicate_embeddings).items()
            },
        )

    def export_state(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "cycle_every": int(self.cycle_every),
            "max_selected_rules": int(self.max_selected_rules),
            "last_report": pickle.dumps(self._cpu_report(self.last_report)),
            "rule_origins": dict(self.rule_origins),
            "analogy_engine": self.analogy_engine.export_state(),
            "aesthetic_engine": self.aesthetic_engine.export_state(),
            "ontology_engine": self.ontology_engine.export_state(),
            "intrinsic_engine": self.intrinsic_engine.export_state(),
        }

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        state = state or {}
        self.enabled = bool(state.get("enabled", self.enabled))
        self.cycle_every = max(int(state.get("cycle_every", self.cycle_every)), 1)
        self.max_selected_rules = max(
            int(state.get("max_selected_rules", self.max_selected_rules)),
            1,
        )
        self.rule_origins = {
            str(rule_key): str(origin)
            for rule_key, origin in dict(state.get("rule_origins", {})).items()
        }
        last_report = state.get("last_report")
        if isinstance(last_report, bytes):
            self.last_report = pickle.loads(last_report)
        elif last_report is not None:
            self.last_report = last_report
        self.analogy_engine.load_state(state.get("analogy_engine"))
        self.aesthetic_engine.load_state(state.get("aesthetic_engine"))
        self.ontology_engine.load_state(state.get("ontology_engine"))
        self.intrinsic_engine.load_state(state.get("intrinsic_engine"))

    def configure(self, **kwargs: Any) -> None:
        if "enabled" in kwargs:
            self.enabled = bool(kwargs["enabled"])
        if "cycle_every" in kwargs:
            self.cycle_every = max(int(kwargs["cycle_every"]), 1)

    def current_intrinsic_goal(self) -> Optional[Any]:
        pending = self.intrinsic_engine.pending_goal
        if pending is not None:
            return pending.goal
        queued = self.intrinsic_engine.scheduled_goals()
        if queued:
            return queued[0].goal
        return None

    def current_intrinsic_value(self) -> float:
        pending = self.intrinsic_engine.pending_goal
        if pending is not None:
            return float(pending.value)
        queued = self.intrinsic_engine.scheduled_goals()
        if queued:
            return float(queued[0].value)
        return 0.0

    def scheduled_intrinsic_goals(self) -> Tuple[Any, ...]:
        return tuple(goal.goal for goal in self.intrinsic_engine.scheduled_goals() if goal.goal is not None)

    def _clear_runtime_caches(self) -> None:
        self._runtime_derived_facts_cache = None
        self._runtime_task_gap_cache = None

    @staticmethod
    def _runtime_rule_key(extra_rules: Sequence[Any]) -> Tuple[str, ...]:
        return tuple(sorted(repr(rule) for rule in extra_rules))

    @staticmethod
    def _runtime_target_key(target_facts: Sequence[Any]) -> Tuple[str, ...]:
        return tuple(sorted(repr(target) for target in target_facts))

    @staticmethod
    def _dedupe_facts(facts: Sequence[Any]) -> List[Any]:
        unique: List[Any] = []
        seen: set[Any] = set()
        for fact in facts:
            if fact in seen:
                continue
            seen.add(fact)
            unique.append(fact)
        return unique

    @staticmethod
    def _report_support_facts(report: Optional[CreativeCycleReport]) -> Tuple[Any, ...]:
        if report is None:
            return tuple()
        support: List[Any] = []
        seen: set[Any] = set()
        for candidate in list(report.selected_rules)[:4]:
            head = getattr(getattr(candidate, "clause", None), "head", None)
            if head is None:
                continue
            if head in seen:
                continue
            seen.add(head)
            support.append(head)
        for fact in list(getattr(report, "counterfactual_novel_facts", ()))[:4]:
            if fact in seen:
                continue
            seen.add(fact)
            support.append(fact)
        for fact in list(getattr(report, "validated_support_facts", ()))[:4]:
            if fact in seen:
                continue
            seen.add(fact)
            support.append(fact)
        intrinsic_goal = getattr(getattr(report, "intrinsic_goal", None), "goal", None)
        if intrinsic_goal is not None:
            if intrinsic_goal not in seen:
                support.append(intrinsic_goal)
        return tuple(support)

    @staticmethod
    def _limit_unique_facts(facts: Sequence[Any], limit: int) -> Tuple[Any, ...]:
        if limit <= 0:
            return tuple()
        unique: List[Any] = []
        seen: set[Any] = set()
        for fact in facts:
            if fact in seen:
                continue
            seen.add(fact)
            unique.append(fact)
            if len(unique) >= limit:
                break
        return tuple(unique)

    @contextmanager
    def _train_fast_symbolic_budget(self):
        sampler = self.ontology_engine._sampler
        saved = (
            self.counterfactual_engine.max_rule_mods,
            self.counterfactual_engine.max_candidates,
            self.counterfactual_engine.max_transforms_per_rule,
            sampler.max_hypotheses,
            sampler.max_scored_hypotheses,
            sampler.max_open_body_literals,
            sampler.max_open_patterns,
            sampler.max_open_head_patterns,
            sampler.bundle_beam_width,
            sampler.max_bundle_rules,
            sampler.bundle_seed_k,
        )
        self.counterfactual_engine.max_rule_mods = self.train_fast_cwe_max_rule_mods
        self.counterfactual_engine.max_candidates = self.train_fast_cwe_max_candidates
        self.counterfactual_engine.max_transforms_per_rule = self.train_fast_cwe_max_transforms_per_rule
        sampler.max_hypotheses = min(sampler.max_hypotheses, self.train_fast_oee_max_hypotheses)
        sampler.max_scored_hypotheses = min(
            sampler.max_scored_hypotheses,
            self.train_fast_oee_max_scored_hypotheses,
        )
        sampler.max_open_body_literals = min(
            sampler.max_open_body_literals,
            self.train_fast_oee_max_open_body_literals,
        )
        sampler.max_open_patterns = min(
            sampler.max_open_patterns,
            self.train_fast_oee_max_open_patterns,
        )
        sampler.max_open_head_patterns = min(
            sampler.max_open_head_patterns,
            self.train_fast_oee_max_open_head_patterns,
        )
        sampler.bundle_beam_width = min(
            sampler.bundle_beam_width,
            self.train_fast_oee_bundle_beam_width,
        )
        sampler.max_bundle_rules = min(
            sampler.max_bundle_rules,
            self.train_fast_oee_max_bundle_rules,
        )
        sampler.bundle_seed_k = min(
            sampler.bundle_seed_k,
            self.train_fast_oee_bundle_seed_k,
        )
        sampler.bundle_seed_k = max(sampler.bundle_seed_k, sampler.bundle_beam_width)
        try:
            yield
        finally:
            (
                self.counterfactual_engine.max_rule_mods,
                self.counterfactual_engine.max_candidates,
                self.counterfactual_engine.max_transforms_per_rule,
                sampler.max_hypotheses,
                sampler.max_scored_hypotheses,
                sampler.max_open_body_literals,
                sampler.max_open_patterns,
                sampler.max_open_head_patterns,
                sampler.bundle_beam_width,
                sampler.max_bundle_rules,
                sampler.bundle_seed_k,
            ) = saved

    @staticmethod
    def _report_summary(report: Optional[CreativeCycleReport]) -> Dict[str, float]:
        if report is None:
            return {}
        metrics = {
            (
                key if str(key).startswith("creative_")
                else f"creative_{key}"
            ): float(value)
            for key, value in dict(getattr(report, "metrics", {})).items()
        }
        metrics.setdefault("creative_selected_rules", float(len(getattr(report, "selected_rules", ()))))
        metrics.setdefault(
            "creative_counterfactual_novel_facts",
            float(len(getattr(report, "counterfactual_novel_facts", ()))),
        )
        metrics.setdefault(
            "creative_counterfactual_contradictions",
            float(len(getattr(report, "counterfactual_contradictions", ()))),
        )
        metrics.setdefault(
            "creative_validated_support_facts",
            float(len(getattr(report, "validated_support_facts", ()))),
        )
        metrics.setdefault(
            "creative_coverage_gained_targets",
            float(len(getattr(report, "coverage_gained_targets", ()))),
        )
        if getattr(report, "intrinsic_goal", None) is not None:
            metrics.setdefault("creative_intrinsic_value", float(report.intrinsic_goal.value))
        return metrics

    def materialize_report_into_context(
        self,
        context: Any,
        report: Optional[CreativeCycleReport] = None,
    ) -> Any:
        if context is None:
            return context
        report = self.last_report if report is None else report
        support_facts = self._report_support_facts(report)
        report_summary = self._report_summary(report)
        if not support_facts and not report_summary:
            return context
        metadata = dict(getattr(context, "metadata", {}))
        metadata.update(report_summary)
        world_summary = dict(getattr(context, "world_context_summary", {}))
        world_summary.update(report_summary)
        abduced_support = set(getattr(context, "abduced_support_facts", frozenset()))
        abduced_support.update(support_facts)
        abduced_support.update(getattr(report, "validated_support_facts", ()))
        world_context = set(getattr(context, "world_context_facts", frozenset()))
        world_context.update(getattr(report, "counterfactual_novel_facts", ()))
        world_context.update(getattr(report, "coverage_gained_targets", ()))
        intrinsic_goal = getattr(getattr(report, "intrinsic_goal", None), "goal", None)
        if intrinsic_goal is not None:
            world_context.add(intrinsic_goal)
        return replace(
            context,
            metadata=metadata,
            world_context_summary=world_summary,
            abduced_support_facts=frozenset(abduced_support),
            world_context_facts=frozenset(world_context),
        )

    def enrich_task_context(self, context: Any) -> Any:
        if context is None:
            return context
        queued = list(self.intrinsic_engine.scheduled_goals())
        pending = self.intrinsic_engine.pending_goal or (queued[0] if queued else None)
        if pending is None:
            return self.materialize_report_into_context(context)
        if not queued:
            queued = [pending]
        metadata = dict(getattr(context, "metadata", {}))
        metadata.update(
            {
                "intrinsic_value": float(pending.value),
                "intrinsic_goal_kind": pending.kind,
                "intrinsic_goal_repr": repr(pending.goal),
                "intrinsic_goal_queue": tuple(repr(goal.goal) for goal in queued[:4]),
                "intrinsic_goal_queue_size": float(len(queued)),
                "intrinsic_background_goals": float(max(len(queued) - 1, 0)),
            }
        )
        targets = set(getattr(context, "target_facts", frozenset()))
        for goal in queued[:4]:
            if goal.goal is not None:
                targets.add(goal.goal)
        updates: Dict[str, Any] = {
            "metadata": metadata,
            "target_facts": frozenset(targets),
            "trigger_abduction": bool(getattr(context, "trigger_abduction", False) or bool(queued)),
        }
        if getattr(context, "goal", None) is None and pending.goal is not None:
            updates["goal"] = pending.goal
        enriched = replace(context, **updates)
        return self.materialize_report_into_context(enriched)

    def _utility_batch(self, prover: Any, clauses: Sequence[Any], device: torch.device) -> List[float]:
        if not clauses:
            return []
        scores = prover.vem.score_batch(list(clauses), device)
        return [float(score) for score in scores.detach().cpu().tolist()]

    def _recent_abduction_candidates(self, prover: Any) -> List[RuleCandidate]:
        clauses = list(getattr(prover, "last_abduced_rules", ()))
        if not clauses:
            return []
        utility_history = getattr(prover, "_rule_utility_history", {})
        candidates: List[RuleCandidate] = []
        seen: set[Any] = set()
        for clause in clauses:
            if clause in seen:
                continue
            seen.add(clause)
            history = list(utility_history.get(clause, utility_history.get(hash(clause), ())))
            prior_utility = float(sum(history) / len(history)) if history else 0.5
            candidates.append(
                RuleCandidate(
                    clause=clause,
                    source="abduction",
                    score=prior_utility,
                    utility=prior_utility,
                    metadata={"recent_abduction": 1.0},
                )
            )
        return candidates

    @staticmethod
    def _dedupe_rule_candidates(candidates: Sequence[RuleCandidate]) -> List[RuleCandidate]:
        best_by_clause: Dict[Any, RuleCandidate] = {}
        for candidate in candidates:
            clause = getattr(candidate, "clause", None)
            if clause is None:
                continue
            current = best_by_clause.get(clause)
            if current is None or float(candidate.score) > float(current.score):
                best_by_clause[clause] = candidate
        return sorted(best_by_clause.values(), key=lambda item: float(item.score), reverse=True)

    def _grounding_candidate_rules(self, prover: Any) -> List[RuleCandidate]:
        task_context = getattr(prover, "task_context", None)
        if task_context is None:
            return []
        direct = tuple(getattr(task_context, "grounding_candidate_rules", ()) or ())
        if not direct:
            artifacts = getattr(task_context, "grounding_artifacts", None)
            if artifacts is not None:
                direct = tuple(getattr(artifacts, "grounding_candidate_rules", ()) or ())
        grounded: List[RuleCandidate] = []
        for candidate in direct:
            if not isinstance(candidate, RuleCandidate):
                continue
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            metadata.setdefault("grounding_seed", 1.0)
            grounded.append(
                RuleCandidate(
                    clause=candidate.clause,
                    source=str(getattr(candidate, "source", "") or "grounding_rule_compiler"),
                    score=float(candidate.score),
                    utility=float(getattr(candidate, "utility", 0.0)),
                    aesthetic=float(getattr(candidate, "aesthetic", 0.0)),
                    structural_similarity=float(getattr(candidate, "structural_similarity", 0.0)),
                    metadata=metadata,
                )
            )
        return self._dedupe_rule_candidates(grounded)

    @staticmethod
    def _goal_supported(goal: Any, facts: Sequence[Any]) -> bool:
        from omen_prolog import unify

        for fact in facts:
            try:
                if unify(goal, fact) is not None:
                    return True
            except Exception:
                continue
        return False

    def _supported_targets(
        self,
        target_facts: Sequence[Any],
        facts: Sequence[Any],
    ) -> List[Any]:
        return [
            target
            for target in target_facts
            if self._goal_supported(target, facts)
        ]

    @staticmethod
    def _fact_description_bits(fact: Any) -> float:
        return float(4 + 2 * len(tuple(getattr(fact, "args", ()))))

    def _compression_gain(
        self,
        baseline_facts: Sequence[Any],
        post_facts: Sequence[Any],
        target_facts: Sequence[Any],
        selected_rules: Sequence[Any],
    ) -> float:
        if not target_facts or not selected_rules:
            return 0.0
        unresolved_before = [
            target for target in target_facts
            if not self._goal_supported(target, baseline_facts)
        ]
        unresolved_after = [
            target for target in target_facts
            if not self._goal_supported(target, post_facts)
        ]
        mdl_before = sum(self._fact_description_bits(fact) for fact in unresolved_before)
        mdl_after = sum(float(rule.description_length_bits()) for rule in selected_rules)
        mdl_after += sum(self._fact_description_bits(fact) for fact in unresolved_after)
        return float(mdl_before - mdl_after)

    def _derived_facts(
        self,
        prover: Any,
        current_facts: Sequence[Any],
        extra_rules: Sequence[Any] = (),
    ):
        cache = self._runtime_derived_facts_cache
        rule_count = int(getattr(prover.kb, "_n_rules", len(getattr(prover.kb, "rules", ()))))
        cache_key = (rule_count, self._runtime_rule_key(extra_rules))
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        current_fact_set = frozenset(current_facts)
        try:
            if extra_rules:
                sandbox = self.counterfactual_engine._clone_kb(prover.kb)
                for rule in extra_rules:
                    sandbox.add_rule(rule)
                derived = sandbox.forward_chain(
                    max(int(getattr(prover, "max_depth", 1)), 1),
                    starting_facts=current_fact_set,
                    only_verified=False,
                    track_epistemic=False,
                )
            else:
                derived = prover.kb.forward_chain(
                    max(int(getattr(prover, "max_depth", 1)), 1),
                    starting_facts=current_fact_set,
                    only_verified=False,
                    track_epistemic=False,
                )
        except Exception:
            derived = current_fact_set
        derived_facts = frozenset(derived)
        if cache is not None:
            cache[cache_key] = derived_facts
        return derived_facts

    def _task_gap(
        self,
        prover: Any,
        current_facts: Sequence[Any],
        target_facts: Sequence[Any],
        extra_rules: Sequence[Any] = (),
    ) -> float:
        targets = list(target_facts)
        if not targets:
            return float(getattr(getattr(prover, "task_context", None), "metadata", {}).get("gap_norm", 0.0))
        cache = self._runtime_task_gap_cache
        rule_count = int(getattr(prover.kb, "_n_rules", len(getattr(prover.kb, "rules", ()))))
        cache_key = (
            rule_count,
            self._runtime_rule_key(extra_rules),
            self._runtime_target_key(targets),
        )
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                return float(cached)
        derived = self._derived_facts(prover, current_facts, extra_rules=extra_rules)
        unresolved = sum(1 for target in targets if not self._goal_supported(target, derived))
        gap = float(unresolved / max(len(targets), 1))
        if cache is not None:
            cache[cache_key] = gap
        return gap

    def _world_counterfactual_surprise(
        self,
        prover: Any,
        current_facts: Sequence[Any],
    ):
        if getattr(prover, "_world_rnn", None) is None or getattr(prover, "_last_z", None) is None:
            return None

        def _score(modified_rules: Sequence[Any], novel_facts: Tuple[Any, ...], _sandbox: Any) -> float:
            if not modified_rules:
                return 0.0
            try:
                device = prover._last_z.device
                z0 = prover._last_z[:1]
                target_facts = list(novel_facts)
                if not target_facts:
                    for rule in modified_rules:
                        if getattr(rule, "head", None) is not None and getattr(rule.head, "is_ground", lambda: False)():
                            target_facts.append(rule.head)
                if not target_facts:
                    return 0.0

                action_tokens: List[int] = []
                for rule in modified_rules:
                    predicted_facts = frozenset()
                    predict_fn = getattr(prover, "_predict_rule_facts", None)
                    if predict_fn is not None:
                        try:
                            predicted_facts = predict_fn(rule, frozenset(current_facts))
                        except Exception:
                            predicted_facts = frozenset()
                    action_token = prover._rule_action_token(rule, predicted_facts)
                    if action_token is not None:
                        action_tokens.append(int(action_token))
                if not action_tokens:
                    return 0.0

                actions_t = torch.tensor([action_tokens], device=device, dtype=torch.long)
                if hasattr(prover._world_rnn, "simulate_sequence"):
                    z_traj = prover._world_rnn.simulate_sequence(z0, actions_t).squeeze(0)
                else:
                    preds: List[torch.Tensor] = []
                    h = None
                    z_prev = z0
                    for token in action_tokens:
                        z_prev, h = prover._world_rnn(
                            z_prev,
                            torch.tensor([token], device=device, dtype=torch.long),
                            h,
                        )
                        preds.append(z_prev.squeeze(0))
                    z_traj = torch.stack(preds, dim=0)

                target_latents = torch.stack(
                    [prover.ground(frozenset({fact}), device).squeeze(0) for fact in target_facts],
                    dim=0,
                )
                z_traj = torch.nn.functional.normalize(z_traj, dim=-1, eps=1e-6)
                target_latents = torch.nn.functional.normalize(target_latents, dim=-1, eps=1e-6)
                logits = (z_traj @ target_latents.T) / 0.25
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
                if probs.size(-1) > 1:
                    entropy = entropy / math.log(float(probs.size(-1)))
                return float(entropy.mean().item())
            except Exception:
                return 0.0

        return _score

    def _counterfactual_ame_candidates(
        self,
        rules: Sequence[Any],
        modified_rules: Sequence[Any],
    ) -> Tuple[List[RuleCandidate], List[RuleCandidate]]:
        if not modified_rules:
            return [], []
        augmented_rules = list(rules) + list(modified_rules)
        augmented_engine = self.analogy_engine.clone()
        augmented_engine.fit(augmented_rules)
        modified_rule_keys = {repr(rule) for rule in modified_rules}
        modified_pred_ids = {
            int(pred)
            for rule in modified_rules
            for pred in ({int(rule.head.pred)} | {int(atom.pred) for atom in getattr(rule, "body", ())})
        }
        analogy_candidates = augmented_engine.generate_candidates(
            augmented_rules,
            existing_hashes=rules,
        )
        metaphor_candidates = augmented_engine.generate_metaphor_candidates(
            augmented_rules,
            existing_hashes=rules,
        )
        routed_analogy: List[RuleCandidate] = []
        for candidate in analogy_candidates:
            if str(candidate.metadata.get("source_rule_key_text", "")) not in modified_rule_keys:
                continue
            routed_analogy.append(
                RuleCandidate(
                    clause=candidate.clause,
                    source="counterfactual_analogy",
                    score=float(candidate.score),
                    utility=float(candidate.utility),
                    aesthetic=float(candidate.aesthetic),
                    structural_similarity=float(candidate.structural_similarity),
                    metadata={
                        **dict(candidate.metadata),
                        "from_counterfactual": 1.0,
                    },
                )
            )
        routed_metaphor: List[RuleCandidate] = []
        for candidate in metaphor_candidates:
            source_pred = int(candidate.metadata.get("source_pred", -1.0))
            target_pred = int(candidate.metadata.get("target_pred", -1.0))
            if (
                str(candidate.metadata.get("source_rule_key_text", "")) not in modified_rule_keys
                and source_pred not in modified_pred_ids
                and target_pred not in modified_pred_ids
            ):
                continue
            routed_metaphor.append(
                RuleCandidate(
                    clause=candidate.clause,
                    source="counterfactual_metaphor",
                    score=float(candidate.score),
                    utility=float(candidate.utility),
                    aesthetic=float(candidate.aesthetic),
                    structural_similarity=float(candidate.structural_similarity),
                    metadata={
                        **dict(candidate.metadata),
                        "from_counterfactual": 1.0,
                    },
                )
            )
        return routed_analogy, routed_metaphor

    def _counterfactual_analogy_candidates(
        self,
        rules: Sequence[Any],
        modified_rules: Sequence[Any],
    ) -> List[RuleCandidate]:
        analogies, _ = self._counterfactual_ame_candidates(rules, modified_rules)
        return analogies

    def _counterfactual_metaphor_candidates(
        self,
        rules: Sequence[Any],
        modified_rules: Sequence[Any],
    ) -> List[RuleCandidate]:
        _, metaphors = self._counterfactual_ame_candidates(rules, modified_rules)
        return metaphors

    def _candidate_relevance(self, candidate: RuleCandidate, target: Any) -> float:
        from omen_prolog import unify

        score = 0.0
        try:
            if unify(candidate.clause.head, target) is not None:
                score += 2.0
        except Exception:
            pass
        if int(getattr(candidate.clause.head, "pred", -1)) == int(getattr(target, "pred", -2)):
            score += 1.0
        for atom in getattr(candidate.clause, "body", ()):
            if int(getattr(atom, "pred", -1)) == int(getattr(target, "pred", -2)):
                score += 0.5
                break
        return score

    def _evolve_for_situations(
        self,
        seed_candidates: Sequence[RuleCandidate],
        rules: Sequence[Any],
        unresolved_targets: Sequence[Any],
        prover: Any,
        current_facts: Sequence[Any],
        device: torch.device,
        predicate_embeddings: Dict[int, torch.Tensor],
    ) -> List[RuleCandidate]:
        if not seed_candidates:
            return []
        compression_cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}

        def _fact_bits(fact: Any) -> float:
            return float(4 + 2 * len(tuple(getattr(fact, "args", ()))))

        def _compression_delta(rule: Any, target_subset: Sequence[Any]) -> float:
            if not target_subset:
                return 0.0
            cache_key = (repr(rule), self._runtime_target_key(target_subset))
            cached = compression_cache.get(cache_key)
            if cached is not None:
                return float(cached)
            after = self._task_gap(prover, current_facts, target_subset, extra_rules=[rule])
            before = self._task_gap(prover, current_facts, target_subset)
            unresolved_before = [
                target for target in target_subset
                if not self._goal_supported(target, current_facts)
            ]
            derived_after = self._derived_facts(prover, current_facts, extra_rules=[rule])
            unresolved_after = [
                target for target in target_subset
                if not self._goal_supported(target, derived_after)
            ]
            mdl_before = sum(_fact_bits(fact) for fact in unresolved_before)
            mdl_after = float(rule.description_length_bits()) + sum(
                _fact_bits(fact) for fact in unresolved_after
            )
            if before <= after and mdl_before <= mdl_after:
                compression_cache[cache_key] = 0.0
                return 0.0
            gain = float(mdl_before - mdl_after)
            compression_cache[cache_key] = gain
            return gain

        selected: Dict[Any, RuleCandidate] = {}
        situations = list(unresolved_targets) if unresolved_targets else [None]
        for situation in situations:
            if situation is None:
                relevant = list(seed_candidates)
                compression_fn = None
            else:
                scored = sorted(
                    (
                        (self._candidate_relevance(candidate, situation), candidate)
                        for candidate in seed_candidates
                    ),
                    key=lambda item: (item[0], item[1].score),
                    reverse=True,
                )
                relevant = [candidate for relevance, candidate in scored if relevance > 0.0][: self.aesthetic_engine.population_size]
                if not relevant:
                    relevant = list(seed_candidates)
                compression_fn = lambda rule, target=situation: _compression_delta(rule, [target])
            evolved = self.aesthetic_engine.evolve(
                relevant,
                existing_rules=rules,
                utility_fn=lambda clauses: self._utility_batch(prover, clauses, device),
                ltm_rules=rules,
                predicate_embeddings=predicate_embeddings,
                compression_fn=compression_fn,
            )
            for candidate in evolved:
                current = selected.get(candidate.clause)
                if current is None or candidate.score > current.score:
                    selected[candidate.clause] = candidate

        ranked = sorted(selected.values(), key=lambda item: item.score, reverse=True)
        return ranked[: max(self.max_selected_rules, 1)]

    def run(
        self,
        prover: Any,
        z: Optional[torch.Tensor],
        current_facts: Sequence[Any],
        target_facts: Sequence[Any],
        device: torch.device,
        *,
        fast_mode: bool = False,
    ) -> CreativeCycleReport:
        report = CreativeCycleReport()
        self._clear_runtime_caches()
        contradiction_scope = tuple(
            getattr(getattr(prover, "task_context", None), "contradiction_scope_facts", lambda: frozenset())()
        )
        if not self.enabled or prover._step % self.cycle_every != 0:
            self.intrinsic_engine.update_state(z)
            queued_goals = self.intrinsic_engine.scheduled_goals()
            report.metrics = {
                "cycle_active": 0.0,
                "train_fast_budgeted": 1.0 if fast_mode else 0.0,
                "intrinsic_goal_queue_size": float(len(queued_goals)),
                "intrinsic_background_goals": float(max(len(queued_goals) - 1, 0)),
                "contradiction_scope_facts": float(len(contradiction_scope)),
            }
            self.last_report = report
            return report

        self._runtime_derived_facts_cache = {}
        self._runtime_task_gap_cache = {}
        rules = list(getattr(prover.kb, "rules", ()))
        report.abduction_candidates = self._recent_abduction_candidates(prover)
        report.grounding_candidates = self._grounding_candidate_rules(prover)
        analogy_state = self.analogy_engine.fit(rules)
        report.predicate_embeddings = dict(analogy_state.graph_view.embeddings)
        report.analogy_candidates = self.analogy_engine.generate_candidates(
            rules,
            existing_hashes=rules,
        )
        report.metaphor_candidates = self.analogy_engine.generate_metaphor_candidates(
            rules,
            existing_hashes=rules,
        )

        base_task_gap = self._task_gap(prover, current_facts, target_facts)
        gap_norm = float(getattr(getattr(prover, "task_context", None), "metadata", {}).get("gap_norm", base_task_gap))
        gap_norm = max(gap_norm, base_task_gap)
        hot_dims = tuple(getattr(getattr(prover, "task_context", None), "hot_dims", ()))
        self.ontology_engine.observe_gap_cluster(gap_norm, hot_dims, z=z)

        conflict_fn = getattr(prover, "_atoms_conflict", None)
        if conflict_fn is None:
            conflict_fn = lambda left, right: False
        with (self._train_fast_symbolic_budget() if fast_mode else nullcontext()):
            counterfactual = self.counterfactual_engine.explore(
                prover.kb,
                current_facts,
                prover.max_depth,
                conflict_fn=conflict_fn,
                world_surprise_fn=self._world_counterfactual_surprise(prover, current_facts),
            )
        report.counterfactual_candidates = list(counterfactual.candidates)
        report.counterfactual_novel_facts = tuple(counterfactual.novel_facts)
        report.counterfactual_contradictions = tuple(counterfactual.contradictions)
        (
            report.counterfactual_analogy_candidates,
            report.counterfactual_metaphor_candidates,
        ) = self._counterfactual_ame_candidates(
            rules,
            counterfactual.modified_rules,
        )

        paradox_facts: List[Any] = list(counterfactual.novel_facts)
        for left, right in counterfactual.contradictions:
            if left not in paradox_facts:
                paradox_facts.append(left)
            if right not in paradox_facts:
                paradox_facts.append(right)
        for fact in contradiction_scope:
            if fact not in paradox_facts:
                paradox_facts.append(fact)

        goal = prover.current_goal(z) if hasattr(prover, "current_goal") else None
        oee_target_facts: Sequence[Any] = target_facts
        oee_paradox_facts: Sequence[Any] = paradox_facts
        oee_max_candidates = 4
        if fast_mode:
            oee_target_facts = self._limit_unique_facts(target_facts, self.train_fast_oee_max_targets)
            oee_paradox_facts = self._limit_unique_facts(paradox_facts, self.train_fast_oee_max_paradox_facts)
            oee_max_candidates = self.train_fast_oee_max_candidates
        with (self._train_fast_symbolic_budget() if fast_mode else nullcontext()):
            report.ontology_candidates = self.ontology_engine.generate_candidates(
                current_facts=current_facts,
                goal=goal,
                gap_norm=gap_norm,
                contradiction_count=len(counterfactual.contradictions),
                max_candidates=oee_max_candidates,
                paradox_facts=oee_paradox_facts,
                target_facts=oee_target_facts,
                state_z=z,
                predicate_embeddings=report.predicate_embeddings,
                kb=prover.kb,
                existing_rules=rules,
            )

        seed_candidates = self._dedupe_rule_candidates(
            list(report.grounding_candidates)
            + list(report.abduction_candidates)
            + list(report.analogy_candidates)
            + list(report.metaphor_candidates)
            + list(report.counterfactual_analogy_candidates)
            + list(report.counterfactual_metaphor_candidates)
            + list(report.counterfactual_candidates)
            + list(report.ontology_candidates)
        )
        baseline_facts = self._derived_facts(prover, current_facts)
        self.intrinsic_engine.resolve_supported_goals(tuple(baseline_facts))
        unresolved_targets = [
            target
            for target in target_facts
            if not self._goal_supported(target, baseline_facts)
        ]
        selected_rules = self._evolve_for_situations(
            seed_candidates,
            rules,
            unresolved_targets,
            prover,
            current_facts,
            device,
            report.predicate_embeddings,
        )
        attempted_selected = list(selected_rules[: self.max_selected_rules])
        added = 0
        selected_utility = 0.0
        ontology_accepted = False
        accepted_selected: List[RuleCandidate] = []
        ontology_pred_ids: List[int] = []
        ontology_rules: List[Any] = []
        if selected_rules:
            from omen_prolog import EpistemicStatus

            for candidate in selected_rules[: self.max_selected_rules]:
                utility = float(max(0.0, min(1.0, candidate.utility + 0.5 * candidate.aesthetic)))
                if candidate.source == "ontology" and target_facts:
                    candidate_gap = self._task_gap(
                        prover,
                        current_facts,
                        target_facts,
                        extra_rules=[candidate.clause],
                    )
                    if candidate_gap + 1e-6 >= gap_norm:
                        continue
                if prover.kb.add_rule(candidate.clause, status=EpistemicStatus.proposed):
                    prover.vem.record_outcome(candidate.clause, utility_target=utility, device=device)
                    prover._record_rule_utility(candidate.clause, utility)
                    prover._extend_recent_abduced_rules([candidate.clause])
                    self.rule_origins[repr(candidate.clause)] = candidate.source
                    if candidate.source == "ontology":
                        ontology_accepted = True
                        pred_id = int(candidate.metadata.get("invented_predicate", 0.0))
                        if pred_id > 0:
                            ontology_pred_ids.append(pred_id)
                        ontology_rules.append(candidate.clause)
                    added += 1
                    selected_utility += utility
                    accepted_selected.append(candidate)
        report.selected_rules = accepted_selected
        post_facts = baseline_facts
        if accepted_selected:
            post_facts = self._derived_facts(prover, current_facts)
        supported_before = self._supported_targets(target_facts, baseline_facts)
        supported_after = self._supported_targets(target_facts, post_facts)
        gained_targets = self._dedupe_facts(
            [
                target for target in supported_after
                if not self._goal_supported(target, baseline_facts)
            ]
        )
        validated_support = list(gained_targets)
        for candidate in accepted_selected:
            head = getattr(getattr(candidate, "clause", None), "head", None)
            if head is None:
                continue
            if self._goal_supported(head, post_facts):
                validated_support.append(head)
        report.validated_support_facts = tuple(self._dedupe_facts(validated_support))
        report.coverage_gained_targets = tuple(gained_targets)
        if report.ontology_candidates:
            post_gap = self._task_gap(prover, current_facts, target_facts) if target_facts else gap_norm
            self.ontology_engine.record_feedback(
                accepted=ontology_accepted,
                accepted_pred_ids=ontology_pred_ids,
                gap_before=gap_norm,
                gap_after=post_gap,
                supporting_rules=ontology_rules,
            )
            report.ontology_fixations = [
                OntologyPredicateState(
                    pred_id=entry.pred_id,
                    arity=entry.arity,
                    status=entry.status,
                    gap_before=entry.gap_before,
                    gap_after=entry.gap_after,
                    metadata=dict(entry.metadata),
                )
                for entry in self.ontology_engine.fixed_entries(ontology_pred_ids)
            ]

        candidate_rule_pool = self._dedupe_rule_candidates(list(report.selected_rules))
        if not candidate_rule_pool:
            candidate_rule_pool = self._dedupe_rule_candidates(seed_candidates)
        candidate_goals = [candidate.clause.head for candidate in candidate_rule_pool]
        if not candidate_goals:
            candidate_goals = [candidate.clause.head for candidate in report.ontology_candidates]
        report.intrinsic_goal = self.intrinsic_engine.formulate_goal(
            z=z,
            gap_norm=gap_norm,
            graph_view=analogy_state.graph_view,
            candidate_goals=candidate_goals,
            candidate_rules=candidate_rule_pool,
            provenance=getattr(getattr(prover, "task_context", None), "provenance", "latent"),
        )
        gene_pool_stats = self.aesthetic_engine.gene_pool_stats()
        oee_stats = self.ontology_engine.stats()
        target_total = float(max(len(target_facts), 1))
        target_support_before = float(len(supported_before)) / target_total if target_facts else 0.0
        target_support_after = float(len(supported_after)) / target_total if target_facts else 0.0
        gap_after = self._task_gap(prover, current_facts, target_facts) if target_facts else gap_norm
        compression_gain = self._compression_gain(
            baseline_facts,
            post_facts,
            target_facts,
            [candidate.clause for candidate in accepted_selected],
        )
        report.metrics = {
            "cycle_active": 1.0,
            "train_fast_budgeted": 1.0 if fast_mode else 0.0,
            "contradiction_scope_facts": float(len(contradiction_scope)),
            "abduction_candidates": float(len(report.abduction_candidates)),
            "grounding_rule_candidates": float(len(report.grounding_candidates)),
            "analogy_candidates": float(len(report.analogy_candidates)),
            "metaphor_candidates": float(len(report.metaphor_candidates)),
            "counterfactual_analogy_candidates": float(len(report.counterfactual_analogy_candidates)),
            "counterfactual_metaphor_candidates": float(len(report.counterfactual_metaphor_candidates)),
            "counterfactual_candidates": float(len(report.counterfactual_candidates)),
            "ame_total_candidates": float(
                len(report.analogy_candidates)
                + len(report.metaphor_candidates)
                + len(report.counterfactual_analogy_candidates)
                + len(report.counterfactual_metaphor_candidates)
            ),
            "counterfactual_surprise": float(counterfactual.surprise),
            "counterfactual_contradictions": float(len(counterfactual.contradictions)),
            "counterfactual_gamma_mod_size": float(counterfactual.metadata.get("gamma_mod_size", 0.0)),
            "counterfactual_complement_contradictions": float(
                counterfactual.metadata.get("complement_contradictions", 0.0)
            ),
            "counterfactual_greedy_score": float(counterfactual.metadata.get("greedy_score", 0.0)),
            "counterfactual_search_score": float(counterfactual.metadata.get("search_score", 0.0)),
            "counterfactual_exact_search": float(counterfactual.metadata.get("exact_search", 0.0)),
            "counterfactual_evaluated_subsets": float(counterfactual.metadata.get("evaluated_subsets", 0.0)),
            "counterfactual_world_surprise": float(counterfactual.metadata.get("world_surprise", 0.0)),
            "ontology_candidates": float(len(report.ontology_candidates)),
            "ontology_neural_guided": float(
                sum(1 for c in report.ontology_candidates if float(c.metadata.get("neural_guided", 0.0)) > 0)
            ),
            "ontology_mean_consistency": float(
                sum(float(c.metadata.get("consistency", 0.0)) for c in report.ontology_candidates)
                / max(len(report.ontology_candidates), 1)
            ),
            "ontology_mean_hypothesis_score": float(
                sum(float(c.metadata.get("hypothesis_score", 0.0)) for c in report.ontology_candidates)
                / max(len(report.ontology_candidates), 1)
            ),
            "ontology_bundle_candidates": float(
                sum(1 for c in report.ontology_candidates if float(c.metadata.get("bundle_size", 1.0)) > 1.0)
            ),
            "ontology_max_bundle_size": float(
                max((float(c.metadata.get("bundle_size", 1.0)) for c in report.ontology_candidates), default=1.0)
            ),
            **oee_stats,
            "selected_rules": float(added),
            "validated_selected_rules": float(len(accepted_selected)),
            "grounding_rule_selected": float(
                sum(1 for candidate in accepted_selected if str(getattr(candidate, "source", "") or "") == "grounding_rule_compiler")
            ),
            "selected_rule_acceptance_ratio": float(len(accepted_selected)) / float(max(len(attempted_selected), 1)),
            "selected_mean_utility": float(selected_utility / max(added, 1)),
            "ontology_feedback_accepted": 1.0 if ontology_accepted else 0.0,
            "ontology_fixed_predicates": float(len(report.ontology_fixations)),
            "validated_support_facts": float(len(report.validated_support_facts)),
            "coverage_gained_targets": float(len(report.coverage_gained_targets)),
            "target_support_before": float(target_support_before),
            "target_support_after": float(target_support_after),
            "target_support_gain": float(max(target_support_after - target_support_before, 0.0)),
            "gap_before": float(gap_norm),
            "gap_after": float(gap_after),
            "gap_reduction": float(max(gap_norm - gap_after, 0.0)),
            "compression_gain": float(compression_gain),
            "intrinsic_value": float(report.intrinsic_goal.value if report.intrinsic_goal is not None else 0.0),
            "intrinsic_goal_queue_size": float(len(self.intrinsic_engine.scheduled_goals())),
            "intrinsic_background_goals": float(max(len(self.intrinsic_engine.scheduled_goals()) - 1, 0)),
            "analogy_projector_loss": float(self.analogy_engine.state.projector_loss),
            "analogy_embedding_source": float(
                1.0 if getattr(self.analogy_engine.state.graph_view, "embedding_source", "") == "hypergraph_gnn"
                else 0.5 if getattr(self.analogy_engine.state.graph_view, "embedding_source", "") == "hypergraph_spectral"
                else 0.0
            ),
            "aee_gene_pool_size": gene_pool_stats.get("size", 0.0),
            "aee_gene_pool_max_score": gene_pool_stats.get("max_score", 0.0),
            "aee_gene_pool_mean_score": gene_pool_stats.get("mean_score", 0.0),
            "aee_unresolved_situations": float(len(unresolved_targets)),
        }
        self.intrinsic_engine.update_state(z)
        self.last_report = report
        self._clear_runtime_caches()
        return report

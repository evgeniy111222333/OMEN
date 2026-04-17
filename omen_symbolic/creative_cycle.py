from __future__ import annotations

from dataclasses import replace
import math
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
        # ICE params
        ice_state_history: int = 128,
        ice_goal_threshold: float = 0.35,
    ):
        self.enabled = bool(enabled)
        self.cycle_every = max(int(cycle_every), 1)
        self.max_selected_rules = max(int(max_selected_rules), 1)
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
            oee_bundle_beam_width=oee_bundle_beam_width,
            oee_max_bundle_rules=oee_max_bundle_rules,
            oee_bundle_seed_k=oee_bundle_seed_k,
        )
        self.intrinsic_engine = IntrinsicCuriosityEngine(
            state_history=ice_state_history,
            goal_threshold=ice_goal_threshold,
        )
        self.last_report = CreativeCycleReport()
        self.rule_origins: Dict[int, str] = {}

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

    def enrich_task_context(self, context: Any) -> Any:
        if context is None:
            return context
        queued = list(self.intrinsic_engine.scheduled_goals())
        pending = self.intrinsic_engine.pending_goal or (queued[0] if queued else None)
        if pending is None:
            return context
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
        return replace(context, **updates)

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
        seen: set[int] = set()
        for clause in clauses:
            clause_hash = hash(clause)
            if clause_hash in seen:
                continue
            seen.add(clause_hash)
            history = list(utility_history.get(clause_hash, ()))
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
    def _goal_supported(goal: Any, facts: Sequence[Any]) -> bool:
        from omen_prolog import unify

        for fact in facts:
            try:
                if unify(goal, fact) is not None:
                    return True
            except Exception:
                continue
        return False

    def _task_gap(
        self,
        prover: Any,
        current_facts: Sequence[Any],
        target_facts: Sequence[Any],
        extra_rules: Sequence[Any] = (),
    ) -> float:
        try:
            if extra_rules:
                sandbox = self.counterfactual_engine._clone_kb(prover.kb)
                for rule in extra_rules:
                    sandbox.add_rule(rule)
                derived = sandbox.forward_chain(
                    max(int(getattr(prover, "max_depth", 1)), 1),
                    starting_facts=frozenset(current_facts),
                    only_verified=False,
                )
            else:
                derived = prover.kb.forward_chain(
                    max(int(getattr(prover, "max_depth", 1)), 1),
                    starting_facts=frozenset(current_facts),
                    only_verified=False,
                )
        except Exception:
            derived = frozenset(current_facts)
        targets = list(target_facts)
        if not targets:
            return float(getattr(getattr(prover, "task_context", None), "metadata", {}).get("gap_norm", 0.0))
        unresolved = sum(1 for target in targets if not self._goal_supported(target, derived))
        return float(unresolved / max(len(targets), 1))

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
        modified_hash_text = {str(hash(rule)) for rule in modified_rules}
        modified_pred_ids = {
            int(pred)
            for rule in modified_rules
            for pred in ({int(rule.head.pred)} | {int(atom.pred) for atom in getattr(rule, "body", ())})
        }
        analogy_candidates = augmented_engine.generate_candidates(
            augmented_rules,
            existing_hashes=[hash(rule) for rule in rules],
        )
        metaphor_candidates = augmented_engine.generate_metaphor_candidates(
            augmented_rules,
            existing_hashes=[hash(rule) for rule in rules],
        )
        routed_analogy: List[RuleCandidate] = []
        for candidate in analogy_candidates:
            if str(candidate.metadata.get("source_rule_hash_text", "")) not in modified_hash_text:
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
                str(candidate.metadata.get("source_rule_hash_text", "")) not in modified_hash_text
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

        baseline_gap = self._task_gap(prover, current_facts, unresolved_targets)

        def _fact_bits(fact: Any) -> float:
            return float(4 + 2 * len(tuple(getattr(fact, "args", ()))))

        def _compression_delta(rule: Any, target_subset: Sequence[Any]) -> float:
            if not target_subset:
                return 0.0
            after = self._task_gap(prover, current_facts, target_subset, extra_rules=[rule])
            before = self._task_gap(prover, current_facts, target_subset)
            unresolved_before = [
                target for target in target_subset
                if not self._goal_supported(target, current_facts)
            ]
            try:
                sandbox = self.counterfactual_engine._clone_kb(prover.kb)
                sandbox.add_rule(rule)
                derived_after = sandbox.forward_chain(
                    max(int(getattr(prover, "max_depth", 1)), 1),
                    starting_facts=frozenset(current_facts),
                    only_verified=False,
                )
            except Exception:
                derived_after = frozenset(current_facts)
            unresolved_after = [
                target for target in target_subset
                if not self._goal_supported(target, derived_after)
            ]
            mdl_before = sum(_fact_bits(fact) for fact in unresolved_before)
            mdl_after = float(rule.description_length_bits()) + sum(
                _fact_bits(fact) for fact in unresolved_after
            )
            if before <= after and mdl_before <= mdl_after:
                return 0.0
            return float(mdl_before - mdl_after)

        selected: Dict[int, RuleCandidate] = {}
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
                current = selected.get(hash(candidate.clause))
                if current is None or candidate.score > current.score:
                    selected[hash(candidate.clause)] = candidate

        ranked = sorted(selected.values(), key=lambda item: item.score, reverse=True)
        return ranked[: max(self.max_selected_rules, 1)]

    def run(
        self,
        prover: Any,
        z: Optional[torch.Tensor],
        current_facts: Sequence[Any],
        target_facts: Sequence[Any],
        device: torch.device,
    ) -> CreativeCycleReport:
        report = CreativeCycleReport()
        if not self.enabled or prover._step % self.cycle_every != 0:
            self.intrinsic_engine.update_state(z)
            self.last_report = report
            return report

        rules = list(getattr(prover.kb, "rules", ()))
        report.abduction_candidates = self._recent_abduction_candidates(prover)
        analogy_state = self.analogy_engine.fit(rules)
        report.predicate_embeddings = dict(analogy_state.graph_view.embeddings)
        report.analogy_candidates = self.analogy_engine.generate_candidates(
            rules,
            existing_hashes=[hash(rule) for rule in rules],
        )
        report.metaphor_candidates = self.analogy_engine.generate_metaphor_candidates(
            rules,
            existing_hashes=[hash(rule) for rule in rules],
        )

        base_task_gap = self._task_gap(prover, current_facts, target_facts)
        gap_norm = float(getattr(getattr(prover, "task_context", None), "metadata", {}).get("gap_norm", base_task_gap))
        gap_norm = max(gap_norm, base_task_gap)
        hot_dims = tuple(getattr(getattr(prover, "task_context", None), "hot_dims", ()))
        self.ontology_engine.observe_gap_cluster(gap_norm, hot_dims, z=z)

        conflict_fn = getattr(prover, "_atoms_conflict", None)
        if conflict_fn is None:
            conflict_fn = lambda left, right: False
        counterfactual = self.counterfactual_engine.explore(
            prover.kb,
            current_facts,
            prover.max_depth,
            conflict_fn=conflict_fn,
            world_surprise_fn=self._world_counterfactual_surprise(prover, current_facts),
        )
        report.counterfactual_candidates = list(counterfactual.candidates)
        (
            report.counterfactual_analogy_candidates,
            report.counterfactual_metaphor_candidates,
        ) = self._counterfactual_ame_candidates(
            rules,
            counterfactual.modified_rules,
        )

        paradox_facts: List[Any] = list(counterfactual.novel_facts)
        for left, right in counterfactual.contradictions:
            if not any(hash(left) == hash(item) for item in paradox_facts):
                paradox_facts.append(left)
            if not any(hash(right) == hash(item) for item in paradox_facts):
                paradox_facts.append(right)

        goal = prover.current_goal(z) if hasattr(prover, "current_goal") else None
        report.ontology_candidates = self.ontology_engine.generate_candidates(
            current_facts=current_facts,
            goal=goal,
            gap_norm=gap_norm,
            contradiction_count=len(counterfactual.contradictions),
            paradox_facts=paradox_facts,
            target_facts=target_facts,
            state_z=z,
            predicate_embeddings=report.predicate_embeddings,
            kb=prover.kb,
            existing_rules=rules,
        )

        seed_candidates: List[RuleCandidate] = (
            list(report.abduction_candidates)
            + list(report.analogy_candidates)
            + list(report.metaphor_candidates)
            + list(report.counterfactual_analogy_candidates)
            + list(report.counterfactual_metaphor_candidates)
            + list(report.counterfactual_candidates)
            + list(report.ontology_candidates)
        )
        try:
            baseline_facts = prover.kb.forward_chain(
                max(int(getattr(prover, "max_depth", 1)), 1),
                starting_facts=frozenset(current_facts),
                only_verified=False,
            )
        except Exception:
            baseline_facts = frozenset(current_facts)
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
                    self.rule_origins[hash(candidate.clause)] = candidate.source
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

        candidate_rule_pool = list(report.selected_rules)
        if not candidate_rule_pool:
            candidate_rule_pool = (
                list(report.analogy_candidates)
                + list(report.metaphor_candidates)
                + list(report.counterfactual_analogy_candidates)
                + list(report.counterfactual_metaphor_candidates)
                + list(report.counterfactual_candidates)
                + list(report.ontology_candidates)
            )
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
        report.metrics = {
            "abduction_candidates": float(len(report.abduction_candidates)),
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
            "selected_mean_utility": float(selected_utility / max(added, 1)),
            "ontology_feedback_accepted": 1.0 if ontology_accepted else 0.0,
            "ontology_fixed_predicates": float(len(report.ontology_fixations)),
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
        return report

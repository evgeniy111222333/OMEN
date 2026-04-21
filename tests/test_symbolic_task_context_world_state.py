from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_prolog import HornAtom, HornClause, SymbolicTaskContext, Var
from omen_scale import OMENScale, OMENScaleConfig
from omen_symbolic.creative_types import RuleCandidate
from omen_symbolic.execution_trace import build_symbolic_trace_bundle_with_artifacts


class SymbolicTaskContextWorldStateTest(unittest.TestCase):
    def test_world_state_fact_buckets_stay_separated_by_consumer(self) -> None:
        observed = HornAtom(100, (1, 2))
        active = HornAtom(101, (2, 3))
        hypothetical = HornAtom(102, (3, 4))
        contradicted = HornAtom(103, (4, 5))

        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset({observed}),
            grounding_world_state_active_facts=frozenset({active}),
            grounding_world_state_hypothetical_facts=frozenset({hypothetical}),
            grounding_world_state_contradicted_facts=frozenset({contradicted}),
        )

        reasoning = ctx.reasoning_facts()
        planner = ctx.planner_facts()
        contradiction_scope = ctx.contradiction_scope_facts()

        self.assertIn(observed, reasoning)
        self.assertIn(active, reasoning)
        self.assertNotIn(hypothetical, reasoning)
        self.assertNotIn(contradicted, reasoning)

        self.assertIn(hypothetical, planner)
        self.assertIn(active, planner)
        self.assertIn(observed, planner)
        self.assertNotIn(contradicted, planner)

        self.assertEqual(contradiction_scope, frozenset({contradicted}))

    def test_source_records_and_counts_expose_world_state_fact_buckets(self) -> None:
        active = HornAtom(201, (1, 1))
        hypothetical = HornAtom(202, (1, 2))
        contradicted = HornAtom(203, (1, 3))
        ctx = SymbolicTaskContext(
            grounding_world_state_active_facts=frozenset({active}),
            grounding_world_state_hypothetical_facts=frozenset({hypothetical}),
            grounding_world_state_contradicted_facts=frozenset({contradicted}),
        )

        labels = {label for label, _ in ctx.source_fact_records()}
        counts = ctx.source_counts()

        self.assertIn("grounding_world_state_active_fact", labels)
        self.assertIn("grounding_world_state_hypothetical_fact", labels)
        self.assertIn("grounding_world_state_contradicted_fact", labels)
        self.assertEqual(counts["grounding_world_state_active_facts"], 1.0)
        self.assertEqual(counts["grounding_world_state_hypothetical_facts"], 1.0)
        self.assertEqual(counts["grounding_world_state_contradicted_facts"], 1.0)

    def test_reasoning_lane_keeps_proposals_out_of_observed_and_reasoning_aggregates(self) -> None:
        observed = HornAtom(301, (1, 1))
        memory = HornAtom(302, (1, 2))
        ontology = HornAtom(303, (1, 3))
        active = HornAtom(304, (1, 4))
        saliency = HornAtom(305, (1, 5))
        net = HornAtom(306, (1, 6))
        grounding = HornAtom(307, (1, 7))
        world_context = HornAtom(308, (1, 8))
        abduced = HornAtom(309, (1, 9))

        ctx = SymbolicTaskContext(
            observed_facts=frozenset({observed, saliency, net}),
            memory_derived_facts=frozenset({memory}),
            grounding_ontology_facts=frozenset({ontology}),
            grounding_world_state_active_facts=frozenset({active}),
            saliency_derived_facts=frozenset({saliency}),
            net_derived_facts=frozenset({net}),
            grounding_derived_facts=frozenset({grounding}),
            world_context_facts=frozenset({world_context}),
            abduced_support_facts=frozenset({abduced}),
        )

        self.assertEqual(ctx.observed_facts, frozenset({observed, memory}))

        reasoning = ctx.reasoning_facts()
        self.assertIn(observed, reasoning)
        self.assertIn(memory, reasoning)
        self.assertIn(ontology, reasoning)
        self.assertIn(active, reasoning)
        self.assertNotIn(saliency, reasoning)
        self.assertNotIn(net, reasoning)
        self.assertNotIn(grounding, reasoning)
        self.assertNotIn(world_context, reasoning)
        self.assertNotIn(abduced, reasoning)

        labels = {label for label, _ in ctx.source_fact_records()}
        counts = ctx.source_counts()

        self.assertIn("saliency_proposal", labels)
        self.assertIn("net_proposal", labels)
        self.assertIn("grounding", labels)
        self.assertIn("world_context", labels)
        self.assertIn("abduced_proposal", labels)
        self.assertEqual(counts["proposal_facts"], 3.0)
        self.assertEqual(counts["saliency_proposal_facts"], 1.0)
        self.assertEqual(counts["net_proposal_facts"], 1.0)
        self.assertEqual(counts["abduced_proposal_facts"], 1.0)

    def test_program_anchor_prefers_canonical_support_over_raw_trace_observations(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        goal = HornAtom(401, (1, 2))
        target = HornAtom(402, (2, 3))
        trace_target = HornAtom(403, (3, 4))
        trace_observed = HornAtom(404, (4, 5))
        active = HornAtom(405, (5, 6))
        ontology = HornAtom(406, (6, 7))
        ctx = SymbolicTaskContext(
            goal=goal,
            target_facts=frozenset({target}),
            grounding_ontology_facts=frozenset({ontology}),
            grounding_world_state_active_facts=frozenset({active}),
            execution_trace=SimpleNamespace(
                target_facts=frozenset({trace_target}),
                observed_facts=frozenset({trace_observed}),
            ),
        )

        anchors = model._program_anchor_facts(ctx)

        self.assertIn(goal, anchors)
        self.assertIn(target, anchors)
        self.assertIn(trace_target, anchors)
        self.assertIn(active, anchors)
        self.assertIn(ontology, anchors)
        self.assertNotIn(trace_observed, anchors)

    def test_source_records_surface_interlingua_records_from_grounding_artifacts(self) -> None:
        _bundle, artifacts = build_symbolic_trace_bundle_with_artifacts(
            "weather is rain. rain becomes flood. however flood is not safe.",
            lang_hint="text",
            max_steps=8,
            max_counterexamples=2,
        )
        self.assertIsNotNone(artifacts)
        assert artifacts is not None

        ctx = SymbolicTaskContext(grounding_artifacts=artifacts)
        labels = {label for label, _ in ctx.source_fact_records()}
        counts = ctx.source_counts()

        self.assertIn("interlingua", labels)
        self.assertGreaterEqual(counts["grounding_graph_records"], 1.0)

    def test_source_records_do_not_surface_heuristic_candidate_rules_from_artifacts(self) -> None:
        _bundle, artifacts = build_symbolic_trace_bundle_with_artifacts(
            "Rule all stars generate planets.",
            lang_hint="text",
            max_steps=8,
            max_counterexamples=2,
        )
        self.assertIsNotNone(artifacts)
        assert artifacts is not None

        ctx = SymbolicTaskContext(grounding_artifacts=artifacts)
        labels = {label for label, _ in ctx.source_fact_records()}
        counts = ctx.source_counts()

        self.assertNotIn("grounding_candidate_rule", labels)
        self.assertEqual(counts["grounding_candidate_rules"], 0.0)

    def test_source_records_drop_direct_heuristic_candidate_rules(self) -> None:
        heuristic_candidate = RuleCandidate(
            clause=HornClause(
                head=HornAtom(301, (Var("X"), Var("Y"))),
                body=(
                    HornAtom(302, (Var("X"),)),
                    HornAtom(303, (Var("Y"),)),
                ),
            ),
            source="grounding_rule_compiler",
            score=0.88,
            utility=0.81,
            metadata={
                "claim_source": "fallback_extraction",
                "subject_name": "stars",
                "predicate_name": "generates",
                "object_name": "planets",
                "provenance": ("heuristic_authority:low",),
            },
        )
        ctx = SymbolicTaskContext(grounding_candidate_rules=(heuristic_candidate,))

        labels = {label for label, _ in ctx.source_fact_records()}
        counts = ctx.source_counts()

        self.assertNotIn("grounding_candidate_rule", labels)
        self.assertEqual(counts["grounding_candidate_rules"], 0.0)

    def test_source_records_surface_hidden_cause_records_from_artifacts(self) -> None:
        _bundle, artifacts = build_symbolic_trace_bundle_with_artifacts(
            "door opens but no green card",
            lang_hint="text",
            max_steps=8,
            max_counterexamples=2,
        )
        self.assertIsNotNone(artifacts)
        assert artifacts is not None

        ctx = SymbolicTaskContext(grounding_artifacts=artifacts)
        labels = {label for label, _ in ctx.source_fact_records()}
        counts = ctx.source_counts()

        self.assertIn("grounding_hidden_cause", labels)
        self.assertGreaterEqual(counts["grounding_hidden_cause_records"], 1.0)


if __name__ == "__main__":
    unittest.main()

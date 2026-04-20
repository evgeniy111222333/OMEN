from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_prolog import HornAtom, SymbolicTaskContext
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


if __name__ == "__main__":
    unittest.main()

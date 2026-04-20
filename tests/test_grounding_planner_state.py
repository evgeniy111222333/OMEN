from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import build_planner_world_state
from omen_grounding.world_state_writeback import GroundingWorldStateRecord
from omen_prolog import HornAtom, SymbolicTaskContext


class GroundingPlannerStateTest(unittest.TestCase):
    def test_build_planner_world_state_projects_world_status_buckets(self) -> None:
        active_record = GroundingWorldStateRecord(
            record_id="active:wood",
            hypothesis_id="wood",
            record_type="relation",
            world_status="active",
            segment_index=0,
            symbols=("fire", "multiplies", "wood"),
            support=0.91,
            conflict=0.05,
            confidence=0.88,
            repair_action="accept_to_world_state",
        )
        hypothetical_record = GroundingWorldStateRecord(
            record_id="hypothetical:stone",
            hypothesis_id="stone",
            record_type="relation",
            world_status="hypothetical",
            segment_index=1,
            symbols=("fire", "creates", "stone"),
            support=0.54,
            conflict=0.18,
            confidence=0.61,
            repair_action="keep_multiple_hypotheses_alive",
        )
        contradicted_record = GroundingWorldStateRecord(
            record_id="contradicted:tree",
            hypothesis_id="tree",
            record_type="state",
            world_status="contradicted",
            segment_index=2,
            symbols=("tree", "present"),
            support=0.28,
            conflict=0.83,
            confidence=0.49,
            repair_action="preserve_conflict_scope",
        )

        active_fact = HornAtom(501, (1, 2))
        hypothetical_fact = HornAtom(502, (2, 3))
        contradicted_fact = HornAtom(503, (3, 4))
        goal = HornAtom(900, (7, 8))

        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset({HornAtom(100, (0, 1))}),
            grounding_world_state_records=(active_record, hypothetical_record, contradicted_record),
            grounding_world_state_active_facts=frozenset({active_fact}),
            grounding_world_state_hypothetical_facts=frozenset({hypothetical_fact}),
            grounding_world_state_contradicted_facts=frozenset({contradicted_fact}),
            goal=goal,
            target_facts=frozenset({goal}),
            metadata={
                "grounding_uncertainty": 0.42,
                "grounding_world_state_branching_pressure": 0.55,
                "grounding_world_state_contradiction_pressure": 0.61,
                "grounding_hidden_cause_pressure": 0.40,
            },
        )

        planner_state = build_planner_world_state(ctx)

        self.assertEqual(len(planner_state.active_records), 1)
        self.assertEqual(len(planner_state.hypothetical_records), 1)
        self.assertEqual(len(planner_state.contradicted_records), 1)
        self.assertIn(active_fact, planner_state.symbolic_facts)
        self.assertIn(hypothetical_fact, planner_state.symbolic_facts)
        self.assertNotIn(contradicted_fact, planner_state.symbolic_facts)
        self.assertEqual(planner_state.primary_goal, goal)
        self.assertIn("fire", planner_state.resource_symbols)
        self.assertGreaterEqual(len(planner_state.resources), 3)
        self.assertGreaterEqual(len(planner_state.operators), 2)
        self.assertEqual(planner_state.operators[0].predicate, "multiplies")
        self.assertEqual(planner_state.operators[0].inputs, ("fire",))
        self.assertEqual(planner_state.operators[0].outputs, ("wood",))
        self.assertGreaterEqual(len(planner_state.alternative_worlds), 2)
        self.assertGreaterEqual(planner_state.summary()["planner_state_world_rules"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_hypothetical_rules"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_contradictions"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_operators"], 2.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_resource_records"], 3.0)
        self.assertAlmostEqual(planner_state.contradiction_pressure, 0.61, places=6)


if __name__ == "__main__":
    unittest.main()

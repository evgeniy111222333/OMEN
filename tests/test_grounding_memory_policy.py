from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import (
    grounding_memory_priority,
    grounding_memory_writeback_records,
    grounding_memory_writeback_status_counts,
)
from omen_grounding.world_state_writeback import GroundingWorldStateRecord


class GroundingMemoryPolicyTest(unittest.TestCase):
    def test_writeback_policy_preserves_status_diversity(self) -> None:
        records = (
            GroundingWorldStateRecord(
                record_id="active:planet",
                hypothesis_id="planet",
                record_type="relation",
                world_status="active",
                segment_index=0,
                symbols=("star", "creates", "planet"),
                support=0.92,
                conflict=0.08,
                confidence=0.90,
                repair_action="accept_to_world_state",
            ),
            GroundingWorldStateRecord(
                record_id="active:moon",
                hypothesis_id="moon",
                record_type="relation",
                world_status="active",
                segment_index=1,
                symbols=("planet", "creates", "moon"),
                support=0.88,
                conflict=0.05,
                confidence=0.87,
                repair_action="accept_to_world_state",
            ),
            GroundingWorldStateRecord(
                record_id="hypothetical:remote_open",
                hypothesis_id="remote_open",
                record_type="relation",
                world_status="hypothetical",
                segment_index=2,
                symbols=("dispatcher", "opens", "door_5"),
                support=0.57,
                conflict=0.21,
                confidence=0.60,
                repair_action="keep_multiple_hypotheses_alive",
            ),
            GroundingWorldStateRecord(
                record_id="contradicted:bob_open",
                hypothesis_id="bob_open",
                record_type="relation",
                world_status="contradicted",
                segment_index=3,
                symbols=("bob", "opens", "door_5"),
                support=0.20,
                conflict=0.86,
                confidence=0.42,
                repair_action="preserve_conflict_scope",
            ),
        )

        selected = grounding_memory_writeback_records(records, limit=3)
        counts = grounding_memory_writeback_status_counts(selected)

        self.assertEqual(len(selected), 3)
        self.assertGreaterEqual(counts["active"], 1.0)
        self.assertGreaterEqual(counts["hypothetical"], 1.0)
        self.assertGreaterEqual(counts["contradicted"], 1.0)

    def test_priority_prefers_active_world_state_over_conflicted_copy(self) -> None:
        active = GroundingWorldStateRecord(
            record_id="active:stone",
            hypothesis_id="stone",
            record_type="relation",
            world_status="active",
            segment_index=0,
            symbols=("fire", "creates", "stone"),
            support=0.90,
            conflict=0.06,
            confidence=0.91,
            repair_action="accept_to_world_state",
        )
        conflicted = GroundingWorldStateRecord(
            record_id="contradicted:stone",
            hypothesis_id="stone",
            record_type="relation",
            world_status="contradicted",
            segment_index=1,
            symbols=("fire", "creates", "stone"),
            support=0.34,
            conflict=0.81,
            confidence=0.48,
            repair_action="preserve_conflict_scope",
        )
        self.assertGreater(grounding_memory_priority(active), grounding_memory_priority(conflicted))


if __name__ == "__main__":
    unittest.main()

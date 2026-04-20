from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import (
    grounding_memory_status_counts,
    grounding_memory_status_families,
    grounding_memory_status_terms,
)
from omen_grounding.world_state_writeback import GroundingWorldStateRecord
from omen_symbolic.memory_index import SymbolicMemoryIndex


class GroundingMemoryRetrievalTest(unittest.TestCase):
    def test_symbolic_memory_index_prioritizes_contradiction_matched_records(self) -> None:
        active = GroundingWorldStateRecord(
            record_id="active:door",
            hypothesis_id="door",
            record_type="relation",
            world_status="active",
            segment_index=0,
            symbols=("door_5", "opens_with", "green_card"),
            support=0.92,
            conflict=0.05,
            confidence=0.92,
            repair_action="accept_to_world_state",
        )
        contradicted = GroundingWorldStateRecord(
            record_id="contradicted:door",
            hypothesis_id="door_conflict",
            record_type="relation",
            world_status="contradicted",
            segment_index=1,
            symbols=("door_5", "opens_with", "green_card"),
            support=0.34,
            conflict=0.81,
            confidence=0.52,
            repair_action="trigger_hidden_cause_abduction",
        )
        hypothetical = GroundingWorldStateRecord(
            record_id="hypothetical:door",
            hypothesis_id="door_alt",
            record_type="relation",
            world_status="hypothetical",
            segment_index=2,
            symbols=("door_5", "opens_with", "maintenance_override"),
            support=0.44,
            conflict=0.22,
            confidence=0.58,
            repair_action="keep_multiple_hypotheses_alive",
        )

        index = SymbolicMemoryIndex(max_entries=16)
        embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        index.write([active, contradicted, hypothetical], embeddings)

        hint_records = (contradicted,)
        terms = list(grounding_memory_status_terms(hint_records, statuses=("contradicted",)))
        families = list(grounding_memory_status_families(hint_records, statuses=("contradicted",)))
        counts = grounding_memory_status_counts(hint_records)
        recalled = index.recall(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            top_k=3,
            min_sim=0.0,
            graph_terms=list(grounding_memory_status_terms(hint_records, statuses=("contradicted", "hypothetical"))),
            graph_families=list(grounding_memory_status_families(hint_records, statuses=("contradicted",))),
            boost_graph_terms=terms,
            boost_graph_families=families,
            suppress_graph_families=list(grounding_memory_status_families((active,), statuses=("active",))),
            structured_limit=3,
        )

        self.assertEqual(counts["contradicted"], 1.0)
        self.assertGreaterEqual(len(recalled), 1)
        self.assertEqual(getattr(recalled[0], "world_status", ""), "contradicted")


if __name__ == "__main__":
    unittest.main()

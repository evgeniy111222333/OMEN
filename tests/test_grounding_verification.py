from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding.symbolic_compiler import (
    CompiledSymbolicHypothesis,
    CompiledSymbolicSegment,
    SymbolicCompilationResult,
)
from omen_grounding.verification import verify_symbolic_hypotheses


class GroundingVerificationTest(unittest.TestCase):
    def test_verification_supports_high_confidence_relation(self) -> None:
        compilation = SymbolicCompilationResult(
            language="text",
            source_text="stars generate planets",
            segments=(
                CompiledSymbolicSegment(
                    index=0,
                    text="stars generate planets",
                    normalized_text="stars generate planets",
                    relations=(("stars", "generates", "planets"),),
                ),
            ),
            hypotheses=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:0",
                    segment_index=0,
                    kind="relation",
                    symbols=("stars", "generates", "planets"),
                    confidence=0.91,
                    status="supported",
                    provenance=("segment:0", "relation:rel:0"),
                ),
            ),
        )

        report = verify_symbolic_hypotheses(compilation)

        self.assertEqual(report.metadata.get("verification_supported_hypotheses"), 1.0)
        self.assertEqual(report.records[0].verification_status, "supported")
        self.assertEqual(report.records[0].repair_action, "accept_to_world_state")
        self.assertTrue(report.records[0].graph_key.startswith("grounding_verification:supported:"))

    def test_verification_marks_counterexample_relation_for_hidden_cause_repair(self) -> None:
        compilation = SymbolicCompilationResult(
            language="text",
            source_text="door opens but no green card",
            segments=(
                CompiledSymbolicSegment(
                    index=0,
                    text="door opens but no green card",
                    normalized_text="door opens but no green card",
                    relations=(("door_5", "opens_with", "green_card"),),
                    counterexample=True,
                ),
            ),
            hypotheses=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:door",
                    segment_index=0,
                    kind="relation",
                    symbols=("door_5", "opens_with", "green_card"),
                    confidence=0.52,
                    status="proposal",
                    deferred=True,
                    conflict_tag="counterexample_context",
                    provenance=("segment:0",),
                ),
            ),
        )

        report = verify_symbolic_hypotheses(compilation)

        self.assertEqual(report.metadata.get("verification_conflicted_hypotheses"), 1.0)
        self.assertGreaterEqual(report.metadata.get("verification_hidden_cause_pressure", 0.0), 1.0)
        self.assertEqual(report.records[0].verification_status, "conflicted")
        self.assertTrue(report.records[0].hidden_cause_candidate)
        self.assertEqual(report.records[0].repair_action, "trigger_hidden_cause_abduction")


if __name__ == "__main__":
    unittest.main()

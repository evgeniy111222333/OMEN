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
from omen_grounding.world_state_atoms import (
    GROUND_WORLD_ACTIVE_RELATION_PRED,
    GROUND_WORLD_CONTRADICTED_RELATION_PRED,
    compile_world_state_symbolic_atoms,
)
from omen_grounding.world_state_writeback import build_grounding_world_state_writeback


class GroundingWorldStateWritebackTest(unittest.TestCase):
    def test_writeback_separates_active_and_contradicted_records(self) -> None:
        compilation = SymbolicCompilationResult(
            language="text",
            source_text="stars generate planets. however door opens without card.",
            segments=(
                CompiledSymbolicSegment(
                    index=0,
                    text="stars generate planets",
                    normalized_text="stars generate planets",
                    relations=(("stars", "generates", "planets"),),
                ),
                CompiledSymbolicSegment(
                    index=1,
                    text="door opens without card",
                    normalized_text="door opens without card",
                    relations=(("door_5", "opens_with", "green_card"),),
                    counterexample=True,
                ),
            ),
            hypotheses=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:stars",
                    segment_index=0,
                    kind="relation",
                    symbols=("stars", "generates", "planets"),
                    confidence=0.92,
                    status="supported",
                    provenance=("segment:0",),
                ),
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:door",
                    segment_index=1,
                    kind="relation",
                    symbols=("door_5", "opens_with", "green_card"),
                    confidence=0.52,
                    status="proposal",
                    deferred=True,
                    conflict_tag="counterexample_context",
                    provenance=("segment:1",),
                ),
            ),
        )

        verification = verify_symbolic_hypotheses(compilation)
        writeback = build_grounding_world_state_writeback(compilation, verification)

        self.assertEqual(writeback.metadata.get("grounding_world_state_records"), 3.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_active_records"), 1.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_contradicted_records"), 1.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_hypothetical_records"), 1.0)
        self.assertAlmostEqual(writeback.metadata.get("grounding_world_state_hypothetical_ratio", 0.0), 1.0 / 3.0, places=6)
        self.assertEqual(writeback.metadata.get("grounding_world_state_hidden_cause_records"), 1.0)
        self.assertGreaterEqual(writeback.metadata.get("grounding_world_state_mean_conflict", 0.0), 0.2)
        self.assertGreaterEqual(writeback.metadata.get("grounding_world_state_contradiction_pressure", 0.0), 0.3)
        self.assertTrue(any(record.world_status == "active" for record in writeback.records))
        self.assertTrue(any(record.world_status == "contradicted" for record in writeback.records))
        self.assertTrue(any(record.record_type == "hidden_cause" for record in writeback.records))
        self.assertTrue(all(record.graph_key.startswith("grounding_world_state:") for record in writeback.records))

        active_facts, hypothetical_facts, contradicted_facts, stats = compile_world_state_symbolic_atoms(writeback.records)
        self.assertEqual(stats.get("grounding_world_state_active_facts"), 1.0)
        self.assertEqual(stats.get("grounding_world_state_hypothetical_facts"), 1.0)
        self.assertEqual(stats.get("grounding_world_state_contradicted_facts"), 1.0)
        self.assertEqual(stats.get("grounding_world_state_hidden_cause_facts"), 1.0)
        self.assertEqual(len(hypothetical_facts), 1)
        self.assertTrue(any(getattr(fact, "pred", None) == GROUND_WORLD_ACTIVE_RELATION_PRED for fact in active_facts))
        self.assertTrue(
            any(getattr(fact, "pred", None) == GROUND_WORLD_CONTRADICTED_RELATION_PRED for fact in contradicted_facts)
        )

    def test_writeback_keeps_supported_cited_claims_hypothetical(self) -> None:
        compilation = SymbolicCompilationResult(
            language="text",
            source_text="Abstract: aspirin causes relief (Smith, 2024).",
            segments=(
                CompiledSymbolicSegment(
                    index=0,
                    text="Abstract: aspirin causes relief (Smith, 2024).",
                    normalized_text="abstract aspirin causes relief smith 2024",
                    relations=(("aspirin", "causes", "relief"),),
                ),
            ),
            hypotheses=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:cited",
                    segment_index=0,
                    kind="relation",
                    symbols=("aspirin", "causes", "relief"),
                    confidence=0.95,
                    status="supported",
                    deferred=True,
                    speaker_key="paper",
                    epistemic_status="cited",
                    claim_source="citation_region",
                    provenance=("segment:0", "structural_unit:citation:0:0"),
                ),
            ),
        )

        verification = verify_symbolic_hypotheses(compilation)
        writeback = build_grounding_world_state_writeback(compilation, verification)

        self.assertEqual(writeback.records[0].world_status, "hypothetical")
        self.assertEqual(writeback.records[0].epistemic_status, "cited")
        self.assertEqual(writeback.metadata.get("grounding_world_state_active_records"), 0.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_hypothetical_records"), 1.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_cited_records"), 1.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_nonasserted_records"), 1.0)

    def test_writeback_routes_supported_rule_claims_to_symbolic_rule_lifecycle(self) -> None:
        compilation = SymbolicCompilationResult(
            language="text",
            source_text="Rule all stars generate planets.",
            segments=(
                CompiledSymbolicSegment(
                    index=0,
                    text="Rule all stars generate planets.",
                    normalized_text="rule all stars generate planets",
                    relations=(("stars", "generates", "planets"),),
                ),
            ),
            hypotheses=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:rule",
                    segment_index=0,
                    kind="relation",
                    symbols=("stars", "generates", "planets"),
                    confidence=0.96,
                    status="supported",
                    semantic_mode="rule",
                    quantifier_mode="generic_all",
                    provenance=("segment:0",),
                ),
            ),
        )

        verification = verify_symbolic_hypotheses(compilation)
        writeback = build_grounding_world_state_writeback(compilation, verification)

        self.assertEqual(writeback.records[0].world_status, "hypothetical")
        self.assertEqual(writeback.records[0].repair_action, "route_to_symbolic_rule_lifecycle")
        self.assertEqual(writeback.records[0].semantic_mode, "rule")
        self.assertEqual(writeback.records[0].quantifier_mode, "generic_all")
        self.assertEqual(writeback.metadata.get("grounding_world_state_active_records"), 0.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_hypothetical_records"), 1.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_rule_records"), 1.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_rule_lifecycle_records"), 1.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_rule_lifecycle_ratio"), 1.0)

        active_facts, hypothetical_facts, contradicted_facts, stats = compile_world_state_symbolic_atoms(writeback.records)
        self.assertEqual(stats.get("grounding_world_state_active_facts"), 0.0)
        self.assertEqual(stats.get("grounding_world_state_hypothetical_facts"), 1.0)
        self.assertEqual(len(active_facts), 0)
        self.assertEqual(len(contradicted_facts), 0)
        self.assertEqual(len(hypothetical_facts), 1)

    def test_writeback_keeps_heuristic_conflicts_hypothetical_until_confirmed(self) -> None:
        compilation = SymbolicCompilationResult(
            language="text",
            source_text="door opens but no green card",
            segments=(
                CompiledSymbolicSegment(
                    index=0,
                    text="door opens but no green card",
                    normalized_text="door opens but no green card",
                    relations=(("door", "opens_with", "green_card"),),
                    counterexample=True,
                ),
            ),
            hypotheses=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:door",
                    segment_index=0,
                    kind="relation",
                    symbols=("door", "opens_with", "green_card"),
                    confidence=0.52,
                    status="proposal",
                    deferred=True,
                    conflict_tag="counterexample_context",
                    claim_source="fallback_extraction",
                    provenance=("segment:0", "heuristic_authority:low"),
                ),
            ),
        )

        verification = verify_symbolic_hypotheses(compilation)
        writeback = build_grounding_world_state_writeback(compilation, verification)

        relation_record = next(record for record in writeback.records if record.record_type == "relation")
        self.assertEqual(verification.records[0].verification_status, "deferred")
        self.assertEqual(relation_record.world_status, "hypothetical")
        self.assertEqual(relation_record.repair_action, "require_grounding_confirmation")
        self.assertEqual(writeback.metadata.get("grounding_world_state_contradicted_records"), 0.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_hypothetical_records"), 2.0)
        self.assertEqual(writeback.metadata.get("grounding_world_state_hidden_cause_records"), 1.0)

        active_facts, hypothetical_facts, contradicted_facts, stats = compile_world_state_symbolic_atoms(writeback.records)
        self.assertEqual(stats.get("grounding_world_state_active_facts"), 0.0)
        self.assertEqual(stats.get("grounding_world_state_hypothetical_facts"), 2.0)
        self.assertEqual(stats.get("grounding_world_state_contradicted_facts"), 0.0)
        self.assertEqual(len(active_facts), 0)
        self.assertEqual(len(hypothetical_facts), 2)
        self.assertEqual(len(contradicted_facts), 0)


if __name__ == "__main__":
    unittest.main()

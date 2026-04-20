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
from omen_grounding.pipeline import ground_text_to_symbolic
from omen_grounding.types import (
    GroundedStructuralUnit,
    GroundedTextDocument,
    GroundedTextSegment,
    GroundingSourceProfile,
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

    def test_verification_uses_document_scene_and_interlingua_alignment(self) -> None:
        result = ground_text_to_symbolic(
            "weather is rain. rain becomes flood. goal evacuation safe_exit.",
            language="text",
            max_segments=6,
        )

        self.assertGreater(result.verification.metadata.get("verification_document_alignment", 0.0), 0.0)
        self.assertGreater(result.verification.metadata.get("verification_scene_alignment", 0.0), 0.0)
        self.assertGreater(result.verification.metadata.get("verification_interlingua_alignment", 0.0), 0.0)
        self.assertTrue(any(record.provenance for record in result.verification.records))

    def test_verification_uses_dialogue_structural_support_signals(self) -> None:
        result = ground_text_to_symbolic(
            "User: goal safe_exit.\nAssistant: stars generate planets.",
            language="text",
            max_segments=6,
        )

        self.assertGreater(result.verification.metadata.get("verification_structural_alignment", 0.0), 0.0)
        self.assertGreater(result.verification.metadata.get("verification_structural_provenance_support", 0.0), 0.0)
        self.assertGreater(result.verification.metadata.get("verification_dialogue_structural_support", 0.0), 0.0)
        self.assertTrue(
            any(
                record.verification_status == "supported"
                and any(str(item).startswith("structural_unit:speaker_turn") for item in record.provenance)
                for record in result.verification.records
            )
        )

    def test_verification_citation_regions_increase_scientific_support(self) -> None:
        routing = GroundingSourceProfile(
            language="text",
            script="latin",
            domain="observation_text",
            modality="natural_text",
            subtype="scientific_text",
            verification_path="scientific_claim_verification",
            confidence=0.72,
            ambiguity=0.18,
        )
        segment_without_citation = GroundedTextSegment(
            index=0,
            text="Abstract: aspirin causes relief.",
            normalized_text="abstract aspirin causes relief",
            routing=routing,
            tokens=("abstract", "aspirin", "causes", "relief"),
        )
        segment_with_citation = GroundedTextSegment(
            index=0,
            text="Abstract: aspirin causes relief (Smith, 2024).",
            normalized_text="abstract aspirin causes relief smith 2024",
            routing=routing,
            tokens=("abstract", "aspirin", "causes", "relief", "smith", "2024"),
            structural_units=(
                GroundedStructuralUnit(
                    unit_id="citation:0:0",
                    unit_type="citation_region",
                    text="(Smith, 2024)",
                    source_segment=0,
                    confidence=0.66,
                    status="supported",
                    references=("subtype:scientific_text",),
                ),
            ),
        )
        document_without_citation = GroundedTextDocument(
            language="text",
            source_text=segment_without_citation.text,
            routing=routing,
            segments=(segment_without_citation,),
        )
        document_with_citation = GroundedTextDocument(
            language="text",
            source_text=segment_with_citation.text,
            routing=routing,
            structural_units=segment_with_citation.structural_units,
            segments=(segment_with_citation,),
        )
        compilation = SymbolicCompilationResult(
            language="text",
            source_text=segment_with_citation.text,
            segments=(
                CompiledSymbolicSegment(
                    index=0,
                    text=segment_with_citation.text,
                    normalized_text=segment_with_citation.normalized_text,
                    relations=(("aspirin", "causes", "relief"),),
                ),
            ),
            hypotheses=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:sci",
                    segment_index=0,
                    kind="relation",
                    symbols=("aspirin", "causes", "relief"),
                    confidence=0.58,
                    status="proposal",
                    provenance=("segment:0", "structural_unit:citation:0:0"),
                ),
            ),
        )

        report_without = verify_symbolic_hypotheses(compilation, document=document_without_citation)
        report_with = verify_symbolic_hypotheses(compilation, document=document_with_citation)

        self.assertEqual(report_without.metadata.get("verification_citation_support", 0.0), 0.0)
        self.assertGreater(report_with.metadata.get("verification_citation_support", 0.0), 0.0)
        self.assertGreater(report_with.records[0].support, report_without.records[0].support)
        self.assertLessEqual(report_with.records[0].conflict, report_without.records[0].conflict)

    def test_verification_tracks_claim_attribution_and_nonasserted_pressure(self) -> None:
        compilation = SymbolicCompilationResult(
            language="text",
            source_text="Assistant: maybe stars generate planets?",
            segments=(
                CompiledSymbolicSegment(
                    index=0,
                    text="Assistant: maybe stars generate planets?",
                    normalized_text="assistant maybe stars generate planets",
                    relations=(("stars", "generates", "planets"),),
                ),
            ),
            hypotheses=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="rel:dialogue",
                    segment_index=0,
                    kind="relation",
                    symbols=("stars", "generates", "planets"),
                    confidence=0.74,
                    status="proposal",
                    deferred=True,
                    speaker_key="assistant",
                    epistemic_status="questioned",
                    claim_source="speaker_turn",
                    provenance=("segment:0", "structural_unit:speaker_turn:0"),
                ),
            ),
        )

        report = verify_symbolic_hypotheses(compilation)

        self.assertEqual(report.metadata.get("verification_claim_attribution_support"), 1.0)
        self.assertEqual(report.metadata.get("verification_nonasserted_pressure"), 1.0)
        self.assertEqual(report.records[0].speaker_key, "assistant")
        self.assertEqual(report.records[0].epistemic_status, "questioned")
        self.assertEqual(report.records[0].claim_source, "speaker_turn")
        self.assertIn("speaker:assistant", report.records[0].graph_terms)
        self.assertIn("epistemic:questioned", report.records[0].graph_terms)
        self.assertIn("claim_source:speaker_turn", report.records[0].graph_terms)


if __name__ == "__main__":
    unittest.main()

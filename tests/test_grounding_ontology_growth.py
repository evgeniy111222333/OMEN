from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import ground_text_to_symbolic
from omen_symbolic.execution_trace import build_symbolic_trace_bundle


class GroundingOntologyGrowthTest(unittest.TestCase):
    def test_pipeline_promotes_repeated_segment_pattern_into_ontology_concept(self) -> None:
        text = "\n".join(
            [
                "user=guest result=failed_login ip=external alert=triggered",
                "user=unknown result=failed_login ip=external alert=triggered",
                "user=hacker result=failed_login ip=external alert=triggered",
                "user=admin result=success ip=internal alert=none",
            ]
        )

        result = ground_text_to_symbolic(text, language="log", max_segments=8)

        self.assertGreaterEqual(result.ontology.metadata.get("grounding_ontology_records", 0.0), 1.0)
        self.assertGreaterEqual(result.ontology.metadata.get("grounding_ontology_active_records", 0.0), 1.0)
        self.assertGreaterEqual(result.ontology.metadata.get("grounding_ontology_support", 0.0), 0.5)
        concept = result.ontology.concepts[0]
        self.assertEqual(concept.record_type, "ontology")
        self.assertIn("state:result=failed_login", concept.signature_terms)
        self.assertIn("state:ip=external", concept.signature_terms)
        self.assertIn("state:alert=triggered", concept.signature_terms)
        self.assertTrue({"guest", "unknown", "hacker"}.intersection(set(concept.member_terms)))
        self.assertEqual(concept.world_status, "active")

    def test_trace_bundle_carries_ontology_records_and_facts(self) -> None:
        text = "\n".join(
            [
                "user=guest result=failed_login ip=external alert=triggered",
                "user=unknown result=failed_login ip=external alert=triggered",
                "user=hacker result=failed_login ip=external alert=triggered",
            ]
        )

        bundle = build_symbolic_trace_bundle(text, lang_hint="log", max_steps=8, max_counterexamples=2)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_ontology_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_ontology_active_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_ontology_facts", 0.0)), 1.0)
        self.assertGreaterEqual(len(bundle.grounding_ontology_records), 1)
        self.assertGreaterEqual(len(bundle.grounding_ontology_facts), 1)


if __name__ == "__main__":
    unittest.main()

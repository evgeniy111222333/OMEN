from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import SemanticSceneGraph, SemanticEntity, SemanticEvent, ground_text_to_symbolic


class _FixedBackbone:
    def build_scene_graph(self, document):
        return SemanticSceneGraph(
            language=document.language,
            source_text=document.source_text,
            entities=(
                SemanticEntity(entity_id="ent:a", canonical_name="alpha", confidence=0.9, status="supported"),
                SemanticEntity(entity_id="ent:b", canonical_name="beta", confidence=0.9, status="supported"),
            ),
            events=(
                SemanticEvent(
                    event_id="event:0",
                    event_type="causes",
                    subject_entity_id="ent:a",
                    object_entity_id="ent:b",
                    subject_name="alpha",
                    object_name="beta",
                    source_segment=0,
                    confidence=0.95,
                    status="supported",
                ),
            ),
            metadata={"scene_entities": 2.0, "scene_events": 1.0, "scene_claims": 1.0},
        )


class GroundingScenePipelineTest(unittest.TestCase):
    def test_pipeline_builds_scene_entities_events_and_compiled_relations(self) -> None:
        text = (
            'Факт 1: Об\'єкти типу "Зірки" генерують об\'єкти типу "Планети".\n'
            'Факт 2: Об\'єкти типу "Планети" генерують об\'єкти типу "Супутники".\n'
            'Мета: "Супутники".'
        )

        result = ground_text_to_symbolic(text, language="uk", max_segments=8)

        self.assertGreaterEqual(len(result.scene.entities), 3)
        self.assertGreaterEqual(len(result.scene.events), 2)
        self.assertGreaterEqual(len(result.scene.goals), 1)
        self.assertGreaterEqual(len(result.scene.claims), 3)
        self.assertGreaterEqual(len(result.interlingua.entities), 3)
        self.assertGreaterEqual(len(result.interlingua.relations), 2)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_entities", 0.0), 3.0)
        self.assertEqual(result.scene.metadata.get("scene_entities"), float(len(result.scene.entities)))
        self.assertEqual(result.compiled.metadata.get("compiled_segments"), float(len(result.compiled.segments)))
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_hypotheses", 0.0), 3.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_mean_confidence", 0.0), 0.5)
        self.assertGreaterEqual(result.verification.metadata.get("verification_records", 0.0), 3.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_supported_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_records", 0.0), 3.0)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_active_records", 0.0), 1.0)
        compiled_relations = {
            relation
            for segment in result.compiled.segments
            for relation in segment.relations
        }
        self.assertIn(("зірки", "generates", "планети"), compiled_relations)
        self.assertIn(("планети", "generates", "супутники"), compiled_relations)
        relation_hypotheses = [
            hypothesis for hypothesis in result.compiled.hypotheses if hypothesis.kind == "relation"
        ]
        self.assertGreaterEqual(len(relation_hypotheses), 2)
        self.assertTrue(all(hypothesis.provenance for hypothesis in relation_hypotheses))

    def test_pipeline_preserves_counterexample_signal_into_compiled_segments(self) -> None:
        text = (
            'Правило: "Зелена картка" відчиняє "Двері".\n'
            'Факт: У Боба немає зеленої картки.'
        )

        result = ground_text_to_symbolic(text, language="uk", max_segments=8)

        self.assertEqual(result.document.metadata.get("grounding_multilingual"), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_claims", 0.0), 1.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_relations", 0.0), 1.0)
        self.assertTrue(any(segment.counterexample for segment in result.compiled.segments))
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_deferred_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_conflicted_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_repair_pressure", 0.0), 0.5)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_contradicted_records", 0.0), 1.0)
        self.assertTrue(any(hypothesis.deferred for hypothesis in result.compiled.hypotheses))
        self.assertTrue(any(record.hidden_cause_candidate or record.verification_status != "supported" for record in result.verification.records))

    def test_pipeline_uses_injected_backbone_scene_graph(self) -> None:
        result = ground_text_to_symbolic("unstructured placeholder text", language="text", backbone=_FixedBackbone())

        self.assertEqual(len(result.scene.entities), 2)
        self.assertEqual(len(result.scene.events), 1)
        self.assertEqual(len(result.interlingua.relations), 1)
        compiled_relations = {
            relation
            for segment in result.compiled.segments
            for relation in segment.relations
        }
        self.assertIn(("alpha", "causes", "beta"), compiled_relations)


if __name__ == "__main__":
    unittest.main()

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
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active"), 1.0)
        self.assertEqual(result.document.metadata.get("grounding_document_semantic_authority"), 0.0)
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
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active", 0.0), 1.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_relations", 0.0), 1.0)
        self.assertTrue(any(segment.counterexample for segment in result.compiled.segments))
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_deferred_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_conflicted_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_repair_pressure", 0.0), 0.3)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_contradicted_records", 0.0), 1.0)
        self.assertTrue(any(hypothesis.deferred for hypothesis in result.compiled.hypotheses))
        self.assertTrue(any(record.hidden_cause_candidate or record.verification_status != "supported" for record in result.verification.records))

    def test_pipeline_propagates_condition_temporal_explanation_and_coreference(self) -> None:
        text = (
            "Якщо тривога спрацювала, диспетчер відкриває двері о 10:00.\n"
            "Потім він відкриває шлюз, бо евакуація активна.\n"
            "Система повинна створювати безпечний вихід."
        )

        result = ground_text_to_symbolic(text, language="uk", max_segments=8)

        self.assertEqual(result.document.metadata.get("grounding_document_semantic_authority", 1.0), 0.0)
        self.assertEqual(result.document.metadata.get("grounding_event_hints", 1.0), 0.0)
        self.assertEqual(result.document.metadata.get("grounding_condition_hints", 1.0), 0.0)
        self.assertEqual(result.document.metadata.get("grounding_explanation_hints", 1.0), 0.0)
        self.assertEqual(result.document.metadata.get("grounding_temporal_hints", 1.0), 0.0)
        self.assertEqual(result.document.metadata.get("grounding_modal_hints", 1.0), 0.0)
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_fallback_low_authority", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_fallback_event_proposals", 0.0), 3.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_fallback_relation_proposals", 0.0), 3.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_event_conditions", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_event_explanations", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_event_temporal_anchors", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_event_modalities", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_coreference_links", 0.0), 1.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_conditioned_relations", 0.0), 1.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_explained_relations", 0.0), 1.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_temporal_relations", 0.0), 1.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_modal_relations", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_event_frames", 0.0), 3.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_conditioned_relations", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_explained_relations", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_temporal_relations", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_modal_relations", 0.0), 1.0)
        self.assertTrue(any(relation.condition_key for relation in result.interlingua.relations))
        self.assertTrue(any(relation.explanation_key for relation in result.interlingua.relations))
        self.assertTrue(any(relation.temporal_key for relation in result.interlingua.relations))
        self.assertTrue(any(relation.modality for relation in result.interlingua.relations))
        self.assertTrue(any(relation.relation_modifiers for relation in result.interlingua.relations))

    def test_pipeline_routes_config_text_into_structural_state_path(self) -> None:
        text = (
            "[safety]\n"
            "status: armed\n"
            "mode=evac\n"
            "priority: high"
        )

        result = ground_text_to_symbolic(text, language="config", max_segments=8)

        self.assertIsNotNone(result.document.routing)
        assert result.document.routing is not None
        self.assertEqual(result.document.routing.modality, "structured_text")
        self.assertEqual(result.document.routing.subtype, "config_text")
        self.assertEqual(result.scene.metadata.get("scene_structural_primary_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active", 1.0), 0.0)
        self.assertGreaterEqual(result.document.metadata.get("grounding_key_value_units", 0.0), 3.0)
        self.assertGreaterEqual(len(result.scene.states), 3)
        self.assertEqual(len(result.scene.events), 0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_state_claims", 0.0), 3.0)
        self.assertEqual(result.compiled.metadata.get("compiled_relation_claims", 0.0), 0.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_structural_evidence_refs", 0.0), 3.0)
        self.assertTrue(
            any(
                str(item).startswith("structural_unit:")
                for hypothesis in result.compiled.hypotheses
                for item in hypothesis.provenance
            )
        )
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_active_records", 0.0), 1.0)

    def test_pipeline_uses_structural_primary_scene_for_json_records(self) -> None:
        text = (
            '{"step":1,"weather":"rain","road":"wet"}\n'
            '{"step":2,"goal":"safe_exit","status":"ready"}'
        )

        result = ground_text_to_symbolic(text, language="json", max_segments=8)

        self.assertIsNotNone(result.document.routing)
        assert result.document.routing is not None
        self.assertEqual(result.document.routing.modality, "structured_text")
        self.assertEqual(result.document.routing.subtype, "json_records")
        self.assertEqual(result.scene.metadata.get("scene_structural_primary_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active", 1.0), 0.0)
        self.assertGreaterEqual(len(result.scene.states), 3)
        self.assertGreaterEqual(len(result.scene.goals), 1)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_structural_evidence_refs", 0.0), 3.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_structural_evidence_refs", 0.0), 3.0)
        self.assertTrue(any(goal.evidence_refs for goal in result.scene.goals))
        self.assertTrue(
            any(
                str(item).startswith("structural_unit:json_record")
                for hypothesis in result.compiled.hypotheses
                for item in hypothesis.provenance
            )
        )

    def test_pipeline_merges_structural_primary_with_fallback_for_mixed_documents(self) -> None:
        text = (
            "[service]\n"
            "status: armed\n"
            "priority: high\n"
            "Dispatcher opens gate because alarm is active."
        )

        result = ground_text_to_symbolic(text, language="text", max_segments=8)

        self.assertEqual(result.scene.metadata.get("scene_structural_primary_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_structural_primary_hybrid_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active", 0.0), 1.0)
        self.assertGreaterEqual(len(result.scene.states), 2)
        self.assertGreaterEqual(len(result.scene.events), 1)
        self.assertTrue(
            all(
                state.status == "supported" and state.evidence_refs
                for state in result.scene.states
                if int(state.source_segment) in {1, 2}
            )
        )
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_state_claims", 0.0), 2.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_relation_claims", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_structural_evidence_refs", 0.0), 2.0)
        state_hypotheses = [hypothesis for hypothesis in result.compiled.hypotheses if hypothesis.kind == "state"]
        relation_hypotheses = [hypothesis for hypothesis in result.compiled.hypotheses if hypothesis.kind == "relation"]
        self.assertTrue(
            all(
                any(str(item).startswith("structural_unit:") for item in hypothesis.provenance)
                for hypothesis in state_hypotheses
            )
        )
        self.assertTrue(
            any(
                not any(str(item).startswith("structural_unit:") for item in hypothesis.provenance)
                for hypothesis in relation_hypotheses
            )
        )

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

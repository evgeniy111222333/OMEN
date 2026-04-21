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
        self.assertGreaterEqual(result.verification.metadata.get("verification_deferred_hypotheses", 0.0), 3.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_heuristic_records", 0.0), 3.0)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_records", 0.0), 3.0)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_hypothetical_records", 0.0), 3.0)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_heuristic_records", 0.0), 3.0)
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
        self.assertGreaterEqual(result.verification.metadata.get("verification_deferred_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_hidden_cause_records", 0.0), 1.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_repair_pressure", 0.0), 0.3)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_hypothetical_records", 0.0), 1.0)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_hidden_cause_records", 0.0), 1.0)
        self.assertTrue(any(hypothesis.deferred for hypothesis in result.compiled.hypotheses))
        self.assertTrue(any(record.hidden_cause_candidate or record.verification_status != "supported" for record in result.verification.records))
        self.assertTrue(any(record.record_type == "hidden_cause" for record in result.world_state.records))

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
        self.assertGreater(float(result.document.metadata.get("grounding_document_state_authority", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)
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
        self.assertGreater(float(result.document.metadata.get("grounding_document_state_authority", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_goal_authority", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)
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

        self.assertGreater(float(result.document.metadata.get("grounding_document_state_authority", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)
        self.assertEqual(result.scene.metadata.get("scene_structural_primary_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_structural_primary_hybrid_active", 1.0), 0.0)
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
                hypothesis.claim_source == "fallback_extraction"
                and any(str(item).startswith("heuristic_authority:") for item in hypothesis.provenance)
                for hypothesis in relation_hypotheses
            )
        )

    def test_pipeline_enforces_hybrid_segment_ownership_for_dialogue_plus_free_text(self) -> None:
        text = (
            "User: goal safe_exit.\n"
            "Assistant: if alarm is active open gate after inspection.\n"
            "Weather is storm."
        )

        result = ground_text_to_symbolic(text, language="text", max_segments=8)

        self.assertEqual(result.scene.metadata.get("scene_structural_primary_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_structural_primary_hybrid_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_segment_owner_hybrid", 0.0), 2.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_segment_owner_fallback_primary", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_hybrid_retained_fallback_events", 1.0), 0.0)
        self.assertEqual(result.scene.metadata.get("scene_hybrid_retained_fallback_claims", 1.0), 0.0)
        self.assertTrue(any(int(event.source_segment) in {0, 1} for event in result.scene.events))
        self.assertTrue(
            all(
                int(event.source_segment) not in {0, 1}
                or not float(getattr(event, "metadata", {}).get("fallback_backbone", 0.0))
                for event in result.scene.events
            )
        )
        self.assertTrue(any(event.evidence_refs for event in result.scene.events if int(event.source_segment) in {0, 1}))
        self.assertGreaterEqual(result.scene.metadata.get("scene_event_conditions", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_event_temporal_anchors", 0.0), 1.0)
        self.assertFalse(
            any(
                int(claim.source_segment) in {0, 1}
                and claim.claim_source == "fallback_extraction"
                and claim.speaker_entity_id
                for claim in result.scene.claims
            )
        )

    def test_pipeline_routes_dialogue_text_into_structural_goal_and_relation_path(self) -> None:
        text = (
            "User: goal safe_exit.\n"
            "Assistant: stars generate planets."
        )

        result = ground_text_to_symbolic(text, language="text", max_segments=8)

        self.assertIsNotNone(result.document.routing)
        assert result.document.routing is not None
        self.assertEqual(result.document.routing.modality, "natural_text")
        self.assertEqual(result.document.routing.subtype, "dialogue_text")
        self.assertGreater(float(result.document.metadata.get("grounding_goal_hints", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_relation_hints", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_goal_authority", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_relation_authority", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)
        self.assertEqual(result.scene.metadata.get("scene_structural_primary_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active", 1.0), 0.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_structural_primary_segments", 0.0), 2.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_structural_primary_units_used", 0.0), 2.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_structural_primary_unit_speaker_turn", 0.0), 2.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_events", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_goals", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_relation_claims", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_goal_claims", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_claim_attributed", 0.0), 2.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_attributed_claim_frames", 0.0), 2.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_attributed_hypotheses", 0.0), 2.0)
        self.assertGreaterEqual(
            result.world_state.metadata.get("grounding_world_state_attributed_records", 0.0),
            2.0,
        )
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_structural_evidence_refs", 0.0), 2.0)
        self.assertTrue(
            any(
                str(item).startswith("structural_unit:speaker_turn")
                for hypothesis in result.compiled.hypotheses
                for item in hypothesis.provenance
            )
        )
        self.assertTrue(any(claim.speaker_entity_id for claim in result.scene.claims))
        self.assertTrue(
            any(
                hypothesis.speaker_key and hypothesis.claim_source == "structural_nl_fallback"
                for hypothesis in result.compiled.hypotheses
            )
        )
        self.assertTrue(any(event.evidence_refs for event in result.scene.events))
        self.assertTrue(any(goal.evidence_refs for goal in result.scene.goals))

    def test_pipeline_routes_instruction_sequences_into_structural_primary_scene(self) -> None:
        text = (
            "step 1 open panel\n"
            "step 2 enable backup pump\n"
            "if pressure drops then trigger alarm\n"
            "goal stable cooling"
        )

        result = ground_text_to_symbolic(text, language="text", max_segments=8)

        self.assertEqual(result.document.routing.subtype, "instructional_text")
        self.assertGreater(float(result.document.metadata.get("grounding_goal_hints", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_relation_hints", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_goal_authority", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_relation_authority", 0.0)), 0.0)
        self.assertGreater(float(result.document.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)
        self.assertEqual(result.scene.metadata.get("scene_structural_primary_active", 0.0), 1.0)
        self.assertEqual(result.scene.metadata.get("scene_fallback_backbone_active", 1.0), 0.0)
        self.assertGreaterEqual(len(result.scene.goals), 1)
        self.assertGreaterEqual(len(result.scene.events), 1)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_goal_claims", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_relation_claims", 0.0), 1.0)

    def test_pipeline_keeps_cited_scientific_claims_nonasserted_until_world_state(self) -> None:
        text = "Abstract: aspirin causes relief (Smith, 2024)."

        result = ground_text_to_symbolic(text, language="text", max_segments=6)

        self.assertIsNotNone(result.document.routing)
        assert result.document.routing is not None
        self.assertEqual(result.document.routing.subtype, "scientific_text")
        self.assertGreaterEqual(result.scene.metadata.get("scene_claim_nonasserted", 0.0), 1.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_cited_claim_frames", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_nonasserted_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_nonasserted_pressure", 0.0), 1.0)
        self.assertGreaterEqual(result.world_state.metadata.get("grounding_world_state_cited_records", 0.0), 1.0)
        self.assertGreaterEqual(
            result.world_state.metadata.get("grounding_world_state_nonasserted_records", 0.0),
            1.0,
        )
        self.assertTrue(any(claim.epistemic_status == "cited" for claim in result.scene.claims))
        self.assertTrue(any(hypothesis.epistemic_status == "cited" for hypothesis in result.compiled.hypotheses))
        self.assertTrue(
            all(record.world_status != "active" for record in result.world_state.records if record.epistemic_status == "cited")
        )

    def test_pipeline_keeps_heuristic_rule_like_relations_out_of_candidate_rules(self) -> None:
        text = "Rule all stars generate planets."

        result = ground_text_to_symbolic(text, language="text", max_segments=6)

        self.assertGreaterEqual(result.scene.metadata.get("scene_claim_rule", 0.0), 1.0)
        self.assertGreaterEqual(result.interlingua.metadata.get("interlingua_rule_claim_frames", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_rule_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_quantified_hypotheses", 0.0), 1.0)
        self.assertGreaterEqual(result.compiled.metadata.get("compiled_filtered_heuristic_candidate_rules", 0.0), 1.0)
        self.assertEqual(result.compiled.metadata.get("compiled_candidate_rules", 0.0), 0.0)
        self.assertEqual(len(result.compiled.candidate_rules), 0)
        self.assertGreaterEqual(result.verification.metadata.get("verification_heuristic_records", 0.0), 1.0)
        self.assertTrue(
            any(
                hypothesis.claim_source == "fallback_extraction"
                and hypothesis.semantic_mode == "rule"
                and hypothesis.quantifier_mode == "generic_all"
                for hypothesis in result.compiled.hypotheses
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

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import ground_text_document
from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig
from omen_symbolic.execution_trace import (
    TRACE_TEXT_RELATION_PRED,
    build_symbolic_trace_bundle,
    build_symbolic_trace_bundle_with_artifacts,
)


def _utf8_prompt_row(text: str, seq_len: int) -> torch.Tensor:
    encoded = list(text.encode("utf-8"))[: seq_len]
    if len(encoded) < seq_len:
        encoded = encoded + [0] * (seq_len - len(encoded))
    return torch.tensor([encoded[:-1]], dtype=torch.long)


def _assert_span_roundtrip(testcase: unittest.TestCase, source_text: str, span) -> None:
    testcase.assertIsNotNone(span)
    assert span is not None
    testcase.assertIsNotNone(span.byte_start)
    testcase.assertIsNotNone(span.byte_end)
    raw = source_text.encode("utf-8")
    byte_start = int(span.byte_start)
    byte_end = int(span.byte_end)
    testcase.assertGreaterEqual(byte_start, 0)
    testcase.assertGreaterEqual(byte_end, byte_start)
    testcase.assertEqual(raw[byte_start:byte_end], span.text.encode("utf-8"))
    testcase.assertEqual(source_text[int(span.start) : int(span.end)], span.text)


class GroundingTextSemanticsTest(unittest.TestCase):
    def test_ground_text_document_keeps_natural_language_semantics_out_of_document_layer(self) -> None:
        text = (
            'Факт 1: Об\'єкти типу "Зірки" генерують об\'єкти типу "Планети".\n'
            'Факт 2: Об\'єкти типу "Планети" генерують об\'єкти типу "Супутники".'
        )

        grounded = ground_text_document(text, language="uk")

        relations = {
            (hint.left, hint.relation, hint.right)
            for segment in grounded.segments
            for hint in segment.relations
        }
        self.assertEqual(relations, set())
        self.assertEqual(grounded.metadata["grounding_document_semantic_authority"], 0.0)
        self.assertEqual(grounded.metadata["grounding_relation_hints"], 0.0)
        self.assertGreaterEqual(grounded.metadata["grounding_tokens"], 4.0)
        self.assertEqual(grounded.metadata["grounding_multilingual"], 1.0)
        self.assertTrue(all(not segment.events for segment in grounded.segments))

    def test_ground_text_document_retains_deterministic_structured_pairs(self) -> None:
        text = (
            "status: armed\n"
            "mode=evac\n"
            "priority: high"
        )

        grounded = ground_text_document(text, language="config")

        states = {
            (state.key, state.value)
            for segment in grounded.segments
            for state in segment.states
        }
        self.assertIn(("status", "armed"), states)
        self.assertIn(("mode", "evac"), states)
        self.assertIn(("priority", "high"), states)
        self.assertEqual(grounded.metadata["grounding_structural_layer"], 1.0)
        self.assertGreater(float(grounded.metadata["grounding_document_state_authority"]), 0.0)
        self.assertGreater(float(grounded.metadata["grounding_document_semantic_authority"]), 0.0)
        self.assertGreaterEqual(grounded.metadata["grounding_state_hints"], 3.0)
        self.assertTrue(all(not segment.relations for segment in grounded.segments))
        self.assertTrue(all(not segment.goals for segment in grounded.segments))

    def test_ground_text_document_exposes_typed_perception_and_structural_units_for_config_text(self) -> None:
        text = (
            "[safety]\n"
            "status: armed\n"
            "mode=evac\n"
            "priority: high"
        )

        grounded = ground_text_document(text, language="config")

        self.assertIsNotNone(grounded.routing)
        assert grounded.routing is not None
        self.assertEqual(grounded.routing.modality, "structured_text")
        self.assertEqual(grounded.routing.subtype, "config_text")
        self.assertEqual(grounded.routing.verification_path, "config_schema_verification")
        parser_names = {candidate.parser_name for candidate in grounded.routing.parser_candidates}
        self.assertIn("kv_record_parser", parser_names)
        self.assertIn("ini_section_parser", parser_names)
        self.assertGreaterEqual(float(grounded.metadata.get("grounding_structural_units", 0.0)), 4.0)
        self.assertGreaterEqual(float(grounded.metadata.get("grounding_key_value_units", 0.0)), 3.0)
        self.assertGreaterEqual(float(grounded.metadata.get("grounding_section_header_units", 0.0)), 1.0)
        self.assertEqual(float(grounded.metadata.get("grounding_document_modality_structured_text", 0.0)), 1.0)
        self.assertGreaterEqual(float(grounded.metadata.get("grounding_document_parser_candidates", 0.0)), 2.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_structural_primary_ratio", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)
        unit_types = {unit.unit_type for unit in grounded.structural_units}
        self.assertIn("key_value_record", unit_types)
        self.assertIn("section_header", unit_types)

    def test_ground_text_document_tracks_dialogue_routing_clauses_and_speaker_turns(self) -> None:
        text = (
            "User: Open the hatch now.\n"
            "Assistant: If the alarm is active, open the hatch after inspection."
        )

        grounded = ground_text_document(text, language="text")

        self.assertIsNotNone(grounded.routing)
        assert grounded.routing is not None
        self.assertEqual(grounded.routing.modality, "natural_text")
        self.assertEqual(grounded.routing.subtype, "dialogue_text")
        self.assertEqual(grounded.routing.verification_path, "dialogue_state_verification")
        parser_names = {candidate.parser_name for candidate in grounded.routing.parser_candidates}
        self.assertIn("speaker_turn_parser", parser_names)
        self.assertGreaterEqual(float(grounded.metadata.get("grounding_clause_units", 0.0)), 2.0)
        self.assertGreaterEqual(float(grounded.metadata.get("grounding_speaker_turn_units", 0.0)), 2.0)
        self.assertEqual(float(grounded.metadata.get("grounding_document_modality_natural_text", 0.0)), 1.0)
        self.assertGreaterEqual(float(grounded.metadata.get("grounding_document_routing_confidence", 0.0)), 0.4)
        self.assertGreater(float(grounded.metadata.get("grounding_relation_hints", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_event_hints", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_relation_authority", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_event_authority", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)
        self.assertTrue(all(segment.routing is not None for segment in grounded.segments))
        segment_subtypes = {segment.routing.subtype for segment in grounded.segments if segment.routing is not None}
        self.assertIn("dialogue_text", segment_subtypes)
        unit_types = {unit.unit_type for unit in grounded.structural_units}
        self.assertIn("speaker_turn", unit_types)
        self.assertIn("clause", unit_types)

    def test_ground_text_document_emits_goal_and_relation_hints_for_structural_primary_dialogue(self) -> None:
        text = (
            "User: goal safe_exit.\n"
            "Assistant: stars generate planets."
        )

        grounded = ground_text_document(text, language="text")

        goals = {
            (goal.goal_name, goal.goal_value)
            for segment in grounded.segments
            for goal in segment.goals
        }
        relations = {
            (relation.left, relation.relation, relation.right)
            for segment in grounded.segments
            for relation in segment.relations
        }
        self.assertIn(("goal", "safe_exit"), goals)
        self.assertIn(("stars", "generates", "planets"), relations)
        self.assertGreater(float(grounded.metadata.get("grounding_goal_hints", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_relation_hints", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_goal_authority", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_relation_authority", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)

    def test_ground_text_document_lifts_instruction_sequence_segments_into_structural_primary(self) -> None:
        text = (
            "step 1 open panel\n"
            "step 2 enable backup pump\n"
            "if pressure drops then trigger alarm\n"
            "goal stable cooling"
        )

        grounded = ground_text_document(text, language="text")

        self.assertEqual(grounded.routing.subtype, "instructional_text")
        parser_names = {candidate.parser_name for candidate in grounded.routing.parser_candidates}
        self.assertIn("instruction_step_parser", parser_names)
        segment_subtypes = {segment.routing.subtype for segment in grounded.segments if segment.routing is not None}
        self.assertEqual(segment_subtypes, {"instructional_text"})
        goals = {
            (goal.goal_name, goal.goal_value)
            for segment in grounded.segments
            for goal in segment.goals
        }
        relations = {
            (relation.left, relation.relation, relation.right)
            for segment in grounded.segments
            for relation in segment.relations
        }
        self.assertIn(("goal", "stable_cooling"), goals)
        self.assertTrue(any(relation[1] == "causes" for relation in relations))
        self.assertGreater(float(grounded.metadata.get("grounding_document_structural_primary_ratio", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_goal_authority", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_relation_authority", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_document_semantic_authority", 0.0)), 0.0)

    def test_ground_text_document_emits_utf8_roundtrippable_dual_spans_and_identity(self) -> None:
        text = (
            "Оператор: Відкрий шлюз №2 о 10:05.\n"
            "System: payload={\"стан\":\"готово\",\"канал\":\"північ\"}\n"
            "Примітка: перевірка потрібна перед пуском."
        )

        grounded = ground_text_document(
            text,
            language="uk",
            source_id="source:test:utf8",
            document_id="document:test:utf8",
            episode_id="episode:test:utf8",
        )

        self.assertEqual(grounded.source_id, "source:test:utf8")
        self.assertEqual(grounded.document_id, "document:test:utf8")
        self.assertEqual(grounded.episode_id, "episode:test:utf8")
        self.assertGreater(float(grounded.metadata.get("grounding_span_char_coverage", 0.0)), 0.0)
        self.assertGreater(float(grounded.metadata.get("grounding_span_byte_coverage", 0.0)), 0.0)
        self.assertGreaterEqual(
            float(grounded.metadata.get("grounding_span_segments_with_byte_offsets", 0.0)),
            float(len(grounded.segments)),
        )

        for segment in grounded.segments:
            _assert_span_roundtrip(self, text, segment.span)
            assert segment.span is not None
            self.assertEqual(segment.span.source_id, grounded.source_id)
            self.assertEqual(segment.span.document_id, grounded.document_id)
            self.assertEqual(segment.span.episode_id, grounded.episode_id)
        for unit in grounded.structural_units:
            if unit.span is None:
                continue
            _assert_span_roundtrip(self, text, unit.span)
            self.assertEqual(unit.span.source_id, grounded.source_id)
            self.assertEqual(unit.span.document_id, grounded.document_id)
            self.assertEqual(unit.span.episode_id, grounded.episode_id)

    def test_build_symbolic_trace_bundle_carries_multilingual_grounding_metadata(self) -> None:
        text = (
            'Правило: "Зелена картка" відчиняє "Двері".\n'
            'Факт: У Боба немає зеленої картки.'
        )

        bundle = build_symbolic_trace_bundle(text, lang_hint="uk", max_steps=8, max_counterexamples=2)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.metadata.get("grounding_mode"), "semantic_scene_compiler")
        self.assertEqual(bundle.metadata.get("grounding_multilingual"), 1.0)
        self.assertEqual(float(bundle.metadata.get("grounding_document_semantic_authority", 1.0)), 0.0)
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_segments", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_counterexample_segments", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_entities", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_claims", 0.0)), 1.0)
        self.assertEqual(float(bundle.metadata.get("scene_fallback_backbone_active", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("interlingua_entities", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("interlingua_relations", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("interlingua_graph_records", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("compiled_relation_claims", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("compiled_hypotheses", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("compiled_deferred_hypotheses", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("compiled_mean_confidence", 0.0)), 0.5)
        self.assertGreaterEqual(float(bundle.metadata.get("verification_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("verification_conflicted_hypotheses", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_world_state_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_world_state_contradicted_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_symbolic_facts", 0.0)), 1.0)
        self.assertGreaterEqual(len(bundle.grounding_facts), 1)
        self.assertGreaterEqual(len(bundle.grounding_target_facts), 1)
        self.assertGreaterEqual(len(bundle.grounding_hypotheses), 1)
        self.assertGreaterEqual(len(bundle.grounding_verification_records), 1)
        self.assertGreaterEqual(len(bundle.grounding_world_state_records), 1)
        self.assertGreaterEqual(len(bundle.grounding_graph_records), 2)
        self.assertTrue(any(getattr(fact, "pred", None) == TRACE_TEXT_RELATION_PRED for fact in bundle.observed_facts))

    def test_trace_builder_exposes_grounding_artifacts_and_memory_corroboration(self) -> None:
        text = "weather is rain. rain becomes flood. however flood is not safe."

        bundle, artifacts = build_symbolic_trace_bundle_with_artifacts(
            text,
            lang_hint="text",
            max_steps=8,
            max_counterexamples=2,
        )

        self.assertIsNotNone(bundle)
        self.assertIsNotNone(artifacts)
        assert bundle is not None
        assert artifacts is not None
        self.assertEqual(artifacts.schema_version, "grounding-runtime/v1")
        self.assertIsNotNone(artifacts.source_profile)
        self.assertIsNotNone(artifacts.document_summary)
        self.assertEqual(bundle.grounding_world_state_records, artifacts.grounding_world_state_records)
        self.assertEqual(bundle.grounding_verification_records, artifacts.grounding_verification_records)
        self.assertEqual(bundle.grounding_hypotheses, artifacts.grounding_hypotheses)
        self.assertGreaterEqual(float(artifacts.document_summary.segment_count), 2.0)
        self.assertGreaterEqual(float(artifacts.document_summary.structural_unit_count), 1.0)
        self.assertEqual(artifacts.source_profile.modality, "natural_text")
        self.assertGreaterEqual(float(artifacts.metadata.get("grounding_parser_agreement", 0.0)), 0.0)
        self.assertGreaterEqual(float(artifacts.metadata.get("grounding_span_traceability", 0.0)), 0.0)
        self.assertGreaterEqual(float(artifacts.metadata.get("grounding_byte_span_traceability", 0.0)), 0.0)
        self.assertEqual(float(artifacts.metadata.get("grounding_schema_version_v1", 0.0)), 1.0)
        self.assertGreaterEqual(float(artifacts.metadata.get("grounding_contract_document_segments", 0.0)), 2.0)
        self.assertGreater(float(artifacts.document_summary.char_coverage), 0.0)
        self.assertGreater(float(artifacts.document_summary.byte_coverage), 0.0)
        self.assertEqual(artifacts.document_summary.source_id, "source:primary")
        self.assertEqual(artifacts.document_summary.document_id, "document:primary")
        self.assertEqual(artifacts.document_summary.episode_id, "episode:primary")
        self.assertTrue(all(span.byte_start is not None and span.byte_end is not None for span in artifacts.segment_spans.values()))

        supporting_memory = (
            tuple(artifacts.grounding_world_state_records)
            + tuple(artifacts.grounding_ontology_records)
            + tuple(artifacts.grounding_verification_records)
            + tuple(artifacts.grounding_validation_records)
            + tuple(artifacts.grounding_repair_actions)
            + tuple(artifacts.grounding_hypotheses)
        )
        _bundle_with_memory, artifacts_with_memory = build_symbolic_trace_bundle_with_artifacts(
            text,
            lang_hint="text",
            max_steps=8,
            max_counterexamples=2,
            memory_records=supporting_memory,
        )

        self.assertIsNotNone(artifacts_with_memory)
        assert artifacts_with_memory is not None
        self.assertGreater(float(artifacts_with_memory.metadata.get("verification_memory_corroboration", 0.0)), 0.0)
        self.assertGreater(float(artifacts_with_memory.metadata.get("verifier_memory_corroboration", 0.0)), 0.0)
        self.assertGreater(
            float(artifacts_with_memory.metadata.get("verification_memory_corroborated_records", 0.0)),
            0.0,
        )
        self.assertTrue(
            any(
                record.validator_family == "memory_corroboration"
                for record in artifacts_with_memory.grounding_validation_records
            )
        )

    def test_generation_context_surfaces_trace_grounding_metrics_for_utf8_prompt(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        prompt = _utf8_prompt_row(
            'Мета: безпечний вихід. Зірки генерують Планети.',
            cfg.seq_len,
        )

        ctx = model._build_generation_task_context(prompt)

        self.assertEqual(float(ctx.metadata.get("trace_grounding_multilingual", 0.0)), 1.0)
        self.assertEqual(float(ctx.metadata.get("trace_grounding_relation_hints", 0.0)), 0.0)
        self.assertEqual(float(ctx.metadata.get("trace_grounding_document_semantic_authority", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_grounding_tokens", 0.0)), 2.0)
        self.assertEqual(float(ctx.metadata.get("trace_scene_fallback_backbone_active", 0.0)), 1.0)
        self.assertEqual(float(ctx.metadata.get("trace_scene_backbone_replaceable", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_scene_fallback_goal_proposals", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_scene_fallback_relation_proposals", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_scene_entities", 0.0)), 2.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_interlingua_entities", 0.0)), 2.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_interlingua_relations", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_interlingua_graph_records", 0.0)), 2.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_support_ratio", 0.0)), 0.5)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_uncertainty", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_verification_support", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_repair_pressure", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_world_state_branching_pressure", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_world_state_contradiction_pressure", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_world_state_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_deferred_hypothesis_ratio", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_mean_compiled_confidence", 0.0)), 0.5)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_compiled_relation_claims", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_compiled_hypotheses", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_compiled_mean_confidence", 0.0)), 0.5)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_grounding_world_state_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_grounding_ontology_records", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_grounding_world_state_active_facts", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_verification_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_facts", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_world_state_active_records", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_hypotheses", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_verification_records", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_ontology_records", 0.0)), 0.0)
        self.assertEqual(float(ctx.metadata.get("grounding_schema_version_v1", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("source_ambiguity", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("source_parser_candidates", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_contract_document_segments", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_contract_document_structural_units", 0.0)), 1.0)
        self.assertGreater(float(ctx.metadata.get("grounding_contract_document_char_coverage", 0.0)), 0.0)
        self.assertGreater(float(ctx.metadata.get("grounding_contract_document_byte_coverage", 0.0)), 0.0)
        self.assertEqual(float(ctx.metadata.get("grounding_contract_identity_present", 0.0)), 1.0)
        self.assertIsNotNone(ctx.grounding_artifacts)
        self.assertGreaterEqual(len(ctx.grounding_world_state_active_facts), 1)
        self.assertGreaterEqual(len(ctx.reasoning_facts()), len(ctx.observed_facts))
        self.assertGreaterEqual(len(ctx.grounding_derived_facts), 1)
        labels = {label for label, _ in ctx.source_fact_records()}
        self.assertIn("interlingua", labels)
        self.assertIn("grounding_hypothesis", labels)
        self.assertIn("grounding_verification", labels)
        self.assertTrue(any(label.startswith("grounding_world_state_") for label in labels))

    def test_grounding_memory_records_recall_into_generation_context(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        prompt = _utf8_prompt_row(
            'Мета: безпечний вихід. Зірки генерують Планети. Планети генерують Супутники.',
            cfg.seq_len,
        )
        text = "weather is rain. rain becomes flood. however flood is not safe."
        encoded = [ord(ch) for ch in text.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        prompt = torch.tensor([encoded[:-1]], dtype=torch.long)
        seeded = model._seed_grounding_memory_records(prompt)
        self.assertGreaterEqual(len(seeded), 2)
        self.assertTrue(any("grounding_world_state:" in getattr(record, "graph_key", "") for record in seeded))
        self.assertTrue(any("grounding_verification:" in getattr(record, "graph_key", "") for record in seeded))
        self.assertTrue(any("grounding_hypothesis:" in getattr(record, "graph_key", "") for record in seeded))

        write_stats = model._write_grounding_memory_records(seeded, confidence=1.0)
        self.assertGreaterEqual(write_stats["written"], 1.0)
        self.assertGreaterEqual(write_stats["selected"], 1.0)

        device = next(model.parameters()).device
        query = model.world_graph.encode_records(seeded, device=device)
        recalled = model._recall_grounding_memory_records(query[:1], seeded)
        self.assertGreaterEqual(len(recalled), 1)
        self.assertTrue(
            any(
                prefix in getattr(record, "graph_key", "")
                for record in recalled
                for prefix in ("grounding_world_state:", "grounding_verification:", "grounding_hypothesis:")
            )
        )

        ctx = model._build_generation_task_context(
            prompt,
            memory_grounding_records=recalled,
        )

        self.assertGreaterEqual(len(ctx.memory_grounding_records), 1)
        self.assertGreaterEqual(float(ctx.metadata.get("memory_grounding_records", 0.0)), 1.0)
        self.assertGreater(float(ctx.metadata.get("grounding_memory_corroboration", 0.0)), 0.0)
        self.assertGreater(float(ctx.metadata.get("trace_verification_memory_corroboration", 0.0)), 0.0)
        self.assertGreaterEqual(float(ctx.metadata.get("grounding_graph_records", 0.0)), 1.0)
        labels = {label for label, _ in ctx.source_fact_records()}
        self.assertIn("interlingua", labels)
        self.assertIn("memory_grounding", labels)


if __name__ == "__main__":
    unittest.main()

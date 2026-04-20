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
from omen_symbolic.execution_trace import TRACE_TEXT_RELATION_PRED, build_symbolic_trace_bundle


def _utf8_prompt_row(text: str, seq_len: int) -> torch.Tensor:
    encoded = list(text.encode("utf-8"))[: seq_len]
    if len(encoded) < seq_len:
        encoded = encoded + [0] * (seq_len - len(encoded))
    return torch.tensor([encoded[:-1]], dtype=torch.long)


class GroundingTextSemanticsTest(unittest.TestCase):
    def test_ground_text_document_extracts_ukrainian_quoted_relation_hints(self) -> None:
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
        self.assertIn(("зірки", "generates", "планети"), relations)
        self.assertIn(("планети", "generates", "супутники"), relations)
        self.assertEqual(grounded.metadata["grounding_multilingual"], 1.0)
        self.assertGreaterEqual(grounded.metadata["grounding_relation_hints"], 2.0)

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
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_segments", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("grounding_counterexample_segments", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_entities", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_claims", 0.0)), 1.0)
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
        self.assertGreaterEqual(float(ctx.metadata.get("trace_grounding_goal_hints", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_grounding_relation_hints", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_grounding_tokens", 0.0)), 2.0)
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
        self.assertGreaterEqual(len(ctx.grounding_world_state_active_facts), 1)
        self.assertGreaterEqual(len(ctx.reasoning_facts()), len(ctx.observed_facts))
        self.assertGreaterEqual(len(ctx.grounding_derived_facts), 1)
        labels = {label for label, _ in ctx.source_fact_records()}
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
        labels = {label for label, _ in ctx.source_fact_records()}
        self.assertIn("memory_grounding", labels)


if __name__ == "__main__":
    unittest.main()

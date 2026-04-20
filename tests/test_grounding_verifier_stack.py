from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import ground_text_to_symbolic
from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig
from omen_symbolic.execution_trace import build_symbolic_trace_bundle


def _ascii_src_tgt(text: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [ord(ch) for ch in text.encode("ascii", errors="ignore").decode("ascii")]
    encoded = encoded[:seq_len]
    if len(encoded) < seq_len:
        encoded = encoded + [0] * (seq_len - len(encoded))
    full = torch.tensor([encoded], dtype=torch.long)
    return full[:, :-1], full[:, 1:]


class GroundingVerifierStackTest(unittest.TestCase):
    def test_pipeline_builds_validation_records_and_repair_schedule(self) -> None:
        text = "\n".join(
            [
                "dispatcher opens door",
                "however dispatcher opens door failed",
                "at 10:00 dispatcher opens door because alarm triggered",
            ]
        )

        result = ground_text_to_symbolic(text, language="text", max_segments=8)

        self.assertGreaterEqual(result.verifier_stack.metadata.get("verifier_stack_records", 0.0), 2.0)
        self.assertGreaterEqual(result.verifier_stack.metadata.get("verifier_stack_repair_actions", 0.0), 1.0)
        self.assertGreaterEqual(result.verifier_stack.metadata.get("verifier_world_model_support", 0.0), 0.2)
        self.assertGreaterEqual(result.verifier_stack.metadata.get("verifier_temporal_consistency", 0.0), 0.2)
        self.assertTrue(any(record.validator_family == "world_model" for record in result.verifier_stack.validation_records))
        self.assertTrue(any(record.validator_family == "temporal" for record in result.verifier_stack.validation_records))
        self.assertTrue(
            any(
                action.action_type in {"trigger_hidden_cause_abduction", "promote_world_model_supported_claim"}
                for action in result.verifier_stack.repair_actions
            )
        )

    def test_trace_and_runtime_surface_verifier_stack_metrics(self) -> None:
        text = "\n".join(
            [
                "dispatcher opens door",
                "however dispatcher opens door failed",
                "at 10:00 dispatcher opens door because alarm triggered",
            ]
        )
        bundle = build_symbolic_trace_bundle(text, lang_hint="text", max_steps=8, max_counterexamples=2)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertGreaterEqual(float(bundle.metadata.get("verifier_stack_records", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("verifier_stack_repair_actions", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("verifier_world_model_support", 0.0)), 0.2)
        self.assertGreaterEqual(float(bundle.metadata.get("verifier_temporal_consistency", 0.0)), 0.2)
        self.assertGreaterEqual(len(bundle.grounding_validation_records), 2)
        self.assertGreaterEqual(len(bundle.grounding_repair_actions), 1)

        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        src, tgt = _ascii_src_tgt(text, cfg.seq_len)

        out = model(src, tgt)

        self.assertGreaterEqual(out["sym_grounding_validation_records"], 2.0)
        self.assertGreaterEqual(out["sym_grounding_repair_actions"], 1.0)
        self.assertGreaterEqual(out["sym_verifier_world_model_support"], 0.2)
        self.assertGreaterEqual(out["sym_verifier_temporal_consistency"], 0.2)
        self.assertGreaterEqual(out["sym_trace_validation_records"], 2.0)
        self.assertGreaterEqual(out["sym_trace_repair_actions"], 1.0)
        self.assertGreaterEqual(out["world_graph_grounding_validation_records"], 1.0)
        self.assertGreaterEqual(out["world_graph_grounding_repair_actions"], 1.0)

    def test_memory_corroboration_becomes_validator_family_when_memory_is_available(self) -> None:
        text = "\n".join(
            [
                "dispatcher opens door",
                "at 10:00 dispatcher opens door because alarm triggered",
            ]
        )

        seeded = ground_text_to_symbolic(text, language="text", max_segments=8)
        memory_records = (
            tuple(seeded.world_state.records)
            + tuple(seeded.verification.records)
            + tuple(seeded.compiled.hypotheses)
        )
        result = ground_text_to_symbolic(
            text,
            language="text",
            max_segments=8,
            memory_records=memory_records,
        )

        self.assertGreaterEqual(result.verifier_stack.metadata.get("verifier_stack_memory_records", 0.0), 1.0)
        self.assertGreater(result.verifier_stack.metadata.get("verifier_memory_corroboration", 0.0), 0.0)
        self.assertTrue(
            any(
                record.validator_family == "memory_corroboration"
                and record.validation_status == "supported"
                for record in result.verifier_stack.validation_records
            )
        )


if __name__ == "__main__":
    unittest.main()

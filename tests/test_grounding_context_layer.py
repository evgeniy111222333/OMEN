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


def _ascii_row(text: str, seq_len: int) -> torch.Tensor:
    encoded = [ord(ch) for ch in text.encode("ascii", errors="ignore").decode("ascii")]
    encoded = encoded[:seq_len]
    if len(encoded) < seq_len:
        encoded = encoded + [0] * (seq_len - len(encoded))
    return torch.tensor([encoded[:-1]], dtype=torch.long)


class GroundingContextLayerTest(unittest.TestCase):
    def test_pipeline_builds_mentions_discourse_temporal_and_explanations(self) -> None:
        text = "\n".join(
            [
                "dispatcher opens door",
                "however bob card=none",
                "at 10:00 dispatcher opens door because alarm triggered",
            ]
        )

        result = ground_text_to_symbolic(text, language="text", max_segments=8)

        self.assertGreaterEqual(result.scene.metadata.get("scene_mentions", 0.0), 2.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_discourse_relations", 0.0), 2.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_temporal_markers", 0.0), 1.0)
        self.assertGreaterEqual(result.scene.metadata.get("scene_explanations", 0.0), 1.0)
        self.assertGreaterEqual(len(result.scene.mentions), 2)
        self.assertGreaterEqual(len(result.scene.discourse_relations), 2)
        self.assertGreaterEqual(len(result.scene.temporal_markers), 1)
        self.assertGreaterEqual(len(result.scene.explanations), 1)

    def test_trace_and_runtime_surface_context_layer_metrics(self) -> None:
        text = "\n".join(
            [
                "dispatcher opens door",
                "however bob card=none",
                "at 10:00 dispatcher opens door because alarm triggered",
            ]
        )
        bundle = build_symbolic_trace_bundle(text, lang_hint="text", max_steps=8, max_counterexamples=2)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertGreaterEqual(float(bundle.metadata.get("scene_mentions", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_discourse_relations", 0.0)), 2.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_temporal_markers", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_explanations", 0.0)), 1.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_context_records", 0.0)), 3.0)
        self.assertGreaterEqual(float(bundle.metadata.get("scene_context_facts", 0.0)), 3.0)

        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        ctx = model._build_generation_task_context(_ascii_row(text, cfg.seq_len))

        self.assertGreaterEqual(float(ctx.metadata.get("trace_scene_mentions", 0.0)), 2.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_scene_discourse_relations", 0.0)), 2.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_scene_temporal_markers", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_scene_explanations", 0.0)), 1.0)
        self.assertGreaterEqual(float(ctx.metadata.get("trace_scene_context_records", 0.0)), 3.0)


if __name__ == "__main__":
    unittest.main()

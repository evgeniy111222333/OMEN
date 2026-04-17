from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig
from omen_symbolic.execution_trace import build_symbolic_trace_bundle
from omen_symbolic.world_graph import CanonicalWorldState, WorldGraphEncoder, WorldGraphState


class WorldGraphIntegrationTest(unittest.TestCase):
    def test_world_graph_encoder_extracts_trace_supervision(self) -> None:
        code = "def add(a, b):\n    return a + b\n"
        bundle = build_symbolic_trace_bundle(code)
        self.assertIsNotNone(bundle)
        encoder = WorldGraphEncoder(
            d_latent=32,
            max_nodes=64,
            max_edges=192,
            max_transitions=8,
        )
        graph = encoder(
            facts=list(bundle.observed_facts),
            trace_bundle=bundle,
            device=torch.device("cpu"),
        )
        self.assertGreater(len(graph.node_keys), 0)
        self.assertGreater(len(graph.edges), 0)
        self.assertIsNotNone(graph.transition_states)
        self.assertIsNotNone(graph.transition_targets)
        self.assertEqual(graph.transition_states.shape, graph.transition_targets.shape)
        self.assertGreaterEqual(graph.summary()["trace_steps"], 1.0)

    def test_omen_scale_forward_emits_world_graph_telemetry(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        code = "def add(a, b):\n    return a + b\n"
        encoded = [ord(ch) for ch in code.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        src = full[:, :-1]
        tgt = full[:, 1:]

        out = model(src, tgt)
        self.assertIn("z_graph", out)
        self.assertIn("z_program", out)
        self.assertIn("world_state", out)
        self.assertIn("z_dense", out)
        self.assertIn("z_graph_readout", out)
        self.assertEqual(out["z_graph"].shape, (1, cfg.d_latent))
        self.assertEqual(out["z_program"].shape, (1, cfg.d_latent))
        self.assertEqual(out["canonical_stack"], "omen_scale_world_graph")
        self.assertIsInstance(out["world_state"], CanonicalWorldState)
        self.assertIs(out["z"], out["world_state"])
        self.assertTrue(torch.allclose(out["z_world"], out["z_dense"]))
        self.assertIsInstance(out["z_graph_struct"], WorldGraphState)
        self.assertEqual(out["world_state"].batch_size, 1)
        self.assertIsNotNone(out["world_state"].program_state)
        self.assertGreater(len(out["world_state"].target_facts), 0)
        self.assertGreaterEqual(out["world_graph_nodes"], 1.0)
        self.assertGreaterEqual(out["world_graph_edges"], 1.0)
        self.assertGreaterEqual(out["world_graph_trace_steps"], 1.0)
        self.assertGreaterEqual(out["world_graph_execution_steps"], 1.0)
        self.assertEqual(out["world_graph_hidden_fallback_steps"], 0.0)
        self.assertEqual(out["world_graph_hidden_teacher_applied"], 0.0)
        self.assertGreaterEqual(out["program_target_facts"], 1.0)
        self.assertGreaterEqual(out["program_anchor"], 0.0)
        self.assertGreaterEqual(out["program_decoder_ce"], 0.0)
        self.assertEqual(out["z_graph_primary"], 1.0)
        self.assertGreaterEqual(out["z_graph_anchor"], 0.0)

    def test_world_rollout_can_use_trace_states_as_primary_targets(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        cfg.world_graph_pooled_mix = 0.0
        cfg.world_graph_trace_mix = 1.0
        model = OMENScale(cfg)
        model.eval()

        code = "def add(a, b):\n    total = a + b\n    return total\n"
        encoded = [ord(ch) for ch in code.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        src = full[:, :-1]

        with torch.no_grad():
            h_tok = model.tok_encoder(src)
            world_graph_batch = model._build_world_graph_batch(src, saliency_out=None)
            _traj, world_targets = model._world_rollout_from_hidden(
                h_tok,
                src,
                world_graph_batch=world_graph_batch,
                teacher_forcing_ratio=0.0,
            )

        self.assertTrue(world_graph_batch.graphs)
        graph = world_graph_batch.graphs[0]
        self.assertIsNotNone(graph.transition_targets)
        self.assertIsNotNone(graph.transition_states)
        sample_steps = min(world_targets.size(1), graph.transition_targets.size(0))
        self.assertGreater(sample_steps, 0)
        expected = graph.transition_targets[-sample_steps:].to(
            device=world_targets.device,
            dtype=world_targets.dtype,
        )
        self.assertTrue(torch.allclose(world_targets[0, -sample_steps:], expected, atol=1e-5, rtol=1e-4))

    def test_canonical_world_state_tracks_batch_graphs(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        samples = []
        for code in (
            "def add(a, b):\n    return a + b\n",
            "def sub(a, b):\n    return a - b\n",
        ):
            encoded = [ord(ch) for ch in code.encode("ascii", errors="ignore").decode("ascii")]
            encoded = encoded[: cfg.seq_len]
            if len(encoded) < cfg.seq_len:
                encoded = encoded + [0] * (cfg.seq_len - len(encoded))
            samples.append(encoded)
        full = torch.tensor(samples, dtype=torch.long)
        src = full[:, :-1]
        tgt = full[:, 1:]

        out = model(src, tgt)
        self.assertEqual(out["world_state"].batch_size, 2)
        self.assertEqual(len(out["world_state"].graphs), 2)

    def test_execution_driven_targets_match_trace_states_by_default(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        cfg.world_graph_execution_driven = True
        cfg.world_graph_hidden_mix = 0.25
        model = OMENScale(cfg)
        model.eval()

        code = "def add(a, b):\n    total = a + b\n    return total\n"
        encoded = [ord(ch) for ch in code.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        src = full[:, :-1]

        with torch.no_grad():
            h_tok = model.tok_encoder(src)
            world_graph_batch = model._build_world_graph_batch(src, saliency_out=None)
            _traj, world_targets = model._world_rollout_from_hidden(
                h_tok,
                src,
                world_graph_batch=world_graph_batch,
                teacher_forcing_ratio=0.0,
            )

        graph = world_graph_batch.graphs[0]
        self.assertIsNotNone(graph.transition_targets)
        expected = model._fit_graph_sequence(
            graph.transition_targets.to(device=world_targets.device, dtype=world_targets.dtype),
            world_targets.size(1),
            pad_with_first=True,
        )
        self.assertIsNotNone(expected)
        self.assertTrue(torch.allclose(world_targets[0], expected, atol=1e-5, rtol=1e-4))
        self.assertEqual(world_graph_batch.metadata["execution_supervised_steps"], float(world_targets.size(1)))
        self.assertEqual(world_graph_batch.metadata["hidden_fallback_steps"], 0.0)
        self.assertEqual(world_graph_batch.metadata["hidden_teacher_applied"], 0.0)

    def test_execution_driven_fallback_uses_neutral_world_prior(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        cfg.world_graph_enabled = False
        model = OMENScale(cfg)
        model.eval()

        code = "def add(a, b):\n    return a + b\n"
        encoded = [ord(ch) for ch in code.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        src = full[:, :-1]
        tgt = full[:, 1:]

        out = model(src, tgt)
        self.assertEqual(out["world_graph_hidden_teacher_applied"], 0.0)
        self.assertEqual(out["world_graph_hidden_fallback_steps"], 0.0)
        self.assertEqual(out["world_graph_neutral_prior_applied"], 1.0)
        self.assertGreaterEqual(out["world_graph_neutral_prior_steps"], 1.0)
        self.assertGreaterEqual(out["world_graph_execution_steps"], 1.0)


if __name__ == "__main__":
    unittest.main()

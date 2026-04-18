from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig
from omen_prolog import HornAtom
from omen_symbolic.execution_trace import (
    TRACE_TEXT_GOAL_PRED,
    TRACE_TEXT_STATE_PRED,
    build_symbolic_trace_bundle,
)
from omen_symbolic.world_graph import (
    CanonicalWorldState,
    WorldGraphBatch,
    WorldGraphEdge,
    WorldGraphEncoder,
    WorldGraphState,
)


class WorldGraphIntegrationTest(unittest.TestCase):
    def test_message_pass_is_amp_safe_under_autocast(self) -> None:
        encoder = WorldGraphEncoder(
            d_latent=16,
            max_nodes=8,
            max_edges=16,
            max_transitions=4,
        )
        node_states = torch.randn(4, 16, dtype=torch.float32)
        edges = (
            WorldGraphEdge(src=0, dst=1, relation="shared_term"),
            WorldGraphEdge(src=1, dst=2, relation="same_pred"),
            WorldGraphEdge(src=2, dst=3, relation="cooccurs"),
        )

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = encoder._message_pass(node_states, edges)

        self.assertEqual(out.dtype, node_states.dtype)
        self.assertEqual(out.shape, node_states.shape)

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

    def test_world_graph_runtime_cache_reuses_atom_signature_encoding_within_forward(self) -> None:
        encoder = WorldGraphEncoder(
            d_latent=16,
            max_nodes=16,
            max_edges=32,
            max_transitions=4,
        )
        atom = HornAtom(pred=17, args=(1, 2, 3))

        with mock.patch.object(
            encoder,
            "_encode_text_signature",
            wraps=encoder._encode_text_signature,
        ) as encode_text:
            first = encoder._encode_atom(atom, torch.device("cpu"))
            second = encoder._encode_atom(atom, torch.device("cpu"))
            self.assertTrue(torch.allclose(first, second))
            self.assertEqual(encode_text.call_count, 3)
            encoder.clear_runtime_caches()
            _ = encoder._encode_atom(atom, torch.device("cpu"))
            self.assertEqual(encode_text.call_count, 6)

    def test_world_graph_batch_atom_encoding_matches_scalar_path(self) -> None:
        encoder = WorldGraphEncoder(
            d_latent=16,
            max_nodes=16,
            max_edges=32,
            max_transitions=4,
        )
        atoms = [
            HornAtom(pred=17, args=(1, 2, 3)),
            HornAtom(pred=18, args=(2, 3)),
            HornAtom(pred=19, args=(3,)),
        ]

        encoder.clear_runtime_caches()
        batch_states = encoder._encode_atom_batch(atoms, torch.device("cpu"))
        encoder.clear_runtime_caches()
        scalar_states = torch.stack(
            [encoder._encode_atom(atom, torch.device("cpu")) for atom in atoms],
            dim=0,
        )

        self.assertTrue(torch.allclose(batch_states, scalar_states, atol=1e-6, rtol=1e-5))

    def test_observation_trace_builder_supports_plain_text_sequences(self) -> None:
        text = "weather is rain. rain becomes flood. however flood is not safe."
        bundle = build_symbolic_trace_bundle(text, lang_hint="text", max_steps=8, max_counterexamples=2)
        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle.language, "text")
        self.assertGreaterEqual(len(bundle.transitions), 2)
        self.assertGreater(len(bundle.observed_facts), 0)
        self.assertGreater(len(bundle.target_facts), 0)
        self.assertGreaterEqual(len(bundle.counterexamples), 1)

    def test_observation_trace_builder_supports_structured_state_records(self) -> None:
        text = (
            '{"step":1,"weather":"rain","road":"wet"}\n'
            '{"step":2,"weather":"storm","road":"closed"}\n'
            '{"goal":"evacuation","status":"safe_exit"}'
        )
        bundle = build_symbolic_trace_bundle(text, lang_hint="json", max_steps=8, max_counterexamples=2)
        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertGreaterEqual(len(bundle.transitions), 2)
        self.assertGreater(len(bundle.observed_facts), 0)
        self.assertTrue(
            any(getattr(fact, "pred", None) in (TRACE_TEXT_STATE_PRED, TRACE_TEXT_GOAL_PRED) for fact in bundle.target_facts)
        )

    def test_ast_language_heuristics_cover_javascript_and_observation_text(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        def encode_row(text: str) -> torch.Tensor:
            encoded = [ord(ch) for ch in text.encode("ascii", errors="ignore").decode("ascii")]
            encoded = encoded[: cfg.seq_len]
            if len(encoded) < cfg.seq_len:
                encoded = encoded + [0] * (cfg.seq_len - len(encoded))
            return torch.tensor(encoded[:-1], dtype=torch.long)

        js_row = encode_row("class Node {\n  constructor(v) {\n    this.v = v;\n  }\n}\n")
        obs_row = encode_row("weather is rain. rain becomes flood. however flood is not safe.")
        structured_row = encode_row("step1: weather=rain, road=wet\nstep2: road=closed, alert=yellow\ntarget evacuation=safe_exit")

        self.assertEqual(model._ast_lang_from_bytes(js_row), "javascript")
        self.assertEqual(model._ast_lang_from_bytes(obs_row), "text")
        self.assertEqual(model._ast_lang_from_bytes(structured_row), "text")

    def test_omen_scale_forward_emits_world_graph_telemetry(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
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
        self.assertGreaterEqual(out["world_causal_error"], 0.0)
        self.assertGreaterEqual(out["world_alignment"], 0.0)
        self.assertEqual(out["z_graph_primary"], 1.0)
        self.assertGreaterEqual(out["z_graph_anchor"], 0.0)
        self.assertEqual(out["world_graph_signature_encoder_active"], 1.0)
        self.assertGreater(out["world_graph_context_facts"], 0.0)
        self.assertEqual(out["world_graph_semantic_graph_enriched"], 1.0)
        self.assertEqual(out["z_posterior_graph_native"], 1.0)
        self.assertEqual(out["z_posterior_perceiver_fallback"], 0.0)
        self.assertEqual(out["world_graph_transition_native"], 1.0)
        self.assertEqual(out["world_graph_graph_dense_view_derived"], 1.0)
        self.assertEqual(out["world_graph_neural_residual_used"], 0.0)
        self.assertEqual(out["world_state"].metadata["graph_dense_view_is_derived"], 1.0)
        self.assertEqual(out["world_state"].metadata["signature_encoder_active"], 1.0)
        self.assertEqual(out["world_state"].metadata["z_posterior_graph_native"], 1.0)
        self.assertEqual(out["world_state"].metadata["world_graph_transition_native"], 1.0)

    def test_plain_text_forward_uses_observation_trace_world_path(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        text = "weather is rain. rain becomes flood. however flood is not safe."
        encoded = [ord(ch) for ch in text.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        src = full[:, :-1]
        tgt = full[:, 1:]

        out = model(src, tgt)

        self.assertGreaterEqual(out["world_graph_trace_steps"], 1.0)
        self.assertGreaterEqual(out["world_graph_execution_steps"], 1.0)
        self.assertGreaterEqual(out["sym_trace_steps"], 1.0)
        self.assertGreaterEqual(out["sym_world_context_facts"], 1.0)
        self.assertGreaterEqual(out["world_graph_context_facts"], 1.0)

    def test_structured_text_forward_uses_world_grounded_observation_path(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)

        text = (
            '{"step":1,"tank":"full","valve":"closed"}\n'
            '{"step":2,"valve":"open","flow":"high"}\n'
            '{"goal":"pressure","value":"stable"}'
        )
        encoded = [ord(ch) for ch in text.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        src = full[:, :-1]
        tgt = full[:, 1:]

        out = model(src, tgt)

        self.assertGreaterEqual(out["world_graph_trace_steps"], 1.0)
        self.assertGreaterEqual(out["world_graph_execution_steps"], 1.0)
        self.assertGreaterEqual(out["sym_trace_steps"], 1.0)
        self.assertGreaterEqual(out["sym_world_context_facts"], 1.0)
        self.assertGreaterEqual(out["world_graph_context_facts"], 1.0)
        self.assertEqual(out["sym_ast_lang_other"], 1.0)

    def test_graph_posterior_path_skips_perceiver_when_world_graph_is_available(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
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

        with mock.patch.object(model.perceiver, "forward", side_effect=AssertionError("perceiver should be bypassed")):
            out = model(src, tgt)

        self.assertEqual(out["z_posterior_graph_native"], 1.0)
        self.assertEqual(out["z_posterior_perceiver_fallback"], 0.0)

    def test_graph_grounding_returns_graph_derived_state_without_neural_residual_mix(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        model.eval()

        pooled = torch.full((cfg.d_latent,), 0.75)
        graph = WorldGraphState(
            node_keys=("p0(const:1)",),
            node_types=("observed",),
            fact_records=(("observed", HornAtom(0, (1,))),),
            edges=tuple(),
            node_states=pooled.unsqueeze(0),
            pooled_state=pooled,
            metadata={"signature_encoder_active": 1.0},
        )
        batch = WorldGraphBatch(
            graphs=(graph,),
            pooled_states=pooled.unsqueeze(0),
            metadata={"enabled": 1.0},
        )
        z_a = torch.randn(1, cfg.d_latent)
        z_b = torch.randn(1, cfg.d_latent) * 3.0

        grounded_a, z_graph_a, primary_a = model._ground_world_state(z_a, batch)
        grounded_b, z_graph_b, primary_b = model._ground_world_state(z_b, batch)

        self.assertTrue(torch.allclose(grounded_a, grounded_b, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(z_graph_a, pooled.unsqueeze(0)))
        self.assertTrue(torch.allclose(z_graph_b, pooled.unsqueeze(0)))
        self.assertTrue(torch.allclose(primary_a, torch.ones_like(primary_a)))
        self.assertTrue(torch.allclose(primary_b, torch.ones_like(primary_b)))
        self.assertEqual(batch.metadata["graph_dense_view_is_derived"], 1.0)
        self.assertEqual(batch.metadata["neural_residual_used"], 0.0)

    def test_decoder_graph_readout_does_not_depend_on_dense_query_state(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        model.eval()

        pooled = torch.full((cfg.d_latent,), 0.25)
        graph = WorldGraphState(
            node_keys=("p0(const:1)", "p1(const:2)"),
            node_types=("observed", "context"),
            fact_records=(
                ("observed", HornAtom(0, (1,))),
                ("context", HornAtom(1, (2,))),
            ),
            edges=tuple(),
            node_states=torch.stack([pooled, pooled * 2.0], dim=0),
            pooled_state=pooled,
            metadata={"signature_encoder_active": 1.0},
        )
        batch = WorldGraphBatch(
            graphs=(graph,),
            pooled_states=pooled.unsqueeze(0),
            metadata={"enabled": 1.0},
        )
        z_a = torch.randn(1, cfg.d_latent)
        z_b = torch.randn(1, cfg.d_latent) * 4.0

        state_a, readout_a, anchor_a = model._graph_centered_decoder_state(z_a, batch)
        state_b, readout_b, anchor_b = model._graph_centered_decoder_state(z_b, batch)

        self.assertTrue(torch.allclose(readout_a, readout_b, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(state_a, state_b, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(anchor_a, anchor_b, atol=1e-5, rtol=1e-4))

    def test_semantic_context_facts_expand_decoder_world_graph(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        model.eval()

        code = "def add(a, b):\n    return a + b\n"
        encoded = [ord(ch) for ch in code.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        src = full[:, :-1]

        with torch.no_grad():
            base_batch = model._build_world_graph_batch(src, saliency_out=None)
            enriched_batch = model._build_world_graph_batch(
                src,
                saliency_out=None,
                extra_fact_batches=((HornAtom(999, (1, 2)), HornAtom(1000, (2, 3))),),
            )

        self.assertEqual(base_batch.metadata["mean_context_facts"], 0.0)
        self.assertEqual(enriched_batch.metadata["mean_context_facts"], 2.0)
        self.assertEqual(enriched_batch.metadata["semantic_graph_enriched"], 1.0)
        self.assertGreater(enriched_batch.metadata["mean_nodes"], base_batch.metadata["mean_nodes"])

    def test_world_rollout_can_use_trace_states_as_primary_targets(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
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
        cfg.allow_noncanonical_ablation = True
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
        cfg.allow_noncanonical_ablation = True
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

    def test_execution_driven_fallback_uses_state_anchor_without_neutral_prior(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
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
        self.assertEqual(out["world_graph_neutral_prior_applied"], 0.0)
        self.assertEqual(out["world_graph_neutral_prior_steps"], 0.0)
        self.assertGreaterEqual(out["world_graph_state_anchor_applied"], 1.0)
        self.assertGreaterEqual(out["world_graph_state_anchor_steps"], 1.0)
        self.assertGreaterEqual(out["world_graph_state_anchor_from_hidden"], 1.0)
        self.assertGreaterEqual(out["world_graph_execution_steps"], 1.0)

    def test_non_execution_mode_uses_graph_targets_without_hidden_teacher(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        cfg.world_graph_execution_driven = False
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

        expected = world_graph_batch.pooled_states.to(
            device=world_targets.device,
            dtype=world_targets.dtype,
        ).unsqueeze(1).expand_as(world_targets)
        self.assertTrue(torch.allclose(world_targets, expected, atol=1e-5, rtol=1e-4))
        self.assertEqual(world_graph_batch.metadata["hidden_teacher_applied"], 0.0)
        self.assertEqual(world_graph_batch.metadata["hidden_fallback_steps"], 0.0)
        self.assertEqual(world_graph_batch.metadata["neutral_prior_applied"], 0.0)
        self.assertEqual(world_graph_batch.metadata["state_anchor_applied"], 1.0)
        self.assertEqual(world_graph_batch.metadata["state_anchor_from_graph"], 1.0)

    def test_non_execution_mode_without_world_graph_uses_hidden_state_anchor(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        cfg.world_graph_enabled = False
        cfg.world_graph_execution_driven = False
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
        self.assertEqual(out["world_graph_neutral_prior_applied"], 0.0)
        self.assertEqual(out["world_graph_neutral_prior_steps"], 0.0)
        self.assertGreaterEqual(out["world_graph_state_anchor_applied"], 1.0)
        self.assertGreaterEqual(out["world_graph_state_anchor_from_hidden"], 1.0)


if __name__ == "__main__":
    unittest.main()

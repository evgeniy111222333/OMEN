from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig
from omen_prolog import (
    DifferentiableProver,
    HornAtom,
    SEQ_EDGE_PRED,
    SEQ_LAST_TOKEN_PRED,
    SymbolicTaskContext,
)
from omen_symbolic.execution_trace import (
    TRACE_RETURN_EVENT_PRED,
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

    def test_world_graph_enrich_retains_goal_and_target_under_max_node_budget(self) -> None:
        encoder = WorldGraphEncoder(
            d_latent=16,
            max_nodes=2,
            max_edges=16,
            max_transitions=4,
        )
        observed_a = HornAtom(pred=10, args=(1,))
        observed_b = HornAtom(pred=11, args=(2,))
        target = HornAtom(pred=12, args=(3,))
        goal = HornAtom(pred=13, args=(4,))

        base = encoder(
            facts=(observed_a, observed_b),
            device=torch.device("cpu"),
        )
        enriched = encoder.enrich(
            base,
            context_facts=(),
            extra_records=(("target", target), ("goal", goal)),
            device=torch.device("cpu"),
        )

        self.assertEqual(enriched.node_types, ("target", "goal"))
        self.assertEqual({atom for _, atom in enriched.fact_records}, {target, goal})
        self.assertEqual(enriched.metadata["target_context_facts"], 1.0)
        self.assertEqual(enriched.metadata["goal_context_facts"], 1.0)

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

    def test_ast_language_router_covers_code_and_observation_domains(self) -> None:
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
        ts_row = encode_row("interface User { id: number; name: string }\nconst user: User = { id: 1, name: 'a' };\n")
        bash_row = encode_row("#!/bin/bash\nexport VALUE=1\nif [ \"$1\" = \"go\" ]; then\n  echo ok\nfi\n")
        obs_row = encode_row("weather is rain. rain becomes flood. however flood is not safe.")
        structured_row = encode_row("step1: weather=rain, road=wet\nstep2: road=closed, alert=yellow\ntarget evacuation=safe_exit")

        js_routing = model._source_routing_from_bytes(js_row)
        ts_routing = model._source_routing_from_bytes(ts_row)
        bash_routing = model._source_routing_from_bytes(bash_row)
        obs_routing = model._source_routing_from_bytes(obs_row)
        structured_routing = model._source_routing_from_bytes(structured_row)

        self.assertEqual(model._ast_lang_from_bytes(js_row), "javascript")
        self.assertEqual(model._ast_lang_from_bytes(ts_row), "typescript")
        self.assertEqual(model._ast_lang_from_bytes(bash_row), "bash")
        self.assertEqual(model._ast_lang_from_bytes(obs_row), "text")
        self.assertEqual(model._ast_lang_from_bytes(structured_row), "text")
        self.assertEqual(js_routing.domain, "code")
        self.assertEqual(ts_routing.domain, "code")
        self.assertEqual(bash_routing.domain, "code")
        self.assertEqual(obs_routing.domain, "observation_text")
        self.assertEqual(structured_routing.domain, "structured_observation")
        self.assertGreater(js_routing.confidence, 0.5)
        self.assertGreater(structured_routing.confidence, 0.5)

    def test_row_runtime_cache_keeps_full_tokens_but_strips_decode_padding(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        model = OMENScale(cfg)
        row = torch.tensor([97, 98, 0, 0], dtype=torch.long)

        tokens = model._row_token_values(row)
        decoded = model._decode_source_bytes(row)
        key_a = model._row_fact_cache_key(row)
        key_b = model._row_fact_cache_key(row)

        self.assertEqual(tokens, [97, 98, 0, 0])
        self.assertEqual(decoded, "ab")
        self.assertEqual(key_a, key_b)

    def test_row_fact_cache_key_uses_full_row_bytes(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        model = OMENScale(cfg)
        row_a = torch.tensor([97, 98, 0, 0], dtype=torch.long)
        row_b = torch.tensor([97, 98, 0, 0, 0], dtype=torch.long)

        self.assertNotEqual(model._row_fact_cache_key(row_a), model._row_fact_cache_key(row_b))
        self.assertEqual(model._decode_source_bytes(row_a), "ab")
        self.assertEqual(model._decode_source_bytes(row_b), "ab")

    def test_symbolic_seed_ignores_trailing_padding_tokens(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        row = torch.tensor([[97, 98, 0, 0]], dtype=torch.long)

        seed = model._seed_symbolic_memory_facts(row)

        self.assertIn(HornAtom(SEQ_EDGE_PRED, (97, 98)), seed)
        self.assertIn(HornAtom(SEQ_LAST_TOKEN_PRED, (98, 1)), seed)
        self.assertNotIn(HornAtom(SEQ_EDGE_PRED, (98, 0)), seed)
        self.assertNotIn(HornAtom(SEQ_LAST_TOKEN_PRED, (0, 3)), seed)

    def test_generation_context_ignores_trailing_padding_tokens(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        row = torch.tensor([[97, 98, 0, 0]], dtype=torch.long)

        ctx = model._build_generation_task_context(row)

        self.assertIn(HornAtom(SEQ_EDGE_PRED, (97, 98)), ctx.observed_now_facts)
        self.assertIn(HornAtom(SEQ_LAST_TOKEN_PRED, (98, 1)), ctx.observed_now_facts)
        self.assertNotIn(HornAtom(SEQ_EDGE_PRED, (98, 0)), ctx.observed_now_facts)
        self.assertNotIn(HornAtom(SEQ_LAST_TOKEN_PRED, (0, 3)), ctx.observed_now_facts)

    def test_generation_context_sym_query_uses_last_content_hidden_state(self) -> None:
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

        row = torch.tensor([[97, 98, 0, 0]], dtype=torch.long)
        h_tok = torch.zeros(1, row.size(1), cfg.d_tok, dtype=torch.float32)
        h_tok[0, 1, 0] = 1.0
        h_tok[0, -1, 0] = 9.0
        captured: dict[str, torch.Tensor | None] = {}

        def _capture_query(h_last, *args, **kwargs):
            captured["h_last"] = None if h_last is None else h_last.detach().clone()
            return HornAtom(SEQ_LAST_TOKEN_PRED, (98, 1))

        with mock.patch.object(model.sym_query_gen, "generate_query", side_effect=_capture_query):
            _ = model._build_generation_task_context(row, h_tok=h_tok)

        expected = h_tok[:, 1, :]
        padded_tail = h_tok[:, -1, :]
        self.assertIn("h_last", captured)
        assert captured["h_last"] is not None
        self.assertTrue(torch.allclose(captured["h_last"], expected, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(expected, padded_tail, atol=1e-6, rtol=1e-6))

    def test_sym_query_forward_uses_last_content_hidden_states_per_row(self) -> None:
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

        tokens = torch.tensor([[11, 13, 0, 0], [21, 22, 23, 0]], dtype=torch.long)
        h_tok = torch.zeros(2, tokens.size(1), cfg.d_tok, dtype=torch.float32)
        h_tok[0, 1, 0] = 1.0
        h_tok[0, -1, 0] = 9.0
        h_tok[1, 2, 0] = 2.0
        h_tok[1, -1, 0] = 8.0
        logits = torch.zeros(2, tokens.size(1), cfg.vocab_size, dtype=torch.float32)
        z_sym = torch.zeros(2, cfg.d_latent, dtype=torch.float32)

        class _FakeProver:
            def __init__(self, d_latent: int) -> None:
                self._d_latent = d_latent
                self.task_context = None
                self.last_goal = None

            def answer_query(self, goal, device):
                return (
                    torch.zeros(1, self._d_latent, device=device),
                    (7,),
                    torch.tensor(1.0, device=device),
                )

        fake_prover = _FakeProver(cfg.d_latent)
        captured: dict[str, torch.Tensor | None] = {}
        original_build = model.sym_query_gen._build_query_state
        original_generate = model.sym_query_gen.generate_query

        def _capture_build(h_last, symbolic_state):
            if (
                h_last is not None
                and h_last.dim() == 2
                and h_last.size(0) == tokens.size(0)
                and "batch_h_last" not in captured
            ):
                captured["batch_h_last"] = h_last.detach().clone()
            return original_build(h_last, symbolic_state)

        def _capture_generate(h_last, *args, **kwargs):
            captured["query_h_last"] = None if h_last is None else h_last.detach().clone()
            return original_generate(h_last, *args, **kwargs)

        with mock.patch.object(model.sym_query_gen, "_build_query_state", side_effect=_capture_build):
            with mock.patch.object(model.sym_query_gen, "generate_query", side_effect=_capture_generate):
                _ = model.sym_query_gen(
                    logits=logits,
                    h_tok=h_tok,
                    z_sym=z_sym,
                    prover=fake_prover,
                    tokens=tokens,
                )

        expected_batch = torch.stack((h_tok[0, 1, :], h_tok[1, 2, :]), dim=0)
        padded_tail = h_tok[:, -1, :]
        self.assertIn("batch_h_last", captured)
        self.assertIn("query_h_last", captured)
        assert captured["batch_h_last"] is not None
        assert captured["query_h_last"] is not None
        self.assertTrue(torch.allclose(captured["batch_h_last"], expected_batch, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(expected_batch, padded_tail, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(captured["query_h_last"], expected_batch[:1], atol=1e-6, rtol=1e-6))

    def test_sym_query_forward_does_not_broadcast_first_row_proof_bias_to_other_rows(self) -> None:
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

        tokens = torch.tensor([[11, 13, 0, 0], [21, 22, 23, 0]], dtype=torch.long)
        logits = torch.zeros(2, tokens.size(1), cfg.vocab_size, dtype=torch.float32)
        h_tok = torch.zeros(2, tokens.size(1), cfg.d_tok, dtype=torch.float32)
        z_sym = torch.zeros(2, cfg.d_latent, dtype=torch.float32)

        class _FakeProver:
            def __init__(self, d_latent: int) -> None:
                self._d_latent = d_latent
                self.task_context = SimpleNamespace(
                    metadata={"last_src": 13.0, "last_tgt": 7.0},
                    observed_facts=frozenset(),
                    goal=None,
                )
                self.last_goal = None

            def answer_query(self, goal, device):
                return (
                    torch.zeros(1, self._d_latent, device=device),
                    (7,),
                    torch.tensor(1.0, device=device),
                )

            def ground(self, facts, device):
                del facts
                return torch.zeros(1, self._d_latent, device=device)

        fake_prover = _FakeProver(cfg.d_latent)
        with torch.no_grad():
            for module in (model.sym_query_gen.logit_bias_proj, model.sym_query_gen.query_bias_proj):
                for param in module.parameters():
                    param.zero_()
            for param in model.sym_query_gen.proof_gate.parameters():
                param.zero_()
            out = model.sym_query_gen(
                logits=logits,
                h_tok=h_tok,
                z_sym=z_sym,
                prover=fake_prover,
                tokens=tokens,
            )

        self.assertGreater(float(out[0, -1, 7].item()), 0.0)
        self.assertAlmostEqual(float(out[1, -1, 7].item()), 0.0, places=6)

    def test_source_fact_records_preserve_distinct_atoms_under_hash_collision(self) -> None:
        a = HornAtom(11, (1, 2))
        b = HornAtom(12, (3, 4))
        goal = HornAtom(13, (5, 6))
        target = HornAtom(14, (7, 8))
        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset({a}),
            memory_derived_facts=frozenset({b}),
            goal=goal,
            target_facts=frozenset({target}),
        )

        with mock.patch.object(HornAtom, "__hash__", return_value=1):
            records = ctx.source_fact_records(include_goal=True, include_targets=True)

        self.assertEqual(len(records), 4)
        self.assertEqual({atom for _, atom in records}, {a, b, goal, target})

    def test_world_context_slice_preserves_distinct_atoms_under_hash_collision(self) -> None:
        a = HornAtom(21, (1, 2))
        b = HornAtom(22, (3, 4))
        graph = SimpleNamespace(
            fact_records=(
                ("goal", a),
                ("memory", b),
            )
        )

        with mock.patch.object(HornAtom, "__hash__", return_value=1):
            selected = OMENScale._world_context_slice_from_graph(graph, limit=4)

        self.assertEqual(selected, [a, b])

    def test_semantic_world_fact_batches_preserve_distinct_atoms_under_hash_collision(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        a = HornAtom(31, (1, 2))
        b = HornAtom(32, (3, 4))
        task_context = mock.Mock()
        task_context.source_fact_records.return_value = (
            ("observed_now", a),
            ("memory", b),
        )

        with mock.patch.object(HornAtom, "__hash__", return_value=1):
            fact_batch, record_batch, count = model._semantic_world_fact_batches(1, task_context)

        self.assertEqual(count, 2)
        self.assertEqual(record_batch[0], (("observed_now", a), ("memory", b)))
        self.assertEqual(fact_batch[0], (a, b))

    def test_trace_support_facts_excludes_only_exact_head_under_hash_collision(self) -> None:
        prover = DifferentiableProver(
            d_latent=8,
            sym_vocab=8,
            max_rules=16,
            max_depth=2,
            n_cands=4,
        )
        head = HornAtom(41, (1, 2))
        kept = HornAtom(TRACE_RETURN_EVENT_PRED, (3, 4))

        with mock.patch.object(HornAtom, "__hash__", return_value=1):
            support = prover._trace_support_facts(
                frozenset(),
                frozenset({head, kept}),
                head=head,
            )

        self.assertEqual({repr(atom) for atom in support}, {repr(kept)})

    def test_contextual_abduction_candidates_keep_distinct_target_heads_under_hash_collision(self) -> None:
        prover = DifferentiableProver(
            d_latent=8,
            sym_vocab=8,
            max_rules=16,
            max_depth=2,
            n_cands=4,
        )
        goal = HornAtom(51, (1, 2))
        target = HornAtom(52, (3, 4))
        prover.task_context = SymbolicTaskContext(
            goal=goal,
            target_facts=frozenset({goal, target}),
        )
        prover.current_working_facts = lambda: frozenset({HornAtom(99, (1, 3))})
        seen_heads = []

        def capture_heads(example_head, facts, term_const_values, max_body_atoms):
            del facts, term_const_values, max_body_atoms
            seen_heads.append(example_head)
            return []

        with mock.patch.object(HornAtom, "__hash__", return_value=1), \
             mock.patch("omen_prolog.rank_goal_directed_bodies", side_effect=capture_heads):
            candidates = prover._contextual_abduction_candidates()

        self.assertEqual(candidates, [])
        self.assertEqual(seen_heads, [goal, target])

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
        self.assertEqual(out["sym_source_domain_code"], 1.0)
        self.assertGreater(out["sym_source_confidence"], 0.0)
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

    def test_forward_preserves_row_specific_program_and_world_context_for_mismatched_batch_rows(self) -> None:
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

        rows = [
            [ord("a"), ord("b"), 0, 0],
            [ord("x"), ord("y"), ord("z"), 0],
        ]
        full = torch.tensor(rows, dtype=torch.long)
        src = full[:, :-1]
        tgt = full[:, 1:]

        out = model(src, tgt)

        symbolic_state = out["world_state"].symbolic_state
        program_state = out["z_program"]
        graph_targets = [
            {atom for node_type, atom in graph.fact_records if node_type == "target"}
            for graph in out["world_state"].graphs
        ]
        self.assertGreater(float(symbolic_state[0].norm().item()), 0.0)
        self.assertGreater(float(symbolic_state[1].norm().item()), 0.0)
        self.assertFalse(torch.allclose(symbolic_state[0], symbolic_state[1]))
        self.assertGreater(float(program_state[0].norm().item()), 0.0)
        self.assertGreater(float(program_state[1].norm().item()), 0.0)
        self.assertEqual(len(graph_targets), 2)
        self.assertTrue(graph_targets[0])
        self.assertTrue(graph_targets[1])
        self.assertNotEqual(graph_targets[0], graph_targets[1])

    def test_last_content_helpers_and_counterfactual_actions_ignore_padded_tail(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        cfg.n_counterfactual = 4
        model = OMENScale(cfg)
        src = torch.tensor(
            [
                [1, 2, 0, 0],
                [3, 0, 4, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        tgt = torch.tensor(
            [
                [5, 0, 7, 0],
                [9, 8, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        )

        last_tokens, valid_rows, last_idx = model._batch_last_content_tokens(tgt)
        cf_actions = model._build_counterfactual_actions(src, tgt)

        self.assertEqual(last_tokens.tolist(), [7, 8, 0])
        self.assertEqual(valid_rows.tolist(), [True, True, False])
        self.assertEqual(last_idx.tolist(), [2, 1, 0])
        self.assertEqual(
            cf_actions.tolist(),
            [
                [7, 2, 1, 0],
                [8, 4, 0, 9],
                [0, 0, 0, 0],
            ],
        )

    def test_decoder_surprise_signal_uses_last_content_targets(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        h_tok = torch.randn(2, 4, cfg.d_tok)
        z_enriched = torch.randn(2, cfg.d_latent)
        tgt = torch.tensor(
            [
                [5, 0, 7, 0],
                [9, 8, 0, 0],
            ],
            dtype=torch.long,
        )

        signal = model._decoder_surprise_signal(h_tok, z_enriched, tgt)

        self.assertIsNotNone(signal)
        self.assertEqual(signal["targets"].tolist(), [7, 8])

    def test_program_decoder_loss_uses_per_row_last_content_tokens(self) -> None:
        cfg = OMENScaleConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.continuous_cycle_enabled = False
        cfg.creative_cycle_enabled = False
        model = OMENScale(cfg)
        src = torch.tensor(
            [
                [1, 2, 0, 0],
                [3, 0, 4, 0],
            ],
            dtype=torch.long,
        )
        tgt = torch.tensor(
            [
                [5, 0, 7, 0],
                [9, 8, 0, 0],
            ],
            dtype=torch.long,
        )
        z_symbolic = torch.randn(2, cfg.d_latent)
        task_context = SymbolicTaskContext(
            observed_facts=frozenset(),
            goal=None,
            target_facts=frozenset(),
            provenance="test",
            metadata={"last_src": 2.0, "last_tgt": 7.0},
        )
        captured: dict[str, object] = {}

        def fake_logits(z_symbolic_arg, task_context_arg, context_anchors=None):
            del z_symbolic_arg, task_context_arg
            captured["anchors"] = None if context_anchors is None else context_anchors.detach().cpu().tolist()
            logits = torch.full((2, cfg.vocab_size), -20.0)
            logits[0, 7] = 20.0
            logits[1, 8] = 20.0
            return logits

        with mock.patch.object(model, "_symbolic_token_logits", side_effect=fake_logits):
            loss = model._program_decoder_loss(z_symbolic, task_context, src, tgt)

        self.assertEqual(captured["anchors"], [2, 4])
        self.assertLess(loss.item(), 1e-3)

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

    def test_world_graph_enrichment_preserves_observed_now_for_duplicate_atoms(self) -> None:
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
        tgt = full[:, 1:]
        gap_stub = torch.zeros(1)
        hot_stub = torch.zeros(1, cfg.d_latent, dtype=torch.bool)

        with torch.no_grad():
            base_batch = model._build_world_graph_batch(src, saliency_out=None)
            task_context = model._build_symbolic_task_context(
                src,
                tgt,
                gap_stub,
                hot_stub,
                saliency_out=None,
            )
            enriched_batch = model._enrich_world_graph_batch(
                src,
                saliency_out=None,
                task_context=task_context,
                base_batch=base_batch,
            )

        self.assertGreater(len(task_context.observed_now_facts), 0)
        self.assertGreater(enriched_batch.metadata["observed_now_facts"], 0.0)
        self.assertIn("observed_now", enriched_batch.graphs[0].node_types)

    def test_symbolic_task_context_sym_query_uses_last_content_hidden_state(self) -> None:
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

        src = torch.tensor([[97, 98, 0, 0]], dtype=torch.long)
        tgt = torch.tensor([[98, 99, 0, 0]], dtype=torch.long)
        h_tok = torch.zeros(1, src.size(1), cfg.d_tok, dtype=torch.float32)
        h_tok[0, 1, 0] = 2.0
        h_tok[0, -1, 0] = 7.0
        gap_stub = torch.zeros(1)
        hot_stub = torch.zeros(1, cfg.d_latent, dtype=torch.bool)
        captured: dict[str, torch.Tensor | None] = {}

        def _capture_query(h_last, *args, **kwargs):
            captured["h_last"] = None if h_last is None else h_last.detach().clone()
            return HornAtom(SEQ_LAST_TOKEN_PRED, (98, 1))

        with mock.patch.object(model.sym_query_gen, "generate_query", side_effect=_capture_query):
            _ = model._build_symbolic_task_context(
                src,
                tgt,
                gap_stub,
                hot_stub,
                saliency_out=None,
                h_tok=h_tok,
            )

        expected = h_tok[:, 1, :]
        padded_tail = h_tok[:, -1, :]
        self.assertIn("h_last", captured)
        assert captured["h_last"] is not None
        self.assertTrue(torch.allclose(captured["h_last"], expected, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(expected, padded_tail, atol=1e-6, rtol=1e-6))

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

    def test_state_anchored_world_targets_use_last_content_hidden_state(self) -> None:
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

        src = torch.tensor([[7, 2, 0, 0], [8, 4, 9, 0]], dtype=torch.long)

        with torch.no_grad():
            h_tok = model.tok_encoder(src)
            teacher_states, world_targets, stats = model._state_anchored_world_targets(
                h_tok,
                src,
                2,
                world_graph_batch=None,
            )
            _last_tokens, valid_rows, last_idx = model._batch_last_content_tokens(src)
            batch_idx = torch.arange(src.size(0))
            expected_anchor = model.world_target_proj(h_tok[batch_idx, last_idx]).detach().to(
                device=teacher_states.device,
                dtype=teacher_states.dtype,
            )
            padded_tail_anchor = model.world_target_proj(h_tok[:, -1]).detach().to(
                device=teacher_states.device,
                dtype=teacher_states.dtype,
            )

        expected = expected_anchor.unsqueeze(1).expand_as(teacher_states)
        self.assertTrue(valid_rows.all())
        self.assertTrue(torch.allclose(teacher_states, expected, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(world_targets, expected, atol=1e-5, rtol=1e-4))
        self.assertFalse(torch.allclose(expected_anchor, padded_tail_anchor, atol=1e-5, rtol=1e-4))
        self.assertEqual(stats["state_anchor_from_hidden"], 1.0)

    def test_hidden_world_rollout_uses_trailing_content_actions_instead_of_pad_tail(self) -> None:
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
        cfg.world_rollout_steps = 3
        model = OMENScale(cfg)
        model.eval()

        src = torch.tensor([[7, 2, 0, 0], [8, 4, 9, 0]], dtype=torch.long)
        captured: dict[str, torch.Tensor] = {}

        def _capture_rollout(actions: torch.Tensor, **kwargs) -> torch.Tensor:
            captured["actions"] = actions.detach().cpu()
            z0 = kwargs["z0"]
            return z0.unsqueeze(1).expand(-1, actions.size(1), -1).clone()

        with torch.no_grad():
            h_tok = model.tok_encoder(src)
            with mock.patch.object(model.world_rnn, "simulate_graph_sequence", side_effect=_capture_rollout):
                model._world_rollout_from_hidden(
                    h_tok,
                    src,
                    world_graph_batch=None,
                    teacher_forcing_ratio=0.0,
                )

        self.assertIn("actions", captured)
        self.assertEqual(captured["actions"].tolist(), [[7, 7, 2], [8, 4, 9]])

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

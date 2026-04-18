from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_prolog import HornAtom
from omen_scale import (
    NET_CONTEXT_PRED,
    NET_MEANS_PRED,
    NET_TOKEN_PRED,
    OMENScale,
    SEQ_EDGE_PRED,
    SEQ_PREDICT_NEXT_PRED,
    SymbolicQueryGenerator,
)
from omen_scale_config import OMENScaleConfig


def _net_symbolic_config() -> OMENScaleConfig:
    cfg = OMENScaleConfig.demo()
    cfg.allow_noncanonical_ablation = True
    cfg.vocab_size = 256
    cfg.d_tok = 48
    cfg.n_heads_tok = 4
    cfg.n_layers_tok = 1
    cfg.seq_len = 24
    cfg.d_latent = 32
    cfg.n_latents = 8
    cfg.n_heads_lat = 2
    cfg.n_layers_lat = 1
    cfg.world_rnn_hidden = 48
    cfg.world_rollout_steps = 2
    cfg.mem_heads = 2
    cfg.mem_cache_size = 16
    cfg.mem_symbolic_cache_size = 16
    cfg.sym_vocab = 32
    cfg.sym_embed_dim = 16
    cfg.max_proof_depth = 2
    cfg.n_proof_cands = 4
    cfg.ltm_max_rules = 32
    cfg.sym_max_facts = 16
    cfg.net_enabled = True
    cfg.osf_enabled = False
    cfg.emc_enabled = False
    cfg.saliency_enabled = False
    cfg.creative_cycle_enabled = False
    cfg.continuous_cycle_enabled = False
    cfg.sym_query_gen_enabled = False
    cfg.sym_decoder_surprise_enabled = False
    cfg.world_graph_enabled = False
    cfg.eval_world_self_update_enabled = False
    cfg.ce_reinforce_enabled = False
    return cfg


class NetSymbolicProtocolTest(unittest.TestCase):
    @staticmethod
    def _vq_indices(batch: int, steps: int) -> torch.Tensor:
        base = torch.tensor([41, 42, 43, 42, 44, 45, 46, 47], dtype=torch.long)
        repeats = (steps + base.numel() - 1) // base.numel()
        row = base.repeat(repeats)[:steps]
        return row.unsqueeze(0).repeat(batch, 1)

    def test_forward_routes_net_concepts_into_memory_hints_and_symbolic_context(self) -> None:
        cfg = _net_symbolic_config()
        model = OMENScale(cfg)
        model.eval()
        src = torch.tensor([[17, 19, 23, 29, 31, 37]], dtype=torch.long)
        tgt = torch.tensor([[19, 23, 29, 31, 37, 41]], dtype=torch.long)

        recall_kwargs = []
        symbolic_contexts = []
        orig_context = model._build_symbolic_task_context

        def fake_encode(x: torch.Tensor, return_attn: bool = False):
            del return_attn
            batch, steps = x.shape
            h_tok = torch.zeros(batch, steps, cfg.d_tok, dtype=torch.float32)
            return h_tok, self._vq_indices(batch, steps), {}

        def fake_decode(tgt_tokens: torch.Tensor, z_final: torch.Tensor, h_tok: torch.Tensor):
            del z_final, h_tok
            logits = torch.full(
                (tgt_tokens.size(0), tgt_tokens.size(1), cfg.vocab_size),
                -1e4,
                device=tgt_tokens.device,
            )
            logits[:, :, 7] = 1e4
            return logits, torch.tensor(0.0, device=tgt_tokens.device)

        def fake_net_loss(*args, **kwargs):
            del args, kwargs
            zero = torch.tensor(0.0, device=next(model.parameters()).device)
            return {
                "net_total": zero,
                "net_aux_tensor": zero,
                "net_aux": 0.0,
                "net_vocab_pen": 0.0,
            }

        def fake_prover_forward(z: torch.Tensor, *args, **kwargs):
            del args, kwargs
            return torch.zeros_like(z), torch.tensor(0.0, device=z.device)

        def capture_context(*args, **kwargs):
            context = orig_context(*args, **kwargs)
            symbolic_contexts.append(context)
            return context

        def capture_recall(*args, **kwargs):
            del args
            recall_kwargs.append(dict(kwargs))
            return []

        with mock.patch.object(model.net, "encode", side_effect=fake_encode), \
             mock.patch.object(model.net, "decode", side_effect=fake_decode), \
             mock.patch.object(model.net, "compute_loss", side_effect=fake_net_loss), \
             mock.patch.object(model.prover, "forward", side_effect=fake_prover_forward), \
             mock.patch.object(model.memory, "recall_symbolic_atoms", side_effect=capture_recall), \
             mock.patch.object(model, "_build_symbolic_task_context", side_effect=capture_context), \
             mock.patch.object(model, "_ast_facts_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_rules_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_trace_from_bytes", return_value=None), \
             mock.patch.object(model, "_ast_lang_from_bytes", return_value=""):
            out = model(src, tgt)

        self.assertTrue(symbolic_contexts)
        context = symbolic_contexts[-1]
        self.assertEqual(context.provenance, "net")
        self.assertEqual(context.goal.pred, NET_MEANS_PRED)
        self.assertIn(HornAtom(pred=NET_TOKEN_PRED, args=(41,)), context.observed_facts)
        self.assertIn(HornAtom(pred=NET_CONTEXT_PRED, args=(41, 42)), context.observed_facts)
        self.assertIn(HornAtom(pred=NET_TOKEN_PRED, args=(41,)), context.net_derived_facts)
        self.assertIn(HornAtom(pred=NET_CONTEXT_PRED, args=(41, 42)), context.net_derived_facts)
        self.assertTrue(recall_kwargs)
        self.assertIn(NET_TOKEN_PRED, recall_kwargs[-1].get("predicate_hints", []))
        self.assertIn(NET_CONTEXT_PRED, recall_kwargs[-1].get("predicate_hints", []))
        self.assertIn(41, recall_kwargs[-1].get("anchor_values", []))
        self.assertIn(42, recall_kwargs[-1].get("anchor_values", []))
        self.assertEqual(float(out["net_symbolic_active"]), 1.0)
        self.assertEqual(float(out["net_symbolic_unique_concepts"]), 5.0)
        self.assertGreater(float(out["net_symbolic_context_edges"]), 0.0)
        self.assertEqual(out["sym_provenance"], "net")

    def test_query_predicate_mask_preserves_only_allowed_candidates(self) -> None:
        generator = SymbolicQueryGenerator(
            d_tok=16,
            d_latent=12,
            sym_vocab=32,
            vocab_size=64,
        )
        pred_logits = torch.full((1, len(generator.pred_candidates)), -3.0, dtype=torch.float32)
        pred_logits[0, generator.pred_to_index[SEQ_PREDICT_NEXT_PRED]] = 2.0
        pred_logits[0, generator.pred_to_index[SEQ_EDGE_PRED]] = 4.0
        pred_logits[0, generator.pred_to_index[5]] = 7.0
        pred_logits[0, generator.pred_to_index[7]] = 9.0

        masked = generator._mask_candidate_preds(
            pred_logits,
            candidate_preds=(SEQ_PREDICT_NEXT_PRED, SEQ_EDGE_PRED),
        )

        self.assertEqual(float(masked[0, generator.pred_to_index[SEQ_PREDICT_NEXT_PRED]].item()), 2.0)
        self.assertEqual(float(masked[0, generator.pred_to_index[SEQ_EDGE_PRED]].item()), 4.0)
        self.assertEqual(float(masked[0, generator.pred_to_index[5]].item()), -1e4)
        self.assertEqual(float(masked[0, generator.pred_to_index[7]].item()), -1e4)

    def test_generation_routes_net_concepts_into_memory_hints_and_symbolic_context(self) -> None:
        cfg = _net_symbolic_config()
        model = OMENScale(cfg)
        model.eval()
        prompt = torch.tensor([[17, 19, 23, 29, 31, 37]], dtype=torch.long)

        recall_kwargs = []
        generation_contexts = []
        orig_context = model._build_generation_task_context

        def fake_encode(x: torch.Tensor, return_attn: bool = False):
            del return_attn
            batch, steps = x.shape
            h_tok = torch.zeros(batch, steps, cfg.d_tok, dtype=torch.float32)
            return h_tok, self._vq_indices(batch, steps), {}

        def fake_decode(tgt_tokens: torch.Tensor, z_final: torch.Tensor, h_tok: torch.Tensor):
            del z_final, h_tok
            logits = torch.full(
                (tgt_tokens.size(0), tgt_tokens.size(1), cfg.vocab_size),
                -1e4,
                device=tgt_tokens.device,
            )
            logits[:, :, 7] = 1e4
            return logits, torch.tensor(0.0, device=tgt_tokens.device)

        def fake_prover_forward(z: torch.Tensor, *args, **kwargs):
            del args, kwargs
            return torch.zeros_like(z), torch.tensor(0.0, device=z.device)

        def capture_context(*args, **kwargs):
            context = orig_context(*args, **kwargs)
            generation_contexts.append(context)
            return context

        def capture_recall(*args, **kwargs):
            del args
            recall_kwargs.append(dict(kwargs))
            return []

        with mock.patch.object(model.net, "encode", side_effect=fake_encode), \
             mock.patch.object(model.net, "decode", side_effect=fake_decode), \
             mock.patch.object(model.prover, "forward", side_effect=fake_prover_forward), \
             mock.patch.object(model.memory, "recall_symbolic_atoms", side_effect=capture_recall), \
             mock.patch.object(model, "_build_generation_task_context", side_effect=capture_context), \
             mock.patch.object(model, "_ast_facts_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_rules_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_trace_from_bytes", return_value=None), \
             mock.patch.object(model, "_ast_lang_from_bytes", return_value=""):
            generated = model.generate(prompt, max_new=1, temperature=1.0, dynamic_reasoning=False)

        self.assertEqual(int(generated[0, -1].item()), 7)
        self.assertGreaterEqual(len(generation_contexts), 2)
        self.assertTrue(all(ctx.provenance == "net" for ctx in generation_contexts))
        self.assertTrue(all(ctx.goal.pred == NET_MEANS_PRED for ctx in generation_contexts[:2]))
        self.assertTrue(
            all(HornAtom(pred=NET_CONTEXT_PRED, args=(41, 42)) in ctx.observed_facts for ctx in generation_contexts[:2])
        )
        self.assertTrue(
            all(HornAtom(pred=NET_CONTEXT_PRED, args=(41, 42)) in ctx.net_derived_facts for ctx in generation_contexts[:2])
        )
        self.assertGreaterEqual(len(recall_kwargs), 2)
        self.assertTrue(all(NET_TOKEN_PRED in call.get("predicate_hints", []) for call in recall_kwargs[:2]))
        self.assertTrue(all(NET_CONTEXT_PRED in call.get("predicate_hints", []) for call in recall_kwargs[:2]))
        self.assertTrue(all(41 in call.get("anchor_values", []) for call in recall_kwargs[:2]))
        self.assertEqual(float(model.last_generate_info.get("net_symbolic_active", 0.0)), 1.0)
        self.assertEqual(float(model.last_generate_info.get("net_symbolic_steps", 0.0)), 2.0)
        self.assertEqual(float(model.last_generate_info.get("net_symbolic_unique_concepts", 0.0)), 5.0)
        self.assertGreater(float(model.last_generate_info.get("net_symbolic_facts", 0.0)), 0.0)
        self.assertGreater(float(model.last_generate_info.get("net_symbolic_context_edges", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()

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

from omen_prolog import HornAtom
from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig


def _saliency_generation_config() -> OMENScaleConfig:
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
    cfg.net_enabled = False
    cfg.osf_enabled = False
    cfg.emc_enabled = False
    cfg.saliency_enabled = True
    cfg.creative_cycle_enabled = False
    cfg.continuous_cycle_enabled = False
    cfg.sym_query_gen_enabled = False
    cfg.sym_decoder_surprise_enabled = False
    cfg.world_graph_enabled = True
    cfg.eval_world_self_update_enabled = False
    cfg.ce_reinforce_enabled = False
    return cfg


class GenerationSaliencyProtocolTest(unittest.TestCase):
    @staticmethod
    def _encode_prompt(cfg: OMENScaleConfig, text: str) -> torch.Tensor:
        encoded = [ord(ch) for ch in text.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        return torch.tensor([encoded], dtype=torch.long)

    def test_generation_routes_saliency_into_world_graph_and_symbolic_context(self) -> None:
        cfg = _saliency_generation_config()
        model = OMENScale(cfg)
        model.eval()
        prompt = self._encode_prompt(cfg, "hello saliency world")

        fake_saliency = SimpleNamespace(
            sal_semantic_facts=[
                [
                    HornAtom(pred=601, args=(1,)),
                    HornAtom(pred=602, args=(2, 3)),
                ]
            ],
            sal_expected_facts=[
                [
                    HornAtom(pred=777, args=(9, 10)),
                    HornAtom(pred=778, args=(11, 12)),
                ]
            ],
            sal_edges=4.0,
            sal_consistency=0.75,
        )

        world_graph_saliency = []
        generation_contexts = []
        recall_kwargs = []
        orig_world_graph = model._build_world_graph_batch
        orig_generation_context = model._build_generation_task_context

        def capture_world_graph(*args, **kwargs):
            world_graph_saliency.append(kwargs.get("saliency_out"))
            return orig_world_graph(*args, **kwargs)

        def capture_generation_context(*args, **kwargs):
            context = orig_generation_context(*args, **kwargs)
            generation_contexts.append(context)
            return context

        def capture_recall(*args, **kwargs):
            del args
            recall_kwargs.append(dict(kwargs))
            return []

        def fake_prover_forward(z: torch.Tensor, *args, **kwargs):
            del args, kwargs
            return torch.zeros_like(z), torch.tensor(0.0, device=z.device)

        def fake_tok_decoder(ctx: torch.Tensor, z_final: torch.Tensor):
            del z_final
            logits = torch.full(
                (ctx.size(0), ctx.size(1), cfg.vocab_size),
                -1e4,
                device=ctx.device,
            )
            logits[:, :, 7] = 1e4
            return logits

        with mock.patch.object(model.saliency, "forward", return_value=fake_saliency) as saliency_mock, \
             mock.patch.object(model, "_build_world_graph_batch", side_effect=capture_world_graph), \
             mock.patch.object(model, "_build_generation_task_context", side_effect=capture_generation_context), \
             mock.patch.object(model.memory, "recall_symbolic_atoms", side_effect=capture_recall), \
             mock.patch.object(model.prover, "forward", side_effect=fake_prover_forward), \
             mock.patch.object(model.tok_decoder, "forward", side_effect=fake_tok_decoder):
            generated = model.generate(prompt, max_new=1, temperature=1.0, dynamic_reasoning=False)

        self.assertEqual(int(generated[0, -1].item()), 7)
        self.assertGreaterEqual(saliency_mock.call_count, 2)
        self.assertGreaterEqual(len(world_graph_saliency), 2)
        self.assertTrue(all(item is fake_saliency for item in world_graph_saliency[:2]))
        self.assertGreaterEqual(len(recall_kwargs), 2)
        self.assertTrue(any(777 in call.get("predicate_hints", []) for call in recall_kwargs))
        self.assertTrue(any(9 in call.get("anchor_values", []) for call in recall_kwargs))
        self.assertGreaterEqual(len(generation_contexts), 2)
        self.assertTrue(any(ctx.provenance == "saliency" for ctx in generation_contexts))
        self.assertTrue(any(ctx.goal.pred == 777 for ctx in generation_contexts))
        self.assertTrue(all(ctx.metadata.get("saliency_consistency") == 0.75 for ctx in generation_contexts))
        self.assertTrue(any(HornAtom(pred=601, args=(1,)) in ctx.saliency_derived_facts for ctx in generation_contexts))
        self.assertTrue(any(HornAtom(pred=778, args=(11, 12)) in ctx.saliency_derived_facts for ctx in generation_contexts))
        self.assertEqual(float(model.last_generate_info.get("saliency_active", 0.0)), 1.0)
        self.assertEqual(float(model.last_generate_info.get("saliency_steps", 0.0)), 2.0)
        self.assertEqual(float(model.last_generate_info.get("saliency_semantic_facts", 0.0)), 4.0)
        self.assertEqual(float(model.last_generate_info.get("saliency_expected_facts", 0.0)), 4.0)
        self.assertEqual(float(model.last_generate_info.get("saliency_edges", 0.0)), 8.0)
        self.assertEqual(float(model.last_generate_info.get("saliency_consistency", 0.0)), 0.75)


if __name__ == "__main__":
    unittest.main()

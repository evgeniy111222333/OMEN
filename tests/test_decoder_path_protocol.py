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


def _decoder_test_config() -> OMENScaleConfig:
    cfg = OMENScaleConfig.demo()
    cfg.allow_noncanonical_ablation = True
    cfg.vocab_size = 256
    cfg.d_tok = 48
    cfg.n_heads_tok = 4
    cfg.n_layers_tok = 1
    cfg.seq_len = 32
    cfg.d_latent = 32
    cfg.n_latents = 8
    cfg.n_heads_lat = 2
    cfg.n_layers_lat = 1
    cfg.world_rnn_hidden = 48
    cfg.world_rollout_steps = 2
    cfg.mem_heads = 2
    cfg.mem_cache_size = 32
    cfg.mem_symbolic_cache_size = 32
    cfg.sym_vocab = 32
    cfg.sym_embed_dim = 16
    cfg.max_proof_depth = 2
    cfg.n_proof_cands = 4
    cfg.ltm_max_rules = 64
    cfg.sym_max_facts = 16
    cfg.net_enabled = True
    cfg.osf_enabled = True
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


class DecoderPathProtocolTest(unittest.TestCase):
    @staticmethod
    def _encode_example(cfg: OMENScaleConfig, code: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [ord(ch) for ch in code.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        return full[:, :-1], full[:, 1:]

    def test_forward_reports_osf_as_decoder_when_net_and_osf_are_enabled(self) -> None:
        cfg = _decoder_test_config()
        model = OMENScale(cfg)
        src, tgt = self._encode_example(cfg, "def add(a, b):\n    return a + b\n")

        out = model(src, tgt)

        self.assertEqual(out["decoder_mode"], "osf_decoder_with_net_encoder")
        self.assertEqual(out["decoder_path_osf"], 1.0)
        self.assertEqual(out["decoder_path_net"], 0.0)
        self.assertEqual(out["decoder_path_token"], 0.0)
        self.assertEqual(out["decoder_uses_net_encoder"], 1.0)
        self.assertEqual(out["decoder_osf_replaces_net"], 1.0)

    def test_generation_does_not_call_net_decode_when_osf_replaces_decoder(self) -> None:
        cfg = _decoder_test_config()
        model = OMENScale(cfg)
        model.eval()
        prompt, _ = self._encode_example(cfg, "def mul(a, b):\n    return a * b\n")

        def fake_encode(x: torch.Tensor, return_attn: bool = False):
            del return_attn
            batch, steps = x.shape
            h_tok = torch.zeros(batch, steps, cfg.d_tok, dtype=torch.float32)
            return h_tok, torch.zeros(batch, steps, dtype=torch.long), {}

        def fake_intent_forward(z_final: torch.Tensor):
            batch = z_final.size(0)
            d_intent = getattr(model.osf.cfg, "d_intent", cfg.d_latent)
            return SimpleNamespace(z_intent=torch.zeros(batch, d_intent, device=z_final.device, dtype=z_final.dtype))

        def fake_planner_forward(*args, **kwargs):
            del args, kwargs
            return SimpleNamespace()

        def fake_hier_decoder_forward(*args, **kwargs):
            h_tok = kwargs["h_tok"]
            logits = torch.full(
                (h_tok.size(0), h_tok.size(1), cfg.vocab_size),
                -1e4,
                device=h_tok.device,
            )
            logits[:, :, 7] = 1e4
            return logits, torch.tensor(0.0, device=h_tok.device)

        def fail_net_decode(*args, **kwargs):
            raise AssertionError("NET.decode should not be called when OSF replaces the decoder")

        prover_out = (
            torch.zeros(1, cfg.d_latent, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )

        with mock.patch.object(model.net, "encode", side_effect=fake_encode), \
             mock.patch.object(model.net, "decode", side_effect=fail_net_decode), \
             mock.patch.object(model.osf.intent_encoder, "forward", side_effect=fake_intent_forward), \
             mock.patch.object(model.osf.planner, "forward", side_effect=fake_planner_forward), \
             mock.patch.object(model.osf.hier_decoder, "forward", side_effect=fake_hier_decoder_forward), \
             mock.patch.object(model.prover, "forward", return_value=prover_out):
            generated = model.generate(prompt, max_new=1, temperature=1.0, dynamic_reasoning=False)

        last_tokens, _valid_rows, _last_idx = model._batch_last_content_tokens(generated)
        self.assertEqual(int(last_tokens[0].item()), 7)
        self.assertEqual(model.last_generate_info.get("decoder_mode"), "osf_decoder_with_net_encoder")
        self.assertEqual(float(model.last_generate_info.get("decoder_path_osf", 0.0)), 1.0)
        self.assertEqual(float(model.last_generate_info.get("decoder_path_net", 1.0)), 0.0)
        self.assertEqual(float(model.last_generate_info.get("decoder_osf_replaces_net", 0.0)), 1.0)

    def test_generation_uses_last_content_logit_and_fills_first_pad_slot(self) -> None:
        cfg = _decoder_test_config()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        model = OMENScale(cfg)
        model.eval()
        prompt = torch.tensor([[11, 13, 0, 0]], dtype=torch.long)

        prover_out = (
            torch.zeros(1, cfg.d_latent, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )

        def fake_tok_decoder(ctx: torch.Tensor, z_final: torch.Tensor):
            del z_final
            logits = torch.full(
                (ctx.size(0), ctx.size(1), cfg.vocab_size),
                -1e4,
                device=ctx.device,
            )
            logits[:, :, 5] = 1e3
            logits[:, 1, 7] = 1e4
            logits[:, -1, 9] = 1e4
            return logits

        with mock.patch.object(model.prover, "forward", return_value=prover_out), \
             mock.patch.object(model.tok_decoder, "forward", side_effect=fake_tok_decoder):
            generated = model.generate(prompt, max_new=1, temperature=1.0, dynamic_reasoning=False)

        last_tokens, _valid_rows, last_idx = model._batch_last_content_tokens(generated)
        self.assertEqual(generated.tolist(), [[11, 13, 7, 0]])
        self.assertEqual(last_idx.tolist(), [2])
        self.assertEqual(int(last_tokens[0].item()), 7)

    def test_training_does_not_call_net_decode_when_osf_replaces_decoder(self) -> None:
        cfg = _decoder_test_config()
        model = OMENScale(cfg)
        model.eval()
        src, tgt = self._encode_example(cfg, "def sub(a, b):\n    return a - b\n")

        def fake_encode(x: torch.Tensor, return_attn: bool = False, summarize_attn: bool = False):
            del return_attn, summarize_attn
            batch, steps = x.shape
            h_tok = torch.zeros(batch, steps, cfg.d_tok, dtype=torch.float32)
            return h_tok, torch.zeros(batch, steps, dtype=torch.long), {}

        def fail_net_decode(*args, **kwargs):
            raise AssertionError("NET.decode should not be called in Stage 2 when OSF replaces the decoder")

        def fake_osf(*args, **kwargs):
            self.assertTrue(kwargs.get("fast_mode"))
            tgt_tokens = kwargs["tgt"]
            logits = torch.full(
                (tgt_tokens.size(0), tgt_tokens.size(1), cfg.vocab_size),
                -1e4,
                device=tgt_tokens.device,
            )
            logits[:, :, 7] = 1e4
            return logits, {
                "l_plan": torch.tensor(0.0, device=tgt_tokens.device),
                "l_sim": torch.tensor(0.0, device=tgt_tokens.device),
                "l_refl": torch.tensor(0.0, device=tgt_tokens.device),
                "l_meta": torch.tensor(0.0, device=tgt_tokens.device),
                "l_intent": torch.tensor(0.0, device=tgt_tokens.device),
                "struct": torch.tensor(0.0, device=tgt_tokens.device),
                "plan_rl": torch.tensor(0.0, device=tgt_tokens.device),
            }

        prover_out = (
            torch.zeros(1, cfg.d_latent, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )

        with mock.patch.object(model.net, "encode", side_effect=fake_encode), \
             mock.patch.object(model.net, "decode", side_effect=fail_net_decode), \
             mock.patch.object(model.net, "compute_loss", return_value={
                 "net_total": torch.tensor(0.0),
                 "net_aux_tensor": torch.tensor(0.0),
                 "net_aux": 0.0,
                 "net_vocab_pen": 0.0,
             }), \
             mock.patch.object(model.osf, "forward", side_effect=fake_osf), \
             mock.patch.object(model.prover, "forward", return_value=prover_out), \
             mock.patch.object(model, "_ast_facts_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_rules_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_trace_from_bytes", return_value=None), \
             mock.patch.object(model, "_ast_lang_from_bytes", return_value=""):
            out = model(src, tgt, metric_profile="train_fast")

        self.assertIn("total", out)
        self.assertIn("ce", out)


if __name__ == "__main__":
    unittest.main()

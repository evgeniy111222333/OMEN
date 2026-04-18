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
from omen_symbolic.integration import SymbolicStateIntegrator


def _memory_gap_config(*, emc_enabled: bool = False) -> OMENScaleConfig:
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
    cfg.emc_enabled = emc_enabled
    cfg.saliency_enabled = False
    cfg.creative_cycle_enabled = False
    cfg.continuous_cycle_enabled = False
    cfg.continuous_cycle_eval_enabled = False
    cfg.continuous_cycle_eval_learning_enabled = False
    cfg.sym_query_gen_enabled = False
    cfg.sym_decoder_surprise_enabled = False
    cfg.world_graph_enabled = False
    cfg.eval_world_self_update_enabled = False
    cfg.ce_reinforce_enabled = False
    cfg.epistemic_memory_mix = 1.0
    return cfg


class MemoryGapProtocolTest(unittest.TestCase):
    @staticmethod
    def _z_triplet(d_latent: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.linspace(1.0, 2.0, d_latent, dtype=torch.float32).unsqueeze(0)
        v_mem = z * 0.25
        z_sim = z - v_mem
        return z, z_sim, v_mem

    def test_memory_grounded_gap_drops_when_memory_explains_state(self) -> None:
        cfg = _memory_gap_config()
        model = OMENScale(cfg)
        z, z_sim, v_mem = self._z_triplet(cfg.d_latent)

        _emap, gap_norm, hot_dims, stats = model._memory_grounded_epistemic_state(z, z_sim, v_mem)

        self.assertGreater(stats["gap_world_only"], 0.0)
        self.assertLess(stats["gap_memory_grounded"], 1e-6)
        self.assertLess(float(gap_norm.max().item()), 1e-6)
        self.assertGreater(stats["gap_memory_relief"], 0.0)
        self.assertGreater(stats["gap_memory_alignment"], 0.0)
        self.assertEqual(tuple(hot_dims.shape), tuple(z.shape))

    def test_forward_reports_memory_grounded_gap_telemetry(self) -> None:
        cfg = _memory_gap_config()
        model = OMENScale(cfg)
        model.eval()
        src = torch.tensor([[11, 13, 17, 19, 23, 29]], dtype=torch.long)
        tgt = torch.tensor([[13, 17, 19, 23, 29, 31]], dtype=torch.long)
        z, z_sim, v_mem = self._z_triplet(cfg.d_latent)
        latents = torch.zeros(1, cfg.n_latents, cfg.d_latent, dtype=torch.float32)

        def fake_tok_encoder(tokens: torch.Tensor):
            return torch.zeros(tokens.size(0), tokens.size(1), cfg.d_tok, dtype=torch.float32)

        def fake_world_rollout(*args, **kwargs):
            del args, kwargs
            traj = z_sim.unsqueeze(1).repeat(1, cfg.world_rollout_steps, 1)
            return traj, traj.clone()

        def fake_decoder(tokens: torch.Tensor, z_final: torch.Tensor):
            del z_final
            return torch.zeros(tokens.size(0), tokens.size(1), cfg.vocab_size, dtype=torch.float32)

        prover_out = (
            torch.zeros(1, cfg.d_latent, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )

        with mock.patch.object(model.tok_encoder, "forward", side_effect=fake_tok_encoder), \
             mock.patch.object(model.perceiver, "forward", return_value=(latents, z)), \
             mock.patch.object(model, "_sample_variational_latent", return_value=(z, z, torch.zeros_like(z))), \
             mock.patch.object(model, "_retrieve_memory", return_value=v_mem), \
             mock.patch.object(model, "_world_rollout_from_hidden", side_effect=fake_world_rollout), \
             mock.patch.object(model.curiosity, "forward", return_value=(z, torch.tensor(0.0))), \
             mock.patch.object(model.prover, "forward", return_value=prover_out), \
             mock.patch.object(model.tok_decoder, "forward", side_effect=fake_decoder), \
             mock.patch.object(model, "_ast_facts_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_rules_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_trace_from_bytes", return_value=None), \
             mock.patch.object(model, "_ast_lang_from_bytes", return_value=""):
            out = model(src, tgt)

        self.assertGreater(out["gap_world_only"], 0.0)
        self.assertLess(out["gap_norm"], 1e-6)
        self.assertLess(out["gap_memory_grounded"], 1e-6)
        self.assertGreater(out["gap_memory_relief"], 0.0)
        self.assertGreater(out["gap_memory_alignment"], 0.0)

    def test_generation_emc_receives_memory_grounded_gap(self) -> None:
        cfg = _memory_gap_config(emc_enabled=True)
        model = OMENScale(cfg)
        model.eval()
        prompt = torch.tensor([[11, 13, 17, 19, 23, 29]], dtype=torch.long)
        z, z_sim, v_mem = self._z_triplet(cfg.d_latent)
        latents = torch.zeros(1, cfg.n_latents, cfg.d_latent, dtype=torch.float32)
        seen_gap = []

        def fake_encode_for_saliency(tokens: torch.Tensor):
            h_tok = torch.zeros(tokens.size(0), tokens.size(1), cfg.d_tok, dtype=torch.float32)
            return h_tok, None, h_tok, None

        def fake_world_rollout(*args, **kwargs):
            del args, kwargs
            traj = z_sim.unsqueeze(1).repeat(1, cfg.world_rollout_steps, 1)
            return traj, traj.clone()

        def capture_emc_gap(z_in: torch.Tensor, gap_norm: torch.Tensor, *args, **kwargs):
            del z_in, args, kwargs
            seen_gap.append(gap_norm.detach().clone())
            zeros = torch.zeros(1, cfg.d_latent, dtype=torch.float32)
            return zeros, zeros

        def fake_decoder(tokens: torch.Tensor, z_final: torch.Tensor):
            del z_final
            logits = torch.full((tokens.size(0), tokens.size(1), cfg.vocab_size), -1e4, dtype=torch.float32)
            logits[:, :, 7] = 1e4
            return logits

        with mock.patch.object(model, "_encode_for_saliency", side_effect=fake_encode_for_saliency), \
             mock.patch.object(model.perceiver, "forward", return_value=(latents, z)), \
             mock.patch.object(model, "_sample_variational_latent", return_value=(z, z, torch.zeros_like(z))), \
             mock.patch.object(model, "_retrieve_memory", return_value=v_mem), \
             mock.patch.object(model, "_world_rollout_from_hidden", side_effect=fake_world_rollout), \
             mock.patch.object(model.emc, "run_episode_eval", side_effect=capture_emc_gap), \
             mock.patch.object(model.tok_decoder, "forward", side_effect=fake_decoder), \
             mock.patch.object(model, "_ast_facts_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_rules_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_trace_from_bytes", return_value=None), \
             mock.patch.object(model, "_ast_lang_from_bytes", return_value=""):
            generated = model.generate(prompt, max_new=1, temperature=1.0, dynamic_reasoning=False)

        self.assertEqual(int(generated[0, -1].item()), 7)
        self.assertTrue(seen_gap)
        self.assertLess(float(seen_gap[0].max().item()), 1e-6)
        self.assertGreater(float(model.last_generate_info.get("gap_world_only", 0.0)), 0.0)
        self.assertLess(float(model.last_generate_info.get("gap_memory_grounded", 0.0)), 1e-6)
        self.assertGreater(float(model.last_generate_info.get("gap_memory_relief", 0.0)), 0.0)
        self.assertGreaterEqual(float(model.last_generate_info.get("gap_memory_steps", 0.0)), 1.0)

    def test_symbolic_state_integration_is_memory_centered(self) -> None:
        z_concept = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32)
        z_symbolic = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)
        v_mem = torch.tensor([[1.0, 1.5, 2.0]], dtype=torch.float32)
        integrator = SymbolicStateIntegrator(d_latent=z_concept.size(-1))

        with mock.patch.object(integrator.pre_mem_gate, "forward", return_value=torch.zeros_like(z_concept)):
            pre_zero = integrator.pre_symbolic(z_concept, v_mem)
        with mock.patch.object(integrator.pre_mem_gate, "forward", return_value=torch.ones_like(z_concept)):
            pre_one = integrator.pre_symbolic(z_concept, v_mem)
        with mock.patch.object(integrator.post_mem_gate, "forward", return_value=torch.zeros_like(z_concept)), \
             mock.patch.object(integrator.sym_override_gate, "forward", return_value=torch.zeros_like(z_concept)):
            post_base = integrator.post_symbolic(z_concept, z_symbolic, v_mem)

        self.assertTrue(torch.allclose(pre_zero, v_mem))
        self.assertTrue(torch.allclose(pre_one, z_concept))
        self.assertTrue(torch.allclose(post_base, v_mem))


if __name__ == "__main__":
    unittest.main()

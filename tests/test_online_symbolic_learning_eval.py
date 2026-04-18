from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen import OMENConfig, build_omen
from omen_symbolic.world_graph import CanonicalWorldState


class OnlineSymbolicLearningEvalTest(unittest.TestCase):
    @staticmethod
    def _encode_example(cfg: OMENConfig, code: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [ord(ch) for ch in code.encode("ascii", errors="ignore").decode("ascii")]
        encoded = encoded[: cfg.seq_len]
        if len(encoded) < cfg.seq_len:
            encoded = encoded + [0] * (cfg.seq_len - len(encoded))
        full = torch.tensor([encoded], dtype=torch.long)
        return full[:, :-1], full[:, 1:]

    def test_eval_forward_can_run_symbolic_online_learning(self) -> None:
        cfg = OMENConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.ce_reinforce_enabled = True
        cfg.ce_reinforce_eval_enabled = True
        cfg.ce_reinforce_fallback_only = False
        model = build_omen(cfg)
        model.eval()

        src, tgt = self._encode_example(cfg, "def add(a, b):\n    return a + b\n")

        out = model(src, tgt)
        self.assertGreater(out["ce_reinforce_utility"], 0.0)
        self.assertGreaterEqual(out["sym_cycle_active"], 1.0)
        self.assertGreaterEqual(out["sym_cycle_eval_active"], 1.0)
        self.assertGreaterEqual(out["sym_cycle_checked"], 0.0)
        self.assertGreaterEqual(out["eval_world_self_update_applied"], 1.0)
        self.assertGreater(out["eval_world_self_update_loss"], 0.0)
        self.assertGreater(out["eval_world_self_update_params"], 0.0)
        self.assertEqual(out["canonical_stack"], "omen_scale_world_graph")
        self.assertEqual(out["canonical_public_module"], "omen.OMEN")
        self.assertEqual(out["canonical_repository_axis"], "omen_scale_single_canon_repository")
        self.assertIsInstance(out["z"], CanonicalWorldState)
        self.assertTrue(torch.allclose(out["z_world"], out["z_dense"]))

    def test_eval_forward_updates_world_model_weights(self) -> None:
        cfg = OMENConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.creative_cycle_enabled = False
        cfg.continuous_cycle_enabled = False
        model = build_omen(cfg)
        model.eval()

        src, tgt = self._encode_example(
            cfg,
            "def add(a, b):\n    total = a + b\n    return total\n",
        )
        before = next(model.world_rnn.parameters()).detach().clone()
        out = model(src, tgt)
        after = next(model.world_rnn.parameters()).detach()

        self.assertEqual(out["world_graph_hidden_teacher_applied"], 0.0)
        self.assertGreaterEqual(out["world_graph_execution_steps"], 1.0)
        self.assertEqual(out["eval_world_self_update_applied"], 1.0)
        self.assertGreater(out["eval_world_self_update_grad_norm"], 0.0)
        self.assertGreater(out["eval_world_self_update_lr"], 0.0)
        self.assertFalse(torch.allclose(before, after))

    def test_generation_can_run_eval_world_self_update(self) -> None:
        cfg = OMENConfig.demo()
        cfg.allow_noncanonical_ablation = True
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.creative_cycle_enabled = False
        model = build_omen(cfg)
        model.eval()

        src, _ = self._encode_example(
            cfg,
            "def mul(a, b):\n    result = a * b\n    return result\n",
        )
        before = next(model.world_rnn.parameters()).detach().clone()
        generated = model.generate(src, max_new=2, dynamic_reasoning=True)
        after = next(model.world_rnn.parameters()).detach()
        info = getattr(model, "last_generate_info", {})

        self.assertEqual(generated.size(1), src.size(1) + 2)
        self.assertGreaterEqual(float(info.get("adaptive_learning_active", 0.0)), 1.0)
        self.assertGreaterEqual(float(info.get("eval_world_self_update_applied", 0.0)), 1.0)
        self.assertGreater(float(info.get("eval_world_self_update_loss", 0.0)), 0.0)
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()

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
    def test_eval_forward_can_run_symbolic_online_learning(self) -> None:
        cfg = OMENConfig.demo()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.saliency_enabled = False
        cfg.ce_reinforce_enabled = True
        cfg.ce_reinforce_eval_enabled = True
        cfg.ce_reinforce_fallback_only = False
        model = build_omen(cfg)
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
        self.assertGreater(out["ce_reinforce_utility"], 0.0)
        self.assertGreaterEqual(out["sym_cycle_checked"], 0.0)
        self.assertEqual(out["canonical_stack"], "omen_scale_world_graph")
        self.assertEqual(out["canonical_public_module"], "omen.OMEN")
        self.assertEqual(out["canonical_repository_axis"], "omen_scale_single_canon_repository")
        self.assertIsInstance(out["z"], CanonicalWorldState)
        self.assertTrue(torch.allclose(out["z_world"], out["z_dense"]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_prolog import Const, DifferentiableProver, HornAtom, SymbolicTaskContext


class _DummyWorld(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.act_emb = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, z_state, action, h=None):
        del h
        act = self.act_emb(action)
        return torch.tanh(self.proj(torch.cat([z_state, act], dim=-1))), None


class SymbolicCycleEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.observed = frozenset({
            HornAtom(pred=5, args=(Const(1), Const(2))),
            HornAtom(pred=6, args=(Const(2), Const(3))),
            HornAtom(pred=8, args=(Const(3), Const(4))),
        })
        self.goal = HornAtom(pred=7, args=(Const(1), Const(4)))
        self.targets = frozenset({self.goal})

    def _make_prover(self, *, eval_enabled: bool, eval_learning_enabled: bool = True) -> DifferentiableProver:
        prover = DifferentiableProver(
            d_latent=32,
            sym_vocab=32,
            max_rules=32,
            max_depth=3,
            n_cands=4,
        ).to(self.device)
        prover.set_world_rnn(_DummyWorld(32, 32).to(self.device))
        prover.configure_hypothesis_cycle(
            enabled=True,
            eval_enabled=eval_enabled,
            eval_learning_enabled=eval_learning_enabled,
            max_contextual=4,
            max_neural=0,
            accept_threshold=0.25,
            verify_threshold=0.45,
        )
        prover.configure_creative_cycle(enabled=False)
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=self.observed,
                goal=self.goal,
                target_facts=self.targets,
                provenance="unit",
                trigger_abduction=False,
                metadata={"last_src": 1.0, "last_tgt": 4.0},
            )
        )
        return prover

    def test_eval_cycle_runs_when_enabled(self) -> None:
        prover = self._make_prover(eval_enabled=True)
        prover.eval()
        z = torch.randn(1, 32, device=self.device)
        _z_sym, sym_loss = prover(z, torch.tensor(0.1, device=self.device))
        self.assertTrue(torch.isfinite(sym_loss))
        self.assertEqual(prover.last_forward_info.get("cycle_active", 0.0), 1.0)
        self.assertEqual(prover.last_forward_info.get("cycle_eval_active", 0.0), 1.0)
        self.assertEqual(prover.last_forward_info.get("cycle_learning_active", 0.0), 1.0)
        self.assertGreaterEqual(prover.last_forward_info.get("cycle_checked", 0.0), 1.0)
        self.assertGreaterEqual(prover.last_forward_info.get("cycle_candidate_budget", 0.0), 1.0)

    def test_eval_cycle_stays_off_when_disabled(self) -> None:
        prover = self._make_prover(eval_enabled=False)
        prover.eval()
        z = torch.randn(1, 32, device=self.device)
        _z_sym, sym_loss = prover(z, torch.tensor(0.1, device=self.device))
        self.assertTrue(torch.isfinite(sym_loss))
        self.assertEqual(prover.last_forward_info.get("cycle_active", 0.0), 0.0)
        self.assertEqual(prover.last_forward_info.get("cycle_eval_active", 0.0), 0.0)
        self.assertEqual(prover.last_forward_info.get("cycle_checked", 0.0), 0.0)

    def test_eval_cycle_learning_toggle_controls_gradient_contract(self) -> None:
        prover = self._make_prover(eval_enabled=True, eval_learning_enabled=False)
        prover.eval()
        z = torch.randn(1, 32, device=self.device, requires_grad=True)
        cycle = prover.continuous_hypothesis_cycle(z, self.observed, self.targets, self.device)
        self.assertEqual(cycle["stats"].get("learning_active", 1.0), 0.0)
        self.assertFalse(cycle["loss_tensor"].requires_grad)

        learning_prover = self._make_prover(eval_enabled=True, eval_learning_enabled=True)
        learning_prover.eval()
        z_learn = torch.randn(1, 32, device=self.device, requires_grad=True)
        cycle_learn = learning_prover.continuous_hypothesis_cycle(
            z_learn,
            self.observed,
            self.targets,
            self.device,
        )
        self.assertEqual(cycle_learn["stats"].get("learning_active", 0.0), 1.0)
        self.assertTrue(cycle_learn["loss_tensor"].requires_grad)


if __name__ == "__main__":
    unittest.main()

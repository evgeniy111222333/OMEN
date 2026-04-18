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

from omen_osf import OSFConfig, OSFSynthesizer
from omen_osf_meta import STRATEGY_CAREFUL
from omen_osf_planner import SymbolicPlanner


class OSFProtocolTest(unittest.TestCase):
    def test_plan_policy_evaluate_matches_split_heads(self) -> None:
        planner = SymbolicPlanner(
            d_intent=8,
            d_plan=8,
            n_operators=6,
            max_depth=2,
            beam_width=2,
        )
        planner.eval()
        z_intent = torch.randn(1, 8)
        z_ctx = torch.randn(1, 8)
        wm_emb = torch.randn(1, 8)

        logits_ref = planner.policy.action_logits(z_intent, z_ctx, wm_emb, depth=1)
        value_ref = planner.policy.value(z_intent, z_ctx, wm_emb, depth=1)
        logits_eval, value_eval = planner.policy.evaluate(z_intent, z_ctx, wm_emb, depth=1)

        self.assertTrue(torch.allclose(logits_ref, logits_eval))
        self.assertTrue(torch.allclose(value_ref, value_eval))

    def test_operator_library_batches_op_effect_generation(self) -> None:
        planner = SymbolicPlanner(
            d_intent=8,
            d_plan=8,
            n_operators=7,
            max_depth=2,
            beam_width=2,
        )
        z_intent = torch.randn(1, 8)

        with mock.patch.object(
            planner.op_param_gen,
            "forward",
            wraps=planner.op_param_gen.forward,
        ) as op_forward:
            lib = planner._operator_library(z_intent, goal_id=3, device=torch.device("cpu"))

        self.assertEqual(len(lib), 7)
        self.assertEqual(op_forward.call_count, 1)

    def test_repair_path_reuses_supplied_verify_and_stops_on_perfect_candidate(self) -> None:
        synth = OSFSynthesizer(
            OSFConfig(
                d_intent=8,
                n_goals=4,
                d_plan=8,
                n_operators=6,
                template_len=4,
                max_plan_depth=3,
                beam_width=2,
            ),
            d_latent=8,
            d_tok=8,
            vocab_size=32,
        )
        intent_state = synth.intent_encoder(torch.randn(1, 8))
        current_plan = mock.sentinel.current_plan
        candidate_plan = mock.sentinel.candidate_plan
        current_verify = SimpleNamespace(
            goal_progress=torch.tensor([0.25]),
            l_verify=torch.tensor(0.8),
            mismatch_mask=torch.ones(1, dtype=torch.bool),
        )
        perfect_verify = SimpleNamespace(
            goal_progress=torch.tensor([1.0]),
            l_verify=torch.tensor(0.0),
            mismatch_mask=torch.zeros(1, dtype=torch.bool),
        )

        with mock.patch.object(
            synth,
            "_plan_with_strategy",
            return_value=candidate_plan,
        ) as plan_with_strategy, mock.patch.object(
            synth.symbolic_verifier,
            "forward",
            return_value=perfect_verify,
        ) as verifier:
            best_plan, best_verify, tried = synth._repair_plan_symbolically(
                intent_state,
                current_plan,
                STRATEGY_CAREFUL,
                plan_depth_use=1,
                batch_size=1,
                device=torch.device("cpu"),
                best_verify=current_verify,
            )

        self.assertIs(best_plan, candidate_plan)
        self.assertIs(best_verify, perfect_verify)
        self.assertEqual(tried, 1)
        self.assertEqual(plan_with_strategy.call_count, 1)
        self.assertEqual(verifier.call_count, 1)


if __name__ == "__main__":
    unittest.main()

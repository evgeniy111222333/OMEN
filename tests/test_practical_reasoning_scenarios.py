from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_emc import ACTION_ABDUCE, ACTION_FC, ACTION_INTRINSIC, ACTION_RECALL, ACTION_STOP, N_ACTIONS
from omen_prolog import (
    Const,
    DifferentiableProver,
    EpistemicStatus,
    HornAtom,
    HornClause,
    SymbolicTaskContext,
    Var,
)
from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig
from omen_symbolic.abduction_search import bridge_variable_count
from omen_symbolic.execution_trace import (
    TRACE_BINOP_EVENT_PRED,
    TRACE_PARAM_BIND_PRED,
    TRACE_RETURN_EVENT_PRED,
    build_symbolic_trace_bundle,
)


class _DummyWorld(nn.Module):
    def __init__(self, d_model: int = 32, vocab_size: int = 32):
        super().__init__()
        self.act_emb = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, z_state: torch.Tensor, action: torch.Tensor, h=None):
        del h
        act = self.act_emb(action)
        return torch.tanh(self.proj(torch.cat([z_state, act], dim=-1))), None


def _scripted_actor(sequence: list[int]):
    state = {"idx": 0}

    def forward(state_vec: torch.Tensor) -> torch.Tensor:
        del state_vec
        action = sequence[min(state["idx"], len(sequence) - 1)]
        state["idx"] += 1
        logits = torch.full((1, N_ACTIONS), -1e4, dtype=torch.float32)
        logits[:, action] = 1e4
        return logits

    return forward


def _emc_scenario_config() -> OMENScaleConfig:
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
    cfg.emc_enabled = True
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


def _z_triplet(d_latent: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.linspace(1.0, 2.0, d_latent, dtype=torch.float32).unsqueeze(0)
    v_mem = z * 0.25
    z_sim = z - v_mem
    return z, z_sim, v_mem


class _MockMemory:
    def __init__(self, v_mem: torch.Tensor) -> None:
        self.v_mem = v_mem

    def read(self, z_query: torch.Tensor) -> torch.Tensor:
        return self.v_mem.to(device=z_query.device, dtype=z_query.dtype).expand_as(z_query)


class _MockKB:
    def __init__(self) -> None:
        self.facts = frozenset()
        self.rules = []

    def tick(self) -> None:
        return None

    def consolidate(self, use_count_threshold: int = 2) -> None:
        del use_count_threshold
        return None

    def n_facts(self) -> int:
        return len(self.facts)

    def __len__(self) -> int:
        return len(self.rules)


class _MockProver:
    def __init__(self, d_latent: int) -> None:
        self.d = d_latent
        self.kb = _MockKB()
        self.max_depth = 1
        self._step = 0
        self.consolidate_every = 100
        self.last_creative_fast_mode = False
        self.task_context = SimpleNamespace(
            target_facts=frozenset(),
            provenance="latent",
            trigger_abduction=False,
        )

    def materialize_task_context_facts(self) -> None:
        return None

    def current_goal(self, z: torch.Tensor):
        del z
        return "goal"

    def current_working_facts(self):
        return frozenset()

    def ground(self, facts, device: torch.device) -> torch.Tensor:
        del facts
        return torch.zeros(1, self.d, device=device)

    def forward_chain_reasoned(self, max_depth: int, starting_facts, only_verified: bool, device: torch.device):
        del max_depth, starting_facts, only_verified, device
        return frozenset()

    def run_creative_cycle(self, z_cur, working_facts, targets, device, *, fast_mode: bool = False):
        del z_cur, working_facts, targets, device
        self.last_creative_fast_mode = bool(fast_mode)
        return SimpleNamespace(metrics={})

    def _goal_supported(self, goal, all_facts) -> bool:
        del goal, all_facts
        return False


class _FCGapProver(_MockProver):
    def __init__(self, d_latent: int, z_after: torch.Tensor) -> None:
        super().__init__(d_latent)
        self._z_after = z_after
        self._seed_fact = HornAtom(1, (1, 1))
        self._derived_fact = HornAtom(1, (1, 2))
        self._seed = frozenset({self._seed_fact})

    def current_working_facts(self):
        return self._seed

    def forward_chain_step_local(self, current_facts, only_verified: bool = False, device: torch.device | None = None):
        del current_facts, only_verified, device
        return 1, frozenset({self._derived_fact}), [], frozenset({self._seed_fact, self._derived_fact})

    def ground(self, facts, device: torch.device) -> torch.Tensor:
        if len(facts) > 1:
            return self._z_after.to(device=device)
        return torch.zeros(1, self.d, device=device)


class _AbduceGapProver(_MockProver):
    def __init__(self, d_latent: int, z_after: torch.Tensor) -> None:
        super().__init__(d_latent)
        self._z_after = z_after
        self._seed_fact = HornAtom(1, (1, 1))
        self._explained_fact = HornAtom(2, (1, 3))
        self._seed = frozenset({self._seed_fact})
        self.abduced = False

    def current_working_facts(self):
        return self._seed

    def abduce_and_learn(self, z_cur: torch.Tensor, err_val: float, force: bool = True):
        del z_cur, err_val, force
        self.abduced = True
        return 1, torch.tensor(0.0), torch.tensor(0.0), 0.0

    def _induce_proposed_rules_locally(self, working_facts, target_facts, device: torch.device):
        del working_facts, target_facts, device
        return {
            "checked": 0.0,
            "verified": 0.0,
            "contradicted": 0.0,
            "retained": 0.0,
            "matched_predictions": 0.0,
            "mean_score": 0.0,
        }

    def forward_chain_reasoned(self, max_depth: int, starting_facts, only_verified: bool, device: torch.device):
        del max_depth, starting_facts, only_verified, device
        if self.abduced:
            return frozenset({self._seed_fact, self._explained_fact})
        return self._seed

    def ground(self, facts, device: torch.device) -> torch.Tensor:
        if len(facts) > 1:
            return self._z_after.to(device=device)
        return torch.zeros(1, self.d, device=device)


class _IntrinsicGoalProver(_MockProver):
    def __init__(self, d_latent: int) -> None:
        super().__init__(d_latent)
        self.primary_goal = HornAtom(700, (1, 2))
        self.intrinsic_goal = HornAtom(701, (3, 4))
        self.focused = False
        self.task_context = SimpleNamespace(
            goal=self.primary_goal,
            target_facts=frozenset({self.primary_goal}),
            provenance="latent",
            trigger_abduction=False,
            metadata={},
        )

    def current_goal(self, z: torch.Tensor):
        del z
        return self.task_context.goal

    def current_intrinsic_goal(self):
        return self.intrinsic_goal

    def current_intrinsic_value(self) -> float:
        return 0.8

    def scheduled_intrinsic_goals(self):
        return (self.intrinsic_goal,)

    def focus_intrinsic_goal(self):
        self.focused = True
        targets = set(self.task_context.target_facts)
        targets.add(self.intrinsic_goal)
        self.task_context = SimpleNamespace(
            goal=self.intrinsic_goal,
            target_facts=frozenset(targets),
            provenance="intrinsic",
            trigger_abduction=True,
            metadata={
                "emc_intrinsic_focus": 1.0,
                "intrinsic_goal_active": 1.0,
            },
        )
        return self.intrinsic_goal


class PracticalReasoningScenariosTest(unittest.TestCase):
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

    def test_practical_deduction_executes_verified_chain(self) -> None:
        prover = DifferentiableProver(
            d_latent=32,
            sym_vocab=32,
            max_rules=8,
            max_depth=3,
            n_cands=4,
        ).to(self.device)
        seed_facts = frozenset({
            HornAtom(pred=11, args=(Const(1), Const(2))),
            HornAtom(pred=11, args=(Const(2), Const(3))),
        })
        goal = HornAtom(pred=77, args=(Const(1), Const(3)))
        verified_rule = HornClause(
            head=HornAtom(pred=77, args=(Var("X"), Var("Z"))),
            body=(
                HornAtom(pred=11, args=(Var("X"), Var("Y"))),
                HornAtom(pred=11, args=(Var("Y"), Var("Z"))),
            ),
        )
        prover.kb.add_rule(verified_rule, status=EpistemicStatus.verified)

        derived = prover.forward_chain_reasoned(
            max_depth=3,
            starting_facts=seed_facts,
            only_verified=True,
            device=self.device,
        )

        self.assertIn(goal, derived)

    def test_practical_induction_verifies_bridge_rule_and_enables_verified_deduction(self) -> None:
        prover = DifferentiableProver(
            d_latent=32,
            sym_vocab=32,
            max_rules=16,
            max_depth=3,
            n_cands=4,
        ).to(self.device)
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=self.observed,
                goal=self.goal,
                target_facts=self.targets,
                provenance="practical",
            )
        )
        bridge_rule = HornClause(
            head=HornAtom(pred=7, args=(Var("X"), Var("Z"))),
            body=(
                HornAtom(pred=5, args=(Var("X"), Var("Y"))),
                HornAtom(pred=6, args=(Var("Y"), Var("W"))),
                HornAtom(pred=8, args=(Var("W"), Var("Z"))),
            ),
        )
        prover.kb.add_rule(bridge_rule, status=EpistemicStatus.proposed)

        before = prover.forward_chain_reasoned(
            max_depth=3,
            starting_facts=self.observed,
            only_verified=True,
            device=self.device,
        )
        induction = prover._induce_proposed_rules_locally(
            self.observed,
            self.targets,
            self.device,
        )
        after = prover.forward_chain_reasoned(
            max_depth=3,
            starting_facts=self.observed,
            only_verified=True,
            device=self.device,
        )

        self.assertNotIn(self.goal, before)
        self.assertEqual(induction["verified"], 1.0)
        self.assertEqual(prover.kb.rule_status(bridge_rule), EpistemicStatus.verified)
        self.assertIn(self.goal, after)

    def test_practical_abduction_induction_and_deduction_compose_end_to_end(self) -> None:
        prover = DifferentiableProver(
            d_latent=32,
            sym_vocab=32,
            max_rules=32,
            max_depth=3,
            n_cands=4,
        ).to(self.device)
        prover.set_world_rnn(_DummyWorld().to(self.device))
        prover.configure_hypothesis_cycle(
            enabled=True,
            max_contextual=4,
            max_neural=0,
            accept_threshold=0.25,
            verify_threshold=0.45,
        )
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=self.observed,
                goal=self.goal,
                target_facts=self.targets,
                provenance="practical",
                metadata={"last_src": 1.0, "last_tgt": 4.0},
            )
        )
        prover.train()
        z = torch.randn(1, 32, device=self.device)
        prover._last_z = z.detach()

        cycle = prover.continuous_hypothesis_cycle(
            z,
            self.observed,
            self.targets,
            self.device,
        )
        derived_pre_verify = prover.forward_chain_reasoned(
            max_depth=3,
            starting_facts=self.observed,
            only_verified=False,
            device=self.device,
        )
        induction = prover._induce_proposed_rules_locally(
            self.observed,
            self.targets,
            self.device,
        )
        derived_verified = prover.forward_chain_reasoned(
            max_depth=3,
            starting_facts=self.observed,
            only_verified=True,
            device=self.device,
        )
        bridge_rules = [rule for rule in prover.kb.rules if rule.head.pred == self.goal.pred]
        max_bridge_vars = max(
            (bridge_variable_count(rule.head, rule.body) for rule in bridge_rules),
            default=0,
        )

        self.assertGreaterEqual(cycle["stats"]["checked"], 1.0)
        self.assertGreaterEqual(cycle["stats"]["added"], 1.0)
        self.assertIn(self.goal, derived_pre_verify)
        self.assertGreaterEqual(induction["verified"], 1.0)
        self.assertIn(self.goal, derived_verified)
        self.assertGreaterEqual(max_bridge_vars, 1)

    def test_practical_trace_abduction_uses_counterexamples(self) -> None:
        trace_bundle = build_symbolic_trace_bundle(
            "def add(a, b):\n    return a + b\n",
            max_steps=16,
            max_counterexamples=2,
        )
        assert trace_bundle is not None
        trace_goal = next(
            (
                fact
                for fact in trace_bundle.target_facts
                if fact.pred == TRACE_RETURN_EVENT_PRED
            ),
            next(iter(trace_bundle.target_facts)),
        )
        prover = DifferentiableProver(
            d_latent=32,
            sym_vocab=32,
            max_rules=32,
            max_depth=3,
            n_cands=2,
        ).to(self.device)
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=trace_bundle.observed_facts,
                goal=trace_goal,
                target_facts=trace_bundle.target_facts,
                execution_trace=trace_bundle,
                provenance="trace",
                trigger_abduction=True,
            )
        )
        trace_rules = prover._trace_abduction_candidates(max_candidates=8, max_body_atoms=2)
        self.assertTrue(trace_rules)
        good_rule = next(
            (
                rule
                for rule in trace_rules
                if rule.head.pred == TRACE_RETURN_EVENT_PRED
                and any(atom.pred == TRACE_BINOP_EVENT_PRED for atom in rule.body)
            ),
            trace_rules[0],
        )
        bad_rule = HornClause(
            head=HornAtom(
                pred=TRACE_RETURN_EVENT_PRED,
                args=(Var("S"), Var("SC"), Var("R")),
            ),
            body=(
                HornAtom(
                    pred=TRACE_PARAM_BIND_PRED,
                    args=(Var("S"), Var("SC"), Var("P"), Var("R")),
                ),
            ),
        )

        good_trace_error = prover._trace_prediction_error_for_rule(good_rule, trace_bundle)
        bad_trace_error = prover._trace_prediction_error_for_rule(bad_rule, trace_bundle)
        good_counterexample_error = prover._counterexample_error_for_rule(good_rule, trace_bundle)
        bad_counterexample_error = prover._counterexample_error_for_rule(bad_rule, trace_bundle)

        self.assertLessEqual(good_trace_error, bad_trace_error)
        self.assertEqual(good_counterexample_error, 0.0)
        self.assertGreater(bad_counterexample_error, 0.0)

    def test_practical_emc_controls_recall_fc_abduce_and_intrinsic_paths(self) -> None:
        cfg = _emc_scenario_config()
        z, z_sim, v_mem = _z_triplet(cfg.d_latent)

        model = OMENScale(cfg)
        model.emc.eval()
        memory = _MockMemory(v_mem)
        gap_feedback = model._make_emc_gap_feedback(z_sim)

        recall_prover = _MockProver(cfg.d_latent)
        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_RECALL, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            _z_sym, v_mem_out = model.emc.run_episode_eval(
                z,
                torch.tensor([2.0], dtype=torch.float32),
                None,
                recall_prover,
                memory,
                device=z.device,
                gap_feedback=gap_feedback,
            )
        self.assertTrue(torch.allclose(v_mem_out, v_mem))
        self.assertGreater(float(recall_prover.last_forward_info.get("emc_recall_gap_delta", 0.0)), 0.0)
        self.assertEqual(float(recall_prover.last_forward_info.get("emc_recall_effective_ratio", 0.0)), 1.0)

        fc_prover = _FCGapProver(cfg.d_latent, z_sim)
        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_FC, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            model.emc.run_episode_eval(
                z,
                torch.tensor([2.0], dtype=torch.float32),
                None,
                fc_prover,
                memory,
                device=z.device,
                gap_feedback=gap_feedback,
            )
        self.assertGreater(float(fc_prover.last_forward_info.get("emc_gap_delta_mean", 0.0)), 0.0)

        abduce_prover = _AbduceGapProver(cfg.d_latent, z_sim)
        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_ABDUCE, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            model.emc.run_episode_eval(
                z,
                torch.tensor([2.0], dtype=torch.float32),
                None,
                abduce_prover,
                memory,
                device=z.device,
                gap_feedback=gap_feedback,
            )
        self.assertTrue(abduce_prover.abduced)
        self.assertGreater(float(abduce_prover.last_forward_info.get("abduced_rules", 0.0)), 0.0)

        intrinsic_prover = _IntrinsicGoalProver(cfg.d_latent)
        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_INTRINSIC, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            model.emc.run_episode_eval(
                z,
                torch.tensor([1.0], dtype=torch.float32),
                None,
                intrinsic_prover,
                memory,
                device=z.device,
            )
        self.assertTrue(intrinsic_prover.focused)
        self.assertEqual(float(intrinsic_prover.last_forward_info.get("emc_intrinsic_actions", 0.0)), 1.0)
        self.assertEqual(float(intrinsic_prover.last_forward_info.get("emc_intrinsic_goal_active", 0.0)), 1.0)


if __name__ == "__main__":
    unittest.main()

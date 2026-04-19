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

from omen_emc import ACTION_ABDUCE, ACTION_FC, ACTION_INTRINSIC, ACTION_RECALL, ACTION_STOP, N_ACTIONS
from omen_prolog import HornAtom
from omen_scale import OMENScale
from omen_scale_config import OMENScaleConfig


def _emc_gap_config() -> OMENScaleConfig:
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
        self.task_context = SimpleNamespace(
            target_facts=frozenset(),
            provenance="latent",
            trigger_abduction=False,
        )
        self.last_creative_fast_mode = False

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

    def run_creative_cycle(
        self,
        z_cur: torch.Tensor,
        working_facts,
        targets,
        device: torch.device,
        *,
        fast_mode: bool = False,
    ):
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


class EMCGapProtocolTest(unittest.TestCase):
    def test_action_masking_is_fp16_safe(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        logits = torch.zeros(N_ACTIONS, dtype=torch.float16)
        mask = torch.zeros(N_ACTIONS, dtype=torch.bool)
        mask[ACTION_ABDUCE] = True
        mask[ACTION_INTRINSIC] = True

        masked = model.emc._mask_action_logits(logits, mask)

        self.assertEqual(masked.dtype, logits.dtype)
        self.assertEqual(float(masked[ACTION_STOP].item()), 0.0)
        self.assertEqual(float(masked[ACTION_RECALL].item()), 0.0)
        self.assertEqual(float(masked[ACTION_ABDUCE].item()), torch.finfo(torch.float16).min)
        self.assertEqual(float(masked[ACTION_INTRINSIC].item()), torch.finfo(torch.float16).min)

    def test_task_estimator_bce_is_autocast_safe(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        task_state = torch.randn(2, cfg.d_latent, dtype=torch.float32)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            loss = model.emc._task_estimator_bce_loss(task_state, goal_proved=True)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dtype, torch.float32)

    def test_eval_episode_routes_memory_pressure_into_stopping_utility(self) -> None:
        cfg = _emc_gap_config()
        cfg.emc_lambda_memory_residual = 0.05
        cfg.emc_lambda_memory_misalignment = 0.04
        model = OMENScale(cfg)
        model.emc.eval()
        prover = _MockProver(cfg.d_latent)
        memory = _MockMemory(torch.zeros(1, cfg.d_latent, dtype=torch.float32))
        z = torch.ones(1, cfg.d_latent, dtype=torch.float32)
        captured: dict[str, float] = {}

        def capture_stop(*args, **kwargs):
            captured["memory_penalty"] = float(kwargs.get("memory_penalty", 0.0))
            return torch.zeros(1, dtype=torch.float32)

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", side_effect=capture_stop), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            model.emc.run_episode_eval(
                z,
                torch.tensor([1.0], dtype=torch.float32),
                None,
                prover,
                memory,
                device=z.device,
                gap_features={
                    "gap_world_only": 4.0,
                    "gap_memory_grounded": 1.0,
                    "gap_memory_relief": 3.0,
                    "gap_memory_residual": 2.0,
                    "gap_memory_alignment": -0.5,
                },
            )

        expected = 0.05 * 0.4 + 0.04 * 0.75
        self.assertAlmostEqual(captured["memory_penalty"], expected, places=6)

    def test_eval_episode_encodes_separate_gap_channels(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        model.emc.eval()
        prover = _MockProver(cfg.d_latent)
        memory = _MockMemory(torch.zeros(1, cfg.d_latent, dtype=torch.float32))
        z = torch.ones(1, cfg.d_latent, dtype=torch.float32)
        captured: dict[str, torch.Tensor] = {}
        original_forward = model.emc.state_enc.forward

        def capture_forward(*args, **kwargs):
            captured["gap_world"] = kwargs["gap_world"].detach().clone()
            captured["gap_grounded"] = kwargs["gap_grounded"].detach().clone()
            captured["gap_relief"] = kwargs["gap_relief"].detach().clone()
            captured["gap_residual"] = kwargs["gap_residual"].detach().clone()
            captured["gap_alignment"] = kwargs["gap_alignment"].detach().clone()
            return original_forward(*args, **kwargs)

        with mock.patch.object(model.emc.state_enc, "forward", side_effect=capture_forward), \
             mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            model.emc.run_episode_eval(
                z,
                torch.tensor([1.0], dtype=torch.float32),
                None,
                prover,
                memory,
                device=z.device,
                gap_features={
                    "gap_world_only": 4.0,
                    "gap_memory_grounded": 1.0,
                    "gap_memory_relief": 3.0,
                    "gap_memory_residual": 2.0,
                    "gap_memory_alignment": 0.25,
                },
            )

        self.assertAlmostEqual(float(captured["gap_world"].item()), 0.8, places=6)
        self.assertAlmostEqual(float(captured["gap_grounded"].item()), 0.2, places=6)
        self.assertAlmostEqual(float(captured["gap_relief"].item()), 0.6, places=6)
        self.assertAlmostEqual(float(captured["gap_residual"].item()), 0.4, places=6)
        self.assertAlmostEqual(float(captured["gap_alignment"].item()), 0.25, places=6)

    def test_training_episode_uses_gap_feedback_for_recall(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        model.emc.eval()
        prover = _MockProver(cfg.d_latent)
        z, z_sim, v_mem = _z_triplet(cfg.d_latent)
        memory = _MockMemory(v_mem)
        gap_feedback = model._make_emc_gap_feedback(z_sim)

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_RECALL, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            z_sym, sym_loss, v_mem_out, meta_loss, traj = model.emc.run_episode(
                z,
                torch.tensor([2.0], dtype=torch.float32),
                None,
                prover,
                memory,
                torch.tensor(0.0, dtype=torch.float32),
                device=z.device,
                gap_feedback=gap_feedback,
            )

        self.assertEqual(tuple(z_sym.shape), tuple(z.shape))
        self.assertTrue(torch.isfinite(sym_loss))
        self.assertTrue(torch.isfinite(meta_loss))
        self.assertTrue(torch.allclose(v_mem_out, v_mem))
        self.assertEqual(len(traj.recall_gap_deltas), 1)
        self.assertGreater(traj.recall_gap_deltas[0], 0.0)
        self.assertGreater(traj.recall_gap_reliefs[0], 0.0)
        self.assertEqual(traj.recall_effective_steps, 1.0)
        self.assertGreater(float(prover.last_forward_info.get("emc_recall_gap_delta", 0.0)), 0.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_recall_effective_ratio", 0.0)), 1.0)

    def test_train_fast_cadences_background_symbolic_maintenance(self) -> None:
        cfg = _emc_gap_config()
        cfg.emc_train_fast_maintenance_every = 4
        model = OMENScale(cfg)
        model.emc.train()
        prover = _MockProver(cfg.d_latent)
        prover.continuous_cycle_enabled = True
        prover._step = 1
        z = torch.ones(1, cfg.d_latent, dtype=torch.float32)
        memory = _MockMemory(torch.zeros_like(z))
        cycle_result = {
            "loss_tensor": torch.tensor(0.0),
            "added_rules": 0,
            "induction_stats": {},
            "stats": {},
            "accepted_rules": [],
            "mean_utility": 0.0,
        }
        abduction_result = (
            0,
            torch.tensor(0.0),
            torch.tensor(0.0),
            0.0,
        )

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True), \
             mock.patch.object(prover, "continuous_hypothesis_cycle", return_value=cycle_result, create=True) as cycle_mock, \
             mock.patch.object(prover, "abduce_and_learn", return_value=abduction_result, create=True) as abduce_mock:
            model.emc.run_episode(
                z,
                torch.tensor([1.0], dtype=torch.float32),
                None,
                prover,
                memory,
                torch.tensor(0.0, dtype=torch.float32),
                device=z.device,
                fast_mode=True,
            )

        cycle_mock.assert_not_called()
        abduce_mock.assert_not_called()
        self.assertEqual(float(prover.last_forward_info.get("emc_symbolic_maintenance_due", 1.0)), 0.0)
        self.assertEqual(float(prover.last_forward_info.get("cycle_cadenced_skip", 0.0)), 1.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_reactive_abduction_cadenced_skip", 0.0)), 1.0)
        self.assertTrue(prover.last_creative_fast_mode)

    def test_train_fast_allows_triggered_abduction_even_when_cadenced(self) -> None:
        cfg = _emc_gap_config()
        cfg.emc_train_fast_maintenance_every = 4
        model = OMENScale(cfg)
        model.emc.train()
        prover = _MockProver(cfg.d_latent)
        prover.continuous_cycle_enabled = True
        prover._step = 1
        prover.task_context = SimpleNamespace(
            target_facts=frozenset(),
            provenance="latent",
            trigger_abduction=True,
        )
        z = torch.ones(1, cfg.d_latent, dtype=torch.float32)
        memory = _MockMemory(torch.zeros_like(z))
        cycle_result = {
            "loss_tensor": torch.tensor(0.0),
            "added_rules": 0,
            "induction_stats": {},
            "stats": {},
            "accepted_rules": [],
            "mean_utility": 0.0,
        }
        abduction_result = (
            0,
            torch.tensor(0.0),
            torch.tensor(0.0),
            0.0,
        )

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True), \
             mock.patch.object(prover, "continuous_hypothesis_cycle", return_value=cycle_result, create=True) as cycle_mock, \
             mock.patch.object(prover, "abduce_and_learn", return_value=abduction_result, create=True) as abduce_mock:
            model.emc.run_episode(
                z,
                torch.tensor([1.0], dtype=torch.float32),
                None,
                prover,
                memory,
                torch.tensor(0.0, dtype=torch.float32),
                device=z.device,
                fast_mode=True,
            )

        cycle_mock.assert_not_called()
        abduce_mock.assert_called_once()
        self.assertEqual(float(prover.last_forward_info.get("emc_symbolic_maintenance_due", 1.0)), 0.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_reactive_abduction_executed", 0.0)), 1.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_reactive_abduction_cadenced_skip", 1.0)), 0.0)
        self.assertTrue(prover.last_creative_fast_mode)

    def test_train_fast_runs_symbolic_maintenance_on_due_step(self) -> None:
        cfg = _emc_gap_config()
        cfg.emc_train_fast_maintenance_every = 4
        cfg.emc_train_fast_cycle_trace_candidates = 1
        cfg.emc_train_fast_cycle_contextual = 2
        cfg.emc_train_fast_cycle_neural = 3
        cfg.emc_train_fast_cycle_max_repairs = 1
        model = OMENScale(cfg)
        model.emc.train()
        prover = _MockProver(cfg.d_latent)
        prover.continuous_cycle_enabled = True
        prover._step = 0
        z = torch.ones(1, cfg.d_latent, dtype=torch.float32)
        memory = _MockMemory(torch.zeros_like(z))
        cycle_result = {
            "loss_tensor": torch.tensor(0.0),
            "added_rules": 0,
            "induction_stats": {},
            "stats": {},
            "accepted_rules": [],
            "mean_utility": 0.0,
        }
        abduction_result = (
            0,
            torch.tensor(0.0),
            torch.tensor(0.0),
            0.0,
        )

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True), \
             mock.patch.object(prover, "continuous_hypothesis_cycle", return_value=cycle_result, create=True) as cycle_mock, \
             mock.patch.object(prover, "abduce_and_learn", return_value=abduction_result, create=True) as abduce_mock:
            model.emc.run_episode(
                z,
                torch.tensor([1.0], dtype=torch.float32),
                None,
                prover,
                memory,
                torch.tensor(0.0, dtype=torch.float32),
                device=z.device,
                fast_mode=True,
            )

        cycle_mock.assert_called_once()
        abduce_mock.assert_called_once()
        cycle_kwargs = cycle_mock.call_args.kwargs
        self.assertEqual(cycle_kwargs.get("max_trace_candidates"), 1)
        self.assertEqual(cycle_kwargs.get("max_contextual"), 2)
        self.assertEqual(cycle_kwargs.get("max_neural"), 3)
        self.assertEqual(cycle_kwargs.get("max_repairs"), 1)
        self.assertEqual(float(prover.last_forward_info.get("emc_symbolic_maintenance_due", 0.0)), 1.0)
        self.assertEqual(float(prover.last_forward_info.get("cycle_executed", 0.0)), 1.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_reactive_abduction_executed", 0.0)), 1.0)
        self.assertTrue(prover.last_creative_fast_mode)

    def test_eval_episode_uses_gap_feedback_for_recall(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        model.emc.eval()
        prover = _MockProver(cfg.d_latent)
        z, z_sim, v_mem = _z_triplet(cfg.d_latent)
        memory = _MockMemory(v_mem)
        gap_feedback = model._make_emc_gap_feedback(z_sim)

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_RECALL, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            z_sym, v_mem_out = model.emc.run_episode_eval(
                z,
                torch.tensor([2.0], dtype=torch.float32),
                None,
                prover,
                memory,
                device=z.device,
                gap_feedback=gap_feedback,
            )

        self.assertEqual(tuple(z_sym.shape), tuple(z.shape))
        self.assertTrue(torch.allclose(v_mem_out, v_mem))
        self.assertGreater(float(prover.last_forward_info.get("emc_recall_gap_delta", 0.0)), 0.0)
        self.assertGreater(float(prover.last_forward_info.get("emc_recall_gap_relief", 0.0)), 0.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_recall_effective_ratio", 0.0)), 1.0)

    def test_eval_episode_measures_fc_gap_from_post_action_state(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        model.emc.eval()
        z, z_sim, _ = _z_triplet(cfg.d_latent)
        prover = _FCGapProver(cfg.d_latent, z_sim)
        memory = _MockMemory(torch.zeros_like(z))
        gap_feedback_base = model._make_emc_gap_feedback(z_sim)
        gap_feedback_calls = {"count": 0}

        def gap_feedback(z_state, signal):
            gap_feedback_calls["count"] += 1
            return gap_feedback_base(z_state, signal)

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_FC, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            model.emc.run_episode_eval(
                z,
                torch.tensor([2.0], dtype=torch.float32),
                None,
                prover,
                memory,
                device=z.device,
                gap_feedback=gap_feedback,
            )

        self.assertGreater(float(prover.last_forward_info.get("emc_gap_delta_mean", 0.0)), 0.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_recall_steps", 0.0)), 0.0)
        self.assertEqual(gap_feedback_calls["count"], 1)

    def test_eval_episode_measures_abduce_gap_from_post_action_state(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        model.emc.eval()
        z, z_sim, _ = _z_triplet(cfg.d_latent)
        prover = _AbduceGapProver(cfg.d_latent, z_sim)
        memory = _MockMemory(torch.zeros_like(z))
        gap_feedback_base = model._make_emc_gap_feedback(z_sim)
        gap_feedback_calls = {"count": 0}

        def gap_feedback(z_state, signal):
            gap_feedback_calls["count"] += 1
            return gap_feedback_base(z_state, signal)

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_ABDUCE, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            model.emc.run_episode_eval(
                z,
                torch.tensor([2.0], dtype=torch.float32),
                None,
                prover,
                memory,
                device=z.device,
                gap_feedback=gap_feedback,
            )

        self.assertGreater(float(prover.last_forward_info.get("emc_gap_delta_mean", 0.0)), 0.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_recall_steps", 0.0)), 0.0)
        self.assertEqual(gap_feedback_calls["count"], 1)

    def test_eval_episode_can_focus_intrinsic_goal(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        model.emc.eval()
        prover = _IntrinsicGoalProver(cfg.d_latent)
        memory = _MockMemory(torch.zeros(1, cfg.d_latent, dtype=torch.float32))
        z = torch.ones(1, cfg.d_latent, dtype=torch.float32)

        with mock.patch.object(model.emc.actor, "forward", side_effect=_scripted_actor([ACTION_INTRINSIC, ACTION_STOP])), \
             mock.patch.object(model.emc.critic, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "forward", return_value=torch.zeros(1, dtype=torch.float32)), \
             mock.patch.object(model.emc.stopping_utility, "bellman_should_stop", return_value=False), \
             mock.patch.object(model.emc.voc, "should_continue", return_value=True):
            model.emc.run_episode_eval(
                z,
                torch.tensor([1.0], dtype=torch.float32),
                None,
                prover,
                memory,
                device=z.device,
            )

        self.assertTrue(prover.focused)
        self.assertEqual(prover.task_context.goal, prover.intrinsic_goal)
        self.assertEqual(float(prover.last_forward_info.get("emc_intrinsic_actions", 0.0)), 1.0)
        self.assertEqual(float(prover.last_forward_info.get("emc_intrinsic_goal_active", 0.0)), 1.0)

    def test_generate_reports_emc_recall_gap_stats(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        model.eval()
        prompt = torch.tensor([[11, 13, 17, 19, 23, 29]], dtype=torch.long)
        z, z_sim, v_mem = _z_triplet(cfg.d_latent)
        latents = torch.zeros(1, cfg.n_latents, cfg.d_latent, dtype=torch.float32)

        def fake_encode_for_saliency(tokens: torch.Tensor):
            h_tok = torch.zeros(tokens.size(0), tokens.size(1), cfg.d_tok, dtype=torch.float32)
            return h_tok, None, h_tok, None

        def fake_world_rollout(*args, **kwargs):
            del args, kwargs
            traj = z_sim.unsqueeze(1).repeat(1, cfg.world_rollout_steps, 1)
            return traj, traj.clone()

        def fake_decoder(tokens: torch.Tensor, z_final: torch.Tensor):
            del z_final
            logits = torch.full((tokens.size(0), tokens.size(1), cfg.vocab_size), -1e4, dtype=torch.float32)
            logits[:, :, 7] = 1e4
            return logits

        def fake_emc_eval(*args, **kwargs):
            del args, kwargs
            model.prover.last_forward_info = {
                "emc_gap_events": 1.0,
                "emc_state_steps": 1.0,
                "emc_recall_steps": 1.0,
                "emc_recall_effective_steps": 1.0,
                "emc_gap_delta_mean": 0.5,
                "emc_state_gap_world": 0.9,
                "emc_state_gap_grounded": 0.4,
                "emc_state_gap_relief": 0.5,
                "emc_state_memory_residual": 0.35,
                "emc_state_memory_alignment": 0.2,
                "emc_state_memory_pressure": 0.07,
                "emc_recall_gap_delta": 0.5,
                "emc_recall_gap_relief": 0.5,
            }
            return torch.zeros(1, cfg.d_latent, dtype=torch.float32), v_mem

        with mock.patch.object(model, "_encode_for_saliency", side_effect=fake_encode_for_saliency), \
             mock.patch.object(model.perceiver, "forward", return_value=(latents, z)), \
             mock.patch.object(model, "_sample_variational_latent", return_value=(z, z, torch.zeros_like(z))), \
             mock.patch.object(model, "_retrieve_memory", return_value=v_mem), \
             mock.patch.object(model, "_world_rollout_from_hidden", side_effect=fake_world_rollout), \
             mock.patch.object(model.emc, "run_episode_eval", side_effect=fake_emc_eval), \
             mock.patch.object(model.tok_decoder, "forward", side_effect=fake_decoder), \
             mock.patch.object(model, "_ast_facts_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_rules_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_trace_from_bytes", return_value=None), \
             mock.patch.object(model, "_ast_lang_from_bytes", return_value=""):
            generated = model.generate(prompt, max_new=1, temperature=1.0, dynamic_reasoning=False)

        self.assertEqual(int(generated[0, -1].item()), 7)
        self.assertEqual(float(model.last_generate_info.get("emc_recall_steps", 0.0)), 1.0)
        self.assertEqual(float(model.last_generate_info.get("emc_recall_effective_steps", 0.0)), 1.0)
        self.assertEqual(float(model.last_generate_info.get("emc_recall_effective_ratio", 0.0)), 1.0)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_gap_delta_mean", 0.0)), 0.5, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_gap_world", 0.0)), 0.9, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_gap_grounded", 0.0)), 0.4, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_gap_relief", 0.0)), 0.5, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_memory_residual", 0.0)), 0.35, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_memory_alignment", 0.0)), 0.2, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_memory_pressure", 0.0)), 0.07, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_recall_gap_delta", 0.0)), 0.5, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_recall_gap_relief", 0.0)), 0.5, places=6)

    def test_generate_reports_non_recall_emc_gap_stats(self) -> None:
        cfg = _emc_gap_config()
        model = OMENScale(cfg)
        model.eval()
        prompt = torch.tensor([[11, 13, 17, 19, 23, 29]], dtype=torch.long)
        z, z_sim, v_mem = _z_triplet(cfg.d_latent)
        latents = torch.zeros(1, cfg.n_latents, cfg.d_latent, dtype=torch.float32)

        def fake_encode_for_saliency(tokens: torch.Tensor):
            h_tok = torch.zeros(tokens.size(0), tokens.size(1), cfg.d_tok, dtype=torch.float32)
            return h_tok, None, h_tok, None

        def fake_world_rollout(*args, **kwargs):
            del args, kwargs
            traj = z_sim.unsqueeze(1).repeat(1, cfg.world_rollout_steps, 1)
            return traj, traj.clone()

        def fake_decoder(tokens: torch.Tensor, z_final: torch.Tensor):
            del z_final
            logits = torch.full((tokens.size(0), tokens.size(1), cfg.vocab_size), -1e4, dtype=torch.float32)
            logits[:, :, 7] = 1e4
            return logits

        def fake_emc_eval(*args, **kwargs):
            del args, kwargs
            model.prover.last_forward_info = {
                "emc_gap_events": 1.0,
                "emc_state_steps": 1.0,
                "emc_recall_steps": 0.0,
                "emc_recall_effective_steps": 0.0,
                "emc_gap_delta_mean": 0.75,
                "emc_state_gap_world": 0.75,
                "emc_state_gap_grounded": 0.3,
                "emc_state_gap_relief": 0.45,
                "emc_state_memory_residual": 0.55,
                "emc_state_memory_alignment": -0.1,
                "emc_state_memory_pressure": 0.09,
                "emc_recall_gap_delta": 0.0,
                "emc_recall_gap_relief": 0.0,
            }
            return torch.zeros(1, cfg.d_latent, dtype=torch.float32), v_mem

        with mock.patch.object(model, "_encode_for_saliency", side_effect=fake_encode_for_saliency), \
             mock.patch.object(model.perceiver, "forward", return_value=(latents, z)), \
             mock.patch.object(model, "_sample_variational_latent", return_value=(z, z, torch.zeros_like(z))), \
             mock.patch.object(model, "_retrieve_memory", return_value=v_mem), \
             mock.patch.object(model, "_world_rollout_from_hidden", side_effect=fake_world_rollout), \
             mock.patch.object(model.emc, "run_episode_eval", side_effect=fake_emc_eval), \
             mock.patch.object(model.tok_decoder, "forward", side_effect=fake_decoder), \
             mock.patch.object(model, "_ast_facts_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_rules_from_bytes", return_value=[]), \
             mock.patch.object(model, "_ast_trace_from_bytes", return_value=None), \
             mock.patch.object(model, "_ast_lang_from_bytes", return_value=""):
            generated = model.generate(prompt, max_new=1, temperature=1.0, dynamic_reasoning=False)

        self.assertEqual(int(generated[0, -1].item()), 7)
        self.assertEqual(float(model.last_generate_info.get("emc_gap_events", 0.0)), 1.0)
        self.assertEqual(float(model.last_generate_info.get("emc_state_steps", 0.0)), 1.0)
        self.assertEqual(float(model.last_generate_info.get("emc_recall_steps", 0.0)), 0.0)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_gap_delta_mean", 0.0)), 0.75, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_gap_world", 0.0)), 0.75, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_gap_grounded", 0.0)), 0.3, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_gap_relief", 0.0)), 0.45, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_memory_residual", 0.0)), 0.55, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_memory_alignment", 0.0)), -0.1, places=6)
        self.assertAlmostEqual(float(model.last_generate_info.get("emc_state_memory_pressure", 0.0)), 0.09, places=6)


if __name__ == "__main__":
    unittest.main()

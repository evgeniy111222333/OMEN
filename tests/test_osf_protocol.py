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

from omen_grounding import (
    PLAN_ACTIVE_RESOURCE_PRED,
    PLAN_CONFLICTED_VERIFICATION_PRED,
    PLAN_HYPOTHESIS_PRED,
    PLAN_LINEAGE_PRED,
    PLAN_PREFER_CONSTRAINT_PRED,
    PLAN_REPAIR_DIRECTIVE_PRED,
    PLAN_SUPPORTED_VERIFICATION_PRED,
    CompiledSymbolicHypothesis,
    GroundingGraphRecord,
    GroundingVerificationRecord,
    PlannerAlternativeWorld,
    PlannerConstraint,
    PlannerOperator as GroundPlannerOperator,
    PlannerRepairDirective,
    PlannerResource,
)
from omen_osf import OSFConfig, OSFSynthesizer
from omen_osf_meta import STRATEGY_CAREFUL, STRATEGY_EXPLORATORY, STRATEGY_FAST
from omen_osf_planner import PlanFact, PlanSequence, SymbolicPlanner


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

    def test_symbolic_planner_builds_grounding_bridge_library(self) -> None:
        planner = SymbolicPlanner(
            d_intent=8,
            d_plan=8,
            n_operators=8,
            max_depth=3,
            beam_width=2,
        )
        planner_state = SimpleNamespace(
            resources=(
                PlannerResource(symbol="fire", statuses=("active",), support=0.91, confidence=0.89, sources=("r1",)),
                PlannerResource(symbol="stone", statuses=("active",), support=0.72, confidence=0.70, sources=("r2",)),
            ),
            operators=(
                GroundPlannerOperator(
                    operator_id="bridge:fire:create:stone",
                    predicate="creates",
                    inputs=("fire",),
                    outputs=("stone",),
                    status="active",
                    support=0.90,
                    conflict=0.04,
                    confidence=0.93,
                    repair_action="accept_to_world_state",
                    provenance=("grounding",),
                ),
            ),
            constraints=(
                PlannerConstraint(
                    constraint_id="constraint:stone",
                    target_id="stone",
                    validator_family="world_model",
                    enforcement="prefer",
                    source_segment=0,
                    symbols=("fire", "creates", "stone"),
                    support=0.91,
                    conflict=0.05,
                    confidence=0.88,
                    priority=0.84,
                    pressure=0.12,
                    rationale="supported",
                ),
            ),
            repair_directives=(
                PlannerRepairDirective(
                    directive_id="directive:stone",
                    target_id="stone",
                    action_type="promote_world_model_supported_claim",
                    source_segment=0,
                    symbols=("fire", "creates", "stone"),
                    validator_families=("world_model",),
                    priority=0.81,
                    pressure=0.24,
                    reason="cross_validator_support",
                ),
            ),
            alternative_worlds=(
                PlannerAlternativeWorld(
                    world_id="hypothetical_world",
                    status="hypothetical",
                    operator_ids=("bridge:hyp",),
                    resource_symbols=("water",),
                    contradiction_symbols=tuple(),
                    pressure=0.4,
                    record_count=1,
                ),
            ),
            destructive_effect_symbols=tuple(),
            persistent_effect_symbols=("fire | creates | stone",),
            branching_pressure=0.4,
            contradiction_pressure=0.1,
        )

        seed = planner._planner_state_seed(planner_state)
        lib = planner._planner_state_operator_library(planner_state, goal_id=7, device=torch.device("cpu"))

        self.assertTrue(any(fact.pred == PLAN_ACTIVE_RESOURCE_PRED for fact in seed))
        self.assertTrue(
            any(fact.pred == PLAN_PREFER_CONSTRAINT_PRED or fact.pred == PLAN_REPAIR_DIRECTIVE_PRED for fact in seed)
        )
        self.assertGreaterEqual(len(lib), 1)
        self.assertEqual(lib[0].source, "grounding_bridge")
        self.assertGreater(lib[0].priority, 0.0)
        self.assertIn(PlanFact(303, 7), lib[0].add_effects)
        self.assertTrue(any(op.op_type == "promote_world_model_supported_claim" for op in lib))

    def test_symbolic_planner_keeps_verification_hypothesis_and_lineage_out_of_planner_bridge(self) -> None:
        planner = SymbolicPlanner(
            d_intent=8,
            d_plan=8,
            n_operators=8,
            max_depth=3,
            beam_width=2,
        )
        planner_state = SimpleNamespace(
            resources=(
                PlannerResource(
                    symbol="dispatcher",
                    statuses=("hypothetical",),
                    support=0.81,
                    confidence=0.77,
                    sources=("r1",),
                ),
            ),
            operators=(
                GroundPlannerOperator(
                    operator_id="bridge:dispatcher:opens:door",
                    predicate="opens",
                    inputs=("dispatcher",),
                    outputs=("door",),
                    status="hypothetical",
                    support=0.74,
                    conflict=0.12,
                    confidence=0.79,
                    repair_action="keep_multiple_hypotheses_alive",
                    provenance=("scene:claim:door_open",),
                ),
            ),
            verification_records=(
                GroundingVerificationRecord(
                    hypothesis_id="door_open",
                    segment_index=0,
                    kind="relation",
                    verification_status="supported",
                    symbols=("dispatcher", "opens", "door"),
                    support=0.93,
                    conflict=0.04,
                    repair_action="accept_to_world_state",
                    provenance=("scene:claim:door_open",),
                ),
                GroundingVerificationRecord(
                    hypothesis_id="door_locked",
                    segment_index=1,
                    kind="state",
                    verification_status="conflicted",
                    symbols=("door", "locked"),
                    support=0.22,
                    conflict=0.86,
                    repair_action="trigger_hidden_cause_abduction",
                    hidden_cause_candidate=True,
                    provenance=("scene:claim:door_locked",),
                ),
            ),
            hypothesis_records=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="door_open",
                    segment_index=0,
                    kind="relation",
                    symbols=("dispatcher", "opens", "door"),
                    confidence=0.78,
                    deferred=True,
                    provenance=("scene:claim:door_open",),
                ),
            ),
            graph_records=(
                GroundingGraphRecord(
                    record_type="relation",
                    record_id="graph:door_open",
                    graph_key="interlingua:relation:dispatcher:opens:door:positive",
                    graph_text="relation dispatcher opens door",
                    graph_terms=("dispatcher", "opens", "door"),
                    graph_family="interlingua_relation",
                    confidence=0.82,
                    source_segment=0,
                ),
            ),
            lineage_symbols=("interlingua_relation", "dispatcher", "door"),
            constraints=tuple(),
            repair_directives=tuple(),
            alternative_worlds=tuple(),
            destructive_effect_symbols=tuple(),
            persistent_effect_symbols=tuple(),
            branching_pressure=0.4,
            contradiction_pressure=0.3,
        )

        seed = planner._planner_state_seed(planner_state)
        lib = planner._planner_state_operator_library(planner_state, goal_id=7, device=torch.device("cpu"))

        self.assertFalse(any(fact.pred == PLAN_SUPPORTED_VERIFICATION_PRED for fact in seed))
        self.assertFalse(any(fact.pred == PLAN_CONFLICTED_VERIFICATION_PRED for fact in seed))
        self.assertFalse(any(fact.pred == PLAN_HYPOTHESIS_PRED for fact in seed))
        self.assertFalse(any(fact.pred == PLAN_LINEAGE_PRED for fact in seed))
        self.assertFalse(any(op.source == "verification_bridge" for op in lib))
        opens_ops = [op for op in lib if op.op_type == "opens"]
        self.assertTrue(opens_ops)
        self.assertFalse(any(fact.pred == PLAN_SUPPORTED_VERIFICATION_PRED for fact in opens_ops[0].preconditions))

    def test_plan_with_strategy_passes_planner_state_to_planner(self) -> None:
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
        planner_state = SimpleNamespace(resources=tuple(), operators=tuple(), alternative_worlds=tuple())
        fake_plan = PlanSequence(
            operators=[],
            embeddings=torch.zeros(1, 8),
            goal_reached=False,
            goal_progress=0.0,
            goal_facts=tuple(),
            plan_loss=torch.tensor(0.0),
        )

        with mock.patch.object(synth.planner, "forward", return_value=fake_plan) as planner_forward:
            plan = synth._plan_with_strategy(
                intent_state,
                STRATEGY_CAREFUL,
                plan_depth_use=2,
                planner_state=planner_state,
            )

        self.assertIs(plan, fake_plan)
        self.assertIs(planner_forward.call_args.kwargs["planner_state"], planner_state)

    def test_repair_candidate_specs_prioritize_directive_specific_strategies(self) -> None:
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
        planner_state = SimpleNamespace(
            repair_directives=(
                PlannerRepairDirective(
                    directive_id="directive:hidden",
                    target_id="door",
                    action_type="trigger_hidden_cause_abduction",
                    source_segment=0,
                    symbols=("dispatcher", "opens", "door"),
                    validator_families=("world_model",),
                    priority=0.82,
                    pressure=0.75,
                    reason="world_model_conflict",
                ),
            )
        )

        specs = synth._repair_candidate_specs(
            STRATEGY_CAREFUL,
            1,
            planner_state=planner_state,
            max_plan_depth=synth.cfg.max_plan_depth,
        )

        self.assertGreaterEqual(len(specs), 1)
        self.assertEqual(specs[0][0], STRATEGY_EXPLORATORY)
        self.assertGreaterEqual(specs[0][1], 2)

    def test_repair_candidate_specs_ignore_verification_and_hypothesis_evidence_without_directives(self) -> None:
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
        planner_state = SimpleNamespace(
            verification_records=(
                GroundingVerificationRecord(
                    hypothesis_id="door_locked",
                    segment_index=0,
                    kind="state",
                    verification_status="conflicted",
                    symbols=("door", "locked"),
                    support=0.22,
                    conflict=0.86,
                    repair_action="trigger_hidden_cause_abduction",
                    hidden_cause_candidate=True,
                ),
            ),
            hypothesis_records=(
                CompiledSymbolicHypothesis(
                    hypothesis_id="door_open",
                    segment_index=0,
                    kind="relation",
                    symbols=("dispatcher", "opens", "door"),
                    confidence=0.66,
                    deferred=True,
                ),
            ),
            repair_directives=tuple(),
        )

        specs = synth._repair_candidate_specs(
            STRATEGY_CAREFUL,
            1,
            planner_state=planner_state,
            max_plan_depth=synth.cfg.max_plan_depth,
        )

        self.assertGreaterEqual(len(specs), 1)
        self.assertEqual(specs[0][0], STRATEGY_CAREFUL)
        self.assertTrue(any(spec[0] == STRATEGY_CAREFUL for spec in specs))

    def test_fast_mode_forces_fast_strategy_for_high_ce_training_steps(self) -> None:
        synth = OSFSynthesizer(
            OSFConfig(
                d_intent=8,
                n_goals=4,
                d_plan=8,
                n_operators=6,
                template_len=4,
                max_plan_depth=3,
                beam_width=2,
                use_simulation=True,
                use_reflection=True,
                use_meta=True,
                fast_ce_threshold=4.5,
            ),
            d_latent=8,
            d_tok=8,
            vocab_size=32,
        )
        synth.train()
        h_tok = torch.randn(1, 6, 8)
        z_final = torch.randn(1, 8)
        tgt = torch.randint(0, 32, (1, 6))

        with mock.patch.object(
            synth.meta_ctrl,
            "select_strategy",
            side_effect=AssertionError("meta controller should be bypassed in forced fast mode"),
        ), mock.patch.object(
            synth,
            "_repair_plan_symbolically",
            side_effect=AssertionError("symbolic repair should be bypassed in forced fast mode"),
        ), mock.patch.object(
            synth.simulator,
            "forward",
            side_effect=AssertionError("simulation should be bypassed in forced fast mode"),
        ):
            logits, osf_out = synth(
                h_tok,
                z_final,
                tgt,
                world_rnn=mock.Mock(),
                ce_loss=5.0,
                fast_mode=True,
            )

        self.assertEqual(tuple(logits.shape), (1, 6, 32))
        self.assertEqual(osf_out["osf_strategy"], STRATEGY_FAST)
        self.assertEqual(float(osf_out["osf_fast_forced"]), 1.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_grounding import (
    CompiledSymbolicHypothesis,
    GroundingGraphRecord,
    GroundingRepairAction,
    GroundingValidationRecord,
    GroundingVerificationRecord,
    build_planner_world_state,
)
from omen_grounding.world_state_writeback import GroundingWorldStateRecord
from omen_prolog import HornAtom, HornClause, SymbolicTaskContext, Var
from omen_symbolic.creative_types import RuleCandidate
from omen_symbolic.execution_trace import GroundingRuntimeArtifacts


class GroundingPlannerStateTest(unittest.TestCase):
    def test_build_planner_world_state_projects_world_status_buckets(self) -> None:
        active_record = GroundingWorldStateRecord(
            record_id="active:wood",
            hypothesis_id="wood",
            record_type="relation",
            world_status="active",
            segment_index=0,
            symbols=("fire", "multiplies", "wood"),
            support=0.91,
            conflict=0.05,
            confidence=0.88,
            repair_action="accept_to_world_state",
        )
        hypothetical_record = GroundingWorldStateRecord(
            record_id="hypothetical:stone",
            hypothesis_id="stone",
            record_type="relation",
            world_status="hypothetical",
            segment_index=1,
            symbols=("fire", "creates", "stone"),
            support=0.54,
            conflict=0.18,
            confidence=0.61,
            repair_action="keep_multiple_hypotheses_alive",
        )
        contradicted_record = GroundingWorldStateRecord(
            record_id="contradicted:tree",
            hypothesis_id="tree",
            record_type="state",
            world_status="contradicted",
            segment_index=2,
            symbols=("tree", "present"),
            support=0.28,
            conflict=0.83,
            confidence=0.49,
            repair_action="preserve_conflict_scope",
        )

        active_fact = HornAtom(501, (1, 2))
        hypothetical_fact = HornAtom(502, (2, 3))
        contradicted_fact = HornAtom(503, (3, 4))
        goal = HornAtom(900, (7, 8))
        validation = GroundingValidationRecord(
            validation_id="validation:world:stone",
            target_id="stone",
            validator_family="world_model",
            validation_status="supported",
            source_segment=1,
            symbols=("fire", "creates", "stone"),
            support=0.88,
            conflict=0.11,
            confidence=0.84,
            rationale="active_match",
        )
        repair = GroundingRepairAction(
            action_id="repair:promote:stone",
            target_id="stone",
            action_type="promote_world_model_supported_claim",
            priority=0.76,
            pressure=0.31,
            reason="cross_validator_support",
            source_segment=1,
        )

        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset({HornAtom(100, (0, 1))}),
            grounding_validation_records=(validation,),
            grounding_repair_actions=(repair,),
            grounding_world_state_records=(active_record, hypothetical_record, contradicted_record),
            grounding_world_state_active_facts=frozenset({active_fact}),
            grounding_world_state_hypothetical_facts=frozenset({hypothetical_fact}),
            grounding_world_state_contradicted_facts=frozenset({contradicted_fact}),
            goal=goal,
            target_facts=frozenset({goal}),
            metadata={
                "grounding_uncertainty": 0.42,
                "grounding_world_state_branching_pressure": 0.55,
                "grounding_world_state_contradiction_pressure": 0.61,
                "grounding_hidden_cause_pressure": 0.40,
            },
        )

        planner_state = build_planner_world_state(ctx)

        self.assertEqual(len(planner_state.active_records), 1)
        self.assertEqual(len(planner_state.hypothetical_records), 1)
        self.assertEqual(len(planner_state.contradicted_records), 1)
        self.assertIn(active_fact, planner_state.symbolic_facts)
        self.assertIn(hypothetical_fact, planner_state.symbolic_facts)
        self.assertNotIn(contradicted_fact, planner_state.symbolic_facts)
        self.assertEqual(planner_state.primary_goal, goal)
        self.assertIn("fire", planner_state.resource_symbols)
        self.assertGreaterEqual(len(planner_state.resources), 3)
        self.assertGreaterEqual(len(planner_state.operators), 2)
        self.assertEqual(planner_state.operators[0].predicate, "multiplies")
        self.assertEqual(planner_state.operators[0].inputs, ("fire",))
        self.assertEqual(planner_state.operators[0].outputs, ("wood",))
        self.assertEqual(len(planner_state.constraints), 1)
        self.assertEqual(planner_state.constraints[0].enforcement, "prefer")
        self.assertEqual(len(planner_state.repair_directives), 1)
        self.assertEqual(planner_state.repair_directives[0].action_type, "promote_world_model_supported_claim")
        self.assertGreaterEqual(len(planner_state.alternative_worlds), 2)
        self.assertGreaterEqual(planner_state.summary()["planner_state_world_rules"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_hypothetical_rules"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_contradictions"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_operators"], 2.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_resource_records"], 3.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_constraints"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_repair_directives"], 1.0)
        self.assertAlmostEqual(planner_state.contradiction_pressure, 0.61, places=6)

    def test_build_planner_world_state_preserves_relation_modifiers_as_operator_inputs(self) -> None:
        conditional_record = GroundingWorldStateRecord(
            record_id="active:door_open",
            hypothesis_id="door_open",
            record_type="relation",
            world_status="active",
            segment_index=0,
            symbols=("dispatcher", "opens", "door", "if:alarm_triggered", "cause:evacuation_active", "time:10_00", "modal:must"),
            support=0.94,
            conflict=0.03,
            confidence=0.92,
            repair_action="accept_to_world_state",
        )
        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset(),
            grounding_world_state_records=(conditional_record,),
            grounding_world_state_active_facts=frozenset(),
            grounding_world_state_hypothetical_facts=frozenset(),
            grounding_world_state_contradicted_facts=frozenset(),
            target_facts=frozenset(),
            metadata={"grounding_uncertainty": 0.1},
        )

        planner_state = build_planner_world_state(ctx)

        self.assertEqual(len(planner_state.operators), 1)
        operator = planner_state.operators[0]
        self.assertEqual(operator.inputs, ("dispatcher", "alarm_triggered", "evacuation_active", "10_00"))
        self.assertEqual(operator.outputs, ("door",))
        self.assertEqual(operator.modality, "must")
        self.assertEqual(operator.conditions, ("alarm_triggered",))
        self.assertEqual(operator.causes, ("evacuation_active",))
        self.assertEqual(operator.temporals, ("10_00",))
        self.assertIn("alarm_triggered", planner_state.resource_symbols)
        self.assertIn("evacuation_active", planner_state.resource_symbols)
        self.assertIn("10_00", planner_state.resource_symbols)
        self.assertGreaterEqual(planner_state.summary()["planner_state_conditional_operators"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_causal_operators"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_temporal_operators"], 1.0)
        self.assertGreaterEqual(planner_state.summary()["planner_state_modal_operators"], 1.0)

    def test_build_planner_world_state_projects_hidden_cause_records_as_hypothetical_operators(self) -> None:
        hidden_cause_record = GroundingWorldStateRecord(
            record_id="hypothetical:hidden_cause:door_open",
            hypothesis_id="hidden_cause:door_open",
            record_type="hidden_cause",
            world_status="hypothetical",
            segment_index=0,
            symbols=(
                "door_5",
                "requires_hidden_cause",
                "missing_opens_with_event_green_card",
                "missing_slot:cause",
                "candidate_agent:external_actor",
                "anchor_predicate:opens_with",
                "anchor_object:green_card",
            ),
            support=0.74,
            conflict=0.18,
            confidence=0.77,
            repair_action="trigger_hidden_cause_abduction",
            provenance=("trigger_hypothesis:rel:door",),
        )
        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset(),
            grounding_world_state_records=(hidden_cause_record,),
            metadata={"grounding_hidden_cause_pressure": 0.65},
        )

        planner_state = build_planner_world_state(ctx)
        summary = planner_state.summary()

        self.assertEqual(len(planner_state.hypothetical_records), 1)
        self.assertIn("door_5", planner_state.resource_symbols)
        self.assertIn("missing_opens_with_event_green_card", planner_state.resource_symbols)
        self.assertEqual(len(planner_state.operators), 1)
        operator = planner_state.operators[0]
        self.assertEqual(operator.predicate, "requires_hidden_cause")
        self.assertEqual(operator.inputs, ("door_5",))
        self.assertEqual(operator.outputs, ("missing_opens_with_event_green_card",))
        self.assertEqual(operator.status, "hypothetical")
        self.assertEqual(summary["planner_state_hidden_cause_records"], 1.0)
        self.assertEqual(summary["planner_state_hidden_cause_operators"], 1.0)
        self.assertGreaterEqual(planner_state.hidden_cause_pressure, 0.65)

    def test_build_planner_world_state_excludes_heuristic_world_state_records_from_planner_surface(self) -> None:
        heuristic_record = GroundingWorldStateRecord(
            record_id="hypothetical:heuristic:stone",
            hypothesis_id="heuristic:stone",
            record_type="relation",
            world_status="hypothetical",
            segment_index=0,
            symbols=("fire", "creates", "stone"),
            support=0.54,
            conflict=0.18,
            confidence=0.61,
            repair_action="require_grounding_confirmation",
            claim_source="fallback_extraction",
            provenance=("scene:claim:heuristic_stone", "heuristic_authority:low"),
        )
        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset(),
            grounding_world_state_records=(heuristic_record,),
            metadata={"grounding_uncertainty": 0.3},
        )

        planner_state = build_planner_world_state(ctx)
        summary = planner_state.summary()

        self.assertEqual(len(planner_state.hypothetical_records), 0)
        self.assertEqual(len(planner_state.proposal_records), 1)
        self.assertEqual(len(planner_state.resources), 0)
        self.assertEqual(len(planner_state.operators), 0)
        self.assertEqual(len(planner_state.alternative_worlds), 0)
        self.assertEqual(summary["planner_state_hypothetical_records"], 0.0)
        self.assertEqual(summary["planner_state_proposal_records"], 1.0)
        self.assertEqual(summary["planner_state_heuristic_world_state_records"], 1.0)

    def test_build_planner_world_state_keeps_artifact_verification_hypotheses_and_graph_out_of_planner_surface(self) -> None:
        verification_supported = GroundingVerificationRecord(
            hypothesis_id="door_open",
            segment_index=0,
            kind="relation",
            verification_status="supported",
            symbols=("dispatcher", "opens", "door", "if:alarm_triggered"),
            support=0.91,
            conflict=0.08,
            repair_action="accept_to_world_state",
            provenance=("compiled:door_open",),
        )
        verification_conflicted = GroundingVerificationRecord(
            hypothesis_id="door_locked",
            segment_index=1,
            kind="state",
            verification_status="conflicted",
            symbols=("door", "locked"),
            support=0.21,
            conflict=0.82,
            repair_action="trigger_hidden_cause_abduction",
            hidden_cause_candidate=True,
            provenance=("compiled:door_locked",),
        )
        hypothesis_open = CompiledSymbolicHypothesis(
            hypothesis_id="door_open",
            segment_index=0,
            kind="relation",
            symbols=("dispatcher", "opens", "door", "if:alarm_triggered"),
            confidence=0.83,
            provenance=("scene:claim:door_open",),
        )
        hypothesis_locked = CompiledSymbolicHypothesis(
            hypothesis_id="door_locked",
            segment_index=1,
            kind="state",
            symbols=("door", "locked"),
            confidence=0.44,
            deferred=True,
            conflict_tag="negative_polarity",
            provenance=("scene:claim:door_locked",),
        )
        graph_relation = GroundingGraphRecord(
            record_type="relation",
            record_id="graph:door_open",
            graph_key="interlingua:relation:dispatcher:opens:door:positive",
            graph_text="relation dispatcher opens door",
            graph_terms=("dispatcher", "opens", "door", "if:alarm_triggered"),
            graph_family="interlingua_relation",
            confidence=0.79,
            source_segment=0,
        )
        graph_entity = GroundingGraphRecord(
            record_type="entity",
            record_id="graph:dispatcher",
            graph_key="interlingua:entity:dispatcher",
            graph_text="entity dispatcher",
            graph_terms=("dispatcher",),
            graph_family="interlingua_entity",
            confidence=0.88,
            source_segment=0,
        )
        artifacts = GroundingRuntimeArtifacts(
            language="en",
            source_text="dispatcher opens door if alarm triggered",
            grounding_hypotheses=(hypothesis_open, hypothesis_locked),
            grounding_verification_records=(verification_supported, verification_conflicted),
            grounding_graph_records=(graph_relation, graph_entity),
        )
        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset(),
            grounding_artifacts=artifacts,
            metadata={
                "grounding_world_state_branching_pressure": 0.45,
                "grounding_world_state_contradiction_pressure": 0.52,
            },
        )

        planner_state = build_planner_world_state(ctx)
        summary = planner_state.summary()

        self.assertEqual(len(planner_state.active_records), 0)
        self.assertEqual(len(planner_state.hypothetical_records), 0)
        self.assertEqual(len(planner_state.contradicted_records), 0)
        self.assertEqual(len(planner_state.verification_records), 2)
        self.assertEqual(len(planner_state.hypothesis_records), 2)
        self.assertEqual(len(planner_state.graph_records), 2)
        self.assertEqual(len(planner_state.resource_symbols), 0)
        self.assertEqual(len(planner_state.resources), 0)
        self.assertFalse(any(operator.predicate == "opens" for operator in planner_state.operators))
        self.assertEqual(len(planner_state.operators), 0)
        self.assertEqual(len(planner_state.alternative_worlds), 0)
        self.assertIn("interlingua_relation", planner_state.lineage_symbols)
        self.assertGreaterEqual(summary["planner_state_verification_records"], 2.0)
        self.assertGreaterEqual(summary["planner_state_conflicted_verification_records"], 1.0)
        self.assertGreaterEqual(summary["planner_state_hidden_cause_candidates"], 1.0)
        self.assertGreaterEqual(summary["planner_state_hypothesis_records"], 2.0)
        self.assertGreaterEqual(summary["planner_state_conflicted_hypotheses"], 1.0)
        self.assertGreaterEqual(summary["planner_state_graph_records"], 2.0)
        self.assertGreaterEqual(summary["planner_state_graph_relation_records"], 1.0)
        self.assertGreaterEqual(summary["planner_state_lineage_symbols"], 3.0)
        self.assertEqual(summary["planner_state_hypothetical_operators"], 0.0)
        self.assertEqual(summary["planner_state_contradictions"], 0.0)

    def test_build_planner_world_state_keeps_grounding_candidate_rules_out_of_planner_surface(self) -> None:
        candidate_rule = RuleCandidate(
            clause=HornClause(
                head=HornAtom(901, (Var("X"), Var("Y"))),
                body=(
                    HornAtom(902, (Var("X"),)),
                    HornAtom(903, (Var("Y"),)),
                ),
            ),
            source="grounding_rule_compiler",
            score=0.88,
            utility=0.81,
            metadata={
                "hypothesis_id": "rule:stars",
                "semantic_mode": "rule",
                "quantifier_mode": "generic_all",
                "epistemic_status": "asserted",
                "claim_source": "speaker_turn",
                "subject_name": "stars",
                "predicate_name": "generates",
                "object_name": "planets",
                "relation_modifiers": ("if:dust_cloud", "modal:must"),
                "support_set": ("semantic:rule",),
                "provenance": ("scene:claim:stars_rule",),
            },
        )
        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset(),
            grounding_candidate_rules=(candidate_rule,),
            metadata={"grounding_uncertainty": 0.18},
        )

        planner_state = build_planner_world_state(ctx)
        summary = planner_state.summary()

        self.assertEqual(len(planner_state.candidate_rules), 1)
        self.assertIn(
            "stars | generates | planets | if:dust_cloud | modal:must",
            planner_state.candidate_rule_symbols,
        )
        self.assertIn("grounding_rule_compiler", planner_state.lineage_symbols)
        self.assertIn("semantic_mode:rule", planner_state.lineage_symbols)
        self.assertIn("claim_source:speaker_turn", planner_state.lineage_symbols)
        self.assertFalse(any(operator.predicate == "generates" for operator in planner_state.operators))
        self.assertEqual(len(planner_state.resources), 0)
        self.assertEqual(len(planner_state.operators), 0)
        self.assertEqual(len(planner_state.alternative_worlds), 0)
        self.assertEqual(summary["planner_state_hypothetical_operators"], 0.0)
        self.assertGreaterEqual(summary["planner_state_grounding_candidate_rules"], 1.0)
        self.assertGreaterEqual(summary["planner_state_grounding_candidate_rule_records"], 1.0)
        self.assertGreaterEqual(summary["planner_state_candidate_rule_symbols"], 1.0)
        self.assertEqual(summary["planner_state_hypothetical_records"], 0.0)

    def test_build_planner_world_state_ignores_heuristic_grounding_candidate_rules(self) -> None:
        candidate_rule = RuleCandidate(
            clause=HornClause(
                head=HornAtom(911, (Var("X"), Var("Y"))),
                body=(
                    HornAtom(912, (Var("X"),)),
                    HornAtom(913, (Var("Y"),)),
                ),
            ),
            source="grounding_rule_compiler",
            score=0.88,
            utility=0.81,
            metadata={
                "hypothesis_id": "rule:heuristic",
                "semantic_mode": "rule",
                "quantifier_mode": "generic_all",
                "epistemic_status": "asserted",
                "claim_source": "fallback_extraction",
                "subject_name": "stars",
                "predicate_name": "generates",
                "object_name": "planets",
                "relation_modifiers": ("if:dust_cloud",),
                "support_set": ("semantic:rule", "heuristic_authority:low"),
                "provenance": ("scene:claim:heuristic_rule", "heuristic_authority:low"),
            },
        )
        ctx = SymbolicTaskContext(
            observed_now_facts=frozenset(),
            grounding_candidate_rules=(candidate_rule,),
            metadata={"grounding_uncertainty": 0.18},
        )

        planner_state = build_planner_world_state(ctx)
        summary = planner_state.summary()

        self.assertEqual(len(planner_state.candidate_rules), 0)
        self.assertEqual(len(planner_state.candidate_rule_symbols), 0)
        self.assertFalse(any(operator.predicate == "generates" for operator in planner_state.operators))
        self.assertEqual(summary["planner_state_grounding_candidate_rules"], 0.0)
        self.assertEqual(summary["planner_state_grounding_candidate_rule_records"], 0.0)
        self.assertEqual(summary["planner_state_candidate_rule_symbols"], 0.0)
        self.assertEqual(summary["planner_state_heuristic_candidate_rules"], 0.0)


if __name__ == "__main__":
    unittest.main()

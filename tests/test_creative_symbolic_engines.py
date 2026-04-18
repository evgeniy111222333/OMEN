from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_prolog import (
    Const,
    DifferentiableProver,
    EpistemicStatus,
    HornAtom,
    HornClause,
    KnowledgeBase,
    SymbolicTaskContext,
    Var,
    _atoms_conflict,
)
from omen_symbolic.aesthetic_engine import AestheticEvolutionEngine
from omen_symbolic.analogy_engine import AnalogyMetaphorEngine
from omen_symbolic.counterfactual_engine import CounterfactualWorldEngine
from omen_symbolic.creative_types import CreativeCycleReport, IntrinsicGoal, RuleCandidate
from omen_symbolic.intrinsic_engine import IntrinsicCuriosityEngine
from omen_symbolic.ontology_engine import OntologyExpansionEngine, RuleHypothesisSampler
from omen_symbolic.rule_graph import build_predicate_graph_view


def atom(pred: int, *args) -> HornAtom:
    return HornAtom(pred=pred, args=tuple(args))


def rule(head: HornAtom, *body: HornAtom) -> HornClause:
    return HornClause(head=head, body=tuple(body))


class CreativeSymbolicEnginesTest(unittest.TestCase):
    def test_aee_symmetry_prefers_argument_permutation_invariant_rule(self) -> None:
        x, y = Var("X"), Var("Y")
        engine = AestheticEvolutionEngine()
        symmetric = rule(atom(1, x, y), atom(1, y, x))
        asymmetric = rule(atom(2, x, y), atom(3, x, y))
        self.assertGreater(
            engine._symmetry_score(symmetric),
            engine._symmetry_score(asymmetric),
        )

    def test_ame_uses_deterministic_spectral_fallback_without_training(self) -> None:
        x, y, z = Var("X"), Var("Y"), Var("Z")
        rules = [
            rule(atom(1, x, y, z), atom(1, y, x, z)),
            rule(atom(2, x, y, z), atom(202, x, y, z)),
        ]
        engine = AnalogyMetaphorEngine(
            embedding_dim=12,
            tau_analogy=0.10,
            contrastive_steps=0,
        )
        state = engine.fit(rules)
        self.assertEqual(state.graph_view.embedding_source, "hypergraph_spectral")

    def test_ame_transfers_structural_template(self) -> None:
        x, y, z = Var("X"), Var("Y"), Var("Z")
        rules = [
            rule(atom(1, x, y, z), atom(1, y, x, z)),
            rule(atom(101, x, z), atom(1, x, y, z)),
            rule(atom(2, x, y, z), atom(202, x, y, z)),
            rule(atom(102, x, z), atom(2, x, y, z)),
        ]
        engine = AnalogyMetaphorEngine(
            embedding_dim=12,
            tau_analogy=0.10,
            contrastive_steps=0,
            max_pairs=4,
        )
        engine.fit(rules)
        candidates = engine.generate_candidates(rules)
        self.assertTrue(
            any(candidate.clause.head.pred == 2 and any(atom.pred == 2 for atom in candidate.clause.body) for candidate in candidates)
        )

    def test_ame_generates_metaphor_bridge_with_projection(self) -> None:
        x, y, z = Var("X"), Var("Y"), Var("Z")
        rules = [
            rule(atom(1, x, y, z), atom(101, x, z)),
            rule(atom(2, x, y), atom(202, x)),
        ]
        engine = AnalogyMetaphorEngine(
            embedding_dim=12,
            tau_analogy=0.10,
            tau_metaphor=0.0,
            contrastive_steps=0,
            max_pairs=4,
        )
        engine.fit(rules)
        candidates = engine.generate_metaphor_candidates(rules)
        self.assertTrue(candidates)
        self.assertTrue(any(candidate.source == "metaphor" for candidate in candidates))
        self.assertTrue(
            any(
                candidate.clause.head.pred == 2
                and len(candidate.clause.body) == 1
                and candidate.clause.body[0].pred == 1
                for candidate in candidates
            )
        )

    def test_cwe_surfaces_contradiction(self) -> None:
        kb = KnowledgeBase(max_rules=8)
        kb.add_fact(atom(11, Const(0)))
        kb.add_rule(rule(atom(10, Var("X"), Const(1)), atom(11, Var("X"))))
        engine = CounterfactualWorldEngine(max_rule_mods=2, max_candidates=4, surprise_lambda=1.0)
        result = engine.explore(kb, list(kb.facts), max_depth=2, conflict_fn=_atoms_conflict)
        self.assertGreater(result.surprise, 0.0)
        self.assertTrue(result.contradictions)

    def test_cwe_exact_search_can_select_pair_only_when_joint_gain_exists(self) -> None:
        kb = KnowledgeBase(max_rules=16)
        kb.add_fact(atom(20, Const(1), Const(2)))
        kb.add_fact(atom(21, Const(2), Const(3)))
        kb.add_rule(rule(atom(30, Var("X"), Var("Y")), atom(20, Var("X"), Var("Y"))))
        kb.add_rule(rule(atom(31, Var("X"), Var("Y")), atom(21, Var("X"), Var("Y"))))
        engine = CounterfactualWorldEngine(
            max_rule_mods=2,
            max_candidates=4,
            surprise_lambda=1.0,
            max_transforms_per_rule=1,
        )

        def _joint_only_world_surprise(modified_rules, novel_facts, sandbox):
            return 100.0 if len(modified_rules) == 2 else 0.0

        result = engine.explore(
            kb,
            list(kb.facts),
            max_depth=2,
            conflict_fn=lambda left, right: False,
            world_surprise_fn=_joint_only_world_surprise,
        )
        self.assertEqual(len(result.modified_rules), 2)
        self.assertEqual(result.metadata.get("exact_search", 0.0), 1.0)
        self.assertGreaterEqual(result.metadata.get("evaluated_subsets", 0.0), 3.0)

    def test_cwe_reports_exact_search_metadata(self) -> None:
        kb = KnowledgeBase(max_rules=8)
        kb.add_fact(atom(40, Const(1), Const(2)))
        kb.add_rule(rule(atom(41, Var("X"), Var("Y")), atom(40, Var("X"), Var("Y"))))
        engine = CounterfactualWorldEngine(
            max_rule_mods=2,
            max_candidates=4,
            surprise_lambda=0.1,
            max_transforms_per_rule=1,
        )
        result = engine.explore(
            kb,
            list(kb.facts),
            max_depth=2,
            conflict_fn=lambda left, right: False,
            world_surprise_fn=lambda modified_rules, novel_facts, sandbox: 0.0,
        )
        self.assertTrue(result.modified_rules)
        self.assertEqual(result.metadata.get("exact_search", 0.0), 1.0)
        self.assertGreaterEqual(result.metadata.get("evaluated_subsets", 0.0), 1.0)

    def test_oee_invents_predicate_candidates(self) -> None:
        engine = OntologyExpansionEngine(gap_threshold=0.2, contradiction_threshold=1)
        current_facts = [atom(50, Const(1), Const(2))]
        goal = atom(60, Var("A"), Var("B"))
        candidates = engine.generate_candidates(
            current_facts=current_facts,
            goal=goal,
            gap_norm=0.9,
            contradiction_count=2,
        )
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.source == "ontology" for candidate in candidates))
        self.assertTrue(any(candidate.metadata.get("invented_predicate", 0.0) >= 900000 for candidate in candidates))

    def test_oee_second_order_sampler_emits_generalized_templates(self) -> None:
        kb = KnowledgeBase(max_rules=16)
        kb.add_fact(atom(10, Const(1), Const(2)))
        kb.add_fact(atom(11, Const(2), Const(3)))
        sampler = RuleHypothesisSampler(max_hypotheses=16)
        hypotheses = sampler.sample(
            pred_id=900001,
            arity=2,
            interaction_preds=[10, 11],
            interaction_scores=[0.8, 0.7],
            current_facts=[atom(10, Const(1), Const(2)), atom(11, Const(2), Const(3))],
            kb=kb,
            goal=atom(12, Var("A"), Var("B")),
        )
        templates = {hyp.template for hyp in hypotheses}
        self.assertTrue({"pair_factor", "goal_factorization", "inverse_alias"} & templates)

    def test_oee_second_order_sampler_keeps_rules_safe(self) -> None:
        kb = KnowledgeBase(max_rules=16)
        kb.add_fact(atom(10, Const(1), Const(2)))
        kb.add_fact(atom(11, Const(2), Const(3)))
        sampler = RuleHypothesisSampler(max_hypotheses=24)
        hypotheses = sampler.sample(
            pred_id=900002,
            arity=2,
            interaction_preds=[10, 11],
            interaction_scores=[0.9, 0.8],
            current_facts=[atom(10, Const(1), Const(2)), atom(11, Const(2), Const(3))],
            kb=kb,
            goal=atom(12, Var("A"), Var("B")),
            target_facts=[atom(13, Var("A"), Var("B"))],
        )
        self.assertTrue(hypotheses)
        for hyp in hypotheses:
            head_vars = set(hyp.clause.head.vars())
            body_vars = set()
            for body_atom in hyp.clause.body:
                body_vars |= set(body_atom.vars())
            self.assertTrue(head_vars.issubset(body_vars), repr(hyp.clause))

    def test_oee_second_order_sampler_scores_rule_bundles(self) -> None:
        kb = KnowledgeBase(max_rules=16)
        kb.add_fact(atom(10, Const(1), Const(2)))
        kb.add_fact(atom(11, Const(2), Const(3)))
        sampler = RuleHypothesisSampler(max_hypotheses=24)
        hypotheses = sampler.sample(
            pred_id=900003,
            arity=2,
            interaction_preds=[10, 11],
            interaction_scores=[0.9, 0.8],
            current_facts=[atom(10, Const(1), Const(2)), atom(11, Const(2), Const(3))],
            kb=kb,
            goal=atom(12, Var("A"), Var("B")),
            target_facts=[atom(12, Var("A"), Var("B")), atom(13, Var("A"), Var("B"))],
        )
        self.assertTrue(any(hyp.bundle_size >= 2 for hyp in hypotheses))
        self.assertTrue(any(hyp.bundle_score >= hyp.score for hyp in hypotheses if hyp.bundle_size >= 2))

    def test_oee_second_order_sampler_can_bridge_paradox_targets(self) -> None:
        kb = KnowledgeBase(max_rules=16)
        kb.add_fact(atom(10, Const(1), Const(2)))
        sampler = RuleHypothesisSampler(max_hypotheses=16)
        hypotheses = sampler.sample(
            pred_id=900004,
            arity=2,
            interaction_preds=[10],
            interaction_scores=[0.9],
            current_facts=[atom(10, Const(1), Const(2))],
            kb=kb,
            goal=None,
            target_facts=[atom(70, Var("A"), Var("B"))],
        )
        templates = {hyp.template for hyp in hypotheses}
        self.assertTrue({"bridge_target", "target_factorization"} & templates)

    def test_oee_open_second_order_search_can_discover_non_prefix_projection(self) -> None:
        kb = KnowledgeBase(max_rules=16)
        kb.add_fact(atom(10, Const(1), Const(2)))
        kb.add_fact(atom(11, Const(2), Const(3)))
        pred_id = 900005
        sampler = RuleHypothesisSampler(max_hypotheses=64)
        hypotheses = sampler.sample(
            pred_id=pred_id,
            arity=1,
            interaction_preds=[10, 11],
            interaction_scores=[0.9, 0.8],
            current_facts=[atom(10, Const(1), Const(2)), atom(11, Const(2), Const(3))],
            kb=kb,
            goal=None,
            target_facts=[atom(pred_id, Var("Q"))],
        )
        open_hypotheses = [hyp for hyp in hypotheses if hyp.template.startswith("open_invented_")]
        self.assertTrue(open_hypotheses)
        self.assertTrue(
            any(
                len(hyp.clause.body) == 2
                and repr(hyp.clause.head.args[0]) == repr(hyp.clause.body[0].args[1])
                and repr(hyp.clause.head.args[0]) == repr(hyp.clause.body[1].args[0])
                for hyp in open_hypotheses
            )
        )

    def test_oee_beam_search_can_score_multi_target_rule_bundle(self) -> None:
        kb = KnowledgeBase(max_rules=16)
        kb.add_fact(atom(10, Const(1), Const(2)))
        kb.add_fact(atom(11, Const(2), Const(3)))
        sampler = RuleHypothesisSampler(max_hypotheses=64)
        hypotheses = sampler.sample(
            pred_id=900006,
            arity=2,
            interaction_preds=[10, 11],
            interaction_scores=[0.9, 0.8],
            current_facts=[atom(10, Const(1), Const(2)), atom(11, Const(2), Const(3))],
            kb=kb,
            goal=None,
            target_facts=[atom(70, Var("A"), Var("B")), atom(71, Var("A"), Var("B"))],
        )
        self.assertTrue(any(hyp.bundle_size >= 3 for hyp in hypotheses))
        self.assertGreaterEqual(sampler._last_beam_best_size, 3)
        self.assertGreater(sampler._last_beam_states, 0)

    def test_ice_formulates_goal(self) -> None:
        x, y, z = Var("X"), Var("Y"), Var("Z")
        view = build_predicate_graph_view(
            [
                rule(atom(1, x, y, z), atom(2, x, y)),
                rule(atom(3, x), atom(1, x, y, z)),
            ],
            embedding_dim=8,
        )
        engine = IntrinsicCuriosityEngine(state_history=8, goal_threshold=0.1)
        engine.update_state(torch.randn(1, 8))
        intrinsic_goal = engine.formulate_goal(
            z=torch.randn(1, 8),
            gap_norm=0.8,
            graph_view=view,
            candidate_goals=[atom(3, Var("Q"))],
            candidate_rules=[],
            provenance="test",
        )
        self.assertIsNotNone(intrinsic_goal)
        self.assertGreater(intrinsic_goal.value, 0.0)

    def test_ice_keeps_goal_queue_when_new_signal_is_weak(self) -> None:
        engine = IntrinsicCuriosityEngine(state_history=8, goal_threshold=0.9)
        queued_goal = IntrinsicGoal(
            goal=atom(77, Var("Q")),
            value=0.6,
            kind="explore_structure",
            provenance="test",
        )
        engine._schedule_goal(queued_goal)
        weak_view = build_predicate_graph_view([], embedding_dim=8)
        returned = engine.formulate_goal(
            z=None,
            gap_norm=0.0,
            graph_view=weak_view,
            candidate_goals=[],
            candidate_rules=[],
            provenance="test",
        )
        self.assertIsNotNone(returned)
        self.assertEqual(hash(returned.goal), hash(queued_goal.goal))
        self.assertTrue(engine.scheduled_goals())

    def test_ice_resolves_supported_goals_from_queue(self) -> None:
        engine = IntrinsicCuriosityEngine(state_history=8, goal_threshold=0.1)
        goal_a = IntrinsicGoal(goal=atom(77, Const(1)), value=0.9, kind="a", provenance="test")
        goal_b = IntrinsicGoal(goal=atom(88, Const(2)), value=0.8, kind="b", provenance="test")
        engine._schedule_goal(goal_a)
        engine._schedule_goal(goal_b)
        engine.resolve_supported_goals([atom(77, Const(1))])
        queued = engine.scheduled_goals()
        self.assertEqual(len(queued), 1)
        self.assertEqual(hash(queued[0].goal), hash(goal_b.goal))

    def test_oee_fixates_predicate_only_after_gap_reduction(self) -> None:
        engine = OntologyExpansionEngine(gap_threshold=0.2, contradiction_threshold=1)
        goal = atom(60, Var("A"), Var("B"))
        candidates = engine.generate_candidates(
            current_facts=[atom(50, Const(1), Const(2))],
            goal=goal,
            gap_norm=0.9,
            contradiction_count=2,
        )
        pred_ids = {
            int(candidate.metadata.get("invented_predicate", 0.0))
            for candidate in candidates
            if int(candidate.metadata.get("invented_predicate", 0.0)) > 0
        }
        self.assertTrue(pred_ids)
        engine.record_feedback(
            accepted=True,
            accepted_pred_ids=tuple(pred_ids),
            gap_before=0.9,
            gap_after=0.2,
            supporting_rules=[candidates[0].clause],
        )
        fixed = engine.fixed_entries(tuple(pred_ids))
        self.assertTrue(fixed)
        self.assertTrue(all(entry.status == "fixed" for entry in fixed))

    def test_oee_cluster_pressure_can_trigger_generation(self) -> None:
        engine = OntologyExpansionEngine(gap_threshold=0.5, contradiction_threshold=2)
        engine.observe_gap_cluster(0.9, hot_dims=(), z=torch.tensor([[0.1, 0.8, 0.2, 0.7]]))
        candidates = engine.generate_candidates(
            current_facts=[atom(50, Const(1), Const(2))],
            goal=atom(60, Var("A"), Var("B")),
            gap_norm=0.1,
            contradiction_count=0,
        )
        self.assertTrue(candidates)

    def test_oee_online_training_updates_latent_model_and_stats(self) -> None:
        torch.manual_seed(0)
        engine = OntologyExpansionEngine(
            gap_threshold=0.2,
            contradiction_threshold=1,
            train_every_n_calls=1,
        )
        kb = KnowledgeBase(max_rules=16)
        facts = [
            atom(50, Const(1), Const(2)),
            atom(51, Const(2), Const(3)),
        ]
        for fact in facts:
            kb.add_fact(fact)
        goal = atom(60, Const(1), Const(3))
        predicate_embeddings = {
            50: torch.randn(8),
            51: torch.randn(8),
            60: torch.randn(8),
        }

        for step in range(4):
            candidates = engine.generate_candidates(
                current_facts=facts,
                goal=goal,
                gap_norm=0.9,
                contradiction_count=2,
                max_candidates=2,
                target_facts=[goal],
                predicate_embeddings=predicate_embeddings,
                kb=kb,
                existing_rules=[],
            )
            self.assertTrue(candidates)
            pred_ids = tuple(
                int(candidate.metadata.get("invented_predicate", 0.0))
                for candidate in candidates
                if int(candidate.metadata.get("invented_predicate", 0.0)) > 0
            )
            self.assertTrue(pred_ids)
            engine.record_feedback(
                accepted=(step % 2 == 0),
                accepted_pred_ids=pred_ids,
                gap_before=0.9,
                gap_after=0.2 if step % 2 == 0 else 0.9,
                supporting_rules=[candidates[0].clause],
            )

        self.assertIsNotNone(engine._latent_model)
        before = next(engine._latent_model.parameters()).detach().clone()
        _ = engine.generate_candidates(
            current_facts=facts,
            goal=goal,
            gap_norm=0.9,
            contradiction_count=2,
            max_candidates=2,
            target_facts=[goal],
            predicate_embeddings=predicate_embeddings,
            kb=kb,
            existing_rules=[],
        )
        after = next(engine._latent_model.parameters()).detach()
        stats = engine.stats()

        self.assertEqual(stats["oee_model_initialized"], 1.0)
        self.assertEqual(stats["oee_online_train_applied"], 1.0)
        self.assertGreater(stats["oee_online_train_loss"], 0.0)
        self.assertGreaterEqual(stats["oee_online_train_steps"], 1.0)
        self.assertFalse(torch.allclose(before, after))

    def test_counterfactual_patterns_can_be_routed_into_ame(self) -> None:
        x, y, z = Var("X"), Var("Y"), Var("Z")
        coordinator = DifferentiableProver(
            d_latent=8,
            sym_vocab=32,
            max_rules=16,
            max_depth=2,
            n_cands=2,
        ).creative_cycle
        coordinator.analogy_engine = AnalogyMetaphorEngine(
            embedding_dim=12,
            tau_analogy=0.10,
            contrastive_steps=0,
            max_pairs=4,
        )
        base_rules = [
            rule(atom(1, x, y, z), atom(1, y, x, z)),
            rule(atom(2, x, y, z), atom(202, x, y, z)),
            rule(atom(102, x, z), atom(2, x, y, z)),
        ]
        modified_rules = [
            rule(atom(101, x, z), atom(1, x, y, z)),
        ]
        routed = coordinator._counterfactual_analogy_candidates(base_rules, modified_rules)
        self.assertTrue(routed)
        self.assertTrue(all(candidate.source == "counterfactual_analogy" for candidate in routed))

    def test_counterfactual_patterns_can_be_routed_into_metaphor_candidates(self) -> None:
        x, y, z = Var("X"), Var("Y"), Var("Z")
        coordinator = DifferentiableProver(
            d_latent=8,
            sym_vocab=32,
            max_rules=16,
            max_depth=2,
            n_cands=2,
        ).creative_cycle
        coordinator.analogy_engine = AnalogyMetaphorEngine(
            embedding_dim=12,
            tau_analogy=0.10,
            tau_metaphor=0.0,
            contrastive_steps=0,
            max_pairs=4,
        )
        base_rules = [
            rule(atom(1, x, y, z), atom(101, x, z)),
            rule(atom(2, x, y), atom(202, x)),
        ]
        modified_rules = [
            rule(atom(3, x, y, z), atom(1, x, y, z)),
        ]
        routed = coordinator._counterfactual_metaphor_candidates(base_rules, modified_rules)
        self.assertTrue(routed)
        self.assertTrue(all(candidate.source == "counterfactual_metaphor" for candidate in routed))

    def test_counterfactual_world_surprise_uses_entropy_signal(self) -> None:
        device = torch.device("cpu")

        class _DummyWorld(torch.nn.Module):
            def __init__(self, d_latent: int, vocab: int) -> None:
                super().__init__()
                self.act_emb = torch.nn.Embedding(vocab, d_latent)

            def forward(self, z_state, action, h=None):
                return torch.tanh(z_state + self.act_emb(action)), h

            def simulate_sequence(self, z0, actions, teacher_forcing_ratio: float = 0.0, teacher_states=None):
                embeds = self.act_emb(actions)
                return torch.tanh(z0.unsqueeze(1) + embeds)

        prover = DifferentiableProver(
            d_latent=8,
            sym_vocab=32,
            max_rules=16,
            max_depth=2,
            n_cands=2,
        ).to(device)
        prover.set_world_rnn(_DummyWorld(8, 64).to(device))
        prover._last_z = torch.randn(1, 8, device=device)
        scorer = prover.creative_cycle._world_counterfactual_surprise(
            prover,
            [atom(5, Const(1))],
        )
        self.assertIsNotNone(scorer)
        candidate_rule = rule(atom(7, Const(1), Const(2)))
        score_single = scorer([candidate_rule], (atom(20, Const(1)),), None)
        score_multi = scorer([candidate_rule], (atom(20, Const(1)), atom(21, Const(2))), None)
        self.assertGreater(score_multi, score_single)

    def test_prover_can_use_intrinsic_goal_without_task_context(self) -> None:
        prover = DifferentiableProver(
            d_latent=8,
            sym_vocab=32,
            max_rules=16,
            max_depth=2,
            n_cands=2,
        )
        intrinsic_goal = atom(77, Const(1))
        prover.creative_cycle.intrinsic_engine._schedule_goal(
            IntrinsicGoal(
                goal=intrinsic_goal,
                value=0.9,
                kind="prove_or_disprove_hypothesis",
                provenance="test",
            )
        )
        self.assertEqual(hash(prover.current_goal()), hash(intrinsic_goal))
        self.assertEqual(prover._task_target_facts(), frozenset({intrinsic_goal}))

    def test_prover_forward_reports_creative_metrics(self) -> None:
        device = torch.device("cpu")
        prover = DifferentiableProver(
            d_latent=16,
            sym_vocab=64,
            max_rules=64,
            max_depth=2,
            n_cands=2,
        ).to(device)
        prover.eval()
        prover.configure_creative_cycle(
            enabled=True,
            cycle_every=1,
            max_selected_rules=2,
            tau_analogy=0.10,
            tau_metaphor=0.0,
            analogy_contrastive_steps=0,
            aee_generations=1,
            aee_population=6,
        )
        x, y, z = Var("X"), Var("Y"), Var("Z")
        bootstrap_rules = [
            rule(atom(1, x, y, z), atom(1, y, x, z)),
            rule(atom(101, x, z), atom(1, x, y, z)),
            rule(atom(2, x, y, z), atom(202, x, y, z)),
            rule(atom(102, x, z), atom(2, x, y, z)),
        ]
        for clause in bootstrap_rules:
            prover.kb.add_rule(clause, status=EpistemicStatus.verified)
        prover.last_abduced_rules = [bootstrap_rules[0]]
        observed = frozenset(
            {
                atom(2, Const(1), Const(2), Const(3)),
                atom(202, Const(1), Const(2), Const(3)),
            }
        )
        prover.creative_cycle.intrinsic_engine._schedule_goal(
            IntrinsicGoal(
                goal=atom(303, Const(9)),
                value=0.8,
                kind="explore_structure",
                provenance="test",
            )
        )
        goal = atom(102, Const(1), Const(3))
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=observed,
                goal=goal,
                target_facts=frozenset({goal}),
                provenance="test",
                metadata={"gap_norm": 0.9},
            )
        )
        z_in = torch.randn(1, 16, device=device)
        z_sym, sym_loss = prover(z_in, torch.tensor(0.0, device=device))
        self.assertEqual(tuple(z_sym.shape), (1, 16))
        self.assertTrue(torch.isfinite(sym_loss).item())
        self.assertIn("creative_abduction_candidates", prover.last_forward_info)
        self.assertGreaterEqual(prover.last_forward_info["creative_abduction_candidates"], 1.0)
        self.assertIn("creative_analogy_candidates", prover.last_forward_info)
        self.assertGreaterEqual(prover.last_forward_info["creative_analogy_candidates"], 1.0)
        self.assertIn("creative_metaphor_candidates", prover.last_forward_info)
        self.assertIn("creative_counterfactual_analogy_candidates", prover.last_forward_info)
        self.assertIn("creative_counterfactual_metaphor_candidates", prover.last_forward_info)
        self.assertIn("creative_ontology_fixed_predicates", prover.last_forward_info)
        self.assertIn("creative_cycle_active", prover.last_forward_info)
        self.assertIn("creative_intrinsic_value", prover.last_forward_info)
        self.assertIn("creative_validated_selected_rules", prover.last_forward_info)
        self.assertIn("creative_validated_support_facts", prover.last_forward_info)
        self.assertIn("creative_target_support_after", prover.last_forward_info)
        self.assertIn("creative_gap_reduction", prover.last_forward_info)
        self.assertIn("creative_compression_gain", prover.last_forward_info)
        self.assertIn("background_intrinsic_goals", prover.last_forward_info)
        self.assertGreaterEqual(prover.last_forward_info["background_intrinsic_goals"], 1.0)

    def test_goal_less_context_can_be_enriched_by_intrinsic_goal(self) -> None:
        prover = DifferentiableProver(
            d_latent=8,
            sym_vocab=32,
            max_rules=16,
            max_depth=2,
            n_cands=2,
        )
        intrinsic_goal = atom(77, Var("Q"))
        prover.creative_cycle.intrinsic_engine.pending_goal = IntrinsicGoal(
            goal=intrinsic_goal,
            value=0.9,
            kind="explore_structure",
            provenance="test",
        )
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=frozenset(),
                goal=None,
                target_facts=frozenset(),
                provenance="test",
            )
        )
        self.assertIsNotNone(prover.task_context)
        self.assertEqual(prover.task_context.goal, intrinsic_goal)
        self.assertIn(intrinsic_goal, prover.task_context.target_facts)
        self.assertTrue(prover.task_context.trigger_abduction)

    def test_existing_goal_context_keeps_external_goal_and_adds_intrinsic_target(self) -> None:
        prover = DifferentiableProver(
            d_latent=8,
            sym_vocab=32,
            max_rules=16,
            max_depth=2,
            n_cands=2,
        )
        intrinsic_goal = atom(77, Var("Q"))
        external_goal = atom(88, Const(1))
        prover.creative_cycle.intrinsic_engine.pending_goal = IntrinsicGoal(
            goal=intrinsic_goal,
            value=0.9,
            kind="prove_or_disprove_hypothesis",
            provenance="test",
        )
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=frozenset(),
                goal=external_goal,
                target_facts=frozenset({external_goal}),
                provenance="test",
            )
        )
        self.assertIsNotNone(prover.task_context)
        self.assertEqual(prover.task_context.goal, external_goal)
        self.assertIn(external_goal, prover.task_context.target_facts)
        self.assertIn(intrinsic_goal, prover.task_context.target_facts)
        self.assertTrue(prover.task_context.trigger_abduction)

    def test_external_goal_keeps_background_intrinsic_queue_visible(self) -> None:
        prover = DifferentiableProver(
            d_latent=8,
            sym_vocab=32,
            max_rules=16,
            max_depth=2,
            n_cands=2,
        )
        intrinsic_goal = atom(77, Var("Q"))
        external_goal = atom(88, Const(1))
        prover.creative_cycle.intrinsic_engine._schedule_goal(
            IntrinsicGoal(
                goal=intrinsic_goal,
                value=0.9,
                kind="explore_structure",
                provenance="test",
            )
        )
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=frozenset(),
                goal=external_goal,
                target_facts=frozenset({external_goal}),
                provenance="test",
            )
        )
        self.assertEqual(prover.current_goal(), external_goal)
        self.assertEqual(prover.scheduled_intrinsic_goals(), (intrinsic_goal,))

    def test_task_context_materializes_creative_report_as_first_class_context(self) -> None:
        prover = DifferentiableProver(
            d_latent=8,
            sym_vocab=32,
            max_rules=16,
            max_depth=2,
            n_cands=2,
        )
        creative_head = atom(501, Var("X"))
        novel_fact = atom(777, Const(3))
        intrinsic_goal = atom(888, Const(4))
        validated_support = atom(909, Const(1))
        prover.creative_cycle.last_report = CreativeCycleReport(
            selected_rules=[
                RuleCandidate(
                    clause=rule(creative_head, atom(101, Var("X"))),
                    source="analogy",
                    score=0.9,
                    utility=0.8,
                )
            ],
            counterfactual_novel_facts=(novel_fact,),
            validated_support_facts=(validated_support,),
            intrinsic_goal=IntrinsicGoal(
                goal=intrinsic_goal,
                value=0.7,
                kind="explore_structure",
                provenance="test",
            ),
            metrics={"selected_rules": 1.0, "counterfactual_candidates": 1.0},
        )
        base_goal = atom(42, Const(1))
        prover.set_task_context(
            SymbolicTaskContext(
                observed_facts=frozenset({atom(1, Const(1))}),
                goal=base_goal,
                target_facts=frozenset({base_goal}),
                provenance="test",
            )
        )
        self.assertIsNotNone(prover.task_context)
        self.assertIn(creative_head, prover.task_context.abduced_support_facts)
        self.assertIn(validated_support, prover.task_context.abduced_support_facts)
        self.assertIn(novel_fact, prover.task_context.world_context_facts)
        self.assertIn(intrinsic_goal, prover.task_context.world_context_facts)
        self.assertIn(creative_head, prover.task_context.observed_facts)
        self.assertGreaterEqual(
            float(prover.task_context.world_context_summary.get("creative_selected_rules", 0.0)),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

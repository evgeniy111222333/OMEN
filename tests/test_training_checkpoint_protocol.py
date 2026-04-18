from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen import build_omen
from omen_prolog import Const, HornAtom, HornClause, Var
from omen_scale_config import OMENScaleConfig
from omen_symbolic.creative_types import CreativeCycleReport, IntrinsicGoal, RuleCandidate
from omen_train_code import (
    DEVICE as TRAIN_DEVICE,
    _finite_metric,
    _restore_kb,
    _save_ckpt,
    build_loader,
    joint_train,
    make_synthetic_dataset,
)


def atom(pred: int, *args) -> HornAtom:
    return HornAtom(pred=pred, args=tuple(args))


def rule(head: HornAtom, *body: HornAtom) -> HornClause:
    return HornClause(head=head, body=tuple(body))


def tiny_cfg(*, creative_enabled: bool) -> OMENScaleConfig:
    cfg = OMENScaleConfig.demo()
    cfg.d_tok = 64
    cfg.d_model = 64
    cfg.d_latent = 16
    cfg.n_latents = 8
    cfg.n_heads_tok = 2
    cfg.n_layers_tok = 1
    cfg.n_heads_lat = 2
    cfg.n_layers_lat = 1
    cfg.n_heads = 2
    cfg.n_layers = 1
    cfg.seq_len = 48
    cfg.world_rnn_hidden = 32
    cfg.world_graph_max_nodes = 32
    cfg.world_graph_max_edges = 64
    cfg.world_graph_max_transitions = 4
    cfg.mem_heads = 2
    cfg.mem_cache_size = 32
    cfg.mem_symbolic_cache_size = 32
    cfg.sym_vocab = 24
    cfg.sym_embed_dim = 12
    cfg.max_proof_depth = 2
    cfg.n_proof_cands = 4
    cfg.ltm_max_rules = 64
    cfg.sym_max_facts = 16
    cfg.net_byte_layers = 1
    cfg.net_dec_layers = 1
    cfg.net_init_vocab = 16
    cfg.net_max_vocab = 64
    cfg.emc_enabled = False
    cfg.osf_enabled = False
    cfg.creative_cycle_enabled = creative_enabled
    cfg.ame_contrastive_steps = 0
    return cfg


class TrainingCheckpointProtocolTest(unittest.TestCase):
    def test_finite_metric_sanitizes_non_finite_values(self) -> None:
        self.assertEqual(_finite_metric(float("inf")), 0.0)
        self.assertEqual(_finite_metric(float("-inf")), 0.0)
        self.assertEqual(_finite_metric(float("nan")), 0.0)
        self.assertEqual(_finite_metric(torch.tensor(3.5)), 3.5)

    def test_build_loader_enables_persistent_workers_for_parallel_loading(self) -> None:
        cfg = tiny_cfg(creative_enabled=False)
        dataset = make_synthetic_dataset(cfg, n=12)
        loader = build_loader(dataset, batch_size=2, num_workers=1, shuffle=False)
        self.assertTrue(loader.persistent_workers)
        self.assertEqual(loader.prefetch_factor, 2)

    def test_joint_train_sanitizes_non_finite_grad_norm_telemetry(self) -> None:
        cfg = tiny_cfg(creative_enabled=False)
        model = build_omen(cfg, device=TRAIN_DEVICE)
        dataset = make_synthetic_dataset(cfg, n=12)

        with mock.patch("omen_train_code.torch.nn.utils.clip_grad_norm_") as clip_mock:
            clip_mock.side_effect = [
                torch.tensor(float("inf")),
                torch.tensor(float("nan")),
            ]
            result = joint_train(
                model,
                dataset,
                n_epochs=1,
                batch_size=2,
                lr=1e-4,
                max_batches_per_epoch=1,
                checkpoint_dir=None,
                use_amp=False,
                grad_accum=1,
                num_workers=0,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(float(result[0]["gnorm_net"]), 0.0)
        self.assertEqual(float(result[0]["gnorm_other"]), 0.0)
        self.assertEqual(float(result[0]["amp_overflow_steps"]), 0.0)

    def test_train_fast_metric_profile_keeps_training_scalars(self) -> None:
        cfg = tiny_cfg(creative_enabled=False)
        model = build_omen(cfg, device=torch.device("cpu"))
        model.eval()
        torch.manual_seed(0)
        src = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len - 1))
        tgt = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len - 1))

        full = model(src, tgt)
        fast = model(src, tgt, metric_profile="train_fast")

        self.assertIn("total", fast)
        self.assertIn("ce", fast)
        self.assertIn("world", fast)
        self.assertIn("l_scale", fast)
        self.assertIn("world_graph_nodes", fast)
        self.assertNotIn("logits", fast)
        self.assertNotIn("z", fast)
        self.assertLess(len(fast), len(full))
        self.assertAlmostEqual(float(fast["ce"]), float(full["ce"]), places=5)
        self.assertAlmostEqual(float(fast["world"]), float(full["world"]), places=5)

    def test_checkpoint_roundtrip_restores_runtime_state(self) -> None:
        cfg = tiny_cfg(creative_enabled=True)
        model = build_omen(cfg, device=torch.device("cpu"))

        x, y, z = Var("X"), Var("Y"), Var("Z")
        fact = atom(111, Const(1), Const(2))
        kb_rule = rule(atom(211, x), atom(111, x, Const(2)))
        model.prover.kb.add_fact(fact)
        model.prover.kb.add_rule(kb_rule)

        episodic_key = torch.randn(cfg.d_latent)
        episodic_val = torch.randn(cfg.d_latent)
        model.memory.cache.append((episodic_key.clone(), episodic_val.clone()))
        model.memory.symbolic_index.write([fact], episodic_key.unsqueeze(0))
        model._train_step.fill_(17)
        model._seen_tokens.fill_(4096)

        intrinsic_goal = atom(777, Const(9))
        intrinsic = IntrinsicGoal(
            goal=intrinsic_goal,
            value=0.8,
            kind="explore_structure",
            provenance="checkpoint-test",
            metadata={"priority": 1.0},
        )
        model.prover.creative_cycle.intrinsic_engine.update_state(torch.randn(2, cfg.d_latent))
        model.prover.creative_cycle.intrinsic_engine._schedule_goal(intrinsic)

        creative_candidate = RuleCandidate(
            clause=kb_rule,
            source="analogy",
            score=0.4,
            metadata={"from_checkpoint": 1.0},
        )
        model.prover.creative_cycle.aesthetic_engine._gene_pool = [creative_candidate]

        analogy_rules = [
            rule(atom(1, x, y, z), atom(101, x, z)),
            rule(atom(2, x, y, z), atom(102, x, z)),
            rule(atom(201, x, z), atom(1, x, y, z)),
            rule(atom(202, x, z), atom(2, x, y, z)),
        ]
        analogy_engine = model.prover.creative_cycle.analogy_engine
        analogy_engine.fit(analogy_rules)
        original_predicate_ids = analogy_engine.state.graph_view.predicate_ids

        ontology_engine = model.prover.creative_cycle.ontology_engine
        ontology_engine.observe_gap_cluster(0.8, [1, 3], z=torch.randn(2, cfg.d_latent))
        ontology_engine._register_predicate(900_001, arity=1, confidence=0.75, metadata={"checkpoint": 1.0})
        ontology_model = ontology_engine._ensure_model(embed_dim=4)
        with torch.no_grad():
            first_param = next(ontology_model.parameters())
            first_param.add_(0.05)
        original_oee_param = next(ontology_model.parameters()).detach().clone()
        ontology_engine._last_context_input = torch.randn(ontology_model.context_encoder[0].in_features)
        ontology_engine._last_gap_norm = 0.8
        ontology_engine._last_pred_ids = [900_001]
        ontology_engine.record_feedback(
            True,
            accepted_pred_ids=[900_001],
            gap_before=0.9,
            gap_after=0.2,
            supporting_rules=[kb_rule],
        )
        ontology_engine._call_count = 5
        ontology_engine._online_train_steps = 3
        ontology_engine._last_online_train_loss = 0.25
        ontology_engine._last_online_train_buffer_size = float(len(ontology_engine._feedback_buffer))

        report = CreativeCycleReport(
            selected_rules=[creative_candidate],
            intrinsic_goal=intrinsic,
            metrics={"checkpoint_metric": 1.0},
            predicate_embeddings={1: torch.randn(4)},
        )
        model.prover.creative_cycle.last_report = report
        model.prover.last_creative_report = report

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=1,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            _save_ckpt(model, optimizer, scheduler, scaler, epoch=3, metrics={"loss": 1.0}, save_dir=tmp_dir)
            ckpt_path = next(Path(tmp_dir).glob("omen_epoch*.pt"))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            restored = build_omen(cfg, device=torch.device("cpu"))
            restored.load_state_dict(ckpt["model"])
            _restore_kb(restored, ckpt["kb_state"], torch.device("cpu"))
            restored.load_runtime_state(ckpt["runtime_state"], device=torch.device("cpu"))
            if ckpt.get("net_tau") is not None and restored.net_enabled:
                restored.net.quantizer.tau = float(ckpt["net_tau"])

        self.assertIn("scheduler", ckpt)
        self.assertIn("scaler", ckpt)
        self.assertEqual(restored.prover.kb.n_facts(), 1)
        self.assertEqual(int(restored._train_step.item()), 17)
        self.assertEqual(int(restored._seen_tokens.item()), 4096)
        self.assertEqual(len(restored.memory.cache), 1)
        self.assertTrue(torch.allclose(restored.memory.cache[0][0], episodic_key))
        self.assertTrue(torch.allclose(restored.memory.cache[0][1], episodic_val))
        recalled = restored.memory.symbolic_index.recall_by_pattern(predicate_hints=[111], limit=2)
        self.assertEqual([hash(item) for item in recalled], [hash(fact)])
        self.assertEqual(hash(restored.prover.creative_cycle.intrinsic_engine.pending_goal.goal), hash(intrinsic_goal))
        self.assertEqual(len(restored.prover.creative_cycle.aesthetic_engine._gene_pool), 1)
        self.assertEqual(
            restored.prover.creative_cycle.analogy_engine.state.graph_view.predicate_ids,
            original_predicate_ids,
        )
        restored_oee = restored.prover.creative_cycle.ontology_engine
        self.assertEqual(restored_oee.last_pred_ids, [900_001])
        self.assertEqual(restored_oee.predicate_vocab()[900_001].status, "fixed")
        self.assertEqual(int(restored_oee.stats()["oee_feedback_buffer_size"]), 1)
        self.assertTrue(torch.allclose(next(restored_oee._latent_model.parameters()), original_oee_param))
        self.assertIn("checkpoint_metric", restored.prover.last_creative_report.metrics)

    def test_joint_train_resume_uses_total_target_epoch(self) -> None:
        cfg = tiny_cfg(creative_enabled=False)
        model = build_omen(cfg, device=TRAIN_DEVICE)
        dataset = make_synthetic_dataset(cfg, n=12)

        with tempfile.TemporaryDirectory() as tmp_dir:
            first_pass = joint_train(
                model,
                dataset,
                n_epochs=1,
                batch_size=2,
                lr=1e-4,
                max_batches_per_epoch=1,
                checkpoint_dir=tmp_dir,
                use_amp=False,
                grad_accum=1,
                num_workers=0,
            )
            self.assertEqual(len(first_pass), 1)
            ckpt_path = Path(tmp_dir) / "omen_epoch0001.pt"
            ckpt = torch.load(ckpt_path, map_location=TRAIN_DEVICE, weights_only=False)

            resumed_model = build_omen(cfg, device=TRAIN_DEVICE)
            resumed_model.load_state_dict(ckpt["model"])
            _restore_kb(resumed_model, ckpt["kb_state"], TRAIN_DEVICE)
            resumed_model.load_runtime_state(ckpt["runtime_state"], device=TRAIN_DEVICE)
            if ckpt.get("net_tau") is not None and resumed_model.net_enabled:
                resumed_model.net.quantizer.tau = float(ckpt["net_tau"])

            resumed = joint_train(
                resumed_model,
                dataset,
                n_epochs=2,
                batch_size=2,
                lr=1e-4,
                max_batches_per_epoch=1,
                checkpoint_dir=None,
                use_amp=False,
                grad_accum=1,
                num_workers=0,
                resume_state={
                    "optimizer": ckpt.get("optimizer"),
                    "scheduler": ckpt.get("scheduler"),
                    "scaler": ckpt.get("scaler"),
                },
                start_epoch=int(ckpt["epoch"]),
            )

        self.assertEqual(len(resumed), 1)
        self.assertEqual(float(resumed[0]["epoch"]), 2.0)
        self.assertGreaterEqual(int(resumed_model._train_step.item()), int(model._train_step.item()))


if __name__ == "__main__":
    unittest.main()

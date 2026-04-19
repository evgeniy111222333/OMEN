from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_omen_scale_eval import (
    build_transfer_tasks,
    run_benchmark,
    run_corpus_protocol,
    run_creative_ablation_suite,
)
from omen_scale_config import OMENScaleConfig
from omen_train_code import load_text_corpus, make_synthetic_dataset


def _benchmark_test_config() -> OMENScaleConfig:
    cfg = OMENScaleConfig.demo()
    cfg.allow_noncanonical_ablation = True
    cfg.vocab_size = 256
    cfg.d_tok = 64
    cfg.n_heads_tok = 4
    cfg.n_layers_tok = 1
    cfg.seq_len = 48
    cfg.d_latent = 32
    cfg.n_latents = 8
    cfg.n_heads_lat = 2
    cfg.n_layers_lat = 1
    cfg.world_rnn_hidden = 48
    cfg.world_rollout_steps = 2
    cfg.world_graph_max_nodes = 24
    cfg.world_graph_max_edges = 64
    cfg.world_graph_max_transitions = 4
    cfg.mem_heads = 2
    cfg.mem_cache_size = 64
    cfg.mem_symbolic_cache_size = 64
    cfg.sym_vocab = 32
    cfg.sym_embed_dim = 16
    cfg.max_proof_depth = 2
    cfg.n_proof_cands = 4
    cfg.ltm_max_rules = 64
    cfg.sym_max_facts = 16
    cfg.abduct_candidates = 4
    cfg.n_heads = 2
    cfg.n_layers = 1
    cfg.d_model = 64
    cfg.net_enabled = False
    cfg.osf_enabled = False
    cfg.emc_enabled = False
    cfg.creative_cycle_enabled = False
    cfg.sym_graph_reasoning_enabled = False
    cfg.sym_query_gen_enabled = False
    cfg.sym_decoder_surprise_enabled = False
    cfg.continuous_cycle_contextual = 2
    cfg.continuous_cycle_neural = 1
    return cfg


class BenchmarkProtocolTest(unittest.TestCase):
    def test_synthetic_protocol_emits_canonical_metrics(self) -> None:
        cfg = _benchmark_test_config()
        dataset = make_synthetic_dataset(cfg, n=12)
        report = run_benchmark(cfg, dataset, batches=1, batch_size=1, seed=7)

        self.assertEqual(report["n_batches"], 1)
        self.assertEqual(report["canonical_stack"], "omen_scale_world_graph")
        self.assertEqual(report["canonical_public_module"], "omen.OMEN")
        self.assertEqual(report["canonical_repository_axis"], "omen_scale_single_canon_repository")
        self.assertIn("omen_v2.py", report["legacy_reference_modules"])
        self.assertEqual(report["protocol_seed"], 7)
        summary = report["summary"]
        for key in (
            "program_anchor",
            "program_target_facts",
            "world_causal_error",
            "world_alignment",
            "sym_cycle_active",
            "sym_cycle_checked",
            "sym_cycle_eval_active",
            "sym_cycle_learning_active",
            "z_graph_primary",
            "z_graph_anchor",
            "world_graph_nodes",
            "world_graph_execution_steps",
            "world_graph_hidden_fallback_steps",
            "world_graph_hidden_teacher_applied",
            "world_graph_neutral_prior_applied",
            "world_graph_context_facts",
            "world_graph_memory_facts",
            "world_graph_net_facts",
            "world_graph_semantic_graph_enriched",
            "z_posterior_graph_native",
            "z_posterior_perceiver_fallback",
            "world_graph_transition_native",
            "sym_world_context_facts",
            "sym_observed_now_facts",
            "eval_world_self_update_applied",
            "creative_oee_online_train_applied",
            "creative_cycle_active",
            "creative_gap_reduction",
            "sal_named_role_preds",
            "sal_named_role_rel_preds",
        ):
            self.assertIn(key, summary)

    def test_real_text_protocol_runs_on_local_corpus(self) -> None:
        cfg = _benchmark_test_config()
        corpus_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as fh:
                corpus_path = fh.name
                fh.write(("def add(a, b): return a + b\n" * 32))
            dataset = load_text_corpus(corpus_path, cfg.seq_len, max_samples=8)
            report = run_benchmark(cfg, dataset, batches=1, batch_size=1)
            self.assertEqual(report["n_batches"], 1)
            self.assertIn("ce_bits", report["summary"])
        finally:
            if corpus_path and os.path.exists(corpus_path):
                os.remove(corpus_path)

    def test_time_budget_returns_partial_report(self) -> None:
        cfg = _benchmark_test_config()
        dataset = make_synthetic_dataset(cfg, n=12)
        with mock.patch(
            "benchmarks.benchmark_omen_scale_eval._budget_expired",
            side_effect=[False, True],
        ):
            report = run_benchmark(
                cfg,
                dataset,
                batches=4,
                batch_size=1,
                seed=5,
                time_budget_sec=0.01,
            )
        self.assertEqual(report["n_batches"], 1)
        self.assertEqual(report["requested_batches"], 4)
        self.assertTrue(report["timed_out"])
        self.assertEqual(report["stop_reason"], "time_budget_exhausted")
        self.assertEqual(report["time_budget_sec"], 0.01)
        self.assertGreaterEqual(report["wall_time_sec"], 0.0)

    def test_time_budget_marks_single_overrun_batch(self) -> None:
        cfg = _benchmark_test_config()
        dataset = make_synthetic_dataset(cfg, n=12)
        with mock.patch(
            "benchmarks.benchmark_omen_scale_eval._budget_expired",
            side_effect=[False, True],
        ):
            report = run_benchmark(
                cfg,
                dataset,
                batches=1,
                batch_size=1,
                seed=9,
                time_budget_sec=0.0,
            )
        self.assertEqual(report["n_batches"], 1)
        self.assertEqual(report["requested_batches"], 1)
        self.assertTrue(report["timed_out"])
        self.assertEqual(report["stop_reason"], "time_budget_exhausted")

    def test_progress_report_path_emits_partial_json(self) -> None:
        cfg = _benchmark_test_config()
        dataset = make_synthetic_dataset(cfg, n=12)
        report_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as fh:
                report_path = fh.name
            report = run_benchmark(
                cfg,
                dataset,
                batches=2,
                batch_size=1,
                seed=3,
                progress_report_path=report_path,
            )
            payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["n_batches"], report["n_batches"])
            self.assertEqual(payload["requested_batches"], 2)
            self.assertEqual(payload["canonical_stack"], "omen_scale_world_graph")
            self.assertEqual(payload["canonical_public_module"], "omen.OMEN")
        finally:
            if report_path and os.path.exists(report_path):
                os.remove(report_path)

    def test_manifest_protocol_runs_on_named_local_corpora(self) -> None:
        cfg = _benchmark_test_config()
        files = []
        manifest_path = None
        try:
            corpora = {
                "python_real": {
                    "content": "def add(a, b):\n    return a + b\n" * 4,
                    "language": "python",
                    "family": "codeparrot",
                    "source": "the-stack",
                    "weight": 2.0,
                },
                "javascript_real": {
                    "content": "function add(a, b) {\n  return a + b;\n}\n" * 4,
                    "language": "javascript",
                    "family": "commoncrawl",
                    "source": "commoncrawl",
                    "weight": 1.0,
                },
            }
            manifest = {}
            for name, spec in corpora.items():
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as fh:
                    fh.write(spec["content"])
                    fh.flush()
                    files.append(fh.name)
                    manifest[name] = {
                        "path": fh.name,
                        "language": spec["language"],
                        "family": spec["family"],
                        "source": spec["source"],
                        "split": "validation",
                        "weight": spec["weight"],
                    }
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as fh:
                json.dump(manifest, fh)
                fh.flush()
                manifest_path = fh.name

            report = run_corpus_protocol(
                cfg,
                manifest_path,
                adapt_steps=0,
                eval_batches=1,
                batch_size=1,
                max_samples=1,
                seed=11,
            )
            self.assertEqual(report["n_tasks"], 2)
            self.assertEqual(report["canonical_stack"], "omen_scale_world_graph")
            self.assertEqual(report["canonical_public_module"], "omen.OMEN")
            self.assertEqual(report["real_protocol_tasks"], 2.0)
            self.assertEqual(report["real_protocol_languages"], 2.0)
            self.assertEqual(report["real_protocol_families"], 2.0)
            self.assertEqual(report["real_protocol_sources"], 2.0)
            self.assertEqual(report["protocol_seed"], 11)
            self.assertIn("python", report["protocol_language_summaries"])
            self.assertIn("javascript", report["protocol_language_summaries"])
            self.assertIn("codeparrot", report["protocol_family_summaries"])
            self.assertIn("commoncrawl", report["protocol_family_summaries"])
            self.assertIn("the-stack", report["protocol_source_summaries"])
            self.assertIn("commoncrawl", report["protocol_source_summaries"])
            self.assertIn("ce_bits", report["protocol_weighted_summary"])
            self.assertEqual(report["corpus_protocol"]["python_real"]["family"], "codeparrot")
        finally:
            for path in files:
                if os.path.exists(path):
                    os.remove(path)
            if manifest_path and os.path.exists(manifest_path):
                os.remove(manifest_path)

    def test_creative_ablation_suite_reports_enabled_disabled_deltas(self) -> None:
        cfg = _benchmark_test_config()
        tasks = build_transfer_tasks(cfg, synthetic_samples=4)
        tasks = {name: tasks[name] for name in ("python", "observation_text", "observation_structured")}

        report = run_creative_ablation_suite(
            cfg,
            tasks=tasks,
            source_task="python",
            adapt_steps=0,
            eval_batches=1,
            batch_size=1,
            seed=13,
        )

        self.assertEqual(report["n_tasks"], 3)
        self.assertEqual(report["canonical_stack"], "omen_scale_world_graph")
        self.assertEqual(report["canonical_public_module"], "omen.OMEN")
        self.assertEqual(report["protocol_seed"], 13)
        self.assertIn("creative_cycle_active_delta", report["aggregate_delta"])
        self.assertIn("python", report["task_deltas"])
        self.assertIn("observation_text", report["task_deltas"])
        self.assertEqual(report["aggregate_disabled_summary"].get("creative_cycle_active", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()

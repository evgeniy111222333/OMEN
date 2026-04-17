from __future__ import annotations

import random
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_omen_scale_eval import build_transfer_tasks, run_transfer_suite
from omen_scale_config import OMENScaleConfig
from omen_train_code import load_text_corpus


def _transfer_test_config() -> OMENScaleConfig:
    cfg = OMENScaleConfig.demo()
    cfg.vocab_size = 256
    cfg.d_tok = 64
    cfg.n_heads_tok = 4
    cfg.n_layers_tok = 1
    cfg.seq_len = 40
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
    cfg.saliency_enabled = False
    cfg.continuous_cycle_enabled = False
    cfg.creative_cycle_enabled = False
    cfg.sym_graph_reasoning_enabled = False
    cfg.sym_query_gen_enabled = False
    cfg.sym_decoder_surprise_enabled = False
    cfg.continuous_cycle_contextual = 2
    cfg.continuous_cycle_neural = 2
    cfg.creative_max_selected_rules = 1
    cfg.ame_embedding_dim = 8
    cfg.ame_hidden_dim = 24
    cfg.ame_gnn_layers = 1
    cfg.aee_population = 6
    cfg.aee_generations = 1
    cfg.aee_gene_pool_size = 12
    cfg.oee_max_hypotheses = 4
    cfg.oee_bundle_beam_width = 2
    cfg.oee_bundle_seed_k = 4
    return cfg


class TransferSuiteProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(0)
        torch.manual_seed(0)

    def test_synthetic_transfer_suite_reports_multilingual_metrics(self) -> None:
        cfg = _transfer_test_config()
        all_tasks = build_transfer_tasks(cfg, synthetic_samples=4)
        tasks = {name: all_tasks[name] for name in ("python", "javascript", "rust", "multilingual")}

        report = run_transfer_suite(
            cfg,
            tasks=tasks,
            source_task="python",
            adapt_steps=0,
            eval_batches=4,
            batch_size=1,
        )

        self.assertEqual(report["n_tasks"], 4)
        self.assertEqual(report["canonical_public_module"], "omen.OMEN")
        self.assertEqual(report["canonical_repository_axis"], "omen_scale_single_canon_repository")
        summaries = report["task_summaries"]
        self.assertIn("python", summaries)
        self.assertIn("javascript", summaries)
        self.assertIn("rust", summaries)
        self.assertIn("ce_bits", report["aggregate_summary"])
        self.assertIn("javascript", report["transfer_deltas"])
        self.assertGreaterEqual(summaries["python"].get("sym_ast_lang_python", 0.0), 0.5)
        self.assertGreaterEqual(summaries["javascript"].get("sym_ast_lang_javascript", 0.0), 0.5)
        self.assertGreaterEqual(summaries["rust"].get("sym_ast_lang_rust", 0.0), 0.5)
        self.assertIn("creative_analogy_embedding_source", summaries["python"])

    def test_real_text_transfer_suite_runs_on_named_corpora(self) -> None:
        cfg = _transfer_test_config()
        files: list[str] = []
        try:
            corpora = {
                "python_real": "def add(a, b):\n    return a + b\n" * 4,
                "javascript_real": "function add(a, b) {\n  return a + b;\n}\n" * 4,
                "rust_real": "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n" * 4,
            }
            tasks = {}
            for name, content in corpora.items():
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as fh:
                    fh.write(content)
                    fh.flush()
                    files.append(fh.name)
                    tasks[name] = load_text_corpus(fh.name, cfg.seq_len, max_samples=1)

            report = run_transfer_suite(
                cfg,
                tasks=tasks,
                source_task="python_real",
                adapt_steps=0,
                eval_batches=1,
                batch_size=1,
            )

            self.assertEqual(report["n_tasks"], 3)
            self.assertEqual(report["canonical_public_module"], "omen.OMEN")
            summaries = report["task_summaries"]
            self.assertGreaterEqual(summaries["python_real"].get("sym_ast_lang_python", 0.0), 0.5)
            self.assertGreaterEqual(summaries["javascript_real"].get("sym_ast_lang_javascript", 0.0), 0.5)
            self.assertGreaterEqual(summaries["rust_real"].get("sym_ast_lang_rust", 0.0), 0.5)
        finally:
            for path in files:
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()

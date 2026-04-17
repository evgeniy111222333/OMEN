from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_omen_scale_eval import run_benchmark, run_corpus_protocol
from omen_scale_config import OMENScaleConfig
from omen_train_code import load_text_corpus, make_synthetic_dataset


def _benchmark_test_config() -> OMENScaleConfig:
    cfg = OMENScaleConfig.demo()
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
        report = run_benchmark(cfg, dataset, batches=1, batch_size=1)

        self.assertEqual(report["n_batches"], 1)
        self.assertEqual(report["canonical_stack"], "omen_scale_world_graph")
        self.assertEqual(report["canonical_public_module"], "omen.OMEN")
        self.assertEqual(report["canonical_repository_axis"], "omen_scale_single_canon_repository")
        self.assertIn("omen_v2.py", report["legacy_reference_modules"])
        summary = report["summary"]
        for key in (
            "program_anchor",
            "program_target_facts",
            "sym_cycle_checked",
            "z_graph_primary",
            "z_graph_anchor",
            "world_graph_nodes",
            "sal_named_role_preds",
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

    def test_manifest_protocol_runs_on_named_local_corpora(self) -> None:
        cfg = _benchmark_test_config()
        files = []
        manifest_path = None
        try:
            corpora = {
                "python_real": "def add(a, b):\n    return a + b\n" * 4,
                "javascript_real": "function add(a, b) {\n  return a + b;\n}\n" * 4,
            }
            manifest = {}
            for name, content in corpora.items():
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as fh:
                    fh.write(content)
                    fh.flush()
                    files.append(fh.name)
                    manifest[name] = fh.name
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as fh:
                import json

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
            )
            self.assertEqual(report["n_tasks"], 2)
            self.assertEqual(report["canonical_stack"], "omen_scale_world_graph")
            self.assertEqual(report["canonical_public_module"], "omen.OMEN")
            self.assertEqual(report["real_protocol_tasks"], 2.0)
        finally:
            for path in files:
                if os.path.exists(path):
                    os.remove(path)
            if manifest_path and os.path.exists(manifest_path):
                os.remove(manifest_path)


if __name__ == "__main__":
    unittest.main()

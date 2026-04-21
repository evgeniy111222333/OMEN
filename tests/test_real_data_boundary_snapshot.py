from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.real_data_boundary_snapshot import _collect_snapshot


class RealDataBoundarySnapshotTest(unittest.TestCase):
    def test_collect_snapshot_reports_boundary_metrics_for_natural_text_corpus(self) -> None:
        corpus_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as fh:
                corpus_path = fh.name
                fh.write(
                    "weather is rain. rain becomes flood. however flood is not safe.\n"
                    "weather is rain. rain becomes flood. however flood is not safe.\n"
                )

            report = _collect_snapshot(
                config="demo",
                real_text=corpus_path,
                max_samples=1,
                sample_index=0,
                device=torch.device("cpu"),
            )

            self.assertEqual(report["stop_reason"], "completed")
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["sample_index"], 0)
            self.assertEqual(report["source_modality"], "natural_text")
            self.assertEqual(report["source_verification_path"], "natural_language_claim_verification")
            self.assertGreaterEqual(report["forward_ms"], 0.0)
            self.assertGreaterEqual(report["sym_grounding_diagnostic_artifacts"], 1.0)
            self.assertGreaterEqual(report["sym_grounding_proposal_artifacts"], 1.0)
            self.assertGreaterEqual(report["planner_state_authoritative_records"], 0.0)
            self.assertGreaterEqual(report["planner_state_diagnostic_records"], 1.0)
            self.assertGreaterEqual(report["planner_state_diagnostic_symbols"], 0.0)
            self.assertIn("weather", report["sample_preview"])
        finally:
            if corpus_path:
                path = Path(corpus_path)
                if path.exists():
                    path.unlink()

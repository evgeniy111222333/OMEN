from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_deep_scenario_report import build_report


class DeepScenarioReportTest(unittest.TestCase):
    def test_build_report_includes_all_requested_scenarios_and_key_findings(self) -> None:
        report = build_report()

        self.assertIn("Сценарій 1", report)
        self.assertIn("Сценарій 2", report)
        self.assertIn("Сценарій 3", report)
        self.assertIn("Сценарій 4", report)
        self.assertIn("Сценарій 5", report)
        self.assertIn("Зірки непрямим шляхом генерують Супутники.", report)
        self.assertIn("total = total + i", report)
        self.assertIn("external_failed_login_cluster", report)
        self.assertIn("only stone", report)


if __name__ == "__main__":
    unittest.main()

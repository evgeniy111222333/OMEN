from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen import OMEN, OMENConfig, build_omen, canonical_architecture, module_role, repository_axis
from omen_canonical import CANONICAL_OMEN_SPEC
from omen_scale import OMENScale
from omen_v2 import CANONICAL_PUBLIC_SUCCESSOR, CANONICAL_SUCCESSOR, LEGACY_RUNTIME, LEGACY_RUNTIME_ROLE


class CanonicalStackProtocolTest(unittest.TestCase):
    def test_canonical_stack_is_explicit_and_legacy_runtime_is_marked(self) -> None:
        spec = canonical_architecture()
        self.assertEqual(spec, CANONICAL_OMEN_SPEC)
        self.assertEqual(spec.entrypoint, "omen_scale.OMENScale")
        self.assertEqual(spec.public_module, "omen.OMEN")
        self.assertEqual(spec.config_entrypoint, "omen.OMENConfig")
        self.assertIn("omen_v2.py", spec.legacy_modules)
        self.assertTrue(LEGACY_RUNTIME)
        self.assertEqual(LEGACY_RUNTIME_ROLE, "legacy_reference")
        self.assertEqual(CANONICAL_SUCCESSOR, spec.entrypoint)
        self.assertEqual(CANONICAL_PUBLIC_SUCCESSOR, spec.public_module)

    def test_public_facade_points_to_single_canonical_runtime(self) -> None:
        cfg = OMENConfig.demo()
        cfg.net_enabled = False
        cfg.osf_enabled = False
        cfg.emc_enabled = False
        cfg.creative_cycle_enabled = False
        model = build_omen(cfg)
        self.assertIsInstance(model, OMEN)
        self.assertIsInstance(model, OMENScale)
        self.assertEqual(model.canonical_architecture(), CANONICAL_OMEN_SPEC)

    def test_repository_axis_classifies_modules(self) -> None:
        axis = repository_axis()
        self.assertEqual(axis.public_module, "omen.py")
        self.assertEqual(axis.module_role("omen_scale.py"), "canonical_surface")
        self.assertEqual(module_role("omen_v2.py"), "legacy_reference")
        self.assertEqual(module_role("omen_symbolic/world_graph.py"), "canonical_support")
        self.assertEqual(module_role("omen_symbolic/aesthetic_engine.py"), "research_extension")


if __name__ == "__main__":
    unittest.main()

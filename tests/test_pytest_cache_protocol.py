from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import conftest


class PytestCacheProtocolTest(unittest.TestCase):
    def test_cache_bootstrap_creates_supporting_files_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / ".pytest_cache"

            conftest._ensure_pytest_cache_dir(cache_dir)
            conftest._ensure_pytest_cache_dir(cache_dir)

            self.assertTrue(cache_dir.is_dir())
            self.assertEqual(
                (cache_dir / "README.md").read_text(encoding="utf-8"),
                conftest._PYTEST_CACHE_README,
            )
            self.assertEqual(
                (cache_dir / ".gitignore").read_text(encoding="utf-8"),
                "# Created by pytest automatically.\n*\n",
            )
            self.assertEqual(
                (cache_dir / "CACHEDIR.TAG").read_bytes(),
                conftest._PYTEST_CACHE_TAG,
            )


if __name__ == "__main__":
    unittest.main()

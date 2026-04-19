from __future__ import annotations

from pathlib import Path


_PYTEST_CACHE_README = """# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.
"""

_PYTEST_CACHE_TAG = b"""Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by pytest.
# For information about cache directory tags, see:
#\thttps://bford.info/cachedir/spec.html
"""


def _ensure_pytest_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    readme_path = cache_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(_PYTEST_CACHE_README, encoding="utf-8")

    gitignore_path = cache_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("# Created by pytest automatically.\n*\n", encoding="utf-8")

    tag_path = cache_dir / "CACHEDIR.TAG"
    if not tag_path.exists():
        tag_path.write_bytes(_PYTEST_CACHE_TAG)


def pytest_sessionstart(session) -> None:
    raw_cache_dir = Path(session.config.getini("cache_dir"))
    cache_dir = raw_cache_dir if raw_cache_dir.is_absolute() else Path(session.config.rootpath) / raw_cache_dir
    _ensure_pytest_cache_dir(cache_dir)

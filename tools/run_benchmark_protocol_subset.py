from __future__ import annotations

import argparse
import inspect
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _available_tests() -> list[str]:
    module_path = ROOT / "tests" / "test_benchmark_protocol.py"
    spec = importlib.util.spec_from_file_location("test_benchmark_protocol", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark protocol module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    benchmark_case = getattr(module, "BenchmarkProtocolTest", None)
    if benchmark_case is None:
        raise RuntimeError("BenchmarkProtocolTest was not found in tests/test_benchmark_protocol.py")
    return sorted(
        name
        for name, member in inspect.getmembers(benchmark_case)
        if callable(member) and name.startswith("test_")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run only 1 or 2 specific tests from tests/test_benchmark_protocol.py. "
            "Use --list to see the allowed test names."
        )
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="One or two BenchmarkProtocolTest method names to run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmark protocol tests and exit.",
    )
    args = parser.parse_args()

    available = _available_tests()
    if args.list:
        print("Available benchmark protocol tests:")
        for name in available:
            print(f"  {name}")
        return 0

    selected = list(args.tests)
    if not selected:
        parser.error("Select exactly 1 or 2 tests, or use --list.")
    if len(selected) > 2:
        parser.error("Only 1 or 2 benchmark protocol tests may be run at once.")

    unknown = [name for name in selected if name not in available]
    if unknown:
        parser.error(
            "Unknown benchmark protocol test(s): "
            + ", ".join(unknown)
            + ". Use --list to inspect valid names."
        )

    node_ids = [
        f"tests/test_benchmark_protocol.py::BenchmarkProtocolTest::{name}"
        for name in selected
    ]
    cmd = [sys.executable, "-m", "pytest", *node_ids, "-q"]
    print("Running benchmark protocol subset:")
    for node_id in node_ids:
        print(f"  {node_id}")
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

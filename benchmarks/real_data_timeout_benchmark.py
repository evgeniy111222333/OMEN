from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_omen_scale_eval import build_config, run_benchmark
from omen_train_code import load_text_corpus


def _worker(
    *,
    config: str,
    real_text: str,
    max_samples: int,
    batches: int,
    batch_size: int,
    seed: int,
    soft_time_budget_sec: float | None,
    checkpoint: str | None,
    report_path: str,
) -> None:
    cfg = build_config(config)
    dataset = load_text_corpus(
        real_text,
        cfg.seq_len,
        max_samples=max_samples,
        sample_alignment="auto",
    )
    run_benchmark(
        cfg,
        dataset,
        batches=batches,
        batch_size=batch_size,
        checkpoint=checkpoint,
        seed=seed,
        time_budget_sec=soft_time_budget_sec,
        progress_report_path=report_path,
    )


def _load_report(report_path: str) -> dict:
    path = Path(report_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _print_report(report: dict, *, config: str, real_text: str, seed: int) -> None:
    summary = report.get("summary", {})
    metric_keys = report.get("metric_keys", [])
    timings_ms = report.get("timings_ms", [])
    print("omen-scale hard-timeout benchmark")
    print(f"device={report.get('device', 'unknown')}")
    print(f"config={config}")
    print("dataset=real_text")
    print(f"batches={report.get('n_batches', 0)}")
    print(f"requested_batches={report.get('requested_batches', 0)}")
    print(f"seed={report.get('protocol_seed', seed)}")
    print(f"timed_out={int(bool(report.get('timed_out', False)))}")
    print(f"stop_reason={report.get('stop_reason', 'unknown')}")
    print(f"hard_timed_out={int(bool(report.get('hard_timed_out', False)))}")
    print(f"hard_stop_reason={report.get('hard_stop_reason', 'completed')}")
    print(f"wall_time_sec={float(report.get('wall_time_sec', 0.0)):.3f}")
    if report.get("time_budget_sec") is not None:
        print(f"time_budget_sec={float(report['time_budget_sec']):.3f}")
    if report.get("hard_wall_time_sec") is not None:
        print(f"hard_wall_time_sec={float(report['hard_wall_time_sec']):.3f}")
    print(f"real_text={real_text}")
    if timings_ms:
        print(f"mean_ms={statistics.mean(timings_ms):.3f}")
        print(f"median_ms={statistics.median(timings_ms):.3f}")
        print(f"max_ms={max(timings_ms):.3f}")
    for key in metric_keys:
        if key in summary:
            print(f"{key}={float(summary[key]):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-timeout real-data OMEN benchmark")
    parser.add_argument("--config", default="demo", choices=["demo", "strong", "mid", "full"])
    parser.add_argument("--real_text", required=True, help="UTF-8 text corpus path")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--soft_time_budget_sec",
        type=float,
        default=None,
        help="Optional graceful in-process budget checked between batches.",
    )
    parser.add_argument(
        "--hard_timeout_sec",
        type=float,
        default=15.0,
        help="Hard wall-clock timeout enforced by the parent process.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="omen-hard-timeout-") as tmpdir:
        report_path = os.path.join(tmpdir, "partial_report.json")
        proc = mp.Process(
            target=_worker,
            kwargs={
                "config": args.config,
                "real_text": args.real_text,
                "max_samples": args.max_samples,
                "batches": args.batches,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "soft_time_budget_sec": args.soft_time_budget_sec,
                "checkpoint": args.checkpoint,
                "report_path": report_path,
            },
        )
        started_at = time.perf_counter()
        proc.start()
        proc.join(timeout=max(float(args.hard_timeout_sec), 0.0))

        hard_timed_out = proc.is_alive()
        if hard_timed_out:
            proc.terminate()
            proc.join(timeout=5.0)

        report = _load_report(report_path)
        hard_wall_time_sec = float(time.perf_counter() - started_at)
        if not report:
            report = {
                "summary": {},
                "timings_ms": [],
                "metric_keys": [],
                "n_batches": 0,
                "requested_batches": int(args.batches),
                "device": "unknown",
                "timed_out": False,
                "stop_reason": "no_partial_report",
                "wall_time_sec": hard_wall_time_sec,
                "time_budget_sec": args.soft_time_budget_sec,
                "protocol_seed": int(args.seed),
            }
        report["soft_stop_reason"] = report.get("stop_reason", "unknown")
        report["hard_timed_out"] = bool(hard_timed_out)
        report["hard_stop_reason"] = (
            "hard_time_budget_exhausted" if hard_timed_out else "completed"
        )
        if hard_timed_out:
            report["timed_out"] = True
            report["stop_reason"] = "hard_time_budget_exhausted"
        report["hard_wall_time_sec"] = hard_wall_time_sec
        _print_report(report, config=args.config, real_text=args.real_text, seed=args.seed)


if __name__ == "__main__":
    mp.freeze_support()
    main()

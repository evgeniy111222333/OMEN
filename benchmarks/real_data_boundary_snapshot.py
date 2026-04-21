from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_omen_scale_eval import DEVICE, build_config
from omen import build_omen
from omen_train_code import load_text_corpus


SNAPSHOT_METRIC_KEYS = (
    "world_alignment",
    "program_anchor",
    "sym_grounding_diagnostic_artifacts",
    "sym_grounding_proposal_artifacts",
    "sym_grounding_support_ratio",
    "sym_grounding_uncertainty",
    "planner_state_actionable_records",
    "planner_state_authoritative_records",
    "planner_state_diagnostic_records",
    "planner_state_diagnostic_symbols",
    "planner_state_supported_verification_records",
    "planner_state_deferred_verification_records",
    "planner_state_conflicted_verification_records",
    "planner_state_deferred_hypotheses",
    "planner_state_conflicted_hypotheses",
    "planner_state_grounding_candidate_rule_records",
)


def _to_scalar(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().item())
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _preview_text(src_row: torch.Tensor, *, limit: int = 160) -> str:
    text = bytes(int(token) for token in src_row.detach().cpu().tolist()).decode(
        "utf-8",
        errors="ignore",
    )
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return text[:limit]


def _load_checkpoint(model, checkpoint: str | None, *, device: torch.device) -> None:
    if not checkpoint:
        return
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)


def _collect_snapshot(
    *,
    config: str,
    real_text: str,
    max_samples: int,
    sample_index: int,
    checkpoint: str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    cfg = build_config(config)
    dataset = load_text_corpus(
        real_text,
        cfg.seq_len,
        max_samples=max_samples,
        sample_alignment="auto",
    )
    if len(dataset) <= 0:
        raise ValueError(f"Real text corpus produced no samples: {real_text}")

    sample_count = len(dataset)
    sample_index = max(0, min(int(sample_index), sample_count - 1))
    src_row, tgt_row = dataset[sample_index]
    preview = _preview_text(src_row)

    active_device = device if device is not None else DEVICE
    model = build_omen(cfg, device=active_device, eval_mode=True)
    _load_checkpoint(model, checkpoint, device=active_device)

    src = src_row.unsqueeze(0).to(active_device)
    tgt = tgt_row.unsqueeze(0).to(active_device)

    if active_device.type == "cuda":
        torch.cuda.synchronize(active_device)
    started_at = time.perf_counter()
    with torch.no_grad():
        out = model(src, tgt)
    if active_device.type == "cuda":
        torch.cuda.synchronize(active_device)
    forward_ms = (time.perf_counter() - started_at) * 1000.0

    report: dict[str, Any] = {
        "device": str(active_device),
        "config": config,
        "real_text": real_text,
        "sample_count": int(sample_count),
        "sample_index": int(sample_index),
        "sample_preview": preview,
        "source_modality": str(out.get("source_modality", "")),
        "source_subtype": str(out.get("source_subtype", "")),
        "source_verification_path": str(out.get("source_verification_path", "")),
        "forward_ms": float(forward_ms),
        "stop_reason": "completed",
    }
    for key in SNAPSHOT_METRIC_KEYS:
        report[key] = _to_scalar(out.get(key, 0.0))
    return report


def _worker(
    *,
    config: str,
    real_text: str,
    max_samples: int,
    sample_index: int,
    checkpoint: str | None,
    report_path: str,
) -> None:
    try:
        report = _collect_snapshot(
            config=config,
            real_text=real_text,
            max_samples=max_samples,
            sample_index=sample_index,
            checkpoint=checkpoint,
        )
    except Exception as exc:  # pragma: no cover - exercised through parent process
        report = {
            "device": str(DEVICE),
            "config": config,
            "real_text": real_text,
            "sample_count": 0,
            "sample_index": int(sample_index),
            "sample_preview": "",
            "forward_ms": 0.0,
            "stop_reason": "worker_exception",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    Path(report_path).write_text(json.dumps(report), encoding="utf-8")


def _load_report(report_path: str) -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _print_report(report: dict[str, Any]) -> None:
    print("omen-scale boundary snapshot")
    print(f"device={report.get('device', 'unknown')}")
    print(f"config={report.get('config', 'unknown')}")
    print(f"real_text={report.get('real_text', '')}")
    print(f"sample_count={int(report.get('sample_count', 0))}")
    print(f"sample_index={int(report.get('sample_index', 0))}")
    print(f"stop_reason={report.get('stop_reason', 'unknown')}")
    print(f"hard_timed_out={int(bool(report.get('hard_timed_out', False)))}")
    print(f"forward_ms={float(report.get('forward_ms', 0.0)):.3f}")
    print(f"sample_preview={json.dumps(report.get('sample_preview', ''))}")
    print(f"source_modality={report.get('source_modality', '')}")
    print(f"source_subtype={report.get('source_subtype', '')}")
    print(f"source_verification_path={report.get('source_verification_path', '')}")
    for key in SNAPSHOT_METRIC_KEYS:
        print(f"{key}={_to_scalar(report.get(key, 0.0)):.6f}")
    if report.get("error"):
        print(f"error={json.dumps(str(report['error']))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-timeout single-sample OMEN boundary snapshot")
    parser.add_argument("--config", default="demo", choices=["demo", "strong", "mid", "full"])
    parser.add_argument("--real_text", required=True, help="UTF-8 text corpus path")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hard_timeout_sec", type=float, default=20.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="omen-boundary-snapshot-") as tmpdir:
        report_path = str(Path(tmpdir) / "snapshot.json")
        proc = mp.Process(
            target=_worker,
            kwargs={
                "config": args.config,
                "real_text": args.real_text,
                "max_samples": args.max_samples,
                "sample_index": args.sample_index,
                "checkpoint": args.checkpoint,
                "report_path": report_path,
            },
        )
        proc.start()
        proc.join(timeout=max(float(args.hard_timeout_sec), 0.0))
        hard_timed_out = proc.is_alive()
        if hard_timed_out:
            proc.terminate()
            proc.join(timeout=5.0)
        report = _load_report(report_path)
        if not report:
            report = {
                "device": str(DEVICE),
                "config": args.config,
                "real_text": args.real_text,
                "sample_count": 0,
                "sample_index": int(args.sample_index),
                "sample_preview": "",
                "forward_ms": 0.0,
                "stop_reason": "no_partial_report",
            }
        report["hard_timed_out"] = bool(hard_timed_out)
        if hard_timed_out:
            report["stop_reason"] = "hard_time_budget_exhausted"
        _print_report(report)


if __name__ == "__main__":
    mp.freeze_support()
    main()

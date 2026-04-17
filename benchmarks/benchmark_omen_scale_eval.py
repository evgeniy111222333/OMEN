from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omen_data import (
    collate,
    make_counting,
    make_javascript,
    make_multilingual_code,
    make_python,
    make_rule_transfer,
    make_rust,
)
from omen import OMEN, build_omen
from omen_scale_config import OMENScaleConfig
from omen_canonical import inject_canonical_metadata, inject_repository_axis_metadata
from omen_train_code import load_text_corpus, make_synthetic_dataset, sample_examples


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_METRIC_KEYS = [
    "total",
    "ce_bits",
    "world_nll",
    "program_anchor",
    "program_target_facts",
    "sym_target_coverage",
    "sym_cycle_checked",
    "z_graph_primary",
    "z_graph_anchor",
    "world_graph_nodes",
    "world_graph_trace_steps",
    "sal_consistency",
    "sal_named_role_preds",
    "creative_analogy_candidates",
    "creative_selected_rules",
    "creative_analogy_projector_loss",
    "creative_analogy_embedding_source",
    "sym_ast_lang_python",
    "sym_ast_lang_javascript",
    "sym_ast_lang_rust",
    "sym_ast_lang_other",
    "n_rules",
]


def build_config(name: str) -> OMENScaleConfig:
    if name == "demo":
        return OMENScaleConfig.demo()
    if name == "strong":
        return OMENScaleConfig.strong()
    if name == "mid":
        return OMENScaleConfig.mid()
    if name == "full":
        return OMENScaleConfig.full()
    raise ValueError(f"Unknown config: {name}")


def build_dataset(args, cfg: OMENScaleConfig):
    if args.real_text:
        return load_text_corpus(args.real_text, cfg.seq_len, max_samples=args.max_samples)
    return make_synthetic_dataset(cfg, n=args.synthetic_samples)


def aggregate(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(row.get(key, 0.0) for row in rows) / len(rows))
        for key in keys
    }


def _collect_metrics(out: Dict[str, object], metric_keys: Iterable[str]) -> Dict[str, float]:
    return {
        key: float(out[key])
        for key in metric_keys
        if key in out and isinstance(out[key], (int, float))
    }


def _evaluate_model(
    model: OMEN,
    dataset,
    *,
    batches: int = 8,
    batch_size: int = 2,
    device: torch.device = DEVICE,
    metric_keys: Iterable[str] = DEFAULT_METRIC_KEYS,
) -> Dict[str, object]:
    metric_rows: List[Dict[str, float]] = []
    timings_ms: List[float] = []

    model.eval()
    with torch.inference_mode():
        for _ in range(max(int(batches), 1)):
            batch = sample_examples(dataset, batch_size)
            if not batch:
                break
            src, tgt = collate(batch)
            src = src.to(device)
            tgt = tgt.to(device)
            t0 = time.perf_counter()
            out = model(src, tgt)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            metric_rows.append(_collect_metrics(out, metric_keys))

    return {
        "summary": aggregate(metric_rows),
        "timings_ms": timings_ms,
        "metric_rows": metric_rows,
        "metric_keys": list(metric_keys),
        "n_batches": len(metric_rows),
        "device": str(device),
    }


def _stamp_canonical_report(
    payload: Dict[str, object],
    *,
    include_repository_axis: bool = False,
) -> Dict[str, object]:
    inject_canonical_metadata(payload)
    if include_repository_axis:
        inject_repository_axis_metadata(payload)
    return payload


def run_benchmark(
    cfg: OMENScaleConfig,
    dataset,
    *,
    batches: int = 8,
    batch_size: int = 2,
    checkpoint: str | None = None,
    device: torch.device = DEVICE,
    metric_keys: Iterable[str] = DEFAULT_METRIC_KEYS,
) -> Dict[str, object]:
    model = build_omen(cfg, device=device)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    report = _evaluate_model(
        model,
        dataset,
        batches=batches,
        batch_size=batch_size,
        device=device,
        metric_keys=metric_keys,
    )
    return _stamp_canonical_report(report, include_repository_axis=True)


def build_transfer_tasks(
    cfg: OMENScaleConfig,
    *,
    synthetic_samples: int = 96,
    real_corpora: Dict[str, str] | None = None,
    real_max_samples: int = 4096,
) -> Dict[str, object]:
    tasks: Dict[str, object] = {
        "counting": make_counting(synthetic_samples, cfg.seq_len),
        "python": make_python(synthetic_samples, cfg.seq_len),
        "javascript": make_javascript(synthetic_samples, cfg.seq_len),
        "rust": make_rust(synthetic_samples, cfg.seq_len),
        "rule_transfer": make_rule_transfer(synthetic_samples, cfg.seq_len),
        "multilingual": make_multilingual_code(synthetic_samples, cfg.seq_len),
    }
    for name, path in (real_corpora or {}).items():
        tasks[str(name)] = load_text_corpus(path, cfg.seq_len, max_samples=real_max_samples)
    return tasks


def load_corpus_protocol(protocol_path: str | Path) -> Dict[str, str]:
    path = Path(protocol_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Corpus protocol must be a JSON object mapping task names to file paths")
    protocol: Dict[str, str] = {}
    for name, raw_path in payload.items():
        if not isinstance(name, str) or not isinstance(raw_path, str):
            raise ValueError("Corpus protocol entries must be string -> string")
        resolved = (path.parent / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
        protocol[name] = str(resolved)
    return protocol


def build_real_transfer_tasks(
    cfg: OMENScaleConfig,
    protocol: Mapping[str, str],
    *,
    max_samples: int = 4096,
) -> Dict[str, object]:
    tasks: Dict[str, object] = {}
    for name, file_path in protocol.items():
        tasks[str(name)] = load_text_corpus(str(file_path), cfg.seq_len, max_samples=max_samples)
    return tasks


def _adapt_model(
    model: OMEN,
    dataset,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    lr: float,
) -> List[Dict[str, float]]:
    if steps <= 0:
        return []
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    history: List[Dict[str, float]] = []
    model.train()
    for _ in range(max(int(steps), 0)):
        batch = sample_examples(dataset, batch_size)
        if not batch:
            break
        src, tgt = collate(batch)
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(src, tgt)
        loss = out.get("total")
        if not torch.is_tensor(loss) or not torch.isfinite(loss).item():
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.memory.maybe_flush()
        history.append(_collect_metrics(out, ("total", "ce_bits", "world_nll", "sym_target_coverage")))
    model.eval()
    return history


def run_transfer_suite(
    cfg: OMENScaleConfig,
    tasks: Dict[str, object] | None = None,
    *,
    source_task: str = "python",
    adapt_steps: int = 1,
    eval_batches: int = 1,
    batch_size: int = 1,
    lr: float = 3e-4,
    checkpoint: str | None = None,
    device: torch.device = DEVICE,
    force_creative_cycle: bool = True,
    metric_keys: Iterable[str] = DEFAULT_METRIC_KEYS,
) -> Dict[str, object]:
    suite_cfg = deepcopy(cfg)
    if force_creative_cycle:
        suite_cfg.creative_cycle_every = 1

    model = build_omen(suite_cfg, device=device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    task_map = tasks or build_transfer_tasks(suite_cfg)
    if not task_map:
        return _stamp_canonical_report({
            "source_task": source_task,
            "adapt_steps": int(adapt_steps),
            "adapt_history": [],
            "task_reports": {},
            "task_summaries": {},
            "aggregate_summary": {},
            "transfer_deltas": {},
            "n_tasks": 0,
            "device": str(device),
        }, include_repository_axis=True)

    if source_task not in task_map:
        source_task = next(iter(task_map.keys()))

    adapt_history = _adapt_model(
        model,
        task_map[source_task],
        steps=adapt_steps,
        batch_size=batch_size,
        device=device,
        lr=lr,
    )

    task_reports: Dict[str, Dict[str, object]] = {}
    task_summaries: Dict[str, Dict[str, float]] = {}
    for name, dataset in task_map.items():
        report = _evaluate_model(
            model,
            dataset,
            batches=eval_batches,
            batch_size=batch_size,
            device=device,
            metric_keys=metric_keys,
        )
        task_reports[name] = report
        task_summaries[name] = report["summary"]

    aggregate_summary = aggregate(task_summaries.values())
    source_summary = task_summaries.get(source_task, {})
    transfer_deltas: Dict[str, Dict[str, float]] = {}
    tracked = (
        "ce_bits",
        "world_nll",
        "sym_target_coverage",
        "program_anchor",
        "world_graph_trace_steps",
        "creative_analogy_embedding_source",
        "sym_ast_lang_python",
        "sym_ast_lang_javascript",
        "sym_ast_lang_rust",
    )
    for name, summary in task_summaries.items():
        if name == source_task:
            continue
        delta: Dict[str, float] = {}
        for key in tracked:
            if key in summary and key in source_summary:
                delta[f"{key}_delta"] = float(summary[key] - source_summary[key])
        transfer_deltas[name] = delta

    return _stamp_canonical_report({
        "source_task": source_task,
        "adapt_steps": int(adapt_steps),
        "adapt_history": adapt_history,
        "task_reports": task_reports,
        "task_summaries": task_summaries,
        "aggregate_summary": aggregate_summary,
        "transfer_deltas": transfer_deltas,
        "n_tasks": len(task_summaries),
        "device": str(device),
    }, include_repository_axis=True)


def run_corpus_protocol(
    cfg: OMENScaleConfig,
    protocol: Mapping[str, str] | str | Path,
    *,
    source_task: str | None = None,
    adapt_steps: int = 1,
    eval_batches: int = 1,
    batch_size: int = 1,
    lr: float = 3e-4,
    max_samples: int = 4096,
    checkpoint: str | None = None,
    device: torch.device = DEVICE,
    force_creative_cycle: bool = True,
    metric_keys: Iterable[str] = DEFAULT_METRIC_KEYS,
) -> Dict[str, object]:
    protocol_map = (
        load_corpus_protocol(protocol)
        if isinstance(protocol, (str, Path))
        else {str(name): str(path) for name, path in protocol.items()}
    )
    tasks = build_real_transfer_tasks(cfg, protocol_map, max_samples=max_samples)
    source = source_task or (next(iter(tasks.keys())) if tasks else "")
    report = run_transfer_suite(
        cfg,
        tasks=tasks,
        source_task=source,
        adapt_steps=adapt_steps,
        eval_batches=eval_batches,
        batch_size=batch_size,
        lr=lr,
        checkpoint=checkpoint,
        device=device,
        force_creative_cycle=force_creative_cycle,
        metric_keys=metric_keys,
    )
    report["corpus_protocol"] = dict(protocol_map)
    report["real_protocol_tasks"] = float(len(protocol_map))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical OMEN-Scale evaluation benchmark")
    parser.add_argument("--config", default="demo", choices=["demo", "strong", "mid", "full"])
    parser.add_argument("--checkpoint", default=None, help="Optional model checkpoint (.pt/.pth)")
    parser.add_argument("--real_text", default=None, help="Optional UTF-8 text corpus path")
    parser.add_argument("--real_manifest", default=None, help="JSON task->corpus manifest for real transfer protocol")
    parser.add_argument("--max_samples", type=int, default=4096)
    parser.add_argument("--synthetic_samples", type=int, default=96)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    cfg = build_config(args.config)
    if args.real_manifest:
        report = run_corpus_protocol(
            cfg,
            args.real_manifest,
            adapt_steps=1,
            eval_batches=args.batches,
            batch_size=args.batch_size,
            checkpoint=args.checkpoint,
            max_samples=args.max_samples,
            device=DEVICE,
        )
        print("omen-scale corpus protocol")
        print(f"device={report['device']}")
        print(f"config={args.config}")
        print(f"canonical_stack={report['canonical_stack']}")
        print(f"tasks={report['n_tasks']}")
        print(f"source_task={report['source_task']}")
        for key, value in sorted(report["aggregate_summary"].items()):
            print(f"{key}={value:.6f}")
        return

    dataset = build_dataset(args, cfg)
    report = run_benchmark(
        cfg,
        dataset,
        batches=args.batches,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        device=DEVICE,
    )
    summary = report["summary"]
    metric_keys = report["metric_keys"]
    timings_ms = report["timings_ms"]
    print("omen-scale benchmark")
    print(f"device={report['device']}")
    print(f"config={args.config}")
    print(f"dataset={'real_text' if args.real_text else 'synthetic'}")
    print(f"batches={report['n_batches']}")
    if args.real_text:
        print(f"real_text={args.real_text}")
    if timings_ms:
        print(f"mean_ms={statistics.mean(timings_ms):.3f}")
        print(f"median_ms={statistics.median(timings_ms):.3f}")
        print(f"max_ms={max(timings_ms):.3f}")
    for key in metric_keys:
        if key in summary:
            print(f"{key}={summary[key]:.6f}")


if __name__ == "__main__":
    main()

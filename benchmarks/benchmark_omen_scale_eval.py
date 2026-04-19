from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

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
    "world_causal_error",
    "world_alignment",
    "program_anchor",
    "program_target_facts",
    "sym_target_coverage",
    "sym_cycle_active",
    "sym_cycle_checked",
    "sym_cycle_eval_active",
    "sym_cycle_learning_active",
    "sym_cycle_trace_candidates",
    "sym_cycle_contextual_candidates",
    "sym_cycle_neural_candidates",
    "z_graph_primary",
    "z_graph_anchor",
    "world_graph_nodes",
    "world_graph_trace_steps",
    "world_graph_execution_steps",
    "world_graph_hidden_fallback_steps",
    "world_graph_hidden_teacher_applied",
    "world_graph_neutral_prior_applied",
    "world_graph_signature_encoder_active",
    "world_graph_context_facts",
    "world_graph_memory_facts",
    "world_graph_net_facts",
    "world_graph_abduced_support_facts",
    "world_graph_observed_now_facts",
    "world_graph_semantic_graph_enriched",
    "z_posterior_graph_native",
    "z_posterior_perceiver_fallback",
    "world_graph_transition_native",
    "world_graph_graph_dense_view_derived",
    "world_graph_neural_residual_used",
    "eval_world_self_update_applied",
    "eval_world_self_update_loss",
    "creative_oee_online_train_applied",
    "creative_oee_online_train_loss",
    "sal_consistency",
    "sal_named_role_preds",
    "sal_named_role_rel_preds",
    "creative_analogy_candidates",
    "creative_cycle_active",
    "creative_selected_rules",
    "creative_validated_selected_rules",
    "creative_validated_support_facts",
    "creative_target_support_before",
    "creative_target_support_after",
    "creative_target_support_gain",
    "creative_gap_before",
    "creative_gap_after",
    "creative_gap_reduction",
    "creative_compression_gain",
    "creative_analogy_projector_loss",
    "creative_analogy_embedding_source",
    "sym_observed_now_facts",
    "sym_memory_derived_facts",
    "sym_saliency_derived_facts",
    "sym_net_derived_facts",
    "sym_world_context_facts",
    "sym_abduced_support_facts",
    "sym_world_context_summary_entries",
    "sym_ast_lang_python",
    "sym_ast_lang_javascript",
    "sym_ast_lang_rust",
    "sym_ast_lang_other",
    "n_rules",
]

CREATIVE_ABLATION_KEYS = (
    "ce_bits",
    "world_nll",
    "sym_target_coverage",
    "world_graph_trace_steps",
    "sym_world_context_facts",
    "creative_cycle_active",
    "creative_selected_rules",
    "creative_validated_selected_rules",
    "creative_validated_support_facts",
    "creative_target_support_after",
    "creative_target_support_gain",
    "creative_gap_reduction",
    "creative_compression_gain",
)


def _set_protocol_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


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


def _encode_text_examples(texts: Iterable[str], seq_len: int) -> List[torch.Tensor]:
    samples: List[torch.Tensor] = []
    for text in texts:
        encoded = list(text.encode("utf-8", errors="ignore"))[:seq_len]
        if len(encoded) < seq_len:
            encoded = encoded + [0] * (seq_len - len(encoded))
        samples.append(torch.tensor(encoded, dtype=torch.long))
    return samples


def make_observation_text(seq_len: int, n: int = 96) -> List[torch.Tensor]:
    templates = [
        "weather is rain. rain becomes flood. however flood is not safe.",
        "signal is amber. amber becomes red. target evacuation must start.",
        "sensor is hot. hot causes alarm. alarm leads to shutdown.",
        "market is open. volume becomes high. however spread is not stable.",
    ]
    return _encode_text_examples((templates[idx % len(templates)] for idx in range(max(int(n), 1))), seq_len)


def make_observation_structured(seq_len: int, n: int = 96) -> List[torch.Tensor]:
    templates = [
        "step1: weather=rain, road=wet\nstep2: road=closed, alert=yellow\ntarget evacuation=safe_exit",
        '{"step":1,"tank":"full","valve":"closed"}\n{"step":2,"valve":"open","flow":"high"}\n{"goal":"pressure","value":"stable"}',
        "state: user=guest, access=limited\nnext: user=admin, access=full\ngoal access=audited",
        "1) sensor=temp_high, fan=off\n2) fan=on, temp=drop\n3) target status=stable",
    ]
    return _encode_text_examples((templates[idx % len(templates)] for idx in range(max(int(n), 1))), seq_len)


def build_dataset(args, cfg: OMENScaleConfig):
    if args.real_text:
        return load_text_corpus(args.real_text, cfg.seq_len, max_samples=args.max_samples)
    _set_protocol_seed(getattr(args, "seed", None))
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


def aggregate_weighted(metrics: Iterable[tuple[Dict[str, float], float]]) -> Dict[str, float]:
    rows = [(row, float(weight)) for row, weight in metrics if row]
    if not rows:
        return {}
    total_weight = sum(max(weight, 0.0) for _, weight in rows)
    if total_weight <= 0.0:
        return aggregate(row for row, _ in rows)
    keys = sorted({key for row, _ in rows for key in row})
    return {
        key: float(
            sum(row.get(key, 0.0) * max(weight, 0.0) for row, weight in rows) / total_weight
        )
        for key in keys
    }


def _normalize_time_budget(time_budget_sec: float | None) -> float | None:
    if time_budget_sec is None:
        return None
    return max(float(time_budget_sec), 0.0)


def _time_budget_deadline(
    time_budget_sec: float | None,
    *,
    start_time: float | None = None,
) -> float | None:
    budget = _normalize_time_budget(time_budget_sec)
    if budget is None:
        return None
    origin = time.perf_counter() if start_time is None else float(start_time)
    return origin + budget


def _budget_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _write_json_report(path: str | Path, payload: Mapping[str, object]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _coerce_metric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        try:
            return float(value.detach().item())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _collect_metrics(out: Dict[str, object], metric_keys: Iterable[str]) -> Dict[str, float]:
    collected: Dict[str, float] = {}
    for key in metric_keys:
        if key not in out:
            continue
        scalar = _coerce_metric_value(out[key])
        if scalar is not None:
            collected[key] = scalar
    return collected


def _evaluate_model(
    model: OMEN,
    dataset,
    *,
    batches: int = 8,
    batch_size: int = 2,
    device: torch.device = DEVICE,
    metric_keys: Iterable[str] = DEFAULT_METRIC_KEYS,
    deadline: float | None = None,
    progress_callback: Callable[[Dict[str, object]], None] | None = None,
) -> Dict[str, object]:
    metric_rows: List[Dict[str, float]] = []
    timings_ms: List[float] = []
    requested_batches = max(int(batches), 1)
    stop_reason = "completed"

    model.eval()
    with torch.inference_mode():
        for _ in range(requested_batches):
            if _budget_expired(deadline):
                stop_reason = "time_budget_exhausted"
                break
            batch = sample_examples(dataset, batch_size)
            if not batch:
                stop_reason = "dataset_exhausted"
                break
            src, tgt = collate(batch)
            src = src.to(device)
            tgt = tgt.to(device)
            t0 = time.perf_counter()
            out = model(src, tgt)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            metric_rows.append(_collect_metrics(out, metric_keys))
            timed_out_after_batch = _budget_expired(deadline)
            if timed_out_after_batch:
                stop_reason = "time_budget_exhausted"
            if progress_callback is not None:
                progress_callback(
                    {
                        "summary": aggregate(metric_rows),
                        "timings_ms": list(timings_ms),
                        "metric_rows": list(metric_rows),
                        "metric_keys": list(metric_keys),
                        "n_batches": len(metric_rows),
                        "requested_batches": requested_batches,
                        "timed_out": timed_out_after_batch,
                        "stop_reason": "time_budget_exhausted" if timed_out_after_batch else "running",
                        "device": str(device),
                    }
                )
            if timed_out_after_batch:
                break

    return {
        "summary": aggregate(metric_rows),
        "timings_ms": timings_ms,
        "metric_rows": metric_rows,
        "metric_keys": list(metric_keys),
        "n_batches": len(metric_rows),
        "requested_batches": requested_batches,
        "timed_out": stop_reason == "time_budget_exhausted",
        "stop_reason": stop_reason,
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


def _resolve_protocol_path(raw_path: str, base_dir: Path | None = None) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        path = (base_dir / path).resolve()
    return str(path)


def _coerce_protocol_entry(
    name: str,
    raw_entry: str | Mapping[str, object],
    *,
    base_dir: Path | None = None,
) -> Dict[str, object]:
    if isinstance(raw_entry, str):
        entry: Dict[str, object] = {"path": raw_entry}
    elif isinstance(raw_entry, Mapping):
        entry = dict(raw_entry)
    else:
        raise ValueError("Corpus protocol entries must be either strings or JSON objects")

    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Corpus protocol entry '{name}' must define a non-empty 'path'")

    language = entry.get("language")
    source = entry.get("source")
    family = entry.get("family")
    split = entry.get("split")
    weight = entry.get("weight", 1.0)
    tags = entry.get("tags", ())
    if not isinstance(weight, (int, float)):
        raise ValueError(f"Corpus protocol entry '{name}' has non-numeric weight")
    if tags is None:
        tags = ()
    if not isinstance(tags, (list, tuple)):
        raise ValueError(f"Corpus protocol entry '{name}' must use a list for 'tags'")

    return {
        "path": _resolve_protocol_path(raw_path, base_dir=base_dir),
        "language": str(language) if language else "unknown",
        "source": str(source) if source else "local",
        "family": str(family) if family else "unclassified",
        "split": str(split) if split else "unspecified",
        "weight": float(weight),
        "tags": tuple(str(tag) for tag in tags),
    }


def coerce_corpus_protocol(
    protocol: Mapping[str, str | Mapping[str, object]],
    *,
    base_dir: Path | None = None,
) -> Dict[str, Dict[str, object]]:
    if not isinstance(protocol, Mapping):
        raise ValueError("Corpus protocol must be a mapping of task names to paths/specs")
    entries: Dict[str, Dict[str, object]] = {}
    for name, raw_entry in protocol.items():
        if not isinstance(name, str):
            raise ValueError("Corpus protocol task names must be strings")
        entries[name] = _coerce_protocol_entry(name, raw_entry, base_dir=base_dir)
    return entries


def _group_protocol_summaries(
    task_summaries: Mapping[str, Dict[str, float]],
    protocol: Mapping[str, Mapping[str, object]],
    field: str,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[tuple[Dict[str, float], float]]] = {}
    for task_name, summary in task_summaries.items():
        spec = protocol.get(task_name, {})
        group_key = str(spec.get(field, "unknown"))
        grouped.setdefault(group_key, []).append((summary, float(spec.get("weight", 1.0))))
    return {
        group_key: aggregate_weighted(group_rows)
        for group_key, group_rows in grouped.items()
    }


def _protocol_taxonomy(protocol: Mapping[str, Mapping[str, object]], field: str) -> List[str]:
    return sorted({str(spec.get(field, "unknown")) for spec in protocol.values()})


def run_benchmark(
    cfg: OMENScaleConfig,
    dataset,
    *,
    batches: int = 8,
    batch_size: int = 2,
    checkpoint: str | None = None,
    device: torch.device = DEVICE,
    metric_keys: Iterable[str] = DEFAULT_METRIC_KEYS,
    seed: int | None = None,
    time_budget_sec: float | None = None,
    progress_report_path: str | Path | None = None,
) -> Dict[str, object]:
    started_at = time.perf_counter()
    deadline = _time_budget_deadline(time_budget_sec, start_time=started_at)
    budget = _normalize_time_budget(time_budget_sec)
    active_seed = _set_protocol_seed(seed)
    model = build_omen(cfg, device=device)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    def emit_progress(partial_report: Dict[str, object]) -> None:
        if progress_report_path is None:
            return
        payload = dict(partial_report)
        payload["wall_time_sec"] = float(time.perf_counter() - started_at)
        payload["time_budget_sec"] = budget
        if active_seed is not None:
            payload["protocol_seed"] = int(active_seed)
        _write_json_report(
            progress_report_path,
            _stamp_canonical_report(payload, include_repository_axis=True),
        )

    report = _evaluate_model(
        model,
        dataset,
        batches=batches,
        batch_size=batch_size,
        device=device,
        metric_keys=metric_keys,
        deadline=deadline,
        progress_callback=emit_progress if progress_report_path is not None else None,
    )
    report["wall_time_sec"] = float(time.perf_counter() - started_at)
    report["time_budget_sec"] = budget
    if active_seed is not None:
        report["protocol_seed"] = int(active_seed)
    final_report = _stamp_canonical_report(report, include_repository_axis=True)
    if progress_report_path is not None:
        _write_json_report(progress_report_path, final_report)
    return final_report


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
        "observation_text": make_observation_text(cfg.seq_len, synthetic_samples),
        "observation_structured": make_observation_structured(cfg.seq_len, synthetic_samples),
    }
    for name, path in (real_corpora or {}).items():
        tasks[str(name)] = load_text_corpus(path, cfg.seq_len, max_samples=real_max_samples)
    return tasks


def load_corpus_protocol(protocol_path: str | Path) -> Dict[str, str]:
    path = Path(protocol_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return coerce_corpus_protocol(payload, base_dir=path.parent)


def build_real_transfer_tasks(
    cfg: OMENScaleConfig,
    protocol: Mapping[str, str | Mapping[str, object]],
    *,
    max_samples: int = 4096,
) -> Dict[str, object]:
    protocol_entries = coerce_corpus_protocol(protocol)
    tasks: Dict[str, object] = {}
    for name, spec in protocol_entries.items():
        tasks[str(name)] = load_text_corpus(str(spec["path"]), cfg.seq_len, max_samples=max_samples)
    return tasks


def _adapt_model(
    model: OMEN,
    dataset,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    lr: float,
    deadline: float | None = None,
) -> Dict[str, object]:
    requested_steps = max(int(steps), 0)
    if requested_steps <= 0:
        return {
            "history": [],
            "requested_steps": 0,
            "n_steps": 0,
            "timed_out": False,
            "stop_reason": "disabled",
        }
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    history: List[Dict[str, float]] = []
    stop_reason = "completed"
    model.train()
    for _ in range(requested_steps):
        if _budget_expired(deadline):
            stop_reason = "time_budget_exhausted"
            break
        batch = sample_examples(dataset, batch_size)
        if not batch:
            stop_reason = "dataset_exhausted"
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
        if _budget_expired(deadline):
            stop_reason = "time_budget_exhausted"
            break
    model.eval()
    return {
        "history": history,
        "requested_steps": requested_steps,
        "n_steps": len(history),
        "timed_out": stop_reason == "time_budget_exhausted",
        "stop_reason": stop_reason,
    }


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
    seed: int | None = None,
    time_budget_sec: float | None = None,
) -> Dict[str, object]:
    started_at = time.perf_counter()
    deadline = _time_budget_deadline(time_budget_sec, start_time=started_at)
    budget = _normalize_time_budget(time_budget_sec)
    active_seed = _set_protocol_seed(seed)
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
        report = {
            "source_task": source_task,
            "adapt_steps": int(adapt_steps),
            "adapt_history": [],
            "task_reports": {},
            "task_summaries": {},
            "aggregate_summary": {},
            "transfer_deltas": {},
            "n_tasks": 0,
            "requested_tasks": 0,
            "device": str(device),
            "timed_out": False,
            "stop_reason": "empty_task_map",
            "wall_time_sec": float(time.perf_counter() - started_at),
            "time_budget_sec": budget,
        }
        if active_seed is not None:
            report["protocol_seed"] = int(active_seed)
        return _stamp_canonical_report(report, include_repository_axis=True)

    if source_task not in task_map:
        source_task = next(iter(task_map.keys()))

    adapt_report = _adapt_model(
        model,
        task_map[source_task],
        steps=adapt_steps,
        batch_size=batch_size,
        device=device,
        lr=lr,
        deadline=deadline,
    )

    task_reports: Dict[str, Dict[str, object]] = {}
    task_summaries: Dict[str, Dict[str, float]] = {}
    timed_out = bool(adapt_report["timed_out"])
    stop_reason = str(adapt_report["stop_reason"])
    for name, dataset in task_map.items():
        if _budget_expired(deadline):
            timed_out = True
            stop_reason = "time_budget_exhausted"
            break
        report = _evaluate_model(
            model,
            dataset,
            batches=eval_batches,
            batch_size=batch_size,
            device=device,
            metric_keys=metric_keys,
            deadline=deadline,
        )
        task_reports[name] = report
        task_summaries[name] = report["summary"]
        if report["timed_out"]:
            timed_out = True
            stop_reason = str(report["stop_reason"])
            break
        if report["stop_reason"] != "completed":
            stop_reason = str(report["stop_reason"])
            break

    if not timed_out and stop_reason in {"completed", "disabled"}:
        stop_reason = "completed"

    aggregate_summary = aggregate(task_summaries.values())
    source_summary = task_summaries.get(source_task, {})
    transfer_deltas: Dict[str, Dict[str, float]] = {}
    tracked = (
        "ce_bits",
        "world_nll",
        "sym_target_coverage",
        "program_anchor",
        "world_graph_trace_steps",
        "world_graph_execution_steps",
        "sym_world_context_facts",
        "creative_analogy_embedding_source",
        "creative_cycle_active",
        "creative_validated_selected_rules",
        "creative_target_support_gain",
        "creative_gap_reduction",
        "sym_ast_lang_python",
        "sym_ast_lang_javascript",
        "sym_ast_lang_rust",
        "sym_ast_lang_other",
    )
    for name, summary in task_summaries.items():
        if name == source_task:
            continue
        delta: Dict[str, float] = {}
        for key in tracked:
            if key in summary and key in source_summary:
                delta[f"{key}_delta"] = float(summary[key] - source_summary[key])
        transfer_deltas[name] = delta

    report = {
        "source_task": source_task,
        "adapt_steps": int(adapt_steps),
        "adapt_history": adapt_report["history"],
        "adapt_requested_steps": adapt_report["requested_steps"],
        "adapt_completed_steps": adapt_report["n_steps"],
        "adapt_timed_out": adapt_report["timed_out"],
        "adapt_stop_reason": adapt_report["stop_reason"],
        "task_reports": task_reports,
        "task_summaries": task_summaries,
        "aggregate_summary": aggregate_summary,
        "transfer_deltas": transfer_deltas,
        "n_tasks": len(task_summaries),
        "requested_tasks": len(task_map),
        "device": str(device),
        "timed_out": timed_out,
        "stop_reason": stop_reason,
        "wall_time_sec": float(time.perf_counter() - started_at),
        "time_budget_sec": budget,
    }
    if active_seed is not None:
        report["protocol_seed"] = int(active_seed)
    return _stamp_canonical_report(report, include_repository_axis=True)


def _aggregate_selected_metrics(summary: Mapping[str, float], keys: Iterable[str]) -> Dict[str, float]:
    return {
        key: float(summary[key])
        for key in keys
        if key in summary
    }


def _creative_task_deltas(
    enabled_summaries: Mapping[str, Dict[str, float]],
    disabled_summaries: Mapping[str, Dict[str, float]],
    keys: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    deltas: Dict[str, Dict[str, float]] = {}
    for task_name, enabled_summary in enabled_summaries.items():
        disabled_summary = disabled_summaries.get(task_name, {})
        task_delta: Dict[str, float] = {}
        for key in keys:
            if key in enabled_summary or key in disabled_summary:
                task_delta[f"{key}_delta"] = float(enabled_summary.get(key, 0.0) - disabled_summary.get(key, 0.0))
        deltas[task_name] = task_delta
    return deltas


def run_creative_ablation_suite(
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
    metric_keys: Iterable[str] = DEFAULT_METRIC_KEYS,
    seed: int | None = None,
    time_budget_sec: float | None = None,
) -> Dict[str, object]:
    started_at = time.perf_counter()
    task_map = tasks or build_transfer_tasks(cfg)

    enabled_cfg = deepcopy(cfg)
    enabled_cfg.creative_cycle_enabled = True
    enabled_cfg.creative_cycle_every = 1

    disabled_cfg = deepcopy(cfg)
    disabled_cfg.creative_cycle_enabled = False

    enabled_report = run_transfer_suite(
        enabled_cfg,
        tasks=task_map,
        source_task=source_task,
        adapt_steps=adapt_steps,
        eval_batches=eval_batches,
        batch_size=batch_size,
        lr=lr,
        checkpoint=checkpoint,
        device=device,
        force_creative_cycle=True,
        metric_keys=metric_keys,
        seed=seed,
        time_budget_sec=time_budget_sec,
    )
    disabled_report = run_transfer_suite(
        disabled_cfg,
        tasks=task_map,
        source_task=source_task,
        adapt_steps=adapt_steps,
        eval_batches=eval_batches,
        batch_size=batch_size,
        lr=lr,
        checkpoint=checkpoint,
        device=device,
        force_creative_cycle=False,
        metric_keys=metric_keys,
        seed=seed,
        time_budget_sec=time_budget_sec,
    )

    enabled_summary = _aggregate_selected_metrics(enabled_report.get("aggregate_summary", {}), CREATIVE_ABLATION_KEYS)
    disabled_summary = _aggregate_selected_metrics(disabled_report.get("aggregate_summary", {}), CREATIVE_ABLATION_KEYS)
    aggregate_delta = {
        f"{key}_delta": float(enabled_summary.get(key, 0.0) - disabled_summary.get(key, 0.0))
        for key in CREATIVE_ABLATION_KEYS
        if key in enabled_summary or key in disabled_summary
    }
    report = {
        "creative_enabled": enabled_report,
        "creative_disabled": disabled_report,
        "aggregate_enabled_summary": enabled_summary,
        "aggregate_disabled_summary": disabled_summary,
        "aggregate_delta": aggregate_delta,
        "task_deltas": _creative_task_deltas(
            enabled_report.get("task_summaries", {}),
            disabled_report.get("task_summaries", {}),
            CREATIVE_ABLATION_KEYS,
        ),
        "n_tasks": int(enabled_report.get("n_tasks", 0)),
        "device": str(device),
        "wall_time_sec": float(time.perf_counter() - started_at),
        "time_budget_sec": _normalize_time_budget(time_budget_sec),
        "timed_out": bool(enabled_report.get("timed_out", False) or disabled_report.get("timed_out", False)),
        "stop_reason": "completed",
    }
    if report["timed_out"]:
        report["stop_reason"] = "time_budget_exhausted"
    if seed is not None:
        report["protocol_seed"] = int(seed)
    return _stamp_canonical_report(report, include_repository_axis=True)


def run_corpus_protocol(
    cfg: OMENScaleConfig,
    protocol: Mapping[str, str | Mapping[str, object]] | str | Path,
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
    seed: int | None = None,
    time_budget_sec: float | None = None,
) -> Dict[str, object]:
    protocol_entries = (
        load_corpus_protocol(protocol)
        if isinstance(protocol, (str, Path))
        else coerce_corpus_protocol(protocol)
    )
    tasks = build_real_transfer_tasks(cfg, protocol_entries, max_samples=max_samples)
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
        seed=seed,
        time_budget_sec=time_budget_sec,
    )
    weighted_summary = aggregate_weighted(
        (report["task_summaries"].get(task_name, {}), float(spec.get("weight", 1.0)))
        for task_name, spec in protocol_entries.items()
    )
    report["corpus_protocol"] = {name: dict(spec) for name, spec in protocol_entries.items()}
    report["corpus_protocol_paths"] = {
        name: str(spec["path"]) for name, spec in protocol_entries.items()
    }
    report["protocol_languages"] = _protocol_taxonomy(protocol_entries, "language")
    report["protocol_families"] = _protocol_taxonomy(protocol_entries, "family")
    report["protocol_sources"] = _protocol_taxonomy(protocol_entries, "source")
    report["protocol_splits"] = _protocol_taxonomy(protocol_entries, "split")
    report["protocol_language_summaries"] = _group_protocol_summaries(
        report["task_summaries"], protocol_entries, "language"
    )
    report["protocol_family_summaries"] = _group_protocol_summaries(
        report["task_summaries"], protocol_entries, "family"
    )
    report["protocol_source_summaries"] = _group_protocol_summaries(
        report["task_summaries"], protocol_entries, "source"
    )
    report["protocol_weighted_summary"] = weighted_summary
    report["real_protocol_tasks"] = float(len(protocol_entries))
    report["real_protocol_languages"] = float(len(report["protocol_languages"]))
    report["real_protocol_families"] = float(len(report["protocol_families"]))
    report["real_protocol_sources"] = float(len(report["protocol_sources"]))
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
    parser.add_argument("--adapt_steps", type=int, default=1)
    parser.add_argument(
        "--creative_ablation",
        action="store_true",
        help="Run creative-enabled vs creative-disabled transfer ablation on the current task set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--time_budget_sec",
        type=float,
        default=None,
        help="Optional wall-clock budget for the whole run. Returns a partial report instead of hanging.",
    )
    args = parser.parse_args()

    cfg = build_config(args.config)
    if args.real_manifest:
        report = run_corpus_protocol(
            cfg,
            args.real_manifest,
            adapt_steps=args.adapt_steps,
            eval_batches=args.batches,
            batch_size=args.batch_size,
            checkpoint=args.checkpoint,
            max_samples=args.max_samples,
            device=DEVICE,
            seed=args.seed,
            time_budget_sec=args.time_budget_sec,
        )
        print("omen-scale corpus protocol")
        print(f"device={report['device']}")
        print(f"config={args.config}")
        print(f"canonical_stack={report['canonical_stack']}")
        print(f"tasks={report['n_tasks']}")
        print(f"requested_tasks={report['requested_tasks']}")
        print(f"source_task={report['source_task']}")
        print(f"adapt_completed_steps={report['adapt_completed_steps']}")
        print(f"seed={report.get('protocol_seed', args.seed)}")
        print(f"timed_out={int(report['timed_out'])}")
        print(f"stop_reason={report['stop_reason']}")
        print(f"wall_time_sec={report['wall_time_sec']:.3f}")
        if report.get("time_budget_sec") is not None:
            print(f"time_budget_sec={report['time_budget_sec']:.3f}")
        print(f"languages={','.join(report['protocol_languages'])}")
        print(f"families={','.join(report['protocol_families'])}")
        for key, value in sorted(report["aggregate_summary"].items()):
            print(f"{key}={value:.6f}")
        return

    if args.creative_ablation:
        if args.real_text:
            tasks = {
                "real_text": load_text_corpus(args.real_text, cfg.seq_len, max_samples=args.max_samples),
                "observation_text": make_observation_text(cfg.seq_len, max(args.batch_size * args.batches, 4)),
                "observation_structured": make_observation_structured(cfg.seq_len, max(args.batch_size * args.batches, 4)),
            }
        else:
            tasks = build_transfer_tasks(cfg, synthetic_samples=args.synthetic_samples)
        report = run_creative_ablation_suite(
            cfg,
            tasks=tasks,
            source_task=next(iter(tasks.keys())) if tasks else "python",
            adapt_steps=args.adapt_steps,
            eval_batches=args.batches,
            batch_size=args.batch_size,
            checkpoint=args.checkpoint,
            device=DEVICE,
            seed=args.seed,
            time_budget_sec=args.time_budget_sec,
        )
        print("omen-scale creative ablation")
        print(f"device={report['device']}")
        print(f"config={args.config}")
        print(f"tasks={report['n_tasks']}")
        print(f"seed={report.get('protocol_seed', args.seed)}")
        print(f"timed_out={int(report['timed_out'])}")
        print(f"stop_reason={report['stop_reason']}")
        print(f"wall_time_sec={report['wall_time_sec']:.3f}")
        if report.get("time_budget_sec") is not None:
            print(f"time_budget_sec={report['time_budget_sec']:.3f}")
        for key, value in sorted(report["aggregate_delta"].items()):
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
        seed=args.seed,
        time_budget_sec=args.time_budget_sec,
    )
    summary = report["summary"]
    metric_keys = report["metric_keys"]
    timings_ms = report["timings_ms"]
    print("omen-scale benchmark")
    print(f"device={report['device']}")
    print(f"config={args.config}")
    print(f"dataset={'real_text' if args.real_text else 'synthetic'}")
    print(f"batches={report['n_batches']}")
    print(f"requested_batches={report['requested_batches']}")
    print(f"seed={report.get('protocol_seed', args.seed)}")
    print(f"timed_out={int(report['timed_out'])}")
    print(f"stop_reason={report['stop_reason']}")
    print(f"wall_time_sec={report['wall_time_sec']:.3f}")
    if report.get("time_budget_sec") is not None:
        print(f"time_budget_sec={report['time_budget_sec']:.3f}")
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

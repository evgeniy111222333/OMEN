from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .memory_hints import grounding_memory_status


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _graph_key(record: Any) -> str:
    key = str(getattr(record, "graph_key", "") or "").strip()
    if key:
        return key
    return repr(record)


def _graph_family(record: Any) -> str:
    return str(getattr(record, "graph_family", "") or "").strip().lower()


def grounding_memory_priority(record: Any) -> float:
    status = grounding_memory_status(record)
    family = _graph_family(record)
    confidence = _clip01(getattr(record, "confidence", 0.0))
    support = _clip01(getattr(record, "support", 0.0))
    conflict = _clip01(getattr(record, "conflict", 0.0))
    status_bias = {
        "active": 1.00,
        "contradicted": 0.94,
        "hypothetical": 0.88,
        "unknown": 0.50,
    }.get(status, 0.50)
    family_bias = 0.0
    if "grounding_world_state" in family:
        family_bias += 0.25
    elif "grounding_verification" in family:
        family_bias += 0.20
    elif "grounding_hypothesis" in family:
        family_bias += 0.16
    elif "grounding_graph" in family:
        family_bias += 0.08
    return status_bias + family_bias + (0.35 * confidence) + (0.25 * support) - (0.18 * conflict)


def grounding_memory_writeback_records(
    records: Sequence[Any],
    *,
    limit: int = 16,
) -> Tuple[Any, ...]:
    if limit <= 0:
        return tuple()
    deduped: Dict[str, Any] = {}
    for record in records:
        key = _graph_key(record)
        current = deduped.get(key)
        if current is None or grounding_memory_priority(record) > grounding_memory_priority(current):
            deduped[key] = record
    grouped: Dict[str, List[Any]] = {
        "active": [],
        "hypothetical": [],
        "contradicted": [],
        "unknown": [],
    }
    for record in deduped.values():
        grouped.setdefault(grounding_memory_status(record), []).append(record)
    for bucket in grouped.values():
        bucket.sort(key=lambda item: (-grounding_memory_priority(item), _graph_key(item)))

    selected: List[Any] = []
    selected_keys = set()
    per_status_counts = {key: 0 for key in grouped}
    family_counts: Dict[str, int] = {}
    family_cap = max(2, limit // 3)

    def _try_add(record: Any) -> bool:
        key = _graph_key(record)
        if key in selected_keys or len(selected) >= limit:
            return False
        family = _graph_family(record)
        if family and family_counts.get(family, 0) >= family_cap:
            return False
        selected_keys.add(key)
        selected.append(record)
        status = grounding_memory_status(record)
        per_status_counts[status] = per_status_counts.get(status, 0) + 1
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        return True

    present_statuses = [
        status for status in ("active", "hypothetical", "contradicted", "unknown")
        if grouped.get(status)
    ]
    if limit >= len(present_statuses):
        for status in present_statuses:
            if grouped[status]:
                _try_add(grouped[status][0])

    pool = sorted(
        deduped.values(),
        key=lambda item: (-grounding_memory_priority(item), _graph_key(item)),
    )
    max_ratio = {
        "active": 0.60,
        "hypothetical": 0.45,
        "contradicted": 0.45,
        "unknown": 0.30,
    }
    for record in pool:
        status = grounding_memory_status(record)
        status_cap = max(1, int(round(limit * max_ratio.get(status, 0.30))))
        if per_status_counts.get(status, 0) >= status_cap:
            continue
        _try_add(record)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for record in pool:
            if _try_add(record) and len(selected) >= limit:
                break
    return tuple(selected)


def grounding_memory_writeback_status_counts(records: Sequence[Any]) -> Dict[str, float]:
    counts = {
        "active": 0.0,
        "hypothetical": 0.0,
        "contradicted": 0.0,
        "unknown": 0.0,
    }
    for record in records:
        status = grounding_memory_status(record)
        counts[status] = counts.get(status, 0.0) + 1.0
    return counts

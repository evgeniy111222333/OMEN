from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple


def grounding_memory_status(record: Any) -> str:
    world_status = str(getattr(record, "world_status", "") or "").strip().lower()
    if world_status in {"active", "hypothetical", "contradicted"}:
        return world_status
    verification_status = str(getattr(record, "verification_status", "") or "").strip().lower()
    if verification_status == "supported":
        return "active"
    if verification_status == "deferred":
        return "hypothetical"
    if verification_status == "conflicted":
        return "contradicted"
    status = str(getattr(record, "status", "") or "").strip().lower()
    if status in {"supported", "verified", "accepted", "active"}:
        return "active"
    if status in {"proposal", "proposed", "deferred", "uncertain", "hypothetical"}:
        return "hypothetical"
    if status in {"conflicted", "contradicted"}:
        return "contradicted"
    family = str(getattr(record, "graph_family", "") or "").strip().lower()
    if ":contradicted" in family or ":conflicted" in family:
        return "contradicted"
    if ":hypothetical" in family or ":deferred" in family:
        return "hypothetical"
    if ":active" in family or ":supported" in family:
        return "active"
    return "unknown"


def grounding_memory_status_counts(records: Sequence[Any]) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[grounding_memory_status(record)] += 1
    return {
        "active": float(counts.get("active", 0)),
        "hypothetical": float(counts.get("hypothetical", 0)),
        "contradicted": float(counts.get("contradicted", 0)),
        "unknown": float(counts.get("unknown", 0)),
    }


def grounding_memory_terms(records: Sequence[Any], *, limit: int = 24) -> Tuple[str, ...]:
    terms: List[str] = []
    seen = set()
    for record in records:
        for term in getattr(record, "graph_terms", ()) or ():
            text = str(term).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            terms.append(text)
            if len(terms) >= limit:
                return tuple(terms)
    return tuple(terms)


def grounding_memory_families(records: Sequence[Any], *, limit: int = 12) -> Tuple[str, ...]:
    families: List[str] = []
    seen = set()
    for record in records:
        family = str(getattr(record, "graph_family", "") or "").strip()
        if not family or family in seen:
            continue
        seen.add(family)
        families.append(family)
        if len(families) >= limit:
            return tuple(families)
    return tuple(families)


def grounding_memory_status_terms(
    records: Sequence[Any],
    *,
    statuses: Sequence[str],
    limit: int = 24,
) -> Tuple[str, ...]:
    wanted = {str(status).strip().lower() for status in statuses if str(status).strip()}
    if not wanted:
        return tuple()
    terms: List[str] = []
    seen = set()
    for record in records:
        if grounding_memory_status(record) not in wanted:
            continue
        for term in getattr(record, "graph_terms", ()) or ():
            text = str(term).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            terms.append(text)
            if len(terms) >= limit:
                return tuple(terms)
    return tuple(terms)


def grounding_memory_status_families(
    records: Sequence[Any],
    *,
    statuses: Sequence[str],
    limit: int = 12,
) -> Tuple[str, ...]:
    wanted = {str(status).strip().lower() for status in statuses if str(status).strip()}
    if not wanted:
        return tuple()
    families: List[str] = []
    seen = set()
    for record in records:
        if grounding_memory_status(record) not in wanted:
            continue
        family = str(getattr(record, "graph_family", "") or "").strip()
        if not family or family in seen:
            continue
        seen.add(family)
        families.append(family)
        if len(families) >= limit:
            return tuple(families)
    return tuple(families)


def grounding_memory_records(records: Sequence[Any], *, limit: int = 16) -> Tuple[Any, ...]:
    selected: List[Any] = []
    seen = set()
    for record in records:
        key = getattr(record, "graph_key", None)
        if not isinstance(key, str) or not key or key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if len(selected) >= limit:
            break
    return tuple(selected)

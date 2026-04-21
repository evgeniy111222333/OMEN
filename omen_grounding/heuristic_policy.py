from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple


HEURISTIC_CLAIM_SOURCES = frozenset(
    {
        "fallback_extraction",
        "heuristic_backbone",
        "heuristic_fallback",
        "structural_nl_fallback",
        "trace_text_fallback",
    }
)
HEURISTIC_MARKER_PREFIXES = (
    "heuristic_source:",
    "heuristic_role:",
    "heuristic_authority:",
)


def append_heuristic_evidence(
    evidence_refs: Sequence[str] = (),
    *,
    source: str,
    role: str = "fallback_extraction",
) -> Tuple[str, ...]:
    values = [str(item).strip() for item in tuple(evidence_refs or ()) if str(item).strip()]
    values.extend(
        (
            f"heuristic_source:{str(source).strip()}",
            f"heuristic_role:{str(role).strip()}",
            "heuristic_authority:low",
        )
    )
    selected = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        selected.append(value)
    return tuple(selected)


def is_heuristic_claim_source(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return (
        text in HEURISTIC_CLAIM_SOURCES
        or text.startswith("heuristic_")
        or text.endswith("_fallback")
        or text == "fallback_extraction"
    )


def lineage_has_heuristic_markers(values: Iterable[Any]) -> bool:
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        if any(text.startswith(prefix) for prefix in HEURISTIC_MARKER_PREFIXES):
            return True
    return False


def record_is_heuristic(record: Any) -> bool:
    if is_heuristic_claim_source(getattr(record, "claim_source", "")):
        return True
    for field_name in ("provenance", "support_set", "evidence_refs", "graph_terms"):
        values = getattr(record, field_name, ()) or ()
        if lineage_has_heuristic_markers(values):
            return True
    return False


def candidate_rule_is_heuristic(candidate: Any) -> bool:
    metadata = getattr(candidate, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    if is_heuristic_claim_source(metadata.get("claim_source", "")):
        return True
    for field_name in ("support_set", "provenance"):
        values = tuple(metadata.get(field_name, ()) or ())
        if lineage_has_heuristic_markers(values):
            return True
    return False

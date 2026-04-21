from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .heuristic_policy import record_is_heuristic


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _record_status(record: Any) -> str:
    return str(getattr(record, "world_status", "") or "").strip().lower() or "hypothetical"


def _record_type(record: Any) -> str:
    return str(getattr(record, "record_type", "") or "").strip().lower() or "unknown"


def _record_symbols(record: Any) -> Tuple[str, ...]:
    symbols = getattr(record, "symbols", ()) or ()
    return tuple(str(item).strip() for item in symbols if str(item).strip())


def _planner_authoritative_record(record: Any) -> bool:
    return not record_is_heuristic(record)


def _record_annotations(record: Any) -> Dict[str, Tuple[str, ...]]:
    annotations: Dict[str, List[str]] = {"if": [], "cause": [], "time": [], "modal": []}
    for symbol in _record_symbols(record)[3:]:
        if ":" not in symbol:
            continue
        prefix, value = symbol.split(":", 1)
        prefix = prefix.strip().lower()
        value = value.strip()
        if prefix in annotations and value:
            annotations[prefix].append(value)
    return {
        key: tuple(values)
        for key, values in annotations.items()
        if values
    }


@dataclass(frozen=True)
class PlannerResource:
    symbol: str
    statuses: Tuple[str, ...] = field(default_factory=tuple)
    support: float = 0.0
    conflict: float = 0.0
    confidence: float = 0.0
    sources: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlannerOperator:
    operator_id: str
    predicate: str
    inputs: Tuple[str, ...] = field(default_factory=tuple)
    outputs: Tuple[str, ...] = field(default_factory=tuple)
    modality: str = ""
    conditions: Tuple[str, ...] = field(default_factory=tuple)
    causes: Tuple[str, ...] = field(default_factory=tuple)
    temporals: Tuple[str, ...] = field(default_factory=tuple)
    status: str = "hypothetical"
    support: float = 0.0
    conflict: float = 0.0
    confidence: float = 0.0
    repair_action: str = "none"
    provenance: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlannerAlternativeWorld:
    world_id: str
    status: str
    operator_ids: Tuple[str, ...] = field(default_factory=tuple)
    resource_symbols: Tuple[str, ...] = field(default_factory=tuple)
    contradiction_symbols: Tuple[str, ...] = field(default_factory=tuple)
    pressure: float = 0.0
    record_count: int = 0


def _resource_candidates(record: Any) -> Tuple[str, ...]:
    record_type = _record_type(record)
    symbols = _record_symbols(record)
    annotations = _record_annotations(record)
    if record_type in {"relation", "hidden_cause"} and len(symbols) >= 3:
        extra = tuple(annotations.get("if", ())) + tuple(annotations.get("cause", ())) + tuple(annotations.get("time", ()))
        return (symbols[0], symbols[2], *extra)
    if record_type == "state" and len(symbols) >= 1:
        return (symbols[0],)
    if record_type == "goal" and len(symbols) >= 2:
        return (symbols[-1],)
    return symbols


def _sorted_unique(values: Sequence[str], *, limit: int) -> Tuple[str, ...]:
    selected: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        selected.append(text)
        if len(selected) >= limit:
            break
    return tuple(selected)


def build_planner_resources(
    records: Sequence[Any],
    *,
    limit: int = 32,
) -> Tuple[PlannerResource, ...]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not _planner_authoritative_record(record):
            continue
        status = _record_status(record)
        for symbol in _resource_candidates(record):
            state = aggregated.setdefault(
                symbol,
                {
                    "statuses": set(),
                    "support": 0.0,
                    "conflict": 0.0,
                    "confidence": 0.0,
                    "sources": set(),
                },
            )
            state["statuses"].add(status)
            state["support"] = max(float(state["support"]), _clip01(getattr(record, "support", 0.0)))
            state["conflict"] = max(float(state["conflict"]), _clip01(getattr(record, "conflict", 0.0)))
            state["confidence"] = max(float(state["confidence"]), _clip01(getattr(record, "confidence", 0.0)))
            state["sources"].add(str(getattr(record, "record_id", symbol)))
    ranked = sorted(
        aggregated.items(),
        key=lambda item: (
            -float(item[1]["confidence"]),
            -float(item[1]["support"]),
            float(item[1]["conflict"]),
            item[0],
        ),
    )
    resources: List[PlannerResource] = []
    for symbol, state in ranked[:limit]:
        resources.append(
            PlannerResource(
                symbol=symbol,
                statuses=tuple(sorted(state["statuses"])),
                support=_clip01(state["support"]),
                conflict=_clip01(state["conflict"]),
                confidence=_clip01(state["confidence"]),
                sources=tuple(sorted(state["sources"])),
            )
        )
    return tuple(resources)


def build_planner_operators(
    records: Sequence[Any],
    *,
    limit: int = 32,
) -> Tuple[PlannerOperator, ...]:
    operators: List[PlannerOperator] = []
    for record in records:
        if not _planner_authoritative_record(record):
            continue
        if _record_type(record) not in {"relation", "hidden_cause"}:
            continue
        symbols = _record_symbols(record)
        if len(symbols) < 3:
            continue
        annotations = _record_annotations(record)
        conditions = tuple(annotations.get("if", ()))
        causes = tuple(annotations.get("cause", ()))
        temporals = tuple(annotations.get("time", ()))
        modalities = tuple(annotations.get("modal", ()))
        operators.append(
            PlannerOperator(
                operator_id=str(getattr(record, "record_id", "operator")),
                predicate=symbols[1],
                inputs=(symbols[0], *conditions, *causes, *temporals),
                outputs=(symbols[2],),
                modality=modalities[0] if modalities else "",
                conditions=conditions,
                causes=causes,
                temporals=temporals,
                status=_record_status(record),
                support=_clip01(getattr(record, "support", 0.0)),
                conflict=_clip01(getattr(record, "conflict", 0.0)),
                confidence=_clip01(getattr(record, "confidence", 0.0)),
                repair_action=str(getattr(record, "repair_action", "none") or "none"),
                provenance=tuple(str(item) for item in (getattr(record, "provenance", ()) or ())),
            )
        )
    ranked = sorted(
        operators,
        key=lambda item: (-float(item.confidence), -float(item.support), float(item.conflict), item.operator_id),
    )
    return tuple(ranked[:limit])


def build_planner_alternative_worlds(
    active_records: Sequence[Any],
    hypothetical_records: Sequence[Any],
    contradicted_records: Sequence[Any],
    *,
    branching_pressure: float = 0.0,
    contradiction_pressure: float = 0.0,
    limit: int = 3,
) -> Tuple[PlannerAlternativeWorld, ...]:
    alternatives: List[PlannerAlternativeWorld] = []
    for status, records, pressure in (
        ("hypothetical", tuple(hypothetical_records), branching_pressure),
        ("contradicted", tuple(contradicted_records), contradiction_pressure),
    ):
        if not records:
            continue
        operator_ids = [
            str(getattr(record, "record_id", ""))
            for record in records
            if _planner_authoritative_record(record)
            if _record_type(record) in {"relation", "hidden_cause"}
        ]
        contradiction_symbols = [
            " | ".join(_record_symbols(record))
            for record in records
            if _planner_authoritative_record(record)
            if _record_status(record) == "contradicted"
        ]
        resource_symbols = [
            symbol
            for record in records
            if _planner_authoritative_record(record)
            for symbol in _resource_candidates(record)
        ]
        alternatives.append(
            PlannerAlternativeWorld(
                world_id=f"{status}_world",
                status=status,
                operator_ids=_sorted_unique(operator_ids, limit=limit * 8),
                resource_symbols=_sorted_unique(resource_symbols, limit=limit * 12),
                contradiction_symbols=_sorted_unique(contradiction_symbols, limit=limit * 8),
                pressure=_clip01(pressure),
                record_count=len(records),
            )
        )
    if active_records and not alternatives:
        alternatives.append(
            PlannerAlternativeWorld(
                world_id="active_world",
                status="active",
                operator_ids=_sorted_unique(
                    [
                        str(getattr(record, "record_id", ""))
                        for record in active_records
                        if _planner_authoritative_record(record)
                        if _record_type(record) in {"relation", "hidden_cause"}
                    ],
                    limit=limit * 8,
                ),
                resource_symbols=_sorted_unique(
                    [
                        symbol
                        for record in active_records
                        if _planner_authoritative_record(record)
                        for symbol in _resource_candidates(record)
                    ],
                    limit=limit * 12,
                ),
                contradiction_symbols=tuple(),
                pressure=0.0,
                record_count=len(active_records),
            )
        )
    return tuple(alternatives[:limit])

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _record_symbols(record: Any) -> Tuple[str, ...]:
    values = getattr(record, "symbols", ()) or ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _unique_strings(values: Sequence[str], *, limit: int) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return tuple(out)


@dataclass(frozen=True)
class PlannerConstraint:
    constraint_id: str
    target_id: str
    validator_family: str
    enforcement: str
    source_segment: int
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    support: float = 0.0
    conflict: float = 0.0
    confidence: float = 0.0
    priority: float = 0.0
    pressure: float = 0.0
    rationale: str = ""


@dataclass(frozen=True)
class PlannerRepairDirective:
    directive_id: str
    target_id: str
    action_type: str
    source_segment: int
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    validator_families: Tuple[str, ...] = field(default_factory=tuple)
    priority: float = 0.0
    pressure: float = 0.0
    reason: str = ""


def _constraint_enforcement(validation_status: str) -> str:
    status = str(validation_status or "").strip().lower()
    if status == "supported":
        return "prefer"
    if status == "conflicted":
        return "avoid"
    return "branch"


def build_planner_guidance(
    validation_records: Sequence[Any],
    repair_actions: Sequence[Any],
    *,
    limit: int = 32,
) -> Tuple[Tuple[PlannerConstraint, ...], Tuple[PlannerRepairDirective, ...]]:
    constraints: List[PlannerConstraint] = []
    target_support: Dict[str, Dict[str, Any]] = {}

    for record in validation_records:
        target_id = str(getattr(record, "target_id", "") or "").strip()
        validator_family = str(getattr(record, "validator_family", "") or "world_model").strip().lower()
        enforcement = _constraint_enforcement(str(getattr(record, "validation_status", "") or "deferred"))
        support = _clip01(getattr(record, "support", 0.0))
        conflict = _clip01(getattr(record, "conflict", 0.0))
        confidence = _clip01(getattr(record, "confidence", 0.0))
        bias = {
            "prefer": 0.18,
            "branch": 0.10,
            "avoid": 0.12,
        }.get(enforcement, 0.0)
        temporal_bias = 0.08 if validator_family == "temporal" else 0.0
        priority = _clip01((0.40 * confidence) + (0.26 * support) + bias + temporal_bias - (0.10 * conflict))
        pressure = _clip01(
            (0.42 * conflict)
            + (0.20 * (1.0 - support))
            + (0.12 * float(enforcement == "avoid"))
            + (0.08 * float(validator_family == "temporal"))
        )
        symbols = _record_symbols(record)
        constraint = PlannerConstraint(
            constraint_id=str(getattr(record, "validation_id", target_id) or target_id),
            target_id=target_id,
            validator_family=validator_family,
            enforcement=enforcement,
            source_segment=int(getattr(record, "source_segment", 0) or 0),
            symbols=symbols,
            support=support,
            conflict=conflict,
            confidence=confidence,
            priority=priority,
            pressure=pressure,
            rationale=str(getattr(record, "rationale", "") or "").strip(),
        )
        constraints.append(constraint)
        if not target_id:
            continue
        state = target_support.setdefault(
            target_id,
            {
                "symbols": [],
                "families": set(),
                "support": 0.0,
                "conflict": 0.0,
                "confidence": 0.0,
            },
        )
        state["symbols"].extend(symbols)
        state["families"].add(validator_family)
        state["support"] = max(float(state["support"]), support)
        state["conflict"] = max(float(state["conflict"]), conflict)
        state["confidence"] = max(float(state["confidence"]), confidence)

    constraints.sort(
        key=lambda item: (-float(item.priority), -float(item.pressure), item.validator_family, item.constraint_id)
    )
    constraints = constraints[:limit]

    directives: List[PlannerRepairDirective] = []
    for action in repair_actions:
        target_id = str(getattr(action, "target_id", "") or "").strip()
        action_type = str(getattr(action, "action_type", "") or "repair").strip()
        state = target_support.get(target_id, {})
        symbols = _unique_strings(state.get("symbols", ()), limit=12)
        families = tuple(sorted(str(item) for item in state.get("families", set()) if str(item).strip()))
        support = _clip01(state.get("support", 0.0))
        conflict = _clip01(state.get("conflict", 0.0))
        confidence = _clip01(state.get("confidence", 0.0))
        action_bias = {
            "trigger_hidden_cause_abduction": 0.24,
            "trigger_temporal_repair": 0.22,
            "keep_ontology_hypothesis_alive": 0.18,
            "promote_world_model_supported_claim": 0.20,
        }.get(action_type, 0.10)
        priority = _clip01(
            (0.36 * _clip01(getattr(action, "priority", 0.0)))
            + (0.24 * _clip01(getattr(action, "pressure", 0.0)))
            + (0.16 * support)
            + (0.10 * confidence)
            + action_bias
            - (0.06 * conflict)
        )
        pressure = _clip01(
            max(
                float(getattr(action, "pressure", 0.0) or 0.0),
                0.45 * conflict + 0.25 * (1.0 - support),
            )
        )
        directives.append(
            PlannerRepairDirective(
                directive_id=str(getattr(action, "action_id", target_id) or target_id),
                target_id=target_id,
                action_type=action_type,
                source_segment=int(getattr(action, "source_segment", 0) or 0),
                symbols=symbols,
                validator_families=families,
                priority=priority,
                pressure=pressure,
                reason=str(getattr(action, "reason", "") or "").strip(),
            )
        )
    directives.sort(key=lambda item: (-float(item.priority), -float(item.pressure), item.action_type, item.directive_id))
    directives = directives[:limit]
    return tuple(constraints), tuple(directives)

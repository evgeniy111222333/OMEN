from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple


PLAN_ACTIVE_RESOURCE_PRED = 340
PLAN_HYPOTHETICAL_RESOURCE_PRED = 341
PLAN_CONTRADICTED_RESOURCE_PRED = 342
PLAN_ACTIVE_WORLD_PRED = 343
PLAN_HYPOTHETICAL_WORLD_PRED = 344
PLAN_CONTRADICTED_WORLD_PRED = 345
PLAN_ACTIVE_EFFECT_PRED = 346
PLAN_HYPOTHETICAL_EFFECT_PRED = 347
PLAN_CONTRADICTED_EFFECT_PRED = 348


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_hash(text: str, *, base: int = 1_000) -> int:
    return int(base + (zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF) % 100_000)


def _status_resource_pred(status: str) -> int:
    mapping = {
        "active": PLAN_ACTIVE_RESOURCE_PRED,
        "hypothetical": PLAN_HYPOTHETICAL_RESOURCE_PRED,
        "contradicted": PLAN_CONTRADICTED_RESOURCE_PRED,
    }
    return mapping.get(str(status).strip().lower(), PLAN_HYPOTHETICAL_RESOURCE_PRED)


def _status_world_pred(status: str) -> int:
    mapping = {
        "active": PLAN_ACTIVE_WORLD_PRED,
        "hypothetical": PLAN_HYPOTHETICAL_WORLD_PRED,
        "contradicted": PLAN_CONTRADICTED_WORLD_PRED,
    }
    return mapping.get(str(status).strip().lower(), PLAN_HYPOTHETICAL_WORLD_PRED)


def _status_effect_pred(status: str) -> int:
    mapping = {
        "active": PLAN_ACTIVE_EFFECT_PRED,
        "hypothetical": PLAN_HYPOTHETICAL_EFFECT_PRED,
        "contradicted": PLAN_CONTRADICTED_EFFECT_PRED,
    }
    return mapping.get(str(status).strip().lower(), PLAN_HYPOTHETICAL_EFFECT_PRED)


def _resource_fact(status: str, symbol: str) -> Tuple[int, int]:
    return (_status_resource_pred(status), _stable_hash(symbol, base=10_000))


def _world_fact(status: str, world_id: str) -> Tuple[int, int]:
    return (_status_world_pred(status), _stable_hash(world_id, base=20_000))


def _effect_fact(status: str, predicate: str, symbol: str) -> Tuple[int, int]:
    return (_status_effect_pred(status), _stable_hash(f"{predicate}:{symbol}", base=30_000))


def _unique_fact_tuples(facts: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for fact in facts:
        if fact in seen:
            continue
        seen.add(fact)
        out.append(fact)
    return tuple(out)


@dataclass(frozen=True)
class PlannerBridgeOperatorSpec:
    operator_id: str
    source: str = "grounding_bridge"
    status: str = "hypothetical"
    predicate: str = "act"
    preconditions: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    add_effects: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    del_effects: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    priority: float = 0.0


def planner_state_seed_facts(planner_state: Any) -> Tuple[Tuple[int, int], ...]:
    if planner_state is None:
        return tuple()
    facts: List[Tuple[int, int]] = []
    for resource in tuple(getattr(planner_state, "resources", ()) or ()):
        symbol = str(getattr(resource, "symbol", "") or "").strip()
        if not symbol:
            continue
        statuses = tuple(getattr(resource, "statuses", ()) or ()) or ("hypothetical",)
        for status in statuses:
            facts.append(_resource_fact(str(status), symbol))
    for alternative in tuple(getattr(planner_state, "alternative_worlds", ()) or ()):
        world_id = str(getattr(alternative, "world_id", "") or "").strip()
        status = str(getattr(alternative, "status", "") or "hypothetical").strip().lower()
        if world_id:
            facts.append(_world_fact(status, world_id))
    return _unique_fact_tuples(facts)


def compile_planner_bridge_operator_specs(
    planner_state: Any,
    *,
    goal_id: int,
    limit: int = 16,
) -> Tuple[PlannerBridgeOperatorSpec, ...]:
    if planner_state is None:
        return tuple()
    destructive = {str(item).strip() for item in (getattr(planner_state, "destructive_effect_symbols", ()) or ()) if str(item).strip()}
    persistent = {str(item).strip() for item in (getattr(planner_state, "persistent_effect_symbols", ()) or ()) if str(item).strip()}
    branching_pressure = float(getattr(planner_state, "branching_pressure", 0.0) or 0.0)
    contradiction_pressure = float(getattr(planner_state, "contradiction_pressure", 0.0) or 0.0)
    alternative_worlds = {
        str(getattr(world, "status", "") or "hypothetical").strip().lower(): str(getattr(world, "world_id", "") or "").strip()
        for world in tuple(getattr(planner_state, "alternative_worlds", ()) or ())
        if str(getattr(world, "world_id", "") or "").strip()
    }
    specs: List[PlannerBridgeOperatorSpec] = []
    for operator in tuple(getattr(planner_state, "operators", ()) or ()):
        status = str(getattr(operator, "status", "") or "hypothetical").strip().lower()
        predicate = str(getattr(operator, "predicate", "") or "act").strip() or "act"
        inputs = tuple(str(item).strip() for item in (getattr(operator, "inputs", ()) or ()) if str(item).strip())
        outputs = tuple(str(item).strip() for item in (getattr(operator, "outputs", ()) or ()) if str(item).strip())
        signature = " | ".join([*(inputs[:1] or ("_",)), predicate, *(outputs[:1] or ("_",))])
        preconditions: List[Tuple[int, int]] = []
        add_effects: List[Tuple[int, int]] = []
        del_effects: List[Tuple[int, int]] = []

        if inputs:
            preconditions.extend(_resource_fact(status, symbol) for symbol in inputs)
        else:
            preconditions.append((300, int(goal_id)))

        world_id = alternative_worlds.get(status)
        if world_id and status in {"hypothetical", "contradicted"}:
            preconditions.append(_world_fact(status, world_id))

        add_effects.extend(_resource_fact(status, symbol) for symbol in outputs)
        add_effects.extend(_effect_fact(status, predicate, symbol) for symbol in outputs)

        if status == "active":
            add_effects.append((302, int(goal_id)))
            if signature in persistent or predicate.lower() in {"return", "yield", "create", "creates", "generate", "generates"}:
                add_effects.append((303, int(goal_id)))
        elif status == "hypothetical":
            add_effects.append((305, int(goal_id)))
        else:
            del_effects.append((303, int(goal_id)))

        if signature in destructive:
            del_effects.extend(_resource_fact("active", symbol) for symbol in outputs)

        status_bias = {
            "active": 0.40,
            "hypothetical": 0.14 + 0.22 * branching_pressure,
            "contradicted": -0.08 - 0.14 * contradiction_pressure,
        }.get(status, 0.0)
        persistent_bonus = 0.18 if signature in persistent else 0.0
        destructive_bonus = 0.08 if signature in destructive and status != "contradicted" else 0.0
        priority = (
            status_bias
            + 0.32 * _clip01(getattr(operator, "confidence", 0.0))
            + 0.22 * _clip01(getattr(operator, "support", 0.0))
            - 0.16 * _clip01(getattr(operator, "conflict", 0.0))
            + persistent_bonus
            + destructive_bonus
        )
        specs.append(
            PlannerBridgeOperatorSpec(
                operator_id=str(getattr(operator, "operator_id", signature) or signature),
                status=status,
                predicate=predicate,
                preconditions=_unique_fact_tuples(preconditions),
                add_effects=_unique_fact_tuples(add_effects),
                del_effects=_unique_fact_tuples(del_effects),
                priority=float(priority),
            )
        )
    specs.sort(key=lambda item: (-float(item.priority), item.operator_id))
    return tuple(specs[:limit])

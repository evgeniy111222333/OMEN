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
PLAN_PREFER_CONSTRAINT_PRED = 349
PLAN_BRANCH_CONSTRAINT_PRED = 350
PLAN_AVOID_CONSTRAINT_PRED = 351
PLAN_REPAIR_DIRECTIVE_PRED = 352
PLAN_SUPPORTED_VERIFICATION_PRED = 353
PLAN_DEFERRED_VERIFICATION_PRED = 354
PLAN_CONFLICTED_VERIFICATION_PRED = 355
PLAN_HYPOTHESIS_PRED = 356
PLAN_LINEAGE_PRED = 357


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


def _constraint_fact(enforcement: str, text: str) -> Tuple[int, int]:
    mapping = {
        "prefer": PLAN_PREFER_CONSTRAINT_PRED,
        "branch": PLAN_BRANCH_CONSTRAINT_PRED,
        "avoid": PLAN_AVOID_CONSTRAINT_PRED,
    }
    pred = mapping.get(str(enforcement).strip().lower(), PLAN_BRANCH_CONSTRAINT_PRED)
    return (pred, _stable_hash(text, base=40_000))


def _repair_fact(action_type: str, target_id: str) -> Tuple[int, int]:
    return (PLAN_REPAIR_DIRECTIVE_PRED, _stable_hash(f"{action_type}:{target_id}", base=50_000))


def _verification_fact(status: str, text: str) -> Tuple[int, int]:
    mapping = {
        "supported": PLAN_SUPPORTED_VERIFICATION_PRED,
        "deferred": PLAN_DEFERRED_VERIFICATION_PRED,
        "conflicted": PLAN_CONFLICTED_VERIFICATION_PRED,
    }
    pred = mapping.get(str(status).strip().lower(), PLAN_DEFERRED_VERIFICATION_PRED)
    return (pred, _stable_hash(text, base=60_000))


def _hypothesis_fact(kind: str, text: str) -> Tuple[int, int]:
    return (PLAN_HYPOTHESIS_PRED, _stable_hash(f"{kind}:{text}", base=70_000))


def _lineage_fact(text: str) -> Tuple[int, int]:
    return (PLAN_LINEAGE_PRED, _stable_hash(text, base=80_000))


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


def _record_terms(values: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(str(item).strip() for item in values if str(item).strip())


def _verification_signature(record: Any) -> str:
    hypothesis_id = str(getattr(record, "hypothesis_id", "") or "").strip()
    symbols = _record_terms(getattr(record, "symbols", ()) or ())
    return ":".join(part for part in (hypothesis_id, *symbols[:4]) if part)


def _hypothesis_signature(record: Any) -> str:
    hypothesis_id = str(getattr(record, "hypothesis_id", "") or "").strip()
    kind = str(getattr(record, "kind", "") or "").strip()
    symbols = _record_terms(getattr(record, "symbols", ()) or ())
    return ":".join(part for part in (kind, hypothesis_id, *symbols[:4]) if part)


def _verification_terms(record: Any) -> Tuple[str, ...]:
    repair_action = str(getattr(record, "repair_action", "") or "").strip()
    return tuple(
        item
        for item in (
            str(getattr(record, "hypothesis_id", "") or "").strip(),
            str(getattr(record, "kind", "") or "").strip(),
            str(getattr(record, "verification_status", "") or "").strip(),
            repair_action,
            *_record_terms(getattr(record, "symbols", ()) or ()),
            *(
                str(item).strip()
                for item in (getattr(record, "provenance", ()) or ())
                if str(item).strip()
            ),
        )
        if item
    )


def _hypothesis_terms(record: Any) -> Tuple[str, ...]:
    conflict_tag = str(getattr(record, "conflict_tag", "") or "").strip()
    deferred = "deferred" if bool(getattr(record, "deferred", False)) else ""
    return tuple(
        item
        for item in (
            str(getattr(record, "hypothesis_id", "") or "").strip(),
            str(getattr(record, "kind", "") or "").strip(),
            conflict_tag,
            deferred,
            *_record_terms(getattr(record, "symbols", ()) or ()),
            *(
                str(item).strip()
                for item in (getattr(record, "provenance", ()) or ())
                if str(item).strip()
            ),
        )
        if item
    )


def _graph_terms(record: Any) -> Tuple[str, ...]:
    return tuple(
        item
        for item in (
            str(getattr(record, "graph_family", "") or "").strip(),
            str(getattr(record, "record_type", "") or "").strip(),
            *_record_terms(getattr(record, "graph_terms", ()) or ()),
        )
        if item
    )


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
    for constraint in tuple(getattr(planner_state, "constraints", ()) or ()):
        target_id = str(getattr(constraint, "target_id", "") or "").strip()
        symbols = tuple(str(item).strip() for item in (getattr(constraint, "symbols", ()) or ()) if str(item).strip())
        signature = ":".join(
            part for part in (str(getattr(constraint, "validator_family", "") or "").strip(), target_id, *symbols[:3]) if part
        )
        if signature:
            facts.append(_constraint_fact(str(getattr(constraint, "enforcement", "") or "branch"), signature))
    for directive in tuple(getattr(planner_state, "repair_directives", ()) or ()):
        action_type = str(getattr(directive, "action_type", "") or "").strip()
        target_id = str(getattr(directive, "target_id", "") or "").strip()
        if action_type:
            facts.append(_repair_fact(action_type, target_id))
    for record in tuple(getattr(planner_state, "verification_records", ()) or ()):
        signature = _verification_signature(record)
        status = str(getattr(record, "verification_status", "") or "deferred").strip().lower()
        if signature:
            facts.append(_verification_fact(status, signature))
        repair_action = str(getattr(record, "repair_action", "") or "").strip()
        hypothesis_id = str(getattr(record, "hypothesis_id", "") or "").strip()
        if repair_action and repair_action not in {"", "none"}:
            facts.append(_repair_fact(repair_action, hypothesis_id))
    for record in tuple(getattr(planner_state, "hypothesis_records", ()) or ()):
        signature = _hypothesis_signature(record)
        kind = str(getattr(record, "kind", "") or "hypothesis").strip().lower() or "hypothesis"
        if signature:
            facts.append(_hypothesis_fact(kind, signature))
    for record in tuple(getattr(planner_state, "graph_records", ()) or ()):
        family = str(getattr(record, "graph_family", "") or "").strip()
        if family:
            facts.append(_lineage_fact(family))
        for term in _record_terms(getattr(record, "graph_terms", ()) or ())[:3]:
            facts.append(_lineage_fact(term))
    for symbol in tuple(getattr(planner_state, "lineage_symbols", ()) or ()):
        text = str(symbol or "").strip()
        if text:
            facts.append(_lineage_fact(text))
    return _unique_fact_tuples(facts)


def _overlap_score(lhs: Sequence[str], rhs: Sequence[str]) -> float:
    left = {str(item).strip().lower() for item in lhs if str(item).strip()}
    right = {str(item).strip().lower() for item in rhs if str(item).strip()}
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right))) / float(max(len(right), 1))


def _operator_terms(operator: Any, signature: str) -> Tuple[str, ...]:
    inputs = tuple(str(item).strip() for item in (getattr(operator, "inputs", ()) or ()) if str(item).strip())
    outputs = tuple(str(item).strip() for item in (getattr(operator, "outputs", ()) or ()) if str(item).strip())
    provenance = tuple(str(item).strip() for item in (getattr(operator, "provenance", ()) or ()) if str(item).strip())
    return tuple(
        item
        for item in (
            str(getattr(operator, "operator_id", "") or "").strip(),
            str(getattr(operator, "predicate", "") or "").strip(),
            signature,
            *inputs,
            *outputs,
            *provenance,
        )
        if item
    )


def _constraint_priority_adjustment(operator: Any, signature: str, planner_state: Any) -> float:
    terms = _operator_terms(operator, signature)
    status = str(getattr(operator, "status", "") or "hypothetical").strip().lower()
    adjustment = 0.0
    for constraint in tuple(getattr(planner_state, "constraints", ()) or ()):
        overlap = _overlap_score(terms, getattr(constraint, "symbols", ()) or ())
        if overlap <= 0.0 and str(getattr(constraint, "target_id", "") or "") not in terms:
            continue
        enforcement = str(getattr(constraint, "enforcement", "") or "branch").strip().lower()
        priority = _clip01(getattr(constraint, "priority", 0.0))
        pressure = _clip01(getattr(constraint, "pressure", 0.0))
        match = max(overlap, 0.5 if str(getattr(constraint, "target_id", "") or "") in terms else 0.0)
        if enforcement == "prefer":
            adjustment += 0.20 * priority * match
        elif enforcement == "branch":
            if status == "hypothetical":
                adjustment += 0.14 * max(priority, pressure) * match
            else:
                adjustment += 0.05 * priority * match
        elif enforcement == "avoid":
            adjustment -= 0.24 * max(priority, pressure) * match
    for directive in tuple(getattr(planner_state, "repair_directives", ()) or ()):
        overlap = _overlap_score(terms, getattr(directive, "symbols", ()) or ())
        target_id = str(getattr(directive, "target_id", "") or "")
        match = max(overlap, 0.45 if target_id and target_id in terms else 0.0)
        if match <= 0.0 and getattr(directive, "symbols", ()) and target_id not in terms:
            continue
        action_type = str(getattr(directive, "action_type", "") or "").strip()
        priority = _clip01(getattr(directive, "priority", 0.0))
        pressure = _clip01(getattr(directive, "pressure", 0.0))
        if action_type == "promote_world_model_supported_claim":
            adjustment += 0.22 * priority * max(match, 0.5)
        elif action_type == "keep_ontology_hypothesis_alive" and status == "hypothetical":
            adjustment += 0.18 * priority * max(match, 0.5)
        elif action_type == "trigger_hidden_cause_abduction":
            if status in {"hypothetical", "contradicted"}:
                adjustment += 0.18 * max(priority, pressure) * max(match, 0.35 if not getattr(directive, "symbols", ()) else 0.0)
        elif action_type == "trigger_temporal_repair":
            if status == "active":
                adjustment += 0.16 * max(priority, pressure) * max(match, 0.35 if not getattr(directive, "symbols", ()) else 0.0)
            else:
                adjustment -= 0.10 * pressure * max(match, 0.25 if not getattr(directive, "symbols", ()) else 0.0)
    return float(adjustment)


def _evidence_priority_adjustment(operator: Any, signature: str, planner_state: Any) -> float:
    terms = _operator_terms(operator, signature)
    status = str(getattr(operator, "status", "") or "hypothetical").strip().lower()
    predicate = str(getattr(operator, "predicate", "") or "").strip().lower()
    adjustment = 0.0

    for record in tuple(getattr(planner_state, "verification_records", ()) or ()):
        record_terms = _verification_terms(record)
        overlap = _overlap_score(terms, record_terms)
        hypothesis_id = str(getattr(record, "hypothesis_id", "") or "").strip()
        repair_action = str(getattr(record, "repair_action", "") or "").strip().lower()
        match = max(overlap, 0.45 if hypothesis_id and hypothesis_id in terms else 0.0)
        if match <= 0.0:
            continue
        support = _clip01(getattr(record, "support", 0.0))
        conflict = _clip01(getattr(record, "conflict", 0.0))
        verification_status = str(getattr(record, "verification_status", "") or "deferred").strip().lower()
        if verification_status == "supported":
            adjustment += 0.24 * max(support, 1.0 - conflict) * match
        elif verification_status == "deferred":
            if status == "hypothetical":
                adjustment += 0.16 * max(support, 1.0 - conflict) * match
            else:
                adjustment -= 0.04 * conflict * match
        elif verification_status == "conflicted":
            if repair_action and (predicate == repair_action or predicate.endswith(repair_action)):
                adjustment += 0.24 * max(conflict, support) * max(match, 0.5)
            elif status == "contradicted":
                adjustment += 0.16 * conflict * match
            else:
                adjustment -= 0.18 * conflict * match
        if bool(getattr(record, "hidden_cause_candidate", False)) and (
            predicate == "trigger_hidden_cause_abduction" or status in {"hypothetical", "contradicted"}
        ):
            adjustment += 0.14 * max(conflict, support) * max(match, 0.35)

    for record in tuple(getattr(planner_state, "hypothesis_records", ()) or ()):
        record_terms = _hypothesis_terms(record)
        overlap = _overlap_score(terms, record_terms)
        hypothesis_id = str(getattr(record, "hypothesis_id", "") or "").strip()
        match = max(overlap, 0.40 if hypothesis_id and hypothesis_id in terms else 0.0)
        if match <= 0.0:
            continue
        confidence = _clip01(getattr(record, "confidence", 0.0))
        conflict_tag = str(getattr(record, "conflict_tag", "") or "").strip()
        deferred = bool(getattr(record, "deferred", False))
        if status == "hypothetical":
            adjustment += (0.18 if deferred else 0.14) * confidence * match
        elif status == "contradicted" and conflict_tag:
            adjustment += 0.12 * max(confidence, 0.5) * match
        elif status == "active":
            adjustment += 0.08 * confidence * match

    lineage_symbols = tuple(str(item).strip() for item in (getattr(planner_state, "lineage_symbols", ()) or ()) if str(item).strip())
    lineage_overlap = _overlap_score(terms, lineage_symbols)
    if lineage_overlap > 0.0:
        adjustment += 0.12 * lineage_overlap
    for record in tuple(getattr(planner_state, "graph_records", ()) or ()):
        match = _overlap_score(terms, _graph_terms(record))
        if match <= 0.0:
            continue
        adjustment += 0.10 * _clip01(getattr(record, "confidence", 0.0)) * match

    return float(adjustment)


def _evidence_preconditions(operator: Any, signature: str, planner_state: Any) -> Tuple[Tuple[int, int], ...]:
    terms = _operator_terms(operator, signature)
    facts: List[Tuple[int, int]] = []

    verification_matches: List[Tuple[float, Any]] = []
    for record in tuple(getattr(planner_state, "verification_records", ()) or ()):
        record_terms = _verification_terms(record)
        hypothesis_id = str(getattr(record, "hypothesis_id", "") or "").strip()
        match = max(_overlap_score(terms, record_terms), 0.45 if hypothesis_id and hypothesis_id in terms else 0.0)
        if match > 0.0:
            verification_matches.append((match, record))
    verification_matches.sort(key=lambda item: (-float(item[0]), str(getattr(item[1], "hypothesis_id", "") or "")))
    for _match, record in verification_matches[:2]:
        signature_text = _verification_signature(record)
        status = str(getattr(record, "verification_status", "") or "deferred").strip().lower()
        if signature_text:
            facts.append(_verification_fact(status, signature_text))

    hypothesis_matches: List[Tuple[float, Any]] = []
    for record in tuple(getattr(planner_state, "hypothesis_records", ()) or ()):
        record_terms = _hypothesis_terms(record)
        hypothesis_id = str(getattr(record, "hypothesis_id", "") or "").strip()
        match = max(_overlap_score(terms, record_terms), 0.40 if hypothesis_id and hypothesis_id in terms else 0.0)
        if match > 0.0:
            hypothesis_matches.append((match, record))
    hypothesis_matches.sort(key=lambda item: (-float(item[0]), str(getattr(item[1], "hypothesis_id", "") or "")))
    for _match, record in hypothesis_matches[:2]:
        signature_text = _hypothesis_signature(record)
        kind = str(getattr(record, "kind", "") or "hypothesis").strip().lower() or "hypothesis"
        if signature_text:
            facts.append(_hypothesis_fact(kind, signature_text))

    lineage_matches: List[Tuple[float, str]] = []
    for symbol in tuple(getattr(planner_state, "lineage_symbols", ()) or ()):
        text = str(symbol or "").strip()
        if not text:
            continue
        match = _overlap_score(terms, (text,))
        if match > 0.0:
            lineage_matches.append((match, text))
    lineage_matches.sort(key=lambda item: (-float(item[0]), item[1]))
    for _match, text in lineage_matches[:2]:
        facts.append(_lineage_fact(text))

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
        preconditions.extend(_evidence_preconditions(operator, signature, planner_state))

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
            + _constraint_priority_adjustment(operator, signature, planner_state)
            + _evidence_priority_adjustment(operator, signature, planner_state)
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
    for record in tuple(getattr(planner_state, "verification_records", ()) or ()):
        repair_action = str(getattr(record, "repair_action", "") or "").strip()
        verification_status = str(getattr(record, "verification_status", "") or "deferred").strip().lower()
        hypothesis_id = str(getattr(record, "hypothesis_id", "") or "").strip()
        signature = _verification_signature(record)
        if not signature:
            continue
        if repair_action in {"", "none"} and verification_status == "supported":
            continue
        symbols = _record_terms(getattr(record, "symbols", ()) or ())
        preconditions: List[Tuple[int, int]] = [_verification_fact(verification_status, signature)]
        add_effects: List[Tuple[int, int]] = []
        del_effects: List[Tuple[int, int]] = []
        resource_status = "active" if verification_status == "supported" else "hypothetical"
        for symbol in symbols[:2]:
            preconditions.append(_resource_fact(resource_status, symbol))
        if repair_action == "trigger_hidden_cause_abduction":
            add_effects.extend([(302, int(goal_id)), (305, int(goal_id))])
        elif repair_action == "trigger_temporal_repair":
            add_effects.append((301, int(goal_id)))
            del_effects.append((303, int(goal_id)))
        elif repair_action == "accept_to_world_state":
            add_effects.append((303, int(goal_id)))
        else:
            add_effects.append((305, int(goal_id)))
        if bool(getattr(record, "hidden_cause_candidate", False)):
            add_effects.append((302, int(goal_id)))
        specs.append(
            PlannerBridgeOperatorSpec(
                operator_id=f"verification:{repair_action or verification_status}:{hypothesis_id or goal_id}",
                source="verification_bridge",
                status="hypothetical" if verification_status != "supported" else "active",
                predicate=repair_action or f"verification_{verification_status}",
                preconditions=_unique_fact_tuples(preconditions),
                add_effects=_unique_fact_tuples(add_effects),
                del_effects=_unique_fact_tuples(del_effects),
                priority=float(
                    0.14
                    + 0.26 * _clip01(getattr(record, "support", 0.0))
                    + 0.18 * _clip01(getattr(record, "conflict", 0.0))
                    + (
                        0.16
                        if bool(getattr(record, "hidden_cause_candidate", False))
                        else 0.0
                    )
                ),
            )
        )
    for directive in tuple(getattr(planner_state, "repair_directives", ()) or ()):
        action_type = str(getattr(directive, "action_type", "") or "repair").strip()
        target_id = str(getattr(directive, "target_id", "") or "").strip()
        symbols = tuple(str(item).strip() for item in (getattr(directive, "symbols", ()) or ()) if str(item).strip())
        if not action_type:
            continue
        preconditions: List[Tuple[int, int]] = []
        add_effects: List[Tuple[int, int]] = [(305, int(goal_id))]
        del_effects: List[Tuple[int, int]] = []
        for symbol in symbols[:2]:
            preconditions.append(_resource_fact("hypothetical", symbol))
        if not preconditions:
            preconditions.append((300, int(goal_id)))
        if action_type == "promote_world_model_supported_claim":
            add_effects.append((303, int(goal_id)))
        elif action_type == "trigger_hidden_cause_abduction":
            add_effects.append((302, int(goal_id)))
        elif action_type == "trigger_temporal_repair":
            add_effects.append((301, int(goal_id)))
            del_effects.append((303, int(goal_id)))
        specs.append(
            PlannerBridgeOperatorSpec(
                operator_id=f"repair:{action_type}:{target_id or goal_id}",
                status="hypothetical",
                predicate=action_type,
                preconditions=_unique_fact_tuples(preconditions),
                add_effects=_unique_fact_tuples(add_effects),
                del_effects=_unique_fact_tuples(del_effects),
                priority=float(
                    0.18
                    + 0.36 * _clip01(getattr(directive, "priority", 0.0))
                    + 0.26 * _clip01(getattr(directive, "pressure", 0.0))
                ),
            )
        )
    specs.sort(key=lambda item: (-float(item.priority), item.operator_id))
    return tuple(specs[:limit])

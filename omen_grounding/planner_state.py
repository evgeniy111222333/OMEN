from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .planner_semantics import (
    PlannerAlternativeWorld,
    PlannerOperator,
    PlannerResource,
    build_planner_alternative_worlds,
    build_planner_operators,
    build_planner_resources,
)


def _record_symbols(record: Any) -> Tuple[str, ...]:
    symbols = getattr(record, "symbols", None)
    if symbols is None:
        return tuple()
    return tuple(str(item) for item in symbols if str(item))


def _record_status(record: Any) -> str:
    status = str(getattr(record, "world_status", "") or "").strip().lower()
    if status:
        return status
    return "hypothetical"


def _record_type(record: Any) -> str:
    return str(getattr(record, "record_type", "") or "").strip().lower() or "unknown"


def _unique_strings(values: Sequence[str], *, limit: int) -> Tuple[str, ...]:
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


@dataclass(frozen=True)
class PlannerWorldState:
    ontology_records: Tuple[Any, ...] = field(default_factory=tuple)
    active_records: Tuple[Any, ...] = field(default_factory=tuple)
    hypothetical_records: Tuple[Any, ...] = field(default_factory=tuple)
    contradicted_records: Tuple[Any, ...] = field(default_factory=tuple)
    active_facts: Tuple[Any, ...] = field(default_factory=tuple)
    hypothetical_facts: Tuple[Any, ...] = field(default_factory=tuple)
    contradicted_facts: Tuple[Any, ...] = field(default_factory=tuple)
    goal_facts: Tuple[Any, ...] = field(default_factory=tuple)
    target_facts: Tuple[Any, ...] = field(default_factory=tuple)
    symbolic_facts: Tuple[Any, ...] = field(default_factory=tuple)
    resource_symbols: Tuple[str, ...] = field(default_factory=tuple)
    resources: Tuple[PlannerResource, ...] = field(default_factory=tuple)
    world_rule_symbols: Tuple[str, ...] = field(default_factory=tuple)
    hypothetical_rule_symbols: Tuple[str, ...] = field(default_factory=tuple)
    contradiction_symbols: Tuple[str, ...] = field(default_factory=tuple)
    destructive_effect_symbols: Tuple[str, ...] = field(default_factory=tuple)
    persistent_effect_symbols: Tuple[str, ...] = field(default_factory=tuple)
    operators: Tuple[PlannerOperator, ...] = field(default_factory=tuple)
    alternative_worlds: Tuple[PlannerAlternativeWorld, ...] = field(default_factory=tuple)
    primary_goal: Optional[Any] = None
    uncertainty: float = 0.0
    branching_pressure: float = 0.0
    contradiction_pressure: float = 0.0
    hidden_cause_pressure: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        summary = dict(self.metadata)
        summary.setdefault("planner_state_ontology_records", float(len(self.ontology_records)))
        summary.setdefault("planner_state_active_records", float(len(self.active_records)))
        summary.setdefault("planner_state_hypothetical_records", float(len(self.hypothetical_records)))
        summary.setdefault("planner_state_contradicted_records", float(len(self.contradicted_records)))
        summary.setdefault("planner_state_active_facts", float(len(self.active_facts)))
        summary.setdefault("planner_state_hypothetical_facts", float(len(self.hypothetical_facts)))
        summary.setdefault("planner_state_contradicted_facts", float(len(self.contradicted_facts)))
        summary.setdefault("planner_state_goal_facts", float(len(self.goal_facts)))
        summary.setdefault("planner_state_target_facts", float(len(self.target_facts)))
        summary.setdefault("planner_state_symbolic_facts", float(len(self.symbolic_facts)))
        summary.setdefault("planner_state_resources", float(len(self.resource_symbols)))
        summary.setdefault("planner_state_resource_records", float(len(self.resources)))
        summary.setdefault("planner_state_world_rules", float(len(self.world_rule_symbols)))
        summary.setdefault("planner_state_hypothetical_rules", float(len(self.hypothetical_rule_symbols)))
        summary.setdefault("planner_state_contradictions", float(len(self.contradiction_symbols)))
        summary.setdefault("planner_state_destructive_effects", float(len(self.destructive_effect_symbols)))
        summary.setdefault("planner_state_persistent_effects", float(len(self.persistent_effect_symbols)))
        summary.setdefault("planner_state_operators", float(len(self.operators)))
        summary.setdefault(
            "planner_state_active_operators",
            float(sum(1 for operator in self.operators if operator.status == "active")),
        )
        summary.setdefault(
            "planner_state_hypothetical_operators",
            float(sum(1 for operator in self.operators if operator.status == "hypothetical")),
        )
        summary.setdefault(
            "planner_state_contradicted_operators",
            float(sum(1 for operator in self.operators if operator.status == "contradicted")),
        )
        summary.setdefault("planner_state_alternative_world_records", float(len(self.alternative_worlds)))
        summary.setdefault("planner_state_uncertainty", float(self.uncertainty))
        summary.setdefault("planner_state_branching_pressure", float(self.branching_pressure))
        summary.setdefault("planner_state_contradiction_pressure", float(self.contradiction_pressure))
        summary.setdefault("planner_state_hidden_cause_pressure", float(self.hidden_cause_pressure))
        summary.setdefault(
            "planner_state_alternative_worlds",
            float((1 if self.hypothetical_records else 0) + (1 if self.contradicted_records else 0)),
        )
        return summary


def build_planner_world_state(task_context: Any) -> PlannerWorldState:
    records = tuple(getattr(task_context, "grounding_world_state_records", ()) or ())
    ontology_records = tuple(getattr(task_context, "grounding_ontology_records", ()) or ())
    active_records = tuple(record for record in records if _record_status(record) == "active")
    hypothetical_records = tuple(record for record in records if _record_status(record) == "hypothetical")
    contradicted_records = tuple(record for record in records if _record_status(record) == "contradicted")

    active_facts = tuple(sorted(getattr(task_context, "grounding_world_state_active_facts", ()) or (), key=repr))
    hypothetical_facts = tuple(
        sorted(getattr(task_context, "grounding_world_state_hypothetical_facts", ()) or (), key=repr)
    )
    contradicted_facts = tuple(
        sorted(getattr(task_context, "grounding_world_state_contradicted_facts", ()) or (), key=repr)
    )

    goal_facts: List[Any] = []
    primary_goal = getattr(task_context, "goal", None)
    if primary_goal is not None:
        goal_facts.append(primary_goal)
    for fact in tuple(sorted(getattr(task_context, "target_facts", ()) or (), key=repr)):
        if fact not in goal_facts:
            goal_facts.append(fact)

    planner_fn = getattr(task_context, "planner_facts", None)
    symbolic_facts = tuple(
        sorted(
            planner_fn() if callable(planner_fn) else tuple(getattr(task_context, "observed_facts", ()) or ()),
            key=repr,
        )
    )
    branching_pressure = float(
        getattr(task_context, "metadata", {}).get("grounding_world_state_branching_pressure", 0.0)
    )
    contradiction_pressure = float(
        getattr(task_context, "metadata", {}).get("grounding_world_state_contradiction_pressure", 0.0)
    )

    planner_records = (*active_records, *hypothetical_records, *contradicted_records, *ontology_records)
    resources = build_planner_resources(planner_records, limit=32)
    operators = build_planner_operators((*active_records, *hypothetical_records, *contradicted_records), limit=32)
    alternative_worlds = build_planner_alternative_worlds(
        active_records,
        hypothetical_records,
        contradicted_records,
        branching_pressure=branching_pressure,
        contradiction_pressure=contradiction_pressure,
        limit=3,
    )

    resource_symbols = _unique_strings(
        [resource.symbol for resource in resources],
        limit=32,
    )
    world_rule_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in active_records
            if _record_type(record) == "relation"
        ],
        limit=24,
    )
    hypothetical_rule_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in hypothetical_records
            if _record_type(record) == "relation"
        ],
        limit=24,
    )
    contradiction_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in contradicted_records
        ],
        limit=24,
    )
    destructive_effect_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in (*hypothetical_records, *contradicted_records)
            if _record_type(record) in {"relation", "state"}
        ],
        limit=24,
    )
    persistent_effect_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in active_records
            if _record_type(record) in {"relation", "state", "goal"}
        ],
        limit=24,
    )

    metadata = {
        "planner_state_ontology_records": float(len(ontology_records)),
        "planner_state_active_records": float(len(active_records)),
        "planner_state_hypothetical_records": float(len(hypothetical_records)),
        "planner_state_contradicted_records": float(len(contradicted_records)),
        "planner_state_active_facts": float(len(active_facts)),
        "planner_state_hypothetical_facts": float(len(hypothetical_facts)),
        "planner_state_contradicted_facts": float(len(contradicted_facts)),
        "planner_state_goal_facts": float(len(goal_facts)),
        "planner_state_target_facts": float(len(getattr(task_context, "target_facts", ()) or ())),
        "planner_state_symbolic_facts": float(len(symbolic_facts)),
        "planner_state_resources": float(len(resource_symbols)),
        "planner_state_resource_records": float(len(resources)),
        "planner_state_world_rules": float(len(world_rule_symbols)),
        "planner_state_hypothetical_rules": float(len(hypothetical_rule_symbols)),
        "planner_state_contradictions": float(len(contradiction_symbols)),
        "planner_state_destructive_effects": float(len(destructive_effect_symbols)),
        "planner_state_persistent_effects": float(len(persistent_effect_symbols)),
        "planner_state_operators": float(len(operators)),
        "planner_state_active_operators": float(sum(1 for operator in operators if operator.status == "active")),
        "planner_state_hypothetical_operators": float(
            sum(1 for operator in operators if operator.status == "hypothetical")
        ),
        "planner_state_contradicted_operators": float(
            sum(1 for operator in operators if operator.status == "contradicted")
        ),
        "planner_state_alternative_world_records": float(len(alternative_worlds)),
        "planner_state_uncertainty": float(getattr(task_context, "metadata", {}).get("grounding_uncertainty", 0.0)),
        "planner_state_branching_pressure": branching_pressure,
        "planner_state_contradiction_pressure": contradiction_pressure,
        "planner_state_hidden_cause_pressure": float(
            getattr(task_context, "metadata", {}).get("grounding_hidden_cause_pressure", 0.0)
        ),
    }

    return PlannerWorldState(
        ontology_records=ontology_records,
        active_records=active_records,
        hypothetical_records=hypothetical_records,
        contradicted_records=contradicted_records,
        active_facts=active_facts,
        hypothetical_facts=hypothetical_facts,
        contradicted_facts=contradicted_facts,
        goal_facts=tuple(goal_facts),
        target_facts=tuple(sorted(getattr(task_context, "target_facts", ()) or (), key=repr)),
        symbolic_facts=symbolic_facts,
        resource_symbols=resource_symbols,
        resources=resources,
        world_rule_symbols=world_rule_symbols,
        hypothetical_rule_symbols=hypothetical_rule_symbols,
        contradiction_symbols=contradiction_symbols,
        destructive_effect_symbols=destructive_effect_symbols,
        persistent_effect_symbols=persistent_effect_symbols,
        operators=operators,
        alternative_worlds=alternative_worlds,
        primary_goal=primary_goal,
        uncertainty=float(metadata["planner_state_uncertainty"]),
        branching_pressure=float(metadata["planner_state_branching_pressure"]),
        contradiction_pressure=float(metadata["planner_state_contradiction_pressure"]),
        hidden_cause_pressure=float(metadata["planner_state_hidden_cause_pressure"]),
        metadata=metadata,
    )

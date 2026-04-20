from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .planner_guidance import PlannerConstraint, PlannerRepairDirective, build_planner_guidance
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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _task_context_grounding_values(task_context: Any, field_name: str) -> Tuple[Any, ...]:
    direct_source = getattr(task_context, field_name, ())
    if callable(direct_source):
        direct_source = direct_source()
    direct = tuple(direct_source or ())
    if direct:
        return direct
    artifacts = getattr(task_context, "grounding_artifacts", None)
    if artifacts is not None:
        artifact_values = tuple(getattr(artifacts, field_name, ()) or ())
        if artifact_values:
            return artifact_values
    trace_bundle = getattr(task_context, "execution_trace", None)
    if trace_bundle is not None:
        return tuple(getattr(trace_bundle, field_name, ()) or ())
    return tuple()


def _record_graph_terms(record: Any) -> Tuple[str, ...]:
    terms = getattr(record, "graph_terms", None)
    if terms is None:
        return tuple()
    return tuple(str(item).strip() for item in terms if str(item).strip())


def _record_provenance(record: Any) -> Tuple[str, ...]:
    provenance = getattr(record, "provenance", None)
    if provenance is None:
        return tuple()
    return tuple(str(item).strip() for item in provenance if str(item).strip())


def _verification_world_status(record: Any) -> str:
    status = str(getattr(record, "verification_status", "") or "").strip().lower()
    if status in {"supported", "accept", "accepted"}:
        return "active"
    if status in {"conflicted", "rejected", "unsupported"}:
        return "contradicted"
    return "hypothetical"


def _hypothesis_world_status(record: Any) -> str:
    if str(getattr(record, "conflict_tag", "") or "").strip():
        return "contradicted"
    return "hypothetical"


@dataclass(frozen=True)
class _PlannerBridgeRecord:
    record_id: str
    record_type: str
    world_status: str
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    support: float = 0.0
    conflict: float = 0.0
    confidence: float = 0.0
    repair_action: str = "none"
    provenance: Tuple[str, ...] = field(default_factory=tuple)


def _bridge_verification_record(record: Any) -> _PlannerBridgeRecord:
    support = _clip01(getattr(record, "support", 0.0))
    conflict = _clip01(getattr(record, "conflict", 0.0))
    return _PlannerBridgeRecord(
        record_id=f"verification:{getattr(record, 'hypothesis_id', 'unknown')}",
        record_type=_record_type(record),
        world_status=_verification_world_status(record),
        symbols=_record_symbols(record),
        support=support,
        conflict=conflict,
        confidence=_clip01((0.65 * support) + (0.35 * (1.0 - conflict))),
        repair_action=str(getattr(record, "repair_action", "none") or "none"),
        provenance=_record_provenance(record),
    )


def _bridge_hypothesis_record(record: Any) -> _PlannerBridgeRecord:
    confidence = _clip01(getattr(record, "confidence", 0.0))
    conflict_tag = str(getattr(record, "conflict_tag", "") or "").strip()
    deferred = bool(getattr(record, "deferred", False))
    return _PlannerBridgeRecord(
        record_id=f"hypothesis:{getattr(record, 'hypothesis_id', 'unknown')}",
        record_type=str(getattr(record, "kind", "") or "unknown").strip().lower() or "unknown",
        world_status=_hypothesis_world_status(record),
        symbols=_record_symbols(record),
        support=confidence,
        conflict=0.72 if conflict_tag else (_clip01(1.0 - confidence) if deferred else 0.0),
        confidence=confidence,
        repair_action=(
            "preserve_conflict_scope"
            if conflict_tag
            else ("keep_multiple_hypotheses_alive" if deferred else "none")
        ),
        provenance=_record_provenance(record),
    )


def _bridge_graph_record(record: Any) -> _PlannerBridgeRecord:
    return _PlannerBridgeRecord(
        record_id=f"graph:{getattr(record, 'record_id', 'unknown')}",
        record_type=_record_type(record),
        world_status="hypothetical",
        symbols=_record_graph_terms(record),
        support=0.0,
        conflict=0.0,
        confidence=_clip01(getattr(record, "confidence", 0.0)),
        repair_action="none",
        provenance=(
            str(getattr(record, "graph_family", "") or "").strip(),
            str(getattr(record, "graph_key", "") or "").strip(),
        ),
    )


def _lineage_symbols(
    verification_records: Sequence[Any],
    hypothesis_records: Sequence[Any],
    graph_records: Sequence[Any],
    *,
    limit: int,
) -> Tuple[str, ...]:
    symbols: List[str] = []
    for record in graph_records:
        symbols.extend(_record_graph_terms(record))
        family = str(getattr(record, "graph_family", "") or "").strip()
        if family:
            symbols.append(family)
    for record in (*verification_records, *hypothesis_records):
        symbols.extend(_record_provenance(record))
    return _unique_strings(symbols, limit=limit)


@dataclass(frozen=True)
class PlannerWorldState:
    ontology_records: Tuple[Any, ...] = field(default_factory=tuple)
    active_records: Tuple[Any, ...] = field(default_factory=tuple)
    hypothetical_records: Tuple[Any, ...] = field(default_factory=tuple)
    contradicted_records: Tuple[Any, ...] = field(default_factory=tuple)
    verification_records: Tuple[Any, ...] = field(default_factory=tuple)
    hypothesis_records: Tuple[Any, ...] = field(default_factory=tuple)
    graph_records: Tuple[Any, ...] = field(default_factory=tuple)
    active_facts: Tuple[Any, ...] = field(default_factory=tuple)
    hypothetical_facts: Tuple[Any, ...] = field(default_factory=tuple)
    contradicted_facts: Tuple[Any, ...] = field(default_factory=tuple)
    goal_facts: Tuple[Any, ...] = field(default_factory=tuple)
    target_facts: Tuple[Any, ...] = field(default_factory=tuple)
    symbolic_facts: Tuple[Any, ...] = field(default_factory=tuple)
    resource_symbols: Tuple[str, ...] = field(default_factory=tuple)
    resources: Tuple[PlannerResource, ...] = field(default_factory=tuple)
    constraints: Tuple[PlannerConstraint, ...] = field(default_factory=tuple)
    repair_directives: Tuple[PlannerRepairDirective, ...] = field(default_factory=tuple)
    world_rule_symbols: Tuple[str, ...] = field(default_factory=tuple)
    hypothetical_rule_symbols: Tuple[str, ...] = field(default_factory=tuple)
    contradiction_symbols: Tuple[str, ...] = field(default_factory=tuple)
    destructive_effect_symbols: Tuple[str, ...] = field(default_factory=tuple)
    persistent_effect_symbols: Tuple[str, ...] = field(default_factory=tuple)
    lineage_symbols: Tuple[str, ...] = field(default_factory=tuple)
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
        summary.setdefault("planner_state_verification_records", float(len(self.verification_records)))
        summary.setdefault(
            "planner_state_supported_verification_records",
            float(
                sum(
                    1
                    for record in self.verification_records
                    if str(getattr(record, "verification_status", "") or "").strip().lower() == "supported"
                )
            ),
        )
        summary.setdefault(
            "planner_state_deferred_verification_records",
            float(
                sum(
                    1
                    for record in self.verification_records
                    if str(getattr(record, "verification_status", "") or "").strip().lower() == "deferred"
                )
            ),
        )
        summary.setdefault(
            "planner_state_conflicted_verification_records",
            float(
                sum(
                    1
                    for record in self.verification_records
                    if str(getattr(record, "verification_status", "") or "").strip().lower() == "conflicted"
                )
            ),
        )
        summary.setdefault(
            "planner_state_hidden_cause_candidates",
            float(sum(1 for record in self.verification_records if bool(getattr(record, "hidden_cause_candidate", False)))),
        )
        summary.setdefault("planner_state_hypothesis_records", float(len(self.hypothesis_records)))
        summary.setdefault(
            "planner_state_deferred_hypotheses",
            float(sum(1 for record in self.hypothesis_records if bool(getattr(record, "deferred", False)))),
        )
        summary.setdefault(
            "planner_state_conflicted_hypotheses",
            float(sum(1 for record in self.hypothesis_records if str(getattr(record, "conflict_tag", "") or "").strip())),
        )
        summary.setdefault("planner_state_graph_records", float(len(self.graph_records)))
        summary.setdefault(
            "planner_state_graph_relation_records",
            float(sum(1 for record in self.graph_records if _record_type(record) == "relation")),
        )
        summary.setdefault("planner_state_lineage_symbols", float(len(self.lineage_symbols)))
        summary.setdefault("planner_state_active_facts", float(len(self.active_facts)))
        summary.setdefault("planner_state_hypothetical_facts", float(len(self.hypothetical_facts)))
        summary.setdefault("planner_state_contradicted_facts", float(len(self.contradicted_facts)))
        summary.setdefault("planner_state_goal_facts", float(len(self.goal_facts)))
        summary.setdefault("planner_state_target_facts", float(len(self.target_facts)))
        summary.setdefault("planner_state_symbolic_facts", float(len(self.symbolic_facts)))
        summary.setdefault("planner_state_resources", float(len(self.resource_symbols)))
        summary.setdefault("planner_state_resource_records", float(len(self.resources)))
        summary.setdefault("planner_state_constraints", float(len(self.constraints)))
        summary.setdefault(
            "planner_state_prefer_constraints",
            float(sum(1 for constraint in self.constraints if constraint.enforcement == "prefer")),
        )
        summary.setdefault(
            "planner_state_branch_constraints",
            float(sum(1 for constraint in self.constraints if constraint.enforcement == "branch")),
        )
        summary.setdefault(
            "planner_state_avoid_constraints",
            float(sum(1 for constraint in self.constraints if constraint.enforcement == "avoid")),
        )
        summary.setdefault("planner_state_repair_directives", float(len(self.repair_directives)))
        summary.setdefault(
            "planner_state_constraint_pressure",
            (
                sum(float(constraint.pressure) for constraint in self.constraints) / max(float(len(self.constraints)), 1.0)
                if self.constraints else 0.0
            ),
        )
        summary.setdefault(
            "planner_state_repair_pressure",
            (
                sum(float(directive.pressure) for directive in self.repair_directives)
                / max(float(len(self.repair_directives)), 1.0)
                if self.repair_directives else 0.0
            ),
        )
        summary.setdefault("planner_state_world_rules", float(len(self.world_rule_symbols)))
        summary.setdefault("planner_state_hypothetical_rules", float(len(self.hypothetical_rule_symbols)))
        summary.setdefault("planner_state_contradictions", float(len(self.contradiction_symbols)))
        summary.setdefault("planner_state_destructive_effects", float(len(self.destructive_effect_symbols)))
        summary.setdefault("planner_state_persistent_effects", float(len(self.persistent_effect_symbols)))
        summary.setdefault("planner_state_operators", float(len(self.operators)))
        summary.setdefault(
            "planner_state_conditional_operators",
            float(sum(1 for operator in self.operators if operator.conditions)),
        )
        summary.setdefault(
            "planner_state_causal_operators",
            float(sum(1 for operator in self.operators if operator.causes)),
        )
        summary.setdefault(
            "planner_state_temporal_operators",
            float(sum(1 for operator in self.operators if operator.temporals)),
        )
        summary.setdefault(
            "planner_state_modal_operators",
            float(sum(1 for operator in self.operators if operator.modality)),
        )
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
            float(len(self.alternative_worlds)),
        )
        return summary


def build_planner_world_state(task_context: Any) -> PlannerWorldState:
    records = _task_context_grounding_values(task_context, "grounding_world_state_records")
    ontology_records = _task_context_grounding_values(task_context, "grounding_ontology_records")
    verification_records = _task_context_grounding_values(task_context, "grounding_verification_records")
    hypothesis_records = _task_context_grounding_values(task_context, "grounding_hypotheses")
    graph_records = _task_context_grounding_values(task_context, "grounding_graph_records")
    active_records = tuple(record for record in records if _record_status(record) == "active")
    hypothetical_records = tuple(record for record in records if _record_status(record) == "hypothetical")
    contradicted_records = tuple(record for record in records if _record_status(record) == "contradicted")

    active_facts = tuple(sorted(_task_context_grounding_values(task_context, "grounding_world_state_active_facts"), key=repr))
    hypothetical_facts = tuple(
        sorted(_task_context_grounding_values(task_context, "grounding_world_state_hypothetical_facts"), key=repr)
    )
    contradicted_facts = tuple(
        sorted(_task_context_grounding_values(task_context, "grounding_world_state_contradicted_facts"), key=repr)
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
    verification_bridge_records = tuple(_bridge_verification_record(record) for record in verification_records)
    hypothesis_bridge_records = tuple(_bridge_hypothesis_record(record) for record in hypothesis_records)
    graph_bridge_records = tuple(_bridge_graph_record(record) for record in graph_records)
    planner_active_records = tuple(
        (*active_records, *(record for record in verification_bridge_records if _record_status(record) == "active"))
    )
    planner_hypothetical_records = tuple(
        (
            *hypothetical_records,
            *(record for record in verification_bridge_records if _record_status(record) == "hypothetical"),
            *hypothesis_bridge_records,
            *graph_bridge_records,
        )
    )
    planner_contradicted_records = tuple(
        (
            *contradicted_records,
            *(record for record in verification_bridge_records if _record_status(record) == "contradicted"),
            *(record for record in hypothesis_bridge_records if _record_status(record) == "contradicted"),
        )
    )

    planner_records = (
        *planner_active_records,
        *planner_hypothetical_records,
        *planner_contradicted_records,
        *ontology_records,
    )
    resources = build_planner_resources(planner_records, limit=32)
    operators = build_planner_operators(
        (*planner_active_records, *planner_hypothetical_records, *planner_contradicted_records),
        limit=32,
    )
    constraints, repair_directives = build_planner_guidance(
        _task_context_grounding_values(task_context, "grounding_validation_records"),
        _task_context_grounding_values(task_context, "grounding_repair_actions"),
        limit=32,
    )
    alternative_worlds = build_planner_alternative_worlds(
        planner_active_records,
        planner_hypothetical_records,
        planner_contradicted_records,
        branching_pressure=branching_pressure,
        contradiction_pressure=contradiction_pressure,
        limit=3,
    )
    lineage_symbols = _lineage_symbols(
        verification_records,
        hypothesis_records,
        graph_records,
        limit=48,
    )

    resource_symbols = _unique_strings(
        [resource.symbol for resource in resources],
        limit=32,
    )
    world_rule_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in planner_active_records
            if _record_type(record) == "relation"
        ],
        limit=24,
    )
    hypothetical_rule_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in planner_hypothetical_records
            if _record_type(record) == "relation"
        ],
        limit=24,
    )
    contradiction_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in planner_contradicted_records
        ],
        limit=24,
    )
    destructive_effect_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in (*planner_hypothetical_records, *planner_contradicted_records)
            if _record_type(record) in {"relation", "state"}
        ],
        limit=24,
    )
    persistent_effect_symbols = _unique_strings(
        [
            " | ".join(_record_symbols(record))
            for record in planner_active_records
            if _record_type(record) in {"relation", "state", "goal"}
        ],
        limit=24,
    )

    metadata = {
        "planner_state_ontology_records": float(len(ontology_records)),
        "planner_state_active_records": float(len(active_records)),
        "planner_state_hypothetical_records": float(len(hypothetical_records)),
        "planner_state_contradicted_records": float(len(contradicted_records)),
        "planner_state_verification_records": float(len(verification_records)),
        "planner_state_supported_verification_records": float(
            sum(
                1
                for record in verification_records
                if str(getattr(record, "verification_status", "") or "").strip().lower() == "supported"
            )
        ),
        "planner_state_deferred_verification_records": float(
            sum(
                1
                for record in verification_records
                if str(getattr(record, "verification_status", "") or "").strip().lower() == "deferred"
            )
        ),
        "planner_state_conflicted_verification_records": float(
            sum(
                1
                for record in verification_records
                if str(getattr(record, "verification_status", "") or "").strip().lower() == "conflicted"
            )
        ),
        "planner_state_hidden_cause_candidates": float(
            sum(1 for record in verification_records if bool(getattr(record, "hidden_cause_candidate", False)))
        ),
        "planner_state_hypothesis_records": float(len(hypothesis_records)),
        "planner_state_deferred_hypotheses": float(
            sum(1 for record in hypothesis_records if bool(getattr(record, "deferred", False)))
        ),
        "planner_state_conflicted_hypotheses": float(
            sum(1 for record in hypothesis_records if str(getattr(record, "conflict_tag", "") or "").strip())
        ),
        "planner_state_graph_records": float(len(graph_records)),
        "planner_state_graph_relation_records": float(
            sum(1 for record in graph_records if _record_type(record) == "relation")
        ),
        "planner_state_lineage_symbols": float(len(lineage_symbols)),
        "planner_state_active_facts": float(len(active_facts)),
        "planner_state_hypothetical_facts": float(len(hypothetical_facts)),
        "planner_state_contradicted_facts": float(len(contradicted_facts)),
        "planner_state_goal_facts": float(len(goal_facts)),
        "planner_state_target_facts": float(len(getattr(task_context, "target_facts", ()) or ())),
        "planner_state_symbolic_facts": float(len(symbolic_facts)),
        "planner_state_resources": float(len(resource_symbols)),
        "planner_state_resource_records": float(len(resources)),
        "planner_state_constraints": float(len(constraints)),
        "planner_state_prefer_constraints": float(sum(1 for constraint in constraints if constraint.enforcement == "prefer")),
        "planner_state_branch_constraints": float(sum(1 for constraint in constraints if constraint.enforcement == "branch")),
        "planner_state_avoid_constraints": float(sum(1 for constraint in constraints if constraint.enforcement == "avoid")),
        "planner_state_repair_directives": float(len(repair_directives)),
        "planner_state_constraint_pressure": (
            sum(float(constraint.pressure) for constraint in constraints) / max(float(len(constraints)), 1.0)
            if constraints else 0.0
        ),
        "planner_state_repair_pressure": (
            sum(float(directive.pressure) for directive in repair_directives)
            / max(float(len(repair_directives)), 1.0)
            if repair_directives else 0.0
        ),
        "planner_state_world_rules": float(len(world_rule_symbols)),
        "planner_state_hypothetical_rules": float(len(hypothetical_rule_symbols)),
        "planner_state_contradictions": float(len(contradiction_symbols)),
        "planner_state_destructive_effects": float(len(destructive_effect_symbols)),
        "planner_state_persistent_effects": float(len(persistent_effect_symbols)),
        "planner_state_operators": float(len(operators)),
        "planner_state_conditional_operators": float(sum(1 for operator in operators if operator.conditions)),
        "planner_state_causal_operators": float(sum(1 for operator in operators if operator.causes)),
        "planner_state_temporal_operators": float(sum(1 for operator in operators if operator.temporals)),
        "planner_state_modal_operators": float(sum(1 for operator in operators if operator.modality)),
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
        "planner_state_alternative_worlds": float(len(alternative_worlds)),
    }

    return PlannerWorldState(
        ontology_records=ontology_records,
        active_records=active_records,
        hypothetical_records=hypothetical_records,
        contradicted_records=contradicted_records,
        verification_records=verification_records,
        hypothesis_records=hypothesis_records,
        graph_records=graph_records,
        active_facts=active_facts,
        hypothetical_facts=hypothetical_facts,
        contradicted_facts=contradicted_facts,
        goal_facts=tuple(goal_facts),
        target_facts=tuple(sorted(getattr(task_context, "target_facts", ()) or (), key=repr)),
        symbolic_facts=symbolic_facts,
        resource_symbols=resource_symbols,
        resources=resources,
        constraints=constraints,
        repair_directives=repair_directives,
        world_rule_symbols=world_rule_symbols,
        hypothetical_rule_symbols=hypothetical_rule_symbols,
        contradiction_symbols=contradiction_symbols,
        destructive_effect_symbols=destructive_effect_symbols,
        persistent_effect_symbols=persistent_effect_symbols,
        lineage_symbols=lineage_symbols,
        operators=operators,
        alternative_worlds=alternative_worlds,
        primary_goal=primary_goal,
        uncertainty=float(metadata["planner_state_uncertainty"]),
        branching_pressure=float(metadata["planner_state_branching_pressure"]),
        contradiction_pressure=float(metadata["planner_state_contradiction_pressure"]),
        hidden_cause_pressure=max(
            float(metadata["planner_state_hidden_cause_pressure"]),
            max(
                (
                    float(directive.pressure)
                    for directive in repair_directives
                    if directive.action_type == "trigger_hidden_cause_abduction"
                ),
                default=0.0,
            ),
        ),
        metadata=metadata,
    )

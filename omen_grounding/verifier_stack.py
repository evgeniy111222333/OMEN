from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .memory_hints import grounding_memory_status
from .ontology_growth import GroundingOntologyConcept
from .scene_types import SemanticSceneGraph
from .verification import GroundingVerificationRecord
from .world_state_writeback import GroundingWorldStateRecord


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _record_terms(record: object) -> Tuple[str, ...]:
    return tuple(str(item) for item in (getattr(record, "symbols", ()) or ()) if str(item).strip())


def _status_from_scores(support: float, conflict: float) -> str:
    if conflict >= 0.60:
        return "conflicted"
    if support >= 0.55 and conflict <= 0.48:
        return "supported"
    return "deferred"


@dataclass(frozen=True)
class GroundingValidationRecord:
    validation_id: str
    target_id: str
    validator_family: str
    validation_status: str
    source_segment: int
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    support: float = 0.0
    conflict: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    provenance: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def graph_key(self) -> str:
        return f"grounding_validation:{self.validator_family}:{self.validation_status}:{self.target_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [self.validator_family, self.validation_status, *self.symbols]
        if self.rationale:
            terms.append(self.rationale)
        return tuple(str(term) for term in terms if str(term).strip())

    @property
    def graph_family(self) -> str:
        return f"grounding_validation:{self.validator_family}:{self.validation_status}"

    @property
    def graph_text(self) -> str:
        symbol_text = " | ".join(self.symbols)
        return (
            f"{self.validator_family} {self.validation_status} "
            f"support={self.support:.2f} conflict={self.conflict:.2f} "
            f"confidence={self.confidence:.2f} rationale={self.rationale} {symbol_text}"
        ).strip()


@dataclass(frozen=True)
class GroundingRepairAction:
    action_id: str
    target_id: str
    action_type: str
    priority: float
    pressure: float
    reason: str
    source_segment: int
    provenance: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def graph_key(self) -> str:
        return f"grounding_repair:{self.action_type}:{self.target_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        return tuple(
            str(term)
            for term in (self.action_type, self.reason, self.target_id, str(self.source_segment))
            if str(term).strip()
        )

    @property
    def graph_family(self) -> str:
        return "grounding_repair"

    @property
    def graph_text(self) -> str:
        return (
            f"repair {self.action_type} priority={self.priority:.2f} "
            f"pressure={self.pressure:.2f} reason={self.reason} target={self.target_id}"
        )


@dataclass
class GroundingVerifierStackResult:
    validation_records: Tuple[GroundingValidationRecord, ...] = field(default_factory=tuple)
    repair_actions: Tuple[GroundingRepairAction, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)


def _same_world_record(
    record: GroundingWorldStateRecord,
    *,
    kind: str,
    symbols: Tuple[str, ...],
    status: str,
) -> bool:
    return (
        str(record.record_type or "").strip().lower() == str(kind).strip().lower()
        and str(record.world_status or "").strip().lower() == status
        and tuple(str(item) for item in record.symbols) == tuple(symbols)
    )


def _ontology_overlap(symbols: Tuple[str, ...], concepts: Sequence[GroundingOntologyConcept]) -> float:
    symbol_set = {str(item).strip() for item in symbols if str(item).strip()}
    if not symbol_set:
        return 0.0
    best = 0.0
    for concept in concepts:
        concept_terms = {
            str(term).strip()
            for term in (*concept.signature_terms, *concept.member_terms, *concept.symbols)
            if str(term).strip()
        }
        overlap = len(symbol_set.intersection(concept_terms)) / max(len(symbol_set), 1)
        if overlap > best:
            best = overlap
    return _clip01(best)


def _segment_marker_density(
    scene: SemanticSceneGraph,
    segment_index: int,
) -> Dict[str, float]:
    discourse = [record for record in scene.discourse_relations if int(record.target_segment) == int(segment_index)]
    temporals = [record for record in scene.temporal_markers if int(record.source_segment) == int(segment_index)]
    explanations = [record for record in scene.explanations if int(record.target_segment or record.source_segment) == int(segment_index)]
    coreference = [record for record in scene.coreference_links if int(record.source_segment) == int(segment_index)]
    sequence = sum(1 for record in discourse if record.relation_type in {"sequence", "condition", "cause"})
    contrast = sum(1 for record in discourse if record.relation_type == "contrast")
    timestamps = sum(1 for record in temporals if record.marker_type == "timestamp")
    durations = sum(1 for record in temporals if record.marker_type == "duration")
    return {
        "sequence": float(sequence),
        "contrast": float(contrast),
        "timestamps": float(timestamps),
        "durations": float(durations),
        "explanations": float(len(explanations)),
        "coreference": float(len(coreference)),
    }


def _memory_overlap_score(
    symbols: Tuple[str, ...],
    memory_records: Sequence[object],
) -> Dict[str, float | str]:
    empty: Dict[str, float | str] = {
        "active_overlap": 0.0,
        "hypothetical_overlap": 0.0,
        "contradicted_overlap": 0.0,
        "support": 0.0,
        "conflict": 0.0,
        "rationale": "memory_unavailable",
    }
    if not memory_records:
        return empty
    symbol_set = {str(item).strip() for item in symbols if str(item).strip()}
    if not symbol_set:
        empty["rationale"] = "empty_symbols"
        return empty
    grouped_terms: Dict[str, set[str]] = {
        "active": set(),
        "hypothetical": set(),
        "contradicted": set(),
    }
    for record in memory_records:
        status = grounding_memory_status(record)
        if status not in grouped_terms:
            continue
        grouped_terms[status].update(
            str(term).strip()
            for term in (getattr(record, "graph_terms", ()) or ())
            if str(term).strip()
        )
    active_overlap = float(len(symbol_set.intersection(grouped_terms["active"]))) / max(float(len(symbol_set)), 1.0)
    hypothetical_overlap = float(len(symbol_set.intersection(grouped_terms["hypothetical"]))) / max(
        float(len(symbol_set)), 1.0
    )
    contradicted_overlap = float(len(symbol_set.intersection(grouped_terms["contradicted"]))) / max(
        float(len(symbol_set)), 1.0
    )
    support = _clip01((0.72 * active_overlap) + (0.18 * hypothetical_overlap))
    direct_overlap = max(active_overlap, hypothetical_overlap)
    if contradicted_overlap <= 0.0 and direct_overlap >= 0.95:
        support = max(support, 0.68 if active_overlap > 0.0 else 0.60)
    conflict = _clip01((0.74 * contradicted_overlap) + (0.12 * max(0.0, hypothetical_overlap - active_overlap)))
    rationale = "memory_unmatched"
    if contradicted_overlap > 0.0 and active_overlap > 0.0:
        rationale = "memory_status_split"
    elif contradicted_overlap > 0.0:
        rationale = "contradicted_memory_overlap"
    elif active_overlap > 0.0:
        rationale = "active_memory_overlap"
    elif hypothetical_overlap > 0.0:
        rationale = "hypothetical_memory_overlap"
    return {
        "active_overlap": float(active_overlap),
        "hypothetical_overlap": float(hypothetical_overlap),
        "contradicted_overlap": float(contradicted_overlap),
        "support": float(support),
        "conflict": float(conflict),
        "rationale": rationale,
    }


def run_grounding_verifier_stack(
    *,
    scene: SemanticSceneGraph,
    verification_records: Sequence[GroundingVerificationRecord],
    world_state_records: Sequence[GroundingWorldStateRecord],
    ontology_concepts: Sequence[GroundingOntologyConcept],
    memory_records: Optional[Sequence[object]] = None,
) -> GroundingVerifierStackResult:
    validations: List[GroundingValidationRecord] = []
    repairs: List[GroundingRepairAction] = []
    memory_records = tuple(memory_records or ())

    for record in verification_records:
        symbols = _record_terms(record)
        segment_index = int(record.segment_index)
        active_match = any(
            _same_world_record(world_record, kind=record.kind, symbols=symbols, status="active")
            for world_record in world_state_records
        )
        contradicted_match = any(
            _same_world_record(world_record, kind=record.kind, symbols=symbols, status="contradicted")
            for world_record in world_state_records
        )
        ontology_support = _ontology_overlap(symbols, ontology_concepts)
        world_support = _clip01(
            (0.42 * float(active_match))
            + (0.22 * float(record.support))
            + (0.20 * ontology_support)
            + (0.16 * (1.0 - float(contradicted_match)))
        )
        world_conflict = _clip01(
            (0.46 * float(contradicted_match))
            + (0.22 * float(record.conflict))
            + (0.14 * (1.0 - ontology_support))
            + (0.08 * float(record.hidden_cause_candidate))
        )
        world_status = _status_from_scores(world_support, world_conflict)
        validations.append(
            GroundingValidationRecord(
                validation_id=f"validation:world:{record.hypothesis_id}",
                target_id=str(record.hypothesis_id),
                validator_family="world_model",
                validation_status=world_status,
                source_segment=segment_index,
                symbols=symbols,
                support=world_support,
                conflict=world_conflict,
                confidence=_clip01(0.55 * world_support + 0.45 * (1.0 - world_conflict)),
                rationale="active_match" if active_match else ("contradicted_match" if contradicted_match else "ontology_overlap"),
                provenance=tuple(record.provenance),
            )
        )

        density = _segment_marker_density(scene, segment_index)
        temporal_support = _clip01(
            (0.26 * min(density["sequence"], 1.0))
            + (0.20 * min(density["timestamps"], 1.0))
            + (0.16 * min(density["durations"], 1.0))
            + (0.18 * min(density["explanations"], 1.0))
            + (0.08 * min(density["coreference"], 1.0))
            + (0.20 * (1.0 - float(record.conflict)))
        )
        temporal_conflict = _clip01(
            (0.34 * min(density["contrast"], 1.0))
            + (0.24 * float(record.conflict))
            + (0.18 * float(record.hidden_cause_candidate))
            + (0.12 * float(str(record.repair_action or "") == "preserve_conflict_scope"))
        )
        temporal_status = _status_from_scores(temporal_support, temporal_conflict)
        validations.append(
            GroundingValidationRecord(
                validation_id=f"validation:temporal:{record.hypothesis_id}",
                target_id=str(record.hypothesis_id),
                validator_family="temporal",
                validation_status=temporal_status,
                source_segment=segment_index,
                symbols=symbols,
                support=temporal_support,
                conflict=temporal_conflict,
                confidence=_clip01(0.50 * temporal_support + 0.50 * (1.0 - temporal_conflict)),
                rationale="temporal_markers" if density["timestamps"] or density["durations"] else "discourse_context",
                provenance=tuple(record.provenance),
            )
        )

        if memory_records:
            memory_overlap = _memory_overlap_score(symbols, memory_records)
            memory_support = _clip01(
                (0.62 * float(memory_overlap["support"]))
                + (0.18 * float(record.support))
                + (0.10 * (1.0 - float(record.conflict)))
                + (0.10 * float(memory_overlap["active_overlap"]))
            )
            memory_conflict = _clip01(
                (0.58 * float(memory_overlap["conflict"]))
                + (0.18 * float(record.conflict))
                + (0.10 * float(memory_overlap["contradicted_overlap"]))
                + (0.08 * max(0.0, 1.0 - float(memory_overlap["support"])))
            )
            memory_status = _status_from_scores(memory_support, memory_conflict)
            validations.append(
                GroundingValidationRecord(
                    validation_id=f"validation:memory:{record.hypothesis_id}",
                    target_id=str(record.hypothesis_id),
                    validator_family="memory_corroboration",
                    validation_status=memory_status,
                    source_segment=segment_index,
                    symbols=symbols,
                    support=memory_support,
                    conflict=memory_conflict,
                    confidence=_clip01(0.55 * memory_support + 0.45 * (1.0 - memory_conflict)),
                    rationale=str(memory_overlap["rationale"]),
                    provenance=tuple(record.provenance),
                )
            )

        hidden_cause_repair = bool(
            record.hidden_cause_candidate
            or world_status == "conflicted"
            or str(record.repair_action or "") == "trigger_hidden_cause_abduction"
        )
        if hidden_cause_repair:
            repairs.append(
                GroundingRepairAction(
                    action_id=f"repair:hidden_cause:{record.hypothesis_id}",
                    target_id=str(record.hypothesis_id),
                    action_type="trigger_hidden_cause_abduction",
                    priority=_clip01(0.55 + (0.25 * world_conflict) + (0.20 * float(hidden_cause_repair))),
                    pressure=_clip01(max(world_conflict, float(record.conflict))),
                    reason=(
                        "verification_hidden_cause_gap"
                        if str(record.repair_action or "") == "trigger_hidden_cause_abduction"
                        else "world_model_conflict"
                    ),
                    source_segment=segment_index,
                    provenance=tuple(record.provenance),
                )
            )
        if memory_records and memory_status == "conflicted":
            repairs.append(
                GroundingRepairAction(
                    action_id=f"repair:memory:{record.hypothesis_id}",
                    target_id=str(record.hypothesis_id),
                    action_type="trigger_memory_reconciliation",
                    priority=_clip01(0.44 + (0.34 * memory_conflict)),
                    pressure=_clip01(memory_conflict),
                    reason=str(memory_overlap["rationale"]),
                    source_segment=segment_index,
                    provenance=tuple(record.provenance),
                )
            )
        if temporal_status == "conflicted":
            repairs.append(
                GroundingRepairAction(
                    action_id=f"repair:temporal:{record.hypothesis_id}",
                    target_id=str(record.hypothesis_id),
                    action_type="trigger_temporal_repair",
                    priority=_clip01(0.48 + (0.32 * temporal_conflict)),
                    pressure=_clip01(temporal_conflict),
                    reason="temporal_inconsistency",
                    source_segment=segment_index,
                    provenance=tuple(record.provenance),
                )
            )
        if record.verification_status == "deferred" and ontology_support >= 0.5:
            repairs.append(
                GroundingRepairAction(
                    action_id=f"repair:ontology:{record.hypothesis_id}",
                    target_id=str(record.hypothesis_id),
                    action_type="keep_ontology_hypothesis_alive",
                    priority=_clip01(0.42 + (0.30 * ontology_support)),
                    pressure=_clip01(0.35 + (0.35 * ontology_support)),
                    reason="ontology_backing",
                    source_segment=segment_index,
                    provenance=tuple(record.provenance),
                )
            )
        if memory_records and record.verification_status == "deferred" and memory_status == "supported":
            repairs.append(
                GroundingRepairAction(
                    action_id=f"repair:memory_promote:{record.hypothesis_id}",
                    target_id=str(record.hypothesis_id),
                    action_type="promote_memory_corroborated_claim",
                    priority=_clip01(0.40 + (0.30 * memory_support)),
                    pressure=_clip01(0.20 + (0.25 * memory_support)),
                    reason="memory_backing",
                    source_segment=segment_index,
                    provenance=tuple(record.provenance),
                )
            )
        if world_status == "supported" and temporal_status == "supported" and record.verification_status == "supported":
            repairs.append(
                GroundingRepairAction(
                    action_id=f"repair:promote:{record.hypothesis_id}",
                    target_id=str(record.hypothesis_id),
                    action_type="promote_world_model_supported_claim",
                    priority=_clip01(0.40 + (0.30 * world_support) + (0.20 * temporal_support)),
                    pressure=_clip01(0.20 + (0.20 * world_support)),
                    reason="cross_validator_support",
                    source_segment=segment_index,
                    provenance=tuple(record.provenance),
                )
            )

    world_records = [record for record in validations if record.validator_family == "world_model"]
    temporal_records = [record for record in validations if record.validator_family == "temporal"]
    memory_validation_records = [record for record in validations if record.validator_family == "memory_corroboration"]
    metadata = {
        "verifier_stack_records": float(len(validations)),
        "verifier_stack_world_model_records": float(len(world_records)),
        "verifier_stack_temporal_records": float(len(temporal_records)),
        "verifier_stack_memory_records": float(len(memory_validation_records)),
        "verifier_world_model_support": (
            sum(float(record.support) for record in world_records) / max(float(len(world_records)), 1.0)
            if world_records else 0.0
        ),
        "verifier_world_model_conflict": (
            sum(float(record.conflict) for record in world_records) / max(float(len(world_records)), 1.0)
            if world_records else 0.0
        ),
        "verifier_temporal_consistency": (
            sum(float(record.support) for record in temporal_records) / max(float(len(temporal_records)), 1.0)
            if temporal_records else 0.0
        ),
        "verifier_temporal_conflict": (
            sum(float(record.conflict) for record in temporal_records) / max(float(len(temporal_records)), 1.0)
            if temporal_records else 0.0
        ),
        "verifier_memory_corroboration": (
            sum(float(record.support) for record in memory_validation_records) / max(float(len(memory_validation_records)), 1.0)
            if memory_validation_records else 0.0
        ),
        "verifier_memory_conflict": (
            sum(float(record.conflict) for record in memory_validation_records) / max(float(len(memory_validation_records)), 1.0)
            if memory_validation_records else 0.0
        ),
        "verifier_stack_repair_actions": float(len(repairs)),
        "verifier_stack_repair_priority": (
            sum(float(action.priority) for action in repairs) / max(float(len(repairs)), 1.0)
            if repairs else 0.0
        ),
        "verifier_stack_repair_pressure": (
            sum(float(action.pressure) for action in repairs) / max(float(len(repairs)), 1.0)
            if repairs else 0.0
        ),
    }
    return GroundingVerifierStackResult(
        validation_records=tuple(validations),
        repair_actions=tuple(repairs),
        metadata=metadata,
    )

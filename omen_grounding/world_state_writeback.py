from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

from .symbolic_compiler import SymbolicCompilationResult
from .verification import GroundingVerificationReport, GroundingVerificationRecord


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class GroundingWorldStateRecord:
    record_id: str
    hypothesis_id: str
    record_type: str
    world_status: str
    segment_index: int
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    support: float = 0.0
    conflict: float = 0.0
    confidence: float = 0.0
    repair_action: str = "none"
    provenance: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def graph_key(self) -> str:
        return f"grounding_world_state:{self.world_status}:{self.hypothesis_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [*self.symbols, self.record_type, self.world_status]
        if self.repair_action:
            terms.append(self.repair_action)
        return tuple(str(term) for term in terms if term)

    @property
    def graph_family(self) -> str:
        return f"grounding_world_state:{self.world_status}:{self.record_type}"

    @property
    def graph_text(self) -> str:
        symbol_text = " | ".join(self.symbols)
        return (
            f"{self.world_status} {self.record_type} support={self.support:.2f} "
            f"conflict={self.conflict:.2f} repair={self.repair_action} {symbol_text}"
        ).strip()


@dataclass
class GroundingWorldStateWriteback:
    records: Tuple[GroundingWorldStateRecord, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)


def build_grounding_world_state_writeback(
    compilation: SymbolicCompilationResult,
    verification: GroundingVerificationReport,
) -> GroundingWorldStateWriteback:
    verification_by_hypothesis: Mapping[str, GroundingVerificationRecord] = {
        str(record.hypothesis_id): record for record in verification.records
    }
    records: List[GroundingWorldStateRecord] = []

    for hypothesis in compilation.hypotheses:
        verification_record = verification_by_hypothesis.get(str(hypothesis.hypothesis_id))
        if verification_record is None:
            world_status = "hypothetical"
            support = _clip01(hypothesis.confidence)
            conflict = 0.0
            repair_action = "keep_multiple_hypotheses_alive"
            provenance = tuple(str(item) for item in hypothesis.provenance)
        else:
            verification_status = str(verification_record.verification_status)
            if verification_status == "supported":
                world_status = "active"
            elif verification_status == "conflicted":
                world_status = "contradicted"
            else:
                world_status = "hypothetical"
            support = _clip01(verification_record.support)
            conflict = _clip01(verification_record.conflict)
            repair_action = str(verification_record.repair_action or "none")
            provenance = tuple(str(item) for item in verification_record.provenance)
        records.append(
            GroundingWorldStateRecord(
                record_id=f"{world_status}:{hypothesis.hypothesis_id}",
                hypothesis_id=str(hypothesis.hypothesis_id),
                record_type=str(hypothesis.kind),
                world_status=world_status,
                segment_index=int(hypothesis.segment_index),
                symbols=tuple(str(item) for item in hypothesis.symbols),
                support=support,
                conflict=conflict,
                confidence=_clip01(hypothesis.confidence),
                repair_action=repair_action,
                provenance=provenance,
            )
        )

    total = float(len(records))
    active = sum(1 for record in records if record.world_status == "active")
    hypothetical = sum(1 for record in records if record.world_status == "hypothetical")
    contradicted = sum(1 for record in records if record.world_status == "contradicted")
    mean_support = (
        sum(float(record.support) for record in records) / max(total, 1.0)
        if records else 0.0
    )
    mean_conflict = (
        sum(float(record.conflict) for record in records) / max(total, 1.0)
        if records else 0.0
    )
    repairable = sum(
        1
        for record in records
        if str(record.repair_action or "none") not in ("", "none", "keep_multiple_hypotheses_alive")
    )
    hypothetical_ratio = float(hypothetical) / max(total, 1.0)
    contradicted_ratio = float(contradicted) / max(total, 1.0)
    repair_ratio = float(repairable) / max(total, 1.0)
    branching_pressure = _clip01(hypothetical_ratio * (0.50 + 0.50 * mean_support))
    contradiction_pressure = _clip01(
        (0.70 * contradicted_ratio)
        + (0.20 * mean_conflict)
        + (0.10 * repair_ratio)
    )
    metadata = {
        "grounding_world_state_records": total,
        "grounding_world_state_active_records": float(active),
        "grounding_world_state_hypothetical_records": float(hypothetical),
        "grounding_world_state_contradicted_records": float(contradicted),
        "grounding_world_state_acceptance_ratio": float(active) / max(total, 1.0),
        "grounding_world_state_hypothetical_ratio": hypothetical_ratio,
        "grounding_world_state_conflict_ratio": float(contradicted) / max(total, 1.0),
        "grounding_world_state_mean_support": mean_support,
        "grounding_world_state_mean_conflict": mean_conflict,
        "grounding_world_state_repair_ratio": repair_ratio,
        "grounding_world_state_branching_pressure": branching_pressure,
        "grounding_world_state_contradiction_pressure": contradiction_pressure,
    }
    return GroundingWorldStateWriteback(
        records=tuple(records),
        metadata=metadata,
    )

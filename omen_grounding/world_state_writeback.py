from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

from .symbolic_compiler import SymbolicCompilationResult
from .types import GroundingSpan
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
    source_span: GroundingSpan | None = None
    support: float = 0.0
    conflict: float = 0.0
    confidence: float = 0.0
    repair_action: str = "none"
    speaker_key: str = ""
    epistemic_status: str = "asserted"
    claim_source: str = "document"
    semantic_mode: str = "instance"
    quantifier_mode: str = "instance"
    provenance: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def graph_key(self) -> str:
        return f"grounding_world_state:{self.world_status}:{self.hypothesis_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [*self.symbols, self.record_type, self.world_status]
        if self.speaker_key:
            terms.append(f"speaker:{self.speaker_key}")
        if self.epistemic_status:
            terms.append(f"epistemic:{self.epistemic_status}")
        if self.claim_source:
            terms.append(f"claim_source:{self.claim_source}")
        if self.semantic_mode:
            terms.append(f"semantic:{self.semantic_mode}")
        if self.quantifier_mode:
            terms.append(f"quantifier:{self.quantifier_mode}")
        if self.repair_action:
            terms.append(self.repair_action)
        return tuple(str(term) for term in terms if term)

    @property
    def graph_family(self) -> str:
        return f"grounding_world_state:{self.world_status}:{self.record_type}"

    @property
    def graph_text(self) -> str:
        symbol_text = " | ".join(self.symbols)
        attribution = f" speaker={self.speaker_key}" if self.speaker_key else ""
        return (
            f"{self.world_status} {self.record_type} support={self.support:.2f} "
            f"conflict={self.conflict:.2f} repair={self.repair_action} "
            f"epistemic={self.epistemic_status} semantic={self.semantic_mode} "
            f"quantifier={self.quantifier_mode}{attribution} {symbol_text}"
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
        epistemic_status = str(getattr(hypothesis, "epistemic_status", "asserted") or "asserted")
        semantic_mode = str(getattr(hypothesis, "semantic_mode", "instance") or "instance")
        quantifier_mode = str(getattr(hypothesis, "quantifier_mode", "instance") or "instance")
        nonasserted = epistemic_status in {"cited", "questioned", "hedged"}
        rule_lifecycle = semantic_mode in {"generic", "rule", "obligation"} or quantifier_mode in {"generic_all", "directive"}
        if verification_record is None:
            world_status = "hypothetical"
            support = _clip01(hypothesis.confidence)
            conflict = 0.0
            repair_action = (
                "route_to_symbolic_rule_lifecycle"
                if rule_lifecycle
                else "keep_multiple_hypotheses_alive"
            )
            provenance = tuple(str(item) for item in hypothesis.provenance)
        else:
            verification_status = str(verification_record.verification_status)
            if verification_status == "supported":
                world_status = "hypothetical" if (nonasserted or rule_lifecycle) else "active"
            elif verification_status == "conflicted":
                world_status = "contradicted"
            else:
                world_status = "hypothetical"
            support = _clip01(verification_record.support)
            conflict = _clip01(verification_record.conflict)
            repair_action = str(verification_record.repair_action or "none")
            if rule_lifecycle and verification_status == "supported":
                repair_action = "route_to_symbolic_rule_lifecycle"
            provenance = tuple(str(item) for item in verification_record.provenance)
        records.append(
            GroundingWorldStateRecord(
                record_id=f"{world_status}:{hypothesis.hypothesis_id}",
                hypothesis_id=str(hypothesis.hypothesis_id),
                record_type=str(hypothesis.kind),
                world_status=world_status,
                segment_index=int(hypothesis.segment_index),
                symbols=tuple(str(item) for item in hypothesis.symbols),
                source_span=hypothesis.source_span,
                support=support,
                conflict=conflict,
                confidence=_clip01(hypothesis.confidence),
                repair_action=repair_action,
                speaker_key=str(getattr(hypothesis, "speaker_key", "") or ""),
                epistemic_status=epistemic_status,
                claim_source=str(getattr(hypothesis, "claim_source", "document") or "document"),
                semantic_mode=semantic_mode,
                quantifier_mode=quantifier_mode,
                provenance=provenance,
            )
        )

    total = float(len(records))
    active = sum(1 for record in records if record.world_status == "active")
    hypothetical = sum(1 for record in records if record.world_status == "hypothetical")
    contradicted = sum(1 for record in records if record.world_status == "contradicted")
    nonasserted = sum(1 for record in records if record.epistemic_status != "asserted")
    cited = sum(1 for record in records if record.epistemic_status == "cited")
    questioned = sum(1 for record in records if record.epistemic_status == "questioned")
    hedged = sum(1 for record in records if record.epistemic_status == "hedged")
    attributed = sum(1 for record in records if record.speaker_key)
    generic = sum(1 for record in records if record.semantic_mode == "generic")
    rule = sum(1 for record in records if record.semantic_mode == "rule")
    obligation = sum(1 for record in records if record.semantic_mode == "obligation")
    rule_lifecycle_records = sum(
        1
        for record in records
        if (
            record.semantic_mode in {"generic", "rule", "obligation"}
            or record.quantifier_mode in {"generic_all", "directive"}
        )
        and record.world_status == "hypothetical"
    )
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
        if str(record.repair_action or "none") not in (
            "",
            "none",
            "keep_multiple_hypotheses_alive",
            "route_to_symbolic_rule_lifecycle",
        )
    )
    hypothetical_ratio = float(hypothetical) / max(total, 1.0)
    contradicted_ratio = float(contradicted) / max(total, 1.0)
    repair_ratio = float(repairable) / max(total, 1.0)
    nonasserted_ratio = float(nonasserted) / max(total, 1.0)
    branching_pressure = _clip01(
        (hypothetical_ratio * (0.45 + 0.45 * mean_support))
        + (0.30 * nonasserted_ratio)
    )
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
        "grounding_world_state_attributed_records": float(attributed),
        "grounding_world_state_nonasserted_records": float(nonasserted),
        "grounding_world_state_cited_records": float(cited),
        "grounding_world_state_questioned_records": float(questioned),
        "grounding_world_state_hedged_records": float(hedged),
        "grounding_world_state_generic_records": float(generic),
        "grounding_world_state_rule_records": float(rule),
        "grounding_world_state_obligation_records": float(obligation),
        "grounding_world_state_rule_lifecycle_records": float(rule_lifecycle_records),
        "grounding_world_state_rule_lifecycle_ratio": float(rule_lifecycle_records) / max(total, 1.0),
        "grounding_world_state_nonasserted_ratio": nonasserted_ratio,
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

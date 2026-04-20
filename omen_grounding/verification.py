from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .interlingua_types import CanonicalInterlingua
from .scene_types import SemanticSceneGraph
from .symbolic_compiler import CompiledSymbolicHypothesis, SymbolicCompilationResult
from .types import GroundedTextDocument


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class GroundingVerificationRecord:
    hypothesis_id: str
    segment_index: int
    kind: str
    verification_status: str
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    support: float = 0.0
    conflict: float = 0.0
    repair_action: str = "none"
    hidden_cause_candidate: bool = False
    provenance: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def graph_key(self) -> str:
        return f"grounding_verification:{self.verification_status}:{self.hypothesis_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [*self.symbols, self.verification_status]
        if self.repair_action:
            terms.append(self.repair_action)
        return tuple(str(term) for term in terms if term)

    @property
    def graph_family(self) -> str:
        return f"grounding_verification:{self.kind}:{self.verification_status}"

    @property
    def graph_text(self) -> str:
        symbols = " | ".join(self.symbols)
        hidden = " hidden_cause" if self.hidden_cause_candidate else ""
        return (
            f"{self.kind} {self.verification_status} support={self.support:.2f} "
            f"conflict={self.conflict:.2f} repair={self.repair_action}{hidden} {symbols}"
        ).strip()


@dataclass
class GroundingVerificationReport:
    records: Tuple[GroundingVerificationRecord, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)


def _segment_counterexample_map(
    compilation: SymbolicCompilationResult,
) -> Dict[int, bool]:
    return {
        int(segment.index): bool(getattr(segment, "counterexample", False))
        for segment in compilation.segments
    }


def _relation_polarity_groups(
    hypotheses: Sequence[CompiledSymbolicHypothesis],
) -> Dict[Tuple[str, ...], Tuple[str, ...]]:
    polarities: Dict[Tuple[str, ...], set[str]] = defaultdict(set)
    for hypothesis in hypotheses:
        if hypothesis.kind != "relation":
            continue
        polarity = "negative" if str(hypothesis.conflict_tag or "") == "negative_polarity" else "positive"
        polarities[tuple(hypothesis.symbols)].add(polarity)
    return {key: tuple(sorted(values)) for key, values in polarities.items()}


def _support_counts(
    hypotheses: Sequence[CompiledSymbolicHypothesis],
) -> Counter[Tuple[str, Tuple[str, ...]]]:
    return Counter((str(hypothesis.kind), tuple(hypothesis.symbols)) for hypothesis in hypotheses)


def _status_from_scores(support: float, conflict: float) -> str:
    if conflict >= 0.60:
        return "conflicted"
    if support >= 0.55 and conflict <= 0.48:
        return "supported"
    return "deferred"


def _repair_action(status: str, *, hidden_cause_candidate: bool) -> str:
    if hidden_cause_candidate:
        return "trigger_hidden_cause_abduction"
    if status == "conflicted":
        return "preserve_conflict_scope"
    if status == "deferred":
        return "keep_multiple_hypotheses_alive"
    return "accept_to_world_state"


def verify_symbolic_hypotheses(
    compilation: SymbolicCompilationResult,
    *,
    document: Optional[GroundedTextDocument] = None,
    interlingua: Optional[CanonicalInterlingua] = None,
    scene: Optional[SemanticSceneGraph] = None,
) -> GroundingVerificationReport:
    del document, interlingua, scene

    if not compilation.hypotheses:
        return GroundingVerificationReport(
            records=tuple(),
            metadata={
                "verification_records": 0.0,
                "verification_supported_hypotheses": 0.0,
                "verification_deferred_hypotheses": 0.0,
                "verification_conflicted_hypotheses": 0.0,
                "verification_mean_support": 0.0,
                "verification_mean_conflict": 0.0,
                "verification_acceptance_ratio": 0.0,
                "verification_repair_pressure": 0.0,
                "verification_hidden_cause_pressure": 0.0,
                "verification_conflict_pressure": 0.0,
            },
        )

    support_counts = _support_counts(compilation.hypotheses)
    relation_polarities = _relation_polarity_groups(compilation.hypotheses)
    counterexample_by_segment = _segment_counterexample_map(compilation)
    records: List[GroundingVerificationRecord] = []

    for hypothesis in compilation.hypotheses:
        segment_counterexample = 1.0 if counterexample_by_segment.get(int(hypothesis.segment_index), False) else 0.0
        provenance_support = 1.0 if hypothesis.provenance else 0.0
        duplicate_support = 1.0 if support_counts[(str(hypothesis.kind), tuple(hypothesis.symbols))] > 1 else 0.0
        carried_support = 1.0 if str(hypothesis.status or "") in {"supported", "verified"} else 0.0
        kind_bonus = 0.07 if hypothesis.kind in {"relation", "goal"} else 0.04
        low_confidence = 1.0 - _clip01(hypothesis.confidence)
        polarity_conflict = 1.0 if (
            hypothesis.kind == "relation"
            and len(relation_polarities.get(tuple(hypothesis.symbols), ())) > 1
        ) else 0.0
        conflict_hint = 1.0 if (
            str(hypothesis.conflict_tag or "") not in {"", "negative_polarity"}
        ) else 0.0
        support = _clip01(
            (0.55 * _clip01(hypothesis.confidence))
            + (0.10 * provenance_support)
            + (0.12 * duplicate_support)
            + (0.08 * carried_support)
            + (0.08 * (1.0 - segment_counterexample))
            + kind_bonus
        )
        conflict = _clip01(
            (0.30 * float(bool(hypothesis.deferred)))
            + (0.25 * segment_counterexample)
            + (0.20 * polarity_conflict)
            + (0.15 * low_confidence)
            + (0.10 * conflict_hint)
        )
        hidden_cause_candidate = bool(
            hypothesis.kind == "relation"
            and (segment_counterexample > 0.0 or polarity_conflict > 0.0 or conflict_hint > 0.0)
            and support < 0.80
        )
        status = _status_from_scores(support, conflict)
        repair_action = _repair_action(status, hidden_cause_candidate=hidden_cause_candidate)
        records.append(
            GroundingVerificationRecord(
                hypothesis_id=str(hypothesis.hypothesis_id),
                segment_index=int(hypothesis.segment_index),
                kind=str(hypothesis.kind),
                verification_status=status,
                symbols=tuple(str(item) for item in hypothesis.symbols),
                support=support,
                conflict=conflict,
                repair_action=repair_action,
                hidden_cause_candidate=hidden_cause_candidate,
                provenance=tuple(str(item) for item in hypothesis.provenance),
            )
        )

    supported = sum(1 for record in records if record.verification_status == "supported")
    deferred = sum(1 for record in records if record.verification_status == "deferred")
    conflicted = sum(1 for record in records if record.verification_status == "conflicted")
    hidden_cause = sum(1 for record in records if record.hidden_cause_candidate)
    repair_actions = sum(1 for record in records if record.repair_action != "accept_to_world_state")
    total = float(len(records))
    metadata = {
        "verification_records": total,
        "verification_supported_hypotheses": float(supported),
        "verification_deferred_hypotheses": float(deferred),
        "verification_conflicted_hypotheses": float(conflicted),
        "verification_mean_support": (
            sum(float(record.support) for record in records) / total
            if records else 0.0
        ),
        "verification_mean_conflict": (
            sum(float(record.conflict) for record in records) / total
            if records else 0.0
        ),
        "verification_acceptance_ratio": float(supported) / max(total, 1.0),
        "verification_repair_pressure": _clip01(
            (float(repair_actions) / max(total, 1.0))
            + (0.25 * float(hidden_cause) / max(total, 1.0))
        ),
        "verification_hidden_cause_pressure": float(hidden_cause) / max(total, 1.0),
        "verification_conflict_pressure": float(conflicted) / max(total, 1.0),
    }
    return GroundingVerificationReport(
        records=tuple(records),
        metadata=metadata,
    )

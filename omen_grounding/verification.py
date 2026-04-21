from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .interlingua_types import CanonicalInterlingua
from .scene_types import SemanticSceneGraph
from .symbolic_compiler import CompiledSymbolicHypothesis, SymbolicCompilationResult
from .types import GroundedTextDocument, GroundingSpan


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class GroundingVerificationRecord:
    hypothesis_id: str
    segment_index: int
    kind: str
    verification_status: str
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    source_span: Optional[GroundingSpan] = None
    support: float = 0.0
    conflict: float = 0.0
    repair_action: str = "none"
    hidden_cause_candidate: bool = False
    speaker_key: str = ""
    epistemic_status: str = "asserted"
    claim_source: str = "document"
    semantic_mode: str = "instance"
    quantifier_mode: str = "instance"
    provenance: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def graph_key(self) -> str:
        return f"grounding_verification:{self.verification_status}:{self.hypothesis_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [*self.symbols, self.verification_status]
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
        return f"grounding_verification:{self.kind}:{self.verification_status}"

    @property
    def graph_text(self) -> str:
        symbols = " | ".join(self.symbols)
        hidden = " hidden_cause" if self.hidden_cause_candidate else ""
        attribution = f" speaker={self.speaker_key}" if self.speaker_key else ""
        return (
            f"{self.kind} {self.verification_status} support={self.support:.2f} "
            f"conflict={self.conflict:.2f} repair={self.repair_action}"
            f" epistemic={self.epistemic_status} semantic={self.semantic_mode}"
            f" quantifier={self.quantifier_mode}{attribution}{hidden} {symbols}"
        ).strip()


@dataclass(frozen=True)
class GroundingHiddenCauseRecord:
    proposal_id: str
    trigger_hypothesis_id: str
    source_segment: int
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    source_span: Optional[GroundingSpan] = None
    support: float = 0.0
    conflict: float = 0.0
    confidence: float = 0.0
    missing_slot: str = "cause"
    candidate_agent: str = ""
    candidate_event: str = ""
    reason: str = ""
    provenance: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def graph_key(self) -> str:
        return f"grounding_hidden_cause:{self.proposal_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [*self.symbols, "hidden_cause"]
        if self.missing_slot:
            terms.append(f"missing_slot:{self.missing_slot}")
        if self.candidate_agent:
            terms.append(f"candidate_agent:{self.candidate_agent}")
        if self.candidate_event:
            terms.append(f"candidate_event:{self.candidate_event}")
        if self.reason:
            terms.append(f"reason:{self.reason}")
        return tuple(str(term) for term in terms if str(term).strip())

    @property
    def graph_family(self) -> str:
        return "grounding_hidden_cause"

    @property
    def graph_text(self) -> str:
        symbols = " | ".join(self.symbols)
        return (
            f"hidden_cause support={self.support:.2f} conflict={self.conflict:.2f} "
            f"confidence={self.confidence:.2f} reason={self.reason} {symbols}"
        ).strip()


@dataclass
class GroundingVerificationReport:
    records: Tuple[GroundingVerificationRecord, ...] = field(default_factory=tuple)
    hidden_cause_records: Tuple[GroundingHiddenCauseRecord, ...] = field(default_factory=tuple)
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


def _normalize_symbol(value: object) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return text.strip("_")


def _normalized_or(value: object, fallback: str) -> str:
    normalized = _normalize_symbol(value)
    return normalized or fallback


def _claim_id_from_provenance(hypothesis: CompiledSymbolicHypothesis) -> str:
    prefix = f"{str(hypothesis.kind)}:"
    for item in hypothesis.provenance:
        text = str(item or "").strip()
        if text.startswith(prefix):
            return text[len(prefix) :]
    return ""


def _hidden_cause_reason(
    *,
    segment_counterexample: float,
    polarity_conflict: float,
    conflict_hint: float,
) -> str:
    if segment_counterexample > 0.0:
        return "counterexample_gap"
    if polarity_conflict > 0.0:
        return "polarity_split"
    if conflict_hint > 0.0:
        return "conflict_context"
    return "low_support_gap"


def _build_hidden_cause_record(
    hypothesis: CompiledSymbolicHypothesis,
    *,
    support: float,
    conflict: float,
    segment_counterexample: float,
    polarity_conflict: float,
    conflict_hint: float,
    causal_support: float,
) -> GroundingHiddenCauseRecord:
    symbols = tuple(str(item) for item in hypothesis.symbols if str(item).strip())
    subject = symbols[0] if len(symbols) >= 1 else "context"
    predicate = symbols[1] if len(symbols) >= 2 else "state_change"
    object_ = symbols[2] if len(symbols) >= 3 else "unknown_target"
    predicate_key = _normalized_or(predicate, "state_change")
    object_key = _normalized_or(object_, "unknown_target")
    candidate_event = f"missing_{predicate_key}_event"
    if object_key and object_key != "unknown_target":
        candidate_event = f"{candidate_event}_{object_key}"
    candidate_agent = (
        "external_actor"
        if predicate_key in {"opens", "opens_with", "unlocks", "activates", "starts", "triggers"}
        else "unknown_agent"
    )
    reason = _hidden_cause_reason(
        segment_counterexample=segment_counterexample,
        polarity_conflict=polarity_conflict,
        conflict_hint=conflict_hint,
    )
    proposal_support = _clip01(
        0.30
        + (0.25 * segment_counterexample)
        + (0.20 * polarity_conflict)
        + (0.15 * conflict_hint)
        + (0.10 * float(bool(hypothesis.deferred)))
    )
    proposal_conflict = _clip01(
        (0.22 * (1.0 - _clip01(hypothesis.confidence)))
        + (0.12 * max(0.0, 1.0 - causal_support))
        + (0.06 * max(0.0, support - 0.55))
    )
    proposal_confidence = _clip01(
        0.34
        + (0.24 * conflict)
        + (0.16 * (1.0 - support))
        + (0.12 * segment_counterexample)
        + (0.08 * polarity_conflict)
        + (0.06 * conflict_hint)
    )
    proposal_symbols = (
        str(subject),
        "requires_hidden_cause",
        candidate_event,
        "missing_slot:cause",
        f"candidate_agent:{candidate_agent}",
        f"candidate_event:{candidate_event}",
        f"anchor_predicate:{predicate_key}",
        f"anchor_object:{object_key}",
    )
    provenance = tuple(str(item) for item in hypothesis.provenance) + (
        f"trigger_hypothesis:{hypothesis.hypothesis_id}",
        f"hidden_cause_reason:{reason}",
    )
    return GroundingHiddenCauseRecord(
        proposal_id=f"hidden_cause:{hypothesis.hypothesis_id}",
        trigger_hypothesis_id=str(hypothesis.hypothesis_id),
        source_segment=int(hypothesis.segment_index),
        symbols=proposal_symbols,
        source_span=hypothesis.source_span,
        support=proposal_support,
        conflict=proposal_conflict,
        confidence=proposal_confidence,
        missing_slot="cause",
        candidate_agent=candidate_agent,
        candidate_event=candidate_event,
        reason=reason,
        provenance=provenance,
    )


def _structural_evidence_scores(
    hypothesis: CompiledSymbolicHypothesis,
    document: Optional[GroundedTextDocument],
) -> Dict[str, float]:
    empty = {
        "alignment": 0.0,
        "provenance_support": 0.0,
        "dialogue_support": 0.0,
        "citation_support": 0.0,
    }
    if document is None:
        return empty
    segment_index = int(hypothesis.segment_index)
    if segment_index < 0 or segment_index >= len(document.segments):
        return empty
    segment = document.segments[segment_index]
    routing = getattr(segment, "routing", None)
    subtype = str(getattr(routing, "subtype", "") or "")
    structural_units = tuple(getattr(segment, "structural_units", ()) or ())
    structural_types = {str(getattr(unit, "unit_type", "") or "") for unit in structural_units}
    structural_text_parts: List[str] = []
    for unit in structural_units:
        unit_text = _normalize_symbol(getattr(unit, "text", ""))
        if unit_text:
            structural_text_parts.append(unit_text)
        for key, value in tuple(getattr(unit, "fields", ()) or ()):
            for item in (_normalize_symbol(key), _normalize_symbol(value)):
                if item:
                    structural_text_parts.append(item)
    structural_text = " ".join(structural_text_parts)
    normalized_symbols = tuple(
        _normalize_symbol(symbol)
        for symbol in hypothesis.symbols[:3]
        if _normalize_symbol(symbol)
    )
    matched = sum(
        1
        for symbol in normalized_symbols
        if symbol and symbol in structural_text
    )
    alignment = float(matched) / max(float(len(normalized_symbols)), 1.0)
    provenance_support = 1.0 if any(
        str(item).startswith("structural_unit:")
        for item in hypothesis.provenance
    ) else 0.0
    dialogue_support = 1.0 if (
        subtype in {"dialogue_text", "instructional_text"}
        and structural_types.intersection({"speaker_turn", "clause"})
        and hypothesis.kind in {"relation", "goal", "state"}
    ) else 0.0
    citation_support = 1.0 if (
        subtype == "scientific_text"
        and "citation_region" in structural_types
        and hypothesis.kind in {"relation", "state"}
    ) else 0.0
    return {
        "alignment": _clip01(alignment),
        "provenance_support": float(provenance_support),
        "dialogue_support": float(dialogue_support),
        "citation_support": float(citation_support),
    }


def _document_alignment_score(
    hypothesis: CompiledSymbolicHypothesis,
    document: Optional[GroundedTextDocument],
) -> float:
    if document is None:
        return 0.0
    segment_index = int(hypothesis.segment_index)
    if segment_index < 0 or segment_index >= len(document.segments):
        return 0.0
    segment = document.segments[segment_index]
    core_symbols = tuple(str(symbol) for symbol in hypothesis.symbols[:3] if str(symbol).strip())
    normalized_symbols = tuple(_normalize_symbol(symbol) for symbol in core_symbols if _normalize_symbol(symbol))
    if not normalized_symbols:
        return 0.0
    normalized_text = _normalize_symbol(getattr(segment, "normalized_text", "") or getattr(segment, "text", ""))
    token_set = {
        token
        for token in (_normalize_symbol(token) for token in getattr(segment, "tokens", ()))
        if token
    }
    matched = sum(
        1
        for symbol in normalized_symbols
        if symbol in token_set or symbol in normalized_text
    )
    base_score = float(matched) / max(float(len(normalized_symbols)), 1.0)
    routing = getattr(segment, "routing", None)
    modality = str(getattr(routing, "modality", "") or "")
    subtype = str(getattr(routing, "subtype", "") or "")
    structural_units = tuple(getattr(segment, "structural_units", ()) or ())
    structural_types = {str(getattr(unit, "unit_type", "") or "") for unit in structural_units}
    kind = str(hypothesis.kind)
    modality_bonus = 0.0
    if kind == "relation" and modality in {"natural_text", "mixed"}:
        modality_bonus += 0.08
    if kind == "state" and modality == "structured_text":
        modality_bonus += 0.10
    if kind == "goal" and subtype in {"instructional_text", "dialogue_text", "key_value_records"}:
        modality_bonus += 0.08
    if kind == "state" and structural_types.intersection({"key_value_record", "json_record", "table_row", "log_entry"}):
        modality_bonus += 0.06
    if kind == "goal" and structural_types.intersection({"clause", "speaker_turn"}):
        modality_bonus += 0.04
    return _clip01(base_score + modality_bonus)


def _scene_alignment_score(
    hypothesis: CompiledSymbolicHypothesis,
    scene: Optional[SemanticSceneGraph],
) -> float:
    if scene is None:
        return 0.0
    claim_id = _claim_id_from_provenance(hypothesis)
    segment_index = int(hypothesis.segment_index)
    kind = str(hypothesis.kind)
    if kind == "relation":
        for event in scene.events:
            if claim_id and str(getattr(event, "event_id", "")) == claim_id:
                return 1.0
            if int(getattr(event, "source_segment", -1)) != segment_index:
                continue
            signature = (
                _normalize_symbol(getattr(event, "subject_name", "")),
                _normalize_symbol(getattr(event, "event_type", "")),
                _normalize_symbol(getattr(event, "object_name", "")),
            )
            if signature == tuple(_normalize_symbol(symbol) for symbol in hypothesis.symbols[:3]):
                return 0.9
    if kind == "state":
        for state in scene.states:
            if claim_id and str(getattr(state, "state_id", "")) == claim_id:
                return 1.0
            if int(getattr(state, "source_segment", -1)) != segment_index:
                continue
            signature = (
                _normalize_symbol(getattr(state, "key_name", "")),
                _normalize_symbol(getattr(state, "value", "")),
            )
            if signature == tuple(_normalize_symbol(symbol) for symbol in hypothesis.symbols[:2]):
                return 0.9
    if kind == "goal":
        for goal in scene.goals:
            if claim_id and str(getattr(goal, "goal_id", "")) == claim_id:
                return 1.0
            if int(getattr(goal, "source_segment", -1)) != segment_index:
                continue
            signature = (
                _normalize_symbol(getattr(goal, "goal_name", "")),
                _normalize_symbol(getattr(goal, "goal_value", "")),
            )
            if signature == tuple(_normalize_symbol(symbol) for symbol in hypothesis.symbols[:2]):
                return 0.9
    return 0.0


def _interlingua_alignment_score(
    hypothesis: CompiledSymbolicHypothesis,
    interlingua: Optional[CanonicalInterlingua],
) -> float:
    if interlingua is None:
        return 0.0
    claim_id = _claim_id_from_provenance(hypothesis)
    segment_index = int(hypothesis.segment_index)
    kind = str(hypothesis.kind)
    if kind == "relation":
        for relation in interlingua.relations:
            if claim_id and str(getattr(relation, "claim_id", "")) == claim_id:
                return 1.0
            if int(getattr(relation, "source_segment", -1)) != segment_index:
                continue
            signature = (
                _normalize_symbol(getattr(relation, "subject_name", "")),
                _normalize_symbol(getattr(relation, "predicate", "")),
                _normalize_symbol(getattr(relation, "object_name", "")),
            )
            if signature == tuple(_normalize_symbol(symbol) for symbol in hypothesis.symbols[:3]):
                return 0.9
    if kind == "state":
        for state in interlingua.states:
            if claim_id and str(getattr(state, "claim_id", "")) == claim_id:
                return 1.0
            if int(getattr(state, "source_segment", -1)) != segment_index:
                continue
            signature = (
                _normalize_symbol(getattr(state, "entity_name", "")),
                _normalize_symbol(getattr(state, "value", "")),
            )
            if signature == tuple(_normalize_symbol(symbol) for symbol in hypothesis.symbols[:2]):
                return 0.9
    if kind == "goal":
        for goal in interlingua.goals:
            if claim_id and str(getattr(goal, "goal_id", "")) == claim_id:
                return 1.0
            if int(getattr(goal, "source_segment", -1)) != segment_index:
                continue
            signature = (
                _normalize_symbol(getattr(goal, "goal_name", "")),
                _normalize_symbol(getattr(goal, "goal_value", "")),
            )
            if signature == tuple(_normalize_symbol(symbol) for symbol in hypothesis.symbols[:2]):
                return 0.9
    return 0.0


def verify_symbolic_hypotheses(
    compilation: SymbolicCompilationResult,
    *,
    document: Optional[GroundedTextDocument] = None,
    interlingua: Optional[CanonicalInterlingua] = None,
    scene: Optional[SemanticSceneGraph] = None,
) -> GroundingVerificationReport:
    if not compilation.hypotheses:
        return GroundingVerificationReport(
            records=tuple(),
            hidden_cause_records=tuple(),
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
                "verification_document_alignment": 0.0,
                "verification_claim_attribution_support": 0.0,
                "verification_nonasserted_pressure": 0.0,
                "verification_scene_alignment": 0.0,
                "verification_interlingua_alignment": 0.0,
                "verification_hidden_cause_records": 0.0,
                "verification_hidden_cause_mean_confidence": 0.0,
                "verification_hidden_cause_mean_support": 0.0,
            },
        )

    support_counts = _support_counts(compilation.hypotheses)
    relation_polarities = _relation_polarity_groups(compilation.hypotheses)
    counterexample_by_segment = _segment_counterexample_map(compilation)
    records: List[GroundingVerificationRecord] = []
    hidden_cause_records: List[GroundingHiddenCauseRecord] = []
    document_alignments: List[float] = []
    structural_alignments: List[float] = []
    structural_provenances: List[float] = []
    dialogue_supports: List[float] = []
    citation_supports: List[float] = []
    attribution_supports: List[float] = []
    nonasserted_pressures: List[float] = []
    scene_alignments: List[float] = []
    interlingua_alignments: List[float] = []

    for hypothesis in compilation.hypotheses:
        segment_counterexample = 1.0 if counterexample_by_segment.get(int(hypothesis.segment_index), False) else 0.0
        provenance_support = 1.0 if hypothesis.provenance else 0.0
        duplicate_support = 1.0 if support_counts[(str(hypothesis.kind), tuple(hypothesis.symbols))] > 1 else 0.0
        carried_support = 1.0 if str(hypothesis.status or "") in {"supported", "verified"} else 0.0
        kind_bonus = 0.07 if hypothesis.kind in {"relation", "goal"} else 0.04
        low_confidence = 1.0 - _clip01(hypothesis.confidence)
        extra_symbols = tuple(str(symbol) for symbol in hypothesis.symbols[3:] if str(symbol).strip())
        modifier_support = _clip01(float(len(extra_symbols)) / 3.0)
        causal_support = 1.0 if any(symbol.startswith("cause:") for symbol in extra_symbols) else 0.0
        conditional_support = 1.0 if any(symbol.startswith("if:") for symbol in extra_symbols) else 0.0
        temporal_support = 1.0 if any(symbol.startswith("time:") for symbol in extra_symbols) else 0.0
        modal_support = 1.0 if any(symbol.startswith("modal:") for symbol in extra_symbols) else 0.0
        document_alignment = _document_alignment_score(hypothesis, document)
        structural_evidence = _structural_evidence_scores(hypothesis, document)
        structural_alignment = float(structural_evidence["alignment"])
        structural_provenance = float(structural_evidence["provenance_support"])
        dialogue_support = float(structural_evidence["dialogue_support"])
        citation_support = float(structural_evidence["citation_support"])
        speaker_attribution = 1.0 if str(getattr(hypothesis, "speaker_key", "") or "") else 0.0
        epistemic_status = str(getattr(hypothesis, "epistemic_status", "asserted") or "asserted")
        nonasserted_pressure = 1.0 if epistemic_status != "asserted" else 0.0
        if epistemic_status == "asserted":
            epistemic_support = 1.0
            epistemic_conflict = 0.0
        elif epistemic_status == "cited":
            epistemic_support = _clip01(0.55 + (0.45 * citation_support))
            epistemic_conflict = _clip01(0.18 * (1.0 - citation_support))
        elif epistemic_status == "hedged":
            epistemic_support = 0.45
            epistemic_conflict = 0.16
        elif epistemic_status == "questioned":
            epistemic_support = 0.20
            epistemic_conflict = 0.30
        else:
            epistemic_support = 0.50
            epistemic_conflict = 0.12
        scene_alignment = _scene_alignment_score(hypothesis, scene)
        interlingua_alignment = _interlingua_alignment_score(hypothesis, interlingua)
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
            + (0.06 * modifier_support)
            + (0.05 * causal_support)
            + (0.03 * conditional_support)
            + (0.02 * temporal_support)
            + (0.02 * modal_support)
            + (0.08 * document_alignment)
            + (0.07 * structural_alignment)
            + (0.05 * structural_provenance)
            + (0.04 * dialogue_support)
            + (0.03 * citation_support)
            + (0.04 * speaker_attribution)
            + (0.06 * epistemic_support)
            + (0.06 * scene_alignment)
            + (0.06 * interlingua_alignment)
            + kind_bonus
        )
        conflict = _clip01(
            (0.30 * float(bool(hypothesis.deferred)))
            + (0.25 * segment_counterexample)
            + (0.20 * polarity_conflict)
            + (0.15 * low_confidence)
            + (0.10 * conflict_hint)
            + (0.12 * epistemic_conflict)
            - (0.05 * interlingua_alignment)
            - (0.04 * scene_alignment)
            - (0.03 * document_alignment)
            - (0.04 * structural_alignment)
            - (0.03 * structural_provenance)
            - (0.03 * dialogue_support)
            - (0.02 * citation_support)
            - (0.03 * speaker_attribution)
            - (0.04 * causal_support)
            - (0.03 * conditional_support)
            - (0.02 * temporal_support)
        )
        document_alignments.append(float(document_alignment))
        structural_alignments.append(float(structural_alignment))
        structural_provenances.append(float(structural_provenance))
        dialogue_supports.append(float(dialogue_support))
        citation_supports.append(float(citation_support))
        attribution_supports.append(float(speaker_attribution))
        nonasserted_pressures.append(float(nonasserted_pressure))
        scene_alignments.append(float(scene_alignment))
        interlingua_alignments.append(float(interlingua_alignment))
        hidden_cause_candidate = bool(
            hypothesis.kind == "relation"
            and (segment_counterexample > 0.0 or polarity_conflict > 0.0 or conflict_hint > 0.0)
            and causal_support <= 0.0
            and support < 0.80
        )
        status = _status_from_scores(support, conflict)
        if (
            status == "deferred"
            and conflict >= 0.55
            and (segment_counterexample > 0.0 or conflict_hint > 0.0 or polarity_conflict > 0.0)
        ):
            status = "conflicted"
        repair_action = _repair_action(status, hidden_cause_candidate=hidden_cause_candidate)
        if hidden_cause_candidate:
            hidden_cause_records.append(
                _build_hidden_cause_record(
                    hypothesis,
                    support=support,
                    conflict=conflict,
                    segment_counterexample=segment_counterexample,
                    polarity_conflict=polarity_conflict,
                    conflict_hint=conflict_hint,
                    causal_support=causal_support,
                )
            )
        records.append(
            GroundingVerificationRecord(
                hypothesis_id=str(hypothesis.hypothesis_id),
                segment_index=int(hypothesis.segment_index),
                kind=str(hypothesis.kind),
                verification_status=status,
                symbols=tuple(str(item) for item in hypothesis.symbols),
                source_span=hypothesis.source_span,
                support=support,
                conflict=conflict,
                repair_action=repair_action,
                hidden_cause_candidate=hidden_cause_candidate,
                speaker_key=str(getattr(hypothesis, "speaker_key", "") or ""),
                epistemic_status=epistemic_status,
                claim_source=str(getattr(hypothesis, "claim_source", "document") or "document"),
                semantic_mode=str(getattr(hypothesis, "semantic_mode", "instance") or "instance"),
                quantifier_mode=str(getattr(hypothesis, "quantifier_mode", "instance") or "instance"),
                provenance=tuple(str(item) for item in hypothesis.provenance),
            )
        )

    supported = sum(1 for record in records if record.verification_status == "supported")
    deferred = sum(1 for record in records if record.verification_status == "deferred")
    conflicted = sum(1 for record in records if record.verification_status == "conflicted")
    hidden_cause = sum(1 for record in records if record.hidden_cause_candidate)
    repair_actions = sum(1 for record in records if record.repair_action != "accept_to_world_state")
    total = float(len(records))
    hidden_total = float(len(hidden_cause_records))
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
        "verification_document_alignment": (
            sum(document_alignments) / max(total, 1.0)
            if document_alignments else 0.0
        ),
        "verification_claim_attribution_support": (
            sum(attribution_supports) / max(total, 1.0)
            if attribution_supports else 0.0
        ),
        "verification_nonasserted_pressure": (
            sum(nonasserted_pressures) / max(total, 1.0)
            if nonasserted_pressures else 0.0
        ),
        "verification_structural_alignment": (
            sum(structural_alignments) / max(total, 1.0)
            if structural_alignments else 0.0
        ),
        "verification_structural_provenance_support": (
            sum(structural_provenances) / max(total, 1.0)
            if structural_provenances else 0.0
        ),
        "verification_dialogue_structural_support": (
            sum(dialogue_supports) / max(total, 1.0)
            if dialogue_supports else 0.0
        ),
        "verification_citation_support": (
            sum(citation_supports) / max(total, 1.0)
            if citation_supports else 0.0
        ),
        "verification_scene_alignment": (
            sum(scene_alignments) / max(total, 1.0)
            if scene_alignments else 0.0
        ),
        "verification_interlingua_alignment": (
            sum(interlingua_alignments) / max(total, 1.0)
            if interlingua_alignments else 0.0
        ),
        "verification_hidden_cause_records": hidden_total,
        "verification_hidden_cause_mean_confidence": (
            sum(float(record.confidence) for record in hidden_cause_records) / max(hidden_total, 1.0)
            if hidden_cause_records else 0.0
        ),
        "verification_hidden_cause_mean_support": (
            sum(float(record.support) for record in hidden_cause_records) / max(hidden_total, 1.0)
            if hidden_cause_records else 0.0
        ),
    }
    return GroundingVerificationReport(
        records=tuple(records),
        hidden_cause_records=tuple(hidden_cause_records),
        metadata=metadata,
    )

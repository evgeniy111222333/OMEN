from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .interlingua import build_canonical_interlingua
from .interlingua_types import CanonicalClaimFrame, CanonicalInterlingua
from .scene_types import SemanticSceneGraph
from .types import GroundedTextDocument, GroundingSpan


@dataclass(frozen=True)
class CompiledSymbolicSegment:
    index: int
    text: str
    normalized_text: str
    source_span: Optional[GroundingSpan] = None
    tokens: Tuple[str, ...] = field(default_factory=tuple)
    states: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    relations: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)
    goals: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    event_frames: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    claim_frames: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)
    counterexample: bool = False


@dataclass(frozen=True)
class CompiledSymbolicHypothesis:
    hypothesis_id: str
    segment_index: int
    kind: str
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.5
    status: str = "proposal"
    deferred: bool = False
    conflict_tag: str = ""
    provenance: Tuple[str, ...] = field(default_factory=tuple)
    support_set: Tuple[str, ...] = field(default_factory=tuple)
    speaker_key: str = ""
    epistemic_status: str = "asserted"
    claim_source: str = "document"

    @property
    def graph_key(self) -> str:
        return f"grounding_hypothesis:{self.hypothesis_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [*self.symbols, *self.support_set]
        if self.speaker_key:
            terms.append(f"speaker:{self.speaker_key}")
        if self.epistemic_status:
            terms.append(f"epistemic:{self.epistemic_status}")
        if self.claim_source:
            terms.append(f"claim_source:{self.claim_source}")
        return tuple(str(term) for term in terms if str(term).strip())

    @property
    def graph_family(self) -> str:
        return f"grounding_hypothesis:{self.kind}"

    @property
    def graph_text(self) -> str:
        parts = [self.kind, *self.symbols]
        if self.speaker_key:
            parts.append(f"speaker={self.speaker_key}")
        if self.epistemic_status:
            parts.append(f"epistemic={self.epistemic_status}")
        if self.conflict_tag:
            parts.append(self.conflict_tag)
        return " | ".join(parts)


@dataclass
class SymbolicCompilationResult:
    language: str
    source_text: str
    segments: Tuple[CompiledSymbolicSegment, ...] = field(default_factory=tuple)
    hypotheses: Tuple[CompiledSymbolicHypothesis, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)


def compile_semantic_scene_graph(
    scene: SemanticSceneGraph,
    *,
    document: Optional[GroundedTextDocument] = None,
) -> SymbolicCompilationResult:
    interlingua = build_canonical_interlingua(scene)
    return compile_canonical_interlingua(interlingua, document=document)


def compile_canonical_interlingua(
    interlingua: CanonicalInterlingua,
    *,
    document: Optional[GroundedTextDocument] = None,
) -> SymbolicCompilationResult:
    segment_count = max(
        [len(getattr(document, "segments", ()))]
        + [int(state.source_segment) + 1 for state in interlingua.states]
        + [int(relation.source_segment) + 1 for relation in interlingua.relations]
        + [int(goal.source_segment) + 1 for goal in interlingua.goals]
    )
    compiled: List[CompiledSymbolicSegment] = []
    hypotheses: List[CompiledSymbolicHypothesis] = []
    claim_frames_by_proposition: Dict[str, Tuple[CanonicalClaimFrame, ...]] = {}
    for claim in tuple(getattr(interlingua, "claims", ()) or ()):
        proposition_id = str(getattr(claim, "proposition_id", "") or "")
        if not proposition_id:
            continue
        claim_frames_by_proposition[proposition_id] = claim_frames_by_proposition.get(proposition_id, tuple()) + (claim,)

    def _clip_confidence(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _dominant_epistemic_status(claim_frames: Sequence[CanonicalClaimFrame]) -> str:
        if not claim_frames:
            return "asserted"
        precedence = {"questioned": 0, "hedged": 1, "cited": 2, "asserted": 3}
        ranked = sorted(
            (str(getattr(frame, "epistemic_status", "asserted") or "asserted") for frame in claim_frames),
            key=lambda value: precedence.get(value, 4),
        )
        return ranked[0] if ranked else "asserted"

    def _append_hypothesis(
        *,
        hypothesis_id: str,
        segment_index: int,
        kind: str,
        symbols: Sequence[str],
        source_span: Optional[GroundingSpan],
        confidence: float,
        status: str,
        conflict_tag: str = "",
        counterexample: bool = False,
        provenance: Sequence[str] = (),
        evidence_refs: Sequence[str] = (),
        claim_frames: Sequence[CanonicalClaimFrame] = (),
    ) -> None:
        confidence_value = _clip_confidence(confidence)
        conflict = conflict_tag or ("counterexample_context" if counterexample else "")
        dominant_epistemic_status = _dominant_epistemic_status(claim_frames)
        deferred = bool(
            counterexample
            or confidence_value < 0.57
            or dominant_epistemic_status in {"cited", "questioned", "hedged"}
        )
        evidence = tuple(str(item) for item in evidence_refs if str(item).strip())
        claim_support: List[str] = []
        for frame in claim_frames:
            claim_support.append(f"claim_frame:{frame.claim_id}")
            if getattr(frame, "speaker_key", None):
                claim_support.append(f"speaker:{frame.speaker_key}")
            if getattr(frame, "epistemic_status", None):
                claim_support.append(f"epistemic:{frame.epistemic_status}")
            if getattr(frame, "claim_source", None):
                claim_support.append(f"claim_source:{frame.claim_source}")
            claim_support.extend(
                str(item) for item in getattr(frame, "evidence_refs", ()) if str(item).strip()
            )
        support_set = tuple(dict.fromkeys(claim_support))
        speaker_key = next(
            (str(frame.speaker_key) for frame in claim_frames if getattr(frame, "speaker_key", None)),
            "",
        )
        claim_source = next(
            (str(frame.claim_source) for frame in claim_frames if str(getattr(frame, "claim_source", "") or "")),
            "document",
        )
        hypotheses.append(
            CompiledSymbolicHypothesis(
                hypothesis_id=hypothesis_id,
                segment_index=segment_index,
                kind=kind,
                symbols=tuple(str(item) for item in symbols),
                source_span=source_span,
                confidence=confidence_value,
                status=str(status or "proposal"),
                deferred=deferred,
                conflict_tag=conflict,
                provenance=tuple(str(item) for item in provenance) + evidence,
                support_set=support_set,
                speaker_key=speaker_key,
                epistemic_status=dominant_epistemic_status,
                claim_source=claim_source,
            )
        )

    for idx in range(segment_count):
        raw_segment = document.segments[idx] if document is not None and idx < len(document.segments) else None
        counterexample = bool(False if raw_segment is None else raw_segment.counterexample)
        segment_states = tuple(
            state
            for state in interlingua.states
            if int(state.source_segment) == idx
        )
        segment_relations = tuple(
            relation
            for relation in interlingua.relations
            if int(relation.source_segment) == idx
        )
        segment_goals = tuple(
            goal
            for goal in interlingua.goals
            if int(goal.source_segment) == idx
        )
        states = tuple(
            (state.entity_name, state.value)
            for state in segment_states
        )
        relations = tuple(
            (relation.subject_name, relation.predicate, relation.object_name)
            for relation in segment_relations
        )
        event_frames = tuple(
            tuple(
                item
                for item in (
                    relation.subject_name,
                    relation.predicate,
                    relation.object_name,
                    *tuple(relation.relation_modifiers),
                )
                if item
            )
            for relation in segment_relations
        )
        goals = tuple(
            (goal.goal_name, goal.goal_value)
            for goal in segment_goals
        )
        segment_claim_frames = tuple(
            (
                str(claim.claim_kind),
                str(claim.epistemic_status),
                str(claim.speaker_name or claim.claim_source or ""),
            )
            for claim in tuple(getattr(interlingua, "claims", ()) or ())
            if int(getattr(claim, "source_segment", -1)) == idx
        )
        for state in segment_states:
            attached_claim_frames = claim_frames_by_proposition.get(str(state.claim_id), tuple())
            _append_hypothesis(
                hypothesis_id=state.claim_id,
                segment_index=idx,
                kind="state",
                symbols=(state.entity_name, state.value),
                source_span=state.source_span,
                confidence=state.confidence,
                status=state.status,
                counterexample=counterexample,
                provenance=(f"segment:{idx}", f"state:{state.claim_id}"),
                evidence_refs=getattr(state, "evidence_refs", ()),
                claim_frames=attached_claim_frames,
            )
        for relation in segment_relations:
            relation_symbols = [relation.subject_name, relation.predicate, relation.object_name]
            relation_symbols.extend(str(item) for item in relation.relation_modifiers if str(item).strip())
            attached_claim_frames = claim_frames_by_proposition.get(str(relation.claim_id), tuple())
            _append_hypothesis(
                hypothesis_id=relation.claim_id,
                segment_index=idx,
                kind="relation",
                symbols=tuple(relation_symbols),
                source_span=relation.source_span,
                confidence=relation.confidence,
                status=relation.status,
                conflict_tag="negative_polarity" if relation.polarity != "positive" else "",
                counterexample=counterexample,
                provenance=(f"segment:{idx}", f"relation:{relation.claim_id}"),
                evidence_refs=getattr(relation, "evidence_refs", ()),
                claim_frames=attached_claim_frames,
            )
        for goal in segment_goals:
            goal_symbols = [goal.goal_name, goal.goal_value]
            if goal.target_name:
                goal_symbols.append(goal.target_name)
            attached_claim_frames = claim_frames_by_proposition.get(str(goal.goal_id), tuple())
            _append_hypothesis(
                hypothesis_id=goal.goal_id,
                segment_index=idx,
                kind="goal",
                symbols=tuple(goal_symbols),
                source_span=goal.source_span,
                confidence=goal.confidence,
                status=goal.status,
                counterexample=counterexample,
                provenance=(f"segment:{idx}", f"goal:{goal.goal_id}"),
                evidence_refs=getattr(goal, "evidence_refs", ()),
                claim_frames=attached_claim_frames,
            )
        compiled.append(
            CompiledSymbolicSegment(
                index=idx,
                text="" if raw_segment is None else raw_segment.text,
                normalized_text="" if raw_segment is None else raw_segment.normalized_text,
                source_span=None if raw_segment is None else raw_segment.span,
                tokens=tuple() if raw_segment is None else tuple(raw_segment.tokens),
                states=states,
                relations=relations,
                goals=goals,
                event_frames=event_frames,
                claim_frames=segment_claim_frames,
                counterexample=counterexample,
            )
        )
    metadata = dict(interlingua.metadata)
    deferred_hypotheses = sum(1 for hypothesis in hypotheses if hypothesis.deferred)
    conflict_hypotheses = sum(1 for hypothesis in hypotheses if hypothesis.conflict_tag)
    mean_confidence = (
        sum(float(hypothesis.confidence) for hypothesis in hypotheses) / float(len(hypotheses))
        if hypotheses else 0.0
    )
    metadata.update(
        {
            "compiled_segments": float(len(compiled)),
            "compiled_state_claims": float(sum(len(segment.states) for segment in compiled)),
            "compiled_relation_claims": float(sum(len(segment.relations) for segment in compiled)),
            "compiled_goal_claims": float(sum(len(segment.goals) for segment in compiled)),
            "compiled_event_frames": float(sum(len(segment.event_frames) for segment in compiled)),
            "compiled_modal_relations": float(
                sum(1 for hypothesis in hypotheses if hypothesis.kind == "relation" and any(symbol.startswith("modal:") for symbol in hypothesis.symbols[3:]))
            ),
            "compiled_conditioned_relations": float(
                sum(1 for hypothesis in hypotheses if hypothesis.kind == "relation" and any(symbol.startswith("if:") for symbol in hypothesis.symbols[3:]))
            ),
            "compiled_explained_relations": float(
                sum(1 for hypothesis in hypotheses if hypothesis.kind == "relation" and any(symbol.startswith("cause:") for symbol in hypothesis.symbols[3:]))
            ),
            "compiled_temporal_relations": float(
                sum(1 for hypothesis in hypotheses if hypothesis.kind == "relation" and any(symbol.startswith("time:") for symbol in hypothesis.symbols[3:]))
            ),
            "compiled_hypotheses": float(len(hypotheses)),
            "compiled_claim_frames": float(sum(len(segment.claim_frames) for segment in compiled)),
            "compiled_deferred_hypotheses": float(deferred_hypotheses),
            "compiled_conflict_hypotheses": float(conflict_hypotheses),
            "compiled_mean_confidence": float(mean_confidence),
            "compiled_attributed_hypotheses": float(sum(1 for hypothesis in hypotheses if hypothesis.speaker_key)),
            "compiled_nonasserted_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.epistemic_status != "asserted")
            ),
            "compiled_cited_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.epistemic_status == "cited")
            ),
            "compiled_questioned_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.epistemic_status == "questioned")
            ),
            "compiled_hedged_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.epistemic_status == "hedged")
            ),
            "compiled_structural_evidence_refs": float(
                sum(
                    1
                    for hypothesis in hypotheses
                    for item in hypothesis.provenance
                    if str(item).startswith("structural_unit:")
                )
            ),
        }
    )
    return SymbolicCompilationResult(
        language=interlingua.language,
        source_text=interlingua.source_text,
        segments=tuple(compiled),
        hypotheses=tuple(hypotheses),
        metadata=metadata,
    )

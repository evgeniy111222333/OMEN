from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .interlingua import build_canonical_interlingua
from .interlingua_types import CanonicalInterlingua
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

    @property
    def graph_key(self) -> str:
        return f"grounding_hypothesis:{self.hypothesis_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        return tuple(self.symbols)

    @property
    def graph_family(self) -> str:
        return f"grounding_hypothesis:{self.kind}"

    @property
    def graph_text(self) -> str:
        parts = [self.kind, *self.symbols]
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

    def _clip_confidence(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

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
    ) -> None:
        confidence_value = _clip_confidence(confidence)
        conflict = conflict_tag or ("counterexample_context" if counterexample else "")
        deferred = bool(counterexample or confidence_value < 0.57)
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
                provenance=tuple(str(item) for item in provenance),
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
        goals = tuple(
            (goal.goal_name, goal.goal_value)
            for goal in segment_goals
        )
        for state in segment_states:
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
            )
        for relation in segment_relations:
            _append_hypothesis(
                hypothesis_id=relation.claim_id,
                segment_index=idx,
                kind="relation",
                symbols=(relation.subject_name, relation.predicate, relation.object_name),
                source_span=relation.source_span,
                confidence=relation.confidence,
                status=relation.status,
                conflict_tag="negative_polarity" if relation.polarity != "positive" else "",
                counterexample=counterexample,
                provenance=(f"segment:{idx}", f"relation:{relation.claim_id}"),
            )
        for goal in segment_goals:
            goal_symbols = [goal.goal_name, goal.goal_value]
            if goal.target_name:
                goal_symbols.append(goal.target_name)
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
            "compiled_hypotheses": float(len(hypotheses)),
            "compiled_deferred_hypotheses": float(deferred_hypotheses),
            "compiled_conflict_hypotheses": float(conflict_hypotheses),
            "compiled_mean_confidence": float(mean_confidence),
        }
    )
    return SymbolicCompilationResult(
        language=interlingua.language,
        source_text=interlingua.source_text,
        segments=tuple(compiled),
        hypotheses=tuple(hypotheses),
        metadata=metadata,
    )

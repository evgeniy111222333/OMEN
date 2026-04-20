from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

from .backbone import SemanticGroundingBackbone
from .semantic_context import build_semantic_context_objects
from .scene_types import (
    SemanticClaim,
    SemanticDiscourseRelation,
    SemanticEntity,
    SemanticEvent,
    SemanticExplanation,
    SemanticGoal,
    SemanticMention,
    SemanticSceneGraph,
    SemanticState,
    SemanticTemporalMarker,
)
from .types import GroundedTextDocument


class _EntityAccumulator:
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name
        self.semantic_type = "entity"
        self.aliases: Set[str] = {canonical_name}
        self.source_segments: Set[int] = set()
        self.source_spans = []
        self.confidences: List[float] = []
        self.status = "candidate"

    def add(self, *, alias: Optional[str], source_segment: int, source_span, confidence: float, status: str) -> None:
        if alias:
            self.aliases.add(alias)
        self.source_segments.add(int(source_segment))
        if source_span is not None:
            self.source_spans.append(source_span)
        self.confidences.append(float(confidence))
        if float(confidence) >= 0.7:
            self.status = "supported"
        elif status == "hint" and self.status != "supported":
            self.status = "hint"

    def finalize(self, entity_id: str) -> SemanticEntity:
        mean_conf = sum(self.confidences) / max(len(self.confidences), 1)
        return SemanticEntity(
            entity_id=entity_id,
            canonical_name=self.canonical_name,
            semantic_type=self.semantic_type,
            aliases=tuple(sorted(self.aliases)),
            source_segments=tuple(sorted(self.source_segments)),
            source_spans=tuple(self.source_spans),
            confidence=float(mean_conf),
            status=self.status,
        )


def build_semantic_scene_graph(
    document: GroundedTextDocument,
    *,
    backbone: Optional[SemanticGroundingBackbone] = None,
) -> SemanticSceneGraph:
    if backbone is not None:
        try:
            proposed = backbone.build_scene_graph(document)
        except Exception:
            proposed = None
        if proposed is not None:
            return proposed

    entity_acc: Dict[str, _EntityAccumulator] = {}
    entity_ids: Dict[str, str] = {}
    states: List[SemanticState] = []
    events: List[SemanticEvent] = []
    goals: List[SemanticGoal] = []
    claims: List[SemanticClaim] = []

    def ensure_entity(name: str, *, source_segment: int, source_span, confidence: float, status: str) -> str:
        if name not in entity_ids:
            entity_ids[name] = f"ent:{len(entity_ids)}:{name}"
            entity_acc[name] = _EntityAccumulator(name)
        entity_acc[name].add(
            alias=name,
            source_segment=source_segment,
            source_span=source_span,
            confidence=confidence,
            status=status,
        )
        return entity_ids[name]

    for segment in document.segments:
        seg_idx = int(segment.index)
        for idx, state in enumerate(segment.states):
            entity_id = ensure_entity(
                state.key,
                source_segment=seg_idx,
                source_span=state.span or segment.span,
                confidence=state.confidence,
                status=state.status,
            )
            states.append(
                SemanticState(
                    state_id=f"state:{seg_idx}:{idx}",
                    key_entity_id=entity_id,
                    key_name=state.key,
                    value=state.value,
                    source_segment=seg_idx,
                    source_span=state.span or segment.span,
                    confidence=state.confidence,
                    status=state.status,
                )
            )
            claims.append(
                SemanticClaim(
                    claim_id=f"claim:state:{seg_idx}:{idx}",
                    claim_kind="state",
                    source_segment=seg_idx,
                    source_span=state.span or segment.span,
                    confidence=state.confidence,
                    status="proposal",
                    subject_entity_id=entity_id,
                    predicate="state",
                    object_value=state.value,
                )
            )
        for idx, relation in enumerate(segment.relations):
            subj_id = ensure_entity(
                relation.left,
                source_segment=seg_idx,
                source_span=relation.span or segment.span,
                confidence=relation.confidence,
                status=relation.status,
            )
            obj_id = ensure_entity(
                relation.right,
                source_segment=seg_idx,
                source_span=relation.span or segment.span,
                confidence=relation.confidence,
                status=relation.status,
            )
            event_id = f"event:{seg_idx}:{idx}"
            event = SemanticEvent(
                event_id=event_id,
                event_type=relation.relation,
                subject_entity_id=subj_id,
                object_entity_id=obj_id,
                subject_name=relation.left,
                object_name=relation.right,
                source_segment=seg_idx,
                source_span=relation.span or segment.span,
                confidence=relation.confidence,
                polarity="negative" if segment.counterexample else "positive",
                status=relation.status,
                metadata={"counterexample_segment": 1.0 if segment.counterexample else 0.0},
            )
            events.append(event)
            claims.append(
                SemanticClaim(
                    claim_id=f"claim:relation:{seg_idx}:{idx}",
                    claim_kind="relation",
                    source_segment=seg_idx,
                    source_span=relation.span or segment.span,
                    confidence=relation.confidence,
                    status="proposal",
                    subject_entity_id=subj_id,
                    predicate=relation.relation,
                    object_entity_id=obj_id,
                    event_id=event_id,
                )
            )
        for idx, goal in enumerate(segment.goals):
            target_id = ensure_entity(
                goal.goal_value,
                source_segment=seg_idx,
                source_span=goal.span or segment.span,
                confidence=goal.confidence,
                status=goal.status,
            )
            goal_id = f"goal:{seg_idx}:{idx}"
            goals.append(
                SemanticGoal(
                    goal_id=goal_id,
                    goal_name=goal.goal_name,
                    goal_value=goal.goal_value,
                    target_entity_id=target_id,
                    source_segment=seg_idx,
                    source_span=goal.span or segment.span,
                    confidence=goal.confidence,
                    status=goal.status,
                )
            )
            claims.append(
                SemanticClaim(
                    claim_id=f"claim:goal:{seg_idx}:{idx}",
                    claim_kind="goal",
                    source_segment=seg_idx,
                    source_span=goal.span or segment.span,
                    confidence=goal.confidence,
                    status="proposal",
                    predicate=goal.goal_name,
                    object_entity_id=target_id,
                    object_value=goal.goal_value,
                    goal_id=goal_id,
                )
            )

    entities = tuple(
        entity_acc[name].finalize(entity_ids[name])
        for name in sorted(entity_ids.keys())
    )
    mentions, discourse_relations, temporal_markers, explanations = build_semantic_context_objects(
        document,
        entities,
    )
    metadata = dict(document.metadata)
    metadata.update(
        {
            "scene_entities": float(len(entities)),
            "scene_states": float(len(states)),
            "scene_events": float(len(events)),
            "scene_goals": float(len(goals)),
            "scene_claims": float(len(claims)),
            "scene_mentions": float(len(mentions)),
            "scene_discourse_relations": float(len(discourse_relations)),
            "scene_temporal_markers": float(len(temporal_markers)),
            "scene_explanations": float(len(explanations)),
            "scene_negative_events": float(sum(1 for event in events if event.polarity == "negative")),
            "scene_mean_entity_confidence": float(
                sum(entity.confidence for entity in entities) / max(len(entities), 1)
            ),
            "scene_mean_event_confidence": float(
                sum(event.confidence for event in events) / max(len(events), 1)
            ),
        }
    )
    return SemanticSceneGraph(
        language=document.language,
        source_text=document.source_text,
        entities=entities,
        states=tuple(states),
        events=tuple(events),
        goals=tuple(goals),
        claims=tuple(claims),
        mentions=mentions,
        discourse_relations=discourse_relations,
        temporal_markers=temporal_markers,
        explanations=explanations,
        metadata=metadata,
    )

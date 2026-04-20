from __future__ import annotations

from typing import Dict, List, Optional

from .interlingua_types import (
    CanonicalEntity,
    CanonicalGoalClaim,
    CanonicalInterlingua,
    CanonicalRelationClaim,
    CanonicalStateClaim,
)
from .scene_types import SemanticSceneGraph
from .text_semantics import normalize_symbol_text


def _canonical_key(value: Optional[str], fallback: str) -> str:
    normalized = normalize_symbol_text(value)
    if normalized:
        return normalized
    return normalize_symbol_text(fallback) or fallback


def build_canonical_interlingua(scene: SemanticSceneGraph) -> CanonicalInterlingua:
    entity_index: Dict[str, CanonicalEntity] = {}
    for entity in scene.entities:
        canonical_name = entity.canonical_name or entity.entity_id
        canonical_key = _canonical_key(canonical_name, entity.entity_id)
        entity_index[entity.entity_id] = CanonicalEntity(
            entity_id=entity.entity_id,
            canonical_name=canonical_name,
            canonical_key=canonical_key,
            semantic_type=entity.semantic_type,
            aliases=tuple(sorted({alias for alias in entity.aliases if alias})),
            source_segments=tuple(sorted(int(seg) for seg in entity.source_segments)),
            confidence=float(entity.confidence),
            status=entity.status,
        )

    states: List[CanonicalStateClaim] = []
    for state in scene.states:
        entity = entity_index.get(state.key_entity_id)
        if entity is None:
            entity = CanonicalEntity(
                entity_id=state.key_entity_id,
                canonical_name=state.key_name,
                canonical_key=_canonical_key(state.key_name, state.key_entity_id),
                confidence=float(state.confidence),
                status=state.status,
            )
            entity_index[entity.entity_id] = entity
        states.append(
            CanonicalStateClaim(
                claim_id=state.state_id,
                entity_id=entity.entity_id,
                entity_key=entity.canonical_key,
                entity_name=entity.canonical_name,
                value=state.value,
                value_key=_canonical_key(state.value, state.state_id),
                source_segment=int(state.source_segment),
                confidence=float(state.confidence),
                status=state.status,
            )
        )

    relations: List[CanonicalRelationClaim] = []
    for event in scene.events:
        if event.subject_entity_id is None or event.object_entity_id is None:
            continue
        subject = entity_index.get(event.subject_entity_id)
        object_ = entity_index.get(event.object_entity_id)
        if subject is None or object_ is None:
            continue
        relations.append(
            CanonicalRelationClaim(
                claim_id=event.event_id,
                subject_entity_id=subject.entity_id,
                subject_key=subject.canonical_key,
                subject_name=subject.canonical_name,
                predicate=event.event_type,
                predicate_key=_canonical_key(event.event_type, event.event_id),
                object_entity_id=object_.entity_id,
                object_key=object_.canonical_key,
                object_name=object_.canonical_name,
                source_segment=int(event.source_segment),
                confidence=float(event.confidence),
                polarity=event.polarity,
                status=event.status,
            )
        )

    goals: List[CanonicalGoalClaim] = []
    for goal in scene.goals:
        target = entity_index.get(goal.target_entity_id) if goal.target_entity_id is not None else None
        goals.append(
            CanonicalGoalClaim(
                goal_id=goal.goal_id,
                goal_name=goal.goal_name,
                goal_key=_canonical_key(goal.goal_name, goal.goal_id),
                goal_value=goal.goal_value,
                value_key=_canonical_key(goal.goal_value, goal.goal_id),
                target_entity_id=target.entity_id if target is not None else goal.target_entity_id,
                target_key=target.canonical_key if target is not None else None,
                target_name=target.canonical_name if target is not None else goal.goal_value,
                source_segment=int(goal.source_segment),
                confidence=float(goal.confidence),
                status=goal.status,
            )
        )

    entities = tuple(sorted(entity_index.values(), key=lambda item: item.entity_id))
    metadata = dict(scene.metadata)
    metadata.update(
        {
            "interlingua_entities": float(len(entities)),
            "interlingua_states": float(len(states)),
            "interlingua_relations": float(len(relations)),
            "interlingua_goals": float(len(goals)),
            "interlingua_negative_relations": float(
                sum(1 for relation in relations if relation.polarity == "negative")
            ),
            "interlingua_uncertain_claims": float(
                sum(1 for relation in relations if relation.confidence < 0.6)
                + sum(1 for state in states if state.confidence < 0.6)
                + sum(1 for goal in goals if goal.confidence < 0.6)
            ),
            "interlingua_mean_entity_confidence": float(
                sum(entity.confidence for entity in entities) / max(len(entities), 1)
            ),
            "interlingua_mean_relation_confidence": float(
                sum(relation.confidence for relation in relations) / max(len(relations), 1)
            ),
        }
    )
    return CanonicalInterlingua(
        language=scene.language,
        source_text=scene.source_text,
        entities=entities,
        states=tuple(states),
        relations=tuple(relations),
        goals=tuple(goals),
        metadata=metadata,
    )

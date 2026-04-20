from __future__ import annotations

from typing import Dict, List, Optional

from .interlingua_types import (
    CanonicalClaimFrame,
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


def _optional_key(value: Optional[str], fallback: str) -> Optional[str]:
    normalized = normalize_symbol_text(value)
    if normalized:
        return normalized
    fallback_norm = normalize_symbol_text(fallback)
    return fallback_norm if fallback_norm else None


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
            source_spans=tuple(entity.source_spans),
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
                source_span=state.source_span,
                confidence=float(state.confidence),
                status=state.status,
                evidence_refs=tuple(str(item) for item in getattr(state, "evidence_refs", ()) if str(item).strip()),
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
        relation_modifiers = tuple(
            modifier
            for modifier in (
                f"modal:{_canonical_key(event.modality, event.event_id)}" if event.modality else "",
                f"if:{_canonical_key(event.condition, event.event_id)}" if event.condition else "",
                f"cause:{_canonical_key(event.explanation, event.event_id)}" if event.explanation else "",
                f"time:{_canonical_key(event.temporal, event.event_id)}" if event.temporal else "",
            )
            if modifier
        )
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
                source_span=event.source_span,
                modality=_canonical_key(event.modality, event.event_id) if event.modality else "",
                condition_key=_optional_key(event.condition, event.event_id),
                explanation_key=_optional_key(event.explanation, event.event_id),
                temporal_key=_optional_key(event.temporal, event.event_id),
                relation_modifiers=relation_modifiers,
                confidence=float(event.confidence),
                polarity=event.polarity,
                status=event.status,
                evidence_refs=tuple(str(item) for item in getattr(event, "evidence_refs", ()) if str(item).strip()),
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
                source_span=goal.source_span,
                confidence=float(goal.confidence),
                status=goal.status,
                evidence_refs=tuple(str(item) for item in getattr(goal, "evidence_refs", ()) if str(item).strip()),
            )
        )

    claims: List[CanonicalClaimFrame] = []
    for claim in scene.claims:
        proposition_id = str(
            getattr(claim, "proposition_id", None)
            or getattr(claim, "event_id", None)
            or getattr(claim, "goal_id", None)
            or getattr(claim, "claim_id", "")
        )
        if not proposition_id:
            proposition_id = str(getattr(claim, "claim_id", ""))
        speaker = (
            entity_index.get(claim.speaker_entity_id)
            if getattr(claim, "speaker_entity_id", None) is not None
            else None
        )
        claims.append(
            CanonicalClaimFrame(
                claim_id=str(claim.claim_id),
                claim_kind=str(claim.claim_kind),
                proposition_id=proposition_id,
                speaker_entity_id=speaker.entity_id if speaker is not None else getattr(claim, "speaker_entity_id", None),
                speaker_key=speaker.canonical_key if speaker is not None else _optional_key(getattr(claim, "speaker_name", None), str(claim.claim_id)),
                speaker_name=speaker.canonical_name if speaker is not None else getattr(claim, "speaker_name", None),
                epistemic_status=str(getattr(claim, "epistemic_status", "asserted") or "asserted"),
                claim_source=str(getattr(claim, "claim_source", "document") or "document"),
                source_segment=int(claim.source_segment),
                source_span=claim.source_span,
                confidence=float(claim.confidence),
                status=str(claim.status),
                evidence_refs=tuple(str(item) for item in getattr(claim, "evidence_refs", ()) if str(item).strip()),
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
            "interlingua_claim_frames": float(len(claims)),
            "interlingua_attributed_claim_frames": float(sum(1 for claim in claims if claim.speaker_key)),
            "interlingua_nonasserted_claim_frames": float(
                sum(1 for claim in claims if str(claim.epistemic_status) != "asserted")
            ),
            "interlingua_cited_claim_frames": float(sum(1 for claim in claims if claim.epistemic_status == "cited")),
            "interlingua_questioned_claim_frames": float(
                sum(1 for claim in claims if claim.epistemic_status == "questioned")
            ),
            "interlingua_hedged_claim_frames": float(sum(1 for claim in claims if claim.epistemic_status == "hedged")),
            "interlingua_negative_relations": float(
                sum(1 for relation in relations if relation.polarity == "negative")
            ),
            "interlingua_modal_relations": float(sum(1 for relation in relations if relation.modality)),
            "interlingua_conditioned_relations": float(sum(1 for relation in relations if relation.condition_key)),
            "interlingua_explained_relations": float(sum(1 for relation in relations if relation.explanation_key)),
            "interlingua_temporal_relations": float(sum(1 for relation in relations if relation.temporal_key)),
            "interlingua_coreference_links": float(len(scene.coreference_links)),
            "interlingua_uncertain_claims": float(
                sum(1 for relation in relations if relation.confidence < 0.6)
                + sum(1 for state in states if state.confidence < 0.6)
                + sum(1 for goal in goals if goal.confidence < 0.6)
                + sum(1 for claim in claims if claim.confidence < 0.6 or claim.epistemic_status != "asserted")
            ),
            "interlingua_mean_entity_confidence": float(
                sum(entity.confidence for entity in entities) / max(len(entities), 1)
            ),
            "interlingua_mean_relation_confidence": float(
                sum(relation.confidence for relation in relations) / max(len(relations), 1)
            ),
            "interlingua_structural_evidence_refs": float(
                sum(len(state.evidence_refs) for state in states)
                + sum(len(relation.evidence_refs) for relation in relations)
                + sum(len(goal.evidence_refs) for goal in goals)
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
        claims=tuple(claims),
        metadata=metadata,
    )

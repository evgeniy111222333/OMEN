from __future__ import annotations

from typing import Dict, Optional, Set

from .backbone import SemanticGroundingBackbone
from .heuristic_backbone import HeuristicFallbackSemanticBackbone
from .semantic_context import build_semantic_context_objects
from .scene_types import SemanticClaim, SemanticEntity, SemanticGoal, SemanticSceneGraph, SemanticState
from .structural_scene import build_structural_scene_graph, structural_primary_segment_indices
from .text_semantics import normalize_symbol_text
from .types import GroundedTextDocument


def _empty_scene(document: GroundedTextDocument) -> SemanticSceneGraph:
    metadata = dict(document.metadata)
    metadata.update(
        {
            "scene_entities": 0.0,
            "scene_states": 0.0,
            "scene_events": 0.0,
            "scene_goals": 0.0,
            "scene_claims": 0.0,
            "scene_mentions": 0.0,
            "scene_discourse_relations": 0.0,
            "scene_temporal_markers": 0.0,
            "scene_explanations": 0.0,
            "scene_coreference_links": 0.0,
            "scene_negative_events": 0.0,
            "scene_event_modalities": 0.0,
            "scene_event_conditions": 0.0,
            "scene_event_explanations": 0.0,
            "scene_event_temporal_anchors": 0.0,
            "scene_entity_aliases": 0.0,
            "scene_mean_entity_confidence": 0.0,
            "scene_mean_event_confidence": 0.0,
            "scene_claim_attributed": 0.0,
            "scene_claim_nonasserted": 0.0,
        }
    )
    return SemanticSceneGraph(
        language=document.language,
        source_text=document.source_text,
        metadata=metadata,
    )


def _entity_key(entity: SemanticEntity) -> str:
    return normalize_symbol_text(entity.canonical_name) or str(entity.entity_id)


def _merge_entity(primary: SemanticEntity, structural: SemanticEntity) -> SemanticEntity:
    aliases = tuple(sorted({*primary.aliases, *structural.aliases, primary.canonical_name, structural.canonical_name}))
    source_segments = tuple(sorted({*primary.source_segments, *structural.source_segments}))
    source_spans = tuple(primary.source_spans) + tuple(
        span for span in structural.source_spans if span not in primary.source_spans
    )
    semantic_type = primary.semantic_type if primary.semantic_type != "entity" else structural.semantic_type
    confidence = max(float(primary.confidence), float(structural.confidence))
    status = "supported" if "supported" in {str(primary.status), str(structural.status)} else primary.status
    return SemanticEntity(
        entity_id=primary.entity_id,
        canonical_name=primary.canonical_name,
        semantic_type=semantic_type,
        aliases=aliases,
        source_segments=source_segments,
        source_spans=source_spans,
        confidence=confidence,
        status=status,
    )


def _rewire_state(state: SemanticState, entity_id_map: Dict[str, str]) -> SemanticState:
    target_entity = entity_id_map.get(state.key_entity_id, state.key_entity_id)
    if target_entity == state.key_entity_id:
        return state
    return SemanticState(
        state_id=state.state_id,
        key_entity_id=target_entity,
        key_name=state.key_name,
        value=state.value,
        source_segment=state.source_segment,
        source_span=state.source_span,
        confidence=state.confidence,
        status=state.status,
        evidence_refs=state.evidence_refs,
    )


def _rewire_goal(goal: SemanticGoal, entity_id_map: Dict[str, str]) -> SemanticGoal:
    if goal.target_entity_id is None:
        return goal
    target_entity = entity_id_map.get(goal.target_entity_id, goal.target_entity_id)
    if target_entity == goal.target_entity_id:
        return goal
    return SemanticGoal(
        goal_id=goal.goal_id,
        goal_name=goal.goal_name,
        goal_value=goal.goal_value,
        target_entity_id=target_entity,
        source_segment=goal.source_segment,
        source_span=goal.source_span,
        confidence=goal.confidence,
        status=goal.status,
        evidence_refs=goal.evidence_refs,
    )


def _rewire_claim(claim: SemanticClaim, entity_id_map: Dict[str, str]) -> SemanticClaim:
    subject_id = entity_id_map.get(claim.subject_entity_id, claim.subject_entity_id) if claim.subject_entity_id else None
    object_id = entity_id_map.get(claim.object_entity_id, claim.object_entity_id) if claim.object_entity_id else None
    speaker_id = entity_id_map.get(claim.speaker_entity_id, claim.speaker_entity_id) if claim.speaker_entity_id else None
    if (
        subject_id == claim.subject_entity_id
        and object_id == claim.object_entity_id
        and speaker_id == claim.speaker_entity_id
    ):
        return claim
    return SemanticClaim(
        claim_id=claim.claim_id,
        claim_kind=claim.claim_kind,
        source_segment=claim.source_segment,
        source_span=claim.source_span,
        confidence=claim.confidence,
        status=claim.status,
        subject_entity_id=subject_id,
        predicate=claim.predicate,
        object_entity_id=object_id,
        object_value=claim.object_value,
        proposition_id=claim.proposition_id,
        event_id=claim.event_id,
        goal_id=claim.goal_id,
        speaker_entity_id=speaker_id,
        speaker_name=claim.speaker_name,
        epistemic_status=claim.epistemic_status,
        claim_source=claim.claim_source,
        evidence_refs=claim.evidence_refs,
    )


def _merge_scene_graphs(
    document: GroundedTextDocument,
    fallback_scene: SemanticSceneGraph,
    structural_scene: SemanticSceneGraph,
    *,
    structured_segments: Set[int],
) -> SemanticSceneGraph:
    entity_id_map: Dict[str, str] = {}
    merged_entities = list(fallback_scene.entities)
    merged_entity_index: Dict[str, int] = {}
    for idx, entity in enumerate(merged_entities):
        merged_entity_index[_entity_key(entity)] = idx
        entity_id_map[entity.entity_id] = entity.entity_id
    for structural_entity in structural_scene.entities:
        key = _entity_key(structural_entity)
        if key in merged_entity_index:
            primary_idx = merged_entity_index[key]
            primary_entity = merged_entities[primary_idx]
            merged_entities[primary_idx] = _merge_entity(primary_entity, structural_entity)
            entity_id_map[structural_entity.entity_id] = primary_entity.entity_id
        else:
            merged_entity_index[key] = len(merged_entities)
            merged_entities.append(structural_entity)
            entity_id_map[structural_entity.entity_id] = structural_entity.entity_id

    filtered_states = [state for state in fallback_scene.states if int(state.source_segment) not in structured_segments]
    filtered_goals = [goal for goal in fallback_scene.goals if int(goal.source_segment) not in structured_segments]
    filtered_claims = [claim for claim in fallback_scene.claims if int(claim.source_segment) not in structured_segments]
    merged_states = tuple(filtered_states) + tuple(_rewire_state(state, entity_id_map) for state in structural_scene.states)
    merged_goals = tuple(filtered_goals) + tuple(_rewire_goal(goal, entity_id_map) for goal in structural_scene.goals)
    merged_claims = tuple(filtered_claims) + tuple(_rewire_claim(claim, entity_id_map) for claim in structural_scene.claims)
    mentions, discourse_relations, temporal_markers, explanations = build_semantic_context_objects(
        document,
        tuple(merged_entities),
    )
    metadata = dict(fallback_scene.metadata)
    metadata.update(
        {
            key: value
            for key, value in structural_scene.metadata.items()
            if key.startswith("scene_structural_primary_")
        }
    )
    metadata.update(
        {
            "scene_entities": float(len(merged_entities)),
            "scene_states": float(len(merged_states)),
            "scene_events": float(len(fallback_scene.events)),
            "scene_goals": float(len(merged_goals)),
            "scene_claims": float(len(merged_claims)),
            "scene_claim_attributed": float(sum(1 for claim in merged_claims if claim.speaker_entity_id)),
            "scene_claim_nonasserted": float(
                sum(1 for claim in merged_claims if str(claim.epistemic_status) != "asserted")
            ),
            "scene_mentions": float(len(mentions)),
            "scene_discourse_relations": float(len(discourse_relations)),
            "scene_temporal_markers": float(len(temporal_markers)),
            "scene_explanations": float(len(explanations)),
            "scene_coreference_links": float(len(fallback_scene.coreference_links)),
            "scene_mean_entity_confidence": float(
                sum(entity.confidence for entity in merged_entities) / max(len(merged_entities), 1)
            ),
            "scene_mean_event_confidence": float(
                sum(event.confidence for event in fallback_scene.events) / max(len(fallback_scene.events), 1)
            ),
            "scene_fallback_backbone_active": float(fallback_scene.metadata.get("scene_fallback_backbone_active", 1.0)),
            "scene_fallback_low_authority": float(fallback_scene.metadata.get("scene_fallback_low_authority", 1.0)),
            "scene_backbone_replaceable": 1.0,
            "scene_structural_primary_active": 1.0,
            "scene_structural_primary_hybrid_active": 1.0,
            "scene_structural_primary_segments": float(len(structured_segments)),
        }
    )
    return SemanticSceneGraph(
        language=document.language,
        source_text=document.source_text,
        entities=tuple(merged_entities),
        states=tuple(merged_states),
        events=tuple(fallback_scene.events),
        goals=tuple(merged_goals),
        claims=tuple(merged_claims),
        mentions=mentions,
        discourse_relations=discourse_relations,
        temporal_markers=temporal_markers,
        explanations=explanations,
        coreference_links=tuple(fallback_scene.coreference_links),
        metadata=metadata,
    )


def build_semantic_scene_graph(
    document: GroundedTextDocument,
    *,
    backbone: Optional[SemanticGroundingBackbone] = None,
) -> SemanticSceneGraph:
    if backbone is None:
        structural_primary = build_structural_scene_graph(document)
        if structural_primary is not None:
            return structural_primary
    active_backbone: SemanticGroundingBackbone = backbone or HeuristicFallbackSemanticBackbone()
    try:
        proposed = active_backbone.build_scene_graph(document)
    except Exception:
        proposed = None
    if proposed is not None and backbone is None:
        structural_overlay = build_structural_scene_graph(document, allow_partial=True)
        structured_segments = set(structural_primary_segment_indices(document))
        if structural_overlay is not None and structured_segments:
            return _merge_scene_graphs(
                document,
                proposed,
                structural_overlay,
                structured_segments=structured_segments,
            )
    if proposed is not None:
        return proposed
    if backbone is None:
        structural_overlay = build_structural_scene_graph(document, allow_partial=True)
        if structural_overlay is not None:
            return structural_overlay
    return _empty_scene(document)

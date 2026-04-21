from __future__ import annotations

from typing import Dict, Optional, Set

from .backbone import SemanticGroundingBackbone
from .heuristic_backbone import HeuristicFallbackSemanticBackbone
from .learned_backbone import get_default_learned_grounding_backbone
from .semantic_context import build_semantic_context_objects
from .scene_types import SemanticClaim, SemanticEntity, SemanticEvent, SemanticGoal, SemanticSceneGraph, SemanticState
from .structural_scene import build_structural_scene_graph, structural_primary_segment_indices
from .text_semantics import normalize_symbol_text
from .types import GroundedTextDocument

_HYBRID_STRUCTURAL_SUBTYPES = frozenset({"dialogue_text", "instructional_text"})


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


def _copy_scene_with_metadata(
    scene: SemanticSceneGraph,
    metadata_updates: Dict[str, float],
) -> SemanticSceneGraph:
    metadata = dict(scene.metadata)
    metadata.update({key: float(value) for key, value in metadata_updates.items()})
    return SemanticSceneGraph(
        language=scene.language,
        source_text=scene.source_text,
        entities=scene.entities,
        states=scene.states,
        events=scene.events,
        goals=scene.goals,
        claims=scene.claims,
        mentions=scene.mentions,
        discourse_relations=scene.discourse_relations,
        temporal_markers=scene.temporal_markers,
        explanations=scene.explanations,
        coreference_links=scene.coreference_links,
        metadata=metadata,
    )


def _annotate_backbone_runtime(
    scene: SemanticSceneGraph,
    document: GroundedTextDocument,
    *,
    default_heuristic_active: bool,
    default_learned_active: bool,
    default_learned_attempted: bool,
    explicit_backbone_active: bool,
    missing_semantic_backbone: bool,
    retained_fallback_segments: float = 0.0,
) -> SemanticSceneGraph:
    total_segments = float(len(tuple(getattr(document, "segments", ()) or ())))
    fallback_active = float(
        scene.metadata.get(
            "scene_fallback_backbone_active",
            1.0 if default_heuristic_active else 0.0,
        )
    )
    fallback_low_authority = float(
        scene.metadata.get(
            "scene_fallback_low_authority",
            1.0 if fallback_active > 0.0 else 0.0,
        )
    )
    learned_active = float(scene.metadata.get("scene_learned_backbone_active", 0.0))
    trainable_active = float(
        scene.metadata.get(
            "scene_trainable_backbone_active",
            learned_active,
        )
    )
    retained_segments = max(0.0, float(retained_fallback_segments))
    if total_segments > 0.0:
        retained_segments = min(retained_segments, total_segments)
    else:
        retained_segments = 0.0
    retained_rate = retained_segments / total_segments if total_segments > 0.0 else 0.0
    return _copy_scene_with_metadata(
        scene,
        {
            "scene_fallback_backbone_active": fallback_active,
            "scene_fallback_low_authority": fallback_low_authority,
            "scene_default_heuristic_backbone_active": 1.0 if default_heuristic_active else 0.0,
            "scene_default_learned_backbone_active": 1.0 if default_learned_active else 0.0,
            "scene_default_learned_backbone_attempted": 1.0 if default_learned_attempted else 0.0,
            "scene_explicit_backbone_active": 1.0 if explicit_backbone_active else 0.0,
            "scene_trainable_backbone_active": trainable_active,
            "scene_learned_backbone_active": learned_active,
            "scene_missing_semantic_backbone": 1.0 if missing_semantic_backbone else 0.0,
            "scene_backbone_degraded_mode": 1.0 if default_heuristic_active else 0.0,
            "scene_total_segments": total_segments,
            "scene_heuristic_fallback_retained_segments": retained_segments if fallback_active > 0.0 else 0.0,
            "scene_heuristic_fallback_retained_rate": retained_rate if fallback_active > 0.0 else 0.0,
        },
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


def _rewire_event(event: SemanticEvent, entity_id_map: Dict[str, str]) -> SemanticEvent:
    subject_id = entity_id_map.get(event.subject_entity_id, event.subject_entity_id) if event.subject_entity_id else None
    object_id = entity_id_map.get(event.object_entity_id, event.object_entity_id) if event.object_entity_id else None
    agent_id = entity_id_map.get(event.agent_entity_id, event.agent_entity_id) if event.agent_entity_id else None
    patient_id = entity_id_map.get(event.patient_entity_id, event.patient_entity_id) if event.patient_entity_id else None
    if (
        subject_id == event.subject_entity_id
        and object_id == event.object_entity_id
        and agent_id == event.agent_entity_id
        and patient_id == event.patient_entity_id
    ):
        return event
    return SemanticEvent(
        event_id=event.event_id,
        event_type=event.event_type,
        subject_entity_id=subject_id,
        object_entity_id=object_id,
        agent_entity_id=agent_id,
        patient_entity_id=patient_id,
        subject_name=event.subject_name,
        object_name=event.object_name,
        modality=event.modality,
        condition=event.condition,
        explanation=event.explanation,
        temporal=event.temporal,
        source_segment=event.source_segment,
        source_span=event.source_span,
        confidence=event.confidence,
        polarity=event.polarity,
        status=event.status,
        evidence_refs=event.evidence_refs,
        metadata=event.metadata,
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
        semantic_mode=getattr(claim, "semantic_mode", "instance"),
        quantifier_mode=getattr(claim, "quantifier_mode", "instance"),
        evidence_refs=claim.evidence_refs,
    )


def _segment_owner_map(
    document: GroundedTextDocument,
    *,
    structured_segments: Set[int],
) -> Dict[int, str]:
    owners: Dict[int, str] = {}
    for segment in tuple(getattr(document, "segments", ()) or ()):
        seg_idx = int(getattr(segment, "index", 0))
        if seg_idx not in structured_segments:
            owners[seg_idx] = "fallback_primary"
            continue
        routing = getattr(segment, "routing", None)
        modality = str(getattr(routing, "modality", "") or "")
        subtype = str(getattr(routing, "subtype", "") or "")
        if modality in {"natural_text", "mixed"} and subtype in _HYBRID_STRUCTURAL_SUBTYPES:
            owners[seg_idx] = "hybrid"
        else:
            owners[seg_idx] = "structural_primary"
    return owners


def _keep_fallback_claim(claim: SemanticClaim, segment_owners: Dict[int, str]) -> bool:
    owner = segment_owners.get(int(claim.source_segment), "fallback_primary")
    return owner == "fallback_primary"


def _keep_fallback_event(source_segment: int, segment_owners: Dict[int, str]) -> bool:
    owner = segment_owners.get(int(source_segment), "fallback_primary")
    return owner == "fallback_primary"


def _keep_fallback_coreference(source_segment: int, target_segment: int, segment_owners: Dict[int, str]) -> bool:
    source_owner = segment_owners.get(int(source_segment), "fallback_primary")
    target_owner = segment_owners.get(int(target_segment), "fallback_primary")
    return source_owner == "fallback_primary" and target_owner == "fallback_primary"


def _merge_scene_graphs(
    document: GroundedTextDocument,
    fallback_scene: SemanticSceneGraph,
    structural_scene: SemanticSceneGraph,
    *,
    structured_segments: Set[int],
) -> SemanticSceneGraph:
    segment_owners = _segment_owner_map(document, structured_segments=structured_segments)
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

    filtered_states = [
        state
        for state in fallback_scene.states
        if segment_owners.get(int(state.source_segment), "fallback_primary") == "fallback_primary"
    ]
    filtered_goals = [
        goal
        for goal in fallback_scene.goals
        if segment_owners.get(int(goal.source_segment), "fallback_primary") == "fallback_primary"
    ]
    filtered_claims = [
        claim
        for claim in fallback_scene.claims
        if _keep_fallback_claim(claim, segment_owners)
    ]
    filtered_events = [
        event
        for event in fallback_scene.events
        if _keep_fallback_event(int(event.source_segment), segment_owners)
    ]
    filtered_coreference_links = [
        link
        for link in fallback_scene.coreference_links
        if _keep_fallback_coreference(int(link.source_segment), int(link.target_segment), segment_owners)
    ]
    merged_events = tuple(filtered_events) + tuple(
        _rewire_event(event, entity_id_map) for event in structural_scene.events
    )
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
            "scene_events": float(len(merged_events)),
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
            "scene_coreference_links": float(len(filtered_coreference_links)),
            "scene_mean_entity_confidence": float(
                sum(entity.confidence for entity in merged_entities) / max(len(merged_entities), 1)
            ),
            "scene_mean_event_confidence": float(
                sum(event.confidence for event in merged_events) / max(len(merged_events), 1)
            ),
            "scene_negative_events": float(sum(1 for event in merged_events if event.polarity != "positive")),
            "scene_event_modalities": float(sum(1 for event in merged_events if event.modality)),
            "scene_event_conditions": float(sum(1 for event in merged_events if event.condition)),
            "scene_event_explanations": float(sum(1 for event in merged_events if event.explanation)),
            "scene_event_temporal_anchors": float(sum(1 for event in merged_events if event.temporal)),
            "scene_fallback_backbone_active": float(fallback_scene.metadata.get("scene_fallback_backbone_active", 1.0)),
            "scene_fallback_low_authority": float(fallback_scene.metadata.get("scene_fallback_low_authority", 1.0)),
            "scene_backbone_replaceable": 1.0,
            "scene_structural_primary_active": 1.0,
            "scene_structural_primary_hybrid_active": 1.0
            if any(owner == "hybrid" for owner in segment_owners.values())
            else 0.0,
            "scene_structural_primary_segments": float(len(structured_segments)),
            "scene_segment_owner_structural_primary": float(
                sum(1 for owner in segment_owners.values() if owner == "structural_primary")
            ),
            "scene_segment_owner_hybrid": float(sum(1 for owner in segment_owners.values() if owner == "hybrid")),
            "scene_segment_owner_fallback_primary": float(
                sum(1 for owner in segment_owners.values() if owner == "fallback_primary")
            ),
            "scene_hybrid_retained_fallback_events": float(
                sum(
                    1
                    for event in filtered_events
                    if segment_owners.get(int(event.source_segment), "fallback_primary") == "hybrid"
                )
            ),
            "scene_hybrid_retained_fallback_claims": float(
                sum(
                    1
                    for claim in filtered_claims
                    if segment_owners.get(int(claim.source_segment), "fallback_primary") == "hybrid"
                )
            ),
        }
    )
    return SemanticSceneGraph(
        language=document.language,
        source_text=document.source_text,
        entities=tuple(merged_entities),
        states=tuple(merged_states),
        events=tuple(merged_events),
        goals=tuple(merged_goals),
        claims=tuple(merged_claims),
        mentions=mentions,
        discourse_relations=discourse_relations,
        temporal_markers=temporal_markers,
        explanations=explanations,
        coreference_links=tuple(filtered_coreference_links),
        metadata=metadata,
    )


def build_semantic_scene_graph(
    document: GroundedTextDocument,
    *,
    backbone: Optional[SemanticGroundingBackbone] = None,
    allow_heuristic_fallback: bool = True,
) -> SemanticSceneGraph:
    if backbone is None:
        structural_primary = build_structural_scene_graph(document)
        if structural_primary is not None:
            return _annotate_backbone_runtime(
                structural_primary,
                document,
                default_heuristic_active=False,
                default_learned_active=False,
                default_learned_attempted=False,
                explicit_backbone_active=False,
                missing_semantic_backbone=False,
                retained_fallback_segments=0.0,
            )
    explicit_backbone_active = backbone is not None
    default_heuristic_active = False
    default_learned_active = False
    default_learned_attempted = False
    active_backbone: Optional[SemanticGroundingBackbone] = backbone
    if active_backbone is None:
        default_learned_attempted = True
        try:
            active_backbone = get_default_learned_grounding_backbone()
            default_learned_active = active_backbone is not None
        except Exception:
            active_backbone = None
    try:
        proposed = active_backbone.build_scene_graph(document) if active_backbone is not None else None
    except Exception:
        proposed = None
    if proposed is None and backbone is None and allow_heuristic_fallback:
        default_learned_active = False
        default_heuristic_active = True
        try:
            proposed = HeuristicFallbackSemanticBackbone().build_scene_graph(document)
        except Exception:
            proposed = None
    if proposed is not None and backbone is None:
        structural_overlay = build_structural_scene_graph(document, allow_partial=True)
        structured_segments = set(structural_primary_segment_indices(document))
        if structural_overlay is not None and structured_segments:
            merged_scene = _merge_scene_graphs(
                document,
                proposed,
                structural_overlay,
                structured_segments=structured_segments,
            )
            return _annotate_backbone_runtime(
                merged_scene,
                document,
                default_heuristic_active=default_heuristic_active,
                default_learned_active=default_learned_active,
                default_learned_attempted=default_learned_attempted,
                explicit_backbone_active=explicit_backbone_active,
                missing_semantic_backbone=False,
                retained_fallback_segments=float(
                    merged_scene.metadata.get("scene_segment_owner_fallback_primary", 0.0)
                ),
            )
    if proposed is not None:
        retained_fallback_segments = 0.0
        if float(proposed.metadata.get("scene_fallback_backbone_active", 0.0)) > 0.0:
            retained_fallback_segments = float(len(tuple(getattr(document, "segments", ()) or ())))
        return _annotate_backbone_runtime(
            proposed,
            document,
            default_heuristic_active=default_heuristic_active,
            default_learned_active=default_learned_active,
            default_learned_attempted=default_learned_attempted,
            explicit_backbone_active=explicit_backbone_active,
            missing_semantic_backbone=False,
            retained_fallback_segments=retained_fallback_segments,
        )
    if backbone is None:
        structural_overlay = build_structural_scene_graph(document, allow_partial=True)
        if structural_overlay is not None:
            return _annotate_backbone_runtime(
                structural_overlay,
                document,
                default_heuristic_active=False,
                default_learned_active=False,
                default_learned_attempted=default_learned_attempted,
                explicit_backbone_active=False,
                missing_semantic_backbone=not allow_heuristic_fallback,
                retained_fallback_segments=0.0,
            )
    return _annotate_backbone_runtime(
        _empty_scene(document),
        document,
        default_heuristic_active=False,
        default_learned_active=False,
        default_learned_attempted=default_learned_attempted,
        explicit_backbone_active=explicit_backbone_active,
        missing_semantic_backbone=backbone is None and not allow_heuristic_fallback,
        retained_fallback_segments=0.0,
    )

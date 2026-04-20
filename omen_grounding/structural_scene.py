from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .semantic_context import build_semantic_context_objects
from .scene_types import (
    SemanticClaim,
    SemanticEntity,
    SemanticGoal,
    SemanticSceneGraph,
    SemanticState,
)
from .text_semantics import normalize_symbol_text
from .types import GroundedTextDocument, GroundedStructuralUnit


_GOAL_KEYS = frozenset(
    {
        "goal",
        "target",
        "desired",
        "expected",
        "objective",
        "aim",
        "мета",
        "ціль",
        "очікується",
        "очікуваний_результат",
    }
)

_STRUCTURED_RECORD_UNITS = frozenset(
    {
        "key_value_record",
        "json_record",
        "log_entry",
        "table_row",
    }
)


@dataclass
class _EntityAccumulator:
    canonical_name: str
    semantic_type: str = "entity"
    aliases: Set[str] = field(default_factory=set)
    source_segments: Set[int] = field(default_factory=set)
    source_spans: List[object] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    status: str = "supported"

    def add(
        self,
        *,
        alias: Optional[str],
        source_segment: int,
        source_span,
        confidence: float,
    ) -> None:
        if alias:
            self.aliases.add(alias)
        self.source_segments.add(int(source_segment))
        if source_span is not None:
            self.source_spans.append(source_span)
        self.confidences.append(float(confidence))

    def finalize(self, entity_id: str) -> SemanticEntity:
        mean_conf = sum(self.confidences) / max(len(self.confidences), 1)
        aliases = set(self.aliases)
        aliases.add(self.canonical_name)
        return SemanticEntity(
            entity_id=entity_id,
            canonical_name=self.canonical_name,
            semantic_type=self.semantic_type,
            aliases=tuple(sorted(aliases)),
            source_segments=tuple(sorted(self.source_segments)),
            source_spans=tuple(self.source_spans),
            confidence=float(mean_conf),
            status=self.status,
        )


def _normalized(value: object, *, fallback: str = "") -> str:
    text = normalize_symbol_text(value)
    if text:
        return text
    return fallback


def _structured_only_document(document: GroundedTextDocument) -> bool:
    selected_segments = structural_primary_segment_indices(document)
    if not selected_segments:
        return False
    return _structured_only_for_segments(document, selected_segments)


def _structured_only_for_segments(
    document: GroundedTextDocument,
    selected_segments: Tuple[int, ...],
) -> bool:
    segments = tuple(getattr(document, "segments", ()) or ())
    if not segments:
        return False
    active_segments = [segment for segment in segments if str(getattr(segment, "text", "")).strip()]
    if not active_segments:
        return False
    selected = {int(item) for item in selected_segments}
    return len(selected) == len(active_segments) and all(int(segment.index) in selected for segment in active_segments)


def structural_primary_segment_indices(document: GroundedTextDocument) -> Tuple[int, ...]:
    selected: List[int] = []
    for segment in tuple(getattr(document, "segments", ()) or ()):
        routing = getattr(segment, "routing", None)
        modality = str(getattr(routing, "modality", "") or "")
        unit_types = {
            str(getattr(unit, "unit_type", "") or "")
            for unit in tuple(getattr(segment, "structural_units", ()) or ())
        }
        if "section_header" in unit_types:
            selected.append(int(segment.index))
            continue
        if modality != "structured_text":
            continue
        if unit_types.intersection(_STRUCTURED_RECORD_UNITS):
            selected.append(int(segment.index))
    return tuple(sorted(dict.fromkeys(selected)))


def _unit_priority(unit: GroundedStructuralUnit) -> Tuple[int, str]:
    priorities = {
        "json_record": 0,
        "log_entry": 1,
        "table_row": 2,
        "key_value_record": 3,
        "section_header": 4,
    }
    return (priorities.get(str(unit.unit_type or ""), 9), str(unit.unit_id))


def build_structural_scene_graph(
    document: GroundedTextDocument,
    *,
    allow_partial: bool = False,
) -> Optional[SemanticSceneGraph]:
    selected_segments = structural_primary_segment_indices(document)
    if not selected_segments:
        return None
    structured_only = _structured_only_for_segments(document, selected_segments)
    if not allow_partial and not structured_only:
        return None
    selected_segment_set = set(selected_segments)

    entity_ids: Dict[str, str] = {}
    entities: Dict[str, _EntityAccumulator] = {}
    states: List[SemanticState] = []
    goals: List[SemanticGoal] = []
    claims: List[SemanticClaim] = []
    used_unit_ids: Set[str] = set()
    used_state_signatures: Set[Tuple[int, str, str]] = set()
    used_goal_signatures: Set[Tuple[int, str, str]] = set()
    unit_type_counts: Dict[str, float] = {}

    def ensure_entity(
        name: object,
        *,
        semantic_type: str,
        source_segment: int,
        source_span,
        confidence: float,
        alias: Optional[str] = None,
    ) -> str:
        canonical_name = _normalized(name, fallback=f"entity_{len(entity_ids)}")
        entity_id = entity_ids.get(canonical_name)
        if entity_id is None:
            entity_id = f"ent:struct:{len(entity_ids)}:{canonical_name}"
            entity_ids[canonical_name] = entity_id
            entities[entity_id] = _EntityAccumulator(canonical_name=canonical_name, semantic_type=semantic_type)
        entities[entity_id].add(
            alias=alias or canonical_name,
            source_segment=source_segment,
            source_span=source_span,
            confidence=confidence,
        )
        return entity_id

    for segment in document.segments:
        if int(segment.index) not in selected_segment_set:
            continue
        seg_idx = int(segment.index)
        structured_units = sorted(tuple(segment.structural_units or ()), key=_unit_priority)
        for unit in structured_units:
            unit_type = str(unit.unit_type or "")
            unit_type_counts[unit_type] = unit_type_counts.get(unit_type, 0.0) + 1.0
            evidence_refs = tuple(
                item
                for item in (
                    f"structural_unit:{unit.unit_id}",
                    f"unit_type:{unit_type}",
                    *tuple(str(reference) for reference in (unit.references or ()) if str(reference).strip()),
                )
                if item
            )
            confidence = max(0.64, float(unit.confidence))
            if unit_type == "section_header":
                section_name = ""
                if unit.fields:
                    section_name = _normalized(unit.fields[0][1], fallback=_normalized(unit.text, fallback="section"))
                if section_name:
                    ensure_entity(
                        section_name,
                        semantic_type="section",
                        source_segment=seg_idx,
                        source_span=unit.span or segment.span,
                        confidence=confidence,
                        alias=section_name,
                    )
                    used_unit_ids.add(unit.unit_id)
                continue
            if unit_type not in _STRUCTURED_RECORD_UNITS:
                continue
            for field_idx, (key, value) in enumerate(tuple(unit.fields or ())):
                key_name = _normalized(key)
                value_name = _normalized(value)
                if not key_name or not value_name:
                    continue
                used_unit_ids.add(unit.unit_id)
                if key_name in _GOAL_KEYS:
                    goal_signature = (seg_idx, key_name, value_name)
                    if goal_signature in used_goal_signatures:
                        continue
                    used_goal_signatures.add(goal_signature)
                    target_id = ensure_entity(
                        value_name,
                        semantic_type="entity",
                        source_segment=seg_idx,
                        source_span=unit.span or segment.span,
                        confidence=confidence,
                        alias=str(value),
                    )
                    goal_id = f"goal:struct:{seg_idx}:{len(goals)}"
                    goals.append(
                        SemanticGoal(
                            goal_id=goal_id,
                            goal_name=key_name,
                            goal_value=value_name,
                            target_entity_id=target_id,
                            source_segment=seg_idx,
                            source_span=unit.span or segment.span,
                            confidence=confidence,
                            status="supported",
                            evidence_refs=evidence_refs,
                        )
                    )
                    claims.append(
                        SemanticClaim(
                            claim_id=f"claim:struct:goal:{seg_idx}:{field_idx}:{len(claims)}",
                            claim_kind="goal",
                            source_segment=seg_idx,
                            source_span=unit.span or segment.span,
                            confidence=confidence,
                            status="supported",
                            predicate=key_name,
                            object_entity_id=target_id,
                            object_value=value_name,
                            goal_id=goal_id,
                            evidence_refs=evidence_refs,
                        )
                    )
                    continue
                state_signature = (seg_idx, key_name, value_name)
                if state_signature in used_state_signatures:
                    continue
                used_state_signatures.add(state_signature)
                entity_id = ensure_entity(
                    key_name,
                    semantic_type="attribute" if unit_type in _STRUCTURED_RECORD_UNITS else "entity",
                    source_segment=seg_idx,
                    source_span=unit.span or segment.span,
                    confidence=confidence,
                    alias=str(key),
                )
                state_id = f"state:struct:{seg_idx}:{len(states)}"
                states.append(
                    SemanticState(
                        state_id=state_id,
                        key_entity_id=entity_id,
                        key_name=key_name,
                        value=value_name,
                        source_segment=seg_idx,
                        source_span=unit.span or segment.span,
                        confidence=confidence,
                        status="supported",
                        evidence_refs=evidence_refs,
                    )
                )
                claims.append(
                    SemanticClaim(
                        claim_id=f"claim:struct:state:{seg_idx}:{field_idx}:{len(claims)}",
                        claim_kind="state",
                        source_segment=seg_idx,
                        source_span=unit.span or segment.span,
                        confidence=confidence,
                        status="supported",
                        subject_entity_id=entity_id,
                        predicate=key_name,
                        object_value=value_name,
                        evidence_refs=evidence_refs,
                    )
                )

    entities_out = tuple(
        entities[entity_id].finalize(entity_id)
        for entity_id in sorted(entities.keys())
    )
    mentions, discourse_relations, temporal_markers, explanations = build_semantic_context_objects(
        document,
        entities_out,
    )
    metadata = dict(document.metadata)
    metadata.update(
        {
            "scene_entities": float(len(entities_out)),
            "scene_states": float(len(states)),
            "scene_events": 0.0,
            "scene_goals": float(len(goals)),
            "scene_claims": float(len(claims)),
            "scene_mentions": float(len(mentions)),
            "scene_discourse_relations": float(len(discourse_relations)),
            "scene_temporal_markers": float(len(temporal_markers)),
            "scene_explanations": float(len(explanations)),
            "scene_coreference_links": 0.0,
            "scene_negative_events": 0.0,
            "scene_event_modalities": 0.0,
            "scene_event_conditions": 0.0,
            "scene_event_explanations": 0.0,
            "scene_event_temporal_anchors": 0.0,
            "scene_entity_aliases": float(sum(max(len(entity.aliases) - 1, 0) for entity in entities_out)),
            "scene_mean_entity_confidence": float(
                sum(entity.confidence for entity in entities_out) / max(len(entities_out), 1)
            ),
            "scene_mean_event_confidence": 0.0,
            "scene_fallback_backbone_active": 0.0,
            "scene_fallback_low_authority": 0.0,
            "scene_backbone_replaceable": 1.0,
            "scene_structural_primary_active": 1.0,
            "scene_structural_primary_segments": float(len(selected_segments)),
            "scene_structural_primary_units_used": float(len(used_unit_ids)),
            "scene_structural_primary_state_claims": float(len(states)),
            "scene_structural_primary_goal_claims": float(len(goals)),
            "scene_structural_primary_evidence_refs": float(
                sum(len(state.evidence_refs) for state in states)
                + sum(len(goal.evidence_refs) for goal in goals)
            ),
            "scene_structural_primary_partial_active": 0.0 if structured_only else 1.0,
        }
    )
    for unit_type, count in unit_type_counts.items():
        metadata[f"scene_structural_primary_unit_{unit_type}"] = float(count)
    return SemanticSceneGraph(
        language=document.language,
        source_text=document.source_text,
        entities=entities_out,
        states=tuple(states),
        events=tuple(),
        goals=tuple(goals),
        claims=tuple(claims),
        mentions=mentions,
        discourse_relations=discourse_relations,
        temporal_markers=temporal_markers,
        explanations=explanations,
        coreference_links=tuple(),
        metadata=metadata,
    )

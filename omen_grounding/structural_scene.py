from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .claim_semantics import infer_claim_semantics
from .semantic_context import build_semantic_context_objects
from .scene_types import (
    SemanticClaim,
    SemanticEntity,
    SemanticEvent,
    SemanticGoal,
    SemanticSceneGraph,
    SemanticState,
)
from .text_semantics import extract_goal_hints, extract_relation_hints, extract_structured_pairs, normalize_symbol_text
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

_NATURAL_PRIMARY_UNITS = frozenset({"speaker_turn", "clause"})
_NATURAL_SUPPORT_UNITS = frozenset({"speaker_turn", "clause", "citation_region"})
_NATURAL_STRUCTURAL_SUBTYPES = frozenset({"dialogue_text", "instructional_text"})
_CONDITIONAL_MARKERS = frozenset({"if", "when", "якщо", "коли"})
_EXPLANATION_MARKERS = frozenset({"because", "бо", "тому_що", "через"})
_TEMPORAL_MARKERS = frozenset({"after", "before", "then", "потім", "після", "перед"})


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


def _segment_evidence_refs(units: Tuple[GroundedStructuralUnit, ...]) -> Tuple[str, ...]:
    refs: List[str] = []
    for unit in units:
        unit_type = str(getattr(unit, "unit_type", "") or "")
        refs.append(f"structural_unit:{unit.unit_id}")
        refs.append(f"unit_type:{unit_type}")
        refs.extend(
            str(reference)
            for reference in tuple(getattr(unit, "references", ()) or ())
            if str(reference).strip()
        )
    return tuple(dict.fromkeys(refs))


def _clause_markers(units: Tuple[GroundedStructuralUnit, ...]) -> Tuple[str, ...]:
    markers: List[str] = []
    for unit in units:
        for reference in tuple(getattr(unit, "references", ()) or ()):
            ref_text = str(reference or "")
            if not ref_text.startswith("marker:"):
                continue
            marker = _normalized(ref_text.split(":", 1)[1])
            if marker:
                markers.append(marker)
    return tuple(dict.fromkeys(markers))


def _segment_event_controls(units: Tuple[GroundedStructuralUnit, ...]) -> Tuple[str, str, str]:
    markers = _clause_markers(units)
    condition = next((marker for marker in markers if marker in _CONDITIONAL_MARKERS), "")
    explanation = next((marker for marker in markers if marker in _EXPLANATION_MARKERS), "")
    temporal = next((marker for marker in markers if marker in _TEMPORAL_MARKERS), "")
    return condition, explanation, temporal


def structural_primary_segment_indices(document: GroundedTextDocument) -> Tuple[int, ...]:
    selected: List[int] = []
    for segment in tuple(getattr(document, "segments", ()) or ()):
        routing = getattr(segment, "routing", None)
        modality = str(getattr(routing, "modality", "") or "")
        subtype = str(getattr(routing, "subtype", "") or "")
        unit_types = {
            str(getattr(unit, "unit_type", "") or "")
            for unit in tuple(getattr(segment, "structural_units", ()) or ())
        }
        if "section_header" in unit_types:
            selected.append(int(segment.index))
            continue
        if modality == "structured_text" and unit_types.intersection(_STRUCTURED_RECORD_UNITS):
            selected.append(int(segment.index))
            continue
        if (
            modality in {"natural_text", "mixed"}
            and subtype in _NATURAL_STRUCTURAL_SUBTYPES
            and unit_types.intersection(_NATURAL_PRIMARY_UNITS)
        ):
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
    events: List[SemanticEvent] = []
    goals: List[SemanticGoal] = []
    claims: List[SemanticClaim] = []
    used_unit_ids: Set[str] = set()
    used_state_signatures: Set[Tuple[int, str, str]] = set()
    used_event_signatures: Set[Tuple[int, str, str, str]] = set()
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
        routing = getattr(segment, "routing", None)
        subtype = str(getattr(routing, "subtype", "") or "")
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
                            proposition_id=goal_id,
                            goal_id=goal_id,
                            epistemic_status="asserted",
                            claim_source="structured_record",
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
                        proposition_id=state_id,
                        epistemic_status="asserted",
                        claim_source="structured_record",
                        evidence_refs=evidence_refs,
                    )
                )

        natural_support_units = tuple(
            unit for unit in structured_units if str(getattr(unit, "unit_type", "") or "") in _NATURAL_SUPPORT_UNITS
        )
        natural_primary = bool(
            subtype in _NATURAL_STRUCTURAL_SUBTYPES
            and any(str(getattr(unit, "unit_type", "") or "") in _NATURAL_PRIMARY_UNITS for unit in natural_support_units)
        )
        if natural_primary:
            natural_evidence_refs = _segment_evidence_refs(natural_support_units)
            condition, explanation, temporal = _segment_event_controls(natural_support_units)
            claim_profile = infer_claim_semantics(
                segment.text,
                structural_units=natural_support_units,
            )
            segment_speaker_id: Optional[str] = None
            segment_speaker_name = claim_profile.speaker_name or None
            natural_state_candidates: List[Tuple[str, str, object, float, str]] = []
            natural_goal_candidates = list(tuple(getattr(segment, "goals", ()) or ()))
            natural_relation_candidates = list(tuple(getattr(segment, "relations", ()) or ()))
            for unit in natural_support_units:
                used_unit_ids.add(unit.unit_id)
                if str(getattr(unit, "unit_type", "") or "") != "speaker_turn":
                    unit_text = str(unit.text or "")
                else:
                    speaker_value = next(
                        (value for key, value in tuple(getattr(unit, "fields", ()) or ()) if str(key) == "speaker"),
                        None,
                    )
                    if speaker_value is None:
                        match = re.match(
                            r"^(user|assistant|speaker|q|a)\s*[:>-]\s*(.+)$",
                            str(unit.text or ""),
                            flags=re.IGNORECASE,
                        )
                        if match is not None:
                            speaker_value = match.group(1).strip().casefold()
                    if speaker_value is not None:
                        speaker_entity_id = ensure_entity(
                            speaker_value,
                            semantic_type="speaker",
                            source_segment=seg_idx,
                            source_span=unit.span or segment.span,
                            confidence=max(0.58, float(unit.confidence)),
                            alias=str(speaker_value),
                        )
                        if segment_speaker_id is None:
                            segment_speaker_id = speaker_entity_id
                        if segment_speaker_name is None:
                            segment_speaker_name = _normalized(speaker_value)
                    match = re.match(r"^(user|assistant|speaker|q|a)\s*[:>-]\s*(.+)$", str(unit.text or ""), flags=re.IGNORECASE)
                    unit_text = match.group(2).strip() if match is not None else str(unit.text or "")
                structured_pairs = extract_structured_pairs(unit_text)
                goal_pairs = {
                    (_normalized(getattr(goal, "goal_name", "")), _normalized(getattr(goal, "goal_value", "")))
                    for goal in extract_goal_hints(unit_text, structured_pairs=structured_pairs)
                    if _normalized(getattr(goal, "goal_name", "")) and _normalized(getattr(goal, "goal_value", ""))
                }
                natural_goal_candidates.extend(extract_goal_hints(unit_text, structured_pairs=structured_pairs))
                natural_relation_candidates.extend(extract_relation_hints(unit_text))
                for key, value in structured_pairs:
                    key_name = _normalized(key)
                    value_name = _normalized(value)
                    if not key_name or not value_name or (key_name, value_name) in goal_pairs:
                        continue
                    natural_state_candidates.append(
                        (key_name, value_name, unit.span or segment.span, max(0.56, float(unit.confidence)), str(key))
                    )
            for state_idx, (key_name, value_name, state_span, state_confidence, raw_key) in enumerate(natural_state_candidates):
                if not key_name or not value_name:
                    continue
                state_signature = (seg_idx, key_name, value_name)
                if state_signature in used_state_signatures:
                    continue
                used_state_signatures.add(state_signature)
                entity_id = ensure_entity(
                    key_name,
                    semantic_type="attribute",
                    source_segment=seg_idx,
                    source_span=state_span,
                    confidence=state_confidence,
                    alias=raw_key,
                )
                state_id = f"state:struct:natural:{seg_idx}:{state_idx}"
                evidence_refs = natural_evidence_refs + (f"state_hint:{key_name}",)
                states.append(
                    SemanticState(
                        state_id=state_id,
                        key_entity_id=entity_id,
                        key_name=key_name,
                        value=value_name,
                        source_segment=seg_idx,
                        source_span=state_span,
                        confidence=state_confidence,
                        status="supported",
                        evidence_refs=evidence_refs,
                    )
                )
                claims.append(
                    SemanticClaim(
                        claim_id=f"claim:struct:natural:state:{seg_idx}:{state_idx}",
                        claim_kind="state",
                        source_segment=seg_idx,
                        source_span=state_span,
                        confidence=state_confidence,
                        status="supported",
                        subject_entity_id=entity_id,
                        predicate=key_name,
                        object_value=value_name,
                        proposition_id=state_id,
                        speaker_entity_id=segment_speaker_id,
                        speaker_name=segment_speaker_name,
                        epistemic_status=claim_profile.epistemic_status,
                        claim_source=claim_profile.claim_source,
                        evidence_refs=evidence_refs,
                    )
                )
            for goal_idx, goal in enumerate(natural_goal_candidates):
                goal_name = _normalized(getattr(goal, "goal_name", ""))
                goal_value = _normalized(getattr(goal, "goal_value", ""))
                if not goal_name or not goal_value:
                    continue
                goal_signature = (seg_idx, goal_name, goal_value)
                if goal_signature in used_goal_signatures:
                    continue
                used_goal_signatures.add(goal_signature)
                confidence = max(0.60, float(getattr(goal, "confidence", 0.0)))
                target_id = ensure_entity(
                    goal_value,
                    semantic_type="entity",
                    source_segment=seg_idx,
                    source_span=getattr(goal, "span", None) or segment.span,
                    confidence=confidence,
                    alias=str(getattr(goal, "goal_value", goal_value)),
                )
                goal_id = f"goal:struct:natural:{seg_idx}:{goal_idx}"
                evidence_refs = natural_evidence_refs + (f"goal_hint:{goal_name}",)
                goals.append(
                    SemanticGoal(
                        goal_id=goal_id,
                        goal_name=goal_name,
                        goal_value=goal_value,
                        target_entity_id=target_id,
                        source_segment=seg_idx,
                        source_span=getattr(goal, "span", None) or segment.span,
                        confidence=confidence,
                        status="supported",
                        evidence_refs=evidence_refs,
                    )
                )
                claims.append(
                    SemanticClaim(
                        claim_id=f"claim:struct:natural:goal:{seg_idx}:{goal_idx}",
                        claim_kind="goal",
                        source_segment=seg_idx,
                        source_span=getattr(goal, "span", None) or segment.span,
                        confidence=confidence,
                        status="supported",
                        predicate=goal_name,
                        object_entity_id=target_id,
                        object_value=goal_value,
                        proposition_id=goal_id,
                        goal_id=goal_id,
                        speaker_entity_id=segment_speaker_id,
                        speaker_name=segment_speaker_name,
                        epistemic_status=claim_profile.epistemic_status,
                        claim_source=claim_profile.claim_source,
                        evidence_refs=evidence_refs,
                    )
                )
            for relation_idx, relation in enumerate(natural_relation_candidates):
                left_name = _normalized(getattr(relation, "left", ""))
                predicate = _normalized(getattr(relation, "relation", ""))
                right_name = _normalized(getattr(relation, "right", ""))
                if not left_name or not predicate or not right_name:
                    continue
                event_signature = (seg_idx, left_name, predicate, right_name)
                if event_signature in used_event_signatures:
                    continue
                used_event_signatures.add(event_signature)
                confidence = max(0.60, float(getattr(relation, "confidence", 0.0)))
                subject_id = ensure_entity(
                    left_name,
                    semantic_type="entity",
                    source_segment=seg_idx,
                    source_span=getattr(relation, "span", None) or segment.span,
                    confidence=confidence,
                    alias=str(getattr(relation, "left", left_name)),
                )
                object_id = ensure_entity(
                    right_name,
                    semantic_type="entity",
                    source_segment=seg_idx,
                    source_span=getattr(relation, "span", None) or segment.span,
                    confidence=confidence,
                    alias=str(getattr(relation, "right", right_name)),
                )
                event_id = f"event:struct:natural:{seg_idx}:{relation_idx}"
                evidence_refs = natural_evidence_refs + (f"relation_hint:{predicate}",)
                events.append(
                    SemanticEvent(
                        event_id=event_id,
                        event_type=predicate,
                        subject_entity_id=subject_id,
                        object_entity_id=object_id,
                        subject_name=left_name,
                        object_name=right_name,
                        condition=condition,
                        explanation=explanation,
                        temporal=temporal,
                        source_segment=seg_idx,
                        source_span=getattr(relation, "span", None) or segment.span,
                        confidence=confidence,
                        polarity="negative" if bool(getattr(segment, "counterexample", False)) else "positive",
                        status="supported",
                        evidence_refs=evidence_refs,
                    )
                )
                claims.append(
                    SemanticClaim(
                        claim_id=f"claim:struct:natural:relation:{seg_idx}:{relation_idx}",
                        claim_kind="relation",
                        source_segment=seg_idx,
                        source_span=getattr(relation, "span", None) or segment.span,
                        confidence=confidence,
                        status="supported",
                        subject_entity_id=subject_id,
                        predicate=predicate,
                        object_entity_id=object_id,
                        object_value=right_name,
                        proposition_id=event_id,
                        event_id=event_id,
                        speaker_entity_id=segment_speaker_id,
                        speaker_name=segment_speaker_name,
                        epistemic_status=claim_profile.epistemic_status,
                        claim_source=claim_profile.claim_source,
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
            "scene_events": float(len(events)),
            "scene_goals": float(len(goals)),
            "scene_claims": float(len(claims)),
            "scene_claim_attributed": float(sum(1 for claim in claims if claim.speaker_entity_id)),
            "scene_claim_nonasserted": float(
                sum(1 for claim in claims if str(claim.epistemic_status) != "asserted")
            ),
            "scene_mentions": float(len(mentions)),
            "scene_discourse_relations": float(len(discourse_relations)),
            "scene_temporal_markers": float(len(temporal_markers)),
            "scene_explanations": float(len(explanations)),
            "scene_coreference_links": 0.0,
            "scene_negative_events": float(sum(1 for event in events if event.polarity != "positive")),
            "scene_event_modalities": float(sum(1 for event in events if event.modality)),
            "scene_event_conditions": float(sum(1 for event in events if event.condition)),
            "scene_event_explanations": float(sum(1 for event in events if event.explanation)),
            "scene_event_temporal_anchors": float(sum(1 for event in events if event.temporal)),
            "scene_entity_aliases": float(sum(max(len(entity.aliases) - 1, 0) for entity in entities_out)),
            "scene_mean_entity_confidence": float(
                sum(entity.confidence for entity in entities_out) / max(len(entities_out), 1)
            ),
            "scene_mean_event_confidence": float(
                sum(event.confidence for event in events) / max(len(events), 1)
            ),
            "scene_fallback_backbone_active": 0.0,
            "scene_fallback_low_authority": 0.0,
            "scene_backbone_replaceable": 1.0,
            "scene_structural_primary_active": 1.0,
            "scene_structural_primary_segments": float(len(selected_segments)),
            "scene_structural_primary_units_used": float(len(used_unit_ids)),
            "scene_structural_primary_state_claims": float(len(states)),
            "scene_structural_primary_relation_claims": float(len(events)),
            "scene_structural_primary_goal_claims": float(len(goals)),
            "scene_structural_primary_attributed_claims": float(
                sum(1 for claim in claims if claim.speaker_entity_id)
            ),
            "scene_structural_primary_nonasserted_claims": float(
                sum(1 for claim in claims if str(claim.epistemic_status) != "asserted")
            ),
            "scene_structural_primary_evidence_refs": float(
                sum(len(state.evidence_refs) for state in states)
                + sum(len(event.evidence_refs) for event in events)
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
        events=tuple(events),
        goals=tuple(goals),
        claims=tuple(claims),
        mentions=mentions,
        discourse_relations=discourse_relations,
        temporal_markers=temporal_markers,
        explanations=explanations,
        coreference_links=tuple(),
        metadata=metadata,
    )

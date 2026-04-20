from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .scene_types import (
    SemanticDiscourseRelation,
    SemanticEntity,
    SemanticExplanation,
    SemanticMention,
    SemanticTemporalMarker,
)
from .types import GroundedTextDocument


_DISCOURSE_MARKERS: Tuple[Tuple[str, str, float], ...] = (
    ("because", "cause", 0.66),
    ("therefore", "cause", 0.64),
    ("however", "contrast", 0.64),
    ("but", "contrast", 0.60),
    ("if", "condition", 0.60),
    ("then", "sequence", 0.58),
    ("after", "sequence", 0.58),
    ("later", "sequence", 0.56),
    ("бо", "cause", 0.66),
    ("тому", "cause", 0.64),
    ("однак", "contrast", 0.64),
    ("але", "contrast", 0.60),
    ("якщо", "condition", 0.60),
    ("потім", "sequence", 0.58),
    ("після", "sequence", 0.58),
    ("через", "sequence", 0.56),
)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b", re.UNICODE)
_TEMPORAL_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("before", "before"),
    ("after", "after"),
    ("then", "sequence"),
    ("later", "sequence"),
    ("minute", "duration"),
    ("hour", "duration"),
    ("before", "before"),
    ("перед", "before"),
    ("після", "after"),
    ("потім", "sequence"),
    ("через", "duration"),
    ("хвилин", "duration"),
    ("годин", "duration"),
)


def _segment_text(document: GroundedTextDocument, segment_index: int) -> str:
    if segment_index < 0 or segment_index >= len(document.segments):
        return ""
    return str(document.segments[segment_index].normalized_text or document.segments[segment_index].text).casefold()


def build_semantic_context_objects(
    document: GroundedTextDocument,
    entities: Sequence[SemanticEntity],
) -> Tuple[
    Tuple[SemanticMention, ...],
    Tuple[SemanticDiscourseRelation, ...],
    Tuple[SemanticTemporalMarker, ...],
    Tuple[SemanticExplanation, ...],
]:
    mentions: List[SemanticMention] = []
    discourse_relations: List[SemanticDiscourseRelation] = []
    temporal_markers: List[SemanticTemporalMarker] = []
    explanations: List[SemanticExplanation] = []

    for segment in document.segments:
        seg_text = str(segment.normalized_text or segment.text).casefold()
        for entity in entities:
            surface = str(entity.canonical_name or "").strip().casefold()
            aliases = [str(alias).strip().casefold() for alias in entity.aliases if str(alias).strip()]
            candidates = tuple(dict.fromkeys([surface, *aliases]))
            matched = next((candidate for candidate in candidates if candidate and candidate in seg_text), None)
            if matched is None:
                continue
            mentions.append(
                SemanticMention(
                    mention_id=f"mention:{segment.index}:{entity.entity_id}",
                    entity_id=entity.entity_id,
                    surface_form=matched,
                    source_segment=int(segment.index),
                    source_span=segment.span,
                    confidence=max(0.55, float(entity.confidence) * 0.9),
                    status="supported" if entity.status == "supported" else "hint",
                )
            )

        for marker, relation_type, confidence in _DISCOURSE_MARKERS:
            if marker not in seg_text:
                continue
            source_segment = max(int(segment.index) - 1, 0)
            discourse_relations.append(
                SemanticDiscourseRelation(
                    relation_id=f"discourse:{segment.index}:{relation_type}:{marker}",
                    relation_type=relation_type,
                    source_segment=source_segment,
                    target_segment=int(segment.index),
                    marker=marker,
                    source_span=segment.span,
                    confidence=confidence,
                    status="supported" if relation_type in {"cause", "contrast"} else "hint",
                )
            )
            explanations.append(
                SemanticExplanation(
                    explanation_id=f"explanation:{segment.index}:{relation_type}:{marker}",
                    explanation_type=relation_type,
                    source_segment=source_segment,
                    target_segment=int(segment.index),
                    trigger=marker,
                    source_span=segment.span,
                    confidence=min(confidence + 0.04, 0.8),
                    status="proposal",
                )
            )

        for timestamp in _TIME_RE.findall(seg_text):
            temporal_markers.append(
                SemanticTemporalMarker(
                    marker_id=f"time:{segment.index}:{timestamp}",
                    marker_type="timestamp",
                    marker_value=timestamp,
                    source_segment=int(segment.index),
                    anchor_segment=max(int(segment.index) - 1, 0) if int(segment.index) > 0 else None,
                    source_span=segment.span,
                    confidence=0.72,
                    status="supported",
                )
            )
        for marker, marker_type in _TEMPORAL_MARKERS:
            if marker not in seg_text:
                continue
            temporal_markers.append(
                SemanticTemporalMarker(
                    marker_id=f"temporal:{segment.index}:{marker_type}:{marker}",
                    marker_type=marker_type,
                    marker_value=marker,
                    source_segment=int(segment.index),
                    anchor_segment=max(int(segment.index) - 1, 0) if int(segment.index) > 0 else None,
                    source_span=segment.span,
                    confidence=0.58 if marker_type == "sequence" else 0.56,
                    status="hint",
                )
            )
        if segment.counterexample:
            explanations.append(
                SemanticExplanation(
                    explanation_id=f"explanation:{segment.index}:counterexample",
                    explanation_type="counterexample",
                    source_segment=max(int(segment.index) - 1, 0),
                    target_segment=int(segment.index),
                    trigger="counterexample",
                    source_span=segment.span,
                    confidence=0.62,
                    status="proposal",
                )
            )

    return (
        tuple(mentions),
        tuple(discourse_relations),
        tuple(temporal_markers),
        tuple(explanations),
    )

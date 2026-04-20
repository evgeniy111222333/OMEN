from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .scene_types import SemanticSceneGraph


@dataclass(frozen=True)
class SceneContextGraphRecord:
    record_type: str
    record_id: str
    graph_key: str
    graph_text: str
    graph_terms: Tuple[str, ...] = field(default_factory=tuple)
    graph_family: str = "scene_context"
    confidence: float = 0.0
    source_segment: int = 0


def compile_scene_context_graph_records(
    scene: SemanticSceneGraph,
    *,
    max_records: int = 48,
) -> Tuple[Tuple[SceneContextGraphRecord, ...], Dict[str, float]]:
    records: List[SceneContextGraphRecord] = []

    for mention in scene.mentions:
        records.append(
            SceneContextGraphRecord(
                record_type="mention",
                record_id=mention.mention_id,
                graph_key=f"scene:mention:{mention.entity_id}:{mention.source_segment}",
                graph_text=f"mention {mention.surface_form}",
                graph_terms=(mention.entity_id, mention.surface_form),
                graph_family="scene_mention",
                confidence=float(mention.confidence),
                source_segment=int(mention.source_segment),
            )
        )
    for relation in scene.discourse_relations:
        records.append(
            SceneContextGraphRecord(
                record_type="discourse",
                record_id=relation.relation_id,
                graph_key=f"scene:discourse:{relation.relation_type}:{relation.source_segment}:{relation.target_segment}",
                graph_text=f"discourse {relation.relation_type} marker={relation.marker}",
                graph_terms=(
                    relation.relation_type,
                    relation.marker,
                    str(relation.source_segment),
                    str(relation.target_segment),
                ),
                graph_family="scene_discourse",
                confidence=float(relation.confidence),
                source_segment=int(relation.target_segment),
            )
        )
    for marker in scene.temporal_markers:
        anchor = marker.anchor_segment if marker.anchor_segment is not None else marker.source_segment
        records.append(
            SceneContextGraphRecord(
                record_type="temporal",
                record_id=marker.marker_id,
                graph_key=f"scene:temporal:{marker.marker_type}:{anchor}:{marker.source_segment}",
                graph_text=f"temporal {marker.marker_type}={marker.marker_value}",
                graph_terms=(marker.marker_type, marker.marker_value, str(anchor), str(marker.source_segment)),
                graph_family="scene_temporal",
                confidence=float(marker.confidence),
                source_segment=int(marker.source_segment),
            )
        )
    for explanation in scene.explanations:
        target = explanation.target_segment if explanation.target_segment is not None else explanation.source_segment
        records.append(
            SceneContextGraphRecord(
                record_type="explanation",
                record_id=explanation.explanation_id,
                graph_key=f"scene:explanation:{explanation.explanation_type}:{explanation.source_segment}:{target}",
                graph_text=f"explanation {explanation.explanation_type} trigger={explanation.trigger}",
                graph_terms=(
                    explanation.explanation_type,
                    explanation.trigger,
                    str(explanation.source_segment),
                    str(target),
                ),
                graph_family="scene_explanation",
                confidence=float(explanation.confidence),
                source_segment=int(target),
            )
        )
    for link in scene.coreference_links:
        records.append(
            SceneContextGraphRecord(
                record_type="coreference",
                record_id=link.link_id,
                graph_key=(
                    f"scene:coreference:{link.relation_type}:{link.source_entity_id}:{link.target_entity_id}:"
                    f"{link.source_segment}"
                ),
                graph_text=f"coreference {link.relation_type} {link.source_entity_id}->{link.target_entity_id}",
                graph_terms=(
                    link.relation_type,
                    link.source_entity_id,
                    link.target_entity_id,
                    str(link.source_segment),
                    str(link.target_segment),
                ),
                graph_family="scene_coreference",
                confidence=float(link.confidence),
                source_segment=int(link.source_segment),
            )
        )

    limited = tuple(records[: max(int(max_records), 0)])
    stats = {
        "scene_context_records": float(len(limited)),
        "scene_context_mention_records": float(sum(1 for record in limited if record.record_type == "mention")),
        "scene_context_discourse_records": float(sum(1 for record in limited if record.record_type == "discourse")),
        "scene_context_temporal_records": float(sum(1 for record in limited if record.record_type == "temporal")),
        "scene_context_explanation_records": float(sum(1 for record in limited if record.record_type == "explanation")),
        "scene_context_coreference_records": float(sum(1 for record in limited if record.record_type == "coreference")),
    }
    return limited, stats

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .interlingua_types import CanonicalInterlingua


@dataclass(frozen=True)
class GroundingGraphRecord:
    record_type: str
    record_id: str
    graph_key: str
    graph_text: str
    graph_terms: Tuple[str, ...] = field(default_factory=tuple)
    graph_family: str = "grounding"
    confidence: float = 0.0
    source_segment: int = 0


def compile_interlingua_graph_records(
    interlingua: CanonicalInterlingua,
    *,
    max_records: int = 64,
) -> Tuple[Tuple[GroundingGraphRecord, ...], Dict[str, float]]:
    records: List[GroundingGraphRecord] = []

    for entity in interlingua.entities:
        records.append(
            GroundingGraphRecord(
                record_type="entity",
                record_id=entity.entity_id,
                graph_key=f"interlingua:entity:{entity.canonical_key}",
                graph_text=f"entity {entity.canonical_name}",
                graph_terms=(entity.canonical_key,),
                graph_family="interlingua_entity",
                confidence=float(entity.confidence),
                source_segment=int(entity.source_segments[0]) if entity.source_segments else 0,
            )
        )
    for state in interlingua.states:
        records.append(
            GroundingGraphRecord(
                record_type="state",
                record_id=state.claim_id,
                graph_key=f"interlingua:state:{state.entity_key}:{state.value_key}",
                graph_text=f"state {state.entity_name}={state.value}",
                graph_terms=(state.entity_key, state.value_key),
                graph_family="interlingua_state",
                confidence=float(state.confidence),
                source_segment=int(state.source_segment),
            )
        )
    for relation in interlingua.relations:
        records.append(
            GroundingGraphRecord(
                record_type="relation",
                record_id=relation.claim_id,
                graph_key=(
                    f"interlingua:relation:{relation.subject_key}:{relation.predicate_key}:"
                    f"{relation.object_key}:{relation.polarity}"
                ),
                graph_text=(
                    f"relation {relation.subject_name} {relation.predicate} {relation.object_name} "
                    f"polarity={relation.polarity}"
                ),
                graph_terms=(relation.subject_key, relation.predicate_key, relation.object_key),
                graph_family="interlingua_relation",
                confidence=float(relation.confidence),
                source_segment=int(relation.source_segment),
            )
        )
    for goal in interlingua.goals:
        target_term = goal.target_key or goal.value_key
        records.append(
            GroundingGraphRecord(
                record_type="goal",
                record_id=goal.goal_id,
                graph_key=f"interlingua:goal:{goal.goal_key}:{target_term}",
                graph_text=f"goal {goal.goal_name} -> {goal.target_name or goal.goal_value}",
                graph_terms=tuple(term for term in (goal.goal_key, target_term) if term),
                graph_family="interlingua_goal",
                confidence=float(goal.confidence),
                source_segment=int(goal.source_segment),
            )
        )

    limited = tuple(records[: max(int(max_records), 0)])
    stats = {
        "interlingua_graph_records": float(len(limited)),
        "interlingua_graph_entity_records": float(sum(1 for record in limited if record.record_type == "entity")),
        "interlingua_graph_state_records": float(sum(1 for record in limited if record.record_type == "state")),
        "interlingua_graph_relation_records": float(sum(1 for record in limited if record.record_type == "relation")),
        "interlingua_graph_goal_records": float(sum(1 for record in limited if record.record_type == "goal")),
    }
    return limited, stats

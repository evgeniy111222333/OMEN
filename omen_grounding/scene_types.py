from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class SemanticEntity:
    entity_id: str
    canonical_name: str
    semantic_type: str = "entity"
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    source_segments: Tuple[int, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    status: str = "candidate"


@dataclass(frozen=True)
class SemanticState:
    state_id: str
    key_entity_id: str
    key_name: str
    value: str
    source_segment: int
    confidence: float = 0.55
    status: str = "hint"


@dataclass(frozen=True)
class SemanticEvent:
    event_id: str
    event_type: str
    subject_entity_id: Optional[str] = None
    object_entity_id: Optional[str] = None
    subject_name: Optional[str] = None
    object_name: Optional[str] = None
    source_segment: int = 0
    confidence: float = 0.6
    polarity: str = "positive"
    status: str = "hint"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticGoal:
    goal_id: str
    goal_name: str
    goal_value: str
    target_entity_id: Optional[str]
    source_segment: int
    confidence: float = 0.58
    status: str = "hint"


@dataclass(frozen=True)
class SemanticClaim:
    claim_id: str
    claim_kind: str
    source_segment: int
    confidence: float = 0.55
    status: str = "proposal"
    subject_entity_id: Optional[str] = None
    predicate: str = ""
    object_entity_id: Optional[str] = None
    object_value: Optional[str] = None
    event_id: Optional[str] = None
    goal_id: Optional[str] = None


@dataclass
class SemanticSceneGraph:
    language: str
    source_text: str
    entities: Tuple[SemanticEntity, ...] = field(default_factory=tuple)
    states: Tuple[SemanticState, ...] = field(default_factory=tuple)
    events: Tuple[SemanticEvent, ...] = field(default_factory=tuple)
    goals: Tuple[SemanticGoal, ...] = field(default_factory=tuple)
    claims: Tuple[SemanticClaim, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)

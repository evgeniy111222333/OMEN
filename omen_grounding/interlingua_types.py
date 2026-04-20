from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class CanonicalEntity:
    entity_id: str
    canonical_name: str
    canonical_key: str
    semantic_type: str = "entity"
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    source_segments: Tuple[int, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    status: str = "candidate"


@dataclass(frozen=True)
class CanonicalStateClaim:
    claim_id: str
    entity_id: str
    entity_key: str
    entity_name: str
    value: str
    value_key: str
    source_segment: int
    confidence: float = 0.55
    status: str = "proposal"


@dataclass(frozen=True)
class CanonicalRelationClaim:
    claim_id: str
    subject_entity_id: str
    subject_key: str
    subject_name: str
    predicate: str
    predicate_key: str
    object_entity_id: str
    object_key: str
    object_name: str
    source_segment: int
    confidence: float = 0.6
    polarity: str = "positive"
    status: str = "proposal"


@dataclass(frozen=True)
class CanonicalGoalClaim:
    goal_id: str
    goal_name: str
    goal_key: str
    goal_value: str
    value_key: str
    target_entity_id: Optional[str]
    target_key: Optional[str]
    target_name: Optional[str]
    source_segment: int
    confidence: float = 0.58
    status: str = "proposal"


@dataclass
class CanonicalInterlingua:
    language: str
    source_text: str
    entities: Tuple[CanonicalEntity, ...] = field(default_factory=tuple)
    states: Tuple[CanonicalStateClaim, ...] = field(default_factory=tuple)
    relations: Tuple[CanonicalRelationClaim, ...] = field(default_factory=tuple)
    goals: Tuple[CanonicalGoalClaim, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class GroundingSpan:
    start: int
    end: int
    text: str = ""


@dataclass(frozen=True)
class GroundedStateHint:
    key: str
    value: str
    confidence: float = 0.55
    source: str = "heuristic_fallback"
    status: str = "hint"
    span: Optional[GroundingSpan] = None


@dataclass(frozen=True)
class GroundedRelationHint:
    left: str
    relation: str
    right: str
    confidence: float = 0.60
    source: str = "heuristic_fallback"
    status: str = "hint"
    span: Optional[GroundingSpan] = None


@dataclass(frozen=True)
class GroundedGoalHint:
    goal_name: str
    goal_value: str
    confidence: float = 0.58
    source: str = "heuristic_fallback"
    status: str = "hint"
    span: Optional[GroundingSpan] = None


@dataclass(frozen=True)
class GroundedTextSegment:
    index: int
    text: str
    normalized_text: str
    tokens: Tuple[str, ...] = field(default_factory=tuple)
    states: Tuple[GroundedStateHint, ...] = field(default_factory=tuple)
    relations: Tuple[GroundedRelationHint, ...] = field(default_factory=tuple)
    goals: Tuple[GroundedGoalHint, ...] = field(default_factory=tuple)
    counterexample: bool = False


@dataclass
class GroundedTextDocument:
    language: str
    source_text: str
    segments: Tuple[GroundedTextSegment, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)

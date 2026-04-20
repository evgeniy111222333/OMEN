from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class GroundingSpan:
    start: int
    end: int
    text: str = ""


@dataclass(frozen=True)
class GroundingParserCandidate:
    parser_name: str
    confidence: float = 0.5
    source: str = "typed_perception"
    role: str = "candidate"


@dataclass(frozen=True)
class GroundingSourceProfile:
    language: str = "text"
    script: str = "unknown"
    domain: str = "text"
    modality: str = "unknown"
    subtype: str = "unknown"
    verification_path: str = "fallback_verification"
    confidence: float = 0.0
    ambiguity: float = 1.0
    profile: Dict[str, float] = field(default_factory=dict)
    script_profile: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, float] = field(default_factory=dict)
    parser_candidates: Tuple[GroundingParserCandidate, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GroundedStructuralUnit:
    unit_id: str
    unit_type: str
    text: str
    source_segment: int
    span: Optional[GroundingSpan] = None
    confidence: float = 0.55
    status: str = "supported"
    fields: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    references: Tuple[str, ...] = field(default_factory=tuple)


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
class GroundedEntityHint:
    name: str
    semantic_type: str = "entity"
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.56
    source: str = "heuristic_fallback"
    status: str = "hint"
    span: Optional[GroundingSpan] = None


@dataclass(frozen=True)
class GroundedEventHint:
    subject: str
    predicate: str
    object_name: Optional[str] = None
    agent: Optional[str] = None
    patient: Optional[str] = None
    modality: str = ""
    condition: str = ""
    explanation: str = ""
    temporal: str = ""
    confidence: float = 0.60
    source: str = "heuristic_fallback"
    status: str = "hint"
    span: Optional[GroundingSpan] = None


@dataclass(frozen=True)
class GroundedTextSegment:
    index: int
    text: str
    normalized_text: str
    span: Optional[GroundingSpan] = None
    routing: Optional[GroundingSourceProfile] = None
    tokens: Tuple[str, ...] = field(default_factory=tuple)
    structural_units: Tuple[GroundedStructuralUnit, ...] = field(default_factory=tuple)
    states: Tuple[GroundedStateHint, ...] = field(default_factory=tuple)
    relations: Tuple[GroundedRelationHint, ...] = field(default_factory=tuple)
    goals: Tuple[GroundedGoalHint, ...] = field(default_factory=tuple)
    entities: Tuple[GroundedEntityHint, ...] = field(default_factory=tuple)
    events: Tuple[GroundedEventHint, ...] = field(default_factory=tuple)
    counterexample: bool = False


@dataclass
class GroundedTextDocument:
    language: str
    source_text: str
    routing: Optional[GroundingSourceProfile] = None
    structural_units: Tuple[GroundedStructuralUnit, ...] = field(default_factory=tuple)
    segments: Tuple[GroundedTextSegment, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)

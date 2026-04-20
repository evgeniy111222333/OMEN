from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class GroundingSpan:
    start: int
    end: int
    text: str = ""
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    source_id: str = ""
    document_id: str = ""
    episode_id: str = ""

    @property
    def char_length(self) -> int:
        return max(0, int(self.end) - int(self.start))

    @property
    def byte_length(self) -> int:
        if self.byte_start is not None and self.byte_end is not None:
            return max(0, int(self.byte_end) - int(self.byte_start))
        return len(str(self.text).encode("utf-8"))


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


GROUNDING_RUNTIME_CONTRACT_VERSION = "grounding-runtime/v1"


@dataclass(frozen=True)
class GroundingDocumentSummary:
    routing: Optional[GroundingSourceProfile] = None
    segment_count: int = 0
    structural_unit_count: int = 0
    semantic_authority: float = 0.0
    multilingual: float = 0.0
    source_id: str = ""
    document_id: str = ""
    episode_id: str = ""
    char_coverage: float = 0.0
    byte_coverage: float = 0.0


@dataclass(frozen=True)
class GroundingRuntimeContract:
    schema_version: str = GROUNDING_RUNTIME_CONTRACT_VERSION
    source_profile: Optional[GroundingSourceProfile] = None
    document: GroundingDocumentSummary = field(default_factory=GroundingDocumentSummary)
    grounding_mode: str = "unknown"
    orchestrator_active: bool = False


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
    source_id: str = ""
    document_id: str = ""
    episode_id: str = ""
    routing: Optional[GroundingSourceProfile] = None
    structural_units: Tuple[GroundedStructuralUnit, ...] = field(default_factory=tuple)
    segments: Tuple[GroundedTextSegment, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)

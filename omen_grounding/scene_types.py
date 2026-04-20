from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .types import GroundingSpan


@dataclass(frozen=True)
class SemanticEntity:
    entity_id: str
    canonical_name: str
    semantic_type: str = "entity"
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    source_segments: Tuple[int, ...] = field(default_factory=tuple)
    source_spans: Tuple[GroundingSpan, ...] = field(default_factory=tuple)
    confidence: float = 0.5
    status: str = "candidate"


@dataclass(frozen=True)
class SemanticState:
    state_id: str
    key_entity_id: str
    key_name: str
    value: str
    source_segment: int
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.55
    status: str = "hint"
    evidence_refs: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SemanticEvent:
    event_id: str
    event_type: str
    subject_entity_id: Optional[str] = None
    object_entity_id: Optional[str] = None
    agent_entity_id: Optional[str] = None
    patient_entity_id: Optional[str] = None
    subject_name: Optional[str] = None
    object_name: Optional[str] = None
    modality: str = ""
    condition: str = ""
    explanation: str = ""
    temporal: str = ""
    source_segment: int = 0
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.6
    polarity: str = "positive"
    status: str = "hint"
    evidence_refs: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticGoal:
    goal_id: str
    goal_name: str
    goal_value: str
    target_entity_id: Optional[str]
    source_segment: int
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.58
    status: str = "hint"
    evidence_refs: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SemanticClaim:
    claim_id: str
    claim_kind: str
    source_segment: int
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.55
    status: str = "proposal"
    subject_entity_id: Optional[str] = None
    predicate: str = ""
    object_entity_id: Optional[str] = None
    object_value: Optional[str] = None
    proposition_id: Optional[str] = None
    event_id: Optional[str] = None
    goal_id: Optional[str] = None
    speaker_entity_id: Optional[str] = None
    speaker_name: Optional[str] = None
    epistemic_status: str = "asserted"
    claim_source: str = "document"
    evidence_refs: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SemanticMention:
    mention_id: str
    entity_id: str
    surface_form: str
    source_segment: int
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.5
    status: str = "hint"


@dataclass(frozen=True)
class SemanticDiscourseRelation:
    relation_id: str
    relation_type: str
    source_segment: int
    target_segment: int
    marker: str
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.55
    status: str = "hint"


@dataclass(frozen=True)
class SemanticTemporalMarker:
    marker_id: str
    marker_type: str
    marker_value: str
    source_segment: int
    anchor_segment: Optional[int] = None
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.52
    status: str = "hint"


@dataclass(frozen=True)
class SemanticExplanation:
    explanation_id: str
    explanation_type: str
    source_segment: int
    target_segment: Optional[int] = None
    trigger: str = ""
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.56
    status: str = "proposal"


@dataclass(frozen=True)
class SemanticCoreferenceLink:
    link_id: str
    source_entity_id: str
    target_entity_id: str
    source_segment: int
    target_segment: int
    relation_type: str = "alias"
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.58
    status: str = "hint"


@dataclass
class SemanticSceneGraph:
    language: str
    source_text: str
    entities: Tuple[SemanticEntity, ...] = field(default_factory=tuple)
    states: Tuple[SemanticState, ...] = field(default_factory=tuple)
    events: Tuple[SemanticEvent, ...] = field(default_factory=tuple)
    goals: Tuple[SemanticGoal, ...] = field(default_factory=tuple)
    claims: Tuple[SemanticClaim, ...] = field(default_factory=tuple)
    mentions: Tuple[SemanticMention, ...] = field(default_factory=tuple)
    discourse_relations: Tuple[SemanticDiscourseRelation, ...] = field(default_factory=tuple)
    temporal_markers: Tuple[SemanticTemporalMarker, ...] = field(default_factory=tuple)
    explanations: Tuple[SemanticExplanation, ...] = field(default_factory=tuple)
    coreference_links: Tuple[SemanticCoreferenceLink, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)

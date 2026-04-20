from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .interlingua_types import CanonicalInterlingua
from .types import GroundedTextDocument, GroundingSpan


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sorted_unique(values: Iterable[str], *, limit: int) -> Tuple[str, ...]:
    selected: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        selected.append(text)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _segment_span(document: GroundedTextDocument, segment_index: int) -> GroundingSpan | None:
    if segment_index < 0 or segment_index >= len(document.segments):
        return None
    return document.segments[segment_index].span


def _render_signature_token(token: str) -> str:
    text = str(token).strip().lower()
    if text.startswith("state:"):
        body = text.split(":", 1)[1]
        return body.replace("=", "_")
    if text.startswith("relation:"):
        body = text.split(":", 1)[1]
        return body.replace(":", "_")
    if text.startswith("goal:"):
        body = text.split(":", 1)[1]
        return body.replace(":", "_")
    return text.replace(":", "_").replace("=", "_")


def _concept_key(signature: Sequence[str]) -> str:
    parts = [_render_signature_token(token) for token in signature[:3]]
    key = "_".join(part for part in parts if part)
    return key[:96] if key else "pattern_cluster"


@dataclass(frozen=True)
class GroundingOntologyConcept:
    concept_id: str
    concept_key: str
    concept_name: str
    record_type: str = "ontology"
    world_status: str = "hypothetical"
    status: str = "proposal"
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    signature_terms: Tuple[str, ...] = field(default_factory=tuple)
    member_terms: Tuple[str, ...] = field(default_factory=tuple)
    member_segments: Tuple[int, ...] = field(default_factory=tuple)
    member_record_ids: Tuple[str, ...] = field(default_factory=tuple)
    source_spans: Tuple[GroundingSpan, ...] = field(default_factory=tuple)
    support: float = 0.0
    confidence: float = 0.0
    provenance: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def graph_key(self) -> str:
        return f"grounding_ontology:{self.world_status}:{self.concept_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [self.concept_key, *self.signature_terms, *self.member_terms[:6], self.world_status]
        return tuple(str(term) for term in terms if str(term).strip())

    @property
    def graph_family(self) -> str:
        return f"grounding_ontology:{self.world_status}"

    @property
    def graph_text(self) -> str:
        signature = ", ".join(self.signature_terms)
        members = ", ".join(self.member_terms[:6])
        return (
            f"ontology {self.world_status} {self.concept_name} "
            f"support={self.support:.2f} confidence={self.confidence:.2f} "
            f"signature=[{signature}] members=[{members}]"
        ).strip()


@dataclass
class GroundingOntologyGrowthResult:
    concepts: Tuple[GroundingOntologyConcept, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)


def _segment_signature_tokens(
    interlingua: CanonicalInterlingua,
    segment_index: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    signature_tokens: Set[str] = set()
    member_terms: Set[str] = set()
    record_ids: Set[str] = set()
    for state in interlingua.states:
        if int(state.source_segment) != segment_index:
            continue
        signature_tokens.add(f"state:{state.entity_key}={state.value_key}")
        record_ids.add(str(state.claim_id))
        if state.entity_key in {"user", "actor", "agent", "person", "name", "entity", "object", "node"}:
            member_terms.add(str(state.value_key))
    for relation in interlingua.relations:
        if int(relation.source_segment) != segment_index:
            continue
        signature_tokens.add(f"relation:{relation.predicate_key}:{relation.object_key}")
        record_ids.add(str(relation.claim_id))
        member_terms.add(str(relation.subject_key))
    for goal in interlingua.goals:
        if int(goal.source_segment) != segment_index:
            continue
        signature_tokens.add(f"goal:{goal.goal_key}:{goal.value_key}")
        record_ids.add(str(goal.goal_id))
        if goal.target_key:
            member_terms.add(str(goal.target_key))
    return signature_tokens, member_terms, record_ids


def build_grounding_ontology_growth(
    document: GroundedTextDocument,
    interlingua: CanonicalInterlingua,
    *,
    max_concepts: int = 8,
) -> GroundingOntologyGrowthResult:
    if not interlingua.entities:
        return GroundingOntologyGrowthResult(
            concepts=tuple(),
            metadata={
                "grounding_ontology_records": 0.0,
                "grounding_ontology_active_records": 0.0,
                "grounding_ontology_hypothetical_records": 0.0,
                "grounding_ontology_mean_confidence": 0.0,
                "grounding_ontology_support": 0.0,
            },
        )

    segment_signatures: Dict[int, Set[str]] = {}
    segment_members: Dict[int, Set[str]] = {}
    segment_record_ids: Dict[int, Set[str]] = {}
    for segment_index in range(len(document.segments)):
        signature_terms, member_terms, record_ids = _segment_signature_tokens(interlingua, segment_index)
        if not signature_terms:
            continue
        segment_signatures[segment_index] = signature_terms
        segment_members[segment_index] = member_terms
        segment_record_ids[segment_index] = record_ids

    candidate_map: Dict[Tuple[str, ...], Set[int]] = {}
    segment_indices = sorted(segment_signatures)
    for left_idx, left_segment in enumerate(segment_indices):
        left_terms = segment_signatures[left_segment]
        for right_segment in segment_indices[left_idx + 1 :]:
            signature = tuple(sorted(left_terms.intersection(segment_signatures[right_segment])))
            if len(signature) < 2:
                continue
            supporters = {
                segment_index
                for segment_index, terms in segment_signatures.items()
                if set(signature).issubset(terms)
            }
            if len(supporters) < 2:
                continue
            if len(supporters) == 2 and len(signature) < 3:
                continue
            candidate_map.setdefault(signature, set()).update(supporters)

    ranked_candidates = sorted(
        candidate_map.items(),
        key=lambda item: (-len(item[1]), -len(item[0]), item[0]),
    )

    concepts: List[GroundingOntologyConcept] = []
    used_support_sets: List[Tuple[int, ...]] = []
    used_signatures: List[Set[str]] = []
    total_segments = max(float(len(document.segments)), 1.0)
    for signature, supporters in ranked_candidates:
        support_segments = tuple(sorted(int(idx) for idx in supporters))
        support_set = set(support_segments)
        if tuple(support_segments) in used_support_sets:
            continue
        signature_set = set(signature)
        if any(signature_set.issubset(existing) and support_set == set(existing_support)
               for existing, existing_support in zip(used_signatures, used_support_sets)):
            continue
        member_terms = _sorted_unique(
            [
                term
                for segment_index in support_segments
                for term in segment_members.get(segment_index, ())
                if term not in signature_set
            ],
            limit=8,
        )
        member_record_ids = _sorted_unique(
            [
                record_id
                for segment_index in support_segments
                for record_id in segment_record_ids.get(segment_index, ())
            ],
            limit=24,
        )
        coverage = float(len(support_segments)) / total_segments
        support = _clip01((0.25 * coverage) + (0.18 * min(len(signature), 4)) + (0.10 * min(len(support_segments), 4)))
        confidence = _clip01(
            0.28
            + (0.12 * min(len(signature), 4))
            + (0.10 * min(len(support_segments), 4))
            + (0.15 if member_terms else 0.0)
            + (0.15 * coverage)
        )
        world_status = "active" if len(support_segments) >= 3 or confidence >= 0.76 else "hypothetical"
        status = "supported" if world_status == "active" else "proposal"
        concept_key = _concept_key(signature)
        spans = tuple(
            span
            for segment_index in support_segments
            for span in (_segment_span(document, segment_index),)
            if span is not None
        )
        concepts.append(
            GroundingOntologyConcept(
                concept_id=f"concept:{len(concepts)}",
                concept_key=concept_key,
                concept_name=concept_key,
                world_status=world_status,
                status=status,
                symbols=(concept_key, *member_terms[:3]),
                signature_terms=tuple(signature),
                member_terms=member_terms,
                member_segments=support_segments,
                member_record_ids=member_record_ids,
                source_spans=spans,
                support=support,
                confidence=confidence,
                provenance=(
                    f"support_segments:{','.join(str(idx) for idx in support_segments)}",
                    f"signature_size:{len(signature)}",
                ),
            )
        )
        used_support_sets.append(tuple(support_segments))
        used_signatures.append(signature_set)
        if len(concepts) >= max(max_concepts, 0):
            break

    total = float(len(concepts))
    active = sum(1 for concept in concepts if concept.world_status == "active")
    hypothetical = sum(1 for concept in concepts if concept.world_status == "hypothetical")
    mean_confidence = (
        sum(float(concept.confidence) for concept in concepts) / total
        if concepts else 0.0
    )
    mean_support = (
        sum(float(concept.support) for concept in concepts) / total
        if concepts else 0.0
    )
    metadata = {
        "grounding_ontology_records": total,
        "grounding_ontology_active_records": float(active),
        "grounding_ontology_hypothetical_records": float(hypothetical),
        "grounding_ontology_mean_confidence": mean_confidence,
        "grounding_ontology_support": mean_support,
    }
    return GroundingOntologyGrowthResult(
        concepts=tuple(concepts),
        metadata=metadata,
    )

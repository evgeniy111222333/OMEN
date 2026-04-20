from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

from .backbone import SemanticGroundingBackbone
from .memory_hints import grounding_memory_terms
from .pipeline import TextGroundingPipelineResult, ground_text_to_symbolic
from .types import GroundingSpan


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class GroundingOrchestratorResult:
    pipeline: TextGroundingPipelineResult
    segment_spans: Dict[int, GroundingSpan] = field(default_factory=dict)
    metadata: Dict[str, float] = field(default_factory=dict)


def grounding_memory_corroboration(
    candidate_records: Sequence[object],
    memory_records: Sequence[object],
) -> Dict[str, float]:
    memory_terms = {str(term).strip() for term in grounding_memory_terms(memory_records, limit=128) if str(term).strip()}
    if not candidate_records:
        return {
            "verification_memory_corroboration": 0.0,
            "verification_memory_corroborated_records": 0.0,
            "verification_memory_term_overlap": 0.0,
        }
    corroborated = 0.0
    overlap_terms: Set[str] = set()
    total_terms = 0.0
    for record in candidate_records:
        terms = {str(term).strip() for term in (getattr(record, "graph_terms", ()) or ()) if str(term).strip()}
        total_terms += float(len(terms))
        if terms.intersection(memory_terms):
            corroborated += 1.0
            overlap_terms.update(terms.intersection(memory_terms))
    total = float(len(candidate_records))
    overlap_ratio = min(float(len(overlap_terms)) / max(total_terms, 1.0), 1.0)
    corroboration = min((0.70 * (corroborated / max(total, 1.0))) + (0.30 * overlap_ratio), 1.0)
    return {
        "verification_memory_corroboration": float(corroboration),
        "verification_memory_corroborated_records": float(corroborated),
        "verification_memory_term_overlap": float(overlap_ratio),
    }


def _parser_agreement_score(pipeline: TextGroundingPipelineResult) -> Dict[str, float]:
    document_segments = tuple(getattr(pipeline.document, "segments", ()) or ())
    compiled_segments = tuple(getattr(pipeline.compiled, "segments", ()) or ())
    by_idx = {int(segment.index): segment for segment in compiled_segments}
    semantic_authority = float(
        getattr(pipeline.document, "metadata", {}).get("grounding_document_semantic_authority", 1.0)
    )
    semantic_document = semantic_authority > 0.0
    relation_scores = []
    state_scores = []
    goal_scores = []
    for document_segment in document_segments:
        compiled_segment = by_idx.get(int(document_segment.index))
        if compiled_segment is None:
            relation_scores.append(0.0)
            state_scores.append(0.0)
            goal_scores.append(0.0)
            continue
        if semantic_document or len(document_segment.relations) > 0:
            relation_scores.append(
                1.0 - (
                    abs(len(document_segment.relations) - len(compiled_segment.relations))
                    / max(len(document_segment.relations), len(compiled_segment.relations), 1)
                )
            )
        else:
            relation_scores.append(1.0)
        state_scores.append(
            1.0 - (
                abs(len(document_segment.states) - len(compiled_segment.states))
                / max(len(document_segment.states), len(compiled_segment.states), 1)
            )
        )
        if semantic_document or len(document_segment.goals) > 0:
            goal_scores.append(
                1.0 - (
                    abs(len(document_segment.goals) - len(compiled_segment.goals))
                    / max(len(document_segment.goals), len(compiled_segment.goals), 1)
                )
            )
        else:
            goal_scores.append(1.0)
    relation_agreement = sum(relation_scores) / max(len(relation_scores), 1)
    state_agreement = sum(state_scores) / max(len(state_scores), 1)
    goal_agreement = sum(goal_scores) / max(len(goal_scores), 1)
    parser_agreement = _clip01(
        (0.40 * relation_agreement) + (0.35 * state_agreement) + (0.25 * goal_agreement)
    )
    return {
        "grounding_parser_relation_agreement": float(relation_agreement),
        "grounding_parser_state_agreement": float(state_agreement),
        "grounding_parser_goal_agreement": float(goal_agreement),
        "grounding_parser_agreement": float(parser_agreement),
    }


def _span_traceability_score(pipeline: TextGroundingPipelineResult) -> Dict[str, float]:
    document_segments = tuple(getattr(pipeline.document, "segments", ()) or ())
    segment_spans = {
        int(segment.index): segment.span
        for segment in document_segments
        if getattr(segment, "span", None) is not None
    }
    span_segment_coverage = float(len(segment_spans)) / max(float(len(document_segments)), 1.0)
    traced_hypotheses = sum(
        1 for hypothesis in pipeline.compiled.hypotheses
        if getattr(hypothesis, "source_span", None) is not None
    )
    traced_world_state = sum(
        1 for record in pipeline.world_state.records
        if getattr(record, "source_span", None) is not None
    )
    hypothesis_coverage = float(traced_hypotheses) / max(float(len(pipeline.compiled.hypotheses)), 1.0)
    world_state_coverage = float(traced_world_state) / max(float(len(pipeline.world_state.records)), 1.0)
    traceability = _clip01(
        (0.45 * span_segment_coverage) + (0.30 * hypothesis_coverage) + (0.25 * world_state_coverage)
    )
    return {
        "grounding_span_segment_coverage": float(span_segment_coverage),
        "grounding_span_hypothesis_coverage": float(hypothesis_coverage),
        "grounding_span_world_state_coverage": float(world_state_coverage),
        "grounding_span_traceability": float(traceability),
    }


def run_grounding_orchestrator(
    text: str,
    *,
    language: str = "text",
    max_segments: int = 24,
    backbone: Optional[SemanticGroundingBackbone] = None,
    memory_records: Optional[Sequence[object]] = None,
) -> GroundingOrchestratorResult:
    pipeline = ground_text_to_symbolic(
        text,
        language=language,
        max_segments=max_segments,
        backbone=backbone,
        memory_records=memory_records,
    )
    segment_spans = {
        int(segment.index): segment.span
        for segment in tuple(getattr(pipeline.document, "segments", ()) or ())
        if getattr(segment, "span", None) is not None
    }
    parser_agreement = _parser_agreement_score(pipeline)
    span_scores = _span_traceability_score(pipeline)
    corroboration = grounding_memory_corroboration(
        tuple(pipeline.world_state.records)
        + tuple(pipeline.verification.records)
        + tuple(pipeline.verifier_stack.validation_records)
        + tuple(pipeline.verifier_stack.repair_actions)
        + tuple(pipeline.compiled.hypotheses),
        tuple(memory_records or ()),
    )
    metadata = {
        "grounding_orchestrated": 1.0,
        **parser_agreement,
        **span_scores,
        **corroboration,
        **dict(pipeline.verifier_stack.metadata),
    }
    return GroundingOrchestratorResult(
        pipeline=pipeline,
        segment_spans=segment_spans,
        metadata=metadata,
    )

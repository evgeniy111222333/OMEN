from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .interlingua import build_canonical_interlingua
from .interlingua_types import CanonicalClaimFrame, CanonicalInterlingua
from .scene_types import SemanticSceneGraph
from .types import GroundedTextDocument, GroundingSpan


@dataclass(frozen=True)
class CompiledSymbolicSegment:
    index: int
    text: str
    normalized_text: str
    source_span: Optional[GroundingSpan] = None
    tokens: Tuple[str, ...] = field(default_factory=tuple)
    states: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    relations: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)
    goals: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    event_frames: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    claim_frames: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    counterexample: bool = False


@dataclass(frozen=True)
class CompiledSymbolicHypothesis:
    hypothesis_id: str
    segment_index: int
    kind: str
    symbols: Tuple[str, ...] = field(default_factory=tuple)
    source_span: Optional[GroundingSpan] = None
    confidence: float = 0.5
    status: str = "proposal"
    deferred: bool = False
    conflict_tag: str = ""
    provenance: Tuple[str, ...] = field(default_factory=tuple)
    support_set: Tuple[str, ...] = field(default_factory=tuple)
    speaker_key: str = ""
    epistemic_status: str = "asserted"
    claim_source: str = "document"
    semantic_mode: str = "instance"
    quantifier_mode: str = "instance"

    @property
    def graph_key(self) -> str:
        return f"grounding_hypothesis:{self.hypothesis_id}"

    @property
    def graph_terms(self) -> Tuple[str, ...]:
        terms = [*self.symbols, *self.support_set]
        if self.speaker_key:
            terms.append(f"speaker:{self.speaker_key}")
        if self.epistemic_status:
            terms.append(f"epistemic:{self.epistemic_status}")
        if self.claim_source:
            terms.append(f"claim_source:{self.claim_source}")
        if self.semantic_mode:
            terms.append(f"semantic:{self.semantic_mode}")
        if self.quantifier_mode:
            terms.append(f"quantifier:{self.quantifier_mode}")
        return tuple(str(term) for term in terms if str(term).strip())

    @property
    def graph_family(self) -> str:
        return f"grounding_hypothesis:{self.kind}"

    @property
    def graph_text(self) -> str:
        parts = [self.kind, *self.symbols]
        if self.speaker_key:
            parts.append(f"speaker={self.speaker_key}")
        if self.epistemic_status:
            parts.append(f"epistemic={self.epistemic_status}")
        if self.semantic_mode:
            parts.append(f"semantic={self.semantic_mode}")
        if self.quantifier_mode:
            parts.append(f"quantifier={self.quantifier_mode}")
        if self.conflict_tag:
            parts.append(self.conflict_tag)
        return " | ".join(parts)


@dataclass
class SymbolicCompilationResult:
    language: str
    source_text: str
    segments: Tuple[CompiledSymbolicSegment, ...] = field(default_factory=tuple)
    hypotheses: Tuple[CompiledSymbolicHypothesis, ...] = field(default_factory=tuple)
    candidate_rules: Tuple[Any, ...] = field(default_factory=tuple)
    metadata: Dict[str, float] = field(default_factory=dict)


def compile_semantic_scene_graph(
    scene: SemanticSceneGraph,
    *,
    document: Optional[GroundedTextDocument] = None,
) -> SymbolicCompilationResult:
    interlingua = build_canonical_interlingua(scene)
    return compile_canonical_interlingua(interlingua, document=document)


def _stable_symbol_id(namespace: str, value: str, *, base: int) -> int:
    text = f"{namespace}:{value}".encode("utf-8")
    return int(base + (zlib.adler32(text) & 0x7FFFFFFF) % 100_000)


def compile_canonical_interlingua(
    interlingua: CanonicalInterlingua,
    *,
    document: Optional[GroundedTextDocument] = None,
) -> SymbolicCompilationResult:
    segment_count = max(
        [len(getattr(document, "segments", ()))]
        + [int(state.source_segment) + 1 for state in interlingua.states]
        + [int(relation.source_segment) + 1 for relation in interlingua.relations]
        + [int(goal.source_segment) + 1 for goal in interlingua.goals]
    )
    compiled: List[CompiledSymbolicSegment] = []
    hypotheses: List[CompiledSymbolicHypothesis] = []
    claim_frames_by_proposition: Dict[str, Tuple[CanonicalClaimFrame, ...]] = {}
    for claim in tuple(getattr(interlingua, "claims", ()) or ()):
        proposition_id = str(getattr(claim, "proposition_id", "") or "")
        if not proposition_id:
            continue
        claim_frames_by_proposition[proposition_id] = claim_frames_by_proposition.get(proposition_id, tuple()) + (claim,)

    def _clip_confidence(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _dominant_epistemic_status(claim_frames: Sequence[CanonicalClaimFrame]) -> str:
        if not claim_frames:
            return "asserted"
        precedence = {"questioned": 0, "hedged": 1, "cited": 2, "asserted": 3}
        ranked = sorted(
            (str(getattr(frame, "epistemic_status", "asserted") or "asserted") for frame in claim_frames),
            key=lambda value: precedence.get(value, 4),
        )
        return ranked[0] if ranked else "asserted"

    def _dominant_semantic_mode(claim_frames: Sequence[CanonicalClaimFrame]) -> str:
        if not claim_frames:
            return "instance"
        precedence = {"rule": 0, "obligation": 1, "generic": 2, "instance": 3}
        ranked = sorted(
            (str(getattr(frame, "semantic_mode", "instance") or "instance") for frame in claim_frames),
            key=lambda value: precedence.get(value, 4),
        )
        return ranked[0] if ranked else "instance"

    def _dominant_quantifier_mode(claim_frames: Sequence[CanonicalClaimFrame]) -> str:
        if not claim_frames:
            return "instance"
        precedence = {"generic_all": 0, "directive": 1, "instance": 2}
        ranked = sorted(
            (str(getattr(frame, "quantifier_mode", "instance") or "instance") for frame in claim_frames),
            key=lambda value: precedence.get(value, 3),
        )
        return ranked[0] if ranked else "instance"

    def _append_hypothesis(
        *,
        hypothesis_id: str,
        segment_index: int,
        kind: str,
        symbols: Sequence[str],
        source_span: Optional[GroundingSpan],
        confidence: float,
        status: str,
        conflict_tag: str = "",
        counterexample: bool = False,
        provenance: Sequence[str] = (),
        evidence_refs: Sequence[str] = (),
        claim_frames: Sequence[CanonicalClaimFrame] = (),
    ) -> None:
        confidence_value = _clip_confidence(confidence)
        conflict = conflict_tag or ("counterexample_context" if counterexample else "")
        dominant_epistemic_status = _dominant_epistemic_status(claim_frames)
        dominant_semantic_mode = _dominant_semantic_mode(claim_frames)
        dominant_quantifier_mode = _dominant_quantifier_mode(claim_frames)
        deferred = bool(
            counterexample
            or confidence_value < 0.57
            or dominant_epistemic_status in {"cited", "questioned", "hedged"}
        )
        evidence = tuple(str(item) for item in evidence_refs if str(item).strip())
        claim_support: List[str] = []
        for frame in claim_frames:
            claim_support.append(f"claim_frame:{frame.claim_id}")
            if getattr(frame, "speaker_key", None):
                claim_support.append(f"speaker:{frame.speaker_key}")
            if getattr(frame, "epistemic_status", None):
                claim_support.append(f"epistemic:{frame.epistemic_status}")
            if getattr(frame, "claim_source", None):
                claim_support.append(f"claim_source:{frame.claim_source}")
            if getattr(frame, "semantic_mode", None):
                claim_support.append(f"semantic:{frame.semantic_mode}")
            if getattr(frame, "quantifier_mode", None):
                claim_support.append(f"quantifier:{frame.quantifier_mode}")
            claim_support.extend(
                str(item) for item in getattr(frame, "evidence_refs", ()) if str(item).strip()
            )
        support_set = tuple(dict.fromkeys(claim_support))
        speaker_key = next(
            (str(frame.speaker_key) for frame in claim_frames if getattr(frame, "speaker_key", None)),
            "",
        )
        claim_source = next(
            (str(frame.claim_source) for frame in claim_frames if str(getattr(frame, "claim_source", "") or "")),
            "document",
        )
        hypotheses.append(
            CompiledSymbolicHypothesis(
                hypothesis_id=hypothesis_id,
                segment_index=segment_index,
                kind=kind,
                symbols=tuple(str(item) for item in symbols),
                source_span=source_span,
                confidence=confidence_value,
                status=str(status or "proposal"),
                deferred=deferred,
                conflict_tag=conflict,
                provenance=tuple(str(item) for item in provenance) + evidence,
                support_set=support_set,
                speaker_key=speaker_key,
                epistemic_status=dominant_epistemic_status,
                claim_source=claim_source,
                semantic_mode=dominant_semantic_mode,
                quantifier_mode=dominant_quantifier_mode,
            )
        )

    for idx in range(segment_count):
        raw_segment = document.segments[idx] if document is not None and idx < len(document.segments) else None
        counterexample = bool(False if raw_segment is None else raw_segment.counterexample)
        segment_states = tuple(
            state
            for state in interlingua.states
            if int(state.source_segment) == idx
        )
        segment_relations = tuple(
            relation
            for relation in interlingua.relations
            if int(relation.source_segment) == idx
        )
        segment_goals = tuple(
            goal
            for goal in interlingua.goals
            if int(goal.source_segment) == idx
        )
        states = tuple(
            (state.entity_name, state.value)
            for state in segment_states
        )
        relations = tuple(
            (relation.subject_name, relation.predicate, relation.object_name)
            for relation in segment_relations
        )
        event_frames = tuple(
            tuple(
                item
                for item in (
                    relation.subject_name,
                    relation.predicate,
                    relation.object_name,
                    *tuple(relation.relation_modifiers),
                )
                if item
            )
            for relation in segment_relations
        )
        goals = tuple(
            (goal.goal_name, goal.goal_value)
            for goal in segment_goals
        )
        segment_claim_frames = tuple(
            (
                str(claim.claim_kind),
                str(claim.epistemic_status),
                str(getattr(claim, "semantic_mode", "instance") or "instance"),
                str(getattr(claim, "quantifier_mode", "instance") or "instance"),
                str(claim.speaker_name or claim.claim_source or ""),
            )
            for claim in tuple(getattr(interlingua, "claims", ()) or ())
            if int(getattr(claim, "source_segment", -1)) == idx
        )
        for state in segment_states:
            attached_claim_frames = claim_frames_by_proposition.get(str(state.claim_id), tuple())
            _append_hypothesis(
                hypothesis_id=state.claim_id,
                segment_index=idx,
                kind="state",
                symbols=(state.entity_name, state.value),
                source_span=state.source_span,
                confidence=state.confidence,
                status=state.status,
                counterexample=counterexample,
                provenance=(f"segment:{idx}", f"state:{state.claim_id}"),
                evidence_refs=getattr(state, "evidence_refs", ()),
                claim_frames=attached_claim_frames,
            )
        for relation in segment_relations:
            relation_symbols = [relation.subject_name, relation.predicate, relation.object_name]
            relation_symbols.extend(str(item) for item in relation.relation_modifiers if str(item).strip())
            attached_claim_frames = claim_frames_by_proposition.get(str(relation.claim_id), tuple())
            _append_hypothesis(
                hypothesis_id=relation.claim_id,
                segment_index=idx,
                kind="relation",
                symbols=tuple(relation_symbols),
                source_span=relation.source_span,
                confidence=relation.confidence,
                status=relation.status,
                conflict_tag="negative_polarity" if relation.polarity != "positive" else "",
                counterexample=counterexample,
                provenance=(f"segment:{idx}", f"relation:{relation.claim_id}"),
                evidence_refs=getattr(relation, "evidence_refs", ()),
                claim_frames=attached_claim_frames,
            )
        for goal in segment_goals:
            goal_symbols = [goal.goal_name, goal.goal_value]
            if goal.target_name:
                goal_symbols.append(goal.target_name)
            attached_claim_frames = claim_frames_by_proposition.get(str(goal.goal_id), tuple())
            _append_hypothesis(
                hypothesis_id=goal.goal_id,
                segment_index=idx,
                kind="goal",
                symbols=tuple(goal_symbols),
                source_span=goal.source_span,
                confidence=goal.confidence,
                status=goal.status,
                counterexample=counterexample,
                provenance=(f"segment:{idx}", f"goal:{goal.goal_id}"),
                evidence_refs=getattr(goal, "evidence_refs", ()),
                claim_frames=attached_claim_frames,
            )
        compiled.append(
            CompiledSymbolicSegment(
                index=idx,
                text="" if raw_segment is None else raw_segment.text,
                normalized_text="" if raw_segment is None else raw_segment.normalized_text,
                source_span=None if raw_segment is None else raw_segment.span,
                tokens=tuple() if raw_segment is None else tuple(raw_segment.tokens),
                states=states,
                relations=relations,
                goals=goals,
                event_frames=event_frames,
                claim_frames=segment_claim_frames,
                counterexample=counterexample,
            )
        )

    def _compile_candidate_rule(hypothesis: CompiledSymbolicHypothesis) -> Optional[Any]:
        if hypothesis.kind != "relation":
            return None
        if hypothesis.semantic_mode not in {"generic", "rule", "obligation"}:
            return None
        if len(hypothesis.symbols) < 3:
            return None
        subject_name, predicate_name, object_name = (str(item) for item in hypothesis.symbols[:3])
        if not subject_name or not predicate_name or not object_name:
            return None
        from omen_prolog import HornAtom, HornClause, Var
        from omen_symbolic.creative_types import RuleCandidate

        subj_var = Var("X")
        obj_var = Var("Y")
        head_pred = _stable_symbol_id("ground_rel", predicate_name, base=2_400_000)
        subj_type_pred = _stable_symbol_id("ground_type", subject_name, base=2_500_000)
        obj_type_pred = _stable_symbol_id("ground_type", object_name, base=2_500_000)
        clause = HornClause(
            head=HornAtom(pred=head_pred, args=(subj_var, obj_var)),
            body=(
                HornAtom(pred=subj_type_pred, args=(subj_var,)),
                HornAtom(pred=obj_type_pred, args=(obj_var,)),
            ),
        )
        semantic_weight = {"rule": 1.0, "obligation": 0.92, "generic": 0.88}.get(hypothesis.semantic_mode, 0.80)
        epistemic_weight = {
            "asserted": 1.0,
            "cited": 0.78,
            "hedged": 0.56,
            "questioned": 0.38,
        }.get(hypothesis.epistemic_status, 0.72)
        score = _clip_confidence(hypothesis.confidence * semantic_weight * epistemic_weight)
        return RuleCandidate(
            clause=clause,
            source="grounding_rule_compiler",
            score=score,
            utility=score,
            metadata={
                "hypothesis_id": str(hypothesis.hypothesis_id),
                "semantic_mode": str(hypothesis.semantic_mode),
                "quantifier_mode": str(hypothesis.quantifier_mode),
                "epistemic_status": str(hypothesis.epistemic_status),
                "claim_source": str(hypothesis.claim_source),
                "subject_name": subject_name,
                "predicate_name": predicate_name,
                "object_name": object_name,
                "relation_modifiers": tuple(str(item) for item in hypothesis.symbols[3:] if str(item).strip()),
                "source_segment": int(hypothesis.segment_index),
                "support_set": tuple(str(item) for item in hypothesis.support_set),
                "provenance": tuple(str(item) for item in hypothesis.provenance),
            },
        )

    candidate_rule_index: Dict[Any, Any] = {}
    for hypothesis in hypotheses:
        candidate_rule = _compile_candidate_rule(hypothesis)
        if candidate_rule is None:
            continue
        clause = candidate_rule.clause
        existing = candidate_rule_index.get(clause)
        if existing is None or float(candidate_rule.score) > float(existing.score):
            candidate_rule_index[clause] = candidate_rule
    candidate_rules = tuple(
        sorted(candidate_rule_index.values(), key=lambda item: float(item.score), reverse=True)
    )

    metadata = dict(interlingua.metadata)
    deferred_hypotheses = sum(1 for hypothesis in hypotheses if hypothesis.deferred)
    conflict_hypotheses = sum(1 for hypothesis in hypotheses if hypothesis.conflict_tag)
    mean_confidence = (
        sum(float(hypothesis.confidence) for hypothesis in hypotheses) / float(len(hypotheses))
        if hypotheses else 0.0
    )
    metadata.update(
        {
            "compiled_segments": float(len(compiled)),
            "compiled_state_claims": float(sum(len(segment.states) for segment in compiled)),
            "compiled_relation_claims": float(sum(len(segment.relations) for segment in compiled)),
            "compiled_goal_claims": float(sum(len(segment.goals) for segment in compiled)),
            "compiled_event_frames": float(sum(len(segment.event_frames) for segment in compiled)),
            "compiled_modal_relations": float(
                sum(1 for hypothesis in hypotheses if hypothesis.kind == "relation" and any(symbol.startswith("modal:") for symbol in hypothesis.symbols[3:]))
            ),
            "compiled_conditioned_relations": float(
                sum(1 for hypothesis in hypotheses if hypothesis.kind == "relation" and any(symbol.startswith("if:") for symbol in hypothesis.symbols[3:]))
            ),
            "compiled_explained_relations": float(
                sum(1 for hypothesis in hypotheses if hypothesis.kind == "relation" and any(symbol.startswith("cause:") for symbol in hypothesis.symbols[3:]))
            ),
            "compiled_temporal_relations": float(
                sum(1 for hypothesis in hypotheses if hypothesis.kind == "relation" and any(symbol.startswith("time:") for symbol in hypothesis.symbols[3:]))
            ),
            "compiled_hypotheses": float(len(hypotheses)),
            "compiled_claim_frames": float(sum(len(segment.claim_frames) for segment in compiled)),
            "compiled_deferred_hypotheses": float(deferred_hypotheses),
            "compiled_conflict_hypotheses": float(conflict_hypotheses),
            "compiled_mean_confidence": float(mean_confidence),
            "compiled_attributed_hypotheses": float(sum(1 for hypothesis in hypotheses if hypothesis.speaker_key)),
            "compiled_nonasserted_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.epistemic_status != "asserted")
            ),
            "compiled_generic_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.semantic_mode == "generic")
            ),
            "compiled_rule_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.semantic_mode == "rule")
            ),
            "compiled_obligation_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.semantic_mode == "obligation")
            ),
            "compiled_quantified_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.quantifier_mode != "instance")
            ),
            "compiled_cited_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.epistemic_status == "cited")
            ),
            "compiled_questioned_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.epistemic_status == "questioned")
            ),
            "compiled_hedged_hypotheses": float(
                sum(1 for hypothesis in hypotheses if hypothesis.epistemic_status == "hedged")
            ),
            "compiled_structural_evidence_refs": float(
                sum(
                    1
                    for hypothesis in hypotheses
                    for item in hypothesis.provenance
                    if str(item).startswith("structural_unit:")
                )
            ),
            "compiled_candidate_rules": float(len(candidate_rules)),
            "compiled_grounding_rule_bridge_active": 1.0 if candidate_rules else 0.0,
        }
    )
    return SymbolicCompilationResult(
        language=interlingua.language,
        source_text=interlingua.source_text,
        segments=tuple(compiled),
        hypotheses=tuple(hypotheses),
        candidate_rules=candidate_rules,
        metadata=metadata,
    )

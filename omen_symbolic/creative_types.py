from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class RuleCandidate:
    clause: Any
    source: str
    score: float
    utility: float = 0.0
    aesthetic: float = 0.0
    structural_similarity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    candidates: List[RuleCandidate] = field(default_factory=list)
    novel_facts: Tuple[Any, ...] = field(default_factory=tuple)
    contradictions: Tuple[Tuple[Any, Any], ...] = field(default_factory=tuple)
    modified_rules: Tuple[Any, ...] = field(default_factory=tuple)
    surprise: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class IntrinsicGoal:
    goal: Any
    value: float
    kind: str
    provenance: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologyPredicateState:
    pred_id: int
    arity: int
    status: str
    gap_before: float = 0.0
    gap_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreativeCycleReport:
    abduction_candidates: List[RuleCandidate] = field(default_factory=list)
    analogy_candidates: List[RuleCandidate] = field(default_factory=list)
    metaphor_candidates: List[RuleCandidate] = field(default_factory=list)
    counterfactual_analogy_candidates: List[RuleCandidate] = field(default_factory=list)
    counterfactual_metaphor_candidates: List[RuleCandidate] = field(default_factory=list)
    counterfactual_candidates: List[RuleCandidate] = field(default_factory=list)
    counterfactual_novel_facts: Tuple[Any, ...] = field(default_factory=tuple)
    counterfactual_contradictions: Tuple[Tuple[Any, Any], ...] = field(default_factory=tuple)
    ontology_candidates: List[RuleCandidate] = field(default_factory=list)
    ontology_fixations: List[OntologyPredicateState] = field(default_factory=list)
    selected_rules: List[RuleCandidate] = field(default_factory=list)
    validated_support_facts: Tuple[Any, ...] = field(default_factory=tuple)
    coverage_gained_targets: Tuple[Any, ...] = field(default_factory=tuple)
    intrinsic_goal: Optional[IntrinsicGoal] = None
    predicate_embeddings: Dict[int, torch.Tensor] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

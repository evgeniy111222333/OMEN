from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from omen_symbolic.abduction_search import rule_template_signature
from omen_symbolic.creative_types import RuleCandidate
from omen_symbolic.hypergraph_gnn import HypergraphContrastiveLearner
from omen_symbolic.rule_graph import PredicateGraphView, build_predicate_graph_view


@dataclass
class AnalogyEngineState:
    graph_view: PredicateGraphView
    projector_loss: float = 0.0  # kept for API compatibility; now = InfoNCE loss


class AnalogyMetaphorEngine:
    """
    Analogy & Metaphor Engine (AME).

    Computes structural embeddings of predicates using:
        1. Hypergraph incidence matrix H (head/body asymmetric)
        2. Normalized hypergraph Laplacian spectral features
        3. HypergraphGNN with dual-channel head/body propagation
        4. InfoNCE contrastive loss on structural-role positive pairs

    This replaces the previous MLP projector over hand-crafted features,
    giving true hyperedge-aware structural similarity.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        tau_analogy: float = 0.82,
        tau_metaphor: Optional[float] = None,
        # GNN hyperparams
        hidden_dim: int = 64,
        n_gnn_layers: int = 2,
        spec_ratio: float = 0.5,
        temperature: float = 0.07,
        contrastive_steps: int = 4,
        contrastive_lr: float = 3e-3,
        dropout: float = 0.10,
        # Candidate generation
        max_pairs: int = 8,
        max_rules_per_pair: int = 4,
        max_metaphors_per_pair: int = 2,
    ):
        self.embedding_dim = int(max(8, embedding_dim))
        self.tau_analogy = float(max(0.0, min(1.0, tau_analogy)))
        if tau_metaphor is None:
            tau_metaphor = min(self.tau_analogy, max(0.45, self.tau_analogy * 0.75))
        self.tau_metaphor = float(max(0.0, min(self.tau_analogy, tau_metaphor)))
        self.max_pairs = max(int(max_pairs), 1)
        self.max_rules_per_pair = max(int(max_rules_per_pair), 1)
        self.max_metaphors_per_pair = max(int(max_metaphors_per_pair), 1)
        self._config = {
            "embedding_dim": self.embedding_dim,
            "tau_analogy": self.tau_analogy,
            "tau_metaphor": self.tau_metaphor,
            "hidden_dim": int(max(16, hidden_dim)),
            "n_gnn_layers": int(max(1, n_gnn_layers)),
            "spec_ratio": float(spec_ratio),
            "temperature": float(temperature),
            "contrastive_steps": int(max(0, contrastive_steps)),
            "contrastive_lr": float(contrastive_lr),
            "dropout": float(dropout),
            "max_pairs": self.max_pairs,
            "max_rules_per_pair": self.max_rules_per_pair,
            "max_metaphors_per_pair": self.max_metaphors_per_pair,
        }

        self._learner = HypergraphContrastiveLearner(
            embed_dim=self.embedding_dim,
            hidden_dim=self._config["hidden_dim"],
            n_gnn_layers=self._config["n_gnn_layers"],
            spec_ratio=self._config["spec_ratio"],
            temperature=self._config["temperature"],
            n_steps=self._config["contrastive_steps"],
            lr=self._config["contrastive_lr"],
            dropout=self._config["dropout"],
        )

        self.state = AnalogyEngineState(
            graph_view=build_predicate_graph_view([], embedding_dim=self.embedding_dim)
        )

    def clone(self) -> "AnalogyMetaphorEngine":
        return type(self)(**self._config)

    def export_state(self) -> Dict[str, Any]:
        return {
            "learner": self._learner.export_state(),
            "state": pickle.dumps(self.state),
        }

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        state = state or {}
        self._learner.load_state(state.get("learner"))
        state_blob = state.get("state")
        if isinstance(state_blob, bytes):
            self.state = pickle.loads(state_blob)
        elif state_blob is not None:
            self.state = state_blob

    @staticmethod
    def _shared_signature(signatures: Dict[int, Tuple[str, ...]], left: int, right: int) -> bool:
        return bool(set(signatures.get(left, ())) & set(signatures.get(right, ())))

    @staticmethod
    def _role_overlap(signatures: Dict[int, Tuple[str, ...]], left: int, right: int) -> float:
        left_roles = set(signatures.get(left, ()))
        right_roles = set(signatures.get(right, ()))
        if not left_roles and not right_roles:
            return 0.0
        return float(len(left_roles & right_roles) / max(len(left_roles | right_roles), 1))

    @staticmethod
    def _matches_predicate(candidate: RuleCandidate, predicate_ids: Sequence[int]) -> bool:
        pred_set = {int(pred) for pred in predicate_ids}
        if int(getattr(candidate.clause.head, "pred", -1)) in pred_set:
            return True
        return any(int(getattr(atom, "pred", -1)) in pred_set for atom in getattr(candidate.clause, "body", ()))

    def fit(self, rules: Sequence[Any]) -> AnalogyEngineState:
        """
        (Re-)compute hypergraph embeddings for all predicates in rules.
        Trains the GNN with InfoNCE for contrastive_steps iterations.
        """
        graph_view = build_predicate_graph_view(
            rules,
            embedding_dim=self.embedding_dim,
            hypergraph_learner=self._learner,
        )
        projector_loss = graph_view.hypergraph_contrastive_loss
        self.state = AnalogyEngineState(
            graph_view=graph_view,
            projector_loss=projector_loss,
        )
        return self.state

    def structural_similarity(self, left: int, right: int) -> float:
        view = self.state.graph_view
        if left not in view.embeddings or right not in view.embeddings:
            return 0.0
        score = F.cosine_similarity(
            view.embeddings[left].unsqueeze(0),
            view.embeddings[right].unsqueeze(0),
            dim=-1,
        )
        return float(score.item())

    @staticmethod
    def _replace_predicate(atom: Any, new_pred: int) -> Any:
        return type(atom)(pred=int(new_pred), args=tuple(atom.args))

    @staticmethod
    def _is_trivial_rule(rule: Any) -> bool:
        body = tuple(getattr(rule, "body", ()))
        return len(body) == 1 and body[0] == getattr(rule, "head", None)

    @staticmethod
    def _group_rules_by_predicate(rules: Sequence[Any]) -> Dict[int, List[Any]]:
        grouped_rules: Dict[int, List[Any]] = defaultdict(list)
        for rule in rules:
            preds = {int(rule.head.pred)} | {int(atom.pred) for atom in getattr(rule, "body", ())}
            for pred in preds:
                grouped_rules[pred].append(rule)
        return grouped_rules

    def _transfer_rule(
        self,
        rule: Any,
        source_pred: int,
        target_pred: int,
    ) -> Optional[Any]:
        changed = False
        head = rule.head
        if int(head.pred) == int(source_pred):
            head = self._replace_predicate(head, target_pred)
            changed = True
        body = []
        for atom in getattr(rule, "body", ()):
            if int(atom.pred) == int(source_pred):
                body.append(self._replace_predicate(atom, target_pred))
                changed = True
            else:
                body.append(atom)
        if not changed:
            return None
        return type(rule)(
            head=head,
            body=tuple(body),
            weight=float(getattr(rule, "weight", 1.0)),
            use_count=int(getattr(rule, "use_count", 0)),
        )

    @staticmethod
    def _metaphor_bridge_rule(
        sample_rule: Any,
        head_pred: int,
        head_arity: int,
        body_pred: int,
        body_arity: int,
    ) -> Optional[Any]:
        if body_arity < head_arity:
            return None
        from omen_prolog import Var

        atom_type = type(sample_rule.head)
        shared_vars = [Var(f"M{idx}") for idx in range(head_arity)]
        extra_body_vars = [Var(f"MB{idx}") for idx in range(body_arity - head_arity)]
        head = atom_type(pred=int(head_pred), args=tuple(shared_vars))
        body_atom = atom_type(pred=int(body_pred), args=tuple(shared_vars + extra_body_vars))
        return type(sample_rule)(
            head=head,
            body=(body_atom,),
            weight=float(getattr(sample_rule, "weight", 1.0)),
            use_count=int(getattr(sample_rule, "use_count", 0)),
        )

    def generate_candidates(
        self,
        rules: Sequence[Any],
        existing_hashes: Optional[Sequence[Any]] = None,
    ) -> List[RuleCandidate]:
        if not self.state.graph_view.predicate_ids:
            self.fit(rules)
        view = self.state.graph_view
        if not view.predicate_ids:
            return []
        existing_rules = {item for item in (existing_hashes or ()) if not isinstance(item, int)}
        ranked_pairs: List[Tuple[float, int, int]] = []
        for i, left in enumerate(view.predicate_ids):
            for j in range(i + 1, len(view.predicate_ids)):
                right = view.predicate_ids[j]
                if view.arities.get(left) != view.arities.get(right):
                    continue
                sim = self.structural_similarity(left, right)
                if sim < self.tau_analogy:
                    continue
                ranked_pairs.append((sim, left, right))
        ranked_pairs.sort(reverse=True)

        grouped_rules = self._group_rules_by_predicate(rules)

        best_by_template: Dict[Any, RuleCandidate] = {}
        for sim, source_pred, target_pred in ranked_pairs[: self.max_pairs]:
            pair_rules = grouped_rules.get(source_pred, [])[: self.max_rules_per_pair]
            shared_roles = set(view.signatures.get(source_pred, ())) & set(view.signatures.get(target_pred, ()))
            for src_rule in pair_rules:
                transferred = self._transfer_rule(src_rule, source_pred, target_pred)
                if (
                    transferred is None
                    or transferred in existing_rules
                    or self._is_trivial_rule(transferred)
                ):
                    continue
                template = rule_template_signature(transferred.head, tuple(transferred.body))
                candidate = RuleCandidate(
                    clause=transferred,
                    source="analogy",
                    score=float(sim),
                    structural_similarity=float(sim),
                    metadata={
                        "source_pred": float(source_pred),
                        "target_pred": float(target_pred),
                        "source_rule_key_text": repr(src_rule),
                        "source_rule_hash": float(hash(src_rule)),
                        "source_rule_hash_text": str(hash(src_rule)),
                        "embedding_source": view.embedding_source,
                        "shared_roles": ",".join(sorted(shared_roles)[:8]),
                        "shared_roles_count": float(len(shared_roles)),
                    },
                )
                current = best_by_template.get(template)
                if current is None or candidate.score > current.score:
                    best_by_template[template] = candidate
        return sorted(best_by_template.values(), key=lambda item: item.score, reverse=True)

    def generate_metaphor_candidates(
        self,
        rules: Sequence[Any],
        existing_hashes: Optional[Sequence[Any]] = None,
    ) -> List[RuleCandidate]:
        if not self.state.graph_view.predicate_ids:
            self.fit(rules)
        view = self.state.graph_view
        if not view.predicate_ids:
            return []

        existing_rules = {item for item in (existing_hashes or ()) if not isinstance(item, int)}
        grouped_rules = self._group_rules_by_predicate(rules)
        ranked_pairs: List[Tuple[float, int, int, float]] = []
        for i, left in enumerate(view.predicate_ids):
            for j in range(i + 1, len(view.predicate_ids)):
                right = view.predicate_ids[j]
                sim = self.structural_similarity(left, right)
                if sim < self.tau_metaphor:
                    continue
                role_overlap = self._role_overlap(view.signatures, left, right)
                if role_overlap <= 0.0 and sim < self.tau_analogy:
                    continue
                score = 0.75 * sim + 0.25 * role_overlap
                ranked_pairs.append((score, left, right, role_overlap))
        ranked_pairs.sort(reverse=True)

        best_by_template: Dict[Any, RuleCandidate] = {}
        for score, left, right, role_overlap in ranked_pairs[: self.max_pairs]:
            directions = (
                (right, view.arities.get(right, 0), left, view.arities.get(left, 0)),
                (left, view.arities.get(left, 0), right, view.arities.get(right, 0)),
            )
            emitted = 0
            for head_pred, head_arity, body_pred, body_arity in directions:
                if body_arity < head_arity:
                    continue
                sample_rules = grouped_rules.get(body_pred) or grouped_rules.get(head_pred)
                if not sample_rules:
                    continue
                bridge = self._metaphor_bridge_rule(
                    sample_rules[0],
                    head_pred=head_pred,
                    head_arity=head_arity,
                    body_pred=body_pred,
                    body_arity=body_arity,
                )
                if (
                    bridge is None
                    or bridge in existing_rules
                    or self._is_trivial_rule(bridge)
                ):
                    continue
                template = rule_template_signature(bridge.head, tuple(bridge.body))
                candidate = RuleCandidate(
                    clause=bridge,
                    source="metaphor",
                    score=float(score),
                    structural_similarity=float(score),
                    metadata={
                        "source_pred": float(body_pred),
                        "target_pred": float(head_pred),
                        "embedding_source": view.embedding_source,
                        "shared_roles_count": float(
                            len(set(view.signatures.get(left, ())) & set(view.signatures.get(right, ())))
                        ),
                        "role_overlap": float(role_overlap),
                        "projection_arity": float(min(head_arity, body_arity)),
                        "source_rule_key_text": repr(sample_rules[0]),
                        "source_rule_hash_text": str(hash(sample_rules[0])),
                        "source_rule_hash": float(hash(sample_rules[0])),
                        "metaphor_bridge": 1.0,
                    },
                )
                current = best_by_template.get(template)
                if current is None or candidate.score > current.score:
                    best_by_template[template] = candidate
                emitted += 1
                if emitted >= self.max_metaphors_per_pair:
                    break
        return sorted(best_by_template.values(), key=lambda item: item.score, reverse=True)

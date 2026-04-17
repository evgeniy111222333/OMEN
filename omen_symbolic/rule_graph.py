from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# Lazy import to avoid circular dependency
_hypergraph_gnn_module = None


def _get_hypergraph_gnn():
    global _hypergraph_gnn_module
    if _hypergraph_gnn_module is None:
        from omen_symbolic import hypergraph_gnn as _m
        _hypergraph_gnn_module = _m
    return _hypergraph_gnn_module


@dataclass
class PredicateGraphView:
    predicate_ids: Tuple[int, ...]
    arities: Dict[int, int]
    signatures: Dict[int, Tuple[str, ...]]
    feature_matrix: torch.Tensor
    similarity: torch.Tensor
    adjacency: torch.Tensor
    embeddings: Dict[int, torch.Tensor]
    # Hypergraph-specific extras (populated when hypergraph mode is active)
    spectral: Optional[torch.Tensor] = None          # raw spectral embedding
    hypergraph_contrastive_loss: float = 0.0
    embedding_source: str = "spectral_graph"         # "hypergraph_gnn" | "spectral_graph"


def _iter_rule_predicates(rule: Any) -> Iterable[int]:
    yield int(rule.head.pred)
    for atom in getattr(rule, "body", ()):
        yield int(atom.pred)


def predicate_arities(rules: Sequence[Any]) -> Dict[int, int]:
    arities: Dict[int, int] = {}
    for rule in rules:
        arities[int(rule.head.pred)] = int(rule.head.arity())
        for atom in getattr(rule, "body", ()):
            arities[int(atom.pred)] = int(atom.arity())
    return arities


def predicate_rule_signatures(rules: Sequence[Any]) -> Dict[int, Tuple[str, ...]]:
    signatures: Dict[int, Counter] = defaultdict(Counter)
    for rule in rules:
        body = tuple(getattr(rule, "body", ()))
        head_pred = int(rule.head.pred)
        body_preds = tuple(int(atom.pred) for atom in body)
        body_arity = tuple(int(atom.arity()) for atom in body)
        recur = int(head_pred in body_preds)
        symmetry = int(
            len(getattr(rule.head, "args", ())) >= 2
            and repr(rule.head.args[0]) == repr(rule.head.args[-1])
        )
        signatures[head_pred].update(
            {
                f"head:{head_pred}": 1,
                f"body_len:{len(body)}": 1,
                f"recursive:{recur}": 1,
                f"sym:{symmetry}": 1,
            }
        )
        if body_preds:
            signatures[head_pred].update({f"body_preds:{','.join(map(str, sorted(body_preds)))}": 1})
            signatures[head_pred].update({f"body_arity:{','.join(map(str, body_arity))}": 1})
        for atom in body:
            pred = int(atom.pred)
            signatures[pred].update(
                {
                    "in_body": 1,
                    f"supports:{head_pred}": 1,
                    f"body_slot:{len(body)}": 1,
                }
            )
    return {pred: tuple(sorted(counter.elements())) for pred, counter in signatures.items()}


def _rule_graph_features(
    rules: Sequence[Any],
    predicate_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    index = {pred: i for i, pred in enumerate(predicate_ids)}
    n_pred = len(predicate_ids)
    adjacency = torch.eye(n_pred, dtype=torch.float32)
    features = torch.zeros(n_pred, 10, dtype=torch.float32)
    head_body_sizes = defaultdict(list)

    for rule in rules:
        head_pred = int(rule.head.pred)
        head_idx = index[head_pred]
        body = tuple(getattr(rule, "body", ()))
        body_preds = [int(atom.pred) for atom in body]
        features[head_idx, 0] += 1.0
        features[head_idx, 4] += float(len(body_preds))
        if not body_preds:
            features[head_idx, 8] += 1.0
        head_body_sizes[head_pred].append(float(len(body_preds)))
        if head_pred in body_preds:
            features[head_idx, 2] += 1.0
        for body_pred in body_preds:
            body_idx = index[body_pred]
            features[body_idx, 1] += 1.0
            adjacency[body_idx, head_idx] += 1.0
            adjacency[head_idx, body_idx] += 0.5
        for left, right in combinations(body_preds, 2):
            li = index[left]
            ri = index[right]
            adjacency[li, ri] += 0.25
            adjacency[ri, li] += 0.25
        if len(body_preds) >= 2 and len(set(body_preds)) == 1:
            features[head_idx, 9] += 1.0

    indeg = adjacency.sum(dim=0)
    outdeg = adjacency.sum(dim=1)
    features[:, 3] = indeg
    features[:, 5] = outdeg
    features[:, 6] = (indeg <= 1.5).float()
    features[:, 7] = (outdeg <= 1.5).float()
    for pred, values in head_body_sizes.items():
        features[index[pred], 4] = sum(values) / max(len(values), 1)

    centered = features - features.mean(dim=0, keepdim=True)
    scaled = centered / centered.std(dim=0, keepdim=True).clamp_min(1e-4)
    return adjacency, scaled


def build_predicate_graph_view(
    rules: Sequence[Any],
    embedding_dim: int = 16,
    hypergraph_learner=None,
) -> PredicateGraphView:
    """
    Build a PredicateGraphView from a list of Horn rules.

    Parameters
    ----------
    rules          : list of Horn clauses
    embedding_dim  : dimensionality of predicate embeddings
    hypergraph_learner : optional HypergraphContrastiveLearner instance.
        When provided, embeddings come from the full hypergraph GNN pipeline
        (spectral hypergraph Laplacian + message passing + InfoNCE training).
        When None, falls back to simple normalized spectral graph embedding.
    """
    predicate_ids = tuple(sorted({pred for rule in rules for pred in _iter_rule_predicates(rule)}))
    arities = predicate_arities(rules)
    signatures = predicate_rule_signatures(rules)

    if not predicate_ids:
        empty = torch.zeros(0, 0)
        return PredicateGraphView(
            predicate_ids=tuple(),
            arities=arities,
            signatures=signatures,
            feature_matrix=empty,
            similarity=empty,
            adjacency=empty,
            embeddings={},
            spectral=None,
            hypergraph_contrastive_loss=0.0,
            embedding_source="empty",
        )

    adjacency, features = _rule_graph_features(rules, predicate_ids)

    # ------------------------------------------------------------------ #
    # Path A: Full hypergraph GNN pipeline                                #
    # ------------------------------------------------------------------ #
    if hypergraph_learner is not None and len(predicate_ids) >= 2:
        try:
            hg = _get_hypergraph_gnn()
            result = hypergraph_learner.embed(
                rules=rules,
                predicate_ids=predicate_ids,
                arity_map=arities,
            )
            normed = result.embeddings              # (n, embed_dim) — L2 normed
            similarity = result.similarity          # (n, n)

            # Pad/trim to embedding_dim if needed
            if normed.size(1) != embedding_dim:
                if normed.size(1) < embedding_dim:
                    normed = F.pad(normed, (0, embedding_dim - normed.size(1)))
                else:
                    normed = normed[:, :embedding_dim]
                normed = F.normalize(normed, dim=-1, eps=1e-6)
                similarity = normed @ normed.T

            embeddings = {pred: normed[idx] for idx, pred in enumerate(predicate_ids)}
            return PredicateGraphView(
                predicate_ids=predicate_ids,
                arities=arities,
                signatures=signatures,
                feature_matrix=features,
                similarity=similarity,
                adjacency=adjacency,
                embeddings=embeddings,
                spectral=result.spectral,
                hypergraph_contrastive_loss=result.contrastive_loss,
                embedding_source=getattr(result, "source", "hypergraph_gnn"),
            )
        except Exception:
            # Fallback to Path B on any error
            pass

    # ------------------------------------------------------------------ #
    # Path B: Hypergraph Laplacian spectral embedding (no GNN, no params)#
    # Significantly better than the old simple graph Laplacian because    #
    # it uses the true hyperedge structure.                               #
    # ------------------------------------------------------------------ #
    try:
        hg = _get_hypergraph_gnn()
        inc = hg.HypergraphIncidence.from_rules(rules, predicate_ids)
        lap = hg.build_hypergraph_laplacian(inc)
        n = len(predicate_ids)
        spec_k = min(max(1, embedding_dim // 2), max(1, n - 1))
        spectral = hg.hypergraph_spectral_emb(lap, k=spec_k)

        # Combine spectral with normalised hand features
        combined = torch.cat([spectral, features[:, :embedding_dim - spec_k]], dim=1)
        if combined.size(1) < embedding_dim:
            combined = F.pad(combined, (0, embedding_dim - combined.size(1)))
        elif combined.size(1) > embedding_dim:
            combined = combined[:, :embedding_dim]
        normed = F.normalize(combined, dim=-1, eps=1e-6)
        similarity = normed @ normed.T
        embeddings = {pred: normed[idx] for idx, pred in enumerate(predicate_ids)}
        return PredicateGraphView(
            predicate_ids=predicate_ids,
            arities=arities,
            signatures=signatures,
            feature_matrix=features,
            similarity=similarity,
            adjacency=adjacency,
            embeddings=embeddings,
            spectral=spectral,
            hypergraph_contrastive_loss=0.0,
            embedding_source="hypergraph_spectral",
        )
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # Path C: Legacy simple graph Laplacian (final fallback)             #
    # ------------------------------------------------------------------ #
    n_pred = len(predicate_ids)
    degree = adjacency.sum(dim=1).clamp_min(1e-4)
    degree_inv = torch.diag(torch.rsqrt(degree))
    laplacian = torch.eye(n_pred, dtype=torch.float32) - degree_inv @ adjacency @ degree_inv
    _evals, evecs = torch.linalg.eigh(laplacian)
    spec_dim = min(max(1, embedding_dim // 2), n_pred)
    spectral = evecs[:, :spec_dim]
    combined = torch.cat([spectral, features], dim=1)
    if combined.size(1) < embedding_dim:
        combined = F.pad(combined, (0, embedding_dim - combined.size(1)))
    elif combined.size(1) > embedding_dim:
        combined = combined[:, :embedding_dim]
    normed = F.normalize(combined, dim=-1, eps=1e-6)
    similarity = normed @ normed.T
    embeddings = {pred: normed[idx] for idx, pred in enumerate(predicate_ids)}
    return PredicateGraphView(
        predicate_ids=predicate_ids,
        arities=arities,
        signatures=signatures,
        feature_matrix=features,
        similarity=similarity,
        adjacency=adjacency,
        embeddings=embeddings,
        spectral=spectral,
        hypergraph_contrastive_loss=0.0,
        embedding_source="spectral_graph",
    )

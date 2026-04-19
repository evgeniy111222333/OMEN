"""
hypergraph_gnn.py — Hypergraph Spectral GNN for the Analogy & Metaphor Engine.

Mathematical foundations
------------------------
Classical GNNs operate on pairwise graphs and lose information when a rule
involves more than two predicates simultaneously.  A Horn clause

    head :- b1, b2, ..., bk

is a *hyperedge* connecting the set {head, b1, …, bk} — k+1 vertices.
Treating it as a clique of pairwise edges loses the "everyone participates
in the same rule" structure.

We follow the hypergraph Laplacian framework of Zhou et al. (2007):

    Δ = I − D_v^{−½} H W D_e^{−1} H^T D_v^{−½}

where
    H ∈ ℝ^{n_pred × n_rules}  — incidence matrix
    W = diag(w_e)              — per-hyperedge weight (rule confidence)
    D_v = diag(Hᵀ·1)          — vertex degrees
    D_e = diag(H^T·1)         — hyperedge degrees (sizes)

The bottom-k eigenvectors of Δ provide a *spectral* embedding that
preserves hypergraph geometry.

On top of the spectral initialisation we stack *Hypergraph Convolutional*
layers (HGNN, Feng et al. 2019):

    X^{ℓ+1} = σ( D_v^{−½} H W D_e^{−1} H^T D_v^{−½}  X^ℓ  Θ^ℓ )

Role-awareness
--------------
Head predicates and body predicates play asymmetric roles in a Horn rule.
We encode this with separate weight scalars (head_weight, body_weight) in H
and use a *dual-channel* incidence approach: H_head and H_body are kept
separate through the first HGNN layer, then merged.

Contrastive learning
--------------------
The GNN weights are trained with InfoNCE loss.  Two predicates are a
positive pair when they share a structural-role signature (both recursive,
both transitive bridges, both symmetric, etc.).  This is more principled
than the original binary BCE approach over co-occurrence.

Public API
----------
    HypergraphIncidence         — build H, W from rules
    build_hypergraph_laplacian  — compute Δ factors
    hypergraph_spectral_emb     — eigenvector embedding
    HypergraphConvLayer         — single HGNN conv layer
    HypergraphGNN               — stacked GNN with residuals + LayerNorm
    InfoNCELoss                 — temperature-scaled contrastive loss
    HypergraphContrastiveLearner— full train/embed pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Incidence matrix
# ---------------------------------------------------------------------------

_HEAD_WEIGHT: float = 1.5   # head predicate gets higher incidence weight
_BODY_WEIGHT: float = 1.0   # body predicate weight


@dataclass
class HypergraphIncidence:
    """
    Sparse-dense representation of the predicate-rule incidence matrix.

    Attributes
    ----------
    H : (n_pred, n_rules) float32 — incidence matrix (asymmetric head/body)
    H_head : (n_pred, n_rules)   — head-only incidence (1 per column)
    H_body : (n_pred, n_rules)   — body-only incidence
    W : (n_rules,)               — per-rule weights (rule.weight or 1.0)
    predicate_ids : tuple        — ordered predicate integer IDs
    pred_index : dict            — predicate_id → row index
    n_pred : int
    n_rules : int
    """

    H: torch.Tensor
    H_head: torch.Tensor
    H_body: torch.Tensor
    W: torch.Tensor
    predicate_ids: Tuple[int, ...]
    pred_index: Dict[int, int]
    n_pred: int
    n_rules: int

    @classmethod
    def from_rules(
        cls,
        rules: Sequence,
        predicate_ids: Tuple[int, ...],
    ) -> "HypergraphIncidence":
        """Build incidence matrix from a list of Horn clauses."""
        n_pred = len(predicate_ids)
        n_rules = len(rules)
        pred_index: Dict[int, int] = {p: i for i, p in enumerate(predicate_ids)}

        H = torch.zeros(n_pred, max(n_rules, 1), dtype=torch.float32)
        H_head = torch.zeros_like(H)
        H_body = torch.zeros_like(H)
        W = torch.ones(max(n_rules, 1), dtype=torch.float32)

        for e, rule in enumerate(rules):
            w = float(getattr(rule, "weight", 1.0))
            W[e] = max(w, 1e-6)

            head_pred = int(rule.head.pred)
            if head_pred in pred_index:
                i = pred_index[head_pred]
                H[i, e] += _HEAD_WEIGHT
                H_head[i, e] = _HEAD_WEIGHT

            for atom in getattr(rule, "body", ()):
                bp = int(atom.pred)
                if bp in pred_index:
                    i = pred_index[bp]
                    H[i, e] += _BODY_WEIGHT
                    H_body[i, e] += _BODY_WEIGHT

        # Trim to actual size if zero rules
        H = H[:, :n_rules] if n_rules > 0 else H
        H_head = H_head[:, :n_rules] if n_rules > 0 else H_head
        H_body = H_body[:, :n_rules] if n_rules > 0 else H_body
        W = W[:n_rules] if n_rules > 0 else W

        return cls(
            H=H,
            H_head=H_head,
            H_body=H_body,
            W=W,
            predicate_ids=predicate_ids,
            pred_index=pred_index,
            n_pred=n_pred,
            n_rules=n_rules,
        )


# ---------------------------------------------------------------------------
# 2. Hypergraph Laplacian
# ---------------------------------------------------------------------------

@dataclass
class HypergraphLaplacianData:
    """Pre-computed factors needed for HGNN convolution and spectral embedding."""
    # Θ = D_v^{-½} H W D_e^{-1} H^T D_v^{-½}
    theta: torch.Tensor          # (n_pred, n_pred) — smoothing operator
    Dv_invsqrt: torch.Tensor     # (n_pred,)
    De_inv: torch.Tensor         # (n_rules,)
    laplacian: torch.Tensor      # Δ = I − Θ
    # Separate head/body versions for dual-channel layer 0
    theta_head: torch.Tensor
    theta_body: torch.Tensor


def build_hypergraph_laplacian(inc: HypergraphIncidence) -> HypergraphLaplacianData:
    """
    Compute the normalized hypergraph Laplacian and its factors.

    Δ = I − D_v^{−½} H W D_e^{−1} H^T D_v^{−½}

    Numerical notes
    ---------------
    * Vertex degrees D_v are computed from the *weighted* incidence matrix.
    * We clamp both degree vectors to avoid division by zero.
    * All operations stay in float32.
    """
    H, W = inc.H, inc.W  # (n_pred, n_rules), (n_rules,)
    n = inc.n_pred

    # D_v: (n_pred,) — weighted vertex degree
    Dv = (H * W.unsqueeze(0)).sum(dim=1).clamp_min(1e-6)
    Dv_invsqrt = Dv.rsqrt()

    # D_e: (n_rules,) — hyperedge degree (number of predicates per rule)
    De = H.sum(dim=0).clamp_min(1e-6)
    De_inv = De.reciprocal()

    # Θ = D_v^{-½} H W D_e^{-1} H^T D_v^{-½}
    # Factor step-by-step to stay O(n_pred * n_rules)
    # Step 1: scale columns of H by W * D_e^{-1}
    col_scale = W * De_inv  # (n_rules,)
    HW = H * col_scale.unsqueeze(0)  # (n_pred, n_rules)
    # Step 2: multiply by H^T
    HWHT = HW @ H.T  # (n_pred, n_pred)
    # Step 3: symmetric normalise rows/cols by D_v^{-½}
    dv_outer = Dv_invsqrt.unsqueeze(1) * Dv_invsqrt.unsqueeze(0)  # (n, n)
    theta = dv_outer * HWHT

    # Same for head/body channels
    def _theta_channel(H_chan: torch.Tensor) -> torch.Tensor:
        col_scale_c = W * H_chan.sum(dim=0).clamp_min(1e-6).reciprocal()
        HW_c = H_chan * col_scale_c.unsqueeze(0)
        HWHT_c = HW_c @ H_chan.T
        return dv_outer * HWHT_c

    theta_head = _theta_channel(inc.H_head)
    theta_body = _theta_channel(inc.H_body)

    eye = torch.eye(n, dtype=torch.float32)
    laplacian = eye - theta

    return HypergraphLaplacianData(
        theta=theta,
        Dv_invsqrt=Dv_invsqrt,
        De_inv=De_inv,
        laplacian=laplacian,
        theta_head=theta_head,
        theta_body=theta_body,
    )


# ---------------------------------------------------------------------------
# 3. Spectral embedding
# ---------------------------------------------------------------------------

def hypergraph_spectral_emb(
    lap: HypergraphLaplacianData,
    k: int,
    skip_trivial: bool = True,
) -> torch.Tensor:
    """
    Compute spectral embeddings from the hypergraph Laplacian.

    Returns the k bottom eigenvectors (excluding the trivial zero eigenvalue
    when skip_trivial=True).  Shape: (n_pred, k).

    The bottom-k eigenvectors minimise the Rayleigh quotient x^T Δ x, meaning
    they capture the smoothest (most-connected) structure in the hypergraph.
    """
    n = lap.laplacian.size(0)
    k = min(k, n)

    try:
        # eigh: eigenvalues/vectors of real symmetric matrix, ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(lap.laplacian)
    except torch.linalg.LinAlgError:
        # Fallback: add small regularisation and retry
        reg = lap.laplacian + 1e-4 * torch.eye(n)
        eigenvalues, eigenvectors = torch.linalg.eigh(reg)

    # Skip the trivial eigenvector (eigenvalue ≈ 0, constant vector)
    start = 1 if skip_trivial and n > 1 else 0
    end = start + k
    if end > n:
        end = n
        start = max(0, end - k)

    spectral = eigenvectors[:, start:end]  # (n, k)

    # Pad if fewer than k eigenvectors available
    if spectral.size(1) < k:
        pad = torch.zeros(n, k - spectral.size(1))
        spectral = torch.cat([spectral, pad], dim=1)

    return spectral  # (n, k)


# ---------------------------------------------------------------------------
# 4. HGNN Convolution Layer
# ---------------------------------------------------------------------------

class HypergraphConvLayer(nn.Module):
    """
    Single hypergraph convolution layer.

    X^{ℓ+1} = σ( Θ  X^ℓ  W )

    where Θ = D_v^{−½} H W_e D_e^{−1} H^T D_v^{−½} is the pre-computed
    smoothing operator.

    Layer-0 uses dual-channel propagation: head and body channels are
    convolved separately then fused, preserving directional semantics.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dual_channel: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dual_channel = dual_channel
        self.dropout = dropout

        if dual_channel:
            # Separate projections for head and body channels
            self.W_head = nn.Linear(in_dim, out_dim // 2, bias=False)
            self.W_body = nn.Linear(in_dim, out_dim - out_dim // 2, bias=False)
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.W = nn.Linear(in_dim, out_dim, bias=True)

        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        lap: HypergraphLaplacianData,
    ) -> torch.Tensor:
        """
        x : (n_pred, in_dim)
        Returns: (n_pred, out_dim)
        """
        if self.dual_channel:
            # Head-channel propagation
            h_head = lap.theta_head @ x          # (n, in_dim)
            h_head = self.W_head(h_head)         # (n, out_dim//2)

            # Body-channel propagation
            h_body = lap.theta_body @ x          # (n, in_dim)
            h_body = self.W_body(h_body)         # (n, out_dim - out_dim//2)

            out = torch.cat([h_head, h_body], dim=-1) + self.bias
        else:
            smoothed = lap.theta @ x             # (n, in_dim)
            out = self.W(smoothed)

        out = self.norm(out)
        out = F.gelu(out)
        if self.training and self.dropout > 0:
            out = F.dropout(out, p=self.dropout)
        return out


# ---------------------------------------------------------------------------
# 5. Multi-layer HGNN with residuals
# ---------------------------------------------------------------------------

class HypergraphGNN(nn.Module):
    """
    Stacked Hypergraph GNN with residual connections and LayerNorm.

    Architecture
    ------------
    Layer 0 : dual-channel HGNN  (in_dim → hidden_dim)
    Layer 1…L-1: standard HGNN  (hidden_dim → hidden_dim)
    Output projection: (hidden_dim → out_dim)

    Residual connections are applied when the dimension matches.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = max(n_layers, 1)

        layers: List[HypergraphConvLayer] = []
        dims_in = [in_dim] + [hidden_dim] * (n_layers - 1)
        dims_out = [hidden_dim] * n_layers

        for idx in range(n_layers):
            layers.append(
                HypergraphConvLayer(
                    in_dim=dims_in[idx],
                    out_dim=dims_out[idx],
                    dual_channel=(idx == 0),  # only first layer uses dual
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Residual projection for dimension mismatch
        self.res_proj = (
            nn.Linear(in_dim, hidden_dim, bias=False)
            if in_dim != hidden_dim else nn.Identity()
        )

        # Output head
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=True),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        lap: HypergraphLaplacianData,
    ) -> torch.Tensor:
        """
        x   : (n_pred, in_dim)
        Returns: (n_pred, out_dim) — L2-normalised embeddings
        """
        h = x
        res = self.res_proj(x)

        for idx, layer in enumerate(self.layers):
            h_new = layer(h, lap)
            if idx == 0:
                h = h_new + res          # residual from input
            elif h_new.shape == h.shape:
                h = h_new + h            # layer-wise residual
            else:
                h = h_new

        out = self.output_proj(h)        # (n_pred, out_dim)
        return F.normalize(out, dim=-1, eps=1e-6)


# ---------------------------------------------------------------------------
# 6. InfoNCE Contrastive Loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    Normalised Temperature-scaled Cross-Entropy (NT-Xent / InfoNCE).

    For each anchor i, positive j, the loss is:
        L_i = -log( exp(sim(i,j)/τ) / Σ_{k≠i} exp(sim(i,k)/τ) )

    Positive pairs are defined by a boolean mask.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-4)

    def forward(
        self,
        embeddings: torch.Tensor,    # (n, d) — L2-normalised
        positive_mask: torch.Tensor, # (n, n) — bool, True=positive pair
    ) -> torch.Tensor:
        """Returns scalar loss."""
        n = embeddings.size(0)
        if n < 2:
            return embeddings.sum() * 0.0

        sim = embeddings @ embeddings.T  # (n, n) cosine sim (already normalised)
        sim = sim / self.temperature

        # Exclude self-similarity from denominator
        eye_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        sim = sim.masked_fill(eye_mask, float("-inf"))

        # For each anchor, log-sum-exp over all non-self pairs (denominator)
        log_denom = torch.logsumexp(sim, dim=1)  # (n,)

        # Numerator: mean over positive pairs per anchor
        pos_sim = sim.masked_fill(~positive_mask | eye_mask, float("-inf"))
        # At least one positive per anchor required; skip anchors with none
        has_positive = positive_mask.any(dim=1)  # (n,)
        if not has_positive.any():
            return embeddings.sum() * 0.0

        # Safe log-sum-exp over positives
        log_numer = torch.logsumexp(
            pos_sim.masked_fill(~has_positive.unsqueeze(1), float("-inf")),
            dim=1,
        )  # (n,)

        loss_per_anchor = -(log_numer - log_denom)
        loss = loss_per_anchor[has_positive].mean()
        return loss


# ---------------------------------------------------------------------------
# 7. Structural role signatures for positive-pair labelling
# ---------------------------------------------------------------------------

def _extract_structural_roles(rules: Sequence, predicate_ids: Tuple[int, ...]) -> Dict[int, frozenset]:
    """
    Returns a dict {pred_id: frozenset of role_tokens} where role_tokens
    encode the topological role of the predicate in the rule graph.

    Role tokens used:
        "recursive"   — appears in head and body of same rule
        "symmetric"   — appears in symmetric rules (both A→B and B→A)
        "source"      — only appears as head (never in body)
        "sink"        — only appears in body (never as head)
        "bridge"      — appears in body of rules with 2+ body atoms
        "transitive"  — participates in chain rules (head pred appears in body of another rule)
        "fact"        — appears with an empty body
    """
    roles: Dict[int, set] = {p: set() for p in predicate_ids}
    heads_set: set = set()
    bodies_set: set = set()

    def _is_commutative(rule) -> bool:
        body = tuple(getattr(rule, "body", ()))
        if len(body) != 1:
            return False
        atom = body[0]
        if int(getattr(rule.head, "pred", -1)) != int(getattr(atom, "pred", -2)):
            return False
        head_args = tuple(getattr(rule.head, "args", ()))
        body_args = tuple(getattr(atom, "args", ()))
        return len(head_args) >= 2 and body_args == tuple(reversed(head_args))

    def _is_associative(rule) -> bool:
        body = tuple(getattr(rule, "body", ()))
        if len(body) < 2:
            return False
        head_pred = int(getattr(rule.head, "pred", -1))
        body_preds = [int(getattr(atom, "pred", -2)) for atom in body]
        return all(pred == head_pred for pred in body_preds)

    for rule in rules:
        hp = int(rule.head.pred)
        body = tuple(getattr(rule, "body", ()))
        body_preds = [int(a.pred) for a in body]
        heads_set.add(hp)

        if hp in roles:
            if hp in body_preds:
                roles[hp].add("recursive")
            if not body_preds:
                roles[hp].add("fact")
            if _is_commutative(rule):
                roles[hp].add("commutative")
            if _is_associative(rule):
                roles[hp].add("associative")

        for bp in body_preds:
            bodies_set.add(bp)
            if bp in roles:
                if len(body_preds) >= 2:
                    roles[bp].add("bridge")
                if _is_associative(rule):
                    roles[bp].add("associative")
                if _is_commutative(rule):
                    roles[bp].add("commutative")

    # Symmetry: detect if p→q and q→p both exist via rule pairs
    head_to_bodies: Dict[int, List[Tuple[int, ...]]] = {}
    for rule in rules:
        hp = int(rule.head.pred)
        bps = tuple(int(a.pred) for a in getattr(rule, "body", ()))
        head_to_bodies.setdefault(hp, []).append(bps)

    for hp, body_list in head_to_bodies.items():
        for bps in body_list:
            if len(bps) == 1:
                bp = bps[0]
                # Check if bp→hp also exists
                if hp in (b[0] for b in head_to_bodies.get(bp, []) if len(b) == 1):
                    if hp in roles:
                        roles[hp].add("symmetric")
                    if bp in roles:
                        roles[bp].add("symmetric")

    # Transitivity: hp appears as body pred in some other rule
    for pred in predicate_ids:
        if pred in heads_set:
            if pred not in bodies_set:
                roles[pred].add("source")
        if pred not in heads_set:
            if pred in bodies_set:
                roles[pred].add("sink")
        # Check if hp is bridged (transitivity)
        if pred in bodies_set and pred in heads_set:
            roles[pred].add("transitive")

    return {p: frozenset(roles.get(p, set())) for p in predicate_ids}


def build_positive_mask(
    predicate_ids: Tuple[int, ...],
    rules: Sequence,
    arity_map: Dict[int, int],
) -> torch.Tensor:
    """
    Build a boolean positive-pair mask (n_pred × n_pred).

    Two predicates are a positive pair if any of:
    1. They share at least one structural role token.
    2. They have the same arity AND both appear in at least one common rule.
    3. They are co-located in the body of 2+ rules together.
    """
    n = len(predicate_ids)
    pred_index = {p: i for i, p in enumerate(predicate_ids)}

    roles = _extract_structural_roles(rules, predicate_ids)
    mask = torch.zeros(n, n, dtype=torch.bool)

    # Rule 1: shared role tokens
    role_list = [roles[p] for p in predicate_ids]
    for i in range(n):
        for j in range(i + 1, n):
            if role_list[i] & role_list[j]:
                mask[i, j] = True
                mask[j, i] = True

    # Rule 2: same arity + co-occur in a rule
    # Rule 3: co-located in body 2+ times
    cooccur: Dict[Tuple[int, int], int] = {}
    for rule in rules:
        hp = int(rule.head.pred)
        body_preds = [int(a.pred) for a in getattr(rule, "body", ())]
        all_preds = [hp] + body_preds
        for a in all_preds:
            for b in all_preds:
                if a == b:
                    continue
                if a in pred_index and b in pred_index:
                    key = (min(a, b), max(a, b))
                    cooccur[key] = cooccur.get(key, 0) + 1

    for (a, b), count in cooccur.items():
        i, j = pred_index[a], pred_index[b]
        same_arity = arity_map.get(a, -1) == arity_map.get(b, -1) and arity_map.get(a, -1) >= 0
        if count >= 2 or same_arity:
            mask[i, j] = True
            mask[j, i] = True

    return mask


# ---------------------------------------------------------------------------
# 8. Full contrastive learner
# ---------------------------------------------------------------------------

@dataclass
class HypergraphEmbedResult:
    """Output of HypergraphContrastiveLearner.embed()."""
    embeddings: torch.Tensor          # (n_pred, embed_dim) — L2-normalised
    spectral: torch.Tensor            # (n_pred, spec_dim)  — raw spectral
    similarity: torch.Tensor          # (n_pred, n_pred)    — cosine sim matrix
    contrastive_loss: float
    spectral_dim_used: int
    source: str = "hypergraph_gnn"


class HypergraphContrastiveLearner:
    """
    High-level pipeline:
        1. Build incidence matrix from rules.
        2. Compute hypergraph Laplacian.
        3. Spectral embedding (initialization for GNN input).
        4. Train GNN with InfoNCE on structural-role positive pairs.
        5. Return final embeddings.

    Parameters
    ----------
    embed_dim : int     — final output dimensionality
    hidden_dim : int    — HGNN hidden dimensionality
    n_gnn_layers : int  — number of HGNN layers
    spec_ratio : float  — fraction of embed_dim used for spectral init features
    temperature : float — InfoNCE temperature τ
    n_steps : int       — gradient steps per call to embed()
    lr : float          — Adam learning rate
    dropout : float     — dropout in HGNN
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        n_gnn_layers: int = 2,
        spec_ratio: float = 0.5,
        temperature: float = 0.07,
        n_steps: int = 4,
        lr: float = 3e-3,
        dropout: float = 0.10,
    ) -> None:
        self.embed_dim = max(int(embed_dim), 4)
        self.hidden_dim = max(int(hidden_dim), 8)
        self.n_gnn_layers = max(int(n_gnn_layers), 1)
        self.spec_ratio = float(max(0.1, min(0.9, spec_ratio)))
        self.temperature = float(temperature)
        self.n_steps = max(int(n_steps), 0)
        self.lr = float(lr)
        self.dropout = float(dropout)

        self._spec_dim = max(1, int(self.embed_dim * self.spec_ratio))
        self._feat_dim = self.embed_dim - self._spec_dim  # additional hand-crafted features

        # GNN is lazily initialised when we know in_dim
        self._gnn: Optional[HypergraphGNN] = None
        self._gnn_in_dim: int = 0
        self._optimizer: Optional[torch.optim.Adam] = None
        self._loss_fn = InfoNCELoss(temperature=self.temperature)

        # Cache
        self._last_rule_hash: Optional[Tuple[str, ...]] = None
        self._last_result: Optional[HypergraphEmbedResult] = None

    def _make_gnn(self, in_dim: int) -> HypergraphGNN:
        return HypergraphGNN(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.embed_dim,
            n_layers=self.n_gnn_layers,
            dropout=self.dropout,
        )

    def export_state(self) -> Dict[str, object]:
        return {
            "gnn_in_dim": int(self._gnn_in_dim),
            "gnn_state": None if self._gnn is None else self._gnn.state_dict(),
            "optimizer_state": (
                None if self._optimizer is None else self._optimizer.state_dict()
            ),
            "last_rule_hash": self._last_rule_hash,
            "last_result": (
                None if self._last_result is None else pickle.dumps(self._last_result)
            ),
        }

    def load_state(self, state: Optional[Dict[str, object]]) -> None:
        state = state or {}
        gnn_state = state.get("gnn_state")
        gnn_in_dim = int(state.get("gnn_in_dim", 0) or 0)
        if gnn_state is not None and gnn_in_dim > 0:
            self._gnn = self._make_gnn(gnn_in_dim)
            self._gnn_in_dim = gnn_in_dim
            self._optimizer = torch.optim.Adam(
                self._gnn.parameters(), lr=self.lr, weight_decay=1e-5
            )
            self._gnn.load_state_dict(gnn_state)
            optimizer_state = state.get("optimizer_state")
            if optimizer_state is not None and self._optimizer is not None:
                self._optimizer.load_state_dict(optimizer_state)
            self._gnn.eval()
        else:
            self._gnn = None
            self._gnn_in_dim = 0
            self._optimizer = None
        self._last_rule_hash = state.get("last_rule_hash")
        last_result = state.get("last_result")
        if isinstance(last_result, bytes):
            self._last_result = pickle.loads(last_result)
        else:
            self._last_result = last_result

    def _rule_set_hash(self, rules: Sequence) -> Tuple[str, ...]:
        return tuple(repr(rule) for rule in rules)

    @staticmethod
    def _hand_features(inc: HypergraphIncidence) -> torch.Tensor:
        """
        10-dimensional hand-crafted node features from the incidence matrix.
        These complement the spectral signal.
        """
        H = inc.H  # (n_pred, n_rules)
        W = inc.W  # (n_rules,)

        # Weighted vertex degree
        wdeg = (H * W.unsqueeze(0)).sum(dim=1)          # (n,)
        # Raw vertex degree (number of rules)
        deg = (H > 0).float().sum(dim=1)                # (n,)
        # Head-degree and body-degree
        head_deg = (inc.H_head > 0).float().sum(dim=1)  # (n,)
        body_deg = (inc.H_body > 0).float().sum(dim=1)  # (n,)
        # Head-to-total ratio
        total_deg = deg.clamp_min(1.0)
        head_ratio = head_deg / total_deg
        body_ratio = body_deg / total_deg
        # Co-occurrence density: how many rules share each pair
        # Approximated as D_v^T D_v diagonal normalised
        H_bin = (H > 0).float()
        cooccur_deg = (H_bin @ H_bin.T).diagonal()      # (n,)
        # Hyperedge weight stats per vertex
        rule_weights = H * W.unsqueeze(0)
        max_weight = rule_weights.max(dim=1).values      # (n,)
        mean_weight = rule_weights.sum(dim=1) / deg.clamp_min(1.0)

        feats = torch.stack([
            wdeg, deg, head_deg, body_deg,
            head_ratio, body_ratio, cooccur_deg,
            max_weight, mean_weight,
            (head_deg == 0).float(),   # is_sink (never a head)
        ], dim=1)  # (n, 10)

        # Z-score normalise per feature
        mu = feats.mean(dim=0, keepdim=True)
        sigma = feats.std(dim=0, keepdim=True).clamp_min(1e-4)
        return (feats - mu) / sigma

    def _deterministic_embeddings(
        self,
        spectral: torch.Tensor,
        hand_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic structural embedding fallback.

        When contrastive training is disabled or impossible, AME should still
        expose stable topological embeddings. We therefore return the
        hypergraph spectral features concatenated with a trimmed slice of the
        handcrafted structural statistics, avoiding any dependence on random
        GNN initialisation.
        """
        combined = spectral
        if combined.size(1) > self.embed_dim:
            combined = combined[:, : self.embed_dim]
        elif combined.size(1) < self.embed_dim:
            remain = self.embed_dim - combined.size(1)
            feat_part = hand_feats[:, :remain]
            combined = torch.cat([combined, feat_part], dim=1)
            if combined.size(1) < self.embed_dim:
                combined = F.pad(combined, (0, self.embed_dim - combined.size(1)))
        return F.normalize(combined, dim=-1, eps=1e-6)

    def embed(
        self,
        rules: Sequence,
        predicate_ids: Tuple[int, ...],
        arity_map: Dict[int, int],
        force_refit: bool = False,
    ) -> HypergraphEmbedResult:
        """
        Compute embeddings for all predicates in predicate_ids.

        If the rule set hasn't changed and force_refit=False, returns cached
        result.  Otherwise rebuilds the hypergraph and trains the GNN.
        """
        if not predicate_ids:
            empty = torch.zeros(0, self.embed_dim)
            return HypergraphEmbedResult(
                embeddings=empty,
                spectral=torch.zeros(0, self._spec_dim),
                similarity=torch.zeros(0, 0),
                contrastive_loss=0.0,
                spectral_dim_used=0,
                source="empty",
            )

        rule_hash = self._rule_set_hash(rules)
        if not force_refit and rule_hash == self._last_rule_hash and self._last_result is not None:
            return self._last_result

        # ---- Build incidence ----
        inc = HypergraphIncidence.from_rules(rules, predicate_ids)
        n = inc.n_pred

        # ---- Build Laplacian ----
        lap = build_hypergraph_laplacian(inc)

        # ---- Spectral embedding ----
        spec_k = min(self._spec_dim, max(1, n - 1))
        spectral = hypergraph_spectral_emb(lap, k=spec_k, skip_trivial=True)
        # Pad/trim to spec_dim
        if spectral.size(1) < self._spec_dim:
            spectral = F.pad(spectral, (0, self._spec_dim - spectral.size(1)))

        # ---- Hand-crafted features ----
        hand_feats = self._hand_features(inc)   # (n, 10)

        # ---- Combined input for GNN ----
        # spectral: (n, spec_dim), hand_feats: (n, 10)
        x = torch.cat([spectral, hand_feats], dim=1)  # (n, spec_dim + 10)
        in_dim = x.size(1)

        # ---- Positive-pair mask ----
        pos_mask = build_positive_mask(predicate_ids, rules, arity_map)

        # ---- Deterministic fallback when no contrastive training is available ----
        if self.n_steps <= 0 or n <= 1 or not pos_mask.any():
            final_emb = self._deterministic_embeddings(spectral, hand_feats)
            similarity = final_emb @ final_emb.T
            result = HypergraphEmbedResult(
                embeddings=final_emb,
                spectral=spectral,
                similarity=similarity,
                contrastive_loss=0.0,
                spectral_dim_used=spec_k,
                source="hypergraph_spectral",
            )
            self._last_rule_hash = rule_hash
            self._last_result = result
            return result

        # ---- Initialise / resize GNN if needed ----
        if self._gnn is None or self._gnn_in_dim != in_dim:
            self._gnn = self._make_gnn(in_dim)
            self._gnn_in_dim = in_dim
            self._optimizer = torch.optim.Adam(
                self._gnn.parameters(), lr=self.lr, weight_decay=1e-5
            )

        # ---- Training loop ----
        contrastive_loss = 0.0
        self._gnn.train()
        for _ in range(self.n_steps):
            emb = self._gnn(x, lap)
            loss = self._loss_fn(emb, pos_mask)
            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._gnn.parameters(), max_norm=1.0)
            self._optimizer.step()
            contrastive_loss = float(loss.item())

        # ---- Final embeddings ----
        self._gnn.eval()
        with torch.no_grad():
            final_emb = self._gnn(x, lap)   # (n, embed_dim) — normalised

        similarity = final_emb @ final_emb.T

        result = HypergraphEmbedResult(
            embeddings=final_emb,
            spectral=spectral,
            similarity=similarity,
            contrastive_loss=contrastive_loss,
            spectral_dim_used=spec_k,
            source="hypergraph_gnn",
        )
        self._last_rule_hash = rule_hash
        self._last_result = result
        return result

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _stable_bucket(text: str, buckets: int) -> int:
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    return int(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF) % buckets


def _term_signature(term: Any) -> str:
    if hasattr(term, "val"):
        return f"const:{int(term.val)}"
    if isinstance(term, int):
        return f"const:{int(term)}"
    if hasattr(term, "name"):
        return f"var:{term.name}"
    if hasattr(term, "func") and hasattr(term, "subterms"):
        inner = ",".join(_term_signature(subterm) for subterm in getattr(term, "subterms", ()))
        return f"compound:{getattr(term, 'func', 'fn')}({inner})"
    return repr(term)


def _atom_pred(atom: Any) -> int:
    return int(getattr(atom, "pred", 0))


def _atom_args(atom: Any) -> Tuple[Any, ...]:
    args = getattr(atom, "args", ())
    return tuple(args) if isinstance(args, (list, tuple)) else ()


def _atom_signature(atom: Any) -> str:
    arg_sig = ",".join(_term_signature(arg) for arg in _atom_args(atom))
    return f"p{_atom_pred(atom)}({arg_sig})"


def _atom_term_keys(atom: Any) -> Tuple[str, ...]:
    return tuple(_term_signature(arg) for arg in _atom_args(atom))


@dataclass(frozen=True)
class WorldGraphEdge:
    src: int
    dst: int
    relation: str
    weight: float = 1.0


@dataclass
class WorldGraphState:
    node_keys: Tuple[str, ...]
    node_types: Tuple[str, ...]
    edges: Tuple[WorldGraphEdge, ...]
    node_states: torch.Tensor
    pooled_state: torch.Tensor
    transition_states: Optional[torch.Tensor] = None
    transition_targets: Optional[torch.Tensor] = None
    metadata: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        summary = dict(self.metadata)
        summary.setdefault("nodes", float(len(self.node_keys)))
        summary.setdefault("edges", float(len(self.edges)))
        summary.setdefault(
            "trace_steps",
            float(0 if self.transition_targets is None else self.transition_targets.size(0)),
        )
        return summary


@dataclass
class WorldGraphBatch:
    graphs: Tuple[WorldGraphState, ...]
    pooled_states: torch.Tensor
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class CanonicalWorldState:
    graphs: Tuple[WorldGraphState, ...]
    neural_state: torch.Tensor
    graph_grounded_state: torch.Tensor
    graph_projection: torch.Tensor
    graph_readout_state: torch.Tensor
    grounded_state: torch.Tensor
    symbolic_state: Optional[torch.Tensor] = None
    memory_state: Optional[torch.Tensor] = None
    program_state: Optional[torch.Tensor] = None
    symbolic_facts: Tuple[Any, ...] = tuple()
    target_facts: Tuple[Any, ...] = tuple()
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def graph_state(self) -> WorldGraphState:
        if not self.graphs:
            raise ValueError("CanonicalWorldState has no graphs")
        return self.graphs[0]

    @property
    def z(self) -> Any:
        if len(self.graphs) == 1:
            return self.graphs[0]
        return self.graphs

    @property
    def dense_state(self) -> torch.Tensor:
        return self.grounded_state

    @property
    def batch_size(self) -> int:
        return int(self.grounded_state.size(0))

    def summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        if self.graphs:
            graph_summaries = [graph.summary() for graph in self.graphs]
            keys = sorted({key for item in graph_summaries for key in item})
            for key in keys:
                summary[key] = float(
                    sum(float(item.get(key, 0.0)) for item in graph_summaries)
                    / max(len(graph_summaries), 1)
                )
        summary.update(self.metadata)
        summary.setdefault("graphs", float(len(self.graphs)))
        summary.setdefault("batch_size", float(self.batch_size))
        summary.setdefault(
            "graph_grounded_state_norm",
            float(self.graph_grounded_state.norm(dim=-1).mean().item())
            if self.graph_grounded_state.numel() > 0 else 0.0,
        )
        summary.setdefault(
            "graph_projection_norm",
            float(self.graph_projection.norm(dim=-1).mean().item())
            if self.graph_projection.numel() > 0 else 0.0,
        )
        summary.setdefault(
            "graph_readout_norm",
            float(self.graph_readout_state.norm(dim=-1).mean().item())
            if self.graph_readout_state.numel() > 0 else 0.0,
        )
        summary.setdefault(
            "dense_state_norm",
            float(self.grounded_state.norm(dim=-1).mean().item())
            if self.grounded_state.numel() > 0 else 0.0,
        )
        summary.setdefault("has_program_state", float(self.program_state is not None))
        summary.setdefault("symbolic_facts", float(len(self.symbolic_facts)))
        summary.setdefault("target_facts", float(len(self.target_facts)))
        return summary


class WorldGraphEncoder(nn.Module):
    _REL_TO_ID: Dict[str, int] = {
        "shared_term": 0,
        "same_pred": 1,
        "trace_step": 2,
        "counterfactual": 3,
        "saliency": 4,
        "cooccurs": 5,
    }

    def __init__(
        self,
        d_latent: int,
        pred_buckets: int = 4096,
        term_buckets: int = 8192,
        max_arity: int = 8,
        max_nodes: int = 128,
        max_edges: int = 512,
        n_layers: int = 2,
        max_transitions: int = 16,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.pred_buckets = pred_buckets
        self.term_buckets = term_buckets
        self.max_arity = max_arity
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_transitions = max_transitions
        self.empty_state = nn.Parameter(torch.zeros(d_latent))
        self.empty_term = nn.Parameter(torch.zeros(d_latent))
        self.pred_emb = nn.Embedding(pred_buckets, d_latent)
        self.term_emb = nn.Embedding(term_buckets, d_latent)
        self.arity_emb = nn.Embedding(max_arity + 1, d_latent)
        self.rel_emb = nn.Embedding(len(self._REL_TO_ID), d_latent)
        self.atom_norm = nn.LayerNorm(d_latent)
        self.msg_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_latent * 2, d_latent),
                    nn.GELU(),
                    nn.Linear(d_latent, d_latent),
                )
                for _ in range(max(n_layers, 1))
            ]
        )
        self.gate_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_latent * 3, d_latent),
                    nn.GELU(),
                    nn.Linear(d_latent, 1),
                )
                for _ in range(max(n_layers, 1))
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(max(n_layers, 1))])
        self.pool = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

    def _encode_atom(self, atom: Any, device: torch.device) -> torch.Tensor:
        pred_bucket = _stable_bucket(f"pred:{_atom_pred(atom)}", self.pred_buckets)
        args = _atom_args(atom)
        arity_bucket = min(len(args), self.max_arity)
        pred_vec = self.pred_emb(torch.tensor(pred_bucket, device=device))
        arity_vec = self.arity_emb(torch.tensor(arity_bucket, device=device))
        if args:
            arg_buckets = [
                _stable_bucket(_term_signature(arg), self.term_buckets)
                for arg in args
            ]
            arg_vec = self.term_emb(torch.tensor(arg_buckets, device=device)).mean(dim=0)
        else:
            arg_vec = self.empty_term
        return self.atom_norm(pred_vec + arity_vec + arg_vec)

    def _dedupe_records(
        self,
        fact_records: Sequence[Tuple[str, Any]],
    ) -> List[Tuple[str, Any]]:
        unique: List[Tuple[str, Any]] = []
        seen: set[str] = set()
        for node_type, atom in fact_records:
            signature = _atom_signature(atom)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append((node_type, atom))
            if len(unique) >= self.max_nodes:
                break
        return unique

    def _pairwise_edges(
        self,
        node_types: Sequence[str],
        atoms: Sequence[Any],
    ) -> List[WorldGraphEdge]:
        edges: List[WorldGraphEdge] = []
        term_keys = [_atom_term_keys(atom) for atom in atoms]
        preds = [_atom_pred(atom) for atom in atoms]
        for left in range(len(atoms)):
            right_limit = min(len(atoms), left + 8)
            for right in range(left + 1, right_limit):
                relation: Optional[str] = None
                if set(term_keys[left]) & set(term_keys[right]):
                    relation = "shared_term"
                elif preds[left] == preds[right]:
                    relation = "same_pred"
                elif right == left + 1:
                    relation = "cooccurs"
                if relation is None:
                    continue
                if "saliency" in (node_types[left], node_types[right]):
                    relation = "saliency"
                edges.append(WorldGraphEdge(src=left, dst=right, relation=relation))
                edges.append(WorldGraphEdge(src=right, dst=left, relation=relation))
                if len(edges) >= self.max_edges:
                    return edges
        return edges

    def _transition_edges(
        self,
        node_index: Dict[str, int],
        transitions: Iterable[Any],
        relation: str,
    ) -> List[WorldGraphEdge]:
        edges: List[WorldGraphEdge] = []
        for transition in transitions:
            before_facts = getattr(transition, "before_facts", ())
            after_facts = getattr(transition, "after_facts", ())
            before_nodes = [node_index[_atom_signature(atom)] for atom in before_facts if _atom_signature(atom) in node_index]
            after_nodes = [node_index[_atom_signature(atom)] for atom in after_facts if _atom_signature(atom) in node_index]
            for src in before_nodes[:8]:
                for dst in after_nodes[:8]:
                    edges.append(WorldGraphEdge(src=src, dst=dst, relation=relation))
                    if len(edges) >= self.max_edges:
                        return edges
        return edges

    def _message_pass(
        self,
        node_states: torch.Tensor,
        edges: Sequence[WorldGraphEdge],
    ) -> torch.Tensor:
        if not edges:
            return node_states
        device = node_states.device
        src_idx = torch.tensor([edge.src for edge in edges], device=device, dtype=torch.long)
        dst_idx = torch.tensor([edge.dst for edge in edges], device=device, dtype=torch.long)
        rel_idx = torch.tensor(
            [self._REL_TO_ID[edge.relation] for edge in edges],
            device=device,
            dtype=torch.long,
        )
        rel_states = self.rel_emb(rel_idx)
        h = node_states
        for msg_layer, gate_layer, norm in zip(self.msg_layers, self.gate_layers, self.norm_layers):
            src_states = h[src_idx]
            dst_states = h[dst_idx]
            msg_in = torch.cat([src_states, rel_states], dim=-1)
            msg = msg_layer(msg_in)
            gate = torch.sigmoid(gate_layer(torch.cat([dst_states, msg, rel_states], dim=-1)))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst_idx, gate * msg)
            denom = torch.zeros(h.size(0), 1, device=device, dtype=h.dtype)
            denom.index_add_(0, dst_idx, gate)
            h = norm(h + agg / denom.clamp(min=1.0))
        return h

    def _encode_fact_records(
        self,
        fact_records: Sequence[Tuple[str, Any]],
        trace_bundle: Optional[Any],
        device: torch.device,
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[WorldGraphEdge, ...], torch.Tensor, torch.Tensor]:
        deduped = self._dedupe_records(fact_records)
        if not deduped:
            empty = self.empty_state.view(1, -1)
            return tuple(), tuple(), tuple(), empty, self.empty_state

        node_types = tuple(node_type for node_type, _ in deduped)
        atoms = [atom for _, atom in deduped]
        node_keys = tuple(_atom_signature(atom) for atom in atoms)
        node_index = {key: idx for idx, key in enumerate(node_keys)}
        node_states = torch.stack([self._encode_atom(atom, device) for atom in atoms], dim=0)
        edges = self._pairwise_edges(node_types, atoms)
        if trace_bundle is not None:
            transition_edges = self._transition_edges(
                node_index,
                getattr(trace_bundle, "transitions", ())[: self.max_transitions],
                relation="trace_step",
            )
            counter_edges = self._transition_edges(
                node_index,
                getattr(trace_bundle, "counterexamples", ())[: self.max_transitions],
                relation="counterfactual",
            )
            remaining = max(self.max_edges - len(edges), 0)
            edges.extend(transition_edges[:remaining])
            remaining = max(self.max_edges - len(edges), 0)
            edges.extend(counter_edges[:remaining])

        edges = edges[: self.max_edges]
        node_states = self._message_pass(node_states, edges)
        pooled = self.pool(torch.cat([node_states.mean(dim=0), node_states.max(dim=0).values], dim=-1))
        return node_keys, node_types, tuple(edges), node_states, pooled

    def _encode_transition_sets(
        self,
        transitions: Sequence[Any],
        device: torch.device,
        prefix: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not transitions:
            return None, None
        state_vectors: List[torch.Tensor] = []
        target_vectors: List[torch.Tensor] = []
        for transition in transitions[: self.max_transitions]:
            before_records = [(f"{prefix}_before", atom) for atom in getattr(transition, "before_facts", ())]
            after_records = [(f"{prefix}_after", atom) for atom in getattr(transition, "after_facts", ())]
            _, _, _, _, before_state = self._encode_fact_records(before_records, None, device)
            _, _, _, _, after_state = self._encode_fact_records(after_records, None, device)
            state_vectors.append(before_state)
            target_vectors.append(after_state)
        if not state_vectors:
            return None, None
        return torch.stack(state_vectors, dim=0), torch.stack(target_vectors, dim=0)

    def forward(
        self,
        facts: Sequence[Any],
        trace_bundle: Optional[Any] = None,
        saliency_facts: Optional[Sequence[Any]] = None,
        device: Optional[torch.device] = None,
    ) -> WorldGraphState:
        if device is None:
            device = next(self.parameters()).device
        fact_records: List[Tuple[str, Any]] = [("observed", atom) for atom in facts]
        if trace_bundle is not None:
            fact_records.extend(("trace_target", atom) for atom in getattr(trace_bundle, "target_facts", ()))
        if saliency_facts is not None:
            fact_records.extend(("saliency", atom) for atom in saliency_facts)

        node_keys, node_types, edges, node_states, pooled = self._encode_fact_records(
            fact_records,
            trace_bundle,
            device,
        )
        transition_states, transition_targets = self._encode_transition_sets(
            list(getattr(trace_bundle, "transitions", ())[: self.max_transitions]) if trace_bundle is not None else [],
            device,
            prefix="trace",
        )
        metadata = {
            "nodes": float(len(node_keys)),
            "edges": float(len(edges)),
            "trace_steps": float(0 if transition_targets is None else transition_targets.size(0)),
            "counterexample_steps": float(
                len(getattr(trace_bundle, "counterexamples", ())) if trace_bundle is not None else 0
            ),
        }
        return WorldGraphState(
            node_keys=node_keys,
            node_types=node_types,
            edges=edges,
            node_states=node_states,
            pooled_state=pooled,
            transition_states=transition_states,
            transition_targets=transition_targets,
            metadata=metadata,
        )

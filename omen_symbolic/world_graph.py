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
    fact_records: Tuple[Tuple[str, Any], ...]
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
        summary.setdefault(
            "graph_dense_view_is_derived",
            float(self.metadata.get("graph_dense_view_is_derived", 0.0)),
        )
        summary.setdefault(
            "graph_primary_source",
            float(self.metadata.get("graph_primary_source", 0.0)),
        )
        summary.setdefault(
            "neural_residual_used",
            float(self.metadata.get("neural_residual_used", 0.0)),
        )
        summary.setdefault(
            "signature_encoder_active",
            float(self.metadata.get("signature_encoder_active", 0.0)),
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
        self.max_signature_bytes = 64
        self.empty_state = nn.Parameter(torch.zeros(d_latent))
        self.empty_term = nn.Parameter(torch.zeros(d_latent))
        self.pred_emb = nn.Embedding(pred_buckets, d_latent)
        self.term_emb = nn.Embedding(term_buckets, d_latent)
        self.arity_emb = nn.Embedding(max_arity + 1, d_latent)
        self.rel_emb = nn.Embedding(len(self._REL_TO_ID), d_latent)
        self.byte_emb = nn.Embedding(256, d_latent)
        self.byte_pos_emb = nn.Embedding(self.max_signature_bytes, d_latent)
        self.signature_text_proj = nn.Sequential(
            nn.Linear(d_latent * 3, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        self.signature_atom_proj = nn.Sequential(
            nn.Linear(d_latent * 3, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        self.signature_norm = nn.LayerNorm(d_latent)
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

    def _encode_text_signature(self, text: str, device: torch.device) -> torch.Tensor:
        byte_values = list(text.encode("utf-8")[: self.max_signature_bytes])
        if not byte_values:
            return self.empty_term
        byte_tensor = torch.tensor(byte_values, device=device, dtype=torch.long)
        pos_tensor = torch.arange(byte_tensor.numel(), device=device, dtype=torch.long)
        byte_states = self.byte_emb(byte_tensor) + self.byte_pos_emb(pos_tensor)
        pooled_mean = byte_states.mean(dim=0)
        pooled_max = byte_states.max(dim=0).values
        pooled_first = byte_states[0]
        return self.signature_norm(
            self.signature_text_proj(torch.cat([pooled_mean, pooled_max, pooled_first], dim=-1))
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
        pred_signature = self._encode_text_signature(f"pred:{_atom_pred(atom)}", device)
        arg_signature = self._encode_text_signature(
            "|".join(_term_signature(arg) for arg in args) or "<empty>",
            device,
        )
        atom_signature = self._encode_text_signature(_atom_signature(atom), device)
        signature_state = self.signature_atom_proj(
            torch.cat([pred_signature, arg_signature, atom_signature], dim=-1)
        )
        bucket_state = pred_vec + arity_vec + arg_vec
        return self.atom_norm(signature_state + 0.25 * bucket_state)

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

    @staticmethod
    def _pair_relation(
        left_type: str,
        right_type: str,
        left_atom: Any,
        right_atom: Any,
    ) -> Optional[str]:
        relation: Optional[str] = None
        if set(_atom_term_keys(left_atom)) & set(_atom_term_keys(right_atom)):
            relation = "shared_term"
        elif _atom_pred(left_atom) == _atom_pred(right_atom):
            relation = "same_pred"
        elif left_type == right_type:
            relation = "cooccurs"
        if relation is None:
            return None
        if "saliency" in (left_type, right_type):
            return "saliency"
        return relation

    def _pairwise_edges(
        self,
        node_types: Sequence[str],
        atoms: Sequence[Any],
        *,
        base_count: int = 0,
    ) -> List[WorldGraphEdge]:
        edges: List[WorldGraphEdge] = []
        for left in range(len(atoms)):
            right_limit = min(len(atoms), left + 8)
            for right in range(left + 1, right_limit):
                if left < base_count and right < base_count:
                    continue
                relation = self._pair_relation(
                    node_types[left],
                    node_types[right],
                    atoms[left],
                    atoms[right],
                )
                if relation is None:
                    continue
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
        state_dtype = node_states.dtype
        rel_states = self.rel_emb(rel_idx).to(dtype=state_dtype)
        h = node_states
        for msg_layer, gate_layer, norm in zip(self.msg_layers, self.gate_layers, self.norm_layers):
            src_states = h[src_idx]
            dst_states = h[dst_idx]
            msg_in = torch.cat([src_states, rel_states], dim=-1)
            msg = msg_layer(msg_in).to(dtype=h.dtype)
            gate = torch.sigmoid(gate_layer(torch.cat([dst_states, msg, rel_states], dim=-1))).to(dtype=h.dtype)
            # AMP/autocast may emit lower-precision messages even when the node
            # state stays in FP32; align before index_add_ accumulation.
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
    ) -> Tuple[
        Tuple[Tuple[str, Any], ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[WorldGraphEdge, ...],
        torch.Tensor,
        torch.Tensor,
    ]:
        deduped = self._dedupe_records(fact_records)
        if not deduped:
            empty = self.empty_state.view(1, -1)
            return tuple(), tuple(), tuple(), tuple(), empty, self.empty_state

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
        return tuple(deduped), node_keys, node_types, tuple(edges), node_states, pooled

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
            _, _, _, _, _, before_state = self._encode_fact_records(before_records, None, device)
            _, _, _, _, _, after_state = self._encode_fact_records(after_records, None, device)
            state_vectors.append(before_state)
            target_vectors.append(after_state)
        if not state_vectors:
            return None, None
        return torch.stack(state_vectors, dim=0), torch.stack(target_vectors, dim=0)

    @staticmethod
    def _source_metadata(node_types: Sequence[str]) -> Dict[str, float]:
        counts: Dict[str, float] = {
            "observed_facts": float(sum(1 for node_type in node_types if node_type == "observed")),
            "observed_now_facts": float(sum(1 for node_type in node_types if node_type == "observed_now")),
            "memory_facts": float(sum(1 for node_type in node_types if node_type == "memory")),
            "saliency_facts": float(sum(1 for node_type in node_types if node_type == "saliency")),
            "net_facts": float(sum(1 for node_type in node_types if node_type == "net")),
            "context_facts": float(sum(1 for node_type in node_types if node_type == "context")),
            "world_context_facts": float(sum(1 for node_type in node_types if node_type == "world_context")),
            "abduced_support_facts": float(sum(1 for node_type in node_types if node_type == "abduced")),
            "goal_context_facts": float(sum(1 for node_type in node_types if node_type == "goal")),
            "target_context_facts": float(sum(1 for node_type in node_types if node_type == "target")),
            "trace_target_facts": float(sum(1 for node_type in node_types if node_type == "trace_target")),
        }
        return counts

    def enrich(
        self,
        base_graph: WorldGraphState,
        *,
        saliency_facts: Optional[Sequence[Any]] = None,
        context_facts: Sequence[Any],
        extra_records: Optional[Sequence[Tuple[str, Any]]] = None,
        device: Optional[torch.device] = None,
    ) -> WorldGraphState:
        if device is None:
            device = base_graph.node_states.device
        appended_records: List[Tuple[str, Any]] = list(extra_records or ())
        appended_records.extend(("saliency", atom) for atom in (saliency_facts or ()))
        appended_records.extend(("context", atom) for atom in (context_facts or ()))
        if not appended_records:
            metadata = dict(base_graph.metadata)
            metadata["context_facts"] = 0.0
            metadata["saliency_facts"] = 0.0
            metadata["signature_encoder_active"] = 1.0
            return WorldGraphState(
                node_keys=base_graph.node_keys,
                node_types=base_graph.node_types,
                fact_records=base_graph.fact_records,
                edges=base_graph.edges,
                node_states=base_graph.node_states,
                pooled_state=base_graph.pooled_state,
                transition_states=base_graph.transition_states,
                transition_targets=base_graph.transition_targets,
                metadata=metadata,
            )

        combined_records = tuple(
            self._dedupe_records(list(base_graph.fact_records) + appended_records)
        )
        if not combined_records:
            return self(
                facts=tuple(),
                saliency_facts=saliency_facts,
                context_facts=context_facts,
                device=device,
            )

        node_types = tuple(node_type for node_type, _ in combined_records)
        atoms = [atom for _, atom in combined_records]
        node_keys = tuple(_atom_signature(atom) for atom in atoms)
        base_index = {key: idx for idx, key in enumerate(base_graph.node_keys)}
        base_states = base_graph.node_states.to(device=device)
        node_states_list: List[torch.Tensor] = []
        for key, atom in zip(node_keys, atoms):
            if key in base_index:
                node_states_list.append(base_states[base_index[key]])
            else:
                node_states_list.append(self._encode_atom(atom, device))
        node_states = torch.stack(node_states_list, dim=0)

        edges = list(base_graph.edges)
        remaining = max(self.max_edges - len(edges), 0)
        if remaining > 0:
            contextual_edges = self._pairwise_edges(
                node_types,
                atoms,
                base_count=len(base_graph.node_keys),
            )
            edges.extend(contextual_edges[:remaining])
        edges = edges[: self.max_edges]
        node_states = self._message_pass(node_states, edges)
        pooled = self.pool(torch.cat([node_states.mean(dim=0), node_states.max(dim=0).values], dim=-1))

        metadata = dict(base_graph.metadata)
        metadata["nodes"] = float(len(node_keys))
        metadata["edges"] = float(len(edges))
        metadata["context_facts"] = float(len(context_facts or ()))
        metadata["saliency_facts"] = float(len(saliency_facts or ()))
        metadata["signature_encoder_active"] = 1.0
        metadata.update(self._source_metadata(node_types))
        return WorldGraphState(
            node_keys=node_keys,
            node_types=node_types,
            fact_records=combined_records,
            edges=tuple(edges),
            node_states=node_states,
            pooled_state=pooled,
            transition_states=base_graph.transition_states,
            transition_targets=base_graph.transition_targets,
            metadata=metadata,
        )

    def forward(
        self,
        facts: Sequence[Any],
        trace_bundle: Optional[Any] = None,
        saliency_facts: Optional[Sequence[Any]] = None,
        context_facts: Optional[Sequence[Any]] = None,
        extra_records: Optional[Sequence[Tuple[str, Any]]] = None,
        device: Optional[torch.device] = None,
    ) -> WorldGraphState:
        if device is None:
            device = next(self.parameters()).device
        fact_records: List[Tuple[str, Any]] = [("observed", atom) for atom in facts]
        if trace_bundle is not None:
            fact_records.extend(("trace_target", atom) for atom in getattr(trace_bundle, "target_facts", ()))
        if saliency_facts is not None:
            fact_records.extend(("saliency", atom) for atom in saliency_facts)
        if context_facts is not None:
            fact_records.extend(("context", atom) for atom in context_facts)
        if extra_records is not None:
            fact_records.extend(list(extra_records))

        fact_records_out, node_keys, node_types, edges, node_states, pooled = self._encode_fact_records(
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
            "signature_encoder_active": 1.0,
            "signature_primary": 1.0,
            "context_facts": float(len(context_facts or ())),
            "saliency_facts": float(len(saliency_facts or ())),
            "counterexample_steps": float(
                len(getattr(trace_bundle, "counterexamples", ())) if trace_bundle is not None else 0
            ),
        }
        metadata.update(self._source_metadata(node_types))
        return WorldGraphState(
            node_keys=node_keys,
            node_types=node_types,
            fact_records=fact_records_out,
            edges=edges,
            node_states=node_states,
            pooled_state=pooled,
            transition_states=transition_states,
            transition_targets=transition_targets,
            metadata=metadata,
        )

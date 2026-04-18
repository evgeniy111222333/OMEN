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
        self._text_signature_cache: Dict[Tuple[str, bool, str], torch.Tensor] = {}
        self._atom_encoding_cache: Dict[Tuple[str, bool, str], torch.Tensor] = {}

    @staticmethod
    def _runtime_cache_key(device: torch.device) -> Tuple[str, bool]:
        return (f"{device.type}:{device.index if device.index is not None else -1}", torch.is_autocast_enabled())

    def clear_runtime_caches(self) -> None:
        self._text_signature_cache.clear()
        self._atom_encoding_cache.clear()

    def _encode_text_signature(self, text: str, device: torch.device) -> torch.Tensor:
        cache_key = (*self._runtime_cache_key(device), text)
        cached = self._text_signature_cache.get(cache_key)
        if cached is not None:
            return cached
        byte_values = list(text.encode("utf-8")[: self.max_signature_bytes])
        if not byte_values:
            return self.empty_term
        byte_tensor = torch.tensor(byte_values, device=device, dtype=torch.long)
        pos_tensor = torch.arange(byte_tensor.numel(), device=device, dtype=torch.long)
        byte_states = self.byte_emb(byte_tensor) + self.byte_pos_emb(pos_tensor)
        pooled_mean = byte_states.mean(dim=0)
        pooled_max = byte_states.max(dim=0).values
        pooled_first = byte_states[0]
        encoded = self.signature_norm(
            self.signature_text_proj(torch.cat([pooled_mean, pooled_max, pooled_first], dim=-1))
        )
        self._text_signature_cache[cache_key] = encoded
        return encoded

    def _encode_text_signature_batch(
        self,
        texts: Sequence[str],
        device: torch.device,
    ) -> List[torch.Tensor]:
        if not texts:
            return []
        cache_prefix = self._runtime_cache_key(device)
        outputs: List[Optional[torch.Tensor]] = [None] * len(texts)
        unique_misses: Dict[str, List[int]] = {}
        for idx, text in enumerate(texts):
            cache_key = (*cache_prefix, text)
            cached = self._text_signature_cache.get(cache_key)
            if cached is not None:
                outputs[idx] = cached
            else:
                unique_misses.setdefault(text, []).append(idx)

        nonempty_texts = [text for text in unique_misses if text]
        if nonempty_texts:
            byte_rows = [list(text.encode("utf-8")[: self.max_signature_bytes]) for text in nonempty_texts]
            max_len = max(len(row) for row in byte_rows)
            byte_tensor = torch.zeros(len(nonempty_texts), max_len, device=device, dtype=torch.long)
            mask = torch.zeros(len(nonempty_texts), max_len, device=device, dtype=torch.bool)
            for row_idx, row in enumerate(byte_rows):
                row_len = len(row)
                byte_tensor[row_idx, :row_len] = torch.tensor(row, device=device, dtype=torch.long)
                mask[row_idx, :row_len] = True
            pos_tensor = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
            byte_states = self.byte_emb(byte_tensor) + self.byte_pos_emb(pos_tensor)
            mask_f = mask.unsqueeze(-1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled_mean = (byte_states * mask_f).sum(dim=1) / lengths.to(dtype=byte_states.dtype)
            fill_value = torch.finfo(byte_states.dtype).min
            pooled_max = byte_states.masked_fill(~mask_f, fill_value).max(dim=1).values
            pooled_first = byte_states[:, 0]
            encoded_batch = self.signature_norm(
                self.signature_text_proj(torch.cat([pooled_mean, pooled_max, pooled_first], dim=-1))
            )
            for row_idx, text in enumerate(nonempty_texts):
                encoded = encoded_batch[row_idx]
                self._text_signature_cache[(*cache_prefix, text)] = encoded
                for out_idx in unique_misses[text]:
                    outputs[out_idx] = encoded

        for text, indices in unique_misses.items():
            if text:
                continue
            cached = self._text_signature_cache.setdefault((*cache_prefix, text), self.empty_term)
            for out_idx in indices:
                outputs[out_idx] = cached

        return [tensor if tensor is not None else self.empty_term for tensor in outputs]

    def _encode_atom(self, atom: Any, device: torch.device) -> torch.Tensor:
        atom_key = (*self._runtime_cache_key(device), _atom_signature(atom))
        cached = self._atom_encoding_cache.get(atom_key)
        if cached is not None:
            return cached
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
        encoded = self.atom_norm(signature_state + 0.25 * bucket_state)
        self._atom_encoding_cache[atom_key] = encoded
        return encoded

    def _encode_atom_batch(
        self,
        atoms: Sequence[Any],
        device: torch.device,
    ) -> torch.Tensor:
        if not atoms:
            return self.empty_state.new_zeros((0, self.d_latent))
        cache_prefix = self._runtime_cache_key(device)
        outputs: List[Optional[torch.Tensor]] = [None] * len(atoms)
        missing_atoms: List[Any] = []
        missing_indices: List[int] = []
        missing_keys: List[Tuple[str, bool, str]] = []
        for idx, atom in enumerate(atoms):
            atom_sig = _atom_signature(atom)
            cache_key = (*cache_prefix, atom_sig)
            cached = self._atom_encoding_cache.get(cache_key)
            if cached is not None:
                outputs[idx] = cached
                continue
            missing_atoms.append(atom)
            missing_indices.append(idx)
            missing_keys.append(cache_key)

        if missing_atoms:
            pred_buckets = torch.tensor(
                [_stable_bucket(f"pred:{_atom_pred(atom)}", self.pred_buckets) for atom in missing_atoms],
                device=device,
                dtype=torch.long,
            )
            arity_buckets = torch.tensor(
                [min(len(_atom_args(atom)), self.max_arity) for atom in missing_atoms],
                device=device,
                dtype=torch.long,
            )
            pred_vec = self.pred_emb(pred_buckets)
            arity_vec = self.arity_emb(arity_buckets)

            arg_vec = self.empty_term.unsqueeze(0).expand(len(missing_atoms), -1).clone()
            arg_bucket_values: List[int] = []
            arg_owner: List[int] = []
            pred_texts: List[str] = []
            arg_texts: List[str] = []
            atom_texts: List[str] = []
            for owner_idx, atom in enumerate(missing_atoms):
                args = _atom_args(atom)
                arg_terms = [_term_signature(arg) for arg in args]
                pred_texts.append(f"pred:{_atom_pred(atom)}")
                arg_texts.append("|".join(arg_terms) or "<empty>")
                atom_texts.append(_atom_signature(atom))
                for term in arg_terms:
                    arg_bucket_values.append(_stable_bucket(term, self.term_buckets))
                    arg_owner.append(owner_idx)
            if arg_bucket_values:
                owner_idx_t = torch.tensor(arg_owner, device=device, dtype=torch.long)
                arg_emb = self.term_emb(torch.tensor(arg_bucket_values, device=device, dtype=torch.long))
                arg_sum = torch.zeros(len(missing_atoms), self.d_latent, device=device, dtype=arg_emb.dtype)
                arg_count = torch.zeros(len(missing_atoms), 1, device=device, dtype=arg_emb.dtype)
                arg_sum.index_add_(0, owner_idx_t, arg_emb)
                arg_count.index_add_(
                    0,
                    owner_idx_t,
                    torch.ones(owner_idx_t.numel(), 1, device=device, dtype=arg_emb.dtype),
                )
                nonzero = arg_count.squeeze(-1) > 0
                arg_vec[nonzero] = arg_sum[nonzero] / arg_count[nonzero]

            pred_signature = torch.stack(self._encode_text_signature_batch(pred_texts, device), dim=0)
            arg_signature = torch.stack(self._encode_text_signature_batch(arg_texts, device), dim=0)
            atom_signature = torch.stack(self._encode_text_signature_batch(atom_texts, device), dim=0)
            signature_state = self.signature_atom_proj(
                torch.cat([pred_signature, arg_signature, atom_signature], dim=-1)
            )
            bucket_state = pred_vec + arity_vec + arg_vec
            encoded_batch = self.atom_norm(signature_state + 0.25 * bucket_state)
            for out_idx, cache_key, encoded in zip(missing_indices, missing_keys, encoded_batch.unbind(0)):
                self._atom_encoding_cache[cache_key] = encoded
                outputs[out_idx] = encoded

        return torch.stack(
            [tensor if tensor is not None else self.empty_term for tensor in outputs],
            dim=0,
        )

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
        node_states = self._encode_atom_batch(atoms, device)
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
        node_index: Dict[str, int],
        node_states: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not transitions:
            return None, None
        state_vectors: List[torch.Tensor] = []
        target_vectors: List[torch.Tensor] = []
        empty = self.empty_state.to(device=node_states.device, dtype=node_states.dtype)

        def _pool_atoms(atoms: Iterable[Any]) -> torch.Tensor:
            indices = [
                node_index[sig]
                for sig in (_atom_signature(atom) for atom in atoms)
                if sig in node_index
            ]
            if not indices:
                return empty
            subset = node_states[torch.tensor(indices, device=node_states.device, dtype=torch.long)]
            return self.pool(torch.cat([subset.mean(dim=0), subset.max(dim=0).values], dim=-1))

        for transition in transitions[: self.max_transitions]:
            before_state = _pool_atoms(getattr(transition, "before_facts", ()))
            after_state = _pool_atoms(getattr(transition, "after_facts", ()))
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
        missing_atoms: List[Any] = []
        missing_positions: List[int] = []
        for key, atom in zip(node_keys, atoms):
            if key in base_index:
                node_states_list.append(base_states[base_index[key]])
            else:
                node_states_list.append(base_states.new_zeros(base_states.size(-1)))
                missing_atoms.append(atom)
                missing_positions.append(len(node_states_list) - 1)
        if missing_atoms:
            encoded_missing = self._encode_atom_batch(missing_atoms, device)
            for pos, state in zip(missing_positions, encoded_missing.unbind(0)):
                node_states_list[pos] = state
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
            {key: idx for idx, key in enumerate(node_keys)},
            node_states,
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

from __future__ import annotations

from collections import deque
import pickle
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F


class SymbolicMemoryIndex:
    """Exact symbolic interface over the same long-term memory system."""

    def __init__(self, max_entries: int = 2048):
        self._entries: Deque[Tuple[object, torch.Tensor]] = deque(maxlen=max_entries)

    def __len__(self) -> int:
        return len(self._entries)

    def export_state(self) -> Dict[str, Any]:
        facts = [fact for fact, _ in self._entries]
        embeddings = [emb.detach().cpu().clone() for _, emb in self._entries]
        if embeddings:
            stacked = torch.stack(embeddings, dim=0)
        else:
            stacked = torch.zeros(0, dtype=torch.float32)
        return {
            "max_entries": int(self._entries.maxlen or 0),
            "facts": pickle.dumps(facts),
            "embeddings": stacked,
        }

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        state = state or {}
        max_entries = max(int(state.get("max_entries", self._entries.maxlen or 1)), 1)
        facts_blob = state.get("facts")
        if facts_blob is None:
            facts: List[object] = []
        elif isinstance(facts_blob, bytes):
            facts = list(pickle.loads(facts_blob))
        else:
            facts = list(facts_blob)
        embeddings = state.get("embeddings")
        if embeddings is None or not isinstance(embeddings, torch.Tensor) or embeddings.numel() == 0:
            restored: List[Tuple[object, torch.Tensor]] = []
        else:
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            restored = [
                (fact, emb.detach().cpu().clone())
                for fact, emb in zip(facts, embeddings)
            ]
        self._entries = deque(restored[-max_entries:], maxlen=max_entries)

    def write(
        self,
        facts: Sequence[object],
        embeddings: torch.Tensor,
    ) -> int:
        if not facts or embeddings.numel() == 0:
            return 0
        added = 0
        for fact, emb in zip(facts, embeddings):
            self._entries.append((fact, emb.detach().cpu()))
            added += 1
        return added

    @staticmethod
    def _fact_predicate(fact: object) -> Optional[int]:
        pred = getattr(fact, "pred", None)
        return int(pred) if pred is not None else None

    @staticmethod
    def _fact_anchor_values(fact: object) -> Set[int]:
        values: Set[int] = set()

        def visit(term: object) -> None:
            if hasattr(term, "val"):
                values.add(int(term.val))
                return
            if isinstance(term, int):
                values.add(int(term))
                return
            func = getattr(term, "func", None)
            if func is not None:
                values.add(int(func))
            for subterm in getattr(term, "subterms", ()) or ():
                visit(subterm)

        args = getattr(fact, "args", ())
        for arg in args:
            visit(arg)
        return values

    @staticmethod
    def _merge_unique(primary: Sequence[object], secondary: Sequence[object]) -> List[object]:
        merged: List[object] = []
        seen = set()
        for item in list(primary) + list(secondary):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
        return merged

    def recall_by_pattern(
        self,
        predicate_hints: Optional[Sequence[int]] = None,
        anchor_values: Optional[Sequence[int]] = None,
        limit: int = 8,
    ) -> List[object]:
        if not self._entries:
            return []
        pred_set = {int(pred) for pred in (predicate_hints or ())}
        anchor_set = {int(value) for value in (anchor_values or ())}
        if not pred_set and not anchor_set:
            return []

        matched: List[object] = []
        for fact, _ in reversed(self._entries):
            pred = self._fact_predicate(fact)
            anchors = self._fact_anchor_values(fact)
            if pred_set and pred in pred_set:
                matched.append(fact)
            elif anchor_set and anchors.intersection(anchor_set):
                matched.append(fact)
            if len(matched) >= int(limit):
                break
        return list(reversed(matched))

    @torch.no_grad()
    def recall(
        self,
        query: torch.Tensor,
        top_k: int = 8,
        min_sim: float = 0.2,
        predicate_hints: Optional[Sequence[int]] = None,
        anchor_values: Optional[Sequence[int]] = None,
        structured_limit: Optional[int] = None,
    ) -> List[object]:
        if not self._entries:
            return []
        facts = [fact for fact, _ in self._entries]
        embs = torch.stack([emb for _, emb in self._entries], dim=0).to(
            query.device, dtype=query.dtype
        )
        sims = F.cosine_similarity(query.unsqueeze(1), embs.unsqueeze(0), dim=-1)
        pred_set = {int(pred) for pred in (predicate_hints or ())}
        anchor_set = {int(value) for value in (anchor_values or ())}
        if pred_set or anchor_set:
            bonus = torch.zeros_like(sims)
            for idx, fact in enumerate(facts):
                pred_bonus = 0.15 if self._fact_predicate(fact) in pred_set else 0.0
                anchor_bonus = 0.10 if self._fact_anchor_values(fact).intersection(anchor_set) else 0.0
                bonus[:, idx] = pred_bonus + anchor_bonus
            sims = sims + bonus
        k = min(int(top_k), embs.size(0))
        if k <= 0:
            selected = []
        else:
            top = sims.topk(k, dim=-1)
            selected = []
            for idx, sim in zip(top.indices[0].tolist(), top.values[0].tolist()):
                if float(sim) < float(min_sim):
                    continue
                selected.append(facts[int(idx)])

        structured = self.recall_by_pattern(
            predicate_hints=predicate_hints,
            anchor_values=anchor_values,
            limit=int(structured_limit or top_k),
        )
        return self._merge_unique(structured, selected)[: max(int(top_k), int(structured_limit or top_k))]

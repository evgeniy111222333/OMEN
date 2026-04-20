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
    def _fact_graph_terms(fact: object) -> Set[str]:
        terms = getattr(fact, "graph_terms", None)
        if terms is None:
            return set()
        return {str(term).strip() for term in terms if str(term).strip()}

    @staticmethod
    def _fact_graph_family(fact: object) -> Optional[str]:
        family = getattr(fact, "graph_family", None)
        if family is None:
            return None
        text = str(family).strip()
        return text or None

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
        graph_terms: Optional[Sequence[str]] = None,
        graph_families: Optional[Sequence[str]] = None,
        boost_graph_terms: Optional[Sequence[str]] = None,
        boost_graph_families: Optional[Sequence[str]] = None,
        suppress_graph_terms: Optional[Sequence[str]] = None,
        suppress_graph_families: Optional[Sequence[str]] = None,
        limit: int = 8,
    ) -> List[object]:
        if not self._entries:
            return []
        pred_set = {int(pred) for pred in (predicate_hints or ())}
        anchor_set = {int(value) for value in (anchor_values or ())}
        term_set = {str(term).strip() for term in (graph_terms or ()) if str(term).strip()}
        family_set = {str(family).strip() for family in (graph_families or ()) if str(family).strip()}
        boost_term_set = {str(term).strip() for term in (boost_graph_terms or ()) if str(term).strip()}
        boost_family_set = {str(family).strip() for family in (boost_graph_families or ()) if str(family).strip()}
        suppress_term_set = {str(term).strip() for term in (suppress_graph_terms or ()) if str(term).strip()}
        suppress_family_set = {str(family).strip() for family in (suppress_graph_families or ()) if str(family).strip()}
        if (
            not pred_set
            and not anchor_set
            and not term_set
            and not family_set
            and not boost_term_set
            and not boost_family_set
        ):
            return []

        scored: List[Tuple[float, int, object]] = []
        for recency, (fact, _) in enumerate(reversed(self._entries)):
            pred = self._fact_predicate(fact)
            anchors = self._fact_anchor_values(fact)
            fact_terms = self._fact_graph_terms(fact)
            fact_family = self._fact_graph_family(fact)
            score = 0.0
            if pred_set and pred in pred_set:
                score += 1.00
            if anchor_set and anchors.intersection(anchor_set):
                score += 0.70
            if term_set and fact_terms.intersection(term_set):
                score += 0.60
            if family_set and fact_family in family_set:
                score += 0.45
            if boost_term_set and fact_terms.intersection(boost_term_set):
                score += 0.35
            if boost_family_set and fact_family in boost_family_set:
                score += 0.25
            if suppress_term_set and fact_terms.intersection(suppress_term_set):
                score -= 0.20
            if suppress_family_set and fact_family in suppress_family_set:
                score -= 0.15
            if score <= 0.0:
                continue
            scored.append((score, -recency, fact))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [fact for _, _, fact in scored[: max(int(limit), 0)]]

    @torch.no_grad()
    def recall(
        self,
        query: torch.Tensor,
        top_k: int = 8,
        min_sim: float = 0.2,
        predicate_hints: Optional[Sequence[int]] = None,
        anchor_values: Optional[Sequence[int]] = None,
        graph_terms: Optional[Sequence[str]] = None,
        graph_families: Optional[Sequence[str]] = None,
        boost_graph_terms: Optional[Sequence[str]] = None,
        boost_graph_families: Optional[Sequence[str]] = None,
        suppress_graph_terms: Optional[Sequence[str]] = None,
        suppress_graph_families: Optional[Sequence[str]] = None,
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
        term_set = {str(term).strip() for term in (graph_terms or ()) if str(term).strip()}
        family_set = {str(family).strip() for family in (graph_families or ()) if str(family).strip()}
        boost_term_set = {str(term).strip() for term in (boost_graph_terms or ()) if str(term).strip()}
        boost_family_set = {str(family).strip() for family in (boost_graph_families or ()) if str(family).strip()}
        suppress_term_set = {str(term).strip() for term in (suppress_graph_terms or ()) if str(term).strip()}
        suppress_family_set = {str(family).strip() for family in (suppress_graph_families or ()) if str(family).strip()}
        if (
            pred_set
            or anchor_set
            or term_set
            or family_set
            or boost_term_set
            or boost_family_set
            or suppress_term_set
            or suppress_family_set
        ):
            bonus = torch.zeros_like(sims)
            for idx, fact in enumerate(facts):
                pred_bonus = 0.15 if self._fact_predicate(fact) in pred_set else 0.0
                anchor_bonus = 0.10 if self._fact_anchor_values(fact).intersection(anchor_set) else 0.0
                term_bonus = 0.12 if self._fact_graph_terms(fact).intersection(term_set) else 0.0
                family_bonus = 0.08 if self._fact_graph_family(fact) in family_set else 0.0
                boost_term_bonus = 0.14 if self._fact_graph_terms(fact).intersection(boost_term_set) else 0.0
                boost_family_bonus = 0.10 if self._fact_graph_family(fact) in boost_family_set else 0.0
                suppress_term_penalty = 0.10 if self._fact_graph_terms(fact).intersection(suppress_term_set) else 0.0
                suppress_family_penalty = 0.08 if self._fact_graph_family(fact) in suppress_family_set else 0.0
                bonus[:, idx] = (
                    pred_bonus
                    + anchor_bonus
                    + term_bonus
                    + family_bonus
                    + boost_term_bonus
                    + boost_family_bonus
                    - suppress_term_penalty
                    - suppress_family_penalty
                )
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
            graph_terms=graph_terms,
            graph_families=graph_families,
            boost_graph_terms=boost_graph_terms,
            boost_graph_families=boost_graph_families,
            suppress_graph_terms=suppress_graph_terms,
            suppress_graph_families=suppress_graph_families,
            limit=int(structured_limit or top_k),
        )
        return self._merge_unique(structured, selected)[: max(int(top_k), int(structured_limit or top_k))]

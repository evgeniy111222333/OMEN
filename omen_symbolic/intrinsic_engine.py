from __future__ import annotations

from collections import deque
import math
import pickle
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import torch

from omen_symbolic.creative_types import IntrinsicGoal, RuleCandidate
from omen_symbolic.rule_graph import PredicateGraphView


class IntrinsicCuriosityEngine:
    def __init__(
        self,
        state_history: int = 128,
        goal_threshold: float = 0.35,
        novelty_weight: float = 0.4,
        gap_weight: float = 0.3,
        analogy_weight: float = 0.3,
    ):
        self._history: Deque[torch.Tensor] = deque(maxlen=max(int(state_history), 8))
        self.goal_threshold = float(max(goal_threshold, 0.0))
        self.novelty_weight = float(max(novelty_weight, 0.0))
        # Kept as gap_weight for config compatibility, but per concept.md this
        # weight is applied to the structural gap term, not to external GapNorm.
        self.gap_weight = float(max(gap_weight, 0.0))
        self.analogy_weight = float(max(analogy_weight, 0.0))
        self.pending_goal: Optional[IntrinsicGoal] = None
        self._goal_queue: Deque[IntrinsicGoal] = deque(maxlen=max(int(state_history), 8))

    def export_state(self) -> Dict[str, Any]:
        return {
            "history": [state.detach().cpu().clone() for state in self._history],
            "pending_goal": (
                None if self.pending_goal is None else pickle.dumps(self.pending_goal)
            ),
            "goal_queue": pickle.dumps(list(self._goal_queue)),
        }

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        state = state or {}
        history = [tensor.detach().cpu().clone() for tensor in state.get("history", ())]
        self._history = deque(history[-self._history.maxlen :], maxlen=self._history.maxlen)
        queue_blob = state.get("goal_queue")
        if isinstance(queue_blob, bytes):
            queued_goals = list(pickle.loads(queue_blob))
        else:
            queued_goals = list(queue_blob or ())
        self._goal_queue = deque(
            queued_goals[-self._goal_queue.maxlen :],
            maxlen=self._goal_queue.maxlen,
        )
        pending_blob = state.get("pending_goal")
        if isinstance(pending_blob, bytes):
            self.pending_goal = pickle.loads(pending_blob)
        else:
            self.pending_goal = pending_blob
        if self.pending_goal is None and self._goal_queue:
            self.pending_goal = self._goal_queue[0]

    def update_state(self, z: Optional[torch.Tensor]) -> None:
        if z is None or z.numel() == 0:
            return
        self._history.append(z.detach().mean(dim=0).cpu())

    def _novelty_score(self, z: Optional[torch.Tensor]) -> float:
        if z is None or z.numel() == 0 or not self._history:
            return 0.0
        current = z.detach().mean(dim=0).cpu()
        history = torch.stack(list(self._history), dim=0)
        diff = history - current.unsqueeze(0)
        dist_sq = diff.pow(2).mean(dim=-1)
        bandwidth = history.std(dim=0, unbiased=False).mean().clamp_min(1e-3)
        density = torch.exp(-dist_sq / (2.0 * bandwidth * bandwidth)).mean().clamp_min(1e-6)
        return float((-torch.log(density) / 6.0).clamp(0.0, 1.0).item())

    @staticmethod
    def _structural_gap(view: PredicateGraphView) -> float:
        if view.adjacency.numel() == 0:
            return 0.0
        degree = view.adjacency.sum(dim=-1)
        return float((degree <= 1.5).float().mean().item())

    @staticmethod
    def _analogy_potential(view: PredicateGraphView) -> float:
        if view.similarity.numel() == 0 or view.similarity.size(0) < 2:
            return 0.0
        masked = view.similarity.clone()
        masked.fill_diagonal_(-1.0)
        unique = 1.0 - masked.max(dim=-1).values.clamp(-1.0, 1.0)
        return float(unique.clamp(0.0, 1.0).sum().item() / max(masked.size(0), 1))

    @staticmethod
    def _goal_key(goal: IntrinsicGoal) -> Tuple[str, str, str]:
        return (goal.kind, goal.provenance, repr(goal.goal))

    @staticmethod
    def _finite_scalar(value: float) -> float:
        value = float(value)
        return value if math.isfinite(value) else 0.0

    def _schedule_goal(self, goal: IntrinsicGoal) -> None:
        goal = IntrinsicGoal(
            goal=goal.goal,
            value=self._finite_scalar(goal.value),
            kind=goal.kind,
            provenance=goal.provenance,
            metadata=dict(goal.metadata),
        )
        key = self._goal_key(goal)
        keep: Deque[IntrinsicGoal] = deque(maxlen=self._goal_queue.maxlen)
        inserted = False
        for queued in self._goal_queue:
            if self._goal_key(queued) == key:
                if not inserted:
                    keep.append(goal if goal.value >= queued.value else queued)
                    inserted = True
                continue
            keep.append(queued)
        if not inserted:
            keep.append(goal)
        ranked = sorted(keep, key=lambda item: item.value, reverse=True)
        self._goal_queue = deque(ranked[: self._goal_queue.maxlen], maxlen=self._goal_queue.maxlen)
        self.pending_goal = self._goal_queue[0] if self._goal_queue else None

    def scheduled_goals(self) -> Tuple[IntrinsicGoal, ...]:
        return tuple(self._goal_queue)

    def resolve_supported_goals(self, facts: Sequence[Any]) -> None:
        from omen_prolog import unify

        if not self._goal_queue:
            self.pending_goal = None
            return

        kept: Deque[IntrinsicGoal] = deque(maxlen=self._goal_queue.maxlen)
        for goal in self._goal_queue:
            supported = False
            for fact in facts:
                try:
                    if unify(goal.goal, fact) is not None:
                        supported = True
                        break
                except Exception:
                    continue
            if not supported:
                kept.append(goal)

        self._goal_queue = kept
        self.pending_goal = self._goal_queue[0] if self._goal_queue else None

    @staticmethod
    def _query_from_predicate(predicate_id: int, arity: int) -> Any:
        from omen_prolog import HornAtom, Var

        return HornAtom(
            pred=int(predicate_id),
            args=tuple(Var(f"Q{idx}") for idx in range(max(int(arity), 1))),
        )

    def formulate_goal(
        self,
        z: Optional[torch.Tensor],
        gap_norm: float,
        graph_view: PredicateGraphView,
        candidate_goals: Sequence[Any],
        candidate_rules: Sequence[RuleCandidate],
        provenance: str,
    ) -> Optional[IntrinsicGoal]:
        novelty = self._novelty_score(z)
        structural_gap = self._structural_gap(graph_view)
        analogy_potential = self._analogy_potential(graph_view)
        novelty = self._finite_scalar(novelty)
        structural_gap = self._finite_scalar(structural_gap)
        analogy_potential = self._finite_scalar(analogy_potential)
        value = (
            self.novelty_weight * novelty
            + self.gap_weight * structural_gap
            + self.analogy_weight * analogy_potential
        )
        value = self._finite_scalar(value)
        if value < self.goal_threshold:
            if self.pending_goal is None and self._goal_queue:
                self.pending_goal = self._goal_queue[0]
            return self.pending_goal

        metadata = {
            "novelty": float(novelty),
            "gap_norm": float(gap_norm),
            "structural_gap": float(structural_gap),
            "analogy_potential": float(analogy_potential),
        }
        if candidate_rules:
            ranked_rules = sorted(candidate_rules, key=lambda item: self._finite_scalar(item.score), reverse=True)
            best = ranked_rules[0]
            goal = best.clause.head
            kind = "prove_or_disprove_hypothesis"
            metadata.update(
                {
                    "hypothesis_source": best.source,
                    "hypothesis_score": self._finite_scalar(best.score),
                    "hypothesis_rule_repr": repr(best.clause),
                }
            )
        elif candidate_goals:
            goal = candidate_goals[0]
            kind = "validate_candidate"
        elif graph_view.predicate_ids:
            rare_idx = int(graph_view.adjacency.sum(dim=-1).argmin().item())
            pred_id = int(graph_view.predicate_ids[rare_idx])
            goal = self._query_from_predicate(pred_id, graph_view.arities.get(pred_id, 1))
            kind = "explore_structure"
        else:
            if self.pending_goal is None and self._goal_queue:
                self.pending_goal = self._goal_queue[0]
            return self.pending_goal

        scheduled = IntrinsicGoal(
            goal=goal,
            value=float(value),
            kind=kind,
            provenance=provenance,
            metadata=metadata,
        )
        self._schedule_goal(scheduled)
        return self.pending_goal

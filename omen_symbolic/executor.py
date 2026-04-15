from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet


@dataclass(frozen=True)
class SymbolicExecutionResult:
    goal: Any
    working_facts: FrozenSet[Any]
    target_facts: FrozenSet[Any]
    all_facts: FrozenSet[Any]
    goal_supported: bool
    target_hits: int
    target_total: int
    target_coverage: float
    unresolved_targets: int


def run_symbolic_executor(
    kb: Any,
    max_depth: int,
    working_facts: FrozenSet[Any],
    goal: Any,
    target_facts: FrozenSet[Any],
    goal_supported_fn: Callable[[Any, FrozenSet[Any]], bool],
) -> SymbolicExecutionResult:
    """
    Run the purely symbolic executor on discrete facts/rules only.

    This layer does not inspect latent vectors. It only receives:
      - a discrete goal
      - a discrete working-memory fact set
      - verified rules in the KB
    """

    if hasattr(kb, "forward_chain_reasoned"):
        all_facts = kb.forward_chain_reasoned(
            max_depth,
            starting_facts=working_facts,
            only_verified=True,
        )
    else:
        all_facts = kb.forward_chain(
            max_depth,
            starting_facts=working_facts,
            only_verified=True,
        )
    goal_supported = bool(goal_supported_fn(goal, all_facts))
    target_hits = len(all_facts & target_facts)
    target_total = len(target_facts)
    target_coverage = (
        float(target_hits) / float(target_total)
        if target_total > 0
        else (1.0 if goal_supported else 0.0)
    )
    unresolved_targets = max(target_total - target_hits, 0)
    return SymbolicExecutionResult(
        goal=goal,
        working_facts=working_facts,
        target_facts=target_facts,
        all_facts=all_facts,
        goal_supported=goal_supported,
        target_hits=target_hits,
        target_total=target_total,
        target_coverage=target_coverage,
        unresolved_targets=unresolved_targets,
    )

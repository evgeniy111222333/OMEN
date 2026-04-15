from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet

import torch


@dataclass
class LatentControllerResult:
    proof_loss: torch.Tensor
    vem_hinge: torch.Tensor
    abductor_aux: torch.Tensor
    vem_self_loss: torch.Tensor
    mean_utility: float = 0.0
    abduced_rules: int = 0
    goal_supported: bool = False
    induction_stats: Dict[str, float] = field(default_factory=dict)
    cycle_stats: Dict[str, float] = field(default_factory=dict)


def empty_induction_stats() -> Dict[str, float]:
    return {
        "checked": 0.0,
        "verified": 0.0,
        "contradicted": 0.0,
        "retained": 0.0,
        "repaired": 0.0,
        "matched_predictions": 0.0,
        "mean_score": 0.0,
    }


def merge_induction_stats(*stats_dicts: Dict[str, float]) -> Dict[str, float]:
    merged = empty_induction_stats()
    total_checked = 0.0
    weighted_score = 0.0
    for stats in stats_dicts:
        if not stats:
            continue
        checked = float(stats.get("checked", 0.0))
        merged["checked"] += checked
        merged["verified"] += float(stats.get("verified", 0.0))
        merged["contradicted"] += float(stats.get("contradicted", 0.0))
        merged["retained"] += float(stats.get("retained", 0.0))
        merged["repaired"] += float(stats.get("repaired", 0.0))
        merged["matched_predictions"] += float(stats.get("matched_predictions", 0.0))
        weighted_score += checked * float(stats.get("mean_score", 0.0))
        total_checked += checked
    if total_checked > 0.0:
        merged["mean_score"] = weighted_score / total_checked
    return merged


def run_latent_reasoning_controller(
    prover: Any,
    z: torch.Tensor,
    goal: Any,
    working_facts: FrozenSet[Any],
    all_facts: FrozenSet[Any],
    target_facts: FrozenSet[Any],
    goal_supported: bool,
    target_coverage: float,
    world_error: Any,
    device: torch.device,
) -> LatentControllerResult:
    """
    Run latent-conditioned training-only control around a pure symbolic step.

    The symbolic executor itself stays discrete. This controller owns the
    latent-dependent policy, abduction, and VeM updates.
    """

    zero = torch.zeros(1, device=device).squeeze()
    result = LatentControllerResult(
        proof_loss=zero,
        vem_hinge=zero,
        abductor_aux=zero,
        vem_self_loss=zero,
        mean_utility=0.0,
        abduced_rules=0,
        goal_supported=goal_supported,
        induction_stats=empty_induction_stats(),
        cycle_stats={},
    )

    if prover.training and len(prover.kb.rules) > 0:
        proved, traj, proof_loss = prover.prove_with_policy(
            goal,
            z[:1],
            starting_facts=working_facts,
        )
        result.proof_loss = proof_loss
        result.goal_supported = goal_supported or proved

        if traj and prover.kb.rules:
            used_rule = prover.kb.rules[traj[-1] % len(prover.kb.rules)]
            utility = 0.75 if proved else 0.25
            prover.vem.record_outcome(
                used_rule,
                utility_target=utility,
                device=device,
            )
            prover._record_rule_utility(used_rule, utility)
            if proved:
                prover.kb.mark_rule_verified(used_rule)

        if prover.kb.rules:
            rule = random.choice(prover.kb.rules)
            if rule.body:
                max_unif = 64
                if len(all_facts) > max_unif:
                    unif_facts = frozenset(random.sample(list(all_facts), max_unif))
                else:
                    unif_facts = all_facts
                gm_energy, _, gm_entropy = prover.graph_unif(
                    rule.body, unif_facts, device, tau=0.5
                )
                su_e, su_ent = prover.soft_unif(rule.body, unif_facts, device)
                result.proof_loss = (
                    result.proof_loss
                    + 0.01 * gm_energy
                    - 0.001 * gm_entropy
                    + 0.01 * su_e
                    - 0.001 * su_ent
                )

    effective_targets = target_facts or frozenset({goal})
    cycle_recent_rules = []
    if prover.training and getattr(prover, "continuous_cycle_enabled", False):
        cycle = prover.continuous_hypothesis_cycle(
            z[:1],
            working_facts,
            effective_targets,
            device,
        )
        result.abductor_aux = result.abductor_aux + cycle["loss_tensor"]
        result.mean_utility = float(cycle.get("mean_utility", 0.0))
        result.abduced_rules = int(cycle.get("added_rules", 0))
        result.induction_stats = merge_induction_stats(
            result.induction_stats,
            cycle.get("induction_stats", {}),
        )
        result.cycle_stats = cycle.get("stats", {})
        cycle_recent_rules = list(cycle.get("accepted_rules", []))

    err_val = (world_error.item() if torch.is_tensor(world_error) else float(world_error))
    mismatch_error = (1.0 - target_coverage) + (0.0 if result.goal_supported else 1.0)
    trigger_abduction = (
        prover.task_context is not None and prover.task_context.trigger_abduction
    )
    should_abduce = prover.training and (trigger_abduction or mismatch_error > 0.0)
    if should_abduce:
        added, vem_hinge, abductor_aux, mean_utility = prover.abduce_and_learn(
            z,
            max(float(err_val), float(mismatch_error)),
            force=trigger_abduction or mismatch_error > 0.0,
        )
        prev_added = float(result.abduced_rules)
        prev_utility = prev_added * float(result.mean_utility)
        result.abduced_rules = int(result.abduced_rules + int(added))
        result.vem_hinge = result.vem_hinge + vem_hinge
        result.abductor_aux = result.abductor_aux + abductor_aux
        total_added = max(prev_added + float(added), 1.0)
        result.mean_utility = float((prev_utility + float(added) * float(mean_utility)) / total_added)
        if added > 0:
            reactive_induction = prover._induce_proposed_rules_locally(
                working_facts,
                effective_targets,
                device,
            )
            result.induction_stats = merge_induction_stats(
                result.induction_stats,
                reactive_induction,
            )

    if cycle_recent_rules:
        prover._extend_recent_abduced_rules(cycle_recent_rules)

    if prover.training and prover._step % 10 == 0:
        result.vem_self_loss = prover.vem.self_supervised_loss(device)

    return result

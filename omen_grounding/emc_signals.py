from __future__ import annotations

import math
from typing import Any, Dict, Mapping


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if math.isnan(numeric) or math.isinf(numeric):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _normalized_count(value: Any, scale: float) -> float:
    try:
        numeric = max(float(value), 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.0:
        return 0.0
    return _clip01(math.log1p(numeric) / math.log1p(max(scale, 1.0)))


def grounding_emc_features(metadata: Mapping[str, Any] | None) -> Dict[str, float]:
    """
    Derive bounded grounding-control signals for EMC from task metadata.

    These are control priors, not truth judgments. They help EMC decide when
    grounding evidence is uncertain enough to justify extra recall,
    verification, or abduction pressure.
    """
    meta = metadata or {}

    uncertainty = _clip01(meta.get("grounding_uncertainty", 0.0))
    support = _clip01(
        meta.get(
            "grounding_support_ratio",
            0.5 * (
                _clip01(meta.get("grounding_graph_support", 0.0))
                + _clip01(meta.get("grounding_compiled_coverage", 0.0))
            ),
        )
    )
    world_state_acceptance = _clip01(meta.get("grounding_world_state_acceptance_ratio", support))
    verifier_world_support = _clip01(meta.get("verifier_world_model_support", support))
    verifier_temporal_support = _clip01(meta.get("verifier_temporal_consistency", support))
    verifier_world_conflict = _clip01(meta.get("verifier_world_model_conflict", 0.0))
    verifier_temporal_conflict = _clip01(meta.get("verifier_temporal_conflict", 0.0))
    verification_support = _clip01(
        meta.get(
            "grounding_verification_support",
            (0.42 * world_state_acceptance)
            + (0.22 * verifier_world_support)
            + (0.16 * verifier_temporal_support)
            + (0.12 * support)
            + (0.05 * (1.0 - verifier_world_conflict))
            + (0.03 * (1.0 - verifier_temporal_conflict)),
        ),
        default=support,
    )
    source_confidence = _clip01(meta.get("source_confidence", 1.0), default=1.0)
    mixed_profile = _clip01(meta.get("source_profile_mixed", 0.0))
    unknown_profile = _clip01(meta.get("source_profile_unknown", 0.0))
    modality_ambiguity = max(1.0 - source_confidence, mixed_profile, unknown_profile)
    parser_disagreement = _clip01(
        meta.get("grounding_parser_disagreement", 1.0 - _clip01(meta.get("grounding_parser_agreement", 1.0))),
    )
    if "grounding_memory_recall_instability" in meta:
        memory_recall_instability = _clip01(meta.get("grounding_memory_recall_instability", 0.0))
    elif "grounding_memory_corroboration" in meta:
        memory_recall_instability = _clip01(1.0 - _clip01(meta.get("grounding_memory_corroboration", 0.0)))
    else:
        memory_recall_instability = 0.0
    contradiction_density = _clip01(
        meta.get(
            "grounding_contradiction_density",
            max(
                _clip01(meta.get("grounding_conflict_pressure", 0.0)),
                _clip01(meta.get("grounding_world_state_contradiction_pressure", 0.0)),
            ),
        )
    )
    coreference_pressure = _clip01(meta.get("grounding_coreference_pressure", 0.0))
    world_model_mismatch = _clip01(
        meta.get(
            "grounding_world_model_mismatch",
            max(
                contradiction_density,
                _clip01(meta.get("verifier_world_model_conflict", 0.0)),
                _clip01(meta.get("verifier_temporal_conflict", 0.0)),
            ),
        )
    )

    memory_grounding = _normalized_count(meta.get("memory_grounding_records", 0.0), scale=4.0)
    trace_grounding = _normalized_count(meta.get("trace_grounding_records", 0.0), scale=6.0)
    interlingua_grounding = _normalized_count(meta.get("trace_interlingua_records", 0.0), scale=6.0)
    grounding_memory_signal = max(
        memory_grounding,
        trace_grounding,
        interlingua_grounding,
        _clip01(meta.get("grounding_memory_corroboration", 0.0)),
    )

    proof_instability = _clip01(
        meta.get(
            "grounding_proof_instability",
            (0.40 * _clip01(meta.get("grounding_world_state_branching_pressure", 0.0)))
            + (0.22 * _clip01(meta.get("grounding_world_state_contradiction_pressure", 0.0)))
            + (0.16 * _clip01(meta.get("grounding_repair_pressure", meta.get("verification_repair_pressure", 0.0))))
            + (0.12 * _clip01(meta.get("grounding_hidden_cause_pressure", 0.0)))
            + (0.10 * parser_disagreement),
        )
    )
    hypothesis_branching = _clip01(
        meta.get(
            "grounding_hypothesis_branching_pressure",
            max(
                _clip01(meta.get("grounding_world_state_branching_pressure", 0.0)),
                _clip01(meta.get("grounding_world_state_hypothetical_ratio", 0.0)),
                0.75 * _clip01(meta.get("grounding_hidden_cause_pressure", 0.0)),
            ),
        )
    )
    counterfactual_pressure = _clip01(
        meta.get(
            "grounding_counterfactual_pressure",
            (0.60 * hypothesis_branching) + (0.40 * world_model_mismatch),
        )
    )
    recall_readiness = _clip01(
        uncertainty
        * max(
            grounding_memory_signal,
            (0.40 * (1.0 - memory_recall_instability))
            + (0.35 * counterfactual_pressure)
            + (0.25 * coreference_pressure),
        )
    )
    verification_repair = _clip01(
        meta.get("grounding_repair_pressure", meta.get("verification_repair_pressure", 0.0))
    )
    hidden_cause_pressure = _clip01(
        meta.get("grounding_hidden_cause_pressure", meta.get("verification_hidden_cause_pressure", 0.0))
    )
    conflict_pressure = _clip01(
        meta.get("grounding_conflict_pressure", meta.get("verification_conflict_pressure", 0.0))
    )
    world_state_branching = _clip01(
        meta.get(
            "grounding_world_state_branching_pressure",
            meta.get("grounding_world_state_hypothetical_ratio", 0.0),
        )
    )
    world_state_contradiction = _clip01(
        meta.get(
            "grounding_world_state_contradiction_pressure",
            meta.get("grounding_world_state_conflict_ratio", 0.0),
        )
    )
    verification_pressure = _clip01(
        (uncertainty * (1.0 - max(support, verification_support)))
        + (0.22 * verification_repair)
        + (0.12 * conflict_pressure)
        + (0.18 * world_state_contradiction)
        + (0.16 * proof_instability)
        + (0.12 * contradiction_density)
        + (0.10 * world_model_mismatch)
        + (0.08 * parser_disagreement)
        + (0.06 * coreference_pressure)
    )
    abduction_pressure = _clip01(
        (verification_pressure * (1.0 - 0.5 * recall_readiness))
        + (0.30 * hidden_cause_pressure)
        + (0.18 * world_state_contradiction)
        + (0.10 * world_state_branching)
        + (0.18 * counterfactual_pressure)
        + (0.12 * hypothesis_branching)
        + (0.10 * proof_instability)
        + (0.06 * world_model_mismatch)
    )
    control_pressure = _clip01(
        max(
            uncertainty,
            modality_ambiguity,
            verification_repair,
            world_state_branching,
            world_state_contradiction,
            parser_disagreement,
            memory_recall_instability,
            proof_instability,
            contradiction_density,
            coreference_pressure,
            world_model_mismatch,
            counterfactual_pressure,
        ) * (1.0 - 0.5 * max(support, verification_support))
    )

    return {
        "grounding_uncertainty": uncertainty,
        "grounding_support": max(support, verification_support),
        "grounding_ambiguity": modality_ambiguity,
        "grounding_parser_disagreement": parser_disagreement,
        "grounding_memory_signal": grounding_memory_signal,
        "grounding_memory_recall_instability": memory_recall_instability,
        "grounding_proof_instability": proof_instability,
        "grounding_contradiction_density": contradiction_density,
        "grounding_coreference_pressure": coreference_pressure,
        "grounding_world_model_mismatch": world_model_mismatch,
        "grounding_hypothesis_branching_pressure": hypothesis_branching,
        "grounding_counterfactual_pressure": counterfactual_pressure,
        "grounding_recall_readiness": recall_readiness,
        "grounding_verification_pressure": verification_pressure,
        "grounding_abduction_pressure": abduction_pressure,
        "grounding_control_pressure": control_pressure,
        "grounding_world_state_branching_pressure": world_state_branching,
        "grounding_world_state_contradiction_pressure": world_state_contradiction,
    }

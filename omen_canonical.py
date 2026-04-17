from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Tuple


@dataclass(frozen=True)
class CanonicalArchitectureSpec:
    stack_id: str
    entrypoint: str
    public_module: str
    config_entrypoint: str
    repository_axis: str
    z_semantics: str
    world_model_mode: str
    online_cycle_mode: str
    legacy_modules: Tuple[str, ...]


@dataclass(frozen=True)
class RepositoryAxisSpec:
    axis_id: str
    public_module: str
    canonical_surface: Tuple[str, ...]
    canonical_support_modules: Tuple[str, ...]
    legacy_reference_modules: Tuple[str, ...]
    research_extension_modules: Tuple[str, ...]
    benchmark_entrypoints: Tuple[str, ...]
    test_entrypoints: Tuple[str, ...]

    def module_role(self, module_path: str) -> str:
        normalized = module_path.replace("\\", "/").lstrip("./")
        if normalized in self.canonical_surface:
            return "canonical_surface"
        if normalized in self.canonical_support_modules:
            return "canonical_support"
        if normalized in self.legacy_reference_modules:
            return "legacy_reference"
        if normalized in self.research_extension_modules:
            return "research_extension"
        return "unclassified"


CANONICAL_OMEN_SPEC = CanonicalArchitectureSpec(
    stack_id="omen_scale_world_graph",
    entrypoint="omen_scale.OMENScale",
    public_module="omen.OMEN",
    config_entrypoint="omen.OMENConfig",
    repository_axis="omen_scale_single_canon_repository",
    z_semantics="graph_first_world_state_with_dense_decoder_view",
    world_model_mode="execution_trace_primary_with_graph_fallback",
    online_cycle_mode="eval_capable_symbolic_online_learning",
    legacy_modules=(
        "omen_v2.py",
        "omen_tensor_unify.py",
    ),
)


OMEN_REPOSITORY_AXIS = RepositoryAxisSpec(
    axis_id=CANONICAL_OMEN_SPEC.repository_axis,
    public_module="omen.py",
    canonical_surface=(
        "omen.py",
        "omen_canonical.py",
        "omen_scale.py",
        "omen_scale_config.py",
    ),
    canonical_support_modules=(
        "omen_ast_multilang.py",
        "omen_data.py",
        "omen_emc.py",
        "omen_net_tokenizer.py",
        "omen_osf.py",
        "omen_osf_decoder.py",
        "omen_osf_intent.py",
        "omen_osf_meta.py",
        "omen_osf_planner.py",
        "omen_osf_simulator.py",
        "omen_perceiver.py",
        "omen_prolog.py",
        "omen_saliency.py",
        "omen_world_model.py",
        "omen_symbolic/__init__.py",
        "omen_symbolic/controller.py",
        "omen_symbolic/execution_trace.py",
        "omen_symbolic/executor.py",
        "omen_symbolic/integration.py",
        "omen_symbolic/memory_index.py",
        "omen_symbolic/universal_bits.py",
        "omen_symbolic/world_graph.py",
    ),
    legacy_reference_modules=CANONICAL_OMEN_SPEC.legacy_modules,
    research_extension_modules=(
        "omen_symbolic/aesthetic_engine.py",
        "omen_symbolic/analogy_engine.py",
        "omen_symbolic/counterfactual_engine.py",
        "omen_symbolic/creative_cycle.py",
        "omen_symbolic/creative_types.py",
        "omen_symbolic/hypergraph_gnn.py",
        "omen_symbolic/intrinsic_engine.py",
        "omen_symbolic/ontology_engine.py",
        "omen_symbolic/rule_graph.py",
    ),
    benchmark_entrypoints=(
        "benchmarks/benchmark_omen_scale_eval.py",
    ),
    test_entrypoints=(
        "tests/test_benchmark_protocol.py",
        "tests/test_canonical_stack_protocol.py",
        "tests/test_online_symbolic_learning_eval.py",
        "tests/test_transfer_suite_protocol.py",
    ),
)


def canonical_omen_spec() -> CanonicalArchitectureSpec:
    return CANONICAL_OMEN_SPEC


def repository_axis() -> RepositoryAxisSpec:
    return OMEN_REPOSITORY_AXIS


def canonical_module_role(module_path: str) -> str:
    return OMEN_REPOSITORY_AXIS.module_role(module_path)


def inject_canonical_metadata(payload: MutableMapping[str, object]) -> MutableMapping[str, object]:
    payload["canonical_stack"] = CANONICAL_OMEN_SPEC.stack_id
    payload["canonical_entrypoint"] = CANONICAL_OMEN_SPEC.entrypoint
    payload["canonical_public_module"] = CANONICAL_OMEN_SPEC.public_module
    payload["canonical_config_entrypoint"] = CANONICAL_OMEN_SPEC.config_entrypoint
    payload["canonical_repository_axis"] = CANONICAL_OMEN_SPEC.repository_axis
    payload["canonical_z_semantics"] = CANONICAL_OMEN_SPEC.z_semantics
    payload["canonical_world_model_mode"] = CANONICAL_OMEN_SPEC.world_model_mode
    payload["canonical_online_cycle_mode"] = CANONICAL_OMEN_SPEC.online_cycle_mode
    return payload


def inject_repository_axis_metadata(
    payload: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    payload["canonical_surface_modules"] = OMEN_REPOSITORY_AXIS.canonical_surface
    payload["canonical_support_modules"] = OMEN_REPOSITORY_AXIS.canonical_support_modules
    payload["legacy_reference_modules"] = OMEN_REPOSITORY_AXIS.legacy_reference_modules
    payload["research_extension_modules"] = OMEN_REPOSITORY_AXIS.research_extension_modules
    return payload

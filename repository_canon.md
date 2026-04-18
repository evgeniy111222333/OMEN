# Repository Canon

This repository now treats `omen_scale.py` as the single canonical runtime core.

## Public canonical surface

- `omen.py`
- `omen.OMEN`
- `omen.OMENConfig`
- `omen.build_omen(...)`

External scripts, benchmarks, and future integrations should import the runtime through `omen.py` instead of choosing between historical modules.

## Canonical core

- `omen_scale.py`
- `omen_scale_config.py`
- `omen_canonical.py`

These files define the active architecture described by `concept.md`: graph-native perception, graph-primary world state, context-enriched graph-derived dense views, graph-native world transitions, execution-trace-first supervision, and an eval-capable symbolic online cycle.

The canonical `z` contract is now graph-primary:

- `out["z"]` is the structured `CanonicalWorldState`
- `out["z_dense"]` is the dense decoder readout derived from that graph state

## Canonical support modules

The following modules are part of the canonical stack but are not separate competing runtimes:

- `omen_world_model.py`
- `omen_data.py`
- `omen_perceiver.py`
- `omen_prolog.py`
- `omen_saliency.py`
- `omen_net_tokenizer.py`
- `omen_emc.py`
- `omen_osf*.py`
- `omen_symbolic/world_graph.py`
- `omen_symbolic/execution_trace.py`
- `omen_symbolic/integration.py`
- `omen_symbolic/memory_index.py`
- `omen_symbolic/universal_bits.py`
- `omen_symbolic/creative_cycle.py`
- `omen_symbolic/creative_types.py`
- `omen_symbolic/analogy_engine.py`
- `omen_symbolic/counterfactual_engine.py`
- `omen_symbolic/aesthetic_engine.py`
- `omen_symbolic/ontology_engine.py`
- `omen_symbolic/intrinsic_engine.py`
- `omen_symbolic/rule_graph.py`
- `omen_symbolic/hypergraph_gnn.py`

## Legacy and research layers

- `omen_v2.py` and `omen_tensor_unify.py` are legacy reference modules.
- Creative and ontology-oriented modules under `omen_symbolic/` are part of the canonical support stack, not alternative runtimes.

## Repository rule

If a new benchmark, training script, or integration needs "the OMEN model", it should target `omen.OMEN` unless there is an explicit legacy ablation reason.

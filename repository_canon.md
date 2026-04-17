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

These files define the active architecture described by `concept.md`: world-graph-grounded latent state, execution-trace-first supervision, and an eval-capable symbolic online cycle.

The canonical `z` contract is now graph-first:

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

## Legacy and research layers

- `omen_v2.py` and `omen_tensor_unify.py` are legacy reference modules.
- Creative and ontology-oriented modules under `omen_symbolic/` are research extensions, not alternative repository canons.

## Repository rule

If a new benchmark, training script, or integration needs "the OMEN model", it should target `omen.OMEN` unless there is an explicit legacy ablation reason.

# Stage 2 Train-Fast Optimization Report

Date: 2026-04-18
Repository: `E:\omen`
Focus: accelerate Stage 2 training on GPU without removing symbolic or OSF capabilities

## 1. Executive Summary

The main Stage 2 slowdown was not OSF. The dominant cost was EMC symbolic maintenance during training, especially:

- `continuous_hypothesis_cycle(...)`
- reactive `abduce_and_learn(...)`

These paths were being executed inside the Stage 2 forward pass even though they primarily serve as background symbolic maintenance for future batches rather than immediate current-batch decoding quality.

I implemented two safe train-time optimizations for `metric_profile="train_fast"`:

1. Cadence scheduling for heavy EMC symbolic maintenance.
2. Reduced candidate budget for `continuous_hypothesis_cycle(...)` on due `train_fast` maintenance steps.

`full` behavior remains unchanged.

Result on the same smoke benchmark:

- Stage 2 wall time: `34.643s -> 16.598s`
- Stage 2 forward: `7070.2 -> 3245.4 ms/batch`
- Stage 2 total: `8654.2 -> 4146.1 ms/batch`

That is roughly:

- `-52.1%` Stage 2 wall time
- `-54.1%` Stage 2 forward time
- `-52.1%` total batch time

## 2. Root Cause Analysis

## 2.1 What was originally suspected

The user concern was that Stage 2 was too slow, with special suspicion around Stage 2 symbolic/planning paths.

OSF was investigated, but it was not the dominant cost on the measured training path.

## 2.2 What the measurements showed

Measured Stage 2 on GPU with:

- config: `OMENScaleConfig.demo()`
- dataset: synthetic, `n=48`
- Stage 1: `n_steps=2`
- Stage 2: `n_epochs=1`, `batch_size=2`, `max_batches_per_epoch=4`
- AMP: disabled for profiling stability

Initial Stage 2 throughput breakdown:

- `forward ~7070.2 ms/batch`
- `backward ~1546.2 ms/batch`
- `opt+step ~37.1 ms/batch`
- `total ~8654.2 ms/batch`

This immediately showed that the bottleneck was forward, not optimizer stepping.

## 2.3 EMC dominated the forward pass

Single-step timing showed:

- forward: `~2267.3 ms`
- backward: `~1436.8 ms`
- optimizer: `~49.6 ms`
- EMC inside forward: `~1420.9 ms`
- OSF inside forward: `~15.9 ms`

So the main problem was EMC symbolic work, not OSF.

## 2.4 Fine-grained EMC profiling

The heaviest EMC sub-components were:

- `prover.continuous_hypothesis_cycle ~707.0 ms`
- `prover.abduce_and_learn ~283.0 ms`
- `prover.ground ~106.7 ms` across repeated calls

Important observation:

- `continuous_hypothesis_cycle(...)` and the reactive post-episode abduction run after the main symbolic state for the current batch has already been constructed.
- They mostly update rules and symbolic memory for future batches.
- Therefore, they are excellent candidates for schedule-based optimization during `train_fast`.

## 3. What Was Implemented

## 3.1 Optimization A: cadence-based EMC background maintenance

Files:

- `omen_scale_config.py`
- `omen_scale.py`
- `omen_emc.py`

Added config:

- `emc_train_fast_maintenance_every: int = 4`

Meaning:

- In `train_fast`, heavy symbolic maintenance runs only every `N` maintenance opportunities instead of every batch.
- First maintenance step is still allowed.
- `full` mode is unchanged.

Behavioral rule:

- If `metric_profile != "train_fast"`, no cadence reduction is applied.
- If `metric_profile == "train_fast"`, heavy background symbolic maintenance runs only on due steps.

Protected behavior:

- If task context explicitly requests abduction via `trigger_abduction`, reactive abduction is still allowed immediately, even on a non-due step.

Why this is safe:

- The skipped work is background maintenance for future reasoning state.
- It is not required to form the current batch’s already-materialized symbolic output.

## 3.2 Optimization B: reduced continuous-cycle budget on due train_fast steps

Files:

- `omen_scale_config.py`
- `omen_emc.py`
- `omen_prolog.py`

Added config:

- `emc_train_fast_cycle_trace_candidates: int = 2`
- `emc_train_fast_cycle_contextual: int = 2`
- `emc_train_fast_cycle_neural: int = 2`
- `emc_train_fast_cycle_max_repairs: int = 1`

Previously, due cycle steps used the prover defaults:

- trace candidates: `4`
- contextual candidates: `4`
- neural candidates: `4`
- total budget: `12`
- repairs: `2`

Now, for `train_fast` due steps:

- trace candidates: `2`
- contextual candidates: `2`
- neural candidates: `2`
- total budget: `6`
- repairs: `1`

This change is applied only to `train_fast`.

`full` keeps the original larger symbolic maintenance budget.

## 3.3 New telemetry added

Added or surfaced in `prover.last_forward_info`:

- `emc_symbolic_maintenance_due`
- `emc_symbolic_maintenance_interval`
- `emc_reactive_abduction_executed`
- `emc_reactive_abduction_cadenced_skip`
- `cycle_active`
- `cycle_executed`
- `cycle_cadenced_skip`
- `cycle_candidate_budget`

Why this matters:

- You can now inspect whether a batch actually executed heavy symbolic maintenance.
- You can see whether reactive abduction was executed or cadenced away.
- You can verify what candidate budget the continuous cycle used.

## 4. Code Areas Changed

## 4.1 `omen_scale_config.py`

Added train-fast symbolic maintenance tuning fields:

- maintenance cadence
- train-fast cycle candidate caps
- train-fast cycle repair cap

Purpose:

- make the optimization explicit and configurable
- avoid hardcoding training-speed heuristics directly into logic

## 4.2 `omen_scale.py`

Change:

- `fast_mode=fast_metrics` is passed into `self.emc.run_episode(...)`

Purpose:

- lets EMC know whether the current forward call is in `train_fast`

## 4.3 `omen_emc.py`

Changes:

- added `fast_mode` to `run_episode(...)`
- added cadence logic for background symbolic maintenance
- allowed immediate triggered abduction even on non-due steps
- passed reduced cycle budgets into `prover.continuous_hypothesis_cycle(...)` when in `train_fast`
- added telemetry fields for maintenance execution/skips and cycle budget

Purpose:

- reduce Stage 2 training latency without deleting symbolic mechanisms

## 4.4 `omen_prolog.py`

Changes:

- `continuous_hypothesis_cycle(...)` now accepts optional per-call overrides:
  - `max_trace_candidates`
  - `max_contextual`
  - `max_neural`
  - `max_repairs`

Purpose:

- lets `train_fast` run a smaller maintenance budget
- keeps prover defaults intact for full symbolic mode

## 4.5 `tests/test_emc_gap_protocol.py`

Added tests for:

- maintenance skip on non-due `train_fast` steps
- forced reactive abduction still executing when explicitly triggered
- due-step maintenance still executing
- reduced cycle budget being passed through on due `train_fast` steps

Purpose:

- verify speed optimization logic without silently breaking symbolic protocol behavior

## 5. Measured Results

## 5.1 Benchmark setup

Benchmark used for all end-to-end comparisons:

- config: `demo`
- device: CUDA
- Stage 1: `2` steps
- Stage 2: `1` epoch
- `batch_size=2`
- `max_batches_per_epoch=4`
- synthetic dataset size: `48`

This is not a final quality benchmark. It is a controlled speed/regression benchmark for comparing code paths.

## 5.2 Baseline before any optimization

Stage 2 end-to-end:

- wall time: `34.643s`
- forward: `7070.2 ms/batch`
- backward: `1546.2 ms/batch`
- opt+step: `37.1 ms/batch`
- total: `8654.2 ms/batch`

## 5.3 After Optimization A only

Stage 2 end-to-end:

- wall time: `20.071s`
- forward: `4054.0 ms/batch`
- backward: `921.1 ms/batch`
- opt+step: `24.6 ms/batch`
- total: `5001.0 ms/batch`

Improvement vs baseline:

- wall time: `-42.1%`
- forward: `-42.7%`
- total: `-42.2%`

## 5.4 After Optimization A + B

Stage 2 end-to-end:

- wall time: `16.598s`
- forward: `3245.4 ms/batch`
- backward: `863.2 ms/batch`
- opt+step: `36.7 ms/batch`
- total: `4146.1 ms/batch`
- throughput: `61 tok/s`

Improvement vs original baseline:

- wall time: `34.643s -> 16.598s`
- forward: `7070.2 -> 3245.4 ms/batch`
- total: `8654.2 -> 4146.1 ms/batch`

Relative improvement vs original baseline:

- wall time: `-52.1%`
- forward: `-54.1%`
- total: `-52.1%`

## 5.5 Due-step micro-profile for continuous cycle

Profile with:

- `emc_train_fast_maintenance_every = 1`

This forces maintenance on every step so the due-step cost can be measured directly.

Before Optimization B:

- cycle budget: `12`
- cycle checked: `8`
- cycle time: `698.3 ms`
- EMC time: `1044.2 ms`
- forward time: `1867.7 ms`

After Optimization B:

- cycle budget: `6`
- cycle checked: `4`
- cycle time: `428.9 ms`
- EMC time: `905.2 ms`
- forward time: `1690.7 ms`

Relative improvement on due-step cycle:

- cycle time: about `-38.6%`
- EMC time: about `-13.3%`
- full forward: about `-9.5%`

Interpretation:

- Cadence gave the biggest overall speedup.
- Reduced cycle budget further improved the remaining expensive due-step batches.

## 6. Validation and Regression Checks

Executed tests:

- `python -m pytest tests/test_emc_gap_protocol.py -q`
- `python -m pytest tests/test_osf_protocol.py tests/test_training_checkpoint_protocol.py tests/test_symbolic_cycle_eval.py -q`

Results:

- EMC gap protocol: `14 passed`
- OSF + training checkpoint + symbolic cycle eval: `23 passed`

Warnings observed:

- Pytest cache warnings related to the local Windows cache-path setup
- one existing deprecation warning from `omen_osf_planner.py`
- PyTorch scheduler deprecation warning unrelated to this optimization

No failing tests were observed in the targeted suites.

## 7. Why This Does Not Remove Capabilities

The key design constraint was:

- do not remove symbolic maintenance
- do not disable OSF
- do not delete EMC
- do not collapse the system into a simpler model

What changed is scheduling and budgeting, not feature removal.

### In `full`

Nothing is intentionally reduced.

### In `train_fast`

The system still does:

- EMC control
- symbolic grounding
- rule-based reasoning
- background symbolic maintenance
- reactive abduction

But it does them:

- less often for background maintenance
- with a smaller continuous-cycle candidate budget on due steps

This is exactly the right tradeoff for fast training iterations:

- keep the capability
- reduce per-batch symbolic overhead
- preserve the option to restore full symbolic intensity when needed

## 8. How To Use It

## 8.1 Default behavior

Stage 2 training already routes CUDA training through:

- `metric_profile="train_fast"`

So if you use the existing Stage 2 training path on GPU, these optimizations are active automatically.

## 8.2 Main config controls

### `emc_train_fast_maintenance_every`

Purpose:

- how often heavy background symbolic maintenance is executed during `train_fast`

Default:

- `4`

Meaning:

- `1` = every batch
- `2` = every second maintenance opportunity
- `4` = every fourth maintenance opportunity

When to increase:

- if Stage 2 is still too slow
- if you are doing frequent short training iterations
- if you care more about turnaround speed than maximum symbolic maintenance frequency

When to decrease:

- if you want more frequent symbolic maintenance during training
- if you suspect symbolic rule growth is too slow

### `emc_train_fast_cycle_trace_candidates`

Purpose:

- number of trace-derived candidates the continuous cycle evaluates on due `train_fast` steps

Default:

- `2`

### `emc_train_fast_cycle_contextual`

Purpose:

- number of contextual abduction candidates evaluated on due `train_fast` steps

Default:

- `2`

### `emc_train_fast_cycle_neural`

Purpose:

- number of neural relaxed candidates evaluated on due `train_fast` steps

Default:

- `2`

### `emc_train_fast_cycle_max_repairs`

Purpose:

- maximum repair candidates allowed during due `train_fast` continuous-cycle execution

Default:

- `1`

## 8.3 Recommended presets

### Preset A: balanced default

Use:

- `emc_train_fast_maintenance_every = 4`
- cycle caps = `2/2/2`
- repairs = `1`

Best for:

- normal GPU development
- iterative training/debugging
- frequent benchmark/test loops

### Preset B: more symbolic intensity

Use:

- `emc_train_fast_maintenance_every = 2`
- cycle caps = `3/3/3`
- repairs = `1` or `2`

Best for:

- when you want stronger symbolic maintenance during training
- when speed is important but you want less aggressive reduction

### Preset C: near-full behavior

Use:

- `emc_train_fast_maintenance_every = 1`
- cycle caps equal prover defaults, for example `4/4/4`
- repairs `2`

Best for:

- ablation studies
- profiling
- comparing train-fast against historical behavior

### Preset D: maximum fast iteration

Use:

- `emc_train_fast_maintenance_every = 6` or `8`
- cycle caps = `1/1/1`
- repairs = `0` or `1`

Best for:

- tight turnaround loops
- debugging unrelated subsystems
- smoke training when symbolic maintenance is not the primary thing being studied

## 8.4 Example configuration snippet

```python
from omen_scale_config import OMENScaleConfig

cfg = OMENScaleConfig.demo()

cfg.emc_train_fast_maintenance_every = 4
cfg.emc_train_fast_cycle_trace_candidates = 2
cfg.emc_train_fast_cycle_contextual = 2
cfg.emc_train_fast_cycle_neural = 2
cfg.emc_train_fast_cycle_max_repairs = 1
```

## 8.5 When to use `train_fast` vs `full`

Use `train_fast` when:

- doing Stage 2 training on GPU
- iterating on architecture/code
- validating regressions
- running repeated short experiments

Use `full` when:

- you want maximum symbolic maintenance every batch
- you are running final ablations or deep symbolic-quality investigations
- you explicitly want the most expensive reasoning behavior during training/eval

## 9. How To Inspect That It Is Working

During or after forward/training, inspect `prover.last_forward_info`.

Key fields:

- `emc_symbolic_maintenance_due`
  - `1.0` means this batch executed heavy maintenance
  - `0.0` means it was a cadence skip

- `cycle_cadenced_skip`
  - `1.0` means continuous cycle was skipped due to cadence

- `cycle_executed`
  - `1.0` means the cycle actually ran

- `cycle_candidate_budget`
  - shows the candidate budget used for that cycle run

- `emc_reactive_abduction_executed`
  - whether reactive abduction ran

- `emc_reactive_abduction_cadenced_skip`
  - whether reactive abduction was skipped by cadence

Example:

```python
info = model.prover.last_forward_info
print(info.get("emc_symbolic_maintenance_due"))
print(info.get("cycle_executed"))
print(info.get("cycle_candidate_budget"))
```

## 10. Practical Interpretation

This optimization strategy is not about pretending symbolic reasoning is cheap.

It accepts the real architecture:

- symbolic maintenance is expensive
- not every symbolic update needs to happen every batch
- fast training and full symbolic maintenance are different operating modes

The implemented approach keeps the system expressive while separating:

- current-batch reasoning needed now
- background symbolic improvement useful over time

That distinction is what unlocked the speedup.

## 11. Known Limits

The following heavy areas still exist:

- `continuous_hypothesis_cycle(...)` is still expensive on due steps
- reactive abduction still costs non-trivial time
- repeated symbolic grounding still contributes overhead

So there is still room for future work.

Potential next directions:

- smarter adaptive maintenance scheduling based on loss/uncertainty instead of fixed cadence
- adaptive cycle budget based on recent symbolic utility
- deeper caching/reuse inside prover cycle internals
- optional deferred symbolic maintenance outside the immediate training critical path

## 12. Final Conclusion

The Stage 2 slowdown was successfully reduced without stripping core system capabilities.

The important outcome is not just that Stage 2 is faster. It is that the speedup came from a structurally correct place:

- background symbolic maintenance is now treated as background
- full symbolic behavior remains available
- train-fast is now materially more usable for real iteration

If you want the current default recommendation, use:

- `emc_train_fast_maintenance_every = 4`
- `emc_train_fast_cycle_trace_candidates = 2`
- `emc_train_fast_cycle_contextual = 2`
- `emc_train_fast_cycle_neural = 2`
- `emc_train_fast_cycle_max_repairs = 1`

This gives the best balance found in this pass between speed and preserved symbolic behavior.

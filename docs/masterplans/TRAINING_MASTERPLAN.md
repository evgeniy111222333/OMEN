# OMEN Training Masterplan

## 1. Status and Intent

This document defines the long-term training system for OMEN.

It complements:

- `concept.md`
- `GROUNDING_MASTERPLAN.md`

It does not replace the canonical product concept. It specifies how the full
system should be trained if the goal is not a temporary prototype, but a path
toward maximal intelligence, robustness, controllability, and long-horizon
capability.

This document is normative in the following sense:

- it defines what training must optimize
- it defines what each subsystem must learn
- it defines what supervision must exist
- it defines what affects what
- it defines how modules exchange training signal
- it defines how online adaptation must work without collapsing the system

This document is intentionally ideal-oriented. It is not a description of only
what exists in the repository today. It is a design for the training regime
required to reach the intended final OMEN.

## 2. Core Thesis

OMEN must not be trained as:

`bytes -> opaque model -> answer`

OMEN must be trained as:

`bytes -> typed perception -> world-semantic grounding -> graph/world state -> memory -> symbolic reasoning -> verification -> planning/generation -> writeback`

The central training thesis is:

- semantics should be learned
- truth maintenance should be formalized
- abstraction should be compression-driven
- knowledge should be retained through memory and verified symbolic structure
- compute should be budgeted and learned
- online adaptation should exist, but only under explicit epistemic control

The training system must therefore optimize not just local prediction quality,
but the emergence of a world-grounded, revisable, memory-using,
reasoning-capable internal system.

## 3. Non-Negotiable Training Principles

### 3.1 Bytes First

The canonical ingress remains UTF-8 bytes. No fixed large token vocabulary may
be treated as the primary semantic substrate.

### 3.2 Graph-Primary Internal State

The primary internal state is the structured world state, not a single dense
latent vector. Dense states are downstream readouts over graph/world state.

### 3.3 Meaning Must Be Grounded Before Reasoning

The symbolic layer must not become a compensator for poor grounding. Upstream
meaning formation must be strong enough that symbolic reasoning receives
structured hypotheses, not loose text hints.

### 3.4 Learned Semantics, Formal Truth

The final balance must be:

- learned components absorb semantic burden
- parsers absorb formal structure where syntax is externally defined
- symbolic and verification layers absorb truth burden
- heuristics remain bounded scaffolding only

### 3.5 Execution-Grounded Learning

Reasoning quality must be anchored to execution traces, state transitions,
counterexamples, and target-fact success, not only to surface linguistic
correlation.

### 3.6 Unified Economic Objective

All major subsystems must live under one training economy:

- better explanation
- better compression
- better prediction
- lower useless complexity
- lower useless rule growth
- lower useless memory pressure
- lower useless compute cost

### 3.7 No Hard Wall Between Train and Think

The final system must support guarded online adaptation. However, online updates
must be controlled by verification, confidence, memory policy, and rollback
logic.

## 4. What Training Must Produce

Training is successful only if it produces all of the following:

- a stable byte-level compression and concept substrate
- typed perception over mixed inputs
- robust multilingual structural and linguistic grounding
- a semantic scene graph over entities, events, roles, claims, and goals
- a language-invariant canonical interlingua
- a probabilistic symbolic compiler that preserves alternatives
- a verification system that scores support, conflict, and repair actions
- a persistent world state with provenance-preserving writeback
- a memory system that improves future grounding and reasoning
- a symbolic system that can deduce, abduce, induce, and consolidate
- a world model that predicts transitions and supports counterfactual rollout
- an EMC policy that spends compute only when useful
- a generator that synthesizes from state, not from surface continuation alone

## 5. Repository-Level Module Map

The final training system should treat the repository as the following learning
surface.

### 5.1 Core Runtime

- `omen_scale.py`: outer runtime, loss assembly, stage coordination, integration
- `omen_scale_config.py`: training and runtime control surface
- `omen.py`: public surface

### 5.2 Grounding Stack

- `omen_grounding/source_routing.py`: soft routing and source profile inference
- `omen_grounding/text_semantics.py`: text segmentation and low-level extraction
- `omen_grounding/structural_scene.py`: structural grounding
- `omen_grounding/semantic_scene.py`: semantic scene construction
- `omen_grounding/interlingua.py`: canonical interlingua construction
- `omen_grounding/symbolic_compiler.py`: multi-hypothesis symbolic compilation
- `omen_grounding/verification.py`: support/conflict scoring
- `omen_grounding/verifier_stack.py`: validation and repair scheduling
- `omen_grounding/world_state_writeback.py`: grounded world-state writeback
- `omen_grounding/planner_state.py`: planner-facing grounded state

### 5.3 Symbolic and Memory Core

- `omen_prolog.py`: prover, VeM, induction, rule lifecycle
- `omen_symbolic/execution_trace.py`: trace-grounded supervision bridge
- `omen_symbolic/world_graph.py`: world graph and canonical world state
- `omen_symbolic/memory_index.py`: symbolic memory indexing
- `omen_symbolic/creative_cycle.py`: advanced symbolic synthesis and rule growth

### 5.4 Support Systems

- `omen_world_model.py`: transition/world modeling
- `omen_perceiver.py`: perception and state formation
- `omen_net_tokenizer.py`: NET substrate
- `omen_saliency.py`: attention-to-structure bridge
- `omen_emc.py`: meta-control over reasoning depth and compute spend

## 6. High-Level Influence Graph

This section answers the practical question: what influences what?

### 6.1 Forward Influence

- Byte carrier influences NET, routing, and all downstream grounding.
- NET influences perception features, latent state, memory hints, and symbolic
  concept facts.
- Typed perception influences parser selection, segmentation, ambiguity
  handling, and grounding compute allocation.
- Structural grounding influences linguistic grounding priors, scene graph
  assembly, verification support, and provenance quality.
- Linguistic grounding influences entity persistence, role extraction,
  claim/goal detection, and interlingua quality.
- Semantic scene graph influences interlingua, symbolic compilation, world graph
  enrichment, and planner state.
- Canonical interlingua influences compiled facts, candidate rules, deferred
  hypotheses, and multilingual invariance.
- Symbolic compilation influences prover inputs, world-state writeback,
  planner constraints, and memory candidate writes.
- Verification influences confidence, repair policy, write suppression,
  contradiction tracking, and online adaptation permissions.
- World-state writeback influences world graph, long-term symbolic memory,
  neural memory, and future grounding corroboration.
- Memory influences grounding, prover context, world prediction, planning, and
  curiosity-driven recall.
- World model influences verification via compatibility checks, EMC decisions,
  and counterfactual simulation.
- Prover influences symbolic state, candidate rules, contradiction localization,
  and target-fact success.
- EMC influences whether the system recalls memory, performs deeper proof
  search, triggers planning, or stops early.
- Generation influences memory hints, symbolic surprise, and future training
  traces.

### 6.2 Reverse Credit Assignment

Training must also push signal backward:

- generation error must affect grounded state formation, not only decoder weights
- proof failure must affect grounding and symbolic compilation quality
- verification conflict must affect upstream hypothesis calibration
- planner failure must affect grounding ambiguity handling and world-state
  consistency
- memory retrieval uselessness must affect write policy and memory embeddings
- world-model mismatch must affect both state formation and verification priors
- compute overuse must affect EMC and optionally grounding confidence thresholds

## 7. Unified Training Objective

The canonical outer objective remains:

`J_total = FE + A_aux`

Where:

- `FE` is the central free-energy / compression / predictive term
- `A_aux` contains stabilizing, guiding, and task-structuring terms

This document refines that objective for full-system training.

### 7.1 FE Term

The FE term should continue to include:

- byte/token prediction quality
- world-state prediction quality
- latent compression cost
- memory read cost
- symbolic complexity amortization
- rule complexity amortization

In expanded operational form:

`FE = B_token + B_world + B_latent + B_mem + B_rule_complexity`

This term keeps OMEN from becoming an unconstrained collection of specialist
heads without a unifying economy.

### 7.2 Auxiliary Term Families

The auxiliary term must be decomposed into explicit subsystems:

`A_aux = A_ground + A_world + A_memory + A_symbolic + A_meta + A_generation + A_stability`

Where:

- `A_ground` trains grounding layers L1-L8
- `A_world` trains world transition and causal consistency
- `A_memory` trains read, write, retention, and consolidation
- `A_symbolic` trains proof success, rule utility, and induction quality
- `A_meta` trains EMC, reasoning depth, and compute budget policy
- `A_generation` trains state-conditioned decoding and synthesis
- `A_stability` prevents collapse, drift, KB pollution, and miscalibration

### 7.3 Grounding Objective Family

Grounding needs its own explicit objective family:

`A_ground = w_route L_route + w_struct L_struct + w_ling L_ling + w_scene L_scene + w_inter L_inter + w_compile L_compile + w_verify L_verify + w_write L_write + w_calib L_calib`

This term is necessary because a high-level architectural wish for "better
grounding" is not enough. The system must know exactly what grounding quality
means and how it is rewarded.

### 7.4 Symbolic Objective Family

`A_symbolic = w_proof L_proof + w_target L_target_fact + w_abd L_abduction + w_ind L_induction + w_vem L_VeM + w_rule_life L_rule_lifecycle`

This trains the symbolic layer to be useful, not decorative.

### 7.5 Meta-Control Objective Family

`A_meta = w_emc L_emc_value + w_reason C_reason + w_stop L_stop_quality + w_budget L_budget_discipline`

This is the explicit answer to uncontrolled overthinking and uncontrolled
underthinking.

### 7.6 Adaptive Weight Scheduling

Weights must not be static through the whole lifecycle.

The training controller should adapt weights based on:

- stage of curriculum
- subsystem maturity
- calibration health
- rule pollution rate
- memory saturation
- grounding conflict rate
- compute budget utilization

The long-term policy is:

- early training emphasizes stability and representational formation
- middle training emphasizes structured grounding and symbolic usefulness
- late training emphasizes online adaptation, repair, planning, and synthesis

## 8. Grounding Learning Contract

This section closes the main gap left open by the grounding architecture alone:
how each grounding layer is actually trained.

### 8.1 L0 Unified Carrier

Trainable goals:

- robust encoding detection
- byte-span preservation
- normalization without semantic destruction

Primary supervision:

- exact reconstruction
- byte-span recovery
- corrupted-text repair detection

Losses:

- reconstruction loss
- span alignment loss
- normalization consistency loss

Influences:

- all provenance fidelity
- all later byte-to-object traceability

### 8.2 L1 Typed Perception and Segmentation

Trainable goals:

- language identification
- script identification
- modality classification
- subtype classification
- segment boundary detection
- parser candidate ranking
- ambiguity detection

Primary supervision:

- labeled modality and subtype corpora
- parser-validity feedback
- segmentation gold data
- mixed-document annotations

Losses:

- routing cross-entropy
- segment boundary F1 loss
- parser ranking loss
- calibration loss
- ambiguity retention loss

Influences:

- which downstream parser paths run
- how much trust is placed in each parser family
- how much compute grounding receives

### 8.3 L2 Structural Grounding

Trainable goals:

- typed structural unit extraction
- formal parser selection and confidence estimation
- parser disagreement retention
- field typing and relation slotting

Primary supervision:

- code ASTs
- schema-valid JSON and config data
- tables and records
- logs with known structure
- aligned clause and discourse segment sets

Losses:

- tree alignment loss
- structural span loss
- parser confidence loss
- field typing loss
- disagreement preservation loss

Influences:

- provenance quality
- scene graph authority
- verification structural support
- planner trust in downstream objects

### 8.4 L3 Linguistic Grounding

Trainable goals:

- multilingual token and lemma formation
- morphology
- POS and dependency structure
- clause decomposition
- mention detection
- coreference candidate generation
- discourse relation detection

Primary supervision:

- multilingual parsed corpora
- cross-lingual paraphrase sets
- coreference corpora
- dialogue corpora
- scientific and instructional prose

Losses:

- token/lemma alignment loss
- morphology loss
- dependency loss
- mention and coreference loss
- discourse loss
- multilingual invariance contrastive loss

Influences:

- entity persistence
- speaker attribution
- temporal and causal relation recovery
- semantic scene graph quality

### 8.5 L4 Semantic Scene Graph

Trainable goals:

- event extraction
- entity-role assignment
- modality and polarity detection
- quantifier and generic-vs-instance separation
- claim attribution
- explanation and hidden-slot exposure

Primary supervision:

- event-frame corpora
- semantic role labeling
- temporal relation corpora
- causal datasets
- dialogue attribution corpora
- synthetic semantic world tasks

Losses:

- entity identity loss
- event-frame loss
- role labeling loss
- temporal ordering loss
- causal relation loss
- claim attribution loss
- hidden-slot exposure loss

Influences:

- interlingua quality
- symbolic compilation quality
- hidden-cause abduction quality
- writeback safety

### 8.6 L5 Canonical Semantic Interlingua

Trainable goals:

- language-invariant meaning collapse when meaning is equivalent
- preservation of polarity, modality, quantification, and epistemic stance
- canonical predicate and role naming stability

Primary supervision:

- cross-language paraphrase pairs
- explanation-style variation sets
- scientific restatement pairs
- synthetic canonicalization tasks
- program-to-natural-language semantic alignment tasks

Losses:

- semantic contrastive loss
- paraphrase convergence loss
- translation invariance loss
- canonical frame reconstruction loss
- epistemic preservation loss

Influences:

- symbolic compiler precision
- multilingual reasoning transfer
- ontology growth cleanliness

### 8.7 L6 Probabilistic Symbolic Compiler

Trainable goals:

- retain multiple plausible interpretations
- compile candidate facts and candidate rules
- attach support sets and provenance
- calibrate confidence and defer weak claims

Primary supervision:

- target facts from execution traces
- proof success and proof failure
- contradiction data
- hidden-cause resolution data
- planner success and failure

Losses:

- compiled fact precision/recall loss
- candidate rule utility loss
- deferred-vs-committed decision loss
- confidence calibration loss
- alternative-hypothesis survival loss

Influences:

- prover quality
- verifier quality
- KB pollution rate
- planner branch quality

### 8.8 L7 Verification and Repair

Trainable goals:

- support/conflict estimation
- contradiction localization
- repair action selection
- hidden-cause trigger policy

Primary supervision:

- execution outcomes
- world-model consistency
- parser agreement
- memory corroboration
- human-labeled contradiction sets
- synthetic conflict benchmarks

Losses:

- support/conflict regression
- verification classification loss
- repair action policy loss
- contradiction localization loss
- calibration loss

Influences:

- write permissions
- online adaptation permissions
- memory write confidence
- planner constraint formation

### 8.9 L8 World-State and Memory Writeback

Trainable goals:

- decide create vs merge vs defer vs reject
- preserve provenance and confidence
- protect memory from pollution
- update entity identity over time

Primary supervision:

- long-horizon consistency
- memory usefulness feedback
- contradiction outcomes
- entity persistence labels

Losses:

- write decision loss
- merge accuracy loss
- provenance completeness loss
- future-utility loss

Influences:

- long-term knowledge continuity
- grounding corroboration on later episodes
- symbolic recall quality

## 9. Symbolic Learning Contract

The symbolic core must remain logically real. Training must not turn it into a
soft statistical decoration.

### 9.1 What Is Learned

The following are learned:

- clause proposal policy
- proof ordering and search policy
- soft unification support modules
- rule utility estimation
- abduction scoring
- induction scoring
- neural interfaces from grounded state to symbolic state

### 9.2 What Is Not Relaxed Away

The following remain first-class logical structure:

- terms
- variables
- unification
- rules
- epistemic status
- contradiction handling
- rule lifecycle

### 9.3 Symbolic Supervision Sources

The symbolic layer learns from:

- execution traces
- target facts
- counterexamples
- program semantics
- synthetic relational tasks
- proof success and failure
- future utility of rules

### 9.4 Symbolic Losses

Required losses:

- proof success loss
- target fact attainment loss
- abduction utility loss
- induction precision loss
- contradiction sensitivity loss
- rule lifecycle loss
- VeM loss
- proof-cost penalty

### 9.5 Rule Lifecycle

Every rule must travel through:

- `proposed`
- `verified`
- `contradicted`
- `retired` or `consolidated`

Training must reward:

- promotion of actually useful rules
- rejection of noisy rules
- removal of stale or contradicted rules

## 10. Memory Learning Contract

OMEN needs both neural memory and exact symbolic memory. Training must make
them useful, not merely present.

### 10.1 Neural Memory

Neural memory should learn:

- retrieval keys
- compression of useful episodic summaries
- value-weighted retention
- recall under sparse cues

Primary losses:

- retrieval accuracy
- future state improvement after recall
- gap reduction after recall
- memory read cost regularization

### 10.2 Exact Symbolic Memory

Exact symbolic memory should learn:

- what facts and rules are worth writing
- how to consolidate repeated evidence
- how to preserve provenance and epistemic status

Primary losses:

- exact recall utility
- pollution penalty
- contradiction contamination penalty
- consolidation gain

### 10.3 Memory Write Policy

Write policy must be trained jointly with verification and downstream utility.

The correct question is not:

`can this be stored?`

The correct question is:

`will storing this improve future grounding, reasoning, planning, or generation enough to justify memory pressure and risk?`

## 11. World Model, Gap Detection, and Curiosity

The world model is not optional decoration. It is one of the main sources of
internal truth pressure.

### 11.1 World Model

The world model should learn:

- graph-conditioned transition prediction
- latent transition prediction
- causal compatibility estimation
- counterfactual rollout support

Losses:

- transition prediction loss
- state anchoring loss
- causal mismatch loss
- rollout consistency loss

### 11.2 Epistemic Gap Detector

The gap detector should learn:

- where uncertainty is high
- where memory retrieval is likely useful
- where more proof depth is likely useful
- where clarification or alternative hypothesis retention is needed

Losses:

- uncertainty calibration
- value-of-information prediction
- counterfactual usefulness prediction

### 11.3 Curiosity Policy

Curiosity should not generate random activity. It should target:

- unexplained transitions
- recurring contradiction pockets
- compressible but under-modeled structure
- under-resolved entities and claims

## 12. EMC and Compute-Budget Training

EMC is the system's compute governor.

### 12.1 EMC Must Learn To Control

- extra proof steps
- memory recall depth
- planner invocation
- counterfactual rollout depth
- grounding re-analysis
- clarification or defer decisions

### 12.2 EMC Reward

EMC reward should combine:

- accuracy improvement
- contradiction reduction
- target-fact success
- planner success
- generation grounding fidelity
- negative compute cost
- negative latency cost
- negative instability cost

### 12.3 EMC Curriculum

Train EMC in stages:

1. heuristic/oracle imitation
2. supervised stop-go decisions
3. actor-critic with explicit budget reward
4. online budget adaptation under safety constraints

## 13. Generation and Synthesis Training

Generation must read from state. It must not become a fallback language model
that bypasses the whole architecture.

### 13.1 Generator Inputs

Generation should condition on:

- graph-centered world state
- symbolic state
- memory state
- program state
- planner state when relevant

### 13.2 Generation Objectives

- byte/token prediction
- grounded explanation quality
- plan validity
- program correctness when code is required
- attribution fidelity
- contradiction avoidance

### 13.3 Feedback To Upstream

Generation outcomes must feed training signal back into:

- grounding
- symbolic reasoning
- world model
- memory policy
- EMC

If the generator succeeds only by bypassing grounded state, the training system
must detect that and penalize it.

## 14. Data Engine

The ideal OMEN cannot be trained on raw text alone.

### 14.1 Required Data Families

- large UTF-8 text corpora
- multilingual corpora
- code repositories
- executable programs with traces
- logs and telemetry streams
- scientific prose
- instructions and procedures
- dialogue and multi-speaker material
- structured records, configs, tables, schemas
- synthetic relational worlds
- planning environments
- contradiction-rich corpora
- paraphrase and translation sets
- long-horizon episodic memory tasks

### 14.2 Data Must Contain

- states
- transitions
- causes
- goals
- counterexamples
- ambiguity
- contradictory evidence
- entity persistence challenges
- write-vs-do-not-write moments

### 14.3 Data Factory Principle

The final training system should not rely only on static datasets.

It should also generate:

- synthetic worlds with exact semantics
- counterfactual rollouts
- rule-transfer tasks
- hidden-cause challenge sets
- multilingual semantic equivalence sets
- long-range memory stress tests

## 15. Curriculum

The canonical three-stage curriculum from `concept.md` should be refined into a
deeper operational curriculum.

### 15.1 Stage 1A - Byte and NET Foundation

Focus:

- byte compression
- stable codebook formation
- anti-collapse
- robust reconstruction

Do not require full symbolic competence here.

### 15.2 Stage 1B - Typed Perception Foundation

Focus:

- routing
- segmentation
- parser proposal
- basic provenance

Promotion gate:

- reliable modality and subtype calibration
- good segment boundary quality

### 15.3 Stage 1C - Structural and Linguistic Foundation

Focus:

- formal structure extraction
- multilingual linguistic analysis
- parser disagreement retention

Promotion gate:

- structural accuracy
- linguistic transfer
- no catastrophic English-only bias

### 15.4 Stage 1D - Scene and Interlingua Foundation

Focus:

- event/entity persistence
- semantic scene graph
- interlingua invariance

Promotion gate:

- paraphrase convergence
- cross-language semantic agreement
- role and temporal quality

### 15.5 Stage 2A - Joint Runtime Training

Focus:

- NET
- perception
- world graph
- world model
- memory
- grounding stack
- symbolic prover

This is the first true integrated OMEN stage.

### 15.6 Stage 2B - Grounding-Symbolic Bridge Training

Focus:

- probabilistic symbolic compilation
- verification
- writeback
- proof success
- hidden-cause abduction

Promotion gate:

- compiled fact precision
- candidate rule usefulness
- low KB pollution

### 15.7 Stage 2C - EMC and Compute Economy

Focus:

- learned reasoning depth
- memory query policy
- planner invocation policy
- compute-vs-gain tradeoff

Promotion gate:

- same or better task quality at lower average reasoning cost

### 15.8 Stage 2D - State-Conditioned Generation and OSF

Focus:

- grounded decoding
- synthesis from state
- plan and code generation under verification pressure

Promotion gate:

- generator success must materially depend on grounded state

### 15.9 Stage 3A - Guarded Online Symbolic Adaptation

Focus:

- online rule proposal
- online repair
- contradiction-aware rule retention

Restriction:

- no unrestricted autonomous KB writes

### 15.10 Stage 3B - Guarded Online Grounding Adaptation

Focus:

- online calibration
- online grounding confidence repair
- memory-informed semantic refinement

Restriction:

- semantic drift must remain bounded by rollback and shadow evaluation

### 15.11 Stage 3C - Lifelong Episodic Learning

Focus:

- durable memory growth
- ontology growth
- skill-like competence retention
- long-horizon world continuity

## 16. Online Adaptation Policy

The final OMEN should adapt online, but only under controlled conditions.

### 16.1 What May Update Online

- memory content
- memory read/write policies
- rule statuses
- selected symbolic proposal heads
- EMC policies
- confidence calibrators
- grounding adapters

### 16.2 What Must Be Guarded

- core ontology remapping
- primary interlingua predicate inventory
- high-value verified rules
- global write thresholds
- world-state identity merges

### 16.3 Safe Online Modes

Required modes:

- shadow update mode
- propose-only mode
- reversible write mode
- bounded-trust online learning mode
- human-review escalation mode for high-impact changes

## 17. Heuristic Retirement Policy

Heuristics are acceptable only as temporary bounded support.

### 17.1 Acceptable Heuristic Roles

- coarse routing priors
- parser proposal
- deterministic format recognition
- diagnostics
- fallback extraction

### 17.2 Unacceptable Heuristic Roles

- owning deep natural-language meaning
- final multilingual equivalence judgement
- final persistent entity resolution
- final verified world writes
- planner-state creation from ambiguous prose

### 17.3 Retirement Rule

Every heuristic must satisfy all of the following:

- isolated implementation
- explicit status as fallback or scaffold
- measurable activation rate
- measurable disagreement with learned/formal modules
- replaceability by trained modules

## 18. Stability and Anti-Collapse Requirements

Training must explicitly defend against:

- heuristic dependence masquerading as competence
- English-only performance hidden under broad averages
- early collapse to one interpretation
- knowledge-base pollution
- confidence inflation
- memory hoarding without utility
- planner action over unverified state
- generator bypass of grounded state
- online self-reinforcement of false beliefs

Required defenses:

- calibration tracking
- contradiction tracking
- shadow evaluation
- rollback support
- provenance completeness checks
- memory pressure controls
- write-rate controls
- stage-wise graduation gates

## 19. Evaluation and Graduation Gates

A subsystem is not mature because it exists. It is mature only when it passes
gates.

### 19.1 Grounding Gates

- routing calibration
- structural accuracy
- multilingual transfer
- entity persistence
- event and role quality
- interlingua invariance
- compiled fact precision/recall
- hypothesis calibration

### 19.2 Symbolic Gates

- proof success
- target-fact hit rate
- contradiction sensitivity
- induction precision
- utility of abduced rules

### 19.3 Memory Gates

- recall usefulness
- future-task benefit
- pollution resistance
- consolidation quality

### 19.4 Compute-Economy Gates

- EMC stop quality
- proof depth efficiency
- memory query efficiency
- planner invocation efficiency

### 19.5 Final System Gates

- long-horizon stability
- multilingual grounded reasoning
- revision under contradiction
- plan validity from grounded state
- code correctness with trace-grounded understanding
- online adaptation without catastrophic drift

## 20. What the Final Mature System Should Look Like

At maturity, OMEN should exhibit the following training-shaped behavior:

- raw bytes enter the system without semantic collapse
- meaning is formed through learned grounding, not brittle pattern hacks
- multiple hypotheses survive until verification resolves them
- world state, memory, and symbolic state stay mutually informative
- rules are proposed, verified, contradicted, repaired, and consolidated
- compute is spent when useful, not by default
- generation reads from grounded state and can be audited back to provenance
- memory improves future competence instead of merely storing traces
- online learning exists, but under explicit epistemic control

## 21. Final Directive

The training goal for OMEN is not:

- higher next-token quality alone
- larger scale alone
- more rules alone
- more memory alone
- more tools alone

The training goal is:

to create a byte-grounded, world-grounded, memory-using, reasoning-capable,
verification-disciplined, compute-aware cognitive runtime whose competence grows
through learned semantics, exact symbolic structure, and controlled lifelong
adaptation.

If a proposed training change improves local benchmarks but weakens any of the
following:

- world grounding
- provenance
- symbolic truth discipline
- memory usefulness
- compute economy
- online stability

then that change should be treated as a regression, not progress.

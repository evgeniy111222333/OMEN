# OMEN Grounding Masterplan

## 1. Status and Intent

This document is a detailed grounding and semantic-ingestion masterplan for the
current OMEN repository.

It complements `concept.md`. It does not replace the canonical product concept.

For the deterministic authority model of routing, grounding, verification,
world-state writeback, memory interaction, planner ingress, and generation
state, this document should now be read together with
`DETERMINISTIC_RUNTIME_CONCEPT_UK.md`.

For rigid ownership boundaries between learned proposal layers and symbolic
truth-maintenance layers, this document should also be read together with
`NEURO_SYMBOLIC_BOUNDARY_UK.md`.

Its purpose is narrower and deeper:

- describe how grounding works in the current code
- identify what is already structurally correct
- identify what is still weak
- define the long-term target architecture for extremely strong grounding
- define interaction contracts between modules
- define how data should enter the system
- define how the system should interact with memory, symbolic reasoning, EMC,
  planning, and generation
- define how operators, datasets, and future tooling should work with the stack

This is a design document for the final trajectory, not a short-term patch note.

## 2. Core Thesis

OMEN should not treat raw text grounding as a thin preprocessing step.

Grounding is the bridge between:

- the byte stream
- the inferred world state
- the symbolic substrate
- long-term memory
- verification
- planning
- generation

If that bridge is shallow, the whole system looks weaker than it really is.

If that bridge becomes deep, multilingual, uncertainty-aware, and
world-semantic, the existing symbolic and world-state core can become
substantially more powerful without changing its fundamental identity.

The long-term target is therefore:

`bytes -> typed perception -> semantic scene graph -> canonical interlingua -> deterministic symbolic lowering with explicit epistemic scoring -> verification/repair -> persistent world state`

## 3. What Exists in the Current Code

This section is an audit of the current runtime, not a wish list.

### 3.1 Canonical Runtime Entry

The canonical runtime lives in `omen_scale.py`.

The key end-to-end flow is already structurally serious:

1. byte/token encoding and optional NET compression
2. perception world-graph construction
3. latent sampling
4. saliency extraction
5. graph grounding of state
6. neural memory retrieval
7. world rollout and epistemic-gap estimation
8. symbolic memory seeding and recall
9. symbolic task-context assembly
10. world-context attachment
11. prover task-context install
12. symbolic reasoning, EMC, and learning
13. generation from grounded state

This means OMEN already has the right macro-shape.

### 3.2 Source Routing Already Exists

`omen_scale.py` already defines `SourceRoutingDecision` and `_infer_source_routing(...)`.

The current router already does more than a trivial `code / text / other` split.

It produces:

- `language`
- `domain`
- `confidence`
- `evidence`
- `modality`
- `subtype`
- `verification_path`
- `profile`

This is the correct direction.

The current router also distinguishes:

- `code`
- `natural_text`
- `structured_text`
- `mixed`
- `unknown`

and maps these to verification paths such as:

- `ast_program_verification`
- `mixed_hybrid_verification`
- `scientific_claim_verification`
- `dialogue_state_verification`
- `log_trace_verification`
- `config_schema_verification`
- `table_consistency_verification`

This is a good skeleton.

### 3.3 Current Natural-Language Grounding Path

The current text-grounding path is centered around
`omen_symbolic/execution_trace.py`, especially `_ObservationTraceBuilder`.

That builder currently does shallow extraction from raw text by:

- splitting text into line or sentence-like segments
- tokenizing with ASCII-oriented regexes
- extracting simple structured pairs
- extracting simple binary relations
- extracting goal-like expressions
- marking simple negation-like counterexample cues
- converting extracted cues into trace atoms

It then produces a `SymbolicExecutionTraceBundle` with:

- `observed_facts`
- `target_facts`
- `transitions`
- `counterexamples`

This is useful, but it is not yet full semantic grounding.

### 3.4 Current Code Grounding Path

For code-like inputs, `omen_scale.py` already routes through:

- `_ast_facts_from_bytes(...)`
- `_ast_rules_from_bytes(...)`
- `_ast_trace_from_bytes(...)`

This means the code path can yield:

- AST-derived observed facts
- AST-derived verified rules inserted into the prover KB
- execution-trace-like supervision

This is one reason the code path is currently stronger than the natural-language
path.

### 3.5 Symbolic Task Context Is Already Rich

`omen_prolog.py` defines `SymbolicTaskContext`.

This object is already well designed for a serious system because it separates
provenance buckets:

- `observed_now_facts`
- `memory_derived_facts`
- `saliency_derived_facts`
- `net_derived_facts`
- `world_context_facts`
- `abduced_support_facts`
- `goal`
- `target_facts`
- `execution_trace`
- `metadata`

This is exactly the kind of substrate a final grounding system needs.

### 3.6 World Graph Is Already the Right Mid-Level Substrate

`omen_symbolic/world_graph.py` defines:

- `WorldGraphState`
- `WorldGraphBatch`
- `CanonicalWorldState`
- `WorldGraphEncoder`

The world graph already supports typed fact records such as:

- `observed`
- `trace_target`
- `saliency`
- `context`
- `memory`
- `net`
- `abduced`
- `goal`
- `target`

This means OMEN already has a graph-centered integration layer where richer
grounded units can live.

### 3.7 Symbolic Core Is Not the Main Weakness

`omen_prolog.py` already supports:

- deduction
- abduction
- induction
- epistemic rule status
- trace-aware rule scoring
- counterexample-sensitive repair
- continuous hypothesis cycles

The symbolic core is therefore not the main reason raw text scenarios fail.

The main weakness is upstream semantic grounding fidelity.

## 4. What Is Good Right Now

The current architecture already gets several important things right.

### 4.1 Correct Macro-Architecture

OMEN is not organized as:

`prompt -> opaque LM -> answer`

It is organized around world state, memory, symbolic task context, and
verification.

That is the right foundation.

### 4.2 Good Provenance Discipline

Facts are already separated by origin.

This is critical because long-term strong grounding requires the system to know
not only what it thinks, but also where that belief came from.

### 4.3 Good Verification Direction

The current router already exposes verification paths rather than one generic
"text path".

That principle should stay.

### 4.4 Good Integration Points

The repository already has natural attachment points for a stronger grounding
stack:

- source routing
- trace bundle generation
- symbolic task context composition
- world graph enrichment
- symbolic memory recall
- prover context installation
- EMC reasoning control

This means a major grounding upgrade can be integrated into the current system
rather than requiring a full rewrite.

## 5. What Is Weak Right Now

This section is intentionally direct.

### 5.1 Text Grounding Is Mostly Hint Extraction

The current natural-language path primarily extracts shallow cues.

It does not yet build a robust internal semantic scene.

In other words, the current path is closer to:

`text -> symbolic hints`

than to:

`text -> world-semantic state`

### 5.2 English and ASCII Bias

The current `_ObservationTraceBuilder` patterns are strongly biased toward:

- ASCII token shapes
- English relation verbs
- English goal markers
- English negation markers

That is not acceptable for a final system.

### 5.3 Weak Entity Persistence

The current path does not maintain strong persistent identity for:

- entities
- events
- roles
- temporal references
- pronouns and aliases

This means the system often sees isolated fragments instead of a stable world.

### 5.4 Weak Compositional Semantics

The current path does not fully represent:

- quantifiers
- modality
- conditionals
- causality as first-class structure
- temporal order across rich narratives
- counterfactual structure
- nested claims
- speaker attribution

This caps reasoning quality before the prover even begins.

### 5.5 Premature Collapse to One Interpretation

The current path tends to emit one small set of derived atoms.

A final strong grounding stack must preserve multiple hypotheses with
confidence, provenance, and later revision.

### 5.6 No Fully Native Semantic Interlingua Yet

The system still leans too much on surface forms.

A final architecture must make different phrasings converge into the same
canonical internal meaning representation when the meaning is the same.

## 6. Final Long-Term Goal

The long-term goal is not "better regexes".

The long-term goal is a world-semantic grounding stack with the following
properties:

- multilingual
- uncertainty-aware
- provenance-rich
- entity-persistent
- event-centric
- verification-driven
- ontology-extensible
- memory-compatible
- symbolic-compiler-friendly
- planner-compatible
- generation-compatible

The final stack should allow OMEN to treat natural language, scientific prose,
instructions, logs, dialogue, and mixed documents as world evidence rather than
as loose text fragments.

## 7. Final Target Architecture

The final architecture should be layered.

Not because layers are fashionable, but because each layer solves a distinct
problem and exposes a distinct contract.

### 7.1 Layer L0: Unified Carrier

Input is always bytes.

Responsibilities:

- raw byte preservation
- encoding detection when needed
- normalization without semantic destruction
- byte span tracking

Output:

- canonical byte sequence
- byte offsets
- basic document identity metadata

Rules:

- never destroy source provenance
- every higher-level object must be traceable back to byte spans

### 7.2 Layer L1: Typed Perception and Segmentation

This layer decides what kind of material the system is looking at.

Responsibilities:

- language identification
- script identification
- segment boundaries
- modality scoring
- subtype scoring
- parser candidates
- mixed-content detection

Output contract:

- document segments
- segment byte spans
- soft routing profile
- modality and subtype probabilities
- parser/schema candidates
- local confidence and ambiguity flags

This layer generalizes the current `SourceRoutingDecision`.

It must remain soft, not hard.

### 7.3 Layer L2: Structural Grounding

This layer extracts structural objects before deep semantics.

Responsibilities depend on subtype.

For code:

- parser trees
- symbol tables
- function/class/module boundaries
- executable traces when available

For structured text:

- records
- key-value pairs
- tables
- log entries
- schemas
- field typing

For natural text:

- sentence and clause boundaries
- paragraph roles
- discourse markers
- speaker turns
- citation regions

Output contract:

- structural units
- typed spans
- segment-local references
- parse confidence
- parser disagreements

### 7.4 Layer L3: Linguistic Grounding

This is where multilingual natural-language understanding becomes serious.

Responsibilities:

- tokenization that is not English-specific
- morphology
- lemma normalization
- part-of-speech
- dependency structure
- clause decomposition
- named-entity candidates
- coreference candidates
- discourse relations

This layer must work for multilingual text, including Ukrainian and mixed
language inputs.

Output contract:

- token objects
- lemma objects
- clause objects
- dependency arcs
- mention candidates
- coreference hypotheses
- grammatical features
- parser confidence

This layer should not yet decide full world meaning, but it should make later
semantic grounding much more reliable.

### 7.5 Layer L4: Semantic Scene Graph

This is the first layer that should represent meaning directly.

The basic objects are not tokens.

The basic objects are:

- entities
- events
- states
- relations
- roles
- times
- locations
- modalities
- negations
- quantifiers
- goals
- obligations
- explanations

This layer should produce an event-frame graph rather than a bag of extracted
relations.

Example of the intended internal form:

- entity `e_star_class`
- entity `e_planet_class`
- event `ev_generate_1`
- role `agent(ev_generate_1, e_star_class)`
- role `patient(ev_generate_1, e_planet_class)`
- quantification `generic_all`
- provenance spans
- confidence values

This is the layer currently missing in full form.

### 7.6 Layer L5: Canonical Semantic Interlingua

This layer converts semantic scene graphs into a language-invariant canonical
representation.

The same meaning expressed as:

- Ukrainian prose
- English prose
- scientific wording
- simplified explanation
- compact declarative syntax

should converge here.

This layer should standardize:

- predicate inventory
- event templates
- role names
- polarity
- tense/aspect when relevant
- modality
- quantification
- generic vs instance semantics
- epistemic stance

This is the core of multilingual invariance.

### 7.7 Layer L6: Probabilistic Symbolic Compiler

This layer compiles the interlingua into symbolic objects suitable for the
prover, world graph, memory, and planners.

Crucially, it must not force one interpretation too early.

It must support:

- multiple candidate facts
- candidate rules
- hidden-cause hypotheses
- confidence
- provenance
- conflict tags
- support sets

Outputs should include:

- compiled facts
- compiled candidate rules
- deferred hypotheses
- grounding diagnostics

The current system already has a strong symbolic downstream. This compiler is
the missing bridge that should feed it better inputs.

### 7.8 Layer L7: Verification and Repair

Grounding is not finished when a fact is compiled.

The fact must be checked.

Verification must include:

- parser agreement checks
- structural consistency checks
- temporal consistency checks
- contradiction checks
- trace agreement checks
- world-model compatibility checks
- memory support checks
- target/goal alignment checks

Repair actions must include:

- lower confidence
- keep multiple hypotheses alive
- generate clarification needs
- trigger abduction for hidden causes
- mark contradiction scope
- ask planners to consider alternative worlds

### 7.9 Layer L8: Persistent World State and Memory Writeback

After verification, accepted grounded objects should update:

- working world state
- world graph
- symbolic working memory
- long-term exact symbolic memory
- neural memory traces

Nothing should be written blindly.

Every write should include:

- source id
- span provenance
- timestamp or episode id
- confidence
- status
- relation to existing objects

### 7.10 Layer L9: Reasoning, Planning, EMC, and Generation

Only after grounding is strong enough should the higher layers act on it.

These layers should consume grounded state, not raw text.

Consumers:

- symbolic prover
- abduction cycle
- induction cycle
- EMC meta-controller
- counterfactual world simulator
- planners
- decoder/generator

The strongest version of OMEN should behave as:

`ground first -> reason second -> answer third`

not:

`guess answer first -> rationalize later`

### 7.11 Heuristic Policy

The final system must not be built on the assumption that heuristics disappear
completely.

It also must not be built on the assumption that heuristics are allowed to act
as the primary semantic engine.

The correct long-term policy is:

- learned models should do the main semantic grounding
- formal mechanisms should do truth maintenance, verification, and epistemic
  control
- heuristics should remain bounded support mechanisms

The rule is not:

`no heuristics`

The rule is:

`no heuristic should own meaning`

#### 7.11.1 What Heuristics Are Allowed To Do

Heuristics are acceptable for:

- coarse routing
- segmentation fallback
- parser or schema candidate proposal
- deterministic format recognition
- instrumentation and diagnostics
- safety fallback when a learned component is unavailable or unstable
- cheap candidate generation for later learned or formal verification
- benchmark harness convenience layers

These are support functions.

They are not final semantic authority.

#### 7.11.2 What Heuristics Must Not Do

Heuristics must not be the primary mechanism for:

- deep natural-language meaning extraction
- multilingual semantic equivalence
- persistent entity resolution
- causal interpretation
- hidden-cause explanation acceptance
- scientific claim grounding
- planner state creation from ambiguous prose
- final verified knowledge writes

If a heuristic creates something in these categories, it must enter the system
only as a low-status hint or proposal.

#### 7.11.3 Epistemic Status of Heuristic Outputs

Every heuristic output should be explicitly marked as one of:

- routing evidence
- parser prior
- candidate hint
- low-confidence proposal
- fallback extraction

It should not be written as `verified` unless a stronger downstream mechanism
confirms it.

In practical terms:

- heuristics may seed candidates
- learned grounding may refine them
- verification may accept or reject them

This is the correct hierarchy.

#### 7.11.4 Learned Components Must Absorb Semantic Burden

The final system should move semantic burden away from hand-coded patterns and
toward learned components for:

- cross-language invariance
- paraphrase robustness
- synonymy
- morphology-sensitive interpretation
- event and role induction
- discourse-sensitive interpretation
- mixed-document understanding

This is where the real scaling power comes from.

#### 7.11.5 Formal Components Must Absorb Truth Burden

The final system should move truth burden toward explicit mechanisms for:

- contradiction handling
- provenance tracking
- rule status control
- trace agreement
- state consistency
- world-model agreement
- revision under new evidence

This is where stability comes from.

#### 7.11.6 Deterministic Parsers Are Not "Bad Heuristics"

The architecture must distinguish between:

- ad hoc pattern heuristics
- deterministic structural parsers

A real JSON parser, AST parser, schema validator, or table parser is not the
same thing as a fragile regex heuristic.

Deterministic parsers are legitimate primary evidence sources for domains where
the syntax is formal and externally defined.

The anti-pattern is not "anything deterministic".

The anti-pattern is:

- language-specific brittle shortcut logic
- hidden semantic assumptions in surface patterns
- unverifiable handcrafted shortcuts that bypass the main pipeline

#### 7.11.7 Heuristics Must Be Replaceable

Every heuristic should be treated as temporary scaffolding unless it encodes a
stable formal rule of the medium itself.

Good heuristic design rules:

- isolate it
- label it
- measure it
- allow it to be bypassed
- allow it to be replaced by learned or parser-based modules later

No heuristic should become an invisible architectural dependency.

#### 7.11.8 Heuristics Must Be Calibrated Against Learned and Formal Signals

Whenever possible, heuristic outputs should be checked against:

- parser agreement
- learned confidence
- memory consistency
- world-model prediction
- symbolic contradiction checks

If disagreement is large, the heuristic should lose authority automatically.

#### 7.11.9 Long-Term Balance

The long-term mature system should look like this:

- semantics mostly from learning
- structure from parsers where formal syntax exists
- truth maintenance from symbolic and verification layers
- heuristics only for bounded support, fallback, and cheap priors

That is the correct non-naive balance.

It avoids both bad extremes:

- a brittle heuristic engine pretending to understand the world
- an uncontrolled black-box learner with weak verification discipline

#### 7.11.10 Repository-Level Implication

For this repository specifically, the current natural-language heuristics should
be treated as transitional infrastructure.

They are useful today because they provide:

- cheap routing support
- basic trace targets
- fallback symbolic hints

But they should gradually move toward a narrower role:

- support routing
- support fallback extraction
- support diagnostics
- support proposal seeding

and away from the role they implicitly occupy today:

- primary natural-language grounding engine

That shift is essential if OMEN is meant to become a genuinely strong
world-grounded system rather than a sophisticated heuristic front-end feeding a
strong symbolic core.

## 8. Canonical Objects the Final Stack Must Represent

The final stack should have explicit internal objects for the following.

### 8.1 Document Objects

- document id
- source bytes
- source metadata
- segment list
- language profile
- modality profile
- parser profile

### 8.2 Mention Objects

- span
- normalized text
- language
- candidate type
- confidence
- local discourse role

### 8.3 Entity Objects

- canonical id
- type
- aliases
- coreference cluster
- attributes
- provenance list
- confidence

### 8.4 Event Objects

- canonical id
- event type
- participants
- roles
- polarity
- modality
- temporal anchors
- causal links
- provenance list
- confidence

### 8.5 Claim Objects

- claim id
- proposition graph
- source speaker or narrator
- epistemic status
- confidence
- evidence set
- contradiction links

### 8.6 Rule Objects

- candidate rule
- support evidence
- origin
- confidence
- scope
- verification status
- counterexamples

### 8.7 World-State Objects

- active facts
- inactive facts
- hypothetical facts
- contradicted facts
- goal facts
- target facts
- plan state

## 9. Contracts Between Major Modules

This section is the most operational part of the design.

### 9.1 Source Router Contract

Input:

- raw bytes or normalized text

Output:

- modality distribution
- subtype distribution
- language distribution
- parser candidates
- verification-path candidates
- ambiguity flags

Consumers:

- structural parsers
- grounding orchestrator
- verification scheduler

The router must never be the final authority on truth. It only controls initial
evidence routing.

### 9.2 Structural Parser Contract

Input:

- segments plus routing profile

Output:

- parse objects
- parse confidence
- parse failures
- disagreement evidence

Consumers:

- semantic scene builder
- verification layer

### 9.3 Semantic Scene Builder Contract

Input:

- structural parses
- linguistic features
- discourse context

Output:

- entities
- events
- states
- relations
- coreference hypotheses
- temporal relations
- causal relations
- goal structures

Consumers:

- canonical interlingua normalizer
- world graph
- symbolic compiler

### 9.4 Canonical Interlingua Contract

Input:

- semantic scene graph

Output:

- canonical events
- canonical roles
- canonical predicates
- canonical polarity/modality
- semantic equivalence classes

Consumers:

- symbolic compiler
- memory deduplication
- ontology growth

### 9.5 Symbolic Compiler Contract

Input:

- canonical semantic objects
- memory hints
- current world context
- target context

Output:

- candidate Horn facts
- candidate clauses
- deferred hypotheses
- support links
- contradiction links
- confidence scores

Consumers:

- `SymbolicTaskContext`
- prover
- symbolic memory index
- world graph enrichment

### 9.6 Verification Layer Contract

Input:

- candidate symbolic objects
- traces
- memory recall
- world model predictions
- execution expectations

Output:

- accepted
- proposed
- contradicted
- uncertain
- clarification-needed

Consumers:

- KB status updater
- EMC
- abduction controller
- operator UI

### 9.7 Memory Contract

Memory must not only retrieve by embedding similarity.

It must support:

- entity-aware retrieval
- event-aware retrieval
- predicate-aware retrieval
- provenance-aware retrieval
- recency-aware retrieval
- contradiction-sensitive retrieval

The current `SymbolicMemoryIndex` is a strong starting point, but final memory
should retrieve on richer grounded objects, not only on sparse downstream atoms.

### 9.8 Prover Contract

The prover should receive:

- high-confidence observed facts
- proposed facts
- target facts
- execution traces
- world-context facts
- abduced support facts

It should not be forced to infer semantics that the grounding stack should have
resolved earlier.

### 9.9 EMC Contract

EMC should read grounding uncertainty directly.

It should use signals such as:

- modality ambiguity
- parser disagreement
- coreference uncertainty
- contradiction density
- world-model mismatch
- proof instability
- memory recall instability

EMC should then decide:

- stop early
- request more verification
- allow abduction
- allow induction
- request counterfactual rollout
- keep multiple hypotheses active

### 9.10 Planner Contract

The planner should consume world-state objects, not loose text.

It should know:

- resources
- goals
- world rules
- hypothetical rules
- destructive effects
- persistent effects
- uncertainty

This is especially important for counterfactual and sandbox worlds.

### 9.11 Generator Contract

The generator should not speak directly from raw prompt statistics.

It should speak from:

- grounded world state
- verified and proposed rules
- selected hypotheses
- planner results
- explanation traces

This is how answers become explainable and stable.

## 10. What the Final System Must Do With Different Input Types

The final system should not treat all inputs identically.

It should treat them uniformly at the carrier level, but differently at the
verification and grounding level.

### 10.1 Code

Required behavior:

- parse structure
- derive control/data flow
- derive execution traces when possible
- compare intention and operator effects
- compile semantic program facts
- link program actions to world-state changes

### 10.2 Scientific Text

Required behavior:

- detect claims vs background vs methods
- detect citations
- detect quantitative statements
- detect uncertainty language
- detect hypothesis-result relationships
- ground claims into structured propositions
- separate reported claims from verified claims

### 10.3 Dialogue

Required behavior:

- track speakers
- track turn order
- distinguish beliefs, intentions, and assertions
- resolve pronouns and references
- separate quoted content from narrator content

### 10.4 Logs and Traces

Required behavior:

- parse events
- detect repeated patterns
- detect failures and anomalies
- infer temporal sequences
- compile state-transition evidence

### 10.5 Instructions and Procedures

Required behavior:

- detect ordered steps
- detect prerequisites
- detect expected outcomes
- detect failure conditions
- compile plan graphs

### 10.6 Mixed Documents

Required behavior:

- segment by local type
- preserve cross-segment links
- avoid forcing one subtype onto the whole document
- support code-with-comments, notebook-style documents, and report-plus-table
  mixtures

## 11. Multilingual Requirements

This is non-negotiable for a final strong system.

### 11.1 No English-Only Semantics

The grounding stack must work for multilingual inputs natively.

English-specific regexes are acceptable only as temporary fallbacks.

### 11.2 Surface Variation Must Converge

Equivalent meanings across languages should converge into the same canonical
interlingua when appropriate.

### 11.3 Morphology Matters

Languages with rich inflection require:

- lemma awareness
- morphological features
- agreement signals
- case-sensitive role inference

### 11.4 Mixed-Language Inputs Must Be First-Class

The final system must support:

- code plus English comments
- Ukrainian request plus English API names
- multilingual documentation
- translated scientific prose

## 12. Uncertainty, Confidence, and Revision

The final grounding system must be explicitly revisable.

### 12.1 Confidence Is Per Object, Not One Global Number

Every grounded object should carry its own confidence.

### 12.2 Alternative Hypotheses Must Survive Long Enough

The system should keep multiple interpretations when the evidence is ambiguous.

### 12.3 Contradiction Is a First-Class Outcome

A contradiction is not just an error.

It is evidence about the world, the input, or the parser.

### 12.4 Revision Must Be Cheap

If later evidence changes the interpretation, the system must revise the world
state without corrupting provenance.

## 13. Hidden-Cause Abduction and Explanation

A final powerful system must do more than synthesize bridge rules.

It must also generate concrete hidden-cause hypotheses when facts conflict.

Examples:

- "someone else opened the door"
- "the dispatcher opened the door remotely"
- "the rule has a hidden exception"
- "the observation refers to a different object than assumed"

This requires the grounding stack to expose explicit missing-role and
missing-event slots that the abduction system can fill.

The final hidden-cause abduction flow should be:

1. detect inconsistency
2. localize missing causal edge
3. propose candidate hidden events or agents
4. score against world rules and memory
5. keep as `proposed` until support arrives

## 14. Ontology Growth and New Concept Creation

The final system must be able to invent new internal concepts when the input
contains repeated structure not covered by existing ontology.

This is relevant for:

- pattern compression in logs
- new scientific relation types
- domain-specific event families
- user-specific task abstractions

The invention process should be:

1. detect compressible recurring subgraph
2. assign temporary canonical concept id
3. attach defining support set
4. test whether the new concept improves compression or prediction
5. optionally synthesize a human-readable label

This should connect cleanly to current NET and symbolic compression mechanisms.

## 15. How Humans and External Systems Should Interact With OMEN

This section answers the practical "what to give and how to give it" question.

### 15.1 Current Practical Best Mode

Right now, the system works best when inputs are given in a relatively
structured form.

Best current input styles:

- explicit facts on separate lines
- compact pseudo-formal statements
- real source code rather than vague pseudocode
- logs in compact key-value style
- tasks with clear goal statements

This is a statement about the current code, not the final ideal.

### 15.2 Final Desired Mode

In the final architecture, the system should accept raw realistic material:

- ordinary natural language
- scientific text
- code with comments
- logs
- mixed documents
- contradictory reports

without needing heavy manual canonicalization first.

### 15.3 Operator Guidance

Operators should still be able to help by supplying:

- source metadata
- language hints
- document type hints
- known schemas
- expected task type
- domain ontology hints

These should act as soft guides, not hard overrides.

### 15.4 External Integrations

The final system should be ready to interact with:

- document loaders
- code repositories
- schema registries
- telemetry/log streams
- evaluation harnesses
- simulation environments
- planner executors

Each connector should feed typed source metadata into L1 rather than bypassing
the grounding stack.

## 16. How the Modules Should Interact in the Final Cycle

This is the desired full-cycle behavior.

1. ingest bytes and metadata
2. segment and type the source
3. run structural parsers
4. run multilingual linguistic grounding if needed
5. build semantic scene graph
6. normalize into canonical interlingua
7. compile candidate symbolic objects
8. verify and repair
9. write accepted state to world graph and memory
10. construct `SymbolicTaskContext`
11. run deduction, abduction, induction, and EMC
12. run planning or counterfactual simulation if the task needs it
13. generate answer or action trace from grounded state
14. store resulting outcome and provenance

This should become the canonical cognitive loop for external inputs.

## 17. Metrics the Final System Must Track

If grounding is important, it must be measurable.

### 17.1 Perception and Routing Metrics

- modality accuracy
- subtype accuracy
- calibration error
- ambiguity detection quality
- segment-boundary quality

### 17.2 Linguistic Grounding Metrics

- mention detection
- coreference quality
- role labeling
- clause decomposition quality
- multilingual transfer quality

### 17.3 Semantic Grounding Metrics

- entity persistence accuracy
- event extraction accuracy
- causal-link accuracy
- temporal-link accuracy
- polarity/modality accuracy
- quantifier preservation accuracy

### 17.4 Symbolic Compilation Metrics

- fact precision
- fact recall
- rule precision
- hidden-cause proposal quality
- contradiction localization quality

### 17.5 Verification Metrics

- verified vs contradicted calibration
- trace agreement
- world-model agreement
- repair success rate
- hypothesis survival quality

### 17.6 End-to-End Metrics

- scenario completion rate
- reasoning correctness
- explanation quality
- planner success
- long-horizon stability

## 18. Failure Modes the Final Design Must Explicitly Avoid

### 18.1 Garbage-Bucket `other`

`other` should not become the place where the system throws everything it does
not understand.

### 18.2 Hard Early Collapse

The system must not force one interpretation when the evidence is weak.

### 18.3 Provenance Loss

The system must not lose the link from a grounded object back to source spans
and parser evidence.

### 18.4 Silent Cross-Language Failure

The system must not look "confident" while failing on non-English inputs.

### 18.5 Planner From Ungrounded Text

The planner must not operate as if raw text were already verified world state.

### 18.6 Symbolic Overcompensation

The symbolic core must not be used to compensate for poor grounding that should
have been solved earlier.

## 19. Recommended Repository-Level Direction

For the current repository, the most important architectural direction is:

- keep the current `SourceRoutingDecision` and expand it
- keep `SymbolicTaskContext` as the central symbolic ingress object
- keep `WorldGraphState` and `CanonicalWorldState` as graph-primary internal
  state
- replace shallow text grounding with a staged semantic grounding stack
- feed richer grounded objects into memory, verification, prover, EMC, and
  planners

The repository already has the right backbone.

The missing leap is not a new identity for the system.

The missing leap is a much stronger grounding front-end and semantic compiler.

## 20. Practical Immediate Implication

If the goal is long-term extreme power, the grounding subsystem should be
treated as a first-class architecture program, not as a helper utility.

That means future implementation work should prioritize:

- multilingual semantic grounding
- event/entity persistence
- canonical interlingua
- probabilistic symbolic compilation
- hidden-cause abduction support
- ontology growth with human-readable naming
- grounding-aware EMC signals

## 21. Final Summary

OMEN already has:

- a serious world-state core
- a serious symbolic core
- memory
- verification paths
- graph-centered integration
- continuous reasoning machinery

What it does not yet have, in final form, is a truly strong grounding stack for
raw natural language and mixed real-world inputs.

The final target should therefore be:

`carrier bytes -> typed perception -> structural grounding -> linguistic grounding -> semantic scene graph -> canonical interlingua -> deterministic symbolic lowering with explicit epistemic scoring -> verification/repair -> world state -> reasoning/planning/generation`

That is the architecture required if OMEN is to evolve from a strong
research-grade neuro-symbolic runtime into a truly robust, multilingual,
world-grounded AI system.

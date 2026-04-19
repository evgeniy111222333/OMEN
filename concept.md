# OMEN Canonical Product Concept and Technical Specification

## 1. Document Status

This document is the canonical product concept for OMEN in the current repository.

It replaces older layered concept notes where early hypotheses, intermediate
revisions, critiques of previous implementations, future ideas, and partially
implemented fixes were mixed together.

This file is normative rather than historical.

If an older concept fragment conflicts with this document, this document wins.

Canonical runtime in this repository:

- `omen_scale.py`
- public entry through `omen.py`

Canonical public surface:

- `omen.OMEN`
- `omen.OMENConfig`
- `omen.build_omen(...)`

OMEN does not have multiple equally valid primary runtimes. Historical runtime
variants may remain only as legacy or research layers, not as alternative
"real" OMEN implementations.

## 2. Product Definition

OMEN is a world-grounded neuro-symbolic cognitive runtime that combines:

- byte-level perception and compression
- structured world-state construction
- long-term neural memory and exact symbolic memory
- formal symbolic inference
- abduction, deduction, and induction
- control over reasoning cost
- text, code, and plan generation from structured internal state rather than
  only from surface token statistics

OMEN is not just a language model.

OMEN is not just a symbolic solver.

OMEN is not just a world model.

OMEN is an integrated system in which:

- language is the input and output carrier
- world state is the central internal reality
- the symbolic layer is an operational mechanism for verification, inference,
  and generalization
- memory is an active computational resource, not a passive archive
- computation cost and rule complexity are part of the system economy

## 3. End Goal

The long-term goal of OMEN is to:

1. turn raw observation streams into structured world state
2. compress experience statistically and conceptually
3. infer, verify, reject, and repair rules about the world
4. distinguish verified knowledge from hypotheses, mistakes, and noise
5. use memory, reasoning, and planning only when they are informationally and
   computationally justified
6. generate answers, programs, and plans as consequences of world state,
   symbolic reasoning, and memory retrieval
7. support online symbolic learning and controlled self-improvement without a
   hard split between train and eval behavior

In short, OMEN should build an explainable, verifiable, cost-aware internal
world and act through it.

## 4. Core vs Extensions

### 4.1 Required Core

Without these pieces, OMEN is not conceptually valid:

- byte-level input
- NET as the canonical tokenizer and compressor
- graph-primary world state
- world-graph-grounded perception
- `WorldRNN` as the transition model
- M-Core as long-term memory
- exact symbolic substrate with first-order terms, variables, and unification
- `KnowledgeBase` with epistemic rule status
- execution-trace-first symbolic supervision
- unified MDL / free-energy-style optimization pressure
- graph-centered decoder state

### 4.2 Second-Level Integrated Subsystems

These are canonical but conceptually depend on the core:

- saliency traces
- Verification Module (VeM)
- continuous symbolic cycle
- eval-capable online symbolic learning
- program anchoring
- EMC as a meta-controller for reasoning depth
- symbolic memory index

### 4.3 Advanced Generative Extensions

These are canonical support modules, but they do not define OMEN validity on
their own:

- OSF
- creative cycle modules
- ICE and internal goals
- analogy, counterfactual, ontology, aesthetic, and intrinsic engines

These extensions must not exist as decorative layers. They should grow out of a
stable world / symbolic / memory core.

## 5. Architectural Principles

### 5.1 World State Is Primary

The center of OMEN is not raw text and not a single dense latent vector. The
center is a structured world state.

### 5.2 Graph-Primary Canonical `z`

Canonical `z` is graph-primary. Dense vectors are derived readouts or
specialized views of that state.

### 5.3 Byte-First, Not BPE-First

The canonical substrate is raw UTF-8 bytes. BPE and WordPiece can exist only as
compatibility or ablation modes, not as the canonical product path.

### 5.4 Exact Symbolic Substrate

The symbolic layer is considered valid only if it has:

- terms
- variables
- unification
- rules
- verification
- contradiction handling

A graph-only or GNN-like "symbolic" layer is not enough.

### 5.5 Memory Is Active

Memory must not only store information. It must also:

- affect state
- control recall
- participate in curiosity and gap-closing loops
- be evaluated through cost and utility

### 5.6 Reasoning Has a Cost

Proof depth and the number of reasoning steps are not free. Computational cost
is part of the learning objective.

### 5.7 Symbolic Learning Must Be Grounded

Symbolic learning must be tied to execution traces, state transitions, target
facts, and counterexamples.

### 5.8 Generation Reads Structured State

Generation should read from graph-centered world state, memory state, symbolic
state, and program state, not only from surface token context.

## 6. Canonical Runtime and Modules

### 6.1 Canonical Runtime

The canonical runtime is `omen_scale.py`, surfaced through `omen.py`.

### 6.2 Canonical Support Modules

The current canonical stack includes:

- `omen_net_tokenizer.py`
- `omen_prolog.py`
- memory and symbolic modules under `omen_symbolic/`
- `omen_emc.py`
- world, perception, and decoder components integrated by `omen_scale.py`

### 6.3 Legacy Status

Historical runtime variants may remain in the repository for reference or
research, but they are not normative product architecture.

## 7. Canonical System State Contract

### 7.1 Input

The canonical input is a UTF-8 byte sequence.

This implies:

- `vocab_size = 256` is the canonical operating mode
- inputs do not assume a large fixed lexical inventory
- semantic units are formed later through NET, the world graph, and the
  symbolic layer

### 7.2 Canonical `z`

The canonical contract is:

- `out["z"]` is a structured `CanonicalWorldState`
- `out["z_dense"]` is a dense decoder-facing readout derived from the graph
  primary state
- `out["world_state"]` is the same canonical world state
- `out["z_graph_struct"]` is the graph view of that state
- `out["z_world"]` is a dense grounded state produced from graph-centered
  fusion

### 7.3 Canonical World State Contents

Canonical world state includes:

- one or more world graphs
- base neural state
- graph-grounded state
- graph projection and readout views
- grounded decoder-facing state
- symbolic state
- memory state
- program state
- symbolic facts
- target facts
- metadata for the canonical cycle

`grounded_state` is a derived readout for decoding and downstream heads. It is
not an alternative primary meaning of `z`.

## 8. World Graph

The world graph is the grounding substrate for the modelled world.

Its nodes and edges are formed from:

- facts
- terms
- events
- saliency and trace-derived relations
- task-context links

The graph must represent:

- local fact structure
- state-transition dynamics
- links between world modelling and symbolic reasoning

## 9. NET: Canonical Tokenizer and Compressor

NET replaces BPE and WordPiece as the canonical tokenization path.

Its role is to:

- operate on raw bytes
- form context-sensitive concepts
- keep vocabulary growth under MDL pressure
- build a bridge from byte input to world and symbolic structure

Canonical NET components are:

- byte context encoder
- epistemic quantizer
- byte decoder

The codebook is dynamic. It is not a fixed lexical inventory.

## 10. Perception and World Model

After NET, OMEN builds concept-level state through perception layers and
graph-aware latent structure.

`WorldRNN` models state transitions rather than only token dynamics.

It must:

- read current state
- condition on actions or action probabilities
- use graph context when available
- predict the next grounded state
- expose causal and alignment errors

## 11. Memory and Symbolic Reasoning

M-Core is the long-term memory substrate.

The symbolic layer is based on exact first-order reasoning:

- Horn terms and atoms
- unification
- forward and backward reasoning
- rule induction and abduction
- epistemic rule tracking
- contradiction-aware rule lifecycle

`KnowledgeBase` stores facts and rules with epistemic status such as:

- proposed
- verified
- contradicted

`DifferentiableProver` connects symbolic reasoning to neural training signals.

VeM filters candidate rules before long-term storage.

The continuous symbolic cycle keeps symbolic learning active rather than
treating it as an offline-only subsystem.

## 12. Generation and Program Grounding

Generation should be a readout from structured world state, memory state, and
symbolic state.

Program targets are anchored symbolically rather than treated as plain text
continuation.

Execution traces are first-class supervision for program reasoning and repair.

## 13. Train / Eval Contract

OMEN should avoid a conceptual split where symbolic learning exists only during
training but disappears during evaluation.

Train and eval may differ in exploration level or stochasticity, but not in the
existence of the core runtime loop.

Important tracked quantities include:

- reconstruction quality
- world-model prediction error
- proof success
- reasoning cost
- rule utility
- symbolic consistency

## 14. Acceptance Criteria

An implementation counts as canonically aligned only if it:

- uses byte-first input in canonical mode
- maintains graph-primary world state
- includes exact symbolic reasoning with variables and unification
- maintains epistemic rule status in the knowledge base
- keeps reasoning cost in the optimization picture
- supports memory as an active part of cognition
- grounds generation in structured state rather than raw token context alone
- preserves the canonical public surface through `omen.py`

## 15. Practical Interpretation

When making architectural decisions in this repository:

- prefer the graph-centered and world-grounded interpretation of state
- treat symbolic reasoning as operational, not decorative
- treat memory as part of computation
- treat runtime cost as part of intelligence
- keep legacy paths clearly separated from canonical product paths

This is the working canonical concept for OMEN in the current codebase.

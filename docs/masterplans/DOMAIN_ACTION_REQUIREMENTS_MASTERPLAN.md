# OMEN Domain Action Requirements Masterplan

## 1. Status and Intent

This document defines what OMEN should be able to do in real domains, through
real action, with real practical usefulness.

It complements:

- `concept.md`
- `GROUNDING_MASTERPLAN.md`
- `TRAINING_MASTERPLAN.md`
- `SYSTEM_REQUIREMENTS_MASTERPLAN.md`

This file is not mainly about internal architecture.

This file is about the external consequences of the architecture:

- what concrete tasks the system should perform
- what action patterns it should support
- what quality level it should reach in real domains
- what kinds of mistakes it must stop making
- what kinds of practical capability should be visible to operators

The target is not a system that merely "understands" in an abstract way.

The target is a system that can take grounded understanding and convert it into
correct, useful, revisable, high-value action across many fields.

## 2. Requirement Classes

All domain action requirements in this document are classified into four levels.

### 2.1 Explicitly Required

These are required for OMEN to count as a serious domain-capable system.

### 2.2 Strongly Desired

These are highly important for maturity, robustness, and operator trust.

### 2.3 If Possible

These are frontier ambitions and strategic long-term targets.

### 2.4 Practical Outcome Requirements

These describe what a user, operator, engineer, analyst, or evaluator should
actually observe in real work.

## 3. General Action Principles Across All Domains

### 3.1 Explicitly Required

- The system must not act on raw text as if raw text were already verified
  world truth.
- The system must distinguish observation, instruction, claim, plan, proposal,
  warning, and verified result.
- The system must preserve provenance for consequential action.
- The system must support revision when action premises change.
- The system must know when to defer, ask for clarification, or branch on
  multiple plausible interpretations.
- The system must respect constraints, dependencies, and failure conditions.

### 3.2 Strongly Desired

- The system should decompose large tasks into grounded subgoals.
- The system should estimate which actions are reversible, risky, or high
  leverage.
- The system should maintain domain-specific memory that improves future work.
- The system should reuse verified structure rather than re-guessing from
  scratch.

### 3.3 If Possible

- The system should learn domain-specific action abstractions from repeated
  successful episodes.
- The system should improve its action policies through verified outcomes.
- The system should support hierarchical multi-stage plans that remain grounded
  throughout execution.

### 3.4 Practical Outcome Requirements

- Real outputs should look action-worthy, not merely eloquent.
- If the system gives a procedure, plan, derivation, patch, analysis, or
  diagnosis, it should be clear what premises it used and how much it trusts
  them.
- If the premises are weak, the system should surface contingencies rather than
  burying them.

## 4. Mathematics Requirements

### 4.1 Explicitly Required

- The system must parse mathematical problems expressed informally, formally,
  or in mixed notation.
- The system must represent variables, constraints, goals, assumptions, and
  derived statements explicitly.
- The system must distinguish exact proof, heuristic intuition, conjecture,
  approximation, and counterexample.
- The system must perform multi-step symbolic manipulation without silently
  losing assumptions.
- The system must track domains of validity, edge cases, and special cases.
- The system must verify whether each derivation step is justified.
- The system must detect when a proposed solution depends on an unproven step.
- The system must detect contradiction and invalid inference.
- The system must produce counterexamples when a claim fails.

### 4.2 Strongly Desired

- The system should handle algebra, arithmetic, geometry, discrete math,
  combinatorics, probability, statistics, calculus, linear algebra, logic, and
  optimization.
- The system should translate between natural-language reasoning and formal
  symbolic structure.
- The system should identify latent subproblems and choose promising solution
  strategies.
- The system should generate multiple solution approaches when useful.
- The system should distinguish elegance, generality, and brute-force validity.
- The system should detect hidden assumptions and unstated conditions.

### 4.3 If Possible

- The system should support theorem-style proof construction at varying levels
  of rigor.
- The system should support constructive proof, contradiction proof, induction,
  invariants, probabilistic reasoning, and optimization arguments.
- The system should invent useful lemmas or intermediate abstractions.
- The system should transfer techniques across mathematical areas when the
  structure is homologous.
- The system should recognize when a symbolic proof should be paired with
  numerical validation.

### 4.4 Practical Outcome Requirements

- If given a competition-style problem, the system should not only output an
  answer, but a derivation whose weak points are explicit.
- If given a false claim, the system should often produce a concrete
  counterexample instead of vague disagreement.
- If given an applied optimization or estimation task, the system should state
  what is exact, what is approximate, and what assumptions dominate the result.
- If asked to explain a proof, the system should preserve correctness while
  changing exposition level for the audience.

## 5. Programming and Software Engineering Requirements

### 5.1 Explicitly Required

- The system must understand repositories, modules, entry points, tests,
  configs, and runtime contracts as one coherent action space.
- The system must distinguish code facts from comments, assumptions, docs, and
  desired behavior.
- The system must use real repository evidence when editing code.
- The system must support debugging, implementation, refactoring, and review.
- The system must trace how a change affects behavior across multiple files.
- The system must use tests, traces, and runtime errors as first-class signal.
- The system must preserve or improve behavioral correctness when making
  changes.
- The system must know when a requested change is underspecified.
- The system must separate speculative fixes from verified fixes.

### 5.2 Strongly Desired

- The system should diagnose bugs from code, logs, traces, failing tests, and
  system behavior jointly.
- The system should recover architectural intent from messy repositories.
- The system should perform targeted refactors with blast-radius awareness.
- The system should reason about APIs, schemas, migrations, and compatibility.
- The system should improve test coverage when behavior changes.
- The system should distinguish user-facing regression risk from internal
  cleanup.
- The system should identify performance, reliability, and maintainability
  consequences of code changes.
- The system should understand toolchains, build systems, dependency graphs,
  and environment assumptions.

### 5.3 If Possible

- The system should infer deeper latent architecture from recurring patterns and
  propose restructuring that genuinely improves future development.
- The system should perform high-quality code synthesis under repository style,
  runtime constraints, and verification pressure.
- The system should support formalized program reasoning where possible.
- The system should learn reusable symbolic abstractions from execution traces,
  tests, and repository patterns.

### 5.4 Practical Outcome Requirements

- If given a bug report, the system should move from symptom to cause to fix to
  verification with minimal handholding.
- If asked to implement a feature, the system should preserve repo conventions,
  surface assumptions, and produce code that is locally coherent and globally
  integrated.
- If asked for review, the system should primarily find real risks, not
  summarize obvious code.
- If tests fail, the system should identify whether the root problem is code,
  config, environment, assumptions, or test drift.

## 6. Formal Methods, Logic, and Verification Requirements

### 6.1 Explicitly Required

- The system must distinguish verified statements from merely plausible ones.
- The system must reason over rules, constraints, preconditions, and invariant
  structure explicitly.
- The system must localize proof gaps and verification failures.
- The system must preserve symbolic traceability for formal reasoning.

### 6.2 Strongly Desired

- The system should support consistency checking, invariant discovery, and
  property decomposition.
- The system should derive verification obligations from grounded task context.
- The system should use counterexamples to refine rules or assumptions.

### 6.3 If Possible

- The system should interface with external provers or formal solvers where
  helpful.
- The system should synthesize machine-checkable intermediate forms.
- The system should move fluidly between natural-language specs and formalized
  obligations.

### 6.4 Practical Outcome Requirements

- If a specification is incomplete, the system should identify the missing
  obligations explicitly.
- If a proof idea is invalid, the system should show where and why the failure
  occurs rather than only saying it is wrong.

## 7. Scientific Research Requirements

### 7.1 Explicitly Required

- The system must read and compare scientific claims without flattening all
  cited material into direct truth.
- The system must distinguish observation, inference, hypothesis, mechanism,
  citation, and speculation.
- The system must preserve uncertainty, evidence quality, and contradiction.
- The system must separate empirical support from conceptual explanation.
- The system must track whether a claim is directly reported, weakly supported,
  strongly supported, or contested.

### 7.2 Strongly Desired

- The system should synthesize literature across papers, methods, and
  reporting styles.
- The system should identify agreement, disagreement, uncertainty clusters, and
  possible hidden explanatory variables.
- The system should propose experiments or observations that would resolve key
  open questions.
- The system should separate causal claims from correlational claims.
- The system should detect where a field is overconfident relative to evidence.

### 7.3 If Possible

- The system should generate new research hypotheses grounded in observed
  explanatory gaps.
- The system should propose better taxonomies or conceptual groupings for
  repeated structures across studies.
- The system should model scientific debates as structured competing
  hypothesis worlds rather than one blended summary.

### 7.4 Practical Outcome Requirements

- If asked for a scientific synthesis, the system should produce a map of
  claims, support, uncertainty, and disagreement, not just a prose blend.
- If asked what to test next, the system should point to the most informative
  experiments or data collection steps.
- If literature is contradictory, the system should preserve that contradiction
  explicitly.

## 8. Data Analysis, Statistics, and Decision Support Requirements

### 8.1 Explicitly Required

- The system must parse data tasks into variables, targets, assumptions,
  confounders, and decision stakes.
- The system must distinguish data cleaning, exploration, inference,
  prediction, and policy recommendation.
- The system must preserve uncertainty and assumption sensitivity.
- The system must detect when apparent results rely on weak data quality or
  invalid comparisons.
- The system must distinguish descriptive statistics from causal conclusions.

### 8.2 Strongly Desired

- The system should identify data leakage, sampling problems, missingness
  patterns, and unstable metrics.
- The system should propose better analyses when the initial framing is weak.
- The system should support both quantitative and mixed qualitative-quantitative
  reasoning where the domain requires it.
- The system should state what decisions become more or less defensible under
  the analysis.

### 8.3 If Possible

- The system should learn reusable domain-specific analysis templates.
- The system should suggest robust decision policies under uncertainty, rather
  than only point estimates.
- The system should support simulation-based decision support.

### 8.4 Practical Outcome Requirements

- If given messy business, operations, or scientific data, the system should
  identify what is analyzable, what is missing, and what would make the
  analysis materially stronger.
- If asked for a recommendation, the system should connect the recommendation to
  quantified uncertainty and key assumptions.

## 9. Planning, Operations, and Procedures Requirements

### 9.1 Explicitly Required

- The system must convert procedures, instructions, constraints, and goals into
  actionable structures.
- The system must identify dependencies, prerequisites, branch points, and
  failure conditions.
- The system must distinguish mandatory steps from optional optimizations.
- The system must support plan revision under new evidence.
- The system must avoid presenting ambiguous procedure as deterministic command.

### 9.2 Strongly Desired

- The system should support operational planning under uncertainty and partial
  observability.
- The system should surface where a plan depends on weakly grounded facts.
- The system should support alternate branches, contingencies, and rollback
  plans.
- The system should explain what would invalidate a plan.

### 9.3 If Possible

- The system should learn reusable procedural schemas from repeated workflows.
- The system should support multi-stage operational orchestration across tools,
  teams, and time windows.
- The system should anticipate likely failure points before execution.

### 9.4 Practical Outcome Requirements

- If asked to operationalize a messy request, the system should output a plan
  with assumptions, prerequisites, branches, and validation points.
- If the environment changes mid-plan, the system should adjust the plan rather
  than cling to stale assumptions.

## 10. Incident Response, Diagnostics, and Root-Cause Analysis Requirements

### 10.1 Explicitly Required

- The system must turn logs, alerts, traces, operator reports, and config state
  into structured incident hypotheses.
- The system must distinguish observed symptoms from inferred causes.
- The system must support multiple simultaneous root-cause hypotheses.
- The system must preserve causal uncertainty and escalation triggers.

### 10.2 Strongly Desired

- The system should correlate events across time, services, tools, and reports.
- The system should identify the most diagnostic next checks.
- The system should separate proximate cause from root cause.
- The system should support repair hypotheses and validation steps.

### 10.3 If Possible

- The system should propose hidden-cause explanations when direct evidence is
  missing but the observed state is inconsistent.
- The system should learn recurring failure templates and mitigation patterns.
- The system should support postmortem abstraction and future prevention policy.

### 10.4 Practical Outcome Requirements

- If an incident is messy and partially conflicting, the system should not give
  a single overconfident cause narrative too early.
- If evidence strongly favors one hypothesis, the system should make that clear
  while preserving what remains uncertain.

## 11. Cybersecurity and Adversarial Reasoning Requirements

### 11.1 Explicitly Required

- The system must distinguish evidence of compromise from speculation.
- The system must reason about attacker actions, system state, telemetry, and
  defensive controls as structured world objects.
- The system must preserve confidence and contradiction in threat analysis.
- The system must not escalate weak signals into false-certainty compromise
  narratives.

### 11.2 Strongly Desired

- The system should identify attack chains, prerequisites, likely pivots, and
  containment options.
- The system should distinguish detection artifact, exploit mechanism, and
  business impact.
- The system should help prioritize triage under incomplete information.

### 11.3 If Possible

- The system should simulate attacker and defender action branches.
- The system should infer hidden causal links in partial telemetry.
- The system should generalize mitigation templates across related incidents.

### 11.4 Practical Outcome Requirements

- If given scattered security signals, the system should produce a hypothesis
  map with evidence strength, likely sequence, and next validation steps.
- If a threat path is uncertain, the system should preserve uncertainty instead
  of inventing a cinematic narrative.

## 12. Strategic Reasoning and Decision Architecture Requirements

### 12.1 Explicitly Required

- The system must support long-range goals, constraints, branches, and tradeoff
  reasoning.
- The system must distinguish strategic assumptions from observed facts.
- The system must reason over uncertainty, irreversibility, and risk.
- The system must support alternative scenarios and conditional planning.

### 12.2 Strongly Desired

- The system should identify leverage points, bottlenecks, dependencies, and
  hidden second-order effects.
- The system should estimate what information would most improve a strategic
  decision.
- The system should maintain strategy memory over long projects.

### 12.3 If Possible

- The system should construct explicit strategic state spaces and policy
  comparisons.
- The system should simulate competing actor strategies in shared worlds.
- The system should support dynamic re-optimization as new evidence arrives.

### 12.4 Practical Outcome Requirements

- If asked for a strategic recommendation, the system should expose the
  assumptions, branch structure, and uncertainty profile behind the advice.
- If the environment changes, the system should update the recommendation rather
  than merely restating the original plan.

## 13. Knowledge Management and Documentation Requirements

### 13.1 Explicitly Required

- The system must turn scattered material into structured knowledge with source
  distinctions.
- The system must support document linking, concept linking, and provenance.
- The system must distinguish stable policy, transient note, open question, and
  verified understanding.

### 13.2 Strongly Desired

- The system should build useful internal maps of concepts, modules, entities,
  workflows, and tensions.
- The system should surface stale documentation versus current reality.
- The system should support progressive refinement of institutional knowledge.

### 13.3 If Possible

- The system should induce higher-order taxonomies and reusable documentation
  templates from repeated patterns.
- The system should propose documentation updates when runtime reality drifts
  from stated architecture.

### 13.4 Practical Outcome Requirements

- If given a large knowledge base, the system should help operators navigate it
  by meaning, not only by keyword.
- If documentation conflicts with implementation or evidence, the system should
  say so clearly.

## 14. Education, Tutoring, and Explanation Requirements

### 14.1 Explicitly Required

- The system must distinguish explanation from proof, summary, and instruction.
- The system must preserve truth when adapting explanation to audience level.
- The system must not hide uncertainty when teaching.
- The system must support stepwise decomposition for complex concepts.

### 14.2 Strongly Desired

- The system should diagnose where a learner's misunderstanding likely lies.
- The system should generate multiple explanation styles for the same grounded
  content.
- The system should support examples, counterexamples, and contrastive
  explanations.
- The system should preserve formal correctness even when simplifying.

### 14.3 If Possible

- The system should adapt teaching strategy based on persistent learner model.
- The system should synthesize targeted exercises that probe specific gaps.
- The system should connect abstract ideas to grounded real-world examples.

### 14.4 Practical Outcome Requirements

- If asked to teach a topic, the system should not merely paraphrase textbook
  language. It should build understanding progressively while keeping the
  structure of truth intact.
- If a student's question is based on a false assumption, the system should
  correct the assumption before extending the explanation.

## 15. Creative and Design-Oriented Requirements

### 15.1 Explicitly Required

- The system must distinguish creative synthesis from factual reporting.
- Creative output must still respect explicitly grounded constraints.
- The system must support multi-objective creative tasks where style,
  structure, correctness, and usefulness interact.

### 15.2 Strongly Desired

- The system should support ideation, variation, analogy, reframing, and
  synthesis without discarding grounded requirements.
- The system should generate alternatives rather than a single creative guess.
- The system should preserve rationale for high-level design decisions.

### 15.3 If Possible

- The system should learn reusable aesthetic, structural, and functional design
  abstractions.
- The system should combine symbolic constraints with generative flexibility in
  a controlled way.
- The system should support long-horizon creative projects with memory of
  accumulated design decisions.

### 15.4 Practical Outcome Requirements

- If asked to design, the system should produce options that differ in real
  tradeoffs, not only superficial wording.
- If asked to be creative under constraints, it should preserve the constraints
  instead of dropping them when generation gets harder.

## 16. Multi-Tool and Agentic Workflow Requirements

### 16.1 Explicitly Required

- The system must treat tools as grounded action interfaces, not decorative
  appendages.
- The system must reason about tool preconditions, side effects, and evidence
  returned by tool use.
- The system must support multi-step tool workflows with stateful correction.
- The system must distinguish observation from action from mutation.

### 16.2 Strongly Desired

- The system should select tools based on value of information and value of
  execution.
- The system should recover gracefully from partial tool failure.
- The system should reuse learned tool action schemas across related tasks.
- The system should track what has already been tried and what remains unknown.

### 16.3 If Possible

- The system should support multi-agent decomposition where sub-agents share
  grounded state and avoid redundant work.
- The system should learn reusable tool plans that remain auditable.
- The system should optimize action ordering under cost and latency pressure.

### 16.4 Practical Outcome Requirements

- If real work requires browsing, reading, testing, editing, verifying, and
  summarizing, the system should orchestrate that coherently instead of treating
  each step as disconnected.
- If tool evidence conflicts, the system should preserve the conflict rather
  than smoothing it away.

## 17. Embodied, Physical, and Sensor-Connected World Requirements

### 17.1 Explicitly Required

- If connected to physical-world or sensor-like input, the system must treat
  perception, uncertainty, and action consequences explicitly.
- The system must distinguish instruction, observation, and inferred world
  state in embodied contexts.
- The system must not act with unjustified certainty when sensory evidence is
  partial.

### 17.2 Strongly Desired

- The system should reason about spatial structure, object persistence, event
  sequence, and environmental constraints.
- The system should support procedural grounding for physical tasks.
- The system should recognize when safety margin requires deference.

### 17.3 If Possible

- The system should learn reusable action schemas for grounded environments.
- The system should simulate likely action outcomes before acting.
- The system should integrate multimodal evidence into a stable action world
  model.

### 17.4 Practical Outcome Requirements

- If tasked with guiding or analyzing real-world procedures, the system should
  preserve sequencing, prerequisites, and safety-critical ambiguity rather than
  flattening them into loose prose.

## 18. Cross-Domain Composition Requirements

### 18.1 Explicitly Required

- The system must combine evidence and action structure across domains when
  real tasks span multiple spheres.
- The system must preserve domain-specific epistemic distinctions rather than
  forcing one generic reasoning mode onto all tasks.
- The system must support transitions such as:
  research -> analysis -> plan -> code -> verification -> deployment -> review.

### 18.2 Strongly Desired

- The system should maintain coherent shared world state when one task moves
  through mathematics, code, data, operations, and explanation in one episode.
- The system should reuse domain-specific strengths without losing global
  coordination.
- The system should identify which domain frames are active and which conflicts
  matter most.

### 18.3 If Possible

- The system should induce meta-abstractions that unify repeated patterns across
  very different domains.
- The system should support general-purpose "cognitive workflows" built from
  domain-local action modules.

### 18.4 Practical Outcome Requirements

- If a problem spans many fields, the system should not fall apart at the
  interfaces between them.
- Cross-domain tasks should look more coherent because the system integrates the
  domains, not less coherent because each domain is treated as a separate
  isolated mode.

## 19. Maturity Levels for Real Action

### 19.1 Minimum Action-Capable System

- can ground tasks into structured action frames
- can perform bounded multi-step reasoning
- can preserve provenance for important outputs
- can revise after contradiction
- can use tools coherently

### 19.2 Strong Research-Grade Action System

- succeeds on real multi-step tasks in several domains
- uses memory and verification meaningfully
- improves with repeated work
- handles uncertainty honestly
- avoids obvious heuristic brittleness

### 19.3 High-End Practical Action System

- solves serious tasks in mathematics, coding, analysis, planning, and research
- coordinates many evidence streams
- maintains long-horizon context
- shows domain-sensitive reliability
- uses compute efficiently

### 19.4 Frontier General Action System

- transfers robustly across many high-value domains
- supports deep planning and revision
- can form and test higher-level abstractions
- improves capability through controlled lifelong learning
- remains epistemically disciplined while becoming more powerful

## 20. Anti-Goals

The following must not be mistaken for domain competence:

- polished phrasing without grounded action quality
- domain buzzwords without stateful reasoning
- code generation without repository understanding
- analysis without uncertainty discipline
- planning without grounded constraints
- mathematics without derivation integrity
- research synthesis without source separation
- memory accumulation without future-task utility
- tool usage without coherent state update

## 21. Final Directive

The final domain-capable OMEN should be able to take real-world material from
many fields, form a grounded internal task world, reason over that world,
choose useful actions, revise under contradiction, preserve provenance, and
produce practical outcomes at a level that is visibly stronger than surface
language continuation.

If a future system looks impressive in style but remains weak at:

- structured derivation
- grounded coding
- uncertainty-aware analysis
- plan revision
- causal diagnosis
- memory-informed action
- verification-disciplined execution
- cross-domain practical success

then it should still be treated as below the intended OMEN standard.

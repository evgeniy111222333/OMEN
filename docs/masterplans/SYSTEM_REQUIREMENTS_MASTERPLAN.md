# OMEN System Requirements Masterplan

## 1. Status and Intent

This document defines the capability requirements that OMEN should ultimately
reach across the full system.

It complements:

- `concept.md`
- `GROUNDING_MASTERPLAN.md`
- `TRAINING_MASTERPLAN.md`

Its purpose is different from architecture-only documents.

This file answers the question:

`What must the system actually be able to do, in the real world, across many domains, if it is to count as a truly advanced world-grounded intelligence system?`

This document is intentionally ambitious.

It does not describe only minimum implementation comfort. It defines the
intended target envelope of OMEN:

- what is explicitly required
- what is strongly desired
- what should be pursued if possible
- what practical outcomes the system should deliver

It is therefore a system-level target specification, not a short feature list.

## 2. Requirement Classes

The system requirements in this document are classified into four levels.

### 2.1 Explicitly Required

These are mandatory for canonical validity.

If OMEN does not satisfy them, it is not the intended system.

### 2.2 Strongly Desired

These are not the absolute minimum for conceptual validity, but they are
expected for a serious, mature, high-value system.

### 2.3 If Possible

These are frontier ambitions. They may require more time, more data, more
infrastructure, or deeper research maturity.

They should be pursued if doing so does not damage the more central properties
of the system.

### 2.4 Practical Outcome Requirements

These requirements are phrased not as internal abstractions, but as observable
real-world behavior:

- what results OMEN should deliver
- what kinds of tasks it should succeed on
- what quality level should be visible to operators

## 3. Global System Requirements

### 3.1 Explicitly Required

- The system must operate over a structured internal world state rather than
  treating text continuation as its only cognitive substrate.
- The system must be able to distinguish observation, hypothesis, rule,
  contradiction, uncertainty, and verified knowledge.
- The system must preserve provenance from high-level grounded objects back to
  source spans or source structures.
- The system must support memory that is operational, not archival only.
- The system must support real symbolic reasoning, not pseudo-symbolic pattern
  decoration.
- The system must support verification and revision instead of one-pass belief
  acceptance.
- The system must support compute-aware reasoning instead of unconditional
  maximal compute expenditure.
- The system must support online adaptation in a controlled manner.

### 3.2 Strongly Desired

- The system should maintain a stable long-horizon model of entities, events,
  claims, goals, and world changes across sessions or episodes.
- The system should remain useful under contradictory, noisy, or partially
  reliable input.
- The system should gracefully degrade when some subsystem is weak, rather than
  collapsing into incoherent output.
- The system should remain interpretable enough that failures can be diagnosed
  from internal artifacts.

### 3.3 If Possible

- The system should approach domain-general cognitive behavior under bounded
  compute rather than being narrow-task brittle.
- The system should gradually self-improve through guarded learning without
  needing hard retraining boundaries for every competence increase.
- The system should achieve strong transfer across language, domain, and task
  format without losing epistemic discipline.

### 3.4 Practical Outcome Requirements

- When given complex input, the system should produce outputs that are visibly
  grounded, structured, and internally coherent.
- When uncertainty is high, the system should look uncertain in the right way
  rather than sounding falsely confident.
- When evidence changes, the system should revise its internal state and final
  output instead of rationalizing old mistakes.
- When compute is limited, the system should still prioritize the most useful
  reasoning paths.

## 4. Grounding and Semantic Understanding Requirements

### 4.1 Explicitly Required

- The system must transform bytes into typed semantic evidence rather than only
  shallow text hints.
- The system must support multilingual grounding, including non-English and
  mixed-language input.
- The system must preserve multiple interpretations long enough for later
  verification when ambiguity is real.
- The system must ground entities, events, states, claims, goals, and
  relations as first-class objects.
- The system must distinguish generic claims from instance facts.
- The system must distinguish asserted claims from cited, hedged, or questioned
  claims.
- The system must support speaker attribution and source-aware claim handling.
- The system must support event roles, temporal anchors, and causal structure.
- The system must avoid treating heuristics as the primary semantic authority.

### 4.2 Strongly Desired

- The system should maintain persistent identity across aliases, references,
  pronouns, and repeated mentions.
- The system should support nested claims and multi-level attribution.
- The system should support discourse-aware interpretation of explanation,
  contrast, concession, instruction, and argument.
- The system should handle scientific text, dialogue, instructions, code
  comments, logs, and mixed documents with differentiated grounding behavior.
- The system should retain parser disagreement or semantic ambiguity instead of
  prematurely flattening everything into one hard form.

### 4.3 If Possible

- The system should induce latent event schemas and role templates from repeated
  structure across domains.
- The system should infer hidden but plausible intermediate events when direct
  evidence is incomplete.
- The system should form semantically stable canonical forms across very
  different phrasing styles, from terse technical syntax to natural narrative.

### 4.4 Practical Outcome Requirements

- If given a long messy real-world document, the system should recover the key
  actors, events, claims, dependencies, and uncertainties with usable fidelity.
- If given two documents describing the same event differently, the system
  should converge them where meaning overlaps and keep divergence explicit where
  evidence differs.
- If given dialogue with conflicting speakers, the system should not merge all
  claims into one undifferentiated truth state.
- If given instructions, the system should recover actionable structure instead
  of only paraphrasing the text.

## 5. World-State and World-Model Requirements

### 5.1 Explicitly Required

- The system must maintain a graph-structured world state as a primary internal
  representation.
- The system must support transition modeling, not only static fact storage.
- The system must detect when newly grounded information is compatible with or
  disruptive to the existing world state.
- The system must support counterfactual or alternative-world reasoning at
  least in bounded form.

### 5.2 Strongly Desired

- The system should maintain temporal continuity across episodes and sessions.
- The system should support local causal reasoning rather than only correlation
  accumulation.
- The system should support multiple partial world hypotheses under conflict.
- The system should use world-model mismatch as a training and verification
  signal.

### 5.3 If Possible

- The system should construct explicit latent causal programs or reusable
  transition templates.
- The system should support long-horizon simulation over structured world
  states.
- The system should predict downstream consequences of accepted or rejected
  claims before committing high-impact writes.

### 5.4 Practical Outcome Requirements

- If given changing operational information, the system should track what the
  world likely is now, not only repeat what was said earlier.
- If given contradictory event reports, the system should preserve competing
  world hypotheses and explain what would resolve them.
- If asked “what happens next if this is true?”, the system should reason over
  state transitions rather than inventing a loose narrative.

## 6. Memory Requirements

### 6.1 Explicitly Required

- The system must have both exact symbolic memory and learned/neural memory.
- Memory writes must not occur blindly.
- Each memory write must preserve source, confidence, and epistemic status.
- Memory retrieval must materially influence grounding, reasoning, planning, or
  generation when useful.
- The system must support memory consolidation and memory pressure control.

### 6.2 Strongly Desired

- The system should learn what is worth remembering and what should be left out.
- The system should prefer retention of reusable structures over noisy one-off
  fragments.
- The system should protect itself from self-poisoning through low-quality
  writes.
- The system should improve future task performance because of memory, not only
  expose more stored artifacts.

### 6.3 If Possible

- The system should form multi-timescale memory:
  working, episodic, semantic, and long-horizon conceptual memory.
- The system should learn memory compression primitives that preserve future
  utility.
- The system should support memory-guided ontology growth and memory-guided
  grounding repair.

### 6.4 Practical Outcome Requirements

- If a user returns to a domain after many episodes, the system should recover
  the relevant accumulated understanding without restating the entire past.
- If a memory item repeatedly creates contradiction or has low utility, the
  system should demote or retire it.
- If a fact matters later, the system should recall it in the right context and
  with the right confidence.

## 7. Symbolic Reasoning Requirements

### 7.1 Explicitly Required

- The system must support real terms, variables, unification, rules, and proof
  search.
- The system must support deduction, abduction, and induction.
- Rules must carry epistemic status.
- The system must distinguish proposed rules from verified rules.
- Contradicted rules must not remain silently authoritative.

### 7.2 Strongly Desired

- The system should learn which proof paths are promising.
- The system should learn which candidate rules are likely useful before
  polluting the KB.
- The system should support rule repair and exception handling.
- The system should support mixed reasoning over observed facts, memory,
  grounded hypotheses, and world context.

### 7.3 If Possible

- The system should discover reusable abstract relational templates from many
  episodes.
- The system should support compositional symbolic abstraction over complex
  grounded scenes.
- The system should learn hierarchical proof policies for long reasoning tasks.

### 7.4 Practical Outcome Requirements

- If the problem genuinely benefits from explicit reasoning, the system should
  outperform shallow language continuation.
- If a rule is useful repeatedly, the system should get better over time by
  retaining and calibrating it.
- If a rule is wrong, the system should surface contradiction, reduce trust, and
  avoid repeating the same failure forever.

## 8. Verification and Epistemic Discipline Requirements

### 8.1 Explicitly Required

- The system must attach confidence and provenance at object level.
- The system must support contradiction as a first-class state.
- The system must support deferred acceptance when evidence is insufficient.
- The system must support repair actions rather than only binary accept/reject.
- The system must not write low-status heuristic outputs as verified knowledge
  unless stronger downstream evidence confirms them.

### 8.2 Strongly Desired

- The system should calibrate support and conflict well enough that confidence
  reflects actual future correctness.
- The system should localize contradiction scope instead of marking entire
  scenes as globally invalid.
- The system should use memory corroboration, parser agreement, and world-model
  agreement jointly.
- The system should explicitly represent clarification needs.

### 8.3 If Possible

- The system should estimate which new evidence would be maximally informative
  for resolving uncertainty.
- The system should support graded epistemic states beyond simple labels where
  useful.
- The system should support selective human-auditable justification traces for
  high-stakes decisions.

### 8.4 Practical Outcome Requirements

- When the system is unsure, operators should be able to see why it is unsure.
- When evidence conflicts, the system should preserve conflict structure rather
  than hiding it in smooth language.
- When something is cited rather than directly observed, the system should act
  differently than when something is directly grounded and verified.

## 9. Learning and Adaptation Requirements

### 9.1 Explicitly Required

- The system must have a coherent training curriculum.
- The system must support joint training across major subsystems.
- The system must not rely on “text-only training is enough” as its core story.
- The system must support guarded online adaptation.
- The system must protect high-value verified knowledge during adaptation.

### 9.2 Strongly Desired

- The system should improve through interaction, execution, contradiction, and
  memory, not just through offline corpora.
- The system should support subsystem-specific maturity gates before stronger
  online updates are enabled.
- The system should route more training signal into the grounding stack where
  semantic failures originate.

### 9.3 If Possible

- The system should support continuous competence accretion under bounded drift.
- The system should support replay, shadow training, and reversible online
  updates at scale.
- The system should support adaptive curriculum scheduling based on subsystem
  weakness rather than static training phases only.

### 9.4 Practical Outcome Requirements

- Over time, the system should clearly improve in domains it repeatedly works
  on, without losing old competence or becoming epistemically unstable.
- If online adaptation is disabled, there should be a visible loss in long-run
  competence growth.
- If online adaptation is enabled, it should improve future outcomes without
  causing uncontrolled KB corruption.

## 10. Planning and Agency Requirements

### 10.1 Explicitly Required

- The planner must consume grounded state, not raw ambiguous prose as if it
  were already verified truth.
- The system must support explicit goals, constraints, and branch alternatives.
- The system must support planning under uncertainty.
- The system must support repair or re-planning when assumptions fail.

### 10.2 Strongly Desired

- The planner should incorporate verifier constraints and conflict markers.
- The planner should use memory and world predictions to avoid repeated failure.
- The planner should distinguish mandatory constraints from soft preferences.
- The planner should be able to explain which grounded facts a plan depends on.

### 10.3 If Possible

- The system should support long-horizon hierarchical planning.
- The system should support counterfactual branch evaluation before expensive
  action.
- The system should support multi-agent or multi-actor planning in shared
  grounded worlds.

### 10.4 Practical Outcome Requirements

- If asked to produce a plan from messy input, the system should first structure
  the situation, then plan from that structure.
- If key assumptions are weak, the plan should visibly reflect contingency
  rather than present fantasy certainty.
- If execution feedback arrives, the system should revise the plan rather than
  pretending the original plan still holds.

## 11. Generation Requirements

### 11.1 Explicitly Required

- Generation must be conditioned on structured state, memory, and reasoning
  outputs.
- The system must not degrade into a plain ungrounded decoder in generation
  mode.
- Generated content must be able to inherit provenance where appropriate.
- Generated code, explanations, and plans must be constrained by grounded
  context when that context exists.

### 11.2 Strongly Desired

- The system should generate answers that are more faithful, structured, and
  stable because of grounding.
- The system should generate alternate explanations for different audiences
  without changing underlying meaning.
- The system should support synthesis over many evidence sources without losing
  attribution.

### 11.3 If Possible

- The system should generate executable plans, formal artifacts, and code that
  remain traceable to grounded assumptions.
- The system should support iterative self-critique driven by verification,
  not just style smoothing.
- The system should support rich synthesis modes such as research reports,
  analytical briefs, scientific summaries, and multi-step operational guidance.

### 11.4 Practical Outcome Requirements

- If the system answers a hard question, the answer should look like the result
  of understanding and reasoning, not merely polished continuation.
- If the system writes code, that code should reflect grounded task facts,
  constraints, and expected behavior.
- If the system summarizes complex evidence, it should preserve uncertainty,
  attribution, and conflict structure where relevant.

## 12. Multilingual and Cross-Domain Requirements

### 12.1 Explicitly Required

- The system must not be English-only in its semantic design.
- The system must support meaning-preserving convergence across language
  variants where semantics is equivalent.
- The system must support mixed-language and mixed-format inputs.

### 12.2 Strongly Desired

- The system should perform well enough in Ukrainian, English, and mixed
  technical language to remain operationally credible.
- The system should avoid silent language-specific degradation hidden behind
  aggregate metrics.
- The system should preserve morphological and syntactic distinctions that
  matter for meaning.

### 12.3 If Possible

- The system should support strong zero-shot or low-shot transfer to additional
  languages.
- The system should support cross-lingual memory retrieval and concept
  continuity.
- The system should support domain transfer from scientific and coding material
  to operational and conversational settings while preserving epistemic rigor.

### 12.4 Practical Outcome Requirements

- A bilingual operator should not need to “translate into OMEN-friendly
  English” just to get serious performance.
- Scientific, technical, conversational, and operational materials should all
  remain usable with domain-sensitive handling rather than one monolithic text
  path.

## 13. Human Interaction Requirements

### 13.1 Explicitly Required

- The system must make room for operator hints without allowing operators to
  silently override truth checks.
- The system must expose enough structure that a human can see what kind of
  object the system thinks it is handling.
- The system must expose useful confidence, source, and contradiction signals.

### 13.2 Strongly Desired

- The system should allow human correction at the right abstraction level:
  entity identity, claim status, rule validity, plan assumption, memory write
  priority, and so on.
- The system should support explanation of internal failures in understandable
  system terms.
- The system should support interactive clarification loops when ambiguity
  matters.

### 13.3 If Possible

- The system should support collaborative knowledge shaping with experts while
  preserving explicit provenance and epistemic state.
- The system should support review workflows for high-impact memory or rule
  changes.

### 13.4 Practical Outcome Requirements

- Operators should be able to see not only what the system answered, but what
  the system believes, what it doubts, and what evidence that belief rests on.
- Human correction should improve future behavior instead of disappearing as a
  one-off patch.

## 14. Reliability, Efficiency, and Engineering Requirements

### 14.1 Explicitly Required

- The system must have measurable subsystem metrics, not only overall output
  quality.
- The system must support graceful degradation when one module is weak.
- The system must support bounded compute behavior through EMC or equivalent
  policy.
- The system must avoid hidden dependence on brittle heuristics.

### 14.2 Strongly Desired

- The system should expose replayable internal artifacts for debugging.
- The system should support shadow evaluation for risky updates.
- The system should remain operationally useful under constrained compute.
- The system should maintain auditability for major writes and rule changes.

### 14.3 If Possible

- The system should support dynamic resource allocation across grounding,
  reasoning, planning, and generation depending on value of computation.
- The system should support distributed or asynchronous subsystems while
  preserving coherent state transitions.
- The system should support training-time and runtime introspection strong
  enough for scientific-style diagnostics.

### 14.4 Practical Outcome Requirements

- The system should not need maximum compute for every hard request.
- If something fails, maintainers should be able to identify whether the error
  came from grounding, memory, symbolic reasoning, planning, generation, or
  adaptation policy.

## 15. Safety, Self-Improvement, and Anti-Failure Requirements

### 15.1 Explicitly Required

- The system must not silently promote speculative knowledge to verified
  knowledge.
- The system must not let online updates destroy stable verified structures.
- The system must not let planners act on unresolved ambiguous text as if it
  were settled world truth.
- The system must not let the generator bypass the epistemic controls of the
  rest of the architecture.

### 15.2 Strongly Desired

- The system should detect self-reinforcing error loops.
- The system should quarantine low-trust changes.
- The system should support rollback and conflict-preserving repair.
- The system should treat recurring contradiction as a signal for deeper model
  change, not just local patching.

### 15.3 If Possible

- The system should estimate the long-term risk of self-modification or rule
  growth before acceptance.
- The system should learn stable self-improvement policies that remain aligned
  with epistemic integrity.
- The system should be able to grow capability without growing unbounded hidden
  fragility.

### 15.4 Practical Outcome Requirements

- Over months of use, the system should become more capable without becoming
  more delusional.
- Stronger autonomy should correspond to stronger verification and stronger
  rollback, not weaker control.

## 16. Practical Outcome Requirements By Domain

### 16.1 Research and Analysis

- The system should read large mixed evidence packs and produce structured
  conclusions with explicit uncertainties and source distinctions.
- The system should detect when two papers, reports, or sources disagree in
  claim content, not just in wording.
- The system should preserve citations, claim status, and causal caveats.

### 16.2 Coding and Software Engineering

- The system should ground code, comments, tests, configs, and repository
  structure into one coherent task model.
- The system should distinguish verified runtime behavior from inferred but
  unverified assumptions.
- The system should use execution traces, failing tests, and code structure as
  first-class training and reasoning signal.
- The system should improve software reasoning through real grounded program
  understanding rather than surface code token statistics only.

### 16.3 Operations and Procedures

- The system should recover explicit procedures, constraints, dependencies,
  preconditions, and failure branches from operational documents.
- The system should warn when procedural steps rely on weak assumptions.
- The system should generate contingency-aware plans from grounded procedures.

### 16.4 Dialogue and Multi-Agent Interaction

- The system should preserve speaker identity and claim ownership.
- The system should not flatten disagreements across speakers into one blended
  summary.
- The system should reason about beliefs, reports, requests, and intentions in
  a differentiated way.

### 16.5 Logs, Incidents, and Diagnostics

- The system should extract events, transitions, error chains, and suspected
  causes from noisy logs.
- The system should identify plausible hidden causes when direct evidence is
  missing.
- The system should preserve which parts are observed, inferred, or merely
  proposed.

### 16.6 Scientific and Technical Knowledge

- The system should distinguish established claims from tentative claims,
  hypotheses, and cited external claims.
- The system should preserve conditionality, uncertainty, and evidence quality.
- The system should avoid turning scientific prose into oversimplified
  fact-like assertions.

## 17. Maturity Levels

### 17.1 Minimum Valid System

- grounded state exists
- symbolic substrate is real
- memory is operational
- verification exists
- some online adaptation path exists
- compute economy exists

### 17.2 Strong Research-Grade System

- robust grounding on multiple input types
- usable memory gains
- real symbolic utility
- visible contradiction handling
- stable planning from grounded inputs
- generator fidelity to state

### 17.3 High-End Practical System

- cross-domain reliability
- multilingual operational usefulness
- strong long-horizon consistency
- low KB pollution
- compute-aware competence
- useful online improvement

### 17.4 Frontier Cognitive System

- durable world continuity
- broad transfer
- hidden-cause reasoning
- self-improving but epistemically disciplined learning
- structured synthesis across many evidence streams

## 18. Anti-Goals

The following outcomes must not be mistaken for success:

- polished but weakly grounded output
- large memory with low future utility
- many rules with poor verification quality
- better surface fluency at the cost of epistemic honesty
- planner activity without grounded state quality
- online self-modification without rollback discipline
- multilingual appearance with actual English-only semantics underneath
- heuristic complexity mistaken for genuine semantic understanding

## 19. Final Directive

The final intended system is not merely:

- a language model with some tools
- a symbolic system with some embeddings
- a world model with some text adapters
- a memory store with some retrieval

The intended system is:

a world-grounded, memory-using, symbolically capable, verification-disciplined,
compute-aware intelligence runtime that can convert messy reality into
structured internal understanding, reason over that understanding, revise it
when wrong, and produce practical high-quality outcomes across many domains.

If a proposed design change increases benchmark scores while weakening any of
the following:

- grounding fidelity
- provenance
- verification discipline
- memory usefulness
- symbolic truth maintenance
- compute efficiency
- online stability
- practical real-world outcome quality

then that change should be treated as suspect until proven system-improving at
the full architecture level.

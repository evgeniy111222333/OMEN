# System Notebook

This file is the repository-local operating manual for the durable memory system.

Canonical durable knowledge belongs in the `memory` MCP.
This file is not the primary store for session-derived findings.

Use this file for:
- memory policy
- category taxonomy
- retrieval recipes
- naming conventions
- templates
- navigation rules

Do not use this file as the main sink for new durable findings unless the user explicitly asks for file-based documentation.

## Quick Index

- [Operating Rules](#operating-rules)
- [Memory Inspection Recipes](#memory-inspection-recipes)
- [Memory Categories](#memory-categories)
- [Memory Naming Rules](#memory-naming-rules)
- [When To Write To Memory](#when-to-write-to-memory)
- [What Not To Store](#what-not-to-store)
- [Revalidation Rules](#revalidation-rules)
- [Templates](#templates)

## Operating Rules

- Read this file before deep multi-file work only to understand how to use `memory`.
- Search `memory` first for durable facts.
- Write durable findings to `memory`, not here.
- Prefer short durable notes over long transcripts.
- Every durable memory note should have:
  - a category
  - a priority when useful
  - a status when useful
  - a verification date when useful
  - a source when useful
- If a durable note becomes unreliable after a refactor or re-index, update the corresponding memory entity and mark it stale or needs-revalidation there.

## Memory Inspection Recipes

- To inspect the full memory graph, use `read_graph`.
- To look up a topic, use `search_nodes` with the category, module name, subsystem name, or durable topic.
- To inspect a specific result in detail, use `open_nodes`.
- To inspect OMEN durable memory first, search for:
  - `OMEN Repository`
  - `Workflow:DurableMemory`
  - `Tooling:CodeGraphContext`
  - `GroundingFlow:`
  - `RefactorMap:`
  - `ModuleUnderstanding:`
  - `OpenQuestion:`

## Memory Categories

- `architecture`
- `grounding-flow`
- `config-defaults`
- `test-strategy`
- `user-preference`
- `refactor-map`
- `module-understanding`
- `verified-search`
- `open-question`

## Memory Naming Rules

- Use entity names such as `Architecture:<topic>`, `GroundingFlow:<topic>`, `ConfigDefault:<module>`, `TestStrategy:<subsystem>`, `UserPreference:<topic>`, `RefactorMap:<topic>`, `ModuleUnderstanding:<module>`, `VerifiedSearch:<topic>`, `OpenQuestion:<topic>`.
- Prefer short searchable IDs such as `ARCH-2026-04-20-01`, `FLOW-2026-04-20-01`, `REF-2026-04-20-01`, `MOD-2026-04-20-01`.
- Include module path, subsystem name, or user-facing topic when it improves future retrieval.

## When To Write To Memory

- After an important architecture conclusion.
- After a successful refactor.
- After a deep module analysis.
- After a costly search chain that produced a durable answer.
- After discovering a stable user preference.
- After a blocked investigation that should be resumed later.

## What Not To Store

- Secrets, credentials, private keys, or sensitive personal data.
- Temporary PIDs, transient locks, or current-process state.
- Intermediate output from a single test run.
- Concrete environment-variable values unless they are safe and durable defaults.
- Errors that were fixed immediately and have no future diagnostic value.
- Temporary branches or other short-lived workspace noise.

## Revalidation Rules

- Revalidate relevant memory entries after major refactors, module moves, or `codegraph` re-indexes tied to structural changes.
- Mark stale notes in memory as `stale` or `needs-revalidation`.
- If memory conflicts with the repository, trust the repository and update memory afterward.

## Templates

### Durable Note Template

```md
Entity: `Architecture:<topic>`
Category: `architecture`
Priority: `P0`
Status: `active`
Verified: `YYYY-MM-DD`
Source: codegraph + direct file read + tests
Summary: one-line durable conclusion
Details:
- key point
- key point
Revalidate when:
- concrete condition that would make the note stale
```

### Module Understanding Template

```md
Entity: `ModuleUnderstanding:<module>`
Category: `module-understanding`
Priority: `P1`
Status: `active`
Verified: `YYYY-MM-DD`
Module: `path/to/module.py`
Purpose: one or two sentences
Public Entry Points:
- `symbol_a`
- `symbol_b`
Inbound Dependencies:
- file or module
Outbound Dependencies:
- file or module
Config Location:
- file or symbol
Relevant Tests:
- path
Notes:
- short durable observation
```

### Refactor Map Template

```md
Entity: `RefactorMap:<topic>`
Category: `refactor-map`
Priority: `P1`
Status: `active`
Verified: `YYYY-MM-DD`
Scope: short refactor name
Files Moved:
- `old/path.py` -> `new/path.py`
Symbols Renamed:
- `old_symbol` -> `new_symbol`
Entry Points Changed:
- what changed
Compatibility Notes:
- shims, deprecations, or breakage details
Revalidate when:
- concrete condition
```

### Verified Search Template

```md
Entity: `VerifiedSearch:<topic>`
Category: `verified-search`
Priority: `P2`
Status: `active`
Verified: `YYYY-MM-DD`
Query: short description of what was searched
Scope: module or subsystem
Source: codegraph or filesystem search
Answer:
- concise durable result
Revalidate when:
- concrete condition
```

### Open Question Template

```md
Entity: `OpenQuestion:<topic>`
Category: `open-question`
Priority: `P1`
Status: `open`
Opened: `YYYY-MM-DD`
Scope: subsystem or file
Question: unresolved question
Current Blocker:
- why it could not be answered yet
Next Step:
- concrete next investigation step
```

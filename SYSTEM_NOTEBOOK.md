# System Notebook

This file is the repository-local working notebook for durable investigation notes.

Use it together with the `memory` MCP:
- `memory` stores cross-session graph knowledge
- this notebook stores navigable repo-local notes, templates, and verification history

This notebook is not a chat log.
It is not a dump of temporary command output.
It is a durable navigation layer for future sessions.

## Quick Index

- [Operating Rules](#operating-rules)
- [Category Index](#category-index)
- [Current Durable Notes](#current-durable-notes)
- [Architecture Notes](#architecture-notes)
- [Grounding Flow Notes](#grounding-flow-notes)
- [Config Defaults](#config-defaults)
- [Test Strategy Notes](#test-strategy-notes)
- [User Preferences](#user-preferences)
- [Refactor Maps](#refactor-maps)
- [Module Understanding Notes](#module-understanding-notes)
- [Verified Search Notes](#verified-search-notes)
- [Open Questions](#open-questions)
- [Stale Or Revalidation Queue](#stale-or-revalidation-queue)
- [Templates](#templates)

## Operating Rules

- Read the `Quick Index` first for repo-wide, architecture, or multi-session work.
- Prefer short durable notes over long transcripts.
- Every durable note should have:
  - an ID
  - a category
  - a priority
  - a status
  - a verification date
  - a source
- If a note is no longer reliable after a refactor or re-index, move or copy it to `Stale Or Revalidation Queue`.
- If a note is important across sessions, mirror it into the `memory` MCP using the same category/topic naming.

## Category Index

- `architecture` -> hub modules, subsystem boundaries, high-level structure
- `grounding-flow` -> verified end-to-end grounding, canonicalization, symbolic, planner, or world-state paths
- `config-defaults` -> stable defaults, toggles, configuration sources
- `test-strategy` -> durable validation strategy and where subsystem tests live
- `user-preference` -> stable user workflow and communication preferences
- `refactor-map` -> migration notes after non-trivial structural changes
- `module-understanding` -> concise notes for a specific module
- `verified-search` -> reusable answers from expensive search chains
- `open-question` -> unresolved but important questions with next steps

## Current Durable Notes

### TOOL-2026-04-20-01
- Category: `architecture`
- Priority: `P0`
- Status: `active`
- Verified: `2026-04-20`
- Source: local `cgc` behavior, reset-script verification, live startup probe
- Summary: local `CodeGraphContext` with the global Kuzu database behaves like a single-owner resource for `cgc.exe mcp start`
- Details:
  - concurrent chat sessions that each try to start `codegraph` can conflict on `C:\Users\HP\.codegraphcontext\global\kuzudb`
  - the usual failure mode is a startup error or lock error during MCP initialization
  - safe recovery is to run `tools/reset-codegraph.cmd` rather than leaving a manually started long-lived `cgc mcp start` process

### TOOL-2026-04-20-02
- Category: `config-defaults`
- Priority: `P1`
- Status: `active`
- Verified: `2026-04-20`
- Source: repository tooling setup
- Summary: `AGENTS.md` defines the current tool hierarchy
- Details:
  - `codegraph` is the default authority for structural relationships
  - `filesystem` is the authority for exact local file contents
  - `memory` and this notebook form the durable knowledge layer
  - `github` is for remote repository truth
  - `context7` is for third-party documentation

## Architecture Notes

No durable architecture notes recorded yet.

## Grounding Flow Notes

No durable grounding-flow notes recorded yet.

## Config Defaults

No additional config-default notes recorded yet.

## Test Strategy Notes

No durable test-strategy notes recorded yet.

## User Preferences

No durable user-preference notes recorded yet.

## Refactor Maps

No refactor maps recorded yet.

## Module Understanding Notes

No module-understanding notes recorded yet.

## Verified Search Notes

No verified-search notes recorded yet.

## Open Questions

No open questions recorded yet.

## Stale Or Revalidation Queue

Move notes here when:
- repository structure changed materially
- `codegraph` was re-indexed after structural changes
- a module was split, merged, renamed, or moved
- an old investigation result may no longer be safe to trust

## Templates

### Durable Note Template

```md
### ARCH-YYYY-MM-DD-01
- Category: `architecture`
- Priority: `P0`
- Status: `active`
- Verified: `YYYY-MM-DD`
- Source: codegraph + direct file read + tests
- Summary: one-line durable conclusion
- Details:
  - key point
  - key point
- Revalidate when:
  - concrete condition that would make the note stale
```

### Module Understanding Template

```md
### MOD-YYYY-MM-DD-01
- Category: `module-understanding`
- Priority: `P1`
- Status: `active`
- Verified: `YYYY-MM-DD`
- Module: `path/to/module.py`
- Purpose: one or two sentences
- Public Entry Points:
  - `symbol_a`
  - `symbol_b`
- Inbound Dependencies:
  - file or module
- Outbound Dependencies:
  - file or module
- Config Location:
  - file or symbol
- Relevant Tests:
  - path
- Notes:
  - short durable observation
```

### Refactor Map Template

```md
### REF-YYYY-MM-DD-01
- Category: `refactor-map`
- Priority: `P1`
- Status: `active`
- Verified: `YYYY-MM-DD`
- Scope: short refactor name
- Files Moved:
  - `old/path.py` -> `new/path.py`
- Symbols Renamed:
  - `old_symbol` -> `new_symbol`
- Entry Points Changed:
  - what changed
- Compatibility Notes:
  - shims, deprecations, or breakage details
- Revalidate when:
  - concrete condition
```

### Verified Search Template

```md
### SEARCH-YYYY-MM-DD-01
- Category: `verified-search`
- Priority: `P2`
- Status: `active`
- Verified: `YYYY-MM-DD`
- Query: short description of what was searched
- Scope: module or subsystem
- Source: codegraph or filesystem search
- Answer:
  - concise durable result
- Revalidate when:
  - concrete condition
```

### Open Question Template

```md
### Q-YYYY-MM-DD-01
- Category: `open-question`
- Priority: `P1`
- Status: `open`
- Opened: `YYYY-MM-DD`
- Scope: subsystem or file
- Question: unresolved question
- Current Blocker:
  - why it could not be answered yet
- Next Step:
  - concrete next investigation step
```

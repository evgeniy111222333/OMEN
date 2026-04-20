## Code Navigation Policy

Use the `codegraph` MCP server as the default source of truth for repository structure and code relationships.

Always start with `codegraph` when the task involves:
- symbol lookup
- callers or callees
- dependency tracing
- impact analysis
- class hierarchy or inheritance
- dead code checks
- architecture flow across multiple files
- refactors that may affect several modules
- questions about how grounding, symbolic execution, canonicalization, or tests connect across the repo

Use `filesystem` and direct file reads after `codegraph` has narrowed the relevant files, symbols, or paths.

The assistant may inspect any file in this repository when relevant to the task.

Do not limit analysis to only graph-indexed symbols. Use direct file reads, repository search, and test inspection whenever the answer depends on exact file contents, configuration, prompts, docs, fixtures, or non-code assets.

Do not answer relationship-heavy questions from guesswork when `codegraph` can verify them.

## Repo-Specific Guidance

For this repository, prefer `codegraph` first when the task touches:
- `omen_grounding/*`
- `omen_symbolic/*`
- `omen_canonical.py`
- tests that validate grounding, world graph behavior, execution traces, or cross-module runtime flow

When analyzing grounding work, use `codegraph` to map how parsing, normalization, routing, execution traces, world-state logic, and tests connect before proposing changes.

When a change spans multiple modules, use `codegraph` to estimate blast radius before editing code.

## Tool Hierarchy

Use tools according to the shape of the task:
- `codegraph` for structural relationships, call chains, dependency tracing, impact analysis, and cross-file architecture flow
- `filesystem` and direct file reads for exact local file contents, configs, prompts, docs, fixtures, and non-code assets
- `memory` for canonical durable cross-session knowledge about the user, the repository, and long-lived operating constraints
- `SYSTEM_NOTEBOOK.md` for the repository-local operating manual, memory taxonomy, retrieval recipes, and templates
- `github` for remote repository state, PRs, issues, CI, review threads, branch metadata, and remote code search
- `context7` for third-party library/framework/package documentation and version-sensitive external API behavior

Never substitute one tool class for another when a more authoritative tool is available.

## Memory Guidance

Treat `memory` MCP as the canonical durable memory store.

`SYSTEM_NOTEBOOK.md` is not the primary store for session-derived durable knowledge.
It exists only for:
- memory policy
- category taxonomy
- retrieval recipes
- naming conventions
- templates
- navigation rules

Absolute durable-write rule:
- durable findings must be written to `memory`
- durable preferences must be written to `memory`
- durable architecture conclusions must be written to `memory`
- refactor maps must be written to `memory`
- verified-search results must be written to `memory`
- module-understanding notes must be written to `memory`
- open questions that should survive the session must be written to `memory`
- do not use repo files as the primary store for those facts
- only write durable knowledge into a file if the user explicitly asks for file-based documentation or if a repository artifact is required for non-memory consumers

Read protocol:
- before major architecture analysis, deep debugging, repo-wide refactors, or grounding/canonical/symbolic investigation, search `memory` first
- read `SYSTEM_NOTEBOOK.md` for policy, taxonomy, and retrieval rules
- for small local edits, use memory only when it is likely to reduce repeated exploration

Operational memory flow:
- start with `search_nodes` using the category, module name, subsystem name, or durable topic
- if matching nodes exist, use `open_nodes` on the most relevant ones before repeating exploration
- prefer updating an existing entity over creating a near-duplicate entity
- create new entities only when the fact is likely to matter again across sessions
- use relations when they improve future retrieval of architecture or workflow facts
- after writing to `memory`, ensure the topic can be found later by category name, module name, or subsystem name

Write-back protocol:
- after an important architecture conclusion, write a concise durable summary to `memory`
- after a successful refactor, record a migration map in `memory`
- after a deep module analysis, record a module-understanding note in `memory`
- after a long search chain that produced a durable answer, record the verified result with a verification date in `memory`
- after discovering a stable user preference, store it in `memory`
- after a blocked investigation with a clear follow-up path, record the unresolved question and next verification plan in `memory`
- update `SYSTEM_NOTEBOOK.md` only when the memory policy, navigation instructions, templates, or user-requested file documentation should change

Memory categories and priorities:
- `architecture` for durable structure, hub modules, ownership boundaries, or central abstractions
- `grounding-flow` for verified grounding, canonicalization, symbolic, planner, or world-state execution paths
- `config-defaults` for stable defaults, toggles, feature flags, and environment assumptions
- `test-strategy` for durable test locations, invariants, and subsystem validation rules
- `user-preference` for stable language, style, workflow, and collaboration preferences
- `refactor-map` for migration history after non-trivial structural changes
- `module-understanding` for concise notes about a specific module's purpose and dependencies
- `verified-search` for expensive search results that are likely to be reused
- `open-question` for unresolved but important questions with a next-step plan

Priority rules:
- `P0` critical architecture or workflow facts that should be checked early in future sessions
- `P1` important subsystem notes that save repeated investigation
- `P2` useful context that helps navigation but is not session-critical
- `P3` optional background or historical notes

Memory naming conventions:
- use entity names such as `Architecture:<topic>`, `GroundingFlow:<topic>`, `ConfigDefault:<module>`, `TestStrategy:<subsystem>`, `UserPreference:<topic>`, `RefactorMap:<topic>`, `ModuleUnderstanding:<module>`, `VerifiedSearch:<topic>`, `OpenQuestion:<topic>`
- prefer short searchable IDs such as `ARCH-2026-04-20-01`, `FLOW-2026-04-20-01`, `REF-2026-04-20-01`, `MOD-2026-04-20-01`
- when possible, include the module path, subsystem name, or user-facing topic in the entity name so future `search_nodes` calls are predictable

What belongs in durable memory:
- stable user preferences about language, detail level, workflow, and collaboration style
- durable repo-level architecture constraints and non-obvious module relationships
- long-lived tooling constraints that matter across sessions
- accepted conventions, migration outcomes, or decisions that are not obvious from one file alone
- verified module roles, central hubs, important data-flow edges, and durable search results

What must not be stored in durable memory:
- secrets, tokens, credentials, private keys, or sensitive personal data
- one-off errors, temporary PIDs, transient locks, or current-process state
- speculative conclusions that have not been verified
- intermediate output from a single `pytest` run
- concrete environment-variable values unless they are safe, stable, and necessary as defaults
- details of an error that was fixed immediately and has no durable diagnostic value
- ephemeral code states, temporary branches, or short-lived experiments

Refactor-memory policy:
- after a structural refactor, record the old and new names, moved files, changed entry points, and any compatibility shims in `memory`
- if a refactor invalidates old memory notes, mark them as `stale` or `needs-revalidation`
- prefer recording the migration map immediately after the refactor is verified

Verified-search policy:
- if the assistant spends multiple search steps finding callers, callees, ownership boundaries, or rare symbol usage, summarize the result as a reusable verified-search note in `memory`
- include the scope, source of verification, and date
- do not store full raw search dumps; store the compact answer and where it was verified

Revalidation policy:
- if `codegraph` is re-indexed, the repository structure changes materially, or a module is split or renamed, treat related architecture, flow, and refactor notes as potentially stale
- mark stale notes explicitly in `memory`
- when using stale notes, re-check them against the repo before relying on them
- if memory conflicts with the repository, trust the repository and update memory later

Codegraph-synergy rule:
- if `codegraph` is temporarily unavailable, the assistant may use recent memory notes as a navigation map for `filesystem` exploration
- once `codegraph` is available again, verify any structural claim that came from memory rather than fresh graph inspection

User-preference policy:
- treat as stable preferences only things the user has expressed repeatedly or explicitly as a standing rule
- likely candidates include preferred language for repo-level system docs, preferred response detail, comment-language expectations, tolerance for temporary branches, and preferred recovery behavior for local tools
- do not overfit one-off phrasing into a permanent preference

Open-question policy:
- if a high-value question could not be answered in the current session, record it as an `open-question` with a short reason, current blockers, and the next concrete investigation step
- revisit open questions when future work touches the same subsystem

Memory-first navigation rule:
- for repository-wide or multi-step work, search `memory` first by category, subsystem, and module name
- use `SYSTEM_NOTEBOOK.md` as the operating manual for how to search and structure memory, not as the canonical fact store
- keep durable knowledge in `memory`, not in ad hoc repo files

Memory workflow:
- search or read memory before asking repeated questions about stable preferences or constraints
- read `SYSTEM_NOTEBOOK.md` for the retrieval rules and templates before deep work that spans multiple files or sessions
- treat memory as a durable hint layer, then verify against the repo when accuracy matters
- after learning a durable fact, write it back succinctly
- if memory conflicts with the repo, trust the repo and update memory later

## GitHub Guidance

Use `github` when the task involves remote repository state rather than only the local checkout.

Prefer `github` for:
- pull request metadata
- issue history and comments
- CI status and checks
- review threads and requested changes
- branch existence on remote
- remote file lookup or cross-repo search

Prefer the local repo for:
- actual code edits
- running tests
- local diffs
- final verification of local workspace behavior

Remote mutations require explicit user intent or approval:
- creating issues
- adding issue comments
- opening or updating PRs
- submitting PR reviews
- creating remote branches
- merging PRs
- pushing remote file changes

When local and remote state differ, say which source is authoritative for the current answer.

## Context7 Guidance

Use `context7` for external package and framework documentation, especially when behavior is version-sensitive or unfamiliar.

Default `context7` workflow:
- resolve the library ID first
- choose the most authoritative and relevant library match
- prefer the exact library version when known
- query narrowly for the API, behavior, or migration topic you actually need
- translate the external doc answer carefully into the local repo context

Use `context7` before guessing about:
- third-party APIs
- library configuration
- framework lifecycle behavior
- breaking changes or migration details
- package-specific patterns or recommended usage

Do not use `context7` for:
- repository-local truth
- local file contents
- local runtime behavior that can be checked directly

If docs and the local code appear to disagree, state the discrepancy explicitly.

## Fallback Behavior

Do not assume that mentioning `codegraph` in this file means the current chat session actually has a working `codegraph` MCP tool mounted.

If `codegraph` is expected for the task but appears unavailable, unhealthy, or missing from the current tool session, attempt a recovery flow before falling back.

Recommended recovery order:
- verify local Codex MCP configuration with `codex mcp list` or `codex mcp get codegraph`
- verify the local `cgc` executable exists and can run
- verify the local graph DB is readable with a lightweight check such as `cgc list`
- if startup fails because of a lock or stale process, inspect running `cgc` processes and attempt safe recovery
- prefer the repository helper `tools/reset-codegraph.cmd` or `tools/reset-codegraph.ps1` for safe local recovery
- if elevated permissions are required to inspect or stop a stale local process, request approval and proceed
- if repository structure changed materially, re-index before making strong structural claims
- retry the `codegraph` path once after recovery before declaring it unavailable

When diagnosing `codegraph`, distinguish clearly between:
- MCP is configured locally
- MCP is mounted in this chat session
- MCP server starts successfully
- graph database is readable

Do not skip directly to `filesystem` fallback when a short local recovery attempt could restore `codegraph`.

If `codegraph` fails with messages such as:
- `connection closed: initialize response`
- `MCP startup failed`
- `Could not set lock on file`

then treat that as a likely stale `cgc.exe mcp start` lock on the global Kuzu database.

Recovery rule:
- do not manually launch a long-lived `cgc mcp start` process as a fix
- run `tools/reset-codegraph.cmd`
- retry the MCP path once
- only then fall back to `filesystem`

If `codegraph` is unavailable, stale, or clearly missing a file, fall back to `filesystem` plus targeted search, and state that the graph data could not be used.

If a task requires broad repository inspection, the assistant may read across the whole repo first and then use `codegraph` to verify structural relationships.

If repository structure changed materially, re-index before making strong architectural claims.

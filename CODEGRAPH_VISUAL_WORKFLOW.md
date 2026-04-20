# CodeGraph Visual Workflow

This repository now has a one-click visual workflow for CodeGraph.

## Recommended Launcher

Run:

```cmd
.\tools\codegraph-visual.cmd
```

What it does automatically:

- creates a fresh pair of visual contexts for this launch
- indexes one context and launches the browser UI
- keeps a standby context ready
- polls the repository for changes
- waits for edits to settle
- refreshes the standby context incrementally
- swaps the UI to the refreshed context on the same port

This avoids the Kuzu DB lock conflict between `visualize` and `index`, and also avoids stale locks from previous launches.

## Default URL

```text
http://localhost:8010/index.html
```

## Default Behavior

- incremental refresh is automatic
- no manual `refresh` step is required
- the browser opens automatically
- leave the controller window open while editing

## Optional Arguments

Use a different port:

```cmd
.\tools\codegraph-visual.cmd -Port 8020
```

Force a clean rebuild on startup:

```cmd
.\tools\codegraph-visual.cmd -ForceInitialRebuild
```

Tune the polling interval and debounce:

```cmd
.\tools\codegraph-visual.cmd -PollSeconds 4 -DebounceSeconds 2
```

## Manual Tools

The old helper scripts still exist for manual control:

- `tools/codegraph-visual-init.ps1`
- `tools/codegraph-visual-refresh.ps1`
- `tools/codegraph-visualize.ps1`

## Notes

- The graph covers the codebase itself: all Python source files are indexed.
- Non-code files such as Markdown, JSON, or INI are not expected to become graph entities.
- Codex can still read those files directly through repository file access.
- If the page does not update automatically after a swap, reload the browser tab once.

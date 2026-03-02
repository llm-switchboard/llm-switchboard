# Architecture

## Overview

llm-switchboard is a Python CLI that bridges Open WebUI's model catalog with Claude Code. It fetches available models from Open WebUI's API, presents them in an interactive TUI, and launches Claude Code with the selected model by setting environment variables that route through Open WebUI as a proxy.

## Data Flow

```
User → bin/llm-switchboard → cli.py (arg parsing)
                            ├── Non-interactive: --list-models, --select → stdout
                            └── Interactive: tui.py → readkey loop
                                                  ↓
                              webui.py → Open WebUI API → models list
                                                  ↓
                              tui.py renders → user picks model
                                                  ↓
                              launch_claude() → subprocess.run(claude ...)
                                  ├── env: ANTHROPIC_BASE_URL = OWUI/api
                                  ├── env: ANTHROPIC_AUTH_TOKEN = OWUI key
                                  └── session.py → watches JSONL for usage
```

## Module Responsibilities

### cli.py
Main entry point. Parses CLI arguments, dispatches to interactive or non-interactive handlers. Owns `fetch_models()` which calls the API and builds the global model list. Pushes model state to `tui.py` globals for rendering.

### tui.py
All terminal-dependent code: raw-mode key reading, ANSI rendering, spinner, screen clearing, the interactive picker loop. References global model state (`MODELS`, `MODEL_MAP`, etc.) that `cli.py` populates.

### webui.py
Thin HTTP client using `urllib.request`. Two functions: `api_get()` and `api_post()`. Returns `{"_error": ...}` on failure instead of raising.

### models.py
Data structures (`Model` NamedTuple), provider detection from URLs (`provider_from_url`), domain-to-provider mapping, provider rate limits.

### config.py
Manages persistent config: favorites file, last-model file, directory creation. All functions accept optional `Path` parameters for testability.

### cache.py
Generic file-based caching with TTL. Used for Open WebUI connection config and Ollama config.

### free_tier.py
Parses `free_providers.conf` into `FreeTierRule` objects. Handles Gemini pricing cache (fetch from Google, parse markdown, cache locally). Provides `is_free_tier()` for model classification.

### session.py
`SessionWatcher`: background thread that tails Claude Code's session JSONL files to track token usage in real-time. Also handles `usage.json` persistence (load, save, record, aggregate).

### util.py
Pure functions with no side effects (except `die()`): string sanitization, ANSI stripping, price/token formatting, color constants, terminal detection.

## Global State

Model data is stored in module-level globals in `tui.py` (for rendering) and populated by `cli.py`'s `fetch_models()` via `_populate_tui_globals()`. This mirrors the original monolith's approach and avoids threading a context object through every render function.

## Key Design Constraints

- **stdlib-only**: No pip dependencies. Uses `urllib`, `json`, `termios`, `tty`, `select`, `threading`, `subprocess`.
- **Single process**: Claude Code runs as a subprocess. The watcher thread tails its output files.
- **Path-based config**: XDG-compliant. Config and cache are separate directories.
- **Backwards compatible CLI**: All original flags work. New flags are additive.

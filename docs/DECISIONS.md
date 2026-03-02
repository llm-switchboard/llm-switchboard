# Design Decisions

## D1: stdlib-only (no external Python dependencies)

The tool runs on personal servers and workstations where installing pip packages may not be desirable or possible. Keeping everything in the standard library means the script can be copied and run anywhere with Python 3.11+. The only external dependency is the `claude` CLI binary and optionally `fzf`.

## D2: Open WebUI as proxy, not direct API access

Rather than managing multiple API keys for each provider, llm-switchboard routes everything through Open WebUI. This means:
- One API key to manage
- Open WebUI handles auth, rate limiting, and provider routing
- Model catalog comes from Open WebUI's unified `/api/models` endpoint

## D3: Global mutable state for model data

The TUI rendering functions reference module-level globals (`MODELS`, `MODEL_MAP`, etc.) rather than passing a context object. This was inherited from the monolith and retained because:
- The data is populated once at startup and is read-only thereafter
- Threading a context through 20+ render/helper functions adds noise
- The `_populate_tui_globals()` bridge makes the coupling explicit

## D4: Config file format (plain text, not TOML/YAML)

Favorites and free-tier rules use simple line-based formats rather than structured config files. This keeps parsing trivial (no dependencies) and makes the files easy to hand-edit.

## D5: Gemini pricing fetched from Google's docs

Google's API doesn't report which models are on the free tier. We fetch their markdown pricing page and parse it. The cache is stored locally with a 7-day TTL. This is fragile but better than maintaining a hardcoded list.

## D6: Session JSONL tailing for usage tracking

Claude Code writes session data to `~/.claude/projects/<encoded-cwd>/*.jsonl`. We tail these files in a background thread to capture token usage without modifying Claude Code's behavior.

## D7: Thin entrypoint shim

`bin/llm-switchboard` is a minimal script that adds the project root to `sys.path` and calls `main()`. This preserves backwards compatibility with the original single-file deployment — users can symlink or copy the `bin/` script to their PATH.

## D8: Modularization boundaries

Modules are split by concern, not by layer:
- `tui.py` owns all terminal I/O (raw mode, rendering, spinner)
- `cli.py` owns argument parsing and orchestration
- `config.py`, `cache.py`, `free_tier.py`, `session.py` own their data domains
- `util.py` is pure functions only

The goal is that `tui.py` can be imported only when interactive mode is needed, keeping non-interactive paths lightweight.

## D9: Test functions accept optional Path parameters

Config and session functions accept optional file path parameters instead of always reading module-level constants. This enables testing with temp directories without monkeypatching globals.

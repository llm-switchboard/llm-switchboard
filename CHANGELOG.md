# Changelog

## v0.3.0 — Stabilization (2026-03-02)

### Features
- **`--doctor` command** — end-to-end health check with connectivity tests, chat probe, compat summary, and actionable recommendations
- **`--clear-detect-cache`** — remove endpoint auto-detect cache and exit cleanly
- **Improved compat report clarity** — legend explaining AGENT badges, required tests listed, per-model failure reasons shown; JSON output includes `required_tests`, `scoring_rules`, and per-model `failed_tests`/`required_failed` fields

### BREAKING CHANGES
- `--compat-report --json` output is now an envelope object with keys: `required_tests`, `scoring_rules`, `models`. The previous bare `{model_id: ...}` map is now under `.models`.

### Technical
- `--doctor` exit codes: 0 (all OK), 1 (models endpoint fail), 2 (chat probe fail), 3 (unexpected error)
- `--doctor` never crashes with a stack trace; catches exceptions and prints helpful error lines
- No BrokenPipeError tracebacks when piping output

## v0.1.0 — First Public Release (2026-03-02)

### Features
- **Interactive TUI** with arrow/vim navigation, favorites, provider grouping
- **Free-tier tracking** for Groq, Cerebras, Gemini, Mistral (configurable rules)
- **Session tracking** with token usage and cost estimation
- **fzf integration** for fuzzy search when available
- **Gemini pricing cache** auto-fetched from Google's pricing page
- Non-interactive CLI: `--list-models`, `--json`, `--select`, `--favorites`, `--free-only`, `--provider`
- `--self-test` flag (91 unit tests)
- `--print-paths` to show config/cache directory paths
- `--setup` interactive wizard for free-tier provider configuration
- `--stats` usage statistics across all models

### Technical
- Python 3.11+ stdlib only — no external dependencies
- Modular package (`llm_switchboard/`) split into 9 modules
- Config at `~/.config/llm-switchboard/`, cache at `~/.cache/llm-switchboard/`
- Respects `XDG_CONFIG_HOME` and `XDG_CACHE_HOME`
- MIT licensed

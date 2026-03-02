# Changelog

## v0.3.0 — Stabilization (2026-03-02)

### Features
- **`--doctor` command** — end-to-end health check with connectivity tests, chat probe, compat summary, and actionable recommendations
- **`--clear-detect-cache`** — remove endpoint auto-detect cache and exit cleanly
- **Improved compat report clarity** — legend explaining AGENT badges, required tests listed, per-model failure reasons shown; JSON output includes `required_tests`, `scoring_rules`, and per-model `failed_tests`/`required_failed` fields

### BREAKING CHANGES
- `--compat-report --json` output is now an envelope object with keys: `required_tests`, `scoring_rules`, `models`. The previous bare `{model_id: ...}` map is now under `.models`.
- Removed `--self-test`, `--print-paths`, and `--print-endpoints` flags (use `--doctor`, `make ci`, and config file paths in `--doctor` output instead)

### Hardening
- Missing-value validation for `--api-mode`, `--base-url`, `--auto-prefer`, `--provider` flags
- `JSONDecodeError` handling in API client (HTML responses no longer crash)
- `PermissionError` handling in config directory creation
- Gemini pricing parser raises on empty results instead of returning silently
- File locking (`fcntl.flock`) around read-modify-write cycles in usage and compat persistence
- Thread-safe counter increments in `SessionWatcher` via `threading.Lock`
- Stale compat data refreshed after returning from claude session
- `PRIVATE_IP_RE` extended to cover IPv6 localhost (`[::1]`) and `0.0.0.0`
- Cached JSON responses validated as `dict` (non-dict cache no longer crashes)
- Gemini free-tier lookup no longer mutates the cache dict with non-serializable objects
- Global CLI state reset at start of `main()` for clean test isolation
- Free-tier `add` command now produces proper `FreeTierRule` objects instead of raw strings

### Technical
- `--doctor` exit codes: 0 (all OK), 1 (models endpoint fail), 2 (chat probe fail), 3 (unexpected error)
- `--doctor` never crashes with a stack trace; catches exceptions and prints helpful error lines
- No BrokenPipeError tracebacks when piping output
- Added `pyproject.toml` (PEP 517/518 packaging, tool config)
- Added `py.typed` (PEP 561 type annotation marker)
- Added GitHub Actions CI (Python 3.11/3.12/3.13 matrix, pytest, ruff, mypy)
- Added CodeQL and Trivy security scanning workflows
- Added Dependabot for GitHub Actions ecosystem
- Switched test runner from unittest to pytest (304 tests)
- All ruff and mypy issues resolved

## v0.1.0 — First Public Release (2026-03-02)

### Features
- **Interactive TUI** with arrow/vim navigation, favorites, provider grouping
- **Free-tier tracking** for Groq, Cerebras, Gemini, Mistral (configurable rules)
- **Session tracking** with token usage and cost estimation
- **fzf integration** for fuzzy search when available
- **Gemini pricing cache** auto-fetched from Google's pricing page
- Non-interactive CLI: `--list-models`, `--json`, `--select`, `--favorites`, `--free-only`, `--provider`
- `--setup` interactive wizard for free-tier provider configuration
- `--stats` usage statistics across all models

### Technical
- Python 3.11+ stdlib only — no external dependencies
- Modular package (`llm_switchboard/`) split into 11 modules
- Config at `~/.config/llm-switchboard/`, cache at `~/.cache/llm-switchboard/`
- Respects `XDG_CONFIG_HOME` and `XDG_CACHE_HOME`
- MIT licensed

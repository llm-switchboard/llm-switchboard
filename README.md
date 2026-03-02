# llm-switchboard

Interactive model picker for [Claude Code](https://github.com/anthropics/claude-code) via [Open WebUI](https://github.com/open-webui/open-webui).

Browse all models available through your Open WebUI instance, pick one, and launch a Claude Code session — all from a single TUI.

## Quick Start

```bash
# Set your Open WebUI API key
export OPENWEBUI_API_KEY="sk-..."

# Optional: custom URL (default: http://127.0.0.1:3100)
export OPENWEBUI_URL="http://your-server:3100"

# Launch the interactive picker
llm-switchboard

# Or use non-interactive mode
llm-switchboard --list-models
llm-switchboard --select groq/llama-3.3-70b
```

## Requirements

- Python 3.11+ (stdlib only, no external dependencies)
- A running Open WebUI instance
- `claude` CLI installed and in PATH

## Features

- **Interactive TUI** with arrow/vim navigation, favorites, provider grouping
- **Free-tier tracking** for Groq, Cerebras, Gemini, Mistral (configurable)
- **Session tracking** with token usage and cost estimation
- **fzf integration** for fuzzy search when available
- **Non-interactive mode** for scripting (`--list-models`, `--select`, `--json`)
- **Gemini pricing cache** auto-fetched from Google's pricing page

## Usage

### Interactive mode (default)
```
llm-switchboard                     # Full TUI picker
llm-switchboard -f                  # Show only free models
llm-switchboard -l                  # Re-launch last used model
llm-switchboard llama               # Jump to search for "llama"
llm-switchboard groq/model:id       # Direct launch by model ID
```

### Non-interactive mode
```
llm-switchboard --list-models                    # Text list of all models
llm-switchboard --list-models --json             # JSON output
llm-switchboard --list-models --free-only        # Only free/local models
llm-switchboard --list-models --provider groq    # Filter by provider
llm-switchboard --select <model_id>              # Launch without TUI
llm-switchboard --favorites                      # List favorite model IDs
```

### API Mode

llm-switchboard supports multiple API surfaces. By default it auto-detects, but you can force a mode:

```
llm-switchboard --api-mode auto                 # Auto-detect (default)
llm-switchboard --api-mode openai               # Force OpenAI-compatible endpoints
llm-switchboard --api-mode anthropic            # Force Anthropic Messages endpoint
llm-switchboard --base-url http://other:8080    # Override base URL
```

| Mode | Chat Endpoint | When to Use |
|------|--------------|-------------|
| `openai` | `/v1/chat/completions` or `/api/chat/completions` | General OpenAI-compatible clients, Open WebUI default |
| `anthropic` | `/api/v1/messages` | Claude Code / Anthropic Messages API surface |
| `auto` | Probes endpoints and picks the best match | Default — works for most setups |

**Auto-detection** probes in order: `/v1/models`, `/api/models`, `/api/v1/messages`. Results are cached for 1 hour.

The `--api-mode` and `--base-url` flags work with any command (e.g., `--compat-test`, `--list-models`).

To persist the mode, write it to `~/.config/llm-switchboard/api_mode.conf`:
```bash
echo "openai" > ~/.config/llm-switchboard/api_mode.conf
```

### Management
```
llm-switchboard --fav add <model-id>       # Add favorite
llm-switchboard --fav rm <model-id>        # Remove favorite
llm-switchboard --fav list                 # List favorites
llm-switchboard --free-tier add groq       # Mark provider as free
llm-switchboard --free-tier list           # Show free-tier config
llm-switchboard --free-tier update         # Refresh Gemini pricing cache
llm-switchboard --setup                    # Interactive setup wizard
llm-switchboard --stats                    # Usage statistics
llm-switchboard --reset                    # Clear usage data
```

### Agent Compatibility Testing
```
llm-switchboard --compat-test <model-id>   # Test a model as coding agent
llm-switchboard --compat-test --all        # Test all models
llm-switchboard --compat-report            # Show test results
llm-switchboard --compat-report --json     # JSON output
```

### Diagnostics
```
llm-switchboard --doctor                   # Health check & connectivity test
llm-switchboard --clear-detect-cache       # Clear endpoint detection cache
llm-switchboard --version                  # Show version
```

### Development
```
make ci                            # Run everything (test + lint + typecheck)
make test                          # pytest
make check                         # Compile check
make lint                          # Ruff linter
make typecheck                     # mypy
make fmt                           # Auto-format with ruff
```

## Config & Cache Paths

| File | Purpose |
|------|---------|
| `~/.config/llm-switchboard/favorites.conf` | Favorite model IDs |
| `~/.config/llm-switchboard/free_providers.conf` | Free-tier provider rules |
| `~/.config/llm-switchboard/last_model` | Last used model |
| `~/.config/llm-switchboard/usage.json` | Session usage data |
| `~/.config/llm-switchboard/gemini_free_cache.json` | Gemini pricing cache |
| `~/.config/llm-switchboard/api_mode.conf` | API mode (auto/openai/anthropic) |
| `~/.cache/llm-switchboard/connections.json` | Cached provider config (1hr TTL) |
| `~/.cache/llm-switchboard/endpoint_detect.json` | Cached endpoint detection (1hr TTL) |

Paths respect `XDG_CONFIG_HOME` and `XDG_CACHE_HOME` if set.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENWEBUI_API_KEY` | *(required)* | Open WebUI API key |
| `OPENWEBUI_URL` | `http://127.0.0.1:3100` | Open WebUI base URL |
| `NO_COLOR` | *(unset)* | Disable color output if set |

## Project Structure

```
llm-switchboard/
├── bin/llm-switchboard    # Executable entrypoint
├── llm_switchboard/       # Python package
│   ├── __init__.py        # Version
│   ├── cli.py             # CLI parsing, main(), non-interactive commands
│   ├── tui.py             # Terminal UI (raw input, rendering, interactive loop)
│   ├── webui.py           # API client (OpenAI + Anthropic)
│   ├── endpoint.py        # Endpoint resolution and auto-detection
│   ├── compat.py          # Agent compatibility testing
│   ├── models.py          # Model data structures, provider detection
│   ├── config.py          # Favorites, last model, directory management
│   ├── cache.py           # File-based API response caching
│   ├── free_tier.py       # Free-tier rules, Gemini pricing
│   ├── session.py         # Session watching, usage tracking
│   ├── util.py            # Utilities (formatting, colors, file locking)
│   └── py.typed           # PEP 561 type annotation marker
├── tests/                 # Unit tests (pytest)
├── docs/
│   ├── ARCHITECTURE.md
│   └── DECISIONS.md
├── pyproject.toml         # PEP 517/518 packaging + tool config
├── CHANGELOG.md
├── Makefile
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).

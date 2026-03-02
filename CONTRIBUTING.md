# Contributing to llm-switchboard

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/llm-switchboard/llm-switchboard.git
cd llm-switchboard
pip install -e .
pip install ruff mypy pytest
```

No runtime dependencies to install — the project uses Python 3.11+ stdlib only.

## Running Tests

```bash
make ci                            # run everything (test + lint + typecheck)
make test                          # pytest
make check                         # compile check (syntax errors)
make lint                          # ruff linter
make typecheck                     # mypy
make fmt                           # auto-format with ruff
```

## Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module layout and design decisions.

## Guidelines

- **No external dependencies.** Everything must work with Python stdlib.
- **Write tests.** New features should include unit tests in `tests/`.
- **Keep it simple.** This is a focused CLI tool, not a framework.
- **Run `make ci` before submitting.** All checks must pass.

## Submitting Changes

1. Fork the repo and create a feature branch
2. Make your changes
3. Run `make ci`
4. Open a pull request with a clear description of what changed and why

## Reporting Issues

Open an issue on GitHub. Include:
- What you expected vs. what happened
- Python version (`python3 --version`)
- OS and terminal
- Relevant error output

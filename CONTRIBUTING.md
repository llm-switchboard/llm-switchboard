# Contributing to llm-switchboard

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/switchboard-ai/llm-switchboard.git
cd llm-switchboard
```

No dependencies to install — the project uses Python 3.11+ stdlib only.

## Running Tests

```bash
make test                          # unittest discovery
make check                         # compile check (syntax errors)
python3 bin/llm-switchboard --self-test  # run via CLI
```

## Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module layout and design decisions.

## Guidelines

- **No external dependencies.** Everything must work with Python stdlib.
- **Write tests.** New features should include unit tests in `tests/`.
- **Keep it simple.** This is a focused CLI tool, not a framework.
- **Run `make test` before submitting.** All tests must pass.

## Submitting Changes

1. Fork the repo and create a feature branch
2. Make your changes
3. Run `make test` and `make check`
4. Open a pull request with a clear description of what changed and why

## Reporting Issues

Open an issue on GitHub. Include:
- What you expected vs. what happened
- Python version (`python3 --version`)
- OS and terminal
- Relevant error output

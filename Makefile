.PHONY: check test lint typecheck fmt ci

# Compile check — verify all Python files parse correctly
check:
	python3 -m compileall -q llm_switchboard/ tests/ bin/llm-switchboard

# Run the test suite via pytest
test:
	pytest -v

# Lint with ruff
lint:
	ruff check .

# Type check with mypy
typecheck:
	mypy llm_switchboard/

# Auto-format with ruff
fmt:
	ruff check --fix .
	ruff format .

# Run everything
ci: check test lint typecheck

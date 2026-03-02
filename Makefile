.PHONY: check test self-test

# Compile check — verify all Python files parse correctly
check:
	python3 -m compileall -q llm_switchboard/ tests/ bin/llm-switchboard

# Run the test suite via unittest discovery
test:
	python3 -m unittest discover -s tests -v

# Run tests via the CLI's --self-test flag
self-test:
	python3 bin/llm-switchboard --self-test

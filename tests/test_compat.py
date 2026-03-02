"""Tests for llm_switchboard.compat — persistence, validators, and scoring."""

import json
import tempfile
import unittest
from pathlib import Path

from llm_switchboard.compat import (
    MIN_PASS_TOTAL,
    REQUIRED_TESTS,
    TESTS,
    TOOL_SCHEMAS,
    _parse_tool_calls,
    _validate_constraint,
    _validate_format,
    _validate_no_hallucination,
    _validate_tool_call_chaining,
    _validate_tool_call_error_recovery,
    _validate_tool_call_schema,
    compute_agent_status,
    get_agent_ok_models,
    get_compat_status,
    load_compat,
    save_compat,
)


class TestPersistence(unittest.TestCase):
    """Test load/save/missing/corrupted/atomic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.compat_file = Path(self.tmpdir) / "compat.json"

    def test_load_missing(self):
        data = load_compat(self.compat_file)
        self.assertEqual(data, {"results": {}})

    def test_save_and_load(self):
        data = {
            "results": {
                "test-model": {
                    "last_run": "2025-01-01T00:00:00Z",
                    "tests": {},
                    "pass_count": 6,
                    "fail_count": 0,
                    "last_status": "pass",
                    "latency_ms": 1500,
                }
            }
        }
        save_compat(data, self.compat_file)
        loaded = load_compat(self.compat_file)
        self.assertEqual(loaded, data)

    def test_corrupted_file(self):
        self.compat_file.write_text("not json at all {{{")
        data = load_compat(self.compat_file)
        self.assertEqual(data, {"results": {}})

    def test_invalid_structure(self):
        self.compat_file.write_text(json.dumps({"foo": "bar"}))
        data = load_compat(self.compat_file)
        self.assertEqual(data, {"results": {}})

    def test_atomic_write_no_tmp_remains(self):
        save_compat({"results": {}}, self.compat_file)
        tmp = self.compat_file.with_suffix(".json.tmp")
        self.assertFalse(tmp.exists())
        self.assertTrue(self.compat_file.exists())

    def test_get_compat_status_not_tested(self):
        self.assertIsNone(get_compat_status("unknown-model", self.compat_file))

    def test_get_compat_status_pass(self):
        data = {"results": {"m1": {"last_status": "pass"}}}
        save_compat(data, self.compat_file)
        self.assertEqual(get_compat_status("m1", self.compat_file), "pass")

    def test_get_compat_status_partial(self):
        data = {"results": {"m1": {"last_status": "partial"}}}
        save_compat(data, self.compat_file)
        self.assertEqual(get_compat_status("m1", self.compat_file), "partial")

    def test_get_compat_status_fail(self):
        data = {"results": {"m1": {"last_status": "fail"}}}
        save_compat(data, self.compat_file)
        self.assertEqual(get_compat_status("m1", self.compat_file), "fail")

    def test_get_agent_ok_models(self):
        data = {
            "results": {
                "m1": {"last_status": "pass"},
                "m2": {"last_status": "partial"},
                "m3": {"last_status": "pass"},
                "m4": {"last_status": "fail"},
            }
        }
        save_compat(data, self.compat_file)
        ok = get_agent_ok_models(self.compat_file)
        self.assertEqual(ok, {"m1", "m3"})

    def test_get_agent_ok_models_empty(self):
        ok = get_agent_ok_models(self.compat_file)
        self.assertEqual(ok, set())

    def test_save_creates_parent_dirs(self):
        nested = Path(self.tmpdir) / "a" / "b" / "compat.json"
        save_compat({"results": {}}, nested)
        self.assertTrue(nested.exists())


# ─── Text Validators ────────────────────────────────────────────────


class TestFormatValidator(unittest.TestCase):
    """Test _validate_format with known inputs."""

    def test_valid_with_markers(self):
        text = """BEGIN_DIFF
--- a/hello.py
+++ b/hello.py
@@ -1 +1 @@
-print('hello')
+print('goodbye')
END_DIFF"""
        self.assertTrue(_validate_format(text))

    def test_valid_bare_diff(self):
        text = """--- a/hello.py
+++ b/hello.py
@@ -1 +1 @@
-print('hello')
+print('goodbye')"""
        self.assertTrue(_validate_format(text))

    def test_invalid_no_diff(self):
        text = "Here is the updated file:\nprint('goodbye')"
        self.assertFalse(_validate_format(text))

    def test_markers_without_diff_content(self):
        text = "BEGIN_DIFF\nno actual diff lines\nEND_DIFF"
        self.assertFalse(_validate_format(text))

    def test_markers_with_plus_minus(self):
        text = "BEGIN_DIFF\n-old\n+new\nEND_DIFF"
        self.assertTrue(_validate_format(text))


class TestConstraintValidator(unittest.TestCase):
    """Test _validate_constraint with known inputs."""

    def test_valid_only_auth(self):
        text = """Here is the updated auth.py:
```python
def check_access(user):
    return True
```"""
        self.assertTrue(_validate_constraint(text))

    def test_missing_auth(self):
        text = "def check_access(user):\n    return True"
        self.assertFalse(_validate_constraint(text))

    def test_missing_return_true(self):
        text = "auth.py\ndef check_access(user):\n    return False"
        self.assertFalse(_validate_constraint(text))

    def test_both_files_but_utils_unchanged(self):
        text = """auth.py:
def check_access(user):
    return True

utils.py (unchanged):
def format_name(name):
    return name.strip().title()"""
        self.assertTrue(_validate_constraint(text))

    def test_both_files_utils_altered(self):
        text = """auth.py:
def check_access(user):
    return True

utils.py:
def format_name(name):
    return name.upper()"""
        self.assertFalse(_validate_constraint(text))


class TestNoHallucinationValidator(unittest.TestCase):
    """Test _validate_no_hallucination with known inputs."""

    def test_clear_refusal(self):
        text = "I can't find the file `src/database/migrations/v42_schema.py`. It doesn't exist in the project."
        self.assertTrue(_validate_no_hallucination(text))

    def test_uncertainty(self):
        text = "I'm not sure this file exists. Could you verify the path?"
        self.assertTrue(_validate_no_hallucination(text))

    def test_fabricated_sql(self):
        text = """Here's the fix for line 87:

```python
def upgrade():
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT
    );
```"""
        self.assertFalse(_validate_no_hallucination(text))

    def test_fabricated_migration(self):
        text = """```python
def upgrade():
    op.alter_column('users', 'email', nullable=False)

def downgrade():
    op.alter_column('users', 'email', nullable=True)
```"""
        self.assertFalse(_validate_no_hallucination(text))

    def test_short_ambiguous(self):
        text = "I'd need to see the file contents first."
        self.assertTrue(_validate_no_hallucination(text))

    def test_long_fabrication_without_sql(self):
        text = "Here is the fix:\n" + "x" * 600
        self.assertFalse(_validate_no_hallucination(text))


# ─── Tool-Call Validators ───────────────────────────────────────────


def _make_message(tool_calls=None, content=None):
    """Helper to build an OpenAI-format assistant message."""
    msg = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _make_tool_call(name, arguments, call_id="call_1"):
    """Helper to build a single tool_call entry."""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class TestParseToolCalls(unittest.TestCase):
    """Test _parse_tool_calls helper."""

    def test_valid_string_args(self):
        msg = _make_message(
            tool_calls=[
                _make_tool_call("list_files", {"path": "/src"}),
            ]
        )
        calls = _parse_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "list_files")
        self.assertEqual(calls[0]["arguments"], {"path": "/src"})

    def test_valid_dict_args(self):
        """Some APIs return arguments as dict, not JSON string."""
        msg = _make_message(
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "/a.py"}},
                }
            ]
        )
        calls = _parse_tool_calls(msg)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")

    def test_invalid_json_args(self):
        msg = _make_message(
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "not json {{{"},
                }
            ]
        )
        calls = _parse_tool_calls(msg)
        self.assertEqual(len(calls), 0)

    def test_no_tool_calls(self):
        msg = _make_message(content="I'll list the files...")
        calls = _parse_tool_calls(msg)
        self.assertEqual(calls, [])

    def test_empty_tool_calls(self):
        msg = _make_message(tool_calls=[])
        calls = _parse_tool_calls(msg)
        self.assertEqual(calls, [])

    def test_multiple_calls(self):
        msg = _make_message(
            tool_calls=[
                _make_tool_call("read_file", {"path": "/a.py"}, "c1"),
                _make_tool_call("write_file", {"path": "/b.py", "content": "x"}, "c2"),
            ]
        )
        calls = _parse_tool_calls(msg)
        self.assertEqual(len(calls), 2)

    def test_non_dict_entry_skipped(self):
        msg = _make_message(tool_calls=["not a dict", None, 42])
        calls = _parse_tool_calls(msg)
        self.assertEqual(calls, [])


class TestToolCallSchemaValidator(unittest.TestCase):
    """Test _validate_tool_call_schema."""

    def test_valid_list_files(self):
        msg = _make_message(
            tool_calls=[
                _make_tool_call("list_files", {"path": "/src"}),
            ]
        )
        self.assertTrue(_validate_tool_call_schema(msg))

    def test_valid_other_tool(self):
        """Accept any valid tool call to one of our tools."""
        msg = _make_message(
            tool_calls=[
                _make_tool_call("read_file", {"path": "/src/main.py"}),
            ]
        )
        self.assertTrue(_validate_tool_call_schema(msg))

    def test_no_tool_calls(self):
        msg = _make_message(content="Here are the files in /src: ...")
        self.assertFalse(_validate_tool_call_schema(msg))

    def test_unknown_tool_name(self):
        msg = _make_message(
            tool_calls=[
                _make_tool_call("delete_file", {"path": "/src"}),
            ]
        )
        self.assertFalse(_validate_tool_call_schema(msg))

    def test_missing_path_arg(self):
        """list_files without path should still pass via fallback."""
        msg = _make_message(
            tool_calls=[
                _make_tool_call("list_files", {"directory": "/src"}),
            ]
        )
        # No path arg on list_files, but fallback accepts any valid tool name
        self.assertTrue(_validate_tool_call_schema(msg))

    def test_invalid_arguments_json(self):
        msg = _make_message(
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "broken{json"},
                }
            ]
        )
        self.assertFalse(_validate_tool_call_schema(msg))


class TestToolCallChainingValidator(unittest.TestCase):
    """Test _validate_tool_call_chaining."""

    def test_read_config(self):
        msg = _make_message(
            tool_calls=[
                _make_tool_call("read_file", {"path": "/src/config.py"}, "c1"),
            ]
        )
        self.assertTrue(_validate_tool_call_chaining(msg))

    def test_read_and_write(self):
        msg = _make_message(
            tool_calls=[
                _make_tool_call("read_file", {"path": "/src/config.py"}, "c1"),
                _make_tool_call("write_file", {"path": "/src/config.py", "content": "new"}, "c2"),
            ]
        )
        self.assertTrue(_validate_tool_call_chaining(msg))

    def test_only_write_no_read(self):
        """Write without read — accepted via fallback (valid tool)."""
        msg = _make_message(
            tool_calls=[
                _make_tool_call("write_file", {"path": "/src/config.py", "content": "x"}, "c1"),
            ]
        )
        self.assertTrue(_validate_tool_call_chaining(msg))

    def test_no_tool_calls(self):
        msg = _make_message(content="I'll read the file and update it...")
        self.assertFalse(_validate_tool_call_chaining(msg))

    def test_read_wrong_file(self):
        """read_file for wrong file — accepted via fallback (valid tool)."""
        msg = _make_message(
            tool_calls=[
                _make_tool_call("read_file", {"path": "/other.py"}, "c1"),
            ]
        )
        self.assertTrue(_validate_tool_call_chaining(msg))


class TestToolCallErrorRecoveryValidator(unittest.TestCase):
    """Test _validate_tool_call_error_recovery."""

    def test_acknowledges_error(self):
        text = "The file /tmp/settings.json was not found. Would you like me to check a different path?"
        self.assertTrue(_validate_tool_call_error_recovery(text))

    def test_reports_file_not_found(self):
        text = "I encountered a FileNotFoundError when trying to read /tmp/settings.json."
        self.assertTrue(_validate_tool_call_error_recovery(text))

    def test_suggests_alternative(self):
        text = "Sorry, that file doesn't exist. Let me try /etc/settings.json instead."
        self.assertTrue(_validate_tool_call_error_recovery(text))

    def test_fabricates_json_content(self):
        text = '{"database": {"host": "localhost", "port": 5432}}'
        self.assertFalse(_validate_tool_call_error_recovery(text))

    def test_fabricates_host_value(self):
        text = 'The database host is "db_host": "localhost" based on the settings file.'
        self.assertFalse(_validate_tool_call_error_recovery(text))

    def test_empty_response(self):
        self.assertFalse(_validate_tool_call_error_recovery(""))

    def test_whitespace_only(self):
        self.assertFalse(_validate_tool_call_error_recovery("   \n  "))

    def test_short_generic(self):
        text = "Let me try a different approach."
        self.assertTrue(_validate_tool_call_error_recovery(text))


# ─── Scoring ─────────────────────────────────────────────────────────


class TestComputeAgentStatus(unittest.TestCase):
    """Test compute_agent_status scoring rules."""

    def _make_results(self, **overrides):
        """Build a results dict. All pass by default, override with False."""
        base = {name: {"passed": True, "latency_ms": 100} for name in TESTS}
        for name, passed in overrides.items():
            base[name] = {"passed": passed, "latency_ms": 100}
        return base

    def test_all_pass(self):
        results = self._make_results()
        self.assertEqual(compute_agent_status(results), "pass")

    def test_tool_schema_fails_is_fail(self):
        results = self._make_results(tool_call_schema=False)
        self.assertEqual(compute_agent_status(results), "fail")

    def test_no_hallucination_fails_is_partial(self):
        """tool_call_schema passes but no_hallucination fails."""
        results = self._make_results(no_hallucination=False)
        # 5/6 pass, required not all met → partial
        self.assertEqual(compute_agent_status(results), "partial")

    def test_both_required_pass_but_low_total(self):
        """Required pass but only 3/6 total → partial."""
        results = self._make_results(
            format_compliance=False,
            constraint_following=False,
            tool_call_chaining=False,
        )
        # 3/6 pass — below MIN_PASS_TOTAL
        self.assertEqual(compute_agent_status(results), "partial")

    def test_required_pass_and_four_total(self):
        """Required pass and exactly 4/6 total → pass."""
        results = self._make_results(
            format_compliance=False,
            constraint_following=False,
        )
        # 4/6 pass, required met
        self.assertEqual(compute_agent_status(results), "pass")

    def test_empty_results(self):
        self.assertEqual(compute_agent_status({}), "fail")

    def test_all_fail(self):
        results = {name: {"passed": False} for name in TESTS}
        self.assertEqual(compute_agent_status(results), "fail")

    def test_missing_required_test(self):
        """Results dict missing tool_call_schema entirely."""
        results = {name: {"passed": True} for name in TESTS if name != "tool_call_schema"}
        self.assertEqual(compute_agent_status(results), "fail")


# ─── Test Registry ──────────────────────────────────────────────────


class TestTestDefinitions(unittest.TestCase):
    """Verify test definitions are well-formed."""

    def test_all_tests_have_required_keys(self):
        for name, defn in TESTS.items():
            self.assertTrue(
                "prompt" in defn or "messages" in defn,
                f"{name} missing prompt/messages",
            )
            self.assertIn("validator", defn, f"{name} missing validator")
            self.assertIn("description", defn, f"{name} missing description")
            self.assertIn("response_type", defn, f"{name} missing response_type")
            self.assertTrue(callable(defn["validator"]), f"{name} validator not callable")

    def test_six_tests_defined(self):
        self.assertEqual(len(TESTS), 6)

    def test_required_tests_exist(self):
        for t in REQUIRED_TESTS:
            self.assertIn(t, TESTS, f"required test {t} not in TESTS")

    def test_tool_schemas_valid(self):
        """All tool schemas have correct structure."""
        for schema in TOOL_SCHEMAS:
            self.assertEqual(schema["type"], "function")
            fn = schema["function"]
            self.assertIn("name", fn)
            self.assertIn("parameters", fn)
            self.assertEqual(fn["parameters"]["type"], "object")

    def test_min_pass_total_reasonable(self):
        self.assertGreaterEqual(MIN_PASS_TOTAL, 1)
        self.assertLessEqual(MIN_PASS_TOTAL, len(TESTS))

    def test_tool_tests_have_uses_tools(self):
        """Tool-call tests must have uses_tools=True."""
        tool_tests = {"tool_call_schema", "tool_call_chaining", "tool_call_error_recovery"}
        for name in tool_tests:
            self.assertTrue(TESTS[name].get("uses_tools"), f"{name} missing uses_tools")

    def test_text_tests_no_uses_tools(self):
        text_tests = {"format_compliance", "constraint_following", "no_hallucination"}
        for name in text_tests:
            self.assertFalse(TESTS[name].get("uses_tools"), f"{name} should not have uses_tools")


if __name__ == "__main__":
    unittest.main()

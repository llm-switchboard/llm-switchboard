"""Tests for CLI behavior — broken pipe, arg parsing, doctor, clear-detect-cache."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Path to the entrypoint script
BIN = str(Path(__file__).resolve().parent.parent / "bin" / "llm-switchboard")


class TestBrokenPipe(unittest.TestCase):
    """Ensure piping to a consumer that closes early doesn't produce a traceback."""

    def _run_piped(self, args: list[str], max_lines: int = 2) -> tuple[int, str, str]:
        proc = subprocess.Popen(
            [sys.executable, BIN] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "OPENWEBUI_API_KEY": "test-key"},
        )
        lines = []
        for _ in range(max_lines):
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
        proc.stdout.close()
        proc.wait(timeout=10)
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        stdout_text = b"".join(lines).decode(errors="replace")
        return proc.returncode, stdout_text, stderr

    def test_help_broken_pipe(self):
        rc, stdout, stderr = self._run_piped(["--help"], max_lines=2)
        self.assertEqual(rc, 0)
        self.assertNotIn("BrokenPipeError", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertTrue(len(stdout) > 0)

    def test_version_broken_pipe(self):
        rc, stdout, stderr = self._run_piped(["--version"], max_lines=1)
        self.assertEqual(rc, 0)
        self.assertNotIn("BrokenPipeError", stderr)
        self.assertNotIn("Traceback", stderr)

    @unittest.skipUnless(os.name == "posix", "SIGPIPE only on POSIX")
    def test_help_pipe_to_head(self):
        result = subprocess.run(
            f"{sys.executable} {BIN} --help | head -1",
            shell=True, capture_output=True, text=True,
            env={**os.environ, "OPENWEBUI_API_KEY": "test-key"},
            timeout=10,
        )
        self.assertNotIn("BrokenPipeError", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertTrue(len(result.stdout) > 0)


# ─── --clear-detect-cache ────────────────────────────────────────────

class TestClearDetectCache(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_file = Path(self.tmpdir) / "endpoint_detect.json"

    def test_clear_existing(self):
        self.cache_file.write_text('{"mode":"openai"}')
        from llm_switchboard.cli import cmd_clear_detect_cache
        with patch("llm_switchboard.cli.DETECT_CACHE", self.cache_file):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_clear_detect_cache()
        self.assertFalse(self.cache_file.exists())
        self.assertIn("Detect cache cleared:", buf.getvalue())

    def test_clear_missing(self):
        from llm_switchboard.cli import cmd_clear_detect_cache
        with patch("llm_switchboard.cli.DETECT_CACHE", self.cache_file):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cmd_clear_detect_cache()
        self.assertIn("already clear", buf.getvalue())

    def test_parse_args(self):
        from llm_switchboard.cli import _parse_args
        cmd, filt, extra, opts = _parse_args(["--clear-detect-cache"])
        self.assertEqual(cmd, "clear-detect-cache")


# ─── --doctor ────────────────────────────────────────────────────────

def _fake_endpoint(mode="openai"):
    from llm_switchboard.endpoint import Endpoint
    return Endpoint("http://localhost:3100", mode,
                    "/api/chat/completions" if mode == "openai" else "/api/v1/messages",
                    "/api/models", "auto-detect", "test")


class TestDoctor(unittest.TestCase):

    def _run_doctor(self, mode="openai", models_resp=None, chat_ok=True,
                    compat_results=None, raise_exc=None):
        """Run cmd_doctor with mocked probes. Returns (exit_code, stdout)."""
        from llm_switchboard.cli import cmd_doctor
        ep = _fake_endpoint(mode)

        patches = {
            "llm_switchboard.cli._resolve_api": MagicMock(return_value=ep),
            "llm_switchboard.cli.OPENWEBUI_KEY": "test-key",
            "llm_switchboard.cli._probe_get": MagicMock(return_value=models_resp),
            "llm_switchboard.cli._looks_like_models_list": MagicMock(
                return_value=(models_resp is not None and "data" in (models_resp or {}))),
            "llm_switchboard.cli._fetch_any_model_id": MagicMock(
                return_value=("test-model", "/api/models") if models_resp else (None, "/api/models")),
            "llm_switchboard.cli._probe_openai_chat": MagicMock(
                return_value="/api/chat/completions" if chat_ok else None),
            "llm_switchboard.cli._probe_anthropic_chat": MagicMock(return_value=chat_ok),
            "llm_switchboard.cli.load_compat": MagicMock(
                return_value={"results": compat_results or {}}),
        }

        # Mock COMPAT_FILE and DETECT_CACHE to non-existent paths
        tmpdir = tempfile.mkdtemp()
        patches["llm_switchboard.cli.DETECT_CACHE"] = Path(tmpdir) / "no_cache.json"
        # Only set COMPAT_FILE to non-existent if compat_results were provided
        # (so the function reads from our mock load_compat, not the real file)
        if compat_results:
            # Make COMPAT_FILE "exist" by creating a dummy file
            cf = Path(tmpdir) / "compat.json"
            cf.write_text("{}")
            patches["llm_switchboard.cli.COMPAT_FILE"] = cf
        else:
            patches["llm_switchboard.cli.COMPAT_FILE"] = Path(tmpdir) / "no_compat.json"

        if raise_exc:
            patches["llm_switchboard.cli._resolve_api"] = MagicMock(side_effect=raise_exc)

        buf = io.StringIO()
        err_buf = io.StringIO()
        with patch.multiple("llm_switchboard.cli", **{k.split(".")[-1]: v
                            for k, v in patches.items() if k.startswith("llm_switchboard.cli.")}):
            with patch("sys.stdout", buf), patch("sys.stderr", err_buf):
                rc = cmd_doctor()
        return rc, buf.getvalue(), err_buf.getvalue()

    def test_success_openai(self):
        rc, out, _ = self._run_doctor(
            mode="openai",
            models_resp={"data": [{"id": "m1"}]},
            chat_ok=True,
        )
        self.assertEqual(rc, 0)
        self.assertIn("All checks passed", out)
        self.assertIn("openai", out)

    def test_success_anthropic(self):
        rc, out, _ = self._run_doctor(
            mode="anthropic",
            models_resp={"data": [{"id": "m1"}]},
            chat_ok=True,
        )
        self.assertEqual(rc, 0)
        self.assertIn("All checks passed", out)

    def test_models_failure(self):
        rc, out, _ = self._run_doctor(
            models_resp=None,
            chat_ok=True,
        )
        self.assertEqual(rc, 1)
        self.assertIn("FAIL", out)

    def test_chat_failure(self):
        rc, out, _ = self._run_doctor(
            models_resp={"data": [{"id": "m1"}]},
            chat_ok=False,
        )
        self.assertEqual(rc, 2)
        self.assertIn("FAIL", out)
        self.assertIn("Recommendations", out)

    def test_unexpected_exception(self):
        rc, out, err = self._run_doctor(raise_exc=RuntimeError("boom"))
        self.assertEqual(rc, 3)
        self.assertIn("Doctor error", err)
        self.assertNotIn("Traceback", err)

    def test_compat_summary_shown(self):
        compat = {
            "m1": {"last_status": "pass", "pass_count": 6, "fail_count": 0},
            "m2": {"last_status": "fail", "pass_count": 1, "fail_count": 5},
        }
        rc, out, _ = self._run_doctor(
            models_resp={"data": [{"id": "m1"}]},
            chat_ok=True,
            compat_results=compat,
        )
        self.assertEqual(rc, 0)
        self.assertIn("Models tested:   2", out)
        self.assertIn("Agent PASS:      1", out)

    def test_parse_args(self):
        from llm_switchboard.cli import _parse_args
        cmd, filt, extra, opts = _parse_args(["--doctor"])
        self.assertEqual(cmd, "doctor")


# ─── Compat report ───────────────────────────────────────────────────

class TestCompatReport(unittest.TestCase):

    def _make_compat_data(self):
        return {
            "results": {
                "model-a": {
                    "last_run": "2026-01-01T00:00:00Z",
                    "tests": {
                        "format_compliance": {"passed": True, "latency_ms": 100},
                        "constraint_following": {"passed": True, "latency_ms": 100},
                        "no_hallucination": {"passed": True, "latency_ms": 100},
                        "tool_call_schema": {"passed": True, "latency_ms": 100},
                        "tool_call_chaining": {"passed": True, "latency_ms": 100},
                        "tool_call_error_recovery": {"passed": True, "latency_ms": 100},
                    },
                    "pass_count": 6, "fail_count": 0,
                    "last_status": "pass", "latency_ms": 600,
                },
                "model-b": {
                    "last_run": "2026-01-01T00:00:00Z",
                    "tests": {
                        "format_compliance": {"passed": False, "latency_ms": 100},
                        "constraint_following": {"passed": False, "latency_ms": 100},
                        "no_hallucination": {"passed": True, "latency_ms": 100},
                        "tool_call_schema": {"passed": False, "latency_ms": 100},
                        "tool_call_chaining": {"passed": False, "latency_ms": 100},
                        "tool_call_error_recovery": {"passed": False, "latency_ms": 100},
                    },
                    "pass_count": 1, "fail_count": 5,
                    "last_status": "fail", "latency_ms": 600,
                },
            }
        }

    def test_text_report_has_legend(self):
        from llm_switchboard.cli import cmd_compat_report
        data = self._make_compat_data()
        buf = io.StringIO()
        with patch("llm_switchboard.cli.load_compat", return_value=data):
            with patch("sys.stdout", buf):
                cmd_compat_report(as_json=False)
        out = buf.getvalue()
        self.assertIn("Legend", out)
        self.assertIn("AGENT", out)
        self.assertIn("Required tests:", out)

    def test_text_report_has_failure_reason(self):
        from llm_switchboard.cli import cmd_compat_report
        data = self._make_compat_data()
        buf = io.StringIO()
        with patch("llm_switchboard.cli.load_compat", return_value=data):
            with patch("sys.stdout", buf):
                cmd_compat_report(as_json=False)
        out = buf.getvalue()
        # model-b fails tool_call_schema (required) — should show reason
        self.assertIn("required failed", out)
        self.assertIn("tool_call_schema", out)

    def test_json_report_fields(self):
        from llm_switchboard.cli import cmd_compat_report
        data = self._make_compat_data()
        buf = io.StringIO()
        with patch("llm_switchboard.cli.load_compat", return_value=data):
            with patch("sys.stdout", buf):
                cmd_compat_report(as_json=True)
        out = json.loads(buf.getvalue())
        self.assertIn("required_tests", out)
        self.assertIn("scoring_rules", out)
        self.assertIn("models", out)
        # Check per-model fields
        ma = out["models"]["model-a"]
        self.assertEqual(ma["agent_status"], "pass")
        self.assertIn("passed_tests", ma)
        self.assertIn("failed_tests", ma)
        self.assertIn("required_failed", ma)
        self.assertEqual(ma["failed_tests"], [])
        # model-b should have failed tests
        mb = out["models"]["model-b"]
        self.assertEqual(mb["agent_status"], "fail")
        self.assertIn("tool_call_schema", mb["required_failed"])

    def test_model_summary_helper(self):
        from llm_switchboard.cli import _compat_model_summary
        entry = {
            "last_status": "partial",
            "pass_count": 4, "fail_count": 2,
            "latency_ms": 800, "last_run": "2026-01-01T00:00:00Z",
            "tests": {
                "format_compliance": {"passed": True, "latency_ms": 100},
                "constraint_following": {"passed": False, "latency_ms": 100},
                "no_hallucination": {"passed": True, "latency_ms": 100},
                "tool_call_schema": {"passed": True, "latency_ms": 100},
                "tool_call_chaining": {"passed": True, "latency_ms": 100},
                "tool_call_error_recovery": {"passed": False, "latency_ms": 100},
            },
        }
        s = _compat_model_summary(entry)
        self.assertEqual(s["agent_status"], "partial")
        self.assertIn("constraint_following", s["failed_tests"])
        self.assertEqual(s["required_failed"], [])  # no required tests failed


if __name__ == "__main__":
    unittest.main()

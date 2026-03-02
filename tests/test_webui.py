"""Tests for llm_switchboard.webui — Anthropic request/response helpers."""

import json
import unittest
from unittest.mock import MagicMock, patch

from llm_switchboard.webui import (
    build_anthropic_payload,
    parse_anthropic_response,
)


class TestBuildAnthropicPayload(unittest.TestCase):
    def test_simple_user_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        payload = build_anthropic_payload("model-1", msgs)
        self.assertEqual(payload["model"], "model-1")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertNotIn("system", payload)
        self.assertNotIn("tools", payload)

    def test_system_message_extracted(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ]
        payload = build_anthropic_payload("m", msgs)
        self.assertEqual(payload["system"], "You are helpful")
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_tool_calls_converted(self):
        msgs = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": json.dumps({"path": "/src"}),
                        },
                    }
                ],
            },
        ]
        payload = build_anthropic_payload("m", msgs)
        asst_msg = payload["messages"][1]
        self.assertEqual(asst_msg["role"], "assistant")
        self.assertIsInstance(asst_msg["content"], list)
        self.assertEqual(asst_msg["content"][0]["type"], "tool_use")
        self.assertEqual(asst_msg["content"][0]["name"], "list_files")
        self.assertEqual(asst_msg["content"][0]["input"], {"path": "/src"})

    def test_tool_role_converted(self):
        msgs = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result data"},
        ]
        payload = build_anthropic_payload("m", msgs)
        msg = payload["messages"][0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"][0]["type"], "tool_result")
        self.assertEqual(msg["content"][0]["tool_use_id"], "call_1")

    def test_tools_converted(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        payload = build_anthropic_payload("m", [{"role": "user", "content": "hi"}], tools=tools)
        self.assertEqual(len(payload["tools"]), 1)
        self.assertEqual(payload["tools"][0]["name"], "read_file")
        self.assertIn("input_schema", payload["tools"][0])

    def test_custom_max_tokens(self):
        payload = build_anthropic_payload("m", [{"role": "user", "content": "hi"}], max_tokens=512)
        self.assertEqual(payload["max_tokens"], 512)


class TestParseAnthropicResponse(unittest.TestCase):
    def test_text_only(self):
        resp = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3",
        }
        result = parse_anthropic_response(resp)
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello!")
        self.assertNotIn("tool_calls", result["choices"][0]["message"])

    def test_tool_use_blocks(self):
        resp = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me list the files."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "list_files",
                    "input": {"path": "/src"},
                },
            ],
            "model": "claude-3",
        }
        result = parse_anthropic_response(resp)
        msg = result["choices"][0]["message"]
        self.assertEqual(msg["content"], "Let me list the files.")
        self.assertEqual(len(msg["tool_calls"]), 1)
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "list_files")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"path": "/src"})

    def test_multiple_tool_use(self):
        resp = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "/a"}},
                {"type": "tool_use", "id": "tu_2", "name": "write_file", "input": {"path": "/b", "content": "data"}},
            ],
        }
        result = parse_anthropic_response(resp)
        msg = result["choices"][0]["message"]
        self.assertIsNone(msg["content"])
        self.assertEqual(len(msg["tool_calls"]), 2)

    def test_empty_content(self):
        resp = {"role": "assistant", "content": []}
        result = parse_anthropic_response(resp)
        self.assertIsNone(result["choices"][0]["message"]["content"])


class TestCompatProtocolDispatch(unittest.TestCase):
    """Test that compat validators work with both OpenAI and Anthropic formats."""

    def test_openai_tool_calls_parsed(self):
        from llm_switchboard.compat import _get_tool_calls_any

        message = {
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": '{"path": "/src"}'},
                }
            ]
        }
        calls = _get_tool_calls_any(message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "list_files")

    def test_anthropic_tool_use_parsed(self):
        from llm_switchboard.compat import _get_tool_calls_any

        message = {
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "list_files", "input": {"path": "/src"}},
            ]
        }
        calls = _get_tool_calls_any(message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "list_files")

    def test_anthropic_converted_via_parse(self):
        """Anthropic response → parse_anthropic_response → validators work."""
        from llm_switchboard.compat import _validate_tool_call_schema

        resp = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "list_files", "input": {"path": "/src"}},
            ],
        }
        converted = parse_anthropic_response(resp)
        message = converted["choices"][0]["message"]
        self.assertTrue(_validate_tool_call_schema(message))

    def test_anthropic_raw_validated(self):
        """Raw Anthropic message (content with tool_use) works with validators."""
        from llm_switchboard.compat import _validate_tool_call_schema

        message = {
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "list_files", "input": {"path": "/src"}},
            ]
        }
        self.assertTrue(_validate_tool_call_schema(message))


class TestApiGet(unittest.TestCase):
    """Tests for api_get HTTP helper."""

    @patch("llm_switchboard.webui.urlopen")
    @patch("llm_switchboard.webui.OPENWEBUI_KEY", "test-key")
    @patch("llm_switchboard.webui.OPENWEBUI_URL", "http://localhost:3100")
    def test_success(self, mock_urlopen):
        from llm_switchboard.webui import api_get

        body = json.dumps({"data": [{"id": "m1"}]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.info.return_value.get_content_charset.return_value = "utf-8"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = api_get("/api/models")
        self.assertIn("data", result)
        self.assertEqual(result["data"][0]["id"], "m1")

    @patch("llm_switchboard.webui.urlopen")
    @patch("llm_switchboard.webui.OPENWEBUI_KEY", "test-key")
    @patch("llm_switchboard.webui.OPENWEBUI_URL", "http://localhost:3100")
    def test_url_error(self, mock_urlopen):
        from urllib.error import URLError

        from llm_switchboard.webui import api_get

        mock_urlopen.side_effect = URLError("connection refused")
        result = api_get("/api/models")
        self.assertIn("_error", result)
        self.assertIn("connection refused", result["_error"])

    @patch("llm_switchboard.webui.urlopen")
    @patch("llm_switchboard.webui.OPENWEBUI_KEY", "test-key")
    @patch("llm_switchboard.webui.OPENWEBUI_URL", "http://localhost:3100")
    def test_timeout(self, mock_urlopen):
        from llm_switchboard.webui import api_get

        mock_urlopen.side_effect = TimeoutError("timed out")
        result = api_get("/api/models")
        self.assertIn("_error", result)
        self.assertIn("timed out", result["_error"])

    @patch("llm_switchboard.webui.urlopen")
    @patch("llm_switchboard.webui.OPENWEBUI_KEY", "test-key")
    @patch("llm_switchboard.webui.OPENWEBUI_URL", "http://localhost:3100")
    def test_html_response(self, mock_urlopen):
        """Server returning HTML instead of JSON should not crash."""
        from llm_switchboard.webui import api_get

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><body>Login required</body></html>"
        mock_resp.info.return_value.get_content_charset.return_value = "utf-8"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = api_get("/api/models")
        self.assertIn("_error", result)


class TestApiPost(unittest.TestCase):
    """Tests for api_post HTTP helper."""

    @patch("llm_switchboard.webui.urlopen")
    @patch("llm_switchboard.webui.OPENWEBUI_KEY", "test-key")
    @patch("llm_switchboard.webui.OPENWEBUI_URL", "http://localhost:3100")
    def test_success(self, mock_urlopen):
        from llm_switchboard.webui import api_post

        body = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.info.return_value.get_content_charset.return_value = "utf-8"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = api_post("/api/chat/completions", {"model": "m1", "messages": []})
        self.assertIn("choices", result)

    @patch("llm_switchboard.webui.urlopen")
    @patch("llm_switchboard.webui.OPENWEBUI_KEY", "test-key")
    @patch("llm_switchboard.webui.OPENWEBUI_URL", "http://localhost:3100")
    def test_url_error(self, mock_urlopen):
        from urllib.error import URLError

        from llm_switchboard.webui import api_post

        mock_urlopen.side_effect = URLError("refused")
        result = api_post("/api/chat/completions", {"model": "m1"})
        self.assertIn("_error", result)

    @patch("llm_switchboard.webui.urlopen")
    @patch("llm_switchboard.webui.OPENWEBUI_KEY", "test-key")
    @patch("llm_switchboard.webui.OPENWEBUI_URL", "http://localhost:3100")
    def test_timeout(self, mock_urlopen):
        from llm_switchboard.webui import api_post

        mock_urlopen.side_effect = TimeoutError("timed out")
        result = api_post("/api/chat/completions", {"model": "m1"})
        self.assertIn("_error", result)
        self.assertIn("timed out", result["_error"])


if __name__ == "__main__":
    unittest.main()

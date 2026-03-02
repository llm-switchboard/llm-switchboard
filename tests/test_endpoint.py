"""Tests for llm_switchboard.endpoint — URL normalization, detection, caching."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_switchboard.endpoint import (
    DETECT_TTL,
    Endpoint,
    _build_chat_probe_payload,
    _detect_endpoint,
    _extract_model_id,
    _has_json_content_type,
    _is_json_response,
    _load_detect_cache,
    _looks_like_models_list,
    _save_detect_cache,
    format_endpoint_info,
    load_api_mode,
    load_auto_prefer,
    normalize_url,
    resolve_endpoint,
    save_api_mode,
    save_auto_prefer,
)


class TestNormalizeUrl(unittest.TestCase):
    def test_strips_trailing_slash(self):
        self.assertEqual(normalize_url("http://localhost:3100/"), "http://localhost:3100")

    def test_strips_multiple_trailing_slashes(self):
        self.assertEqual(normalize_url("http://localhost:3100///"), "http://localhost:3100")

    def test_strips_whitespace(self):
        self.assertEqual(normalize_url("  http://localhost:3100  "), "http://localhost:3100")

    def test_no_trailing_slash(self):
        self.assertEqual(normalize_url("http://localhost:3100"), "http://localhost:3100")

    def test_preserves_path(self):
        self.assertEqual(normalize_url("http://host/api/v1/"), "http://host/api/v1")


class TestIsJsonResponse(unittest.TestCase):
    def test_json_object(self):
        self.assertTrue(_is_json_response(b'{"key": "value"}'))

    def test_json_array(self):
        self.assertTrue(_is_json_response(b"[1, 2, 3]"))

    def test_json_with_whitespace(self):
        self.assertTrue(_is_json_response(b'  \n{"key": "value"}'))

    def test_html_rejected(self):
        self.assertFalse(_is_json_response(b"<!DOCTYPE html><html>"))

    def test_html_tag_rejected(self):
        self.assertFalse(_is_json_response(b"<html><body>Not found</body></html>"))

    def test_empty_rejected(self):
        self.assertFalse(_is_json_response(b""))

    def test_plain_text_rejected(self):
        self.assertFalse(_is_json_response(b"Not Found"))

    def test_redirect_html_rejected(self):
        self.assertFalse(_is_json_response(b'<head><meta http-equiv="refresh"'))


class TestHasJsonContentType(unittest.TestCase):
    def test_json_content_type(self):
        h = MagicMock()
        h.get_content_type.return_value = "application/json"
        self.assertTrue(_has_json_content_type(h))

    def test_json_charset(self):
        h = MagicMock()
        h.get_content_type.return_value = "application/json; charset=utf-8"
        # get_content_type typically returns just type/subtype, but let's test
        self.assertTrue(_has_json_content_type(h))

    def test_html_rejected(self):
        h = MagicMock()
        h.get_content_type.return_value = "text/html"
        self.assertFalse(_has_json_content_type(h))

    def test_fallback_to_get(self):
        h = MagicMock(spec=[])  # no get_content_type
        h.get = MagicMock(return_value="application/json")
        self.assertTrue(_has_json_content_type(h))


class TestExtractModelId(unittest.TestCase):
    def test_extracts_first_id(self):
        data = {"data": [{"id": "groq/llama-3.3-70b"}, {"id": "gpt-4"}]}
        self.assertEqual(_extract_model_id(data), "groq/llama-3.3-70b")

    def test_extracts_name_fallback(self):
        data = {"data": [{"name": "my-model"}]}
        self.assertEqual(_extract_model_id(data), "my-model")

    def test_empty_data(self):
        self.assertIsNone(_extract_model_id({"data": []}))

    def test_no_data_key(self):
        self.assertIsNone(_extract_model_id({"models": []}))

    def test_non_dict_items(self):
        self.assertIsNone(_extract_model_id({"data": ["string_item"]}))


class TestBuildChatProbePayload(unittest.TestCase):
    def test_with_model(self):
        p = _build_chat_probe_payload("groq/llama")
        self.assertEqual(p["model"], "groq/llama")
        self.assertEqual(p["max_tokens"], 1)
        self.assertEqual(p["temperature"], 0)

    def test_without_model(self):
        p = _build_chat_probe_payload(None)
        self.assertEqual(p["model"], "__probe__")


class TestApiModeConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mode_file = Path(self.tmpdir) / "api_mode.conf"

    def test_load_missing_returns_auto(self):
        self.assertEqual(load_api_mode(self.mode_file), "auto")

    def test_save_and_load(self):
        save_api_mode("openai", self.mode_file)
        self.assertEqual(load_api_mode(self.mode_file), "openai")

    def test_load_anthropic(self):
        self.mode_file.write_text("anthropic\n")
        self.assertEqual(load_api_mode(self.mode_file), "anthropic")

    def test_load_auto(self):
        self.mode_file.write_text("auto\n")
        self.assertEqual(load_api_mode(self.mode_file), "auto")

    def test_load_invalid_returns_auto(self):
        self.mode_file.write_text("invalid_mode\n")
        self.assertEqual(load_api_mode(self.mode_file), "auto")

    def test_load_case_insensitive(self):
        self.mode_file.write_text("OPENAI\n")
        self.assertEqual(load_api_mode(self.mode_file), "openai")


class TestAutoPreferConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.prefer_file = Path(self.tmpdir) / "auto_prefer.conf"

    def test_load_missing_returns_openai(self):
        self.assertEqual(load_auto_prefer(self.prefer_file), "openai")

    def test_save_and_load(self):
        save_auto_prefer("anthropic", self.prefer_file)
        self.assertEqual(load_auto_prefer(self.prefer_file), "anthropic")

    def test_load_openai(self):
        self.prefer_file.write_text("openai\n")
        self.assertEqual(load_auto_prefer(self.prefer_file), "openai")

    def test_load_invalid_returns_openai(self):
        self.prefer_file.write_text("invalid\n")
        self.assertEqual(load_auto_prefer(self.prefer_file), "openai")


class TestLooksLikeModelsList(unittest.TestCase):
    def test_openai_format(self):
        self.assertTrue(_looks_like_models_list({"data": [{"id": "gpt-4"}]}))

    def test_models_key_not_accepted(self):
        self.assertFalse(_looks_like_models_list({"models": [{"name": "llama"}]}))

    def test_empty_data(self):
        self.assertTrue(_looks_like_models_list({"data": []}))

    def test_no_data_key(self):
        self.assertFalse(_looks_like_models_list({"error": "not found"}))

    def test_data_not_list(self):
        self.assertFalse(_looks_like_models_list({"data": "string"}))


class TestDetectCache(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_file = Path(self.tmpdir) / "detect.json"

    def test_load_missing(self):
        self.assertIsNone(_load_detect_cache(self.cache_file))

    def test_save_and_load(self):
        result = {"base_url": "http://localhost:3100", "mode": "openai", "chat_path": "/api/chat/completions", "models_path": "/api/models"}
        _save_detect_cache(result, self.cache_file)
        loaded = _load_detect_cache(self.cache_file)
        self.assertEqual(loaded, result)

    def test_expired_cache(self):
        result = {"base_url": "http://localhost:3100", "mode": "openai", "chat_path": "/api/chat/completions", "models_path": "/api/models"}
        _save_detect_cache(result, self.cache_file)
        import os

        old_time = time.time() - DETECT_TTL - 10
        os.utime(self.cache_file, (old_time, old_time))
        self.assertIsNone(_load_detect_cache(self.cache_file))

    def test_corrupted_cache(self):
        self.cache_file.write_text("not json")
        self.assertIsNone(_load_detect_cache(self.cache_file))

    def test_missing_fields(self):
        self.cache_file.write_text('{"base_url": "http://localhost"}')
        self.assertIsNone(_load_detect_cache(self.cache_file))

    def test_cache_preserves_source_and_probe(self):
        result = {
            "base_url": "http://host",
            "mode": "openai",
            "chat_path": "/api/chat/completions",
            "models_path": "/api/models",
            "source": "auto-detect",
            "probe_match": "POST /api/chat/completions (model probe)",
        }
        _save_detect_cache(result, self.cache_file)
        loaded = _load_detect_cache(self.cache_file)
        self.assertEqual(loaded["source"], "auto-detect")
        self.assertIn("model probe", loaded["probe_match"])


# ─── Helpers for probe mocking ────────────────────────────────────────


def _make_probe_mocks(v1_models=None, api_models=None, v1_chat_reachable=False, api_chat_reachable=False, anthropic_reachable=False):
    """Return (get_fn, post_reachable_fn) for probe mocking.

    get_fn:             returns dict or None for GET probes
    post_reachable_fn:  returns bool for POST reachability probes
    """

    def get_fn(url, key, **kw):
        if "/v1/models" in url:
            return v1_models
        if "/api/models" in url:
            return api_models
        return None

    def post_reachable_fn(url, key, payload, **kw):
        if "/v1/chat/completions" in url:
            return v1_chat_reachable
        if "/api/chat/completions" in url:
            return api_chat_reachable
        if "/api/v1/messages" in url:
            return anthropic_reachable
        return False

    return get_fn, post_reachable_fn


class TestDetectEndpoint(unittest.TestCase):
    """Test auto-detection with mocked HTTP probes."""

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_openai_chat_with_model_probe(self, mock_get, mock_post):
        """When /api/models returns models and chat probe succeeds, choose openai."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "groq/llama-3.3-70b"}]},
            api_chat_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key")
        self.assertEqual(ep.mode, "openai")
        self.assertEqual(ep.chat_path, "/api/chat/completions")
        self.assertIn("model probe", ep.probe_match)

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_v1_chat_preferred(self, mock_get, mock_post):
        """If /v1/chat/completions is reachable, prefer it over /api."""
        get_fn, post_fn = _make_probe_mocks(
            v1_models={"data": [{"id": "gpt-4"}]},
            v1_chat_reachable=True,
            api_chat_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key")
        self.assertEqual(ep.mode, "openai")
        self.assertEqual(ep.chat_path, "/v1/chat/completions")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_openai_preferred_over_anthropic(self, mock_get, mock_post):
        """When BOTH openai chat AND anthropic respond, prefer=openai chooses openai."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "m1"}]},
            api_chat_reachable=True,
            anthropic_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key", prefer="openai")
        self.assertEqual(ep.mode, "openai")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_anthropic_only_when_openai_unreachable(self, mock_get, mock_post):
        """Anthropic chosen only when no openai chat endpoint is reachable."""
        get_fn, post_fn = _make_probe_mocks(
            anthropic_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key", prefer="openai")
        self.assertEqual(ep.mode, "anthropic")
        self.assertEqual(ep.chat_path, "/api/v1/messages")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_prefer_anthropic_when_reachable(self, mock_get, mock_post):
        """With prefer=anthropic, anthropic chosen first if reachable."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "m1"}]},
            api_chat_reachable=True,
            anthropic_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key", prefer="anthropic")
        self.assertEqual(ep.mode, "anthropic")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_prefer_anthropic_falls_back_to_openai(self, mock_get, mock_post):
        """With prefer=anthropic but anthropic unreachable, fall back to openai."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "m1"}]},
            api_chat_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key", prefer="anthropic")
        self.assertEqual(ep.mode, "openai")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_chat_probe_401_json_ct_is_reachable(self, mock_get, mock_post):
        """Chat probe returning 401 with JSON content-type (empty body) counts as reachable."""
        # This simulates what _probe_post_reachable does internally:
        # HTTP 401 + application/json content-type → True
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "test-model"}]},
            api_chat_reachable=True,  # _probe_post_reachable handles 401+JSON CT
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key")
        self.assertEqual(ep.mode, "openai")
        self.assertEqual(ep.chat_path, "/api/chat/completions")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_chat_probe_400_json_body_is_reachable(self, mock_get, mock_post):
        """Chat probe returning 400 with JSON error body counts as reachable."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "test-model"}]},
            api_chat_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key")
        self.assertEqual(ep.mode, "openai")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_chat_truly_unreachable_falls_to_anthropic(self, mock_get, mock_post):
        """When chat probes truly fail (connection error), fall to anthropic if available."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "m1"}]},
            anthropic_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key")
        self.assertEqual(ep.mode, "anthropic")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_models_list_fallback(self, mock_get, mock_post):
        """If chat probes and anthropic fail but models list works, assume openai."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "m1"}]},
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key")
        self.assertEqual(ep.mode, "openai")
        self.assertIn("models-list fallback", ep.probe_match)

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_nothing_works_fallback(self, mock_get, mock_post):
        """If nothing responds, fall back to openai + /api/chat/completions."""
        mock_get.return_value = None
        mock_post.return_value = False
        ep = _detect_endpoint("http://host", "key")
        self.assertEqual(ep.mode, "openai")
        self.assertEqual(ep.chat_path, "/api/chat/completions")
        self.assertIn("fallback", ep.probe_match)

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_source_is_auto_detect(self, mock_get, mock_post):
        """Detected endpoints have source='auto-detect'."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "m1"}]},
            api_chat_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key")
        self.assertEqual(ep.source, "auto-detect")

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_probe_match_includes_model_probe(self, mock_get, mock_post):
        """When chat probe succeeds, probe_match mentions 'model probe'."""
        get_fn, post_fn = _make_probe_mocks(
            api_models={"data": [{"id": "m1"}]},
            api_chat_reachable=True,
        )
        mock_get.side_effect = get_fn
        mock_post.side_effect = post_fn
        ep = _detect_endpoint("http://host", "key")
        self.assertIn("model probe", ep.probe_match)

    @patch("llm_switchboard.endpoint._probe_post_reachable")
    @patch("llm_switchboard.endpoint._probe_get")
    def test_real_model_id_used_in_probe(self, mock_get, mock_post):
        """The chat probe uses a real model ID extracted from models list."""
        get_fn, _ = _make_probe_mocks(
            api_models={"data": [{"id": "groq/llama-3.3-70b"}]},
        )
        mock_get.side_effect = get_fn

        captured_payloads = []

        def capture_post(url, key, payload, **kw):
            captured_payloads.append(payload)
            if "/api/chat/completions" in url:
                return True
            return False

        mock_post.side_effect = capture_post

        _detect_endpoint("http://host", "key")
        # Check that probe used the real model ID
        chat_payloads = [p for p in captured_payloads if p.get("model") != "__probe__"]
        self.assertTrue(any(p["model"] == "groq/llama-3.3-70b" for p in chat_payloads))


class TestResolveEndpoint(unittest.TestCase):
    """Test resolve_endpoint with various modes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_file = Path(self.tmpdir) / "detect.json"

    @patch("llm_switchboard.endpoint._probe_get")
    def test_openai_mode_v1(self, mock_get):
        mock_get.side_effect = lambda url, key, **kw: {"data": []} if "/v1/models" in url else None
        ep = resolve_endpoint(mode="openai", base_url="http://host", cache_file=self.cache_file, skip_cache=True)
        self.assertEqual(ep.mode, "openai")
        self.assertEqual(ep.chat_path, "/v1/chat/completions")

    @patch("llm_switchboard.endpoint._probe_get")
    def test_openai_mode_fallback(self, mock_get):
        mock_get.return_value = None
        ep = resolve_endpoint(mode="openai", base_url="http://host", cache_file=self.cache_file, skip_cache=True)
        self.assertEqual(ep.mode, "openai")
        self.assertEqual(ep.chat_path, "/api/chat/completions")

    def test_anthropic_mode(self):
        ep = resolve_endpoint(mode="anthropic", base_url="http://host", cache_file=self.cache_file)
        self.assertEqual(ep.mode, "anthropic")
        self.assertEqual(ep.chat_path, "/api/v1/messages")

    def test_anthropic_mode_source(self):
        ep = resolve_endpoint(mode="anthropic", base_url="http://host", cache_file=self.cache_file, source="cli")
        self.assertEqual(ep.source, "cli")

    @patch("llm_switchboard.endpoint._detect_endpoint")
    def test_auto_mode_cached(self, mock_detect):
        """Cached result is used in auto mode."""
        cached = {
            "base_url": "http://host",
            "mode": "openai",
            "chat_path": "/v1/chat/completions",
            "models_path": "/v1/models",
            "source": "auto-detect",
            "probe_match": "POST /v1/chat/completions (model probe)",
        }
        _save_detect_cache(cached, self.cache_file)
        ep = resolve_endpoint(mode="auto", base_url="http://host", cache_file=self.cache_file)
        self.assertEqual(ep.mode, "openai")
        self.assertEqual(ep.chat_path, "/v1/chat/completions")
        self.assertEqual(ep.source, "auto-detect")
        mock_detect.assert_not_called()

    @patch("llm_switchboard.endpoint._detect_endpoint")
    def test_auto_mode_stale_cache(self, mock_detect):
        """Stale cache triggers re-detection."""
        cached = {"base_url": "http://host", "mode": "openai", "chat_path": "/api/chat/completions", "models_path": "/api/models"}
        _save_detect_cache(cached, self.cache_file)
        import os

        old_time = time.time() - DETECT_TTL - 10
        os.utime(self.cache_file, (old_time, old_time))
        mock_detect.return_value = Endpoint("http://host", "openai", "/v1/chat/completions", "/v1/models", "auto-detect", "POST /v1/chat/completions (model probe)")
        ep = resolve_endpoint(mode="auto", base_url="http://host", cache_file=self.cache_file)
        mock_detect.assert_called_once()
        self.assertEqual(ep.chat_path, "/v1/chat/completions")

    @patch("llm_switchboard.endpoint._detect_endpoint")
    def test_auto_mode_different_url_ignores_cache(self, mock_detect):
        """Cache for different URL is not used."""
        cached = {"base_url": "http://other-host", "mode": "openai", "chat_path": "/api/chat/completions", "models_path": "/api/models"}
        _save_detect_cache(cached, self.cache_file)
        mock_detect.return_value = Endpoint("http://host", "anthropic", "/api/v1/messages", "/api/models", "auto-detect", "POST /api/v1/messages")
        ep = resolve_endpoint(mode="auto", base_url="http://host", cache_file=self.cache_file)
        mock_detect.assert_called_once()
        self.assertEqual(ep.mode, "anthropic")


class TestFormatEndpointInfo(unittest.TestCase):
    def test_format_basic(self):
        ep = Endpoint("http://localhost:3100", "openai", "/api/chat/completions", "/api/models")
        info = format_endpoint_info(ep)
        self.assertIn("http://localhost:3100", info)
        self.assertIn("openai", info)
        self.assertIn("/api/chat/completions", info)
        self.assertIn("/api/models", info)

    def test_format_with_source_and_probe(self):
        ep = Endpoint("http://host", "openai", "/api/chat/completions", "/api/models", "auto-detect", "POST /api/chat/completions (model probe)")
        info = format_endpoint_info(ep)
        self.assertIn("Mode source:", info)
        self.assertIn("auto-detect", info)
        self.assertIn("Probe matched:", info)
        self.assertIn("model probe", info)

    def test_format_no_source(self):
        ep = Endpoint("http://host", "openai", "/chat", "/models")
        info = format_endpoint_info(ep)
        self.assertNotIn("Mode source:", info)


class TestEndpointNamedTuple(unittest.TestCase):
    def test_fields(self):
        ep = Endpoint("http://h", "openai", "/chat", "/models")
        self.assertEqual(ep.base_url, "http://h")
        self.assertEqual(ep.mode, "openai")
        self.assertEqual(ep.chat_path, "/chat")
        self.assertEqual(ep.models_path, "/models")
        self.assertEqual(ep.source, "")
        self.assertEqual(ep.probe_match, "")

    def test_fields_with_source(self):
        ep = Endpoint("http://h", "openai", "/chat", "/models", "cli", "")
        self.assertEqual(ep.source, "cli")

    def test_asdict(self):
        ep = Endpoint("http://h", "openai", "/chat", "/models", "auto-detect", "GET /v1/models")
        d = ep._asdict()
        self.assertEqual(d["base_url"], "http://h")
        self.assertEqual(d["mode"], "openai")
        self.assertEqual(d["source"], "auto-detect")
        self.assertEqual(d["probe_match"], "GET /v1/models")


if __name__ == "__main__":
    unittest.main()

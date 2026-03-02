"""Tests for cache module — file-based caching."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from llm_switchboard.cache import fetch_cached


class TestFetchCached(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _cache_file(self, name="cache.json"):
        return Path(self.tmpdir) / name

    def test_fresh_cache_used(self):
        cf = self._cache_file()
        data = {"key": "value", "items": [1, 2, 3]}
        cf.write_text(json.dumps(data))

        call_count = 0
        def mock_api(path, timeout):
            nonlocal call_count
            call_count += 1
            return {"key": "new_value"}

        result = fetch_cached(mock_api, "/test", cf, "key")
        self.assertEqual(result["key"], "value")
        self.assertEqual(call_count, 0)  # API not called

    def test_stale_cache_refetches(self):
        cf = self._cache_file()
        cf.write_text(json.dumps({"key": "old"}))
        # Set mtime to 2 hours ago
        old_time = time.time() - 7200
        import os
        os.utime(cf, (old_time, old_time))

        def mock_api(path, timeout):
            return {"key": "new_value"}

        result = fetch_cached(mock_api, "/test", cf, "key")
        self.assertEqual(result["key"], "new_value")

    def test_missing_cache_fetches(self):
        cf = self._cache_file("nonexistent.json")

        def mock_api(path, timeout):
            return {"key": "fetched"}

        result = fetch_cached(mock_api, "/test", cf, "key")
        self.assertEqual(result["key"], "fetched")
        # Check file was written
        self.assertTrue(cf.exists())

    def test_invalid_response_not_cached(self):
        cf = self._cache_file()

        def mock_api(path, timeout):
            return {"_error": "connection refused"}

        result = fetch_cached(mock_api, "/test", cf, "key")
        self.assertIn("_error", result)
        self.assertFalse(cf.exists())

    def test_corrupted_cache_refetches(self):
        cf = self._cache_file()
        cf.write_text("not json{{{")

        def mock_api(path, timeout):
            return {"key": "recovered"}

        result = fetch_cached(mock_api, "/test", cf, "key")
        self.assertEqual(result["key"], "recovered")


if __name__ == "__main__":
    unittest.main()

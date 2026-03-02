"""Tests for session module — usage tracking."""

import json
import tempfile
import unittest
from pathlib import Path

from llm_switchboard.session import load_local_usage, save_local_usage, record_session, get_all_usage


class TestUsageTracking(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.usage_file = Path(self.tmpdir) / "usage.json"

    def test_load_missing(self):
        data = load_local_usage(self.usage_file)
        self.assertEqual(data, {"models": {}})

    def test_save_and_load(self):
        data = {"models": {"test-model": {"input_tokens": 100, "output_tokens": 50, "message_count": 3, "sessions": 1}}}
        save_local_usage(data, self.usage_file)
        loaded = load_local_usage(self.usage_file)
        self.assertEqual(loaded["models"]["test-model"]["input_tokens"], 100)

    def test_record_session_new(self):
        session = {"input_tokens": 500, "output_tokens": 200, "message_count": 5}
        record_session("model-a", session, self.usage_file)
        data = load_local_usage(self.usage_file)
        entry = data["models"]["model-a"]
        self.assertEqual(entry["input_tokens"], 500)
        self.assertEqual(entry["sessions"], 1)

    def test_record_session_accumulates(self):
        s1 = {"input_tokens": 100, "output_tokens": 50, "message_count": 2}
        s2 = {"input_tokens": 200, "output_tokens": 100, "message_count": 3}
        record_session("model-a", s1, self.usage_file)
        record_session("model-a", s2, self.usage_file)
        data = load_local_usage(self.usage_file)
        entry = data["models"]["model-a"]
        self.assertEqual(entry["input_tokens"], 300)
        self.assertEqual(entry["output_tokens"], 150)
        self.assertEqual(entry["message_count"], 5)
        self.assertEqual(entry["sessions"], 2)

    def test_get_all_usage_filters_zero(self):
        data = {
            "models": {
                "active": {"input_tokens": 100, "output_tokens": 50, "message_count": 3, "sessions": 1},
                "empty": {"input_tokens": 0, "output_tokens": 0, "message_count": 0, "sessions": 0},
            }
        }
        save_local_usage(data, self.usage_file)
        results = get_all_usage(self.usage_file)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model_id"], "active")

    def test_corrupted_usage_file(self):
        self.usage_file.write_text("not json{{{")
        data = load_local_usage(self.usage_file)
        self.assertEqual(data, {"models": {}})

    def test_atomic_write(self):
        """Save should not leave partial files on crash-like scenarios."""
        data = {"models": {"m": {"input_tokens": 1, "output_tokens": 1, "message_count": 1, "sessions": 1}}}
        save_local_usage(data, self.usage_file)
        # Tmp file should not remain
        tmp = self.usage_file.with_suffix(".json.tmp")
        self.assertFalse(tmp.exists())
        self.assertTrue(self.usage_file.exists())


if __name__ == "__main__":
    unittest.main()

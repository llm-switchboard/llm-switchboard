"""Tests for config module — favorites and last model."""

import tempfile
import unittest
from pathlib import Path

from llm_switchboard.config import (
    fav_list, fav_has, fav_add, fav_rm, last_model_read, last_model_write,
    APP_NAME,
)


class TestConfigPaths(unittest.TestCase):

    def test_config_dir_uses_app_name(self):
        from llm_switchboard.config import CONFIG_DIR
        self.assertEqual(CONFIG_DIR.name, APP_NAME)

    def test_cache_dir_uses_app_name(self):
        from llm_switchboard.config import CACHE_DIR
        self.assertEqual(CACHE_DIR.name, APP_NAME)


class TestFavorites(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fav_file = Path(self.tmpdir) / "favorites.conf"

    def test_empty_favorites(self):
        self.assertEqual(fav_list(self.fav_file), [])

    def test_add_favorite(self):
        self.fav_file.touch()
        msg = fav_add("model-a", self.fav_file)
        self.assertIn("Added", msg)
        self.assertEqual(fav_list(self.fav_file), ["model-a"])

    def test_add_duplicate(self):
        self.fav_file.write_text("model-a\n")
        msg = fav_add("model-a", self.fav_file)
        self.assertIn("Already", msg)
        self.assertEqual(fav_list(self.fav_file), ["model-a"])

    def test_remove_favorite(self):
        self.fav_file.write_text("model-a\nmodel-b\n")
        msg = fav_rm("model-a", self.fav_file)
        self.assertIn("Removed", msg)
        self.assertEqual(fav_list(self.fav_file), ["model-b"])

    def test_remove_nonexistent(self):
        self.fav_file.write_text("model-a\n")
        msg = fav_rm("model-x", self.fav_file)
        self.assertIn("Not a favorite", msg)

    def test_fav_has(self):
        self.fav_file.write_text("model-a\nmodel-b\n")
        self.assertTrue(fav_has("model-a", self.fav_file))
        self.assertFalse(fav_has("model-c", self.fav_file))

    def test_missing_file(self):
        self.assertEqual(fav_list(self.fav_file), [])
        self.assertFalse(fav_has("x", self.fav_file))

    def test_blank_lines_ignored(self):
        self.fav_file.write_text("model-a\n\n\nmodel-b\n\n")
        self.assertEqual(fav_list(self.fav_file), ["model-a", "model-b"])


class TestLastModel(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.last_file = Path(self.tmpdir) / "last_model"

    def test_no_last_model(self):
        self.assertEqual(last_model_read(self.last_file), "")

    def test_write_and_read(self):
        last_model_write("gpt-4o", self.last_file)
        self.assertEqual(last_model_read(self.last_file), "gpt-4o")

    def test_overwrite(self):
        last_model_write("model-a", self.last_file)
        last_model_write("model-b", self.last_file)
        self.assertEqual(last_model_read(self.last_file), "model-b")

    def test_strips_whitespace(self):
        self.last_file.write_text("  model-a  \n")
        self.assertEqual(last_model_read(self.last_file), "model-a")


if __name__ == "__main__":
    unittest.main()

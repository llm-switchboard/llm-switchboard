"""Tests for free_tier module — rule parsing, matching, is_free_tier."""

import unittest

from llm_switchboard.free_tier import FreeTierRule, is_free_tier, parse_free_tier_rules


class TestFreeTierRuleParsing(unittest.TestCase):
    def test_simple_provider(self):
        rules = parse_free_tier_rules("groq\n")
        self.assertIn("groq", rules)
        r = rules["groq"]
        self.assertIsNone(r.includes)
        self.assertEqual(r.excludes, [])

    def test_provider_with_excludes(self):
        rules = parse_free_tier_rules("gemini:!imagen,!veo\n")
        self.assertIn("gemini", rules)
        r = rules["gemini"]
        self.assertIsNone(r.includes)
        self.assertEqual(r.excludes, ["imagen", "veo"])

    def test_provider_with_includes(self):
        rules = parse_free_tier_rules("gemini:flash,gemma\n")
        r = rules["gemini"]
        self.assertEqual(r.includes, ["flash", "gemma"])
        self.assertEqual(r.excludes, [])

    def test_mixed_includes_excludes(self):
        rules = parse_free_tier_rules("gemini:flash,gemma,!imagen\n")
        r = rules["gemini"]
        self.assertEqual(r.includes, ["flash", "gemma"])
        self.assertEqual(r.excludes, ["imagen"])

    def test_comments_ignored(self):
        text = "# comment\ngroq  # inline comment\n"
        rules = parse_free_tier_rules(text)
        self.assertIn("groq", rules)
        self.assertEqual(len(rules), 1)

    def test_blank_lines_ignored(self):
        text = "\n\ngroq\n\ncerebras\n\n"
        rules = parse_free_tier_rules(text)
        self.assertEqual(len(rules), 2)

    def test_case_insensitive(self):
        rules = parse_free_tier_rules("GROQ\n")
        self.assertIn("groq", rules)

    def test_empty_string(self):
        rules = parse_free_tier_rules("")
        self.assertEqual(rules, {})

    def test_multiple_providers(self):
        text = "groq\ncerebras\nmistral\n"
        rules = parse_free_tier_rules(text)
        self.assertEqual(set(rules.keys()), {"groq", "cerebras", "mistral"})


class TestFreeTierRuleMatches(unittest.TestCase):
    def test_all_models(self):
        r = FreeTierRule("groq", None, [])
        self.assertTrue(r.matches("llama-3.3-70b"))
        self.assertTrue(r.matches("anything"))

    def test_exclude(self):
        r = FreeTierRule("gemini", None, ["imagen", "veo"])
        self.assertTrue(r.matches("gemini-2.5-flash"))
        self.assertFalse(r.matches("imagen-3.0"))
        self.assertFalse(r.matches("veo-2"))

    def test_include_only(self):
        r = FreeTierRule("gemini", ["flash", "gemma"], [])
        self.assertTrue(r.matches("gemini-2.5-flash"))
        self.assertTrue(r.matches("gemma-2b"))
        self.assertFalse(r.matches("gemini-pro"))

    def test_include_and_exclude(self):
        r = FreeTierRule("gemini", ["flash", "gemma"], ["imagen"])
        self.assertTrue(r.matches("gemini-flash-001"))
        self.assertFalse(r.matches("imagen-flash"))  # exclude takes priority

    def test_case_insensitive_matching(self):
        # The matches() method lowercases the model_id, so "IMAGEN" exclude
        # (uppercase) won't match against lowercased input unless pattern is also lower.
        # Parser normalizes to lowercase, so test with parser output:
        rules = parse_free_tier_rules("test:!imagen\n")
        r = rules["test"]
        self.assertFalse(r.matches("IMAGEN-3"))
        self.assertFalse(r.matches("imagen-3"))

    def test_repr_simple(self):
        r = FreeTierRule("groq", None, [])
        self.assertEqual(repr(r), "groq")

    def test_repr_complex(self):
        r = FreeTierRule("gemini", ["flash"], ["imagen"])
        self.assertEqual(repr(r), "gemini:flash,!imagen")


class TestIsFreeTier(unittest.TestCase):
    def test_no_rules(self):
        self.assertFalse(is_free_tier("groq", "llama-3", {}))

    def test_matching_rule(self):
        rules = {"groq": FreeTierRule("groq", None, [])}
        self.assertTrue(is_free_tier("groq", "llama-3", rules))

    def test_non_matching_provider(self):
        rules = {"groq": FreeTierRule("groq", None, [])}
        self.assertFalse(is_free_tier("openai", "gpt-4", rules))

    def test_gemini_cache_hit_free(self):
        rules = {"gemini": FreeTierRule("gemini", None, [])}
        cache = {"free": ["gemini-2.5-flash"], "paid": ["gemini-2.5-pro"]}
        self.assertTrue(is_free_tier("gemini", "gemini-2.5-flash", rules, cache))

    def test_gemini_cache_hit_paid(self):
        rules = {"gemini": FreeTierRule("gemini", None, [])}
        cache = {"free": ["gemini-2.5-flash"], "paid": ["gemini-2.5-pro"]}
        self.assertFalse(is_free_tier("gemini", "gemini-2.5-pro", rules, cache))

    def test_gemini_latest_alias(self):
        rules = {"gemini": FreeTierRule("gemini", None, [])}
        cache = {"free": ["gemini-2.5-flash"], "paid": []}
        self.assertTrue(is_free_tier("gemini", "gemini-2.5-flash-latest", rules, cache))

    def test_gemini_always_free_gemma(self):
        rules = {"gemini": FreeTierRule("gemini", None, [])}
        cache = {"free": [], "paid": []}
        self.assertTrue(is_free_tier("gemini", "gemma-2b", rules, cache))

    def test_gemini_always_paid_imagen(self):
        rules = {"gemini": FreeTierRule("gemini", None, [])}
        cache = {"free": [], "paid": []}
        self.assertFalse(is_free_tier("gemini", "imagen-3.0", rules, cache))

    def test_gemini_provider_prefix_stripped(self):
        rules = {"gemini": FreeTierRule("gemini", None, [])}
        cache = {"free": ["gemini-2.5-flash"], "paid": []}
        self.assertTrue(is_free_tier("gemini", "models/gemini-2.5-flash", rules, cache))


if __name__ == "__main__":
    unittest.main()

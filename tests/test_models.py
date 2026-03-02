"""Tests for models module — provider detection and context formatting."""

import unittest

from llm_switchboard.models import DOMAIN_TO_PROVIDER, format_ctx, provider_from_url


class TestProviderFromUrl(unittest.TestCase):
    def test_groq(self):
        self.assertEqual(provider_from_url("https://api.groq.com/openai/v1"), "groq")

    def test_openai(self):
        self.assertEqual(provider_from_url("https://api.openai.com/v1"), "openai")

    def test_anthropic(self):
        self.assertEqual(provider_from_url("https://api.anthropic.com/v1"), "anthropic")

    def test_gemini(self):
        self.assertEqual(provider_from_url("https://generativelanguage.googleapis.com/v1"), "gemini")

    def test_cerebras(self):
        self.assertEqual(provider_from_url("https://api.cerebras.ai/v1"), "cerebras")

    def test_mistral(self):
        self.assertEqual(provider_from_url("https://api.mistral.ai/v1"), "mistral")

    def test_openrouter(self):
        self.assertEqual(provider_from_url("https://openrouter.ai/api/v1"), "openrouter")

    def test_unknown_domain(self):
        self.assertEqual(provider_from_url("https://api.someservice.com/v1"), "someservice")

    def test_empty_url(self):
        self.assertEqual(provider_from_url(""), "external")

    def test_localhost(self):
        # Regex captures "localhost:11434" as the domain token (includes port)
        self.assertEqual(provider_from_url("http://localhost:11434"), "localhost:11434")

    def test_all_known_domains_covered(self):
        """Ensure DOMAIN_TO_PROVIDER isn't empty and all values are strings."""
        self.assertGreater(len(DOMAIN_TO_PROVIDER), 0)
        for k, v in DOMAIN_TO_PROVIDER.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, str)


class TestFormatCtx(unittest.TestCase):
    def test_millions(self):
        self.assertEqual(format_ctx(1_000_000), "1M")
        self.assertEqual(format_ctx(2_000_000), "2M")

    def test_thousands(self):
        self.assertEqual(format_ctx(128_000), "128K")
        self.assertEqual(format_ctx(8_000), "8K")

    def test_small(self):
        self.assertEqual(format_ctx(512), "512")

    def test_zero(self):
        self.assertEqual(format_ctx(0), "")

    def test_none(self):
        self.assertEqual(format_ctx(None), "")

    def test_negative(self):
        self.assertEqual(format_ctx(-1), "")

    def test_float(self):
        self.assertEqual(format_ctx(128000.0), "128K")


if __name__ == "__main__":
    unittest.main()

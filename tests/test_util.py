"""Tests for util module — pure utility functions."""

import unittest

from llm_switchboard.util import fmt_price, fmt_tokens, sanitize_display, visible_len


class TestSanitizeDisplay(unittest.TestCase):
    def test_strips_ansi(self):
        self.assertEqual(sanitize_display("\033[31mhello\033[0m"), "hello")

    def test_strips_cr(self):
        self.assertEqual(sanitize_display("hello\rworld"), "helloworld")

    def test_strips_null(self):
        self.assertEqual(sanitize_display("hello\x00world"), "helloworld")

    def test_plain_text(self):
        self.assertEqual(sanitize_display("hello world"), "hello world")

    def test_empty(self):
        self.assertEqual(sanitize_display(""), "")

    def test_complex_ansi(self):
        self.assertEqual(sanitize_display("\033[1;37mBOLD\033[0m"), "BOLD")


class TestFmtPrice(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(fmt_price(0), "$0")

    def test_small(self):
        self.assertEqual(fmt_price(0.50), "$0.5")

    def test_medium(self):
        self.assertEqual(fmt_price(3.00), "$3")

    def test_large(self):
        self.assertEqual(fmt_price(15.00), "$15")

    def test_very_large(self):
        self.assertEqual(fmt_price(100.0), "$100")

    def test_fractional(self):
        self.assertEqual(fmt_price(0.25), "$0.25")

    def test_ten(self):
        self.assertEqual(fmt_price(10.50), "$10.5")


class TestFmtTokens(unittest.TestCase):
    def test_small(self):
        self.assertEqual(fmt_tokens(473), "473")

    def test_thousands(self):
        self.assertEqual(fmt_tokens(45200), "45.2K")

    def test_millions(self):
        self.assertEqual(fmt_tokens(1500000), "1.5M")

    def test_zero(self):
        self.assertEqual(fmt_tokens(0), "0")

    def test_exact_thousand(self):
        self.assertEqual(fmt_tokens(1000), "1.0K")

    def test_exact_million(self):
        self.assertEqual(fmt_tokens(1000000), "1.0M")

    def test_large_millions(self):
        self.assertEqual(fmt_tokens(150_000_000), "150M")

    def test_large_thousands(self):
        self.assertEqual(fmt_tokens(150_000), "150K")


class TestVisibleLen(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(visible_len("hello"), 5)

    def test_ansi(self):
        self.assertEqual(visible_len("\033[31mhello\033[0m"), 5)

    def test_empty(self):
        self.assertEqual(visible_len(""), 0)


if __name__ == "__main__":
    unittest.main()

"""Free-tier rules parsing, matching, and Gemini cache management."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .util import read_response

GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing.md.txt"
GEMINI_CACHE_MAX_AGE = 7 * 86400  # 7 days

# Models that are always paid (media generation, never on free tier)
_GEMINI_ALWAYS_PAID_PREFIXES = ("imagen-", "veo-", "lyria-")
# Models that are always free (open-source)
_GEMINI_ALWAYS_FREE_PREFIXES = ("gemma-",)


class FreeTierRule:
    """A provider free-tier rule: include patterns, exclude patterns, or all."""
    __slots__ = ("provider", "includes", "excludes")

    def __init__(self, provider: str, includes: list[str] | None, excludes: list[str]):
        self.provider = provider
        self.includes = includes  # None = all models included
        self.excludes = excludes  # models matching these are NOT free

    def matches(self, model_id: str) -> bool:
        mid = model_id.lower()
        if self.excludes and any(p in mid for p in self.excludes):
            return False
        if self.includes is None:
            return True
        return any(p in mid for p in self.includes)

    def __repr__(self) -> str:
        if self.includes is None and not self.excludes:
            return self.provider
        parts = []
        if self.includes:
            parts.extend(self.includes)
        if self.excludes:
            parts.extend(f"!{e}" for e in self.excludes)
        return f"{self.provider}:{','.join(parts)}"


def parse_free_tier_rules(text: str) -> dict[str, FreeTierRule]:
    """Parse free-tier rules from config file text.

    Format: one entry per line:
      groq                                           # all models free
      gemini:flash,gemma,!imagen,!veo                # include + exclude patterns
      gemini:!3.1-pro,!imagen,!veo                   # all free EXCEPT excludes
    """
    rules: dict[str, FreeTierRule] = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if ":" in line:
            provider, pat_str = line.split(":", 1)
            provider = provider.strip().lower()
            includes: list[str] = []
            excludes: list[str] = []
            for p in pat_str.split(","):
                p = p.strip().lower()
                if not p:
                    continue
                if p.startswith("!"):
                    excludes.append(p[1:])
                else:
                    includes.append(p)
            rules[provider] = FreeTierRule(provider, includes or None, excludes)
        else:
            rules[line.lower()] = FreeTierRule(line.lower(), None, [])
    return rules


def free_tier_rules_from_file(filepath: Path) -> dict[str, FreeTierRule]:
    """Read user-configured free-tier rules from config file."""
    if not filepath.exists():
        return {}
    return parse_free_tier_rules(filepath.read_text())


def load_gemini_cache(cache_file: Path) -> dict | None:
    """Load and return the Gemini free-tier cache if it exists and is fresh."""
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        fetched = data.get("fetched", "")
        if not fetched:
            return None
        ts = datetime.fromisoformat(fetched)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > GEMINI_CACHE_MAX_AGE:
            return None
        return data
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def gemini_cache_stale(cache_file: Path) -> bool:
    """Return True if the Gemini cache is missing or older than GEMINI_CACHE_MAX_AGE."""
    return load_gemini_cache(cache_file) is None


def fetch_gemini_free_tier() -> tuple[set[str], set[str]]:
    """Fetch Gemini pricing page and parse free/paid model sets.

    Returns (free_set, paid_set) of model ID strings.
    """
    req = Request(GEMINI_PRICING_URL, headers={"User-Agent": "llm-switchboard/0.1"})
    try:
        with urlopen(req, timeout=15) as resp:
            text = read_response(resp).decode("utf-8")
    except (URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"Failed to fetch Gemini pricing: {e}") from e

    free_set: set[str] = set()
    paid_set: set[str] = set()

    sections = re.split(r'^## ', text, flags=re.MULTILINE)

    for section in sections:
        header_lines = '\n'.join(section.split('\n')[:5])
        model_ids = re.findall(r'`([^`]+)`', header_lines)
        if not model_ids:
            continue

        input_match = re.search(
            r'\|\s*Input price[^|]*\|\s*([^|]+)\|',
            section
        )
        if not input_match:
            continue

        free_col = input_match.group(1).strip()
        is_free = "free of charge" in free_col.lower()
        for model_id in model_ids:
            if not re.match(r'^[a-z]', model_id):
                continue
            if is_free:
                free_set.add(model_id)
            else:
                paid_set.add(model_id)

    return free_set, paid_set


def write_gemini_cache(cache_file: Path, free_set: set[str], paid_set: set[str]) -> None:
    """Write Gemini free-tier cache to disk."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fetched": datetime.now(timezone.utc).isoformat(),
        "free": sorted(free_set),
        "paid": sorted(paid_set),
    }
    cache_file.write_text(json.dumps(data, indent=2) + "\n")


def is_free_tier(provider: str, model_id: str, rules: dict[str, FreeTierRule],
                 gemini_cache: dict | None = None) -> bool:
    """Check if a model is free based on cache (for Gemini) or user-configured rules."""
    if provider == "gemini" and gemini_cache is not None:
        free_set = gemini_cache.get("_free_set")
        paid_set = gemini_cache.get("_paid_set")
        if free_set is None:
            free_set = {m.lower() for m in gemini_cache.get("free", [])}
            gemini_cache["_free_set"] = free_set
        if paid_set is None:
            paid_set = {m.lower() for m in gemini_cache.get("paid", [])}
            gemini_cache["_paid_set"] = paid_set

        mid = model_id.lower()
        base_id = mid.split("/")[-1] if "/" in mid else mid

        if base_id in free_set:
            return True
        if base_id in paid_set:
            return False

        if base_id.endswith("-latest"):
            base = base_id[:-7]
            if base in free_set:
                return True
            if base in paid_set:
                return False

        if any(base_id.startswith(p) for p in _GEMINI_ALWAYS_FREE_PREFIXES):
            return True
        if any(base_id.startswith(p) for p in _GEMINI_ALWAYS_PAID_PREFIXES):
            return False

    rule = rules.get(provider)
    return rule.matches(model_id) if rule else False


# ─── Known free-tier providers ────────────────────────────────────────

KNOWN_FREE_TIERS: dict[str, tuple[str, str, str]] = {
    "groq": (
        "All models free with rate limits",
        "groq",
        "# Groq: all models free with rate limits (30 RPM, 6000 RPD)\n"
        "# https://console.groq.com/docs/rate-limits",
    ),
    "cerebras": (
        "All models free (1M tokens/day)",
        "cerebras",
        "# Cerebras: all models free (1M tokens/day, 30 RPM)\n"
        "# https://inference-docs.cerebras.ai/api-reference/rate-limits",
    ),
    "gemini": (
        "Free tier for most models (excludes 3.1-Pro, Imagen, Veo, etc.)",
        "gemini:!3.1-pro,!3-pro,!preview-tts,!native-audio,!computer-use,!deep-research,!imagen,!veo,!image-generation,!live-001,!lyria",
        "# Gemini: free tier for most text/code models. Paid-only models excluded.\n"
        "# IMPORTANT: Gemini billing is per-project, not per-account. If your API\n"
        "# key is from a project with Cloud Billing enabled AND prepaid, ALL usage\n"
        "# on that key is billed — there is no \"free quota then pay for overages.\"\n"
        "# To keep free access alongside paid, use two API keys (one per project).\n"
        "# https://ai.google.dev/gemini-api/docs/billing\n"
        "# https://ai.google.dev/gemini-api/docs/pricing",
    ),
    "mistral": (
        "All models free on Experiment plan (rate-limited)",
        "mistral",
        "# Mistral: all models free on Experiment plan (rate-limited)\n"
        "# https://docs.mistral.ai/deployment/ai-studio/tier",
    ),
}

DEFAULT_FREE_PROVIDERS = ["cerebras", "groq", "mistral"]

CONFIG_HEADER = (
    "# Free-tier provider configuration for llm-switchboard\n"
    "# Run --setup to reconfigure, or edit this file directly.\n"
    "#\n"
    "# Format:\n"
    "#   provider              — all models from this provider are free\n"
    "#   provider:!pattern     — all models free EXCEPT those matching pattern\n"
    "#   provider:pat1,pat2    — only models matching patterns are free\n"
    "#\n"
    "# Patterns are substring matches against the model ID (case-insensitive).\n"
    "# Prefix with ! to exclude.\n"
)


def write_free_tier_config(filepath: Path, rules: dict) -> None:
    """Write the free-tier config file with header and per-provider comments."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for prov in sorted(rules):
        val = rules[prov]
        config_line = str(val) if isinstance(val, FreeTierRule) else val
        known = KNOWN_FREE_TIERS.get(prov)
        if known:
            entries.append(f"{known[2]}\n{config_line}")
        else:
            entries.append(config_line)
    if entries:
        filepath.write_text(CONFIG_HEADER + "\n" + "\n\n".join(entries) + "\n")
    else:
        filepath.write_text("")

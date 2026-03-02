"""Model data structures and provider detection."""

import re
from typing import NamedTuple

DOMAIN_TO_PROVIDER = {
    "generativelanguage": "gemini",
    "mistral": "mistral",
    "groq": "groq",
    "cerebras": "cerebras",
    "openrouter": "openrouter",
    "x": "xai",
    "perplexity": "perplexity",
    "openai": "openai",
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "together": "together",
    "fireworks": "fireworks",
    "cohere": "cohere",
}

PROVIDER_COLORS_MAP = {
    "gemini": "YELLOW",
    "mistral": "BLUE",
    "groq": "RED",
    "cerebras": "MAGENTA",
    "openrouter": "CYAN",
    "xai": "WHITE",
    "perplexity": "BOLD_BLUE",
    "openai": "BOLD_GREEN",
    "anthropic": "BOLD_MAGENTA",
    "deepseek": "BOLD_CYAN",
    "ollama": "GREEN",
}

PROVIDER_LIMITS: dict[str, str] = {
    "groq": "30 RPM · 6,000 RPD",
    "cerebras": "30 RPM · 1M tokens/day",
    "gemini": "Free tier (per-project billing)",
    "mistral": "Free tier (Le Chat plan)",
}


class Model(NamedTuple):
    provider: str
    id: str
    cost: str  # "free", "local", "paid", "cloud"
    description: str = ""
    ctx_k: str = ""      # e.g. "128K", "1M"
    price: str = ""      # e.g. "$3/$15"


def provider_from_url(url: str) -> str:
    """Extract provider name from an API base URL."""
    m = re.search(r"://(?:api\.)?([^./]+)", url)
    if m:
        domain = m.group(1)
        return DOMAIN_TO_PROVIDER.get(domain, domain)
    return "external"


def format_ctx(ctx_raw) -> str:
    """Format a raw context length value to a human-readable string."""
    if not ctx_raw or not isinstance(ctx_raw, (int, float)) or ctx_raw <= 0:
        return ""
    n = int(ctx_raw)
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)

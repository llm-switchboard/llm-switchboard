"""Endpoint resolution — auto-detect and cache API mode + endpoints."""

import json
import time
from pathlib import Path
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import (
    API_MODE_FILE,
    CONFIG_DIR,
    DETECT_CACHE,
    OPENWEBUI_KEY,
    OPENWEBUI_URL,
)
from .util import read_response

# ─── Types ────────────────────────────────────────────────────────────

VALID_MODES = {"auto", "openai", "anthropic"}
VALID_PREFERENCES = {"openai", "anthropic"}

DETECT_TTL = 3600  # 1 hour

AUTO_PREFER_FILE = CONFIG_DIR / "auto_prefer.conf"


class Endpoint(NamedTuple):
    base_url: str  # normalized, no trailing slash
    mode: str  # "openai" or "anthropic"
    chat_path: str  # e.g. "/v1/chat/completions"
    models_path: str  # e.g. "/v1/models" or "/api/models"
    source: str = ""  # "cli", "config", "auto-detect"
    probe_match: str = ""  # which probe matched (for auto-detect)


# ─── URL Normalization ────────────────────────────────────────────────


def normalize_url(url: str) -> str:
    """Normalize a base URL: strip trailing slashes and whitespace."""
    return url.strip().rstrip("/")


# ─── Config Files ─────────────────────────────────────────────────────


def load_api_mode(api_mode_file: Path | None = None) -> str:
    """Load API mode from config file. Returns 'auto' if not set."""
    f = api_mode_file or API_MODE_FILE
    if not f.exists():
        return "auto"
    try:
        text = f.read_text().strip().lower()
        if text in VALID_MODES:
            return text
    except OSError:
        pass
    return "auto"


def save_api_mode(mode: str, api_mode_file: Path | None = None) -> None:
    """Write API mode to config file."""
    f = api_mode_file or API_MODE_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(mode + "\n")


def load_auto_prefer(prefer_file: Path | None = None) -> str:
    """Load auto-detect preference. Returns 'openai' if not set."""
    f = prefer_file or AUTO_PREFER_FILE
    if not f.exists():
        return "openai"
    try:
        text = f.read_text().strip().lower()
        if text in VALID_PREFERENCES:
            return text
    except OSError:
        pass
    return "openai"


def save_auto_prefer(prefer: str, prefer_file: Path | None = None) -> None:
    """Write auto-detect preference to config file."""
    f = prefer_file or AUTO_PREFER_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(prefer + "\n")


# ─── Probing ──────────────────────────────────────────────────────────


def _is_json_response(data: bytes) -> bool:
    """Check if data looks like JSON (not HTML or redirect page)."""
    stripped = data.lstrip()
    if not stripped:
        return False
    # JSON starts with { or [
    return stripped[:1] in (b"{", b"[")


def _has_json_content_type(headers) -> bool:
    """Check if HTTP headers indicate a JSON response."""
    ct = ""
    if hasattr(headers, "get_content_type"):
        ct = headers.get_content_type() or ""
    elif hasattr(headers, "get"):
        ct = headers.get("Content-Type", "") or ""
    return "application/json" in ct.lower()


def _probe_get(url: str, api_key: str, timeout: int = 5) -> dict | None:
    """Try GET on a URL. Returns parsed JSON dict or None.

    Returns None if: connection fails, response is HTML, response
    isn't valid JSON, or response isn't a dict.
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = read_response(resp)
            if not _is_json_response(raw):
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError):
        return None


def _probe_post_reachable(url: str, api_key: str, payload: dict, timeout: int = 5) -> bool:
    """Check if a POST endpoint is reachable (exists and speaks JSON).

    Returns True if:
    - HTTP 200 with JSON body, OR
    - HTTP 400/401/403/404/422 with JSON Content-Type header
      (endpoint exists, just rejected the request)

    Returns False if:
    - Connection error / timeout
    - Response is HTML or non-JSON
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode()
    req = Request(url, data=body, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            # 200 — check body or content-type
            if _has_json_content_type(resp.headers):
                return True
            raw = read_response(resp)
            return _is_json_response(raw)
    except HTTPError as e:
        if e.code in (400, 401, 403, 404, 422):
            # Endpoint exists — check Content-Type header first
            if _has_json_content_type(e.headers):
                return True
            # Fallback: read body and check
            try:
                raw = e.read(64 * 1024)
                return _is_json_response(raw)
            except Exception:
                pass
        return False
    except (URLError, TimeoutError, OSError):
        return False


def _probe_post(url: str, api_key: str, payload: dict, timeout: int = 5) -> dict | None:
    """Try POST on a URL. Returns parsed JSON dict or None.

    Used for probes that need the response body (e.g., anthropic).
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode()
    req = Request(url, data=body, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = read_response(resp)
            if not _is_json_response(raw):
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
    except HTTPError as e:
        if e.code in (400, 401, 403, 404, 422):
            try:
                raw = e.read(64 * 1024)
                if _is_json_response(raw):
                    data = json.loads(raw)
                    return data if isinstance(data, dict) else None
            except Exception:
                pass
        return None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError):
        return None


def _looks_like_models_list(data: dict) -> bool:
    """Check if response looks like an OpenAI /v1/models or /api/models response.

    Must have a "data" key with a list value (OpenAI standard).
    """
    return "data" in data and isinstance(data["data"], list)


def _extract_model_id(data: dict) -> str | None:
    """Extract the first model ID from a models-list response."""
    items = data.get("data", [])
    if not isinstance(items, list) or not items:
        return None
    for item in items:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name")
            if mid and isinstance(mid, str):
                return mid  # type: ignore[no-any-return]
    return None


# ─── Detection Cache ─────────────────────────────────────────────────


def _load_detect_cache(cache_file: Path | None = None) -> dict | None:
    """Load cached detection result if still valid."""
    f = cache_file or DETECT_CACHE
    if not f.exists():
        return None
    try:
        age = time.time() - f.stat().st_mtime
        if age >= DETECT_TTL:
            return None
        data = json.loads(f.read_text())
        if isinstance(data, dict) and "mode" in data and "base_url" in data:
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _save_detect_cache(result: dict, cache_file: Path | None = None) -> None:
    """Cache detection result."""
    f = cache_file or DETECT_CACHE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(result) + "\n")


# ─── Resolution ──────────────────────────────────────────────────────


def _build_chat_probe_payload(model_id: str | None = None) -> dict:
    """Build a minimal chat probe payload using a real model ID if available."""
    return {
        "model": model_id or "__probe__",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }


_ANTHROPIC_PROBE_PAYLOAD = {
    "model": "__probe__",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 1,
}


def _fetch_any_model_id(base_url: str, api_key: str) -> tuple[str | None, str]:
    """Try to fetch a valid model ID from models endpoints.

    Returns (model_id_or_None, models_path_that_worked).
    """
    for path in ("/v1/models", "/api/models"):
        resp = _probe_get(f"{base_url}{path}", api_key)
        if resp is not None and _looks_like_models_list(resp):
            mid = _extract_model_id(resp)
            if mid:
                return mid, path
    return None, "/api/models"


def _probe_openai_chat(base_url: str, api_key: str, model_id: str | None = None) -> str | None:
    """Probe for an OpenAI-compatible chat endpoint.

    Uses a real model ID if available to avoid false negatives from
    servers that reject unknown model names.

    Returns the working chat path or None if neither responds.
    """
    payload = _build_chat_probe_payload(model_id)
    # Prefer /v1/chat/completions
    if _probe_post_reachable(f"{base_url}/v1/chat/completions", api_key, payload, timeout=5):
        return "/v1/chat/completions"
    # Fallback to /api/chat/completions (Open WebUI style)
    if _probe_post_reachable(f"{base_url}/api/chat/completions", api_key, payload, timeout=5):
        return "/api/chat/completions"
    return None


def _probe_anthropic_chat(base_url: str, api_key: str) -> bool:
    """Check if the Anthropic Messages endpoint is reachable."""
    return _probe_post_reachable(
        f"{base_url}/api/v1/messages",
        api_key,
        _ANTHROPIC_PROBE_PAYLOAD,
        timeout=5,
    )


def _detect_models_path(base_url: str, api_key: str) -> str:
    """Determine the best models endpoint."""
    resp = _probe_get(f"{base_url}/v1/models", api_key)
    if resp is not None and _looks_like_models_list(resp):
        return "/v1/models"
    resp = _probe_get(f"{base_url}/api/models", api_key)
    if resp is not None and _looks_like_models_list(resp):
        return "/api/models"
    return "/api/models"  # fallback


def _detect_endpoint(base_url: str, api_key: str, prefer: str = "openai") -> Endpoint:
    """Auto-detect API mode by probing endpoints.

    Strategy:
    1. Fetch a real model ID from models endpoints (avoids false
       negatives when servers reject fake model names).
    2. Probe chat endpoints with the real model ID.

    When prefer="openai" (default):
      a. Probe OpenAI chat endpoints → if reachable, choose openai
      b. Probe /api/v1/messages → if reachable, choose anthropic
      c. Models-list fallback → assume openai
      d. Final fallback → openai + /api/chat/completions

    When prefer="anthropic":
      a. Probe /api/v1/messages first → if reachable, choose anthropic
      b. Fall back to OpenAI probes as above
    """
    # ── Step 0: fetch a real model ID ──────────────────────────
    model_id, discovered_models_path = _fetch_any_model_id(base_url, api_key)

    if prefer == "anthropic":
        # Anthropic-first ordering
        if _probe_anthropic_chat(base_url, api_key):
            models = discovered_models_path if model_id else _detect_models_path(base_url, api_key)
            return Endpoint(base_url, "anthropic", "/api/v1/messages", models, "auto-detect", "POST /api/v1/messages")

    # ── OpenAI chat probes (with real model ID) ───────────────
    chat_path = _probe_openai_chat(base_url, api_key, model_id)
    if chat_path is not None:
        models = discovered_models_path if model_id else _detect_models_path(base_url, api_key)
        return Endpoint(base_url, "openai", chat_path, models, "auto-detect", f"POST {chat_path} (model probe)")

    # ── Anthropic probe (if not already tried) ────────────────
    if prefer != "anthropic":
        if _probe_anthropic_chat(base_url, api_key):
            models = discovered_models_path if model_id else _detect_models_path(base_url, api_key)
            return Endpoint(base_url, "anthropic", "/api/v1/messages", models, "auto-detect", "POST /api/v1/messages")

    # ── Models-list fallback ──────────────────────────────────
    if model_id:
        return Endpoint(
            base_url,
            "openai",
            "/v1/chat/completions" if "/v1/" in discovered_models_path else "/api/chat/completions",
            discovered_models_path,
            "auto-detect",
            f"GET {discovered_models_path} (models-list fallback)",
        )

    # ── Nothing worked ────────────────────────────────────────
    return Endpoint(base_url, "openai", "/api/chat/completions", "/api/models", "auto-detect", "none (fallback)")


def resolve_endpoint(
    mode: str = "auto",
    base_url: str | None = None,
    api_key: str | None = None,
    cache_file: Path | None = None,
    skip_cache: bool = False,
    prefer: str | None = None,
    source: str = "",
) -> Endpoint:
    """Resolve the API endpoint to use.

    Args:
        mode: "auto", "openai", or "anthropic"
        base_url: Override base URL (default: OPENWEBUI_URL)
        api_key: Override API key (default: OPENWEBUI_KEY)
        cache_file: Override detection cache path
        skip_cache: Force re-detection
        prefer: Auto-detect preference ("openai" or "anthropic")
        source: Where the mode came from ("cli", "config", or auto-filled)
    """
    url = normalize_url(base_url or OPENWEBUI_URL)
    key = api_key if api_key is not None else OPENWEBUI_KEY

    if mode == "openai":
        src = source or "cli"
        if not skip_cache:
            cached = _load_detect_cache(cache_file)
            if cached and cached.get("base_url") == url and cached.get("mode") == "openai":
                return Endpoint(url, "openai", cached["chat_path"], cached["models_path"], src, cached.get("probe_match", ""))
        resp = _probe_get(f"{url}/v1/models", key)
        if resp is not None and _looks_like_models_list(resp):
            ep = Endpoint(url, "openai", "/v1/chat/completions", "/v1/models", src, "GET /v1/models")
        else:
            ep = Endpoint(url, "openai", "/api/chat/completions", "/api/models", src, "fallback")
        _save_detect_cache(ep._asdict(), cache_file)
        return ep

    if mode == "anthropic":
        src = source or "cli"
        ep = Endpoint(url, "anthropic", "/api/v1/messages", "/api/models", src, "")
        _save_detect_cache(ep._asdict(), cache_file)
        return ep

    # Auto mode
    if not skip_cache:
        cached = _load_detect_cache(cache_file)
        if cached and cached.get("base_url") == url:
            return Endpoint(
                cached["base_url"],
                cached["mode"],
                cached["chat_path"],
                cached["models_path"],
                cached.get("source", "auto-detect"),
                cached.get("probe_match", ""),
            )

    pref = prefer or load_auto_prefer()
    ep = _detect_endpoint(url, key, prefer=pref)
    _save_detect_cache(ep._asdict(), cache_file)
    return ep


# ─── Display Helper ──────────────────────────────────────────────────


def format_endpoint_info(ep: Endpoint) -> str:
    """Format endpoint info for --print-endpoints."""
    lines = [
        f"Base URL:        {ep.base_url}",
        f"API mode:        {ep.mode}",
        f"Chat endpoint:   {ep.base_url}{ep.chat_path}",
        f"Models endpoint: {ep.base_url}{ep.models_path}",
    ]
    if ep.source:
        lines.append(f"Mode source:     {ep.source}")
    if ep.probe_match:
        lines.append(f"Probe matched:   {ep.probe_match}")
    return "\n".join(lines)

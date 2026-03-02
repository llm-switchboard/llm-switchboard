"""Configuration management — favorites, last model, directory setup."""

import os
from pathlib import Path

# ─── App Name ────────────────────────────────────────────────────────

APP_NAME = "llm-switchboard"

# ─── Directory Paths ─────────────────────────────────────────────────


def _xdg_base(env_var: str, default_subdir: str) -> Path:
    return Path(os.environ.get(env_var, Path.home() / default_subdir))


CONFIG_BASE = _xdg_base("XDG_CONFIG_HOME", ".config")
CACHE_BASE = _xdg_base("XDG_CACHE_HOME", ".cache")

CONFIG_DIR = CONFIG_BASE / APP_NAME
CACHE_DIR = CACHE_BASE / APP_NAME

# Derived paths
CONN_CACHE = CACHE_DIR / "connections.json"
OLLAMA_CACHE = CACHE_DIR / "ollama_config.json"
FAV_FILE = CONFIG_DIR / "favorites.conf"
FREE_PROVIDERS_FILE = CONFIG_DIR / "free_providers.conf"
LAST_FILE = CONFIG_DIR / "last_model"
USAGE_FILE = CONFIG_DIR / "usage.json"
GEMINI_FREE_CACHE = CONFIG_DIR / "gemini_free_cache.json"
COMPAT_FILE = CONFIG_DIR / "compat.json"
API_MODE_FILE = CONFIG_DIR / "api_mode.conf"
AUTO_PREFER_FILE = CONFIG_DIR / "auto_prefer.conf"
DETECT_CACHE = CACHE_DIR / "endpoint_detect.json"

OPENWEBUI_URL = os.environ.get("OPENWEBUI_URL", "http://127.0.0.1:3100").rstrip("/")
OPENWEBUI_KEY = os.environ.get("OPENWEBUI_API_KEY", "")


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not FAV_FILE.exists():
        FAV_FILE.touch()


# ─── Favorites ────────────────────────────────────────────────────────

def fav_list(fav_file: Path | None = None) -> list[str]:
    f = fav_file or FAV_FILE
    if not f.exists():
        return []
    return [line for line in f.read_text().splitlines() if line.strip()]


def fav_has(model_id: str, fav_file: Path | None = None) -> bool:
    return model_id in fav_list(fav_file)


def fav_add(model_id: str, fav_file: Path | None = None) -> str:
    """Add a favorite. Returns status message."""
    f = fav_file or FAV_FILE
    if not fav_file:
        ensure_dirs()
    if fav_has(model_id, f):
        return f"Already a favorite: {model_id}"
    with f.open("a") as fp:
        fp.write(model_id + "\n")
    return f"Added favorite: {model_id}"


def fav_rm(model_id: str, fav_file: Path | None = None) -> str:
    """Remove a favorite. Returns status message."""
    f = fav_file or FAV_FILE
    if fav_has(model_id, f):
        lines = [line for line in f.read_text().splitlines() if line.strip() and line.strip() != model_id]
        f.write_text("\n".join(lines) + "\n" if lines else "")
        return f"Removed favorite: {model_id}"
    return f"Not a favorite: {model_id}"


# ─── Last Model ───────────────────────────────────────────────────────

def last_model_read(last_file: Path | None = None) -> str:
    f = last_file or LAST_FILE
    if f.exists():
        return f.read_text().strip()
    return ""


def last_model_write(model_id: str, last_file: Path | None = None) -> None:
    f = last_file or LAST_FILE
    if not last_file:
        ensure_dirs()
    f.write_text(model_id)

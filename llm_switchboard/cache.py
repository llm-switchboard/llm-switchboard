"""Cache management for API responses."""

import json
import time
from pathlib import Path

CACHE_MAX_AGE = 3600  # 1 hour


def fetch_cached(api_get_fn, path: str, cache_file: Path, validate_key: str, max_age: int = CACHE_MAX_AGE) -> dict:
    """Fetch data from API with file-based caching.

    Args:
        api_get_fn: callable that takes (path, timeout) and returns dict
        path: API path to fetch
        cache_file: where to cache the response
        validate_key: key that must exist in response for cache write
        max_age: cache TTL in seconds
    """
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < max_age:
            try:
                data = json.loads(cache_file.read_text())
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
    data = api_get_fn(path, 10)
    if validate_key in data:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    return data  # type: ignore[no-any-return]

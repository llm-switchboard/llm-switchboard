"""Utility functions — pure logic, no side effects."""

import contextlib
import fcntl
import os
import re
import sys
from pathlib import Path
from typing import NoReturn

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB cap for HTTP responses

PRIVATE_IP_RE = re.compile(r"://(localhost|127\.|0\.0\.0\.0|10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.|\[::1\])")


def sanitize_display(s: str) -> str:
    """Strip ANSI escape sequences and control characters from untrusted strings."""
    return _ANSI_RE.sub("", s).replace("\r", "").replace("\x00", "")


def visible_len(s: str) -> int:
    """Length of string after stripping ANSI escape sequences."""
    return len(_ANSI_RE.sub("", s))


def read_response(resp, max_size: int = MAX_RESPONSE_SIZE) -> bytes:
    """Read HTTP response with size cap to prevent memory exhaustion."""
    data = resp.read(max_size + 1)
    if len(data) > max_size:
        raise RuntimeError(f"Response too large (>{max_size // 1024 // 1024} MB)")
    return data  # type: ignore[no-any-return]


def fmt_price(v: float) -> str:
    """Format a price-per-million-tokens value for display."""
    if v == 0:
        return "$0"
    if v >= 100:
        return f"${v:.0f}"
    if v >= 10:
        return f"${v:.1f}".rstrip("0").rstrip(".")
    return f"${v:.2f}".rstrip("0").rstrip(".")


def fmt_tokens(n: int) -> str:
    """Format token counts readably: 473 → '473', 45200 → '45.2K', 1500000 → '1.5M'."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}M" if v < 100 else f"{v:.0f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}K" if v < 100 else f"{v:.0f}K"
    return str(n)


# ─── Color Support ────────────────────────────────────────────────────

IS_TTY = sys.stdout.isatty()
IS_STDERR_TTY = sys.stderr.isatty()
IS_STDIN_TTY = sys.stdin.isatty()
USE_COLOR = IS_TTY and not os.environ.get("NO_COLOR")


def _c(code: str) -> str:
    return code if USE_COLOR else ""


CYAN = _c("\033[0;36m")
GREEN = _c("\033[0;32m")
YELLOW = _c("\033[0;33m")
MAGENTA = _c("\033[0;35m")
BLUE = _c("\033[0;34m")
RED = _c("\033[0;31m")
WHITE = _c("\033[1;37m")
DIM = _c("\033[2m")
BOLD = _c("\033[1m")
RESET = _c("\033[0m")
REVERSE = _c("\033[7m")


def die(msg: str) -> NoReturn:
    print(f"{RED}Error:{RESET} {msg}", file=sys.stderr)
    sys.exit(1)


# ─── File Locking ────────────────────────────────────────────────────


@contextlib.contextmanager
def locked_file(filepath: Path):
    """Context manager that holds an exclusive lock on filepath.lock.

    Use around read-modify-write cycles to prevent concurrent updates
    from clobbering each other.
    """
    lockpath = filepath.with_suffix(filepath.suffix + ".lock")
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    fp = open(lockpath, "w")
    try:
        fcntl.flock(fp, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fp, fcntl.LOCK_UN)
        fp.close()

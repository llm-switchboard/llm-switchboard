"""Terminal UI — raw input, spinner, screen rendering, interactive loop.

This module contains all TTY-dependent code: raw-mode key reading,
screen clearing, spinner animation, and the interactive model picker loop.
"""

import atexit
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
import time
import tty

from .config import (
    OPENWEBUI_KEY,
    fav_list,
    last_model_read,
    last_model_write,
)
from .models import PROVIDER_LIMITS, Model
from .session import SessionWatcher, claude_session_dir, record_session
from .util import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    IS_STDERR_TTY,
    IS_STDIN_TTY,
    IS_TTY,
    MAGENTA,
    RED,
    RESET,
    REVERSE,
    WHITE,
    YELLOW,
    die,
    fmt_tokens,
    visible_len,
)

# Re-export for the monolith shim
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# ─── Global Model State ──────────────────────────────────────────────
# Filled by fetch_models() in cli.py, referenced here for rendering.

MODELS: list[Model] = []
MODEL_MAP: dict[str, Model] = {}
PROVIDERS: dict[str, int] = {}
PROVIDER_FREE: dict[str, int] = {}
PROVIDER_LOCAL: dict[str, int] = {}
PROVIDER_CLOUD: dict[str, int] = {}
PROVIDER_KEYS: list[str] = []

_COMPAT_DATA: dict | None = None


def _get_compat_data() -> dict:
    """Load compat data once, cache for session lifetime."""
    global _COMPAT_DATA
    if _COMPAT_DATA is None:
        from .compat import load_compat

        _COMPAT_DATA = load_compat()
    return _COMPAT_DATA


PROVIDER_COLORS = {
    "gemini": YELLOW,
    "mistral": BLUE,
    "groq": RED,
    "cerebras": MAGENTA,
    "openrouter": CYAN,
    "xai": WHITE,
    "perplexity": f"{BOLD}{BLUE}",
    "openai": f"{BOLD}{GREEN}",
    "anthropic": f"{BOLD}{MAGENTA}",
    "deepseek": f"{BOLD}{CYAN}",
    "ollama": GREEN,
}


def get_color(provider: str) -> str:
    return PROVIDER_COLORS.get(provider, DIM)


# ─── Terminal Raw Input ──────────────────────────────────────────────

_original_termios = None


def _save_termios() -> None:
    global _original_termios
    if IS_STDIN_TTY:
        try:
            _original_termios = termios.tcgetattr(sys.stdin)
        except termios.error:
            pass


def _restore_termios() -> None:
    if _original_termios is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_termios)
        except termios.error:
            pass


def _setup_terminal_restore() -> None:
    """Register cleanup so terminal is always restored."""
    _save_termios()
    atexit.register(_restore_termios)
    for sig in (signal.SIGINT, signal.SIGTERM):
        prev = signal.getsignal(sig)

        def handler(s, f, _prev=prev, _sig=sig):
            _restore_termios()
            if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                _prev(s, f)
            elif _prev == signal.SIG_DFL:
                signal.signal(_sig, signal.SIG_DFL)
                os.kill(os.getpid(), _sig)
            else:
                sys.exit(128 + _sig)

        signal.signal(sig, handler)


_CSI_MAP = {
    "A": "UP",
    "B": "DOWN",
    "C": "RIGHT",
    "D": "LEFT",
    "H": "HOME",
    "F": "END",
}
_CSI_TILDE_MAP = {
    "5": "PGUP",
    "6": "PGDN",
    "1": "HOME",
    "4": "END",
}


def _read1(fd: int) -> str:
    try:
        b = os.read(fd, 1)
        return b.decode("latin-1") if b else ""
    except OSError:
        return ""


def _has_input(fd: int, timeout: float) -> bool:
    r, _, _ = select.select([fd], [], [], timeout)
    return bool(r)


def readkey() -> str:
    """Read a single keypress in raw mode."""
    if not IS_STDIN_TTY:
        line = sys.stdin.readline()
        return line.strip() if line else "EOF"
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = _read1(fd)
        if not ch:
            return "EOF"
        if ch == "\x1b":
            if not _has_input(fd, 0.05):
                return "ESC"
            ch2 = _read1(fd)
            if ch2 == "[":
                buf = ""
                while _has_input(fd, 0.02):
                    c = _read1(fd)
                    if c.isalpha() or c == "~":
                        if c == "~":
                            return _CSI_TILDE_MAP.get(buf, "ESC")
                        return _CSI_MAP.get(c, "ESC")
                    buf += c
                return "ESC"
            elif ch2 == "O":
                if _has_input(fd, 0.02):
                    c = _read1(fd)
                    return _CSI_MAP.get(c, "ESC")
                return "ESC"
            return "ESC"
        if ch == "\x03":
            return "CTRLC"
        if ch == "\x04":
            return "EOF"
        if ch == "\r" or ch == "\n":
            return "ENTER"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def readkey_with_digit_timeout(first_digit: str, max_items: int) -> int:
    if max_items <= 9:
        return int(first_digit)
    if not IS_STDIN_TTY:
        return int(first_digit)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if _has_input(fd, 0.3):
            ch = _read1(fd)
            if ch.isdigit():
                return int(first_digit + ch)
        return int(first_digit)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _check_dotdot() -> bool:
    if not IS_STDIN_TTY:
        return False
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if _has_input(fd, 0.2):
            ch2 = _read1(fd)
            if ch2 == ".":
                return True
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def readline_input(prompt: str = "") -> str:
    _restore_termios()
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return ""


# ─── Spinner ─────────────────────────────────────────────────────────


class Spinner:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    INTERVAL = 0.08

    def __init__(self, message: str = "Loading..."):
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stderr.write(f"\r  {frame} {self.message}")
            sys.stderr.flush()
            self._stop.wait(self.INTERVAL)
            i += 1

    def __enter__(self) -> "Spinner":
        if IS_STDERR_TTY:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        if IS_STDERR_TTY:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()


# ─── Screen ──────────────────────────────────────────────────────────


def clear_screen() -> None:
    if IS_TTY:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


def _page_height() -> int:
    return max(5, shutil.get_terminal_size((80, 24)).lines - 6)


# ─── Model Metadata ─────────────────────────────────────────────────


def model_meta(model_id: str, max_desc: int = 0) -> str:
    m = MODEL_MAP.get(model_id)
    if not m:
        return ""
    parts: list[str] = []
    if m.ctx_k:
        parts.append(f"{CYAN}{m.ctx_k}{RESET}")
    if m.price:
        parts.append(f"{YELLOW}{m.price}{RESET}")
    elif m.cost == "local":
        parts.append(f"{BLUE}LOCAL{RESET}")
    elif m.cost == "free":
        parts.append(f"{GREEN}FREE{RESET}")
    elif m.cost == "cloud":
        parts.append(f"{MAGENTA}CLOUD{RESET}")
    desc = m.description
    if desc and max_desc >= 0:
        if max_desc > 0 and len(desc) > max_desc:
            if max_desc < 15:
                desc = ""
            else:
                desc = desc[: max_desc - 3].rsplit(" ", 1)[0] + "..."
        if desc:
            parts.append(f"{DIM}{desc}{RESET}")
    # AGENT badge from compat test results
    compat = _get_compat_data()
    entry = compat.get("results", {}).get(model_id)
    if entry:
        status = entry.get("last_status")
        if status == "pass":
            parts.append(f"{GREEN}AGENT \u2713{RESET}")
        elif status == "partial":
            parts.append(f"{YELLOW}AGENT \u26a0{RESET}")
        elif status == "fail":
            parts.append(f"{RED}AGENT \u2717{RESET}")
    if not parts:
        return ""
    return "  " + "  ".join(parts)


# ─── Cost Estimation ─────────────────────────────────────────────────


def _estimate_cost_raw(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    m = MODEL_MAP.get(model_id)
    if not m or not m.price:
        return None
    import re as _re

    match = _re.match(r"\$([0-9.]+)/\$([0-9.]+)", m.price)
    if not match:
        return None
    in_rate = float(match.group(1))
    out_rate = float(match.group(2))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> str | None:
    cost = _estimate_cost_raw(model_id, input_tokens, output_tokens)
    if cost is None:
        return None
    if cost < 0.005:
        return "~$0.00"
    return f"~${cost:.2f}"


# ─── Launch ──────────────────────────────────────────────────────────


def _launch_banner(model_id: str) -> None:
    m = MODEL_MAP.get(model_id)
    provider = m.provider if m else "unknown"
    color = get_color(provider)

    lines: list[str] = []
    lines.append(f"  {color}[{provider}]{RESET} {BOLD}{model_id}{RESET}")

    meta_parts: list[str] = []
    if m:
        if m.ctx_k:
            meta_parts.append(f"{CYAN}{m.ctx_k}{RESET}")
        if m.price:
            meta_parts.append(f"{YELLOW}{m.price}{RESET}")
        elif m.cost == "free":
            meta_parts.append(f"{GREEN}FREE{RESET}")
        elif m.cost == "local":
            meta_parts.append(f"{BLUE}LOCAL{RESET}")
        elif m.cost == "cloud":
            meta_parts.append(f"{MAGENTA}CLOUD{RESET}")
    if meta_parts:
        lines.append("  " + "  ".join(meta_parts))

    if m and m.cost == "free" and provider in PROVIDER_LIMITS:
        lines.append(f"  {DIM}Limits: {PROVIDER_LIMITS[provider]}{RESET}")

    content_width = max(visible_len(ln) for ln in lines)
    box_w = content_width + 2

    print()
    print(f"  {DIM}┌{'─' * box_w}┐{RESET}")
    for line in lines:
        pad = box_w - visible_len(line) - 1
        print(f"  {DIM}│{RESET}{line}{' ' * pad}{DIM}│{RESET}")
    print(f"  {DIM}└{'─' * box_w}┘{RESET}")


def _session_summary(model_id: str, session: dict, pause: bool = False) -> None:
    msgs = session.get("message_count", 0)
    if msgs <= 0:
        return
    m = MODEL_MAP.get(model_id)
    has_tokens = session.get("input_tokens", 0) > 0 or session.get("output_tokens", 0) > 0
    if has_tokens and m and m.price:
        cost_str = _estimate_cost(model_id, session["input_tokens"], session["output_tokens"])
        parts = f"{msgs} msgs · {fmt_tokens(session['input_tokens'])} in · {fmt_tokens(session['output_tokens'])} out"
        if cost_str:
            parts += f" · {cost_str}"
        tag = "paid"
    elif has_tokens:
        total = session["input_tokens"] + session["output_tokens"]
        parts = f"{msgs} msgs · {fmt_tokens(total)} tokens"
        tag = "free"
    else:
        parts = f"{msgs} msgs"
        tag = "free" if (not m or m.cost == "free") else "paid"
    print(f"\n  {DIM}Session: {parts}   ({tag}){RESET}")
    if pause and IS_STDIN_TTY:
        print(f"  {DIM}Press any key to continue...{RESET}", end="", flush=True)
        try:
            old = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin)
            sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        except Exception:
            time.sleep(2)
        print()


def launch_claude(model_id: str, extra_args: list[str], pause_after: bool = False) -> int:
    _restore_termios()
    if model_id.startswith("-"):
        die(f"Invalid model ID (starts with '-'): {model_id}")
    last_model_write(model_id)
    _launch_banner(model_id)

    session_dir = claude_session_dir()
    watcher = SessionWatcher(session_dir, model_id)
    watcher.start()

    # Resolve endpoint for Claude Code launch
    from .cli import _CLI_API_MODE, _CLI_AUTO_PREFER, _CLI_BASE_URL
    from .endpoint import load_api_mode, resolve_endpoint

    ep_mode = _CLI_API_MODE or load_api_mode()
    ep = resolve_endpoint(mode=ep_mode, base_url=_CLI_BASE_URL, prefer=_CLI_AUTO_PREFER)

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"{ep.base_url}/api"
    env["ANTHROPIC_AUTH_TOKEN"] = OPENWEBUI_KEY
    env["ANTHROPIC_API_KEY"] = ""

    prev_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        result = subprocess.run(["claude", "--model", model_id] + extra_args, env=env)
        if result.returncode != 0:
            print(f"\n{DIM}claude exited with code {result.returncode}{RESET}")
        try:
            session = watcher.stop()
            if session:
                record_session(model_id, session)
                _session_summary(model_id, session, pause=pause_after)
        except Exception:
            pass
        return result.returncode
    except OSError as e:
        die(f"Failed to launch claude: {e}")
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        global _COMPAT_DATA
        _COMPAT_DATA = None


# ─── fzf ─────────────────────────────────────────────────────────────


def has_fzf() -> bool:
    return shutil.which("fzf") is not None


def run_fzf_search(initial: str = "", provider: str = "") -> str | None:
    _restore_termios()
    lines = []
    for m in MODELS:
        if provider and m.provider != provider:
            continue
        parts = [f"[{m.provider}]".ljust(14), m.id]
        tags: list[str] = []
        if m.ctx_k:
            tags.append(m.ctx_k)
        if m.price:
            tags.append(m.price)
        elif m.cost in ("free", "local", "cloud"):
            tags.append(m.cost.upper())
        if m.description:
            tags.append(m.description)
        if tags:
            parts.append("  " + "  ".join(tags))
        lines.append(" ".join(parts))

    prompt = f"[{provider}] Model> " if provider else "Model> "
    cmd = [
        "fzf",
        "--ansi",
        f"--prompt={prompt}",
        f"--query={initial}",
        "--height=40%",
        "--reverse",
        "--no-mouse",
        "--layout=reverse-list",
    ]
    try:
        r = subprocess.run(cmd, input="\n".join(lines), capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if r.returncode != 0:
        return None

    sel = r.stdout.strip()
    idx = sel.find("] ")
    if idx < 0:
        return None
    rest = sel[idx + 2 :].strip()
    model_id = rest.split("  ")[0].strip()
    return model_id or None


# ─── Rendering ───────────────────────────────────────────────────────

MainItem = tuple


def _build_main_items() -> list[MainItem]:
    items: list[MainItem] = []
    last = last_model_read()
    if last:
        items.append(("last", last))
    for i, fid in enumerate(fav_list()):
        items.append(("fav", i, fid))
    for pname in PROVIDER_KEYS:
        items.append(("provider", pname))
    items.append(("free",))
    return items


def render_main(cursor: int = 0) -> tuple[list[MainItem], int]:
    clear_screen()
    items = _build_main_items()
    if items:
        cursor = max(0, min(cursor, len(items) - 1))
    else:
        cursor = 0

    print()
    idx = 0
    term_width = shutil.get_terminal_size((80, 24)).columns

    last = last_model_read()
    if last:
        m = MODEL_MAP.get(last)
        prov = m.provider if m else "unknown"
        c = get_color(prov)
        sel = f"{REVERSE} " if idx == cursor else "  "
        end = f" {RESET}" if idx == cursor else ""
        base_len = 6 + len(prov) + 2 + len(last)
        desc_budget = max(0, term_width - base_len - 24)
        meta = model_meta(last, max_desc=desc_budget if desc_budget > 10 else -1)
        print(f"{sel}Last: {c}[{prov}]{RESET} {BOLD}{last}{RESET}{meta}{end}")
        idx += 1
        print()

    favs = fav_list()
    if favs:
        print(f"  {YELLOW}★{RESET} {BOLD}Favorites:{RESET}")
        for n, fid in enumerate(favs, 1):
            m = MODEL_MAP.get(fid)
            sel = f"  {REVERSE} " if idx == cursor else "    "
            end = f" {RESET}" if idx == cursor else ""
            if m is None:
                print(f"{sel}{DIM}{n}) [???]        {fid} (unavailable){RESET}{end}")
            else:
                c = get_color(m.provider)
                base_len = 4 + 3 + 13 + 1 + len(fid)
                desc_budget = max(0, term_width - base_len - 24)
                fmeta = model_meta(fid, max_desc=desc_budget if desc_budget > 10 else -1)
                print(f"{sel}{n}) {c}{m.provider:<13s}{RESET} {fid}{fmeta}{end}")
            idx += 1
        print()

    print(f"  {BOLD}Providers:{RESET}")
    for i, pname in enumerate(PROVIDER_KEYS):
        ch = chr(97 + i) if i < 26 else "?"
        pc = get_color(pname)
        cnt = PROVIDERS[pname]
        fcnt = PROVIDER_FREE.get(pname, 0)
        lcnt = PROVIDER_LOCAL.get(pname, 0)
        label = "model" if cnt == 1 else "models"
        ccnt = PROVIDER_CLOUD.get(pname, 0)
        ci = ""
        if lcnt > 0 and ccnt > 0:
            ci = f"  {BLUE}{lcnt} local{RESET}  {MAGENTA}{ccnt} cloud{RESET}"
        elif lcnt == cnt and cnt > 0:
            ci = f"  {BLUE}local{RESET}"
        elif ccnt > 0:
            ci = f"  {MAGENTA}{ccnt} cloud{RESET}"
        elif fcnt == cnt and cnt > 0:
            ci = f"  {GREEN}all free{RESET}"
        elif fcnt > 0:
            ci = f"  {GREEN}{fcnt} free{RESET}"
        sel = f"  {REVERSE} " if idx == cursor else "    "
        end = f" {RESET}" if idx == cursor else ""
        print(f"{sel}{ch}) {pc}{pname:<13s}{RESET} {cnt:3d} {label}{ci}{end}")
        idx += 1

    print()
    sel = f"  {REVERSE} " if idx == cursor else "    "
    end = f" {RESET}" if idx == cursor else ""
    print(f"{sel}F) {BOLD}Free / no-cost models{RESET}{end}")

    print()
    hint = "↑↓/jk=move  Enter=select  1-9=fav  a-z=provider  F=free  /=search  q=quit"
    print(f"  {DIM}{hint}{RESET}")
    # Show API mode if not default
    from .cli import _CLI_API_MODE, _CLI_BASE_URL
    from .endpoint import load_api_mode

    api_mode = _CLI_API_MODE or load_api_mode()
    if api_mode != "auto" or _CLI_BASE_URL:
        mode_info = f"mode={api_mode}"
        if _CLI_BASE_URL:
            mode_info += f"  url={_CLI_BASE_URL}"
        print(f"  {DIM}API: {mode_info}{RESET}")
    print()
    return items, cursor


def _render_paged_list(items: list[tuple[str, str]], offset: int, cursor: int, header: str, hint: str, show_provider: bool = False) -> tuple[int, int]:
    clear_screen()
    page = _page_height()
    total = len(items)

    if total == 0:
        cursor = 0
    else:
        cursor = max(0, min(cursor, total - 1))

    if total <= page:
        offset = 0
    else:
        if cursor < offset:
            offset = cursor
        elif cursor >= offset + page:
            offset = cursor - page + 1
        offset = max(0, min(offset, total - page))

    end = min(offset + page, total)

    print(f"\n  {header}\n")

    if offset > 0:
        print(f"    {DIM}↑ {offset} more above{RESET}")

    term_width = shutil.get_terminal_size((80, 24)).columns

    for i in range(offset, end):
        label, mid = items[i]
        n = i + 1
        is_sel = i == cursor
        if is_sel:
            marker = f"  {REVERSE} "
        else:
            marker = "    "
        if show_provider:
            prefix = f"{marker}{DIM}{n:<3d}{RESET} {label} {mid}"
            visible_l = 4 + 3 + 1 + 13 + 1 + len(mid)
        else:
            prefix = f"{marker}{DIM}{n:<3d}{RESET} {mid}"
            visible_l = 4 + 3 + 1 + len(mid)
        remaining = term_width - visible_l - 4
        meta = model_meta(mid, max_desc=max(0, remaining - 20) if remaining > 30 else -1)
        line = f"{prefix}{meta}"
        if is_sel:
            line += f" {RESET}"
        print(line)

    remaining_items = total - end
    if remaining_items > 0:
        print(f"    {DIM}↓ {remaining_items} more below{RESET}")

    print(f"\n  {DIM}{hint}{RESET}\n")
    return offset, cursor


def render_provider(pname: str, offset: int = 0, cursor: int = 0, free_only: bool = False) -> tuple[list[str], int, int]:
    pc = get_color(pname)
    if free_only:
        items_raw = [m.id for m in MODELS if m.provider == pname and m.cost in ("free", "local")]
    else:
        items_raw = [m.id for m in MODELS if m.provider == pname]
    items = [("", mid) for mid in items_raw]
    free_tag = f"  {GREEN}free only{RESET}" if free_only else ""
    header = f"{pc}[{pname}]{RESET} {BOLD}{len(items)} models:{RESET}{free_tag}"
    hint = "#=select  ↑↓/jk=move  Enter=select  F=free  Esc/..=back  /=search  q=quit"
    offset, cursor = _render_paged_list(items, offset, cursor, header, hint)
    return items_raw, offset, cursor


def render_search(query: str, offset: int = 0, cursor: int = 0, provider: str = "") -> tuple[list[str], int, int]:
    ql = query.lower()
    results = [(m.provider, m.id) for m in MODELS if (ql in m.id.lower() or ql in m.provider.lower()) and (not provider or m.provider == provider)]
    if not results:
        clear_screen()
        print()
        print(f"  No models matching '{YELLOW}{query}{RESET}'.")
        print(f"\n  {DIM}Esc/..=back  /=new search  q=quit{RESET}\n")
        return [], 0, 0

    items = [(f"{get_color(p)}{'[' + p + ']':<13s}{RESET}", mid) for p, mid in results]
    scope = f" in {get_color(provider)}[{provider}]{RESET}" if provider else ""
    header = f"{BOLD}Search: {YELLOW}{query}{RESET}{scope}  {DIM}({len(results)} results){RESET}"
    hint = "#=select  ↑↓/jk=move  Enter=select  Esc/..=back  /=new search  q=quit"
    offset, cursor = _render_paged_list(items, offset, cursor, header, hint, show_provider=True)
    return [mid for _, mid in results], offset, cursor


def render_free(offset: int = 0, cursor: int = 0) -> tuple[list[str], int, int]:
    local_items = [(m.provider, m.id) for m in MODELS if m.cost == "local"]
    free_items = [(m.provider, m.id) for m in MODELS if m.cost == "free"]
    total = len(local_items) + len(free_items)

    items: list[tuple[str, str]] = []
    ids: list[str] = []

    for prov, mid in local_items + free_items:
        c = get_color(prov)
        items.append((f"{c}{'[' + prov + ']':<13s}{RESET}", mid))
        ids.append(mid)

    header = f"{BOLD}No-cost models{RESET} {DIM}({total} total){RESET}"
    hint = "#=select  ↑↓/jk=move  Enter=select  Esc/..=back  /=search  q=quit"
    offset, cursor = _render_paged_list(items, offset, cursor, header, hint, show_provider=True)
    return ids, offset, cursor


# ─── Interactive Loop ────────────────────────────────────────────────


def _handle_list_selection(key: str, items: list[str], extra_args: list[str]) -> None:
    if not items:
        return
    num = readkey_with_digit_timeout(key, len(items))
    idx = num - 1
    if 0 <= idx < len(items):
        launch_claude(items[idx], extra_args, pause_after=True)
    else:
        print(f"  Invalid number: {num}")


def run_interactive(state: str = "main", search_query: str = "", extra_args: list[str] | None = None) -> None:
    if extra_args is None:
        extra_args = []

    if IS_STDIN_TTY:
        _setup_terminal_restore()

    current_provider = ""
    provider_free_only = False
    search_provider = ""
    pview_ids: list[str] = []
    search_ids: list[str] = []
    free_ids: list[str] = []
    main_items: list[MainItem] = []
    scroll_offset = 0
    cursor = 0
    main_cursor = 0

    def _activate_main_item(item: MainItem) -> None:
        nonlocal state, current_provider, provider_free_only, scroll_offset, cursor
        kind = item[0]
        if kind == "last":
            model_id = item[1]
            if model_id:
                launch_claude(model_id, extra_args, pause_after=True)
        elif kind == "fav":
            fid = item[2]
            if fid not in MODEL_MAP:
                return
            launch_claude(fid, extra_args, pause_after=True)
        elif kind == "provider":
            current_provider = item[1]
            provider_free_only = False
            state = "provider"
            scroll_offset = 0
            cursor = 0
        elif kind == "free":
            state = "free"
            scroll_offset = 0
            cursor = 0

    while True:
        _restore_termios()
        if state == "main":
            main_items, main_cursor = render_main(main_cursor)
            key = readkey()
            if key in ("EOF", "CTRLC"):
                _restore_termios()
                sys.exit(0)
            if key == "q":
                _restore_termios()
                sys.exit(0)
            if key == "ESC":
                _restore_termios()
                sys.exit(0)
            if key in ("DOWN", "j"):
                main_cursor += 1
                continue
            if key in ("UP", "k"):
                main_cursor -= 1
                continue
            if key == "HOME":
                main_cursor = 0
                continue
            if key == "END":
                main_cursor = len(main_items) - 1
                continue
            if key == "ENTER":
                if main_items and 0 <= main_cursor < len(main_items):
                    _activate_main_item(main_items[main_cursor])
                continue
            if key == "F":
                state = "free"
                scroll_offset = 0
                cursor = 0
                continue
            if key == "/":
                if has_fzf():
                    pick = run_fzf_search("")
                    if pick:
                        launch_claude(pick, extra_args, pause_after=True)
                else:
                    q = readline_input("  Search: ")
                    if q:
                        search_query = q
                        search_provider = ""
                        state = "search"
                        scroll_offset = 0
                        cursor = 0
                continue
            if key.isdigit() and key != "0":
                favs = fav_list()
                num = readkey_with_digit_timeout(key, len(favs))
                idx = num - 1
                if 0 <= idx < len(favs):
                    fid = favs[idx]
                    if fid not in MODEL_MAP:
                        print(f"  {RED}Model unavailable:{RESET} {fid}")
                        continue
                    launch_claude(fid, extra_args, pause_after=True)
                else:
                    print(f"  Invalid favorite number: {num}")
                continue
            if len(key) == 1 and key.islower():
                pidx = ord(key) - 97
                if 0 <= pidx < len(PROVIDER_KEYS):
                    current_provider = PROVIDER_KEYS[pidx]
                    state = "provider"
                    scroll_offset = 0
                    cursor = 0
                else:
                    print("  Invalid provider letter.")
                continue
            if key == ".":
                if _check_dotdot():
                    _restore_termios()
                    sys.exit(0)
                continue
            continue

        else:
            if state == "provider":
                pview_ids, scroll_offset, cursor = render_provider(current_provider, scroll_offset, cursor, provider_free_only)
                list_ids = pview_ids
                fzf_query = current_provider
            elif state == "search":
                search_ids, scroll_offset, cursor = render_search(search_query, scroll_offset, cursor, provider=search_provider)
                list_ids = search_ids
                fzf_query = ""
            elif state == "free":
                free_ids, scroll_offset, cursor = render_free(scroll_offset, cursor)
                list_ids = free_ids
                fzf_query = "free"
            else:
                continue

            key = readkey()
            if key in ("EOF", "CTRLC"):
                _restore_termios()
                sys.exit(0)
            if key == "q":
                _restore_termios()
                sys.exit(0)
            if key == "ESC":
                if state == "search" and search_provider:
                    state = "provider"
                    search_provider = ""
                else:
                    state = "main"
                    provider_free_only = False
                scroll_offset = 0
                cursor = 0
                continue
            if key == ".":
                if _check_dotdot():
                    if state == "search" and search_provider:
                        state = "provider"
                        search_provider = ""
                    else:
                        state = "main"
                        provider_free_only = False
                    scroll_offset = 0
                    cursor = 0
                continue
            if key == "F" and state == "provider":
                provider_free_only = not provider_free_only
                scroll_offset = 0
                cursor = 0
                continue
            if key in ("DOWN", "j"):
                cursor += 1
                continue
            if key in ("UP", "k"):
                cursor -= 1
                continue
            if key == "PGDN":
                cursor += _page_height()
                continue
            if key == "PGUP":
                cursor -= _page_height()
                continue
            if key == "HOME":
                cursor = 0
                continue
            if key == "END":
                cursor = max(0, len(list_ids) - 1)
                continue
            if key == "ENTER":
                if list_ids and 0 <= cursor < len(list_ids):
                    launch_claude(list_ids[cursor], extra_args, pause_after=True)
                continue
            if key == "/":
                sp = current_provider if state == "provider" else ""
                if has_fzf():
                    pick = run_fzf_search(fzf_query, provider=sp)
                    if pick:
                        launch_claude(pick, extra_args, pause_after=True)
                else:
                    q = readline_input("  Search: ")
                    if q:
                        search_query = q
                        search_provider = sp
                        state = "search"
                        scroll_offset = 0
                        cursor = 0
                continue
            if key.isdigit() and key != "0":
                _handle_list_selection(key, list_ids, extra_args)
                continue

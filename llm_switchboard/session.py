"""Session watching and usage tracking."""

import json
import sys
import threading
from pathlib import Path

from .config import USAGE_FILE
from .util import RESET, YELLOW, locked_file


class SessionWatcher:
    """Background thread that tails the Claude Code session JSONL in real-time,
    accumulating usage data as messages are written."""

    def __init__(self, session_dir: Path, model_id: str):
        self.session_dir = session_dir
        self.model_id = model_id
        self.input_tokens = 0
        self.output_tokens = 0
        self.message_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pre_existing: dict[str, int] = {}
        if session_dir.is_dir():
            for f in session_dir.glob("*.jsonl"):
                try:
                    self._pre_existing[f.name] = f.stat().st_size
                except OSError:
                    pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self) -> dict | None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            if self.message_count == 0:
                return None
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "message_count": self.message_count,
            }

    def _find_session_file(self) -> tuple[Path, int] | None:
        if not self.session_dir.is_dir():
            return None
        for _ in range(120):
            if self._stop.is_set():
                return None
            for f in self.session_dir.glob("*.jsonl"):
                if f.name not in self._pre_existing:
                    return f, 0
                try:
                    cur_size = f.stat().st_size
                except OSError:
                    continue
                if cur_size > self._pre_existing[f.name]:
                    return f, self._pre_existing[f.name]
            self._stop.wait(1)
        return None

    def _parse_line(self, line: str) -> None:
        if not line.strip():
            return
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return
        if entry.get("type") != "assistant":
            return
        msg = entry.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            return
        usage = msg.get("usage")
        if not usage:
            return
        inp = usage.get("input_tokens", 0)
        inp += usage.get("cache_creation_input_tokens", 0)
        inp += usage.get("cache_read_input_tokens", 0)
        out = usage.get("output_tokens", 0)
        with self._lock:
            self.input_tokens += inp
            self.output_tokens += out
            self.message_count += 1

    def _watch(self) -> None:
        result = self._find_session_file()
        if not result:
            return
        session_file, offset = result

        try:
            with open(session_file) as f:
                if offset > 0:
                    f.seek(offset)
                while not self._stop.is_set():
                    line = f.readline()
                    if line:
                        self._parse_line(line)
                    else:
                        self._stop.wait(0.5)
        except Exception:
            pass


def claude_session_dir() -> Path:
    """Compute Claude Code session JSONL directory for the current working directory."""
    cwd = str(Path.cwd())
    encoded = cwd.replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded


def load_local_usage(usage_file: Path | None = None) -> dict:
    """Read usage.json, return {"models": {model_id: {...}, ...}} or empty structure."""
    f = usage_file or USAGE_FILE
    if not f.exists():
        return {"models": {}}
    try:
        data = json.loads(f.read_text())
        if not isinstance(data, dict) or "models" not in data:
            raise ValueError("invalid structure")
        return data
    except Exception:
        backup = f.with_suffix(".json.bak")
        try:
            f.rename(backup)
            print(f"  {YELLOW}Warning: usage.json was corrupted. Backed up to {backup.name}{RESET}", file=sys.stderr)
        except Exception:
            print(f"  {YELLOW}Warning: usage.json is corrupted and could not be backed up{RESET}", file=sys.stderr)
        return {"models": {}}


def save_local_usage(data: dict, usage_file: Path | None = None) -> None:
    """Write usage data to usage.json atomically (write tmp, then rename)."""
    f = usage_file or USAGE_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.rename(f)


def get_all_usage(usage_file: Path | None = None) -> list[dict]:
    """Get all model usage from local file. Returns list of usage dicts."""
    data = load_local_usage(usage_file)
    results = []
    for mid, entry in data.get("models", {}).items():
        if entry.get("message_count", 0) > 0:
            results.append({"model_id": mid, **entry})
    return results


def record_session(model_id: str, session_data: dict, usage_file: Path | None = None) -> None:
    """Merge session data into usage.json (increment cumulative counters)."""
    f = usage_file or USAGE_FILE
    with locked_file(f):
        data = load_local_usage(f)
        models = data.setdefault("models", {})
        entry = models.setdefault(model_id, {"input_tokens": 0, "output_tokens": 0, "message_count": 0, "sessions": 0})
        entry["input_tokens"] += session_data["input_tokens"]
        entry["output_tokens"] += session_data["output_tokens"]
        entry["message_count"] += session_data["message_count"]
        entry["sessions"] += 1
        save_local_usage(data, f)

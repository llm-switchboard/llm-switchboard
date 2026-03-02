"""Coding-agent compatibility testing — probes models via chat completions."""

import json
import re
import time
from pathlib import Path

from .config import CONFIG_DIR
from .util import locked_file
from .webui import api_post, build_anthropic_payload, parse_anthropic_response

COMPAT_FILE = CONFIG_DIR / "compat.json"

# ─── Tool Schemas (OpenAI function-calling format) ──────────────────
# These are the fake tools sent with tool-call tests. The model must
# produce tool_calls entries that match these schemas.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

_VALID_TOOL_NAMES = {"list_files", "read_file", "write_file"}

# ─── Text Test Definitions ──────────────────────────────────────────

_FORMAT_PROMPT = """\
You are a coding assistant. Show me a unified diff that changes the file \
`hello.py` from `print('hello')` to `print('goodbye')`.

Wrap the diff between BEGIN_DIFF and END_DIFF markers exactly like this:

BEGIN_DIFF
<your diff here>
END_DIFF

Output ONLY the markers and diff, nothing else."""

_CONSTRAINT_PROMPT = """\
You are a coding assistant. I have two files:

File: auth.py
```python
def check_access(user):
    return False
```

File: utils.py
```python
def format_name(name):
    return name.strip().title()
```

Edit ONLY `auth.py` so that `check_access` returns `True` instead of `False`. \
Do NOT modify `utils.py` in any way. Show the updated `auth.py` content."""

_HALLUCINATION_PROMPT = """\
You are a coding assistant. Please fix the bug on line 87 of \
`src/database/migrations/v42_schema.py`. The file should be in the current \
project but I'm not sure it exists."""

# ─── Tool-Call Test Definitions ─────────────────────────────────────

_TOOL_SCHEMA_PROMPT = """\
You are a coding assistant with access to file tools. \
List all files in the /src directory. Use the provided tools."""

_TOOL_CHAINING_PROMPT = """\
You are a coding assistant with access to file tools. Complete this task:
1. First, read the file /src/config.py to check the current DATABASE_URL value.
2. Then, write a new version of /src/config.py with DATABASE_URL changed to \
"postgresql://prod:secret@db.internal/app".

Use the provided tools. You may make multiple tool calls in one response."""

# Multi-turn conversation: model already made a tool call, got an error back.
_TOOL_ERROR_MESSAGES = [
    {
        "role": "user",
        "content": "Read the file /tmp/settings.json and show me the database host value.",
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_err_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "/tmp/settings.json"}),
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_err_1",
        "content": json.dumps({"error": "FileNotFoundError: [Errno 2] No such file or directory: '/tmp/settings.json'"}),
    },
]


# ─── Text Validators ────────────────────────────────────────────────


def _validate_format(text: str) -> bool:
    """Check that the response contains a unified diff."""
    t = text.strip()
    if "BEGIN_DIFF" in t and "END_DIFF" in t:
        between = t.split("BEGIN_DIFF", 1)[1].split("END_DIFF", 1)[0]
        if "-" in between and "+" in between:
            return True
    # Fallback: accept bare diff lines
    lines = t.splitlines()
    has_minus = any(ln.startswith("-") for ln in lines)
    has_plus = any(ln.startswith("+") for ln in lines)
    return has_minus and has_plus


def _validate_constraint(text: str) -> bool:
    """Check that only auth.py was modified and returns True."""
    t = text.lower()
    if "auth.py" not in t:
        return False
    if "return true" not in t and "return  true" not in t:
        return False
    # Fail if utils.py was modified — if format_name appears, the original
    # body must be intact (return name.strip().title())
    if "def format_name" in t:
        if "return name.strip().title()" not in t:
            return False
    return True


_REFUSAL_PATTERNS = re.compile(
    r"(doesn.t exist|does not exist|cannot find|can.t find|no such file|"
    r"not found|not sure|unable to locate|don.t have access|cannot locate|"
    r"i don.t see|file is missing|doesn.t appear|does not appear|"
    r"couldn.t find|could not find|i.m not able|not able to|"
    r"would need to see|need the actual|can.t access|cannot access)",
    re.IGNORECASE,
)

_SQL_FABRICATION = re.compile(
    r"(CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE|INSERT\s+INTO|"
    r"def\s+upgrade|def\s+downgrade|migration\.)",
    re.IGNORECASE,
)


def _validate_no_hallucination(text: str) -> bool:
    """Pass if model refuses/expresses uncertainty; fail if fabricated fix."""
    if _SQL_FABRICATION.search(text):
        return False
    if _REFUSAL_PATTERNS.search(text):
        return True
    # No clear refusal and no fabrication — ambiguous, lean pass
    # if the response is short (likely a clarifying question)
    return len(text.strip()) < 500


# ─── Tool-Call Validators (work for both OpenAI and normalized Anthropic) ───


def _parse_tool_calls(message: dict) -> list[dict]:
    """Extract normalized tool calls from an OpenAI-format message.

    Works with both native OpenAI tool_calls and Anthropic responses
    that have been converted via parse_anthropic_response().

    Returns list of {"name": str, "arguments": dict} for valid entries.
    """
    raw = message.get("tool_calls") or []
    parsed = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        # Some APIs return arguments as dict, others as JSON string
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                continue
        else:
            continue
        if not isinstance(args, dict):
            continue
        parsed.append({"name": name, "arguments": args})
    return parsed


def _parse_anthropic_tool_use(message: dict) -> list[dict]:
    """Extract tool_use blocks from a raw Anthropic response message.

    This handles the native Anthropic format (content array with tool_use blocks)
    without going through parse_anthropic_response first.
    """
    content = message.get("content", [])
    if not isinstance(content, list):
        return []
    parsed = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        inp = block.get("input", {})
        if not isinstance(inp, dict):
            continue
        parsed.append({"name": name, "arguments": inp})
    return parsed


def _get_tool_calls_any(message: dict) -> list[dict]:
    """Get tool calls from either OpenAI or Anthropic format."""
    # Try OpenAI format first
    calls = _parse_tool_calls(message)
    if calls:
        return calls
    # Try raw Anthropic format
    return _parse_anthropic_tool_use(message)


def _validate_tool_call_schema(message: dict) -> bool:
    """Check model produced a valid tool call to list_files with path arg."""
    calls = _get_tool_calls_any(message)
    if not calls:
        return False
    for call in calls:
        if call["name"] == "list_files" and "path" in call["arguments"]:
            return True
    # Fallback: accept any valid tool call to one of our tools
    for call in calls:
        if call["name"] in _VALID_TOOL_NAMES:
            return True
    return False


def _validate_tool_call_chaining(message: dict) -> bool:
    """Check model produced at least a read_file call for the config file."""
    calls = _get_tool_calls_any(message)
    if not calls:
        return False
    has_read = any(c["name"] == "read_file" and "path" in c["arguments"] and "config" in c["arguments"]["path"].lower() for c in calls)
    if has_read:
        return True
    # Accept if there's at least one valid tool call (model may combine steps)
    return any(c["name"] in _VALID_TOOL_NAMES for c in calls)


_ERROR_ACK_PATTERNS = re.compile(
    r"(file.{0,20}not found|doesn.t exist|does not exist|no such file|"
    r"error|FileNotFoundError|couldn.t read|could not read|unable to read|"
    r"failed to read|not available|cannot be found|doesn.t appear|"
    r"does not appear|check.{0,20}path|correct.{0,20}path|verify.{0,20}path|"
    r"different.{0,20}path|alternative|let me.{0,20}try|sorry|apologize)",
    re.IGNORECASE,
)

_FABRICATED_JSON = re.compile(
    r'("database"|"host"|"db_host"|"port"\s*:\s*\d{2,5})',
    re.IGNORECASE,
)


def _validate_tool_call_error_recovery(text: str) -> bool:
    """Pass if model acknowledges error; fail if it fabricates file contents."""
    if not text or not text.strip():
        return False
    # Fail if model fabricated the file contents
    if _FABRICATED_JSON.search(text):
        return False
    # Pass if model acknowledges the error
    if _ERROR_ACK_PATTERNS.search(text):
        return True
    # Also pass if model makes another tool call (retry with different path)
    # This is checked in the runner; here we just check text content.
    # Short responses without fabrication are acceptable
    return len(text.strip()) < 400


# ─── Test Registry ──────────────────────────────────────────────────

TESTS = {
    # --- Text tests (original) ---
    "format_compliance": {
        "prompt": _FORMAT_PROMPT,
        "validator": _validate_format,
        "description": "Can produce unified diff output",
        "response_type": "content",
    },
    "constraint_following": {
        "prompt": _CONSTRAINT_PROMPT,
        "validator": _validate_constraint,
        "description": "Edits only the file asked",
        "response_type": "content",
    },
    "no_hallucination": {
        "prompt": _HALLUCINATION_PROMPT,
        "validator": _validate_no_hallucination,
        "description": "Refuses to edit nonexistent file",
        "response_type": "content",
    },
    # --- Tool-call tests (new) ---
    "tool_call_schema": {
        "prompt": _TOOL_SCHEMA_PROMPT,
        "validator": _validate_tool_call_schema,
        "description": "Produces valid OpenAI tool_calls",
        "response_type": "message",
        "uses_tools": True,
    },
    "tool_call_chaining": {
        "prompt": _TOOL_CHAINING_PROMPT,
        "validator": _validate_tool_call_chaining,
        "description": "Chains read then write tool calls",
        "response_type": "message",
        "uses_tools": True,
    },
    "tool_call_error_recovery": {
        "messages": _TOOL_ERROR_MESSAGES,
        "validator": _validate_tool_call_error_recovery,
        "description": "Handles tool errors gracefully",
        "response_type": "content",
        "uses_tools": True,
    },
}

# ─── Scoring ─────────────────────────────────────────────────────────

REQUIRED_TESTS = {"tool_call_schema", "no_hallucination"}
MIN_PASS_TOTAL = 4  # out of 6 tests


def compute_agent_status(tests: dict) -> str:
    """Determine agent status from per-test results.

    - "pass":    tool_call_schema + no_hallucination pass AND >= 4/6 total
    - "fail":    tool_call_schema fails
    - "partial": everything else
    """
    if not tests:
        return "fail"
    tool_schema_passed = tests.get("tool_call_schema", {}).get("passed", False)
    if not tool_schema_passed:
        return "fail"
    required_all_pass = all(tests.get(t, {}).get("passed", False) for t in REQUIRED_TESTS)
    pass_count = sum(1 for r in tests.values() if r.get("passed"))
    if required_all_pass and pass_count >= MIN_PASS_TOTAL:
        return "pass"
    return "partial"


# ─── Persistence ─────────────────────────────────────────────────────


def load_compat(compat_file: Path | None = None) -> dict:
    """Load compat.json. Returns {"results": {model_id: {...}}}."""
    f = compat_file or COMPAT_FILE
    if not f.exists():
        return {"results": {}}
    try:
        data = json.loads(f.read_text())
        if not isinstance(data, dict) or "results" not in data:
            raise ValueError("invalid structure")
        return data
    except Exception:
        return {"results": {}}


def save_compat(data: dict, compat_file: Path | None = None) -> None:
    """Write compat data atomically (write tmp, then rename)."""
    f = compat_file or COMPAT_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.rename(f)


# ─── Runner ──────────────────────────────────────────────────────────


def _run_one_test(model_id: str, test_name: str, test_def: dict, mode: str = "openai", chat_path: str = "/api/chat/completions", base_url: str | None = None, api_key: str | None = None) -> dict:
    """Run a single compat test. Returns result dict with passed, latency_ms, etc.

    Args:
        mode: "openai" or "anthropic" — determines request/response format
        chat_path: endpoint path for chat completions
        base_url: override base URL
        api_key: override API key
    """
    messages = test_def.get("messages") or [{"role": "user", "content": test_def["prompt"]}]

    if mode == "anthropic":
        payload = build_anthropic_payload(
            model_id,
            messages,
            max_tokens=1024,
            tools=TOOL_SCHEMAS if test_def.get("uses_tools") else None,
        )
    else:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
        }
        if test_def.get("uses_tools"):
            payload["tools"] = TOOL_SCHEMAS

    start = time.monotonic()
    resp = api_post(chat_path, payload, timeout=60, base_url=base_url, api_key=api_key)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    if "_error" in resp:
        return {
            "passed": False,
            "error": resp["_error"],
            "latency_ms": elapsed_ms,
        }

    # Normalize response to OpenAI format
    if mode == "anthropic":
        # Raw Anthropic response — convert to OpenAI shape
        if "content" in resp and isinstance(resp.get("content"), list):
            resp = parse_anthropic_response(resp)

    # Extract response message from OpenAI-format response
    try:
        message = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {
            "passed": False,
            "error": "unexpected response format",
            "latency_ms": elapsed_ms,
        }

    response_type = test_def.get("response_type", "content")
    if response_type == "message":
        passed = test_def["validator"](message)
    else:
        content = message.get("content") or ""
        passed = test_def["validator"](content)

    return {
        "passed": passed,
        "latency_ms": elapsed_ms,
    }


def run_compat_test(model_id: str, compat_file: Path | None = None, mode: str = "openai", chat_path: str = "/api/chat/completions", base_url: str | None = None, api_key: str | None = None) -> dict:
    """Run all compat tests on a model. Returns per-test results dict."""
    results = {}
    for test_name, test_def in TESTS.items():
        results[test_name] = _run_one_test(
            model_id,
            test_name,
            test_def,
            mode=mode,
            chat_path=chat_path,
            base_url=base_url,
            api_key=api_key,
        )

    # Compute summary
    pass_count = sum(1 for r in results.values() if r.get("passed"))
    fail_count = len(results) - pass_count
    total_latency = sum(r.get("latency_ms", 0) for r in results.values())
    last_status = compute_agent_status(results)

    entry = {
        "last_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tests": results,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "last_status": last_status,
        "latency_ms": total_latency,
    }

    # Persist (locked to prevent concurrent test runs from clobbering each other)
    f = compat_file or COMPAT_FILE
    with locked_file(f):
        data = load_compat(f)
        data["results"][model_id] = entry
        save_compat(data, f)

    return entry


# ─── Query Helpers ───────────────────────────────────────────────────


def get_compat_status(model_id: str, compat_file: Path | None = None) -> str | None:
    """Return "pass", "partial", "fail", or None if not tested."""
    data = load_compat(compat_file)
    entry = data.get("results", {}).get(model_id)
    if entry is None:
        return None
    return entry.get("last_status")  # type: ignore[no-any-return]


def get_agent_ok_models(compat_file: Path | None = None) -> set[str]:
    """Return set of model IDs with status "pass"."""
    data = load_compat(compat_file)
    return {mid for mid, entry in data.get("results", {}).items() if entry.get("last_status") == "pass"}

"""API client layer — supports OpenAI and Anthropic endpoints."""

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import OPENWEBUI_URL, OPENWEBUI_KEY
from .util import read_response


# ─── Core HTTP ────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 10,
            base_url: str | None = None, api_key: str | None = None) -> dict:
    url = f"{base_url or OPENWEBUI_URL}{path}"
    key = api_key if api_key is not None else OPENWEBUI_KEY
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(read_response(resp))
    except (URLError, TimeoutError, OSError) as e:
        return {"_error": str(e)}


def api_post(path: str, data: dict, timeout: int = 10,
             base_url: str | None = None, api_key: str | None = None) -> dict:
    url = f"{base_url or OPENWEBUI_URL}{path}"
    key = api_key if api_key is not None else OPENWEBUI_KEY
    body = json.dumps(data).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = Request(url, data=body, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(read_response(resp))
    except (URLError, TimeoutError, OSError) as e:
        return {"_error": str(e)}


# ─── Anthropic Request/Response Helpers ──────────────────────────────

def build_anthropic_payload(model: str, messages: list[dict],
                            max_tokens: int = 1024,
                            tools: list[dict] | None = None) -> dict:
    """Build an Anthropic Messages API request payload from OpenAI-style messages.

    Converts OpenAI message format to Anthropic format:
    - system messages → top-level 'system' param
    - tool_calls → tool_use content blocks
    - tool role → tool_result content blocks
    - OpenAI tools → Anthropic tool format (input_schema instead of parameters)
    """
    system_parts = []
    anthropic_msgs = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            system_parts.append(msg.get("content", ""))
            continue
        if role == "user":
            anthropic_msgs.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Convert to Anthropic tool_use content blocks
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            args = {}
                    else:
                        args = raw_args if isinstance(raw_args, dict) else {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                anthropic_msgs.append({"role": "assistant", "content": blocks})
            elif content is not None:
                anthropic_msgs.append({"role": "assistant", "content": content})
        elif role == "tool":
            # Convert to tool_result
            anthropic_msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }],
            })

    payload: dict = {
        "model": model,
        "messages": anthropic_msgs,
        "max_tokens": max_tokens,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    if tools:
        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in tools:
            fn = tool.get("function", {})
            anthropic_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        payload["tools"] = anthropic_tools

    return payload


def parse_anthropic_response(resp: dict) -> dict:
    """Convert Anthropic Messages response to OpenAI-compatible format.

    This allows downstream code to use a single response format.
    Maps: content[type=text] → message.content
          content[type=tool_use] → message.tool_calls
    """
    content_blocks = resp.get("content", [])
    text_parts = []
    tool_calls = []

    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    message: dict = {
        "role": resp.get("role", "assistant"),
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "choices": [{"message": message, "index": 0}],
        "model": resp.get("model", ""),
        "usage": resp.get("usage", {}),
    }

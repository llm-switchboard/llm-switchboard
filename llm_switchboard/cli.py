"""CLI entry point — argument parsing, model fetching, non-interactive modes."""

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__
from .cache import fetch_cached
from .config import (
    OPENWEBUI_URL, OPENWEBUI_KEY,
    CONFIG_DIR, CACHE_DIR, DETECT_CACHE,
    FAV_FILE, FREE_PROVIDERS_FILE, USAGE_FILE,
    ensure_dirs, fav_list, fav_add, fav_rm,
    last_model_read, last_model_write,
)
from .free_tier import (
    FreeTierRule, KNOWN_FREE_TIERS, DEFAULT_FREE_PROVIDERS, CONFIG_HEADER,
    free_tier_rules_from_file, parse_free_tier_rules,
    load_gemini_cache, gemini_cache_stale, fetch_gemini_free_tier,
    write_gemini_cache, write_free_tier_config, is_free_tier,
)
from .models import Model, provider_from_url, format_ctx, PROVIDER_LIMITS
from .compat import (
    run_compat_test, load_compat, get_agent_ok_models, TESTS,
    _run_one_test, compute_agent_status, save_compat,
    REQUIRED_TESTS, MIN_PASS_TOTAL, COMPAT_FILE,
)
from .session import load_local_usage, save_local_usage, get_all_usage
from .util import (
    IS_TTY, IS_STDIN_TTY, PRIVATE_IP_RE,
    CYAN, GREEN, YELLOW, MAGENTA, BLUE, RED,
    DIM, BOLD, RESET,
    sanitize_display, fmt_price, fmt_tokens, die,
)
from .endpoint import (
    resolve_endpoint, format_endpoint_info, load_api_mode, normalize_url,
    Endpoint, VALID_MODES, VALID_PREFERENCES,
    _probe_get, _looks_like_models_list, _extract_model_id,
    _fetch_any_model_id, _probe_openai_chat, _probe_anthropic_chat,
)
from .webui import api_get, api_post

# ─── Global Model State ──────────────────────────────────────────────
# These are populated by fetch_models() and shared with the TUI module.

_OWN_FLAGS = {
    "-h", "--help", "-f", "--free", "-l", "--last", "--refresh",
    "--fav", "--free-tier", "--setup", "-s", "--stats", "--reset",
    "--list-models", "--select", "--favorites", "--free-only",
    "--provider", "--json", "--self-test", "--version",
    "--print-paths", "--print-endpoints",
    "--compat-test", "--compat-report", "--agent-ok",
    "--all", "--api-mode", "--base-url", "--auto-prefer",
    "--doctor", "--clear-detect-cache",
}

# ─── Resolved Endpoint (lazily initialized) ─────────────────────────
# Set by _resolve_api() before any API call that needs it.

_RESOLVED_EP: Endpoint | None = None
_CLI_API_MODE: str | None = None
_CLI_BASE_URL: str | None = None
_CLI_AUTO_PREFER: str | None = None


def _resolve_api() -> Endpoint:
    """Resolve and cache the API endpoint."""
    global _RESOLVED_EP
    if _RESOLVED_EP is not None:
        return _RESOLVED_EP
    mode = _CLI_API_MODE or load_api_mode()
    base = _CLI_BASE_URL
    source = "cli" if _CLI_API_MODE else ("config" if load_api_mode() != "auto" else "")
    _RESOLVED_EP = resolve_endpoint(
        mode=mode, base_url=base, prefer=_CLI_AUTO_PREFER, source=source,
    )
    return _RESOLVED_EP


def _populate_tui_globals(models, model_map, providers, provider_free,
                          provider_local, provider_cloud, provider_keys):
    """Push model data into the tui module's global state."""
    from . import tui
    tui.MODELS = models
    tui.MODEL_MAP = model_map
    tui.PROVIDERS = providers
    tui.PROVIDER_FREE = provider_free
    tui.PROVIDER_LOCAL = provider_local
    tui.PROVIDER_CLOUD = provider_cloud
    tui.PROVIDER_KEYS = provider_keys


def fetch_models() -> tuple[list[Model], dict[str, Model]]:
    """Fetch models from Open WebUI and populate global state.

    Returns (MODELS, MODEL_MAP) for non-interactive use.
    """
    from .tui import Spinner

    with Spinner("Fetching models..."):
        with ThreadPoolExecutor(max_workers=3) as pool:
            from .config import CONN_CACHE, OLLAMA_CACHE
            f_models = pool.submit(api_get, "/api/models", 30)
            f_openai = pool.submit(fetch_cached, api_get, "/openai/config", CONN_CACHE, "OPENAI_API_BASE_URLS")
            f_ollama = pool.submit(fetch_cached, api_get, "/ollama/config", OLLAMA_CACHE, "ENABLE_OLLAMA_API")

            models_json = f_models.result(timeout=35)
            openai_cfg = f_openai.result(timeout=35)
            ollama_cfg = f_ollama.result(timeout=35)

    if "_error" in models_json or "data" not in models_json:
        die(f"Failed to fetch models: {models_json.get('_error', 'no data field')}")

    ollama_enabled = ollama_cfg.get("ENABLE_OLLAMA_API", False)

    url_map: dict[int, str] = {}
    for i, url in enumerate(openai_cfg.get("OPENAI_API_BASE_URLS", [])):
        url_map[i] = provider_from_url(url)

    ollama_urls = ollama_cfg.get("OLLAMA_BASE_URLS", []) if ollama_enabled else []
    ollama_url_local: dict[int, bool] = {}
    for i, ourl in enumerate(ollama_urls):
        ollama_url_local[i] = bool(PRIVATE_IP_RE.search(ourl))

    user_free = free_tier_rules_from_file(FREE_PROVIDERS_FILE)
    from .config import GEMINI_FREE_CACHE
    gemini_cache = load_gemini_cache(GEMINI_FREE_CACHE) if "gemini" in user_free else None

    models: list[Model] = []
    for item in models_json["data"]:
        mid = item.get("id", "")
        owner = item.get("owned_by", "")
        raw_idx = item.get("urlIdx")

        conn_type = item.get("connection_type")
        is_preset = item.get("preset", False)
        ollama_obj = item.get("ollama") or {}
        is_ollama_cloud = bool(ollama_obj.get("remote_host"))
        if owner == "arena" or item.get("arena"):
            provider = "arena"
        elif owner == "ollama" or (raw_idx is None and ollama_enabled):
            provider = "ollama"
        elif raw_idx is not None and raw_idx in url_map:
            provider = url_map[raw_idx]
        elif is_preset and conn_type is None:
            provider = "ollama"
        else:
            provider = "external"

        oai = item.get("openai", {}) or {}
        info = item.get("info", {}) or {}
        info_params = info.get("params", {}) or {}
        meta = info.get("meta", {}) or {}

        ctx_raw = (item.get("context_length")
                   or oai.get("context_length")
                   or item.get("max_context_length")
                   or oai.get("max_context_length")
                   or item.get("context_window")
                   or info_params.get("max_context_length"))
        ctx_k = format_ctx(ctx_raw)

        desc_raw = (meta.get("description")
                    or item.get("description")
                    or oai.get("description")
                    or "")
        desc = sanitize_display(desc_raw).replace("\n", " ").strip()
        if len(desc) > 120:
            desc = desc[:117].rsplit(" ", 1)[0] + "..."

        pricing = oai.get("pricing") or item.get("pricing") or {}
        price_str = ""
        pricing_is_free = False
        if isinstance(pricing, dict):
            p_in = pricing.get("prompt") if pricing.get("prompt") is not None else pricing.get("input")
            p_out = pricing.get("completion") if pricing.get("completion") is not None else pricing.get("output")
            if p_in is not None and p_out is not None:
                try:
                    vi = float(p_in) * 1_000_000
                    vo = float(p_out) * 1_000_000
                    if vi == 0 and vo == 0:
                        pricing_is_free = True
                    else:
                        price_str = f"{fmt_price(vi)}/{fmt_price(vo)}"
                except (ValueError, TypeError):
                    pass

        if provider == "ollama" and is_ollama_cloud:
            cost = "cloud"
        elif provider == "ollama":
            cost = "local"
        elif pricing_is_free:
            cost = "free"
        elif is_free_tier(provider, mid, user_free, gemini_cache):
            cost = "free"
        else:
            cost = "paid"

        models.append(Model(provider, mid, cost, desc, ctx_k, price_str))

    models.sort(key=lambda m: m.id)

    if not models:
        die("No models returned from Open WebUI.")

    model_map = {m.id: m for m in models}

    providers: dict[str, int] = {}
    provider_free: dict[str, int] = {}
    provider_local: dict[str, int] = {}
    provider_cloud: dict[str, int] = {}
    for m in models:
        providers[m.provider] = providers.get(m.provider, 0) + 1
        if m.cost == "free":
            provider_free[m.provider] = provider_free.get(m.provider, 0) + 1
        elif m.cost == "local":
            provider_local[m.provider] = provider_local.get(m.provider, 0) + 1
        elif m.cost == "cloud":
            provider_cloud[m.provider] = provider_cloud.get(m.provider, 0) + 1

    provider_keys = sorted(providers.keys())

    _populate_tui_globals(models, model_map, providers, provider_free,
                          provider_local, provider_cloud, provider_keys)

    if "gemini" in user_free and "gemini" in providers and gemini_cache_stale(GEMINI_FREE_CACHE):
        print(f"  {YELLOW}⚠ Gemini free-tier data is stale.{RESET} "
              f"Run {BOLD}--free-tier update{RESET} to refresh.",
              file=sys.stderr)

    return models, model_map


# ─── Non-Interactive Commands ────────────────────────────────────────

def cmd_list_models(as_json: bool = False, free_only: bool = False,
                    provider_filter: str = "", agent_ok: bool = False) -> None:
    """List models in text or JSON format."""
    models, model_map = fetch_models()

    filtered = models
    if free_only:
        filtered = [m for m in filtered if m.cost in ("free", "local")]
    if agent_ok:
        ok_set = get_agent_ok_models()
        filtered = [m for m in filtered if m.id in ok_set]
    if provider_filter:
        pf = provider_filter.lower()
        filtered = [m for m in filtered if m.provider.lower() == pf]

    if as_json:
        out = []
        for m in filtered:
            entry = {
                "id": m.id,
                "provider": m.provider,
                "cost": m.cost,
            }
            if m.description:
                entry["description"] = m.description
            if m.ctx_k:
                entry["context"] = m.ctx_k
            if m.price:
                entry["price"] = m.price
            out.append(entry)
        print(json.dumps(out, indent=2))
    else:
        if not filtered:
            print("No models found.", file=sys.stderr)
            sys.exit(1)
        for m in filtered:
            tag = ""
            if m.cost == "free":
                tag = " [FREE]"
            elif m.cost == "local":
                tag = " [LOCAL]"
            elif m.cost == "cloud":
                tag = " [CLOUD]"
            ctx = f" ({m.ctx_k})" if m.ctx_k else ""
            price = f" {m.price}" if m.price else ""
            print(f"{m.provider:<13s} {m.id}{tag}{ctx}{price}")


def cmd_select(model_id: str, extra_args: list[str]) -> None:
    """Launch a session with a specific model without TUI."""
    models, model_map = fetch_models()
    if model_id not in model_map:
        # Try fuzzy match
        matches = [m.id for m in models if model_id.lower() in m.id.lower()]
        if len(matches) == 1:
            model_id = matches[0]
        elif matches:
            print(f"Ambiguous model '{model_id}'. Matches:", file=sys.stderr)
            for m in matches[:10]:
                print(f"  {m}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Model not found: {model_id}", file=sys.stderr)
            sys.exit(1)
    from .tui import launch_claude
    sys.exit(launch_claude(model_id, extra_args))


def cmd_favorites() -> None:
    """List favorites."""
    favs = fav_list()
    if not favs:
        print("No favorites. Add with: llm-switchboard --fav add <model-id>")
    else:
        for f in favs:
            print(f)


def cmd_clear_detect_cache() -> None:
    """Remove endpoint auto-detect cache."""
    if not DETECT_CACHE.exists():
        print("Detect cache already clear.")
    else:
        path = str(DETECT_CACHE)
        DETECT_CACHE.unlink()
        print(f"Detect cache cleared: {path}")


def cmd_doctor() -> int:
    """Run health checks and print a diagnostic report.

    Returns exit code: 0=OK, 1=models fail, 2=chat fail, 3=unexpected error.
    """
    try:
        return _doctor_report()
    except BrokenPipeError:
        return 0
    except Exception as exc:
        print(f"\nDoctor error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


def _doctor_report() -> int:
    """Internal doctor implementation. Raises on unexpected errors."""
    ep = _resolve_api()
    key = OPENWEBUI_KEY

    print(f"\n{BOLD}LLM Switchboard Doctor{RESET}")
    print(f"{DIM}{'─' * 40}{RESET}")
    print(f"Base URL:        {ep.base_url}")
    print(f"API mode:        {ep.mode}")
    if ep.source:
        print(f"Mode source:     {ep.source}")
    if ep.probe_match:
        print(f"Probe matched:   {ep.probe_match}")

    # ── Paths ──
    print(f"\n{BOLD}Paths{RESET}")
    print(f"  Config:        {CONFIG_DIR}")
    print(f"  Cache:         {CACHE_DIR}")
    cache_exists = DETECT_CACHE.exists()
    cache_note = "" if cache_exists else f"  {DIM}(not present){RESET}"
    print(f"  Detect cache:  {DETECT_CACHE}{cache_note}")

    # ── Connectivity ──
    exit_code = 0
    print(f"\n{BOLD}Connectivity{RESET}")

    # Models endpoint
    models_url = f"{ep.base_url}{ep.models_path}"
    models_resp = _probe_get(models_url, key)
    if models_resp is not None and _looks_like_models_list(models_resp):
        model_count = len(models_resp.get("data", []))
        print(f"  Models endpoint: {GREEN}OK{RESET}  {models_url}  ({model_count} models)")
    elif models_resp is not None:
        print(f"  Models endpoint: {YELLOW}WARN{RESET}  {models_url}  "
              f"(responded but unexpected shape)")
        exit_code = max(exit_code, 1)
    else:
        print(f"  Models endpoint: {RED}FAIL{RESET}  {models_url}  (no response or not JSON)")
        exit_code = max(exit_code, 1)

    # Chat endpoint probe
    print(f"\n{BOLD}Chat Probe{RESET}")
    if ep.mode == "anthropic":
        chat_url = f"{ep.base_url}{ep.chat_path}"
        chat_ok = _probe_anthropic_chat(ep.base_url, key)
        if chat_ok:
            print(f"  Chat endpoint:   {GREEN}OK{RESET}  {chat_url}")
        else:
            print(f"  Chat endpoint:   {RED}FAIL{RESET}  {chat_url}  (not reachable)")
            exit_code = max(exit_code, 2)
    else:
        # OpenAI: fetch model ID and probe with it
        model_id, _ = _fetch_any_model_id(ep.base_url, key)
        if model_id:
            print(f"  Using model:     {model_id}")
        else:
            print(f"  Using model:     {DIM}(none found, using __probe__){RESET}")
        chat_path = _probe_openai_chat(ep.base_url, key, model_id)
        if chat_path:
            print(f"  Chat endpoint:   {GREEN}OK{RESET}  {ep.base_url}{chat_path}")
        else:
            print(f"  Chat endpoint:   {RED}FAIL{RESET}  {ep.base_url}{ep.chat_path}  (not reachable)")
            exit_code = max(exit_code, 2)

    # ── Agent Compatibility ──
    print(f"\n{BOLD}Agent Compatibility{RESET}")
    compat_data = load_compat()
    results = compat_data.get("results", {})
    compat_exists = COMPAT_FILE.exists()
    print(f"  Compat data:     {COMPAT_FILE}  {'(exists)' if compat_exists else '(not present)'}")
    print(f"  Models tested:   {len(results)}")
    if results:
        n_pass = sum(1 for e in results.values() if e.get("last_status") == "pass")
        n_partial = sum(1 for e in results.values() if e.get("last_status") == "partial")
        n_fail = sum(1 for e in results.values() if e.get("last_status") == "fail")
        print(f"  Agent PASS:      {n_pass}")
        print(f"  Agent PARTIAL:   {n_partial}")
        print(f"  Agent FAIL:      {n_fail}")
        # Top agent-ok models
        ok_models = sorted(mid for mid, e in results.items() if e.get("last_status") == "pass")
        if ok_models:
            top = ok_models[:5]
            more = f"  (+{len(ok_models) - 5} more)" if len(ok_models) > 5 else ""
            print(f"  Top agent-ok:    {', '.join(top)}{more}")

    # ── Recommendations ──
    recs = []
    if exit_code >= 2:
        recs.append(f"Chat probe failed. Try: --api-mode openai  or  --api-mode anthropic")
        recs.append(f"Run --print-endpoints to see detected paths")
    if exit_code >= 1 and exit_code < 2:
        recs.append(f"Models endpoint issue. Check that OPENWEBUI_URL is correct")
    if cache_exists:
        import time
        age = time.time() - DETECT_CACHE.stat().st_mtime
        if age > 3600:
            recs.append(f"Detect cache is stale ({int(age / 60)}m old). Run: --clear-detect-cache")
    if not results:
        recs.append(f"No agent compat data. Run: --compat-test --all")
    elif n_pass == 0:
        recs.append(f"No agent-ok models. Run: --compat-test --all  then  --list-models --agent-ok")

    if recs:
        print(f"\n{BOLD}Recommendations{RESET}")
        for r in recs:
            print(f"  {YELLOW}→{RESET} {r}")

    # ── Summary ──
    if exit_code == 0:
        print(f"\n{GREEN}All checks passed.{RESET}\n")
    else:
        print()

    return exit_code


def cmd_compat_test(model_id: str) -> None:
    """Run compat tests on a single model, print per-test pass/fail."""
    from .tui import Spinner
    ep = _resolve_api()
    print(f"\n  {BOLD}Compatibility test: {model_id}{RESET}")
    print(f"  {DIM}6 tests: 3 text + 3 tool-call  mode={ep.mode}  "
          f"(required: {', '.join(sorted(REQUIRED_TESTS))}){RESET}\n")

    results = {}
    for test_name, test_def in TESTS.items():
        desc = test_def["description"]
        tool_tag = f" {CYAN}[tool]{RESET}" if test_def.get("uses_tools") else ""
        req_tag = f" {YELLOW}*{RESET}" if test_name in REQUIRED_TESTS else ""
        with Spinner(f"Testing {test_name}..."):
            result = _run_one_test(model_id, test_name, test_def,
                                   mode=ep.mode, chat_path=ep.chat_path,
                                   base_url=ep.base_url)
        results[test_name] = result

        err = result.get("error", "")
        ms = result.get("latency_ms", 0)
        if err:
            print(f"  {RED}FAIL{RESET}  {test_name}{tool_tag}{req_tag} — {desc}  "
                  f"{DIM}(error: {err}){RESET}")
        else:
            tag = f"{GREEN}PASS{RESET}" if result["passed"] else f"{RED}FAIL{RESET}"
            print(f"  {tag}  {test_name}{tool_tag}{req_tag} — {desc}  {DIM}({ms}ms){RESET}")

    # Compute and persist
    pass_count = sum(1 for r in results.values() if r.get("passed"))
    fail_count = len(results) - pass_count
    total_latency = sum(r.get("latency_ms", 0) for r in results.values())
    status = compute_agent_status(results)

    import time as _time
    entry = {
        "last_run": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "tests": results,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "last_status": status,
        "latency_ms": total_latency,
    }
    data = load_compat()
    data["results"][model_id] = entry
    save_compat(data)

    if status == "pass":
        color = GREEN
    elif status == "partial":
        color = YELLOW
    else:
        color = RED
    print(f"\n  Result: {color}{status.upper()}{RESET}  "
          f"({pass_count}/{pass_count + fail_count} passed, "
          f"{total_latency}ms total)")
    if status == "pass":
        print(f"  {GREEN}AGENT ✓{RESET}  Model is tool-call compatible\n")
    elif status == "partial":
        print(f"  {YELLOW}AGENT ⚠{RESET}  Partial compatibility — "
              f"required: {', '.join(t for t in REQUIRED_TESTS if not results.get(t, {}).get('passed'))}\n")
    else:
        print(f"  {RED}AGENT ✗{RESET}  Not tool-call compatible\n")


def cmd_compat_test_all() -> None:
    """Run compat tests on all models."""
    ep = _resolve_api()
    models, model_map = fetch_models()
    print(f"\n  {BOLD}Compatibility testing all {len(models)} models{RESET}  "
          f"{DIM}(mode={ep.mode}){RESET}\n")
    for m in models:
        entry = run_compat_test(m.id, mode=ep.mode, chat_path=ep.chat_path,
                                base_url=ep.base_url)
        status = entry["last_status"]
        if status == "pass":
            color = GREEN
        elif status == "partial":
            color = YELLOW
        else:
            color = RED
        print(f"  {color}{status.upper():<7s}{RESET} {m.id}  "
              f"{DIM}({entry['pass_count']}/{entry['pass_count'] + entry['fail_count']} passed, "
              f"{entry['latency_ms']}ms){RESET}")
    print()


def _compat_model_summary(entry: dict) -> dict:
    """Build a per-model summary dict for JSON/report use."""
    tests = entry.get("tests", {})
    test_names = list(TESTS.keys())
    passed = [t for t in test_names if tests.get(t, {}).get("passed")]
    failed = [t for t in test_names if t in tests and not tests[t].get("passed")]
    req_failed = [t for t in REQUIRED_TESTS if t in failed]
    return {
        "agent_status": entry.get("last_status", "fail"),
        "pass_count": entry.get("pass_count", 0),
        "fail_count": entry.get("fail_count", 0),
        "latency_ms": entry.get("latency_ms", 0),
        "last_run": entry.get("last_run", ""),
        "passed_tests": passed,
        "failed_tests": failed,
        "required_failed": req_failed,
    }


_SCORING_RULES = (
    f"PASS requires: all required tests pass AND >= {MIN_PASS_TOTAL}/{len(TESTS)} total. "
    f"FAIL if tool_call_schema fails. PARTIAL otherwise."
)


def cmd_compat_report(as_json: bool = False) -> None:
    """Show compat test results from compat.json."""
    data = load_compat()
    results = data.get("results", {})
    if not results:
        print("No compatibility test results. Run: llm-switchboard --compat-test <model-id>")
        return

    if as_json:
        out = {
            "required_tests": sorted(REQUIRED_TESTS),
            "scoring_rules": _SCORING_RULES,
            "models": {},
        }
        for mid in sorted(results):
            out["models"][mid] = _compat_model_summary(results[mid])
        print(json.dumps(out, indent=2))
        return

    test_names = list(TESTS.keys())

    print(f"\n  {BOLD}Compatibility Test Results{RESET}")
    print(f"  {DIM}* = required for AGENT pass  [tool] = tool-call test{RESET}")
    print(f"  {DIM}Required tests: {', '.join(sorted(REQUIRED_TESTS))}{RESET}\n")

    for mid in sorted(results):
        entry = results[mid]
        status = entry.get("last_status", "?")
        if status == "pass":
            color, badge = GREEN, "AGENT \u2713"
        elif status == "partial":
            color, badge = YELLOW, "AGENT \u26a0"
        else:
            color, badge = RED, "AGENT \u2717"
        pc = entry.get("pass_count", 0)
        fc = entry.get("fail_count", 0)
        ms = entry.get("latency_ms", 0)
        run = entry.get("last_run", "")

        # Build failure reason for non-pass models
        summary = _compat_model_summary(entry)
        why = ""
        if status != "pass" and summary["failed_tests"]:
            req_f = summary["required_failed"]
            if req_f:
                why = f"  {DIM}(required failed: {', '.join(req_f)}){RESET}"
            else:
                why = f"  {DIM}(failed: {', '.join(summary['failed_tests'][:3])}){RESET}"

        print(f"  {color}{badge:<8s}{RESET} {mid}  "
              f"{DIM}{pc}/{pc + fc} passed  {ms}ms  {run}{RESET}{why}")

        tests = entry.get("tests", {})
        for tname in test_names:
            tresult = tests.get(tname)
            if tresult is None:
                print(f"    {DIM}----  {tname}{RESET}")
                continue
            p = tresult.get("passed", False)
            tag = f"{GREEN}PASS{RESET}" if p else f"{RED}FAIL{RESET}"
            tdef = TESTS.get(tname, {})
            req = f" {YELLOW}*{RESET}" if tname in REQUIRED_TESTS else ""
            tool = f" {CYAN}[tool]{RESET}" if tdef.get("uses_tools") else ""
            err = tresult.get("error", "")
            extra = f"  {DIM}({err}){RESET}" if err else ""
            tms = tresult.get("latency_ms", 0)
            print(f"    {tag}  {tname}{tool}{req}  {DIM}{tms}ms{RESET}{extra}")

    # Legend
    print(f"\n  {BOLD}Legend{RESET}")
    print(f"  {GREEN}AGENT \u2713{RESET} = required tests pass + >= {MIN_PASS_TOTAL}/{len(TESTS)} total")
    print(f"  {YELLOW}AGENT \u26a0{RESET} = partial compatibility (see failed tests)")
    print(f"  {RED}AGENT \u2717{RESET} = tool schema failed (not agent compatible)")
    print()


# ─── Legacy Commands ─────────────────────────────────────────────────

def do_refresh() -> None:
    from .tui import Spinner
    from .config import CONN_CACHE
    with Spinner("Refreshing model list from all providers..."):
        if CONN_CACHE.exists():
            CONN_CACHE.unlink()
        conn_path = "/api/v1/configs/connections"
        api_post(conn_path, {"ENABLE_DIRECT_CONNECTIONS": False, "ENABLE_BASE_MODELS_CACHE": False})
        api_get("/api/models")
        api_post(conn_path, {"ENABLE_DIRECT_CONNECTIONS": False, "ENABLE_BASE_MODELS_CACHE": True})
    print("Cache refreshed.")


def run_reset() -> None:
    if not USAGE_FILE.exists():
        print(f"  {DIM}No usage data to reset.{RESET}")
        return
    USAGE_FILE.unlink()
    print(f"  {GREEN}Usage data cleared.{RESET}")


def run_stats() -> None:
    from .tui import get_color, MODEL_MAP
    models_data = get_all_usage()
    if not models_data:
        print(f"\n  {YELLOW}No usage data available.{RESET}")
        print(f"  {DIM}(No sessions recorded yet){RESET}\n")
        return

    by_provider: dict[str, list[dict]] = defaultdict(list)
    total_msgs = 0
    total_sessions = 0
    total_in = 0
    total_out = 0
    total_cost = 0.0

    for entry in models_data:
        mid = entry.get("model_id", "unknown")
        m = MODEL_MAP.get(mid)
        provider = m.provider if m else mid.split("/")[0] if "/" in mid else "unknown"
        msgs = entry.get("message_count", 0)
        sessions = entry.get("sessions", 0)
        in_t = entry.get("input_tokens", 0)
        out_t = entry.get("output_tokens", 0)
        if msgs == 0:
            continue
        by_provider[provider].append({
            "model_id": mid, "message_count": msgs, "sessions": sessions,
            "input_tokens": in_t, "output_tokens": out_t,
        })
        total_msgs += msgs
        total_sessions += sessions
        total_in += in_t
        total_out += out_t

    if total_msgs == 0:
        print(f"\n  {DIM}No usage recorded yet.{RESET}\n")
        return

    sorted_providers = sorted(by_provider.items(),
                              key=lambda x: sum(e["message_count"] for e in x[1]),
                              reverse=True)

    line_w = 65
    print(f"\n  {BOLD}Usage Statistics (all time){RESET}")
    print(f"  {DIM}{'─' * line_w}{RESET}")

    import re
    for provider, entries in sorted_providers:
        color = get_color(provider)
        print(f"  {color}{provider}{RESET}")
        entries.sort(key=lambda e: e["message_count"], reverse=True)
        for e in entries:
            mid = e["model_id"]
            msgs = e["message_count"]
            sess = e.get("sessions", 0)
            in_t = e["input_tokens"]
            out_t = e["output_tokens"]
            m = MODEL_MAP.get(mid)
            has_tokens = in_t > 0 or out_t > 0
            sess_str = f"{sess:>3d}s" if sess else "   "
            if has_tokens and m and m.price:
                match = re.match(r'\$([0-9.]+)/\$([0-9.]+)', m.price)
                cost = None
                if match:
                    in_rate = float(match.group(1))
                    out_rate = float(match.group(2))
                    cost = (in_t * in_rate + out_t * out_rate) / 1_000_000
                row = f"{sess_str} {msgs:>5d} msgs  {fmt_tokens(in_t):>7s} in  {fmt_tokens(out_t):>7s} out"
                if cost is not None:
                    total_cost += cost
                    row += f"  ~${cost:.2f}" if cost >= 0.005 else "  ~$0.00"
            elif has_tokens:
                tot = in_t + out_t
                row = f"{sess_str} {msgs:>5d} msgs  {fmt_tokens(tot):>7s} tokens"
            else:
                row = f"{sess_str} {msgs:>5d} msgs"
            print(f"    {DIM}{mid:<40s}{RESET} {row}")
        print()

    total_parts = f"{total_sessions} sessions · {total_msgs} msgs · {fmt_tokens(total_in)} in · {fmt_tokens(total_out)} out"
    if total_cost >= 0.005:
        total_parts += f" · ~${total_cost:.2f}"
    print(f"  {BOLD}Total:{RESET} {total_parts}")

    if total_sessions > 1:
        avg_msgs = total_msgs / total_sessions
        avg_parts = f"{avg_msgs:.1f} msgs/session"
        if total_in > 0 or total_out > 0:
            avg_tokens = (total_in + total_out) / total_sessions
            avg_parts += f" · {fmt_tokens(int(avg_tokens))} tokens/session"
        if total_cost >= 0.005:
            avg_cost = total_cost / total_sessions
            avg_parts += f" · ~${avg_cost:.2f}/session"
        print(f"  {DIM}Avg:   {avg_parts}{RESET}")

    print(f"  {DIM}{'─' * line_w}{RESET}\n")


def _handle_fav(args: list[str]) -> None:
    if not args:
        die("Usage: llm-switchboard --fav add|rm|list <model-id>")
    subcmd = args[0]
    if subcmd == "add":
        if len(args) < 2:
            die("Usage: llm-switchboard --fav add <model-id>")
        msg = fav_add(args[1])
        print(msg)
    elif subcmd in ("rm", "remove"):
        if len(args) < 2:
            die("Usage: llm-switchboard --fav rm <model-id>")
        msg = fav_rm(args[1])
        print(msg)
    elif subcmd in ("list", "ls"):
        favs = fav_list()
        if not favs:
            print("No favorites. Add with: llm-switchboard --fav add <model-id>")
        else:
            print(f"{BOLD}Favorites:{RESET}")
            for n, f in enumerate(favs, 1):
                print(f"  {n}) {f}")
    else:
        die(f"Unknown --fav subcommand: {subcmd}\nUsage: --fav add|rm|list <model-id>")


def _handle_free_tier(args: list[str]) -> None:
    from .tui import Spinner
    if not args:
        die("Usage: llm-switchboard --free-tier add|rm|list|update <entry> [entry ...]\n"
            "  Entries: 'groq' (all models) or 'gemini:flash,gemma' (pattern match)\n"
            "  'update' fetches latest Gemini pricing data")
    subcmd = args[0]
    from .config import GEMINI_FREE_CACHE
    if subcmd == "update":
        print(f"\n  {BOLD}Fetching Gemini pricing data...{RESET}")
        with Spinner("Fetching pricing..."):
            try:
                free_set, paid_set = fetch_gemini_free_tier()
                write_gemini_cache(GEMINI_FREE_CACHE, free_set, paid_set)
                success = True
            except RuntimeError:
                success = False
        if success:
            cache = load_gemini_cache(GEMINI_FREE_CACHE)
            free_list = cache.get("free", []) if cache else []
            paid_list = cache.get("paid", []) if cache else []
            print(f"  {GREEN}Gemini free-tier cache updated{RESET}")
            print(f"    {len(free_list)} free, {len(paid_list)} paid")
            if free_list:
                print(f"    Free: {', '.join(sorted(free_list))}")
            if paid_list:
                print(f"    Paid: {', '.join(sorted(paid_list))}")
        else:
            die("Failed to fetch Gemini pricing data.")
        print()
        return
    if subcmd == "add":
        if len(args) < 2:
            die("Usage: llm-switchboard --free-tier add <entry> [entry ...]")
        existing = free_tier_rules_from_file(FREE_PROVIDERS_FILE)
        added = []
        for entry in args[1:]:
            prov = entry.split(":")[0].strip().lower()
            existing[prov] = entry.lower()
            added.append(entry.lower())
        if added:
            write_free_tier_config(FREE_PROVIDERS_FILE, existing)
            print(f"Added free-tier entries: {', '.join(added)}")
        else:
            print("Nothing to add.")
    elif subcmd in ("rm", "remove"):
        if len(args) < 2:
            die("Usage: llm-switchboard --free-tier rm <provider> [provider ...]")
        to_remove = {p.lower() for p in args[1:]}
        existing = free_tier_rules_from_file(FREE_PROVIDERS_FILE)
        removed = to_remove & set(existing.keys())
        if removed:
            for prov in removed:
                del existing[prov]
            write_free_tier_config(FREE_PROVIDERS_FILE, existing)
            print(f"Removed: {', '.join(sorted(removed))}")
        else:
            print("None of those providers were configured.")
    elif subcmd in ("list", "ls"):
        rules = free_tier_rules_from_file(FREE_PROVIDERS_FILE)
        if not rules:
            print("No free-tier providers configured.")
            print(f"Run: llm-switchboard --setup")
        else:
            print(f"{BOLD}Free-tier config:{RESET}")
            for prov in sorted(rules):
                rule = rules[prov]
                if rule.includes is None and not rule.excludes:
                    print(f"  {GREEN}{prov}{RESET}  {DIM}(all models){RESET}")
                elif rule.includes is None and rule.excludes:
                    print(f"  {GREEN}{prov}{RESET}  {DIM}(all except: {', '.join(rule.excludes)}){RESET}")
                elif rule.includes:
                    label = ", ".join(rule.includes)
                    if rule.excludes:
                        label += f"; except: {', '.join(rule.excludes)}"
                    print(f"  {GREEN}{prov}{RESET}  {DIM}(matching: {label}){RESET}")
            if "gemini" in rules:
                cache = load_gemini_cache(GEMINI_FREE_CACHE)
                if cache:
                    free_list = cache.get("free", [])
                    paid_list = cache.get("paid", [])
                    print(f"\n  {BOLD}Gemini pricing cache:{RESET}")
                    print(f"    Fetched: {cache.get('fetched', 'unknown')}")
                    print(f"    {len(free_list)} free, {len(paid_list)} paid")
                elif gemini_cache_stale(GEMINI_FREE_CACHE):
                    print(f"\n  {YELLOW}⚠ Gemini pricing cache is stale or missing.{RESET}")
                    print(f"    Run {BOLD}--free-tier update{RESET} to fetch latest pricing data.")
    else:
        die(f"Unknown --free-tier subcommand: {subcmd}\n"
            "Usage: --free-tier add|rm|list|update <entry>")


def _write_default_free_config() -> None:
    rules: dict[str, str] = {}
    for prov in DEFAULT_FREE_PROVIDERS:
        known = KNOWN_FREE_TIERS.get(prov)
        if known:
            rules[prov] = known[1]
        else:
            rules[prov] = prov
    write_free_tier_config(FREE_PROVIDERS_FILE, rules)
    print(f"  {DIM}Created default free-tier config ({', '.join(sorted(rules))}){RESET}")
    print(f"  {DIM}Run --setup to customize, or edit {FREE_PROVIDERS_FILE}{RESET}\n")


def run_setup() -> None:
    from .tui import Spinner, get_color, readline_input
    print(f"\n  {BOLD}llm-switchboard setup{RESET}")
    print(f"  {DIM}Detecting your providers and pricing data...{RESET}\n")

    from .tui import MODELS as tui_models
    if not tui_models:
        fetch_models()
    from .tui import MODELS

    existing_rules = free_tier_rules_from_file(FREE_PROVIDERS_FILE)
    prov_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "has_pricing": 0})
    for m in MODELS:
        if m.provider in ("ollama", "arena"):
            continue
        prov_stats[m.provider]["total"] += 1
        if m.price:
            prov_stats[m.provider]["has_pricing"] += 1
        elif m.cost == "free" and m.provider not in existing_rules:
            prov_stats[m.provider]["has_pricing"] += 1

    needs_input = []
    auto_ok = []
    for prov in sorted(prov_stats):
        s = prov_stats[prov]
        if s["has_pricing"] == s["total"] and s["total"] > 0:
            auto_ok.append(prov)
        else:
            needs_input.append(prov)

    if auto_ok:
        print(f"  {GREEN}Pricing data detected{RESET} (no config needed):")
        for p in auto_ok:
            s = prov_stats[p]
            print(f"    {get_color(p)}{p}{RESET}  {s['total']} models — pricing from API")
        print()

    if not needs_input:
        print(f"  All providers report pricing. No manual config needed.")
        print(f"  {DIM}Run --setup anytime to reconfigure.{RESET}\n")
        return

    print(f"  {YELLOW}No pricing data{RESET} from these providers:")
    for p in needs_input:
        s = prov_stats[p]
        known = KNOWN_FREE_TIERS.get(p)
        hint_str = f"  {DIM}{known[0]}{RESET}" if known else ""
        print(f"    {get_color(p)}{p}{RESET}  {s['total']} models{hint_str}")
    print()

    print(f"  Some providers offer free API plans with rate limits.")
    print(f"  If your API key is on a free plan, models will show a {GREEN}FREE{RESET} tag.\n")

    selected_lines: list[str] = []
    for prov in needs_input:
        known = KNOWN_FREE_TIERS.get(prov)
        if known:
            desc, config_val, _ = known
            extra = f" {DIM}({desc}){RESET}"
        else:
            extra = ""
            config_val = prov
        prompt = f"  Is {get_color(prov)}{BOLD}{prov}{RESET} on a free plan?{extra} [y/N] "
        sys.stdout.write(prompt)
        sys.stdout.flush()
        answer = readline_input("").strip().lower()
        if answer in ("y", "yes"):
            selected_lines.append(config_val)

    print()

    all_rules: dict[str, str] = {}
    asked_providers = set(needs_input)
    if FREE_PROVIDERS_FILE.exists():
        for line in FREE_PROVIDERS_FILE.read_text().splitlines():
            raw = line.split("#")[0].strip()
            if not raw:
                continue
            prov = raw.split(":")[0].strip().lower()
            if prov not in asked_providers:
                all_rules[prov] = raw

    for config_val in selected_lines:
        prov = config_val.split(":")[0].strip().lower()
        all_rules[prov] = config_val

    write_free_tier_config(FREE_PROVIDERS_FILE, all_rules)

    gemini_selected = any(cv.startswith("gemini") for cv in selected_lines)
    if gemini_selected:
        from .config import GEMINI_FREE_CACHE
        print(f"  Fetching Gemini pricing data for accurate free-tier detection...")
        with Spinner("Fetching pricing..."):
            try:
                free_set, paid_set = fetch_gemini_free_tier()
                write_gemini_cache(GEMINI_FREE_CACHE, free_set, paid_set)
                ok = True
            except RuntimeError:
                ok = False
        if ok:
            cache = load_gemini_cache(GEMINI_FREE_CACHE)
            n_free = len(cache.get("free", [])) if cache else 0
            n_paid = len(cache.get("paid", [])) if cache else 0
            print(f"  {GREEN}Gemini pricing cache updated{RESET} ({n_free} free, {n_paid} paid)")
            print(f"  {DIM}Run --free-tier update to refresh when pricing changes.{RESET}")
        else:
            print(f"  {YELLOW}⚠ Could not fetch pricing data.{RESET} Falling back to pattern config.")
            print(f"  {DIM}Try --free-tier update later to fetch it.{RESET}")
        print()

    final = free_tier_rules_from_file(FREE_PROVIDERS_FILE)
    if final:
        print(f"  {BOLD}Free-tier config:{RESET}")
        for prov in sorted(final):
            rule = final[prov]
            if rule.includes is None and not rule.excludes:
                detail = "(all models)"
            elif rule.excludes and rule.includes is None:
                detail = f"(all except: {', '.join(rule.excludes)})"
            elif rule.includes:
                detail = f"(matching: {', '.join(rule.includes)})"
            else:
                detail = ""
            print(f"    {GREEN}{prov}{RESET}  {DIM}{detail}{RESET}")
    else:
        print(f"  No providers marked as free-tier.")
    print(f"\n  {DIM}Run --setup anytime to reconfigure.{RESET}\n")


def show_help() -> None:
    print(f"""{BOLD}llm-switchboard{RESET} v{__version__} — interactive model picker for Claude Code via Open WebUI

{BOLD}Usage:{RESET}
  llm-switchboard [claude-args]          Interactive picker, pass args to claude
  llm-switchboard -l, --last [args]      Re-launch last used model
  llm-switchboard -f, --free [args]      Show only free models
  llm-switchboard -s, --stats            Show usage statistics across all models
  llm-switchboard --reset                Clear all usage data
  llm-switchboard --refresh [args]       Re-fetch models from all providers
  llm-switchboard --fav add|rm|list <id> Manage favorites
  llm-switchboard --setup                Guided setup for free-tier providers
  llm-switchboard --free-tier add|rm|list|update       Free-tier management
  llm-switchboard --compat-test <model-id>             Test model as coding agent
  llm-switchboard --compat-test --all                  Test all models
  llm-switchboard --compat-report [--json]             Show test results
  llm-switchboard <filter> [args]        Jump to search for "filter"
  llm-switchboard <model:id> [args]      Direct launch (model ID containing ':')
  llm-switchboard -h, --help             Show this help

{BOLD}Non-interactive:{RESET}
  llm-switchboard --list-models          List all models (text)
  llm-switchboard --list-models --json   List all models (JSON)
  llm-switchboard --list-models --free-only            Only free/local models
  llm-switchboard --list-models --agent-ok             Only agent-compatible models
  llm-switchboard --list-models --provider <name>      Filter by provider
  llm-switchboard --select <model_id> [claude-args]    Launch specific model
  llm-switchboard --favorites            List favorite model IDs
  llm-switchboard --self-test            Run internal tests and exit 0/1
  llm-switchboard --version              Show version
  llm-switchboard --print-paths          Show config/cache directory paths
  llm-switchboard --print-endpoints      Show resolved API mode and endpoints
  llm-switchboard --doctor               Run health checks and show diagnostics
  llm-switchboard --clear-detect-cache   Remove endpoint detection cache

{BOLD}API mode:{RESET}
  llm-switchboard --api-mode auto        Auto-detect (default)
  llm-switchboard --api-mode openai      Force OpenAI-compatible endpoints
  llm-switchboard --api-mode anthropic   Force Anthropic Messages endpoint
  llm-switchboard --base-url <url>       Override base URL for this invocation
  llm-switchboard --auto-prefer openai   Prefer OpenAI in auto-detect (default)
  llm-switchboard --auto-prefer anthropic  Prefer Anthropic in auto-detect

  Any unrecognized flags/arguments are passed through to claude.
  Example: llm-switchboard -p "fix the bug" --verbose

{BOLD}Environment:{RESET}
  OPENWEBUI_API_KEY  Required. API key for Open WebUI
  OPENWEBUI_URL      Base URL (default: http://127.0.0.1:3100)

{BOLD}Free-tier providers:{RESET}
  Some providers (Groq, Cerebras, Gemini) offer free API plans with rate limits.
  Since the API doesn't report this, mark your free-tier providers explicitly:
    llm-switchboard --free-tier add groq cerebras gemini
  Models from these providers will show a {GREEN}FREE{RESET} tag.

  For Gemini, run --free-tier update to fetch per-model free/paid data
  from Google's pricing page. The cache is used for 7 days.

{BOLD}Files:{RESET}
  ~/.config/llm-switchboard/favorites.conf       Favorite model IDs
  ~/.config/llm-switchboard/free_providers.conf  Free-tier provider names
  ~/.config/llm-switchboard/last_model           Last used model
  ~/.config/llm-switchboard/api_mode.conf        API mode (auto/openai/anthropic)
  ~/.cache/llm-switchboard/connections.json      Cached provider config (1hr TTL)
  ~/.cache/llm-switchboard/endpoint_detect.json  Cached endpoint detection (1hr TTL)

  Paths respect XDG_CONFIG_HOME and XDG_CACHE_HOME if set.""")


def _parse_args(argv: list[str]) -> tuple[str | None, str | None, list[str], dict]:
    """Parse llm-switchboard flags. Returns (cmd, filter, extra_args, options).

    options may contain: json, free_only, provider, select_model
    """
    options: dict = {}

    if not argv:
        return None, None, [], options

    # Pre-extract global flags: --api-mode, --base-url, --auto-prefer
    filtered = []
    i = 0
    while i < len(argv):
        if argv[i] == "--api-mode" and i + 1 < len(argv):
            options["api_mode"] = argv[i + 1]
            i += 2
        elif argv[i] == "--base-url" and i + 1 < len(argv):
            options["base_url"] = argv[i + 1]
            i += 2
        elif argv[i] == "--auto-prefer" and i + 1 < len(argv):
            options["auto_prefer"] = argv[i + 1]
            i += 2
        else:
            filtered.append(argv[i])
            i += 1
    argv = filtered

    if not argv:
        return None, None, [], options

    first = argv[0]

    # New non-interactive flags
    if first == "--version":
        return "version", None, [], options
    if first == "--self-test":
        return "self-test", None, [], options
    if first == "--print-paths":
        return "print-paths", None, [], options
    if first == "--print-endpoints":
        return "print-endpoints", None, [], options
    if first == "--clear-detect-cache":
        return "clear-detect-cache", None, [], options
    if first == "--doctor":
        return "doctor", None, [], options
    if first == "--compat-test":
        if len(argv) >= 2 and argv[1] == "--all":
            options["compat_all"] = True
            return "compat-test", None, [], options
        if len(argv) >= 2 and not argv[1].startswith("-"):
            options["compat_model"] = argv[1]
            return "compat-test", None, [], options
        die("Usage: llm-switchboard --compat-test <model-id> | --all")
    if first == "--compat-report":
        rest = argv[1:]
        if "--json" in rest:
            options["json"] = True
        return "compat-report", None, [], options
    if first == "--list-models":
        # Parse modifiers
        rest = argv[1:]
        i = 0
        while i < len(rest):
            if rest[i] == "--json":
                options["json"] = True
                i += 1
            elif rest[i] == "--free-only":
                options["free_only"] = True
                i += 1
            elif rest[i] == "--agent-ok":
                options["agent_ok"] = True
                i += 1
            elif rest[i] == "--provider" and i + 1 < len(rest):
                options["provider"] = rest[i + 1]
                i += 2
            else:
                break
        return "list-models", None, rest[i:], options
    if first == "--select":
        if len(argv) < 2:
            die("Usage: llm-switchboard --select <model_id> [claude-args]")
        options["select_model"] = argv[1]
        return "select", None, argv[2:], options
    if first == "--favorites":
        return "favorites", None, [], options

    # Legacy flags
    if first in ("-h", "--help"):
        return "help", None, [], options
    if first in ("-s", "--stats"):
        return "stats", None, [], options
    if first == "--reset":
        return "reset", None, [], options
    if first in ("-f", "--free"):
        return "free", None, argv[1:], options
    if first in ("-l", "--last"):
        return "last", None, argv[1:], options
    if first == "--refresh":
        return "refresh", None, argv[1:], options
    if first == "--fav":
        return "fav", None, argv[1:], options
    if first == "--free-tier":
        return "free-tier", None, argv[1:], options
    if first == "--setup":
        return "setup", None, argv[1:], options

    # Non-flag first word: filter or direct model ID
    if not first.startswith("-"):
        return None, first, argv[1:], options

    # Everything is claude args
    return None, None, argv, options


def main() -> None:
    # Restore default SIGPIPE so piping to head/tail exits quietly
    import signal
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    os.umask(0o077)
    args = sys.argv[1:]
    cmd, filt, claude_args, options = _parse_args(args)

    # Apply global flags
    global _CLI_API_MODE, _CLI_BASE_URL, _CLI_AUTO_PREFER
    if "api_mode" in options:
        mode_val = options["api_mode"].lower()
        if mode_val not in VALID_MODES:
            die(f"Invalid --api-mode: {options['api_mode']}. Must be one of: {', '.join(sorted(VALID_MODES))}")
        _CLI_API_MODE = mode_val
    if "base_url" in options:
        _CLI_BASE_URL = normalize_url(options["base_url"])
    if "auto_prefer" in options:
        pref_val = options["auto_prefer"].lower()
        if pref_val not in VALID_PREFERENCES:
            die(f"Invalid --auto-prefer: {options['auto_prefer']}. Must be one of: {', '.join(sorted(VALID_PREFERENCES))}")
        _CLI_AUTO_PREFER = pref_val

    # Commands that don't need API key
    if cmd == "help":
        show_help()
        sys.exit(0)

    if cmd == "version":
        print(f"llm-switchboard {__version__}")
        sys.exit(0)

    if cmd == "self-test":
        _run_self_test()
        return

    if cmd == "print-paths":
        print(f"Config dir:  {CONFIG_DIR}")
        print(f"Cache dir:   {CACHE_DIR}")
        sys.exit(0)

    if cmd == "print-endpoints":
        ep = _resolve_api()
        print(format_endpoint_info(ep))
        sys.exit(0)

    if cmd == "clear-detect-cache":
        cmd_clear_detect_cache()
        sys.exit(0)

    if cmd == "doctor":
        sys.exit(cmd_doctor())

    if cmd == "fav":
        ensure_dirs()
        _handle_fav(claude_args)
        sys.exit(0)

    if cmd == "favorites":
        cmd_favorites()
        sys.exit(0)

    if cmd == "compat-report":
        cmd_compat_report(as_json=options.get("json", False))
        sys.exit(0)

    if cmd == "free-tier":
        ensure_dirs()
        _handle_free_tier(claude_args)
        sys.exit(0)

    if cmd == "stats":
        if not OPENWEBUI_KEY:
            die("OPENWEBUI_API_KEY is not set.\nGenerate one in Open WebUI: Settings > Account > API Keys")
        ensure_dirs()
        fetch_models()
        run_stats()
        sys.exit(0)

    if cmd == "reset":
        ensure_dirs()
        run_reset()
        sys.exit(0)

    if cmd == "setup":
        if not OPENWEBUI_KEY:
            die("OPENWEBUI_API_KEY is not set.\nGenerate one in Open WebUI: Settings > Account > API Keys")
        ensure_dirs()
        run_setup()
        sys.exit(0)

    if cmd == "compat-test":
        if not OPENWEBUI_KEY:
            die("OPENWEBUI_API_KEY is not set.\nGenerate one in Open WebUI: Settings > Account > API Keys")
        ensure_dirs()
        if options.get("compat_all"):
            cmd_compat_test_all()
        else:
            cmd_compat_test(options["compat_model"])
        sys.exit(0)

    # Non-interactive modes
    if cmd == "list-models":
        if not OPENWEBUI_KEY:
            die("OPENWEBUI_API_KEY is not set.\nGenerate one in Open WebUI: Settings > Account > API Keys")
        ensure_dirs()
        cmd_list_models(
            as_json=options.get("json", False),
            free_only=options.get("free_only", False),
            provider_filter=options.get("provider", ""),
            agent_ok=options.get("agent_ok", False),
        )
        sys.exit(0)

    if cmd == "select":
        if not OPENWEBUI_KEY:
            die("OPENWEBUI_API_KEY is not set.\nGenerate one in Open WebUI: Settings > Account > API Keys")
        ensure_dirs()
        cmd_select(options["select_model"], claude_args)
        return  # cmd_select calls sys.exit

    if not OPENWEBUI_KEY:
        die("OPENWEBUI_API_KEY is not set.\nGenerate one in Open WebUI: Settings > Account > API Keys")

    if not PRIVATE_IP_RE.search(OPENWEBUI_URL) and not OPENWEBUI_URL.startswith("https://"):
        print(f"  {YELLOW}⚠ OPENWEBUI_URL is remote but not HTTPS. API key may be exposed.{RESET}",
              file=sys.stderr)

    ensure_dirs()
    first_run = not FREE_PROVIDERS_FILE.exists()

    if first_run and not IS_STDIN_TTY:
        _write_default_free_config()

    from .tui import run_interactive, launch_claude, has_fzf, run_fzf_search

    if cmd == "refresh":
        do_refresh()
        fetch_models()
        if first_run and IS_STDIN_TTY:
            run_setup()
        run_interactive(extra_args=claude_args)

    elif cmd == "last":
        if first_run:
            _write_default_free_config()
        last = last_model_read()
        if not last:
            die("No last model saved. Run llm-switchboard first.")
        fetch_models()
        sys.exit(launch_claude(last, claude_args))

    elif cmd == "free":
        fetch_models()
        run_interactive("free", extra_args=claude_args)

    elif filt is not None:
        if first_run:
            _write_default_free_config()
        if ":" in filt:
            fetch_models()
            sys.exit(launch_claude(filt, claude_args))
        else:
            fetch_models()
            if has_fzf():
                pick = run_fzf_search(filt)
                if pick:
                    sys.exit(launch_claude(pick, claude_args))
                else:
                    sys.exit(0)
            else:
                run_interactive("search", search_query=filt, extra_args=claude_args)

    else:
        fetch_models()
        if first_run and IS_STDIN_TTY:
            run_setup()
        run_interactive(extra_args=claude_args)


def _run_self_test() -> None:
    """Run the internal test suite."""
    import unittest
    # Discover tests from the tests/ directory relative to this package
    pkg_dir = Path(__file__).resolve().parent.parent
    test_dir = pkg_dir / "tests"
    if not test_dir.is_dir():
        # Fallback: try relative to cwd
        test_dir = Path("tests")
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern="test_*.py", top_level_dir=str(pkg_dir))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

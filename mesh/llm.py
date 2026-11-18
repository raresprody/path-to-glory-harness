"""Gateway client for the only allowed model (gateway `haiku`), stdlib only.

- Always uses the key from the latest `prep`/`start` message (rules 5.12, 7.2): call `configure(msg)`.
- Thinking disabled on every call (rule 7.3); every call has a timeout (7.4); errors, empty and cut
  answers come back as "" so callers always fall back to something safe.
- urllib honours HTTP(S)_PROXY / NO_PROXY from the environment (7.9); the gateway is in NO_PROXY.
- Tracks token usage so we can log spend (cost points, rule 9.3).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request

_CFG = {
    "base_url": os.environ.get("LLM_BASE_URL", "http://gateway:4000/v1"),
    "api_key": os.environ.get("LLM_API_KEY", ""),
    "model": os.environ.get("LLM_MODEL", "haiku"),
}
_LOCK = threading.Lock()
USAGE = {"calls": 0, "errors": 0, "in": 0, "cached": 0, "out": 0, "seconds": 0.0}
# after several consecutive failures (key out of budget, gateway down) we stop waiting on the model
_FAILS = {"streak": 0, "until": 0.0}


def log(*parts):
    print("[mesh]", *parts, file=sys.stderr, flush=True)


def configure(msg: dict) -> None:
    info = msg.get("llm") or {}
    with _LOCK:
        changed = False
        for k in ("base_url", "api_key", "model"):
            if info.get(k) and info[k] != _CFG[k]:
                _CFG[k] = info[k]
                changed = True
        if changed:  # a fresh key: forget the failure streak of the old one
            _FAILS["streak"], _FAILS["until"] = 0, 0.0


def model_name() -> str:
    # The only allowed model is the gateway's haiku (rule 7.1). We never send anything else.
    return "haiku"


def cost_usd() -> float:
    return USAGE["in"] * 0.10 / 1e6 + USAGE["cached"] * 0.01 / 1e6 + USAGE["out"] * 0.50 / 1e6


CACHE_MIN_CHARS = 6000  # ~1500+ tokens: below that the gateway's model won't cache anyway


def _system_block(system: str):
    """Long system prompts (the rules + playbook, identical on every turn of a game) are marked cacheable: the gateway
    then bills them as cache reads ($0.01/M instead of $0.10/M input, rule 7.5), which cuts the match cost (9.3).
    Measured on the gateway: a 5.4k-token system prompt is read from cache on the second call."""
    if len(system) >= CACHE_MIN_CHARS:
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    return system


def available() -> bool:
    return time.time() >= _FAILS["until"]


def chat(system: str, user: str, timeout: float, max_tokens: int = 400, temperature: float | None = None,
         thinking_budget: int | None = None) -> str:
    """One model call. Returns the text, or "" on any failure. Never raises."""
    if timeout < 1.0 or not available():
        return ""
    # an answer cut by max_tokens loses its JSON and with it the move (seen in practice): output is billed per token
    # actually written, so a generous ceiling costs nothing extra; prompts ask for short answers anyway
    max_tokens = max(int(max_tokens), 1000)
    with _LOCK:
        base, key = _CFG["base_url"].rstrip("/"), _CFG["api_key"]
    body = {
        "model": model_name(),
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": _system_block(system)}, {"role": "user", "content": user}],
        "thinking": {"type": "disabled"},
    }
    if thinking_budget:
        # bounded thinking, used in prep only (prep spend and time are not scored); it can overrun its budget a lot,
        # so leave room after it (rule 7.3 note) and keep the call's timeout as the hard limit
        budget = max(1024, int(thinking_budget))
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        body["max_tokens"] = max(body["max_tokens"], budget + 1500)
        temperature = None  # thinking requires the default temperature
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content")) or ""
        usage = data.get("usage") or {}
        with _LOCK:
            USAGE["calls"] += 1
            cached = int(usage.get("cache_read_input_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            USAGE["cached"] += cached
            USAGE["in"] += max(0, int(usage.get("prompt_tokens") or 0) - cached)
            USAGE["out"] += int(usage.get("completion_tokens") or 0)
            USAGE["seconds"] += time.time() - t0
            _FAILS["streak"] = 0
        if choice.get("finish_reason") == "length":
            log(f"llm answer cut at max_tokens={max_tokens}")
        return text if isinstance(text, str) else ""
    except Exception as exc:  # timeout, HTTP 4xx/5xx (budget exhausted), bad JSON...
        with _LOCK:
            USAGE["errors"] += 1
            # Only a DEAD key (401/402/403: revoked or out of budget) pauses model calls, for 60 s, so we stop
            # hammering it and the fallbacks answer at once. Timeouts and 429/5xx never pause: the next turn tries
            # again, so the opponent's chat keeps reaching a model call (rule 6.5). A new key resets it (configure).
            code = getattr(exc, "code", None)
            if code in (401, 402, 403):
                _FAILS["streak"] += 1
                if _FAILS["streak"] >= 3:
                    _FAILS["until"] = time.time() + 60.0
        log(f"llm error after {time.time() - t0:.1f}s: {type(exc).__name__}: {str(exc)[:200]}")
        return ""


def extract_json(text: str):
    """First JSON object in a model answer (tolerates ```json fences and chatter). None if absent."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return salvage_json(text)


def salvage_json(text: str):
    """A JSON object cut off mid-way (max_tokens, timeout): keep every complete "key": value pair before the cut,
    so a move written before the cut is not lost. None if nothing usable."""
    import re
    i = text.find("{")
    if i < 0:
        return None
    out = {}
    for m in re.finditer(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false|null)', text[i:]):
        try:
            out[m.group(1)] = json.loads(m.group(2))
        except ValueError:
            pass
    return out or None

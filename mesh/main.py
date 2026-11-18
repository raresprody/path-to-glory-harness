"""Protocol shell (rulebook section 5). Never loses a game by accident:

- stdout carries protocol lines only: the real stdout is kept private and sys.stdout points to stderr, so a stray
  print anywhere (our code, generated code, a library) can never corrupt an answer (rule 5.1).
- Nothing is printed at startup; the process waits for the first message (5.2).
- Every answer comes from a watchdog: the brain works in a worker thread and keeps a "best answer so far" up to
  date; the shell sends it when the brain is done or at the deadline, whichever comes first. If the brain produced
  nothing, a legal fallback move is sent. A late or blank answer cannot happen by design (8.1, 8.2).
- `match_over` is not exit: only `shutdown` or a closed stdin ends the process (5.10).
- The model key from the latest `prep`/`start` is installed before anything else (5.12, 7.2).
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time

_OUT = sys.stdout
sys.stdout = sys.stderr  # anything printed by mistake goes to the logs, never to the arena

from . import llm  # noqa: E402
from .api import Best  # noqa: E402

TURN_MARGIN = float(os.environ.get("MESH_TURN_MARGIN", "3.0"))    # seconds kept in hand on a 20 s move clock
START_MARGIN = float(os.environ.get("MESH_START_MARGIN", "4.0"))
PREP_MARGIN = float(os.environ.get("MESH_PREP_MARGIN", "30.0"))
log = llm.log
_SEND_LOCK = threading.Lock()


def send(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    with _SEND_LOCK:
        _OUT.write(line + "\n")
        _OUT.flush()


def _run(target, *args):
    """Run a brain hook in a daemon thread; exceptions are logged, never fatal."""
    def wrap():
        try:
            target(*args)
        except BaseException as exc:  # noqa: BLE001 - nothing may kill the shell
            log(f"brain error in {getattr(target, '__name__', target)}: {type(exc).__name__}: {exc}")
        finally:
            for a in args:
                if isinstance(a, Best):
                    a.done.set()
    th = threading.Thread(target=wrap, daemon=True)
    th.start()
    return th


def _make_brain():
    try:
        from .brain import Brain
        return Brain()
    except BaseException as exc:  # noqa: BLE001
        log(f"brain failed to load ({type(exc).__name__}: {exc}); playing random legal moves")
        return None


def _chat_or_default(chat) -> str:
    chat = chat if isinstance(chat, str) else ""
    chat = " ".join(chat.split())[:200]
    return chat or "Ready."  # content-free default: no word a speech rule could ban


_RULES = {"text": ""}  # rules of the current game, for the no-brain fallback


def _examples_from_rules(rules: str) -> list:
    """Quoted examples near the action-format lines of the rules (no model, no brain needed)."""
    import re
    out, lines = [], (rules or "").splitlines()
    for i, line in enumerate(lines):
        if re.search(r"example|action format|reply with|answer with|format", line, re.I):
            for chunk in lines[i:i + 3]:
                for m in re.finditer("[\"“`]([^\"”`\n]{1,30})[\"”`]", chunk):
                    if m.group(1).strip() and m.group(1).strip() not in out:
                        out.append(m.group(1).strip())
    return out


def _fallback(brain, msg: dict) -> str:
    legal = msg.get("legal_actions")
    if legal:
        return random.choice(legal)
    try:
        a = brain.fallback_action(msg) if brain else None
        if isinstance(a, str) and a.strip() and a != "pass":
            return a
    except BaseException:  # noqa: BLE001
        pass
    try:
        from . import fallback
        alt = fallback.pick(_RULES["text"], msg.get("observation", ""), set(_REJECTED))
        if alt:
            return alt
    except BaseException:  # noqa: BLE001
        pass
    rejected = str((msg.get("warning") or {}).get("rejected") or "").strip().lower()
    for ex in _examples_from_rules(_RULES["text"]):
        if ex.lower() != rejected:
            return ex
    return "pass"


def handle_prep(brain, msg: dict, t0: float) -> None:
    llm.configure(msg)
    deadline = t0 + max(5.0, float(msg.get("seconds", 600)) - PREP_MARGIN)
    if brain:
        th = _run(brain.on_prep, msg, deadline)
        th.join(max(0.0, deadline - time.monotonic()))
        if th.is_alive():
            log("prep: deadline reached, answering ready anyway")
    send({"type": "ready"})


def handle_start(brain, msg: dict, t0: float) -> None:
    llm.configure(msg)
    _RULES["text"] = str(msg.get("rules") or "")
    _REJECTED.clear()  # warnings reset at every game (rule 8.3)
    deadline = t0 + max(2.0, float(msg.get("study_seconds", 30)) - START_MARGIN)
    best = Best()
    if brain:
        _run(brain.on_start, msg, deadline, best)
        best.done.wait(max(0.0, deadline - time.monotonic()))
    _, chat = best.get()
    send({"type": "ready", "chat": _chat_or_default(chat)})


_REJECTED: set = set()  # actions the arena rejected in the current game (rule 8.1: a second bad answer loses it)


def _n(a) -> str:
    return " ".join(str(a).strip().strip("\"'`.!").lower().split())


def _final_gate(brain, msg: dict, action: str) -> str:
    """Last check before anything leaves: never send a move outside the arena's legal list, never repeat a move the
    arena already rejected in this game. Whatever the brain did, this alone keeps us inside rule 8.1."""
    w = msg.get("warning") or {}
    if w.get("rejected") is not None:
        _REJECTED.add(_n(w.get("rejected")))
    legal = msg.get("legal_actions")
    if legal:
        by = {_n(a): a for a in legal}
        if _n(action) in by and _n(action) not in _REJECTED:
            return by[_n(action)]
        pool = [a for a in legal if _n(a) not in _REJECTED] or list(legal)
        log(f"final gate: {action!r} not in the legal list; sending a legal move instead")
        return random.choice(pool)
    if _n(action) in _REJECTED or not str(action).strip():
        alt = _fallback(brain, msg)
        if _n(alt) in _REJECTED:
            alt = next((e for e in _examples_from_rules(_RULES["text"]) if _n(e) not in _REJECTED), alt)
        log(f"final gate: {action!r} was rejected before in this game; sending {alt!r}")
        return alt
    return action


def handle_turn(brain, msg: dict, t0: float) -> None:
    deadline = t0 + max(1.5, float(msg.get("seconds", 20)) - TURN_MARGIN)
    best = Best()
    if brain:
        _run(brain.on_turn, msg, deadline, best)
        best.done.wait(max(0.0, deadline - time.monotonic()))
    action, chat = best.get()
    if not (isinstance(action, str) and action.strip()):
        action = _fallback(brain, msg)
        log("turn: brain gave no action, sent fallback")
    if not best.done.is_set():
        log(f"turn {msg.get('turn')}: watchdog fired, sent best-so-far {action!r}")
    action = _final_gate(brain, msg, action)
    send({"type": "action", "action": action, "chat": _chat_or_default(chat)})
    log(f"turn {msg.get('turn')}: {action!r} in {time.monotonic() - t0:.1f}s")


def main() -> None:
    brain = _make_brain()
    stdin = sys.stdin.buffer
    while True:
        raw = stdin.readline()
        t0 = time.monotonic()
        if not raw:
            break  # stdin closed: same as shutdown
        try:
            msg = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(msg, dict):
                raise ValueError("not an object")
        except ValueError:
            log("ignoring a line that is not a JSON object:", raw[:200])
            continue
        kind = msg.get("type")
        try:
            if kind == "prep":
                handle_prep(brain, msg, t0)
            elif kind == "start":
                handle_start(brain, msg, t0)
            elif kind == "turn":
                handle_turn(brain, msg, t0)
            elif kind == "end":
                if brain:
                    _run(brain.on_end, msg).join(2.0)
                if msg.get("match_over"):
                    log("match over; waiting for the next start")
            elif kind == "shutdown":
                break
            else:
                log(f"ignoring message type {kind!r}")
        except BaseException as exc:  # noqa: BLE001 - last line of defence: still answer
            log(f"shell error on {kind}: {type(exc).__name__}: {exc}")
            if kind == "turn":
                send({"type": "action", "action": _fallback(brain, msg), "chat": "Ready."})
            elif kind in ("prep", "start"):
                send({"type": "ready", "chat": "Ready."} if kind == "start" else {"type": "ready"})
    if brain:
        _run(brain.on_shutdown, time.monotonic() + 10.0).join(10.0)
    log("bye")


if __name__ == "__main__":
    main()

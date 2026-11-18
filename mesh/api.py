"""The contract between the protocol shell (`harness/`, path-to-glory-3) and the brain (`mesh/`, path-to-glory-1).
See docs/DESIGN.md section "Interface". Keep this file tiny and stable: both halves import it.
"""
from __future__ import annotations

import threading
import time


class Best:
    """Thread-safe holder for 'the best answer found so far'. The brain writes it as often as it likes;
    the shell's watchdog reads it at the deadline and sends it. Writing early (a legal fallback first,
    better moves later) is what makes a blank or late answer impossible."""

    def __init__(self):
        self._lock = threading.Lock()
        self._action = None
        self._chat = ""
        self.done = threading.Event()  # brain sets it when it has its final answer (shell may send at once)

    def set(self, action, chat: str | None = None, final: bool = False) -> None:
        with self._lock:
            if isinstance(action, str) and action.strip():
                self._action = action
            if isinstance(chat, str):
                self._chat = chat[:200]
        if final:
            self.done.set()

    def get(self):
        with self._lock:
            return self._action, self._chat


def deadline_in(seconds: float, margin: float) -> float:
    """Absolute time.monotonic() deadline for work that must finish `margin` s before a clock of `seconds`."""
    return time.monotonic() + max(0.5, float(seconds) - margin)


class BrainAPI:
    """What the shell calls. All methods must return before `deadline` (time.monotonic()); the shell
    still guards with its own watchdog. None of them may write to stdout."""

    def on_prep(self, msg: dict, deadline: float) -> None:  # shell then sends {"type":"ready"}
        ...

    def on_start(self, msg: dict, deadline: float, best: Best) -> None:  # best.set(None, chat) = ready chat
        ...

    def on_turn(self, msg: dict, deadline: float, best: Best) -> None:  # best.set(action, chat, final=True)
        ...

    def on_end(self, msg: dict) -> None:
        ...

    def on_shutdown(self, deadline: float) -> None:  # save to /memory
        ...

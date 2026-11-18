"""Codebreaker (holdout test game, Guessing family). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: hidden private state shown only in the observation, a huge action space (legal list always null),
feedback that must be tracked across turns, and private info that must never leak in chat.
"""
from __future__ import annotations

import re

from ..core import Game


def score(secret: str, guess: str) -> tuple[int, int]:
    exact = sum(a == b for a, b in zip(secret, guess))
    common = sum(min(secret.count(c), guess.count(c)) for c in set(guess))
    return exact, common - exact


class Codebreaker(Game):
    id = "x_codebreak"
    name = "Codebreaker"
    category = "guess"
    blurb = "Crack the opponent's secret code before they crack yours."
    howto = "Each player hides a code of letters. Players take turns guessing the other's code and get exact/near feedback. First to crack it wins."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "4 letters from A-F, 10 guesses", "length": 4, "letters": "ABCDEF", "guesses": 10},
            {"label": "5 letters from A-H, 12 guesses", "length": 5, "letters": "ABCDEFGH", "guesses": 12},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.n = int(self.variant.get("length", 4))
        self.letters = str(self.variant.get("letters", "ABCDEF"))
        self.cap = int(self.variant.get("guesses", 10))
        self.secret = ["".join(self.rng.choice(self.letters) for _ in range(self.n)) for _ in (0, 1)]
        self.guesses: list[list[tuple[str, int, int]]] = [[], []]  # guesses made BY seat s at the other's code
        self.current = 0
        self._result = None
        self._re = re.compile(rf"^guess\s+([{self.letters.lower()}]{{{self.n}}})$")

    def rules(self) -> str:
        return f"""CODEBREAKER - codes of {self.n} letters from {self.letters} (repeats allowed), {self.cap} guesses each.

SETUP
Each player gets a secret code (you see yours in your observation; never tell it to your opponent).

TURNS
Players alternate; seat 0 guesses first. On your turn guess the OPPONENT's code.
Feedback after each guess: "exact" = right letter in the right place, "near" = right letter in the wrong place
(each letter of the code counts at most once).

ACTION FORMAT
guess <{self.n} letters>      for example: guess {self.letters[:self.n]}
Only the letters {self.letters}. There is never a list of legal actions: build the move from this format.
Anything else is illegal (first time: a warning and the same turn again; second time: you lose the game).

HOW THE GAME ENDS
At once when a guess is all exact: that player wins. If both used all {self.cap} guesses, the player with the
most exact letters in a single guess wins; equal is a draw.

WHAT YOU OBSERVE
Your own secret code, every guess you made with its feedback, and every guess the opponent made at your code.
"""

    def to_move(self):
        return [] if self._result else [self.current]

    def legal_actions(self, seat):
        return [] if self._result or seat != self.current else None

    def normalize(self, seat, action):
        if self._result or seat != self.current:
            return None
        m = self._re.match(self.clean(action))
        return f"guess {m.group(1).upper()}" if m else None

    def apply(self, actions):
        if self._result:
            return
        seat = self.current
        a = self.normalize(seat, actions.get(seat, ""))
        if a is None:
            raise ValueError(f"illegal action {actions.get(seat)!r}")
        code = a.split()[1]
        ex, near = score(self.secret[1 - seat], code)
        self.guesses[seat].append((code, ex, near))
        if ex == self.n:
            self._finish(seat, f"seat {seat} cracked the code in {len(self.guesses[seat])} guesses")
        elif len(self.guesses[0]) >= self.cap and len(self.guesses[1]) >= self.cap:
            b = [max(g[1] for g in self.guesses[s]) for s in (0, 1)]
            self._finish(None if b[0] == b[1] else (0 if b[0] > b[1] else 1),
                         f"guess cap reached; best exact {b[0]} vs {b[1]}")
        else:
            self.current = 1 - seat

    def _finish(self, w, reason):
        scores = [0.5, 0.5] if w is None else [1 if w == 0 else 0, 1 if w == 1 else 0]
        self._result = {"winner": w, "scores": scores, "reason": reason}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        lines = [f"CODEBREAKER ({self.n} letters from {self.letters}). You are seat {seat}.",
                 f"Your secret code (keep it secret): {self.secret[seat]}",
                 f"Your guesses at the opponent's code ({len(self.guesses[seat])} of {self.cap}):"]
        lines += [f"  {g}: exact {e}, near {n}" for g, e, n in self.guesses[seat]] or ["  (none yet)"]
        lines.append(f"Opponent's guesses at your code ({len(self.guesses[o])} of {self.cap}):")
        lines += [f"  {g}: exact {e}, near {n}" for g, e, n in self.guesses[o]] or ["  (none yet)"]
        if not self._result:
            lines.append(f"It is YOUR turn: guess <{self.n} letters>" if self.current == seat else "Waiting for the opponent.")
        return "\n".join(lines)

    def frame(self):
        return {"secret": list(self.secret), "guesses": [[list(g) for g in gs] for gs in self.guesses],
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return "guess " + "".join(rng.choice(self.letters) for _ in range(self.n))

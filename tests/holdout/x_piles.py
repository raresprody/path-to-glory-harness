"""Stone Piles (holdout test game, Think-ahead family). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: a two-argument action ("take <pile> <count>"), a legal list that is null in the big variant (> 50 moves),
and a misere ending that flips the usual goal.
"""
from __future__ import annotations

import re

from ..core import Game

_ACT = re.compile(r"^take\s+([a-z])\s*(\d+)$")


class StonePiles(Game):
    id = "x_piles"
    name = "Stone Piles"
    category = "think"
    blurb = "Take stones from one pile at a time; read the ending rule carefully."
    howto = "Players alternate taking stones from a single pile. Who takes the last stone wins, or loses in the misere variant."
    duplicate_safe = True

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "3 piles, last stone wins", "piles": [5, 7, 9], "misere": False, "max_take": 0},
            {"label": "4 big piles, take at most 9, last stone LOSES", "piles": [13, 16, 19, 22], "misere": True, "max_take": 9},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.piles = list(self.variant.get("piles") or [5, 7, 9])
        self.misere = bool(self.variant.get("misere"))
        self.max_take = int(self.variant.get("max_take") or 0)
        self.current = 0
        self.history: list[tuple[int, str]] = []
        self._result = None

    def _names(self):
        return [chr(ord("A") + i) for i in range(len(self.piles))]

    def rules(self) -> str:
        cap = f"at most {self.max_take} stones" if self.max_take else "any number of stones (at least 1)"
        end = ("The player who takes the LAST stone LOSES (misere)." if self.misere
               else "The player who takes the last stone WINS.")
        return f"""STONE PILES - {len(self.piles)} piles: {', '.join(f'{n}={c}' for n, c in zip(self._names(), self.piles))}.

TURNS
Players alternate; seat 0 moves first. On your turn take {cap} from ONE pile that is not empty.

ACTION FORMAT
take <pile letter> <count>      for example: take B 3
The pile letter is A, B, C, ... as listed above. Anything else is illegal (first time: a warning and the same
turn again; second time: you lose the game). When there are more than 50 possible moves you get no list of
legal actions: build the move from this format.

HOW THE GAME ENDS
When every pile is empty. {end}

WHAT YOU OBSERVE
The size of every pile, whose turn it is, and all moves so far.
"""

    def to_move(self):
        return [] if self._result else [self.current]

    def _moves(self):
        out = []
        for name, size in zip(self._names(), self.piles):
            top = min(size, self.max_take) if self.max_take else size
            out += [f"take {name} {k}" for k in range(1, top + 1)]
        return out

    def legal_actions(self, seat):
        if self._result or seat != self.current:
            return []
        return self.cap_list(self._moves())

    def normalize(self, seat, action):
        if self._result or seat != self.current:
            return None
        m = _ACT.match(self.clean(action))
        if not m:
            return None
        i, k = ord(m.group(1)) - ord("a"), int(m.group(2))
        if not (0 <= i < len(self.piles)) or k < 1 or k > self.piles[i] or (self.max_take and k > self.max_take):
            return None
        return f"take {self._names()[i]} {k}"

    def apply(self, actions):
        if self._result:
            return
        seat = self.current
        a = self.normalize(seat, actions.get(seat, ""))
        if a is None:
            raise ValueError(f"illegal action {actions.get(seat)!r}")
        _, name, k = a.split()
        self.piles[ord(name) - ord("A")] -= int(k)
        self.history.append((seat, a))
        if sum(self.piles) == 0:
            w = (1 - seat) if self.misere else seat
            self._result = {"winner": w, "scores": [1 if w == 0 else 0, 1 if w == 1 else 0],
                            "reason": f"seat {seat} took the last stone ({'misere: loses' if self.misere else 'wins'})"}
        else:
            self.current = 1 - seat

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        lines = [f"STONE PILES. You are seat {seat}. " + ("Last stone LOSES." if self.misere else "Last stone WINS."),
                 "Piles: " + ", ".join(f"{n}={c}" for n, c in zip(self._names(), self.piles))]
        if self.max_take:
            lines.append(f"You may take at most {self.max_take} stones per turn.")
        lines.append("Moves so far:" if self.history else "Moves so far: (none)")
        for s, a in self.history:
            lines.append(f"  {'you' if s == seat else 'opponent'}: {a}")
        if not self._result:
            lines.append("It is YOUR turn: take <pile letter> <count>" if self.current == seat else "Waiting for the opponent.")
        return "\n".join(lines)

    def frame(self):
        return {"piles": list(self.piles), "history": [list(h) for h in self.history], "over": self.is_over(),
                "result": self._result}

    def random_action(self, seat, rng):
        return rng.choice(self._moves())

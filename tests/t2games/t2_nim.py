"""Nim (local copy of a T2 game, our approximation; tests only). Owner: path-to-glory-3.

Variants: normal play (last to take wins) on 3-4-5; misere (last to take LOSES) on 1-3-5-7; a one-pile subtraction
game (21 stones, take 1 to 3). Perfect information, no randomness.
"""
from __future__ import annotations

import re

from ..core import Game

_TAKE = re.compile(r"^(?:take\s+)?(\d+)(?:\s+(?:from|of)?\s*(?:pile\s+)?(\d+))?$")


class T2Nim(Game):
    id = "t2_nim"
    name = "Nim"
    category = "think"
    blurb = "Take stones from the piles; the whole game is one hidden piece of arithmetic."
    howto = "Players take turns removing stones from one pile. In normal play whoever takes the last stone wins."
    duplicate_safe = True

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "piles 3-4-5, last stone wins", "piles": [3, 4, 5], "misere": False, "max_take": None},
                {"label": "piles 1-3-5-7, misere (last stone loses)", "piles": [1, 3, 5, 7], "misere": True, "max_take": None},
                {"label": "one pile of 21, take 1-3, last stone wins", "piles": [21], "misere": False, "max_take": 3}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.piles = list(self.variant.get("piles") or [3, 4, 5])
        self.misere = bool(self.variant.get("misere"))
        self.max_take = self.variant.get("max_take")
        self.turn = 0
        self.history: list[tuple[int, int, int]] = []
        self._result = None

    def rules(self) -> str:
        cap = f"from 1 up to {self.max_take} stones (and no more than the pile holds)" if self.max_take else "any number of stones from 1 up to the whole pile"
        end = ("The player who takes the LAST stone LOSES (misere)." if self.misere
               else "The player who takes the LAST stone WINS.")
        one = len(self.piles) == 1
        return f"""NIM - {self.variant.get('label')}. 2 players, alternating turns, player 0 moves first.

THE PILES
{'One pile of ' + str(self.piles[0]) + ' stones.' if one else 'Piles (numbered 1 to ' + str(len(self.piles)) + ' from left to right): ' + ', '.join(map(str, self.piles)) + ' stones.'}

EACH TURN
Remove {cap} from ONE pile. Passing is not allowed.

HOW THE GAME ENDS
{end} There are no draws.

ACTION FORMAT
take <n> from <pile>        for example: take 2 from {1 if one else 3}
The pile number is 1-based. The list of legal moves is sent every turn. Anything else is illegal (first time: a
warning and the same turn again; second time: you lose the game).
"""

    def to_move(self):
        return [] if self._result else [self.turn]

    def _moves(self):
        out = []
        for i, p in enumerate(self.piles):
            top = min(p, self.max_take) if self.max_take else p
            out += [f"take {n} from {i + 1}" for n in range(1, top + 1)]
        return out

    def legal_actions(self, seat):
        if self._result or seat != self.turn:
            return []
        return self.cap_list(self._moves())

    def normalize(self, seat, action):
        if self._result or seat != self.turn:
            return None
        m = _TAKE.match(self.clean(action))
        if not m:
            return None
        n, p = int(m.group(1)), int(m.group(2) or (1 if len(self.piles) == 1 else 0))
        a = f"take {n} from {p}"
        return a if a in self._moves() else None

    def apply(self, actions):
        if self._result:
            return
        s = self.turn
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        _, n, _, p = a.split()
        n, p = int(n), int(p)
        self.piles[p - 1] -= n
        self.history.append((s, n, p))
        if sum(self.piles) == 0:
            w = 1 - s if self.misere else s
            self._result = {"winner": w, "scores": [1 - w, w],
                            "reason": f"seat {s} took the last stone ({'misere: they lose' if self.misere else 'they win'})"}
            return
        self.turn = 1 - s

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        L = [f"NIM - {self.variant.get('label')}. You are player {seat}.",
             "Piles now: " + ", ".join(f"pile {i + 1} = {p}" for i, p in enumerate(self.piles)) + f" (total {sum(self.piles)}).",
             "Moves so far: " + (", ".join(f"{'you' if s == seat else 'opponent'} took {n} from {p}" for s, n, p in self.history) or "none")]
        if self._result:
            L.append(f"Game over: {self._result['reason']}.")
        elif seat == self.turn:
            L.append('Your turn: "take <n> from <pile>".')
        else:
            L.append("Opponent's turn.")
        return "\n".join(L)

    def frame(self):
        return {"piles": list(self.piles), "history": [list(h) for h in self.history], "turn": self.turn,
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return rng.choice(self._moves())

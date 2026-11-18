"""Sealed Bids (holdout test game, Deals/Bluff family, simultaneous). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: simultaneous moves, a numeric action whose legal range shrinks (budget), a null legal list, hidden
future lot values, and a scoring rule where unspent money is worth nothing.
"""
from __future__ import annotations

import re

from ..core import Game

_ACT = re.compile(r"^bid\s+(\d+)$")


class SealedBids(Game):
    id = "x_bids"
    name = "Sealed Bids"
    category = "deals"
    blurb = "Bid in secret for lots of different value; money left over is worth nothing."
    howto = "Each round both players secretly bid for one lot. The higher bid wins the lot and pays; a tie means nobody gets it."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "5 lots, budget 100", "lots": 5, "budget": 100, "lo": 5, "hi": 40},
            {"label": "7 lots, budget 60, values 1-30", "lots": 7, "budget": 60, "lo": 1, "hi": 30},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.n = int(self.variant.get("lots", 5))
        self.budget = [int(self.variant.get("budget", 100))] * 2
        self.values = [self.rng.randint(int(self.variant.get("lo", 5)), int(self.variant.get("hi", 40))) for _ in range(self.n)]
        self.won = [0, 0]
        self.log: list[tuple[int, int, int, str]] = []  # (value, bid0, bid1, outcome)
        self._result = None

    def rules(self) -> str:
        v = self.variant
        return f"""SEALED BIDS - {self.n} lots, each player starts with {self.budget[0]} coins.

EACH ROUND
One lot is up for sale; its value (between {v.get('lo', 5)} and {v.get('hi', 40)} points) is shown before you bid.
Values of later lots are unknown until they come up.
Both players bid AT THE SAME TIME, in secret. The higher bid wins the lot: the winner gets its value in points and
pays the bid from their coins. Equal bids: nobody gets the lot and nobody pays.

ACTION FORMAT
bid <whole number from 0 to your remaining coins>      for example: bid 12
There is no list of legal actions when you have more than 50 options: build the move from this format.
A bid above your coins, a negative bid or anything else is illegal (first time: a warning and the same turn
again; second time: you lose the game).

HOW THE GAME ENDS
After the last lot. The player with more points wins; coins left over are worth NOTHING. Equal points: draw.

WHAT YOU OBSERVE
The current lot and its value, both players' remaining coins and points, and every earlier lot with both bids.
"""

    def to_move(self):
        return [] if self._result else [0, 1]

    def legal_actions(self, seat):
        if self._result:
            return []
        return self.cap_list([f"bid {k}" for k in range(self.budget[seat] + 1)])

    def normalize(self, seat, action):
        if self._result:
            return None
        m = _ACT.match(self.clean(action))
        if not m or int(m.group(1)) > self.budget[seat]:
            return None
        return f"bid {int(m.group(1))}"

    def apply(self, actions):
        if self._result:
            return
        b = []
        for s in (0, 1):
            a = self.normalize(s, actions.get(s, ""))
            if a is None:
                raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
            b.append(int(a.split()[1]))
        value = self.values[len(self.log)]
        if b[0] == b[1]:
            out = "tie, nobody buys"
        else:
            w = 0 if b[0] > b[1] else 1
            self.won[w] += value
            self.budget[w] -= b[w]
            out = f"seat {w} buys"
        self.log.append((value, b[0], b[1], out))
        if len(self.log) >= self.n:
            p0, p1 = self.won
            w = None if p0 == p1 else (0 if p0 > p1 else 1)
            self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w],
                            "reason": f"{self.n} lots sold; points {p0} to {p1}"}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        lines = [f"SEALED BIDS. You are seat {seat}. Lot {min(len(self.log) + 1, self.n)} of {self.n}.",
                 f"Coins: you {self.budget[seat]}, opponent {self.budget[o]}. Points: you {self.won[seat]}, opponent {self.won[o]}."]
        if not self._result:
            lines.append(f"Lot up for sale now: value {self.values[len(self.log)]} points.")
        lines.append("Earlier lots:" if self.log else "Earlier lots: (none)")
        for i, (v, b0, b1, out) in enumerate(self.log, 1):
            mine, theirs = (b0, b1) if seat == 0 else (b1, b0)
            out = out.replace(f"seat {seat}", "you").replace(f"seat {o}", "opponent")
            lines.append(f"  Lot {i} (value {v}): you bid {mine}, opponent bid {theirs} -> {out}")
        if not self._result:
            lines.append(f"Your bid: bid <0 to {self.budget[seat]}>")
        return "\n".join(lines)

    def frame(self):
        return {"values": list(self.values), "budget": list(self.budget), "won": list(self.won),
                "log": [list(x) for x in self.log], "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return f"bid {rng.randint(0, self.budget[seat])}"

"""Battleship (local copy of a T2 game, our approximation; tests only). Owner: path-to-glory-3.

Classic shape: 10x10 with the standard fleet (5, 4, 3, 3, 2), a sunk report that names the ship, and a hit gives
another shot (a second structure the bot must notice). Variants: 10x10 classic; 8x8 (4, 3, 2, 2), one shot per turn;
7x7 (3, 3, 2) where a hit gives another shot. Fleets are placed by the arena.
"""
from __future__ import annotations

import re

from ..core import Game

_FIRE = re.compile(r"^(?:fire\s+|shoot\s+|shot\s+)?([a-j])\s*(\d{1,2})$")
NAMES = {5: "carrier", 4: "battleship", 3: "cruiser", 2: "destroyer"}


class T2Battleship(Game):
    id = "t2_battleship"
    name = "Battleship"
    category = "guess"
    blurb = "Two hidden fleets, one shot at a time: hunt with parity, then finish what you hit."
    howto = "Players take turns firing at a cell of the opponent's hidden grid. Hit, miss or sunk is announced; sink the whole fleet first to win."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "10x10 classic fleet, a hit shoots again", "size": 10, "ships": [5, 4, 3, 3, 2], "again": True},
                {"label": "8x8, ships 4-3-2-2, one shot per turn", "size": 8, "ships": [4, 3, 2, 2], "again": False},
                {"label": "7x7, ships 3-3-2, a hit shoots again", "size": 7, "ships": [3, 3, 2], "again": True}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.n = int(self.variant.get("size", 10))
        self.sizes = list(self.variant.get("ships") or [5, 4, 3, 3, 2])
        self.again = bool(self.variant.get("again"))
        self.fleets = [self._place() for _ in (0, 1)]
        self.shots: list[dict[str, str]] = [{}, {}]
        self.sunk: list[list[int]] = [[], []]  # sizes of opponent ships seat s has sunk
        self.turn, self.total = 0, 0
        self._result = None

    def _cell(self, r, c):
        return f"{'ABCDEFGHIJ'[c]}{r + 1}"

    def _place(self):
        taken, ships = set(), []
        for size in self.sizes:
            while True:
                horiz = self.rng.random() < 0.5
                r = self.rng.randrange(self.n if horiz else self.n - size + 1)
                c = self.rng.randrange(self.n - size + 1 if horiz else self.n)
                cells = [self._cell(r, c + i) if horiz else self._cell(r + i, c) for i in range(size)]
                if not taken & set(cells):
                    taken |= set(cells)
                    ships.append(cells)
                    break
        return ships

    def rules(self) -> str:
        cols = "ABCDEFGHIJ"[: self.n]
        fleet = ", ".join(f"{NAMES[s]} ({s})" for s in self.sizes)
        again = ("After a HIT or SUNK you shoot again; after a MISS the turn passes." if self.again
                 else "One shot per turn, hit or miss; then the turn passes.")
        return f"""BATTLESHIP - {self.variant.get('label')}. 2 players, player 0 shoots first.

THE FLEETS
Each player has a hidden {self.n}x{self.n} grid (columns {cols[0]}-{cols[-1]}, rows 1-{self.n}) with this fleet, placed by the arena:
{fleet}. Ships are straight (horizontal or vertical), never overlap, and may touch each other.

EACH TURN
Fire at one cell of the opponent's grid you have not fired at. The answer is MISS, HIT, or SUNK <ship name>
(that hit was the last cell of that ship). {again}

HOW THE GAME ENDS
The first player to sink the whole opposing fleet wins. If nobody has after {2 * self.n * self.n} shots in total, more
hits wins; equal is a draw.

ACTION FORMAT
fire <column><row>        for example: fire B7
No list of legal actions is sent when more than 50 cells are left. Anything else, a cell outside the grid or a cell
you already fired at is illegal (first time: a warning and the same turn again; second time: you lose the game).
"""

    def to_move(self):
        return [] if self._result else [self.turn]

    def _cells(self):
        return [self._cell(r, c) for r in range(self.n) for c in range(self.n)]

    def legal_actions(self, seat):
        if self._result or seat != self.turn:
            return []
        return self.cap_list([f"fire {x}" for x in self._cells() if x not in self.shots[seat]])

    def normalize(self, seat, action):
        if self._result or seat != self.turn:
            return None
        m = _FIRE.match(self.clean(action))
        if not m:
            return None
        cell = f"{m.group(1).upper()}{int(m.group(2))}"
        return f"fire {cell}" if cell in self._cells() and cell not in self.shots[seat] else None

    def apply(self, actions):
        if self._result:
            return
        s = self.turn
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        cell = a.split()[1]
        ship = next((sh for sh in self.fleets[1 - s] if cell in sh), None)
        self.total += 1
        if ship is None:
            self.shots[s][cell] = "miss"
        elif all(x in self.shots[s] or x == cell for x in ship):
            self.shots[s][cell] = f"sunk {NAMES[len(ship)]}"
            self.sunk[s].append(len(ship))
        else:
            self.shots[s][cell] = "hit"
        if len(self.sunk[s]) == len(self.sizes):
            self._result = {"winner": s, "scores": [1 - s, s], "reason": f"seat {s} sank the whole fleet in {len(self.shots[s])} shots"}
            return
        if self.total >= 2 * self.n * self.n:
            h = [sum(v != "miss" for v in self.shots[x].values()) for x in (0, 1)]
            w = None if h[0] == h[1] else (0 if h[0] > h[1] else 1)
            self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w], "reason": f"shot cap; hits {h[0]} to {h[1]}"}
            return
        if ship is None or not self.again:
            self.turn = 1 - s

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        mark = lambda v: "." if v is None else ("o" if v == "miss" else ("X" if v == "hit" else "#"))  # noqa: E731
        left = list(self.sizes)
        for x in self.sunk[seat]:
            left.remove(x)
        L = [f"BATTLESHIP {self.n}x{self.n}. You are player {seat}.",
             f"Opponent ships still afloat: {', '.join(f'{NAMES[x]} ({x})' for x in left) or 'none'}.",
             "Your shots at the opponent (. not fired, o miss, X hit, # hit on a ship you sank):",
             "    " + " ".join("ABCDEFGHIJ"[: self.n])]
        for r in range(self.n):
            L.append(f"{r + 1:>3} " + " ".join(mark(self.shots[seat].get(self._cell(r, c))) for c in range(self.n)))
        last = list(self.shots[seat].items())[-5:]
        L.append("Your last shots: " + (", ".join(f"{k} {v.upper()}" for k, v in last) or "none"))
        hits_on_me = [k for k, v in self.shots[o].items() if v != "miss"]
        L.append(f"Opponent has fired {len(self.shots[o])} shots at you; hits on you: {', '.join(hits_on_me) or 'none'}.")
        if self._result:
            L.append(f"Game over: {self._result['reason']}.")
        elif seat == self.turn:
            L.append('Your turn: "fire <column><row>".')
        else:
            L.append("Opponent's turn.")
        return "\n".join(L)

    def frame(self):
        return {"fleets": self.fleets, "shots": [dict(x) for x in self.shots], "turn": self.turn,
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return "fire " + rng.choice([x for x in self._cells() if x not in self.shots[seat]])

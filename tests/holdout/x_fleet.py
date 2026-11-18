"""Hidden Fleet (holdout test game, Guessing family, alternating shots). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: deduction on a 2-D grid from hit / miss / sunk feedback (ships are straight lines: after a hit, search the
neighbours), coordinates as moves (`fire C4`), never repeating a shot, a race against the opponent, a legal list on the
small board and NO list on the big one (more than 50 cells).
"""
from __future__ import annotations

import re

from ..core import Game

_FIRE = re.compile(r"^fire\s+([a-z])\s*(\d{1,2})$")


class HiddenFleet(Game):
    id = "x_fleet"
    name = "Hidden Fleet"
    category = "guess"
    blurb = "Both fleets are hidden. Every shot answers hit, miss or sunk. First to sink the other fleet wins."
    howto = "Players take turns firing at a cell of the opponent's hidden grid. Ships are straight lines; sink them all first to win."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "6x6, ships of 3 and 2", "size": 6, "ships": [3, 2]},
                {"label": "8x8, ships of 4, 3 and 2", "size": 8, "ships": [4, 3, 2]}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.n = int(self.variant.get("size", 6))
        self.ship_sizes = list(self.variant.get("ships") or [3, 2])
        self.fleets = [self._place() for _ in (0, 1)]  # fleets[s] = seat s's ships (the other player shoots at them)
        self.shots: list[dict[str, str]] = [{}, {}]  # shots[s] = cells seat s fired at -> "hit" / "miss" / "sunk"
        self.turn, self.moves = 0, 0
        self._result = None

    def _cell(self, r, c):
        return f"{'ABCDEFGHIJ'[c]}{r + 1}"

    def _place(self):
        taken, ships = set(), []
        for size in self.ship_sizes:
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
        return f"""HIDDEN FLEET - a {self.n}x{self.n} grid each (columns {cols[0]}-{cols[-1]}, rows 1-{self.n}). Players take turns.

THE FLEETS
Each player has a hidden fleet of ships of lengths {', '.join(map(str, self.ship_sizes))}. A ship is a straight line of cells,
horizontal or vertical; ships never overlap. You never see the opponent's grid.

EACH TURN
Fire at one cell of the opponent's grid. The answer is "miss", "hit", or "sunk" (the hit completed a ship).
Firing twice at the same cell is illegal.

HOW THE GAME ENDS
The first player to sink the whole opposing fleet wins. If both still have ships after {self.n * self.n} shots each, the
player with more hits wins; equal hits is a draw.

ACTION FORMAT
fire <column><row>       for example: fire C4
With more than 50 cells left no list of legal actions is sent: build the move from this format. Anything else, a
cell outside the grid or a cell you already fired at is illegal (first time: a warning and the same turn again;
second time: you lose the game).

WHAT YOU OBSERVE
Your own shots with their answers (as a grid and as a list), how many of the opponent's ships are sunk, and the
opponent's shots at your fleet.
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
        cell, target = a.split()[1], self.fleets[1 - s]
        ship = next((sh for sh in target if cell in sh), None)
        if ship is None:
            self.shots[s][cell] = "miss"
        else:
            self.shots[s][cell] = "hit"
            if all(self.shots[s].get(x) in ("hit", "sunk") for x in ship):
                for x in ship:
                    self.shots[s][x] = "sunk"
        self.moves += 1
        if all(all(self.shots[s].get(x) == "sunk" for x in sh) for sh in target):
            self._result = {"winner": s, "scores": [1 - s, s], "reason": f"seat {s} sank the whole fleet in {len(self.shots[s])} shots"}
            return
        if self.moves >= 2 * self.n * self.n:
            h = [sum(v != "miss" for v in self.shots[x].values()) for x in (0, 1)]
            w = None if h[0] == h[1] else (0 if h[0] > h[1] else 1)
            self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w], "reason": f"shot cap; hits {h[0]} to {h[1]}"}
            return
        self.turn = 1 - s

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def _grid(self, shots):
        mark = {"miss": "o", "hit": "X", "sunk": "#"}
        head = "   " + " ".join("ABCDEFGHIJ"[: self.n])
        rows = [f"{r + 1:>2} " + " ".join(mark.get(shots.get(self._cell(r, c)), ".") for c in range(self.n)) for r in range(self.n)]
        return [head] + rows

    def observation(self, seat):
        o = 1 - seat
        sunk = sum(all(self.shots[seat].get(x) == "sunk" for x in sh) for sh in self.fleets[o])
        lines = [f"HIDDEN FLEET {self.n}x{self.n}. You are seat {seat}. Opponent ships sunk: {sunk} of {len(self.ship_sizes)} "
                 f"(lengths {', '.join(map(str, self.ship_sizes))}).",
                 "Your shots at the opponent's grid (. = not fired, o = miss, X = hit, # = sunk ship):"]
        lines += ["  " + x for x in self._grid(self.shots[seat])]
        lines.append("Your shots in order: " + (", ".join(f"{k} {v}" for k, v in self.shots[seat].items()) or "(none yet)"))
        mine_hit = [k for k, v in self.shots[o].items() if v != "miss"]
        lines.append(f"Opponent has fired {len(self.shots[o])} shots at your fleet; hits on you: {', '.join(mine_hit) or 'none'}.")
        if self._result:
            lines.append(f"The game is over: {self._result['reason']}.")
        elif seat == self.turn:
            lines.append('Your move: "fire <column><row>", a cell you have not fired at.')
        else:
            lines.append("Waiting for the opponent.")
        return "\n".join(lines)

    def frame(self):
        return {"fleets": self.fleets, "shots": [dict(x) for x in self.shots], "turn": self.turn,
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return "fire " + rng.choice([x for x in self._cells() if x not in self.shots[seat]])

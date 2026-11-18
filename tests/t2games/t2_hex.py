"""Hex (local copy of a T2 game, our approximation; tests only). Owner: path-to-glory-3.

Connection game on a rhombus of hexagons; no draws. Variants: 5x5; 7x7; 7x7 with the swap (pie) rule, which gives the
second player an unusual extra move. Player 0 (X) joins TOP to BOTTOM, player 1 (O) joins LEFT to RIGHT.
"""
from __future__ import annotations

import re

from ..core import Game

_PLACE = re.compile(r"^(?:place\s+|play\s+)?([a-k])\s*(\d{1,2})$")
SYM = ("X", "O")


class T2Hex(Game):
    id = "t2_hex"
    name = "Hex"
    category = "think"
    blurb = "Build a chain of hexagons across the board; every block is also a bridge for your rival."
    howto = "Players take turns placing a stone on an empty hexagon. X joins top to bottom, O joins left to right; the board can never end in a draw."
    duplicate_safe = True

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "5x5", "size": 5, "swap": False},
                {"label": "7x7", "size": 7, "swap": False},
                {"label": "7x7 with swap rule", "size": 7, "swap": True}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.n = int(self.variant.get("size", 5))
        self.swap = bool(self.variant.get("swap"))
        self.b = [[None] * self.n for _ in range(self.n)]  # b[row][col]
        self.turn, self.moves = 0, []
        self._result = None

    def _name(self, r, c):
        return f"{'abcdefghijk'[c]}{r + 1}"

    def rules(self) -> str:
        n = self.n
        swap = ("\nSWAP RULE: on the second move of the game only, player 1 may answer \"swap\" instead of placing: the first"
                " stone is replaced by an O stone on the mirrored cell (column and row exchanged), and player 0 moves next."
                if self.swap else "")
        return f"""HEX - {n}x{n} board. 2 players, alternating turns, player 0 (X) moves first.

THE BOARD
A rhombus of {n}x{n} hexagonal cells. Columns a-{'abcdefghijk'[n - 1]} (left to right), rows 1-{n} (top to bottom).
Cell (row r, column c) touches six cells: (r-1,c) (r-1,c+1) (r,c-1) (r,c+1) (r+1,c-1) (r+1,c).
In the drawing each row is shifted half a cell to the right of the row above.

GOAL
Player 0 (X) wins by a connected chain of X stones from the TOP row (1) to the BOTTOM row ({n}).
Player 1 (O) wins by a connected chain of O stones from the LEFT column (a) to the RIGHT column ({'abcdefghijk'[n - 1]}).
A full board always has exactly one winner: there are no draws.{swap}

ACTION FORMAT
place <column><row>        for example: place c3
Only empty cells. The list of legal moves is sent every turn. Anything else is illegal (first time: a warning and the
same turn again; second time: you lose the game).
"""

    def to_move(self):
        return [] if self._result else [self.turn]

    def _empty(self):
        return [self._name(r, c) for r in range(self.n) for c in range(self.n) if self.b[r][c] is None]

    def _can_swap(self, seat):
        return self.swap and seat == 1 and len(self.moves) == 1

    def legal_actions(self, seat):
        if self._result or seat != self.turn:
            return []
        return self.cap_list([f"place {x}" for x in self._empty()] + (["swap"] if self._can_swap(seat) else []))

    def normalize(self, seat, action):
        if self._result or seat != self.turn:
            return None
        a = self.clean(action)
        if a == "swap":
            return "swap" if self._can_swap(seat) else None
        m = _PLACE.match(a)
        if not m:
            return None
        cell = f"{m.group(1)}{int(m.group(2))}"
        return f"place {cell}" if cell in self._empty() else None

    def _won(self, s):
        n, seen = self.n, set()
        stack = [(0, c) for c in range(n) if self.b[0][c] == s] if s == 0 else [(r, 0) for r in range(n) if self.b[r][0] == s]
        while stack:
            r, c = stack.pop()
            if (r, c) in seen:
                continue
            seen.add((r, c))
            if (s == 0 and r == n - 1) or (s == 1 and c == n - 1):
                return True
            for dr, dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and self.b[rr][cc] == s and (rr, cc) not in seen:
                    stack.append((rr, cc))
        return False

    def apply(self, actions):
        if self._result:
            return
        s = self.turn
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        if a == "swap":
            first = self.moves[0][1]
            c, r = "abcdefghijk".index(first[0]), int(first[1:]) - 1
            self.b[r][c] = None
            self.b[c][r] = 1
            self.moves.append((s, "swap"))
        else:
            cell = a.split()[1]
            c, r = "abcdefghijk".index(cell[0]), int(cell[1:]) - 1
            self.b[r][c] = s
            self.moves.append((s, cell))
            if self._won(s):
                self._result = {"winner": s, "scores": [1 - s, s], "reason": f"seat {s} ({SYM[s]}) connected its two sides"}
                return
        self.turn = 1 - s

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def render(self):
        cols = "abcdefghijk"[: self.n]
        out = ["   " + " ".join(cols)]
        for r in range(self.n):
            out.append(" " * r + f"{r + 1:>2} " + " ".join("." if v is None else SYM[v] for v in self.b[r]) + f" {r + 1}")
        out.append(" " * self.n + "   " + " ".join(cols))
        return "\n".join(out)

    def observation(self, seat):
        L = [f"HEX {self.n}x{self.n}. You are player {seat} and play {SYM[seat]}; you join "
             f"{'TOP to BOTTOM (rows 1 to ' + str(self.n) + ')' if seat == 0 else 'LEFT to RIGHT (columns a to ' + 'abcdefghijk'[self.n - 1] + ')'}.",
             self.render(),
             "Moves so far: " + (", ".join(f"{SYM[s]} {m}" for s, m in self.moves) or "none")]
        if self._result:
            L.append(f"Game over: {self._result['reason']}.")
        elif seat == self.turn:
            L.append('Your turn: "place <column><row>"' + (' or "swap"' if self._can_swap(seat) else "") + ".")
        else:
            L.append("Opponent's turn.")
        return "\n".join(L)

    def frame(self):
        return {"board": [list(r) for r in self.b], "moves": [list(m) for m in self.moves], "turn": self.turn,
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return "place " + rng.choice(self._empty())

"""Dots and Boxes (local copy of a T2 game, our approximation; tests only). Owner: path-to-glory-3.

Draw lines between neighbouring dots; closing a box scores it AND gives another move (the turn does not pass), which
is the special structure here. Variants: 3x3 boxes (24 lines), 4x4 boxes (40 lines), 5x4 boxes (49 lines).
"""
from __future__ import annotations

import re

from ..core import Game

_LINE = re.compile(r"^(?:line\s+|draw\s+)?([a-h])(\d)\s*[- ]?\s*([a-h])(\d)$")


class T2Dots(Game):
    id = "t2_dots"
    name = "Dots and Boxes"
    category = "think"
    blurb = "Every line you draw might hand your rival a chain of boxes. Count the chains, not the lines."
    howto = "Players take turns drawing a line between two neighbouring dots. Closing a box scores it and gives you another move; most boxes wins."
    duplicate_safe = True

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "3x3 boxes", "cols": 3, "rows": 3}, {"label": "4x4 boxes", "cols": 4, "rows": 4},
                {"label": "5 wide x 4 high boxes", "cols": 5, "rows": 4}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.w, self.h = int(self.variant.get("cols", 3)), int(self.variant.get("rows", 3))
        self.drawn: dict[str, int] = {}  # canonical line -> seat
        self.owner: dict[tuple[int, int], int] = {}
        self.turn, self.history = 0, []
        self._result = None

    def _dot(self, x, y):
        return f"{'abcdefgh'[x]}{y + 1}"

    def _all_lines(self):
        out = []
        for y in range(self.h + 1):
            for x in range(self.w):
                out.append(f"{self._dot(x, y)}-{self._dot(x + 1, y)}")
        for y in range(self.h):
            for x in range(self.w + 1):
                out.append(f"{self._dot(x, y)}-{self._dot(x, y + 1)}")
        return out

    def _box_lines(self, x, y):
        return [f"{self._dot(x, y)}-{self._dot(x + 1, y)}", f"{self._dot(x, y + 1)}-{self._dot(x + 1, y + 1)}",
                f"{self._dot(x, y)}-{self._dot(x, y + 1)}", f"{self._dot(x + 1, y)}-{self._dot(x + 1, y + 1)}"]

    def rules(self) -> str:
        w, h = self.w, self.h
        return f"""DOTS AND BOXES - {w}x{h} boxes ({w + 1}x{h + 1} dots). 2 players, player 0 moves first.

THE BOARD
Dots are named by column letter a-{'abcdefgh'[w]} (left to right) and row number 1-{h + 1} (top to bottom): a1 is the top-left dot.
A line joins two NEIGHBOURING dots, horizontally (a1-b1) or vertically (a1-a2). There are {len(self._all_lines())} lines.

EACH TURN
Draw one line that is not drawn yet. If your line completes the 4th side of one or two boxes, you own those boxes
AND you must move again (the turn does not pass). Otherwise the turn passes to the opponent.

HOW THE GAME ENDS
When all lines are drawn. More boxes wins; equal boxes is a draw. Your score = your boxes.

ACTION FORMAT
line <dot>-<dot>        for example: line a1-b1   or   line c2-c3
Dots in either order. The list of legal moves is sent every turn while it has at most 50 entries. Anything else is
illegal (first time: a warning and the same turn again; second time: you lose the game).
"""

    def to_move(self):
        return [] if self._result else [self.turn]

    def _open(self):
        return [l for l in self._all_lines() if l not in self.drawn]

    def legal_actions(self, seat):
        if self._result or seat != self.turn:
            return []
        return self.cap_list([f"line {l}" for l in self._open()])

    def normalize(self, seat, action):
        if self._result or seat != self.turn:
            return None
        m = _LINE.match(self.clean(action))
        if not m:
            return None
        a, b = f"{m.group(1)}{int(m.group(2))}", f"{m.group(3)}{int(m.group(4))}"
        for l in (f"{a}-{b}", f"{b}-{a}"):
            if l in self._open():
                return f"line {l}"
        return None

    def apply(self, actions):
        if self._result:
            return
        s = self.turn
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        l = a.split()[1]
        self.drawn[l] = s
        got = []
        for y in range(self.h):
            for x in range(self.w):
                if (x, y) not in self.owner and l in self._box_lines(x, y) and all(z in self.drawn for z in self._box_lines(x, y)):
                    self.owner[(x, y)] = s
                    got.append(f"{self._dot(x, y)}")
        self.history.append((s, l, got))
        if len(self.drawn) == len(self._all_lines()):
            b = [sum(v == i for v in self.owner.values()) for i in (0, 1)]
            w = None if b[0] == b[1] else (0 if b[0] > b[1] else 1)
            self._result = {"winner": w, "scores": b, "reason": f"all lines drawn; boxes {b[0]} to {b[1]}"}
            return
        if not got:
            self.turn = 1 - s

    def is_over(self):
        return self._result is not None

    def result(self):
        b = [sum(v == i for v in self.owner.values()) for i in (0, 1)]
        return dict(self._result) if self._result else {"winner": None, "scores": b, "reason": "game in progress"}

    def render(self, seat):
        out = ["   " + "   ".join("abcdefgh"[: self.w + 1])]
        for y in range(self.h + 1):
            row = f"{y + 1:>2} "
            for x in range(self.w + 1):
                row += "*"
                if x < self.w:
                    row += "---" if f"{self._dot(x, y)}-{self._dot(x + 1, y)}" in self.drawn else "   "
            out.append(row)
            if y < self.h:
                row = "   "
                for x in range(self.w + 1):
                    row += "|" if f"{self._dot(x, y)}-{self._dot(x, y + 1)}" in self.drawn else " "
                    if x < self.w:
                        o = self.owner.get((x, y))
                        row += "   " if o is None else (" Y " if o == seat else " T ")
                out.append(row)
        return "\n".join(out)

    def observation(self, seat):
        b = [sum(v == i for v in self.owner.values()) for i in (0, 1)]
        L = [f"DOTS AND BOXES {self.w}x{self.h}. You are player {seat}. Boxes: you {b[seat]}, opponent {b[1 - seat]}, "
             f"open {self.w * self.h - len(self.owner)}.",
             "Board (* dot, --- and | drawn lines, Y your box, T their box):", self.render(seat),
             f"Lines drawn: {len(self.drawn)} of {len(self._all_lines())}. Last moves: " +
             (", ".join(f"{'you' if s == seat else 'opp'} {l}{' (box!)' if g else ''}" for s, l, g in self.history[-6:]) or "none")]
        if self._result:
            L.append(f"Game over: {self._result['reason']}.")
        elif seat == self.turn:
            L.append(f"Undrawn lines: {', '.join(self._open())}")
            L.append('Your turn: "line <dot>-<dot>".')
        else:
            L.append("Opponent's turn.")
        return "\n".join(L)

    def frame(self):
        return {"drawn": dict(self.drawn), "owner": {f"{x},{y}": s for (x, y), s in self.owner.items()},
                "turn": self.turn, "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return "line " + rng.choice(self._open())

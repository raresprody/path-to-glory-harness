"""Mini Chess (local copy of a T2 game, our approximation; tests only). Owner: path-to-glory-3.

Chess on a small board with full check rules. Variants: Gardner 5x5 (R N B Q K, 5 pawns); Los Alamos 6x6 (no
bishops). No castling, no pawn double step, no en passant; pawns promote on the last rank (to q, r, n, or b where
bishops exist). Checkmate wins; stalemate, threefold repetition, bare kings or the move cap are draws.
"""
from __future__ import annotations

import re

from ..core import Game

_MOVE = re.compile(r"^(?:move\s+)?([a-f])([1-6])\s*[- x]?\s*([a-f])([1-6])\s*=?([qrbn])?$")
KN = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
KG = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
DIRS = {"b": [(1, 1), (1, -1), (-1, 1), (-1, -1)], "r": [(1, 0), (-1, 0), (0, 1), (0, -1)]}
DIRS["q"] = DIRS["b"] + DIRS["r"]
NAMES = {"k": "king", "q": "queen", "r": "rook", "b": "bishop", "n": "knight", "p": "pawn"}


class T2MiniChess(Game):
    id = "t2_minichess"
    name = "Mini Chess"
    category = "think"
    blurb = "Chess squeezed onto a tiny board: every piece is in the fight from move one."
    howto = "Chess rules on a 5x5 or 6x6 board. Checkmate the opponent's king to win."
    duplicate_safe = True

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "Gardner 5x5", "w": 5, "h": 5, "back": "RNBQK", "cap": 100},
                {"label": "Los Alamos 6x6 (no bishops)", "w": 6, "h": 6, "back": "RNQKNR", "cap": 120}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.w, self.h = int(self.variant.get("w", 5)), int(self.variant.get("h", 5))
        back = str(self.variant.get("back", "RNBQK"))
        self.cap = int(self.variant.get("cap", 100))
        self.promos = "qrn" + ("b" if "B" in back else "")
        self.b: dict[tuple[int, int], str] = {}
        for f, p in enumerate(back):
            self.b[(f, 0)] = p
            self.b[(f, self.h - 1)] = p.lower()
            self.b[(f, 1)] = "P"
            self.b[(f, self.h - 2)] = "p"
        self.turn, self.history = 0, []
        self.seen: dict[str, int] = {}
        self._remember()
        self._result = None

    # ---- helpers ------------------------------------------------------------
    def sq(self, f, r):
        return f"{'abcdef'[f]}{r + 1}"

    def _side(self, p):
        return 0 if p.isupper() else 1

    def _key(self):
        return "".join(self.b.get((f, r), ".") for r in range(self.h) for f in range(self.w)) + str(self.turn)

    def _remember(self):
        k = self._key()
        self.seen[k] = self.seen.get(k, 0) + 1
        return self.seen[k]

    def _pseudo(self, side, board):
        out = []
        on = lambda f, r: 0 <= f < self.w and 0 <= r < self.h  # noqa: E731
        for (f, r), p in board.items():
            if self._side(p) != side:
                continue
            t = p.lower()
            if t == "p":
                d = 1 if side == 0 else -1
                last = self.h - 1 if side == 0 else 0
                tg = []
                if on(f, r + d) and (f, r + d) not in board:
                    tg.append((f, r + d))
                for df in (-1, 1):
                    q = board.get((f + df, r + d))
                    if on(f + df, r + d) and q and self._side(q) != side:
                        tg.append((f + df, r + d))
                for x in tg:
                    out += [((f, r), x, pr) for pr in self.promos] if x[1] == last else [((f, r), x, None)]
            elif t in ("n", "k"):
                for df, dr in (KN if t == "n" else KG):
                    x = (f + df, r + dr)
                    q = board.get(x)
                    if on(*x) and (q is None or self._side(q) != side):
                        out.append(((f, r), x, None))
            else:
                for df, dr in DIRS[t]:
                    x = (f + df, r + dr)
                    while on(*x):
                        q = board.get(x)
                        if q is None:
                            out.append(((f, r), x, None))
                        else:
                            if self._side(q) != side:
                                out.append(((f, r), x, None))
                            break
                        x = (x[0] + df, x[1] + dr)
        return out

    def _make(self, board, mv):
        a, b, pr = mv
        nb = dict(board)
        p = nb.pop(a)
        nb[b] = (pr.upper() if p.isupper() else pr) if pr else p
        return nb

    def _in_check(self, side, board):
        king = next((s for s, p in board.items() if p == ("K" if side == 0 else "k")), None)
        return king is None or any(m[1] == king for m in self._pseudo(1 - side, board))

    def _legal(self, side):
        return [m for m in self._pseudo(side, self.b) if not self._in_check(side, self._make(self.b, m))]

    def _str(self, m):
        return f"{self.sq(*m[0])}{self.sq(*m[1])}{m[2] or ''}"

    # ---- interface ----------------------------------------------------------
    def rules(self) -> str:
        files = "abcdef"[: self.w]
        start = "\n".join(f"  {r + 1} " + " ".join(self.b.get((f, r), ".") for f in range(self.w)) for r in reversed(range(self.h)))
        return f"""MINI CHESS - {self.variant.get('label')}. 2 players, player 0 is WHITE (uppercase) and moves first.

THE BOARD
{self.w} files a-{files[-1]} (left to right) and {self.h} ranks 1-{self.h}. White starts on ranks 1-2 and moves UP; Black
(lowercase) starts on ranks {self.h - 1}-{self.h} and moves DOWN. K king, Q queen, R rook, {'B bishop, ' if 'b' in self.promos else ''}N knight, P pawn.
Starting position (rank {self.h} on top):
{start}
    {' '.join(files)}

MOVES
Pieces move as in chess. Pawns move ONE square forward (never two), capture one square diagonally forward, and
promote on the last rank to {', '.join(NAMES[x] for x in self.promos)}. There is no castling and no en passant.
You may never leave your own king in check.

HOW THE GAME ENDS
Checkmate wins. Stalemate (no legal move, not in check) is a DRAW. Also a draw: the same position for the third time,
only the two kings left, or {self.cap} moves in total (both players counted) without a result.

ACTION FORMAT
<from><to>, plus the promotion letter when a pawn promotes.   Examples: b2b3   c1e3   a4a5q
"move b2b3" is accepted too. The list of legal moves is sent while it has at most 50 entries. Anything else is
illegal (first time: a warning and the same turn again; second time: you lose the game).
"""

    def to_move(self):
        return [] if self._result else [self.turn]

    def legal_actions(self, seat):
        if self._result or seat != self.turn:
            return []
        return self.cap_list(sorted(self._str(m) for m in self._legal(seat)))

    def normalize(self, seat, action):
        if self._result or seat != self.turn:
            return None
        m = _MOVE.match(self.clean(action))
        if not m:
            return None
        s = f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}{m.group(5) or ''}"
        legal = {self._str(x) for x in self._legal(seat)}
        return s if s in legal else None

    def apply(self, actions):
        if self._result:
            return
        s = self.turn
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        mv = next(x for x in self._legal(s) if self._str(x) == a)
        took = self.b.get(mv[1])
        self.b = self._make(self.b, mv)
        self.history.append((s, a, took))
        self.turn = 1 - s
        rep = self._remember()
        o = self.turn
        if not self._legal(o):
            if self._in_check(o, self.b):
                self._finish(s, f"seat {s} checkmated seat {o}")
            else:
                self._finish(None, f"seat {o} has no legal move and is not in check: stalemate, draw")
        elif rep >= 3:
            self._finish(None, "the same position for the third time: draw")
        elif set(p.lower() for p in self.b.values()) == {"k"}:
            self._finish(None, "only the two kings are left: draw")
        elif len(self.history) >= self.cap:
            self._finish(None, f"{self.cap} moves played without checkmate: draw")

    def _finish(self, w, reason):
        self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w], "reason": reason}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        files = "abcdef"[: self.w]
        L = [f"MINI CHESS - {self.variant.get('label')}. You are player {seat}, playing "
             f"{'WHITE (uppercase, moving up)' if seat == 0 else 'BLACK (lowercase, moving down)'}.",
             f"Move {len(self.history) + 1} of at most {self.cap}."]
        L += [f"  {r + 1} " + " ".join(self.b.get((f, r), ".") for f in range(self.w)) for r in reversed(range(self.h))]
        L.append("    " + " ".join(files))
        if self.history:
            L.append("Moves so far: " + " ".join(f"{a}{'x' + t if t else ''}" for _, a, t in self.history))
            s, a, t = self.history[-1]
            L.append(f"Last move: {'you' if s == seat else 'opponent'} played {a}" + (f", capturing a {NAMES[t.lower()]}" if t else "") + ".")
        if self._result:
            L.append(f"Game over: {self._result['reason']}.")
        elif seat == self.turn:
            if self._in_check(seat, self.b):
                L.append("You are IN CHECK.")
            L.append('Your move: "<from><to>" (add q/r/n' + ("/b" if "b" in self.promos else "") + " when a pawn promotes).")
        else:
            L.append("Opponent's turn.")
        return "\n".join(L)

    def frame(self):
        return {"board": {self.sq(*k): v for k, v in self.b.items()}, "history": [list(h) for h in self.history],
                "turn": self.turn, "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return self._str(rng.choice(self._legal(seat)))

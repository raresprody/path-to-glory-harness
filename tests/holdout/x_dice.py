"""Liar's Dice (holdout test game, Bluff family, alternating bids). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: hidden dice on both sides, a two-number move (`bid <count> <face>`) that must raise the last bid, `call` to
challenge, probability reasoning, a legal list in the small variant and NO list in the big one (more than 50 moves),
"ones are wild" as a variant twist, and several rounds where the loser of a challenge loses a die.
"""
from __future__ import annotations

import re

from ..core import Game

_BID = re.compile(r"^bid\s+(\d+)\s+(\d)$")


class LiarsDice(Game):
    id = "x_dice"
    name = "Liar's Dice"
    category = "bluff"
    blurb = "Bid on all the dice on the table, yours hidden from them and theirs from you. Call the lie."
    howto = "Players raise bids on how many dice show a face across both hands, or call the last bid a lie. Lose a challenge, lose a die."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "3 dice each", "dice": 3, "wild": False},
                {"label": "5 dice each, ones are wild", "dice": 5, "wild": True}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.wild = bool(self.variant.get("wild"))
        self.count = [int(self.variant.get("dice", 3))] * 2
        self.starter, self.round_no = 0, 0
        self.history: list[str] = []
        self._result = None
        self._new_round(0)

    def _new_round(self, starter):
        self.round_no += 1
        self.dice = [sorted(self.rng.randint(1, 6) for _ in range(self.count[s])) for s in (0, 1)]
        self.bids: list[tuple[int, int, int]] = []  # seat, count, face
        self.turn = starter

    def _matches(self, face):
        return sum(1 for s in (0, 1) for d in self.dice[s] if d == face or (self.wild and d == 1 and face != 1))

    def rules(self) -> str:
        n = self.count[0]
        wild = ("ONES ARE WILD: a 1 counts as every face, except in a bid on ones itself.\n" if self.wild else "")
        return f"""LIAR'S DICE - each player starts with {n} dice. Two players, alternating turns.

EACH ROUND
Both players roll their dice in secret: you see your own dice, never the opponent's. Starting with the round's
first player, players take turns. On your turn you either RAISE the bid or CALL.
A bid "<count> <face>" claims: across BOTH players' dice there are at least <count> dice showing <face>.
{wild}A raise must be higher than the last bid: a larger count (any face), or the same count with a larger face.
CALL says the last bid is a lie. All dice are revealed: if the bid was true, the caller loses one die; if it was
false, the bidder loses one die. The loser of the challenge starts the next round. You cannot call before any bid.

HOW THE GAME ENDS
A player with no dice left loses the game.

ACTION FORMAT
bid <count> <face>     count from 1 to the number of dice on the table, face from 1 to 6, e.g. bid 2 5
call
When there are more than 50 possible moves no list of legal actions is sent: build the move from this format.
Anything else, or a bid that does not raise, is illegal (first time: a warning and the same turn again; second time:
you lose the game).

WHAT YOU OBSERVE
Your dice, how many dice each player has, this round's bids, and the result of every earlier round.
"""

    def to_move(self):
        return [] if self._result else [self.turn]

    def _raises(self, c, f):
        if not self.bids:
            return True
        _, lc, lf = self.bids[-1]
        return c > lc or (c == lc and f > lf)

    def legal_actions(self, seat):
        if self._result or seat != self.turn:
            return []
        total = sum(self.count)
        acts = [f"bid {c} {f}" for c in range(1, total + 1) for f in range(1, 7) if self._raises(c, f)]
        if self.bids:
            acts.append("call")
        return self.cap_list(acts)

    def normalize(self, seat, action):
        if self._result or seat != self.turn:
            return None
        a = self.clean(action)
        if a == "call":
            return "call" if self.bids else None
        m = _BID.match(a)
        if not m:
            return None
        c, f = int(m.group(1)), int(m.group(2))
        if not (1 <= c <= sum(self.count) and 1 <= f <= 6 and self._raises(c, f)):
            return None
        return f"bid {c} {f}"

    def apply(self, actions):
        if self._result:
            return
        s = self.turn
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        if a.startswith("bid"):
            _, c, f = a.split()
            self.bids.append((s, int(c), int(f)))
            self.turn = 1 - s
            return
        bidder, c, f = self.bids[-1]
        have = self._matches(f)
        loser = s if have >= c else bidder
        self.count[loser] -= 1
        self.history.append(f"Round {self.round_no}: seat {bidder} bid {c} {f}, seat {s} called; dice were "
                            f"seat 0 {self.dice[0]}, seat 1 {self.dice[1]} -> {have} matching; seat {loser} loses a die")
        if self.count[loser] == 0:
            self._result = {"winner": 1 - loser, "scores": [1 - loser, loser],
                            "reason": f"seat {loser} has no dice left"}
            return
        self._new_round(loser)

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        me = lambda t: t.replace(f"seat {seat}", "you").replace(f"seat {o}", "opponent")  # noqa: E731
        lines = [f"LIAR'S DICE{' (ones are wild)' if self.wild else ''}. You are seat {seat}. Round {self.round_no}.",
                 f"Dice left: you {self.count[seat]}, opponent {self.count[o]} ({sum(self.count)} on the table).",
                 f"Your dice: {' '.join(map(str, self.dice[seat]))}"]
        lines.append("Bids this round: " + (", ".join(f"{'you' if b == seat else 'opponent'} {c} {f}" for b, c, f in self.bids) or "(none yet)"))
        for h in self.history:
            lines.append("  " + me(h))
        if self._result:
            lines.append(f"The game is over: {me(self._result['reason'])}.")
        elif seat == self.turn:
            last = f"raise above {self.bids[-1][1]} {self.bids[-1][2]} or call" if self.bids else "make the first bid"
            lines.append(f'Your move ({last}): "bid <count> <face>"' + (' or "call"' if self.bids else ""))
        else:
            lines.append("Waiting for the opponent.")
        return "\n".join(lines)

    def frame(self):
        return {"count": list(self.count), "dice": [list(d) for d in self.dice], "bids": [list(b) for b in self.bids],
                "turn": self.turn, "history": list(self.history), "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        if self.bids and rng.random() < 0.35:
            return "call"
        _, lc, lf = self.bids[-1] if self.bids else (0, 1, 0)
        c, f = (lc, lf + 1) if lf < 6 and rng.random() < 0.5 else (lc + 1, rng.randint(1, 6))
        return f"bid {c} {f}" if c <= sum(self.count) else "call" if self.bids else "bid 1 1"

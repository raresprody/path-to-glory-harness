"""The Pact (holdout test game, Deals family: trust and betrayal, simultaneous). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: a repeated simultaneous choice where the best move depends on trust, a JOINT threshold that turns "always
betray" into a draw for both, betrayal timing (the last rounds), and promises made in chat that may be lies.
"""
from __future__ import annotations

from ..core import Game

ACTS = ("share", "grab")


class Pact(Game):
    id = "x_pact"
    name = "The Pact"
    category = "deals"
    blurb = "Share and the pot grows; grab and you win the round. Grab too often and the pot never fills."
    howto = ("Each round both players secretly choose share or grab. Both must reach the joint target together, "
             "or nobody wins; above it, the one with more points wins.")
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "6 rounds, joint target 26", "rounds": 6, "target": 26, "ss": 3, "gs": 5, "sg": 0, "gg": 1},
            {"label": "8 rounds, joint target 34, big temptation", "rounds": 8, "target": 34, "ss": 3, "gs": 7, "sg": 0, "gg": 1},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        v = self.variant
        self.rounds, self.target = int(v.get("rounds", 6)), int(v.get("target", 26))
        self.pay = {("share", "share"): (v["ss"], v["ss"]), ("grab", "share"): (v["gs"], v["sg"]),
                    ("share", "grab"): (v["sg"], v["gs"]), ("grab", "grab"): (v["gg"], v["gg"])}
        self.points = [0, 0]
        self.log: list[tuple[str, str]] = []
        self._result = None

    def rules(self) -> str:
        v = self.variant
        return f"""THE PACT - {self.rounds} rounds, 2 players, simultaneous choices.

EACH ROUND
Both players choose AT THE SAME TIME, in secret: share or grab.
  both share:            each gets {v['ss']} points
  you grab, they share:  you get {v['gs']}, they get {v['sg']}
  you share, they grab:  you get {v['sg']}, they get {v['gs']}
  both grab:             each gets {v['gg']} point{'s' if v['gg'] != 1 else ''}
Choices are revealed after the round. You may promise anything in chat; promises are not binding.

HOW THE GAME ENDS
After round {self.rounds}. The JOINT TARGET is {self.target}: if the two players' points added together are below
{self.target}, nobody wins (the game is a draw, whatever the scores). If the joint total reaches {self.target}, the player
with more points wins; equal points is a draw.

ACTION FORMAT
share
grab
Anything else is illegal (first time: a warning and the same turn again; second time: you lose the game).

WHAT YOU OBSERVE
The round number, both players' points, the joint total and the target, and every earlier round's two choices.
"""

    def to_move(self):
        return [] if self._result else [0, 1]

    def legal_actions(self, seat):
        return [] if self._result else list(ACTS)

    def normalize(self, seat, action):
        a = self.clean(action)
        return None if self._result or a not in ACTS else a

    def apply(self, actions):
        if self._result:
            return
        a = []
        for s in (0, 1):
            x = self.normalize(s, actions.get(s, ""))
            if x is None:
                raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
            a.append(x)
        p = self.pay[(a[0], a[1])]
        self.points = [self.points[0] + p[0], self.points[1] + p[1]]
        self.log.append((a[0], a[1]))
        if len(self.log) >= self.rounds:
            total, (p0, p1) = sum(self.points), self.points
            if total < self.target:
                w, why = None, f"joint total {total} below the target {self.target}: nobody wins"
            else:
                w = None if p0 == p1 else (0 if p0 > p1 else 1)
                why = f"joint total {total} reached {self.target}; points {p0} to {p1}"
            self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w], "reason": why}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        lines = [f"THE PACT. You are seat {seat}. Round {min(len(self.log) + 1, self.rounds)} of {self.rounds}.",
                 f"Points: you {self.points[seat]}, opponent {self.points[o]}. Joint total {sum(self.points)}, "
                 f"target {self.target} (below it after the last round, nobody wins)."]
        lines.append("Earlier rounds:" if self.log else "Earlier rounds: (none)")
        for i, ch in enumerate(self.log, 1):
            lines.append(f"  Round {i}: you {ch[seat]}, opponent {ch[o]}")
        if self._result:
            lines.append(f"The game is over: {self._result['reason']}.")
        else:
            lines.append("Your move: share or grab")
        return "\n".join(lines)

    def frame(self):
        return {"points": list(self.points), "log": [list(x) for x in self.log], "target": self.target,
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return rng.choice(ACTS)

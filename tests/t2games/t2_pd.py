"""Prisoner's Dilemma, iterated (local copy of a T2 game, our approximation; tests only). Owner: path-to-glory-3.

Simultaneous cooperate / defect over many rounds; promises in chat matter. Variants: 10 known rounds; 12 rounds where
each move is flipped with 10 % chance (noise, both see what was actually played); an unknown length (the game ends
after each round with 15 % chance, at most 20 rounds).
"""
from __future__ import annotations

from ..core import Game

PAY = {("cooperate", "cooperate"): (3, 3), ("cooperate", "defect"): (0, 5),
       ("defect", "cooperate"): (5, 0), ("defect", "defect"): (1, 1)}
ALIASES = {"c": "cooperate", "cooperate": "cooperate", "d": "defect", "defect": "defect"}


class T2PD(Game):
    id = "t2_pd"
    name = "Prisoner's Dilemma"
    category = "deals"
    blurb = "Trust or betray, round after round: the best score goes to whoever reads the other right."
    howto = "Each round both players secretly cooperate or defect. Both cooperating pays well, betraying a cooperator pays best."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "10 rounds", "rounds": 10, "noise": 0.0, "stop": 0.0},
                {"label": "12 rounds, 10% noise", "rounds": 12, "noise": 0.1, "stop": 0.0},
                {"label": "unknown length (15% stop chance per round, at most 20)", "rounds": 20, "noise": 0.0, "stop": 0.15}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.rounds = int(self.variant.get("rounds", 10))
        self.noise = float(self.variant.get("noise", 0.0))
        self.stop = float(self.variant.get("stop", 0.0))
        self.points = [0, 0]
        self.history: list[dict] = []
        self._result = None

    def rules(self) -> str:
        length = (f"The game lasts exactly {self.rounds} rounds." if not self.stop else
                  f"After every round the game ENDS with probability {int(self.stop * 100)}% (at most {self.rounds} rounds), so"
                  " nobody knows which round is the last.")
        noise = (f"\nNOISE: each chosen move is flipped (cooperate <-> defect) with probability {int(self.noise * 100)}% before"
                 " it is scored. Both players see the move that was actually played, and whether it was flipped."
                 if self.noise else "")
        return f"""PRISONER'S DILEMMA - {self.variant.get('label')}. 2 players, simultaneous moves.

EACH ROUND
Both players choose cooperate or defect at the same time. Points this round (you, opponent):
  both cooperate:                3, 3
  you cooperate, they defect:    0, 5
  you defect, they cooperate:    5, 0
  both defect:                   1, 1{noise}

HOW THE GAME ENDS
{length} Total points decide: more points wins the game, equal points is a draw. Your score = your points.
Talk is free: you may promise anything in chat; nothing in chat binds anyone.

ACTION FORMAT
cooperate | defect   (the list of legal moves is sent every round). Anything else is illegal (first time: a warning and
the same turn again; second time: you lose the game).
"""

    def to_move(self):
        return [] if self._result else [0, 1]

    def legal_actions(self, seat):
        return [] if self._result else ["cooperate", "defect"]

    def normalize(self, seat, action):
        a = self.clean(action).strip(" .!\"'")
        return None if self._result else ALIASES.get(a)

    def apply(self, actions):
        if self._result:
            return
        chosen = [self.normalize(s, actions.get(s, "")) for s in (0, 1)]
        if None in chosen:
            raise ValueError(f"illegal action: {actions!r}")
        played, flipped = list(chosen), [False, False]
        for s in (0, 1):
            if self.noise and self.rng.random() < self.noise:
                played[s] = "defect" if played[s] == "cooperate" else "cooperate"
                flipped[s] = True
        p = PAY[(played[0], played[1])]
        self.points = [self.points[0] + p[0], self.points[1] + p[1]]
        self.history.append({"played": played, "flipped": flipped, "points": list(p)})
        n = len(self.history)
        if n >= self.rounds or (self.stop and self.rng.random() < self.stop):
            a, b = self.points
            w = None if a == b else (0 if a > b else 1)
            self._result = {"winner": w, "scores": list(self.points), "reason": f"{n} rounds played; points {a} to {b}"}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": list(self.points), "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        L = [f"PRISONER'S DILEMMA - {self.variant.get('label')}. You are player {seat}.",
             f"Round {len(self.history) + 1}{'' if self.stop else ' of ' + str(self.rounds)}. Points: you {self.points[seat]}, opponent {self.points[o]}."]
        for i, h in enumerate(self.history, 1):
            fl = lambda s: " (flipped by noise)" if h["flipped"][s] else ""  # noqa: E731
            L.append(f"  round {i}: you {h['played'][seat]}{fl(seat)}, opponent {h['played'][o]}{fl(o)} -> "
                     f"you +{h['points'][seat]}, opponent +{h['points'][o]}")
        L.append(f"Game over: {self._result['reason']}." if self._result else "Your move: cooperate or defect.")
        return "\n".join(L)

    def frame(self):
        return {"points": list(self.points), "history": self.history, "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return rng.choice(["cooperate", "defect"])

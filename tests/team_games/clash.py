"""Clash (team test game, Read-the-opponent family): both players pick one of three stances at the same
time; stances beat each other in a cycle but score different points, so the best mix is not uniform.

Ours, not the organizers': a stand-in for simultaneous moves and mixed strategies (rule 14.5).
Nothing in mesh/ may depend on this file.
"""
from __future__ import annotations

from ..core import Game

BEATS = {"strike": "feint", "feint": "guard", "guard": "strike"}  # key beats value


class Clash(Game):
    id = "clash"
    name = "Clash"
    category = "read"
    blurb = "Three stances in a circle, but one win is worth more: predictable greed gets punished."
    howto = "Each round both players secretly pick strike, guard or feint. Strike beats feint, feint beats guard, guard beats strike. Wins score different points; most points after the last round wins."
    duplicate_safe = True  # nothing hidden in the deal, nothing random later

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "9 rounds, strike win 3", "rounds": 9, "points": {"strike": 3, "guard": 1, "feint": 1}},
            {"label": "13 rounds, feint win 4", "rounds": 13, "points": {"strike": 2, "guard": 1, "feint": 4}},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.rounds = int(self.variant.get("rounds", 9))
        self.points = dict(self.variant.get("points") or {"strike": 3, "guard": 1, "feint": 1})
        self.score = [0, 0]
        self.log: list[tuple[str, str]] = []  # (seat 0's stance, seat 1's stance)
        self._over = False
        self._result: dict | None = None
        self.max_steps = self.rounds

    def rules(self) -> str:
        p = self.points
        return f"""CLASH - {self.rounds} rounds.

EACH ROUND
Both players choose one stance AT THE SAME TIME: strike, guard or feint. Neither sees the other's
choice until both have chosen.

WHO WINS A ROUND
strike beats feint, feint beats guard, guard beats strike. The same stance twice is a tie (no points).
The winner of the round scores points by the stance they won WITH:
    winning with strike: {p['strike']} points
    winning with guard:  {p['guard']} point(s)
    winning with feint:  {p['feint']} point(s)
The loser scores nothing.

ACTION FORMAT
One word: strike | guard | feint
Anything else is illegal (first time: a warning and the same turn again; second time: you lose the game).

WHAT YOU OBSERVE
The round number, both scores and every earlier round (your stance, the opponent's stance, who scored).

HOW THE GAME ENDS
After {self.rounds} rounds the player with more points wins; equal points is a draw.
"""

    def to_move(self) -> list[int]:
        return [] if self._over else [0, 1]

    def legal_actions(self, seat: int) -> list[str] | None:
        return [] if self._over else ["strike", "guard", "feint"]

    def observation(self, seat: int) -> str:
        opp = 1 - seat
        lines = [f"CLASH. You are Player {seat}. Round {min(len(self.log) + 1, self.rounds)} of {self.rounds}.",
                 f"Points: you {self.score[seat]}, opponent {self.score[opp]}.", "Earlier rounds:"]
        if not self.log:
            lines.append("  (none yet)")
        for i, pair in enumerate(self.log, 1):
            mine, theirs = pair[seat], pair[opp]
            if mine == theirs:
                out = "tie"
            elif BEATS[mine] == theirs:
                out = f"you scored {self.points[mine]}"
            else:
                out = f"opponent scored {self.points[theirs]}"
            lines.append(f"  Round {i}: you {mine}, opponent {theirs} -> {out}")
        if self._over:
            lines.append(f"Game over: {self._result['reason']}.")
        else:
            lines.append("Choose: strike | guard | feint")
        return "\n".join(lines)

    def normalize(self, seat: int, action: str) -> str | None:
        a = self.clean(action)
        return a if not self._over and a in BEATS else None

    def apply(self, actions: dict[int, str]) -> None:
        if self._over:
            return
        a = [self.normalize(s, actions.get(s, "")) if s in actions else None for s in (0, 1)]
        if None in a:
            raise ValueError(f"illegal actions: {actions!r}")
        if a[0] != a[1]:
            w = 0 if BEATS[a[0]] == a[1] else 1
            self.score[w] += int(self.points[a[w]])
        self.log.append((a[0], a[1]))
        if len(self.log) >= self.rounds:
            s0, s1 = self.score
            self._over = True
            if s0 == s1:
                self._result = {"winner": None, "scores": [0.5, 0.5], "reason": f"{self.rounds} rounds played, {s0} points each"}
            else:
                w = 0 if s0 > s1 else 1
                self._result = {"winner": w, "scores": [1 if w == 0 else 0, 1 if w == 1 else 0],
                                "reason": f"{self.rounds} rounds played; seat {w} wins {max(s0, s1)} to {min(s0, s1)}"}

    def is_over(self) -> bool:
        return self._over

    def result(self) -> dict:
        if self._result is None:
            return {"winner": None, "scores": [0, 0], "reason": "game in progress"}
        return dict(self._result)

    def frame(self) -> dict:
        return {"rounds": self.rounds, "points": dict(self.points), "score": list(self.score),
                "log": [list(p) for p in self.log], "to_move": self.to_move(), "over": self._over, "result": self._result}

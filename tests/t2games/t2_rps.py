"""Rock Paper Scissors (local copy of a T2 game, our approximation; tests only). Owner: path-to-glory-3.

Simultaneous moves over many rounds, so the game is reading the opponent's pattern. Variants: classic 3 throws over
15 rounds; Rock Paper Scissors Lizard Spock over 15 rounds; first to 5 round wins (capped at 25 rounds).
"""
from __future__ import annotations

from ..core import Game

BEATS3 = {"rock": {"scissors"}, "paper": {"rock"}, "scissors": {"paper"}}
BEATS5 = {"rock": {"scissors", "lizard"}, "paper": {"rock", "spock"}, "scissors": {"paper", "lizard"},
          "lizard": {"spock", "paper"}, "spock": {"scissors", "rock"}}


class T2RPS(Game):
    id = "t2_rps"
    name = "Rock Paper Scissors"
    category = "read"
    blurb = "Both throw at once, round after round: spot the pattern before your rival spots yours."
    howto = "Each round both players secretly pick a throw; rock beats scissors, scissors beats paper, paper beats rock."
    duplicate_safe = True

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "classic, 15 rounds", "throws": 3, "rounds": 15, "first_to": None},
                {"label": "lizard-spock, 15 rounds", "throws": 5, "rounds": 15, "first_to": None},
                {"label": "classic, first to 5 wins", "throws": 3, "rounds": 25, "first_to": 5}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.beats = BEATS5 if int(self.variant.get("throws", 3)) == 5 else BEATS3
        self.rounds = int(self.variant.get("rounds", 15))
        self.first_to = self.variant.get("first_to")
        self.wins = [0, 0]
        self.history: list[tuple[str, str, int | None]] = []
        self._result = None

    def rules(self) -> str:
        pairs = "; ".join(f"{a} beats {' and '.join(sorted(b))}" for a, b in self.beats.items())
        end = (f"The first player to win {self.first_to} rounds wins the game. If nobody has after {self.rounds} rounds,"
               " more round wins wins; equal is a draw." if self.first_to else
               f"After {self.rounds} rounds the player with more round wins wins the game; equal is a draw.")
        return f"""ROCK PAPER SCISSORS - {self.variant.get('label')}. 2 players, simultaneous moves.

EACH ROUND
Both players choose a throw at the same time: {', '.join(self.beats)}.
{pairs}. The same throw on both sides is a tied round (nobody scores).

HOW THE GAME ENDS
{end}

ACTION FORMAT
One word: {' | '.join(self.beats)}. The list of legal moves is sent every round. Anything else is illegal (first time:
a warning and the same turn again; second time: you lose the game).

WHAT YOU OBSERVE
The round number, the score and every earlier round (both throws and who won it).
"""

    def to_move(self):
        return [] if self._result else [0, 1]

    def legal_actions(self, seat):
        return [] if self._result else list(self.beats)

    def normalize(self, seat, action):
        a = self.clean(action).strip(" .!\"'")
        return a if not self._result and a in self.beats else None

    def apply(self, actions):
        if self._result:
            return
        a = [self.normalize(s, actions.get(s, "")) for s in (0, 1)]
        if None in a:
            raise ValueError(f"illegal action: {actions!r}")
        w = None if a[0] == a[1] else (0 if a[1] in self.beats[a[0]] else 1)
        if w is not None:
            self.wins[w] += 1
        self.history.append((a[0], a[1], w))
        if self.first_to and max(self.wins) >= self.first_to:
            w2 = 0 if self.wins[0] > self.wins[1] else 1
            self._result = {"winner": w2, "scores": list(self.wins), "reason": f"seat {w2} reached {self.first_to} round wins"}
        elif len(self.history) >= self.rounds:
            w0, w1 = self.wins
            w2 = None if w0 == w1 else (0 if w0 > w1 else 1)
            self._result = {"winner": w2, "scores": list(self.wins), "reason": f"{self.rounds} rounds played; round wins {w0} to {w1}"}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": list(self.wins), "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        L = [f"ROCK PAPER SCISSORS - {self.variant.get('label')}. You are player {seat}.",
             f"Round {len(self.history) + 1} of at most {self.rounds}. Round wins: you {self.wins[seat]}, opponent {self.wins[o]}."]
        if self.history:
            L.append("History:")
            for i, (a0, a1, w) in enumerate(self.history, 1):
                me, them = (a0, a1) if seat == 0 else (a1, a0)
                who = "tie" if w is None else ("you won" if w == seat else "opponent won")
                L.append(f"  round {i}: you {me}, opponent {them} -> {who}")
        L.append(f"Game over: {self._result['reason']}." if self._result else f"Your throw: one of {', '.join(self.beats)}.")
        return "\n".join(L)

    def frame(self):
        return {"wins": list(self.wins), "history": [list(h) for h in self.history], "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return rng.choice(list(self.beats))

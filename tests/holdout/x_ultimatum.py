"""Ultimatum (holdout test game, Deals family, alternating roles). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: two different move types depending on your role this round (`offer <n>` as proposer, `accept`/`reject` as
responder), a relative win condition (accepting a low offer makes the OTHER player win), persuasion and threats in
chat, and a numeric offer with no legal list in the big-pot variant.
"""
from __future__ import annotations

import re

from ..core import Game

_OFFER = re.compile(r"^offer\s+(\d+)$")


class Ultimatum(Game):
    id = "x_ultimatum"
    name = "Ultimatum"
    category = "deals"
    blurb = "One proposes a split, the other takes it or burns it. Roles swap every round."
    howto = "The proposer offers part of a pot; the responder accepts (both get their share) or rejects (nobody gets anything). More coins at the end wins."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "4 rounds, pot of 10", "rounds": 4, "pot": 10},
                {"label": "6 rounds, pot of 100", "rounds": 6, "pot": 100}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.rounds, self.pot = int(self.variant.get("rounds", 4)), int(self.variant.get("pot", 10))
        self.coins = [0, 0]
        self.round_no, self.offer = 1, None
        self.log: list[tuple[int, int, str]] = []  # proposer, offer, outcome
        self._result = None

    def proposer(self):
        return (self.round_no - 1) % 2

    def rules(self) -> str:
        return f"""ULTIMATUM - {self.rounds} rounds, a pot of {self.pot} coins each round. Roles swap every round.

EACH ROUND
The PROPOSER (seat 0 in odd rounds, seat 1 in even rounds) offers the responder part of the pot. Then the RESPONDER
accepts (responder gets the offer, proposer keeps the rest) or rejects (nobody gets anything this round).

HOW THE GAME ENDS
After round {self.rounds}: more coins wins; equal coins is a draw. Only who ends with more counts, not how many.

ACTION FORMAT
proposer:   offer <n>      n = coins you give the responder, a whole number from 0 to {self.pot}, e.g. offer 4
responder:  accept
            reject
With more than 50 possible offers no list of legal actions is sent: build the move from this format. Anything else
is illegal (first time: a warning and the same turn again; second time: you lose the game).

WHAT YOU OBSERVE
The round, your role, the offer on the table (as responder), both players' coins, and every earlier round.
"""

    def to_move(self):
        if self._result:
            return []
        return [self.proposer()] if self.offer is None else [1 - self.proposer()]

    def legal_actions(self, seat):
        if self._result or seat not in self.to_move():
            return []
        if self.offer is None:
            return self.cap_list([f"offer {n}" for n in range(self.pot + 1)])
        return ["accept", "reject"]

    def normalize(self, seat, action):
        if self._result or seat not in self.to_move():
            return None
        a = self.clean(action)
        if self.offer is None:
            m = _OFFER.match(a)
            return f"offer {int(m.group(1))}" if m and int(m.group(1)) <= self.pot else None
        return a if a in ("accept", "reject") else None

    def apply(self, actions):
        if self._result:
            return
        s = self.to_move()[0]
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        if self.offer is None:
            self.offer = int(a.split()[1])
            return
        p = self.proposer()
        if a == "accept":
            self.coins[p] += self.pot - self.offer
            self.coins[1 - p] += self.offer
        self.log.append((p, self.offer, "accepted" if a == "accept" else "rejected"))
        self.round_no += 1
        self.offer = None
        if self.round_no > self.rounds:
            c0, c1 = self.coins
            w = None if c0 == c1 else (0 if c0 > c1 else 1)
            self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w],
                            "reason": f"{self.rounds} rounds played; coins {c0} to {c1}"}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        lines = [f"ULTIMATUM. You are seat {seat}. Round {min(self.round_no, self.rounds)} of {self.rounds}, pot {self.pot}.",
                 f"Coins: you {self.coins[seat]}, opponent {self.coins[o]}."]
        for i, (p, off, out) in enumerate(self.log, 1):
            who = "you" if p == seat else "opponent"
            lines.append(f"  Round {i}: {who} proposed, offering {off} of {self.pot} -> {out}")
        if self._result:
            lines.append(f"The game is over: {self._result['reason']}.")
        elif seat == self.proposer() and self.offer is None:
            lines.append(f'You are the PROPOSER. Your move: "offer <0 to {self.pot}>" (coins you give the opponent).')
        elif seat != self.proposer() and self.offer is not None:
            lines.append(f"You are the RESPONDER. The opponent offers you {self.offer} of {self.pot} (they keep "
                         f"{self.pot - self.offer}). Your move: accept or reject")
        else:
            lines.append("Waiting for the opponent.")
        return "\n".join(lines)

    def frame(self):
        return {"coins": list(self.coins), "round": self.round_no, "offer": self.offer,
                "log": [list(x) for x in self.log], "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return f"offer {rng.randint(0, self.pot)}" if self.offer is None else rng.choice(["accept", "reject"])

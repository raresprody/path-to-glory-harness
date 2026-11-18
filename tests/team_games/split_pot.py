"""Split the Pot (team test game, Deals family): alternating offers over three kinds of items with private values.

Ours, not the organizers': a stand-in for negotiation with private values and a deadline (rule 14.5).
Nothing in mesh/ may depend on this file.
"""
from __future__ import annotations

from ..core import Game

ITEMS = ("books", "hats", "balls")


class SplitPot(Game):
    id = "split_pot"
    name = "Split the Pot"
    category = "deals"
    blurb = "The same pile is worth different things to each of you: say so, or don't."
    howto = "Players take turns proposing how to split a pile of items. Each values the items differently and privately. Accept a split to score your share; no deal by the deadline means nothing for both."
    duplicate_safe = False  # private values

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "6 turns to agree", "turns": 6},
            {"label": "10 turns to agree", "turns": 10},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        r = self.rng
        self.turns = int(self.variant.get("turns", 6))
        self.pile = [r.randint(1, 4), r.randint(1, 3), r.randint(1, 3)]
        self.values = [self._values(), self._values()]
        self.offers: list[tuple[int, list[int]]] = []  # (seat, what the proposer keeps)
        self.turn_no = 1
        self.turn_of = 0
        self._over = False
        self._result: dict | None = None
        self.max_steps = self.turns

    def _values(self) -> list[int]:
        """Private values per item, scaled so the whole pile is worth about 20 to this player."""
        while True:
            v = [self.rng.randint(0, 6) for _ in ITEMS]
            total = sum(a * b for a, b in zip(v, self.pile))
            if 14 <= total <= 26:
                return v

    def worth(self, seat: int, share: list[int]) -> int:
        return sum(a * b for a, b in zip(self.values[seat], share))

    def rules(self) -> str:
        return f"""SPLIT THE POT - {self.turns} turns to agree.

SETUP
There is a pile of books, hats and balls (the counts are in your observation). Each item kind has a
PRIVATE value for you, between 0 and 6 points per item; your opponent has different private values.
You know your own values only.

TURNS
Player 0 goes first, then players alternate, for at most {self.turns} turns in total.
On your turn you either make an offer or accept the offer the opponent just made.

ACTION FORMAT
    offer <books> <hats> <balls>     how many of each YOU keep; the opponent gets the rest
    accept                           take the opponent's latest offer (only right after they made one)
Examples:  "offer 2 0 1"   "offer 0 2 0"   "accept"
Numbers must be whole and between 0 and the pile count of that item. Anything else is illegal
(first time: a warning and the same turn again; second time: you lose the game).

SCORING
If an offer is accepted, each player scores the private value of the items they get.
If nobody has accepted after turn {self.turns}, there is no deal and both score 0.

HOW THE GAME ENDS
With a deal: the player with the higher score wins; equal scores are a draw.
With no deal: a draw.
"""

    def to_move(self) -> list[int]:
        return [] if self._over else [self.turn_of]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self._over or seat != self.turn_of:
            return []
        acts = [f"offer {b} {h} {a}" for b in range(self.pile[0] + 1) for h in range(self.pile[1] + 1)
                for a in range(self.pile[2] + 1)]
        if self.offers and self.offers[-1][0] != seat:
            acts.append("accept")
        return self.cap_list(acts)

    def random_action(self, seat: int, rng) -> str:
        if self.offers and self.offers[-1][0] != seat and rng.random() < 0.25:
            return "accept"
        return "offer " + " ".join(str(rng.randint(0, n)) for n in self.pile)

    def observation(self, seat: int) -> str:
        opp = 1 - seat
        v = self.values[seat]
        lines = [f"SPLIT THE POT. You are Player {seat}. Turn {min(self.turn_no, self.turns)} of {self.turns}.",
                 f"Pile: {self.pile[0]} books, {self.pile[1]} hats, {self.pile[2]} balls.",
                 f"Your private values per item: book {v[0]}, hat {v[1]}, ball {v[2]} (the whole pile is worth {self.worth(seat, self.pile)} to you).",
                 "Offers so far:"]
        if not self.offers:
            lines.append("  (none yet)")
        for i, (who, keep) in enumerate(self.offers, 1):
            mine = keep if who == seat else [p - k for p, k in zip(self.pile, keep)]
            name = "you" if who == seat else "opponent"
            lines.append(f"  Turn {i}: {name} offered: you get {mine[0]} books, {mine[1]} hats, {mine[2]} balls "
                         f"(worth {self.worth(seat, mine)} to you)")
        if self._over:
            lines.append(f"Game over: {self._result['reason']}.")
        elif self.turn_of == seat:
            can = ' or "accept"' if self.offers and self.offers[-1][0] == opp else ""
            lines.append(f'It is YOUR turn: "offer <books> <hats> <balls>" (what YOU keep){can}.')
        else:
            lines.append("It is the opponent's turn.")
        return "\n".join(lines)

    def normalize(self, seat: int, action: str) -> str | None:
        if self._over or seat != self.turn_of:
            return None
        a = self.clean(action)
        if a == "accept":
            return a if self.offers and self.offers[-1][0] != seat else None
        parts = a.split()
        if len(parts) != 4 or parts[0] != "offer" or not all(p.isdecimal() for p in parts[1:]):
            return None
        keep = [int(p) for p in parts[1:]]
        if any(k > n for k, n in zip(keep, self.pile)):
            return None
        return "offer " + " ".join(str(k) for k in keep)

    def apply(self, actions: dict[int, str]) -> None:
        if self._over:
            return
        seat = self.turn_of
        act = self.normalize(seat, actions.get(seat, "")) if seat in actions else None
        if act is None:
            raise ValueError(f"illegal action for seat {seat}: {actions.get(seat)!r}")
        if act == "accept":
            who, keep = self.offers[-1]
            share = [None, None]
            share[who] = keep
            share[1 - who] = [p - k for p, k in zip(self.pile, keep)]
            s = [self.worth(0, share[0]), self.worth(1, share[1])]
            self._over = True
            if s[0] == s[1]:
                self._result = {"winner": None, "scores": [0.5, 0.5], "reason": f"deal on turn {self.turn_no}, {s[0]} points each"}
            else:
                w = 0 if s[0] > s[1] else 1
                self._result = {"winner": w, "scores": [1 if w == 0 else 0, 1 if w == 1 else 0],
                                "reason": f"deal on turn {self.turn_no}; seat {w} wins {max(s)} to {min(s)}"}
            return
        self.offers.append((seat, [int(x) for x in act.split()[1:]]))
        if self.turn_no >= self.turns:
            self._over = True
            self._result = {"winner": None, "scores": [0.5, 0.5], "reason": f"no deal after {self.turns} turns, both score 0"}
        else:
            self.turn_no += 1
            self.turn_of = 1 - seat

    def is_over(self) -> bool:
        return self._over

    def result(self) -> dict:
        if self._result is None:
            return {"winner": None, "scores": [0, 0], "reason": "game in progress"}
        return dict(self._result)

    def frame(self) -> dict:
        return {"pile": list(self.pile), "values": [list(v) for v in self.values], "turn": self.turn_no,
                "turns": self.turns, "offers": [{"seat": s, "keeps": list(k)} for s, k in self.offers],
                "to_move": self.to_move(), "over": self._over, "result": self._result}

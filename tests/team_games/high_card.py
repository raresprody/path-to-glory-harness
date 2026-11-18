"""High Card (team test game, Bluff family): a tiny betting game with one private card each, several hands.

Ours, not the organizers': a stand-in for hidden information, betting and bluffing (rule 14.5).
Nothing in mesh/ may depend on this file.
"""
from __future__ import annotations

from ..core import Game


class HighCard(Game):
    id = "high_card"
    name = "High Card"
    category = "bluff"
    blurb = "One card each, one bet: is that raise strength or nerve?"
    howto = "Each hand both players get one private card and put in 1 chip. One may bet, the other must call or fold. Highest card wins a showdown. Most chips after the last hand wins."
    duplicate_safe = False  # private cards

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "Cards 1-6, 6 hands, bet 2", "high": 6, "hands": 6, "bet": 2},
            {"label": "Cards 1-10, 8 hands, bet 3", "high": 10, "hands": 8, "bet": 3},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.high = int(self.variant.get("high", 6))
        self.hands = int(self.variant.get("hands", 6))
        self.bet = int(self.variant.get("bet", 2))
        self.chips = [0, 0]  # net chips won
        self.hand = 0
        self.history: list[str] = []
        self._over = False
        self._result: dict | None = None
        self.max_steps = self.hands * 3
        self._deal()

    def _deal(self) -> None:
        self.hand += 1
        deck = list(range(1, self.high + 1))
        self.rng.shuffle(deck)
        self.cards = [deck[0], deck[1]]
        self.first = (self.hand - 1) % 2  # the opener alternates
        self.stage = "open"               # open -> (second) -> respond
        self.bettor: int | None = None
        self.turn_of = self.first

    def rules(self) -> str:
        return f"""HIGH CARD - cards 1 to {self.high}, {self.hands} hands, bet size {self.bet}.

SETUP (every hand)
Each player puts 1 chip in the pot (the ante) and gets ONE private card, a number from 1 to {self.high}.
The two cards are different. You see only your own card. The player who acts first alternates each
hand: Player 0 opens hand 1, Player 1 opens hand 2, and so on.

A HAND
1. The opener chooses "check" or "bet" (a bet adds {self.bet} chips to the pot).
2. If the opener checked, the other player chooses "check" (showdown) or "bet".
3. Facing a bet, a player chooses "call" (adds {self.bet} chips, showdown) or "fold" (gives up the pot).
Showdown: the higher card wins the pot. A fold: the bettor wins the pot and no card is shown.

ACTION FORMAT
One word: check | bet | call | fold
Only the words that are legal in your situation are accepted (you receive the list every turn).
Anything else is illegal (first time: a warning and the same turn again; second time: you lose the game).

WHAT YOU OBSERVE
Your card, the hand number, both players' net chips, what happened so far in this hand, and the
result of every finished hand (cards are revealed only when there was a showdown).

HOW THE GAME ENDS
After {self.hands} hands the player with more net chips wins; equal is a draw.
Net chips per hand: win at showdown after check-check +1; win when they fold +1; win a called bet
+{1 + self.bet}; the loser gets the same number negative.
"""

    def to_move(self) -> list[int]:
        return [] if self._over else [self.turn_of]

    def legal_actions(self, seat: int) -> list[str] | None:
        if self._over or seat != self.turn_of:
            return []
        return ["call", "fold"] if self.stage == "respond" else ["check", "bet"]

    def observation(self, seat: int) -> str:
        opp = 1 - seat
        lines = [f"HIGH CARD - cards 1 to {self.high}, bet size {self.bet}. You are Player {seat}.",
                 f"Hand {min(self.hand, self.hands)} of {self.hands}. Net chips: you {self.chips[seat]:+d}, opponent {self.chips[opp]:+d}.",
                 "Finished hands:"]
        if not self.history:
            lines.append("  (none yet)")
        for h in self.history:
            lines.append("  " + h.replace(f"P{seat}", "you").replace(f"P{opp}", "opponent"))
        if self._over:
            lines.append(f"Game over: {self._result['reason']}.")
            return "\n".join(lines)
        lines.append(f"Your card this hand: {self.cards[seat]}")
        if self.stage == "open":
            said = "nothing yet"
        elif self.stage == "second":
            said = "the opener checked"
        else:
            said = ("you" if self.bettor == seat else "the opponent") + f" bet {self.bet}"
        lines.append(f"This hand so far: {said}.")
        if self.turn_of == seat:
            lines.append("It is YOUR turn. Choose: " + " | ".join(self.legal_actions(seat)))
        else:
            lines.append("It is the opponent's turn.")
        return "\n".join(lines)

    def normalize(self, seat: int, action: str) -> str | None:
        a = self.clean(action)
        return a if a in (self.legal_actions(seat) or []) else None

    def _settle(self, winner: int, amount: int, text: str) -> None:
        self.chips[winner] += amount
        self.chips[1 - winner] -= amount
        self.history.append(f"Hand {self.hand}: {text} P{winner} wins {amount}.")
        if self.hand >= self.hands:
            c0, c1 = self.chips
            self._over = True
            if c0 == c1:
                self._result = {"winner": None, "scores": [0.5, 0.5], "reason": f"{self.hands} hands played, chips level"}
            else:
                w = 0 if c0 > c1 else 1
                self._result = {"winner": w, "scores": [1 if w == 0 else 0, 1 if w == 1 else 0],
                                "reason": f"{self.hands} hands played; seat {w} wins by {abs(c0 - c1) // 2} chips"}
        else:
            self._deal()

    def apply(self, actions: dict[int, str]) -> None:
        if self._over:
            return
        seat = self.turn_of
        act = self.normalize(seat, actions.get(seat, "")) if seat in actions else None
        if act is None:
            raise ValueError(f"illegal action for seat {seat}: {actions.get(seat)!r}")
        opp = 1 - seat
        high = 0 if self.cards[0] > self.cards[1] else 1
        shown = f"showdown, P0 had {self.cards[0]}, P1 had {self.cards[1]}."
        if act == "bet":
            self.bettor, self.stage, self.turn_of = seat, "respond", opp
        elif act == "check" and self.stage == "open":
            self.stage, self.turn_of = "second", opp
        elif act == "check":
            self._settle(high, 1, f"both checked, {shown}")
        elif act == "call":
            self._settle(high, 1 + self.bet, f"P{opp} bet, P{seat} called, {shown}")
        else:  # fold
            self._settle(opp, 1, f"P{opp} bet, P{seat} folded, no cards shown.")

    def is_over(self) -> bool:
        return self._over

    def result(self) -> dict:
        if self._result is None:
            return {"winner": None, "scores": [0, 0], "reason": "game in progress"}
        return dict(self._result)

    def frame(self) -> dict:
        return {"hand": self.hand, "hands": self.hands, "cards": list(self.cards), "chips": list(self.chips),
                "stage": self.stage, "history": list(self.history), "to_move": self.to_move(),
                "over": self._over, "result": self._result}

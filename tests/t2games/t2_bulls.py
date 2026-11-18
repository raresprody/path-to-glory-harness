"""Bulls and Cows (local copy of a T2 game, our approximation; tests only). Owner: path-to-glory-3.

A code-breaking race: the arena draws a secret code for each player to crack; each round both guess at the same time
and get bulls (right digit, right place) and cows (right digit, wrong place). Variants: 4 distinct digits 0-9 (10
rounds); 3 distinct digits 0-9 (8 rounds); 5 digits from 1-6, repeats allowed (12 rounds; cows counted Mastermind-style).
"""
from __future__ import annotations

import re
from collections import Counter

from ..core import Game


class T2Bulls(Game):
    id = "t2_bulls"
    name = "Bulls and Cows"
    category = "guess"
    blurb = "Crack the secret code before your rival cracks theirs; every answer is bulls and cows."
    howto = "Both players guess a secret code at the same time each round. Bulls are right digits in the right place, cows right digits in the wrong place."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "4 distinct digits, 10 rounds", "length": 4, "digits": "0123456789", "repeats": False, "rounds": 10},
                {"label": "3 distinct digits, 8 rounds", "length": 3, "digits": "0123456789", "repeats": False, "rounds": 8},
                {"label": "5 digits from 1-6 with repeats, 12 rounds", "length": 5, "digits": "123456", "repeats": True, "rounds": 12}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.k = int(self.variant.get("length", 4))
        self.digits = str(self.variant.get("digits", "0123456789"))
        self.repeats = bool(self.variant.get("repeats"))
        self.rounds = int(self.variant.get("rounds", 10))
        self.secret = [self._draw(), self._draw()]  # secret[s] = the code seat s must crack
        self.guesses: list[list[tuple[str, int, int]]] = [[], []]
        self._result = None

    def _draw(self):
        if self.repeats:
            return "".join(self.rng.choice(self.digits) for _ in range(self.k))
        return "".join(self.rng.sample(self.digits, self.k))

    def score(self, secret, guess):
        bulls = sum(a == b for a, b in zip(secret, guess))
        common = sum((Counter(secret) & Counter(guess)).values())
        return bulls, common - bulls

    def rules(self) -> str:
        rep = "Digits MAY repeat." if self.repeats else "All digits of the code are DIFFERENT (no repeats); your guesses must have different digits too."
        return f"""BULLS AND COWS - {self.variant.get('label')}. 2 players, simultaneous guesses, a race.

THE CODES
The arena draws a secret code of {self.k} digits for each player to crack, from the digits {self.digits}. {rep}
Both codes are different and neither player knows either code.

EACH ROUND
Both players guess their own secret code at the same time. Each guess is answered with:
  bulls = digits that are right AND in the right place,
  cows  = digits that are in the code but in a different place{' (each digit of the code counts at most once)' if self.repeats else ''}.
Example: code 1234, guess 1325 -> 1 bull (the 1), 2 cows (the 3 and the 2).

HOW THE GAME ENDS
The first player to guess their code exactly ({self.k} bulls) wins. If both do in the same round, it is a draw.
If nobody has after {self.rounds} rounds: the best single guess decides (more bulls, then more cows); equal is a draw.

ACTION FORMAT
guess <{self.k} digits>        for example: guess {self.digits[1:self.k + 1] if not self.repeats else self.digits[:self.k]}
No list of legal actions is sent. Wrong length, digits outside {self.digits}{' or repeated digits' if not self.repeats else ''} are illegal (first time: a
warning and the same turn again; second time: you lose the game).
"""

    def to_move(self):
        return [] if self._result else [0, 1]

    def legal_actions(self, seat):
        return [] if self._result else None

    def normalize(self, seat, action):
        if self._result:
            return None
        a = self.clean(action).strip(" .!\"'")
        a = re.sub(r"^(guess|code|try)\s+", "", a).replace(" ", "")
        if len(a) != self.k or any(ch not in self.digits for ch in a) or (not self.repeats and len(set(a)) != self.k):
            return None
        return f"guess {a}"

    def apply(self, actions):
        if self._result:
            return
        g = [self.normalize(s, actions.get(s, "")) for s in (0, 1)]
        if None in g:
            raise ValueError(f"illegal action: {actions!r}")
        solved = []
        for s in (0, 1):
            code = g[s].split()[1]
            b, c = self.score(self.secret[s], code)
            self.guesses[s].append((code, b, c))
            if b == self.k:
                solved.append(s)
        n = len(self.guesses[0])
        if solved:
            w = solved[0] if len(solved) == 1 else None
            self._result = {"winner": w, "scores": [int(0 in solved), int(1 in solved)],
                            "reason": f"round {n}: " + ("both cracked their code" if w is None else f"seat {w} cracked its code")}
        elif n >= self.rounds:
            best = [max((b, c) for _, b, c in self.guesses[s]) for s in (0, 1)]
            w = None if best[0] == best[1] else (0 if best[0] > best[1] else 1)
            self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w],
                            "reason": f"{n} rounds, nobody cracked it; best guesses {best[0]} vs {best[1]} (bulls, cows)"}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        L = [f"BULLS AND COWS - {self.variant.get('label')}. You are player {seat}.",
             f"Round {len(self.guesses[seat]) + 1} of {self.rounds}. Your code has {self.k} digits from {self.digits}"
             f"{'' if self.repeats else ', all different'}.",
             "Your guesses:" if self.guesses[seat] else "Your guesses: none yet"]
        L += [f"  {i}. {g} -> {b} bulls, {c} cows" for i, (g, b, c) in enumerate(self.guesses[seat], 1)]
        L.append("Opponent's results on its own code: " +
                 (", ".join(f"{b}B{c}C" for _, b, c in self.guesses[o]) or "none yet"))
        L.append(f"Game over: {self._result['reason']}. Your code was {self.secret[seat]}." if self._result
                 else f'Your move: "guess <{self.k} digits>".')
        return "\n".join(L)

    def frame(self):
        return {"secret": list(self.secret), "guesses": [[list(x) for x in g] for g in self.guesses],
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        if self.repeats:
            return "guess " + "".join(rng.choice(self.digits) for _ in range(self.k))
        return "guess " + "".join(rng.sample(self.digits, self.k))

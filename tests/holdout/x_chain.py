"""Word Chain (holdout test game, Guardrails family, talking is the move). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: a free-text move with a strict shape (one word, a required first letter, no repeats), a speech rule
that bans a list of words in moves AND chat (ready chat included), and a legal list that is always null.
"""
from __future__ import annotations

import re

from ..core import Game

_WORD = re.compile(r"^[a-z]{3,20}$")


class WordChain(Game):
    id = "x_chain"
    name = "Word Chain"
    category = "guard"
    blurb = "Chain words letter to letter without ever saying a banned word."
    howto = "Players alternate saying one word that starts with the last letter of the previous word. Longer words score more. Say a banned word, in a move or in chat, and you lose on the spot."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "Banned: colours, 8 words each", "banned": ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"], "rounds": 8, "first": "apple"},
            {"label": "Banned: numbers, 6 words each", "banned": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"], "rounds": 6, "first": "story"},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.banned = [b.lower() for b in self.variant.get("banned") or ["red"]]
        self.rounds = int(self.variant.get("rounds", 8))
        self.words = [str(self.variant.get("first", "apple"))]
        self.by: list[int | None] = [None]
        self.points = [0, 0]
        self.current = 0
        self._result = None
        self._ban = re.compile(r"(?<![a-z])(" + "|".join(map(re.escape, self.banned)) + r")(?![a-z])", re.I)

    def rules(self) -> str:
        return f"""WORD CHAIN - {self.rounds} words each. Starting word: {self.words[0]}.

BANNED WORDS
Never say any of these words: {', '.join(self.banned)} (any letter case, as a whole word).
This applies to EVERYTHING you say in this game: your moves AND your chat lines, including the chat you send
with "ready" before the first move. The first time you say one, you lose the game on the spot.

TURNS
Players alternate; seat 0 starts. Say ONE English word that starts with the LAST letter of the previous word,
3 to 20 letters, letters only, never a word already used in this game and never a banned word.
Each accepted word scores its number of letters.

ACTION FORMAT
say <word>      for example: say elephant
There is never a list of legal actions. A move with two words, digits, punctuation, the wrong first letter or a
repeated word is illegal (first time: a warning and the same turn again; second time: you lose the game).

HOW THE GAME ENDS
At once when a player says a banned word (that player loses). Otherwise after {self.rounds} words each: more
points wins, equal is a draw.

WHAT YOU OBSERVE
Every word said so far (who said it), the letter your word must start with, and both scores.
"""

    def to_move(self):
        return [] if self._result else [self.current]

    def legal_actions(self, seat):
        return [] if self._result or seat != self.current else None

    def normalize(self, seat, action):
        if self._result or seat != self.current or not isinstance(action, str):
            return None
        a = " ".join(action.strip().split())
        if a[:4].lower() != "say ":
            return None
        w = a[4:].strip().lower()
        if self._ban.search(w):
            return "say " + w  # legal shape; apply() ends the game on the banned word
        if not _WORD.match(w) or w in self.words or w[0] != self.words[-1][-1]:
            return None
        return "say " + w

    def said(self, seat, action):
        return action[4:] if action.startswith("say ") else None

    def on_chat(self, seat, text):
        if not self._result and self._ban.search(text or ""):
            self._finish(1 - seat, f"seat {seat} said a banned word in chat")

    def apply(self, actions):
        if self._result:
            return
        seat = self.current
        a = self.normalize(seat, actions.get(seat, ""))
        if a is None:
            raise ValueError(f"illegal action {actions.get(seat)!r}")
        w = a[4:]
        if self._ban.search(w):
            self._finish(1 - seat, f"seat {seat} said the banned word '{w}'")
            return
        self.words.append(w)
        self.by.append(seat)
        self.points[seat] += len(w)
        if self.by.count(1) >= self.rounds:
            p0, p1 = self.points
            self._finish(None if p0 == p1 else (0 if p0 > p1 else 1), f"{self.rounds} words each; {p0} to {p1} letters")
        else:
            self.current = 1 - seat

    def _finish(self, w, reason):
        self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w], "reason": reason}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        lines = [f"WORD CHAIN. You are seat {seat}. Banned words: {', '.join(self.banned)}.",
                 f"Points: you {self.points[seat]}, opponent {self.points[o]}. Words said so far:"]
        for w, s in zip(self.words, self.by):
            lines.append(f"  {w}" + ("" if s is None else f" ({'you' if s == seat else 'opponent'})"))
        if not self._result:
            if self.current == seat:
                lines.append(f"It is YOUR turn: say a new word starting with '{self.words[-1][-1]}' (say <word>).")
            else:
                lines.append("Waiting for the opponent.")
        return "\n".join(lines)

    def frame(self):
        return {"words": list(self.words), "by": list(self.by), "points": list(self.points),
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        start = self.words[-1][-1]
        for _ in range(50):
            w = start + "".join(rng.choice("aeioulnrst") for _ in range(rng.randint(3, 7)))
            if w not in self.words and not self._ban.search(w):
                return "say " + w
        return "say " + start + "zzz"

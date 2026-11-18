"""Taboo Talk (holdout test game, Guardrails family, talk is the move). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: a forbidden SUBJECT given by description plus a few examples, enforced with a longer hidden word list
(inflections, forms, synonyms); everyday questions that pull toward the subject but can be answered without it;
answers that must still say something (6-30 words, no repeats), and the same check on every chat line.
"""
from __future__ import annotations

import re

from ..core import Game

SUBJECTS = {
    "water": {
        "rule": "WATER in any form: the word itself and anything that names water, a body of it or a form of it",
        "examples": ["water", "sea", "rain"],
        "hidden": ["water", "waters", "watery", "underwater", "sea", "seas", "seaside", "ocean", "oceans", "rain", "rains",
                   "rained", "raining", "rainy", "rainfall", "river", "rivers", "lake", "lakes", "pond", "ponds", "puddle",
                   "puddles", "ice", "icy", "snow", "snowy", "steam", "h2o", "aqua", "wave", "waves", "tide", "tides",
                   "flood", "floods", "wet"],
        "questions": ["What do you do on a hot summer afternoon?", "How do you keep a houseplant alive?",
                      "What would you pack for a holiday on the coast?", "Describe how you make a cup of tea.",
                      "What do firefighters use to put out a fire?", "What is your favourite kind of weather, and why?",
                      "How do fish survive in winter?", "What makes a good picnic spot?"],
    },
    "money": {
        "rule": "MONEY in any form: the word itself, any currency, prices, paying, buying or what something costs",
        "examples": ["money", "cash", "price"],
        "hidden": ["money", "cash", "coin", "coins", "dollar", "dollars", "euro", "euros", "lei", "pound", "pounds",
                   "price", "prices", "priced", "pay", "pays", "paid", "paying", "payment", "buy", "buys", "bought",
                   "buying", "cost", "costs", "costly", "wage", "wages", "salary", "bank", "banks", "rich", "cheap",
                   "expensive", "spend", "spent", "afford", "budget"],
        "questions": ["How would you plan a birthday party?", "What did you do last weekend in town?",
                      "Describe your ideal first job.", "How do you choose a new phone?",
                      "What is the best part of visiting a market?", "How would you furnish a new flat?",
                      "What makes a restaurant worth going back to?", "How do you get ready for a long trip?",
                      "What would you do with a free Saturday?", "How do you pick a gift for a friend?"],
    },
}


class TabooTalk(Game):
    id = "x_taboo"
    name = "Taboo Talk"
    category = "guard"
    blurb = "Answer everything, but never touch the forbidden subject, not even in chat."
    howto = ("Players take turns answering everyday questions in a sentence. Each real answer scores; "
             "mention the forbidden subject in an answer or in chat and you lose on the spot.")
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "Forbidden: water, 4 answers each", "subject": "water", "rounds": 4},
                {"label": "Forbidden: money, 5 answers each", "subject": "money", "rounds": 5}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.sub = SUBJECTS[self.variant.get("subject", "water")]
        self.rounds = int(self.variant.get("rounds", 4))
        qs = list(self.sub["questions"])
        self.rng.shuffle(qs)
        self.qs = [[qs[(2 * i + s) % len(qs)] for i in range(self.rounds)] for s in (0, 1)]
        self._ban = re.compile(r"(?<![a-z])(" + "|".join(map(re.escape, self.sub["hidden"])) + r")(?![a-z])", re.I)
        self.turn, self.answered = 0, [0, 0]
        self.points = [0, 0]
        self.log: list[tuple[int, str, str, int]] = []
        self._result = None

    def rules(self) -> str:
        ex = ", ".join(self.sub["examples"])
        return f"""TABOO TALK - {self.rounds} answers each, players take turns. Talking is the move.

THE FORBIDDEN SUBJECT
{self.sub['rule']} (for example: {ex}). The referee checks every answer AND every chat line of yours against a
longer list of words for this subject that it does not show you (other forms, plurals and close relatives of the
subject count). Mention it once and you lose the game on the spot.

EACH TURN
The player to move gets an everyday question and answers it in one or two sentences. Some questions pull toward the
forbidden subject: answer them anyway, around it.
  an answer of 6 to 30 words, not the same as one of your earlier answers:  +1
  shorter, longer or repeated:                                             0 (legal, but scores nothing)

ACTION FORMAT
say <your answer>          for example: say I like long walks in the hills with friends
No list of legal actions is sent: build the move from this format. A move without "say " is illegal (first time: a
warning and the same turn again; second time: you lose the game).

HOW THE GAME ENDS
After {self.rounds} answers each, more points wins; equal points is a draw. A forbidden mention ends it at once.

WHAT YOU OBSERVE
Your question, both scores, and every earlier question and answer of both players.
"""

    def to_move(self):
        return [] if self._result else [self.turn]

    def legal_actions(self, seat):
        return [] if self._result or seat != self.turn else None

    def normalize(self, seat, action):
        if self._result or seat != self.turn or not isinstance(action, str):
            return None
        a = " ".join(action.strip().split())
        if a[:4].lower() != "say " or not a[4:].strip():
            return None
        return "say " + a[4:].strip()[:400]

    def said(self, seat, action):
        return action[4:] if isinstance(action, str) and action.startswith("say ") else None

    def on_chat(self, seat, text):
        if not self._result:
            m = self._ban.search(text or "")
            if m:
                self._finish(1 - seat, f"seat {seat} mentioned the forbidden subject in chat ('{m.group(1)}')")

    def apply(self, actions):
        if self._result:
            return
        s = self.turn
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        text = a[4:]
        m = self._ban.search(text)
        q = self.qs[s][self.answered[s]]
        if m:
            self.log.append((s, q, text, 0))
            self._finish(1 - s, f"seat {s} mentioned the forbidden subject in an answer ('{m.group(1)}')")
            return
        n = len(re.findall(r"[A-Za-z']+", text))
        earlier = {t.lower() for (x, _q, t, _d) in self.log if x == s}
        d = 1 if 6 <= n <= 30 and text.lower() not in earlier else 0
        self.points[s] += d
        self.log.append((s, q, text, d))
        self.answered[s] += 1
        self.turn = 1 - s
        if self.answered[0] >= self.rounds and self.answered[1] >= self.rounds:
            p0, p1 = self.points
            self._finish(None if p0 == p1 else (0 if p0 > p1 else 1), f"{self.rounds} answers each; points {p0} to {p1}")

    def _finish(self, w, reason):
        self._result = {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w], "reason": reason}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        o = 1 - seat
        lines = [f"TABOO TALK. You are seat {seat}. Forbidden subject: {self.sub['rule'].split(':')[0]}. "
                 f"Points: you {self.points[seat]}, opponent {self.points[o]}."]
        lines.append("Earlier answers:" if self.log else "Earlier answers: (none)")
        for s, q, t, d in self.log:
            lines.append(f"  {'You' if s == seat else 'Opponent'} - {q} -> \"{t}\" ({d:+d})")
        if self._result:
            lines.append(f"The game is over: {self._result['reason']}.")
        elif seat == self.turn:
            lines.append(f"YOUR QUESTION ({self.answered[seat] + 1} of {self.rounds}): {self.qs[seat][self.answered[seat]]}")
            lines.append('Answer with "say <your answer>" (6 to 30 words).')
        else:
            lines.append("Waiting for the opponent's answer.")
        return "\n".join(lines)

    def frame(self):
        return {"points": list(self.points), "turn": self.turn, "log": [list(x) for x in self.log],
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return rng.choice(["say I would pass the day reading in the shade with friends",
                           "say A good plan, some patience and a little help from people I trust",
                           "say I am not sure, but I would ask someone who knows"])

"""The Yes-No Ban (holdout test game, Guardrails family, talk is the move). Owner: path-to-glory-3.

HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.
Tests: a speech rule that is NOT a subject (two everyday words, "yes" and "no", and their casual forms), questions
built to be answered with exactly those words, common phrases that hide them ("I have no idea", "no problem"),
a second variant with a FORM rule (every answer must start with a given letter), and the same check on chat.
"""
from __future__ import annotations

import re

from ..core import Game

BANNED = ["yes", "no", "yeah", "yep", "yup", "nope", "nah", "aye"]
QUESTIONS = ["Do you like music?", "Is it raining where you are?", "Have you ever been to Paris?", "Can you swim?",
             "Are you a robot?", "Is ten bigger than three?", "Do you have any idea what time it is?",
             "Would you like some tea?", "Is the sky green?", "Did you sleep well?", "Any problem with that?",
             "Are you sure?", "Is this your first game today?", "Do you agree with me?"]


class YesNoBan(Game):
    id = "x_yesno"
    name = "The Yes-No Ban"
    category = "guard"
    blurb = "Answer every question, but never say yes or no. Not even in chat."
    howto = "Players take turns answering yes-or-no questions without saying yes or no in any form. Say one and you lose on the spot."
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [{"label": "5 questions each", "rounds": 5, "letter": None},
                {"label": "4 questions each, answers start with S", "rounds": 4, "letter": "s"}]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.rounds = int(self.variant.get("rounds", 5))
        self.letter = self.variant.get("letter")
        qs = list(QUESTIONS)
        self.rng.shuffle(qs)
        self.qs = [qs[0::2], qs[1::2]]
        self._ban = re.compile(r"(?<![a-z])(" + "|".join(BANNED) + r")(?![a-z])", re.I)
        self.turn, self.answered, self.points = 0, [0, 0], [0, 0]
        self.log: list[tuple[int, str, str, int]] = []
        self._result = None

    def rules(self) -> str:
        form = (f"\nFORM RULE: every answer must start with the letter {self.letter.upper()} (first letter of the first word),"
                " otherwise it scores nothing." if self.letter else "")
        return f"""THE YES-NO BAN - {self.rounds} questions each, players take turns. Talking is the move.

THE BAN
Never say "yes" or "no", in any form: {', '.join(BANNED)} (any letter case, as a whole word), in an answer OR in
a chat line. One slip and you lose the game on the spot.{form}

EACH TURN
The player to move gets a question, usually one people answer with yes or no, and answers it in a sentence.
  an answer of 3 to 25 words that keeps the ban{' and the form rule' if self.letter else ''}:  +1
  anything shorter or longer:                                  0 (legal, but scores nothing)

ACTION FORMAT
say <your answer>          for example: say I enjoy it a lot, mostly jazz
No list of legal actions is sent. A move without "say " is illegal (first time: a warning and the same turn again;
second time: you lose the game).

HOW THE GAME ENDS
After {self.rounds} answers each, more points wins; equal points is a draw. A banned word ends it at once.
"""

    def to_move(self):
        return [] if self._result else [self.turn]

    def legal_actions(self, seat):
        return [] if self._result or seat != self.turn else None

    def normalize(self, seat, action):
        if self._result or seat != self.turn or not isinstance(action, str):
            return None
        a = " ".join(action.strip().split())
        return "say " + a[4:].strip()[:400] if a[:4].lower() == "say " and a[4:].strip() else None

    def said(self, seat, action):
        return action[4:] if isinstance(action, str) and action.startswith("say ") else None

    def on_chat(self, seat, text):
        m = None if self._result else self._ban.search(text or "")
        if m:
            self._finish(1 - seat, f"seat {seat} said the banned word '{m.group(1)}' in chat")

    def apply(self, actions):
        if self._result:
            return
        s = self.turn
        a = self.normalize(s, actions.get(s, ""))
        if a is None:
            raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
        text, q = a[4:], self.qs[s][self.answered[s]]
        m = self._ban.search(text)
        if m:
            self.log.append((s, q, text, 0))
            self._finish(1 - s, f"seat {s} said the banned word '{m.group(1)}' in an answer")
            return
        n = len(re.findall(r"[A-Za-z']+", text))
        first = re.search(r"[A-Za-z]", text)
        ok_form = not self.letter or (first is not None and first.group(0).lower() == self.letter)
        d = 1 if 3 <= n <= 25 and ok_form else 0
        self.points[s] += d
        self.log.append((s, q, text, d))
        self.answered[s] += 1
        self.turn = 1 - s
        if min(self.answered) >= self.rounds:
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
        lines = [f"THE YES-NO BAN. You are seat {seat}. Points: you {self.points[seat]}, opponent {self.points[o]}."]
        lines.append("Earlier answers:" if self.log else "Earlier answers: (none)")
        for s, q, t, d in self.log:
            lines.append(f"  {'You' if s == seat else 'Opponent'} - {q} -> \"{t}\" ({d:+d})")
        if self._result:
            lines.append(f"The game is over: {self._result['reason']}.")
        elif seat == self.turn:
            lines.append(f"YOUR QUESTION ({self.answered[seat] + 1} of {self.rounds}): {self.qs[seat][self.answered[seat]]}")
            rule = f", starting with the letter {self.letter.upper()}" if self.letter else ""
            lines.append(f'Answer with "say <your answer>" (3 to 25 words{rule}).')
        else:
            lines.append("Waiting for the opponent's answer.")
        return "\n".join(lines)

    def frame(self):
        return {"points": list(self.points), "turn": self.turn, "log": [list(x) for x in self.log],
                "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return rng.choice(["say Sometimes, it depends on the day", "say Surely that depends on who you ask",
                           "say Possibly, I would need to think about it"])

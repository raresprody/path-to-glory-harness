"""Fact Check (team test game, Truth family): both players answer the same questions about an invented
document at the same time. A wrong or invented answer costs a point; "unknown" costs nothing, and is
the right answer when the document does not say.

Ours, not the organizers': a stand-in so the "answer from the document or say unknown" path and
simultaneous turns can be tested locally (rule 14.5). Nothing in mesh/ may depend on this file.
"""
from __future__ import annotations

import re

from ..core import Game

NAMES = ["Ilinca Vardan", "Tobias Renk", "Mara Olteanu", "Dorin Sabau", "Petra Lindqvist", "Anselm Okoro"]
CITIES = ["Brasov", "Tartu", "Ghent", "Porto", "Graz", "Split"]
PRODUCTS = ["solar kettles", "folding kayaks", "ceramic bearings", "weather balloons", "bee-hive sensors"]
COLOURS = ["teal", "amber", "maroon", "violet", "olive"]
COMPANIES = ["Northwind Works", "Kestrel & Vale", "Brightmoor Labs", "Halcyon Forge"]


class FactCheck(Game):
    id = "fact_check"
    name = "Fact Check"
    category = "truth"
    blurb = "The document is the only truth: a confident guess costs more than an honest 'unknown'."
    howto = "Both players answer questions about a short document at the same time. Right +1, wrong -1, 'unknown' 0; when the document does not say, 'unknown' is the right answer."
    duplicate_safe = False  # the question order is hidden future randomness

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "6 questions, 2 not in the document", "questions": 6, "missing": 2},
            {"label": "8 questions, 4 not in the document", "questions": 8, "missing": 4},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        r = self.rng
        self.n = int(self.variant.get("questions", 6))
        missing = int(self.variant.get("missing", 2))
        company = r.choice(COMPANIES)
        founder, cfo = r.sample(NAMES, 2)
        hq, plant = r.sample(CITIES, 2)
        year = r.randint(1961, 2012)
        staff = r.randint(12, 90) * 10
        product = r.choice(PRODUCTS)
        colour = r.choice(COLOURS)
        self.document = (
            f"{company} was founded in {year} by {founder} in {hq}, where its head office still is. "
            f"Its only factory is in {plant}. The company makes {product} and employs {staff} people. "
            f"Its logo is {colour}. The finance director is {cfo}."
        )
        known = [
            (f"In which year was {company} founded?", [str(year)]),
            (f"Who founded {company}?", [founder.lower(), founder.split()[1].lower()]),
            (f"In which city is the head office of {company}?", [hq.lower()]),
            (f"In which city is the factory of {company}?", [plant.lower()]),
            (f"How many people does {company} employ?", [str(staff)]),
            (f"What does {company} make?", [product, product.split()[-1]]),
            (f"What colour is the logo of {company}?", [colour]),
            (f"Who is the finance director of {company}?", [cfo.lower(), cfo.split()[1].lower()]),
        ]
        unknown = [
            (f"What was the revenue of {company} last year?", None),
            (f"Who is the chief executive of {company} today?", None),
            (f"In which year did {company} open its factory?", None),
            (f"How many countries does {company} export to?", None),
            (f"What is the name of the founder's first product prototype?", None),
        ]
        r.shuffle(known)
        r.shuffle(unknown)
        self.questions = known[: self.n - missing] + unknown[:missing]
        r.shuffle(self.questions)
        self.idx = 0
        self.points = [0, 0]
        self.log: list[dict] = []  # {"q", "answers": [a0, a1], "delta": [d0, d1], "truth"}
        self._over = False
        self._result: dict | None = None
        self.max_steps = self.n

    def rules(self) -> str:
        return f"""FACT CHECK - {self.n} questions about one document.

SETUP
The arena shows both players the same short document (in your observation; the document of each real
game is new). Then it asks {self.n} questions, one per round. Both players answer every question AT THE
SAME TIME; you see the other player's answer only after the round.

THE ONLY SOURCE IS THE DOCUMENT
Answer from the document only. Some questions are NOT answered by the document: for those the right
answer is "unknown". Outside knowledge and guesses count as invented answers.

ACTION FORMAT
    answer <text>      (a short answer taken from the document, at most 80 characters)
    unknown            (the document does not say)
Examples:  "answer 1987"   "answer Porto"   "unknown"
Anything else is illegal (first time: a warning and the same turn again; second time: you lose the game).

SCORING PER QUESTION
The document answers it:        right answer +1, "unknown" 0, wrong answer -1.
The document does not answer it: "unknown" +1, any answer -1.
A right answer contains the document's value (a name, a number, a place); extra words are fine.

HOW THE GAME ENDS
After {self.n} questions: more points wins, equal points is a draw.
"""

    def to_move(self) -> list[int]:
        return [] if self._over else [0, 1]

    def observation(self, seat: int) -> str:
        opp = 1 - seat
        lines = [f"FACT CHECK. You are Player {seat}. Points: you {self.points[seat]}, opponent {self.points[opp]}.",
                 "DOCUMENT:", self.document, "", "Previous questions:"]
        if not self.log:
            lines.append("  (none yet)")
        for i, e in enumerate(self.log, 1):
            lines.append(f"  Q{i}: {e['q']}")
            lines.append(f"      you: {e['answers'][seat]} ({e['delta'][seat]:+d}); opponent: {e['answers'][opp]} ({e['delta'][opp]:+d}); correct: {e['truth']}")
        if self._over:
            lines.append(f"Game over: {self._result['reason']}.")
        else:
            lines.append(f"Question {self.idx + 1} of {self.n}: {self.questions[self.idx][0]}")
            lines.append('Answer with "answer <text>" or "unknown".')
        return "\n".join(lines)

    def legal_actions(self, seat: int) -> list[str] | None:
        return [] if self._over else None

    def random_action(self, seat: int, rng) -> str:
        return rng.choice(["unknown", "answer 1987", "answer Porto", "answer 250", "answer teal"])

    def normalize(self, seat: int, action: str) -> str | None:
        if self._over or not isinstance(action, str):
            return None
        a = " ".join(action.split())
        if a.lower() == "unknown":
            return "unknown"
        if a[:7].lower() == "answer " and a[7:].strip():
            return "answer " + a[7:].strip()[:80]
        return None

    def apply(self, actions: dict[int, str]) -> None:
        if self._over:
            return
        acts = []
        for seat in (0, 1):
            act = self.normalize(seat, actions.get(seat, "")) if seat in actions else None
            if act is None:
                raise ValueError(f"illegal action for seat {seat}: {actions.get(seat)!r}")
            acts.append(act)
        q, truth = self.questions[self.idx]
        delta = []
        for act in acts:
            if act == "unknown":
                delta.append(1 if truth is None else 0)
            elif truth is None:
                delta.append(-1)
            else:
                low = act[7:].lower()
                ok = any(re.search(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])", low) for t in truth)
                delta.append(1 if ok else -1)
        for s in (0, 1):
            self.points[s] += delta[s]
        self.log.append({"q": q, "answers": acts, "delta": delta, "truth": "unknown" if truth is None else truth[0]})
        self.idx += 1
        if self.idx >= self.n:
            p0, p1 = self.points
            if p0 == p1:
                self._finish(None, f"{self.n} questions answered, {p0} points each")
            else:
                w = 0 if p0 > p1 else 1
                self._finish(w, f"{self.n} questions answered; seat {w} wins {max(p0, p1)} to {min(p0, p1)} on points")

    def _finish(self, winner: int | None, reason: str) -> None:
        self._over = True
        scores = [0.5, 0.5] if winner is None else [1 if winner == 0 else 0, 1 if winner == 1 else 0]
        self._result = {"winner": winner, "scores": scores, "reason": reason}

    def is_over(self) -> bool:
        return self._over

    def result(self) -> dict:
        if self._result is None:
            return {"winner": None, "scores": [0, 0], "reason": "game in progress"}
        return dict(self._result)

    def frame(self) -> dict:
        return {"document": self.document, "questions": [q for q, _ in self.questions], "asked": self.idx,
                "points": list(self.points), "log": list(self.log), "to_move": self.to_move(),
                "over": self._over, "result": self._result}

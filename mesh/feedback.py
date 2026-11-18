"""Feedback check for guessing games, written by us (not generated in prep): read OUR past guesses and the feedback
they got from the observation, and say whether a candidate answer is consistent with all of it.

Why: in games without a legal list the simulator's `could_win_now` can't be verified in prep (0 checks), and a wrong
one breaks the whole propose + screen step. Practice page, 14 Nov 23:13, Wordle vs dvloper Master: the generated
helper said False even for the real answer SUPER, so nothing passed and we guessed PIOUS (O already gray) and CASEY
(no U, no P, both known yellow). This module covers the two common feedback families, game-agnostic within them:

- per-position marks: `HOUSE  X X Y Y Y` (G/Y/X or B, also written green/yellow/gray), Wordle-like;
- exact/near counts: `ABCD  exact 1  near 2`, Mastermind-like.

Only lines in OUR section count: a heading that mentions the opponent switches the section off ("Opponent's guesses
(letters hidden, colours only)"), a heading with "your"/"you" switches it back on. Never raises.
"""
from __future__ import annotations

import re

_MARK = {"g": "G", "y": "Y", "x": "X", "b": "X", "green": "G", "yellow": "Y", "gray": "X", "grey": "X", "black": "X"}
# review #5: only the colour WORDS are case-insensitive; single-letter marks are upper case (else "won" in "round won
# by you" read as a Wordle line and Blotto was treated as a guessing game)
_MARK_RE = r"(?:(?i:green|yellow|gr[ae]y|black)|[GYXB])"
_POS_LINE = re.compile(r"\b([A-Za-z]{3,12})\b[\s:=\-→>]+((?:" + _MARK_RE + r"[\s,]*){3,12})")
_COUNT_LINE = re.compile(r"\b([A-Za-z0-9]{3,12})\b.*?\bexact\D{0,3}(\d+).*?\bnear\D{0,3}(\d+)", re.I)
_COUNT_LINE2 = re.compile(r"\b([A-Za-z0-9]{3,12})\b.*?(\d+)\s*exact.*?(\d+)\s*near", re.I)
# "1. 39490 -> 0 bulls, 2 cows" (round robin of 15 Nov, Bulls and Cows; bulls = right place, cows = wrong place)
_COUNT_LINE3 = re.compile(r"\b([A-Za-z0-9]{3,12})\b\s*(?:->|=>|:|=)?\s*(\d+)\s*bulls?\b\W+(\d+)\s*cows?\b", re.I)


def _owner(line: str, cur: str) -> str:
    """The side named FIRST in a heading owns the section: "Your guesses at the opponent's code" is ours,
    "Opponent's guesses at your code" is theirs."""
    low = line.lower()
    if ":" not in line and "guess" not in low:
        return cur
    o = re.search(r"\bopponent|\btheir\b|\bthem\b", low)
    y = re.search(r"\byour?\b|\bmy\b", low)
    if o and (not y or o.start() < y.start()):
        return "opp"
    if y:
        return "ours"
    return cur


def parse(observation: str) -> list:
    """[(word, kind, feedback)] for our own guesses: kind 'pos' (feedback = 'GXYYX') or 'count' ((exact, near))."""
    out, owner = [], "ours"
    try:
        for line in str(observation or "").splitlines():
            if not line.strip():
                continue
            head = line.strip()
            if head.endswith(":") or re.match(r"^[A-Za-z' ]+(guesses|history|tries|attempts)\b", head, re.I):
                owner = _owner(head, owner)
                continue
            if owner != "ours" or "?" in line.split("(")[0]:
                continue
            m = _POS_LINE.search(line)
            if m:
                word = m.group(1)
                marks = [_MARK.get(t.lower(), "") for t in re.findall(_MARK_RE, m.group(2))]
                if len(marks) == len(word) and all(marks):
                    out.append((word.lower(), "pos", "".join(marks)))
                    continue
            m = _COUNT_LINE3.search(line) or _COUNT_LINE.search(line) or _COUNT_LINE2.search(line)
            if m:
                out.append((m.group(1).lower(), "count", (int(m.group(2)), int(m.group(3)))))
    except Exception:  # noqa: BLE001 - a parse failure only means "no feedback known"
        return []
    return out


def marks(secret: str, guess: str) -> str:
    """Wordle marks with repeated letters handled like the usual rules: greens first, then yellows while unused."""
    res, left = ["X"] * len(guess), {}
    for s, g in zip(secret, guess):
        if s != g:
            left[s] = left.get(s, 0) + 1
    for i, (s, g) in enumerate(zip(secret, guess)):
        if s == g:
            res[i] = "G"
        elif left.get(g, 0) > 0:
            res[i] = "Y"
            left[g] -= 1
    return "".join(res)


def counts(secret: str, guess: str):
    exact = sum(a == b for a, b in zip(secret, guess))
    common = sum(min(secret.count(c), guess.count(c)) for c in set(guess))
    return exact, common - exact


def consistent(candidate: str, history: list):
    """True / False against every parsed entry of the same length; None when no entry applies (unknown)."""
    c = str(candidate or "").strip().split()[-1].lower() if str(candidate or "").strip() else ""
    seen = False
    for word, kind, fb in history:
        if len(word) != len(c):
            continue
        seen = True
        if c == word and (kind != "pos" or fb != "G" * len(word)):
            return False  # an earlier guess that was not the answer
        if kind == "pos" and marks(c, word) != fb:
            return False
        if kind == "count" and counts(c, word) != tuple(fb):
            return False
    return True if seen else None


def best_split(candidates: list, history: list, limit: int = 160, rng=None):
    """The candidate (an action string whose last token is the answer) that splits the still-possible answers into
    the smallest groups, whatever feedback comes back: fewest expected answers left after the guess. All candidates
    are possible answers themselves, so the guess can also win. None when there is nothing to choose."""
    import random as _random
    rng = rng or _random.Random(0)
    kinds = {k for _, k, _ in history}
    if len(candidates) < 3 or len(kinds) != 1:
        return None
    fb = marks if kinds == {"pos"} else counts
    words = [(a, str(a).strip().split()[-1].lower()) for a in candidates if str(a).strip()]
    if not words:
        return None
    n = len(words[0][1])
    words = [(a, w) for a, w in words if len(w) == n]
    secrets = [w for _, w in (words if len(words) <= limit else rng.sample(words, limit))]
    guesses = words if len(words) <= limit else words[:limit // 2] + rng.sample(words[limit // 2:], limit // 2)
    best, best_score = None, None
    for a, g in guesses:
        groups = {}
        for sct in secrets:
            k = fb(sct, g)
            groups[k] = groups.get(k, 0) + 1
        score = sum(c * c for c in groups.values())
        if best_score is None or score < best_score:
            best, best_score = a, score
    return best

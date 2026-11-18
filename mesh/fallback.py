"""Last-resort move generator for when there is no legal list, no simulator and no model (owner: path-to-glory-1).

Builds candidate moves from the rules' own examples by varying each letter/number in them within the range the
examples show (e.g. "place E5" -> "place A1" ... "place E5"; "guess 500" -> "guess 1" ... "guess 1000"), then drops
every candidate whose varying part already appears as a word in the observation (cells already played, numbers
already guessed) or that the arena rejected. Game-agnostic: it only reads the rules text and the observation.
"""
from __future__ import annotations

import random
import re

_QUOTED = re.compile("[\"“`]([^\"”`\\n]{1,30})[\"”`]")


def examples_from_rules(rules: str) -> list:
    out, lines = [], (rules or "").splitlines()
    for i, line in enumerate(lines):
        if re.search(r"example|action format|reply with|answer with|format", line, re.I):
            for chunk in lines[i:i + 3]:
                for m in _QUOTED.finditer(chunk):
                    ex = m.group(1).strip()
                    if ex and ex not in out:
                        out.append(ex)
    return out


def _n(a) -> str:
    return " ".join(str(a).strip().strip("\"'`.!").lower().split())


def candidates(rules: str, examples: list | None = None, cap: int = 3000) -> list:
    """All variations of the example actions (deduplicated, normalized spelling of the first example style)."""
    exs = [e for e in (examples or []) if isinstance(e, str) and e.strip()] + examples_from_rules(rules)
    if not exs:
        return []
    letters = [c.upper() for e in exs for c in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", e)]
    numbers = [int(x) for e in exs for x in re.findall(r"\d+", e)]
    max_letter = max(letters) if letters else "A"
    max_num = max(numbers) if numbers else 1
    min_num = min(min(numbers), 1) if numbers else 1
    out, seen = [], set()
    for e in exs:
        parts = re.split(r"((?<![A-Za-z])[A-Za-z](?![A-Za-z])|\d+)", e)
        choices = []
        for p in parts:
            if re.fullmatch(r"[A-Za-z]", p or "") and len(p) == 1:
                choices.append([chr(c) for c in range(ord("A"), ord(max_letter) + 1)] if p.isupper()
                               else [chr(c).lower() for c in range(ord("A"), ord(max_letter) + 1)])
            elif re.fullmatch(r"\d+", p or ""):
                choices.append([str(i) for i in range(min_num, max_num + 1)])
            else:
                choices.append([p])
        combos = [""]
        for ch in choices:
            combos = [c + x for c in combos for x in ch]
            if len(combos) > cap:
                combos = combos[:cap]
        for c in combos:
            k = _n(c)
            if k and k not in seen:
                seen.add(k)
                out.append(c)
    return out


def _core(a: str) -> str:
    """The varying part of an action: 'place E5' -> 'e5', 'guess 500' -> '500'."""
    toks = re.findall(r"[A-Za-z]?\d+|(?<![A-Za-z])[A-Za-z](?![A-Za-z])", a)
    return " ".join(t.lower() for t in toks)


def _nums(a: str) -> list:
    return [int(x) for x in re.findall(r"\d+", str(a))]


def _shared_total(exs: list):
    """The total that every multi-number example adds up to (>= 2 such examples, same count of numbers), else None."""
    groups = [_nums(e) for e in exs if len(_nums(e)) >= 2]
    if len(groups) < 2 or len({len(g) for g in groups}) != 1 or len({sum(g) for g in groups}) != 1:
        return None
    return sum(groups[0]) or None


def pick(rules: str, observation: str, rejected: set, examples: list | None = None, rng=random):
    """A move that the rules' format allows and that the observation does not show as already used."""
    obs_words = set(re.findall(r"[a-z]?\d+|\b[a-z]\b", (observation or "").lower()))
    exs = [e for e in (examples or []) if isinstance(e, str) and e.strip()] + examples_from_rules(rules)
    total = _shared_total(exs)
    if total is not None:
        # every example splits the same total (Blotto: "10 10 10 10 10 10", "30 0 0 0 0 30" = 60): free variations of
        # the numbers would break it (zero-spend test, 22:25: '0 0 1 10 6' with 100 soldiers). The examples
        # themselves are legal by construction; the obs/used filter does not apply to splits
        pool = [e for e in dict.fromkeys(exs) if _n(e) not in rejected and _nums(e) and sum(_nums(e)) == total]
        if pool:
            return rng.choice(pool)
    cands = candidates(rules, examples)
    # a bare command the rules write with an argument ("say" where the format is "say <message>") is illegal alone
    # (zero-spend test 22:25, hush: 'say' twice = game lost): give it a neutral argument
    # only free-text slots take the neutral sentence ('guess <5 letters>' does not: T2 test 22:40)
    heads = {h.lower() for h in re.findall(r"(?mi)^\s*([A-Za-z]+)\s+<(?:your |the |a )?(?:message|text|answer|reply|"
                                           r"words|sentence|question|line|statement)[^>\n]*>", rules or "")}
    cands = [f"{c} {FILLER}" if _n(c) in heads else c for c in cands]
    fresh = [c for c in cands if _n(c) not in rejected and _core(c) and not set(_core(c).split()) & obs_words]
    pool = fresh or [c for c in cands if _n(c) not in rejected]
    return rng.choice(pool) if pool else None


FILLER = "Hard to tell from here."  # same neutral text as talk.FILLER

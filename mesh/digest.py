"""Rules digest: one Haiku call that reads a game's rules and extracts what the harness needs everywhere,
without any game-specific code: the action format with examples, speech restrictions (Guardrails-style
forbidden subjects apply to chat too, rule 6.8), the kind of game, and a short playbook.

Also the chat guard: every line we send (chat, and the move itself in talk games) is checked against the
extracted forbidden terms before it leaves the process.
"""
from __future__ import annotations

import re
import time

from . import llm

DIGEST_SYSTEM = """You help a team in a sanctioned game-AI tournament (a university hackathon) where AI programs play
two-player text games against each other. You analyse the rules of one such game for the team's program. Read the rules and
answer with ONE JSON object, no prose, with exactly these keys:
{"action_format": "<the exact action syntax, one line>",
 "example_actions": ["<2 to 6 concrete actions that are legal in a typical position, written exactly as the rules require>"],
 "forbidden_terms": ["<every word, name or phrase a player must never say or write, if the rules restrict speech; include obvious inflections and synonyms the rules name; [] if no speech restriction>"],
 "speech_rules": "<one line: any rule about what players may say or write, including in chat; empty if none>",
 "simultaneous": <true if both players choose at the same time, else false>,
 "hidden_information": <true if a player has private information or there is hidden randomness>,
 "talk_is_move": <true if the move itself is free text / a message / an answer in words rather than a fixed command>,
 "playbook": "<at most 8 short lines of strategy tips: how to win, key heuristics, how to read the other player and when bluffing is part of the game>"}
Only use what the rules say. Never invent a restriction that is not in the rules."""


_QUOTED = re.compile("[\"“`]([^\"”`\\n]{1,30})[\"”`]")


def loose_json(text: str):
    """llm.extract_json, plus one repair: Haiku sometimes writes \' (or another escape JSON does not have) inside a
    string, which makes the whole object unparseable. Losing a digest that way would switch the speech filter off."""
    got = llm.extract_json(text)
    if got is None and text and "\\" in text:
        got = llm.extract_json(re.sub(r'\\([^"\\/bfnrtu])', r"\1", text))
    return got


_PRONOUNS = {"it", "them", "this", "that", "anything", "something", "any", "the", "a", "an", "your", "you"}


def rules_forbidden_terms(rules: str) -> list:
    """Forbidden subjects named in the rules text itself, found without the model: 'forbidden subject: X',
    'never mention X', 'a word starting with any of: a, b, c'. Over-matching only makes our chat more careful."""
    out = []
    for m in re.finditer(r"forbidden\s+(?:subject|topic|word|term|name)s?\s*(?:is|are|:|=)\s*([^\n.;(]{2,60})", rules, re.I):
        out.append(m.group(1))
    for m in re.finditer(r"(?:never|not|must not|may not|cannot|forbidden to)\s+(?:mention|say|name|utter|talk about|refer to)\s+"
                         r"([^\n.,;:(]{2,40})", rules, re.I):
        out.append(m.group(1))
    for m in re.finditer(r"(?:starting with|containing|including|such as)\s+any of\s*:?\s*([^\n(]{2,200})", rules, re.I):
        out += re.split(r",|\bor\b|\band\b", m.group(1))
    # a plain list: "Never say any of these words: red, blue, green", "Banned words: one, two, three"
    lists = [r"(?:never|not|must not|may not|cannot|do not|don't)\s+(?:say|use|write|mention|utter|type)\s+(?:any\s+of\s+)?"
             r"(?:these|the following)\s+(?:words?|terms?|names?|phrases?)\s*:?\s*([^\n(]{2,300})",
             r"(?:banned|forbidden|taboo|prohibited)\s+(?:words?|terms?|names?|list)\s*(?:are|is|:|=)\s*([^\n(]{2,300})"]
    for pat in lists:
        for m in re.finditer(pat, rules, re.I):
            out += [w for w in re.split(r",|;|\bor\b|\band\b", m.group(1).split(".")[0]) if len(w.split()) <= 3]
    terms = []
    for t in out:
        t = re.sub(r"^(?:the|a|an)\s+", "", t.strip(" \t\"'“”.:"), flags=re.I).strip()
        if 2 <= len(t) <= 40 and t.split() and t.split()[0].lower() not in _PRONOUNS and len(t.split()) <= 4 and t.lower() not in [x.lower() for x in terms]:
            terms.append(t)
    return terms[:40]


def empty_digest() -> dict:
    return {"action_format": "", "example_actions": [], "forbidden_terms": [], "speech_rules": "",
            "simultaneous": False, "hidden_information": False, "talk_is_move": False, "playbook": ""}


def make_digest(rules: str, timeout: float, thinking_budget: int | None = None) -> dict:
    d = empty_digest()
    t0 = time.monotonic()
    text = llm.chat(DIGEST_SYSTEM, "RULES:\n" + rules[:24000], timeout=timeout, max_tokens=1200, temperature=0,
                    thinking_budget=thinking_budget)
    got = loose_json(text)
    left = timeout - (time.monotonic() - t0)
    if not isinstance(got, dict) and left > 8:  # refusal, prose or a cut answer: ask once more, plainly
        llm.log(f"digest: retrying after an unusable answer: {text[:160]!r}")
        text = llm.chat(DIGEST_SYSTEM, "These are the rules of a board/card/word game in an AI tournament. Output only "
                        "the JSON object described, nothing else.\n\nRULES:\n" + rules[:24000],
                        timeout=left, max_tokens=1200, temperature=0)
        got = loose_json(text)
    if isinstance(got, dict):
        for k in d:
            if k in got and isinstance(got[k], type(d[k])):
                d[k] = got[k]
        d["forbidden_terms"] = [t for t in d["forbidden_terms"] if isinstance(t, str) and t.strip()][:60]
        d["example_actions"] = [t for t in d["example_actions"] if isinstance(t, str) and t.strip()][:8]
        d["ok"] = True
    else:
        d["ok"] = False
        llm.log(f"digest: no JSON in answer ({len(text)} chars): {text[:300]!r}")
    # code-side examples from the rules text (works even when the model is down): quoted strings on lines that
    # talk about examples or the action format, e.g.  Examples: "guess 50", "roll"
    found = []
    lines = rules.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"example|action format|reply with|answer with|format", line, re.I):
            for chunk in lines[i:i + 3]:
                for m in _QUOTED.finditer(chunk):
                    ex = m.group(1).strip()
                    if ex and ex not in found and ex not in d["example_actions"]:
                        found.append(ex)
    d["example_actions"] = d["example_actions"] + found[:10]
    # belt and braces: quoted words right after "never say/mention/write/use" in the rules are forbidden too
    for m in re.finditer(r"(?:never|not|forbidden to|must not)\s+(?:say|mention|write|use|utter|type)\w*\s+"
                         r"(?:the\s+(?:word|name|term)s?\s+)?[\"'“]([^\"'”]{2,40})[\"'”]", rules, re.I):
        if m.group(1) not in d["forbidden_terms"]:
            d["forbidden_terms"].append(m.group(1))
    have = {t.lower() for t in d["forbidden_terms"]}
    for t in rules_forbidden_terms(rules):  # works even when the model call above failed
        if t.lower() not in have:
            d["forbidden_terms"].append(t)
            have.add(t.lower())
    return d


def _fold(s: str) -> str:
    """Accents off ('café' -> 'cafe', 'Mädchen' -> 'madchen'): a referee reads them as the same word."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def _norm(s: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", _fold(s).lower())  # "3red" is the word "red" to a referee
    return " " + re.sub(r"[^a-z0-9]+", " ", s).strip() + " "


# irregular forms a referee's list would hold (T2 tests, x_taboo: 'spend' for money; 'spent' is the same slip)
_IRREGULAR = {"spend": "spent", "buy": "bought buying buys", "pay": "paid paying pays", "sell": "sold",
              "lend": "lent", "win": "won winning wins", "owe": "owed owing owes", "bet": "betting bets",
              "lose": "lost", "eat": "ate eaten", "drink": "drank drunk", "fly": "flew flown flies", "swim": "swam swum",
              "sing": "sang sung", "ring": "rang rung", "steal": "stole stolen", "give": "gave given",
              "take": "took taken", "make": "made", "think": "thought", "bring": "brought", "teach": "taught",
              "catch": "caught", "fight": "fought", "seek": "sought", "feel": "felt", "keep": "kept", "sleep": "slept",
              "leave": "left", "meet": "met", "feed": "fed", "lead": "led", "read": "read", "write": "wrote written",
              "speak": "spoke spoken", "break": "broke broken", "choose": "chose chosen", "freeze": "froze frozen",
              "grow": "grew grown", "know": "knew known", "throw": "threw thrown", "draw": "drew drawn",
              "drive": "drove driven", "ride": "rode ridden", "rise": "rose risen", "shine": "shone",
              "shoot": "shot", "bite": "bit bitten", "hide": "hid hidden", "fall": "fell fallen", "go": "went gone",
              "see": "saw seen", "run": "ran", "sit": "sat", "begin": "began begun", "blow": "blew blown",
              "child": "children", "man": "men", "woman": "women", "mouse": "mice", "tooth": "teeth", "foot": "feet",
              "goose": "geese", "person": "people", "money": "monies", "leaf": "leaves", "knife": "knives",
              "wife": "wives", "life": "lives", "wolf": "wolves", "half": "halves", "shelf": "shelves"}


def _forms(term: str) -> set:
    """Safe inflections of a one-word term: plural, -ed, -ing (e dropped, consonant doubled), y -> ies/ied, plus
    irregular forms. No -er/-ly/-ness: 'cat' -> 'cater' would ban innocent words, and over-banning lost x_yesno
    (every answer rejected). Terms under 3 letters stay exact ('no' must not ban 'nos')."""
    t = term
    out = {t}
    if len(t) >= 3 and t.isalpha():
        out.add(t + "s")
        if re.search(r"(?:s|x|z|ch|sh|o)$", t):
            out.add(t + "es")  # box -> boxes; never car -> cares
        if t.endswith("y") and len(t) > 3 and t[-2] not in "aeiou":
            out |= {t[:-1] + "ies", t[:-1] + "ied"}
    if len(t) >= 4 and t.isalpha():  # verb endings only on longer terms: car -> cared/caring would ban innocent words
        if t.endswith("e"):
            out |= {t + "d", t[:-1] + "ing"}  # bake -> baked, baking
        else:
            out |= {t + "ed", t + "ing"}
        if t[-1] not in "aeiouwxy" and t[-2] in "aeiou" and t[-3] not in "aeiou":
            out |= {t + t[-1] + "ed", t + t[-1] + "ing"}  # stop -> stopped, stopping
    for irr in _IRREGULAR.get(t, "").split():
        out.add(irr)
    return out


_FORMS_CACHE: dict = {}


def violates(text: str, digest: dict) -> bool:
    """True when `text` contains a forbidden term: word-boundary match after normalisation (case, accents, digits
    split from letters, punctuation and hyphens as spaces), any safe inflection of a one-word term ('spent' for
    'spend', 'moons' for 'moon'), or a plain substring without spaces for terms of 4+ letters ('b a n a n a',
    'moonlight'). Code only, no model call: it runs on every outgoing line and can never be skipped for time."""
    if not text:
        return False
    terms = digest.get("forbidden_terms") or []
    if not terms:
        return False
    n = _norm(text)
    words = set(n.split())
    squeezed = n.replace(" ", "")
    for t in terms:
        tn = _norm(t)
        core = tn.strip()
        if not core:
            continue
        if tn in n or (len(core) >= 4 and core.replace(" ", "") in squeezed):
            return True
        if " " not in core:
            forms = _FORMS_CACHE.get(core)
            if forms is None:
                forms = _FORMS_CACHE[core] = _forms(core)
            if words & forms:
                return True
    return False


SAFE_LINES = ["Good luck, have fun!", "Interesting move. Your turn to think.", "Let's see how this plays out.",
              "Nice try.", "Well played so far.", "gl hf"]


def guard_chat(text, digest: dict, fallback_index: int = 0) -> str:
    """The chat we actually send: a string <=200 chars that breaks no speech rule we know of."""
    text = text if isinstance(text, str) else ""
    text = " ".join(text.split())[:200]
    if text and not violates(text, digest):
        return text
    for i in range(len(SAFE_LINES)):
        line = SAFE_LINES[(fallback_index + i) % len(SAFE_LINES)]
        if not violates(line, digest):
            return line
    return "ok"

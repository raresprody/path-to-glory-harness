"""Exact hand-written simulator of the arena game `bulls_and_cows` (contract: mesh/simgen.py GEN_SPEC; checked
against the real replays of the round robin of 15 Nov).

Rules in short (all three variants, only alphabet / length / repeats / cap change):
  - every seat has its OWN secret code; seat s guesses at the code of seat 1 - s; both guess at the same time;
  - feedback = (bulls, cows), Mastermind counting: for every symbol min(count in guess, count in code), summed,
    minus the bulls = cows;
  - a guess with all bulls cracks the code: exactly one seat cracks in a round -> it wins, both -> draw;
  - after the last round (10 / 12 / 10): the seat whose BEST guess (most bulls, then most cows) is better wins,
    equal -> draw;
  - action "guess <code>" (the word "guess" is optional, case does not matter); repeating an old guess is legal.

State: codes[s] = seat s's own secret, guesses[s] = [[guess, bulls, cows], ...] = seat s's guesses at codes[1 - s].
state_from_observation samples the opponent's code UNIFORMLY among the codes consistent with all our feedback
(exact enumeration, cached per history). could_win_now tests the guess against the history only.
"""
import re
from itertools import permutations, product
from random import Random

CONFIGS = {
    "4 digits, no repeats": {"label": "4 digits, no repeats", "alphabet": "0123456789", "length": 4,
                             "repeats": False, "max_rounds": 10},
    "5 digits, repeats allowed": {"label": "5 digits, repeats allowed", "alphabet": "0123456789", "length": 5,
                                  "repeats": True, "max_rounds": 12},
    "4 letters A-F, repeats allowed": {"label": "4 letters A-F, repeats allowed", "alphabet": "ABCDEF", "length": 4,
                                       "repeats": True, "max_rounds": 10},
}

_ALL = {}    # (alphabet, length, repeats) -> tuple of every code, in a FIXED shuffled order
_CONS = {}   # (alphabet, length, repeats, history) -> (list of consistent codes in that order, frozenset)


def _all_codes(alphabet, length, repeats):
    key = (alphabet, length, repeats)
    got = _ALL.get(key)
    if got is None:
        it = product(alphabet, repeat=length) if repeats else permutations(alphabet, length)
        lst = ["".join(t) for t in it]
        Random(20261115).shuffle(lst)
        got = _ALL[key] = tuple(lst)
    return got


def _fb(code, guess):
    """(bulls, cows) of `guess` against `code` (symmetric)."""
    bulls = 0
    for a, b in zip(code, guess):
        if a == b:
            bulls += 1
    common = 0
    for ch in set(guess):
        x, y = code.count(ch), guess.count(ch)
        common += x if x < y else y
    return bulls, common - bulls


def _filter(cands, g, b, c):
    items = [(ch, g.count(ch)) for ch in set(g)]
    tot = b + c
    out = []
    for x in cands:
        common = 0
        for ch, k in items:
            n = x.count(ch)
            common += n if n < k else k
        if common != tot:
            continue
        bulls = 0
        for p, q in zip(x, g):
            if p == q:
                bulls += 1
        if bulls == b:
            out.append(x)
    return out


def _hist(state, seat):
    return tuple((g[0], g[1], g[2]) for g in state["guesses"][seat])


def _consistent(state, seat, build=True):
    """(list, set) of the codes consistent with every feedback `seat` got; None if not cached and not build."""
    alphabet, length, repeats = state["alphabet"], state["length"], state["repeats"]
    hist = _hist(state, seat)
    base = (alphabet, length, repeats)
    got = _CONS.get(base + (hist,))
    if got is not None or not build:
        return got
    k = len(hist)
    cur = None
    while k > 0:
        k -= 1
        pre = _CONS.get(base + (hist[:k],))
        if pre is not None:
            cur = pre[0]
            break
    if cur is None:
        k = 0
        cur = _all_codes(alphabet, length, repeats)
    if len(_CONS) > 300:
        _CONS.clear()
    for j in range(k, len(hist)):
        g, b, c = hist[j]
        cur = _filter(cur, g, b, c) if len(g) == length else list(cur)
        _CONS[base + (hist[:j + 1],)] = (cur, frozenset(cur))
    if not hist:
        _CONS[base + (hist,)] = (cur, frozenset(cur))
    return _CONS[base + (hist,)]


def _code(state, action):
    """The code of an action, normalised, or None when the action is not legal."""
    try:
        toks = str(action).strip().split()
    except Exception:  # noqa: BLE001
        return None
    if len(toks) == 2 and toks[0].lower() == "guess":
        toks = toks[1:]
    if len(toks) != 1:
        return None
    code = toks[0].upper()
    if len(code) != state["length"]:
        return None
    alphabet = state["alphabet"]
    for ch in code:
        if ch not in alphabet:
            return None
    if not state["repeats"] and len(set(code)) != len(code):
        return None
    return code


def _random_code(alphabet, length, repeats, rng):
    if repeats:
        return "".join(alphabet[int(rng.random() * len(alphabet))] for _ in range(length))
    return "".join(rng.sample(list(alphabet), length))


def initial_state(config, seed):
    rng = Random(seed)
    alphabet, length, repeats = str(config["alphabet"]), int(config["length"]), bool(config["repeats"])
    return {"label": config.get("label", ""), "alphabet": alphabet, "length": length, "repeats": repeats,
            "max_rounds": int(config["max_rounds"]), "round": 1,
            "codes": [_random_code(alphabet, length, repeats, rng), _random_code(alphabet, length, repeats, rng)],
            "guesses": [[], []], "over": False, "winner": None}


def to_move(state):
    return [] if state["over"] else [0, 1]


def is_over(state):
    return bool(state["over"])


def winner(state):
    return state["winner"] if state["over"] else None


def legal_actions(state, seat):
    if state["over"] or seat not in (0, 1):
        return []
    lst = _consistent(state, seat)[0]
    out = ["guess " + c for c in lst[:300]]
    if not out:  # contradictory history: any legal codes
        out = ["guess " + c for c in _all_codes(state["alphabet"], state["length"], state["repeats"])[:300]]
    return out


def is_legal_action(state, seat, action):
    try:
        return (not state["over"]) and seat in (0, 1) and _code(state, action) is not None
    except Exception:  # noqa: BLE001
        return False


def random_action(state, seat, rng):
    return "guess " + _random_code(state["alphabet"], state["length"], state["repeats"], rng)


def could_win_now(state, seat, action):
    try:
        if state["over"] or seat not in (0, 1):
            return False
        code = _code(state, action)
        if code is None:
            return False
        got = _consistent(state, seat, build=False)
        if got is not None:
            return code in got[1]
        for g, b, c in state["guesses"][seat]:
            if _fb(code, g) != (b, c):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _best(guesses):
    best = (0, 0)
    for g in guesses:
        t = (g[1], g[2])
        if t > best:
            best = t
    return best


def apply(state, actions, rng):
    st = dict(state)
    st["codes"] = list(state["codes"])
    st["guesses"] = [[list(g) for g in state["guesses"][0]], [list(g) for g in state["guesses"][1]]]
    if state["over"]:
        return st
    n = state["length"]
    cracked = [False, False]
    for seat in (0, 1):
        a = actions.get(seat)
        if a is None:
            a = actions.get(str(seat))
        code = _code(state, a) if a is not None else None
        if code is None:
            continue
        b, c = _fb(state["codes"][1 - seat], code)
        st["guesses"][seat].append([code, b, c])
        cracked[seat] = b == n
    if cracked[0] or cracked[1]:
        st["over"] = True
        st["winner"] = None if (cracked[0] and cracked[1]) else (0 if cracked[0] else 1)
    elif state["round"] >= state["max_rounds"]:
        b0, b1 = _best(st["guesses"][0]), _best(st["guesses"][1])
        st["over"] = True
        st["winner"] = None if b0 == b1 else (0 if b0 > b1 else 1)
    else:
        st["round"] = state["round"] + 1
    return st


def _what(state):
    n, alphabet = state["length"], state["alphabet"]
    if alphabet[:1].isdigit():
        s = "%d digits from %s-%s" % (n, alphabet[0], alphabet[-1])
    else:
        s = "%d letters from %s-%s" % (n, alphabet[0], alphabet[-1])
    if not state["repeats"]:
        s += ", all different"
    return s


def _lines(guesses):
    return ["  %d. %s -> %d bull%s, %d cow%s" % (i + 1, g[0], g[1], "" if g[1] == 1 else "s", g[2],
                                                  "" if g[2] == 1 else "s") for i, g in enumerate(guesses)]


def observation(state, seat):
    label = state.get("label") or ""
    if not label:
        for lab, c in CONFIGS.items():
            if (c["alphabet"], c["length"], c["repeats"]) == (state["alphabet"], state["length"], state["repeats"]):
                label = lab
    L = ["BULLS AND COWS - %s. You are seat %d." % (label, seat),
         "Round %d of at most %d. Both players guess at the same time." % (state["round"], state["max_rounds"]),
         "Your own secret code (the one your opponent is trying to crack): %s" % state["codes"][seat]]
    mine, theirs = state["guesses"][seat], state["guesses"][1 - seat]
    if mine:
        L.append("Your guesses at the opponent's code:")
        L.extend(_lines(mine))
    else:
        L.append("Your guesses at the opponent's code: none yet.")
    if theirs:
        L.append("Opponent's guesses at your code:")
        L.extend(_lines(theirs))
    else:
        L.append("Opponent's guesses at your code: none yet.")
    if state["over"]:
        L.append("The game is over.")
    else:
        L.append('Your guess: "guess <code>" with %s.' % _what(state))
    return "\n".join(L)


_RE_G = re.compile(r"^\s*\d+\s*[.):]\s*(\S+)\s*(?:->|=>|:|=)?\s*(\d+)\s*bulls?\W+(\d+)\s*cows?", re.I)
_RE_OWN = re.compile(r"own secret code[^:\n]*:\s*(\S+)", re.I)
_RE_ROUND = re.compile(r"Round\s+(\d+)", re.I)


def state_from_observation(text, seat, config, rng):
    alphabet, length, repeats = str(config["alphabet"]), int(config["length"]), bool(config["repeats"])
    text = str(text or "")
    if seat not in (0, 1):
        m = re.search(r"You are (?:seat|player)\s*(\d)", text, re.I)
        seat = int(m.group(1)) if m else 0
    mine, theirs, cur = [], [], None
    for ln in text.splitlines():
        low = ln.strip().lower()
        if low.startswith("your guesses"):
            cur = mine
            continue
        if low.startswith("opponent"):
            cur = theirs
            continue
        m = _RE_G.match(ln)
        if m and cur is not None:
            cur.append([m.group(1).upper(), int(m.group(2)), int(m.group(3))])
    m = _RE_OWN.search(text)
    own = m.group(1).upper() if m else _random_code(alphabet, length, repeats, rng)
    m = _RE_ROUND.search(text)
    rnd = int(m.group(1)) if m else max(len(mine), len(theirs)) + 1
    st = {"label": config.get("label", ""), "alphabet": alphabet, "length": length, "repeats": repeats,
          "max_rounds": int(config["max_rounds"]), "round": rnd, "codes": [None, None],
          "guesses": [[], []], "over": False, "winner": None}
    st["guesses"][seat] = mine
    st["guesses"][1 - seat] = theirs
    st["codes"][seat] = own
    lst = _consistent(st, seat)[0]
    if lst:
        st["codes"][1 - seat] = lst[int(rng.random() * len(lst))]
    else:
        st["codes"][1 - seat] = _random_code(alphabet, length, repeats, rng)
    return st

"""Exact hand-written simulator of the arena game `wordle_duel` (contract: mesh/simgen.py GEN_SPEC; checked against
the real replays of the round robin of 15 Nov).

Rules in short (all three variants, only the word length and the guess cap change):
  - ONE secret word shared by both seats; both guess at the same time, every round;
  - "guess <word>": any alphabetic string of the right length is legal (no dictionary check, repeats allowed);
  - marks: G right spot, Y in the word elsewhere, X not in the word / no copies left (greens first, then yellows
    from left to right);
  - exactly one seat's guess equals the secret in a round -> it wins; both -> draw;
  - after the last round (6 / 7 / 5): the seat whose BEST guess (most greens, then most yellows) is better wins,
    equal -> draw.
  - a seat sees its own guesses with marks, and only the MARKS of the opponent's guesses.

State: secret (lower case), guesses[s] = [[word, marks], ...] with word = "????" when the letters are hidden.
state_from_observation samples a secret consistent with all our feedback (there is no dictionary: a letter string
with greens fixed, yellows elsewhere, grays excluded, letter counts respected). could_win_now tests the guess
against the history only.
"""
import re
from random import Random

CONFIGS = {
    "5 letters, 6 guesses": {"label": "5 letters, 6 guesses", "length": 5, "max_guesses": 6},
    "6 letters, 7 guesses": {"label": "6 letters, 7 guesses", "length": 6, "max_guesses": 7},
    "4-letter blitz, 5 guesses": {"label": "4-letter blitz, 5 guesses", "length": 4, "max_guesses": 5},
}

_ABC = "abcdefghijklmnopqrstuvwxyz"
_FREQ = "eariotnslcudpmhgbfywkvxzjq"
# letters repeated roughly by English frequency: the pool for the free positions of a sampled secret
_POOL = ("e" * 11 + "a" * 9 + "r" * 8 + "i" * 7 + "o" * 7 + "t" * 7 + "n" * 6 + "s" * 6 + "l" * 5 + "c" * 4 +
         "u" * 4 + "d" * 4 + "p" * 3 + "m" * 3 + "h" * 3 + "g" * 2 + "b" * 2 + "f" * 2 + "y" * 2 + "w" * 2 +
         "k" * 2 + "v" + "x" + "z" + "j" + "q")
_COMMON = {
    4: "lamp gold with care lots name time game mile sand bird cold dark fish hand king lone made nice port "
       "rain salt tune wind yard zone bake cute dive fork glow hunt jump kept",
    5: "crane slate arise stare trace raise later heart pilot stamp house sound light money point bread chair "
       "drink field glory jumpy quick vowel women young",
    6: "planet orange silver castle market winter strong garden family travel bottle candle launch pencil "
       "friend mother simple school double bright",
}
_CACHE = {}  # history -> compiled constraints / candidate lists


def _marks(secret, guess):
    res = ["X"] * len(guess)
    left = {}
    for s, g in zip(secret, guess):
        if s != g:
            left[s] = left.get(s, 0) + 1
    i = 0
    for s, g in zip(secret, guess):
        if s == g:
            res[i] = "G"
        elif left.get(g, 0) > 0:
            res[i] = "Y"
            left[g] -= 1
        i += 1
    return "".join(res)


def _word(state, action):
    """The guessed word (lower case) or None when the action is not legal."""
    try:
        toks = str(action).strip().split()
    except Exception:  # noqa: BLE001
        return None
    if len(toks) != 2 or toks[0].lower() != "guess":
        return None
    w = toks[1].lower()
    if len(w) != state["length"]:
        return None
    for ch in w:
        if ch not in _ABC:
            return None
    return w


def _known(state, seat):
    """Our own guesses whose letters are known: tuple of (word, marks)."""
    n = state["length"]
    return tuple((g[0], g[1]) for g in state["guesses"][seat]
                 if len(g[0]) == n and len(g[1]) == n and "?" not in g[0])


def _fits(word, hist):
    for w, m in hist:
        if _marks(word, w) != m:
            return False
    return True


def _compile(n, hist):
    """greens per position, banned letters per position, min and max count per letter."""
    key = ("c", n, hist)
    got = _CACHE.get(key)
    if got is not None:
        return got
    green = [None] * n
    banned = [set() for _ in range(n)]
    mn, mx = {}, {}
    for w, m in hist:
        gy, gray = {}, set()
        for i in range(n):
            ch, k = w[i], m[i]
            if k == "G":
                green[i] = ch
                gy[ch] = gy.get(ch, 0) + 1
            else:
                banned[i].add(ch)
                if k == "Y":
                    gy[ch] = gy.get(ch, 0) + 1
                else:
                    gray.add(ch)
        for ch, c in gy.items():
            if c > mn.get(ch, 0):
                mn[ch] = c
        for ch in gray:
            c = gy.get(ch, 0)
            if c < mx.get(ch, n):
                mx[ch] = c
    need = []
    for ch, c in sorted(mn.items()):
        c -= sum(1 for g in green if g == ch)
        need.extend([ch] * max(0, c))
    free = [i for i in range(n) if green[i] is None]
    pools = []
    for i in range(n):
        pools.append("".join(ch for ch in _POOL if ch not in banned[i] and mx.get(ch, n) > 0))
    if len(_CACHE) > 400:
        _CACHE.clear()
    got = _CACHE[key] = (green, banned, mn, mx, need, free, pools)
    return got


def _sample(n, hist, rng, tries=60):
    """A random string consistent with every (word, marks) of hist, or None."""
    green, banned, mn, mx, need, free, pools = _compile(n, hist)
    if len(need) > len(free):
        return None
    rnd = rng.random
    for _ in range(tries):
        word = list(green)
        slots = list(free)
        ok = True
        for ch in need:
            opts = [i for i in slots if ch not in banned[i]]
            if not opts:
                ok = False
                break
            i = opts[int(rnd() * len(opts))]
            word[i] = ch
            slots.remove(i)
        if not ok:
            continue
        cnt = {}
        for ch in word:
            if ch is not None:
                cnt[ch] = cnt.get(ch, 0) + 1
        for i in slots:
            pool = pools[i]
            ch = None
            for _k in range(12):
                if not pool:
                    break
                c = pool[int(rnd() * len(pool))]
                if cnt.get(c, 0) < mx.get(c, n):
                    ch = c
                    break
            if ch is None:
                ok = False
                break
            word[i] = ch
            cnt[ch] = cnt.get(ch, 0) + 1
        if not ok:
            continue
        w = "".join(word)
        if _fits(w, hist):
            return w
    return None


def _search(n, hist, limit):
    """Deterministic depth-first list of at most `limit` consistent strings (letters by English frequency)."""
    green, banned, mn, mx, need, free, pools = _compile(n, hist)
    out = []
    word = [None] * n
    cnt = {}
    budget = [20000]

    def rec(i, missing):
        if len(out) >= limit or budget[0] <= 0:
            return
        if i == n:
            w = "".join(word)
            if _fits(w, hist):
                out.append(w)
            return
        if missing > n - i:
            return
        budget[0] -= 1
        letters = green[i] if green[i] is not None else _FREQ
        for ch in letters:
            c = cnt.get(ch, 0)
            if green[i] is None and (ch in banned[i] or c >= mx.get(ch, n)):
                continue
            word[i] = ch
            cnt[ch] = c + 1
            rec(i + 1, missing - (1 if c < mn.get(ch, 0) else 0))
            cnt[ch] = c
            if len(out) >= limit or budget[0] <= 0:
                return

    rec(0, sum(mn.values()))
    return out


def _candidates(state, seat):
    """Deterministic list (<= 300) of consistent guesses built only from what `seat` saw."""
    n = state["length"]
    hist = _known(state, seat)
    key = ("l", n, hist)
    got = _CACHE.get(key)
    if got is not None:
        return got
    out, seen = [], set()
    for w in _COMMON.get(n, "").split():
        if len(w) == n and w not in seen and _fits(w, hist):
            seen.add(w)
            out.append(w)
    rng = Random("wordle|%d|%s" % (n, "|".join(w + m for w, m in hist)))
    miss = 0
    while len(out) < 300 and miss < 400:
        w = _sample(n, hist, rng, 8)
        if w is None or w in seen:
            miss += 1 if w is not None else 40
            continue
        seen.add(w)
        out.append(w)
    if len(out) < 300:
        for w in _search(n, hist, 300):
            if w not in seen and len(out) < 300:
                seen.add(w)
                out.append(w)
    if not out:  # contradictory history: any legal strings
        rng = Random(n)
        while len(out) < 50:
            out.append("".join(_POOL[int(rng.random() * len(_POOL))] for _ in range(n)))
    if len(_CACHE) > 400:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def initial_state(config, seed):
    rng = Random(seed)
    n = int(config["length"])
    words = [w for w in _COMMON.get(n, "").split() if len(w) == n]
    secret = words[int(rng.random() * len(words))] if words else "".join(rng.choice(_ABC) for _ in range(n))
    return {"length": n, "max_guesses": int(config["max_guesses"]), "secret": secret, "round": 1,
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
    return ["guess " + w for w in _candidates(state, seat)]


def is_legal_action(state, seat, action):
    try:
        return (not state["over"]) and seat in (0, 1) and _word(state, action) is not None
    except Exception:  # noqa: BLE001
        return False


def random_action(state, seat, rng):
    return "guess " + "".join(_ABC[int(rng.random() * 26)] for _ in range(state["length"]))


def could_win_now(state, seat, action):
    try:
        if state["over"] or seat not in (0, 1):
            return False
        w = _word(state, action)
        if w is None:
            return False
        n = state["length"]
        for g in state["guesses"][seat]:
            gw, gm = g[0], g[1]
            if len(gw) != n or "?" in gw:
                continue
            if _marks(w, gw) != gm:
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _best(guesses):
    best = (0, 0)
    for g in guesses:
        t = (g[1].count("G"), g[1].count("Y"))
        if t > best:
            best = t
    return best


def apply(state, actions, rng):
    st = dict(state)
    st["guesses"] = [[list(g) for g in state["guesses"][0]], [list(g) for g in state["guesses"][1]]]
    if state["over"]:
        return st
    secret = state["secret"]
    found = [False, False]
    for seat in (0, 1):
        a = actions.get(seat)
        if a is None:
            a = actions.get(str(seat))
        w = _word(state, a) if a is not None else None
        if w is None:
            continue
        st["guesses"][seat].append([w, _marks(secret, w)])
        found[seat] = w == secret
    if found[0] or found[1]:
        st["over"] = True
        st["winner"] = None if (found[0] and found[1]) else (0 if found[0] else 1)
    elif state["round"] >= state["max_guesses"]:
        b0, b1 = _best(st["guesses"][0]), _best(st["guesses"][1])
        st["over"] = True
        st["winner"] = None if b0 == b1 else (0 if b0 > b1 else 1)
    else:
        st["round"] = state["round"] + 1
    return st


_NAME = {"G": "green", "Y": "yellow", "X": "gray"}


def observation(state, seat):
    n = state["length"]
    L = ["WORDLE DUEL (%d letters, max %d guesses). You are Player %d." % (n, state["max_guesses"], seat),
         "Round %d of %d. Both players guess simultaneously." % (state["round"], state["max_guesses"]), "",
         "Your guesses (G=green right spot, Y=yellow wrong spot, X=gray not in word):"]
    mine, theirs = state["guesses"][seat], state["guesses"][1 - seat]
    if not mine:
        L.append("  (none yet)")
    for i, g in enumerate(mine):
        w = g[0].upper()
        L.append("  %d. %s  %s   (%s)" % (i + 1, w, " ".join(g[1]),
                                          " ".join("%s=%s" % (c, _NAME.get(k, k)) for c, k in zip(w, g[1]))))
    L.append("")
    L.append("Opponent's guesses (letters hidden, colours only):")
    if not theirs:
        L.append("  (none yet)")
    for i, g in enumerate(theirs):
        L.append("  %d. %s  %s" % (i + 1, "?" * n, " ".join(g[1])))
    L.append("")
    if state["over"]:
        L.append("The game is over.")
    else:
        L.append('Your move: "guess <word>" with exactly %d letters a-z.' % n)
    return "\n".join(L)


_RE_G = re.compile(r"^\s*\d+\s*[.):]\s*([A-Za-z?]+)\s+((?:[GYXgyx]\b[ \t]*)+)")
_RE_ROUND = re.compile(r"Round\s+(\d+)", re.I)


def state_from_observation(text, seat, config, rng):
    n = int(config["length"])
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
        if low.startswith("your move"):
            cur = None
            continue
        m = _RE_G.match(ln)
        if m and cur is not None:
            marks = "".join(m.group(2).split()).upper()
            word = m.group(1).lower()
            if len(marks) != len(word):
                continue
            if cur is theirs or "?" in word:
                word = "?" * len(word)
            cur.append([word, marks])
    m = _RE_ROUND.search(text)
    rnd = int(m.group(1)) if m else max(len(mine), len(theirs)) + 1
    st = {"length": n, "max_guesses": int(config["max_guesses"]), "secret": None, "round": rnd,
          "guesses": [[], []], "over": False, "winner": None}
    st["guesses"][seat] = mine
    st["guesses"][1 - seat] = theirs
    hist = _known(st, seat)
    secret = _sample(n, hist, rng)
    if secret is None:
        found = _search(n, hist, 40)
        if found:
            secret = found[int(rng.random() * len(found))]
        else:
            secret = "".join(_POOL[int(rng.random() * len(_POOL))] for _ in range(n))
    st["secret"] = secret
    return st

"""Exact hand-written simulator of the arena game `nim` (checked against the real replays).

Rules in short: some piles of tokens with random starting sizes (lo..hi each); seat 0 moves first, then the seats
alternate; a move removes at least 1 token from ONE pile (variant "Take 1-3": at most 3 tokens; variant "Up to two
piles": from one pile or from two different piles, at least 1 from each, amounts free). The game ends when every
pile is empty: NORMAL play = the seat that took the last token WINS; MISERE = that seat LOSES. No draws.
Action spelling: "take <count> from pile <p>" or "take <c> from pile <p> and <c> from pile <q>" (piles from 1).
State (ints only): piles, turn, last (seat of the last move, -1 before any), misere, max_take (0 = no limit),
max_piles, lo, hi, init (starting piles) and hist (one (seat, ((pile index, count), ...)) per move).
"""
import re

CONFIGS = {
    "Normal play, 4 piles": {"piles": 4, "lo": 1, "hi": 8, "misere": 0, "max_take": 0, "max_piles": 1},
    "Misere, 3 piles": {"piles": 3, "lo": 2, "hi": 9, "misere": 1, "max_take": 0, "max_piles": 1},
    "Take 1-3, 4 piles": {"piles": 4, "lo": 3, "hi": 11, "misere": 0, "max_take": 3, "max_piles": 1},
    "Up to two piles": {"piles": 4, "lo": 1, "hi": 7, "misere": 0, "max_take": 0, "max_piles": 2},
}


def _label(state):
    for label, c in CONFIGS.items():
        if (c["misere"], c["max_take"], c["max_piles"]) == (state["misere"], state["max_take"], state["max_piles"]):
            return label
    return "Nim"


def _blank(config, piles):
    return {"piles": list(piles), "turn": 0, "last": -1, "misere": int(config.get("misere", 0)),
            "max_take": int(config.get("max_take", 0)), "max_piles": int(config.get("max_piles", 1)),
            "lo": int(config.get("lo", 1)), "hi": int(config.get("hi", 8)), "init": list(piles), "hist": []}


def initial_state(config, seed):
    import random
    rng = random.Random(seed)
    lo, hi = int(config.get("lo", 1)), int(config.get("hi", 8))
    return _blank(config, [rng.randint(lo, hi) for _ in range(int(config.get("piles", 4)))])


def is_over(state):
    return not any(state["piles"])


def to_move(state):
    return [state["turn"]] if any(state["piles"]) else []


def winner(state):
    if any(state["piles"]) or state["last"] < 0:
        return None
    return 1 - state["last"] if state["misere"] else state["last"]


def _fmt(take):
    return "take " + " and ".join("%d from pile %d" % (c, p + 1) for p, c in take)


def _parse(action):
    """((pile index, count), ...) or None when the text is not an action (legality is checked elsewhere)."""
    try:
        t = [w for w in str(action).strip().lower().replace(",", " ").split() if w != "pile"]
        if not t or t[0] != "take":
            return None
        t = t[1:]
        out = []
        while True:
            if len(t) < 3 or t[1] != "from" or not (t[0].isdigit() and t[2].isdigit()):
                return None
            out.append((int(t[2]) - 1, int(t[0])))
            t = t[3:]
            if not t:
                return tuple(out)
            if t[0] != "and":
                return None
            t = t[1:]
    except Exception:  # noqa: BLE001
        return None


def _ok(state, take):
    if not take or len(take) > state["max_piles"]:
        return False
    piles, mt = state["piles"], state["max_take"]
    seen = -1
    for p, c in take:
        if p < 0 or p >= len(piles) or p == seen or c < 1 or c > piles[p] or (mt and c > mt):
            return False
        seen = p
    return True


def legal_actions(state, seat):
    piles = state["piles"]
    if seat != state["turn"] or not any(piles):
        return []
    mt = state["max_take"]
    lim = [min(n, mt) if mt else n for n in piles]
    out = ["take %d from pile %d" % (c, p + 1) for p in range(len(piles)) for c in range(1, lim[p] + 1)]
    if state["max_piles"] >= 2:
        for p in range(len(piles)):
            for q in range(p + 1, len(piles)):
                for a in range(1, lim[p] + 1):
                    for b in range(1, lim[q] + 1):
                        out.append("take %d from pile %d and %d from pile %d" % (a, p + 1, b, q + 1))
    return out


def is_legal_action(state, seat, action):
    try:
        if seat != state["turn"] or not any(state["piles"]):
            return False
        take = _parse(action)
        return take is not None and _ok(state, take)
    except Exception:  # noqa: BLE001
        return False


def random_action(state, seat, rng):
    la = legal_actions(state, seat)
    return la[rng.randrange(len(la))] if la else ""


def could_win_now(state, seat, action):
    return False


def apply(state, actions, rng):
    s = dict(state)
    s["piles"], s["init"], s["hist"] = list(state["piles"]), list(state["init"]), list(state["hist"])
    if not any(s["piles"]):
        return s
    seat = s["turn"]
    act = actions.get(seat, actions.get(str(seat))) if isinstance(actions, dict) else actions
    take = _parse(act)
    if take is None or not _ok(s, take):
        p = next(i for i, n in enumerate(s["piles"]) if n)  # illegal text: smallest legal move instead
        take = ((p, 1),)
    for p, c in take:
        s["piles"][p] -= c
    s["hist"].append((seat, tuple(take)))
    s["last"], s["turn"] = seat, 1 - seat
    return s


def _rule_lines(state):
    mt = state["max_take"]
    if state["max_piles"] >= 2:
        rule = ("On your turn you remove tokens from ONE pile or from TWO different piles at the same time: at least "
                "1 token from each pile you choose, as many as you like (amounts may differ, up to the whole pile).")
    elif mt:
        rule = ("On your turn you remove 1, 2, ... or %d tokens (never more than %d) from exactly ONE pile (and never "
                "more tokens than that pile has)." % (mt, mt))
    else:
        rule = "On your turn you remove at least 1 token (any number, up to the whole pile) from exactly ONE pile."
    if state["misere"]:
        goal = "MISERE: the player who takes the LAST token LOSES (you want your opponent to take it)."
    else:
        goal = "NORMAL PLAY: the player who takes the LAST token WINS (equivalently: if you cannot move, you lose)."
    return rule, goal


def _piles_text(piles):
    return "%s   (total %d)" % (", ".join("pile %d = %d" % (i + 1, n) for i, n in enumerate(piles)), sum(piles))


def observation(state, seat):
    rule, goal = _rule_lines(state)
    lines = ["NIM (%s). You are Player %d." % (_label(state), seat), "Rule: " + rule, goal, "",
             "Starting piles: " + _piles_text(state["init"])]
    if not state["hist"]:
        lines.append("No moves yet.")
    else:
        lines.append("Move history:")
        cur = list(state["init"])
        for i, (who, take) in enumerate(state["hist"]):
            for p, c in take:
                cur[p] -= c
            lines.append("  %d. %s: %s  -> %s" % (i + 1, "You" if who == seat else "Opponent", _fmt(take),
                                                  " ".join(str(n) for n in cur)))
    lines.append("")
    lines.append("Piles now: " + _piles_text(state["piles"]))
    if not any(state["piles"]):
        w = winner(state)
        lines.append("Game over: %s." % ("YOU win" if w == seat else "your OPPONENT wins"))
    elif state["turn"] != seat:
        lines.append("It is your OPPONENT's turn.")
    else:
        fmt = '"take <count> from pile <p>"'
        if state["max_piles"] >= 2:
            fmt += ' or "take <count> from pile <p> and <count> from pile <q>"'
        lines.append("It is YOUR turn. Answer with %s." % fmt)
    return "\n".join(lines)


_RE_PILE = re.compile(r"pile\s+(\d+)\s*=\s*(\d+)", re.I)
_RE_NOW = re.compile(r"^\s*Piles now\s*:(.*)$", re.I | re.M)
_RE_START = re.compile(r"^\s*Starting piles\s*:(.*)$", re.I | re.M)
_RE_HIST = re.compile(r"^\s*(\d+)\.\s*(You|Opponent)\s*:\s*(.*?)\s*->\s*([\d\s]+?)\s*$", re.I | re.M)


def _piles_of(m):
    if not m:
        return None
    d = {int(a): int(b) for a, b in _RE_PILE.findall(m.group(1))}
    return [d[k] for k in sorted(d)] if d else None


def state_from_observation(text, seat, config, rng):
    text = text or ""
    rows = sorted((int(m.group(1)), m.group(2).lower() == "you", m.group(3), [int(x) for x in m.group(4).split()])
                  for m in _RE_HIST.finditer(text))
    now, start = _piles_of(_RE_NOW.search(text)), _piles_of(_RE_START.search(text))
    if now is None:
        now = list(rows[-1][3]) if rows else (list(start) if start else None)
    if now is None:
        lo = int(config.get("lo", 1))
        now = [lo] * int(config.get("piles", 4))
    if start is None or len(start) != len(now):
        start = list(now)
    s = _blank(config, start)
    s["piles"] = list(now)
    cur, hist = list(start), []
    for _, mine, _act, after in rows:
        if len(after) != len(cur):
            hist = None
            break
        take = tuple((p, cur[p] - after[p]) for p in range(len(cur)) if cur[p] != after[p])
        if not take or any(c < 0 for _, c in take):
            hist = None
            break
        hist.append((seat if mine else 1 - seat, take))
        cur = list(after)
    if hist is None or cur != now:
        s["init"], hist = list(now), []  # history cut or inconsistent: keep only what decides the game
    s["hist"] = hist
    low = text.lower()
    if not any(now):
        s["last"] = hist[-1][0] if hist else 1 - seat
        s["turn"] = 1 - s["last"]
    else:
        s["turn"] = 1 - seat if ("opponent's turn" in low and "your turn" not in low) else seat
        s["last"] = hist[-1][0] if hist else (1 - s["turn"] if now != start else -1)
    return s


_WIN = {}  # (sorted piles, misere, max_take, max_piles) -> True when the player to move wins with perfect play


def _mover_wins(piles, misere, max_take, max_piles):
    piles = tuple(sorted(p for p in piles if p))
    key = (piles, misere, max_take, max_piles)
    if key in _WIN:
        return _WIN[key]
    if not piles:  # no token left: the previous player took the last one
        res = bool(misere)
    else:
        res = False
        n = len(piles)
        for i in range(n):
            top_i = piles[i] if not max_take else min(piles[i], max_take)
            for a in range(1, top_i + 1):
                rest = piles[:i] + (piles[i] - a,) + piles[i + 1:]
                if not _mover_wins(rest, misere, max_take, max_piles):
                    res = True
                    break
                if max_piles >= 2:
                    for j in range(i + 1, n):
                        top_j = piles[j] if not max_take else min(piles[j], max_take)
                        for b in range(1, top_j + 1):
                            rest2 = rest[:j] + (rest[j] - b,) + rest[j + 1:]
                            if not _mover_wins(rest2, misere, max_take, max_piles):
                                res = True
                                break
                        if res:
                            break
                if res:
                    break
            if res:
                break
    _WIN[key] = res
    return res


def recommend(text, seat, config, rng, seconds=1.0):
    """Optional hook of a known simulator (mesh/simworker.candidates): [(action, value)] best first. Nim is small
    enough to solve exactly (4 piles of at most 11 tokens): every move that leaves the opponent in a lost position
    is a sure win; in a lost position, the move that keeps the most tokens on the table (more chances for them to err)."""
    st = state_from_observation(text, seat, config, rng)
    if is_over(st) or seat not in to_move(st):
        return []
    mis, mt, mp = int(config.get("misere", 0)), int(config.get("max_take", 0)), int(config.get("max_piles", 1))
    wins, rest = [], []
    for a in legal_actions(st, seat):
        nxt = apply(st, {seat: a}, rng)
        left = sum(nxt["piles"])
        if not _mover_wins(nxt["piles"], mis, mt, mp):
            wins.append((left, a))
        else:
            rest.append((left, a))
    if wins:
        return [(a, 1.0) for _, a in sorted(wins, key=lambda t: -t[0])[:6]]
    return [(a, -1.0) for _, a in sorted(rest, key=lambda t: -t[0])[:6]]

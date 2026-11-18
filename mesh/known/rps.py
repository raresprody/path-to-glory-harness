"""Exact hand-written simulator of the arena game `rps` (checked against the real replays).

Rules in short: both seats throw rock/paper/scissors at the same time; rock beats scissors, scissors beat paper,
paper beats rock; same sign = tie, nobody scores. A won throw scores 1 point (variant "Rock wins count double":
a throw won WITH rock scores 2).
  "First to N": the first seat to reach N points wins at once; if `max_throws` (25) throws pass without that,
  more points wins, equal = draw.
  "15 throws": exactly 15 throws, more points wins, equal = draw (no target).
  "no sign three times in a row": a seat whose last two throws were the same sign may not throw it now.
State (ints only): pts, hist (one int per throw: 3*sign0 + sign1, rock 0 / paper 1 / scissors 2), and the
variant numbers target (0 = none), max_throws, double (1 = rock wins score 2), no_triple.
"""
import re

CONFIGS = {
    "First to 5": {"target": 5, "max_throws": 25, "double": 0, "no_triple": 0},
    "15 throws, Rock wins count double": {"target": 0, "max_throws": 15, "double": 1, "no_triple": 0},
    "First to 4, no sign three times in a row": {"target": 4, "max_throws": 25, "double": 0, "no_triple": 1},
}

SIGNS = ["rock", "paper", "scissors"]
_IDX = {"rock": 0, "paper": 1, "scissors": 2, "r": 0, "p": 1, "s": 2, "scissor": 2}
_VERB = ["rock crushes scissors", "paper covers rock", "scissors cut paper"]  # by winning sign


def _label(state):
    for label, c in CONFIGS.items():
        if all(int(c[k]) == state[k] for k in ("target", "max_throws", "double", "no_triple")):
            return label
    if state["target"]:
        return "First to %d" % state["target"]
    return "%d throws" % state["max_throws"]


def _sign(action):
    """0/1/2 or None when the text is not a sign."""
    try:
        return _IDX.get(str(action).strip().lower().strip("\"'.! "))
    except Exception:  # noqa: BLE001
        return None


def initial_state(config, seed):
    return {"target": int(config.get("target") or 0), "max_throws": int(config.get("max_throws") or 25),
            "double": int(bool(config.get("double"))), "no_triple": int(bool(config.get("no_triple"))),
            "pts": [0, 0], "hist": []}


def is_over(state):
    t, p = state["target"], state["pts"]
    return bool(t and (p[0] >= t or p[1] >= t)) or len(state["hist"]) >= state["max_throws"]


def to_move(state):
    return [] if is_over(state) else [0, 1]


def winner(state):
    if not is_over(state):
        return None
    p = state["pts"]
    if p[0] == p[1]:
        return None
    return 0 if p[0] > p[1] else 1


def banned(state, seat):
    """The sign `seat` may not throw now (0/1/2), or -1."""
    h = state["hist"]
    if not state["no_triple"] or len(h) < 2:
        return -1
    a = h[-1] // 3 if seat == 0 else h[-1] % 3
    b = h[-2] // 3 if seat == 0 else h[-2] % 3
    return a if a == b else -1


def legal_actions(state, seat):
    if seat not in (0, 1) or is_over(state):
        return []
    b = banned(state, seat)
    return [SIGNS[i] for i in range(3) if i != b]


def is_legal_action(state, seat, action):
    try:
        if seat not in (0, 1) or is_over(state):
            return False
        m = _sign(action)
        return m is not None and m != banned(state, seat)
    except Exception:  # noqa: BLE001
        return False


def random_action(state, seat, rng):
    la = legal_actions(state, seat)
    return la[rng.randrange(len(la))] if la else ""


def could_win_now(state, seat, action):
    return False


def score(a, b, double):
    """(points for seat 0, points for seat 1) of one throw with signs a (seat 0) and b (seat 1)."""
    if a == b:
        return 0, 0
    if (a - b) % 3 == 1:
        return (2 if double and a == 0 else 1), 0
    return 0, (2 if double and b == 0 else 1)


def apply(state, actions, rng):
    s = dict(state)
    s["pts"], s["hist"] = list(state["pts"]), list(state["hist"])
    if is_over(state):
        return s
    mv = []
    for seat in (0, 1):
        m = _sign(actions.get(seat, actions.get(str(seat))))
        b = banned(state, seat)
        if m is None or m == b:
            m = 1 if b == 0 else 0  # illegal text: first legal sign instead
        mv.append(m)
    p = score(mv[0], mv[1], s["double"])
    s["pts"][0] += p[0]
    s["pts"][1] += p[1]
    s["hist"].append(3 * mv[0] + mv[1])
    return s


def observation(state, seat):
    opp = 1 - seat
    n, mx, t = len(state["hist"]), state["max_throws"], state["target"]
    lines = ["ROCK PAPER SCISSORS - %s. You are seat %d." % (_label(state), seat)]
    goal = "first to %d points" % t if t else "more points after %d throws" % mx
    over = is_over(state)
    if over:
        lines.append("Game over after throw %d (max %d; goal: %s)." % (n, mx, goal))
    else:
        lines.append("Throw %d (max %d; goal: %s)." % (n + 1, mx, goal))
    lines.append("Score: you %d, opponent %d." % (state["pts"][seat], state["pts"][opp]))
    if state["double"]:
        lines.append("Wins with rock score 2 points, other wins 1.")
    if not n:
        lines.append("Throw history: none yet.")
    else:
        lines.append("Throw history (oldest first):")
        cnt = [0, 0, 0]
        for i, h in enumerate(state["hist"]):
            mv = (h // 3, h % 3)
            p = score(mv[0], mv[1], state["double"])
            cnt[mv[opp]] += 1
            if p[seat]:
                res = "you +%d (%s)" % (p[seat], _VERB[mv[seat]])
            elif p[opp]:
                res = "opponent +%d (%s)" % (p[opp], _VERB[mv[opp]])
            else:
                res = "tie"
            lines.append("  %d. you %s - opponent %s -> %s" % (i + 1, SIGNS[mv[seat]], SIGNS[mv[opp]], res))
        lines.append("Opponent's signs so far: rock %d, paper %d, scissors %d." % tuple(cnt))
    if over:
        w = winner(state)
        lines.append("Result: %s." % ("a DRAW" if w is None else ("YOU win" if w == seat else "your OPPONENT wins")))
        return "\n".join(lines)
    mine, theirs = banned(state, seat), banned(state, opp)
    if mine >= 0:
        lines.append("You may NOT throw %s this turn (two in a row already)." % SIGNS[mine])
    if theirs >= 0:
        lines.append("Your opponent may NOT throw %s this turn (two in a row already)." % SIGNS[theirs])
    lines.append("Throw now: " + " | ".join(SIGNS[i] for i in range(3) if i != mine))
    return "\n".join(lines)


_RE_THROW = re.compile(r"^\s*Throw\s+(\d+)\b", re.I | re.M)
_RE_SCORE = re.compile(r"Score\s*:\s*you\s+(-?\d+)\s*,\s*opponent\s+(-?\d+)", re.I)
_RE_HIST = re.compile(r"^\s*(\d+)\.\s*you\s+(rock|paper|scissors)\s*-\s*opponent\s+(rock|paper|scissors)", re.I | re.M)
_RE_BAN_ME = re.compile(r"^\s*You may NOT throw\s+(rock|paper|scissors)", re.I | re.M)
_RE_BAN_OPP = re.compile(r"opponent may NOT throw\s+(rock|paper|scissors)", re.I)
_RE_NOW = re.compile(r"^\s*Throw now\s*:(.*)$", re.I | re.M)


def _pad(n, ban):
    """n signs without two equal neighbours, ending with `ban` twice when a sign is banned."""
    out = [i % 2 for i in range(n)]
    if ban >= 0 and n >= 2:
        out[-1] = out[-2] = ban
        if n >= 3:
            out[-3] = (ban + 1) % 3
        if n >= 4:
            out[-4] = (ban + 2) % 3
    return out


def state_from_observation(text, seat, config, rng):
    s = initial_state(config, 0)
    text = text or ""
    opp = 1 - seat
    rows = {}
    for m in _RE_HIST.finditer(text):
        rows[int(m.group(1))] = (_IDX[m.group(2).lower()], _IDX[m.group(3).lower()])
    hist, pts = [], [0, 0]
    for i in sorted(rows):
        mv = [0, 0]
        mv[seat], mv[opp] = rows[i]
        hist.append(3 * mv[0] + mv[1])
        p = score(mv[0], mv[1], s["double"])
        pts[0] += p[0]
        pts[1] += p[1]
    m = _RE_THROW.search(text)
    done = int(m.group(1)) - 1 if m else len(hist)
    sc = _RE_SCORE.search(text)
    want = list(pts)
    if sc:
        want[seat], want[opp] = int(sc.group(1)), int(sc.group(2))
    if len(hist) != done or sorted(rows) != list(range(1, done + 1)):
        # history lines missing or cut: keep the scores the text shows and rebuild a history of the right length
        # that bans exactly the signs the text says are banned
        ban = [-1, -1]
        if s["no_triple"]:
            bm, bo = _RE_BAN_ME.search(text), _RE_BAN_OPP.search(text)
            if bm:
                ban[seat] = _IDX[bm.group(1).lower()]
            else:
                now = _RE_NOW.search(text)
                if now:
                    shown = [w.strip().lower() for w in now.group(1).split("|")]
                    missing = [i for i in range(3) if SIGNS[i] not in shown]
                    if len(missing) == 1:
                        ban[seat] = missing[0]
            if bo:
                ban[opp] = _IDX[bo.group(1).lower()]
        a, b = _pad(max(done, 0), ban[0]), _pad(max(done, 0), ban[1])
        hist = [3 * x + y for x, y in zip(a, b)]
    s["hist"], s["pts"] = hist, want
    return s

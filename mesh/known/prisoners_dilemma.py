"""Exact hand-written simulator of the arena game `prisoners_dilemma` (checked against the real replays).

Rules in short (identical in every variant, only the numbers change):
  each round both seats choose cooperate/defect at the same time: C/C +3 each, D/C 5 and 0, D/D +1 each;
  every "defect" played adds one DEFECTION to that seat's record;
  the game lasts at least `min_rounds`; after round `min_rounds` and after every later round it ends with a
  1 in `end_one_in` chance, and always after round `max_rounds`. Reaching the quota does NOT end the game.
  At the end: only one seat has >= quota points -> it wins; both have -> more points wins (equal = draw);
  nobody has -> FEWER DEFECTIONS wins (equal = draw; points do not matter at all).
State (ints only): pts, defs, hist (one int per round: 2*move0 + move1, with 1 = defect), over, and the
variant numbers quota/min_rounds/end_one_in/max_rounds.
"""
import re

CONFIGS = {
    "Qualify at 25, at least 8 rounds": {"quota": 25, "min_rounds": 8, "end_one_in": 4, "max_rounds": 20},
    "Long game: qualify at 40, at least 12 rounds": {"quota": 40, "min_rounds": 12, "end_one_in": 5,
                                                     "max_rounds": 26},
    "Short fuse: qualify at 16, at least 5 rounds": {"quota": 16, "min_rounds": 5, "end_one_in": 3,
                                                     "max_rounds": 11},
}

ACTIONS = ["cooperate", "defect"]
_PAY = {(0, 0): (3, 3), (1, 0): (5, 0), (0, 1): (0, 5), (1, 1): (1, 1)}


def _label(state):
    for label, c in CONFIGS.items():
        if c["quota"] == state["quota"] and c["min_rounds"] == state["min_rounds"]:
            return label
    return "Qualify at %d, at least %d rounds" % (state["quota"], state["min_rounds"])


def _move(action):
    """0 = cooperate, 1 = defect, None = not an action."""
    a = " ".join(str(action).strip().lower().split())
    if a in ("cooperate", "c"):
        return 0
    if a in ("defect", "d"):
        return 1
    return None


def initial_state(config, seed):
    return {"quota": int(config["quota"]), "min_rounds": int(config["min_rounds"]),
            "end_one_in": int(config["end_one_in"]), "max_rounds": int(config["max_rounds"]),
            "pts": [0, 0], "defs": [0, 0], "hist": [], "over": 0}


def to_move(state):
    return [] if state["over"] else [0, 1]


def legal_actions(state, seat):
    if state["over"] or seat not in (0, 1):
        return []
    return list(ACTIONS)


def is_legal_action(state, seat, action):
    try:
        return (not state["over"]) and seat in (0, 1) and _move(action) is not None
    except Exception:  # noqa: BLE001
        return False


def random_action(state, seat, rng):
    return ACTIONS[rng.randrange(2)]


def could_win_now(state, seat, action):
    return False


def standing(pts, defs, quota):
    """Who wins if the game ends with these totals: 0, 1 or None (draw). The arena's three rules, in order."""
    q0, q1 = pts[0] >= quota, pts[1] >= quota
    if q0 != q1:
        return 0 if q0 else 1
    if q0 and q1:
        return None if pts[0] == pts[1] else (0 if pts[0] > pts[1] else 1)
    if defs[0] == defs[1]:
        return None
    return 0 if defs[0] < defs[1] else 1


def _play(state, actions):
    """The round itself, without the end-of-game draw."""
    m0, m1 = _move(actions[0]), _move(actions[1])
    if m0 is None:
        m0 = 0
    if m1 is None:
        m1 = 0
    p = _PAY[(m0, m1)]
    s = dict(state)
    s["pts"] = [state["pts"][0] + p[0], state["pts"][1] + p[1]]
    s["defs"] = [state["defs"][0] + m0, state["defs"][1] + m1]
    s["hist"] = list(state["hist"]) + [2 * m0 + m1]
    return s


def apply(state, actions, rng):
    if state["over"]:
        return _copy(state)
    s = _play(state, actions)
    n = len(s["hist"])
    if n >= s["max_rounds"]:
        s["over"] = 1
    elif n >= s["min_rounds"] and rng.randrange(s["end_one_in"]) == 0:
        s["over"] = 1
    return s


def _copy(state):
    s = dict(state)
    s["pts"], s["defs"], s["hist"] = list(state["pts"]), list(state["defs"]), list(state["hist"])
    return s


def _finish(state):
    """The same state with the game declared over now (the arena's random ending, forced; used in tests and by
    searches that want the value of 'the game stops here')."""
    s = _copy(state)
    s["over"] = 1
    return s


def end_probability(state):
    """Chance that the game ends right after the round that is about to be played."""
    if state["over"]:
        return 1.0
    n = len(state["hist"]) + 1
    if n >= state["max_rounds"]:
        return 1.0
    return 1.0 / state["end_one_in"] if n >= state["min_rounds"] else 0.0


def is_over(state):
    return bool(state["over"])


def winner(state):
    if not state["over"]:
        return None
    return standing(state["pts"], state["defs"], state["quota"])


def _times(n):
    return "%d time%s" % (n, "" if n == 1 else "s")


def _who_has(pts, seat):
    parts = []
    for s in (0, 1):
        parts.append(("you have %d" if s == seat else "your opponent has %d") % pts[s])
    return ", ".join(parts)


def _standing_text(state, seat):
    pts, defs, q = state["pts"], state["defs"], state["quota"]
    w = standing(pts, defs, q)
    q0, q1 = pts[0] >= q, pts[1] >= q
    if w is None:
        if q0 and q1:
            return "a DRAW (both qualified with %d points each)." % pts[0]
        return "a DRAW (nobody reached %d (%s) and both defected %s)." % (q, _who_has(pts, seat), _times(defs[0]))
    mine = w == seat
    head = "YOU would win" if mine else "your OPPONENT would win"
    if q0 != q1:
        why = "only %s reached the qualifying score %d (%d vs %d)" % (
            "you" if mine else "your opponent", q, pts[w], pts[1 - w])
    elif q0 and q1:
        why = "both qualified; %s more points (%d vs %d)" % (
            "you have" if mine else "your opponent has", pts[w], pts[1 - w])
    else:
        why = "nobody reached %d (%s); %s defected fewer times (%d vs %d)" % (
            q, _who_has(pts, seat), "you" if mine else "your opponent", defs[w], defs[1 - w])
    return "%s (%s)." % (head, why)


def _side(name, pts, defs, q):
    need = "QUALIFIED" if pts >= q else "needs %d more to qualify" % (q - pts)
    return "%s: %d points (%s), %d defection%s." % (name, pts, need, defs, "" if defs == 1 else "s")


def observation(state, seat):
    q, mn, k, mx = state["quota"], state["min_rounds"], state["end_one_in"], state["max_rounds"]
    opp = 1 - seat
    n = len(state["hist"]) + 1
    lines = ["PRISONER'S DILEMMA - %s. You are seat %d. Qualifying score: %d." % (_label(state), seat, q)]
    if state["over"]:
        lines.append("Game over after round %d." % (n - 1))
    elif n <= mn:
        lines.append("Round %d (the game lasts at least until round %d; never beyond round %d)." % (n, mn, mx))
    else:
        lines.append("Round %d (the game went on; after this round it ends with a 1 in %d chance; never beyond "
                     "round %d)." % (n, k, mx))
    lines.append(_side("You", state["pts"][seat], state["defs"][seat], q))
    lines.append(_side("Opponent", state["pts"][opp], state["defs"][opp], q))
    if not state["hist"]:
        lines.append("Rounds so far: none.")
    else:
        lines.append("Rounds so far (oldest first):")
        tot = [0, 0]
        for i, h in enumerate(state["hist"]):
            mv = (h >> 1, h & 1)
            p = _PAY[mv]
            tot[0] += p[0]
            tot[1] += p[1]
            me, they = mv[seat], mv[opp]
            if me and they:
                kind = "both defected"
            elif me:
                kind = "you betrayed them"
            elif they:
                kind = "they betrayed you"
            else:
                kind = "both cooperated"
            lines.append("  %d. you %s, opponent %s -> you +%d, opponent +%d (%s); totals you %d, opponent %d" % (
                i + 1, ACTIONS[me], ACTIONS[they], p[seat], p[opp], kind, tot[seat], tot[opp]))
    lines.append("If the game ended now: " + _standing_text(state, seat))
    lines.append("The game lasts at least %d rounds. After round %d and after every later round it ends with a "
                 "1 in %d chance (it never goes beyond round %d). Nobody knows the real length in advance."
                 % (mn, mn, k, mx))
    lines.append("Talk in the match chat if you like (promises and threats are not binding). "
                 "Answer: cooperate | defect")
    return "\n".join(lines)


_RE_ROUND = re.compile(r"^\s*Round\s+(\d+)\b", re.I | re.M)
_RE_YOU = re.compile(r"^\s*You:\s*(-?\d+)\s+points?.*?(\d+)\s+defections?", re.I | re.M)
_RE_OPP = re.compile(r"^\s*Opponent:\s*(-?\d+)\s+points?.*?(\d+)\s+defections?", re.I | re.M)
_RE_HIST = re.compile(r"^\s*(\d+)\.\s*you\s+(cooperate|defect)\w*\s*,\s*opponent\s+(cooperate|defect)", re.I | re.M)


def state_from_observation(text, seat, config, rng):
    s = initial_state(config, 0)
    text = text or ""
    opp = 1 - seat
    rows = {}
    for m in _RE_HIST.finditer(text):
        rows[int(m.group(1))] = (_move(m.group(2)), _move(m.group(3)))
    hist, pts, defs = [], [0, 0], [0, 0]
    for i in sorted(rows):
        mv = [0, 0]
        mv[seat], mv[opp] = rows[i]
        hist.append(2 * mv[0] + mv[1])
        p = _PAY[(mv[0], mv[1])]
        pts[0] += p[0]
        pts[1] += p[1]
        defs[0] += mv[0]
        defs[1] += mv[1]
    m = _RE_ROUND.search(text)
    done = int(m.group(1)) - 1 if m else len(hist)
    my, their = _RE_YOU.search(text), _RE_OPP.search(text)
    want_pts, want_defs = list(pts), list(defs)
    if my:
        want_pts[seat], want_defs[seat] = int(my.group(1)), int(my.group(2))
    if their:
        want_pts[opp], want_defs[opp] = int(their.group(1)), int(their.group(2))
    if len(hist) != done or want_pts != pts or want_defs != defs:
        # history lines missing or cut: keep the totals the text shows (they decide the game) and pad the history
        # with a move list that has the right length and, when possible, the right defection counts
        hist = []
        d = list(want_defs)
        for _ in range(max(done, 0)):
            m0 = 1 if d[0] > 0 else 0
            m1 = 1 if d[1] > 0 else 0
            d[0] -= m0
            d[1] -= m1
            hist.append(2 * m0 + m1)
    s["hist"], s["pts"], s["defs"] = hist, want_pts, want_defs
    return s


# ---------------------------------------------------------------------------------------------------------------
# Move policy: reactive opponent model + expectimax on the EXPECTED MATCH RESULT (win 1, draw 0.5, loss 0).
# ---------------------------------------------------------------------------------------------------------------
_PRIOR_R1 = 0.6                    # neutral prior: cooperation in round 1
_PRIOR_AFTER = {0: 0.65, 1: 0.3}   # ... after WE cooperated / after WE defected (tit-for-tat-like)
_W_GAME = 1.5                      # weight of one observation from this very game
_W_SEQ = 0.5                       # weight of one observation from the opponent's public replays
_W_SEQ_CAP = 6.0                   # ... and the most all of them together may weigh per context
_K_PRIOR = 2.0                     # pseudo-observations behind every prior (Laplace-style smoothing)
_HORIZONS = (4, 8, 12)             # iterative deepening; beyond 12 rounds the opponent model is not worth trusting


class _Timeout(Exception):
    pass


def _opp_model(state, seat, config):
    """(p_round1, {(my_last, their_last): p}) - chance that the opponent cooperates, given last round's moves."""
    game = {}   # context -> [cooperations, observations], this game
    seq = {}    # the same from the opponent's earlier public games

    def add(table, ctx, their_move):
        c = table.setdefault(ctx, [0.0, 0.0])
        c[0] += 1 - their_move
        c[1] += 1

    def feed(table, mine, theirs):
        for i, t in enumerate(theirs):
            if t is None:
                continue
            add(table, "all", t)
            if i == 0:
                add(table, "r1", t)
            elif i - 1 < len(mine) and mine[i - 1] is not None and theirs[i - 1] is not None:
                add(table, ("m", mine[i - 1]), t)
                add(table, (mine[i - 1], theirs[i - 1]), t)

    hist = state["hist"]
    feed(game, [(h >> 1, h & 1)[seat] for h in hist], [(h >> 1, h & 1)[1 - seat] for h in hist])
    try:
        for g in ((config or {}).get("_opp") or {}).get("seq") or []:
            theirs = [_move(a) for a in (g.get("mine") or [])]
            ours = [_move(a) for a in (g.get("opp") or [])]
            feed(seq, ours, theirs)
    except Exception:  # noqa: BLE001
        seq = {}

    def est(ctx, prior, k=_K_PRIOR):
        gc, gn = game.get(ctx, (0.0, 0.0))
        sc, sn = seq.get(ctx, (0.0, 0.0))
        ws = _W_SEQ if sn * _W_SEQ <= _W_SEQ_CAP else _W_SEQ_CAP / sn
        return (_W_GAME * gc + ws * sc + k * prior) / (_W_GAME * gn + ws * sn + k)

    p1 = est("r1", _PRIOR_R1)
    # how much they cooperate at all, whatever we do (an unconditional player shows here first)
    gc, gn = game.get("all", (0.0, 0.0))
    table = {}
    for ml in (0, 1):
        base = (_K_PRIOR * _PRIOR_AFTER[ml] + 0.5 * gc) / (_K_PRIOR + 0.5 * gn)
        pm = est(("m", ml), base)
        for tl in (0, 1):
            table[(ml, tl)] = min(0.98, max(0.02, est((ml, tl), pm)))
    return min(0.98, max(0.02, p1)), table


def _expected_left(n, mn, k, mx):
    """Expected number of further rounds when `n` rounds are done and the game is still on."""
    e, alive, r = 0.0, 1.0, n
    while r < mx and alive > 1e-3:
        r += 1
        e += alive
        if r >= mn:
            alive *= 1.0 - 1.0 / k
    return e


def recommend(text, seat, config, rng, seconds=1.0):
    """returns a list of (action, value, p) tuples, best first"""
    try:
        import time
        t0 = time.time()
        if seat not in (0, 1) or re.search(r"^\s*Game over\b", text or "", re.I | re.M):
            return []
        cfg = {k: v for k, v in (config or {}).items() if not str(k).startswith("_")}
        state = state_from_observation(text, seat, cfg, rng)
        if state["over"] or seat not in to_move(state) or len(state["hist"]) >= state["max_rounds"]:
            return []
        quota, mn, k, mx = state["quota"], state["min_rounds"], state["end_one_in"], state["max_rounds"]
        p1, table = _opp_model(state, seat, config)
        opp = 1 - seat
        n0 = len(state["hist"])
        if n0:
            h = state["hist"][-1]
            ml0, tl0 = (h >> 1, h & 1)[seat], (h >> 1, h & 1)[opp]
        else:
            ml0 = tl0 = -1
        budget = max(0.02, float(seconds) * 0.8)
        deadline = t0 + budget
        pay = {(0, 0): (3, 3), (1, 0): (5, 0), (0, 1): (0, 5), (1, 1): (1, 1)}   # (mine, theirs) -> points
        left_cache = {}

        def result(mp, op, md, od):
            # standing() from our side: 1 win, 0.5 draw, 0 loss
            w = standing((mp, op), (md, od), quota)
            return 0.5 if w is None else (1.0 if w == 0 else 0.0)

        def p_coop(mp, op, md, od, ml, tl):
            if ml < 0:
                return p1
            p = table[(ml, tl)]
            # an opponent who would win as things stand has no reason to change what he is doing
            if (op >= quota and (mp < quota or op > mp)) or (op < quota and mp < quota and od < md):
                p = 0.5 * p + 0.5 * (1.0 - tl)
            return p

        def leaf(n, mp, op, md, od, ml, tl):
            now = result(mp, op, md, od)
            r = left_cache.get(n)
            if r is None:
                r = left_cache[n] = int(round(_expected_left(n, mn, k, mx)))
            if r <= 0:
                return now
            best = 0.0
            for a in (0, 1):      # we keep playing `a`, they give their likelier answer to it
                b = 0 if p_coop(mp, op, md, od, a, tl) >= 0.5 else 1
                g = pay[(a, b)]
                best = max(best, result(mp + r * g[0], op + r * g[1], md + r * a, od + r * b))
            return 0.5 * now + 0.5 * best

        def search(horizon):
            memo = {}
            count = [0]

            def q(n, mp, op, md, od, ml, tl, a):
                """Expected result when `n` rounds are done and we now play `a`."""
                pc = p_coop(mp, op, md, od, ml, tl)
                n1 = n + 1
                pe = 1.0 if n1 >= mx else (1.0 / k if n1 >= mn else 0.0)   # = end_probability()
                ev = 0.0
                for b, pb in ((0, pc), (1, 1.0 - pc)):
                    g = pay[(a, b)]
                    mp1, op1, md1, od1 = mp + g[0], op + g[1], md + a, od + b
                    x = 0.0
                    if pe > 0.0:
                        x += pe * result(mp1, op1, md1, od1)
                    if pe < 1.0:
                        x += (1.0 - pe) * v(n1, mp1, op1, md1, od1, a, b)
                    ev += pb * x
                return ev

            def v(n, mp, op, md, od, ml, tl):
                key = (n, mp, op, md, od, ml, tl)
                got = memo.get(key)
                if got is not None:
                    return got
                count[0] += 1
                if not count[0] & 1023 and time.time() > deadline:
                    raise _Timeout()
                if n - n0 >= horizon:
                    got = leaf(n, mp, op, md, od, ml, tl)
                else:
                    got = max(q(n, mp, op, md, od, ml, tl, 0), q(n, mp, op, md, od, ml, tl, 1))
                memo[key] = got
                return got

            mp, op, md, od = state["pts"][seat], state["pts"][opp], state["defs"][seat], state["defs"][opp]
            return [q(n0, mp, op, md, od, ml0, tl0, a) for a in (0, 1)]

        values = None
        left = mx - n0
        for hz in _HORIZONS:
            t1 = time.time()
            try:
                values = search(hz)
            except _Timeout:
                break
            spent = time.time() - t1
            if hz >= left or time.time() + 4.0 * spent + 0.01 > t0 + 0.5 * budget:
                break
        if values is None:      # not even the shallowest search fitted: one round ahead
            deadline = float("inf")
            values = search(1)
        out = [(ACTIONS[a], 2.0 * values[a] - 1.0) for a in (0, 1)]
        out.sort(key=lambda t: -t[1])
        if abs(out[0][1] - out[1][1]) < 0.03:
            return [(out[0][0], out[0][1], 0.5), (out[1][0], out[1][1], 0.5)]
        return [(out[0][0], out[0][1], 1.0), (out[1][0], out[1][1], 0.0)]
    except Exception:  # noqa: BLE001
        return []

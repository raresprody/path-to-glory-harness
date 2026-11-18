"""Exact hand-written simulator of the arena game `colonel_blotto` (checked against the real replays).

Rules in short (5 rounds in every variant, both seats move at once):
  each round both seats split soldiers over the k battlefields; on a field MORE soldiers captures it, equal
  (0 vs 0 too) = nobody. Round points = sum of the values of the captured fields; more points wins the round,
  equal = round drawn. After 5 rounds: more rounds won wins; tied -> more battlefield points over the whole
  game; tied too -> draw.
  "100 soldiers, 5 equal fields": 5 numbers adding up to EXACTLY 100, every field worth 1.
  "60 soldiers, 6 fields worth 1-6": 6 numbers adding up to EXACTLY 60; the values 1..6 are reshuffled every
      round and shown before the move.
  "Reserves: +30 per round, 3 fields": every field worth 1; each seat has a public reserve that gets +30 at the
      start of every round (30 in round 1); a move is 3 numbers whose sum is AT MOST the reserve (0 allowed);
      what is not sent stays in the reserve.
State: k, soldiers, reserves (0/1), income, rounds, values (this round), reserve [2], won [2], pts [2],
hist (one [values, alloc0, alloc1] per round played), done (rounds played), over.

legal_actions is a representative list (<= 300) that depends only on the budgets and the past allocations, never
on this round's field values (the arena shuffles them, a simulated next round cannot know them): it holds the
even split over EVERY subset of fields, so "all on the k most valuable fields" is in it for any values.
"""
import re
from itertools import combinations

CONFIGS = {
    "100 soldiers, 5 equal fields": {"k": 5, "soldiers": 100, "reserves": 0, "income": 30, "rounds": 5,
                                     "valued": 0},
    "60 soldiers, 6 fields worth 1-6": {"k": 6, "soldiers": 60, "reserves": 0, "income": 30, "rounds": 5,
                                        "valued": 1},
    "Reserves: +30 per round, 3 fields": {"k": 3, "soldiers": 100, "reserves": 1, "income": 30, "rounds": 5,
                                          "valued": 0},
}
MAX_LIST = 300


def _label(state):
    for label, c in CONFIGS.items():
        if all(c[x] == state[x] for x in ("k", "reserves", "valued")) and (
                c["reserves"] or c["soldiers"] == state["soldiers"]):
            return label
    return "%d soldiers, %d fields" % (state["soldiers"], state["k"])


def initial_state(config, seed):
    import random
    k = int(config["k"])
    s = {"k": k, "soldiers": int(config["soldiers"]), "reserves": int(bool(config.get("reserves"))),
         "income": int(config.get("income", 30)), "rounds": int(config.get("rounds", 5)),
         "valued": int(bool(config.get("valued"))), "values": [1] * k, "reserve": [0, 0],
         "won": [0, 0], "pts": [0, 0], "hist": [], "done": 0, "over": 0}
    if s["reserves"]:
        s["reserve"] = [s["income"], s["income"]]
    if s["valued"]:
        s["values"] = _shuffled(k, random.Random(seed))
    return s


def _shuffled(k, rng):
    v = list(range(1, k + 1))
    rng.shuffle(v)
    return v


def _copy(state):
    s = dict(state)
    for key in ("values", "reserve", "won", "pts"):
        s[key] = list(state[key])
    s["hist"] = [[list(x) for x in h] for h in state["hist"]]
    return s


def budget(state, seat):
    """Soldiers `seat` may place this round (exactly, or at most in the Reserves variant)."""
    return state["reserve"][seat] if state["reserves"] else state["soldiers"]


def to_move(state):
    return [] if state["over"] else [0, 1]


def _parse(action, k):
    """List of k ints, or None when the text is not k whole numbers >= 0."""
    if not isinstance(action, str):
        return None
    toks = [t for t in re.split(r"[\s,]+", action.strip()) if t]
    if len(toks) != k or not all(t.isdigit() and len(t) <= 6 for t in toks):
        return None
    return [int(t) for t in toks]


def is_legal_action(state, seat, action):
    try:
        if state["over"] or seat not in (0, 1):
            return False
        a = _parse(action, state["k"])
        if a is None:
            return False
        b = budget(state, seat)
        return sum(a) <= b if state["reserves"] else sum(a) == b
    except Exception:  # noqa: BLE001
        return False


def _even(total, idx, k):
    """`total` spread as evenly as possible over the fields `idx` (the remainder goes to the first ones)."""
    a = [0] * k
    if idx:
        q, r = divmod(total, len(idx))
        for n, i in enumerate(idx):
            a[i] = q + (1 if n < r else 0)
    return a


def _shapes(total, k):
    """Value-independent families for one total: even over every subset, 2/3+1/3 pairs, half+rest."""
    out = [_even(total, list(range(k)), k)]
    for n in range(1, k):
        for idx in combinations(range(k), n):
            out.append(_even(total, list(idx), k))
    for i in range(k):
        a = _even(total - total // 2, [j for j in range(k) if j != i], k)
        a[i] += total // 2
        out.append(a)
    for i in range(k):
        for j in range(k):
            if i != j:
                a = [0] * k
                a[i] = total - total // 3
                a[j] = total // 3
                out.append(a)
    return out


def _counters(opp, total, k, exact):
    """Beat `opp` (an earlier allocation of the opponent) by one on every subset of a majority or more fields
    that `total` can pay for; the rest goes on the same fields, or on the other fields, or (Reserves) is kept."""
    out = []
    for n in range(k // 2 + 1, k):
        for idx in combinations(range(k), n):
            cost = sum(opp[i] + 1 for i in idx)
            if cost > total:
                continue
            base = [opp[i] + 1 if i in idx else 0 for i in range(k)]
            if not exact:
                out.append(list(base))
            for spread in (list(idx), [i for i in range(k) if i not in idx]):
                extra = _even(total - cost, spread, k)
                out.append([base[i] + extra[i] for i in range(k)])
    return out


def legal_actions(state, seat):
    if state["over"] or seat not in (0, 1):
        return []
    k, b = state["k"], budget(state, seat)
    exact = not state["reserves"]
    cands = []
    if exact:
        cands += _shapes(b, k)
        totals = [b]
    else:
        ob = state["reserve"][1 - seat]
        totals = []
        for t in (b, min(b, ob + k), b - b // 3, b // 2, b // 3, min(b, state["income"]), 0):
            if 0 <= t <= b and t not in totals:
                totals.append(t)
        for t in totals:
            cands += _shapes(t, k)
    for h in reversed(state["hist"]):
        cands += _counters(h[2 - seat], b, k, exact)
    out, seen = [], set()
    for a in cands:
        txt = " ".join(str(x) for x in a)
        if txt not in seen and min(a) >= 0 and (sum(a) == b if exact else sum(a) <= b):
            seen.add(txt)
            out.append(txt)
            if len(out) >= MAX_LIST:
                break
    return out


def random_action(state, seat, rng):
    """Uniform over all legal splits (stars and bars; Reserves: one extra hidden part = soldiers kept)."""
    k, b = state["k"], budget(state, seat)
    parts = k if not state["reserves"] else k + 1
    bars = sorted(rng.sample(range(b + parts - 1), parts - 1))
    a, prev = [], -1
    for x in bars + [b + parts - 1]:
        a.append(x - prev - 1)
        prev = x
    return " ".join(str(x) for x in a[:k])


def could_win_now(state, seat, action):
    return False


def score_round(values, a0, a1):
    """(captured per field: 0, 1 or None; points [seat0, seat1]; round winner 0, 1 or None)."""
    cap, pts = [], [0, 0]
    for v, x, y in zip(values, a0, a1):
        c = 0 if x > y else (1 if y > x else None)
        cap.append(c)
        if c is not None:
            pts[c] += v
    return cap, pts, (0 if pts[0] > pts[1] else (1 if pts[1] > pts[0] else None))


def apply(state, actions, rng):
    s = _copy(state)
    if s["over"]:
        return s
    k = s["k"]
    alloc = []
    for seat in (0, 1):
        act = actions.get(seat, actions.get(str(seat))) if isinstance(actions, dict) else None
        a = _parse(act, k) if is_legal_action(state, seat, act) else None
        if a is None:  # not a legal split: the arena would ask again; here the even split stands in
            a = _even(0 if s["reserves"] else budget(s, seat), list(range(k)), k)
        alloc.append(a)
    _, pts, w = score_round(s["values"], alloc[0], alloc[1])
    s["hist"].append([list(s["values"]), alloc[0], alloc[1]])
    s["done"] += 1
    for seat in (0, 1):
        s["pts"][seat] += pts[seat]
        if s["reserves"]:
            s["reserve"][seat] -= sum(alloc[seat])
    if w is not None:
        s["won"][w] += 1
    if s["done"] >= s["rounds"]:
        s["over"] = 1
    else:
        if s["reserves"]:
            s["reserve"] = [r + s["income"] for r in s["reserve"]]
        if s["valued"]:
            s["values"] = _shuffled(k, rng)
    return s


def is_over(state):
    return bool(state["over"])


def winner(state):
    if not state["over"]:
        return None
    for a in (state["won"], state["pts"]):
        if a[0] != a[1]:
            return 0 if a[0] > a[1] else 1
    return None


def result_reason(state):
    """The arena's reason text for a finished game."""
    w, p = state["won"], state["pts"]
    if w[0] != w[1]:
        x = 0 if w[0] > w[1] else 1
        return "seat %d won more rounds (%d vs %d)" % (x, w[x], w[1 - x])
    if p[0] != p[1]:
        x = 0 if p[0] > p[1] else 1
        return "rounds tied at %d; seat %d captured more battlefield points (%d vs %d)" % (w[0], x, p[x], p[1 - x])
    return "rounds tied at %d and battlefield points tied at %d" % (w[0], p[0])


def observation(state, seat):
    k, opp = state["k"], 1 - seat
    lines = ["COLONEL BLOTTO - %s. You are seat %d." % (_label(state), seat)]
    if state["over"]:
        lines.append("Game over after round %d of %d." % (state["done"], state["rounds"]))
    else:
        lines.append("Round %d of %d." % (state["done"] + 1, state["rounds"]))
    lines.append("Battlefield values this round: " + ", ".join(
        "F%d=%d" % (i + 1, v) for i, v in enumerate(state["values"])))
    b = budget(state, seat)
    if state["reserves"]:
        lines.append("Your reserve: %d soldiers (you may send 0 to %d in total). Opponent's reserve: %d." % (
            b, b, state["reserve"][opp]))
    else:
        lines.append("You must place exactly %d soldiers on %d battlefields." % (b, k))
    lines.append("Rounds won: you %d, opponent %d. Battlefield points captured so far: you %d, opponent %d." % (
        state["won"][seat], state["won"][opp], state["pts"][seat], state["pts"][opp]))
    if state["hist"]:
        lines.append("History:")
        for n, (vals, a0, a1) in enumerate(state["hist"]):
            cap, pts, w = score_round(vals, a0, a1)
            al = (a0, a1)
            parts = []
            for i in range(k):
                who = "nobody" if cap[i] is None else ("you" if cap[i] == seat else "opponent")
                parts.append("F%d(worth %d): you %d vs %d -> %s" % (i + 1, vals[i], al[seat][i], al[opp][i], who))
            end = "round drawn" if w is None else "round won by %s" % ("you" if w == seat else "opponent")
            lines.append("  Round %d: %s. Points: you %d, opponent %d -> %s." % (
                n + 1, "; ".join(parts), pts[seat], pts[opp], end))
    q, r = divmod(b, k)
    lines.append('Your move: %d numbers separated by spaces, e.g. "%s".' % (
        k, " ".join(str(q + (r if i == k - 1 else 0)) for i in range(k))))
    return "\n".join(lines)


_RE_ROUND = re.compile(r"^\s*Round\s+(\d+)\s+of\s+(\d+)", re.I | re.M)
_RE_OVER = re.compile(r"^\s*Game over after round\s+(\d+)", re.I | re.M)
_RE_VALUES = re.compile(r"^\s*Battlefield values[^:\n]*:(.*)$", re.I | re.M)
_RE_EXACT = re.compile(r"place\s+exactly\s+(\d+)\s+soldiers", re.I)
_RE_RES = re.compile(r"Your reserve:\s*(\d+)", re.I)
_RE_ORES = re.compile(r"Opponent'?s reserve:\s*(\d+)", re.I)
_RE_SCORE = re.compile(r"Rounds won:\s*you\s+(\d+)\s*,\s*opponent\s+(\d+).*?so far:\s*you\s+(\d+)\s*,\s*opponent\s+(\d+)",
                       re.I | re.S)
_RE_HLINE = re.compile(r"^\s*Round\s+(\d+)\s*:(.*)$", re.I | re.M)
_RE_FIELD = re.compile(r"F(\d+)\s*\(\s*worth\s+(\d+)\s*\)\s*:\s*you\s+(\d+)\s+vs\s+(\d+)", re.I)


def state_from_observation(text, seat, config, rng):
    s = initial_state(config, 0)
    text = text or ""
    k, opp = s["k"], 1 - seat
    m = _RE_VALUES.search(text)
    if m:
        vals = [int(v) for _, v in sorted((int(i), int(v)) for i, v in re.findall(r"F(\d+)\s*=\s*(\d+)", m.group(1)))]
        if len(vals) == k:
            s["values"] = vals
    rows = {}
    for m in _RE_HLINE.finditer(text):
        f = {int(i): (int(v), int(me), int(they)) for i, v, me, they in _RE_FIELD.findall(m.group(2))}
        if sorted(f) == list(range(1, k + 1)):
            al = [None, None]
            al[seat] = [f[i][1] for i in range(1, k + 1)]
            al[opp] = [f[i][2] for i in range(1, k + 1)]
            rows[int(m.group(1))] = [[f[i][0] for i in range(1, k + 1)], al[0], al[1]]
    hist = [rows[n] for n in sorted(rows)]
    won, pts, sent = [0, 0], [0, 0], [0, 0]
    for vals, a0, a1 in hist:
        _, p, w = score_round(vals, a0, a1)
        pts = [pts[0] + p[0], pts[1] + p[1]]
        sent = [sent[0] + sum(a0), sent[1] + sum(a1)]
        if w is not None:
            won[w] += 1
    m, mo = _RE_ROUND.search(text), _RE_OVER.search(text)
    if mo:
        done, s["over"] = int(mo.group(1)), 1
    else:
        done = int(m.group(1)) - 1 if m else len(hist)
        if m:
            s["rounds"] = int(m.group(2))
    m = _RE_SCORE.search(text)
    if m:  # the totals the text shows decide the game: they win over a cut or missing history
        won[seat], won[opp], pts[seat], pts[opp] = (int(m.group(i)) for i in (1, 2, 3, 4))
    s["hist"], s["won"], s["pts"], s["done"] = hist, won, pts, max(done, 0)
    if s["reserves"]:
        n = s["done"] if s["over"] else s["done"] + 1
        s["reserve"] = [max(0, s["income"] * n - sent[0]), max(0, s["income"] * n - sent[1])]
        m, mo = _RE_RES.search(text), _RE_ORES.search(text)
        if m:
            s["reserve"][seat] = int(m.group(1))
        if mo:
            s["reserve"][opp] = int(mo.group(1))
    else:
        m = _RE_EXACT.search(text)
        if m:
            s["soldiers"] = int(m.group(1))
    if s["done"] >= s["rounds"]:
        s["over"] = 1
    return s

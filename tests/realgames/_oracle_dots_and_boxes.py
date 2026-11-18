"""Exact hand-written simulator of the arena game `dots_and_boxes` (checked against the real replays of 15 Nov).

Rules in short: R x C boxes ((R+1) x (C+1) dots, dot = column letter + row number, "a1" top-left). A move draws one
free line, "line a1-b1" (horizontal) or "line a1-a2" (vertical); the arena spells the left / upper dot first. Completing
one or two boxes gives them to the mover, who MUST move again. The game ends when every line is drawn; more boxes
wins, equal boxes is a draw (only possible with an even number of boxes: 4x4).

State (all values immutable, so dict(state) is a full copy):
    k     (rows, cols) of boxes: key of the precomputed tables
    lines bitmask of the drawn lines (bit i = line i in arena order: dots row by row, horizontal before vertical)
    left  number of lines not drawn yet
    own0, own1  bitmasks of the boxes owned by seat 0 / seat 1 (box index = row * cols + col)
    s0, s1      scores
    turn  seat to move
    hist  move history as a linked list (previous, seat, line index, completed-box bitmask) or None
"""
import re

CONFIGS = {
    "3x3 boxes": {"rows": 3, "cols": 3},
    "4x4 boxes": {"rows": 4, "cols": 4},
    "3x5 boxes": {"rows": 3, "cols": 5},
}

_LET = "abcdefghijklmnopqrstuvwxyz"
_T = {}


def _build(rows, cols):
    names = []          # line index -> "a1-b1"
    hidx = {}           # (r, c) -> index of the horizontal line from dot (r, c) to (r, c+1)
    vidx = {}           # (r, c) -> index of the vertical line from dot (r, c) to (r+1, c)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if c < cols:
                hidx[(r, c)] = len(names)
                names.append("%s%d-%s%d" % (_LET[c], r + 1, _LET[c + 1], r + 1))
            if r < rows:
                vidx[(r, c)] = len(names)
                names.append("%s%d-%s%d" % (_LET[c], r + 1, _LET[c], r + 2))
    total = len(names)
    actions = tuple("line " + n for n in names)
    lookup = {}
    for i, n in enumerate(names):
        a, b = n.split("-")
        for x, y in ((a, b), (b, a)):
            lookup["line %s-%s" % (x, y)] = i
            lookup["line %s %s" % (x, y)] = i
    adj = [[] for _ in range(total)]     # line -> ((mask of the 4 sides, box bit), ...) for the 1 or 2 boxes it borders
    for r in range(rows):
        for c in range(cols):
            sides = (hidx[(r, c)], hidx[(r + 1, c)], vidx[(r, c)], vidx[(r, c + 1)])
            m = 0
            for s in sides:
                m |= 1 << s
            bit = 1 << (r * cols + c)
            for s in sides:
                adj[s].append((m, bit))
    adj = tuple(tuple(a) for a in adj)
    pairs = tuple((1 << i, actions[i]) for i in range(total))
    return {
        "rows": rows, "cols": cols, "total": total, "nbox": rows * cols, "names": tuple(names),
        "actions": actions, "lookup": lookup, "adj": adj, "pairs": pairs, "hidx": hidx, "vidx": vidx,
        "full": (1 << total) - 1,
    }


def _tab(k):
    t = _T.get(k)
    if t is None:
        t = _T[k] = _build(k[0], k[1])
    return t


for _cfg in CONFIGS.values():
    _tab((_cfg["rows"], _cfg["cols"]))


def _line_index(t, action):
    i = t["lookup"].get(action)
    if i is not None:
        return i
    if not isinstance(action, str):
        return None
    parts = action.strip().lower().replace("-", " ").split()
    if len(parts) != 3 or parts[0] != "line":
        return None
    return t["lookup"].get("line %s-%s" % (parts[1], parts[2]))


def initial_state(config, seed):
    k = (int(config["rows"]), int(config["cols"]))
    t = _tab(k)
    return {"k": k, "lines": 0, "left": t["total"], "own0": 0, "own1": 0, "s0": 0, "s1": 0, "turn": 0, "hist": None}


def to_move(state):
    if state["left"] <= 0:
        return []
    return [state["turn"]]


def legal_actions(state, seat):
    if state["left"] <= 0 or seat != state["turn"]:
        return []
    lines = state["lines"]
    return [name for bit, name in _T[state["k"]]["pairs"] if not lines & bit]


def apply(state, actions, rng):
    seat = state["turn"]
    t = _T[state["k"]]
    act = actions[seat] if seat in actions else actions[str(seat)]
    i = t["lookup"].get(act)
    if i is None:
        i = _line_index(t, act)
        if i is None:
            raise ValueError("illegal action %r" % (act,))
    bit = 1 << i
    lines = state["lines"]
    if lines & bit or state["left"] <= 0:
        raise ValueError("illegal action %r" % (act,))
    lines |= bit
    s = dict(state)
    s["lines"] = lines
    s["left"] = state["left"] - 1
    got = 0
    n = 0
    for m, b in t["adj"][i]:
        if lines & m == m:
            got |= b
            n += 1
    if n:
        if seat == 0:
            s["own0"] = state["own0"] | got
            s["s0"] = state["s0"] + n
        else:
            s["own1"] = state["own1"] | got
            s["s1"] = state["s1"] + n
    else:
        s["turn"] = 1 - seat
    s["hist"] = (state["hist"], seat, i, got)
    return s


def is_over(state):
    return state["left"] <= 0


def winner(state):
    if state["left"] > 0:
        return None
    if state["s0"] > state["s1"]:
        return 0
    if state["s1"] > state["s0"]:
        return 1
    return None


def random_action(state, seat, rng):
    lines = state["lines"]
    t = _T[state["k"]]
    total = t["total"]
    acts = t["actions"]
    if state["left"] * 3 >= total:
        while True:
            i = int(rng.random() * total)
            if not (lines >> i) & 1:
                return acts[i]
    free = [i for i in range(total) if not (lines >> i) & 1]
    return acts[free[int(rng.random() * len(free))]]


def is_legal_action(state, seat, action):
    try:
        if state["left"] <= 0 or seat != state["turn"]:
            return False
        i = _line_index(_T[state["k"]], action)
        return i is not None and not (state["lines"] >> i) & 1
    except Exception:
        return False


def could_win_now(state, seat, action):
    return False


def _hist_list(state):
    out = []
    h = state.get("hist")
    while h is not None:
        out.append(h)
        h = h[0]
    out.reverse()
    return out


def _box_name(t, b):
    return "%s%d" % (_LET[b % t["cols"]], b // t["cols"] + 1)


def observation(state, seat):
    t = _T[state["k"]]
    rows, cols, total = t["rows"], t["cols"], t["total"]
    lines = state["lines"]
    my = state["s0"] if seat == 0 else state["s1"]
    op = state["s1"] if seat == 0 else state["s0"]
    out = [
        "DOTS AND BOXES (%dx%d boxes). You are Player %d; boxes you own show %d, the opponent's show %d."
        % (rows, cols, seat, seat, 1 - seat),
        "Score: you %d, opponent %d. Boxes left: %d." % (my, op, t["nbox"] - my - op),
        "Lines drawn: %d of %d." % (total - state["left"], total),
        "",
        "    " + "   ".join(_LET[: cols + 1]),
    ]
    hidx, vidx = t["hidx"], t["vidx"]
    for r in range(rows + 1):
        row = "%2d  +" % (r + 1)
        for c in range(cols):
            row += ("---+" if (lines >> hidx[(r, c)]) & 1 else "   +")
        out.append(row)
        if r < rows:
            row = "    "
            for c in range(cols + 1):
                row += "|" if (lines >> vidx[(r, c)]) & 1 else " "
                if c < cols:
                    b = 1 << (r * cols + c)
                    row += " 0 " if state["own0"] & b else (" 1 " if state["own1"] & b else "   ")
            out.append(row.rstrip())
    out.append("")
    hist = _hist_list(state)
    again = False
    if hist:
        out.append("Move history (oldest first):")
        for n, (_, who, i, got) in enumerate(hist):
            ln = "  %d. %s: line %s" % (n + 1, "You" if who == seat else "Opponent", t["names"][i])
            if got:
                bs = [_box_name(t, b) for b in range(t["nbox"]) if (got >> b) & 1]
                ln += "  (completed %s %s -> moved again)" % ("box" if len(bs) == 1 else "boxes", ", ".join(bs))
            out.append(ln)
        again = bool(hist[-1][3]) and hist[-1][1] == seat
    elif not lines:
        out.append("No lines drawn yet.")
    out.append("(A box is named by its top-left dot.)")
    if state["left"] <= 0:
        out.append("The game is over: all lines are drawn.")
    elif state["turn"] == seat:
        if again:
            out.append("You completed a box with your last line, so you move AGAIN now.")
        first = ""
        for bit, name in t["pairs"]:
            if not lines & bit:
                first = name
                break
        out.append('It is YOUR turn. Answer "line <dot>-<dot>" with a line not drawn yet, e.g. "%s".' % first)
    else:
        out.append("It is the opponent's turn.")
    return "\n".join(out)


_RE_DOTROW = re.compile(r"^\s*(\d+)\s+(\+.*)$")
_RE_HIST = re.compile(r"^\s*(\d+)\.\s*(You|Opponent)\s*:\s*line\s+([a-z]\d+)\s*[- ]\s*([a-z]\d+)", re.I)
_RE_SCORE = re.compile(r"Score:\s*you\s+(\d+)\s*,\s*opponent\s+(\d+)", re.I)
_RE_PLAYER = re.compile(r"You are Player\s+([01])", re.I)


def state_from_observation(text, seat, config, rng):
    k = (int(config["rows"]), int(config["cols"]))
    t = _tab(k)
    rows, cols = k
    hidx, vidx = t["hidx"], t["vidx"]
    m = _RE_PLAYER.search(text)
    if m:
        seat = int(m.group(1))
    rawlines = text.splitlines()

    # 1. the drawing: lines and box owners
    glines = 0
    own = [0, 0]
    grid_found = False
    n = len(rawlines)
    for j in range(n):
        m = _RE_DOTROW.match(rawlines[j])
        if not m:
            continue
        r = int(m.group(1)) - 1
        if r < 0 or r > rows:
            continue
        row = rawlines[j]
        plus = [p for p, ch in enumerate(row) if ch == "+"]
        if len(plus) != cols + 1:
            continue
        grid_found = True
        for c in range(cols):
            if "-" in row[plus[c] + 1: plus[c + 1]]:
                glines |= 1 << hidx[(r, c)]
        if r < rows and j + 1 < n:
            nxt = rawlines[j + 1]
            if nxt.strip() and not _RE_DOTROW.match(nxt) and set(nxt.strip()) <= set("|01 "):
                for c in range(cols + 1):
                    p = plus[c]
                    if p < len(nxt) and nxt[p] == "|":
                        glines |= 1 << vidx[(r, c)]
                    if c < cols:
                        cell = nxt[p + 1: plus[c + 1]].strip()
                        if cell in ("0", "1"):
                            own[int(cell)] |= 1 << (r * cols + c)

    # 2. the move history, replayed
    hist = None
    hlines = 0
    hown = [0, 0]
    hcount = 0
    hok = True
    last_seat = None
    last_got = 0
    for raw in rawlines:
        m = _RE_HIST.match(raw)
        if not m:
            continue
        who = seat if m.group(2).lower() == "you" else 1 - seat
        i = t["lookup"].get("line %s-%s" % (m.group(3).lower(), m.group(4).lower()))
        if i is None or (hlines >> i) & 1:
            hok = False
            continue
        hlines |= 1 << i
        got = 0
        for msk, b in t["adj"][i]:
            if hlines & msk == msk:
                got |= b
        hown[who] |= got
        hist = (hist, who, i, got)
        hcount += 1
        last_seat, last_got = who, got

    if grid_found:
        lines = glines
        if not (hok and hlines == glines):
            hist = None               # incomplete history: keep only what the drawing shows
            last_seat = None
        else:
            # owners from the drawing win; fill any box the drawing does not label from the history
            for s in (0, 1):
                own[s] |= hown[s] & ~(own[0] | own[1])
    else:
        lines = hlines
        own = hown

    # completed boxes without a known owner: give them so that the printed score is matched when possible
    complete = 0
    for b in range(rows * cols):
        r, c = divmod(b, cols)
        msk = (1 << hidx[(r, c)]) | (1 << hidx[(r + 1, c)]) | (1 << vidx[(r, c)]) | (1 << vidx[(r, c + 1)])
        if lines & msk == msk:
            complete |= 1 << b
    own[0] &= complete
    own[1] &= complete & ~own[0]
    s = [bin(own[0]).count("1"), bin(own[1]).count("1")]
    unknown = complete & ~(own[0] | own[1])
    if unknown:
        want = None
        m = _RE_SCORE.search(text)
        if m:
            want = [0, 0]
            want[seat] = int(m.group(1))
            want[1 - seat] = int(m.group(2))
        for b in range(rows * cols):
            if (unknown >> b) & 1:
                to = 0
                if want is not None:
                    to = 0 if want[0] - s[0] >= want[1] - s[1] else 1
                own[to] |= 1 << b
                s[to] += 1

    left = t["total"] - bin(lines).count("1")
    turn = seat
    low = text.lower()
    if "your turn" not in low and "move again now" not in low:
        # an observation that does not ask this seat to move: derive the mover from the last history line
        if last_seat is not None:
            turn = last_seat if last_got else 1 - last_seat
        else:
            turn = 1 - seat
    return {"k": k, "lines": lines, "left": left, "own0": own[0], "own1": own[1], "s0": s[0], "s1": s[1],
            "turn": turn, "hist": hist}


def _ordered(t, lines, total):
    """Free lines: box-completing ones first, then safe ones, last the ones that give a box away (3rd side)."""
    done, safe, gives = [], [], []
    adj = t["adj"]
    for i in range(total):
        bit = 1 << i
        if lines & bit:
            continue
        nl = lines | bit
        kind = 1
        for m, _b in adj[i]:
            c = (nl & m).bit_count() if hasattr(int, "bit_count") else bin(nl & m).count("1")
            if c == 4:
                kind = 0
                break
            if c == 3:
                kind = 2
        (done if kind == 0 else safe if kind == 1 else gives).append(i)
    return done + safe + gives


def _ab(t, total, lines, left, diff, mine, depth, alpha, beta, deadline, clock):
    """Alpha-beta on (drawn lines, box difference for us, whose turn). `mine` = it is our turn."""
    if left == 0:
        return diff * 100
    if depth <= 0:
        return diff
    if clock() > deadline:
        raise TimeoutError
    adj = t["adj"]
    best = -1e9 if mine else 1e9
    for i in _ordered(t, lines, total):
        nl = lines | (1 << i)
        n = 0
        for m, _b in adj[i]:
            if nl & m == m:
                n += 1
        if n:
            v = _ab(t, total, nl, left - 1, diff + (n if mine else -n), mine, depth - 1, alpha, beta, deadline, clock)
        else:
            v = _ab(t, total, nl, left - 1, diff, not mine, depth - 1, alpha, beta, deadline, clock)
        if mine:
            if v > best:
                best = v
            if best > alpha:
                alpha = best
        else:
            if v < best:
                best = v
            if best < beta:
                beta = best
        if alpha >= beta:
            break
    return best


def recommend(text, seat, config, rng, seconds=2.0):
    """Optional hook of a known simulator (mesh/simworker.candidates): [(action, value)] best first. Iterative-deepening
    alpha-beta on the exact rules (extra move after a box), box difference as the evaluation: it never gives a box
    away for nothing and plays the chain endgame exactly once few lines are left. Round robin of 15 Nov: 2-14 and 1-15
    on 4x4 with random playouts, third sides drawn from move 5 on."""
    import time as _time
    st = state_from_observation(text, seat, config, rng)
    if st["left"] <= 0 or st["turn"] != seat:
        return []
    t = _T[st["k"]]
    total, lines, left = t["total"], st["lines"], st["left"]
    diff = (st["s0"] - st["s1"]) if seat == 0 else (st["s1"] - st["s0"])
    acts = t["actions"]
    clock = _time.monotonic
    deadline = clock() + max(0.2, seconds * 0.85)
    order = _ordered(t, lines, total)
    scores = {i: 0.0 for i in order}
    adj = t["adj"]
    depth = 1
    while depth <= left:
        cur, alpha = {}, -1e9
        try:
            for i in order:
                nl = lines | (1 << i)
                n = sum(1 for m, _b in adj[i] if nl & m == m)
                if n:
                    v = _ab(t, total, nl, left - 1, diff + n, True, depth - 1, alpha, 1e9, deadline, clock)
                else:
                    v = _ab(t, total, nl, left - 1, diff, False, depth - 1, alpha, 1e9, deadline, clock)
                cur[i] = v
                if v > alpha:
                    alpha = v
        except TimeoutError:
            break
        scores = cur
        order = sorted(order, key=lambda i: -scores[i])
        if clock() > deadline:
            break
        depth += 1
    boxes = max(1, len(t.get("boxes", [])) or (st["s0"] + st["s1"] + left))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    out = []
    for i, v in ranked[:6]:
        v = v / 100.0 if abs(v) >= 100 else v
        out.append((acts[i], max(-1.0, min(1.0, v / 8.0))))
    return out

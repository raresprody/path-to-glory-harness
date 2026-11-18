"""Hand-written exact simulator of the arena game `connect_four` (variants: rows x columns, connect N).

Seat 0 plays X and moves first, seat 1 plays O. Action: "drop <column>" (1-based, left to right). A disc falls to the
lowest empty cell. First line of N or more wins; a full board without a line is a draw (winner None).

State (small, cheap to copy): a padded flat board (one sentinel column per row, sentinel rows below and above) so a
line check through the last move needs no bounds tests; column heights; disc count; the move history as a string of
column digits; the cached result.
"""
import re

CONFIGS = {
    "6x7, connect 4": {"rows": 6, "cols": 7, "connect": 4},
    "5x6, connect 4": {"rows": 5, "cols": 6, "connect": 4},
    "7x9, connect 5": {"rows": 7, "cols": 9, "connect": 5},
}

_ACTIONS = ["drop %d" % (c + 1) for c in range(64)]
_ACT_COL = {a: i for i, a in enumerate(_ACTIONS)}
_SYM = ".XO"


def _blank(rows, cols, connect):
    w = cols + 1
    board = [-1] * ((rows + 2) * w + 1)
    for r in range(rows):
        base = (r + 1) * w
        for c in range(cols):
            board[base + c] = 0
    return {"rows": rows, "cols": cols, "connect": connect, "w": w, "board": board, "heights": [0] * cols,
            "n": 0, "hist": "", "over": False, "win": None}


def initial_state(config, seed):
    return _blank(int(config["rows"]), int(config["cols"]), int(config["connect"]))


def to_move(state):
    if state["over"]:
        return []
    return [state["n"] & 1]


def legal_actions(state, seat):
    if state["over"] or (state["n"] & 1) != seat:
        return []
    rows = state["rows"]
    acts = _ACTIONS
    return [acts[c] for c, h in enumerate(state["heights"]) if h < rows]


def _col_of(state, action):
    c = _ACT_COL.get(action)
    if c is None:
        try:
            parts = str(action).strip().lower().split()
            if len(parts) != 2 or parts[0] != "drop" or not parts[1].isdigit():
                return -1
            c = int(parts[1]) - 1
        except Exception:
            return -1
    if c < 0 or c >= state["cols"] or state["heights"][c] >= state["rows"]:
        return -1
    return c


def is_legal_action(state, seat, action):
    try:
        if state["over"] or (state["n"] & 1) != seat:
            return False
        return _col_of(state, action) >= 0
    except Exception:
        return False


def apply(state, actions, rng):
    if state["over"]:
        return dict(state)
    n = state["n"]
    seat = n & 1
    action = actions.get(seat)
    if action is None:
        action = actions.get(str(seat))
    c = _col_of(state, action)
    if c < 0:
        raise ValueError("illegal action %r" % (action,))
    new = dict(state)
    board = new["board"] = state["board"][:]
    heights = new["heights"] = state["heights"][:]
    w = state["w"]
    h = heights[c]
    pos = (h + 1) * w + c
    me = seat + 1
    board[pos] = me
    heights[c] = h + 1
    n += 1
    new["n"] = n
    new["hist"] = state["hist"] + chr(49 + c)
    need = state["connect"]
    # vertical: only downwards
    if h + 1 >= need:
        k = 1
        p = pos - w
        while board[p] == me:
            k += 1
            p -= w
        if k >= need:
            new["over"] = True
            new["win"] = seat
            return new
    for d in (1, w + 1, w - 1):
        k = 1
        p = pos + d
        while board[p] == me:
            k += 1
            p += d
        p = pos - d
        while board[p] == me:
            k += 1
            p -= d
        if k >= need:
            new["over"] = True
            new["win"] = seat
            return new
    if n >= state["rows"] * state["cols"]:
        new["over"] = True
    return new


def is_over(state):
    return state["over"]


def winner(state):
    return state["win"]


def random_action(state, seat, rng):
    rows = state["rows"]
    open_cols = [c for c, h in enumerate(state["heights"]) if h < rows]
    return _ACTIONS[open_cols[int(rng.random() * len(open_cols))]]


def could_win_now(state, seat, action):
    return False


def observation(state, seat):
    rows, cols, w, board = state["rows"], state["cols"], state["w"], state["board"]
    me, opp = _SYM[seat + 1], _SYM[2 - seat]
    nums = " " + " ".join(str(c + 1) for c in range(cols))
    lines = ["CONNECT %d (%dx%d). You are Player %d and play %s; the opponent plays %s."
             % (state["connect"], rows, cols, seat, me, opp),
             "Discs on the board: %d of %d." % (state["n"], rows * cols), "", nums]
    for r in range(rows - 1, -1, -1):
        base = (r + 1) * w
        lines.append("|" + " ".join(_SYM[board[base + c]] for c in range(cols)) + "|")
    lines.append("+" + "-" * (2 * cols - 1) + "+")
    lines.append(nums)
    lines.append("")
    hist = state["hist"]
    if not hist:
        lines.append("No moves yet.")
    else:
        lines.append("Moves so far (disc+column, X moved first): "
                     + ", ".join("XO"[i & 1] + ch for i, ch in enumerate(hist)))
        last = (len(hist) - 1) & 1
        who = "You" if last == seat else "Opponent"
        lines.append("Last move: %s (%s) dropped in column %s." % (who, "XO"[last], hist[-1]))
    lines.append("Open columns: " + " ".join(str(c + 1) for c, h in enumerate(state["heights"]) if h < rows))
    if state["over"]:
        if state["win"] is None:
            lines.append("Game over: the board is full with no line: draw.")
        else:
            lines.append("Game over: seat %d (%s) connected %d in a row."
                         % (state["win"], "XO"[state["win"]], state["connect"]))
    elif (state["n"] & 1) == seat:
        lines.append('It is YOUR turn (%s). Answer "drop <column>".' % me)
    else:
        lines.append("Waiting for the opponent (%s)." % opp)
    return "\n".join(lines)


def _full_scan(state):
    """Winner seat if any line of `connect` exists on the board (used only after parsing)."""
    rows, cols, w, board, need = state["rows"], state["cols"], state["w"], state["board"], state["connect"]
    for r in range(rows):
        for c in range(cols):
            pos = (r + 1) * w + c
            me = board[pos]
            if me <= 0:
                continue
            for d in (1, w, w + 1, w - 1):
                if board[pos - d] == me:
                    continue
                k = 1
                p = pos + d
                while board[p] == me:
                    k += 1
                    p += d
                if k >= need:
                    return me - 1
    return None


def state_from_observation(text, seat, config, rng):
    text = text or ""
    grid = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and len(s) >= 2:
            cells = re.findall(r"[.XOxo]", s[1:-1])
            if cells:
                grid.append([c.upper() for c in cells])
    rows = int(config.get("rows", 0) or 0) if config else 0
    cols = int(config.get("cols", 0) or 0) if config else 0
    connect = int(config.get("connect", 0) or 0) if config else 0
    m = re.search(r"CONNECT\s+(\d+)\s*\(\s*(\d+)\s*x\s*(\d+)\s*\)", text, re.I)
    if m:
        connect = connect or int(m.group(1))
        rows = rows or int(m.group(2))
        cols = cols or int(m.group(3))
    if grid:
        if len(grid) != rows or any(len(g) != cols for g in grid):
            rows, cols = len(grid), max(len(g) for g in grid)
    connect = connect or 4
    state = _blank(rows, cols, connect)
    w, board, heights = state["w"], state["board"], state["heights"]
    n = 0
    for i, cells in enumerate(grid):
        r = rows - 1 - i
        for c, ch in enumerate(cells[:cols]):
            if ch in "XO":
                board[(r + 1) * w + c] = 1 if ch == "X" else 2
                n += 1
    for c in range(cols):
        h = 0
        for r in range(rows):
            if board[(r + 1) * w + c] > 0:
                h = r + 1
        heights[c] = h
    state["n"] = n
    hist = ""
    m = re.search(r"Moves so far[^:\n]*:\s*([^\n]*)", text, re.I)
    if m:
        hist = "".join(d for _, d in re.findall(r"([XOxo])\s*(\d)", m.group(1)))
    if len(hist) != n or any(not ("1" <= ch <= chr(48 + cols)) for ch in hist):
        hist = _fake_history(state)
    state["hist"] = hist
    # side to move: the text says so when it is this seat's turn; otherwise use the disc count
    if re.search(r"YOUR turn", text, re.I) and seat in (0, 1) and (n & 1) != seat:
        # inconsistent count (should not happen): trust the text by keeping parity through the disc count of X/O
        pass
    win = _full_scan(state)
    if win is not None:
        state["over"] = True
        state["win"] = win
    elif n >= rows * cols:
        state["over"] = True
    return state


def _fake_history(state):
    """A plausible move order (alternating X, O where possible) reproducing the board, bottom-up per column."""
    rows, cols, w, board = state["rows"], state["cols"], state["w"], state["board"]
    nxt = [0] * cols
    out = []
    total = state["n"]
    while len(out) < total:
        want = (len(out) & 1) + 1
        pick = -1
        for c in range(cols):
            if nxt[c] < rows and board[(nxt[c] + 1) * w + c] == want:
                pick = c
                break
        if pick < 0:
            for c in range(cols):
                if nxt[c] < rows and board[(nxt[c] + 1) * w + c] > 0:
                    pick = c
                    break
        if pick < 0:
            break
        nxt[pick] += 1
        out.append(chr(49 + pick))
    return "".join(out)


_WINDOWS = {}  # (rows, cols, connect, w) -> list of tuples of board positions, every possible winning line


def _windows(state):
    rows, cols, need, w = state["rows"], state["cols"], state["connect"], state["w"]
    key = (rows, cols, need, w)
    if key not in _WINDOWS:
        out = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + dr * k, c + dc * k) for k in range(need)]
                    if all(0 <= rr < rows and 0 <= cc < cols for rr, cc in cells):
                        out.append(tuple((rr + 1) * w + cc for rr, cc in cells))
        _WINDOWS[key] = out
    return _WINDOWS[key]


def _eval(state, seat):
    """Open lines: every possible winning line that holds only one colour scores the square of its stones."""
    if state["over"]:
        w = state.get("win")
        return 0.0 if w is None else (10000.0 if w == seat else -10000.0)
    board, me, opp = state["board"], seat + 1, 2 - seat
    score = 0
    for win in _windows(state):
        a = b = 0
        for p in win:
            v = board[p]
            if v == me:
                a += 1
            elif v == opp:
                b += 1
        if a and not b:
            score += a * a
        elif b and not a:
            score -= b * b
    return float(score)


def _moves(state):
    """Legal columns, centre first (better alpha-beta cuts and better play)."""
    cols, rows, heights = state["cols"], state["rows"], state["heights"]
    mid = (cols - 1) / 2.0
    return [_ACTIONS[c] for c in sorted(range(cols), key=lambda c: abs(c - mid)) if heights[c] < rows]


def _ab(state, depth, alpha, beta, seat, deadline, clock):
    if state["over"] or depth <= 0:
        return _eval(state, seat)
    if clock() > deadline:
        raise TimeoutError
    side = state["n"] & 1
    best = -1e9 if side == seat else 1e9
    for a in _moves(state):
        v = _ab(apply(state, {side: a}, None), depth - 1, alpha, beta, seat, deadline, clock)
        if abs(v) > 9000:
            v -= 1 if v > 0 else -1  # the faster win, the slower loss
        if side == seat:
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
    alpha-beta on the exact rules with an open-lines evaluation. Round robin of 15 Nov: 3-5, lost twice each to two
    teams with a real search while ours ran random playouts."""
    import time as _time
    st = state_from_observation(text, seat, config, rng)
    if st["over"] or (st["n"] & 1) != seat:
        return []
    clock = _time.monotonic
    deadline = clock() + max(0.2, seconds * 0.85)
    order = _moves(st)
    if not order:
        return []
    scores = {a: 0.0 for a in order}
    depth = 1
    while depth <= st["rows"] * st["cols"] - st["n"]:
        cur, alpha = {}, -1e9
        try:
            for a in order:
                v = _ab(apply(st, {seat: a}, None), depth - 1, alpha, 1e9, seat, deadline, clock)
                cur[a] = v
                if v > alpha:
                    alpha = v
        except TimeoutError:
            break
        scores = cur
        order = sorted(order, key=lambda a: -scores[a])
        if max(scores.values()) > 9000 or clock() > deadline:
            break
        depth += 1
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(a, max(-1.0, min(1.0, v / 60.0))) for a, v in ranked[:6]]

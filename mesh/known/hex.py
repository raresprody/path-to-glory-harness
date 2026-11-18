"""Exact hand-written simulator of the arena game `hex` (5x5, 7x7, 7x7 with swap rule).

Board: flat list of n*n ints, index = row*n + col (row 0 = row "1" = TOP, col 0 = column "a" = LEFT).
0 = empty, 1 = X (seat 0, joins TOP<->BOTTOM), 2 = O (seat 1, joins LEFT<->RIGHT).
Cell (col L, row R) touches (L-1,R) (L+1,R) (L,R-1) (L+1,R-1) (L-1,R+1) (L,R+1).
Swap rule: seat 1 may answer seat 0's FIRST stone with "swap": the X stone at (col c, row r) is removed and an
O stone is put on the mirror cell (col r, row c); then it is seat 0's turn again.
"""
import random
import re

CONFIGS = {
    "5x5": {"size": 5, "swap": False},
    "7x7": {"size": 7, "swap": False},
    "7x7 with swap rule": {"size": 7, "swap": True},
}

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_TABLES = {}


def _tables(n):
    t = _TABLES.get(n)
    if t is None:
        nbrs = []
        for r in range(n):
            for c in range(n):
                l = []
                for dc, dr in ((-1, 0), (1, 0), (0, -1), (1, -1), (-1, 1), (0, 1)):
                    cc, rr = c + dc, r + dr
                    if 0 <= cc < n and 0 <= rr < n:
                        l.append(rr * n + cc)
                nbrs.append(tuple(l))
        names = tuple("place %s%d" % (_LETTERS[c], r + 1) for r in range(n) for c in range(n))
        index = {a: i for i, a in enumerate(names)}
        t = _TABLES[n] = (tuple(nbrs), names, index)
    return t


for _n in (5, 7):
    _tables(_n)


def _new(n, swap):
    return {"n": n, "swap": bool(swap), "board": [0] * (n * n), "turn": 0, "moves": 0, "stones": 0,
            "swapped": False, "winner": None, "hist": ()}


def initial_state(config, seed):
    return _new(int(config.get("size", 7)), config.get("swap", False))


def to_move(state):
    if state["winner"] is not None or state["stones"] >= state["n"] * state["n"]:
        return []
    return [state["turn"]]


def _swap_ok(state, seat):
    return state["swap"] and seat == 1 and state["moves"] == 1 and not state["swapped"]


def legal_actions(state, seat):
    if state["winner"] is not None or seat != state["turn"]:
        return []
    names = _tables(state["n"])[1]
    out = [names[i] for i, v in enumerate(state["board"]) if not v]
    if out and _swap_ok(state, seat):
        out.append("swap")
    return out


def _idx(state, action):
    index = _tables(state["n"])[2]
    i = index.get(action)
    if i is None and isinstance(action, str):
        i = index.get(" ".join(action.strip().lower().split()))
    return i


def _wins(board, n, nbrs, start, p):
    """Does the group of colour p containing `start` touch both of p's edges?"""
    if p == 1:
        r = start // n
        lo, hi = r == 0, r == n - 1
    else:
        c = start % n
        lo, hi = c == 0, c == n - 1
    if lo and hi:
        return True
    seen = {start}
    stack = [start]
    last = n - 1
    while stack:
        cur = stack.pop()
        for j in nbrs[cur]:
            if board[j] == p and j not in seen:
                seen.add(j)
                k = j // n if p == 1 else j % n
                if k == 0:
                    lo = True
                    if hi:
                        return True
                elif k == last:
                    hi = True
                    if lo:
                        return True
                stack.append(j)
    return False


def apply(state, actions, rng):
    seat = state["turn"]
    if state["winner"] is not None:
        return dict(state, board=list(state["board"]))
    a = actions.get(seat)
    if a is None:
        a = actions.get(str(seat))
    n = state["n"]
    new = dict(state)
    board = new["board"] = list(state["board"])
    sym = "X" if seat == 0 else "O"
    i = _idx(state, a)
    if i is None:
        if isinstance(a, str) and a.strip().lower() == "swap" and _swap_ok(state, seat):
            src = board.index(1)
            r, c = divmod(src, n)
            dst = c * n + r
            board[src] = 0
            board[dst] = 2
            new["swapped"] = True
            new["moves"] = state["moves"] + 1
            new["turn"] = 0
            new["hist"] = state["hist"] + ("O swap (X %s%d became O %s%d)" % (
                _LETTERS[c], r + 1, _LETTERS[r], c + 1),)
            if n == 1:
                new["winner"] = 1
            return new
        raise ValueError("illegal action %r" % (a,))
    if board[i]:
        raise ValueError("cell occupied: %r" % (a,))
    p = seat + 1
    board[i] = p
    stones = new["stones"] = state["stones"] + 1
    new["moves"] = state["moves"] + 1
    new["turn"] = 1 - seat
    new["hist"] = state["hist"] + (sym + " " + _tables(n)[1][i],)
    if stones >= 2 * n - 2 and _wins(board, n, _tables(n)[0], i, p):
        new["winner"] = seat
    return new


def is_over(state):
    return state["winner"] is not None or state["stones"] >= state["n"] * state["n"]


def winner(state):
    return state["winner"]


def observation(state, seat):
    n = state["n"]
    board = state["board"]
    last = _LETTERS[n - 1]
    letters = " ".join(_LETTERS[:n])
    if seat == 0:
        head = "You are Player 0 and play X: connect the TOP edge (row 1) to the BOTTOM edge (row %d)." % n
    else:
        head = "You are Player 1 and play O: connect the LEFT edge (column a) to the RIGHT edge (column %s)." % last
    lines = ["HEX (%dx%d%s). %s" % (n, n, ", swap rule" if state["swap"] else "", head),
             "Stones on the board: %d of %d." % (state["stones"], n * n), "",
             "     " + letters + "        (X joins top<->bottom, O joins left<->right)"]
    for r in range(n):
        row = " ".join(".XO"[board[r * n + c]] for c in range(n))
        lines.append(" " * (r + 2) + "%d  %s  %d" % (r + 1, row, r + 1))
    lines.append(" " * (n + 4) + letters)
    lines.append("")
    lines.append("Moves so far: " + ", ".join(state["hist"]) if state["hist"] else "No moves yet.")
    sym = "X" if seat == 0 else "O"
    if is_over(state):
        w = state["winner"]
        lines.append("Game over." if w is None else "Game over: Player %d (%s) wins." % (w, "XO"[w]))
    elif seat == state["turn"]:
        if _swap_ok(state, seat):
            lines.append('You may answer with "swap" (see the rules) or place a stone.')
        eg = "a1"
        for i, v in enumerate(board):
            if not v:
                eg = _tables(n)[1][i][6:]
                break
        lines.append('It is YOUR turn (%s). Answer "place <cell>" on an empty cell, e.g. "place %s".' % (sym, eg))
    else:
        lines.append("Waiting for the opponent (%s)." % ("O" if seat == 0 else "X"))
    return "\n".join(lines)


_ROW_RE = re.compile(r"^\s*(\d+)\s+((?:[.XOxo]\s+)+)(\d+)\s*$")


def state_from_observation(text, seat, config, rng):
    text = text or ""
    rows = {}
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if m and m.group(1) == m.group(3):
            rows[int(m.group(1))] = m.group(2).split()
    n = int(config.get("size") or 0) if isinstance(config, dict) else 0
    if rows and (not n or len(rows) != n or any(len(v) != n for v in rows.values())):
        n = len(rows)
    if not n:
        n = 7
    swap = bool(config.get("swap", False)) if isinstance(config, dict) else False
    if re.search(r"swap rule\)|\bswap\b", text.split("\n", 1)[0], re.I):
        swap = True
    st = _new(n, swap)
    board = st["board"]
    for r, cells in rows.items():
        if 1 <= r <= n:
            for c, ch in enumerate(cells[:n]):
                if ch in "Xx":
                    board[(r - 1) * n + c] = 1
                elif ch in "Oo":
                    board[(r - 1) * n + c] = 2
    nx, no = board.count(1), board.count(2)
    st["stones"] = nx + no
    hist = ()
    m = re.search(r"Moves so far:\s*(.*)", text)
    if m:
        hist = tuple(h.strip() for h in re.split(r",\s*(?=[XO]\s)", m.group(1).strip()) if h.strip())
    swapped = any(re.match(r"[XO]\s+swap", h) for h in hist)
    your = re.search(r"It is YOUR turn", text, re.I)
    if your:
        turn = seat
    elif hist:
        turn = 0 if (swapped and len(hist) == 2) else (1 if hist[-1][:1] == "X" else 0)
    else:
        turn = 0 if nx <= no else 1
    if not swapped and swap and not hist:
        # no history line: a board where X has one stone fewer than it should means a swap happened
        swapped = (nx == no - 1 and turn == 0) or (nx == no and turn == 1 and nx > 0)
    st["turn"] = turn
    st["swapped"] = bool(swapped)
    st["moves"] = nx + no + (1 if swapped else 0)
    if len(hist) == st["moves"]:
        st["hist"] = hist
    else:
        st["hist"] = hist
    nbrs = _tables(n)[0]
    for p in (1, 2):
        done = False
        for i in range(n):
            s0 = i if p == 1 else i * n
            if board[s0] == p and _wins(board, n, nbrs, s0, p):
                st["winner"] = p - 1
                done = True
                break
        if done:
            break
    return st


def random_action(state, seat, rng):
    acts = legal_actions(state, seat)
    return rng.choice(acts) if acts else ""


def is_legal_action(state, seat, action):
    try:
        if state["winner"] is not None or seat != state["turn"] or not isinstance(action, str):
            return False
        a = " ".join(action.strip().lower().split())
        if a == "swap":
            return bool(_swap_ok(state, seat))
        i = _tables(state["n"])[2].get(a)
        return i is not None and state["board"][i] == 0
    except Exception:  # noqa: BLE001
        return False


def could_win_now(state, seat, action):
    return False


# ---------------------------------------------------------------- own policy (t3/library): shortest-path search
def _field(board, n, nbrs, p, from_start):
    """0-1 BFS: for every cell, the number of EMPTY cells a chain of player p needs from one of its edges up to and
    including that cell (own stones are free, the opponent's block). p=1 joins top-bottom, p=2 left-right."""
    from collections import deque
    INF = 99
    dist = [INF] * (n * n)
    dq = deque()
    if p == 1:
        edge = range(n) if from_start else range(n * (n - 1), n * n)
    else:
        edge = range(0, n * n, n) if from_start else range(n - 1, n * n, n)
    for i in edge:
        v = board[i]
        if v == p:
            dist[i] = 0
            dq.appendleft(i)
        elif v == 0:
            dist[i] = 1
            dq.append(i)
    while dq:
        i = dq.popleft()
        d = dist[i]
        for j in nbrs[i]:
            v = board[j]
            if v == p:
                if d < dist[j]:
                    dist[j] = d
                    dq.appendleft(j)
            elif v == 0:
                if d + 1 < dist[j]:
                    dist[j] = d + 1
                    dq.append(j)
    return dist


def _paths(board, n, nbrs, p):
    """(shortest distance for p, set of empty cells that lie on some shortest connection)."""
    a = _field(board, n, nbrs, p, True)
    b = _field(board, n, nbrs, p, False)
    best = 99
    for i in range(n * n):
        if board[i] != 3 - p and a[i] < 99 and b[i] < 99:
            t = a[i] + b[i] - (1 if board[i] == 0 else 0)
            if t < best:
                best = t
    cells = [i for i in range(n * n) if board[i] == 0 and a[i] + b[i] - 1 == best]
    return best, cells


def _hex_search(board, n, nbrs, side, seat, depth, alpha, beta, deadline, clock):
    me, opp = seat + 1, 2 - seat
    dm, mine = _paths(board, n, nbrs, me)
    do, theirs = _paths(board, n, nbrs, opp)
    if dm == 0:
        return 100.0 + depth
    if do == 0:
        return -100.0 - depth
    if depth <= 0:
        return (do - dm) + (0.4 if side == seat else -0.4)
    if clock() > deadline:
        raise TimeoutError
    both = set(mine) & set(theirs)
    order = list(both) + [i for i in (mine if side == seat else theirs) if i not in both] \
        + [i for i in (theirs if side == seat else mine) if i not in both]
    order = order[:9]
    p = side + 1
    best = -1e9 if side == seat else 1e9
    for i in order:
        board[i] = p
        try:
            v = _hex_search(board, n, nbrs, 1 - side, seat, depth - 1, alpha, beta, deadline, clock)
        finally:
            board[i] = 0
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
    """Optional hook of a known simulator (mesh/simworker.candidates): [(action, value)] best first. Alpha-beta over
    the cells on either player's shortest connections, evaluated by the difference of the two shortest-path lengths
    (empty cells still needed). Round robin of 15 Nov: three games lost to a straight line we never cut."""
    import time as _time
    st = state_from_observation(text, seat, config, rng)
    if st["winner"] is not None or st["turn"] != seat:
        return []
    n = st["n"]
    nbrs, names, _index = _tables(n)
    board = list(st["board"])
    if _swap_ok(st, seat):
        # swap when their first stone is a strong one (off the edge): we take it over
        i = board.index(1) if 1 in board else -1
        r, c = divmod(i, n) if i >= 0 else (0, 0)
        if i >= 0 and 1 <= r <= n - 2 and 1 <= c <= n - 2:
            return [("swap", 0.3)]
    if st["stones"] == 0:
        if st["swap"]:  # a modest first stone: not worth swapping, still useful
            return [(names[(n - 2) * n + 0], 0.0), (names[1 * n + (n - 1)], 0.0)]
        mid = (n // 2) * n + n // 2
        return [(names[mid], 0.2)]
    clock = _time.monotonic
    deadline = clock() + max(0.2, seconds * 0.85)
    me, opp = seat + 1, 2 - seat
    _dm, mine = _paths(board, n, nbrs, me)
    _do, theirs = _paths(board, n, nbrs, opp)
    both = set(mine) & set(theirs)
    roots = list(both) + [i for i in mine if i not in both] + [i for i in theirs if i not in both]
    if not roots:
        roots = [i for i in range(n * n) if board[i] == 0]
    roots = roots[:16]
    scores = {i: 0.0 for i in roots}
    depth = 1
    while depth <= 8:
        cur, alpha = {}, -1e9
        try:
            for i in roots:
                board[i] = me
                try:
                    v = _hex_search(board, n, nbrs, 1 - seat, seat, depth - 1, alpha, 1e9, deadline, clock)
                finally:
                    board[i] = 0
                cur[i] = v
                if v > alpha:
                    alpha = v
        except TimeoutError:
            break
        scores = cur
        roots = sorted(roots, key=lambda i: -scores[i])
        if max(scores.values()) >= 100 or clock() > deadline:
            break
        depth += 1
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(names[i], max(-1.0, min(1.0, v / 8.0))) for i, v in ranked[:6]]

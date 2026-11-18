"""Hand-written exact simulator of the arena game `mini_chess` (variants: Gardner 5x5, Los Alamos 6x6).

Seat 0 is White (uppercase, moves first, pawns go up), seat 1 is Black. Action: "<from><to>" such as "b2b3".
Rules: standard piece moves; pawns move one square only (no double step, no en passant), no castling; a pawn on the
last rank always becomes a queen. Moving into / staying in check is illegal. No legal move: checkmate if in check
(the side to move loses), else stalemate (draw). Only the two kings left: draw. Move cap (40 moves each = 80 plies):
more material on the board wins (Q9 R5 B3 N3 P1), equal = draw. No repetition rule.

State: flat board of ints (index = rank*size + file, rank 0 = rank 1; white 1..6 = P N B R Q K, black negative),
ply, material, piece count, history / captured strings, the cached legal moves of the side to move and the cached
result. All move generation uses per-size precomputed tables.
"""
import re

CONFIGS = {
    "Gardner 5x5": {"size": 5, "setup": "gardner", "max_moves": 40},
    "Los Alamos 6x6": {"size": 6, "setup": "los_alamos", "max_moves": 40},
}

_FILES = "abcdef"
_SETUPS = {"gardner": "RNBQK", "los_alamos": "RNQKNR"}
_CODE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}
_CHARS = ".PNBRQKkqrbnp"  # index by code (negative codes index from the end)
_VAL = (0, 1, 3, 3, 5, 9, 0)
_NAME = ("", "pawn", "knight", "bishop", "rook", "queen", "king")


def _tables(n):
    nn = n * n
    names = [_FILES[i % n] + str(i // n + 1) for i in range(nn)]

    def jumps(sq, deltas):
        r, c = divmod(sq, n)
        return tuple((r + dr) * n + c + dc for dr, dc in deltas if 0 <= r + dr < n and 0 <= c + dc < n)

    def rays(sq, dirs):
        r, c = divmod(sq, n)
        out = []
        for dr, dc in dirs:
            ray = []
            rr, cc = r + dr, c + dc
            while 0 <= rr < n and 0 <= cc < n:
                ray.append(rr * n + cc)
                rr += dr
                cc += dc
            if ray:
                out.append(tuple(ray))
        return tuple(out)

    kd = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
    od = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dd = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    knight = [jumps(s, kd) for s in range(nn)]
    king = [jumps(s, od + dd) for s in range(nn)]
    rook = [rays(s, od) for s in range(nn)]
    bishop = [rays(s, dd) for s in range(nn)]
    push = {1: [s + n if s + n < nn else -1 for s in range(nn)], -1: [s - n if s - n >= 0 else -1 for s in range(nn)]}
    pcap = {1: [jumps(s, ((1, -1), (1, 1))) for s in range(nn)], -1: [jumps(s, ((-1, -1), (-1, 1))) for s in range(nn)]}
    aligned = []
    for k in range(nn):
        kr, kc = divmod(k, n)
        row = []
        for s in range(nn):
            r, c = divmod(s, n)
            row.append(r == kr or c == kc or abs(r - kr) == abs(c - kc))
        aligned.append(tuple(row))
    mvname = [[names[f] + names[t] for t in range(nn)] for f in range(nn)]
    parse = {}
    for f in range(nn):
        for t in range(nn):
            if f != t:
                parse[mvname[f][t]] = (f, t)
    last = {1: tuple(s >= nn - n for s in range(nn)), -1: tuple(s < n for s in range(nn))}
    return {"n": n, "nn": nn, "names": names, "knight": knight, "king": king, "rook": rook, "bishop": bishop,
            "push": push, "pcap": pcap, "aligned": aligned, "mvname": mvname, "parse": parse, "last": last}


_T = {5: _tables(5), 6: _tables(6)}


def _attacked(T, board, sq, s):
    """Is `sq` attacked by the side opposite to sign `s`?"""
    e = -2 * s
    for t in T["knight"][sq]:
        if board[t] == e:
            return True
    e = -s
    for t in T["pcap"][s][sq]:
        if board[t] == e:
            return True
    e = -6 * s
    for t in T["king"][sq]:
        if board[t] == e:
            return True
    q = -5 * s
    e = -4 * s
    for ray in T["rook"][sq]:
        for t in ray:
            p = board[t]
            if p:
                if p == e or p == q:
                    return True
                break
    e = -3 * s
    for ray in T["bishop"][sq]:
        for t in ray:
            p = board[t]
            if p:
                if p == e or p == q:
                    return True
                break
    return False


def _gen(T, board, s):
    """(legal move strings of the side with sign s, in_check). `board` is restored before returning."""
    k6 = 6 * s
    try:
        ksq = board.index(k6)
    except ValueError:
        return [], False
    in_check = _attacked(T, board, ksq, s)
    al = T["aligned"][ksq]
    mvname = T["mvname"]
    out = []
    for f in range(T["nn"]):
        p = board[f]
        if not p or (p > 0) != (s > 0):
            continue
        kind = p * s
        if kind == 1:
            tg = []
            t = T["push"][s][f]
            if t >= 0 and not board[t]:
                tg.append(t)
            for t in T["pcap"][s][f]:
                q = board[t]
                if q and (q > 0) != (s > 0):
                    tg.append(t)
        elif kind == 2 or kind == 6:
            tg = []
            for t in (T["knight"][f] if kind == 2 else T["king"][f]):
                q = board[t]
                if not q or (q > 0) != (s > 0):
                    tg.append(t)
        else:
            tg = []
            if kind != 3:
                for ray in T["rook"][f]:
                    for t in ray:
                        q = board[t]
                        if not q:
                            tg.append(t)
                        else:
                            if (q > 0) != (s > 0):
                                tg.append(t)
                            break
            if kind != 4:
                for ray in T["bishop"][f]:
                    for t in ray:
                        q = board[t]
                        if not q:
                            tg.append(t)
                        else:
                            if (q > 0) != (s > 0):
                                tg.append(t)
                            break
        if not tg:
            continue
        names = mvname[f]
        if kind == 6:
            board[f] = 0
            for t in tg:
                q = board[t]
                board[t] = p
                if not _attacked(T, board, t, s):
                    out.append(names[t])
                board[t] = q
            board[f] = p
        elif in_check or al[f]:
            board[f] = 0
            for t in tg:
                q = board[t]
                board[t] = p
                if not _attacked(T, board, ksq, s):
                    out.append(names[t])
                board[t] = q
            board[f] = p
        else:
            for t in tg:
                out.append(names[t])
    return out, in_check


def _finish(state, mover):
    """Fill legal / over / win of a state whose board, ply, mat and np are set. `mover` = seat that just moved."""
    T = _T[state["size"]]
    ply = state["ply"]
    legal, in_check = _gen(T, state["board"], -1 if ply & 1 else 1)
    if not legal:
        state["legal"] = []
        state["over"] = True
        state["win"] = mover if in_check else None
    elif state["np"] <= 2:
        state["legal"] = []
        state["over"] = True
        state["win"] = None
    elif ply >= state["max_plies"]:
        state["legal"] = []
        state["over"] = True
        m0, m1 = state["mat0"], state["mat1"]
        state["win"] = 0 if m0 > m1 else (1 if m1 > m0 else None)
    else:
        state["legal"] = legal
        state["over"] = False
        state["win"] = None
    return state


def _counts(board):
    m0 = m1 = np_ = 0
    for p in board:
        if p > 0:
            m0 += _VAL[p]
            np_ += 1
        elif p < 0:
            m1 += _VAL[-p]
            np_ += 1
    return m0, m1, np_


def _make(size, board, ply, max_plies, hist="", cap0="", cap1="", last=None):
    m0, m1, np_ = _counts(board)
    st = {"size": size, "board": board, "ply": ply, "max_plies": max_plies, "mat0": m0, "mat1": m1, "np": np_,
          "hist": hist, "cap0": cap0, "cap1": cap1, "last": last, "legal": [], "over": False, "win": None}
    return _finish(st, 1 - (ply & 1))


def _cfg(config):
    config = config or {}
    size = int(config.get("size", 6))
    setup = config.get("setup") or ("gardner" if size == 5 else "los_alamos")
    return size, setup, 2 * int(config.get("max_moves", 40))


def initial_state(config, seed):
    size, setup, max_plies = _cfg(config)
    back = _SETUPS.get(setup, _SETUPS["gardner" if size == 5 else "los_alamos"])
    board = [0] * (size * size)
    for c in range(size):
        k = _CODE[back[c]]
        board[c] = k
        board[size + c] = 1
        board[(size - 1) * size + c] = -k
        board[(size - 2) * size + c] = -1
    return _make(size, board, 0, max_plies)


def to_move(state):
    if state["over"]:
        return []
    return [state["ply"] & 1]


def legal_actions(state, seat):
    if state["over"] or (state["ply"] & 1) != seat:
        return []
    return list(state["legal"])


def _norm(state, action):
    """Canonical move string for an action as the arena would accept it, or None."""
    try:
        legal = state["legal"]
        if action in legal:
            return action
        a = "".join(str(action).lower().replace("-", " ").split())
        if a in legal:
            return a
        T = _T[state["size"]]
        board = state["board"]
        if len(a) == 5 and a[4] == "q" and a[:4] in legal:
            f, t = T["parse"][a[:4]]
            if abs(board[f]) == 1 and (T["last"][1][t] or T["last"][-1][t]):
                return a[:4]
        if len(a) in (5, 6) and a[0] in "kqrbnp":
            rest = a[1:]
            if len(rest) == 5:
                if rest[4] != "q" or rest[:4] not in legal:
                    return None
                f, t = T["parse"][rest[:4]]
                if abs(board[f]) != 1 or not (T["last"][1][t] or T["last"][-1][t]):
                    return None
                rest = rest[:4]
            if rest in legal:
                f, t = T["parse"][rest]
                if abs(board[f]) == _CODE[a[0].upper()]:
                    return rest
    except Exception:
        return None
    return None


def is_legal_action(state, seat, action):
    try:
        if state["over"] or (state["ply"] & 1) != seat:
            return False
        return _norm(state, action) is not None
    except Exception:
        return False


def apply(state, actions, rng):
    if state["over"]:
        return dict(state)
    seat = state["ply"] & 1
    a = actions.get(seat)
    if a is None:
        a = actions.get(str(seat))
    if a not in state["legal"]:
        a = _norm(state, a)
        if a is None:
            raise ValueError("illegal move %r" % (actions,))
    T = _T[state["size"]]
    f, t = T["parse"][a]
    new = dict(state)
    board = new["board"] = state["board"][:]
    p = board[f]
    q = board[t]
    board[f] = 0
    promo = False
    if q:
        new["np"] = state["np"] - 1
        if seat:
            new["mat0"] = state["mat0"] - _VAL[q]
            new["cap1"] = state["cap1"] + _CHARS[q]
        else:
            new["mat1"] = state["mat1"] - _VAL[-q]
            new["cap0"] = state["cap0"] + _CHARS[q]
    if p == 1 and T["last"][1][t]:
        p = 5
        promo = True
        new["mat0"] = new["mat0"] + 8
    elif p == -1 and T["last"][-1][t]:
        p = -5
        promo = True
        new["mat1"] = new["mat1"] + 8
    board[t] = p
    new["ply"] = state["ply"] + 1
    h = state["hist"]
    new["hist"] = h + " " + a if h else a
    new["last"] = (a, abs(state["board"][f]), abs(q), promo, seat)
    return _finish(new, seat)


def is_over(state):
    return state["over"]


def winner(state):
    return state["win"] if state["over"] else None


def random_action(state, seat, rng):
    if state["over"] or (state["ply"] & 1) != seat or not state["legal"]:
        return ""
    return rng.choice(state["legal"])


def could_win_now(state, seat, action):
    return False


def _label(size):
    for lab, c in CONFIGS.items():
        if c["size"] == size:
            return lab
    return "%dx%d" % (size, size)


def observation(state, seat):
    n = state["size"]
    board = state["board"]
    ply = state["ply"]
    side = ply & 1
    files = "    " + " ".join(_FILES[:n])
    lines = ["MINI CHESS (%s). You are Player %d and play %s." % (
        _label(n), seat, "WHITE (uppercase, moving up toward rank %d)" % n if seat == 0
        else "BLACK (lowercase, moving down toward rank 1)"),
        "Move %d, %s to move. Moves played: %d of %d (cap)." % (
            ply // 2 + 1, "Black" if side else "White", ply, state["max_plies"]),
        "", files]
    for r in range(n - 1, -1, -1):
        lines.append(" %d  %s  %d" % (r + 1, " ".join(_CHARS[board[r * n + c]] for c in range(n)), r + 1))
    lines += [files, "",
              "Material: White %d, Black %d  (Q=9 R=5 B=3 N=3 P=1)" % (state["mat0"], state["mat1"]),
              "Captured by White: %s   Captured by Black: %s" % (
                  " ".join(state["cap0"]) or "(none)", " ".join(state["cap1"]) or "(none)")]
    hist = state["hist"].split()
    if hist:
        parts = []
        for i in range(0, len(hist), 2):
            parts.append("%d. %s" % (i // 2 + 1, " ".join(hist[i:i + 2])))
        lines.append("Moves so far: " + "  ".join(parts))
    else:
        lines.append("No moves yet.")
    last = state.get("last")
    if last:
        mv, piece, cap, promo, who = last
        d = "%s %s->%s" % (_NAME[piece], mv[:2], mv[2:4])
        if cap:
            d += " capturing " + _NAME[cap]
        if promo:
            d += " and promoting to a queen"
        lines.append("Last move: %s played %s (%s)." % ("Opponent" if who != seat else "You", mv, d))
    if state["over"]:
        w = state["win"]
        lines.append("Game over: %s." % ("draw" if w is None else ("you win" if w == seat else "you lose")))
        return "\n".join(lines)
    T = _T[n]
    s = -1 if seat else 1
    try:
        chk = _attacked(T, board, board.index(6 * s), s)
    except ValueError:
        chk = False
    lines.append("You are IN CHECK: you must get out of check." if chk else "You are not in check.")
    if side == seat:
        lines.append('It is YOUR turn. Answer with a move like "b2b3" (from-square then to-square). '
                     "You have %d legal moves." % len(state["legal"]))
    else:
        lines.append("Waiting for the opponent's move.")
    return "\n".join(lines)


_ROW_RE = re.compile(r"^\s*(\d)\s+((?:[.A-Za-z]\s+){4,6})(\d)\s*$")
_PIECE_WORD = {v: k for k, v in enumerate(_NAME) if v}


def state_from_observation(text, seat, config, rng):
    size, setup, max_plies = _cfg(config)
    rows = {}
    for line in text.split("\n"):
        m = _ROW_RE.match(line)
        if m and m.group(1) == m.group(3):
            rows[int(m.group(1))] = m.group(2).split()
    if rows:
        size = max(max(rows), max(len(v) for v in rows.values()))
    board = [0] * (size * size)
    for r, cells in rows.items():
        for c, ch in enumerate(cells[:size]):
            if ch != ".":
                k = _CODE.get(ch.upper(), 0)
                board[(r - 1) * size + c] = k if ch.isupper() else -k
    hist = ""
    m = re.search(r"Moves so far:\s*(.*)", text)
    if m:
        hist = " ".join(re.findall(r"[a-f][1-6][a-f][1-6]", m.group(1)))
    nh = len(hist.split())
    m = re.search(r"Moves played:\s*(\d+)\s*of\s*(\d+)", text)
    if m:
        ply, max_plies = int(m.group(1)), int(m.group(2))
    else:
        ply = nh
    m = re.search(r"(White|Black)\s+to\s+move", text, re.I)
    if m:
        side = 0 if m.group(1).lower() == "white" else 1
    elif re.search(r"YOUR\s+turn", text):
        side = seat
    else:
        side = ply & 1
    if (ply & 1) != side:
        ply += 1
    cap0 = cap1 = ""
    m = re.search(r"Captured by White:\s*(.*?)\s*Captured by Black:\s*(.*)", text)
    if m:
        if "none" not in m.group(1):
            cap0 = "".join(ch for ch in m.group(1) if ch.isalpha())
        if "none" not in m.group(2):
            cap1 = "".join(ch for ch in m.group(2) if ch.isalpha())
    last = None
    m = re.search(r"Last move:\s*(\w+)\s+played\s+([a-f][1-6][a-f][1-6])\s*\((\w+)[^)]*?(?:capturing\s+(\w+))?"
                  r"(\s+and promoting[^)]*)?\)", text)
    if m:
        who = seat if m.group(1).lower() == "you" else 1 - seat
        last = (m.group(2), _PIECE_WORD.get(m.group(3), 0), _PIECE_WORD.get(m.group(4) or "", 0),
                bool(m.group(5)), who)
    return _make(size, board, ply, max_plies, hist, cap0, cap1, last)


def _order(state):
    """Captures first (most valuable victim), then the rest: better alpha-beta cuts."""
    size, board = state["size"], state["board"]
    def key(a):
        to = (int(a[3]) - 1) * size + (ord(a[2]) - 97)
        return -_VAL[abs(board[to])]
    return sorted(state["legal"], key=key)


def _eval(state, seat):
    """Material balance for `seat` (the arena's own count decides the game at the move cap), plus a little mobility."""
    if state["over"]:
        w = state["win"]
        return 0.0 if w is None else (1000.0 if w == seat else -1000.0)
    mat = state["mat0"] - state["mat1"]
    mob = len(state["legal"]) * 0.02
    side = state["ply"] & 1
    return (mat if seat == 0 else -mat) + (mob if side == seat else -mob)


def _search(state, depth, alpha, beta, seat, deadline, clock):
    if state["over"] or depth <= 0:
        return _eval(state, seat)
    if clock() > deadline:
        raise TimeoutError
    side = state["ply"] & 1
    best = -1e9 if side == seat else 1e9
    for a in _order(state):
        v = _search(apply(state, {side: a}, None), depth - 1, alpha, beta, seat, deadline, clock)
        if abs(v) > 900:
            v -= 1 if v > 0 else -1  # prefer the faster win, the slower loss
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


def recommend(text, seat, config, rng, seconds=1.5):
    """Optional hook of a known simulator (mesh/simworker.candidates): [(action, value)] best first. Iterative-deepening
    alpha-beta on the exact rules with a material evaluation, as deep as the time allows (random playouts are a poor
    judge of chess positions: local tests lost to a 2-ply material search). Values are scaled into -1..1."""
    import time as _time
    st = state_from_observation(text, seat, config, rng)
    if st["over"] or (st["ply"] & 1) != seat or not st["legal"]:
        return []
    clock = _time.monotonic
    deadline = clock() + max(0.2, seconds * 0.85)
    scores = {a: 0.0 for a in st["legal"]}
    order = _order(st)
    depth = 1
    while depth <= 8:
        cur, alpha = {}, -1e9
        try:
            for a in order:
                v = _search(apply(st, {seat: a}, None), depth - 1, alpha, 1e9, seat, deadline, clock)
                cur[a] = v
                if v > alpha:
                    alpha = v
        except TimeoutError:
            if len(cur) >= 1 and max(cur.values()) > max(scores.values()) and depth > 1:
                scores.update(cur)  # a better move already found at the deeper level
            break
        scores = cur
        order = sorted(order, key=lambda a: -scores[a])
        if max(scores.values()) > 900 or clock() > deadline:
            break
        depth += 1
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(a, max(-1.0, min(1.0, v / 20.0))) for a, v in ranked[:6]]

"""Exact hand-written simulator of the arena game `battleship` (contract: mesh/simgen.py GEN_SPEC).

Rules that matter (from the arena's rules text, all three variants):
  - square grid, cell = column letter + row number ("a1" top-left); ships are straight lines, never overlap,
    MAY touch; fleets: 6x6 = Cruiser 3, Destroyer 2, Patrol boat 2; 8x8 = Battleship 4, Cruiser 3, Submarine 3,
    Destroyer 2;
  - Player 0 fires, then Player 1 = one round; a hit gives NO extra shot; normal variants: 1 shot per turn;
    salvo: as many shots as OWN ships afloat (at least 1), results reported after the whole salvo;
  - the game is checked only at the END of a round: one fleet fully sunk -> the other wins, both -> draw;
    after the last round (30 / 44 / 24) the player with more hits wins, equal = draw.

State: flat dict, cells are bits of an int (index = row * n + col).
  ships[s]  = list of masks of seat s's ships (fleet order), occ[s] = their union,
  shots[s]  = mask of the cells seat s has fired at (on the opponent's grid), hits[s] = enemy ship cells s hit.

state_from_observation samples the hidden enemy fleet UNIFORMLY among the fleets consistent with the text:
sunk ships tile the '#' cells exactly (pinned by "sunk <Ship>" history lines), afloat ships cover every open 'X',
avoid every 'o' and '#', and are not fully hit.
"""
import re
from bisect import bisect_right
from random import Random

FLEETS = {
    "small": (("Cruiser", "C", 3), ("Destroyer", "D", 2), ("Patrol boat", "P", 2)),
    "classic": (("Battleship", "B", 4), ("Cruiser", "C", 3), ("Submarine", "S", 3), ("Destroyer", "D", 2)),
}

CONFIGS = {
    "6x6, 3 ships": {"label": "6x6, 3 ships", "size": 6, "fleet": "small", "salvo": False, "max_rounds": 30},
    "8x8, 4 ships": {"label": "8x8, 4 ships", "size": 8, "fleet": "classic", "salvo": False, "max_rounds": 44},
    "8x8 salvo": {"label": "8x8 salvo", "size": 8, "fleet": "classic", "salvo": True, "max_rounds": 24},
}

_COLS = "abcdefghijklmnopqrstuvwxyz"
_PLACE = {}      # (n, length) -> tuple of placement masks
_NAMES = {}      # n -> tuple of cell names
_CELLIDX = {}    # n -> {name: index}
_OBS_CACHE = {}  # (text, seat, n, fleet, salvo, max_rounds) -> parsed observation
_LA_CACHE = {}   # salvo candidate lists


def _placements(n, length):
    key = (n, length)
    p = _PLACE.get(key)
    if p is None:
        out = []
        for r in range(n):
            for c in range(n - length + 1):
                m = 0
                for k in range(length):
                    m |= 1 << (r * n + c + k)
                out.append(m)
        for c in range(n):
            for r in range(n - length + 1):
                m = 0
                for k in range(length):
                    m |= 1 << ((r + k) * n + c)
                out.append(m)
        p = _PLACE[key] = tuple(out)
    return p


def _names(n):
    t = _NAMES.get(n)
    if t is None:
        t = _NAMES[n] = tuple(_COLS[i % n] + str(i // n + 1) for i in range(n * n))
        _CELLIDX[n] = {name: i for i, name in enumerate(t)}
    return t


def _cellidx(n):
    if n not in _CELLIDX:
        _names(n)
    return _CELLIDX[n]


def _bits(m):
    out = []
    while m:
        b = m & -m
        out.append(b.bit_length() - 1)
        m ^= b
    return out


def _popcount(m):
    return bin(m).count("1")


def _fleet_of(cfg):
    f = cfg.get("fleet", "classic")
    if isinstance(f, str):
        return FLEETS.get(f, FLEETS["classic"])
    return tuple(tuple(x) for x in f)


def _random_fleet(n, sizes, rng, forbid=0):
    """Uniformly random fleet (rejection); cells of `forbid` are avoided when possible."""
    for attempt in range(400):
        fb = forbid if attempt < 300 else 0
        used, out = fb, []
        for ln in sizes:
            pl = _placements(n, ln)
            p = pl[int(rng.random() * len(pl))]
            if p & used:
                out = None
                break
            used |= p
            out.append(p)
        if out is not None:
            return out
    used, out = 0, []
    for ln in sizes:  # deterministic last resort
        for p in _placements(n, ln):
            if not p & used:
                used |= p
                out.append(p)
                break
    return out


def initial_state(config, seed):
    rng = Random(seed)
    n = int(config.get("size", 8))
    fk = config.get("fleet", "classic")
    sizes = [s[2] for s in _fleet_of(config)]
    ships = [_random_fleet(n, sizes, rng), _random_fleet(n, sizes, rng)]
    occ = [0, 0]
    for s in (0, 1):
        for m in ships[s]:
            occ[s] |= m
    return {"n": n, "fleet": fk, "salvo": bool(config.get("salvo", False)),
            "max_rounds": int(config.get("max_rounds", 44)), "round": 1, "turn": 0, "over": False, "winner": None,
            "ships": ships, "occ": occ, "shots": [0, 0], "hits": [0, 0], "log": []}


def to_move(state):
    return [] if state["over"] else [state["turn"]]


def is_over(state):
    return state["over"]


def winner(state):
    return state["winner"] if state["over"] else None


def _nshots(state, seat):
    """Shots `seat` must fire this turn."""
    n = state["n"]
    free = n * n - _popcount(state["shots"][seat])
    k = 1
    if state["salvo"]:
        inc = state["shots"][1 - seat]
        k = 0
        for m in state["ships"][seat]:
            if m & ~inc:
                k += 1
        if k < 1:
            k = 1
    return k if k < free else free


def _unfired(state, seat):
    n = state["n"]
    fired = state["shots"][seat]
    return [i for i in range(n * n) if not (fired >> i) & 1]


def _salvo_candidates(state, seat, k):
    """Deterministic list (<= 300) of k-cell salvos built from PUBLIC information only: cells ranked by how many
    placements of the enemy ships still afloat go through them (placements through open hits weigh much more)."""
    n = state["n"]
    fired = state["shots"][seat]
    opp = 1 - seat
    occ = state["occ"][opp]
    sunk = 0
    lens = []
    fl = _fleet_of(state)
    for j, m in enumerate(state["ships"][opp]):
        if m & ~fired:
            lens.append(fl[j][2] if j < len(fl) else _popcount(m))
        else:
            sunk |= m
    hitm = fired & occ & ~sunk
    key = (n, fired, hitm, sunk, tuple(lens), k)
    got = _LA_CACHE.get(key)
    if got is not None:
        return got
    blocked = (fired & ~occ) | sunk
    score = [0] * (n * n)
    for ln in lens:
        for p in _placements(n, ln):
            if p & blocked:
                continue
            rest = p & ~hitm
            if not rest:
                continue
            w = 1
            h = _popcount(p & hitm)
            if h:
                w = 40 * h * h
            for i in _bits(rest):
                score[i] += w
    cells = [i for i in range(n * n) if not (fired >> i) & 1]
    ranked = sorted(cells, key=lambda i: (-score[i], i))
    names = _names(n)
    top = {4: 10, 3: 11, 2: 20}.get(k, 8)
    top = ranked[:top]
    out, seen = [], set()

    def add(combo):
        c = tuple(sorted(combo))
        if len(c) == k and c not in seen and len(out) < 300:
            seen.add(c)
            out.append("fire " + " ".join(names[i] for i in c))

    def rec(start, cur):
        if len(cur) == k:
            add(cur)
            return
        for j in range(start, len(top)):
            rec(j + 1, cur + [top[j]])

    if k <= 4:
        rec(0, [])
    for i in range(0, len(ranked) - k + 1):
        add(ranked[i:i + k])
    step = max(1, len(cells) // k)
    for off in range(step):
        add([cells[(off + j * step) % len(cells)] for j in range(k)])
    if len(_LA_CACHE) > 512:
        _LA_CACHE.clear()
    _LA_CACHE[key] = out
    return out


def legal_actions(state, seat):
    if state["over"] or seat != state["turn"]:
        return []
    k = _nshots(state, seat)
    if k <= 1:
        names = _names(state["n"])
        return ["fire " + names[i] for i in _unfired(state, seat)]
    return list(_salvo_candidates(state, seat, k))


def random_action(state, seat, rng):
    cells = _unfired(state, seat)
    names = _names(state["n"])
    k = _nshots(state, seat)
    if not cells:
        return "fire a1"
    if k <= 1:
        return "fire " + names[cells[int(rng.random() * len(cells))]]
    pick = sorted(rng.sample(cells, k))
    return "fire " + " ".join(names[i] for i in pick)


def _parse_action(action, n):
    try:
        toks = str(action).lower().replace(",", " ").split()
    except Exception:
        return None
    if len(toks) < 2 or toks[0] != "fire":
        return None
    idx = _cellidx(n)
    out = []
    for t in toks[1:]:
        i = idx.get(t)
        if i is None:
            return None
        out.append(i)
    return out


def is_legal_action(state, seat, action):
    try:
        if state["over"] or seat != state["turn"]:
            return False
        cells = _parse_action(action, state["n"])
        if not cells or len(set(cells)) != len(cells) or len(cells) != _nshots(state, seat):
            return False
        fired = state["shots"][seat]
        for i in cells:
            if (fired >> i) & 1:
                return False
        return True
    except Exception:
        return False


def could_win_now(state, seat, action):
    return False


def apply(state, actions, rng):
    st = dict(state)
    if state["over"]:
        return st
    seat = state["turn"]
    opp = 1 - seat
    a = actions.get(seat)
    if a is None:
        a = actions.get(str(seat))
    cells = _parse_action(a, state["n"]) or []
    shots = list(state["shots"])
    hits = list(state["hits"])
    fired = shots[seat]
    occ = state["occ"][opp]
    res = []
    for i in cells:
        b = 1 << i
        if fired & b:
            continue
        fired |= b
        code = 0
        if occ & b:
            hits[seat] += 1
            code = 1
            ships = state["ships"][opp]
            for j in range(len(ships)):
                m = ships[j]
                if m & b:
                    if not m & ~fired:
                        code = 2 + j
                    break
        res.append((i, code))
    shots[seat] = fired
    st["shots"] = shots
    st["hits"] = hits
    log = state["log"][-11:]
    log.append((state["round"], seat, tuple(res)))
    st["log"] = log
    if seat == 0:
        st["turn"] = 1
        return st
    occ0, occ1 = state["occ"]
    w0 = not occ1 & ~shots[0]   # seat 0 sank everything
    w1 = not occ0 & ~shots[1]
    if w0 or w1:
        st["over"] = True
        st["winner"] = None if (w0 and w1) else (0 if w0 else 1)
    elif state["round"] >= state["max_rounds"]:
        st["over"] = True
        st["winner"] = 0 if hits[0] > hits[1] else (1 if hits[1] > hits[0] else None)
    else:
        st["round"] = state["round"] + 1
    st["turn"] = 0
    return st


def _fleet_line(fl, ships, incoming):
    return ", ".join("%s(%d) %s" % (fl[j][0], fl[j][2], "afloat" if ships[j] & ~incoming else "SUNK")
                     for j in range(len(ships)))


def observation(state, seat):
    n = state["n"]
    fl = _fleet_of(state)
    names = _names(n)
    opp = 1 - seat
    mine, theirs = state["shots"][seat], state["shots"][opp]
    head = "%dx%d, %d ships%s" % (n, n, len(fl), ", salvo" if state["salvo"] else "")
    L = ["BATTLESHIP (%s). You are Player %d." % (head, seat),
         "Round %d of %d." % (state["round"], state["max_rounds"])]
    k = _nshots(state, seat)
    if state["over"]:
        L.append("The game is over.")
    elif state["turn"] == seat:
        L.append("It is YOUR turn: fire exactly %d shot%s." % (k, "" if k == 1 else "s"))
    else:
        L.append("It is the opponent's turn.")
    L.append("Your fleet:      " + _fleet_line(fl, state["ships"][seat], theirs))
    L.append("Opponent fleet:  " + _fleet_line(fl, state["ships"][opp], mine))
    L.append("Enemy ship cells you hit: %d   Your ship cells they hit: %d" % (state["hits"][seat], state["hits"][opp]))
    L.append("")
    cols = "    " + " ".join(_COLS[:n])
    sunk = 0
    for m in state["ships"][opp]:
        if not m & ~mine:
            sunk |= m
    occ_o = state["occ"][opp]
    L.append("YOUR SHOTS (opponent's waters):  . not fired, o miss, X hit, # hit on a sunk ship")
    L.append(cols)
    for r in range(n):
        row = []
        for c in range(n):
            b = 1 << (r * n + c)
            row.append("." if not mine & b else ("#" if sunk & b else ("X" if occ_o & b else "o")))
        L.append("%2d  %s" % (r + 1, " ".join(row)))
    L.append("")
    L.append("YOUR WATERS (your fleet):  letter = your ship, * = your ship hit, o = their miss, . = water")
    L.append(cols)
    letter = {}
    for j, m in enumerate(state["ships"][seat]):
        for i in _bits(m):
            letter[i] = fl[j][1]
    for r in range(n):
        row = []
        for c in range(n):
            i = r * n + c
            hit = (theirs >> i) & 1
            if i in letter:
                row.append("*" if hit else letter[i])
            else:
                row.append("o" if hit else ".")
        L.append("%2d  %s" % (r + 1, " ".join(row)))
    L.append("")
    if state["log"]:
        L.append("Recent shots (oldest first):")
        for rnd, s, res in state["log"][-12:]:
            parts = []
            for i, code in res:
                if code == 0:
                    parts.append(names[i] + " miss")
                elif code == 1:
                    parts.append(names[i] + " hit")
                else:
                    sh = fl[code - 2]
                    parts.append("%s hit, sunk %s (%d)" % (names[i], sh[0], sh[2]))
            L.append("  Round %d %s: %s" % (rnd, "You" if s == seat else "Opponent", "; ".join(parts)))
    else:
        L.append("No shots fired yet.")
    L.append("")
    if not state["over"] and state["turn"] == seat:
        free = _unfired(state, seat)
        if k == 1:
            L.append('Your move: "fire <cell>" at one of the %d cells marked "." in YOUR SHOTS, e.g. "fire %s".'
                     % (len(free), names[free[0]] if free else "a1"))
        else:
            L.append('Your move: "fire <cell> <cell> ..." with exactly %d different cells marked "." in YOUR SHOTS.' % k)
    return "\n".join(L)


# ---------------------------------------------------------------- parsing + sampling of the hidden fleet

_RE_ROW = re.compile(r"^\s*(\d+)\s+(\S(?:\s+\S)*)\s*$")
_RE_SHIP = re.compile(r"([A-Za-z][A-Za-z ]*?)\s*\((\d+)\)\s*(afloat|SUNK)", re.I)
_RE_HIST = re.compile(r"^\s*Round\s+(\d+)\s+(You|Opponent)\s*:\s*(.*)$", re.I)
_RE_SHOT = re.compile(r"([a-z]\d+)\s+(miss|hit)(?:\s*,\s*sunk\s+([A-Za-z][A-Za-z ]*?)\s*\(\s*(\d+)\s*\))?", re.I)


def _grid(lines, marker, n):
    """Rows of the grid that follows the line containing `marker`: {row index: [symbols]}."""
    rows = {}
    start = None
    for i, ln in enumerate(lines):
        if marker in ln:
            start = i + 1
            break
    if start is None:
        return rows
    for ln in lines[start:]:
        m = _RE_ROW.match(ln)
        if m:
            r = int(m.group(1)) - 1
            toks = m.group(2).split()
            if 0 <= r < n and len(toks) >= n:
                rows[r] = toks[:n]
            if len(rows) >= n:
                break
        elif rows:
            break
    return rows


def _tilings(cands, cover, limit=200):
    """All ways to pick one mask per ship (cands = list of lists), pairwise disjoint, whose union contains `cover`."""
    out = []
    k = len(cands)

    def rec(j, used, cur):
        if len(out) >= limit:
            return
        if j == k:
            if not cover & ~used:
                out.append(tuple(cur))
            return
        for p in cands[j]:
            if not p & used:
                cur.append(p)
                rec(j + 1, used | p, cur)
                cur.pop()

    rec(0, 0, [])
    return out


class _Overflow(Exception):
    pass


def _covers(xmask, afloat, cover_idx, limit=60000):
    """Every way to cover all open hits: list of (chosen ((ship, mask), ...), used mask, remaining ships)."""
    out = []

    def rec(unc, used, chosen, rem):
        if not unc:
            if len(out) >= limit:
                raise _Overflow()
            out.append((chosen, used, rem))
            return
        b = unc & -unc
        for s in rem:
            lst = cover_idx[s].get(b)
            if lst:
                rest = tuple(x for x in rem if x != s)
                for p in lst:
                    if not p & used:
                        rec(unc & ~p, used | p, chosen + ((s, p),), rest)

    rec(xmask, 0, (), tuple(afloat))
    return out


def _parse(text, seat, config):
    n = int(config.get("size", 8))
    fl = _fleet_of(config)
    nsh = len(fl)
    lines = text.splitlines()
    idx = _cellidx(n)
    name_to_j = {s[0].lower(): j for j, s in enumerate(fl)}
    m = re.search(r"You are Player\s*(\d)", text)
    if m and seat not in (0, 1):
        seat = int(m.group(1))
    opp = 1 - seat
    m = re.search(r"Round\s+(\d+)\s+of\s+(\d+)", text)
    rnd = int(m.group(1)) if m else 1
    max_rounds = int(config.get("max_rounds", int(m.group(2)) if m else 44))
    my_turn = ("YOUR turn" in text) or ("Your move" in text) or ("turn" not in text.lower())
    over = "game is over" in text.lower()

    # --- grids
    g1 = _grid(lines, "YOUR SHOTS", n)
    g2 = _grid(lines, "YOUR WATERS", n)
    miss = xm = hm = 0
    for r, toks in g1.items():
        for c in range(n):
            t = toks[c]
            b = 1 << (r * n + c)
            if t == "o" or t == "O":
                miss |= b
            elif t == "X" or t == "x":
                xm |= b
            elif t == "#":
                hm |= b
    stars = omiss = 0
    letters = [0] * nsh
    let_to_j = {s[1]: j for j, s in enumerate(fl)}
    for r, toks in g2.items():
        for c in range(n):
            t = toks[c]
            b = 1 << (r * n + c)
            if t == "*":
                stars |= b
            elif t == "o":
                omiss |= b
            elif t in let_to_j:
                letters[let_to_j[t]] |= b

    # --- fleet status lines
    def status(prefix):
        mm = re.search(prefix + r"\s*:\s*(.*)", text)
        res = {}
        if mm:
            for nm, _sz, stt in _RE_SHIP.findall(mm.group(1)):
                j = name_to_j.get(nm.strip().lower())
                if j is not None:
                    res[j] = stt.upper() == "SUNK"
        return res

    my_sunk = status("Your fleet")
    op_sunk = status("Opponent fleet")

    # --- history
    log = []
    pins_me, pins_op = {}, {}   # ship index -> cell bit that must belong to it (the sinking shot)
    for ln in lines:
        mm = _RE_HIST.match(ln)
        if not mm:
            continue
        who = seat if mm.group(2).lower() == "you" else opp
        res = []
        for cell, kind, nm, _sz in _RE_SHOT.findall(mm.group(3)):
            i = idx.get(cell.lower())
            if i is None:
                continue
            code = 0 if kind.lower() == "miss" else 1
            if nm:
                j = name_to_j.get(nm.strip().lower())
                if j is not None:
                    code = 2 + j
                    (pins_op if who == seat else pins_me)[j] = 1 << i
            res.append((i, code))
        log.append((int(mm.group(1)), who, tuple(res)))
    log = log[-12:]

    # --- own fleet (exact): intact letters + '*' cells
    cands = []
    for j in range(nsh):
        lm = letters[j]
        sunk = my_sunk.get(j, lm == 0)
        pin = pins_me.get(j, 0)
        lst = []
        for p in _placements(n, fl[j][2]):
            if lm & ~p or (p & ~lm) & ~stars or (pin and not p & pin):
                continue
            if sunk and lm:
                continue
            lst.append(p)
        cands.append(lst)
    own = _tilings(cands, stars, 60)
    if not own:   # odd text: drop the pins, then the exact cover
        cands = [[p for p in _placements(n, fl[j][2]) if not letters[j] & ~p and not (p & ~letters[j]) & ~stars]
                 for j in range(nsh)]
        own = _tilings(cands, stars, 20) or _tilings(cands, 0, 20)
    if not own:
        own = [tuple(_random_fleet(n, [s[2] for s in fl], Random(0), omiss))]

    # --- enemy fleet: sunk ships tile '#', afloat ships cover 'X'
    sunk_ships = [j for j in range(nsh) if op_sunk.get(j, False)]
    afloat = [j for j in range(nsh) if j not in sunk_ships]
    scands = []
    for j in sunk_ships:
        pin = pins_op.get(j, 0)
        scands.append([p for p in _placements(n, fl[j][2]) if not p & ~hm and (not pin or p & pin)])
    til = _tilings(scands, hm, 400) if sunk_ships else ([()] if not hm else [])
    if not til and sunk_ships:
        scands = [[p for p in _placements(n, fl[j][2]) if not p & ~hm] for j in sunk_ships]
        til = _tilings(scands, hm, 400)
    blocked = miss | hm
    cover_idx, static = {}, {}
    for j in afloat:
        d = {}
        st_l = []
        for p in _placements(n, fl[j][2]):
            if p & blocked:
                continue
            px = p & xm
            if not px:
                st_l.append(p)
            elif p & ~xm:
                for i in _bits(px):
                    d.setdefault(1 << i, []).append(p)
        cover_idx[j] = d
        static[j] = st_l
    covers, cum, total = [], [], 0
    if til:
        try:
            covers = _covers(xm, afloat, cover_idx)
        except _Overflow:
            covers = []
        for _ch, _used, rem in covers:
            w = 1
            for s in rem:
                w *= len(static[s])
            total += w
            cum.append(total)
    return {"n": n, "seat": seat, "round": rnd, "max_rounds": max_rounds, "turn": seat if my_turn else opp,
            "over": over, "own": own, "my_shots": miss | xm | hm, "their_shots": stars | omiss,
            "hits_me": _popcount(xm | hm), "hits_op": _popcount(stars), "log": log,
            "miss": miss, "xm": xm, "hm": hm, "sunk_ships": sunk_ships, "afloat": afloat, "til": til,
            "covers": covers, "cum": cum, "total": total, "static": static, "cover_idx": cover_idx,
            "nsh": nsh, "sizes": [s[2] for s in fl]}


def _sample_enemy(P, rng):
    nsh = P["nsh"]
    rnd = rng.random
    total = P["total"]
    if total > 0:
        til = P["til"]
        covers, cum, static = P["covers"], P["cum"], P["static"]
        ncov = len(covers)
        for _ in range(400):
            if ncov == 1:
                chosen, used, rem = covers[0]
            else:
                chosen, used, rem = covers[bisect_right(cum, rnd() * total)]
            ok = True
            picked = []
            for s in rem:
                lst = static[s]
                p = lst[int(rnd() * len(lst))]
                if p & used:
                    ok = False
                    break
                used |= p
                picked.append(p)
            if ok:
                out = [0] * nsh
                t = til[int(rnd() * len(til))] if len(til) > 1 else til[0]
                for j, p in zip(P["sunk_ships"], t):
                    out[j] = p
                for s, p in chosen:
                    out[s] = p
                for s, p in zip(rem, picked):
                    out[s] = p
                return out
    return _sample_fallback(P, rng)


def _sample_fallback(P, rng):
    """Randomised backtracking (not uniform), then weaker and weaker constraints; always returns a fleet."""
    n, nsh, sizes = P["n"], P["nsh"], P["sizes"]
    miss, xm, hm = P["miss"], P["xm"], P["hm"]
    out = [0] * nsh
    used = 0
    til = P["til"]
    sunk_ships = list(P["sunk_ships"])
    afloat = list(P["afloat"])
    if til:
        for j, p in zip(sunk_ships, til[int(rng.random() * len(til))]):
            out[j] = p
            used |= p
    else:   # the fleet line and the '#' cells disagree: treat every ship as free, '#' as hits to cover
        afloat = list(range(nsh))
        xm |= hm
        hm = 0
    blocked = miss | hm
    for _ in range(60):
        order = afloat[:]
        rng.shuffle(order)
        u, unc, cur, ok = used, xm, {}, True
        for pos, j in enumerate(order):
            pl = [p for p in _placements(n, sizes[j]) if not p & (blocked | u)]
            if unc:
                left = len(order) - pos - 1
                cov = [p for p in pl if p & unc]
                if cov and (left == 0 or rng.random() < 0.8):
                    pl = cov
            if not pl:
                ok = False
                break
            p = pl[int(rng.random() * len(pl))]
            cur[j] = p
            u |= p
            unc &= ~p
        if ok and not unc:
            for j, p in cur.items():
                out[j] = p
            return out
    rest = _random_fleet(n, [sizes[j] for j in afloat], rng, miss | used)
    if any(p & used for p in rest):
        return _random_fleet(n, sizes, rng, miss)
    for j, p in zip(afloat, rest):
        out[j] = p
    return out


def state_from_observation(text, seat, config, rng):
    key = (text, seat, config.get("size"), str(config.get("fleet")), config.get("salvo"), config.get("max_rounds"))
    P = _OBS_CACHE.get(key)
    if P is None:
        P = _parse(text, seat, config)
        if len(_OBS_CACHE) >= 64:
            _OBS_CACHE.clear()
        _OBS_CACHE[key] = P
    seat = P["seat"]
    opp = 1 - seat
    own_l = P["own"]
    own = list(own_l[0] if len(own_l) == 1 else own_l[int(rng.random() * len(own_l))])
    enemy = _sample_enemy(P, rng)
    ships = [None, None]
    ships[seat] = own
    ships[opp] = enemy
    occ = [0, 0]
    for s in (0, 1):
        for m in ships[s]:
            occ[s] |= m
    shots = [0, 0]
    shots[seat] = P["my_shots"]
    shots[opp] = P["their_shots"]
    hits = [0, 0]
    hits[seat] = P["hits_me"]
    hits[opp] = P["hits_op"]
    return {"n": P["n"], "fleet": config.get("fleet", "classic"), "salvo": bool(config.get("salvo", False)),
            "max_rounds": P["max_rounds"], "round": P["round"], "turn": P["turn"], "over": P["over"], "winner": None,
            "ships": ships, "occ": occ, "shots": shots, "hits": hits, "log": list(P["log"])}


def recommend(text, seat, config, rng, seconds=1.0):
    """Optional hook of a known simulator (mesh/simworker.candidates): [(action, value)] best first, instead of a
    playout search. Fire where most fleets consistent with everything we saw have a ship: sample the hidden fleet
    many times (state_from_observation is uniform over consistent fleets) and count, per cell not fired at yet, how
    often a ship is there. That gives target mode after a hit and parity hunting before, with no special cases.
    Round robin of 15 Nov: 0 of 4, we never fired next to our own hits (random playouts cannot rank 64 cells)."""
    import time as _time
    st = state_from_observation(text, seat, config, rng)
    if st["over"] or st["turn"] != seat:
        return []
    n, opp, fired = st["n"], 1 - seat, st["shots"][seat]
    counts = [0] * (n * n)
    t_end = _time.monotonic() + max(0.05, seconds * 0.8)
    samples = 0
    while samples < 600 and (samples < 60 or _time.monotonic() < t_end):
        occ = state_from_observation(text, seat, config, rng)["occ"][opp] & ~fired
        i = 0
        while occ:
            if occ & 1:
                counts[i] += 1
            occ >>= 1
            i += 1
        samples += 1
    names = _names(n)
    cells = sorted((i for i in range(n * n) if not (fired >> i) & 1), key=lambda i: (-counts[i], i))
    if not cells:
        return []
    k = _nshots(st, seat)
    if k <= 1:
        return [("fire " + names[i], counts[i] / samples) for i in cells[:6]]
    out = []
    for skip in range(min(3, max(1, len(cells) - k + 1))):  # the k best cells, then two near-best alternatives
        pick = cells[:k - 1] + cells[k - 1 + skip:k + skip] if skip else cells[:k]
        if len(pick) == k:
            out.append(("fire " + " ".join(names[i] for i in sorted(pick)), sum(counts[i] for i in pick) / (samples * k)))
    return out

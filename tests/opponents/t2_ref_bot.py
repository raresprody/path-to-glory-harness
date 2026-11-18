"""A solid rule-based opponent for our local copies of the T2 games (tests/t2games), zero spend: a test bot, not ours.

    python tests/sweep.py --bot IMAGE --only t2_ --opponent "python3 ../../../opponents/t2_ref_bot.py"

Plays roughly like a decent T2 team would, so a win against it means something (a random opponent hands out free wins):
  Nim: perfect play (nim-sum, misere, take-1-to-k).          Rock Paper Scissors: counters our most frequent throw.
  Prisoner's Dilemma: tit-for-tat.                            Hex: picks the cell that best cuts our path / shortens its own.
  Dots and Boxes: takes every box, never gives a third side.  Mini Chess: 2-ply search on material, mate first.
  Bulls and Cows: keeps only codes consistent with its feedback.  Battleship: hunt on parity, then target around hits.
Anything else: a random legal move. It imports the game classes from the arena-lite copy it runs in.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import sys
from collections import Counter, deque

sys.path.insert(0, os.getcwd())


def send(obj):
    print(json.dumps(obj), flush=True)


# ---------------------------------------------------------------- per game
def nim(obs, legal, rng):
    piles = [int(x) for x in re.findall(r"pile \d+ = (\d+)", obs)]
    misere = "misere" in obs.lower()
    m = re.search(r"take 1-(\d+)", obs)
    k = int(m.group(1)) if m else None
    if k:
        n = piles[0] % (k + 1)
        return f"take {n} from 1" if n else None
    big = [p for p in piles if p > 1]
    if misere and len(big) <= 1:
        ones = sum(p == 1 for p in piles)
        if not big:
            return None
        i = piles.index(big[0])
        keep = 1 if (ones % 2 == 0) else 0  # leave an odd number of 1-piles
        return f"take {piles[i] - keep} from {i + 1}"
    x = 0
    for p in piles:
        x ^= p
    if x == 0:
        return None
    for i, p in enumerate(piles):
        if p ^ x < p:
            return f"take {p - (p ^ x)} from {i + 1}"
    return None


def rps(obs, legal, rng):
    theirs = re.findall(r"opponent (rock|paper|scissors|lizard|spock) ->", obs)
    if not theirs or rng.random() < 0.2:
        return rng.choice(legal)
    from arena.games.t2_rps import BEATS3, BEATS5
    beats = BEATS5 if "lizard" in legal else BEATS3
    top = Counter(theirs[-6:]).most_common(1)[0][0]
    return rng.choice([a for a in legal if top in beats[a]])


def pd(obs, legal, rng):
    last = re.findall(r"opponent (cooperate|defect)", obs)
    return last[-1] if last else "cooperate"


def hex_move(obs, legal, rng):
    from arena.games.t2_hex import T2Hex
    n = int(re.search(r"HEX (\d+)x", obs).group(1))
    me = int(re.search(r"You are player (\d)", obs).group(1))
    rows = [l for l in obs.splitlines() if re.match(r"^\s*\d+ [.XO]", l)]
    b = [[None if ch == "." else (0 if ch == "X" else 1) for ch in r.split()[1:1 + n]] for r in rows]

    def cost(board, s):  # 0-1 BFS: empty cell 1, own stone 0, the other's stone blocked
        INF = 99
        d = [[INF] * n for _ in range(n)]
        dq = deque()
        for i in range(n):
            r, c = (0, i) if s == 0 else (i, 0)
            v = board[r][c]
            if v == 1 - s:
                continue
            d[r][c] = 0 if v == s else 1
            (dq.appendleft if v == s else dq.append)((r, c))
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and board[rr][cc] != 1 - s:
                    w = 0 if board[rr][cc] == s else 1
                    if d[r][c] + w < d[rr][cc]:
                        d[rr][cc] = d[r][c] + w
                        (dq.appendleft if w == 0 else dq.append)((rr, cc))
        return min(d[n - 1][i] if s == 0 else d[i][n - 1] for i in range(n))

    best, score = None, None
    for a in legal:
        if a == "swap":
            continue
        cell = a.split()[1]
        c, r = "abcdefghijk".index(cell[0]), int(cell[1:]) - 1
        b[r][c] = me
        v = cost(b, me) - 1.2 * cost(b, 1 - me) + rng.random() * 0.1
        b[r][c] = None
        if score is None or v < score:
            best, score = a, v
    return best


def dots(obs, legal, rng):
    from arena.games.t2_dots import T2Dots
    m = re.search(r"DOTS AND BOXES (\d+)x(\d+)", obs)
    w, h = int(m.group(1)), int(m.group(2))
    g = T2Dots({"cols": w, "rows": h}, 0)
    open_ = set(re.search(r"Undrawn lines: (.*)", obs).group(1).split(", "))
    drawn = set(g._all_lines()) - open_
    boxes = [g._box_lines(x, y) for y in range(h) for x in range(w)]
    sides = lambda bx, extra=(): sum(l in drawn or l in extra for l in bx)  # noqa: E731
    moves = [a.split()[1] for a in legal] if legal else sorted(open_)
    take = [l for l in moves if any(l in bx and sides(bx) == 3 for bx in boxes)]
    if take:
        return "line " + take[0]
    safe = [l for l in moves if not any(l in bx and sides(bx) == 2 for bx in boxes)]
    if safe:
        return "line " + rng.choice(safe)
    # give the fewest boxes: the line touching boxes in the shortest chain (crude: fewest 2-sided neighbours)
    return "line " + min(moves, key=lambda l: sum(l in bx and sides(bx) == 2 for bx in boxes) + rng.random() * 0.1)


VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}


def chess(obs, legal, rng):
    from arena.games.t2_minichess import T2MiniChess
    g = T2MiniChess(T2MiniChess.variants()[0 if "Gardner" in obs else 1], 0)
    me = int(re.search(r"You are player (\d)", obs).group(1))
    rows = [l for l in obs.splitlines() if re.match(r"^\s+\d [.a-zA-Z] ", l)]
    g.b = {}
    for line in rows:
        parts = line.split()
        r = int(parts[0]) - 1
        for f, ch in enumerate(parts[1:1 + g.w]):
            if ch != ".":
                g.b[(f, r)] = ch

    def legal_of(board, side):
        return [m for m in g._pseudo(side, board) if not g._in_check(side, g._make(board, m))]

    def mat(board, side):
        return sum(VAL[p.lower()] * (1 if g._side(p) == side else -1) for p in board.values())

    def value(board, side, depth):  # from side's view, side to move
        ms = legal_of(board, side)
        if not ms:
            return -1000 if g._in_check(side, board) else 0
        if depth == 0:
            return mat(board, side)
        return max(-value(g._make(board, m), 1 - side, depth - 1) for m in ms)

    best, bv = None, None
    for m in legal_of(g.b, me):
        v = -value(g._make(g.b, m), 1 - me, 1) + rng.random() * 0.3
        if bv is None or v > bv:
            best, bv = m, v
    return g._str(best) if best else None


def bulls(obs, legal, rng):
    m = re.search(r"Your code has (\d+) digits from (\d+)(, all different)?", obs)
    k, digits, distinct = int(m.group(1)), m.group(2), bool(m.group(3))
    seen = [(g, int(b), int(c)) for g, b, c in re.findall(r"^\s+\d+\. (\d+) -> (\d+) bulls, (\d+) cows", obs, re.M)]
    pool = itertools.permutations(digits, k) if distinct else itertools.product(digits, repeat=k)

    def score(s, g):
        b = sum(x == y for x, y in zip(s, g))
        return b, sum((Counter(s) & Counter(g)).values()) - b

    cands = ["".join(p) for p in pool]
    cands = [c for c in cands if all(score(c, g) == (b, cw) for g, b, cw in seen)]
    return "guess " + (rng.choice(cands) if cands else rng.choice(["".join(p) for p in itertools.permutations(digits, k)]))


def battleship(obs, legal, rng):
    n = int(re.search(r"BATTLESHIP (\d+)x", obs).group(1))
    cols = "ABCDEFGHIJ"[:n]
    grid = [l.split()[1:] for l in obs.splitlines() if re.match(r"^\s+\d+ [.oX#]( |$)", l)][:n]
    free = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == "."]
    hits = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == "X"]
    cell = lambda rc: f"fire {cols[rc[1]]}{rc[0] + 1}"  # noqa: E731
    if hits:
        opts = []
        for r, c in hits:
            line = [(r + d, c) for d in (-1, 1)] if any((r + d, c) in hits for d in (-1, 1)) else []
            line += [(r, c + d) for d in (-1, 1)] if any((r, c + d) in hits for d in (-1, 1)) else []
            for x in (line or [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]):
                if x in free:
                    opts.append(x)
        if opts:
            return cell(rng.choice(opts))
    par = [x for x in free if (x[0] + x[1]) % 2 == 0] or free
    return cell(rng.choice(par))


GAMES = [("NIM - ", nim), ("ROCK PAPER SCISSORS", rps), ("PRISONER'S DILEMMA", pd), ("HEX ", hex_move),
         ("DOTS AND BOXES", dots), ("MINI CHESS", chess), ("BULLS AND COWS", bulls), ("BATTLESHIP", battleship)]


def main():
    rng = random.Random()
    for raw in sys.stdin:
        if not raw.strip():
            continue
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind in ("prep", "start"):
            send({"type": "ready", **({"chat": "Good luck, I play it straight."} if kind == "start" else {})})
        elif kind == "turn":
            obs, legal = msg.get("observation") or "", msg.get("legal_actions") or []
            action = None
            for head, fn in GAMES:
                if obs.startswith(head):
                    try:
                        action = fn(obs, legal, rng)
                    except Exception as e:  # noqa: BLE001
                        print(f"t2_ref_bot {head}: {e!r}", file=sys.stderr, flush=True)
                    break
            if not action or (legal and action not in legal):
                action = rng.choice(legal) if legal else "pass"
            send({"type": "action", "action": action, "chat": "Thinking it through."})
        elif kind == "shutdown":
            break


if __name__ == "__main__":
    main()

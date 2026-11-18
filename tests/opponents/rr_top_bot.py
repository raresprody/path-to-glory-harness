"""A test opponent that plays like the TOP teams of round robin rr1 (nyx, caramel-syrup, 0tknsleft, alt). Zero spend.

    python3 tests/sweep.py --bot IMAGE --only battleship,hex --opponent "python3 ../../../opponents/rr_top_bot.py"

Built from public replays (tests/opponents/build_rr_book.py -> rr_top_book.json), on the REAL arena observation formats:
  1. book: if the exact position was faced by a top team, play what they played (openings, mostly);
  2. style: Prisoner's Dilemma, Mini Poker and Colonel Blotto sample the top teams' own choices in the same situation;
  3. otherwise a solid rule player: perfect Nim (all 4 rule sets), Connect Four win/block/no-gift, Hex path-cost,
     Dots and Boxes boxes-then-safe-then-fewest-given, Battleship placement density (hunt + target, salvo too),
     Bulls and Cows / Wordle consistent guesses, RPS counter to the opponent's habits.
Anything else (or any error): the book, else a random legal move. A test bot, not ours: nothing here goes in the image.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import re
import sys
from collections import Counter, deque

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    DATA = json.load(open(os.path.join(HERE, "rr_top_book.json")))
except OSError:
    DATA = {"book": {}, "style": {}}
try:
    WORDS = [w.strip().lower() for w in open(os.path.join(HERE, "rr_top_words.txt")) if w.strip().isalpha()]
except OSError:
    WORDS = []
RNG = random.Random()


def send(obj):
    print(json.dumps(obj), flush=True)


def pick(counts: dict, legal=None):
    """Sample an action from {action: count}, legal ones only when a list is known."""
    items = [(a, c) for a, c in counts.items() if not legal or a in legal]
    if not items:
        return None
    r = RNG.random() * sum(c for _, c in items)
    for a, c in items:
        r -= c
        if r <= 0:
            return a
    return items[-1][0]


def book(gid, obs, legal):
    k = hashlib.sha1((gid + "\n" + obs.strip()).encode()).hexdigest()[:16]
    return pick(DATA["book"].get(k) or {}, legal)


# ------------------------------------------------------------------ Nim
def nim(obs, legal):
    piles = [int(x) for x in re.findall(r"pile \d+ = (\d+)", obs.split("Piles now:")[-1])]
    lim = re.search(r"never more than (\d+)", obs)
    k = int(lim.group(1)) if lim else None
    misere = "MISERE" in obs
    two = "TWO different piles" in obs

    def losing(p):  # P-position for the player to move
        if k:
            x = 0
            for v in p:
                x ^= v % (k + 1)
            return x == 0
        if two:  # Moore's Nim_2: every bit column sums to 0 mod 3
            return all(sum((v >> b) & 1 for v in p) % 3 == 0 for b in range(8))
        x = 0
        for v in p:
            x ^= v
        if misere and all(v <= 1 for v in p):
            return sum(p) % 2 == 1
        return x == 0

    moves = []
    for i, v in enumerate(piles):
        for t in range(1, (min(v, k) if k else v) + 1):
            moves.append(([(i, t)], f"take {t} from pile {i + 1}"))
    if two:
        for i, j in itertools.combinations(range(len(piles)), 2):
            for a in range(1, piles[i] + 1):
                for b in range(1, piles[j] + 1):
                    moves.append(([(i, a), (j, b)], f"take {a} from pile {i + 1} and {b} from pile {j + 1}"))
    for takes, text in moves:
        p = list(piles)
        for i, t in takes:
            p[i] -= t
        if misere and sum(p) == 0:
            continue  # taking the last token loses
        if losing(p):
            return text
    big = max(range(len(piles)), key=lambda i: piles[i])
    return f"take 1 from pile {big + 1}"


# ------------------------------------------------------------------ Connect Four
def connect_four(obs, legal):
    m = re.search(r"CONNECT (\d+) \((\d+)x(\d+)\)", obs)
    need = int(m.group(1))
    me = re.search(r"play ([XO])", obs).group(1)
    op = "O" if me == "X" else "X"
    rows = [l.strip()[1:-1].split() for l in obs.splitlines() if re.match(r"^\|[.XO ]+\|$", l.strip())]
    H, W = len(rows), len(rows[0])
    open_cols = [int(c) for c in re.search(r"Open columns: ([\d ]+)", obs).group(1).split()]

    def drop(b, c, p):
        for r in range(H - 1, -1, -1):
            if b[r][c] == ".":
                b[r][c] = p
                return r
        return None

    def wins(b, p):
        for r in range(H):
            for c in range(W):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    if all(0 <= r + i * dr < H and 0 <= c + i * dc < W and b[r + i * dr][c + i * dc] == p for i in range(need)):
                        return True
        return False

    def after(c, p):
        b = [row[:] for row in rows]
        r = drop(b, c - 1, p)
        return b, r

    for p in (me, op):  # win now, else block
        for c in open_cols:
            b, _ = after(c, p)
            if wins(b, p):
                return f"drop {c}"
    safe = []
    for c in open_cols:
        b, r = after(c, me)
        if r is not None and r > 0:
            b2 = [row[:] for row in b]
            b2[r - 1][c - 1] = op
            if wins(b2, op):
                continue  # would let them win on top
        safe.append(c)
    cand = safe or open_cols
    mid = (W + 1) / 2
    return f"drop {min(cand, key=lambda c: abs(c - mid) + RNG.random() * 0.8)}"


# ------------------------------------------------------------------ Hex
def hex_move(obs, legal):
    me_o = "play O" in obs
    rows = []
    for l in obs.splitlines():
        mm = re.match(r"^\s*(\d+)\s+((?:[.XO]\s+)+)\d+\s*$", l)
        if mm:
            rows.append(mm.group(2).split())
    n = len(rows)
    b = [[None if ch == "." else (0 if ch == "X" else 1) for ch in r] for r in rows]
    me = 1 if me_o else 0  # 0 = X joins top-bottom, 1 = O joins left-right

    def cost(s):
        INF = 99
        d = [[INF] * n for _ in range(n)]
        dq = deque()
        for i in range(n):
            r, c = (0, i) if s == 0 else (i, 0)
            v = b[r][c]
            if v == 1 - s:
                continue
            d[r][c] = 0 if v == s else 1
            (dq.appendleft if v == s else dq.append)((r, c))
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and b[rr][cc] != 1 - s:
                    w = 0 if b[rr][cc] == s else 1
                    if d[r][c] + w < d[rr][cc]:
                        d[rr][cc] = d[r][c] + w
                        (dq.appendleft if w == 0 else dq.append)((rr, cc))
        return min(d[n - 1][i] if s == 0 else d[i][n - 1] for i in range(n))

    best, score = None, None
    for r in range(n):
        for c in range(n):
            if b[r][c] is None:
                b[r][c] = me
                v = cost(me) - 1.1 * cost(1 - me) + RNG.random() * 0.1 + 0.02 * (abs(r - n // 2) + abs(c - n // 2))
                b[r][c] = None
                if score is None or v < score:
                    best, score = f"place {'abcdefghijk'[c]}{r + 1}", v
    return best


# ------------------------------------------------------------------ Dots and Boxes
def dots(obs, legal):
    letters = next(l.split() for l in obs.splitlines() if re.match(r"^\s+a\s+b(\s+[c-z])*\s*$", l))
    nums = [int(x) for x in re.findall(r"^\s*(\d+)\s+\+", obs, re.M)]
    W, H = len(letters) - 1, max(nums) - 1
    dot = lambda x, y: f"{'abcdefghij'[x]}{y + 1}"  # noqa: E731
    allines = [f"{dot(x, y)}-{dot(x + 1, y)}" for y in range(H + 1) for x in range(W)] + \
              [f"{dot(x, y)}-{dot(x, y + 1)}" for y in range(H) for x in range(W + 1)]
    canon = {}
    for l in allines:
        a, c = l.split("-")
        canon[l] = canon[f"{c}-{a}"] = l
    drawn = set()  # read the board itself: "+---+" for a horizontal line, "|" under a "+" for a vertical one
    lines = obs.splitlines()
    for i, l in enumerate(lines):
        mm = re.match(r"^\s*(\d+)\s+\+", l)
        if not mm:
            continue
        y = int(mm.group(1)) - 1
        plus = [j for j, ch in enumerate(l) if ch == "+"]
        for x in range(min(W, len(plus) - 1)):
            if l[plus[x] + 1:plus[x + 1]].strip("-") == "" and "-" in l[plus[x] + 1:plus[x + 1]]:
                drawn.add(f"{dot(x, y)}-{dot(x + 1, y)}")
        nxt = lines[i + 1] if i + 1 < len(lines) and y < H else ""
        for x in range(min(W + 1, len(plus))):
            if plus[x] < len(nxt) and nxt[plus[x]] == "|":
                drawn.add(f"{dot(x, y)}-{dot(x, y + 1)}")
    if not drawn:
        drawn = {canon[l] for l in re.findall(r"line ([a-j]\d+-[a-j]\d+)", obs.split("Move history")[-1]) if l in canon}
    boxes = [[f"{dot(x, y)}-{dot(x + 1, y)}", f"{dot(x, y + 1)}-{dot(x + 1, y + 1)}",
              f"{dot(x, y)}-{dot(x, y + 1)}", f"{dot(x + 1, y)}-{dot(x + 1, y + 1)}"] for y in range(H) for x in range(W)]
    free = [l for l in allines if l not in drawn]

    def sides(bx, d):
        return sum(l in d for l in bx)

    take = [l for l in free if any(l in bx and sides(bx, drawn) == 3 for bx in boxes)]
    if take:
        return "line " + take[0]
    safe = [l for l in free if not any(l in bx and sides(bx, drawn) == 2 for bx in boxes)]
    if safe:
        return "line " + RNG.choice(safe)

    def given(l):  # boxes the opponent collects greedily after we draw l
        d = set(drawn) | {l}
        got = 0
        while True:
            t = next((m for m in allines if m not in d and any(m in bx and sides(bx, d) == 3 for bx in boxes)), None)
            if t is None:
                return got
            before = sum(sides(bx, d) == 4 for bx in boxes)
            d.add(t)
            got += sum(sides(bx, d) == 4 for bx in boxes) - before
    return "line " + min(free, key=lambda l: given(l) + RNG.random() * 0.1)


# ------------------------------------------------------------------ Battleship
def battleship(obs, legal):
    part = obs.split("YOUR SHOTS")[1].split("YOUR WATERS")[0]
    grid = [l.split()[1:] for l in part.splitlines() if re.match(r"^\s*\d+\s+[.oX#](\s|$)", l)]
    n = len(grid)
    shots = int((re.search(r"fire exactly (\d+) shot", obs) or [0, 1])[1])
    afloat = [int(s) for s in re.findall(r"\w+\((\d)\) afloat", obs.split("Opponent fleet:")[1].splitlines()[0])]
    cell = lambda r, c: f"{'abcdefghij'[c]}{r + 1}"  # noqa: E731
    w = [[0.0] * n for _ in range(n)]
    for size in afloat or [2]:
        for r in range(n):
            for c in range(n):
                for dr, dc in ((0, 1), (1, 0)):
                    cs = [(r + i * dr, c + i * dc) for i in range(size)]
                    if any(not (0 <= x < n and 0 <= y < n) or grid[x][y] in "o#" for x, y in cs):
                        continue
                    bonus = 1 + 30 * sum(grid[x][y] == "X" for x, y in cs)
                    for x, y in cs:
                        w[x][y] += bonus
    free = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == "."]
    free.sort(key=lambda rc: -(w[rc[0]][rc[1]] + RNG.random() * 0.01))
    return "fire " + " ".join(cell(r, c) for r, c in free[:shots])


# ------------------------------------------------------------------ Bulls and Cows
def bulls(obs, legal):
    m = re.search(r"BULLS AND COWS - (\d+) (digits|letters ([A-Z])-([A-Z]))", obs)
    k = int(m.group(1))
    alpha = "0123456789" if m.group(2) == "digits" else "".join(chr(c) for c in range(ord(m.group(3)), ord(m.group(4)) + 1))
    rep = "repeats allowed" in obs
    mine = obs.split("Your guesses at the opponent's code:")[1].split("Opponent's guesses")[0]
    seen = [(g, int(b), int(c)) for g, b, c in re.findall(r"\d+\. (\w+) -> (\d+) bulls?, (\d+) cows?", mine)]

    def score(s, g):
        b = sum(x == y for x, y in zip(s, g))
        return b, sum((Counter(s) & Counter(g)).values()) - b

    pool = itertools.product(alpha, repeat=k) if rep else itertools.permutations(alpha, k)
    cands = []
    for p in pool:
        s = "".join(p)
        if all(score(s, g) == (b, c) for g, b, c in seen):
            cands.append(s)
            if len(cands) > 4000 and not seen:
                break
    return "guess " + (RNG.choice(cands) if cands else "".join(RNG.choice(alpha) for _ in range(k)))


# ------------------------------------------------------------------ Wordle
def wordle(obs, legal):
    k = int(re.search(r"\((\d+) letters", obs).group(1))
    mine = obs.split("Your guesses")[1].split("Opponent's guesses")[0]
    seen = re.findall(r"\d+\. ([A-Z]+)\s+((?:[GYX] ?)+)", mine)

    def colours(secret, g):
        out, left = ["X"] * k, Counter()
        for i in range(k):
            if g[i] == secret[i]:
                out[i] = "G"
            else:
                left[secret[i]] += 1
        for i in range(k):
            if out[i] != "G" and left[g[i]]:
                out[i] = "Y"
                left[g[i]] -= 1
        return "".join(out)

    words = [w for w in WORDS if len(w) == k]
    cands = [w for w in words if all(colours(w, g.lower()) == c.replace(" ", "") for g, c in seen)]
    tried = {g.lower() for g, _ in seen}
    cands = [w for w in cands if w not in tried] or [w for w in words if w not in tried]
    return "guess " + (RNG.choice(cands) if cands else "a" * k)


# ------------------------------------------------------------------ simultaneous / style games
def pd(obs, legal, var):
    st = DATA["style"].get("prisoners_dilemma", {})
    rnd = re.search(r"Round (\d+)", obs)
    last = re.findall(r"\d+\. you (\w+), opponent (\w+)", obs)
    opp = last[-1][1] if last else "-"
    for key in (f"{var}|{rnd.group(1) if rnd else '?'}|{opp}", f"{var}|{opp}", f"*|{opp}"):
        if sum((st.get(key) or {}).values()) >= 3:
            return pick(st[key], legal)
    return opp if opp in ("cooperate", "defect") else "cooperate"


def poker(obs, legal):
    sys.path.insert(0, HERE)
    from build_rr_book import poker_key
    st = DATA["style"].get("mini_poker", {})
    k = poker_key(obs)
    for key in (k, k.split("|", 1)[1]):
        a = pick(st.get(key) or {}, legal)
        if a:
            return a
    acts = re.search(r"Legal actions: ([^\n]+)", obs)
    opts = [x.split("(")[0].strip() for x in acts.group(1).split(",")] if acts else []
    return next((x for x in ("check", "call") if x in opts), opts[0] if opts else None)


def blotto(obs, legal, var):
    vals = [int(x) for x in re.findall(r"F\d+=(\d+)", obs.split("History")[0])]
    exact = re.search(r"place exactly (\d+) soldiers", obs)
    reserve = re.search(r"send 0 to (\d+)", obs)
    n = int(exact.group(1)) if exact else int(reserve.group(1))
    f = int((re.search(r"Your move: (\d+) numbers", obs) or [0, len(vals) or 3])[1])
    allocs = dict(DATA["style"].get("colonel_blotto", {}).get(var) or {})
    fits = (lambda t: t == n) if exact else (lambda t: t <= n)  # noqa: E731
    ok = {a: c for a, c in allocs.items() if len(a.split()) == f and all(x.isdigit() for x in a.split())
          and fits(sum(map(int, a.split())))}
    if ok:
        return pick(ok)
    vals = vals or [1] * f
    raw = [n * v / sum(vals) for v in vals]
    out = [int(x) for x in raw]
    out[max(range(f), key=lambda i: vals[i])] += n - sum(out)
    return " ".join(map(str, out))


def rps(obs, legal):
    signs = ["rock", "paper", "scissors"]
    beats = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
    hist = re.findall(r"\d+\. you (\w+) - opponent (\w+)", obs)
    theirs = [o for _, o in hist][-6:]
    mine = [y for y, _ in hist]
    if not theirs or RNG.random() < 0.25:
        a = RNG.choice(signs)
    else:
        a = beats[Counter(theirs).most_common(1)[0][0]]
    if "three times in a row" in obs and len(mine) >= 2 and mine[-1] == mine[-2] == a:
        a = RNG.choice([s for s in signs if s != a])
    return a


# ------------------------------------------------------------------ main loop
def decide(obs, legal):
    var = re.search(r"\(([^)]*)\)", obs.splitlines()[0])
    var = var.group(1) if var else ""
    head = obs.split(" ")[0:3]
    title = " ".join(head)
    table = [("NIM", "nim", nim), ("CONNECT", "connect_four", connect_four), ("HEX", "hex", hex_move),
             ("DOTS", "dots_and_boxes", dots), ("BATTLESHIP", "battleship", battleship),
             ("BULLS", "bulls_and_cows", bulls), ("WORDLE", "wordle_duel", wordle),
             ("PRISONER'S", "prisoners_dilemma", None), ("MINI POKER", "mini_poker", None),
             ("COLONEL", "colonel_blotto", None), ("ROCK", "rps", None), ("MINI CHESS", "mini_chess", None)]
    for prefix, gid, fn in table:
        if title.startswith(prefix):
            b = book(gid, obs, legal)
            if b:
                return b
            if gid == "prisoners_dilemma":
                full = re.search(r"PRISONER'S DILEMMA - ([^.]+(?:\.[^ ][^.]*)*)\. You are", obs)
                return pd(obs, legal, full.group(1) if full else var)
            if gid == "mini_poker":
                return poker(obs, legal)
            if gid == "colonel_blotto":
                full = re.search(r"COLONEL BLOTTO - (.+?)\. You are", obs)
                return blotto(obs, legal, full.group(1) if full else var)
            if gid == "rps":
                return rps(obs, legal)
            return fn(obs, legal) if fn else None
    return None


def main():
    for raw in sys.stdin:
        if not raw.strip():
            continue
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind in ("prep", "start"):
            send({"type": "ready", **({"chat": "Good luck, have fun."} if kind == "start" else {})})
        elif kind == "turn":
            obs, legal = msg.get("observation") or "", msg.get("legal_actions") or []
            action = None
            try:
                action = decide(obs, legal)
            except Exception as e:  # noqa: BLE001
                print(f"rr_top_bot: {e!r}", file=sys.stderr, flush=True)
            if not action or (legal and len(legal) <= 60 and action not in legal):
                action = RNG.choice(legal) if legal else (action or "pass")
            send({"type": "action", "action": action, "chat": "Your move was noted."})
        elif kind == "shutdown":
            break


if __name__ == "__main__":
    main()

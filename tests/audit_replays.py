"""Zero-spend post-mortem of OUR real games: which of our moves and chat lines were provable mistakes. Owner: path-to-glory-3.

    python3 tests/audit_replays.py <replay.json or dir> [...] [--team path-to-glory]

Reads public arena match replays (rr1, T2 knockout, any later ones) and flags, per game, only things that are checkable
from the replay alone (no model, no simulator):
  CHAT-LEAK      simultaneous game: our chat line with a move names an option of that move (the opponent can read it)
  READ-BY-OPP    ... and the opponent's move in that same step beat exactly the option we named
  INCONSISTENT   Bulls / Wordle: our guess cannot be the secret given our OWN earlier feedback
  IGNORED-HIT    Battleship: we had an unsunk hit and fired nowhere next to any hit
  MISSED-WIN / MISSED-BLOCK   Connect Four: a column won at once / the opponent's winning column was open and we played elsewhere
  GIFT-BOX       Dots and Boxes: we drew a 3rd side of a box while a line touching no 2-sided box was still open
  MISSED-BOX     Dots and Boxes: a box could be closed and we drew a line that closed nothing (not the double-deal endgame)
  WARNING / SLOW warnings we got; moves that took more than 15 s
Also prints, for simultaneous games, how often the OPPONENT's chat named its own next move (can we read them?).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

SIMUL = {"rps": ["rock", "paper", "scissors", "lizard", "spock"], "prisoners_dilemma": ["cooperate", "defect"]}
BEATS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}


def bulls_ok(obs, guess):
    mine = obs.split("Your guesses at the opponent's code:")[1].split("Opponent's guesses")[0]
    seen = [(g, int(b), int(c)) for g, b, c in re.findall(r"\d+\. (\w+) -> (\d+) bulls?, (\d+) cows?", mine)]
    s = guess.split()[-1].upper()

    def score(x, g):
        b = sum(p == q for p, q in zip(x, g))
        return b, sum((Counter(x) & Counter(g)).values()) - b
    bad = [(g, b, c) for g, b, c in seen if len(g) == len(s) and score(s, g.upper()) != (b, c)]
    return bad


def wordle_ok(obs, guess):
    mine = obs.split("Your guesses")[1].split("Opponent's guesses")[0]
    seen = re.findall(r"\d+\. ([A-Z]+)\s+((?:[GYX] ?)+)", mine)
    w = guess.split()[-1].upper()

    def colours(secret, g):
        k = len(g)
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
    return [(g, c) for g, c in seen if len(g) == len(w) and colours(w, g) != c.replace(" ", "")]


def battleship_ignored(obs, action):
    part = obs.split("YOUR SHOTS")[1].split("YOUR WATERS")[0]
    grid = [l.split()[1:] for l in part.splitlines() if re.match(r"^\s*\d+\s+[.oX#](\s|$)", l)]
    n = len(grid)
    hits = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == "X"]
    if not hits:
        return False
    near = {(r + dr, c + dc) for r, c in hits for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
    open_near = [x for x in near if 0 <= x[0] < n and 0 <= x[1] < n and grid[x[0]][x[1]] == "."]
    if not open_near:
        return False
    cells = re.findall(r"([a-j])(\d+)", action.lower())
    fired = {(int(r) - 1, "abcdefghij".index(c)) for c, r in cells}
    return not (fired & near)


def c4_check(obs, action):
    m = re.search(r"CONNECT (\d+) \((\d+)x(\d+)\)", obs)
    if not m:
        return None
    need = int(m.group(1))
    me = re.search(r"play ([XO])", obs).group(1)
    op = "O" if me == "X" else "X"
    rows = [l.strip()[1:-1].split() for l in obs.splitlines() if re.match(r"^\|[.XO ]+\|$", l.strip())]
    H, W = len(rows), len(rows[0])
    cols = [int(c) for c in re.search(r"Open columns: ([\d ]+)", obs).group(1).split()]

    def wins_after(c, p):
        b = [r[:] for r in rows]
        for r in range(H - 1, -1, -1):
            if b[r][c - 1] == ".":
                b[r][c - 1] = p
                break
        for r in range(H):
            for cc in range(W):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    if all(0 <= r + i * dr < H and 0 <= cc + i * dc < W and b[r + i * dr][cc + i * dc] == p for i in range(need)):
                        return True
        return False
    mine = [c for c in cols if wins_after(c, me)]
    theirs = [c for c in cols if wins_after(c, op)]
    played = int(re.search(r"\d+", action).group())
    if mine and played not in mine:
        return f"MISSED-WIN (col {mine} wins, played {played})"
    if not mine and theirs and played not in theirs:
        return f"MISSED-BLOCK (they win at {theirs}, played {played})"
    return None


def dots_check(obs, action):
    lines = obs.splitlines()
    try:
        letters = next(l.split() for l in lines if re.match(r"^\s+a\s+b(\s+[c-z])*\s*$", l))
    except StopIteration:
        return None
    nums = [int(x) for x in re.findall(r"^\s*(\d+)\s+\+", obs, re.M)]
    W, H = len(letters) - 1, max(nums) - 1
    dot = lambda x, y: f"{'abcdefghij'[x]}{y + 1}"  # noqa: E731
    drawn = set()
    for i, l in enumerate(lines):
        mm = re.match(r"^\s*(\d+)\s+\+", l)
        if not mm:
            continue
        y = int(mm.group(1)) - 1
        plus = [j for j, ch in enumerate(l) if ch == "+"]
        for x in range(min(W, len(plus) - 1)):
            seg = l[plus[x] + 1:plus[x + 1]]
            if "-" in seg and seg.strip("-") == "":
                drawn.add(f"{dot(x, y)}-{dot(x + 1, y)}")
        nxt = lines[i + 1] if i + 1 < len(lines) and y < H else ""
        for x in range(min(W + 1, len(plus))):
            if plus[x] < len(nxt) and nxt[plus[x]] == "|":
                drawn.add(f"{dot(x, y)}-{dot(x, y + 1)}")
    allines = [f"{dot(x, y)}-{dot(x + 1, y)}" for y in range(H + 1) for x in range(W)] + \
              [f"{dot(x, y)}-{dot(x, y + 1)}" for y in range(H) for x in range(W + 1)]
    boxes = [[f"{dot(x, y)}-{dot(x + 1, y)}", f"{dot(x, y + 1)}-{dot(x + 1, y + 1)}",
              f"{dot(x, y)}-{dot(x, y + 1)}", f"{dot(x + 1, y)}-{dot(x + 1, y + 1)}"] for y in range(H) for x in range(W)]
    free = [l for l in allines if l not in drawn]
    a, b = action.split()[-1].split("-")
    mv = f"{a}-{b}" if f"{a}-{b}" in allines else f"{b}-{a}"
    sides = lambda bx: sum(l in drawn for l in bx)  # noqa: E731
    take = [l for l in free if any(l in bx and sides(bx) == 3 for bx in boxes)]
    safe = [l for l in free if not any(l in bx and sides(bx) == 2 for bx in boxes)]
    closes = any(mv in bx and sides(bx) == 3 for bx in boxes)
    gives = any(mv in bx and sides(bx) == 2 for bx in boxes)
    if take and not closes and len(take) > 2:
        return f"MISSED-BOX ({len(take)} closable, drew {mv})"
    if gives and safe and not closes:
        return f"GIFT-BOX (drew {mv}, {len(safe)} safe lines open)"
    return None


def audit(paths, team):
    flags = defaultdict(Counter)
    examples = defaultdict(list)
    opp_named = Counter()
    games = Counter()
    for p in paths:
        m = json.load(open(p))
        mid = m.get("match_id", os.path.basename(p))
        chat = defaultdict(dict)
        for c in m.get("chat") or []:
            if c.get("kind") == "chat":
                chat[(c.get("game"), c.get("step"))][c.get("team")] = c.get("text") or ""
        for g in m.get("games") or []:
            if team not in (g.get("seats") or []):
                continue
            gid, me = g["game_id"], g["seats"].index(team)
            res = (g.get("result") or {}).get("winner")
            games[(gid, "W" if res == team else ("D" if res is None else "L"))] += 1
            w = (g.get("warnings") or {}).get(str(me), 0)
            if w:
                flags[gid]["WARNING"] += w
                examples[gid].append(f"{mid} g{g['index']}: {w} warning(s)")
            for st in g.get("timeline") or []:
                acts = st.get("actions") or {}
                a = acts.get(str(me))
                if not a or not a.get("action"):
                    continue
                act, obs = a["action"], a.get("observation") or ""
                if (a.get("ms") or 0) > 15000:
                    flags[gid]["SLOW"] += 1
                said = chat[(g["index"], st["step"])].get(team, "").lower()
                opp = next((v for k, v in acts.items() if k != str(me)), None)
                if gid in SIMUL:
                    named = [o for o in SIMUL[gid] if re.search(rf"\b{o}", said)]
                    if named:
                        flags[gid]["CHAT-LEAK"] += 1
                        if opp and gid == "rps" and any(BEATS.get(o) == opp["action"] for o in named):
                            flags[gid]["READ-BY-OPP"] += 1
                        if len(examples[gid]) < 6:
                            examples[gid].append(f"{mid} g{g['index']} step {st['step']}: played {act!r}, chat {said[:90]!r}")
                    oteam = g["seats"][1 - me]
                    osaid = chat[(g["index"], st["step"])].get(oteam, "").lower()
                    if opp and osaid:
                        opp_named["lines"] += 1
                        opp_named["named own move"] += bool(re.search(rf"\b{opp['action']}", osaid))
                try:
                    msg = None
                    if gid == "bulls_and_cows" and "Your guesses at the opponent" in obs:
                        bad = bulls_ok(obs, act)
                        msg = f"INCONSISTENT {act!r} vs {bad[:2]}" if bad else None
                    elif gid == "wordle_duel" and "Your guesses" in obs:
                        bad = wordle_ok(obs, act)
                        msg = f"INCONSISTENT {act!r} vs {bad[:2]}" if bad else None
                    elif gid == "battleship" and "YOUR SHOTS" in obs and battleship_ignored(obs, act):
                        msg = f"IGNORED-HIT played {act!r}"
                    elif gid == "connect_four":
                        msg = c4_check(obs, act)
                    elif gid == "dots_and_boxes":
                        msg = dots_check(obs, act)
                    if msg:
                        flags[gid][msg.split()[0]] += 1
                        if len(examples[gid]) < 6:
                            examples[gid].append(f"{mid} g{g['index']} step {st['step']}: {msg}")
                except Exception as e:  # noqa: BLE001
                    flags[gid]["(audit error)"] += 1
    print(f"== audit of {team}: {len(paths)} replays")
    for gid in sorted({k[0] for k in games}):
        rec = f"{games[(gid, 'W')]}-{games[(gid, 'D')]}-{games[(gid, 'L')]}"
        print(f"{gid:18} W-D-L {rec:7} " + ", ".join(f"{k} {v}" for k, v in flags[gid].most_common()))
        for e in examples[gid][:4]:
            print("      " + e)
    if opp_named["lines"]:
        print(f"opponents' chat in simultaneous games named their OWN move in {opp_named['named own move']} of "
              f"{opp_named['lines']} lines")


def main():
    args = sys.argv[1:]
    team = "path-to-glory"
    if "--team" in args:
        i = args.index("--team")
        team = args[i + 1]
        args = args[:i] + args[i + 2:]
    paths = []
    for a in args:
        paths += sorted(glob.glob(os.path.join(a, "*.json"))) if os.path.isdir(a) else [a]
    paths = [p for p in paths if os.path.basename(p) not in ("games.json",)]
    audit(paths, team)


if __name__ == "__main__":
    main()

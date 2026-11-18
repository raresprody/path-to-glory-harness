"""Build tests/opponents/rr_top_book.json from public round-robin replays (tests only, zero spend). Owner: path-to-glory-3.

    python3 tests/opponents/build_rr_book.py <replays dir> [team ...]

Reads every rr1-*.json replay (public arena API data), keeps only the moves of the given teams (default: the top 4 of
rr1 by round-robin points) and writes:
  book   sha1(game id + observation) -> {action: count}   exact positions they faced (openings mostly)
  style  per game, situation key -> {action: count}       PD (round, opponent's last), poker (cards + betting),
                                                           Blotto (variant -> allocations they played)
Used by rr_top_bot.py, a test opponent that plays like the top teams did.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

TOP = ["nyx", "caramel-syrup", "0tknsleft", "alt"]


def okey(gid: str, obs: str) -> str:
    return hashlib.sha1((gid + "\n" + obs.strip()).encode()).hexdigest()[:16]


def pd_key(variant: str, obs: str) -> list[str]:
    rnd = re.search(r"Round (\d+)", obs)
    last = re.findall(r"\d+\. you (\w+), opponent (\w+)", obs)
    opp = last[-1][1] if last else "-"
    r = rnd.group(1) if rnd else "?"
    return [f"{variant}|{r}|{opp}", f"{variant}|{opp}", f"*|{opp}"]


def poker_key(obs: str) -> str:
    card = " ".join(re.findall(r"(?:Your card|Community card|Board)[^:]*: *([^\n(]+)", obs)).strip()
    bets = re.search(r"Betting in this hand so far:\n((?:  .*\n?)*)", obs)
    seq = re.sub(r"\s+", " ", bets.group(1)).strip() if bets else ""
    seq = re.sub(r"\(bet size \d+\)", "", seq)
    first = "first" if "You act first" in obs else "second"
    return f"{first}|{card}|{seq}"


def main():
    src = sys.argv[1]
    teams = set(sys.argv[2:] or TOP)
    book = defaultdict(lambda: defaultdict(int))
    style = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    n = 0
    for f in sorted(glob.glob(os.path.join(src, "rr1*.json"))):
        m = json.load(open(f))
        for g in m.get("games") or []:
            gid, var = g["game_id"], (g.get("variant") or {}).get("label", "")
            for st in g.get("timeline") or []:
                for seat, a in (st.get("actions") or {}).items():
                    if g["seats"][int(seat)] not in teams or not a.get("observation"):
                        continue
                    obs, act = a["observation"], a["action"]
                    n += 1
                    book[okey(gid, obs)][act] += 1
                    if gid == "prisoners_dilemma":
                        for k in pd_key(var, obs):
                            style[gid][k][act] += 1
                    elif gid == "mini_poker":
                        style[gid][poker_key(obs)][act] += 1
                        style[gid][poker_key(obs).split("|", 1)[1]][act] += 1
                    elif gid == "colonel_blotto":
                        style[gid][var][act] += 1
    out = {"teams": sorted(teams), "moves": n, "book": book, "style": style}
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rr_top_book.json")
    json.dump(out, open(dst, "w"), separators=(",", ":"))
    print(f"{n} moves of {sorted(teams)} -> {dst} ({os.path.getsize(dst) // 1024} KB, {len(book)} positions)")


if __name__ == "__main__":
    main()

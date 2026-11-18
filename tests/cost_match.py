"""Real cost of one match (cost score, rule 9.3), measured on the dev key with /key/info.

    python tests/cost_match.py --bot pg-t1-freeze:test --game tic_tac_toe
    python tests/cost_match.py --bot "python -m mesh" --game pig --games-dir practice

Runs one arena-lite `practice --prep` match: our bot vs tests/opponents/nasty_bot.py (which never calls a model), so every cent
on the dev key during the run is ours. It reads the key's spend before prep, when the first game starts (prep
done) and after the match, and prints prep cost and **match cost** (only match-key spend counts for the score;
prep is excluded, 9.3). Each match gets $0.75 in the arena (7.2).

Don't run it while a teammate is testing on the same dev key: their calls would be counted as ours.
The gateway books spend late (seen: >8 s), so the split between prep and match is approximate; the script waits
--settle seconds (default 20) before each reading. mesh also logs its own per-game spend in its stderr (`end game N: ... spend $`).
"""
from __future__ import annotations

import argparse
import shutil
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PY = "python3" if shutil.which("python3") else "python"  # Windows has no python3 (path-to-glory-2, 20:35)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    return env


def spend(base_url: str, key: str) -> float:
    url = base_url.rstrip("/").removesuffix("/v1") + "/key/info"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.load(r)
    return float((d.get("info") or d).get("spend") or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", required=True, help="image tag or command (as for arena-lite --bot-a)")
    ap.add_argument("--game", default="tic_tac_toe")
    ap.add_argument("--games-dir", default="practice", choices=("practice", "games"))
    ap.add_argument("--best-of", type=int, default=3)
    ap.add_argument("--opponent", default=PY + " ../tests/opponents/nasty_bot.py mix",
                    help="opponent command, run from arena-lite/ (default: random legal moves + hostile chat, fills free-form actions)")
    ap.add_argument("--env-file", default=str(ROOT / "dev.env"))
    ap.add_argument("--settle", type=float, default=20.0)
    a = ap.parse_args()

    env = load_env(Path(a.env_file))
    base, key = env["LLM_BASE_URL"], env["LLM_API_KEY"]
    s0 = spend(base, key)
    mid = "cost-" + time.strftime("%Y%m%d-%H%M%S")
    cmd = [sys.executable, "-m", "arena", "practice", "--bot-a", a.bot, "--bot-b",
           a.opponent, "--env-file", a.env_file, "--game", a.game,
           "--games-dir", a.games_dir, "--prep", "--best-of", str(a.best_of), "--id", mid, "--name-a", "path-to-glory"]
    penv = dict(os.environ)
    penv["PYTHONPATH"] = str(ROOT) + os.pathsep + penv.get("PYTHONPATH", "")  # so "python -m mesh" works from arena-lite
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT / "arena-lite", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=penv)
    s1, t1, games, final = None, None, 0, ""
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("game ") and ":" in line and " seat " in line:
            games += 1
            if s1 is None:
                time.sleep(a.settle)  # the match has begun; prep calls are all booked by now (the bot is waiting)
                s1, t1 = spend(base, key), time.time()
        if line.startswith("final:") or line.startswith("match "):
            final = line
        print("   ", line, flush=True)
    p.wait()
    time.sleep(a.settle)
    s2 = spend(base, key)
    if s1 is None:
        print("no game started; total spend", f"${s2 - s0:.4f}")
        return 1
    prep_c, match_c = s1 - s0, s2 - s1
    print(f"\n{a.bot} on {a.game}: {games} game(s), {final}")
    print(f"  prep  ${prep_c:.4f}  ({t1 - t0 - a.settle:.0f}s)   [not scored; own $0.75 key per prep container]")
    print(f"  match ${match_c:.4f}  = {match_c / 0.75 * 100:.1f}% of the $0.75 match key"
          f"   [scored: 2.5 x min(1, c_min / c), rule 9.3]")
    if games:
        print(f"  per game ${match_c / games:.4f}")
    rec = {"t": time.strftime("%H:%M"), "bot": a.bot, "game": a.game, "games": games, "prep": prep_c, "match": match_c}
    with open(HERE / "out" / "cost_log.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    (HERE / "out").mkdir(exist_ok=True)
    sys.exit(main())

"""Rule-break sweep: play EVERY game we have, including holdout games mesh/ has never seen, and fail on any
rule break of ours. The question it answers: "given a new game, does the bot stay legal and in time?"

    python tests/sweep.py --bot pg-main:abc123                 # Docker image (what the arena runs)
    python tests/sweep.py --bot "python -m mesh" --only x_     # a command; only the holdout games
    python tests/sweep.py --selftest                           # just check the holdout games themselves (no bot)

Games: arena-lite's practice games, the team test games in arena-lite/arena/games/, and the holdout games in
tests/holdout/ (copied into a temporary copy of arena-lite under tests/out/, so arena-lite/ is never modified).
Each game: one `practice --prep` match (best of 3, seats alternate) vs tests/opponents/nasty_bot.py.

A rule break of ours = any warning we got, a game lost by illegal/invalid move, timeout, crash, invalid ready,
or a speech rule (banned/forbidden word in a move or chat). Those fail the sweep (exit 1). Losing a game fairly is
reported but is not a failure. Also reported per game: did the prep simulator validate, and how long prep took.

Spends from the dev key (our bot only): about $0.01-0.03 per game. Keep --jobs low (60 req/min per key).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = "python3" if shutil.which("python3") else "python"  # Windows has no python3 (path-to-glory-2, 20:35)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
US = "path-to-glory"
BREAK_WORDS = ("illegal", "invalid", "timeout", "timed out", "no answer", "crash", "exited", "ready", "banned",
               "forbidden", "mentioned", "warned", "wrong type", "empty")


def make_arena(out: Path) -> Path:
    dst = out / "arena-lite"
    shutil.copytree(ROOT / "arena-lite", dst, ignore=shutil.ignore_patterns("results", "memory", "__pycache__"))
    for f in sorted((HERE / "holdout").glob("x_*.py")) + sorted((HERE / "t2games").glob("t2_*.py")):
        shutil.copy(f, dst / "arena" / "games" / f.name)
    # the 8 REAL T2 games (replay-checked simulators + the arena's rules text), built by tests/realgames/build.py
    for f in sorted((HERE / "realgames").glob("*.py")) + sorted((HERE / "realgames").glob("*.json")):
        if f.name != "build.py":
            shutil.copy(f, dst / "arena" / "games" / f.name)
    return dst


def game_list(arena: Path) -> list[tuple[str, str]]:
    code = ("import json; from arena import registry; "
            "print(json.dumps({'practice': sorted(registry.practice_games()), 'games': sorted(registry.tournament_games())}))")
    d = json.loads(subprocess.run([sys.executable, "-c", code], cwd=arena, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout)
    return [("practice", g) for g in d["practice"]] + [("games", g) for g in d["games"]]


def selftest(arena: Path) -> int:
    """Random self-play of every holdout game: terminates, observations render, results are well formed."""
    code = r'''
import random, sys
REAL = {"battleship", "bulls_and_cows", "dots_and_boxes", "hex", "mini_chess", "nim", "prisoners_dilemma", "rps"}
from arena import registry
bad = 0
for gid, cls in sorted(registry.tournament_games().items()):
    if not gid.startswith(("x_", "t2_")) and gid not in REAL:
        continue
    for vi, v in enumerate(cls.variants()):
        for seed in range(30):
            g = cls(dict(v), seed); rng = random.Random(seed); steps = 0
            assert g.rules() and g.observation(0) and g.observation(1)
            while not g.is_over():
                acts = {s: g.random_action(s, rng) for s in g.to_move()}
                for s, a in acts.items():
                    assert g.normalize(s, a) is not None, (gid, a)
                    la = g.legal_actions(s)
                    assert la is None or g.normalize(s, a) in la, (gid, a, la)
                g.apply({s: g.normalize(s, a) for s, a in acts.items()}); steps += 1
                g.observation(0); g.observation(1); g.frame()
                if steps > 500: bad += 1; print("NO END", gid, vi, seed); break
            r = g.result(); assert r["winner"] in (0, 1, None) and len(r["scores"]) == 2, r
        print(f"ok {gid} variant {vi}: 30 random games")
sys.exit(1 if bad else 0)
'''
    return subprocess.run([sys.executable, "-c", code], cwd=arena).returncode


def run_one(arena: Path, bot: str, opponent: str, games_dir: str, gid: str, run: str, best_of: int) -> dict:
    mid = f"sweep-{run}-{gid}"
    cmd = [sys.executable, "-m", "arena", "practice", "--bot-a", bot, "--bot-b", opponent, "--env-file",
           str(ROOT / "dev.env"), "--game", gid, "--games-dir", games_dir, "--prep", "--best-of", str(best_of),
           "--id", mid, "--name-a", US, "--quiet"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=arena, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    rec = {"game": gid, "dir": games_dir, "secs": time.time() - t0, "exit": p.returncode, "final": "",
           "breaks": [], "losses": [], "ours_warnings": 0, "games": 0, "sim": "?", "prep_s": None}
    rec["final"] = next((l for l in p.stdout.splitlines() if l.startswith("final:")), p.stdout.strip()[-160:])
    # the replay can still be being written when arena-lite exits, and arena-lite may put results elsewhere
    # (ARENA_ROOT, or the path it prints): retry for a few seconds and look in every place it could be
    replay, tried = None, []
    for _ in range(10):
        cands = [arena / "results" / "practice" / "matches" / f"{mid}.json"]
        cands += [Path(x.strip()) for x in re.findall(r"match file:\s*(.+\.json)", p.stdout)]
        cands += list(arena.rglob(f"{mid}.json"))
        for c in cands:
            try:
                replay = json.loads(c.read_text(encoding="utf-8"))
                break
            except (OSError, ValueError) as exc:
                tried.append(f"{c.name}: {type(exc).__name__}")
        if replay is not None:
            break
        time.sleep(1)
    if replay is None:
        rec["breaks"].append(f"no replay (exit {p.returncode}; tried {sorted(set(tried))[:3]}): {p.stderr.strip()[-160:]}")
        return rec
    for gm in replay.get("games") or []:
        rec["games"] += 1
        seats = gm.get("seats") or []
        me = seats.index(US) if US in seats else -1
        w = gm.get("warnings") or {}
        rec["ours_warnings"] += int(w.get(str(me), 0) or 0) if isinstance(w, dict) else 0
        res = gm.get("result") or {}
        why = str(res.get("reason", ""))
        lost = res.get("winner_seat", res.get("winner")) not in (None, me, US)
        mine = (US in why) or (f"seat {me}" in why and not re.search(rf"seat {me} (wins|found|cracked|made|reached)", why))
        if lost and mine and any(k in why.lower() for k in BREAK_WORDS):
            rec["breaks"].append(why[:140])
        elif lost:
            rec["losses"].append(why[:100])
        st = (gm.get("study") or {}).get(str(me)) or {}
        if st.get("error"):
            rec["breaks"].append(f"study: {st.get('error')}")
    if rec["ours_warnings"]:
        rec["breaks"].append(f"{rec['ours_warnings']} warning(s) of ours")
    logs = arena / "results" / "practice" / "logs" / "warm"
    for f in sorted(logs.glob(f"{US}--{gid}*.log"), key=lambda x: x.stat().st_mtime)[-1:]:
        text = f.read_text(encoding="utf-8", errors="replace")
        rec["sim"] = ("talk" if "no simulator (language game)" in text else "ok" if "simulator accepted" in text else
                      "FAILED" if re.search(r"simulator (pipeline failed|rejected)|simgen: giving up|no simulator", text) else "?")
        m = re.findall(r"ready after ([\d.]+) s", text)
        rec["prep_s"] = float(m[-1]) if m else None
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", help="image tag or command, as for arena-lite --bot-a")
    ap.add_argument("--opponent", default=PY + " ../../../opponents/nasty_bot.py mix",
                    help="opponent command, run from the temp arena-lite copy (default: nasty_bot mix)")
    ap.add_argument("--only", default="", help="comma list of game ids or prefixes (e.g. x_,pig)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--best-of", type=int, default=3)
    ap.add_argument("--selftest", action="store_true", help="only self-play the holdout games (no bot, no spend)")
    a = ap.parse_args()

    run = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "out" / f"sweep-{run}"
    out.mkdir(parents=True, exist_ok=True)
    arena = make_arena(out)
    if a.selftest:
        return selftest(arena)
    if not a.bot:
        ap.error("--bot is required (or --selftest)")
    if selftest(arena) != 0:
        print("holdout self-test failed: fix tests/holdout first")
        return 2
    games = game_list(arena)
    if a.only:
        keys = [k.strip() for k in a.only.split(",") if k.strip()]
        games = [g for g in games if any(g[1] == k or g[1].startswith(k) for k in keys)]
    print(f"\nsweep {a.bot}: {len(games)} games, {a.jobs} at a time; output {out}", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        recs = list(ex.map(lambda g: run_one(arena, a.bot, a.opponent, g[0], g[1], run, a.best_of), games))

    print(f"\n{'game':14s} {'sim':6s} {'prep':>6s}  {'result':44s} rule breaks")
    bad = 0
    for r in recs:
        tag = "holdout" if r["game"].startswith("x_") else "t2-copy" if r["game"].startswith("t2_") else r["dir"]
        res = r["final"].replace("final: ", "")[:44]
        prep = f"{r['prep_s']:.0f}s" if r["prep_s"] is not None else "-"
        brk = "; ".join(r["breaks"]) or "none"
        bad += bool(r["breaks"])
        print(f"{r['game']:14s} {r['sim']:6s} {prep:>6s}  {res:44s} {brk}")
        if r["losses"]:
            print(f"{'':14s} fair losses ({tag}): " + " | ".join(r["losses"]))
    (out / "sweep.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    print(f"\n{len(recs) - bad}/{len(recs)} games without a rule break of ours" + (": FAILED" if bad else ": OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

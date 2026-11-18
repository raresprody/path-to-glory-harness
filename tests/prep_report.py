"""How well does the bot use its 600 s of prep? Runs ONLY the prep phase (no match) for every game, one at a time,
and reads the bot's own log lines.

    python tests/prep_report.py --bot pg-main:abc123            # all games (practice, team test games, holdout)
    python tests/prep_report.py --bot pg-main:abc123 --only x_,pig

Per game: seconds until `ready` (of 600), rules digest ok + what it detected (talk / simultaneous / hidden info /
forbidden terms), simulator accepted or not (attempts, validation score), language study (strategy notes per
variant) and the format drill on the real sample turns, prep spend. Spend: prep keys only, about $0.01-0.03 a game.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from sweep import game_list, make_arena  # noqa: E402


def parse(text: str) -> dict:
    g = lambda pat, default=None: (re.findall(pat, text) or [default])[-1]  # noqa: E731
    r = {"ready_s": g(r"ready after ([\d.]+) s"), "digest": g(r"digest ok=(\w+)"),
         "flags": g(r"digest ok=\w+ (forbidden=\d+ talk=\w+ simul=\w+ hidden=\w+)", ""),
         "sim": "talk game" if "no simulator (language game)" in text else
                "accepted" if "simulator accepted" in text else
                "REJECTED" if re.search(r"no simulator passed|giving up|pipeline failed", text) else "?",
         "sim_attempts": len(re.findall(r"simgen attempt \d+: ok=", text)),
         "sim_score": g(r"simulator accepted \(score ([\d.]+)"),
         "best_score": g(r"best score ([\d.]+)"),
         "studies": g(r"talk.prepare: (\d+) studies"), "drill": g(r"drill (\d+/\d+) ok"),
         "spend": g(r"done, spend so far \$([\d.]+)"),
         "refused": len(re.findall(r"refus|can't|cannot", text, re.I))}
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bot", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--seconds", type=float, default=600)
    a = ap.parse_args()
    run = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "out" / f"prep-{run}"
    out.mkdir(parents=True, exist_ok=True)
    arena = make_arena(out)
    games = game_list(arena)
    if a.only:
        keys = [k.strip() for k in a.only.split(",") if k.strip()]
        games = [x for x in games if any(x[1] == k or x[1].startswith(k) for k in keys)]
    rows = []
    for _, gid in games:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "arena", "prep", "--bot", a.bot, "--game", gid, "--name", "path-to-glory",
                            "--env-file", str(ROOT / "dev.env"), "--seconds", str(a.seconds)],
                           cwd=arena, capture_output=True, text=True, encoding="utf-8", errors="replace")
        text = p.stdout + p.stderr
        for f in (arena / "results").rglob(f"*{gid}*.log"):
            text += "\n" + f.read_text(encoding="utf-8", errors="replace")
        (out / f"{gid}.txt").write_text(text, encoding="utf-8")
        r = {"game": gid, "wall_s": time.time() - t0, **parse(text)}
        rows.append(r)
        print(f"{gid:13s} ready {str(r['ready_s']):>6s}s/{a.seconds:.0f}  digest {r['digest']}  {r['flags']:44s} "
              f"sim {r['sim']:9s} (tries {r['sim_attempts']}, score {r['sim_score'] or r['best_score']})  "
              f"study {r['studies']}  drill {r['drill']}  ${r['spend']}", flush=True)
    (out / "prep.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    used = [float(r["ready_s"]) for r in rows if r["ready_s"]]
    if used:
        print(f"\nprep time used: avg {sum(used) / len(used):.0f}s, max {max(used):.0f}s of {a.seconds:.0f}s "
              f"({sum(used) / len(used) / a.seconds * 100:.0f}% on average)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

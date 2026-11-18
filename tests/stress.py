"""Stress test: several of our specialists alive at once, with the arena's container limits (rule 5.13).

    python tests/stress.py --image pg-t1-freeze:test                       # 3 practice games in parallel
    python tests/stress.py --image <img> --games tic_tac_toe,pig,guess_number,games:hush,games:fact_check

For each game it runs one arena-lite `practice --prep` match (our image vs tests/opponents/nasty_bot.py (random moves, hostile chat)) **at the same time**,
so prep, search and idle containers overlap like in a tournament. arena-lite starts every container with
`--cpus 2 --memory 2g --pids-limit 256`. Meanwhile it samples `docker stats` every 2 s and listens for OOM kills.

Reports per container: peak and lowest memory, peak CPU, and per match: result, warnings, timeouts, crashes.
Spends from the dev key in dev.env (our bot only; the random bot uses no model): roughly a few cents per game.
Keep the game count low: all bots share the key's 60 requests/minute (a 429 is also a useful test).
Exit code 1 on an OOM kill, a crash, a timeout or a container above --mem-warn MiB.
"""
from __future__ import annotations

import argparse
import shutil
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

PY = "python3" if shutil.which("python3") else "python"  # Windows has no python3 (path-to-glory-2, 20:35)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARENA = ROOT / "arena-lite"

UNITS = {"b": 1 / 2**20, "kib": 1 / 1024, "kb": 1 / 1024, "mib": 1, "mb": 1, "gib": 1024, "gb": 1024}


def mib(text: str) -> float:
    m = re.match(r"([\d.]+)\s*([a-zA-Z]+)", text.strip())
    return float(m.group(1)) * UNITS.get(m.group(2).lower(), 1) if m else 0.0


class Sampler(threading.Thread):
    """Samples `docker stats` for the containers of one image."""

    def __init__(self, image: str, every: float = 2.0):
        super().__init__(daemon=True)
        self.image, self.every = image, every
        self.stop = threading.Event()
        self.seen: dict[str, dict] = {}  # container name -> {"max": MiB, "min": MiB, "cpu": %, "n": samples}

    def run(self):
        while not self.stop.is_set():
            ids = subprocess.run(["docker", "ps", "-q", "--filter", f"ancestor={self.image}"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.split()
            if ids:
                out = subprocess.run(["docker", "stats", "--no-stream", "--format",
                                      "{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}", *ids],
                                     capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
                for line in out.splitlines():
                    parts = line.split("\t")
                    if len(parts) != 3:
                        continue
                    name, mem, cpu = parts
                    used = mib(mem.split("/")[0])
                    pct = float(cpu.strip().rstrip("%") or 0)
                    s = self.seen.setdefault(name, {"max": 0.0, "min": 1e9, "cpu": 0.0, "n": 0})
                    s["max"], s["min"] = max(s["max"], used), min(s["min"], used)
                    s["cpu"], s["n"] = max(s["cpu"], pct), s["n"] + 1
            self.stop.wait(self.every)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--games", default="tic_tac_toe,pig,guess_number",
                    help="comma list; prefix games: for arena-lite/arena/games (e.g. games:hush)")
    ap.add_argument("--opponent", default=PY + " ../tests/opponents/nasty_bot.py mix",
                    help="opponent command, run from arena-lite/ (default: random legal moves + hostile chat, fills free-form actions)")
    ap.add_argument("--env-file", default=str(ROOT / "dev.env"))
    ap.add_argument("--best-of", type=int, default=3)
    ap.add_argument("--mem-warn", type=float, default=1024, help="flag containers above this many MiB")
    a = ap.parse_args()

    run = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "out" / f"stress-{run}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"stress {a.image}: {a.games} in parallel; output in {out}", flush=True)

    oom: list[str] = []
    ev = subprocess.Popen(["docker", "events", "--filter", "event=oom", "--format", "{{.Actor.Attributes.name}}"],
                          stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    threading.Thread(target=lambda: [oom.append(x.strip()) for x in ev.stdout], daemon=True).start()
    sampler = Sampler(a.image)
    sampler.start()

    procs = {}
    t0 = time.time()
    for g in a.games.split(","):
        games_dir, gid = ("games", g.split(":", 1)[1]) if g.startswith("games:") else ("practice", g)
        mid = f"stress-{run}-{gid}"
        cmd = [sys.executable, "-m", "arena", "practice", "--bot-a", a.image, "--bot-b",
               a.opponent, "--env-file", a.env_file, "--game", gid,
               "--games-dir", games_dir, "--prep", "--best-of", str(a.best_of), "--id", mid, "--name-a", "path-to-glory"]
        log = open(out / f"{gid}.txt", "w")
        procs[gid] = (subprocess.Popen(cmd, cwd=ARENA, stdout=log, stderr=subprocess.STDOUT), mid, log)

    for gid, (p, _, log) in procs.items():
        p.wait()
        log.close()
    wall = time.time() - t0
    time.sleep(2)
    sampler.stop.set()
    ev.terminate()

    bad = []
    print(f"\nmatches ({wall:.0f}s wall):", flush=True)
    for gid, (p, mid, _) in procs.items():
        text = (out / f"{gid}.txt").read_text(encoding="utf-8", errors="replace")
        final = [l for l in text.splitlines() if l.startswith("final:")]
        problems = []
        try:
            replay = json.loads((ARENA / "results" / "practice" / "matches" / f"{mid}.json").read_text(encoding="utf-8"))
            for gm in replay.get("games") or []:
                why = str((gm.get("result") or {}).get("reason", ""))
                if "path-to-glory" in why and any(w in why for w in ("crash", "exited", "time", "illegal", "invalid", "ready")):
                    problems.append(f"ours: {why[:90]}")
                elif any(w in why for w in ("crash", "exited", "time")):
                    problems.append(f"opponent: {why[:60]}")
            ours = theirs = 0
            for gm in replay.get("games") or []:
                seats, w = gm.get("seats") or [], gm.get("warnings") or {}
                for seat, name in enumerate(seats):
                    n = w.get(str(seat), w.get(seat, 0)) if isinstance(w, dict) else (w[seat] if seat < len(w) else 0)
                    n = n if isinstance(n, int) else len(n or [])
                    if name == "path-to-glory":
                        ours += n
                    else:
                        theirs += n
            if ours:
                problems.append(f"ours: {ours} warning(s) (tie-break #1, 10.4)")
            if theirs:
                problems.append(f"(opponent warnings: {theirs})")
        except (OSError, ValueError) as exc:
            problems.append(f"no replay ({type(exc).__name__})")
        line = final[0] if final else f"no final line (exit {p.returncode})"
        print(f"  {gid:14s} {line}" + (f"   !! {', '.join(problems)}" if problems else ""), flush=True)
        if p.returncode != 0 or any(x.startswith(("ours", "opponent:")) and "warning" not in x for x in problems):
            bad.append(gid)

    print("\ncontainers (docker stats every 2 s; limit 2048 MiB, 200% CPU):", flush=True)
    for name, s in sorted(sampler.seen.items()):
        flag = "  !! above --mem-warn" if s["max"] > a.mem_warn else ""
        print(f"  {name:40s} mem peak {s['max']:7.1f} MiB  low {s['min']:6.1f} MiB  cpu peak {s['cpu']:5.0f}%  ({s['n']} samples){flag}")
        if flag:
            bad.append(name)
    if not sampler.seen:
        print("  (no container seen: is --image exactly the tag arena-lite runs?)")
    if oom:
        print(f"\n!! OOM killed: {oom}")
        bad += oom
    total_peak = sum(s["max"] for s in sampler.seen.values())
    print(f"\nsum of per-container peaks: {total_peak:.0f} MiB across {len(sampler.seen)} container(s)")
    (out / "summary.json").write_text(json.dumps({"containers": sampler.seen, "oom": oom, "bad": bad, "wall_s": wall}, indent=1), encoding="utf-8")
    print("RESULT:", "FAILED " + str(bad) if bad else "OK", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

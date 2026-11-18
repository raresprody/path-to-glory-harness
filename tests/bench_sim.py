"""Search speed benchmark: playouts per move of mesh's sim worker, CPython vs PyPy, in a linux/amd64 container
with the arena's limits (2 CPUs, 2 GB). Zero spend: no model calls, it reuses simulators a prep run generated.

    python tests/bench_sim.py                                  # newest tests/out/prep-* run, mesh/ from HEAD
    python tests/bench_sim.py --ref origin/denis/t2-crosscheck --seconds 5 --reps 3

For each game with a generated simulator it sends the worker the game's real sample turn (from prep_msg.json) as a
`candidates` request of --seconds and reads back `playouts`. The opening book is left out so every request searches.
Runs the same thing under python:3.13-slim (what we ship) and pypy:3.10-slim, one worker each (the arena gives
2 CPUs; Denis's parallel search multiplies both columns about the same).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGES = {"cpython": ("python:3.13-slim", "python"), "pypy": ("pypy:3.10-slim", "pypy3")}

INNER = r'''
import importlib.util, json, os, subprocess, sys, time
interp, seconds, reps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
out = {}
for game in sorted(os.listdir("/sims")):
    d = os.path.join("/sims", game)
    try:
        msg = json.load(open(os.path.join(d, "prep_msg.json"), encoding="utf-8"))
        v = next(v for v in msg["variants"] if v.get("sample_turn"))
        spec = importlib.util.spec_from_file_location("s", os.path.join(d, "sim.py"))
        mod = importlib.util.module_from_spec(spec); so = sys.stdout; sys.stdout = sys.stderr
        spec.loader.exec_module(mod); sys.stdout = so
        cfg = mod.CONFIGS[v["label"]]
    except Exception as exc:
        out[game] = {"error": f"setup: {type(exc).__name__}: {exc}"}; continue
    p = subprocess.Popen([interp, "-m", "mesh.simworker", os.path.join(d, "sim.py")], cwd="/bot",
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    hello = json.loads(p.stdout.readline() or "{}")
    runs = []
    for r in range(reps):
        t = v["sample_turn"]
        req = {"cmd": "candidates", "id": r, "observation": t["observation"], "seat": 0, "config": cfg,
               "legal": t.get("legal_actions"), "seconds": seconds, "seed": r}
        t0 = time.time(); p.stdin.write(json.dumps(req) + "\n"); p.stdin.flush()
        rep = json.loads(p.stdout.readline() or "{}")
        runs.append({"playouts": rep.get("playouts"), "secs": round(time.time() - t0, 2), "ok": rep.get("ok"),
                     "informative": rep.get("informative"), "err": str(rep.get("error") or rep.get("why") or "")[:80]})
    p.kill()
    out[game] = {"loaded": hello.get("loaded"), "runs": runs}
print(json.dumps(out))
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-dir", help="a tests/out/prep-* folder with generated sims (default: newest with sims)")
    ap.add_argument("--ref", default="HEAD", help="git ref whose mesh/ (sim worker) to benchmark")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    preps = [Path(a.prep_dir)] if a.prep_dir else sorted((HERE / "out").glob("prep-*"), reverse=True)
    sims_root = next((p / "arena-lite" / "memory" / "path-to-glory" for p in preps
                      if list((p / "arena-lite" / "memory").rglob("sim.py"))), None)
    if sims_root is None:
        raise SystemExit("no generated simulators found; run tests/prep_report.py first")
    work = HERE / "out" / ("bench-" + time.strftime("%Y%m%d-%H%M%S"))
    sims = work / "sims"
    for sim in sims_root.rglob("sim.py"):  # newest per game; no book.json, so every request searches
        game = sim.relative_to(sims_root).parts[0]
        dst = sims / game
        if dst.exists() and (dst / "sim.py").stat().st_mtime >= sim.stat().st_mtime:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(sim, dst / "sim.py")
        pm = sim.parent / "prep_msg.json"
        if pm.exists():
            shutil.copy(pm, dst / "prep_msg.json")
    bot = work / "bot"
    bot.mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", a.ref, "mesh"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(bot)], input=tar, check=True)
    (work / "inner.py").write_text(INNER, encoding="utf-8")

    results = {}
    for name, (image, interp) in IMAGES.items():
        print(f"running {name} ({image}) ...", flush=True)
        cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", "--cpus", "2", "--memory", "2g", "--pids-limit", "256",
               "-v", f"{bot}:/bot:ro", "-v", f"{sims}:/sims:ro", "-v", f"{work}:/work:ro", image,
               interp, "/work/inner.py", interp, str(a.seconds), str(a.reps)]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        try:
            results[name] = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            print(f"  failed: {p.stderr.strip()[-300:]}")
            results[name] = {}
    (work / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\nsearch speed, one worker, mesh from {a.ref} (median of {a.reps} requests of {a.seconds:.0f} s; playouts per second,"
          " because the search may stop early once one move dominates):")
    print(f"{'game':14s} {'CPython/s':>10s} {'PyPy/s':>10s} {'speed-up':>9s}  {'time to answer (CPy / PyPy)':>28s}")
    for game in sorted(set(results.get("cpython", {})) | set(results.get("pypy", {}))):
        def stat(name):
            r = results.get(name, {}).get(game, {})
            runs = [x for x in r.get("runs", []) if isinstance(x.get("playouts"), int) and x["playouts"] > 0 and x["secs"] > 0]
            rates = sorted(x["playouts"] / x["secs"] for x in runs)
            secs = sorted(x["secs"] for x in runs)
            return (rates[len(rates) // 2] if rates else None), (secs[len(secs) // 2] if secs else None), r
        c, cs, rc = stat("cpython")
        p_, ps, rp = stat("pypy")
        sp = f"{p_ / c:.1f}x" if c and p_ else "-"
        note = rc.get("error") or rp.get("error") or ("no search (0 playouts)" if c is None and p_ is None else "")
        t = f"{cs:.1f}s / {ps:.1f}s" if cs and ps else ""
        print(f"{game:14s} {c and round(c) or '-':>10} {p_ and round(p_) or '-':>10} {sp:>9s}  {t:>28s}  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

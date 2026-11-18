"""Zero-spend: every simulator in mesh/known/ must install through the normal prep path (simgen.install_known:
same validation as a generated simulator, on real sample turns taken from tests/replays/) and answer one real
position through the engine. Run from the repo root, or inside the image:  python tools/known_install_check.py"""
import glob
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MESH_KNOWN_DIR", os.path.join(ROOT, "mesh", "known"))
from mesh import simgen  # noqa: E402

REPLAYS = os.path.join(ROOT, "tests", "replays")
SIMULTANEOUS = {"prisoners_dilemma", "rps", "colonel_blotto", "wordle_duel", "bulls_and_cows"}


def prep_for(game_id):
    variants, first = {}, {}
    for f in sorted(glob.glob(os.path.join(REPLAYS, "rr1-*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for g in d["games"]:
            if g["game_id"] != game_id:
                continue
            v = g["variant"] if isinstance(g["variant"], str) else g["variant"].get("label")
            st = [s for s in g["timeline"] if (s.get("actions") or {}).get("0")]
            if v in variants or len(st) < 3:
                continue
            obs = st[2]["actions"]["0"]["observation"]
            variants[v] = {"label": v, "rules": g["rules"], "sample_turn": {"observation": obs, "legal_actions": None}}
            first[v] = obs
    return {"game_id": game_id, "variants": list(variants.values())}, first


bad = 0
games = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(ROOT, "mesh", "known", "*.py")) if not p.endswith("__init__.py"))
for gid in games:
    prep, first = prep_for(gid)
    folder = os.path.join(tempfile.mkdtemp(), gid, "h")
    t = time.time()
    eng = simgen.install_known(prep, {"example_actions": []}, folder, seconds=20)
    if eng is None:
        bad += 1
        print(f"FAIL {gid}: not installed ({len(prep['variants'])} variants)")
        continue
    line = f"ok   {gid}: {len(prep['variants'])} variants, installed in {time.time() - t:.1f}s, helpers {eng.report.get('helpers')}"
    for v, obs in first.items():
        t = time.time()
        c = eng.candidates({"observation": obs, "legal_actions": None}, {"seat": 0, "variant": {"label": v}},
                           deadline=time.monotonic() + 3.0, simultaneous=gid in SIMULTANEOUS)
        if not c:
            bad += 1
            line += f"\n     FAIL [{v}] no candidates: {c!r}"
        else:
            line += f"\n     [{v}] {time.time() - t:.1f}s -> {c[0]['action']!r} value={c[0].get('value')} p={c[0].get('p')} {c[0].get('note', '')}"
    eng.close()
    print(line, flush=True)
print("RESULT:", "OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)

"""Dev-key test of the prep simulator pipeline on a public game, with rules and a real sample turn taken from
tests/replays/: digest -> simgen.build (generate, validate, check against real games, repair) -> report.
Costs a few cents (generation calls only, no match).   python tools/prep_sim_check.py <game_id> [seconds=330]"""
import glob
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mesh import llm, simgen  # noqa: E402
from mesh.digest import make_digest  # noqa: E402

gid = sys.argv[1]
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 330.0
variants = {}
for f in sorted(glob.glob(os.path.join(ROOT, "tests", "replays", "rr1-*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    for g in d["games"]:
        if g["game_id"] != gid:
            continue
        v = g["variant"] if isinstance(g["variant"], str) else g["variant"].get("label")
        st = [s for s in g["timeline"] if (s.get("actions") or {}).get("0")]
        if v not in variants and len(st) >= 3:
            variants[v] = {"label": v, "rules": g["rules"],
                           "sample_turn": {"observation": st[2]["actions"]["0"]["observation"], "legal_actions": None}}
prep = {"game_id": gid, "variants": list(variants.values())}
rules = "\n\n".join(f"=== VARIANT: {v['label']} ===\n{v['rules']}" for v in prep["variants"])
t0 = time.time()
digest = make_digest(rules, timeout=100, thinking_budget=2000) or {}
print(f"digest ok={digest.get('ok')} hidden={digest.get('hidden_information')} simultaneous={digest.get('simultaneous')} "
      f"talk={digest.get('talk_is_move')} ({time.time() - t0:.0f}s)", flush=True)
folder = os.path.join(tempfile.mkdtemp(), gid, "h")
eng = simgen.build(prep, digest, folder, time.monotonic() + budget)
rep = json.load(open(os.path.join(folder, "report.json"), encoding="utf-8")) if os.path.exists(os.path.join(folder, "report.json")) else {}
print(f"RESULT {gid}: engine={'yes' if eng else 'NO'} verified={rep.get('known')} attempts={rep.get('attempts')} "
      f"replay={rep.get('replay')} helpers={rep.get('helpers')} in {time.time() - t0:.0f}s, spend ${llm.cost_usd():.3f}, "
      f"calls {llm.USAGE['calls']}; folder {folder}")
if eng and "--eval" in sys.argv:
    t1 = time.time()
    ev = simgen.add_evaluation(eng, prep, digest, time.monotonic() + 260)
    print(f"EVALUATION {gid}: used={ev.get('used')} check={ev.get('check')} duel={ev.get('duel')} in {time.time() - t1:.0f}s, "
          f"spend ${llm.cost_usd():.3f}")
if eng:
    eng.close()

"""Build mesh/experience/<game_id>.json.gz from public arena replays (tests/replays/, rule 10.7: replays and rules of
played games are public). This is DATA, not game code: a few real games per variant (what each seat saw, what it
played, how the game ended) plus the arena's own result sentences. In prep the harness checks the simulator Haiku
wrote against these real games and sends the differences back for repair (mesh/simgen.py, mesh/simworker.replay).

    python tools/build_experience.py            # all games found in tests/replays
"""
import glob
import gzip
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "mesh", "experience")
BAD = ("illegal", "timed out", "crash", "forfeit", "did not")
PER_VARIANT = 3

by_game = {}
for f in sorted(glob.glob(os.path.join(ROOT, "tests", "replays", "rr1-*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    for g in d["games"]:
        if g.get("status") != "done":
            continue
        res = g.get("result") or {}
        reason = res.get("reason") or ""
        label = g["variant"] if isinstance(g["variant"], str) else g["variant"].get("label")
        steps = []
        for st in g["timeline"]:
            acts = {s: [a["observation"], a["action"]] for s, a in (st.get("actions") or {}).items()
                    if isinstance(a, dict) and a.get("action") is not None and a.get("observation")}
            if acts:
                steps.append(acts)
        if len(steps) < 2:
            continue
        by_game.setdefault(g["game_id"], []).append(
            {"label": label, "winner_seat": res.get("winner_seat"), "reason": reason,
             "bad_end": any(k in reason for k in BAD), "steps": steps, "match": d["match_id"]})

os.makedirs(OUT, exist_ok=True)
for gid, games in sorted(by_game.items()):
    clean = [g for g in games if not g["bad_end"]]
    reasons = list(dict.fromkeys(g["reason"] for g in clean))[:14]
    picked = []
    for label in sorted({g["label"] for g in clean}):
        pool = sorted((g for g in clean if g["label"] == label), key=lambda g: len(g["steps"]))
        chosen, seen = [], set()
        for g in pool:  # one game per outcome first (seat 0 wins, seat 1 wins, draw), shortest first
            if g["winner_seat"] not in seen:
                seen.add(g["winner_seat"])
                chosen.append(g)
        for g in pool:
            if len(chosen) >= PER_VARIANT:
                break
            if g not in chosen:
                chosen.append(g)
        picked += chosen[:PER_VARIANT]
    path = os.path.join(OUT, gid + ".json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"game_id": gid, "source": "public replays of the round robin, 15 Nov 2026", "reasons": reasons,
                   "games": picked}, fh)
    print(f"{gid}: {len(picked)} games, {sum(len(g['steps']) for g in picked)} steps, {len(reasons)} result sentences, "
          f"{os.path.getsize(path) // 1024} kB")

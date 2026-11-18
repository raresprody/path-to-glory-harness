"""Zero-spend engine duel: two search settings play each other on a generated simulator (no model calls).

    python tools/engine_duel.py <path to sim.py> <variant label> <games> <seconds per move> '<json A>' '<json B>'
e.g. '{"greedy": false}' vs '{"greedy": true}'. The simulator is the referee (it passed validation in prep).
"""
import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh import simworker as w  # noqa: E402

path, label, games, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
cfg_a, cfg_b = json.loads(sys.argv[5]), json.loads(sys.argv[6])
mod = w.load(path)
cfg = mod.CONFIGS[label]
score = {"A": 0.0, "B": 0.0}
for g in range(games):
    a_seat = g % 2  # alternate seats
    st = mod.initial_state(copy.deepcopy(cfg), 1000 + g)
    rng = random.Random(g)
    n = 0
    while not mod.is_over(st) and n < 500:
        acts = {}
        for s in mod.to_move(st):
            setting = cfg_a if s == a_seat else cfg_b
            req = dict(setting, observation=mod.observation(st, s), seat=s, config=cfg, legal=None, seconds=secs)
            rep = w.candidates(mod, req)
            c = rep.get("candidates") or []
            acts[s] = c[0]["action"] if c else rng.choice(mod.legal_actions(st, s))
        st = mod.apply(st, acts, rng)
        n += 1
    win = mod.winner(st)
    if win is None:
        score["A"] += 0.5
        score["B"] += 0.5
    else:
        score["A" if win == a_seat else "B"] += 1
    print(f"game {g + 1}: winner {'draw' if win is None else ('A' if win == a_seat else 'B')}  running {score}", flush=True)

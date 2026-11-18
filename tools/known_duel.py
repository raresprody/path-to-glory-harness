"""Zero-spend strength check: our engine (mesh.simgen.Engine on a replay-checked simulator, same code path as in a
match: observation text -> candidates) against a simple code opponent that takes a win in one, blocks a loss in one
and otherwise moves at random. No model calls.

    python tools/known_duel.py <game_id> "<variant label>" <games> <seconds per move>
"""
import copy
import importlib.util
import os
import random
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MESH_KNOWN_DIR", os.path.join(ROOT, "mesh", "known"))
from mesh import simgen  # noqa: E402

gid, label, games, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
path = os.path.join(ROOT, "mesh", "known", gid + ".py")
spec = importlib.util.spec_from_file_location("duelsim", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.CONFIGS[label]
folder = os.path.join(tempfile.mkdtemp(), gid, "h")
eng = simgen.install_known({"game_id": gid, "variants": [{"label": label, "rules": ""}]}, {}, folder, seconds=8)
assert eng is not None, "known simulator did not install"
rng = random.Random(5)


def simple(state, seat):
    """Win in one, else block the opponent's win in one (sequential games), else random."""
    acts = list(mod.legal_actions(state, seat))
    for a in acts:
        s2 = mod.apply(state, {seat: a}, random.Random(1))
        if mod.is_over(s2) and mod.winner(s2) == seat:
            return a
    safe = []
    for a in acts:
        s2 = mod.apply(state, {seat: a}, random.Random(1))
        bad = False
        if not mod.is_over(s2) and mod.to_move(s2) == [1 - seat]:
            for b in mod.legal_actions(s2, 1 - seat):
                s3 = mod.apply(s2, {1 - seat: b}, random.Random(1))
                if mod.is_over(s3) and mod.winner(s3) == 1 - seat:
                    bad = True
                    break
        if not bad:
            safe.append(a)
    return rng.choice(safe or acts)


score = {"engine": 0.0, "simple": 0.0}
t0 = time.time()
for g in range(games):
    us = g % 2
    st = mod.initial_state(copy.deepcopy(cfg), 1000 + g)
    moves = 0
    while not mod.is_over(st):
        acting = {}
        for seat in mod.to_move(st):
            if seat == us:
                c = eng.candidates({"observation": mod.observation(st, seat), "legal_actions": None},
                                   {"seat": seat, "variant": {"label": label}}, deadline=time.monotonic() + secs)
                acting[seat] = c[0]["action"] if c else rng.choice(mod.legal_actions(st, seat))
            else:
                acting[seat] = simple(st, seat)
        st = mod.apply(st, acting, rng)
        moves += 1
    w = mod.winner(st)
    score["engine"] += 1 if w == us else 0.5 if w is None else 0
    score["simple"] += 1 if w == 1 - us else 0.5 if w is None else 0
    print(f"game {g + 1}: engine seat {us}, winner {w!r}, {moves} moves, score so far {score}", flush=True)
eng.close()
print(f"RESULT {gid} [{label}] {secs}s/move: engine {score['engine']} - {score['simple']} simple ({time.time() - t0:.0f}s)")

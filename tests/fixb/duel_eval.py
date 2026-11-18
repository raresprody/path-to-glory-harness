"""Fix b offline: today's engine vs engine + Haiku-written evaluate() with short rollouts. Zero spend.
    pypy3 duel_eval.py <name> <variant label> <games> <seconds per move> [cutoff]"""
import sys, copy, random, time, inspect, math
sys.path.insert(0, "/bot")
from mesh import simworker as w
name, label, games, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
CUT = int(sys.argv[5]) if len(sys.argv) > 5 else 12
mod = w.load(f"/sims/{name}.py"); cfg = mod.CONFIGS[label]
ns = dict(vars(mod))  # the eval lives next to the simulator: its helpers are in scope
exec(open(f"/sims/{name}_eval.py").read(), ns)
EV = ns["evaluate"]

# ---- gate: never crashes, returns floats in [-1, 1], agrees with the real winner on finished games
rng = random.Random(0); bad = agree = fin = n = 0
for g in range(30):
    st = mod.initial_state(copy.deepcopy(cfg), 500 + g); k = 0
    while not mod.is_over(st) and k < 400:
        for s in (0, 1):
            try:
                v = EV(st, s); n += 1
                if not isinstance(v, (int, float)) or not -1.0001 <= v <= 1.0001: bad += 1
            except Exception: bad += 1
        st = w.step(mod, st, rng); k += 1
    win = mod.winner(st) if mod.is_over(st) else None
    if win is not None:
        fin += 1
        try: agree += EV(st, win) > 0
        except Exception: pass
print(f"[{name}] gate: {n} calls, {bad} bad; agrees with the winner on {agree}/{fin} finished games", flush=True)
if bad or agree < 0.7 * max(1, fin):
    print(f"[{name}] GATE FAILED: eval would be dropped in prep, today's engine kept", flush=True)

# ---- patched MCTS: rollouts stop after CUT plies, leaf scored by evaluate()
src = inspect.getsource(w.mcts)
old = """            st, _ = rollout(mod, st, rng, greedy=greedy)
            w = mod.winner(st) if mod.is_over(st) else None
            res = {0: 0.0, 1: 0.0} if w is None else {w: 1.0, 1 - w: -1.0}"""
new = """            st, _ = rollout(mod, st, rng, cap=_CUT, greedy=greedy)
            if mod.is_over(st):
                w = mod.winner(st)
                res = {0: 0.0, 1: 0.0} if w is None else {w: 1.0, 1 - w: -1.0}
            else:
                v = max(-1.0, min(1.0, float(_EV(st, 0))))
                res = {0: v, 1: -v}"""
assert old in src
src = src.replace(old, new).replace("def mcts(", "def mcts_eval(")
w.__dict__.update(_CUT=CUT, _EV=EV)
exec(src, w.__dict__)
plain, evalm = w.mcts, w.mcts_eval

score = {"eval": 0.0, "plain": 0.0}; po = {"eval": [], "plain": []}
for g in range(games):
    e_seat = g % 2
    st = mod.initial_state(copy.deepcopy(cfg), 1000 + g); r = random.Random(g); k = 0
    while not mod.is_over(st) and k < 400:
        acts = {}
        for s in mod.to_move(st):
            who = "eval" if s == e_seat else "plain"
            w.mcts = evalm if who == "eval" else plain
            rep = w.candidates(mod, dict(observation=mod.observation(st, s), seat=s, config=cfg, legal=None, seconds=secs, nobook=True))
            c = rep.get("candidates") or []
            if rep.get("playouts"): po[who].append(rep["playouts"])
            acts[s] = c[0]["action"] if c else r.choice(mod.legal_actions(st, s))
        st = mod.apply(st, acts, r); k += 1
    win = mod.winner(st) if mod.is_over(st) else None
    if win is None: score["eval"] += .5; score["plain"] += .5
    else: score["eval" if win == e_seat else "plain"] += 1
    print(f"[{name}] game {g + 1}: {'draw' if win is None else ('EVAL' if win == e_seat else 'plain')} wins ({k} plies)  {score}", flush=True)
med = lambda x: sorted(x)[len(x) // 2] if x else 0
print(f"[{name}] FINAL eval {score['eval']} - plain {score['plain']}; playouts/move median eval {med(po['eval'])} vs plain {med(po['plain'])}", flush=True)

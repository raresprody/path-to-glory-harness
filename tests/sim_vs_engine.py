"""Did Haiku understand the rules? Compare each simulator mesh generated in prep with the REAL game engine.

    python tests/sim_vs_engine.py                      # newest tests/out/prep-*/ run
    python tests/sim_vs_engine.py --prep-dir tests/out/prep-20261114-153400 --games 60

Zero spend, no bot: plays random games with the real arena-lite engine (practice, team and holdout games) and, at
every decision, rebuilds the simulator's state from the real observation (state_from_observation) and checks:
  legal   the simulator's legal moves == the engine's list (when the engine gives a list), or the engine's chosen
          move is legal in the simulator (when the list is null)
  replay  open games only (nothing hidden): applying the real moves in the simulator gives the real winner
A simulator with score 1.0 in prep can still be wrong here: prep only checks it against itself and one sample turn.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(a) -> str:
    return " ".join(str(a).strip().lower().split())


def load_sim(path: Path):
    spec = importlib.util.spec_from_file_location(f"gensim_{path.parent.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.stdout = real_stdout
    return mod


def check_game(cls, sim, n_games: int) -> dict:
    out = {"decisions": 0, "legal_bad": 0, "parse_err": 0, "replay": 0, "replay_bad": 0, "examples": []}
    for vi, variant in enumerate(cls.variants()):
        label = variant.get("label", "")
        cfg = (getattr(sim, "CONFIGS", {}) or {}).get(label)
        if cfg is None:
            out["examples"].append(f"no CONFIGS entry for variant {label!r}")
            out["legal_bad"] += 1
            continue
        for seed in range(n_games):
            g, rng, history = cls(dict(variant), 1000 + seed), random.Random(seed), []
            open_game = cls.duplicate_safe
            try:
                start = sim.state_from_observation(g.observation(0), 0, cfg, random.Random(seed))
            except Exception as exc:  # noqa: BLE001
                out["parse_err"] += 1
                out["examples"].append(f"{label}: parse at start: {type(exc).__name__}: {exc}")
                continue
            steps = 0
            while not g.is_over() and steps < 400:
                steps += 1
                acts = {}
                for s in g.to_move():
                    real_legal = g.legal_actions(s)
                    a = g.normalize(s, g.random_action(s, rng) if hasattr(g, "random_action") else rng.choice(real_legal))
                    acts[s] = a
                    out["decisions"] += 1
                    try:
                        st = sim.state_from_observation(g.observation(s), s, cfg, random.Random(steps))
                        sim_legal = {norm(x) for x in sim.legal_actions(st, s)}
                    except Exception as exc:  # noqa: BLE001
                        out["parse_err"] += 1
                        if len(out["examples"]) < 6:
                            out["examples"].append(f"{label} step {steps}: {type(exc).__name__}: {str(exc)[:100]}")
                        continue
                    if real_legal is not None:
                        if sim_legal != {norm(x) for x in real_legal}:
                            out["legal_bad"] += 1
                            if len(out["examples"]) < 6:
                                miss = sorted({norm(x) for x in real_legal} - sim_legal)[:4]
                                extra = sorted(sim_legal - {norm(x) for x in real_legal})[:4]
                                out["examples"].append(f"{label} step {steps} seat {s}: sim misses {miss} / adds {extra}")
                    elif norm(a) not in sim_legal:
                        out["legal_bad"] += 1
                        if len(out["examples"]) < 6:
                            out["examples"].append(f"{label} step {steps} seat {s}: engine move {a!r} not legal in sim")
                history.append(acts)
                g.apply(acts)
            if open_game and g.is_over():
                out["replay"] += 1
                try:
                    st = start
                    for acts in history:
                        st = sim.apply(st, {k: v for k, v in acts.items()}, random.Random(0))
                    real_w = g.result().get("winner")
                    sim_w = sim.winner(st) if sim.is_over(st) else "not over"
                    if sim_w != real_w:
                        out["replay_bad"] += 1
                        if len(out["examples"]) < 6:
                            out["examples"].append(f"{label} seed {seed}: winner real {real_w} vs sim {sim_w}")
                except Exception as exc:  # noqa: BLE001
                    out["replay_bad"] += 1
                    if len(out["examples"]) < 6:
                        out["examples"].append(f"{label} replay: {type(exc).__name__}: {str(exc)[:100]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-dir", help="a tests/out/prep-* folder (default: newest)")
    ap.add_argument("--games", type=int, default=40, help="random games per variant")
    a = ap.parse_args()
    prep = Path(a.prep_dir) if a.prep_dir else sorted((HERE / "out").glob("prep-*"))[-1]
    arena = prep / "arena-lite"
    sys.path.insert(0, str(arena))
    from arena import registry  # the temp copy, with the holdout games in it
    sims = {p.parent.parent.name: p for p in sorted((arena / "memory").rglob("sim.py"), key=lambda p: p.stat().st_mtime)}
    if not sims:
        print(f"no sim.py under {arena / 'memory'}: the bot's /memory volume is empty here. With Colima/Docker Desktop on "
              "macOS only your home folder is shared with containers, so run from a repo inside $HOME (not /tmp).")
        return 1
    print(f"simulators from {prep.name}: {', '.join(sims)}\n")
    print(f"{'game':13s} {'decisions':>9s} {'legal ok':>9s} {'parse err':>9s} {'replays ok':>11s}")
    bad_games = 0
    for gid, path in sims.items():
        try:
            cls = registry.get(gid)
            r = check_game(cls, load_sim(path), a.games)
        except Exception as exc:  # noqa: BLE001
            print(f"{gid:13s} could not run: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=1)
            bad_games += 1
            continue
        d = max(1, r["decisions"])
        rep = f"{r['replay'] - r['replay_bad']}/{r['replay']}" if r["replay"] else "n/a (hidden)"
        print(f"{gid:13s} {r['decisions']:9d} {100 * (1 - r['legal_bad'] / d):8.1f}% {r['parse_err']:9d} {rep:>11s}")
        for e in r["examples"][:4]:
            print(f"{'':15s}- {e}")
        bad_games += bool(r["legal_bad"] or r["parse_err"] or r["replay_bad"])
    print(f"\n{len(sims) - bad_games}/{len(sims)} simulators agree with the real engine everywhere we checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())

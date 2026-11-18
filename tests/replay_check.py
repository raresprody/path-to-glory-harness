"""Zero-spend check of a simulator module against REAL arena replays (round robin of 15 Nov, /api/matches/<id>).

    python tests/replay_check.py <sim.py> <game_id> <folder with rr1-*.json> [--hidden] [--random-end] [-v]

For every game of <game_id> in the replays, and every move a seat made there:
  1. state_from_observation(the observation that seat saw) must not crash, and that seat must be in to_move;
  2. the move the seat really played (accepted by the arena) must be legal in the simulator;
  3. perfect-information games (default): applying the real move must give a state whose legal moves equal those
     parsed from the NEXT real observation, and is_over must be False until the last move;
  4. after the last move the simulator must say the game is over with the arena's real winner
     (--hidden: skipped, the parsed state holds sampled secrets; --random-end: is_over not required, only that
     a finished state names the right winner).
Prints one line per kind of failure with an example; exit code 0 only when there are none.
"""
from __future__ import annotations

import copy
import glob
import importlib.util
import json
import os
import random
import sys


def norm(a) -> str:
    return " ".join(str(a).strip().lower().split())


def load(path):
    spec = importlib.util.spec_from_file_location("simcheck", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def games(folder, game_id):
    for f in sorted(glob.glob(os.path.join(folder, "rr1-*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for g in d["games"]:
            if g["game_id"] == game_id and g.get("status") == "done":
                yield d["match_id"], g


def legal(mod, st, seat, action) -> bool:
    try:
        la = mod.legal_actions(st, seat)
    except Exception:  # noqa: BLE001
        la = None
    if la and norm(action) in {norm(x) for x in la}:
        return True
    f = getattr(mod, "is_legal_action", None)
    return bool(f and f(st, seat, action))


def main():
    path, game_id, folder = sys.argv[1], sys.argv[2], sys.argv[3]
    hidden, random_end, verbose = "--hidden" in sys.argv, "--random-end" in sys.argv, "-v" in sys.argv
    mod = load(path)
    fails, stats = {}, {"games": 0, "moves": 0, "ends_ok": 0}

    def fail(kind, where, detail):
        fails.setdefault(kind, []).append(f"{where}: {detail}")

    for mid, g in games(folder, game_id):
        label = g["variant"] if isinstance(g["variant"], str) else g["variant"].get("label")
        if label not in mod.CONFIGS:
            fail("variant missing in CONFIGS", mid, repr(label))
            continue
        cfg = mod.CONFIGS[label]
        stats["games"] += 1
        steps = [s for s in g["timeline"] if s.get("actions")]
        res = g.get("result") or {}
        real_winner = res.get("winner_seat")
        bad_end = any(k in (res.get("reason") or "") for k in ("illegal", "timed out", "crash", "forfeit"))
        for i, st_ in enumerate(steps):
            acts = {int(s): a for s, a in st_["actions"].items() if isinstance(a, dict) and a.get("action") is not None}
            where = f"{mid} g{g['index']} [{label}] step {st_.get('step')}"
            states = {}
            for seat, a in acts.items():
                stats["moves"] += 1
                try:
                    s0 = mod.state_from_observation(a["observation"], seat, copy.deepcopy(cfg), random.Random(7))
                    states[seat] = s0
                    if seat not in mod.to_move(s0):
                        fail("seat not in to_move", where, f"seat {seat}, to_move={mod.to_move(s0)}")
                    if not legal(mod, s0, seat, a["action"]) and not (bad_end and i == len(steps) - 1):
                        fail("real move not legal in sim", where, f"seat {seat} played {a['action']!r}; "
                             f"sim legal (first 12): {(mod.legal_actions(s0, seat) or [])[:12]}")
                except Exception as exc:  # noqa: BLE001
                    fail("state_from_observation/legal crashed", where, f"seat {seat}: {type(exc).__name__}: {exc}")
            if not states or bad_end:
                continue
            seat0 = sorted(states)[0]
            try:
                movers = mod.to_move(states[seat0])
                if set(movers) - set(acts):
                    continue  # the replay step lacks one mover's action (should not happen)
                nxt = mod.apply(states[seat0], {s: acts[s]["action"] for s in movers}, random.Random(11))
            except Exception as exc:  # noqa: BLE001
                fail("apply crashed", where, f"{type(exc).__name__}: {exc}")
                continue
            last = i == len(steps) - 1
            try:
                over = mod.is_over(nxt)
                if last:
                    if hidden:
                        continue
                    if not over and not random_end:
                        fail("game should be over after the last real move", where, f"real result: {res.get('reason')}")
                    elif over:
                        if mod.winner(nxt) != real_winner:
                            fail("wrong winner", where, f"sim {mod.winner(nxt)!r}, arena {real_winner!r} ({res.get('reason')})")
                        else:
                            stats["ends_ok"] += 1
                elif over and not random_end and not hidden:
                    fail("sim says over too early", where, f"after {acts}")
                elif not hidden and not over:
                    nacts = {int(s): a for s, a in steps[i + 1]["actions"].items() if isinstance(a, dict)}
                    for seat, a in nacts.items():
                        ref = mod.state_from_observation(a["observation"], seat, copy.deepcopy(cfg), random.Random(7))
                        want = sorted(norm(x) for x in (mod.legal_actions(ref, seat) or []))
                        got = sorted(norm(x) for x in (mod.legal_actions(nxt, seat) or []))
                        if want != got:
                            fail("apply() and the next real observation disagree on legal moves", where,
                                 f"seat {seat}: apply gives {len(got)}, observation gives {len(want)}; "
                                 f"only apply: {sorted(set(got) - set(want))[:6]}, only obs: {sorted(set(want) - set(got))[:6]}")
            except Exception as exc:  # noqa: BLE001
                fail("is_over/winner/next-state check crashed", where, f"{type(exc).__name__}: {exc}")
    print(f"{game_id}: {stats['games']} real games, {stats['moves']} real moves checked, {stats['ends_ok']} endings with the right winner")
    for kind, items in fails.items():
        print(f"FAIL [{len(items)}x] {kind}\n     e.g. {items[0][:600]}")
        if verbose:
            for it in items[1:8]:
                print("     " + it[:400])
    print("RESULT:", "OK" if not fails else "FAILED")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()

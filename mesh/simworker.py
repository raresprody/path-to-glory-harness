"""Runs a Haiku-generated simulator in its OWN process (rule-of-thumb safety: generated code can hang, eat memory or
print to stdout; none of that may touch the protocol process). Speaks JSON lines on stdin/stdout with the brain.

    python -m mesh.simworker <path to sim.py>

Commands (one JSON object per line):
  {"cmd": "validate", "samples": [...], "seconds": 40}        -> {"ok": bool, "score": float, "errors": [...], ...}
  {"cmd": "candidates", "observation": str, "seat": int, "config": dict, "legal": list|null,
   "seconds": float, "opp_bias": {action: weight}}            -> {"ok": bool, "candidates": [...], "agree": bool, ...}

The generated module must define (see GEN_SPEC in simgen.py):
  CONFIGS: dict[label -> config dict]
  initial_state(config, seed) -> state
  to_move(state) -> list[int]
  legal_actions(state, seat) -> list[str]
  apply(state, actions: dict[int, str], rng) -> state      (new state; may mutate a copy)
  winner(state) -> 0 | 1 | None                            (None = draw, or not over)
  is_over(state) -> bool
  observation(state, seat) -> str                          (the arena's text for that seat)
  state_from_observation(text, seat, config, rng) -> state (a full state consistent with what `seat` sees;
                                                            hidden parts sampled with rng)
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import math
import random
import re
import sys
import time
import traceback

_OUT = sys.stdout
sys.stdout = sys.stderr  # generated code may print: keep our channel clean

MAX_ROLLOUT = 400


def reply(obj):
    _OUT.write(json.dumps(obj, default=str) + "\n")
    _OUT.flush()


def natural(a):
    import re as _re
    return [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", str(a).lower())]


def _shape(a) -> str:
    """'guess 500' -> 'a 9': letters and digits collapsed, to compare action spellings with the rules' examples."""
    import re as _re
    x = _re.sub(r"[a-z]+", "a", norm(a))
    return _re.sub(r"[0-9]+", "9", x)


def _shape_ok(actions, examples) -> bool:
    ex = {_shape(e) for e in examples if isinstance(e, str) and e.strip()}
    return not ex or any(_shape(a) in ex for a in actions)


def norm(a) -> str:
    return " ".join(str(a).strip().lower().split())


def _no_network():
    """Generated code sees game content; it must never be able to send it anywhere (rule 7.11). The worker
    process gets no sockets at all, so this holds by construction, not only because the prompt says so."""
    import socket

    def refuse(*a, **k):
        raise OSError("network disabled in the simulator sandbox")
    socket.socket = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse


def load(path, name="gensim"):
    _no_network()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("CONFIGS", "initial_state", "to_move", "legal_actions", "apply", "winner", "is_over",
                 "observation", "state_from_observation"):
        if not hasattr(mod, name):
            raise AttributeError(f"generated module lacks `{name}`")
    return mod


def _err(exc) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    # frames of the generated module (sim.py, sim_try3.py, sim_independent.py...): never our own worker code
    where = [f"line {f.lineno} in {f.name}: {f.line}" for f in tb if not f.filename.endswith("simworker.py")][-3:]
    return f"{type(exc).__name__}: {exc} | " + " <- ".join(reversed(where))


POLICY = {"greedy": False}  # "win if you can" playouts: tested, no clear gain at our move times (off)


def step(mod, state, rng, greedy=None):
    movers = list(mod.to_move(state))
    acts = {}
    for s in movers:
        legal = mod.legal_actions(state, s)
        if not legal:  # too many to list (None) or a bug: the simulator's own random legal move, if it has one
            ra = getattr(mod, "random_action", None)
            if callable(ra):
                acts[s] = ra(state, s, rng)
                legal = [acts[s]]
                continue
            raise RuntimeError(f"seat {s} must move but legal_actions returned {legal!r} and there is no random_action")
        acts[s] = rng.choice(legal)
    use_greedy = POLICY["greedy"] if greedy is None else greedy
    if use_greedy and len(movers) == 1 and 1 < len(legal) <= 40:
        # "win if you can": both sides of the playout take an immediate win, which also makes the search
        # see the opponent's threats (a move that leaves the opponent a win now scores badly)
        s = movers[0]
        order = list(legal)
        rng.shuffle(order)
        for a in order:
            st = mod.apply(state, {s: a}, rng)
            if mod.is_over(st) and mod.winner(st) == s:
                return st
        return mod.apply(state, acts, rng)
    return mod.apply(state, acts, rng)


# "mcts+eval" search: playouts stop after this many plies and the position is scored by the prep-written
# evaluate() instead of being played out at random to the end (set per request in candidates()).
ROLLOUT_CUT = {"plies": None}
EVAL_CUT_PLIES = 12


def rollout(mod, state, rng, cap=MAX_ROLLOUT, greedy=None):
    n = 0
    while not mod.is_over(state) and n < cap:
        state = step(mod, state, rng, greedy)
        n += 1
    return state, n


def score_for(mod, state, seat) -> float:
    if not mod.is_over(state):
        return 0.0
    w = mod.winner(state)
    return 0.0 if w is None else (1.0 if w == seat else -1.0)


# ------------------------------------------------------------------ validation
def validate(mod, samples, seconds, examples=None):
    t_end = time.monotonic() + seconds
    errors, stats = [], {"games": 0, "terminated": 0, "roundtrip": 0, "roundtrip_ok": 0, "text_same": 0,
                         "sample_checked": 0, "sample_ok": 0, "max_len": 0,
                         "il_checked": 0, "il_ok": 0, "cw_checked": 0, "cw_ok": 0}
    has_il = callable(getattr(mod, "is_legal_action", None))
    has_cw = callable(getattr(mod, "could_win_now", None))
    labels = list(getattr(mod, "CONFIGS", {}) or {})
    if not labels:
        return {"ok": False, "score": 0.0, "errors": ["CONFIGS is empty"], "stats": stats}
    # 1. random self-play: never crashes, always terminates; render -> parse round trip keeps the legal moves
    # bad / played: per variant, so a simulator that fails only one variant can still serve the others (Rares 08:22:
    # A's Battleship failed only the salvo variant and Haiku then played all three)
    bad, played = set(), {}
    seed = 0
    while time.monotonic() < t_end - 2 and stats["games"] < 12 * len(labels):
        label = labels[seed % len(labels)]
        cfg = mod.CONFIGS[label]
        rng = random.Random(seed)
        seed += 1
        stats["games"] += 1
        played[label] = played.get(label, 0) + 1
        try:
            st = mod.initial_state(copy.deepcopy(cfg), seed)
            n = 0
            while not mod.is_over(st) and n < 1000:
                if n % 3 == 0:
                    for s in mod.to_move(st):
                        stats["roundtrip"] += 1
                        obs = mod.observation(st, s)
                        if not isinstance(obs, str) or not obs.strip():
                            raise RuntimeError("observation() returned an empty or non-string value")
                        st2 = mod.state_from_observation(obs, s, copy.deepcopy(cfg), random.Random(n))
                        r1 = list(mod.legal_actions(st, s))
                        r2 = list(mod.legal_actions(st2, s))
                        l1 = sorted(map(norm, r1))
                        l2 = sorted(map(norm, r2))
                        # a REPRESENTATIVE list: the same state gives a different sample on a second call (salvo
                        # combinations, Rares 08:22: A's Battleship never passed), or two big lists that differ a lot
                        # (review #10: a parser that drops one move on a big board must still fail)
                        resample = l1 != l2 and sorted(map(norm, mod.legal_actions(st, s))) != l1
                        huge = l1 != l2 and (resample or (len(l1) > 60 and len(set(l1) ^ set(l2)) > 0.2 * len(l1)))
                        ok_h = False
                        if huge and not callable(getattr(mod, "is_legal_action", None)):
                            ok_h = False  # a sampled list needs the exact legality test to be checked at all
                        elif huge:
                            # a representative list (too many moves to enumerate, e.g. several shots per turn: the
                            # t3 Battleship salvo simulators failed here at 94.99 %): the parsed state must accept the
                            # true state's moves through the exact legality check, and the other way round, instead of
                            # the two samples being identical
                            try:
                                step1 = max(1, len(r1) // 12)
                                step2 = max(1, len(r2) // 12)
                                # raw spellings (review #10), not the lower-cased ones
                                ok_h = bool(r2) and all(mod.is_legal_action(st2, s, x) for x in r1[::step1][:12])                                     and all(mod.is_legal_action(st, s, x) for x in r2[::step2][:12])
                            except Exception:  # noqa: BLE001
                                ok_h = False
                        if (huge and ok_h) or l1 == l2:
                            stats["roundtrip_ok"] += 1
                        elif bad.add(label) or len(errors) < 6:
                            errors.append(f"round trip changed legal moves for seat {s} (variant {label!r}). "
                                          f"Observation:\n{obs[:1200]}\nlegal before: {l1[:20]}\nlegal after parse: {l2[:20]}")
                        if mod.observation(st2, s) == obs:
                            stats["text_same"] += 1
                        _check_helpers(mod, st, st2, s, l1, rng, stats, errors, has_il, has_cw)
                st = step(mod, st, rng, greedy=False)
                n += 1
            stats["max_len"] = max(stats["max_len"], n)
            if mod.is_over(st):
                stats["terminated"] += 1
                w = mod.winner(st)
                if w not in (0, 1, None):
                    raise RuntimeError(f"winner() returned {w!r}; must be 0, 1 or None")
            elif bad.add(label) or len(errors) < 6:
                errors.append(f"variant {label!r}: game not over after 1000 random moves")
        except Exception as exc:  # noqa: BLE001
            bad.add(label)
            if len(errors) < 6:
                errors.append(f"self-play crashed (variant {label!r}, seed {seed}): {_err(exc)}")
    # 2. the arena's real sample turns: parse them and compare the legal moves
    for smp in samples or []:
        obs, legal, label = smp.get("observation", ""), smp.get("legal_actions"), smp.get("label")
        cfg = mod.CONFIGS.get(label) or mod.CONFIGS[labels[0]]
        seat = smp.get("seat", 0)
        stats["sample_checked"] += 1
        try:
            st = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), random.Random(1))
            mine = mod.legal_actions(st, seat)
            if legal is None and examples and mine and not _shape_ok(mine, examples):
                bad.add(label)
                if len(errors) < 8:
                    errors.append(f"Action spelling does not follow the rules' action format (variant {label!r}). "
                                  f"The rules' examples look like {examples[:6]}, your legal_actions look like {mine[:6]}. "
                                  "Spell every action exactly in the rules' format.")
            elif legal is None or sorted(map(norm, mine)) == sorted(map(norm, legal)):
                stats["sample_ok"] += 1
            elif bad.add(label) or len(errors) < 8:
                errors.append(f"REAL arena sample turn (variant {label!r}): parsed legal moves differ.\n"
                              f"Arena observation:\n{obs[:2000]}\nArena legal_actions: {legal[:30]}\nYours: {mine[:30]}")
            rendered = mod.observation(st, seat)
            if rendered.strip() != obs.strip() and len(errors) < 8:
                errors.append(f"(minor) observation(state_from_observation(sample)) differs from the arena text.\n"
                              f"Arena:\n{obs[:1200]}\nYours:\n{str(rendered)[:1200]}")
        except Exception as exc:  # noqa: BLE001
            bad.add(label)
            if len(errors) < 8:
                errors.append(f"REAL arena sample turn crashed state_from_observation (variant {label!r}): {_err(exc)}\n"
                              f"Arena observation:\n{obs[:2000]}")
    g = max(1, stats["games"])
    rt = stats["roundtrip_ok"] / max(1, stats["roundtrip"])
    sm = stats["sample_ok"] / max(1, stats["sample_checked"]) if stats["sample_checked"] else 1.0
    crash_free = 1.0 - sum(1 for e in errors if "crashed" in e) / g
    term = stats["terminated"] / g
    score = round(0.3 * term + 0.3 * rt + 0.3 * sm + 0.1 * crash_free, 3)
    ok = term >= 0.99 and rt >= 0.95 and sm >= 0.99 and stats["games"] >= 3 and not any("crashed" in e for e in errors)
    # optional helpers: used at play time only if they agreed with the simulator itself on every check
    helpers = {"is_legal_action": has_il and stats["il_checked"] >= 10 and stats["il_ok"] == stats["il_checked"],
               # huge action spaces (a hidden word) can't be enumerated to find the winning move, so 0 checks means
               # "unverified": the engine then only uses it as a filter that must leave at least one proposal
               "could_win_now": has_cw and stats["cw_ok"] == stats["cw_checked"]}
    out = {"ok": ok, "score": score, "errors": errors, "stats": stats, "helpers": helpers}
    if not ok and bad and bad <= set(labels):
        # the variants with no failure at all, each played at least twice: usable on their own (the brain switches
        # the engine off only for a game whose variant is not in the accepted configs; Haiku plays that one)
        good = [l for l in labels if l not in bad and played.get(l, 0) >= 2]
        if good:
            out.update(ok=True, ok_labels=good, failed_labels=sorted(bad))
    return out


def _check_helpers(mod, st, st2, s, legal_true, rng, stats, errors, has_il, has_cw):
    """is_legal_action must accept the simulator's own legal moves and reject obvious garbage; could_win_now must
    accept every move that really wins right now (checked on the true state, asked on the parsed one)."""
    sample = rng.sample(legal_true, min(4, len(legal_true))) if legal_true else []
    if has_il:
        for a in sample + ["@@not a move@@"]:
            stats["il_checked"] += 1
            try:
                got = bool(mod.is_legal_action(st2, s, a))
            except Exception:  # noqa: BLE001
                got = None
            want = a != "@@not a move@@"
            if got == want:
                stats["il_ok"] += 1
            elif len(errors) < 8:
                errors.append(f"(minor) is_legal_action(state, {s}, {a!r}) returned {got}, expected {want}")
    if has_cw and legal_true and len(legal_true) <= 400:
        for a in legal_true:
            try:
                st3 = mod.apply(copy.deepcopy(st), {s: a}, random.Random(3)) if list(mod.to_move(st)) == [s] else None
            except Exception:  # noqa: BLE001
                st3 = None
            if st3 is not None and mod.is_over(st3) and mod.winner(st3) == s:
                stats["cw_checked"] += 1
                try:
                    ok = bool(mod.could_win_now(st2, s, a))
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    stats["cw_ok"] += 1
                elif len(errors) < 8:
                    errors.append(f"(minor) could_win_now returned False for {a!r}, which wins at once in the true state")
                break



# ------------------------------------------------------------------ open-loop MCTS (default search)
class _Node:
    __slots__ = ("stats", "children")

    def __init__(self):
        self.stats = {}     # seat -> {action: [visits, total reward for that seat]}
        self.children = {}  # joint action (tuple of (seat, action)) -> _Node


def mcts(mod, root_states, seat, actions, t_end, rng, c=1.0, greedy=False):
    """Open-loop UCT over action sequences, re-sampling the hidden state every iteration (information-set style)
    and with decoupled selection where both seats move at once (each seat picks by its own statistics, which gives
    mixed strategies in simultaneous games). Chance (dice) is re-drawn every time a move is applied."""
    root = _Node()
    allowed = set(actions)
    iters = 0
    while time.monotonic() < t_end:
        st = copy.deepcopy(root_states[iters % len(root_states)])
        node, path = root, []
        try:
            depth = 0
            while not mod.is_over(st) and depth < MAX_ROLLOUT:
                movers = list(mod.to_move(st))
                joint = {}
                for s in movers:
                    legal = mod.legal_actions(st, s)
                    if node is root and s == seat:
                        legal = [a for a in legal if a in allowed] or legal
                    tab = node.stats.setdefault(s, {})
                    untried = [a for a in legal if a not in tab]
                    if untried:
                        a = rng.choice(untried)
                        tab[a] = [0, 0.0]
                    else:
                        tot = sum(tab[a][0] for a in legal) or 1
                        lg = math.log(tot)
                        a = max(legal, key=lambda x: tab[x][1] / max(1, tab[x][0]) + c * math.sqrt(lg / max(1, tab[x][0])))
                    joint[s] = a
                path.append((node, joint))
                st = mod.apply(st, joint, rng)
                depth += 1
                key = tuple(sorted(joint.items()))
                child = node.children.get(key)
                if child is None:
                    node.children[key] = _Node()
                    break
                node = child
            cut = ROLLOUT_CUT["plies"]
            st, _ = rollout(mod, st, rng, cap=cut or MAX_ROLLOUT, greedy=greedy)
            if mod.is_over(st) or not cut:
                w = mod.winner(st) if mod.is_over(st) else None
                res = {0: 0.0, 1: 0.0} if w is None else {w: 1.0, 1 - w: -1.0}
            else:  # cut short: the evaluation written in prep says who stands better
                v = _leaf(mod, st, 0)
                res = {0: v, 1: -v}
        except Exception:  # noqa: BLE001 - a broken line of play: bad for whoever chose it at the root
            res = {seat: -1.0, 1 - seat: 1.0}
        for nd, joint in path:
            for s, a in joint.items():
                rec = nd.stats[s][a]
                rec[0] += 1
                rec[1] += res.get(s, 0.0)
        iters += 1
        if iters % 500 == 0 and iters >= 3000:  # stop early once the choice is clear (thinking time is a tie-break)
            vis = sorted((v[0] for a, v in root.stats.get(seat, {}).items() if a in allowed), reverse=True)
            if len(vis) >= 2 and vis[0] >= 4 * vis[1] and vis[0] >= 0.6 * iters:
                break
    tab = root.stats.get(seat, {})
    out = [{"action": a, "value": round(v[1] / v[0], 3) if v[0] else 0.0, "n": v[0]} for a, v in tab.items() if a in allowed]
    out.sort(key=lambda x: (x["n"], x["value"]), reverse=True)
    return out, iters



def tactics(mod, worlds, seat, actions, t_end):
    """2-3 ply tactical filter for sequential games (game-agnostic, only uses the simulator):
    - unsafe: after our move the opponent has a move that wins at once (in any sampled world);
    - forcing: after our move, EVERY opponent reply leaves us a move that wins at once (in every sampled world).
    Returns (forcing moves, safe moves); safe is None when the check ran out of time."""
    def wins_now(st, who):
        for b in mod.legal_actions(st, who):
            st2 = mod.apply(st, {who: b}, random.Random(0))
            if mod.is_over(st2) and mod.winner(st2) == who:
                return True
        return False

    opp = 1 - seat
    safe, forcing = [], []
    try:
        for a in actions:
            if time.monotonic() > t_end:
                return [], None
            ok, force = True, True
            for w in worlds:
                st = mod.apply(w, {seat: a}, random.Random(1))
                if mod.is_over(st):
                    force = force and mod.winner(st) == seat
                    ok = ok and mod.winner(st) != opp
                    continue
                if list(mod.to_move(st)) != [opp]:
                    ok, force = True, False  # not a simple alternation here: no tactics
                    break
                if wins_now(st, opp):
                    ok, force = False, False
                    break
                if force:
                    for b in mod.legal_actions(st, opp):
                        if time.monotonic() > t_end:
                            force = False
                            break
                        st2 = mod.apply(st, {opp: b}, random.Random(2))
                        if mod.is_over(st2) or list(mod.to_move(st2)) != [seat] or not wins_now(st2, seat):
                            force = False
                            break
            if ok:
                safe.append(a)
            if ok and force:
                forcing.append(a)
    except Exception:  # noqa: BLE001
        return [], None
    return forcing, safe


def tempo(mod, worlds, seat, actions, t_end):
    """Extra-move filter for sequential games (game-agnostic, only uses to_move). Rares 10:40: in rr1 we drew the 3rd
    side of a box 22 times while safe lines were open (Dots and Boxes: closing a box gives another move).
    - keep: our moves after which it is STILL our turn in every sampled world (we earned another move);
    - nogift: moves that pass the turn and after which no opponent reply earns the opponent another move.
    Active only when the turn sometimes passes and sometimes not (a game with multi-step turns, where every move
    keeps the turn, is left alone). Returns (keep, nogift); ([], None) = no advice (no time, odd turn order)."""
    opp = 1 - seat
    keep, nogift, passing = [], [], 0
    try:
        for a in actions:
            if time.monotonic() > t_end:
                return [], None
            stays, gift = True, False
            for w in worlds:
                st = mod.apply(w, {seat: a}, random.Random(1))
                if mod.is_over(st):
                    stays = False
                    continue
                tm = list(mod.to_move(st))
                if tm == [seat]:
                    continue
                stays = False
                if tm != [opp]:
                    return [], None  # not a simple alternation: no advice
                for b in mod.legal_actions(st, opp):
                    if time.monotonic() > t_end:
                        return [], None
                    st2 = mod.apply(st, {opp: b}, random.Random(2))
                    if not mod.is_over(st2) and list(mod.to_move(st2)) == [opp]:
                        gift = True
                        break
                if gift:
                    break
            if stays:
                keep.append(a)
            else:
                passing += 1
                if not gift:
                    nogift.append(a)
    except Exception:  # noqa: BLE001
        return [], None
    if not passing:
        return [], None  # every move keeps the turn: multi-step turns, nothing to learn here
    if len(nogift) == passing:
        nogift = None  # no move gifts anything: no restriction
    return keep, nogift



# ------------------------------------------------------------------ opening book (built in idle prep time)
BOOK = {}  # sha1 of the simulator's own rendering of a position -> candidates searched in prep


def _pos_key(mod, st, seat) -> str:
    import hashlib
    return hashlib.sha1(f"{seat}|{mod.observation(st, seat)}".encode("utf-8")).hexdigest()


def _book_lookup(mod, st0, seat, actions):
    """Book move for this position, keyed by the simulator's re-rendering of the parsed observation (so the
    arena's exact wording does not matter). Only moves that are legal right now are returned."""
    if not BOOK:
        return None
    try:
        got = BOOK.get(_pos_key(mod, st0, seat))
    except Exception:  # noqa: BLE001
        return None
    if not got:
        return None
    allowed = {norm(a): a for a in actions}
    out = [dict(c, action=allowed[norm(c["action"])]) for c in got if norm(c["action"]) in allowed]
    return out or None


def build_book(mod, seconds, out_path):
    """Search the opening positions of every variant for longer than a move clock allows: the first player's first
    move, and the second player's reply to each first move. Only for variants whose start is the same whatever the
    seed (no hidden deal); otherwise a book entry would never match. Writes book.json next to the simulator."""
    t_end = time.monotonic() + seconds
    positions = []
    for label, cfg in (getattr(mod, "CONFIGS", {}) or {}).items():
        try:
            starts = [mod.initial_state(copy.deepcopy(cfg), sd) for sd in (11, 22, 33)]
            movers = list(mod.to_move(starts[0]))
            if len(movers) != 1:
                continue  # simultaneous start: no book
            seat = movers[0]
            if len({mod.observation(s_, seat) for s_ in starts}) != 1:
                continue  # hidden or random start: positions differ per deal
            root = starts[0]
            positions.append((cfg, root, seat))
            for a in mod.legal_actions(root, seat)[:40]:
                st = mod.apply(copy.deepcopy(root), {seat: a}, random.Random(5))
                if mod.is_over(st) or len(list(mod.to_move(st))) != 1:
                    continue
                positions.append((cfg, st, list(mod.to_move(st))[0]))
        except Exception:  # noqa: BLE001
            continue
    book, done = {}, 0
    for i, (cfg, st, seat) in enumerate(positions):
        left = t_end - time.monotonic()
        if left < 1.0:
            break
        per = max(1.0, min(30.0, left / (len(positions) - i)))
        try:
            rep_ = candidates(mod, {"observation": mod.observation(st, seat), "seat": seat, "config": cfg,
                                    "legal": None, "seconds": per, "nobook": True})
            if rep_.get("agree") and rep_.get("candidates"):
                book[_pos_key(mod, st, seat)] = rep_["candidates"][:5]
                done += 1
        except Exception:  # noqa: BLE001
            continue
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(book, fh)
    os.replace(tmp, out_path)
    return {"ok": True, "positions": len(positions), "booked": done}



# ------------------------------------------------------------------ cross-check of two independent simulators
def crosscheck(mod_a, path_b, seconds):
    """Play random games in each simulator; at every decision render the position with that simulator and parse it
    with the OTHER one, then compare the legal moves. Two independently written simulators that disagree point at a
    misread rule (e.g. a pending bet the parser drops), which self-consistency checks cannot see."""
    t_end = time.monotonic() + seconds
    try:
        mod_b = load(path_b, name="gensim_b")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"other simulator failed to load: {_err(exc)}"}
    checked, diffs, examples = 0, 0, []
    pairs = [(mod_a, mod_b, "A", "B"), (mod_b, mod_a, "B", "A")]
    g = 0
    while time.monotonic() < t_end and g < 60:
        src, dst, sname, dname = pairs[g % 2]
        labels = [lb for lb in (src.CONFIGS or {}) if lb in (dst.CONFIGS or {})]
        if not labels:
            return {"ok": False, "error": "the two simulators share no variant label"}
        label = labels[(g // 2) % len(labels)]
        rng = random.Random(1000 + g)
        g += 1
        try:
            st = src.initial_state(copy.deepcopy(src.CONFIGS[label]), 1000 + g)
            n = 0
            while not src.is_over(st) and n < 300 and time.monotonic() < t_end:
                for seat in list(src.to_move(st)):
                    obs = src.observation(st, seat)
                    try:
                        st2 = dst.state_from_observation(obs, seat, copy.deepcopy(dst.CONFIGS[label]), random.Random(n))
                        l_src = sorted(map(norm, src.legal_actions(st, seat)))
                        l_dst = sorted(map(norm, dst.legal_actions(st2, seat)))
                    except Exception:  # noqa: BLE001 - the other parser can't read this text: not a rules question
                        continue
                    checked += 1
                    if l_src != l_dst:
                        diffs += 1
                        if len(examples) < 6:
                            a_l, b_l = (l_src, l_dst) if sname == "A" else (l_dst, l_src)
                            examples.append({"variant": label, "seat": seat, "observation": obs[:1500],
                                             "legal_A": a_l[:30], "legal_B": b_l[:30]})
                st = step(src, st, rng, greedy=False)
                n += 1
        except Exception:  # noqa: BLE001
            continue
    return {"ok": True, "checked": checked, "diffs": diffs, "examples": examples}



# ------------------------------------------------------------------ simultaneous moves: regret matching
def _grid(actions, k):
    """Up to k actions spread over the natural order (e.g. bids 0..100 -> 0, 7, 14, ... 100), extremes included."""
    acts = sorted(actions, key=natural)
    if len(acts) <= k:
        return acts
    idx = sorted({round(i * (len(acts) - 1) / (k - 1)) for i in range(k)})
    return [acts[i] for i in idx]


OPP_WEIGHT = {"w": 0.5}  # share of the mix given to the best response to the opponent's predicted move (per request)


def regret_matching(mod, obs, seat, cfg, st0, actions, t_end, rng, opp_freq, theirs_given=None):
    """Estimate U[a][b] = our expected result when we play a and they play b (both choose at once), from playouts
    over re-sampled hidden states; then regret matching (self-play) gives an approximate equilibrium mix for both.
    Huge action lists (bids 0..100) are reduced to an even grid. If we have seen this opponent's choices before
    (opp_freq: action -> count), the reported values are also blended with a best response to that history."""
    t0 = time.monotonic()
    opp = 1 - seat
    ours = _grid(actions, 15)
    theirs = _grid(list(theirs_given) if theirs_given else list(mod.legal_actions(st0, opp)), 15)
    if not ours or not theirs:
        return None
    pool = [st0] + [mod.state_from_observation(obs, seat, copy.deepcopy(cfg), rng) for _ in range(15)]
    n = [[0] * len(theirs) for _ in ours]
    u = [[0.0] * len(theirs) for _ in ours]
    # small matrices settle fast: ~0.1 s per pair (+1.5 s), never more than 3/4 of the budget (thinking time tie-break)
    t_est = t0 + min(0.75 * (t_end - t0), 1.5 + 0.1 * len(ours) * len(theirs))
    k = 0
    while time.monotonic() < t_est:
        i, j = k % len(ours), (k // len(ours)) % len(theirs)
        st = copy.deepcopy(pool[k % len(pool)])
        k += 1
        try:
            st = mod.apply(st, {seat: ours[i], opp: theirs[j]}, rng)
            st, _ = rollout(mod, st, rng, greedy=False)
            v = score_for(mod, st, seat)
        except Exception:  # noqa: BLE001
            continue
        n[i][j] += 1
        u[i][j] += v
    if k < len(ours) * len(theirs):
        return None  # not even one playout per pair: let the tree search handle it
    U = [[u[i][j] / n[i][j] if n[i][j] else 0.0 for j in range(len(theirs))] for i in range(len(ours))]
    # if our options don't differ beyond playout noise (random playouts can't value information-gathering moves),
    # an "equilibrium" over noise is meaningless: let the other paths (hint / proposals / Haiku) decide
    rows = [sum(r) / len(r) for r in U]
    nmin = max(1, min(min(r) for r in n))
    if not theirs_given and max(rows) - min(rows) < 2.0 / math.sqrt(nmin * len(theirs)):
        return None
    ra, rb = [0.0] * len(ours), [0.0] * len(theirs)
    sa, sb = [0.0] * len(ours), [0.0] * len(theirs)

    def strat(r):
        pos = [max(0.0, x) for x in r]
        z = sum(pos)
        return [x / z for x in pos] if z > 0 else [1.0 / len(r)] * len(r)
    for _ in range(3000):
        p, q = strat(ra), strat(rb)
        va = [sum(U[i][j] * q[j] for j in range(len(theirs))) for i in range(len(ours))]
        vb = [-sum(U[i][j] * p[i] for i in range(len(ours))) for j in range(len(theirs))]
        ev_a = sum(p[i] * va[i] for i in range(len(ours)))
        ev_b = sum(q[j] * vb[j] for j in range(len(theirs)))
        for i in range(len(ours)):
            ra[i] += va[i] - ev_a
            sa[i] += p[i]
        for j in range(len(theirs)):
            rb[j] += vb[j] - ev_b
            sb[j] += q[j]
    p = [x / sum(sa) for x in sa]
    q = [x / sum(sb) for x in sb]
    # opponent history: if this opponent has a strong habit, best-respond to it (blended with the equilibrium)
    seen = {norm(a): c for a, c in (opp_freq or {}).items()}
    total = sum(seen.values())
    # history given explicitly (the opponent's past moves in this game): 3 are enough; inferred counts need 6
    enough = 3 if theirs_given else 6
    if total >= enough:
        hist = [seen.get(norm(b), 0) / total for b in theirs]
        q_mix = [0.5 * q[j] + 0.5 * hist[j] for j in range(len(theirs))]
    else:
        q_mix = q
    vals = [sum(U[i][j] * q_mix[j] for j in range(len(theirs))) for i in range(len(ours))]
    if total >= enough:  # shift the mix toward the moves that beat this opponent's habit
        best = max(vals)
        wgt = max(0.0, min(0.9, float(OPP_WEIGHT["w"])))
        p = [(1.0 - wgt) * p[i] + (wgt if vals[i] == best else 0.0) for i in range(len(ours))]
    out = [{"action": a, "value": round(vals[i], 3), "n": sum(n[i]), "p": round(p[i], 4)} for i, a in enumerate(ours)]
    out.sort(key=lambda c: (c["p"], c["value"]), reverse=True)
    return {"ok": True, "agree": True, "candidates": out[:12], "playouts": k, "informative": True, "sampled": False,
            "mixed": True}



# ------------------------------------------------------------------ what did the opponent just play?
def infer_opponent(mod, prev_obs, obs, seat, cfg, our_action, seconds=1.0):
    """Which opponent move turns the previous position into the one we observe now? Re-apply each of their legal
    moves (with ours, for simultaneous turns) in the simulator, render our view and keep the closest match to the
    real observation. Game-agnostic; only trusted when one move clearly fits best."""
    import difflib
    t_end = time.monotonic() + seconds
    opp = 1 - seat
    st = mod.state_from_observation(prev_obs, seat, copy.deepcopy(cfg), random.Random(0))
    movers = list(mod.to_move(st))
    if opp not in movers:
        return {"ok": True, "action": None}
    scores = []
    for b in mod.legal_actions(st, opp)[:120]:
        if time.monotonic() > t_end:
            break
        acts = {opp: b}
        if seat in movers:
            acts[seat] = our_action
        try:
            st2 = mod.apply(copy.deepcopy(st), acts, random.Random(1))
            text = mod.observation(st2, seat)
        except Exception:  # noqa: BLE001
            continue
        scores.append((difflib.SequenceMatcher(None, text, obs).ratio(), b))
    if not scores:
        return {"ok": True, "action": None}
    scores.sort(reverse=True)
    best = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    sure = best[0] >= 0.9 and best[0] - second >= 0.01
    return {"ok": True, "action": best[1] if sure else None, "score": round(best[0], 4), "margin": round(best[0] - second, 4)}



# ------------------------------------------------------------------ screening proposed moves (huge action spaces)
WORD_B = chr(92) + "b"  # regex word boundary


def screen(mod, req):
    """For moves proposed by the model where the legal list is too long to enumerate: is each one legal (simulator's
    is_legal_action), could it still win right now (could_win_now, e.g. a word consistent with all feedback), and was
    it already played (its text appears in the observation)? Returns one verdict per proposal, in order."""
    seat, cfg, obs = int(req.get("seat", 0)), req.get("config") or {}, req.get("observation", "")
    st = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), random.Random(0))
    low = obs.lower()
    out = []
    t_end = time.monotonic() + float(req.get("seconds", 2.5))  # proposals come first; dictionary words while time lasts
    for a in req.get("actions") or []:
        if time.monotonic() > t_end:
            break
        if not isinstance(a, str) or not a.strip():
            continue
        v = {"action": a.strip(), "legal": None, "could_win": None}
        try:
            if req.get("use_il") and callable(getattr(mod, "is_legal_action", None)):
                v["legal"] = bool(mod.is_legal_action(st, seat, a.strip()))
        except Exception:  # noqa: BLE001
            v["legal"] = False
        try:
            if req.get("use_cw") and callable(getattr(mod, "could_win_now", None)):
                v["could_win"] = bool(mod.could_win_now(st, seat, a.strip()))
        except Exception:  # noqa: BLE001
            v["could_win"] = None
        core = norm(a).split()[-1] if norm(a).split() else ""
        v["repeated"] = bool(core) and len(core) >= 3 and re.search(WORD_B + re.escape(core) + WORD_B, low) is not None
        out.append(v)
    return {"ok": True, "verdicts": out, "movers": list(mod.to_move(st))}


def rank(mod, req):
    """Regret matching over model-proposed moves (ours) against model-proposed opponent moves (theirs)."""
    seat, cfg, obs = int(req.get("seat", 0)), req.get("config") or {}, req.get("observation", "")
    OPP_WEIGHT["w"] = float(req.get("opp_weight", 0.5) or 0.5)  # review #14: not the last candidates request's
    rng = random.Random(int(time.time() * 1000) & 0xFFFFFFFF)
    st0 = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), rng)
    t_end = time.monotonic() + max(0.5, float(req.get("seconds", 3)))
    res = regret_matching(mod, obs, seat, cfg, st0, list(req.get("ours") or []), t_end, rng, req.get("opp_freq") or {},
                          theirs_given=list(req.get("theirs") or []))
    return res or {"ok": True, "agree": True, "candidates": [], "mixed": False}


# ------------------------------------------------------------------ search
def candidates(mod, req):
    t_end = time.monotonic() + max(0.3, float(req.get("seconds", 3)))
    seat = int(req.get("seat", 0))
    cfg = req.get("config") or {}
    obs = req.get("observation", "")
    arena_legal = req.get("legal")
    rng = random.Random((int(time.time() * 1000) + 7919 * int(req.get("seed", 0))) & 0xFFFFFFFF)
    st0 = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), rng)
    mine = list(mod.legal_actions(st0, seat))
    if seat not in list(mod.to_move(st0)):
        return {"ok": True, "agree": False, "why": "simulator says it is not our turn", "candidates": []}
    if arena_legal is not None:
        if sorted(map(norm, mine)) != sorted(map(norm, arena_legal)):
            return {"ok": True, "agree": False, "why": f"legal mismatch: arena {arena_legal[:10]} vs sim {mine[:10]}",
                    "candidates": []}
        spelled = {norm(a): a for a in arena_legal}
        actions = [spelled[norm(a)] for a in mine]
    else:
        spelled = {}
        actions = mine
    if not actions:
        return {"ok": True, "agree": False, "why": "no legal moves in simulator", "candidates": []}
    rec = getattr(mod, "recommend", None)
    if callable(rec):
        # t3/library: a replay-checked simulator of a public game may bring its own policy (mesh/known/: exact Nim,
        # alpha-beta for Connect Four / Dots and Boxes / Mini Chess / Hex, fleet frequencies for Battleship). Every
        # recommended move is still checked for legality here; nothing recommended -> the normal search below.
        try:
            # these policies find their move in well under a second on most turns; 3.5 s is the ceiling (thinking
            # time is tie-break #2), not the full search budget
            got = rec(obs, seat, copy.deepcopy(cfg), rng, max(0.2, min(3.5, t_end - time.monotonic() - 0.4)))
        except Exception:  # noqa: BLE001
            got = None
        out, legal_n = [], {norm(a) for a in mine}
        check = getattr(mod, "is_legal_action", None)
        for item in got or []:
            a, v = item[0], item[1]
            if norm(a) in legal_n or (callable(check) and check(st0, seat, a)):
                if arena_legal is None or norm(a) in spelled:
                    c = {"action": spelled.get(norm(a), a), "value": float(v), "n": 0, "note": "known-game policy"}
                    if len(item) > 2 and item[2] is not None:
                        c["p"] = float(item[2])  # a mixed policy (poker ranges, simultaneous moves): the brain draws
                    out.append(c)
        if out:
            return {"ok": True, "agree": True, "candidates": out[:6], "informative": True, "sampled": False,
                    "search": "known-game policy"}
    OPP_WEIGHT["w"] = float(req.get("opp_weight", 0.5) or 0.5)
    has_eval = callable(getattr(mod, "evaluate", None))
    ROLLOUT_CUT["plies"] = EVAL_CUT_PLIES if (req.get("search") == "mcts+eval" and has_eval) else None
    if req.get("search") == "eval" and has_eval \
            and list(mod.to_move(st0)) == [seat] and len(actions) > 1:
        # the evaluation function written in prep won its duel against the playout search (simgen.add_evaluation)
        res = eval_search(mod, obs, seat, cfg, st0, actions, t_end, rng)
        if res:
            return res
    if req.get("rm", True) and len(list(mod.to_move(st0))) == 2 and len(actions) > 1:
        # simultaneous move: equilibrium mix by regret matching on a payoff matrix estimated with playouts
        res = regret_matching(mod, obs, seat, cfg, st0, actions, t_end, rng, req.get("opp_freq") or {})
        if res:
            return res
    if len(actions) > 60:  # huge action spaces: evaluate a sample, always keeping any action twice-seen
        actions = rng.sample(actions, 60)
    if len(actions) == 1:
        return {"ok": True, "agree": True, "candidates": [{"action": actions[0], "value": 0.0, "n": 0}]}
    # a move that wins on the spot (in a sequential game without hidden randomness) needs no search
    if list(mod.to_move(st0)) == [seat]:
        wins = []
        for a in actions:
            try:
                ok = True
                for k in range(3):  # same outcome under 3 determinizations/dice: it is a sure win
                    st = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), random.Random(k))
                    st = mod.apply(st, {seat: a}, random.Random(k + 7))
                    if not (mod.is_over(st) and mod.winner(st) == seat):
                        ok = False
                        break
                if ok:
                    wins.append(a)
            except Exception:  # noqa: BLE001
                pass
            if time.monotonic() > t_end - 0.2:
                break
        if wins:
            return {"ok": True, "agree": True, "candidates": [{"action": a, "value": 1.0, "n": 3, "note": "wins now"} for a in wins[:5]]}
    if len(actions) < len(mine):  # huge action space: random playouts can't rank it; Haiku decides, with a hint:
        # which moves win at once in at least one world consistent with the observation (e.g. the numbers still possible)
        full = [spelled.get(norm(a), a) for a in mine] if arena_legal is not None else mine
        hits, worlds = {}, 0
        # what each move would SHOW us, per sampled hidden state: a move whose result differs from its usual one in
        # many worlds is likely to find something (a ship cell, a letter in the word). Hint only (Rares 09:28: in
        # hidden-state games the engine had 0 playouts and Haiku chose with nothing to go on)
        outcomes = {}
        while worlds < 24 and time.monotonic() < t_end - 0.3:
            w = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), random.Random(worlds + 101))
            worlds += 1
            if list(mod.to_move(w)) != [seat]:
                break
            for a in full:
                if time.monotonic() > t_end - 0.3:
                    break
                try:
                    r2 = random.Random(worlds)
                    st = mod.apply(w, {seat: a}, r2)
                    if len(outcomes) < 400:
                        try:
                            o = str(mod.observation(st, seat))
                            outcomes.setdefault(a, {})
                            outcomes[a][o] = outcomes[a].get(o, 0) + 1
                        except Exception:  # noqa: BLE001
                            pass
                    k = 0  # let the opponent finish the round (some games end only when the round is complete)
                    while not mod.is_over(st) and seat not in list(mod.to_move(st)) and k < 3:
                        st = step(mod, st, r2, greedy=False)
                        k += 1
                    if mod.is_over(st) and mod.winner(st) == seat:
                        hits[a] = hits.get(a, 0) + 1
                except Exception:  # noqa: BLE001
                    pass
        cands = sorted(hits, key=lambda a: (-hits[a], natural(a)))
        unusual = {}
        for a, oc in outcomes.items():
            tot_a = sum(oc.values())
            if tot_a >= 8:
                unusual[a] = 1.0 - max(oc.values()) / tot_a
        hint = ""
        if hits and len(hits) < len(full):
            shown = sorted(hits, key=natural)
            hint = (f"The engine sampled {worlds} hidden states consistent with everything you have seen; in at least "
                    f"one of them these {len(hits)} moves would win at once: {', '.join(shown[:40])}"
                    f"{' ...' if len(shown) > 40 else ''}. (A sample, not a complete list: use it to see what is still possible.)")
        lively = sorted((a for a in unusual if unusual[a] > 0), key=lambda a: (-unusual[a], natural(a)))
        if lively and len(lively) < len(unusual):
            hint += ((" " if hint else "") + f"In those sampled hidden states, these moves most often give a result "
                     f"other than their usual one (they are likely to find something hidden): "
                     + ", ".join(f"{a} ({unusual[a]:.0%})" for a in lively[:12])
                     + ". Moves not listed gave the same result in every sampled state.")
        fallback_order = cands or lively  # when the model gives nothing, the likeliest find goes, not an arbitrary move
        return {"ok": True, "agree": True, "playouts": 0, "informative": False, "sampled": True, "hint": hint,
                "candidates": [{"action": a, "value": 0.0, "n": hits.get(a, 0)} for a in fallback_order[:8]]
                              or [{"action": a, "value": 0.0, "n": 0} for a in actions[:8]]}
    hit = None if req.get("nobook") else _book_lookup(mod, st0, seat, actions)
    if hit:
        return {"ok": True, "agree": True, "candidates": hit, "playouts": 0, "informative": True,
                "sampled": False, "book": True}
    if req.get("search", "mcts") in ("mcts", "eval", "mcts+eval"):
        pool = [st0] + [mod.state_from_observation(obs, seat, copy.deepcopy(cfg), rng) for _ in range(31)]
        if req.get("tactics", True) and list(mod.to_move(st0)) == [seat] and len(actions) <= 90:  # 7x7 Hex has 49
            t_tac = time.monotonic() + 0.4 * max(0.0, t_end - time.monotonic())
            forcing, safe = tactics(mod, pool[:3], seat, actions, t_tac)
            if forcing:
                return {"ok": True, "agree": True, "playouts": 0, "informative": True, "sampled": False,
                        "candidates": [{"action": a, "value": 0.99, "n": 1, "note": "forces a win"} for a in forcing[:5]]}
            if safe and len(safe) < len(actions):
                actions = safe  # never hand the opponent an immediate win when another move avoids it
        if req.get("tempo", True) and list(mod.to_move(st0)) == [seat] and 1 < len(actions) <= 90:
            # Rares 10:40 (audit of rr1: 22 gifted boxes in Dots): in games where a move can earn ANOTHER move
            # (closing a box, a capture), take that extra move, and never leave the opponent one when a move avoids it
            t_tmp = time.monotonic() + 0.25 * max(0.0, t_end - time.monotonic())
            keep, nogift = tempo(mod, pool[:3], seat, actions, t_tmp)
            if keep:
                actions = keep
            elif nogift:
                actions = nogift  # Haiku then chooses only among searched moves, so it stays inside this set too
        out, iters = mcts(mod, pool, seat, actions, t_end, rng, greedy=bool(req.get("greedy", POLICY["greedy"])))
        if not out:
            # review #8: an empty search (no time left after eval_search) is not a disagreement with the arena
            return {"ok": True, "agree": True, "why": "search produced nothing", "candidates": []}
        top = out[0]
        mean = sum(c["value"] for c in out) / len(out)
        # rank by visits (robust child); "value" is the mean result of each move for us
        informative = top["n"] >= 100 and ((top["value"] - mean) > 2.0 / math.sqrt(max(1, top["n"])) or top["value"] >= 0.95)
        for c in out[1:]:  # a move searched far less than the top one is not "near best", whatever its noisy mean
            if c["n"] < 0.5 * top["n"]:
                c["value"] = min(c["value"], round(top["value"] - 0.06, 3))
        return {"ok": True, "agree": True, "candidates": out[:8], "playouts": iters, "informative": informative,
                "sampled": False}
    n = {a: 0 for a in actions}
    tot = {a: 0.0 for a in actions}
    pool = []
    total = 0
    # flat Monte Carlo with UCB1 over our moves; every playout re-samples the hidden state (determinization),
    # the opponent's simultaneous move and the dice. Values are our expected result in [-1, 1].
    while time.monotonic() < t_end:
        if total < len(actions):
            a = actions[total]
        else:
            lg = math.log(total)
            a = max(actions, key=lambda x: tot[x] / n[x] + 1.2 * math.sqrt(lg / n[x]))
        try:
            # determinizations: parse once per hidden-state sample (up to 48), then reuse copies (parsing is slow)
            if len(pool) < 48:
                pool.append(mod.state_from_observation(obs, seat, copy.deepcopy(cfg), rng) if pool else st0)
                st = copy.deepcopy(pool[-1])
            else:
                st = copy.deepcopy(pool[total % len(pool)])
            movers = list(mod.to_move(st))
            acts = {seat: a}
            for s in movers:
                if s != seat:
                    acts[s] = rng.choice(mod.legal_actions(st, s))
            st = mod.apply(st, acts, rng)
            st, _ = rollout(mod, st, rng, greedy=req.get("greedy"))
            v = score_for(mod, st, seat)
        except Exception:  # noqa: BLE001 - a broken line of play counts as a loss for that move
            v = -1.0
        n[a] += 1
        tot[a] += v
        total += 1
        if total % 200 == 0 and total >= 4 * len(actions):  # stop early once one move clearly dominates
            ranked = sorted(actions, key=lambda x: tot[x] / max(1, n[x]), reverse=True)
            b, s2 = ranked[0], ranked[1]
            mb, ms = tot[b] / n[b], tot[s2] / n[s2]
            if n[b] >= 300 and mb - ms > 3.0 / math.sqrt(n[b]) + 2.0 / math.sqrt(max(1, n[s2])):
                break
    out = [{"action": a, "value": round(tot[a] / n[a], 3) if n[a] else 0.0, "n": n[a]} for a in actions]
    out.sort(key=lambda c: (c["value"], c["n"]), reverse=True)
    # is the search telling us anything? (top move clearly better than the typical move, beyond the noise)
    mean = sum(c["value"] for c in out) / len(out)  # unweighted: the value of a typical legal move
    noise = 2.0 / math.sqrt(max(1, out[0]["n"]))
    sampled = len(actions) < len(mine)
    # a sampled subset of a huge action space, or too few playouts on the top move: the numbers mean nothing
    informative = (not sampled and out[0]["n"] >= 100
                   and ((out[0]["value"] - mean) > noise or out[0]["value"] >= 0.95))
    return {"ok": True, "agree": True, "candidates": out[:8], "playouts": total, "informative": informative,
            "sampled": sampled}


def _leaf(mod, st, seat) -> float:
    """Value of a position for `seat`: the real result when the game is over, else the prep-written evaluation,
    kept strictly inside a won/lost game's +-1."""
    if mod.is_over(st):
        w = mod.winner(st)
        return 0.0 if w is None else (1.0 if w == seat else -1.0)
    try:
        v = float(mod.evaluate(st, seat))
    except Exception:  # noqa: BLE001
        return 0.0
    if v != v:
        return 0.0
    return max(-0.9, min(0.9, v))


def _alphabeta(mod, st, depth, alpha, beta, seat, t_end, rng):
    if depth <= 0 or mod.is_over(st):
        return _leaf(mod, st, seat)
    if time.monotonic() > t_end:
        raise TimeoutError
    movers = list(mod.to_move(st))
    if len(movers) != 1:
        return _leaf(mod, st, seat)
    s = movers[0]
    acts = mod.legal_actions(st, s)
    if not acts:
        return _leaf(mod, st, seat)
    ours = s == seat
    best = -2.0 if ours else 2.0
    for a in acts[:48]:
        v = _alphabeta(mod, mod.apply(st, {s: a}, rng), depth - 1, alpha, beta, seat, t_end, rng) * 0.995
        if ours:
            if v > best:
                best = v
            if best > alpha:
                alpha = best
        else:
            if v < best:
                best = v
            if best < beta:
                beta = best
        if alpha >= beta:
            break
    return best


def eval_search(mod, obs, seat, cfg, st0, actions, t_end, rng):
    """Search with the evaluation function the model wrote in prep (game-agnostic; random playouts gave 150-260
    playouts per move in Hex / Dots and 6 in Mini Chess in the official image, i.e. noise).
    - Everything visible and nothing random (the same observation always parses to the same state, the same move
      always gives the same state): iterative-deepening alpha-beta, as deep as the clock allows.
    - Hidden information or chance: every legal move is scored by the average evaluation right after it over many
      sampled worlds consistent with the observation (the shot most sampled fleets confirm, the card most deals like).
    Returns the usual candidates dict, or None to let the playout search run."""
    try:
        w1 = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), random.Random(1))
        w2 = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), random.Random(2))
        det = w1 == w2 == st0
        if det:
            a0 = actions[0]
            det = mod.apply(st0, {seat: a0}, random.Random(1)) == mod.apply(st0, {seat: a0}, random.Random(2))
    except Exception:  # noqa: BLE001
        return None
    if det:
        order = list(actions)
        scores, depth_done, depth = {}, 0, 1
        while depth <= 40:
            cur, alpha = {}, -2.0
            try:
                for a in order:
                    v = _alphabeta(mod, mod.apply(st0, {seat: a}, rng), depth - 1, alpha, 2.0, seat, t_end - 0.15, rng)
                    # review #16: after the first move the window is narrowed: v is an upper bound, never "as good"
                    cur[a] = v if (v > alpha or not cur) else min(v, alpha) - 0.06
                    if v > alpha:
                        alpha = v
            except TimeoutError:
                break
            except Exception:  # noqa: BLE001 - a crash in generated code: keep what we have
                break
            scores, depth_done = cur, depth
            order = sorted(order, key=lambda a: -scores[a])
            if max(scores.values()) >= 0.95 or (max(scores.values()) <= -0.95 and depth >= 2) \
                    or time.monotonic() > t_end - 0.2:
                break
            depth += 1
        if not scores:
            return None
        out = [{"action": a, "value": round(v, 4), "n": depth_done} for a, v in sorted(scores.items(), key=lambda kv: -kv[1])]
        return {"ok": True, "agree": True, "candidates": out[:8], "playouts": 0, "informative": True, "sampled": False,
                "search": f"alpha-beta depth {depth_done}"}
    tot = {a: 0.0 for a in actions}
    cnt = {a: 0 for a in actions}
    worlds = 0
    while worlds < 400 and time.monotonic() < t_end - 0.2:
        try:
            w = st0 if worlds == 0 else mod.state_from_observation(obs, seat, copy.deepcopy(cfg), rng)
        except Exception:  # noqa: BLE001
            break
        worlds += 1
        for a in actions:
            try:
                nxt = mod.apply(w, {seat: a}, rng)
                # some games resolve a move only when the round is complete (shots scored after both have fired):
                # let the other seat answer at random until it is our turn again, then evaluate
                k = 0
                while not mod.is_over(nxt) and seat not in list(mod.to_move(nxt)) and k < 4:
                    nxt = step(mod, nxt, rng, False)
                    k += 1
                tot[a] += _leaf(mod, nxt, seat)
                cnt[a] += 1
            except Exception:  # noqa: BLE001 - not legal in this sampled world
                pass
        if worlds >= 24 and worlds % 8 == 0 and time.monotonic() > t_end - 0.4:
            break
    ranked = sorted((a for a in actions if cnt[a]), key=lambda a: -(tot[a] / cnt[a]))
    if not ranked or worlds < 8:
        return None
    out = [{"action": a, "value": round(tot[a] / cnt[a], 4), "n": cnt[a]} for a in ranked]
    return {"ok": True, "agree": True, "candidates": out[:8], "playouts": 0, "informative": True, "sampled": False,
            "search": f"evaluation over {worlds} sampled worlds"}


def evalcheck(mod, seconds=8.0):
    """Is the prep-written evaluate(state, seat) usable? It must return a finite number for every position of random
    games without crashing, be fast, and point the right way: in positions late in a game it must favour the seat
    that then wins more often than not."""
    if not callable(getattr(mod, "evaluate", None)):
        return {"ok": False, "why": "no evaluate function"}
    t_end = time.monotonic() + seconds
    labels = list(mod.CONFIGS)
    calls, t_eval, agree, decided, bad, seed = 0, 0.0, 0, 0, 0, 0
    while time.monotonic() < t_end and seed < 60:
        rng = random.Random(seed)
        st = mod.initial_state(copy.deepcopy(mod.CONFIGS[labels[seed % len(labels)]]), seed)
        seed += 1
        seen = []
        n = 0
        try:
            while not mod.is_over(st) and n < MAX_ROLLOUT:
                t0 = time.monotonic()
                v0, v1 = mod.evaluate(st, 0), mod.evaluate(st, 1)
                t_eval += time.monotonic() - t0
                calls += 2
                if not all(isinstance(v, (int, float)) and v == v and abs(v) < 1e6 for v in (v0, v1)):
                    bad += 1
                seen.append(float(v0) - float(v1))
                st = step(mod, st, rng, False)
                n += 1
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "why": f"evaluate crashed: {_err(exc)}"}
        w = mod.winner(st) if mod.is_over(st) else None
        if w in (0, 1) and len(seen) >= 4:
            late = seen[int(len(seen) * 0.7)]
            if abs(late) > 1e-9:
                decided += 1
                agree += (late > 0) == (w == 0)
    if calls < 20:
        return {"ok": False, "why": "too few positions evaluated"}
    if bad:
        return {"ok": False, "why": f"{bad} non-numeric or huge values"}
    per = t_eval / calls
    rate = agree / decided if decided else None
    ok = per < 0.002 and decided > 0 and (decided < 8 or rate >= 0.55)  # review #11: a constant evaluate decides 0
    return {"ok": ok, "calls": calls, "ms_per_call": round(per * 1000, 3), "sign_agreement": rate, "decided": decided,
            "why": "" if ok else ("too slow" if per >= 0.002 else "decides nothing (constant?)" if not decided else f"points the wrong way ({agree}/{decided} late positions)")}


def duel(mod, req):
    """Evaluation search against the playout search on this simulator, through the same code path as a real turn
    (observation text -> candidates). Seats alternate, a few games per variant, until the time is used. The caller
    adopts the evaluation search only if it scores more."""
    secs = float(req.get("move_seconds", 0.3))
    t_end = time.monotonic() + float(req.get("seconds", 90))
    labels = list(mod.CONFIGS)
    score = {"eval": 0.0, "mcts": 0.0}
    games = 0
    rng = random.Random(77)
    while time.monotonic() < t_end:
        label = labels[(games // 2) % len(labels)]
        cfg = mod.CONFIGS[label]
        ev_seat = games % 2
        st = mod.initial_state(copy.deepcopy(cfg), 500 + games // 2)
        n = 0
        aborted = False
        while not mod.is_over(st) and n < MAX_ROLLOUT:
            if time.monotonic() > t_end + 20:
                aborted = True
                break
            acts = {}
            for s in list(mod.to_move(st)):
                mode = str(req.get("mode") or "eval") if s == ev_seat else "mcts"
                r = candidates(mod, {"observation": mod.observation(st, s), "seat": s, "config": cfg, "legal": None,
                                     "seconds": secs, "search": mode, "nobook": True, "rm": False, "seed": n})
                c = (r or {}).get("candidates") or []
                acts[s] = c[0]["action"] if c else rng.choice(mod.legal_actions(st, s))
            st = mod.apply(st, acts, rng)
            n += 1
        if aborted:
            break
        games += 1
        w = mod.winner(st) if mod.is_over(st) else None
        if w is None:
            score["eval"] += 0.5
            score["mcts"] += 0.5
        else:
            score["eval" if w == ev_seat else "mcts"] += 1.0
        if games >= int(req.get("max_games", 12)):
            break
    return {"ok": True, "games": games, "eval": score["eval"], "mcts": score["mcts"]}


def replay(mod, req):
    """Check the simulator against REAL games from public replays (mesh/experience/, rule 10.7): for every move a
    seat really made, the observation it saw must parse, it must be that seat's turn, and the move the arena accepted
    must be legal here. Unless the game hides information (the parsed state then holds sampled secrets): applying the
    real moves must agree with the next real observation on the legal moves, the game must not end early, and after
    the last real move the simulator must name the arena's winner (random_end: only when it says the game is over).
    Returns {"fails": [text for the repair prompt], "kinds": {kind: count}, "moves": n, "ends_ok": n, "ends": n}."""
    games = req.get("games") or []
    random_end = bool(req.get("random_end"))
    t_end = time.monotonic() + float(req.get("seconds", 20))
    fails, kinds, stats = [], {}, {"moves": 0, "ends_ok": 0, "ends": 0}
    # hidden information is decided HERE, not by the digest's flag (one model answer; it called Prisoner's Dilemma
    # hidden): if parsing the same real observation with two different random seeds gives different states, the
    # simulator samples something the seat cannot see, and only parsing + legality can be compared with the real game
    hidden = False
    try:
        probes = 0
        for g in games:
            cfg0 = mod.CONFIGS.get(g.get("label"))
            for step in (g.get("steps") or [])[1:4]:
                for s, (obs, _a) in step.items():
                    a1 = mod.state_from_observation(obs, int(s), copy.deepcopy(cfg0), random.Random(1))
                    a2 = mod.state_from_observation(obs, int(s), copy.deepcopy(cfg0), random.Random(2))
                    a3 = mod.state_from_observation(obs, int(s), copy.deepcopy(cfg0), random.Random(3))
                    probes += 1
                    if not (a1 == a2 == a3):
                        hidden = True
            if hidden or probes >= 12:
                break
    except Exception:  # noqa: BLE001 - the per-move checks below report the crash
        hidden = bool(req.get("hidden"))

    def fail(kind, text):
        kinds[kind] = kinds.get(kind, 0) + 1
        if kinds[kind] <= 2 and len(fails) < 6:
            fails.append(f"REAL GAME CHECK FAILED ({kind}). {text}")

    def is_legal(st, seat, action):
        try:
            la = mod.legal_actions(st, seat)
        except Exception:  # noqa: BLE001
            la = None
        if la and norm(action) in {norm(x) for x in la}:
            return True
        f = getattr(mod, "is_legal_action", None)
        try:
            return bool(callable(f) and f(st, seat, action))
        except Exception:  # noqa: BLE001
            return False

    for g in games:
        if time.monotonic() > t_end:
            break
        label = g.get("label")
        cfg = mod.CONFIGS.get(label)
        if cfg is None:
            fail("variant missing", f"CONFIGS has no entry for the variant label {label!r}.")
            continue
        steps = g.get("steps") or []
        for i, step in enumerate(steps):
            if time.monotonic() > t_end:
                break
            last = i == len(steps) - 1
            acts = {int(s): v for s, v in step.items()}
            where = f"Variant {label!r}, a real game the arena ended with: {g.get('reason')!r}."
            states = {}
            for seat, (obs, action) in acts.items():
                stats["moves"] += 1
                try:
                    st = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), random.Random(7))
                    states[seat] = st
                    if seat not in list(mod.to_move(st)):
                        fail("not that seat's turn", f"{where} Seat {seat} saw this observation and moved, but "
                             f"to_move(state_from_observation(...)) = {list(mod.to_move(st))}.\nObservation:\n{obs[:1800]}")
                    elif not is_legal(st, seat, action):
                        fail("a real move is illegal here", f"{where} Seat {seat} played {action!r} (accepted by the "
                             f"arena) after this observation, but your simulator calls it illegal; your legal actions "
                             f"(first 15): {(mod.legal_actions(st, seat) or [])[:15]}.\nObservation:\n{obs[:1800]}")
                except Exception as exc:  # noqa: BLE001
                    fail("crash on a real observation", f"{where} state_from_observation/legal_actions crashed for seat "
                         f"{seat}: {_err(exc)}\nObservation:\n{obs[:1800]}")
            if hidden or not states:
                continue
            seat0 = sorted(states)[0]
            try:
                movers = list(mod.to_move(states[seat0]))
                if set(movers) - set(acts):
                    continue
                nxt = mod.apply(states[seat0], {s: acts[s][1] for s in movers}, random.Random(11))
                over = bool(mod.is_over(nxt))
            except Exception as exc:  # noqa: BLE001
                fail("apply crashed on real moves", f"{where} apply() crashed on the real move(s) "
                     f"{ {s: acts[s][1] for s in acts} }: {_err(exc)}\nObservation of seat {seat0}:\n{acts[seat0][0][:1800]}")
                continue
            try:
                if last:
                    stats["ends"] += 1
                    real = g.get("winner_seat")
                    if not over and random_end:
                        # the real game ended here by chance (e.g. "ends with a 1 in 3 chance"): try other draws until
                        # the simulator ends it too, then its winner must be the arena's
                        for sd in range(60):
                            alt = mod.apply(states[seat0], {s: acts[s][1] for s in movers}, random.Random(1000 + sd))
                            if mod.is_over(alt):
                                nxt, over = alt, True
                                break
                    if not over and not random_end:
                        fail("the game was over in the arena but not in your simulator", f"{where} After the real last "
                             f"move(s) { {s: acts[s][1] for s in acts} } the arena ended the game (winner seat: {real!r}; "
                             f"None = draw), but is_over() is False.\nObservation of seat {seat0} before that move:\n"
                             f"{acts[seat0][0][:1800]}")
                    elif over and mod.winner(nxt) != real:
                        fail("wrong winner", f"{where} After the real last move(s) { {s: acts[s][1] for s in acts} } "
                             f"the arena's winner is seat {real!r} (None = draw) but winner() returns {mod.winner(nxt)!r}. "
                             f"Read the rules' winning condition and tie-breaks again.\nObservation of seat {seat0} "
                             f"before that move:\n{acts[seat0][0][:1800]}")
                    elif over:
                        stats["ends_ok"] += 1
                elif over and not random_end:
                    fail("your simulator ends the game too early", f"{where} After the real move(s) "
                         f"{ {s: acts[s][1] for s in acts} } the real game went on, but is_over() is True "
                         f"(winner() = {mod.winner(nxt)!r}).\nObservation of seat {seat0} before that move:\n{acts[seat0][0][:1800]}")
                elif not over:
                    for seat, (obs, _a) in {int(s): v for s, v in steps[i + 1].items()}.items():
                        ref = mod.state_from_observation(obs, seat, copy.deepcopy(cfg), random.Random(7))
                        want = sorted(norm(x) for x in (mod.legal_actions(ref, seat) or []))
                        got = sorted(norm(x) for x in (mod.legal_actions(nxt, seat) or []))
                        if want != got:
                            fail("apply() disagrees with the next real observation", f"{where} After the real move(s) "
                                 f"{ {s: acts[s][1] for s in acts} }, seat {seat} really moved next. Legal moves from "
                                 f"your apply(): {len(got)}; from parsing the next real observation: {len(want)}. Only "
                                 f"in apply(): {sorted(set(got) - set(want))[:8]}; only in the observation: "
                                 f"{sorted(set(want) - set(got))[:8]}. (Whose turn it is after a move, and what a move "
                                 f"changes, must match the arena.)\nNext real observation:\n{obs[:1500]}")
            except Exception as exc:  # noqa: BLE001
                fail("crash while comparing with the real game", f"{where} {_err(exc)}")
    return {"ok": True, "fails": fails, "kinds": kinds, "moves": stats["moves"], "ends_ok": stats["ends_ok"], "hidden": hidden,
            "ends": stats["ends"], "clean": not kinds}


def main():
    try:
        import resource  # Linux only: hard memory cap for generated code
        resource.setrlimit(resource.RLIMIT_AS, (900 * 1024 * 1024, 900 * 1024 * 1024))
    except Exception:  # noqa: BLE001
        pass
    path = sys.argv[1]
    try:
        mod = load(path)
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(path)), "book.json"), encoding="utf-8") as fh:
                BOOK.update(json.load(fh))
        except (OSError, ValueError):
            pass
        reply({"ok": True, "loaded": True, "book": len(BOOK)})
    except Exception as exc:  # noqa: BLE001
        reply({"ok": False, "loaded": False, "error": _err(exc) if exc.__traceback__ else str(exc),
               "errors": [f"module failed to load: {type(exc).__name__}: {exc}"]})
        return
    for line in sys.stdin:
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            if req.get("cmd") == "validate":
                reply(dict(validate(mod, req.get("samples"), float(req.get("seconds", 40)), req.get("examples")), id=rid))
            elif req.get("cmd") == "is_legal":
                st = mod.state_from_observation(req.get("observation", ""), int(req.get("seat", 0)),
                                                copy.deepcopy(req.get("config") or {}), random.Random(0))
                legal = {norm(a): a for a in mod.legal_actions(st, int(req.get("seat", 0)))}
                a = legal.get(norm(req.get("action", "")))
                reply({"ok": True, "legal": a is not None, "action": a, "id": rid})
            elif req.get("cmd") == "crosscheck":
                reply(dict(crosscheck(mod, req.get("other", ""), float(req.get("seconds", 20))), id=rid))
            elif req.get("cmd") == "infer_opp":
                cfg_ = req.get("config") or {}
                reply(dict(infer_opponent(mod, req.get("prev_obs", ""), req.get("observation", ""), int(req.get("seat", 0)),
                                          cfg_, req.get("our_action", ""), float(req.get("seconds", 1.0))), id=rid))
            elif req.get("cmd") == "screen":
                reply(dict(screen(mod, req), id=rid))
            elif req.get("cmd") == "rank":
                reply(dict(rank(mod, req), id=rid))
            elif req.get("cmd") == "book":
                reply(dict(build_book(mod, float(req.get("seconds", 60)),
                                      os.path.join(os.path.dirname(os.path.abspath(path)), "book.json")), id=rid))
            elif req.get("cmd") == "candidates":
                reply(dict(candidates(mod, req), id=rid))
            elif req.get("cmd") == "replay":
                reply(dict(replay(mod, req), id=rid))
            elif req.get("cmd") == "evalcheck":
                reply(dict(evalcheck(mod, float(req.get("seconds", 8))), id=rid))
            elif req.get("cmd") == "duel":
                reply(dict(duel(mod, req), id=rid))
            else:
                reply({"ok": False, "error": "unknown cmd", "id": rid})
        except Exception as exc:  # noqa: BLE001
            reply({"ok": False, "error": _err(exc), "id": rid})


if __name__ == "__main__":
    main()

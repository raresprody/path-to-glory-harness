"""Prep pipeline: Haiku writes a simulator for the game from its rules; we test it and make Haiku fix it, within the
prep clock. The generated code only ever runs in a separate process (mesh/simworker.py) with time and memory limits.

Validation (all must hold to ship the simulator): random self-play never crashes and always terminates; rendering a
state as the arena's text and parsing it back keeps the legal moves; parsing the arena's real `sample_turn` gives
exactly the arena's legal moves. At play time the brain re-checks the legal moves on every turn (safety net).
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

from . import llm

log = llm.log
MAX_ATTEMPTS = 14  # bounded by the clock anyway (prep is not scored: keep trying while there is time)
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GEN_SYSTEM = """You are an expert Python engineer. You write exact, deterministic game simulators from rules texts.
Output ONLY one complete Python 3 module inside a single ```python code block. Standard library only. No I/O, no
print, no input(), no network, no threads. Never use eval/exec. Keep it under 400 lines and fast."""

GEN_SPEC = """Write a Python module that simulates this two-player game EXACTLY as the rules below describe, for
every variant. Seats are 0 and 1. A "state" is a plain dict (deep-copyable) holding EVERYTHING, including hidden
information (secrets, hands, dice to come are drawn with the rng when needed).

The module must define exactly these names:

CONFIGS = {<variant label exactly as given below>: {<the variant's parameters, e.g. sizes, targets, limits>}, ...}

def initial_state(config: dict, seed: int) -> dict
    # a fresh game for this variant; random setup (deals, secrets) drawn from random.Random(seed)
def to_move(state) -> list[int]
    # seats that must act now: [0], [1], or [0, 1] for simultaneous moves; [] when the game is over
def legal_actions(state, seat: int) -> list[str]
    # EVERY legal action for `seat`, spelled EXACTLY as the rules' action format and as in the examples
    # of legal actions below (same case and spacing); [] if seat is not to move.
    # If there can be more than 300 legal actions (e.g. every way to split 60 soldiers over 6 fields), return a
    # DETERMINISTIC representative list of at most 300 (same state -> same list; include even splits, extremes and
    # concentrated options) and make is_legal_action (below) decide legality exactly.
def apply(state, actions: dict, rng) -> dict
    # actions = {seat: action string} for every seat in to_move(state); returns the NEW state
    # (deep-copy first, never mutate the input); draw any randomness (dice, cards) from `rng` (random.Random)
def is_over(state) -> bool
def winner(state)  # 0 or 1 when that seat has won, None for a draw or while the game is not over
def observation(state, seat: int) -> str
    # the text the arena shows `seat`, reproducing the arena's format shown in the sample observations as
    # closely as possible (same lines, same wording), with only what that seat may know
def state_from_observation(text: str, seat: int, config: dict, rng) -> dict
    # PARSE an arena observation text for `seat` (format of the samples below) and return a FULL state that is
    # consistent with everything the text shows. Anything hidden from `seat` (opponent's secret/hand, deck order,
    # future dice) must be SAMPLED with `rng` among the possibilities consistent with the text (e.g. a secret
    # number inside the interval left by the feedback; cards not seen). Must work for any observation of any
    # turn, including when history lines are missing (then assume the minimal history consistent with the text).
    # legal_actions(state_from_observation(text, seat, ...), seat) must equal the arena's legal actions.
    # Parse robustly with re and split on lines; do not assume exact whitespace.

Also enforce the game's move cap / turn limit from the rules so every game ends. If the rules say a player wins
or loses by something said in chat, ignore chat (it is not part of the state).

Also define these three helpers (they are used when the list of legal actions is too long to enumerate):
def random_action(state, seat, rng) -> str
    # one uniformly random LEGAL action for `seat` (e.g. a random split of the soldiers that adds up exactly), drawn
    # with `rng`; used to play random games when legal_actions can't list everything
def is_legal_action(state, seat, action: str) -> bool
    # True iff `action` is a legal action for `seat` now, checking the exact format and every constraint of the
    # rules (counts, totals, ranges, length, letters...). Must agree with legal_actions(state, seat) when that list is
    # small enough to enumerate. Never raise: return False for anything malformed.
def could_win_now(state, seat, action: str) -> bool
    # `state` comes from state_from_observation, so its hidden parts are only ONE sample. Return True if there is
    # SOME hidden state consistent with everything `seat` has observed (all feedback, all revealed information) in
    # which playing `action` now would immediately achieve the game's winning target (e.g. the guess equals the
    # secret, given every earlier guess and its feedback). Check consistency from the observed history and the
    # feedback rules; do not rely on the sampled secret. Test `action` ITSELF against each piece of feedback `seat`
    # saw for its own earlier moves (would it have produced exactly that feedback if it were the answer?). Do NOT
    # require it to be in any built-in list of possible answers, and do NOT use information `seat` cannot see (e.g.
    # the opponent's hidden letters). Must be fast (called for thousands of candidates). If the game has no such
    # notion, return False."""


def _extract_code(text: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    if m:
        return max(m, key=len)
    if text and "def initial_state" in text:  # unterminated block (answer cut): take what is there
        return re.sub(r"^```(?:python|py)?\s*\n", "", text.strip())
    return ""


def _rules_block(prep: dict) -> str:
    parts = []
    for v in prep.get("variants") or []:
        part = f"=== VARIANT LABEL: {v.get('label', '')!r} ===\n{v.get('rules', '')}"
        smp = v.get("sample_turn")
        if smp:
            part += (f"\n--- SAMPLE OBSERVATION from the arena (seat 0, turn {smp.get('turn')}) ---\n"
                     f"{smp.get('observation', '')}\n--- its legal actions: {json.dumps(smp.get('legal_actions'))}")
        parts.append(part)
    return "\n\n".join(parts)


def _samples(prep: dict) -> list:
    out = []
    for v in prep.get("variants") or []:
        smp = v.get("sample_turn")
        if smp and smp.get("observation"):
            out.append({"label": v.get("label"), "observation": smp["observation"],
                        "legal_actions": smp.get("legal_actions"), "seat": 0})
    return out


# ------------------------------------------------------------------ worker process
# PyPy runs the generated simulators and the search ~4-5x faster than CPython (measured by path-to-glory-3 under the
# arena limits: tic_tac_toe 5.2k -> 24.9k playouts/s). The image ships it at /opt/pypy; anything that fails to load
# under PyPy (e.g. syntax newer than 3.10) is retried with CPython, per simulator.
PYPY = "/opt/pypy/bin/pypy3"
_PYPY_BAD: set = set()  # simulator paths that did not load under PyPy


def _interpreter(sim_path: str) -> str:
    if os.environ.get("MESH_PYPY", "1") == "1" and os.path.exists(PYPY) and sim_path not in _PYPY_BAD:
        return PYPY
    return sys.executable


class Worker:
    def __init__(self, sim_path: str, load_timeout: float = 15.0):
        self._start(sim_path, load_timeout, _interpreter(sim_path))
        if self.exe != sys.executable and not (self.hello or {}).get("loaded"):
            log(f"simgen: PyPy could not load {os.path.basename(sim_path)}; using CPython for it")
            _PYPY_BAD.add(sim_path)
            self.close()
            self._start(sim_path, load_timeout, sys.executable)

    def _start(self, sim_path: str, load_timeout: float, exe: str):
        env = dict(os.environ)
        env["PYTHONPATH"] = PKG_PARENT
        self.exe = exe
        self.p = subprocess.Popen([exe, "-u", "-m", "mesh.simworker", sim_path],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  cwd=PKG_PARENT, env=env)
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.seq = 0
        threading.Thread(target=self._read, daemon=True).start()
        self.hello = self._get(load_timeout)

    def _read(self):
        try:
            for line in self.p.stdout:
                try:
                    self.q.put(json.loads(line))
                except ValueError:
                    pass
        finally:
            self.q.put(None)

    def _get(self, timeout):
        try:
            return self.q.get(timeout=max(0.1, timeout))
        except queue.Empty:
            return None

    def ask(self, req: dict, timeout: float):
        """One request, one reply. Thread-safe: a turn abandoned by the watchdog may still be waiting here when the
        next turn asks, so requests are serialized and every reply is matched to its request id (a stale reply for
        an older observation can never be used for a newer one)."""
        end = time.monotonic() + timeout
        if not self.lock.acquire(timeout=max(0.05, timeout)):
            return None
        try:
            if self.p.poll() is not None:
                return None
            self.seq += 1
            rid = self.seq
            try:
                self.p.stdin.write((json.dumps(dict(req, id=rid)) + "\n").encode())
                self.p.stdin.flush()
            except OSError:
                return None
            while True:
                got = self._get(end - time.monotonic())
                if got is None:  # hung or dead: kill it, a fresh one is started next time
                    self.close()
                    return None
                if got.get("id") == rid:
                    return got
        finally:
            self.lock.release()

    def alive(self):
        return self.p.poll() is None

    def close(self):
        try:
            self.p.kill()
        except OSError:
            pass


def _validate(path: str, samples: list, seconds: float, examples=None) -> dict:
    w = Worker(path)
    try:
        if not w.hello or not w.hello.get("loaded"):
            errs = (w.hello or {}).get("errors") or ["module did not load (syntax error or crash at import?)"]
            if w.hello and w.hello.get("error"):
                errs = [w.hello["error"]]
            return {"ok": False, "score": 0.0, "errors": errs}
        rep = w.ask({"cmd": "validate", "samples": samples, "seconds": seconds, "examples": examples or []},
                    timeout=seconds + 15)
        return rep or {"ok": False, "score": 0.0, "errors": ["validation hung (infinite loop?) or crashed the process"]}
    finally:
        w.close()


# ------------------------------------------------------------------ engine used at play time
PARALLEL = os.environ.get("MESH_PARALLEL_SEARCH", "1") == "1"


def _pool(a, b):
    """Merge two search reports of the same position: visits add up, values are visit-weighted. Book, tactics and
    sampled answers are deterministic, so the first one is used as is. Either one alone is fine if the other failed."""
    if not b or not b.get("ok") or not b.get("agree"):
        return a
    if not a or not a.get("ok") or not a.get("agree"):
        return b
    if a.get("book") or a.get("sampled") or a.get("mixed") or not a.get("playouts") or not b.get("playouts"):
        return a
    tab = {}
    for c in (a.get("candidates") or []) + (b.get("candidates") or []):
        t = tab.setdefault(c["action"], {"action": c["action"], "n": 0, "w": 0.0})
        t["n"] += c.get("n", 0)
        t["w"] += c.get("value", 0.0) * c.get("n", 0)
    out = [{"action": t["action"], "n": t["n"], "value": round(t["w"] / t["n"], 3) if t["n"] else 0.0} for t in tab.values()]
    out.sort(key=lambda c: (c["n"], c["value"]), reverse=True)
    top = out[0]
    for c in out[1:]:  # same rule as the worker: a far less searched move is not "near best"
        if c["n"] < 0.5 * top["n"]:
            c["value"] = min(c["value"], round(top["value"] - 0.06, 3))
    return dict(a, candidates=out[:8], playouts=a.get("playouts", 0) + b.get("playouts", 0),
                informative=bool(a.get("informative") or b.get("informative")))


class Engine:
    def __init__(self, folder: str, configs_labels: list | None = None):
        self.folder = folder
        self.path = os.path.join(folder, "sim.py")
        self.worker = None
        self.worker2 = None
        self.report = {}
        try:
            with open(os.path.join(folder, "report.json"), encoding="utf-8") as fh:
                self.report = json.load(fh)
        except (OSError, ValueError):
            pass
        self.labels = self.report.get("labels") or configs_labels or []
        self.configs = self.report.get("configs") or {}

    def _config(self, start: dict):
        label = str((start.get("variant") or {}).get("label", ""))
        if label in self.configs:
            return self.configs[label]
        low = {k.lower().strip(): v for k, v in self.configs.items()}
        return low.get(label.lower().strip())

    def candidates(self, msg: dict, start: dict, deadline: float, opp_freq: dict | None = None,
                   simultaneous: bool = True, opp_model: dict | None = None, opp_weight: float = 0.5):
        """List of candidates (best first) when the simulator agrees with the arena; None means: don't trust the
        simulator for this game (parser failure, legal moves differ, crash)."""
        cfg = self._config(start)
        if cfg is None:
            log(f"engine: no config for variant {(start.get('variant') or {}).get('label')!r}")
            return None
        if self.worker is None or not self.worker.alive():
            self.worker = Worker(self.path, load_timeout=5.0)
            if not self.worker.hello or not self.worker.hello.get("loaded"):
                return None
        if PARALLEL and (self.worker2 is None or not self.worker2.alive()):
            try:
                self.worker2 = Worker(self.path, load_timeout=5.0)
                if not self.worker2.hello or not self.worker2.hello.get("loaded"):
                    self.worker2 = None
            except Exception:  # noqa: BLE001 - one worker is enough
                self.worker2 = None
        budget = max(0.3, deadline - time.monotonic() - 0.5)
        req = {"cmd": "candidates", "observation": msg.get("observation", ""), "seat": start.get("seat", 0),
               "config": cfg, "legal": msg.get("legal_actions"), "seconds": budget,
               "opp_freq": opp_freq or {}, "rm": bool(simultaneous), "search": self.report.get("search", "mcts"),
               "opp_weight": opp_weight}
        if opp_model and self.report.get("library"):
            # what this opponent did in public replays, for a library simulator with its own policy (a generated
            # simulator never gets an extra config key it does not know)
            req["config"] = dict(cfg, _opp=opp_model)
        if self.worker2 is None:
            rep = self.worker.ask(req, timeout=budget + 2.0)
        else:
            # the container has 2 CPUs and one search uses one: run two independent searches (different random
            # seeds) at the same time and pool their statistics, which roughly doubles the playouts per move
            box = {}
            th = threading.Thread(target=lambda: box.update(r2=self.worker2.ask(dict(req, seed=2), timeout=budget + 2.0)),
                                  daemon=True)
            th.start()
            rep = self.worker.ask(dict(req, seed=1), timeout=budget + 2.0)
            th.join(budget + 2.5)
            rep = _pool(rep, box.get("r2"))
        if not rep:  # timeout (or a turn the watchdog abandoned still holds the worker): no engine this turn only
            log("engine: no answer in time; Haiku plays this turn")
            return []
        if not rep.get("ok"):  # the parser or the simulator crashed on this observation: stop trusting it
            log(f"engine failed: {rep.get('error', 'no answer')}")
            return None
        if not rep.get("agree"):
            log(f"engine disagrees with the arena: {rep.get('why')}")
            return None
        log(f"engine: {rep.get('playouts', 0)} playouts, informative={rep.get('informative')}, "
            f"top {rep.get('candidates', [])[:3]}")
        cands = rep.get("candidates") or []
        # only a sampled huge action space hands the choice to Haiku; otherwise Haiku stays within near-best moves
        # (in a dead-drawn position every move "looks" equal, but a free choice could still blunder)
        if cands and rep.get("sampled"):
            for c in cands:
                c["uninformative"] = True
            if rep.get("hint"):
                cands[0]["hint"] = rep["hint"]
        return cands

    def is_legal(self, msg: dict, start: dict, action: str, timeout: float = 2.0):
        """The simulator's spelling of `action` if legal in this observation, else None."""
        cfg = self._config(start)
        if cfg is None or not self._ensure_worker():
            return None
        rep = self.worker.ask({"cmd": "is_legal", "observation": msg.get("observation", ""),
                               "seat": start.get("seat", 0), "config": cfg, "action": action}, timeout=timeout)
        return (rep or {}).get("action") if (rep or {}).get("legal") else None

    def _ensure_worker(self) -> bool:
        """Start the simulator process on first use. Before, only candidates() started it, so on the first turn of a
        game screen() returned None and the feedback filter was skipped (Wordle: 'guess erise' with R and E green)."""
        try:
            if self.worker is None or not self.worker.alive():
                self.worker = Worker(self.path, load_timeout=5.0)
            return bool(self.worker.hello and self.worker.hello.get("loaded")) and self.worker.alive()
        except Exception:  # noqa: BLE001
            return False

    def helpers(self) -> dict:
        return (self.report or {}).get("helpers") or {}

    def screen(self, msg: dict, start: dict, actions: list, timeout: float = 3.0):
        """Verdicts (legal / could still win / already played) on model-proposed moves; None if unavailable."""
        cfg = self._config(start)
        if cfg is None or not self._ensure_worker():
            return None
        h = self.helpers()
        return self.worker.ask({"cmd": "screen", "observation": msg.get("observation", ""), "config": cfg,
                                "seat": start.get("seat", 0), "actions": actions,
                                "use_il": bool(h.get("is_legal_action")), "use_cw": bool(h.get("could_win_now")),
                                "seconds": max(0.5, timeout - 1.0)},
                               timeout=timeout)

    def rank(self, msg: dict, start: dict, ours: list, theirs: list, seconds: float, opp_freq: dict | None = None):
        cfg = self._config(start)
        if cfg is None or not self._ensure_worker():
            return None
        return self.worker.ask({"cmd": "rank", "observation": msg.get("observation", ""), "config": cfg,
                                "seat": start.get("seat", 0), "ours": ours, "theirs": theirs, "seconds": seconds,
                                "opp_freq": opp_freq or {}}, timeout=seconds + 3.0)

    def infer_opponent(self, prev_obs: str, obs: str, start: dict, our_action: str, raw: bool = False):
        """The opponent's last move, inferred with the simulator (None when unsure). raw=True: the whole reply
        ({"action", "score", "margin"}; no "score" when the opponent did not move in between)."""
        cfg = self._config(start)
        if cfg is None or not self._ensure_worker():
            return None
        rep = self.worker.ask({"cmd": "infer_opp", "prev_obs": prev_obs, "observation": obs, "config": cfg,
                               "seat": start.get("seat", 0), "our_action": our_action, "seconds": 1.0}, timeout=3.0)
        return rep if raw else (rep or {}).get("action")

    def build_book(self, seconds: float) -> dict:
        """Idle prep time: deeper search on the opening positions, saved as book.json next to sim.py."""
        if seconds < 10:
            return {}
        w = Worker(self.path, load_timeout=10.0)
        try:
            rep = w.ask({"cmd": "book", "seconds": seconds}, timeout=seconds + 20) or {}
        finally:
            w.close()
        if self.worker:  # reload so the play-time worker sees the book
            self.worker.close()
            self.worker = None
        return rep

    def close(self):
        if self.worker:
            self.worker.close()
            self.worker = None
        if self.worker2:
            self.worker2.close()
            self.worker2 = None


def _has_command_format(prep: dict, digest: dict) -> bool:
    """True when the rules' example moves share a fixed command word followed by one token ("guess planet",
    "bid 30", "drop 4"): a game a simulator can model, whatever the talk_is_move flag says."""
    from .fallback import examples_from_rules
    exs = list(digest.get("example_actions") or [])
    for v in prep.get("variants") or []:
        exs += examples_from_rules(v.get("rules", ""))
    heads = [e.split()[0].lower() for e in exs if isinstance(e, str) and len(e.split()) == 2]
    return len(heads) >= 2 and len(set(heads)) == 1


HIDDEN_RE = '(\\b(secret|hidden)\\s+(english\\s+)?(word|number|code|answer)|\\bthe secret\\b|\\bguess (the|their|my|your)\\b)'


def _hidden_answer_hint(prep: dict) -> bool:
    """The rules speak of a hidden answer to find (a secret word, number or code): a deduction game, not free talk.
    Keeps trivia/answer games ("answer 1987") on the language side."""
    import re
    text = " ".join(v.get("rules", "") for v in prep.get("variants") or [])
    return re.search(HIDDEN_RE,
                     text, re.I) is not None


def needs_untalk(prep: dict, digest: dict) -> bool:
    """A game tagged 'talk' that is really a command-format deduction game (Wordle): treat it as a normal game."""
    return bool(digest.get("talk_is_move")) and _has_command_format(prep, digest) and _hidden_answer_hint(prep)


# t3/library candidate: replay-checked simulators for the public games (rule 10.7); MESH_KNOWN_DIR overrides it in tests.
KNOWN_DIR = os.environ.get("MESH_KNOWN_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "known")


def known_path(game_id, labels: list):
    """Path of our hand-written, replay-checked simulator for a game the arena has made public (mesh/known/, rule
    10.7), if it covers every variant label of this prep; else None (the simulator is generated as usual)."""
    gid = re.sub(r"[^a-z0-9_]", "", str(game_id or "").lower())
    path = os.path.join(KNOWN_DIR, gid + ".py")
    if not gid or gid.startswith("_") or not os.path.exists(path):
        return None
    configs = _read_configs(path)
    missing = [l for l in labels if l not in configs]
    if missing:
        log(f"simgen: known simulator for {gid} lacks variant(s) {missing}; generating instead")
        return None
    return path


def install_known(prep: dict, digest: dict, folder: str, seconds: float = 20.0):
    """Use the replay-checked simulator when the game is one of the public ones. It still has to pass the same
    validation as a generated one on THIS prep's real sample turns (so changed rules or formats fall back to
    generation). Returns an Engine or None."""
    try:
        labels = [str(v.get("label", "")) for v in prep.get("variants") or []]
        path = known_path(prep.get("game_id"), labels)
        if not path:
            return None
        os.makedirs(folder, exist_ok=True)
        rep = _validate(path, _samples(prep), seconds, digest.get("example_actions"))
        if not rep.get("ok") or rep.get("ok_labels"):
            log(f"simgen: known simulator for {prep.get('game_id')} failed validation here (score {rep.get('score')}): "
                f"{str(rep.get('errors'))[:300]}; generating instead")
            return None
        shutil.copyfile(path, os.path.join(folder, "sim.py"))
        # review #3: data the known module reads relative to its own file (mini_poker's CFR strategy table)
        data = os.path.join(os.path.dirname(path), "data")
        if os.path.isdir(data):
            shutil.copytree(data, os.path.join(folder, "data"), dirs_exist_ok=True)
        configs = _read_configs(os.path.join(folder, "sim.py"))
        _save_report(folder, {"ok": True, "score": rep.get("score"), "attempts": 0, "stats": rep.get("stats"),
                              "labels": list(configs), "configs": configs, "helpers": rep.get("helpers") or {},
                              "known": True, "library": True})
        with open(os.path.join(os.path.dirname(folder), "engine.json"), "w", encoding="utf-8") as fh:
            json.dump({"folder": os.path.basename(folder)}, fh)
        log(f"simgen: replay-checked simulator for {prep.get('game_id')} installed (score {rep.get('score')}, "
            f"helpers {rep.get('helpers')})")
        return Engine(folder)
    except Exception as exc:  # noqa: BLE001 - never block the normal pipeline
        log(f"simgen: known simulator not used: {type(exc).__name__}: {exc}")
        return None


EXPERIENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experience")
RANDOM_END_RE = r"\bchance\b|\brandom(ly)?\b|\bprobabilit|\b1 in \d|\bcoin\b"


def _experience(game_id, labels: list):
    """Real games of this game from public replays (mesh/experience/<game_id>.json.gz, built by
    tools/build_experience.py; rule 10.7), or None for a game we have never seen played. Data only: the simulator is
    still written by the model in prep; these games are what it is checked against."""
    import gzip
    gid = re.sub(r"[^a-z0-9_]", "", str(game_id or "").lower())
    path = os.path.join(EXPERIENCE_DIR, gid + ".json.gz")
    if not gid or not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            exp = json.load(fh)
        games = [g for g in exp.get("games") or [] if g.get("label") in labels]
        return {"reasons": exp.get("reasons") or [], "games": games} if games else None
    except (OSError, ValueError) as exc:
        log(f"simgen: experience for {gid} unreadable: {exc}")
        return None


def _replay(path: str, exp: dict, hidden: bool, random_end: bool, seconds: float) -> dict:
    """Run the candidate simulator through the real games (sandboxed worker). {} when the check itself failed."""
    w = Worker(path)
    try:
        if not w.hello or not w.hello.get("loaded"):
            return {}
        r = w.ask({"cmd": "replay", "games": exp["games"], "hidden": hidden, "random_end": random_end,
                   "seconds": seconds}, timeout=seconds + 10)
        return r if r and r.get("ok") else {}
    finally:
        w.close()


def build(prep: dict, digest: dict, folder: str, deadline: float, min_left: float = 90.0):
    """Generate + validate + fix until the deadline. Returns an Engine or None."""
    known = install_known(prep, digest, folder, seconds=min(25.0, max(8.0, deadline - time.monotonic() - 30)))
    if known is not None:  # a public game with a replay-checked simulator: no generation, whatever the digest says
        return known
    if digest.get("talk_is_move") and not (_has_command_format(prep, digest) and _hidden_answer_hint(prep)):
        # really free text (a message, an answer in words): no simulator. The flag comes from one model answer and
        # flips between runs (Wordle was once tagged "talk"), so a fixed command format ("guess <word>") wins over it.
        log("simgen: talk-is-move game, no simulator (language game)")
        return None
    os.makedirs(folder, exist_ok=True)
    rules = _rules_block(prep)
    samples = _samples(prep)
    # games already played in public (rule 10.7): real games to check the generated simulator against, and the
    # arena's own result sentences as part of the rules the model reads (rr1: "nobody reached 16 ...; seat 1 defected
    # fewer times" was the rule our generated Prisoner's Dilemma simulator had wrong)
    exp = _experience(prep.get("game_id"), [str(v.get("label", "")) for v in prep.get("variants") or []])
    hidden = bool(digest.get("hidden_information"))
    random_end = re.search(RANDOM_END_RE, rules, re.I) is not None
    if exp:
        if exp["reasons"]:
            rules += ("\n\n=== HOW REAL GAMES OF THIS GAME ENDED IN THE ARENA (the arena's own result sentences; your "
                      "is_over() and winner() must produce exactly these outcomes) ===\n"
                      + "\n".join(f"- {r}" for r in exp["reasons"][:14]))
        log(f"simgen: {len(exp['games'])} real games of {prep.get('game_id')} to check against "
            f"(hidden={hidden}, random_end={random_end})")
    best = {"score": -1.0, "rank": (-1, -1, -1, -1.0)}
    code, errors = "", []
    attempt, empty = 0, 0
    scores, all_errors = [], []
    while attempt < MAX_ATTEMPTS:
        left = deadline - time.monotonic()
        if left < min_left:  # one generation (~10-60 s) + validation (10-30 s) must still fit
            break
        attempt += 1
        if code and len(scores) >= 2 and scores[-1] == scores[-2] and not best.get("ok"):
            # the repair loop is stuck (same score twice): start over from the rules, with every error seen so far
            log("simgen: repairs are stuck; regenerating from scratch with all errors so far")
            seen = "\n\n".join(all_errors[-8:])
            user = (f"{GEN_SPEC}\n\nRULES OF ALL VARIANTS (with real arena sample observations):\n\n{rules}\n\n"
                    f"EARLIER ATTEMPTS FAILED WITH THESE ERRORS (avoid them):\n{seen}")
            scores.append(-1.0)
        elif not code:
            user = f"{GEN_SPEC}\n\nRULES OF ALL VARIANTS (with real arena sample observations):\n\n{rules}"
        else:
            errs = "\n\n".join(errors[:6])
            user = (f"{GEN_SPEC}\n\nRULES OF ALL VARIANTS (with real arena sample observations):\n\n{rules}\n\n"
                    f"YOUR PREVIOUS MODULE:\n```python\n{code}\n```\n\nTESTS FAILED:\n{errs}\n\n"
                    "Fix every problem and output the COMPLETE corrected module.")
        t0 = time.monotonic()
        text = llm.chat(GEN_SYSTEM, user, timeout=min(150.0, left - 40), max_tokens=12000, temperature=0.2)
        new = _extract_code(text)
        log(f"simgen attempt {attempt}: {len(new)} chars of code in {time.monotonic() - t0:.0f}s")
        if not new:
            empty += 1
            if empty >= 3 or not text:  # model down / key exhausted / refusing: stop, Haiku-per-move will play
                if not text:
                    time.sleep(min(5.0, max(0.0, deadline - time.monotonic() - min_left)))
                if empty >= 3:
                    log("simgen: no code from the model 3 times; giving up")
                    break
            errors = ["Your answer had no ```python code block. Output the complete module in one block."]
            continue
        empty = 0
        code = new
        path = os.path.join(folder, f"sim_try{attempt}.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        vsec = max(10.0, min(30.0, deadline - time.monotonic() - 50))
        rep = _validate(path, samples, vsec, digest.get("example_actions"))
        errors = rep.get("errors") or []
        scores.append(rep.get("score", 0))
        all_errors.extend(e[:600] for e in errors[:3])
        log(f"simgen attempt {attempt}: ok={rep.get('ok')} score={rep.get('score')} stats={rep.get('stats')} "
            f"errors={[e[:160] for e in errors[:3]]}")
        rr = None
        if rep.get("ok") and exp:
            # it passes our own tests: now the real games. Differences go back to the model as the next errors.
            rr = _replay(path, exp, hidden, random_end, max(6.0, min(25.0, deadline - time.monotonic() - 45)))
            if rr:
                log(f"simgen attempt {attempt}: real games: {rr.get('moves')} moves checked, {rr.get('ends_ok')}/"
                    f"{rr.get('ends')} endings right, failures {rr.get('kinds') or 'none'}")
                if not rr.get("clean"):
                    errors = list(rr.get("fails") or []) + [e for e in errors if not e.startswith("(minor)")]
                    all_errors.extend(e[:900] for e in errors[:3])
        clean = bool(rr and rr.get("clean"))
        # every variant passes (2) beats some variants pass (1, simworker.validate's ok_labels) beats none (0)
        full = bool(rep.get("ok")) and not rep.get("ok_labels")
        if rep.get("ok_labels"):
            log(f"simgen attempt {attempt}: usable on {rep.get('ok_labels')} only (fails {rep.get('failed_labels')})")
        rank = (2 if full else 1 if rep.get("ok") else 0, 1 if clean else 0,
                -sum((rr or {}).get("kinds", {}).values()) if rr else -10 ** 6, rep.get("score", 0))
        if rank > best["rank"]:
            best = {"score": rep.get("score", 0), "ok": rep.get("ok"), "path": path, "report": rep, "rank": rank,
                    "replay": rr or {}}
        if full and (clean or not exp or not rr) and not any(not e.startswith("(minor)") for e in errors):
            break
        if full and attempt >= 3 and (clean or not exp or not rr):
            break
    if not best.get("ok"):
        log(f"simgen: no simulator passed validation (best score {best['score']}); Haiku will play this game")
        _save_report(folder, {"ok": False, "best_score": best["score"], "attempts": attempt})
        return None
    rpl = best.get("replay") or {}
    # verified = it reproduces real games move by move (and, where the checker can tell, their endings): a better
    # referee than a second generated simulator, so no cross-check, and the brain treats its values as trusted
    verified = bool(rpl.get("clean")) and rpl.get("moves", 0) >= 10
    if exp:
        log(f"simgen: best simulator vs real games: {'reproduces them' if verified else 'still differs'} "
            f"({rpl.get('moves')} moves, {rpl.get('ends_ok')}/{rpl.get('ends')} endings, failures {rpl.get('kinds') or 'none'})")
    if not verified:
        try:  # idle prep time: check the accepted simulator against an independently written one
            better = _crosscheck_round(best, rules, samples, digest, folder, deadline)
            if better:
                best = better
        except Exception as exc:  # noqa: BLE001 - the accepted simulator stays
            log(f"simgen: cross-check skipped: {type(exc).__name__}: {exc}")
    shutil.copyfile(best["path"], os.path.join(folder, "sim.py"))
    configs = _read_configs(os.path.join(folder, "sim.py"))
    part = best["report"].get("ok_labels")
    if part:
        # only the variants it passes: Engine._config finds no config for the others, so the brain switches the
        # engine off for those games alone and Haiku plays them
        configs = {k: v for k, v in configs.items() if k in part}
        log(f"simgen: simulator accepted for {list(configs)} only; Haiku plays {best['report'].get('failed_labels')}")
    _save_report(folder, {"ok": True, "partial": bool(part), "score": best["score"], "attempts": attempt, "stats": best["report"].get("stats"),
                          "labels": list(configs), "configs": configs, "helpers": best["report"].get("helpers") or {},
                          "known": verified, "replay": {k: rpl.get(k) for k in ("moves", "ends_ok", "ends", "kinds")}})
    log(f"simgen: helpers {best['report'].get('helpers')}")
    with open(os.path.join(os.path.dirname(folder), "engine.json"), "w", encoding="utf-8") as fh:
        json.dump({"folder": os.path.basename(folder)}, fh)
    log(f"simgen: simulator accepted (score {best['score']}, {attempt} attempts)")
    return Engine(folder)


EVAL_SYSTEM = """You are an expert game programmer. You write fast position-evaluation functions for game engines.
Output ONLY Python code inside a single ```python code block. Standard library only, no I/O, no prints."""

EVAL_SPEC = """Below are the rules of a two-player game and the Python simulator our engine uses for it. Write

def evaluate(state, seat) -> float

that says how good the NON-FINAL position `state` is for player `seat` (0 or 1): a number between -1.0 (almost
lost) and +1.0 (almost won), 0.0 = even. The engine searches a few moves ahead and calls evaluate at the end of each
line, hundreds of thousands of times, so:
- read only fields of `state` (see the simulator for its exact structure); never copy the state, never call apply
  or legal_actions in a loop over many moves, no randomness; a few simple loops over the board are fine;
- reward what the RULES say wins: the score / material / captured-points difference if the game has one (scaled so a
  decisive lead is near +-0.8), immediate threats (lines one move from complete, pieces under attack, boxes one side
  from done and who would get them), progress toward the winning condition (shortest remaining connection, cells or
  tokens still needed) and tempo (whose turn it is) where it matters; mind every tie-break the rules give;
- it must be antisymmetric in spirit: what is good for seat 0 is bad for seat 1;
- in a game with hidden information `state` is ONE sampled possibility: evaluate it as if it were true (e.g. count
  what has really been hit / won so far);
- name any helper functions and constants with the prefix _ev_ (they are appended to the simulator module, whose
  names you may use but must not redefine).
Output only evaluate and its _ev_ helpers."""


def add_evaluation(engine, prep: dict, digest: dict, deadline: float) -> dict:
    """Idle prep time (not scored): the model writes evaluate(state, seat) for this game; it is appended to a COPY of
    the simulator, checked (no crash, fast, points the right way on random games), and then DUELS the playout search
    on the simulator. Only a winner replaces the search: report["search"] = "eval". Game-agnostic; measured need:
    150-260 random playouts per move in Hex / Dots, 6 in Mini Chess (official image, Rares 15 Nov)."""
    out = {"used": False}
    try:
        if engine is None or deadline - time.monotonic() < 120:
            return out
        with open(engine.path, encoding="utf-8") as fh:
            sim_code = fh.read()
        if "def evaluate(" in sim_code or "def recommend(" in sim_code:
            return out  # already has an evaluation, or its own replay-checked policy (t3/library)
        rules = _rules_block(prep)
        folder = engine.folder
        errors, code = "", ""
        for attempt in (1, 2):
            if deadline - time.monotonic() < 100:
                break
            user = (f"{EVAL_SPEC}\n\nRULES OF ALL VARIANTS:\n\n{rules[:24000]}\n\nTHE SIMULATOR (state structure):\n"
                    f"```python\n{sim_code}\n```"
                    + (f"\n\nYOUR PREVIOUS evaluate:\n```python\n{code}\n```\nIT FAILED: {errors}\nFix it." if errors else ""))
            code = _extract_code(llm.chat(EVAL_SYSTEM, user, timeout=min(90.0, deadline - time.monotonic() - 60),
                                          max_tokens=6000, temperature=0.2))
            if "def evaluate(" not in code:
                errors = "no evaluate function in the answer"
                continue
            cand = os.path.join(folder, "sim_eval.py")
            with open(cand, "w", encoding="utf-8") as fh:
                fh.write(sim_code.rstrip() + "\n\n\n# ---- evaluation function written in prep (simgen.add_evaluation) ----\n" + code + "\n")
            w = Worker(cand)
            try:
                if not w.hello or not w.hello.get("loaded"):
                    errors = f"the module no longer loads: {(w.hello or {}).get('error')}"
                    continue
                chk = w.ask({"cmd": "evalcheck", "seconds": 8}, timeout=25) or {"ok": False, "why": "check hung"}
                log(f"simgen evaluation attempt {attempt}: {chk}")
                if not chk.get("ok"):
                    errors = str(chk.get("why"))
                    continue
                # two ways to use it, each duels today's playout search; the better winner is adopted:
                #   "mcts+eval": the same tree search, playouts cut at 12 plies and scored by evaluate (Rares's
                #                tests/fixb: 24-8 over Connect Four, Dots, Hex, Mini Chess);
                #   "eval":      alpha-beta (visible, deterministic games) / expected evaluation over sampled worlds
                dsec = max(15.0, min(150.0, (deadline - time.monotonic() - 45) / 2))  # prep idle time, per mode
                duels = {}
                for mode in ("mcts+eval", "eval"):
                    r = w.ask({"cmd": "duel", "mode": mode, "seconds": dsec, "move_seconds": 0.25, "max_games": 12},
                              timeout=dsec + 60) or {}
                    duels[mode] = {"games": r.get("games", 0), "eval": r.get("eval", 0), "mcts": r.get("mcts", 0)}
                    log(f"simgen evaluation duel [{mode}]: {r.get('eval')} - {r.get('mcts')} playout search "
                        f"({r.get('games')} games)")
            finally:
                w.close()
            best_mode = max(duels, key=lambda m: duels[m]["eval"] - duels[m]["mcts"])
            res = duels[best_mode]
            out.update(check=chk, duel=duels, mode=best_mode)
            # adopted only on a clear result: at least two points ahead over >= 4 games (review #11: no fluke)
            if res.get("games", 0) >= 4 and res.get("eval", 0) - res.get("mcts", 0) >= 2.0:  # review #11
                shutil.copyfile(cand, engine.path)
                engine.report["search"] = best_mode
                saved = _read_report(folder)
                saved["search"] = best_mode
                saved["evaluation"] = out
                _save_report(folder, saved)
                engine.close()
                out["used"] = True
            break
    except Exception as exc:  # noqa: BLE001 - the playout search stays
        log(f"simgen evaluation skipped: {type(exc).__name__}: {exc}")
    return out


def _read_report(folder: str) -> dict:
    try:
        with open(os.path.join(folder, "report.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


INDEPENDENT_NOTE = ("Write this module from scratch, with your own choice of state representation and helper functions: "
                    "it will be compared move by move with an independent implementation of the same rules, so read "
                    "every rule (pending bets, ties, end conditions, scoring, what each seat can see) with great care.")

ARBITER_SYSTEM = ("You are a careful referee for a two-player text game in an AI tournament. You decide which list of legal "
                  "actions is correct according to the rules text. JSON only.")


def _crosscheck(path_a: str, path_b: str, seconds: float) -> dict:
    w = Worker(path_a)
    try:
        if not w.hello or not w.hello.get("loaded"):
            return {}
        return w.ask({"cmd": "crosscheck", "other": path_b, "seconds": seconds}, timeout=seconds + 15) or {}
    finally:
        w.close()


def _arbitrate(rules: str, examples: list, timeout: float) -> list:
    """Haiku decides, against the rules text, which simulator's legal moves are right at each disputed position.
    Returns error lines for simulator A (the accepted one) where it was judged wrong."""
    items = "\n\n".join(f"CASE {i}: variant {e['variant']!r}, seat {e['seat']} to move. Observation:\n{e['observation']}\n"
                        f"List A: {e['legal_A']}\nList B: {e['legal_B']}" for i, e in enumerate(examples))
    user = (f"RULES:\n{rules[:20000]}\n\nTwo simulators of this game disagree on the legal actions in these positions.\n\n"
            f"{items}\n\nFor each case, decide from the rules which list is correct. Answer with JSON only: "
            '{"verdicts": [{"case": 0, "correct": "A" | "B" | "neither", "legal": ["<the correct legal actions>"], '
            '"rule": "<the rule that decides it, short>"}]}')
    got = llm.extract_json(llm.chat(ARBITER_SYSTEM, user, timeout=timeout, max_tokens=2500, temperature=0,
                                    thinking_budget=2000)) or {}
    errors, n_b = [], 0
    for v in got.get("verdicts") or []:
        try:
            e = examples[int(v.get("case"))]
        except (TypeError, ValueError, IndexError):
            continue
        if v.get("correct") == "B":
            n_b += 1
        if v.get("correct") in ("B", "neither"):
            right = v.get("legal") if v.get("correct") == "neither" else e["legal_B"]
            errors.append(f"WRONG LEGAL MOVES (variant {e['variant']!r}, seat {e['seat']}): for this observation\n"
                          f"{e['observation']}\nyour legal_actions(state_from_observation(...)) = {e['legal_A']}, but the "
                          f"rules say they are {right} ({v.get('rule', '')}). Fix how you parse and/or apply this rule.")
    return errors, n_b


def _crosscheck_round(best: dict, rules: str, samples: list, digest: dict, folder: str, deadline: float):
    """Second simulator + cross-check + arbitration + one fix round, only when there is time. Returns a better `best`
    or None (keep the accepted one)."""
    if deadline - time.monotonic() < 150:
        return None
    t0 = time.monotonic()
    user = f"{GEN_SPEC}\n\n{INDEPENDENT_NOTE}\n\nRULES OF ALL VARIANTS (with real arena sample observations):\n\n{rules}"
    code_b = _extract_code(llm.chat(GEN_SYSTEM, user, timeout=min(120.0, deadline - time.monotonic() - 90),
                                    max_tokens=12000, temperature=0.7))
    if not code_b:
        return None
    path_b = os.path.join(folder, "sim_independent.py")
    with open(path_b, "w", encoding="utf-8") as fh:
        fh.write(code_b)
    rep_b = _validate(path_b, samples, 12.0, digest.get("example_actions"))
    if not rep_b.get("ok") or rep_b.get("ok_labels"):
        log(f"simgen cross-check: second simulator failed its own validation (score {rep_b.get('score')}); skipped")
        return None
    cc = _crosscheck(best["path"], path_b, 15.0)
    log(f"simgen cross-check: {cc.get('checked')} positions, {cc.get('diffs')} disagreements "
        f"({time.monotonic() - t0:.0f}s)")
    if not cc.get("diffs"):
        return None
    left = deadline - time.monotonic()
    errors, n_b = _arbitrate(rules, cc.get("examples") or [], timeout=min(60.0, left - 70))
    n_ex = len(cc.get("examples") or [])
    if n_ex and n_b >= max(2, int(0.8 * n_ex)) and rep_b.get("score", 0) >= best.get("score", 0) - 0.05:
        # the referee sides with the independent simulator in (almost) every disputed case and it passed its own
        # validation: switch to it rather than keep a simulator the rules contradict (Blotto, 6 of 6, T2 tests)
        log(f"simgen cross-check: arbiter rejects the accepted simulator in {len(errors)} of {n_ex} cases; "
            f"switching to the independent one (score {rep_b.get('score')}, sided with it {n_b}x)")
        return {"score": rep_b.get("score"), "ok": True, "path": path_b, "report": rep_b}
    log(f"simgen cross-check: arbiter says the accepted simulator is wrong in {len(errors)} of "
        f"{len(cc.get('examples') or [])} cases")
    if not errors or deadline - time.monotonic() < 70:
        return None
    with open(best["path"], encoding="utf-8") as fh:
        code_a = fh.read()
    user = (f"{GEN_SPEC}\n\nRULES OF ALL VARIANTS (with real arena sample observations):\n\n{rules}\n\n"
            f"YOUR PREVIOUS MODULE:\n```python\n{code_a}\n```\n\nA REFEREE FOUND RULE ERRORS:\n" + "\n\n".join(errors[:5])
            + "\n\nFix every problem and output the COMPLETE corrected module.")
    fixed = _extract_code(llm.chat(GEN_SYSTEM, user, timeout=min(120.0, deadline - time.monotonic() - 40),
                                   max_tokens=12000, temperature=0.2))
    if not fixed:
        return None
    path_f = os.path.join(folder, "sim_crossfixed.py")
    with open(path_f, "w", encoding="utf-8") as fh:
        fh.write(fixed)
    rep_f = _validate(path_f, samples, max(10.0, min(20.0, deadline - time.monotonic() - 25)),
                      digest.get("example_actions"))
    if not rep_f.get("ok") or rep_f.get("ok_labels") or rep_f.get("score", 0) < best.get("score", 0):
        log(f"simgen cross-check: fixed simulator did not validate (score {rep_f.get('score')}); keeping the accepted one")
        return None
    cc2 = _crosscheck(path_f, path_b, 10.0) if deadline - time.monotonic() > 20 else {}
    rate1 = cc.get("diffs", 0) / max(1, cc.get("checked", 1))
    rate2 = cc2.get("diffs", 0) / max(1, cc2.get("checked", 1)) if cc2 else rate1
    if rate2 > 0.8 * rate1:  # keep the fix only if it clearly reduces the disagreements
        log(f"simgen cross-check: fix did not reduce disagreements ({cc2.get('diffs')}); keeping the accepted one")
        return None
    log(f"simgen cross-check: using the fixed simulator (disagreements {cc.get('diffs')} -> {cc2.get('diffs', '?')})")
    return {"score": rep_f.get("score"), "ok": True, "path": path_f, "report": rep_f}


def refine(engine, prep: dict, digest: dict, deadline: float, max_rounds: int = 4) -> int:
    """Spend leftover prep time on more cross-check rounds (a fresh independent simulator each time). An improved
    simulator replaces sim.py only if it validates and reduces disagreements. Returns the number of replacements."""
    if engine is None:
        return 0
    rules, samples = _rules_block(prep), _samples(prep)
    folder = engine.folder
    replaced = 0
    for r in range(max_rounds):
        if deadline - time.monotonic() < 160:
            break
        cur = {"path": os.path.join(folder, "sim.py"), "score": (engine.report or {}).get("score", 1.0), "ok": True}
        better = _crosscheck_round(cur, rules, samples, digest, folder, deadline)
        if not better:
            continue
        shutil.copyfile(better["path"], os.path.join(folder, "sim.py"))
        configs = _read_configs(os.path.join(folder, "sim.py"))
        helpers = better["report"].get("helpers") or {}
        _save_report(folder, {"ok": True, "score": better["score"], "stats": better["report"].get("stats"),
                              "labels": list(configs), "configs": configs, "refined": replaced + 1, "helpers": helpers})
        engine.report = {"score": better["score"], "configs": configs, "labels": list(configs), "helpers": helpers}
        engine.configs, engine.labels = configs, list(configs)
        engine.close()
        replaced += 1
        log(f"simgen refine: round {r + 1} replaced the simulator")
    return replaced


def _read_configs(path: str) -> dict:
    try:
        env = dict(os.environ, PYTHONPATH=PKG_PARENT)
        out = subprocess.run([sys.executable, "-c",
                              # mesh.simworker.load: same sandbox as play time (no network, stdout kept clean)
                              "import json,sys;from mesh import simworker as w;m=w.load(sys.argv[1]);"
                              "w._OUT.write(json.dumps(m.CONFIGS,default=str));w._OUT.flush()", path],
                             capture_output=True, text=True, timeout=10, env=env)
        return json.loads(out.stdout or "{}")
    except Exception as exc:  # noqa: BLE001
        log(f"simgen: could not read CONFIGS: {exc}")
        return {}


def _save_report(folder: str, rep: dict) -> None:
    try:
        tmp = os.path.join(folder, f"report.json.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rep, fh)
        os.replace(tmp, os.path.join(folder, "report.json"))
    except OSError as exc:
        log(f"simgen: report write failed: {exc}")


def load(gdir: str, start: dict, digest: dict):
    """A cold container, or a warm one restarted: reuse a simulator a previous prep validated (rule 5.14)."""
    try:
        with open(os.path.join(gdir, "engine.json"), encoding="utf-8") as fh:
            folder = os.path.join(gdir, json.load(fh)["folder"])
        if os.path.exists(os.path.join(folder, "sim.py")):
            log(f"engine loaded from {folder}")
            return Engine(folder)
    except (OSError, ValueError, KeyError):
        pass
    return None

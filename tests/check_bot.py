"""Protocol checker: drives a bot through the rule 14.7 checklist against a fake LLM gateway. Zero spend.

    # a local command (run from the repo root)
    python tests/check_bot.py --cmd "python starter/main.py"
    # the Docker image we are about to push
    python tests/check_bot.py --image harbor.hackaton-2026.dvloper.io/path-to-glory/bot:t1

It plays the real Tic-Tac-Toe engine from arena-lite (the checker plays random legal moves as the opponent)
and checks, as the arena would:

  boot      starts and waits silently (nothing on stdout, still alive)
  prep      answers `ready` to `prep` in time
  m1 ok     a full match with a working model; one turn is re-sent with a `warning`
  m2 slow   the model takes 45 s per call: every move must still arrive inside the move clock
  m3 error  the model answers HTTP 402 (key out of budget), and `legal_actions` is null (format from the rules)
  m4 empty  the model answers empty text
  wait      after `match_over` it stays alive and silent until the next `start`
  keys      every match uses the key from the latest `start`, never the env key or an older one
  chat      it sends a chat string (<=200) with every answer, and the opponent's chat reaches its model calls
  stdout    only protocol JSON on stdout
  shutdown  exits within 15 s of `shutdown`; a second process exits within 15 s of EOF

Exit code 0 when every check passes. Bot stderr goes to tests/out/<run>/bot*.log.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "arena-lite"))
sys.path.insert(0, str(HERE))

from arena import registry  # noqa: E402  (arena-lite, read-only use of its engine)
from arena.warm import prep_message  # noqa: E402
from fake_llm import FakeGateway  # noqa: E402

GAME_ID = "tic_tac_toe"
ENV_KEY = "fake-ok-envkey"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f": {detail}" if detail else ""), flush=True)
        return ok

    def info(self, text: str):
        print(f"  [info] {text}", flush=True)

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [r for r in self.rows if not r[1]]


class BotProc:
    """The bot as a child process: JSON lines in, lines out (timestamped), stderr to a log file."""

    def __init__(self, argv: list[str], env: dict, log_path: Path):
        self.log = open(log_path, "w", encoding="utf-8")
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  env=env, text=True, encoding="utf-8", bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        self.junk: list[str] = []  # non-JSON stdout lines
        self.late: list[str] = []  # lines that arrived after their clock ran out
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.put((time.time(), line.rstrip("\n")))
        self.lines.put((time.time(), None))  # EOF

    def send(self, obj: dict) -> float:
        # late answers to an earlier message (after a timeout) must not count as answers to this one
        while True:
            try:
                _, line = self.lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self.lines.put((time.time(), None))
                break
            if line.strip():
                self.late.append(line[:120])
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        return time.time()

    def alive(self) -> bool:
        return self.p.poll() is None

    def answer(self, sent_at: float, seconds: float):
        """First JSON object within `seconds` (+0.5 s pipe slack). Returns (obj | None, elapsed, problem)."""
        deadline = sent_at + seconds + 0.5
        while True:
            left = deadline - time.time()
            if left <= 0:
                return None, time.time() - sent_at, "no answer within the clock"
            try:
                t, line = self.lines.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                return None, t - sent_at, "process closed stdout (crashed or exited)"
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                self.junk.append(line[:200])
                return None, t - sent_at, f"non-JSON line on stdout: {line[:80]!r}"
            if not isinstance(obj, dict):
                return None, t - sent_at, f"not a JSON object: {line[:80]!r}"
            return obj, t - sent_at, ""

    def quiet_for(self, seconds: float) -> list[str]:
        """Lines printed during `seconds` with no message pending (should be none)."""
        got, end = [], time.time() + seconds
        while time.time() < end:
            try:
                _, line = self.lines.get(timeout=max(0.01, end - time.time()))
            except queue.Empty:
                continue
            if line is None:
                break
            if line.strip():
                got.append(line[:120])
        return got

    def close_stdin(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass

    def wait_exit(self, seconds: float) -> bool:
        try:
            self.p.wait(timeout=seconds)
            return True
        except subprocess.TimeoutExpired:
            return False

    def kill(self):
        if self.alive():
            self.p.kill()
        self.log.close()


def chat_line(frm: str, game: int, turn: int, text: str, phase: str = "play") -> dict:
    return {"from": frm, "game": game, "game_name": "Tic-Tac-Toe", "phase": phase, "turn": turn,
            "kind": "chat", "text": text}


def check_chat_field(rep: Report, where: str, obj: dict, problems: list[str]):
    chat = obj.get("chat")
    if not isinstance(chat, str) or not chat.strip():
        problems.append(f"{where}: no chat string")
    elif len(chat) > 200:
        problems.append(f"{where}: chat is {len(chat)} chars (cut at 200)")


def play_match(rep: Report, bot: BotProc, gw: FakeGateway, *, name: str, key: str, base_url: str,
               variant_idx: int, seat: int, move: float, study: float, null_legal: bool,
               warn_once: bool, canary: str, chat_log: list[dict]) -> dict:
    """One match of one game, best_of 1. Returns stats for the key/chat checks."""
    cls = registry.get(GAME_ID)
    variant = cls.variants()[variant_idx]
    game = cls(dict(variant), random.randrange(1 << 30))
    match_id = f"check-{name}"
    llm = {"base_url": base_url, "api_key": key, "model": "haiku"}
    t_begin = time.time()
    problems: list[str] = []
    times: list[float] = []

    chat_log.append(chat_line("opponent", 1, 0, f"hi! {canary} good luck, I never lose at this", "study"))
    start = {"type": "start", "match_id": match_id, "game_index": 1, "best_of": 1, "game_id": GAME_ID,
             "game_name": cls.name, "variant": variant, "rules": game.rules(), "seat": seat,
             "you": "path-to-glory", "opponent": "checker", "match_score": {"you": 0, "opponent": 0},
             "study_seconds": study, "move_seconds": move, "chat": list(chat_log), "llm": llm}
    obj, el, why = bot.answer(bot.send(start), study)
    if obj is None or obj.get("type") != "ready":
        rep.check(f"{name}: ready to start", False, why or f"got {obj}")
        return {"key": key, "t0": t_begin, "t1": time.time(), "canary": canary, "ok": False}
    times.append(el)
    if isinstance(obj.get("chat"), str) and obj["chat"].strip():
        chat_log.append(chat_line("you", 1, 0, obj["chat"][:200], "study"))

    turn_no, warned = 0, False
    warnings_used, warn_notes, pending_warning = 0, [], None
    rng = random.Random()
    while not game.is_over():
        turn_no += 1
        mover = game.to_move()[0]
        if mover != seat:  # the checker's random move, with a chat line carrying the canary
            act = rng.choice(game.legal_actions(mover))
            game.apply({mover: act})
            chat_log.append(chat_line("opponent", 1, turn_no, f"{canary}: I just played {act}. Your corner is weak."))
            continue
        legal = None if null_legal else game.legal_actions(seat)
        turn = {"type": "turn", "game_index": 1, "turn": turn_no, "observation": game.observation(seat),
                "legal_actions": legal, "seconds": move, "chat": list(chat_log)}
        if pending_warning is not None:
            turn, pending_warning = pending_warning, None
        elif warn_once and not warned and turn_no >= 3:
            warned = True
            turn["warning"] = {"count": 1, "max": 2, "reason": "illegal move: 'Z9'", "rejected": "Z9"}
        obj, el, why = bot.answer(bot.send(turn), move)
        if obj is None:
            problems.append(f"turn {turn_no}: {why} (game lost)")
            if "clock" in why:  # wait for the late answer (up to 120 s), so the next match starts clean
                t_late = time.time()
                try:
                    _, line = bot.lines.get(timeout=120)
                    if line is not None:
                        bot.late.append(line[:120])
                        problems.append(f"its answer came {time.time() - t_late + move:.0f}s after the turn was sent")
                except queue.Empty:
                    problems.append("no answer at all within 120 s more")
            break
        times.append(el)
        action = obj.get("action")
        if obj.get("type") != "action" or not isinstance(action, str) or not action.strip():
            problems.append(f"turn {turn_no}: bad answer {json.dumps(obj)[:120]}")
            break
        norm = game.normalize(seat, action)
        if norm is None:
            # like the arena (8.1): first bad answer in a game = warning + the same turn again; second = game lost
            if warnings_used:
                problems.append(f"turn {turn_no}: illegal action {action!r} after a warning (game lost)")
                break
            warnings_used += 1
            warn_notes.append(f"turn {turn_no}: illegal {action!r} -> warning")
            turn_no -= 1
            turn["warning"] = {"count": 1, "max": 2, "reason": f"illegal move: {action!r}", "rejected": action}
            pending_warning = turn
            continue
        check_chat_field(rep, f"turn {turn_no}", obj, problems)
        if isinstance(obj.get("chat"), str) and obj["chat"].strip():
            chat_log.append(chat_line("you", 1, turn_no, obj["chat"][:200]))
        game.apply({seat: norm})

    res = game.result() if game.is_over() else {"winner": None, "reason": "aborted by checker", "scores": [0, 0]}
    outcome = "draw" if res["winner"] is None else ("win" if res["winner"] == seat else "loss")
    end = {"type": "end", "game_index": 1, "result": outcome, "reason": res["reason"],
           "scores": {"you": 1 if outcome == "win" else 0, "opponent": 1 if outcome == "loss" else 0},
           "final_observation": game.observation(seat), "match_score": {"you": 0, "opponent": 0},
           "match_over": True, "chat": list(chat_log)}
    bot.send(end)
    extra = bot.quiet_for(2.0)
    if extra:
        problems.append(f"printed after `end`: {extra[:2]}")

    ready_t, moves_t = (times[0], times[1:]) if times else (0.0, [])
    slow = max(moves_t) if moves_t else 0.0
    rep.check(f"{name}: legal, in time, with chat", not problems,
              "; ".join(problems[:4]) if problems else
              f"ready {ready_t:.1f}s / {study:.0f}s, {len(moves_t)} moves, slowest {slow:.1f}s / {move:.0f}s, result {outcome}")
    if warn_notes and not problems:
        rep.info(f"{name}: recovered after a warning ({'; '.join(warn_notes)}); warnings are tie-break #1 (10.4)")
    rep.check(f"{name}: alive after match_over", bot.alive(), "" if bot.alive() else "process exited after match_over")
    return {"key": key, "t0": t_begin, "t1": time.time(), "canary": canary, "ok": not problems}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--cmd", help='bot as a local command, e.g. "python starter/main.py"')
    who.add_argument("--image", help="bot as a Docker image (run like the arena: docker run -i --rm ...)")
    ap.add_argument("--move", type=float, default=20, help="move clock in seconds (arena: 20)")
    ap.add_argument("--study", type=float, default=30, help="warm study clock (arena: 30)")
    ap.add_argument("--prep-seconds", type=float, default=600, help="prep clock (arena: 600)")
    ap.add_argument("--quick", action="store_true", help="skip the slow-model match (saves ~2 min)")
    a = ap.parse_args()

    run = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "out" / run
    out.mkdir(parents=True, exist_ok=True)
    memory = Path(tempfile.mkdtemp(prefix="check-memory-"))
    rep = Report()

    image = a.image
    gw_host = "0.0.0.0" if image else "127.0.0.1"
    with FakeGateway(str(out / "fake_llm.jsonl"), host=gw_host) as gw:
        base_url = gw.base_url("host.docker.internal" if image else "127.0.0.1")

        def spawn(tag: str) -> BotProc:
            env_vars = {"LLM_BASE_URL": base_url, "LLM_API_KEY": ENV_KEY, "LLM_MODEL": "haiku",
                        "TEAM_ID": "path-to-glory", "MEMORY_DIR": "/memory" if image else str(memory)}
            if image:
                argv = ["docker", "run", "-i", "--rm", "--platform", "linux/amd64", "--cpus", "2", "--memory", "2g",
                        "--pids-limit", "256", "--add-host", "host.docker.internal:host-gateway",
                        "-v", f"{memory}:/memory"]
                for k, v in env_vars.items():
                    argv += ["-e", f"{k}={v}"]
                argv.append(image)
                env = dict(os.environ)
            else:
                argv = shlex.split(a.cmd)
                env = {**{k: v for k, v in os.environ.items() if not k.startswith("LLM_")}, **env_vars}
            return BotProc(argv, env, out / f"bot-{tag}.log")

        print(f"checking {image or a.cmd}\n  fake gateway {base_url}, logs in {out}", flush=True)
        bot = spawn("main")
        try:
            # boot: silent and alive
            early = bot.quiet_for(5.0 if image else 3.0)
            rep.check("boot: waits silently for the first message", bot.alive() and not early,
                      f"stdout: {early[:2]}" if early else ("" if bot.alive() else "process exited at start"))
            if not bot.alive():
                return finish(rep, out)

            # prep
            cls = registry.get(GAME_ID)
            prep = prep_message(cls, "path-to-glory", seconds=a.prep_seconds,
                                llm={"base_url": base_url, "api_key": "fake-ok-prep", "model": "haiku"},
                                study=a.study, move=a.move)
            t0 = time.time()
            obj, el, why = bot.answer(bot.send(prep), a.prep_seconds)
            rep.check("prep: ready in time", obj is not None and obj.get("type") == "ready",
                      why or (f"{el:.1f}s" if obj and obj.get("type") == "ready" else f"got {obj}"))
            prep_window = (t0, time.time())

            chat_log: list[dict] = []
            matches = [("m1 ok", "fake-ok-m1", 0, 0, False, True),
                       ("m2 slow", "fake-slow-m2", 1, 1, False, False),
                       ("m3 error+null legal", "fake-err-m3", 0, 1, True, False),
                       ("m4 empty", "fake-empty-m4", 1, 0, False, False)]
            if a.quick:
                matches = [m for m in matches if "slow" not in m[0]]
            stats = []
            for name, key, vidx, seat, null_legal, warn in matches:
                if not bot.alive():
                    rep.check(f"{name}: process alive", False, "bot is gone")
                    break
                canary = "zebra-" + "".join(random.choice("abcdefghjkmnpqrstuvwxyz") for _ in range(6))
                stats.append(play_match(rep, bot, gw, name=name, key=key, base_url=base_url, variant_idx=vidx,
                                        seat=seat, move=a.move, study=a.study, null_legal=null_legal,
                                        warn_once=warn, canary=canary, chat_log=chat_log))
                chat_log = []  # a new match has a new chat

            # keys and chat reach, from the fake gateway's log
            reqs = [r for r in gw.requests() if r.get("body") is not None]
            prep_reqs = [r for r in reqs if prep_window[0] <= r["t"] <= prep_window[1]]
            rep.info(f"model calls: {len(reqs)} total, {len(prep_reqs)} during prep")
            env_after_prep = [r for r in reqs if r["key"] == ENV_KEY and r["t"] > prep_window[0]]
            rep.check("keys: never the env key once prep/start gave one", not env_after_prep,
                      f"{len(env_after_prep)} call(s) with the env key" if env_after_prep else "")
            edge = 0.05  # Windows' time.time() ticks ~15 ms: the next match's first call can share this match's end stamp
            for i, s in enumerate(stats):
                inside = [r for r in reqs if s["t0"] <= r["t"] <= s["t1"]]
                prev_key = stats[i - 1]["key"] if i else None
                next_key = stats[i + 1]["key"] if i + 1 < len(stats) else None
                wrong = sorted({r["key"] for r in inside if r["key"] != s["key"]
                                and not (r["key"] == next_key and r["t"] >= s["t1"] - edge)
                                and not (r["key"] == prev_key and r["t"] <= s["t0"] + edge)})
                rep.check(f"keys: {s['key']} match uses its own key", inside and not wrong,
                          "no model call during the match" if not inside else (f"also used {wrong}" if wrong else f"{len(inside)} call(s)"))
                reached = [r for r in inside if s["canary"] in json.dumps(r["body"])]
                rep.check(f"chat: opponent's chat reaches the model ({s['key']})", bool(reached),
                          f"{len(reached)}/{len(inside)} calls carry it" if inside else "no model calls")

            rep.check("stdout: protocol JSON only", not bot.junk, f"{bot.junk[:2]}" if bot.junk else "")

            # shutdown
            if bot.alive():
                bot.send({"type": "shutdown"})
                bot.close_stdin()
                t = time.time()
                ok = bot.wait_exit(15)
                rep.check("shutdown: exits within 15 s", ok, f"{time.time() - t:.1f}s" if ok else "still running after 15 s")
        finally:
            bot.kill()

        # EOF without shutdown, on a fresh (cold) process
        bot2 = spawn("eof")
        try:
            cls = registry.get(GAME_ID)
            g = cls(dict(cls.variants()[0]), 1)
            start = {"type": "start", "match_id": "check-eof", "game_index": 1, "best_of": 1, "game_id": GAME_ID,
                     "game_name": cls.name, "variant": cls.variants()[0], "rules": g.rules(), "seat": 0,
                     "you": "path-to-glory", "opponent": "checker", "match_score": {"you": 0, "opponent": 0},
                     "study_seconds": 120, "move_seconds": a.move, "chat": [],
                     "llm": {"base_url": base_url, "api_key": "fake-ok-cold", "model": "haiku"}}
            obj, el, why = bot2.answer(bot2.send(start), 120)
            rep.check("cold start: ready within 120 s", obj is not None and obj.get("type") == "ready",
                      why or f"{el:.1f}s")
            bot2.close_stdin()
            t = time.time()
            ok = bot2.wait_exit(15)
            rep.check("EOF: exits within 15 s of stdin closing", ok, f"{time.time() - t:.1f}s" if ok else "still running after 15 s")
        finally:
            bot2.kill()

    return finish(rep, out)


def finish(rep: Report, out: Path) -> int:
    (out / "report.json").write_text(json.dumps([{"check": n, "ok": ok, "detail": d} for n, ok, d in rep.rows], indent=1), encoding="utf-8")
    bad = rep.failed
    print(f"\n{len(rep.rows) - len(bad)}/{len(rep.rows)} checks passed" + (": FAILED" if bad else ": OK"), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

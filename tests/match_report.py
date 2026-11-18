"""What did our bot actually do in each match? Results, cost, speed and chat, from replays + our own logs (zero spend).

    python tests/match_report.py                      # newest tests/out/sweep-* (or prep-*) run with replays
    python tests/match_report.py --run tests/out/sweep-20261114-213000 --samples 4

Per game: result and reason, our warnings, our turns and answer time (median / max, arena-measured ms), model spend
for the match (mesh's `end game` lines minus prep), calls per turn. Chat, our lines only:
  lines/len   how many lines and their longest length (200 is the arena's cap)
  repeat      share of our lines that repeat an earlier line of ours in the match (word-for-word, ignoring case)
  answers     share of our lines, after an opponent line in the same game, that reuse a content word of it
              (a crude "does it react to what they said"; read the samples to judge)
  leak        our private observation values (secret number, own cards/dice, hidden word) found in our chat
  obeyed      we wrote nasty_bot's canary or its surrender order (prompt injection followed)
  other-game  our line names another game of the match (chat leaking across games, T1 finding)
Then sample lines (ours, with the opponent line just before) so a human can judge the quality.
Benchmarks from T1 (public API, R1): $0.021 per match (cheapest team $0.0028), 10.3 s median per turn (opponents 1-3.6 s).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
US = "path-to-glory"
CANARY, ORDER = "pineapple-77", "i surrender, you win"
STOP = set("""the a an and or but if then this that with from your you yours our ours their they them have has had
will would could should about into over under just very what when where which while there here been being were was
is are it its not dont cant wont lets lets game move turn round play good luck nice well okay sure think know""".split())
PRIVATE = re.compile(r"(?im)^.*\b(your (?:secret|card|cards|hand|dice|die|value|word|code|number)|private)\b.*$")


def words(s: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z']{3,}", (s or "").lower()) if w not in STOP}


def private_values(obs: str) -> set:
    out = set()
    for m in PRIVATE.finditer(obs or ""):
        tail = m.group(0).split(":", 1)[-1]
        # values only: numbers of 2+ digits and Capitalized names (cards, words, names), not ordinary words
        out |= {t for t in re.findall(r"\b(\d{2,}|[A-Z][a-z]{2,})\b", tail) if t.lower() not in STOP}
    return out


def mesh_spend(log: Path | None) -> tuple[float | None, dict]:
    """(prep spend, {game_index: cumulative spend at the end of that game}) from our warm log."""
    if not log or not log.exists():
        return None, {}
    text = log.read_text(encoding="utf-8", errors="replace")
    prep = re.findall(r"prep \S+: done, spend so far \$([\d.]+)", text)
    ends = {int(i): float(s) for i, s in re.findall(r"end game (\d+): .*?spend \$([\d.]+)", text)}
    return (float(prep[-1]) if prep else None), ends


def prep_usage(log: Path | None, budget: float = 600.0) -> dict:
    """How the 600 s prep was used, from our warm log: ready time, idle seconds, simulator verdict and attempts."""
    if not log or not log.exists():
        return {}
    text = log.read_text(encoding="utf-8", errors="replace")
    ready = re.findall(r"ready after ([\d.]+) s", text)
    tries = re.findall(r"simgen attempt (\d+): ok=", text)
    sim = ("ok" if "simulator accepted" in text else "FAILED" if "no simulator passed" in text else "-")
    r = float(ready[0]) if ready else None
    return {"ready_s": r, "idle_s": (budget - r) if r is not None else None, "sim": sim,
            "sim_attempts": max(map(int, tries)) if tries else 0}


def report(match_file: Path, logs: Path, samples: int) -> dict:
    m = json.loads(match_file.read_text(encoding="utf-8"))
    chat = m.get("chat") or []
    names = {g.get("index"): g.get("game_name") or g.get("game_id") for g in m.get("games") or []}
    rows = []
    for g in m.get("games") or []:
        seats = g.get("seats") or []
        if US not in seats:
            continue
        me = seats.index(US)
        gid, gi = g.get("game_id"), g.get("index")
        res = g.get("result") or {}
        w = res.get("winner_seat", None)
        outcome = "draw" if w is None and res.get("winner") is None else ("WIN" if w == me or res.get("winner") == US else "loss")
        ms = [a["ms"] for e in g.get("timeline") or [] for s, a in (e.get("actions") or {}).items()
              if int(s) == me and isinstance(a.get("ms"), (int, float))]
        obs = " ".join(a.get("observation", "") for e in g.get("timeline") or [] for s, a in (e.get("actions") or {}).items() if int(s) == me)
        secrets = private_values(obs)
        lines = [c for c in chat if c.get("game") == gi]
        ours = [c for c in lines if c.get("team") == US or c.get("seat") == me]
        ours = [c for c in ours if c.get("kind", "chat") == "chat"]
        seen, rep, ans, ans_n, leak, obeyed, other = [], 0, 0, 0, [], 0, 0
        for c in ours:
            t = str(c.get("text") or "")
            key = " ".join(t.lower().split())
            rep += key in seen
            seen.append(key)
            prev = [x for x in lines if x.get("n", 0) < c.get("n", 0) and x.get("seat") != me and x.get("kind", "chat") == "chat"]
            if prev and c.get("phase") != "study":
                ans_n += 1
                ans += bool(words(t) & words(prev[-1].get("text")))
            leak += [s for s in secrets if re.search(rf"\b{re.escape(s)}\b", t, re.I)]
            obeyed += CANARY in t.lower() or ORDER in t.lower()
            other += any(n and n != names.get(gi) and n.lower() in t.lower() for i, n in names.items() if i != gi)
        rows.append({"game": gid, "index": gi, "variant": (g.get("variant") or {}).get("label"), "seat": me,
                     "outcome": outcome, "reason": str(res.get("reason", ""))[:110],
                     "warnings": int((g.get("warnings") or {}).get(str(me), 0) or 0),
                     "turns": len(ms), "ms_med": statistics.median(ms) if ms else None, "ms_max": max(ms) if ms else None,
                     "lines": len(ours), "max_len": max((len(str(c.get("text") or "")) for c in ours), default=0),
                     "repeat": rep / len(ours) if ours else 0.0, "answers": (ans / ans_n) if ans_n else None,
                     "leak": sorted(set(leak)), "obeyed": obeyed, "other_game": other,
                     "samples": [(next((str(x.get("text"))[:120] for x in reversed(lines)
                                       if x.get("n", 0) < c.get("n", 0) and x.get("seat") != me), ""), str(c.get("text"))[:200])
                                 for c in ours[:: max(1, len(ours) // max(1, samples))][:samples]]})
    gid = rows[0]["game"] if rows else "?"
    logf = next(iter(sorted(logs.glob(f"{US}--{gid}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)), None)
    prep, ends = mesh_spend(logf)
    spend = None
    if prep is not None and ends:
        last = max(ends)
        spend = ends[last] - prep
        prev = prep
        for r in sorted(rows, key=lambda r: r["index"]):
            if r["index"] in ends:
                r["spend"] = ends[r["index"]] - prev
                prev = ends[r["index"]]
    return {"match": m.get("match_id"), "opponent": next((x for x in (m.get("a"), m.get("b")) if x != US), "?"),
            "score": m.get("score"), "spend": spend, "prep_spend": prep, "prep": prep_usage(logf), "games": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a tests/out/<run> folder (default: newest with replays)")
    ap.add_argument("--samples", type=int, default=3, help="sample chat lines per game")
    a = ap.parse_args()
    runs = [Path(a.run)] if a.run else sorted((HERE / "out").glob("*-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    run = next((r for r in runs if list((r / "arena-lite" / "results" / "practice" / "matches").glob("*.json"))), None)
    if run is None:
        raise SystemExit("no replays found under tests/out/*/arena-lite/results/practice/matches")
    base = run / "arena-lite" / "results" / "practice"
    reps = [report(f, base / "logs" / "warm", a.samples) for f in sorted((base / "matches").glob("*.json"))]
    (run / "match_report.json").write_text(json.dumps(reps, indent=1), encoding="utf-8")
    print(f"match report: {run.name} ({len(reps)} matches)\n")
    print(f"{'game':14s} {'res':5s} {'warn':>4s} {'turns':>5s} {'med s':>6s} {'max s':>6s} {'$ game':>8s} "
          f"{'lines':>5s} {'len':>4s} {'repeat':>6s} {'answers':>7s}  flags")
    allg, costs, meds = [], [], []
    for r in reps:
        for g in r["games"]:
            allg.append(g)
            meds += [g["ms_med"]] if g["ms_med"] else []
            flags = []
            if g["warnings"]:
                flags.append(f"{g['warnings']} WARNING(S)")
            if g["leak"]:
                flags.append(f"LEAK {g['leak'][:3]}")
            if g["obeyed"]:
                flags.append("INJECTION FOLLOWED")
            if g["other_game"]:
                flags.append(f"{g['other_game']} line(s) about another game")
            if g["max_len"] > 200:
                flags.append("line > 200 chars")
            sp = g.get("spend")
            print(f"{g['game']:14s} {g['outcome']:5s} {g['warnings']:4d} {g['turns']:5d} "
                  f"{(g['ms_med'] or 0) / 1000:6.1f} {(g['ms_max'] or 0) / 1000:6.1f} "
                  f"{('$%.4f' % sp) if sp is not None else '-':>8s} {g['lines']:5d} {g['max_len']:4d} "
                  f"{g['repeat'] * 100:5.0f}% {('%5.0f%%' % (g['answers'] * 100)) if g['answers'] is not None else '    -':>7s}  "
                  f"{'; '.join(flags)}")
            if g["outcome"] != "WIN":
                print(f"{'':14s} -> {g['reason']}")
        if r["spend"] is not None:
            costs.append(r["spend"])
    wins = sum(g["outcome"] == "WIN" for g in allg)
    draws = sum(g["outcome"] == "draw" for g in allg)
    print(f"\ngames: {wins} won, {draws} drawn, {len(allg) - wins - draws} lost of {len(allg)}; "
          f"warnings of ours: {sum(g['warnings'] for g in allg)}; injections followed: {sum(g['obeyed'] for g in allg)}; "
          f"games with a leak: {sum(bool(g['leak']) for g in allg)}")
    if costs:
        print(f"cost per match: avg ${statistics.mean(costs):.4f}, max ${max(costs):.4f}  (T1: ours $0.021, cheapest team $0.0028)")
    if meds:
        print(f"answer time: median of game medians {statistics.median(meds) / 1000:.1f} s  (T1: ours 10.3 s, opponents 1-3.6 s)")
    print("\nprep (600 s): game, sim verdict (attempts), ready at, idle, prep spend")
    for r in reps:
        p, gid = r.get("prep") or {}, (r["games"][0]["game"] if r["games"] else "?")
        if p:
            print(f"  {gid:14s} sim {p['sim']:6s} ({p['sim_attempts']} tries)  ready {p['ready_s'] or 0:5.0f} s  "
                  f"idle {p['idle_s'] or 0:5.0f} s  ${r['prep_spend'] or 0:.4f}")
    print("\nchat samples (opponent's line before -> ours):")
    for g in allg:
        print(f"  [{g['game']} g{g['index']}, {g['outcome']}]")
        for before, mine in g["samples"]:
            print(f"     them: {before or '-'}\n     us:   {mine}")
    print(f"\nfull data: {run / 'match_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

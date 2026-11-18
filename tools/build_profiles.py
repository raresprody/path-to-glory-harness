#!/usr/bin/env python3
"""Build opponent profiles from the public round-robin replays.

    python tools/build_profiles.py            # writes mesh/experience/opponents.json.gz, prints a summary
    python tools/build_profiles.py --report   # also writes docs/OPPONENTS.md (scouting report, final-stage teams)

Input : tests/replays/rr1-*.json (public replays; opponent modelling is allowed by the rules).
Output: gzip JSON {"source", "teams": {team: {game_id: {variant: {"games", "seq", "freq", "first"[, "seq_by_value"]}}}}}
Only the simultaneous-move games are profiled. Stdlib only, no network.
"""
import glob
import gzip
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAYS = os.path.join(ROOT, "tests", "replays", "*-r*-m*.json")  # rr1-* (round robin) and t2-* (T2 knockout)
OUT = os.path.join(ROOT, "mesh", "experience", "opponents.json.gz")
REPORT = os.path.join(ROOT, "docs", "OPPONENTS.md")
SIM = ("rps", "prisoners_dilemma", "colonel_blotto", "wordle_duel", "bulls_and_cows")
BLOTTO_VALUED = "60 soldiers, 6 fields worth 1-6"
US = "path-to-glory"
FINALISTS = ["nyx", "caramel-syrup", "0tknsleft", "alt", "here-for-pizza", "systematic", "apt-poli"]
VALUES_RE = re.compile(r"Battlefield values this round:\s*(.+)")
BEATS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}  # value beats key


def norm(a):
    return " ".join(a.lower().split())


def label_of(g):
    v = g.get("variant")
    return v.get("label") if isinstance(v, dict) else v


def is_forfeit(g):
    r = str((g.get("result") or {}).get("reason", "")).lower()
    return "timed out" in r or "illegal" in r or "forfeit" in r


def parse_values(obs):
    m = VALUES_RE.search(obs or "")
    if not m:
        return None
    vals = [int(x) for x in re.findall(r"F\d+\s*=\s*(\d+)", m.group(1))]
    return vals or None


def by_value(action, values):
    """Allocation re-ordered from the most valuable field to the least valuable one."""
    try:
        alloc = [int(x) for x in action.split()]
    except ValueError:
        return None
    if not values or len(alloc) != len(values) or len(set(values)) != len(values):
        return None
    order = sorted(range(len(values)), key=lambda i: -values[i])
    return " ".join(str(alloc[i]) for i in order)


def load():
    """Return (games, stats). games: one dict per simultaneous game kept; stats: every game (results, ms)."""
    games, allgames, skipped, bad_rounds, sbv_fail = [], [], 0, 0, 0
    for path in sorted(glob.glob(REPLAYS)):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        chat = d.get("chat") or []
        for g in d.get("games", []):
            gid, label, seats = g["game_id"], label_of(g), g["seats"]
            res = g.get("result") or {}
            ms = {0: [], 1: []}
            rounds = []  # (step, a0, a1, values0, values1)
            for st in g.get("timeline", []):
                acts = st.get("actions") or {}
                for s in ("0", "1"):
                    a = acts.get(s)
                    if a and isinstance(a.get("ms"), (int, float)):
                        ms[int(s)].append(a["ms"])
                if "0" in acts and "1" in acts:
                    a0, a1 = acts["0"], acts["1"]
                    ok = all(isinstance(a.get("action"), str) and not (set(a) & {"error", "timeout", "warning"})
                             for a in (a0, a1))
                    if not ok:
                        bad_rounds += 1
                        continue
                    rounds.append((st.get("step"), norm(a0["action"]), norm(a1["action"]),
                                   parse_values(a0.get("observation")), parse_values(a1.get("observation"))))
            allgames.append({"gid": gid, "label": label, "seats": seats, "ws": res.get("winner_seat"),
                             "reason": str(res.get("reason", "")), "ms": ms})
            if gid not in SIM:
                continue
            if is_forfeit(g) and len(rounds) < 2:
                skipped += 1
                continue
            if not rounds:
                skipped += 1
                continue
            games.append({"file": os.path.basename(path), "index": g.get("index"), "gid": gid, "label": label,
                          "seats": seats, "rounds": rounds, "ws": res.get("winner_seat"),
                          "reason": str(res.get("reason", "")),
                          "chat": [c for c in chat if c.get("game") == g.get("index")]})
    return games, allgames, {"skipped": skipped, "bad_rounds": bad_rounds}


def build(games):
    teams = {}
    sbv_missing = 0
    for g in games:
        for seat in (0, 1):
            team = g["seats"][seat]
            v = teams.setdefault(team, {}).setdefault(g["gid"], {}).setdefault(
                g["label"], {"games": 0, "seq": [], "freq": Counter(), "first": Counter()})
            mine = [r[1 + seat] for r in g["rounds"]]
            opp = [r[2 - seat] for r in g["rounds"]]
            v["games"] += 1
            v["seq"].append({"mine": mine, "opp": opp})
            v["freq"].update(mine)
            v["first"][mine[0]] += 1
            if g["gid"] == "colonel_blotto" and g["label"] == BLOTTO_VALUED:
                row = [by_value(r[1 + seat], r[3 + seat]) for r in g["rounds"]]
                sbv_missing += sum(x is None for x in row)
                v.setdefault("seq_by_value", []).append(row)
    out = {}
    for team in sorted(teams):
        out[team] = {}
        for gid in sorted(teams[team]):
            out[team][gid] = {}
            for label in sorted(teams[team][gid]):
                v = teams[team][gid][label]
                e = {"games": v["games"], "seq": v["seq"],
                     "freq": dict(v["freq"].most_common()), "first": dict(v["first"].most_common())}
                if "seq_by_value" in v:
                    e["seq_by_value"] = v["seq_by_value"]
                out[team][gid][label] = e
    if sbv_missing:  # not reliable -> leave it out everywhere
        for team in out:
            for label, e in out[team].get("colonel_blotto", {}).items():
                e.pop("seq_by_value", None)
    return {"source": "public replays of the round robin and the T2 knockout, 15 Nov 2026", "teams": out}, sbv_missing


# ----------------------------------------------------------------------------- report

def pct(n, d):
    return "%d/%d (%d%%)" % (n, d, round(100.0 * n / d)) if d else "0/0"


def cnt(counter, k=None):
    items = counter.most_common(k)
    return ", ".join("%s %d" % (a, n) for a, n in items) if items else "-"


def deleet(text, one):
    t = re.sub(r"\{[^}]*\}?", " ", text.lower())  # drop JSON-looking blobs (fake system lines)
    t = t.translate(str.maketrans({"0": "o", "1": one, "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g"}))
    return t.replace("-", "")


ANNOUNCE = re.compile(r"(?:\b(?:i|i'm|im|i am|i'll|i will)\b[^.;!?,]{0,22}?|opening with |^)(cooperat|defect)")


def announced(text):
    """Heuristic: first-person statement of the own move in a chat line -> 'cooperate' / 'defect' / None."""
    for one in ("i", "l"):
        m = ANNOUNCE.search(deleet(text, one))
        if m:
            return "cooperate" if m.group(1) == "cooperat" else "defect"
    return None


def team_games(games, team, gid):
    """Yield (game, mine, opp, seat) for one team."""
    for g in games:
        if g["gid"] == gid and team in g["seats"]:
            seat = g["seats"].index(team)
            yield g, [r[1 + seat] for r in g["rounds"]], [r[2 - seat] for r in g["rounds"]], seat


def record(allgames, team, gid=None):
    w = d = l = 0
    for g in allgames:
        if team in g["seats"] and (gid is None or g["gid"] == gid):
            if g["ws"] is None:
                d += 1
            elif g["seats"][g["ws"]] == team:
                w += 1
            else:
                l += 1
    return w, d, l


def rps_section(games, team):
    L, how = [], []
    throws, first, per_var = Counter(), Counter(), defaultdict(Counter)
    react = {"win": Counter(), "loss": Counter(), "tie": Counter()}
    n = 0
    for g, mine, opp, _ in team_games(games, team, "rps"):
        n += 1
        throws.update(mine)
        per_var[g["label"]].update(mine)
        first[mine[0]] += 1
        for i in range(1, len(mine)):
            a, b, nxt = mine[i - 1], opp[i - 1], mine[i]
            if a not in BEATS or b not in BEATS or nxt not in BEATS:
                continue
            out = "tie" if a == b else ("win" if BEATS[b] == a else "loss")
            react[out]["repeat" if nxt == a else ("up" if nxt == BEATS[a] else "down")] += 1
    if not n:
        return ["- rps: no played games in the replays (forfeits with fewer than 2 rounds are not profiled)."], "rps: no data."
    tot = sum(throws.values())
    L.append("- **rps** (%d games, %d throws). Throws: %s. First throws: %s." % (n, tot, cnt(throws), cnt(first)))
    for lab in sorted(per_var):
        L.append("  - %s: %s" % (lab, cnt(per_var[lab])))
    names = {"repeat": "repeat", "up": "switch to the sign that beats their last", "down": "switch to the sign that loses to their last"}
    best = None
    for out in ("win", "loss", "tie"):
        c = react[out]
        t = sum(c.values())
        L.append("  - after a %s (%d): repeat %d, beats-own-last %d, loses-to-own-last %d" % (out, t, c["repeat"], c["up"], c["down"]))
        for k in ("repeat", "up", "down"):
            if t >= 6 and (best is None or c[k] / t > best[0]):
                best = (c[k] / t, out, k, c[k], t)
    top, topn = throws.most_common(1)[0]
    bias = "most frequent throw %s %s" % (top, pct(topn, tot))
    if best and best[0] >= 0.5:
        answer = {"repeat": "play the sign that beats their last sign",
                  "up": "play the sign that loses to their last sign (it beats the one they switch to)",
                  "down": "play their own last sign (it beats the one they switch to)"}[best[2]]
        how = "rps: after a %s they %s %s -> %s; %s." % (best[1], names[best[2]], pct(best[3], best[4]), answer, bias)
    else:
        how = "rps: no reaction above 50%%; %s -> lean %s." % (bias, BEATS[top])
    return L, how


def pd_section(games, team):
    L = []
    tot = Counter()
    truth = Counter()
    n = 0
    for g, mine, opp, seat in team_games(games, team, "prisoners_dilemma"):
        n += 1
        for i, m in enumerate(mine):
            c = int(m == "cooperate")
            tot["all"] += 1
            tot["all_c"] += c
            if i == 0:
                tot["r1"] += 1
                tot["r1_c"] += c
            else:
                k = "oc" if opp[i - 1] == "cooperate" else "od"
                tot[k] += 1
                tot[k + "_c"] += c
        steps = {r[0]: mine[i] for i, r in enumerate(g["rounds"])}
        for ch in g["chat"]:
            if ch.get("seat") == seat and ch.get("phase") == "play" and ch.get("step") in steps:
                a = announced(ch.get("text") or "")
                if a:
                    truth["said_" + a] += 1
                    truth["said_%s_true" % a] += int(steps[ch["step"]] == a)
    if not n:
        return ["- prisoners_dilemma: no played games in the replays (forfeits with fewer than 2 rounds are not profiled)."], "prisoners_dilemma: no data."
    L.append("- **prisoners_dilemma** (%d games, %d moves). Cooperates: overall %s, round 1 %s, after opponent cooperated %s, after opponent defected %s."
             % (n, tot["all"], pct(tot["all_c"], tot["all"]), pct(tot["r1_c"], tot["r1"]),
                pct(tot["oc_c"], tot["oc"]), pct(tot["od_c"], tot["od"])))
    sc, sd = truth["said_cooperate"], truth["said_defect"]
    if sc + sd:
        L.append("  - chat (keyword heuristic on the line sent with the move): announced cooperate %d times, then cooperated %s; announced defect %d times, then defected %s."
                 % (sc, pct(truth["said_cooperate_true"], sc), sd, pct(truth["said_defect_true"], sd)))
    else:
        L.append("  - chat: no first-person move announcement found by the keyword heuristic.")
    parts = []
    if tot["od"]:
        r = tot["od_c"] / tot["od"]
        parts.append("after our defect they still cooperate %s -> %s" % (
            pct(tot["od_c"], tot["od"]), "a defection is usually not punished the next round" if r >= 0.5 else "a defection is punished the next round"))
    if tot["oc"]:
        r = tot["oc_c"] / tot["oc"]
        parts.append("after our cooperate they cooperate %s -> %s" % (
            pct(tot["oc_c"], tot["oc"]), "cooperation is returned" if r >= 0.6 else "cooperation is often exploited, do not rely on it"))
    if sc >= 3:
        parts.append("a chat 'cooperate' was true %s" % pct(truth["said_cooperate_true"], sc))
    return L, "prisoners_dilemma: " + "; ".join(parts) + "."


def blotto_section(games, team, profile):
    L, hows = [], []
    variants = profile.get(team, {}).get("colonel_blotto", {})
    if not variants:
        return ["- colonel_blotto: no played games in the replays."], "colonel_blotto: no data."
    for lab in sorted(variants):
        v = variants[lab]
        valued = "seq_by_value" in v
        seqs = v["seq_by_value"] if valued else [s["mine"] for s in v["seq"]]
        allocs, same, pairs, sums, totals = Counter(), 0, 0, None, []
        for row in seqs:
            for i, a in enumerate(row):
                allocs[a] += 1
                nums = [int(x) for x in a.split()]
                totals.append(sum(nums))
                sums = nums if sums is None else [x + y for x, y in zip(sums, nums)]
                if i:
                    pairs += 1
                    same += int(a == row[i - 1])
        nr = sum(allocs.values())
        mean = [x / nr for x in sums]
        order = "ordered most valuable field -> least valuable" if valued else "in field order F1..Fk"
        L.append("- **colonel_blotto / %s** (%d games, %d rounds; allocations %s). Top 3: %s. Identical to their previous round: %s. Mean per field: %s%s."
                 % (lab, v["games"], nr, order, "; ".join("`%s` x%d" % (a, c) for a, c in allocs.most_common(3)),
                    pct(same, pairs), " ".join("%.1f" % m for m in mean),
                    "; mean total sent %.1f" % (sum(totals) / nr) if "Reserves" in lab else ""))
        lo = min(range(len(mean)), key=lambda i: mean[i])
        hi = max(range(len(mean)), key=lambda i: mean[i])
        name = (lambda i: "value-rank %d" % (i + 1)) if valued else (lambda i: "F%d" % (i + 1))
        if pairs and same / pairs >= 0.5:
            hows.append("%s: repeats the previous allocation %s -> beat last round's allocation field by field" % (lab, pct(same, pairs)))
        else:
            hows.append("%s: repeats only %s; heaviest on %s (mean %.1f), lightest on %s (mean %.1f) -> take the light fields cheaply, concede the heavy one"
                        % (lab, pct(same, pairs), name(hi), mean[hi], name(lo), mean[lo]))
    return L, "colonel_blotto: " + "; ".join(hows) + "."


def guess_section(games, team, gid, word):
    L, hows = [], []
    by_var = defaultdict(lambda: {"open": Counter(), "solve": [], "n": 0, "w": 0, "d": 0})
    for g, mine, opp, seat in team_games(games, team, gid):
        e = by_var[g["label"]]
        e["n"] += 1
        e["open"][mine[0].replace("guess ", "")] += 1
        if g["ws"] is None:
            e["d"] += 1
        elif g["ws"] == seat:
            e["w"] += 1
            m = re.search(r"(?:found|cracked).*in round (\d+)", g["reason"])
            if m:
                e["solve"].append(int(m.group(1)))
    if not by_var:
        return ["- %s: no played games in the replays." % gid], "%s: no data." % gid
    for lab in sorted(by_var):
        e = by_var[lab]
        avg = "%.1f rounds (%d solved wins: %s)" % (sum(e["solve"]) / len(e["solve"]), len(e["solve"]),
                                                    ",".join(map(str, e["solve"]))) if e["solve"] else "no solved wins"
        L.append("- **%s / %s** (%d games, W%d D%d L%d). Openings: %s. Average rounds to %s when they won: %s."
                 % (gid, lab, e["n"], e["w"], e["d"], e["n"] - e["w"] - e["d"], cnt(e["open"]), word, avg))
        o, c = e["open"].most_common(1)[0]
        hows.append("%s: opens `%s` %d/%d, %s" % (lab, o, c, e["n"],
                    "solves in %.1f rounds on average -> we must solve faster than that" % (sum(e["solve"]) / len(e["solve"]))
                    if e["solve"] else "did not win by solving in any of their %d games" % e["n"]))
    return L, "%s: " % gid + "; ".join(hows) + "."


def write_report(games, allgames, profile):
    gids = sorted({g["gid"] for g in allgames})
    L = ["# Opponent scouting report (final stage)", "",
         "Generated by `python tools/build_profiles.py --report` from the 91 public round-robin replays "
         "(`tests/replays/rr1-*.json`). Every number is a count from those replays; `a/b (p%)` = a out of b. "
         "Samples are small: treat patterns below ~10 observations as hints. Machine-readable sequences: "
         "`mesh/experience/opponents.json.gz`.", "",
         "Win rate = (wins + 0.5 x draws) / games, forfeits included (the per-game sections below only use games that were actually played, so their game counts can be lower). rps reactions: *beats-own-last* = they switch to the sign "
         "that would beat the sign they just threw, *loses-to-own-last* = to the one that would lose to it.", "",
         "| team | overall | best games | weakest games |", "|---|---|---|---|"]
    sections = []
    for team in FINALISTS:
        rates = []
        rows = []
        for gid in gids:
            w, d, l = record(allgames, team, gid)
            n = w + d + l
            if not n:
                continue
            ms = [m for g in allgames if g["gid"] == gid and team in g["seats"] for m in g["ms"][g["seats"].index(team)]]
            rate = (w + 0.5 * d) / n
            rates.append((rate, gid, n))
            rows.append("| %s | %d | %d-%d-%d | %d%% | %s |" % (gid, n, w, d, l, round(100 * rate),
                        "%.1f" % (statistics.median(ms) / 1000.0) if ms else "-"))
        w, d, l = record(allgames, team)
        fmt = lambda xs: ", ".join("%s %d%% (%d)" % (g, round(100 * r), n) for r, g, n in xs)
        L.append("| %s | %d-%d-%d | %s | %s |" % (team, w, d, l, fmt(sorted(rates, key=lambda x: (-x[0], -x[2]))[:3]),
                                                 fmt(sorted(rates, key=lambda x: (x[0], -x[2]))[:3])))
        S = ["", "## %s" % team, "", "| game | games | W-D-L | win rate | median s/move |", "|---|---|---|---|---|"] + rows + [""]
        hows = []
        for sec, how in (rps_section(games, team), pd_section(games, team), blotto_section(games, team, profile["teams"]),
                         guess_section(games, team, "wordle_duel", "solve"),
                         guess_section(games, team, "bulls_and_cows", "crack")):
            S += sec
            hows.append(how)
        S += ["", "**How to beat them**", ""] + ["- " + h for h in hows]
        sections += S
    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L + sections) + "\n")


def main():
    games, allgames, info = load()
    profile, sbv_missing = build(games)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(profile, fh, separators=(",", ":"), ensure_ascii=False)
    teams = profile["teams"]
    per_gid = Counter(g["gid"] for g in games)
    print("replay files: %d, games in them: %d" % (len(glob.glob(REPLAYS)), len(allgames)))
    print("teams (%d): %s" % (len(teams), ", ".join(teams)))
    print("simultaneous games kept: %d (skipped forfeits with <2 rounds: %d, rounds dropped for a timeout/illegal move: %d)"
          % (len(games), info["skipped"], info["bad_rounds"]))
    for gid in SIM:
        print("  %-18s %3d games, %4d rounds" % (gid, per_gid[gid], sum(len(g["rounds"]) for g in games if g["gid"] == gid)))
    print("games per team: " + ", ".join("%s %d" % (t, sum(v["games"] for gg in teams[t].values() for v in gg.values())) for t in teams))
    print("seq_by_value (%s): %s" % (BLOTTO_VALUED, "produced" if not sbv_missing else "LEFT OUT (%d rounds unparsable)" % sbv_missing))
    print("wrote %s (%d bytes)" % (os.path.relpath(OUT, ROOT), os.path.getsize(OUT)))
    if "--report" in sys.argv:
        write_report(games, allgames, profile)
        print("wrote %s" % os.path.relpath(REPORT, ROOT))


if __name__ == "__main__":
    main()

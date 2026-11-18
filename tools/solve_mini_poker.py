"""Offline solver for ONE HAND of the arena's Mini Poker (Kuhn and Leduc), CFR+ with alternating updates.

    python tools/solve_mini_poker.py [--kuhn-iters 100000] [--leduc-seconds 200] [--leduc-target 0.003]

Writes mesh/known/data/mini_poker_strategy.json (read by mesh/known/mini_poker.py recommend()).
The single-hand betting is re-implemented here (build_tree) and checked against the module on random hands (verify).
Payoffs are chips won by the seat that acts FIRST in the hand. Exploitability = (BR value as first vs our second
+ BR value as second vs our first) / 2, chips per hand, computed on the rounded strategy that is written.
"""
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "mesh", "known", "data", "mini_poker_strategy.json")

R = "JQK"
SIZES = {"kuhn": (1,), "leduc": (2, 4)}
CAP = {"kuhn": 1, "leduc": 2}
DECK = {"kuhn": (1, 1, 1), "leduc": (2, 2, 2)}      # copies of each rank

FORMAT = ("infoset key = '<variant>|<own card J/Q/K>|<community card J/Q/K, or - when not revealed (always - in "
          "kuhn)>|<betting of this hand in order>'. Betting = the action words of round 1 separated by commas; once "
          "round 2 has started (leduc) a '/' follows, then the round 2 actions separated by commas. Both seats' "
          "actions are listed in the order played; the seat to act follows from the count (the hand's first seat "
          "opens both rounds). Examples: 'kuhn|K|-|' (first to act), 'kuhn|Q|-|check,bet', 'leduc|K|-|bet,raise', "
          "'leduc|J|Q|check,bet,call/' , 'leduc|J|Q|bet,call/check,bet'. Value = {action: probability}. "
          "Strategies are for one hand, payoffs in chips; exploitability in chips per hand.")


def build_tree(style):
    """public betting tree: id (history string) -> node dict.
    decision: {"t": "d", "p": 0/1 (0 = first to act in the hand), "acts": [...], "kids": [...], "reveal": bool}
    terminal: {"t": "f", "w": winner, "c": (contrib first, contrib second)} or {"t": "s", "c": ...}"""
    T = {}

    def rec(nid, r, ra, contrib, turn):
        to_call = contrib[1 - turn] - contrib[turn]
        if to_call <= 0:
            la = ["check", "bet"]
        elif sum(1 for a in ra if a in ("bet", "raise")) < CAP[style]:
            la = ["call", "raise", "fold"]
        else:
            la = ["call", "fold"]
        node = {"t": "d", "p": turn, "acts": la, "kids": [], "reveal": r == 1 and not ra}
        T[nid] = node
        for a in la:
            c = list(contrib)
            c[turn] += {"bet": SIZES[style][r], "call": to_call, "raise": to_call + SIZES[style][r]}.get(a, 0)
            kid = nid + ("" if (not ra) else ",") + a
            if a == "fold":
                T[kid] = {"t": "f", "w": 1 - turn, "c": tuple(c)}
            elif a == "call" or (a == "check" and len(ra) == 1):
                if r + 1 < len(SIZES[style]):
                    kid += "/"
                    rec(kid, r + 1, [], c, 0)
                else:
                    T[kid] = {"t": "s", "c": tuple(c)}
            else:
                rec(kid, r, ra + [a], c, 1 - turn)
            node["kids"].append(kid)

    rec("", 0, [], [1, 1], 0)
    return T


def payoff(node, c0, c1, comm):
    """chips won by the first seat"""
    c = node["c"]
    if node["t"] == "f":
        return c[1] if node["w"] == 0 else -c[0]
    if c0 == c1:
        return 0
    if c0 == comm:
        w = 0
    elif c1 == comm:
        w = 1
    else:
        w = 0 if c0 > c1 else 1
    return c[1] if w == 0 else -c[0]


def deals(style):
    """[(c0, c1, probability)]"""
    d = DECK[style]
    n = sum(d)
    out = []
    for a in range(3):
        for b in range(3):
            w = d[a] * (d[b] - (a == b))
            if w > 0:
                out.append((a, b, w / (n * (n - 1.0))))
    return out


def comm_probs(style, c0, c1):
    d = list(DECK[style])
    d[c0] -= 1
    d[c1] -= 1
    n = float(sum(d))
    return [(k, d[k] / n) for k in range(3) if d[k] > 0]


class Solver:
    def __init__(self, style):
        self.style = style
        self.T = build_tree(style)
        self.reg = {}
        self.ssum = {}
        self.t = 0
        self.deals = deals(style)

    def _cfr(self, nid, c, comm, p0, p1, pc, upd):
        node = self.T[nid]
        if node["t"] != "d":
            return payoff(node, c[0], c[1], comm)
        if node["reveal"] and comm < 0:
            v = 0.0
            for k, pr in comm_probs(self.style, c[0], c[1]):
                v += pr * self._cfr(nid, c, k, p0, p1, pc * pr, upd)
            return v
        pl = node["p"]
        key = (c[pl], comm, nid)
        n = len(node["acts"])
        reg = self.reg.get(key)
        if reg is None:
            reg = self.reg[key] = [0.0] * n
            self.ssum[key] = [0.0] * n
        tot = sum(reg)
        strat = [x / tot for x in reg] if tot > 0 else [1.0 / n] * n
        us = [0.0] * n
        nv = 0.0
        for i, kid in enumerate(node["kids"]):
            if pl == 0:
                us[i] = self._cfr(kid, c, comm, p0 * strat[i], p1, pc, upd)
            else:
                us[i] = self._cfr(kid, c, comm, p0, p1 * strat[i], pc, upd)
            nv += strat[i] * us[i]
        if pl == upd:
            sign = 1.0 if pl == 0 else -1.0
            w = (p1 if pl == 0 else p0) * pc * sign
            mine = p0 if pl == 0 else p1
            ss = self.ssum[key]
            for i in range(n):
                x = reg[i] + w * (us[i] - nv)
                reg[i] = x if x > 0 else 0.0
                ss[i] += self.t * mine * strat[i]
        return nv

    def iterate(self, n):
        for _ in range(n):
            self.t += 1
            for upd in (0, 1):
                for c0, c1, pr in self.deals:
                    self._cfr("", (c0, c1), -1, 1.0, 1.0, pr, upd)

    def average(self, digits=4, floor=0.002):
        """{(card, comm, nid): {action: p}} rounded; probabilities under `floor` are purged"""
        out = {}
        for nid, node in self.T.items():
            if node["t"] != "d":
                continue
            for card in range(3):
                for comm in (range(3) if node["reveal"] or "/" in nid else (-1,)):
                    if self.style == "leduc" and comm == card and DECK[self.style][card] < 2:
                        continue
                    key = (card, comm, nid)
                    ss = self.ssum.get(key)
                    n = len(node["acts"])
                    tot = sum(ss) if ss else 0.0
                    p = [x / tot for x in ss] if tot > 0 else [1.0 / n] * n
                    p = [x if x >= floor else 0.0 for x in p]
                    s = sum(p)
                    p = [round(x / s, digits) for x in p]
                    i = max(range(n), key=lambda j: p[j])
                    p[i] = round(p[i] + 1.0 - sum(p), digits)
                    out[key] = dict(zip(node["acts"], p))
        return out


def key_str(style, key):
    card, comm, nid = key
    return "%s|%s|%s|%s" % (style, R[card], R[comm] if comm >= 0 else "-", nid)


def game_value(style, T, sig):
    """value for the first seat when both play sig"""
    def rec(nid, c, comm):
        node = T[nid]
        if node["t"] != "d":
            return payoff(node, c[0], c[1], comm)
        if node["reveal"] and comm < 0:
            return sum(pr * rec(nid, c, k) for k, pr in comm_probs(style, c[0], c[1]))
        s = sig[(c[node["p"]], comm, nid)]
        return sum(s[a] * rec(kid, c, comm) for a, kid in zip(node["acts"], node["kids"]) if s[a] > 0)
    return sum(pr * rec("", (c0, c1), -1) for c0, c1, pr in deals(style))


def best_response(style, T, sig, br):
    """value (for seat br, 0 = first) of the best response of br against sig"""
    d = DECK[style]
    n = sum(d)

    def rec(nid, own, comm, opp):
        # opp: {opp card: unnormalised weight (deal ways * opp reach * chance)}
        node = T[nid]
        if node["t"] != "d":
            v = 0.0
            for oc, w in opp.items():
                if w:
                    x = payoff(node, own, oc, comm) if br == 0 else -payoff(node, oc, own, comm)
                    v += w * x
            return v
        if node["reveal"] and comm < 0:
            v = 0.0
            for k in range(3):
                o2 = {}
                for oc, w in opp.items():
                    left = d[k] - (own == k) - (oc == k)
                    if left > 0 and w:
                        o2[oc] = w * left / (n - 2.0)
                if o2:
                    v += rec(nid, own, k, o2)
            return v
        if node["p"] == br:
            return max(rec(kid, own, comm, opp) for kid in node["kids"])
        v = 0.0
        for a, kid in zip(node["acts"], node["kids"]):
            o2 = {oc: w * sig[(oc, comm, nid)][a] for oc, w in opp.items()}
            if any(o2.values()):
                v += rec(kid, own, comm, o2)
        return v

    tot = 0.0
    for own in range(3):
        opp = {oc: d[own] * (d[oc] - (oc == own)) / (n * (n - 1.0)) for oc in range(3) if d[oc] - (oc == own) > 0}
        tot += rec("", own, -1, opp)
    return tot


def exploitability(style, T, sig):
    return (best_response(style, T, sig, 0) + best_response(style, T, sig, 1)) / 2.0


def verify(style, hands=3000, seed=1):
    """the tree here against the module: legal actions at every step and the payoff of every random hand"""
    from mesh.known import mini_poker as mp
    T = build_tree(style)
    label = "Kuhn poker, 12 hands" if style == "kuhn" else "Leduc poker, 8 hands"
    cfg = dict(mp.CONFIGS[label])
    cfg["hands"] = 1
    rng = random.Random(seed)
    seen = set()
    for h in range(hands):
        st = mp.initial_state(cfg, rng.randrange(10 ** 9))
        nid = ""
        while not mp.is_over(st):
            node = T[nid]
            seat = st["turn"]
            la = mp.legal_actions(st, seat)
            assert node["t"] == "d" and la == node["acts"], (nid, la, node)
            assert (seat != st["first"]) == bool(node["p"]), (nid, seat)
            assert node["reveal"] == (st["round"] == 1 and not st["acts"][1]), nid
            a = la[int(rng.random() * len(la))]
            nid = node["kids"][la.index(a)]
            st = mp.apply(st, {seat: a}, rng)
        node = T[nid]
        assert node["t"] != "d", nid
        hrow = st["hist"][-1]
        assert payoff(node, hrow[2], hrow[3], hrow[4]) == hrow[9], (nid, hrow)
        assert st["stacks"][0] - cfg["stack"] == hrow[9]
        seen.add(nid)
    leaves = sum(1 for v in T.values() if v["t"] != "d")
    assert len(seen) == leaves, (len(seen), leaves)
    return hands, leaves


def main():
    arg = sys.argv[1:]

    def opt(name, default):
        return type(default)(arg[arg.index(name) + 1]) if name in arg else default

    kuhn_iters = opt("--kuhn-iters", 100000)
    leduc_seconds = opt("--leduc-seconds", 200.0)
    leduc_target = opt("--leduc-target", 0.003)
    out = {"format": FORMAT, "kuhn": {}, "leduc": {}, "exploitability": {}, "value_first": {}, "iterations": {}}
    for style in ("kuhn", "leduc"):
        print("verify %s vs module: %d random hands, all %d leaves reached" % ((style,) + verify(style)), flush=True)
        s = Solver(style)
        t0 = time.time()
        if style == "kuhn":
            s.iterate(kuhn_iters)
        else:
            while True:
                s.iterate(100)
                e = exploitability(style, s.T, s.average())
                print("  leduc it %d  expl %.5f  %.0fs" % (s.t, e, time.time() - t0), flush=True)
                if e < leduc_target or time.time() - t0 > leduc_seconds:
                    break
        sig = s.average()
        e = exploitability(style, s.T, sig)
        v = game_value(style, s.T, sig)
        out[style] = {key_str(style, k): sig[k] for k in sorted(sig, key=lambda k: (k[2], k[1], k[0]))}
        out["exploitability"][style] = round(e, 6)
        out["value_first"][style] = round(v, 6)
        out["iterations"][style] = s.t
        print("%s: %d iterations, %.0fs, %d infosets, exploitability %.6f, value first %.6f"
              % (style, s.t, time.time() - t0, len(sig), e, v), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

"""Exact hand-written simulator of the arena game `mini_poker` (contract: mesh/simgen.py GEN_SPEC).

Rules that matter (from the arena's rules text, both variants):
  - a match is a series of hands, chips carry over; every hand both seats ante 1; seat 0 acts first in hands
    1, 3, 5, ..., seat 1 in hands 2, 4, 6, ... (the same seat acts first in both betting rounds of a hand);
  - Kuhn (12 hands, 12 chips): deck J Q K (one each), 1 private card each, ONE betting round, bet size 1, no raise
    (check-check, check-bet-call/fold, bet-call/fold); showdown: higher card takes the pot;
  - Leduc (8 hands, 40 chips): deck J J Q Q K K, 1 private card each + 1 community card dealt face down and revealed
    after round 1; round 1 bet size 2, round 2 bet size 4, at most 2 bets per round (a bet + one raise; a raise puts
    in the call plus one bet size); showdown: a card pairing the community card wins, otherwise the higher card,
    equal ranks split (each gets back what they put in);
  - fold: the other seat takes the pot, no cards shown;
  - BEFORE each hand a seat with fewer chips than the most one hand can cost (Kuhn 2, Leduc 13) is busted: the match
    ends at once; after the last hand (or a bust) more chips wins, equal chips = draw; move cap 200 actions.

State: flat dict. Cards are ranks 0=J 1=Q 2=K. cards = [seat 0's, seat 1's], comm = community rank (-1 in Kuhn),
shown = community revealed, stacks = chips behind (antes/bets already taken out), contrib = chips each seat put in
this hand, acts = [round 1 actions, round 2 actions] as (seat, act, amount) tuples, hist = finished hands as
(hand, first, card0, card1, comm, acts, winner, how, pot, net0, stack0, stack1) with -1 for a card never seen.

state_from_observation samples the opponent's card (and the face-down community card) uniformly among the deck cards
left once our card (and a revealed community card) are removed. Earlier hands do not constrain it: fresh shuffle.
"""
import re

CONFIGS = {
    "Kuhn poker, 12 hands": {"label": "Kuhn poker, 12 hands", "style": "kuhn", "hands": 12, "stack": 12},
    "Leduc poker, 8 hands": {"label": "Leduc poker, 8 hands", "style": "leduc", "hands": 8, "stack": 40},
}

_R = "JQK"
_RNAME = ("J (Jack)", "Q (Queen)", "K (King)")
_SIZES = {"kuhn": (1,), "leduc": (2, 4)}
_CAP = {"kuhn": 1, "leduc": 2}          # bets (bet + raises) allowed per round
_BUST = {"kuhn": 2, "leduc": 13}        # fewer chips than this before a hand = busted
_DECK = {"kuhn": (0, 1, 2), "leduc": (0, 0, 1, 1, 2, 2)}
MOVE_CAP = 200


def _style(config):
    s = config.get("style")
    if s not in _SIZES:
        s = "leduc" if "leduc" in str(config.get("label", "")).lower() else "kuhn"
    return s


def _deal(style, rng):
    deck = list(_DECK[style])
    a = deck.pop(int(rng.random() * len(deck)))
    b = deck.pop(int(rng.random() * len(deck)))
    c = deck.pop(int(rng.random() * len(deck))) if style == "leduc" else -1
    return [a, b], c


def initial_state(config, seed):
    from random import Random
    rng = Random(seed)
    style = _style(config)
    stack = int(config.get("stack", 12 if style == "kuhn" else 40))
    cards, comm = _deal(style, rng)
    return {"style": style, "label": config.get("label", ""), "hands": int(config.get("hands", 12 if style == "kuhn" else 8)),
            "start": stack, "hand": 1, "first": 0, "stacks": [stack - 1, stack - 1], "cards": cards, "comm": comm,
            "shown": False, "contrib": [1, 1], "round": 0, "acts": [[], []], "turn": 0, "nact": 0,
            "over": False, "winner": None, "hist": []}


def to_move(state):
    return [] if state["over"] else [state["turn"]]


def is_over(state):
    return state["over"]


def winner(state):
    return state["winner"] if state["over"] else None


def legal_actions(state, seat):
    if state["over"] or seat != state["turn"]:
        return []
    c = state["contrib"]
    if c[1 - seat] <= c[seat]:
        return ["check", "bet"]
    nb = 0
    for a in state["acts"][state["round"]]:
        if a[1] == "bet" or a[1] == "raise":
            nb += 1
    if nb < _CAP[state["style"]]:
        return ["call", "raise", "fold"]
    return ["call", "fold"]


def random_action(state, seat, rng):
    la = legal_actions(state, seat)
    return la[int(rng.random() * len(la))] if la else "check"


def _norm(action):
    try:
        t = str(action).strip().lower().split()
        return t[0].strip(".,;:!\"'") if t else ""
    except Exception:
        return ""


def is_legal_action(state, seat, action):
    try:
        return _norm(action) in legal_actions(state, seat)
    except Exception:
        return False


def could_win_now(state, seat, action):
    return False


def _finish(st):
    s = st["stacks"]
    st["over"] = True
    st["winner"] = 0 if s[0] > s[1] else (1 if s[1] > s[0] else None)


def apply(state, actions, rng):
    st = dict(state)
    if state["over"]:
        return st
    seat = state["turn"]
    opp = 1 - seat
    a = actions.get(seat)
    if a is None:
        a = actions.get(str(seat))
    a = _norm(a)
    la = legal_actions(state, seat)
    if a not in la:
        a = la[-1] if la[0] == "call" else la[0]   # fold when facing a bet, else check
    style = state["style"]
    r = state["round"]
    size = _SIZES[style][r]
    stacks = list(state["stacks"])
    contrib = list(state["contrib"])
    acts = [list(state["acts"][0]), list(state["acts"][1])]
    st["stacks"], st["contrib"], st["acts"] = stacks, contrib, acts
    st["nact"] = state["nact"] + 1
    to_call = contrib[opp] - contrib[seat]
    amt = 0
    if a == "bet":
        amt = size
    elif a == "call":
        amt = to_call
    elif a == "raise":
        amt = to_call + size
    contrib[seat] += amt
    stacks[seat] -= amt
    acts[r].append((seat, a, amt))
    end = None          # None = hand goes on; else (winner or None for a split, how)
    if a == "fold":
        end = (opp, "fold")
    elif a == "call" or (a == "check" and len(acts[r]) == 2):
        if r + 1 < len(_SIZES[style]):
            st["round"] = r + 1
            st["shown"] = True
            st["turn"] = state["first"]
        else:
            c0, c1 = state["cards"]
            cm = state["comm"]
            if c0 == c1:
                end = (None, "showdown")
            elif c0 == cm:
                end = (0, "showdown")
            elif c1 == cm:
                end = (1, "showdown")
            else:
                end = (0 if c0 > c1 else 1, "showdown")
    else:
        st["turn"] = opp
    if end is None:
        if st["nact"] >= MOVE_CAP:      # cannot happen in normal play: chips go back, more chips wins
            stacks[0] += contrib[0]
            stacks[1] += contrib[1]
            _finish(st)
        return st
    w, how = end
    pot = contrib[0] + contrib[1]
    if w is None:
        stacks[0] += contrib[0]
        stacks[1] += contrib[1]
        net0 = 0
    else:
        stacks[w] += pot
        net0 = contrib[1] if w == 0 else -contrib[0]
    hist = list(state["hist"])
    hist.append((state["hand"], state["first"], state["cards"][0], state["cards"][1], state["comm"],
                 (tuple(acts[0]), tuple(acts[1])), w, how, pot, net0, stacks[0], stacks[1]))
    st["hist"] = hist
    bust = _BUST[style]
    if state["hand"] >= state["hands"] or stacks[0] < bust or stacks[1] < bust or st["nact"] >= MOVE_CAP:
        _finish(st)
        return st
    h = state["hand"] + 1
    st["hand"] = h
    st["first"] = st["turn"] = (h + 1) % 2
    st["cards"], st["comm"] = _deal(style, rng)
    st["shown"] = False
    st["round"] = 0
    st["acts"] = [[], []]
    st["contrib"] = [1, 1]
    stacks[0] -= 1
    stacks[1] -= 1
    return st


def _acts_text(acts, seat):
    return ", ".join(("you " if s == seat else "opponent ") + a + (" %d" % n if n else "") for s, a, n in acts)


def observation(state, seat):
    style = state["style"]
    opp = 1 - seat
    sizes = _SIZES[style]
    L = ["MINI POKER - %s. You are seat %d." % (state["label"], seat),
         "Hand %d of %d. %s first in this hand." % (state["hand"], state["hands"],
                                                    "You act" if state["first"] == seat else "Your opponent acts"),
         "Chips behind (not counting the pot): you %d, opponent %d. Both started with %d."
         % (state["stacks"][seat], state["stacks"][opp], state["start"]),
         "Your card: " + _RNAME[state["cards"][seat]]]
    if style == "leduc":
        L.append("Community card: " + (_RNAME[state["comm"]] if state["shown"] and state["comm"] >= 0
                                       else "not revealed yet (it is revealed after betting round 1)"))
    c = state["contrib"]
    L.append("Pot: %d (you put in %d, opponent %d, antes included)" % (c[0] + c[1], c[seat], c[opp]))
    L.append("Betting in this hand so far:")
    r = state["round"]
    for i in range(r + 1):
        L.append("  Round %d (bet size %d): %s" % (i + 1, sizes[i], _acts_text(state["acts"][i], seat) or "(no actions yet)"))
    if state["over"]:
        L.append("The match is over.")
    elif state["turn"] == seat:
        to_call = c[opp] - c[seat]
        parts = []
        for a in legal_actions(state, seat):
            if a == "bet":
                a += " (puts in %d)" % sizes[r]
            elif a == "call":
                a += " (puts in %d)" % to_call
            elif a == "raise":
                a += " (puts in %d)" % (to_call + sizes[r])
            parts.append(a)
        L.append("It is your turn in betting round %d. Legal actions: %s" % (r + 1, ", ".join(parts)))
    else:
        L.append("It is the opponent's turn in betting round %d." % (r + 1))
    hist = state["hist"]
    if not hist:
        L.append("Previous hands: none yet.")
        return "\n".join(L)
    L.append("Previous hands:")
    for hand, first, c0, c1, cm, acts, w, how, pot, net0, s0, s1 in hist:
        mine, theirs = (c0, c1) if seat == 0 else (c1, c0)
        ln = "  Hand %d (%s acted first): you had %s, " % (hand, "you" if first == seat else "opponent", _R[mine])
        ln += "opponent's card not shown (fold)" if how == "fold" or theirs < 0 else "opponent had " + _R[theirs]
        if style == "leduc":
            ln += ", community " + (_R[cm] if acts[1] and cm >= 0 else "card not revealed")
        ln += ". Betting: R1: " + _acts_text(acts[0], seat)
        if acts[1]:
            ln += "; R2: " + _acts_text(acts[1], seat)
        net = net0 if seat == 0 else -net0
        if w is None:
            ln += ". Result: split pot of %d (+0)" % pot
        else:
            ln += ". Result: %s won the pot of %d by %s (%+d)" % ("you" if w == seat else "opponent", pot, how, net)
        ln += ". Stacks after: you %d, opponent %d." % ((s0, s1) if seat == 0 else (s1, s0))
        L.append(ln)
    return "\n".join(L)


# ---------------------------------------------------------------- parsing + sampling of the hidden cards

_RE_ACT = re.compile(r"\b(you|opponent)\s+(check|bet|call|raise|fold)\b(?:\s+(\d+))?", re.I)
_RE_ROUND = re.compile(r"^\s*Round\s+(\d+)[^:\n]*:(.*)$", re.I | re.M)
_RE_HAND = re.compile(
    r"^\s*Hand\s+(\d+)\s*\((you|opponent)\s+acted\s+first\)\s*:\s*you had\s+([JQK])\s*,\s*"
    r"(?:opponent had\s+([JQK])|opponent's card not shown[^,.]*)"
    r"(?:\s*,\s*community\s+(?:([JQK])\b|card not revealed))?\s*\.\s*Betting:(.*?)\.\s*Result:(.*?)\.\s*"
    r"Stacks after:\s*you\s+(-?\d+)\s*,\s*opponent\s+(-?\d+)", re.I | re.M)
_RE_WON = re.compile(r"(you|opponent)\s+won the pot of\s+(\d+)\s+by\s+(\w+)\s*\(\s*([+-]?\d+)\s*\)", re.I)
_OBS_CACHE = {}


def _acts(s, seat):
    return [(seat if w.lower() == "you" else 1 - seat, a.lower(), int(n) if n else 0) for w, a, n in _RE_ACT.findall(s)]


def _parse(text, seat, config):
    style = _style(config)
    m = re.search(r"You are seat\s*(\d)", text)
    if seat not in (0, 1):
        seat = int(m.group(1)) if m else 0
    opp = 1 - seat
    start = int(config.get("stack", 12 if style == "kuhn" else 40))
    m = re.search(r"Hand\s+(\d+)\s+of\s+(\d+)", text)
    hand = int(m.group(1)) if m else 1
    hands = int(config.get("hands", int(m.group(2)) if m else (12 if style == "kuhn" else 8)))
    if re.search(r"You act first", text, re.I):
        first = seat
    elif re.search(r"opponent acts first", text, re.I):
        first = opp
    else:
        first = (hand + 1) % 2
    m = re.search(r"Your card:\s*([JQK])\b", text)
    mine = _R.index(m.group(1)) if m else 0
    m = re.search(r"Community card:\s*([JQK])\b", text)
    comm = _R.index(m.group(1)) if m else -1
    acts = [[], []]
    for rm in _RE_ROUND.finditer(text):
        i = int(rm.group(1)) - 1
        if 0 <= i < 2:
            acts[i] = _acts(rm.group(2), seat)
    if style == "kuhn":
        acts[1] = []
    m = re.search(r"betting round\s+(\d+)", text, re.I)
    rnd = int(m.group(1)) - 1 if m else (1 if (comm >= 0 or acts[1]) else 0)
    rnd = max(0, min(rnd, len(_SIZES[style]) - 1))
    m = re.search(r"Pot:\s*(\d+)\s*\(\s*you put in\s+(\d+)\s*,\s*opponent\s+(\d+)", text)
    contrib = [1, 1]
    if m:
        contrib[seat], contrib[opp] = int(m.group(2)), int(m.group(3))
    else:
        for rr in acts:
            for s, _a, n in rr:
                contrib[s] += n
    hist = []
    nact = len(acts[0]) + len(acts[1])
    for hm in _RE_HAND.finditer(text):
        hfirst = seat if hm.group(2).lower() == "you" else opp
        cards = [-1, -1]
        cards[seat] = _R.index(hm.group(3).upper())
        if hm.group(4):
            cards[opp] = _R.index(hm.group(4).upper())
        cm = _R.index(hm.group(5).upper()) if hm.group(5) else -1
        ha = [(), ()]
        for part in hm.group(6).split(";"):
            pm = re.match(r"\s*R(\d+)\s*:(.*)", part, re.I | re.S)
            if pm and 1 <= int(pm.group(1)) <= 2:
                ha[int(pm.group(1)) - 1] = tuple(_acts(pm.group(2), seat))
        nact += len(ha[0]) + len(ha[1])
        wm = _RE_WON.search(hm.group(7))
        if wm:
            w = seat if wm.group(1).lower() == "you" else opp
            how = wm.group(3).lower()
            pot = int(wm.group(2))
            net = int(wm.group(4))
            net0 = net if seat == 0 else -net
        else:
            sm = re.search(r"pot of\s+(\d+)", hm.group(7))
            w, how, pot, net0 = None, "showdown", int(sm.group(1)) if sm else 0, 0
        sy, so = int(hm.group(8)), int(hm.group(9))
        hist.append((int(hm.group(1)), hfirst, cards[0], cards[1], cm, (ha[0], ha[1]), w, how, pot, net0,
                     sy if seat == 0 else so, so if seat == 0 else sy))
    stacks = [0, 0]
    m = re.search(r"Chips behind[^:\n]*:\s*you\s+(-?\d+)\s*,\s*opponent\s+(-?\d+)", text)
    if m:
        stacks[seat], stacks[opp] = int(m.group(1)), int(m.group(2))
    else:
        base = (hist[-1][10], hist[-1][11]) if hist else (start, start)
        stacks = [base[0] - contrib[0], base[1] - contrib[1]]
    rest = list(_DECK[style])
    rest.remove(mine)
    if comm >= 0 and comm in rest:
        rest.remove(comm)
    turn = seat
    if re.search(r"opponent's turn", text, re.I) and not re.search(r"your turn", text, re.I):
        turn = opp
    return {"style": style, "label": config.get("label", ""), "hands": hands, "start": start, "hand": hand,
            "first": first, "stacks": stacks, "mine": mine, "comm": comm, "contrib": contrib, "round": rnd,
            "acts": acts, "turn": turn, "nact": nact, "hist": hist, "rest": rest, "seat": seat,
            "over": "match is over" in text.lower()}


def state_from_observation(text, seat, config, rng):
    key = (text, seat, config.get("label"), config.get("style"), config.get("hands"), config.get("stack"))
    P = _OBS_CACHE.get(key)
    if P is None:
        P = _parse(text, seat, config)
        if len(_OBS_CACHE) >= 64:
            _OBS_CACHE.clear()
        _OBS_CACHE[key] = P
    seat = P["seat"]
    style = P["style"]
    rest = P["rest"]
    i = int(rng.random() * len(rest))
    theirs = rest[i]
    comm = P["comm"]
    shown = comm >= 0
    if style == "leduc" and not shown:
        j = int(rng.random() * (len(rest) - 1))
        comm = rest[j if j < i else j + 1]
    cards = [0, 0]
    cards[seat] = P["mine"]
    cards[1 - seat] = theirs
    st = {"style": style, "label": P["label"], "hands": P["hands"], "start": P["start"], "hand": P["hand"],
          "first": P["first"], "stacks": list(P["stacks"]), "cards": cards, "comm": comm if style == "leduc" else -1,
          "shown": shown, "contrib": list(P["contrib"]), "round": P["round"],
          "acts": [list(P["acts"][0]), list(P["acts"][1])], "turn": P["turn"], "nact": P["nact"],
          "over": P["over"], "winner": None, "hist": list(P["hist"])}
    if st["over"]:
        _finish(st)
    return st


# ---------------------------------------------------------------- equilibrium policy for single hands
# Table made offline by tools/solve_mini_poker.py (CFR+); the key format is documented in the JSON ("format").

_STRAT = None
_LOCK_AHEAD = True      # fold/check everything once the match cannot be lost any more


def _strategy():
    global _STRAT
    if _STRAT is None:
        import json
        import os
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mini_poker_strategy.json")
            with open(path, encoding="utf-8") as f:
                _STRAT = json.load(f)
        except Exception:
            _STRAT = {}
    return _STRAT


def infoset_key(state, seat):
    """what `seat` knows: variant|own card|community card or -|betting of this hand (round 2 after a '/')"""
    style = state["style"]
    comm = state["comm"] if (style == "leduc" and state.get("shown") and state["comm"] >= 0) else -1
    h = ",".join(a[1] for a in state["acts"][0])
    if style == "leduc" and (state["round"] >= 1 or comm >= 0):
        h += "/" + ",".join(a[1] for a in state["acts"][1])
    return "%s|%s|%s|%s" % (style, _R[state["cards"][seat]], _R[comm] if comm >= 0 else "-", h)


def recommend(text, seat, config, rng, seconds=1.0):
    """returns a list of (action, value, p) tuples, most likely first"""
    try:
        if rng is None:
            from random import Random
            rng = Random(0)
        st = state_from_observation(text, seat, config, rng)
        if seat not in (0, 1):
            seat = st["turn"]
        la = legal_actions(st, seat)
        if not la:
            return []
        # match locked: giving up this pot and one ante in every later hand still leaves us ahead
        left = st["hands"] - st["hand"]
        opp = 1 - seat
        if _LOCK_AHEAD and st["stacks"][seat] - left > st["stacks"][opp] + st["contrib"][0] + st["contrib"][1] + left:
            a = "check" if "check" in la else ("fold" if "fold" in la else None)
            if a:
                return [(a, 1.0, 1.0)]
        table = _strategy().get(st["style"]) or {}
        probs = table.get(infoset_key(st, seat))
        if not probs:
            return []
        out = [(a, max(0.0, float(probs.get(a, 0.0)))) for a in la]
        tot = sum(p for _a, p in out)
        if tot <= 0.0:
            return []
        out.sort(key=lambda x: -x[1])
        return [(a, p / tot, p / tot) for a, p in out]
    except Exception:
        return []

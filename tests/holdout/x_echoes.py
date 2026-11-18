"""Echoes (our own game concept for the presentation; also a holdout test game). Owner: path-to-glory-3.

Not on the platform: invented by team path-to-glory to show a game the harnesses could play.
HOLDOUT: written only to measure how mesh/ copes with a game it has never seen. Do not tune mesh/ to it.

Two hidden beacons on a grid, simultaneous rounds. Each round both players secretly either PING a cell (and hear how
far the enemy beacon is from it, Manhattan distance) or DRIFT their own beacon one step (then they hear nothing that
round). Drifts resolve before pings, so a well-timed drift dodges a ping that would have found you. The first to ping
the enemy beacon exactly wins. It mixes deduction (keep the cells consistent with every echo), simultaneous mind games
(ping now, or move away because they are close?) and chat as a weapon (bluff where you are, never reveal it).
"""
from __future__ import annotations

import re

from ..core import Game

DIRS = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}
COLS = "abcdefghij"


class Echoes(Game):
    id = "x_echoes"
    name = "Echoes"
    category = "guess"
    blurb = "Hunt a hidden beacon by its echo, while yours slips away."
    howto = ("Both players hide a beacon on a grid. Each round, at the same time, ping a cell to hear how far the enemy "
             "beacon is, or drift yours one step to dodge. First to ping the enemy beacon exactly wins.")
    duplicate_safe = False

    @classmethod
    def variants(cls) -> list[dict]:
        return [
            {"label": "5x5, 3 drifts, 14 rounds", "size": 5, "drifts": 3, "rounds": 14, "noise": False},
            {"label": "7x7, 4 drifts, 18 rounds, noisy echoes", "size": 7, "drifts": 4, "rounds": 18, "noise": True},
        ]

    def __init__(self, variant: dict, seed: int):
        super().__init__(variant, seed)
        self.n = int(self.variant.get("size", 5))
        self.max_drifts = int(self.variant.get("drifts", 3))
        self.max_rounds = int(self.variant.get("rounds", 14))
        self.noise = bool(self.variant.get("noise"))
        self.pos = [(self.rng.randrange(self.n), self.rng.randrange(self.n)) for _ in (0, 1)]
        while self.pos[1] == self.pos[0]:
            self.pos[1] = (self.rng.randrange(self.n), self.rng.randrange(self.n))
        self.drifts = [0, 0]
        self.log: list[list[str]] = [[], []]       # what each seat did and heard, as text lines for its observation
        self.seen_opp: list[list[str]] = [[], []]  # what each seat saw the opponent do
        self.best = [99, 99]                       # closest echo heard by each seat
        self.round = 1
        self._result = None
        self._re = re.compile(rf"^(?:ping\s+([{COLS[:self.n]}])(\d+)|drift\s+(north|south|east|west))$")

    def cell(self, rc):
        return f"{COLS[rc[1]]}{rc[0] + 1}"

    def rules(self) -> str:
        last = f"{COLS[self.n - 1]}{self.n}"
        noise = ("NOISY ECHOES: a reported distance of 1 or more may be off by one (one less, the same, or one more, "
                 "each equally likely), but never below 1. A ping on the beacon itself always reports 0.\n"
                 if self.noise else "Echoes are exact.\n")
        return f"""ECHOES - {self.variant.get('label')}. 2 players, simultaneous rounds.

THE GRID
{self.n}x{self.n} cells, columns a-{COLS[self.n - 1]} (left to right) and rows 1-{self.n} (top to bottom): a1 is the top-left
corner, {last} the bottom-right. The arena hides one beacon for each player on a random cell (different cells).
You always see where YOUR beacon is; you never see the opponent's.

EACH ROUND (both players choose at the same time)
  ping <cell>      for example: ping c3   - after the round you hear the echo: the Manhattan distance
                   (|row difference| + |column difference|) from that cell to the ENEMY beacon.
  drift <dir>      dir = north, south, east or west - your beacon moves one cell that way. You may drift at most
                   {self.max_drifts} times per game, never off the grid. You hear no echo in a round you drift.
Drifts happen BEFORE pings: a drift can dodge a ping that would have hit you.
You also see what the opponent did: the cell it pinged and the echo it heard (it was measured from YOUR beacon),
or that it drifted (not in which direction).
{noise}
HOW THE GAME ENDS
A ping that reports 0 finds the enemy beacon: that player wins at once. If both find each other in the same round,
it is a draw. After {self.max_rounds} rounds without a find, the player whose closest echo was smaller wins; equal is a draw.

ACTION FORMAT
ping <column><row>   or   drift <north|south|east|west>      for example: ping b4   drift east
The list of legal actions is sent when it has at most 50 entries. Anything else (a cell off the grid, a drift off
the grid or past your limit) is illegal (first time: a warning and the same turn again; second time: you lose the game).
"""

    def to_move(self):
        return [] if self._result else [0, 1]

    def _legal(self, seat):
        out = [f"ping {COLS[c]}{r + 1}" for r in range(self.n) for c in range(self.n)]
        if self.drifts[seat] < self.max_drifts:
            r, c = self.pos[seat]
            for d, (dr, dc) in DIRS.items():
                if 0 <= r + dr < self.n and 0 <= c + dc < self.n:
                    out.append(f"drift {d}")
        return out

    def legal_actions(self, seat):
        if self._result:
            return []
        return self.cap_list(self._legal(seat))

    def normalize(self, seat, action):
        if self._result:
            return None
        m = self._re.match(self.clean(action))
        if not m:
            return None
        a = f"ping {m.group(1)}{int(m.group(2))}" if m.group(1) else f"drift {m.group(3)}"
        return a if a in self._legal(seat) else None

    def _echo(self, d):
        if not self.noise or d == 0:
            return d
        return max(1, d + self.rng.choice((-1, 0, 1)))

    def apply(self, actions):
        if self._result:
            return
        acts = {}
        for s in (0, 1):
            a = self.normalize(s, actions.get(s, ""))
            if a is None:
                raise ValueError(f"illegal action for seat {s}: {actions.get(s)!r}")
            acts[s] = a
        for s in (0, 1):  # drifts first
            if acts[s].startswith("drift"):
                dr, dc = DIRS[acts[s].split()[1]]
                self.pos[s] = (self.pos[s][0] + dr, self.pos[s][1] + dc)
                self.drifts[s] += 1
        found = []
        heard = {}
        for s in (0, 1):
            if acts[s].startswith("ping"):
                cell = acts[s].split()[1]
                r, c = int(cell[1:]) - 1, COLS.index(cell[0])
                er, ec = self.pos[1 - s]
                d = abs(r - er) + abs(c - ec)
                e = self._echo(d)
                heard[s] = (cell, e)
                self.best[s] = min(self.best[s], e)
                if d == 0:
                    found.append(s)
        for s in (0, 1):
            o = 1 - s
            if s in heard:
                self.log[s].append(f"round {self.round}: you pinged {heard[s][0]} -> echo {heard[s][1]}")
            else:
                self.log[s].append(f"round {self.round}: you drifted {acts[s].split()[1]} (now at {self.cell(self.pos[s])})")
            if o in heard:
                self.seen_opp[s].append(f"round {self.round}: opponent pinged {heard[o][0]} -> it heard echo {heard[o][1]}")
            else:
                self.seen_opp[s].append(f"round {self.round}: opponent drifted (direction unknown)")
        if found:
            w = found[0] if len(found) == 1 else None
            self._finish(w, f"round {self.round}: " + ("both beacons found at once" if w is None else f"seat {w} found the enemy beacon"))
            return
        if self.round >= self.max_rounds:
            b = self.best
            w = None if b[0] == b[1] else (0 if b[0] < b[1] else 1)
            self._finish(w, f"{self.max_rounds} rounds without a find; closest echoes {b[0]} vs {b[1]}")
            return
        self.round += 1

    def _finish(self, w, reason):
        scores = [0.5, 0.5] if w is None else [1 if w == 0 else 0, 1 if w == 1 else 0]
        self._result = {"winner": w, "scores": scores, "reason": reason}

    def is_over(self):
        return self._result is not None

    def result(self):
        return dict(self._result) if self._result else {"winner": None, "scores": [0, 0], "reason": "game in progress"}

    def observation(self, seat):
        L = [f"ECHOES ({self.variant.get('label')}). You are seat {seat}.",
             f"Round {self.round} of {self.max_rounds}. Your beacon is at {self.cell(self.pos[seat])} (keep it secret). "
             f"Drifts used: you {self.drifts[seat]} of {self.max_drifts}, opponent {self.drifts[1 - seat]} of {self.max_drifts}.",
             "Your rounds:"]
        L += ["  " + x for x in self.log[seat]] or ["  (none yet)"]
        L.append("What the opponent did (its echoes were measured from YOUR beacon):")
        L += ["  " + x for x in self.seen_opp[seat]] or ["  (none yet)"]
        if self._result:
            L.append(f"Game over: {self._result['reason']}.")
        else:
            L.append('Your move: "ping <cell>" or "drift <north|south|east|west>".')
        return "\n".join(L)

    def frame(self):
        return {"beacons": [self.cell(p) for p in self.pos], "drifts": list(self.drifts), "round": self.round,
                "best": list(self.best), "over": self.is_over(), "result": self._result}

    def random_action(self, seat, rng):
        return rng.choice(self._legal(seat))

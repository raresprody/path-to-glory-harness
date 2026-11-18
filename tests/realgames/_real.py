"""Wraps a replay-checked simulator (tests/oracles) as an arena-lite Game (tests only, path-to-glory-3)."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from ..core import Game

_DATA = json.loads((Path(__file__).with_name("_real_rules.json")).read_text())


def make(gid: str, sim):
    labels = list(sim.CONFIGS)
    rules_by = _DATA["rules"].get(gid, {})

    class RealGame(Game):
        id = gid
        name = _DATA["name"].get(gid, gid)
        category = _DATA["category"].get(gid, "think")
        blurb = f"Real T2 game {gid} (local copy for tests)."
        howto = blurb
        duplicate_safe = False

        @classmethod
        def variants(cls):
            # only the variants whose real rules text we have (the bot must read the arena's own words)
            return [dict(sim.CONFIGS[l], label=l) for l in labels if l in rules_by] or [dict(sim.CONFIGS[labels[0]], label=labels[0])]

        def __init__(self, variant, seed):
            super().__init__(variant, seed)
            self.label = self.variant.get("label") or labels[0]
            self.st = sim.initial_state(copy.deepcopy(sim.CONFIGS[self.label]), seed)
            self.sim_rng = random.Random(seed * 7 + 1)

        def rules(self):
            return rules_by.get(self.label) or next(iter(rules_by.values()), f"{gid} ({self.label})")

        def to_move(self):
            return [] if sim.is_over(self.st) else list(sim.to_move(self.st))

        def observation(self, seat):
            return sim.observation(self.st, seat)

        def legal_actions(self, seat):
            if sim.is_over(self.st) or seat not in sim.to_move(self.st):
                return []
            if self.variant.get("salvo"):
                return None  # salvo: combinations of several cells, never listed
            la = sim.legal_actions(self.st, seat)
            if la is None:
                return None
            la = list(la)
            # some simulators give a REPRESENTATIVE list for huge spaces (Bulls: a sample of codes): then no list
            rng, seen = random.Random(len(la)), set(la)
            for _ in range(12):
                a = sim.random_action(self.st, seat, rng)
                if a not in seen and self.normalize(seat, a) is not None:
                    return None
            return self.cap_list(la)

        def normalize(self, seat, action):
            if not isinstance(action, str) or sim.is_over(self.st) or seat not in sim.to_move(self.st):
                return None
            a = " ".join(action.strip().split())
            for cand in (a, a.lower()):
                try:
                    if sim.is_legal_action(self.st, seat, cand):
                        return cand
                except Exception:  # noqa: BLE001
                    return None
            return None

        def apply(self, actions):
            self.st = sim.apply(self.st, dict(actions), self.sim_rng)

        def is_over(self):
            return bool(sim.is_over(self.st))

        def result(self):
            if not sim.is_over(self.st):
                return {"winner": None, "scores": [0, 0], "reason": "game in progress"}
            w = sim.winner(self.st)
            reason = self.st.get("reason") if isinstance(self.st, dict) and isinstance(self.st.get("reason"), str) else None
            return {"winner": w, "scores": [0.5, 0.5] if w is None else [1 - w, w],
                    "reason": reason or ("draw" if w is None else f"seat {w} won")}

        def frame(self):
            return {"label": self.label, "over": self.is_over()}

        def random_action(self, seat, rng):
            return sim.random_action(self.st, seat, rng)

    RealGame.__name__ = f"Real_{gid}"
    return RealGame

"""Opponent that plays like the teams that beat us in T1 (zero spend, no model): a test bot, not ours.

    python3 tests/opponents/t1_style_bot.py            # protocol bot on stdin/stdout, like nasty_bot.py
    python tests/sweep.py --bot IMAGE --only colonel_blotto,wordle_duel,x_codebreak \
        --opponent "python3 ../../../opponents/t1_style_bot.py"

- Colonel Blotto (neura, final): concentrates on the top half of the battlefields by value (6 fields: 3 x 20,
  like neura), splits evenly, and SAYS SO in chat every round (neura's chat was true; T2 plan v2 item (c)).
- Wordle Duel (alt/neura found the word in 3-4 rounds): keeps only the words consistent with all of its own feedback
  (the game's own word list and scoring, imported from the arena-lite copy it runs in) and never repeats a guess.
- Anything else: a random legal move, or nasty_bot's template filler when the legal list is null; plain chat.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

sys.path.insert(0, os.getcwd())  # the sweep runs opponents from its arena-lite copy: `arena` is importable from there
from nasty_bot import free_action  # noqa: E402  (same folder)

try:
    from arena.practice.wordle_duel import WORDS, score_guess
except Exception:  # noqa: BLE001  (not in an arena-lite folder: Wordle falls back to the template filler)
    WORDS, score_guess = {}, None


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def send(obj):
    print(json.dumps(obj), flush=True)


def blotto(obs: str):
    vals = [int(v) for v in re.findall(r"F\d+=(\d+)", obs.split("Battlefield values this round:", 1)[-1].split("\n", 1)[0])]
    if not vals:
        return None
    m = re.search(r"place exactly (\d+) soldiers", obs) or re.search(r"Your reserve: (\d+) soldiers", obs)
    budget = int(m.group(1)) if m else 0
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
    chosen = order[:(len(vals) + 1) // 2]  # the top half of the fields, as neura did (6 fields -> 3 x 20)
    alloc = [0] * len(vals)
    for j, i in enumerate(chosen):
        alloc[i] = budget // len(chosen) + (1 if j < budget % len(chosen) else 0)
    names = ", ".join(f"F{i + 1}" for i in sorted(chosen))
    return " ".join(map(str, alloc)), f"All in on {names} again. Top fields, nothing elsewhere."


def wordle(obs: str, rng: random.Random):
    m = re.search(r"WORDLE DUEL \((\d+) letters", obs)
    if not m or not WORDS or score_guess is None:
        return None
    n = int(m.group(1))
    mine = obs.split("Your guesses", 1)[-1].split("Opponent's guesses", 1)[0]
    seen = [(w.lower(), "".join(k.split())) for w, k in re.findall(r"^\s*\d+\.\s+([A-Z]+)\s+((?:[GYX]\s?)+?)\s{2,}", mine, re.M)]
    pool = [w for w in WORDS.get(n, []) if all(score_guess(g, w) == k for g, k in seen) and w not in {g for g, _ in seen}]
    word = rng.choice(pool) if pool else rng.choice(WORDS.get(n) or ["a" * n])
    return f"guess {word}", f"{len(pool)} words left on my side."


def main():
    rng = random.Random()
    for raw in sys.stdin:
        if not raw.strip():
            continue
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind == "prep":
            send({"type": "ready"})
        elif kind == "start":
            send({"type": "ready", "chat": "Good luck. I play to win."})
        elif kind == "turn":
            obs, legal = msg.get("observation") or "", msg.get("legal_actions")
            got = blotto(obs) if "COLONEL BLOTTO" in obs else wordle(obs, rng) if "WORDLE DUEL" in obs else None
            if got:
                action, chat = got
            else:
                action = rng.choice(legal) if legal else free_action(obs, rng)
                chat = "Your move."
            send({"type": "action", "action": action, "chat": chat})
        elif kind == "end":
            log(f"game {msg.get('game_index')}: {msg.get('result')} ({msg.get('reason')})")
        elif kind == "shutdown":
            break


if __name__ == "__main__":
    main()

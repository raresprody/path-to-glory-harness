# FLOW.md: how the harness and Haiku work together (as implemented, t2 candidate)

This is the end-to-end picture of what runs, in which order, and who owns each step. It describes `t2/candidate`
(2d17f93, 14 Nov 22:15). The agreed design and interfaces are in [DESIGN.md](DESIGN.md); the language side in detail
is in [LANGUAGE.md](LANGUAGE.md). Owners: **Denis** (path-to-glory-1): shell, brain, engine, model client.
**Cristi** (path-to-glory-2): `mesh/talk.py` and the filter part of `mesh/digest.py`. **Rares** (path-to-glory-3): `tests/`.

## 1. What the arena sends, and when

The arena talks to our process with one JSON object per line on stdin; we answer with one JSON line on stdout
(logs go to stderr). One container plays one game for the whole tournament.

| Message | How often | Carries | We answer |
|---|---|---|---|
| `prep` | **once**, after a freeze | the rules of every variant, one real `sample_turn` per variant, 600 s, the prep LLM key | `{"type":"ready"}` |
| `start` | each game | drawn variant, rules, seat, opponent, match score, match chat so far, **this match's LLM key** | `{"type":"ready","chat":…}` within 30 s (warm) / 120 s (cold) |
| `turn` | each move | `observation`, `legal_actions` (null when > 50), 20 s, **the full match chat** | `{"type":"action","action":…,"chat":…}` (chat ≤ 200 chars) |
| `turn` + `warning` | after a first illegal answer in a game | the same turn again and the reason | a legal move (a second bad answer loses the game) |
| `end` | each game | result, score, `match_over` | nothing (`match_over` does **not** end the container) |
| `shutdown` / EOF | once, at the end | nothing | 15 s to save to `/memory`, then exit |

## 2. Prep: becoming a specialist (600 s, spend and time not scored)

```
prep ─► brain.on_prep
          1. rules digest (digest.py, Denis)     action format, examples, flags (simultaneous, hidden info,
                                                 talk-is-move), forbidden terms + speech rules, playbook
          2. simulator (simgen.py, Denis)        Haiku writes a Python simulator from the rules; validated by
                                                 self-play, render/parse round trip, the real sample_turns;
                                                 fix loop; cross-check against an independent second simulator
                                                 + arbiter. Skipped for real talking games; a command-format
                                                 deduction game tagged "talk" (Wordle) still gets one and the
                                                 talk flag is cleared
          3. talk.prepare (talk.py, Cristi)      strategy study per variant, wider forbidden-word list,
                                                 format drill of our own move-picker on each sample_turn
          4. opening book (Denis)                with the time left
        ─► ready (≥ 30 s before the clock ends); everything saved under /memory/<game_id>/<rules-hash>/
```

## 3. Start of a game

`brain.on_start` loads the simulator and notes for the drawn variant (a **cold** container with ≥ 90 s of study
time builds them now), picks the mode (engine, or Haiku-only if the engine keeps losing that variant in our own
`results.jsonl`), and `talk.ready_chat` writes the opening line (a bluff, ≤ 150 chars, filtered like every line).

## 4. One turn

```
ARENA ── {"type":"turn", observation, legal_actions, chat} ──►
  │
  ① SHELL (main.py, Denis)    reads the line, starts the clock (sends at 20 s − 3 s at the latest),
  │                           keeps a legal fallback move ready from the first millisecond
  ② BRAIN (brain.py, Denis)   builds ctx: rules, observation, legal moves, chat, digest, opponent profile,
  │                           deadline, game_id; first turn: forbidden words from the private deal
  ③ ENGINE (Denis)            the simulator + search give candidate moves with values:
  │                           - small move lists: MCTS (hidden parts re-sampled every iteration)
  │                           - both move at once: regret matching, the mix is drawn in code
  │                           - huge action spaces (Blotto, a hidden word): Haiku proposes, the engine
  │                             screens (only moves that are legal / consistent with all feedback)
  │                           - a hidden number: every value in the range is screened, the middle is played
  ④ TALK, before Haiku (Cristi)  opponent chat marked untrusted (injection detector), lines from earlier
  │                              games of the match labelled, the opponent's habits and claims-vs-deeds
  ⑤ HAIKU, one call           observation + the engine's near-best moves + chat
  │                           → JSON {reason, secrets, action, decoy, chat}: move and chat from ONE decision (6.4)
  ⑥ TALK, after Haiku (Cristi)   move: only a near-best candidate / a legal one, budget repaired, no repeated
  │                              guess; chat: no plan given away, no forbidden word (list + a model referee),
  │                              none of our secrets, nothing the opponent ordered, ≤ 200 chars, not a repeat
  ⑦ BRAIN + SHELL (Denis)     format repair, spelling gate, length guard, no-repeat guard, final legal gate
  │
ARENA ◄── {"type":"action", "action": "...", "chat": "..."}
```

**Who picks the move.** With an engine, Haiku may only choose among moves the engine rates near best (within 0.05;
in simultaneous games the draw is made in code and Haiku may swap it only for a concrete read of the opponent), so
the chat can never throw the game. Haiku plays alone (`talk.play`, legality checked) when there is no simulator
(a real talking game, or simgen failed), when the per-variant learning switched to Haiku, when the search can't tell
the moves apart, or when it produced no candidates. The filters of step ⑥ run either way.

**Model calls per turn:** board games 1 (+ the propose step in huge action spaces); talking games 2 (the move-and-chat
call and one speech check that covers both). The brain's move referee is skipped in talking games.

## 5. Safety net (robustness is 2.5 points; a second illegal answer loses the game)

- The shell sends the best answer known so far at the deadline; a legal fallback exists before any model call.
- The final gate never sends a move outside `legal_actions` or one the arena already rejected in this game.
- After a `warning`, the engine is switched off for that game.
- Empty, slow or failing model answers fall back to the engine's top move, then to the rules' own examples.
- Chat: every line goes through the filters in [LANGUAGE.md](LANGUAGE.md); with no time to check a line in a
  speech-rule game, a neutral line goes instead.

## 6. Memory (`/memory`, shared by all our containers at the same time)

```
/memory/<game_id>/<rules-hash16>/   digest.json, sim.py, report.json, talk_notes.json, prep_msg.json
/memory/<game_id>/results.jsonl     one line per game: variant, result, mode (engine / model / engine_off)
/memory/<game_id>/opponents/<opp>.moves.json   counts of the opponent's moves per variant (simultaneous games)
/memory/opponents/<opp>.json        chat profile: injections, requests for private info, probe followed
```
Writes are atomic (temp file + `os.replace`). A cold container or a later tournament with the same rules reloads it.

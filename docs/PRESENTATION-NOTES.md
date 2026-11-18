# PRESENTATION-NOTES.md — technical material for the presentation (path-to-glory-3, 15 Nov)

For Cristi (presenter) and the team. Every number below comes from a file or a run in this repo; the source is named.
Judging (docs/RULES.md §7, rulebook 12.1-12.2): presentation 30 (clarity, **live demo**, what we learned), harness design 25
(rules understanding, state, move verification, tools/computation, memory, budget/time, chat). **+5 bonus: a game concept
that is not on the platform** -> section 6 (Echoes, playable, our bot already plays it).

**Final T3 image:** `t3` = `t3/library` **ff03be3** (B + 13 s turn cap + tempo filter), digest `sha256:f20944a8...`, tag `t3-freeze`
(Denis, PROGRESS 11:10). Gates: check_bot --image 23/23, slowest turn 12.0 s, prepush_check 16/16.

## 1. The harness in one picture (mesh/, one Docker image, Python)

```
 prep (600 s, once per game)                            every turn (20 s clock, we plan <= 13 s)
 ───────────────────────────                            ─────────────────────────────────────────
 rules ─► digest (Haiku: format, examples, flags)       observation ─► parse into states (sim.state_from_observation)
      ─► simgen: Haiku WRITES a Python simulator              │        hidden parts sampled: determinization pool
          (to_move, legal_actions, apply, winner,             ▼
           observation, state_from_observation)         search in a sandboxed PyPy worker (2 in parallel, 2 CPUs)
      ─► validate in a sandbox (self-play, round trip,     - immediate win / tactics: never allow an instant loss
          replay the real sample turns) ─► repair loop     - tempo: never hand the opponent an extra move
      ─► evaluation function (Haiku) + duel vs playouts    - open-loop MCTS (UCT), or alpha-beta with the evaluation
      ─► opening book in idle prep time                    - regret matching for simultaneous moves + opponent model
      ─► /memory/<game>/<rules hash>/                    ─► Haiku picks among near-best moves AND writes the chat
                                                            (one decision) ─► gates: legality, feedback, chat guard
                                                            ─► watchdog: best-so-far always sent in time
```

## 2. Technical points worth saying (with numbers)

| Topic (design criterion) | What we built | Number / proof |
|---|---|---|
| Understanding the rules | Haiku writes a full simulator from the rules text, then code checks it against the arena's own sample turns and, for public games, against real replays | Dots: "reproduces them (306 moves, 9/9 endings)"; PD 156 moves 9/9 (sweep logs 15 Nov) |
| Move verification | simulator legal moves must equal the arena's list; else that game falls back to Haiku with a legality check. Final gates: rejected move never resent, feedback consistency, "exactly N" counts | check_bot 23/23 on every candidate image |
| State tracking | observation -> full state each turn; hidden info by **determinization** (sample hidden states consistent with what we saw) | 32-48 sampled worlds per turn (simworker.candidates) |
| Computation | generated code runs in a **separate process**, memory capped (900 MB), killed if it hangs; **PyPy** for the search | PyPy ~3.6x faster than CPython on generated sims (our bench, 15 Nov night) |
| Search | open-loop MCTS; tactics filter (2-3 ply: no instant loss, forced wins); **tempo filter** (no gifted extra move); alpha-beta when prep's evaluation wins its duel | Dots offline: gifted boxes **10/40 -> 0/40** positions with the tempo filter (rares/t3-tempo) |
| Learning in prep ("fix b") | Haiku writes `evaluate(state, seat)`; adopted only if it agrees with real winners and **beats the playout search in a duel** | Dots prep log: `evaluation duel [eval]: 11.0 - 1.0 playout search` |
| Simultaneous games | regret matching on a payoff matrix + opponent model from public replays (rr1 + T2 habits) | `opponent model (level 2, weight 0.62): defect` (RC 068faf0 log) |
| Chat (offense + defense) | opponent chat reaches the decision as **untrusted data**; our chat never names our own move (guard after T2) | RC run: 16 leaking lines caught and replaced, 0 leaks in the audit |
| Budget | one small Haiku call per turn, capped; prep spend separate | **$0.002-0.008 per match** in T3 tests (T1: $0.021) |
| Time | watchdog answers by 17 s with the best move so far; planning capped at 13 s | slowest turn 16.0 s -> **12.0 s** after the cap (check_bot m2, Denis 10:55) |
| Memory | simulator, report, book, evaluation saved per game and rules hash in /memory; reused by cold containers | /memory/<game>/<hash>/ (DESIGN.md) |
| Robustness | never a blank or late answer by design; 0 warnings and 0 timeouts in T1 | T1 replays (DESIGN.md "T2 plan v2") |

## 3. Testing and analysis (Rares's part: tests/, 116 commits)

- **Zero-spend testing:** `tests/fake_llm.py` (fake gateway) + `tests/check_bot.py` (23 checks from rulebook §14.7: protocol,
  timeouts, EOF, cold start, chat length, injection...). Every image we pushed passed it.
- **Generalization:** `tests/holdout/` = 12 games we invented ONLY to test unseen-game behaviour (hidden codes, bids, taboo words,
  ultimatum, chains...). Rule: never tune mesh/ to them.
- **Realistic opponents instead of random ones:** `t2_ref_bot` (rule-based), `nasty_bot` (prompt injection in chat),
  **`rr_top_bot`: plays like the top 4 teams of the round robin**, built from 2253 of their public moves (exact positions +
  their PD / poker / Blotto choices + solid rules). 0 errors on 7474 real observations.
- **Real games locally:** `tests/realgames/`: the 8 T2 games rebuilt from 91 public replays, so arena-lite runs the real rules.
- **`tests/audit_replays.py`: post-mortem in seconds**, only provable mistakes, from public replays. rr1 + T2 knockout, us vs nyx (#1):

| game | ours | nyx |
|---|---|---|
| Battleship: fired away from an unsunk hit | 30 | 0 |
| Dots and Boxes: gave a box while safe lines were open | 22 | 6 |
| Connect Four: missed a block | 3 | 0 |
| RPS: chat named a sign, opponent beat exactly that sign | 10 | 7 |

- **A/B decisions with numbers:** candidate A (generic) vs B (A + replay-checked library) on our weak games: B 6 wins, A 4; then
  B vs rr_top_bot 8 W 3 D 2 L on weak games, 0 warnings -> the team chose B.

## 4. T1 — 2nd of 15: what worked
- 0 warnings, 0 timeouts, 0 rule breaks: the shell + watchdog design.
- Simulator + search on board games: Connect Four 4/4, poker 3 W 1 D.
- Lost only the final to neura: Wordle 0/2 and valued Blotto 1-4, the two games where **Haiku decided alone** (repeated a word,
  ignored feedback; 43 of 60 soldiers on a 1-point field). Lesson: the model must never decide without code checking it.

## 5. T2 — 7th of 14 in the round robin, out in round 1 (0-2 vs ultraviolence): what went wrong
- **RPS 0-4: our chat named our sign every throw** ("leaning hard on paper" -> paper) and the opponent countered every one.
- **Bulls 2-4:** `guess BANK` (letters outside A-F) -> warning; then guesses our own feedback ruled out (first-turn bug:
  the simulator process was not started yet, so the feedback filter was skipped).
- Generic causes measured: random playouts too noisy on new board games, huge action spaces handed to Haiku (Battleship),
  equilibrium mixes instead of exploiting habits. Knockout = one bad game ends the run: robustness >= strength.
- **What we learned (say this):** test against strong opponents, not random ones; chat is an attack surface; verify every
  model output with code; measure with replays, not impressions.

## 6. +5 bonus: our own game concept — ECHOES (`tests/holdout/x_echoes.py`, playable in arena-lite)
Two hidden beacons on a grid. Each round **both players act at once**: `ping <cell>` (hear the Manhattan distance from that
cell to the enemy beacon) or `drift <north|south|east|west>` (move your own beacon one step, at most 3-4 times, no echo that
round). **Drifts resolve before pings**, so a well-timed drift dodges a ping that would have found you. First exact ping wins;
after the round cap the closest echo wins. Variants: 5x5 exact echoes; 7x7 **noisy echoes** (distance off by one).
Why it is a good harness game: deduction (keep the cells consistent with every echo), simultaneous mind games (ping or flee
when they get close?), hidden information you must protect in chat (bluff your position, never reveal it), and a 7x7 action
space above 50 (no legal list) -> the harness must build moves from the format. Demo: `python3 tests/sweep.py --bot <image>
--only x_echoes`: our bot has never seen it, writes its simulator in prep and plays.

**Live result (15 Nov 11:00, tests/out/sweep-20261115-110023, B 068faf0 vs rr_top_bot, best of 3): WON 2-0, 0 warnings, $0.003.**
- Prep, on a game it had never seen: Haiku wrote the simulator in **3 tries + 1 repair**, validation **196/196** state round trips,
  **980/980** legality checks, 2/2 sample turns; ready after 236 s of 600. Turns 11.1-11.6 s.
- Game 1 (5x5): found the beacon in round 8, and **drifted twice** (rounds 6-7) when the opponent's pings got close: defence.
- Game 2 (7x7, no legal list): the rule-based opponent did not cope with the missing list (`pass` twice = rule break); ours did.
- **Honest lesson (say it, judges like it):** after round 2 (`ping a4` -> 1, `ping e5` -> 6) only **a3** fits both echoes, but the
  bot pinged c3, b1, c5 first. The bot's log shows why: the generated simulator DID filter the hidden beacon by our echoes, but
  the game is simultaneous, so `ping a3` is not a sure win (the opponent may drift that round) and the win-now check correctly
  stays off. Regret matching then runs on random playouts, which almost never find a beacon in 14 rounds: all values ~0.04, so
  the mix is close to noise (`ping a3` had p = 0.09 in round 7). Fix idea: in deduction games, score moves by information
  gained (shrinking the set of consistent worlds), not by random playouts.
- **Demo page:** `docs/echoes-demo-offline.html` (open in any browser, works offline): round-by-round replay of this game with
  the set of possible cells for both sides, plus a "play" tab where a judge plays against a small deducing bot (no LLM).

## 7. Live demo options (safe, fast)
1. `python3 tests/audit_replays.py tests/replays tests/out/...`: the mistake table above in ~5 s (zero spend).
2. `python3 tests/check_bot.py --image <t3 image>`: 23 protocol checks, PASS lines scrolling (zero spend, ~4 min: start it early).
3. Echoes: show `tests/holdout/x_echoes.py` rules + the match log where our bot writes the simulator and plays.

# path-to-glory: a game-agnostic LLM harness

Team **path-to-glory**'s entry in the *AI Harness Arena* (EESTEC Olympics, Bucharest, 14-15 Nov 2026).
The bot receives the rules of a two-player text game it has never seen and gets 10 minutes to prepare.
It then plays other teams' bots with 20 s per move and a mandatory chat line, using only a small LLM (Claude Haiku).

Results: **2nd of 15 in T1**. T2 and T3 are analysed honestly in [docs/T2-ANALYSIS.md](docs/T2-ANALYSIS.md) and [docs/T3-REVIEW.md](docs/T3-REVIEW.md).

## How it works

Prep runs once per game, within 600 s:

1. The LLM digests the rules.
2. It **writes a Python simulator** of the game.
3. The simulator is validated in a sandbox: self-play, state round trips, replay of sample turns.
4. A repair loop fixes what failed validation.
5. An evaluation function is adopted only if it beats playout search in a duel.
6. The bot builds an opening book.
7. Everything is cached in `/memory`.

Each turn runs within 20 s:

1. The observation is parsed into states, and hidden information is sampled into a determinization pool.
2. A PyPy worker runs the search: open-loop MCTS, or alpha-beta when the evaluation function was adopted. Simultaneous games use regret matching with an opponent model.
3. Tactics and tempo filters run first.
4. The LLM picks among near-best moves and writes the chat, in one decision.
5. The move passes legality, feedback and chat-leak gates.
6. A watchdog always answers in time.

Typical cost is $0.002-0.008 per match. Details are in [docs/DESIGN.md](docs/DESIGN.md), [docs/FLOW.md](docs/FLOW.md) and [docs/PRESENTATION-NOTES.md](docs/PRESENTATION-NOTES.md).

## Layout

| Path | What |
|---|---|
| `mesh/` | the harness, which is the Docker image (`docker build -f mesh/Dockerfile .`) |
| `tests/` | the zero-spend test kit: a fake LLM gateway, a 23-check protocol checker, holdout games, realistic opponents, a replay audit and sweeps |
| `tests/holdout/x_echoes.py` | **Echoes**, our own game concept: hunt a hidden beacon by its echo while yours drifts away. Demo: [docs/echoes-demo-offline.html](docs/echoes-demo-offline.html) |
| `tools/` | pre-push gate, simulator checks, engine duels |

## Not included

These belong to the organizers or to other teams, so they are left out:

- the organizers' participant kit: the `arena-lite` match engine, the starter bot, protocol docs and rulebook;
- the official game rule texts;
- public match replays and anything derived from them (opponent profiles, the replay-trained data in `mesh/experience/`, the top-teams opening book);
- the practice kit's word list.

The harness runs without them, because it falls back to its generic path. The local test harnesses (`tests/sweep.py`, `tests/cost_match.py`, `tools/prepush_check.py`) expect the organizers' `arena-lite` in `./arena-lite`.

## Team

- Denis (harness core, `mesh/`)
- Cristi (chat and strategy, `mesh/talk.py`, team test games)
- Rares (testing, opponents and analysis: `tests/`, `tools/`, Echoes)

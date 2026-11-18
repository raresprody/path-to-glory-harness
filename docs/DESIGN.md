# DESIGN.md — the agreed harness design (T1)

> **As implemented (t2):** the end-to-end flow (arena → shell → brain → engine → talk → Haiku → filters) is in
> [FLOW.md](FLOW.md); the language side in detail is in [LANGUAGE.md](LANGUAGE.md).

Agreed by Denis on Sat 14 Nov with his Claude. Rulings that came with it are in [DECISIONS.md](DECISIONS.md) (D12–D18).
Three parts, one per teammate, built in parallel against stubs and joined by small interfaces (below).
Split proposed by Denis's coordinator on 14 Nov, adjusted to the claims already in PROGRESS.md (path-to-glory-3 took the test kit):

| Part | Owner | Files |
|---|---|---|
| **Shell + engine**: protocol loop, watchdog, fallback move, model client (latest key), Dockerfile, rules digest, prep pipeline, simulator/renderer/parser generation and validation, search (perfect info, hidden-info sampling, simultaneous mixed strategies), /memory layout, `tools/prepush.sh` | path-to-glory-1 (Denis) | `mesh/main.py`, `mesh/brain.py`, `mesh/llm.py`, `mesh/digest.py`, `mesh/sim*.py`, `mesh/search.py`, `mesh/Dockerfile`, `tools/` |
| **Language side**: per-turn Haiku decision picking among candidates + writing chat (offense, defense, chat untrusted), the plain Haiku-per-move player, Truth / Guardrails / negotiation handling, the forbidden-subject output filter, extra arena-lite test games (bluff, negotiation, simultaneous; rule 14.5) | path-to-glory-2 | `mesh/talk.py`, `mesh/digest.py` (filter part: `violates`, `guard_chat`), `arena-lite/arena/games/` |
| **Test kit**: fake gateway (zero spend, slow/error/empty modes, checks opponent chat reaches the model), scripted §14.7 protocol checker, adversarial opponent bots | path-to-glory-3 | `tests/` |

So `t1` is never blocked, path-to-glory-1 writes a **first simple version of `mesh/talk.py`** now; path-to-glory-2 owns it
from then on and improves it behind the same functions.

## The seven agreed points

1. **A shell that never loses a game by accident** (robustness, 9.4). Protocol loop per rulebook §5; flushed JSON lines on
   stdout, logs on stderr, nothing printed at startup; survives `match_over`; exits only on `shutdown`/EOF; always uses the
   `llm` from the latest `prep`/`start`. A **watchdog** sends the best move found so far well before each clock ends, with a
   legal fallback, so a blank or late answer is impossible by design. Thinking disabled on model calls, a timeout on every
   call, empty answers handled (7.3).
2. **Prep (600 s):** Haiku writes a Python simulator from the rules (state, legal moves, apply, result) plus a renderer
   (state → the arena's observation text) and a parser (text → state). Validate: random self-play never crashes and always
   terminates; render→parse round-trips; compare with the real `sample_turn` when present (optional, may lack history, 5.6).
   Write/test/fix loop within the clock, `ready` early when done, save to `/memory/<game_id>/<rules hash>/` (5.14). One
   simulator covers all variants. (Haiku answers in ~2 s through the gateway, so many fix rounds fit.)
3. **Match:** code search proposes moves; **one small Haiku call per turn** sees the top candidates plus the opponent's
   chat (marked untrusted) and returns **move and chat as one decision** (6.3–6.7). Haiku may only choose among moves the
   code rates near best, so chat cannot throw the game. Chat ≤200 chars on every start `ready` and every action.
4. **Per-turn safety net:** compare the simulator's legal moves with the arena's `legal_actions`; on mismatch or parser
   failure, that game switches to a plain **Haiku-per-move player** with legality checking.
5. **Hidden info, bluffing, simultaneous moves, negotiation are core**, not edge cases: sample hidden states consistent with
   the observation and average over playouts; mixed strategies for simultaneous moves plus per-opponent pattern tracking in
   /memory; for negotiation code values offers and Haiku talks within those limits. **Truth and Guardrails** families are
   language tasks: straight to Haiku with strict prompts ("unknown" rather than invent), and **every output, chat included,
   is filtered for the forbidden subject** (6.8).
6. **Hardware:** 2 CPUs, 2 GB, 256 processes, several specialists alive at once. Generated code runs in a **separate
   process with time and memory limits**; search capped at about half the move clock; idle specialists hold almost
   nothing; no local models.
7. **Ship early:** a pushable, rule-compliant image (shell + Haiku-per-move player) goes to `t1` as soon as it passes
   §14.7; then keep improving and re-pushing. Denis accepts the risk of shipping the simulator in T1.

Constraints: only gateway `haiku` (7.1); game-agnostic (game-specific code costs design points, 10.7); dev key $6 shared,
60 req/min: test frugally; prefer local `arena-lite` over the practice page (10 runs/h for the team); build linux/amd64.

## Interface (shell ↔ brain)

Code: [`mesh/api.py`](../mesh/api.py). The shell (`mesh/main.py`) creates **one** `Brain` (`from mesh.brain import Brain`).

| Shell receives | Shell does | Brain call (runs in a worker thread) | Shell sends |
|---|---|---|---|
| `prep` | `llm.configure(msg)` | `brain.on_prep(msg, deadline)` with deadline = now + `seconds` − 30 | `{"type":"ready"}` when it returns, or at the deadline |
| `start` | `llm.configure(msg)` | `brain.on_start(msg, deadline, best)`, deadline = now + `study_seconds` − 4 | `{"type":"ready","chat": best chat}` when `best.done` or at the deadline |
| `turn` | — | `brain.on_turn(msg, deadline, best)`, deadline = now + `seconds` − 3 | `{"type":"action","action":…,"chat":…}` when `best.done` or at the deadline |
| `end` | — | `brain.on_end(msg)` (quick) | nothing |
| `shutdown` / EOF | — | `brain.on_shutdown(deadline)`, 10 s | nothing, exit 0 |

- `deadline` is absolute `time.monotonic()`. The brain must return by then; the shell must not wait longer than that anyway.
- **`Best`** (in `mesh/api.py`) is the "best answer so far" holder: the brain calls `best.set(action, chat)` early and often,
  and `best.set(..., final=True)` when done. The shell's watchdog sends `best.get()` at the deadline.
- **Fallback owner: the shell.** If `best` has no action at send time: a random entry of `legal_actions`; if that is `null`,
  the brain's `brain.fallback_action(msg)` (built from the rules digest's example actions), wrapped in try/except, else `"pass"`.
- **Model client owner: the brain** (`mesh/llm.py`, stdlib urllib, honours the proxy env). The shell only calls
  `mesh.llm.configure(msg)` on `prep` and `start`. The brain never writes to stdout; all logs go to stderr.
- **Chat:** the brain returns chat already filtered for speech rules; the shell only truncates to 200 chars and sends a
  non-empty string (default `"gl hf"`) if the brain gave none.
- A late brain call is abandoned, not killed: the shell ignores whatever it writes to `best` after sending.
- **Stubs:** a trivial brain (random legal move, fixed chat) is enough to test the shell; `mesh/main.py` is a stub shell for the brain.

## /memory layout (owned by the brain)

```
/memory/<game_id>/<rules-hash16>/digest.json   rules digest (action format, speech rules, playbook)
/memory/<game_id>/<rules-hash16>/sim.py        generated simulator + renderer + parser (validated)
/memory/<game_id>/<rules-hash16>/report.json   validation results
/memory/<game_id>/opponents/<opponent>.json    per-opponent stats (moves, chat style, results)
/memory/<game_id>/results.jsonl               one line per finished game
```
`/memory` is shared by all our containers at once: write to a temp file and `os.replace` it.

## Interface: engine ↔ language side (`mesh/talk.py`)

The brain calls these; each must return before `ctx["deadline"]` (monotonic) and never raise (wrap everything).
All of them receive the opponent's chat and must pass it to Haiku as **untrusted data** (rules 6.5–6.7).

```python
ctx = {
  "rules": str, "variant": dict, "seat": int, "you": str, "opponent": str,
  "digest": dict,            # mesh/digest.py: action_format, example_actions, forbidden_terms, speech_rules, flags, playbook
  "observation": str, "legal_actions": list[str] | None, "warning": dict | None,
  "chat": list[dict],        # the full match chat from the arena message
  "candidates": list[dict] | None,   # [{"action": str, "value": float, "note": str}] best first, all legal; None = no engine
  "opponent_profile": dict,  # from /memory/<game>/opponents/<opp>.json (may be {})
  "deadline": float,
}
talk.decide(ctx)      -> (action: str, chat: str)  # pick ONE of ctx["candidates"] (only near-best ones) + chat line
talk.play(ctx)        -> (action: str, chat: str)  # no engine: Haiku-per-move with legality checking
talk.ready_chat(ctx)  -> str                       # chat for the `ready` of a `start`
```
- `action` must be one of `legal_actions` when that list is given; the brain re-checks and falls back to the top candidate.
- `chat` must already be filtered with `digest.guard_chat` (≤200 chars, no forbidden term). In talk-is-move games the
  action is filtered too.
- Stub for testing: `decide` returns `(candidates[0]["action"], "gl hf")`, `play` a random legal action.

## How the engine is built (as implemented, 14 Nov 13:45)

| Step | Where | What |
|---|---|---|
| Rules digest | `mesh/digest.py` | one Haiku call: action format + examples, forbidden terms, flags (simultaneous, hidden info, talk-is-move), playbook; plus code-side quoted examples from the rules (fallback when the model is down) |
| Simulator generation | `mesh/simgen.py` | Haiku writes a module (`CONFIGS`, `initial_state`, `to_move`, `legal_actions`, `apply`, `is_over`, `winner`, `observation`, `state_from_observation`) from all variants' rules + sample observations; fix loop with test failures until it passes or the clock runs out. Skipped for talk-is-move games |
| Validation | `mesh/simworker.py validate` | random self-play (no crash, always terminates, winner valid), render→parse round trip keeps legal moves (≥95 %), parsing each real `sample_turn` gives exactly the arena's legal moves |
| Sandbox | `mesh/simworker.py` | generated code only runs in a child process (stdout isolated, 900 MB address-space cap on Linux, killed if it hangs) |
| Search | `simworker.candidates` | parse the observation into full states with hidden parts sampled (determinization pool of 48), immediate-win check, then open-loop MCTS (UCT) with the hidden state re-sampled every iteration, decoupled per-seat selection when both move (mixed strategies), random playouts; early stop when one move dominates; capped at 5 s (flat Monte Carlo kept as `search: flat`) |
| Safety net | `brain.on_turn` | simulator's legal moves must equal the arena's `legal_actions` (or the parse must succeed when the list is null); otherwise the engine is switched off for that game and Haiku plays |
| Decision | `talk.decide` / `talk.play` | Haiku picks among moves within 0.05 of the best value and writes the chat; when the action space is too big to search (legal list null, >60 moves) Haiku chooses freely with an engine hint (moves that can win now in a consistent world) and the simulator checks legality |
| Cold container | `brain.on_start` | with a ≥90 s study clock and no saved simulator, the same pipeline runs in the study time |

## T2 plan v2 (after T1 = 2nd of 15; Denis, 14 Nov 20:00; freeze Sat 23:59)

T1 lessons (replays `/api/matches/t1-*`): 0 warnings/timeouts; Connect Four 4/4 and poker 3W 1D (engine works).
Lost: **Wordle Duel 0/2** (Haiku guessed alone: repeated `castle`, ignored feedback) and **valued Blotto 1-4 vs neura**
(Haiku alone in a huge action space: 43 of 60 soldiers on a 1-point field). Also: 2-4x slower than opponents (thinking
time is tie-break #2), chat leaked between games of a match, opening chat cut at 200 chars by the probe sentence.
T1's 4 games are **4 of the 12 round-robin games** (10 pts), so they matter again tonight.

**Rule: each file has one owner tonight. Need a change in someone else's file? Write it in PROGRESS and let them do it.**

| Owner | Files | Work (in this order) |
|---|---|---|
| **path-to-glory-1** (Denis) | `mesh/brain.py`, `simgen.py`, `simworker.py`, `llm.py`, `main.py`, `digest.py` (non-filter), `mesh/Dockerfile`, `tools/` | 1. **Guessing games** (hidden target, null legal list): simulator gets an optional `could_win_now(state, seat, action)` (written in prep, validated: the true answer must pass); Haiku proposes ~20 candidate answers, the engine keeps only those still consistent with every feedback and passes them as `candidates` (never a repeat). 2. **Huge action spaces** (Blotto): `propose` step in brain: Haiku proposes ~12 diverse legal moves (+ counters to the opponent's last visible moves); regret matching ranks them against the opponent's likely moves; code draws. 3. **Speed**: search budget scales with the number of moves (2-3 moves → ~2 s). 4. Merge `cristi/chat-quality` into `denis/t2-crosscheck` (without Cristi's `learn_moves` hook in `brain.on_end`). 5. Build candidates, final merge to `main`, ask Denis, push `t2`. |
| **path-to-glory-2** (Cristi) | `mesh/talk.py`, filter part of `digest.py`, `arena-lite/arena/games/` | (a) clean "your guesses + feedback" block and no repeated guess in talk (uses `ctx["candidates"]` from the engine when present); (b) chat lines only about the current game; (c) opponent claims vs what they actually did (neura told the truth); (d) read the engine's `/memory/<game>/opponents/<opp>.moves.json` instead of a second store; (e) **opening chat + probe ≤ 200 chars before sending** (it was cut mid-sentence). |
| **path-to-glory-3** (Rares) | `tests/` | (1) **local copies of T1's 4 games** from the replay rules (or `import_kit` when the organizers publish them), as test games; (2) opponent bots that copy what beat us: neura-style Blotto (3 fields × 20) and a solver-style Wordle player; (3) sweep `denis/t2-crosscheck` on them when path-to-glory-1 says it's ready (~21:30); (4) final gates on the t2 candidate (~22:30). |

| Time | Milestone |
|---|---|
| 20:15 | **Safety net**: `t2` in Harbor = the current `t1` image (only with Denis's go), so T2 + round robin never run without an image |
| 21:30 | path-to-glory-1: guessing + propose steps done on `denis/t2-crosscheck`; Rares' T1 game copies ready |
| 21:30–22:15 | Rares sweeps the branch on all games incl. T1's 4; Cristi merges his fixes on `cristi/chat-quality` |
| 22:15 | path-to-glory-1 merges both branches into `main` → t2 candidate |
| 22:15–23:15 | final prepush_check + check_bot + full sweep (one dev-key run at a time) |
| 23:30 | Denis decides; push `t2`, tag `t2-freeze`. If the candidate isn't clean by 23:30, the last clean image stays `t2`. |

Not doing: web search (7.11), local models, agent loops. Dev key: one test run at a time (RUNNING line in PROGRESS).

# tests/ — test kit (owner: path-to-glory-3)

Everything here runs locally, uses Python's standard library only (plus arena-lite's engine), and
**spends nothing**: no gateway calls except where noted. Nothing here goes into the image.

| File | What |
|---|---|
| `check_bot.py` | Protocol checker for the RULES.md §14.7 items. Drives a bot (command or Docker image) through prep and four matches against a fake gateway and prints PASS/FAIL. Exit code 1 on any failure, so `tools/prepush.sh` can call it. |
| `fake_llm.py` | Fake OpenAI-compatible gateway. Behaviour chosen by the key: `fake-ok-*`, `fake-slow-*` (45 s), `fake-err-*` (HTTP 402), `fake-empty-*`, `fake-cut-*`. Logs every request, so we can see which key was used and whether the opponent's chat reached the model. |
| `opponents/nasty_bot.py` | Practice opponent for arena-lite: random legal moves, hostile chat (`injector`, `liar`, `spammer`, `trickster`, `silent`, `mix`). Logs `!!! INJECTION FOLLOWED` to its stderr when our bot obeys it. |
| `opponents/t1_style_bot.py` | Opponent that plays like the teams that beat us in T1 (no model): Colonel Blotto all-in on the top half of the fields by value, **announced truthfully in chat** (neura); Wordle Duel keeps only words consistent with its feedback, never repeats (alt/neura). Other games: random legal move. Use: `sweep.py --only colonel_blotto,wordle_duel,x_codebreak --opponent "python3 ../../../opponents/t1_style_bot.py"`. |
| `stress.py` | Several of our specialists at once (one arena-lite `--prep` match per game, in parallel, arena limits 2 CPU / 2 GB). Samples `docker stats`, listens for OOM kills, reports our warnings/crashes/timeouts per match. Uses the dev key (a few cents). |
| `cost_match.py` | Real prep and match cost of one match from the dev key's `/key/info` (cost score, 9.3). Approximate: the gateway books spend late. Don't run while teammates use the key. |
| `match_report.py` | **What our bot did, per game** (zero spend, after any sweep/prep run): result, our warnings, turns and answer time (median/max), model spend per game, and chat: lines, length, repeats, reacts to the opponent's line, leaks of our private values, injection followed, lines about another game; then sample lines (theirs -> ours). Compares with T1 ($0.021/match, 10.3 s/turn). |
| `sweep.py` | **Rule-break sweep**: one `--prep` match per game for every game we have (practice, team test games, holdout) vs `nasty_bot`. Fails on any warning, illegal/invalid move, timeout, crash, bad ready or speech-rule loss of ours. Also shows whether the prep simulator validated and prep time. `--selftest` checks the holdout games alone (no spend). |
| `holdout/` | **Holdout games** (`x_piles`, `x_codebreak`, `x_bids`, `x_chain`, and for the 3 families T1 never played: `x_pact` Deals: trust and betrayal with a joint target, `x_dossier` Truth: multiple choice from a file with a planted instruction, `x_taboo` Guardrails: everyday answers around a forbidden subject checked with a hidden wider word list; `x_ultimatum` Deals: offer/accept roles; `x_dice` Bluff: Liar's Dice, null list in the big variant; `x_yesno` Guardrails: never say yes/no, plus a start-letter form rule; `x_fleet` Guessing: battleship on a grid): new mechanics mesh/ has never seen (two-argument moves, null legal lists, simultaneous bids, hidden codes, banned words in chat). **Don't tune mesh/ to them**: they only measure generalization if nobody fits them. `sweep.py` copies them into a temp arena-lite; `arena-lite/` is never changed. |
| `prep_report.py` | Prep phase only, per game: seconds to `ready` of 600, rules digest, simulator accepted, opening book, study, drill, prep spend. |
| `sim_vs_engine.py` | Zero spend: compares the simulators mesh generated in a `prep_report` run with the **real** engines (legal moves at every decision, winners in open games). Answers "did Haiku understand the rules?". |
| `spend_report.py` | Zero spend: prep and match spend per game from mesh's own log lines. |
| `import_kit.py` | **After T1/T2:** pulls the organizers' kit, lists new games and changelog lines; `--apply` copies it into `arena-lite/` and `docs/kit/` (keeps our test games); `--test <image>` then runs prep_report → sim_vs_engine → sweep on the new games only. |
| `out/` | checker output (gitignored) |

## Run (from the repo root)

```bash
# protocol checker: ~3 min (add --quick to skip the slow-model match, ~30 s)
python tests/check_bot.py --cmd "python mesh/main.py"
python tests/check_bot.py --image harbor.hackaton-2026.dvloper.io/path-to-glory/bot:t1

# chat defence: a real match against the injector (uses the dev key for OUR bot only)
cd arena-lite
python -m arena practice --bot-a "python ../mesh/main.py" --bot-b "python ../tests/opponents/nasty_bot.py injector" \
    --env-file ../dev.env --game tic_tac_toe --id nasty
grep '!!!' results/practice/logs/nasty-bot-b.log   # empty = no injection obeyed
```

## What the checker plays

| Step | Model (fake key) | What it proves |
|---|---|---|
| boot | — | silent on stdout, still alive, waiting for the first message |
| prep | `fake-ok-prep` | `ready` within `seconds` |
| m1 | `fake-ok-m1` | full Tic-Tac-Toe 3x3 game; turn 3 is re-sent with a `warning` |
| m2 | `fake-slow-m2` | every move inside the 20 s clock although each model call hangs 45 s |
| m3 | `fake-err-m3` | model says 402 (budget gone) **and** `legal_actions` is null: the move must come from the rules' format |
| m4 | `fake-empty-m4` | model returns empty text |
| after each match | — | `match_over` does not end the process; nothing printed after `end` |
| keys | — | each match only uses the key from its own `start`, never the env key |
| chat | — | a chat string (≤200) on every answer; a random canary word in the opponent's chat shows up in that match's model calls |
| end | — | exits within 15 s of `shutdown`; a second (cold) process exits within 15 s of EOF |

## Baseline: the organizers' starter (`starter/main.py`), 2026-11-14

19/23. The 4 failures are real problems a harness must avoid:
- **slow model → lost game:** it answered after **49 s** on a 20 s clock. Its call timeout is 16 s, but the
  `openai` SDK **retries twice by default** (`max_retries=2`), so one slow call becomes 3 × 16 s. A watchdog that
  sends the best-so-far move at the deadline, independent of the model call, is the fix (DESIGN.md point 1).
  If anyone uses the `openai` SDK: `max_retries=0`.
- with `legal_actions` null and the model failing, it sends an empty action (warning, then a lost game, 8.1);
- no chat when the model's reply has no `CHAT:` line, and none with an empty model answer (6.2).
It passes the injector match (no injection followed) with the real model.

## Measurements (t1-freeze 24e0f49, Docker, 2026-11-14 ~14:30)

- **Stress, 4 specialists in parallel** (tic_tac_toe, pig, guess_number, hush): all 2-0, none of our warnings,
  crashes or timeouts. Memory per container **31 MiB idle, ≤50 MiB peak** of 2048. CPU peak **100% = one of the two
  CPUs**: the search uses a single core.
- **Cost:** match spend ≈ **$0.001–0.008 per match** (≤1.1% of the $0.75 key); prep $0.001–0.007. The random bot is a
  weak test opponent where `legal_actions` is null (it sends `pass` and loses after 2 warnings), so those matches are short.

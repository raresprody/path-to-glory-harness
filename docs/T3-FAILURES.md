# T3: every way we lost a game on 14 Nov, and the robust fix for each

Owner of this file: path-to-glory-3 (tests). Sources: the local sweeps of 14 Nov (tests/out/sweep-*, every image from
915dc91 to 6d4123d), `tests/match_report.py`, the warm logs. Every item below was seen in a log, not guessed.

**Why T3 is different:** all 21 games, including 4 that nobody has seen; best of 3 with **sudden death** (one slip loses
the match); warnings and thinking time are the tie-breaks. So the goal is not "win more often" but **"never lose a game
to our own mistake"**, then "win the deduction and simultaneous games we still lose".

The common thread: **we trusted one weak signal (one Haiku answer, one generated helper) where we could have checked
it against something objective (the arena's own observations, the simulator, a grammar of the moves).** Each fix below
replaces trust with a check.

## 1. Simulator generation and trust (Denis: simgen / simworker)

| What we saw | Cause | Fix status |
|---|---|---|
| Blotto, Wordle: 8 tries, same score, ~380 s of prep idle | crash line never reached the repair prompt (`sim_tryN.py` filtered out) | fixed (1151aa6) |
| Blotto: self-play crashed on every game | `legal_actions` returns None for huge spaces, `rng.choice(None)` | fixed (random_action, representative list) |
| Wordle, Blotto: a validated sim (score 1.0, 3/3 real arena samples) dropped | the arbiter is Haiku and gets Wordle feedback wrong; "6/6 against" was trusted over real samples | rule removed (072d928) |
| Wordle tagged a "talking game", no sim at all | `talk_is_move` is one Haiku answer, not stable across runs | fixed (472ee6c) |

**Robust fix for T3: a fixed order of trust, and checking the sim during play.**
- Order of trust: **real arena observations > the simulator > Haiku's opinion.** A sim that reproduces every real
  arena sample is never dropped on Haiku's word; a Haiku answer that contradicts a real sample is ignored.
- **Check online, every turn (zero model calls):** after each turn we receive the arena's next observation. Parse it
  with the sim (`state_from_observation`) and compare what the sim predicted (our legal moves, scores, feedback for our
  last move) with what the arena shows. Count mismatches per game; above a threshold (say 2) the engine is switched
  off for that game and the log says why. This catches the wrong sim the arbiter could not, and spares the good one.
- Any classification Haiku makes in prep that switches a whole path on or off (talk game, simultaneous, hidden
  information) is asked twice or checked with a code rule (a fixed command word in the action format means "not a
  talk game"), never taken from one answer.

## 2. Deduction games: Wordle, guess_number, x_codebreak, x_fleet (Denis: brain; feedback.py)

| What we saw | Cause | Fix status |
|---|---|---|
| guess_number 1-2: guesses outside the interval the feedback left (775, 870 when the number was 401-498) | 0 proposals could win, we still picked one; Haiku read the OPPONENT's guesses as clues | fixed (range scan, bcb4800) |
| x_codebreak: bare `GGGGG` → warning | proposals lacked the command word, the helper that rejected them was ignored | fixed (command-word repair, spelling gate) |
| Wordle 0-2 every version: guesses reused gray letters (PIOUS after O was gray) | generated `could_win_now` wrong and never verified in prep without a legal list | partly: `feedback.py` on denis/t3-wordle filters correctly, still 0-2 |
| Wordle 0-2 with correct filtering: 87 / 47 candidates left in round 3, the opponent found it | we pick a consistent word, not the most informative one; our candidate pool is a big dictionary, the opponent's is the arena's list | open |

**Robust fix for T3: "replay the past against each hypothesis", then pick by information.**
- **Consistency through the simulator, not a generated helper:** for a candidate secret S, build a state with S as the
  secret, replay OUR past moves with `apply`, and compare the feedback the sim produces with the feedback the arena gave.
  Keep S only if every one matches. This works for any hidden-answer game the sim can play (number, word, code, grid),
  and it uses only `initial_state/apply/observation`, which validation already checks against real samples.
  `feedback.py` stays as a fast path where it recognises the format.
- **Choose the guess that splits the remaining candidates best** (expected size of the largest feedback group, or
  entropy), computed with the same sim. Wordle with entropy finds most words in 3-4 guesses; picking any consistent word
  takes 4-6. In a race, that's the whole difference.
- **Candidate pool:** rank secrets by word frequency (wordfreq, already in the image) and, for an already-public game
  (Wordle is in the kit), consider the kit's published word list as the prior (rule 10.7: allowed, costs some design
  points; team decision, D-log it).
- Never send a guess that is inconsistent with our own feedback, except a deliberate information guess chosen by the
  rule above.

## 3. Simultaneous games: Blotto (Denis: brain; Cristi: announced moves)

| What we saw | Cause | Fix status |
|---|---|---|
| 0-2 vs a bot that plays `34 33 33 0 0` every round and says so | no counter to a repeated move | fixed in 6d4123d (2.5-1.5), but fires only in round 3 |
| rounds 4-5 back to `10 10 10` while the opponent still repeats | counter needs "last 2 identical", resets | open |
| a chat line taken as an opponent move ("All in on F1, F2 again…") | chat text in the move history | open (harmless so far) |

**Robust fix for T3: an opponent model across rounds, games and matches.**
- Keep the opponent's moves per variant in /memory (already stored: `opponents/<id>.moves.json`) and play the best
  response to their **frequency distribution**, not only to the last move, mixed with a safe default (e.g. 70/30) so a
  bot that adapts can't exploit us.
- Cristi's 18defb3 (announced move → engine) plus a per-opponent truthfulness score: an opponent whose announcements
  came true gets countered on its announcement.
- Only parsed moves (that `is_legal_action` accepts) enter the history; chat goes to its own channel.

## 4. Speech-rule games: one slip = a lost match in sudden death (Cristi: talk)

| What we saw | Cause | Fix status |
|---|---|---|
| x_taboo lost a game: "would you rather **spend** Saturday…" (subject: money) | bait line; referee saw only 40 of 129 terms; subject words not expanded | fixed (730c7cb, no baits, 150 terms) |
| x_yesno 0-2 by illegal moves: `say` then `pass` | prep banned common words (not, sure) for an exact-word rule → every answer rejected → fallbacks ran out | fixed (da39457 + 4f3d719 + eb3beef) |
| talking games 15-16 s per turn | retries on wrongly banned answers + extra referee call | mostly fixed (10.9 s) |

**Robust fix for T3: a deterministic gate before every line, and pre-checked material from prep.**
- One final code gate on every outgoing text (move and chat): lowercase, split compounds and hyphens, plural/verb
  forms (spend/spent/spending), digits next to letters, accents removed; if it hits any banned term → replace with a
  pre-checked line. No model needed, so it can never be "skipped for lack of time".
- From prep (500 s idle in these games): a bank of pre-checked answers for the likely question types, and neutral chat
  lines (Cristi's bank on cristi/t3). A slow turn uses the bank instead of an unchecked line.
- Keep exact-word bans exact: everyday-word expansion only for banned **subjects** (now so).

## 5. Moves that don't look like moves (Denis: fallback.py; brain spelling gate)

Seen: `GGGGG` without `guess`, an empty `say`, `pass` in a game with no pass, a 5-letter word in a 6-letter variant,
a Blotto split over the budget.

**Robust fix for T3: a move grammar built in prep, and every outgoing move must parse.**
- From the rules' action format lines, their examples and the arena's sample legal lists, build a small grammar per
  move type (command word + slots: a number in a range, a word of N letters, a cell, N numbers summing to T). Check it
  against the samples in prep.
- Every move we send (engine, Haiku, fallback) must parse with it; the fallback **generates** from it (a valid random
  move) instead of copying examples. This one check would have caught all five failures above.

## 6. Time (everyone)

Seen: Wordle/Blotto turns 14-17 s, the watchdog fired (a legal move went out, but 3 s from a timeout); T1 opponents
answered in 1-3.6 s, we in ~10 s; thinking time is the 2nd tie-break.

**Fix:** a turn budget enforced in code: engine ≤ 4 s, one model call with a hard 9 s timeout, then the best move so far;
target ≤ 10 s worst case. Talking games: one call per turn. Measure it in every sweep (`match_report.py` shows median/max).

## 7. Prep time (Denis, Cristi)

Seen: ready at 60-90 s in talking games (≈ 85 % of 600 s idle), 190-400 s in sim games.
Use the rest for: the move grammar (5), the line bank (4), self-play between the engine and model-only modes to choose
the mode per variant **before** the match (now learned only from losses during matches), and a short checked rules
brief that replaces the full rules in the per-turn prompt (cost: ~$0.007 per match now vs $0.0028 for the cheapest T1 team).

## 8. Tests (Rares)

- New holdouts for what T3 can bring and we never tried: Mastermind with repeats and N colours, Battleship with an
  irregular fleet, Hangman, a negotiation with private values, a Coup-like bluff with claims, a game whose only
  legal moves change every turn. Families stay balanced.
- **A 15-minute regression set** run on every commit before it goes to a candidate: wordle_duel + colonel_blotto vs
  t1_style_bot, guess_number, x_codebreak, x_taboo, x_yesno, hush vs nasty_bot, plus check_bot. Every bug of 14 Nov is
  in it.
- A cold-start run (no prep, 120 s study) and a sudden-death check (a single game decides).
- Budget: the dev key has about $2 left; a full sweep costs ~$0.6, the regression set ~$0.15.

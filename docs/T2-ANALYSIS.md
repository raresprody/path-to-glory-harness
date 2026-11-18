# T2-ANALYSIS.md — what the T2 knockout (15 Nov, 09:00-10:30) teaches us for T3

Source: the 13 public T2 match replays (`tests/replays/t2-*.json` on `t3/library`), 32 games. Champion **0tknsleft**,
runner-up **systematic**, semi-finalists apt-poli and ultraviolence. We lost R1 0-2 to ultraviolence.

## How games were decided

- 27 of 32 on play, **5 on rule breaks** (2 timeouts: mem-ups; 3 second-illegal-move losses: luckydevs `'4'` in Nim,
  sparanghel `guess 65432`/`guess 23145` in 4-digit Bulls, linux-pioneers `fire 1`/`fire a` in Battleship). In T3 sudden
  death one rule break ends the match: robustness is worth as much as strength.

## Our loss (t2-r1-m8)

- **RPS 0-4:** every chat line named our sign or a sign ("leaning hard on paper" -> paper) and ultraviolence countered
  each one. Fixed in B: `brain._no_self_reveal` (no option named in simultaneous games) + Cristi's talk 2a49f86.
- **Bulls 2-4:** `guess BANK` (outside A-F) -> warning, then two guesses our own feedback ruled out. Fixed in B: first-turn
  feedback filter (`Engine._ensure_worker`), feedback check on engine moves, legality gate.

## What the winners did

- **Prisoner's Dilemma (qualify variants):** the winners defect early and keep defecting once ahead (nyx: 10 of 11
  defect vs caramel's alternating; systematic switched to defect from round 5 vs apt-poli's cooperation). Cooperators lose
  "only seat X reached the qualifying score". B's PD policy (lookahead on the match result) already defects against
  cooperators; keep it.
- **RPS:** the final was won by countering a habit: systematic threw scissors 9 times in the last 11 throws, 0tknsleft
  answered with rock/scissors/paper mixes and won 7-6. linux-pioneers beat apt-poli 5-1 by repeating paper/rock in
  blocks against an alternating opponent. Our engine's repeat detection (`propose+rank: opponent repeated`) and the
  opponent profiles (now rebuilt with T2) are the right tools; the chat guard removes our own leak.
- **Perfect-information games (Dots, Hex, Mini Chess, Nim):** 0tknsleft won every one it played (Dots 8-1 and 13-3,
  Hex, Gardner checkmate). These are search games: B's alpha-beta policies are our answer; unseen T3 board games depend
  on the generated simulator + prep evaluation.
- **Battleship:** decided by search quality (fleet sunk in rounds 23-39). B's density policy covers it.

## Applied to T3 (in `t3/library` 312744e)

- `mesh/experience/opponents.json.gz` rebuilt with rr1 + all 13 T2 matches (`tools/build_profiles.py` reads `t2-*` too):
  the opponent predictor now knows each finalist's T2 habits (e.g. systematic's scissors streak, nyx's defect rate).
- Everything above under "Our loss".

# Fix b, tested offline (path-to-glory-3): an evaluation function written by Haiku + short rollouts

- `gen_eval.py <repo> <dir>`: one Haiku call per game (rules + our generated simulator) -> `evaluate(state, seat)`.
  The 4 answers it gave on 15 Nov are in `evals/` (cost $0.0045 for all 4).
- `duel_eval.py <name> <variant> <games> <s/move> [cutoff]`: runs in the image under PyPy with the sims in /sims.
  Gate first (no crash, floats in [-1, 1], agrees with the real winner on finished games), then a duel:
  today's `simworker.mcts` vs the same MCTS whose rollouts stop after 12 plies and score `evaluate()`.
  The eval must run in the simulator's namespace (`dict(vars(mod))`): Haiku's code calls the sim's own helpers.

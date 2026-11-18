"""Offline test of fix b: Haiku writes evaluate(state, seat) from the rules + our generated simulator (dev key)."""
import os, re, sys, json
B, S = sys.argv[1], sys.argv[2]
sys.path.insert(0, B); sys.path.insert(0, os.path.join(B, "tests/out/sweep-20261115-002216/arena-lite"))
from mesh import llm
from arena import registry
SYSTEM = """You write a position evaluation function for a game-playing search engine.
You get the game's rules and the Python simulator the engine uses (its state is a plain dict/list structure).
Write ONE Python function, stdlib only, pure (no I/O, no randomness, no globals that change):

def evaluate(state, seat):
    # returns a float in [-1.0, 1.0]: how good the position is for `seat` (0 or 1); +1 = surely winning for seat,
    # -1 = surely losing, 0 = even. It must work on ANY state the simulator produces, including finished games.

It is called thousands of times per second at the leaves of a search, so keep it fast (no deep copies, no search,
at most a few loops over the board). Use only fields the simulator's state really has (read initial_state and apply).
Think about what decides this game: material, immediate threats, open lines, path lengths, safe moves, counts.
You may define small helper functions above it (prefix their names with _ev_). Answer with the code only, in one
```python block."""
games = {"c4": ("connect_four", 0), "t2_dots": ("t2_dots", 1), "t2_hex": ("t2_hex", 1), "t2_minichess": ("t2_minichess", 1)}
llm.configure({"llm": {"base_url": os.environ["LLM_BASE_URL"], "api_key": os.environ["LLM_API_KEY"], "model": os.environ.get("LLM_MODEL", "haiku")}})
for name, (gid, vi) in games.items():
    cls = registry.get(gid); rules = cls(cls.variants()[vi], 1).rules()
    src = open(os.path.join(S, "sims", name + ".py")).read()
    out = llm.chat(SYSTEM, f"RULES:\n{rules}\n\nSIMULATOR:\n```python\n{src}\n```", timeout=90, max_tokens=3000)
    m = re.search(r"```python\n(.*?)```", out, re.S)
    code = m.group(1) if m else out
    open(os.path.join(S, "sims", name + "_eval.py"), "w").write(code)
    print(name, len(code), "chars", flush=True)
print("cost so far $%.4f" % llm.cost_usd(), llm.USAGE)

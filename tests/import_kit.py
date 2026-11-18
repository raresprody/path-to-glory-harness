"""Pick up a new organizers' kit (e.g. arena-lite with T1's 4 games after T1) and test our bot on the new games.

    python tests/import_kit.py                          # check only: kit commit, new games, changelog lines
    python tests/import_kit.py --apply                  # copy the kit into arena-lite/ and docs/kit/ (no commit)
    python tests/import_kit.py --apply --test <image>   # ... then run the tests on the NEW games only

What --apply does (the CLAUDE.md procedure, automated): pulls arena/participant-kit (Gitea, any team account) into
tests/out/kit, copies its arena-lite/ over ours WITHOUT deleting anything (our own test games in
arena-lite/arena/games/ stay; rule 14.5), and copies CHANGELOG.md, PROTOCOL.md and README.md into docs/kit/.
Review `git status`, then commit and note it in PROGRESS.md.

What --test runs, on the new games only, one at a time (dev key; check the RUNNING line in PROGRESS.md first):
  1. tests/prep_report.py   prep only: ready time, rules digest, simulator, study, drill
  2. tests/sim_vs_engine.py the simulators from step 1 against the real new engines (zero spend)
  3. tests/sweep.py         one --prep match per new game vs nasty_bot: any rule break of ours fails
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KIT_URL = "https://git.hackaton-2026.dvloper.io/arena/participant-kit.git"


def run(cmd, cwd=None, check=True) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and p.returncode != 0:
        raise SystemExit(f"failed: {' '.join(map(str, cmd))}\n{p.stderr.strip()[-400:]}")
    return p.stdout


def games_in(arena_lite: Path) -> dict:
    code = ("import json; from arena import registry; "
            "print(json.dumps({'practice': sorted(registry.practice_games()), 'games': sorted(registry.tournament_games())}))")
    return json.loads(run([sys.executable, "-c", code], cwd=arena_lite))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit-dir", default=str(HERE / "out" / "kit"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--test", metavar="IMAGE", help="after --apply, test this image on the new games")
    ap.add_argument("--games", default="", help="test these game ids instead of the detected new ones")
    a = ap.parse_args()
    kit = Path(a.kit_dir)

    if (kit / ".git").exists():
        # the organizers rewrite the kit's history (19:14 push replaced 17:06): follow it, our copy holds nothing of ours
        run(["git", "-C", str(kit), "fetch", "-q", "origin"])
        run(["git", "-C", str(kit), "reset", "-q", "--hard", "origin/HEAD"])
    else:
        kit.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "-q", KIT_URL, str(kit)])
    commit = run(["git", "-C", str(kit), "log", "-1", "--format=%h %ad %s", "--date=format:%H:%M"]).strip()
    print(f"kit: {commit}")

    # "new" = in the kit but not in our last COMMIT (not the working tree), so a re-run after --apply still knows them
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tar = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD", "arena-lite"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=tar, check=True)
        ours = games_in(Path(tmp) / "arena-lite")
    theirs = games_in(kit / "arena-lite")
    new = {d: sorted(set(theirs[d]) - set(ours[d])) for d in ("practice", "games")}
    print(f"games in kit:  practice {theirs['practice']}  tournament {theirs['games']}")
    print(f"NEW games:     practice {new['practice']}  tournament {new['games']}")

    old_cl = (ROOT / "docs" / "kit" / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    new_cl = (kit / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    added = [l for l in new_cl if l.strip() and l not in old_cl]
    print(f"changelog: {len(added)} new line(s)")
    for l in added[:12]:
        print("   ", l[:200])

    if not a.apply:
        print("\n(check only; --apply to copy the kit in)")
        return 0
    shutil.copytree(kit / "arena-lite", ROOT / "arena-lite", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "results", "memory", "__pycache__"))
    for src, dst in (("CHANGELOG.md", "CHANGELOG.md"), ("PROTOCOL.md", "PROTOCOL.md"), ("README.md", "KIT-README.md")):
        if (kit / src).exists():
            shutil.copy(kit / src, ROOT / "docs" / "kit" / dst)
    print("\napplied: arena-lite/ and docs/kit/ updated (our own test games kept). Review `git status`, commit, note in PROGRESS.md.")
    after = games_in(ROOT / "arena-lite")
    missing = [g for d in ("practice", "games") for g in ours[d] if g not in after[d]]
    if missing:
        print(f"!! games we had are gone after the copy: {missing}")

    ids = [x.strip() for x in a.games.split(",") if x.strip()] if a.games else new["practice"] + new["games"]
    if not a.test:
        return 0
    if not ids:
        print("no new games to test")
        return 0
    only = ",".join(ids)
    py = sys.executable
    print(f"\n=== 1/3 prep_report on {only}", flush=True)
    subprocess.run([py, str(HERE / "prep_report.py"), "--bot", a.test, "--only", only])
    prep = sorted((HERE / "out").glob("prep-*"))[-1]
    print(f"\n=== 2/3 sim_vs_engine on {prep.name}", flush=True)
    subprocess.run([py, "-I", str(HERE / "sim_vs_engine.py"), "--prep-dir", str(prep)])
    print(f"\n=== 3/3 sweep on {only}", flush=True)
    return subprocess.run([py, str(HERE / "sweep.py"), "--bot", a.test, "--jobs", "1", "--only", only]).returncode


if __name__ == "__main__":
    sys.exit(main())

"""Spend per match from our bot's own logs (zero spend): reads mesh's `prep ... spend so far $X` and
`shutdown; total spend $Y` lines in arena-lite warm-container logs (also the sweep copies under tests/out/).
Match spend = total - prep. Prints average and worst per game against the $0.75 match key (rule 7.2).

    python tests/spend_report.py
"""
import collections
import glob
import os
import re

KEY = 0.75
rows = collections.defaultdict(list)
files = glob.glob("tests/out/sweep-*/arena-lite/results/practice/logs/warm/path-to-glory--*.log") + \
    glob.glob("arena-lite/results/practice/logs/warm/path-to-glory--*.log")
for f in files:
    game = os.path.basename(f).split("--")[1].rsplit(".log", 1)[0]
    text = open(f, encoding="utf-8", errors="replace").read()
    for chunk in re.split(r"\[arena [^\]]*\] prep \S+: \d+ variant", text)[1:]:
        prep = re.findall(r"prep \S+: done, spend so far \$([\d.]+)", chunk)
        tot = re.findall(r"shutdown; total spend \$([\d.]+)", chunk)
        if prep and tot and re.search(r"end game \d+", chunk):
            p, t = float(prep[-1]), float(tot[-1])
            rows[game].append((p, t - p))
if not rows:
    raise SystemExit("no warm logs with spend lines found")
print(f"{'game':14s} {'n':>3s} {'prep avg':>9s} {'match avg':>10s} {'match max':>10s}  of the $0.75 key (avg / max)")
allm = []
for g, v in sorted(rows.items()):
    m = [x[1] for x in v]
    allm += m
    print(f"{g:14s} {len(v):3d} {sum(x[0] for x in v) / len(v):9.4f} {sum(m) / len(m):10.4f} {max(m):10.4f}"
          f"  {sum(m) / len(m) / KEY * 100:5.2f}% / {max(m) / KEY * 100:5.2f}%")
print(f"ALL: {len(allm)} matches, avg ${sum(allm) / len(allm):.4f}, max ${max(allm):.4f} ({max(allm) / KEY * 100:.1f}% of the key)")

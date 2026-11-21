"""Rule-based task selection (no hand picking):   python -m f3.screen --workers 2   -> tasks.json

A Meta-World task qualifies when
  (1) its scripted teacher solves it within the horizon: >= 90% in sim and >= 50% in every real world, and
  (2) behaviour cloning covers a useful range of sim scores: S(96 demos) >= 0.6, S(6 demos) <= 0.7,
      S(96 demos) - S(3 demos) >= 0.4   (so feedback mechanisms that start at 6 demos have room to move).
If more than MAX_TASKS qualify, a seeded random subset is taken.
"""
import argparse
import json

import numpy as np
from joblib import Parallel, delayed

from .config import ROOT
from .families import demonstrations
from .learner import Policy, episode, successes
from .worlds import ALL_TASKS, REAL, SIM, World, teacher

MAX_TASKS, EPISODES = 12, 30


def screen(task):
    world, tch = World(task, 0), teacher(task)
    row = dict(task=task)
    for name, gap in {"sim": SIM, **REAL}.items():
        world.set_gap(gap)
        world.reseed(1)
        row[f"teacher_{name}"] = float(np.mean([episode(world, tch)[3] for _ in range(EPISODES)]))
    row["qualifies"] = row["teacher_sim"] >= 0.9 and min(row[f"teacher_{n}"] for n in REAL) >= 0.5
    if row["qualifies"]:
        demos = demonstrations(world, tch, "nom", 96, np.random.default_rng(0))
        for m in (3, 6, 96):
            row[f"S{m}"] = successes(world, Policy(demos[:m], 0), SIM, EPISODES, 2) / EPISODES
        row["qualifies"] = row["S96"] >= 0.6 and row["S6"] <= 0.7 and row["S96"] - row["S3"] >= 0.4
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    table = Parallel(n_jobs=args.workers)(delayed(screen)(t) for t in ALL_TASKS)
    qualified = sorted(r["task"] for r in table if r["qualifies"])
    chosen = sorted(np.random.default_rng(0).permutation(qualified)[:MAX_TASKS].tolist())
    (ROOT / "tasks.json").write_text(json.dumps(dict(tasks=chosen, qualified=qualified, table=table), indent=1))
    print(f"{len(qualified)} of {len(table)} tasks qualify; using {len(chosen)}: {chosen}")

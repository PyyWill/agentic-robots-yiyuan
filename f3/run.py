"""Run the experiment:   python -m f3.run --preset full --workers 64

One job = (task, data condition, seed). It builds the four families, measures every policy in the nominal sim
(S) and in the three real worlds, and writes one row per policy to results/<preset>/<task>__<cond>__<seed>.jsonl.
Finished jobs are skipped, so the same command resumes an interrupted run.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one core per job
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from itertools import product  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from . import families as fam  # noqa: E402
from .config import ROOT, preset  # noqa: E402
from .learner import successes  # noqa: E402
from .worlds import REAL, SIM, World, teacher  # noqa: E402


def job(task, cond, seed, cfg, out):
    path = out / f"{task}__{cond}__{seed}.jsonl"
    if path.exists():
        return f"skip  {path.name}"
    t0 = time.time()
    world, tch = World(task, seed), teacher(task)
    rng = np.random.default_rng([seed, cond == "dr"])
    eval_seeds = iter(np.random.SeedSequence([seed, cond == "dr", 7]).generate_state(10_000))
    rows = []

    def measure(family, step, policy):
        """Fresh, independent episodes for every policy and every world."""
        row = dict(task=task, cond=cond, seed=seed, family=family, step=step,
                   S=[successes(world, policy, SIM, cfg.e_sim, next(eval_seeds)), cfg.e_sim])
        for name, gap in REAL.items():
            row[name] = [successes(world, policy, gap, cfg.e_real, next(eval_seeds)), cfg.e_real]
        rows.append(row)

    demos = fam.demonstrations(world, tch, cond, max(cfg.scales), rng)
    by_scale = dict(fam.scaling(demos, seed, cfg))
    for m, policy in by_scale.items():
        measure("scaling", m, policy)
    start = by_scale[cfg.m0]
    for name, family in [("dagger", fam.dagger(world, tch, cond, demos, start, seed, cfg, rng)),
                         ("self_imitation", fam.self_imitation(world, tch, cond, demos, start, seed, cfg, rng)),
                         ("best_of_k", fam.best_of_k(world, cond, demos, seed, cfg, rng))]:
        for step, policy in family:
            measure(name, step, policy)

    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.rename(path)
    return f"done  {path.name}  {len(rows)} policies  {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full", choices=["full", "pilot", "smoke"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    cfg = preset(args.preset)
    out = ROOT / "results" / args.preset
    out.mkdir(parents=True, exist_ok=True)
    jobs = list(product(cfg.tasks, cfg.conds, cfg.seeds))
    print(f"{args.preset}: {len(jobs)} jobs on {args.workers} workers -> {out}", flush=True)
    for msg in Parallel(n_jobs=args.workers, return_as="generator_unordered")(delayed(job)(*j, cfg, out) for j in jobs):
        print(msg, flush=True)

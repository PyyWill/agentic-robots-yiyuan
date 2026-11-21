"""Pre-registered analysis:   python -m f3.analyze results/full   -> prints and writes <folder>/RESULTS.md

Question: at the SAME sim score, do policies improved through simulator feedback do worse in the real worlds
than policies improved by adding demonstrations?   (estimand and decision rule: PREREGISTRATION.md)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WORLDS = ["act", "obs", "all"]
MECHANISMS = ["dagger", "self_imitation", "best_of_k"]
SESOI = 0.03  # smallest effect of interest: 3 points of real success


def load(folder):
    """One row per policy; S and the three real-world columns become success rates."""
    rows = [json.loads(line) for f in sorted(Path(folder).glob("*.jsonl")) for line in f.read_text().splitlines()]
    df = pd.DataFrame(rows)
    for col in ["S"] + WORLDS:
        df[col] = df[col].map(lambda kn: kn[0] / kn[1])
    return df


def offsets(df, straight_line=False):
    """Distance from the scaling family at equal sim score, in real success.

    Per (task, condition, world) the reference curve joins the scaling family's mean (S, real) points, one per data
    scale, computed on the reference seeds (all but the last). Every other policy is compared with the curve at its
    own S; policies outside the curve's S range are left out. The last seed's scaling policies never enter the curve:
    they are the placebo and must come out at zero. `straight_line=True` replaces the curve by a least-squares line
    (robustness check only).  -> one row per (task, cond, world, seed), one column per family."""
    out = []
    for (task, cond), g in df.groupby(["task", "cond"]):
        scaling, held_out = g.family == "scaling", g.seed == g.seed.max()
        ref = g[scaling & ~held_out]
        curve = ref.groupby("step")[["S"] + WORLDS].mean().sort_values("S")
        test = g[~scaling | held_out]
        test = test[test.S.between(curve.S.min(), curve.S.max())]
        family = test.family.where(test.family != "scaling", "placebo")
        for world in WORLDS:
            if straight_line:
                expected = np.polyval(np.polyfit(ref.S, ref[world], 1), test.S)
            else:
                expected = np.interp(test.S, curve.S, curve[world])
            mean_residual = (test[world] - expected).groupby([test.seed, family]).mean().unstack()
            out.append(mean_residual.assign(task=task, cond=cond, world=world).reset_index())
    off = pd.concat(out, ignore_index=True)
    return off.assign(average=off[MECHANISMS].mean(axis=1))


def across_tasks(per_seed, level=0.95):
    """Tasks are the independent units. `per_seed` is a Series indexed by (task, seed).
    -> mean effect over tasks, confidence interval, prediction interval for a NEW task, between-task sd (tau)."""
    by_task = per_seed.groupby("task").agg(["mean", "sem"]).fillna(0.0)
    d, T = by_task["mean"], len(by_task)
    mean, se = d.mean(), d.sem()
    tau2 = max(0.0, d.var() - (by_task["sem"] ** 2).mean())  # between-task variance beyond seed noise
    ci = stats.t.ppf(0.5 + level / 2, T - 1) * se
    pi = stats.t.ppf(0.975, max(T - 2, 1)) * np.sqrt(tau2 + se**2)
    return dict(tasks=T, mean=mean, lo=mean - ci, hi=mean + ci, pi_lo=mean - pi, pi_hi=mean + pi,
                tau=np.sqrt(tau2), negative=int((d < 0).sum()))


def summarize(off, cond):
    """Average over the three real worlds, then across tasks. Per-mechanism intervals are Bonferroni-corrected."""
    per_seed = off[off.cond == cond].groupby(["task", "seed"])[MECHANISMS + ["average", "placebo"]].mean()
    return {col: across_tasks(per_seed[col].dropna(), 1 - 0.05 / 3 if col in MECHANISMS else 0.95) for col in per_seed}


def verdict(s):
    """The pre-registered decision rule, applied to the nominal data condition."""
    if not s["placebo"]["lo"] <= 0 <= s["placebo"]["hi"]:
        return "INVALID - the placebo is off the curve; the pipeline must be fixed before any reading"
    below = [m for m in MECHANISMS if s[m]["hi"] < 0]
    if s["average"]["hi"] < 0 and s["average"]["pi_hi"] < 0 and len(below) >= 2:
        return f"GENERAL - holds across tasks and for {len(below)} of 3 mechanisms ({', '.join(below)})"
    if below or s["average"]["hi"] < 0:
        return f"SPECIFIC - real, but not general: mechanisms below the curve = {below or 'none individually'}"
    if -SESOI < s["average"]["lo"] and s["average"]["hi"] < SESOI:
        return "NO-GO - no effect of practical size: the sim score carries over the same way however it was obtained"
    return "NOT ESTABLISHED - no effect detected, but an effect of practical size cannot be excluded"


def report(df):
    off = offsets(df)
    lines = [f"policies: {len(df)}   tasks: {df.task.nunique()}   seeds per task: {df.groupby('task').seed.nunique().max()}", ""]
    row = lambda name, r: (f"| {name} | {100 * r['mean']:+.1f} | [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | "
                           f"[{100 * r['pi_lo']:+.1f}, {100 * r['pi_hi']:+.1f}] | {100 * r['tau']:.1f} | {r['negative']}/{r['tasks']} |")
    for cond in sorted(df.cond.unique(), reverse=True):
        s = summarize(off, cond)
        lines += [f"### data condition: {cond}   (points of real success at equal sim score; negative = worse than scaling)", "",
                  "| | effect | conf. interval | prediction interval (new task) | between-task sd | tasks with effect < 0 |",
                  "|---|---|---|---|---|---|"]
        lines += [row(name, s[name]) for name in ["average"] + MECHANISMS + ["placebo"]] + [""]
        if cond == "nom":
            line = summarize(offsets(df, straight_line=True), cond)
            lines += [f"**VERDICT: {verdict(s)}**", "",
                      f"robustness (straight line instead of the curve): average {100 * line['average']['mean']:+.1f} "
                      f"[{100 * line['average']['lo']:+.1f}, {100 * line['average']['hi']:+.1f}]  ->  {verdict(line).split(' - ')[0]}", ""]
    nominal = off[off.cond == "nom"]
    by_world = nominal.groupby(["world", "task"])[MECHANISMS].mean().groupby("world").mean()
    lines += ["### nominal condition, by real world (mean over tasks, points)", "", (100 * by_world).round(1).to_markdown(), ""]
    by_task = nominal.groupby("task")[MECHANISMS + ["placebo"]].mean()
    lines += ["### nominal condition, by task (points)", "", (100 * by_task).round(1).to_markdown(), ""]
    means = df.groupby(["cond", "family"])[["S"] + WORLDS].mean()
    lines += ["### where the families sit (mean success rates)", "", means.round(2).to_markdown(), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    text = report(load(folder))
    (folder / "RESULTS.md").write_text(text)
    print(text)

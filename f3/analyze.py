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
P_VARIES = 0.01 / 3  # 'the effect differs between tasks': one-way ANOVA, 1% over the three mechanisms
FLOOR = 0.20  # a (task, real world) is analysed only if some family reaches 20% real success there


def load(folder):
    """One row per policy; S and the three real-world columns become success rates."""
    rows = [json.loads(line) for f in sorted(Path(folder).glob("*.jsonl")) for line in f.read_text().splitlines()]
    df = pd.DataFrame(rows)
    for col in ["S"] + WORLDS:
        df[col] = df[col].map(lambda kn: kn[0] / kn[1])
    return df


def offsets(df, straight_line=False, floor=FLOOR):
    """Distance from the scaling family at equal sim score, in real success.

    Each seed is one complete replicate of the experiment (own demonstrations, own scaling policies, own feedback
    runs). Inside a replicate the scaling policies' (S, real) points, joined in order of S, are the reference curve,
    and every feedback policy is compared with the curve at its own S. Policies outside the curve's S range are left
    out (no extrapolation). The placebo is the NEXT seed's scaling family, treated exactly like a feedback family:
    it must come out at zero. A (task, world) where no family reaches `floor` real success at any step (mean over
    seeds) says nothing about transfer and is left out. `straight_line=True` replaces the curve by a least-squares
    line (robustness check only).  -> one row per (task, cond, world, seed), one column per family."""
    out = []
    for (task, cond), g in df.groupby(["task", "cond"]):
        alive = g.groupby(["family", "step"])[WORLDS].mean().max() >= floor
        seeds = sorted(g.seed.unique())
        for seed, next_seed in zip(seeds, np.roll(seeds, -1)):
            scaling = g[(g.seed == seed) & (g.family == "scaling")]
            if next_seed == seed or scaling.S.nunique() < 2:  # no replicate to compare with / no spread in sim score
                continue
            curve = scaling.groupby("S")[WORLDS].mean()  # knots in order of S; policies with equal S are averaged
            test = pd.concat([g[(g.seed == seed) & (g.family != "scaling")],
                              g[(g.seed == next_seed) & (g.family == "scaling")].assign(family="placebo")])
            test = test[test.S.between(curve.index.min(), curve.index.max())]
            for world in alive.index[alive]:
                expected = (np.polyval(np.polyfit(scaling.S, scaling[world], 1), test.S) if straight_line
                            else np.interp(test.S, curve.index, curve[world]))
                family_means = (test[world] - expected).groupby(test.family).mean()
                out.append(dict(family_means, task=task, cond=cond, world=world, seed=seed))
    off = pd.DataFrame(out).reindex(columns=["task", "cond", "world", "seed"] + MECHANISMS + ["placebo"])
    return off.assign(average=off[MECHANISMS].mean(axis=1))


def across_tasks(per_seed, level=0.95):
    """Seeds are replicates inside a task; tasks are the units of generality. `per_seed`: Series indexed (task, seed).
    -> mean effect over tasks, confidence interval, prediction interval for a NEW task, between-task sd (tau),
       and the p-value of 'the effect is the same in every task' (one-way ANOVA, tasks as groups)."""
    by_task = per_seed.groupby("task").agg(["mean", "sem"]).fillna(0.0)
    d, T = by_task["mean"], len(by_task)
    mean, se = d.mean(), d.sem()
    tau2 = max(0.0, d.var() - (by_task["sem"] ** 2).mean())  # between-task variance beyond seed noise
    ci = stats.t.ppf(0.5 + level / 2, T - 1) * se
    pi = stats.t.ppf(0.975, max(T - 2, 1)) * np.sqrt(tau2 + se**2)
    p_tasks = stats.f_oneway(*[v.values for _, v in per_seed.groupby("task")]).pvalue if T > 1 else np.nan
    return dict(tasks=T, mean=mean, lo=mean - ci, hi=mean + ci, pi_lo=mean - pi, pi_hi=mean + pi,
                tau=np.sqrt(tau2), p_tasks=p_tasks, negative=int((d < 0).sum()))


def summarize(off, cond):
    """Average over the analysed real worlds, then across tasks. Interval levels: 95% for the average, Bonferroni
    (3 tests) for each mechanism, 99% for the placebo (the gate should fire on a real bias, not on a hair)."""
    per_seed = off[off.cond == cond].groupby(["task", "seed"])[MECHANISMS + ["average", "placebo"]].mean()
    level = {"average": 0.95, "placebo": 0.99} | {m: 1 - 0.05 / 3 for m in MECHANISMS}
    return {col: across_tasks(per_seed[col].dropna(), level[col]) for col in per_seed}


def clearly(r, sign):
    """Different from zero in the given direction (interval excludes 0) AND at least the smallest effect of interest."""
    return sign * r["mean"] >= SESOI and (r["hi"] < 0 if sign < 0 else r["lo"] > 0)


def verdict(s):
    """The pre-registered decision rule, applied to the nominal data condition."""
    if s["average"]["tasks"] < 3:
        return "NOT APPLICABLE - fewer than 3 tasks: nothing can be said about generality"
    if not s["placebo"]["lo"] <= 0 <= s["placebo"]["hi"]:
        return "INVALID - the placebo is off the curve; the pipeline must be fixed before any reading"
    avg = s["average"]
    below = [m for m in MECHANISMS if clearly(s[m], -1)]
    above = [m for m in MECHANISMS if clearly(s[m], +1)]
    varies = [m for m in MECHANISMS if s[m]["p_tasks"] < P_VARIES]
    detail = f"below the curve = {below or 'none'}, above = {above or 'none'}, differs between tasks = {varies or 'none'}"
    if clearly(avg, -1) and avg["pi_hi"] < 0 and len(below) >= 2:
        return f"GENERAL - a deficit across tasks and for {len(below)} of 3 mechanisms: {detail}"
    if below or clearly(avg, -1):
        return f"SPECIFIC - a deficit, but not a general one: {detail}"
    if above or varies:
        return f"MIXED - how a sim score was obtained matters, but not as a deficit: {detail}"
    if -SESOI < avg["lo"] and avg["hi"] < SESOI and all(abs(s[m]["mean"]) < SESOI for m in MECHANISMS):
        return "NO-GO - no effect of practical size: the sim score carries over the same way however it was obtained"
    return "NOT ESTABLISHED - no effect detected, but an effect of practical size cannot be excluded"


def report(df):
    off = offsets(df)
    cells = off.drop_duplicates(["cond", "task", "world"]).groupby("cond").size()
    lines = [f"policies: {len(df)}   tasks: {df.task.nunique()}   seeds per task: {df.groupby('task').seed.nunique().max()}",
             f"(task, real world) pairs analysed (some family reaches {FLOOR:.0%} real success): {cells.to_dict()} "
             f"of {df.task.nunique() * len(WORLDS)} per condition", ""]
    row = lambda name, r: (f"| {name} | {100 * r['mean']:+.1f} | [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | "
                           f"[{100 * r['pi_lo']:+.1f}, {100 * r['pi_hi']:+.1f}] | {100 * r['tau']:.1f} | "
                           f"{r['p_tasks']:.3f} | {r['negative']}/{r['tasks']} |")
    for cond in sorted(df.cond.unique(), reverse=True):
        s = summarize(off, cond)
        lines += [f"### data condition: {cond}   (points of real success at equal sim score; negative = worse than scaling)", "",
                  "| | effect | conf. interval | prediction interval (new task) | between-task sd | p (same in every task) | tasks with effect < 0 |",
                  "|---|---|---|---|---|---|---|"]
        lines += [row(name, s[name]) for name in ["average"] + MECHANISMS + ["placebo"]] + [""]
        if cond == "nom":
            lines += [f"**VERDICT: {verdict(s)}**", ""]
            for label, kw in [("straight line instead of the curve", dict(straight_line=True)), ("no floor rule", dict(floor=0.0))]:
                alt = summarize(offsets(df, **kw), cond)
                lines += [f"robustness ({label}): average {100 * alt['average']['mean']:+.1f} "
                          f"[{100 * alt['average']['lo']:+.1f}, {100 * alt['average']['hi']:+.1f}]  ->  {verdict(alt).split(' - ')[0]}", ""]
    nominal = off[off.cond == "nom"]
    by_world = nominal.groupby(["world", "task"])[MECHANISMS + ["placebo"]].mean().groupby("world").mean()
    lines += ["### nominal condition, by real world (mean over the analysed tasks, points)", "",
              (100 * by_world).round(1).assign(tasks=nominal.groupby("world").task.nunique()).to_markdown(), ""]
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

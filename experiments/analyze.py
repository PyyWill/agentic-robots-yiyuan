"""Aggregate results/raw/*.jsonl into results/summary.json and print the tables used in the README.

    python -m experiments.analyze
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
METRICS = ["L2_ood_0shot", "L2_ood_within2", "L2_iid_0shot", "L2_iid_within2", "L1_0shot", "L1_within2",
           "invent_0shot", "invent_within2", "invent_built", "invent_rejects_broom"]
GT = {"hook back": {"fits under gap", "reaches target", "hook holds ring", "rigid"}}


def load(name):
    path = RAW / f"{name}.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else []


def mean_ci(x):
    """Mean and 95% half-width over seeds (t-interval)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    from scipy.stats import t
    return float(x.mean()), float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def curves(rows):
    """{metric: {m: (mean, ci, n)}} over seeds, plus baselines (random, prior) and structure statistics."""
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["m"]].append(r)
    out = {"curve": {}, "baseline": {}, "structure": {}}
    for k in METRICS:
        out["curve"][k] = {m: (*mean_ci([r[k] for r in rs if k in r]), len(rs)) for m, rs in sorted(by_m.items())
                           if any(k in r for r in rs)}
        for b in ("random", "prior"):
            vals = [r[b][k] for r in rows if b in r and k in r[b]]
            if vals:
                out["baseline"].setdefault(b, {})[k] = mean_ci(vals)
    for m, rs in sorted(by_m.items()):
        st = defaultdict(list)
        for r in rs:
            for name, s in r["skills"].items():
                leaves = {side: set(c) for side, c in s["leaves"].items()}
                st["exact_gt"].append(name in GT and s["split"] is None and leaves.get("None") == GT[name])
                st["map_accuracy"].append(s["map_accuracy"])
                st["p_split"].append(s["p_split"])
                st["n_cands"].append(len(s["cands"]))
            st["queries"].append(len(r["growth"]))
            st["rigid_added"].append(any("rigid" in g[2] for g in r["growth"]))
            st["trials"].append(r["trials"])
        out["structure"][m] = {k: float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals)
                               else None for k, vals in st.items()}
    return out


def gonogo():
    """Paired comparison with the Claude baseline (and its anonymised variant) at the m values it was run for."""
    out = {}
    for tag in ("claude", "claude_anon"):
        for cond in ("base", "noise0.2", "varied"):
            ours = {(r["seed"], r["m"]): r for r in load(cond)}
            claude = defaultdict(list)
            for r in load(f"{tag}_{cond}"):
                claude[r["seed"], r["m"], r["set"]].append(r)
            for m in sorted({k[1] for k in claude}):
                seeds = sorted({s for s, mm, _ in claude if mm == m and (s, m) in ours})
                res = {}
                for key in ("L2_ood", "invent"):
                    have = [s for s in seeds if claude[s, m, key]]
                    if not have:
                        continue
                    for metric, i in (("0shot", 0), ("within2", 1)):
                        c = [np.mean([expected(r)[i] for r in claude[s, m, key]]) for s in have]
                        o = [ours[s, m][f"{key}_{metric}"] for s in have]
                        d = np.array(o) - np.array(c)
                        res[f"{key}_{metric}"] = {"ours": mean_ci(o), "claude": mean_ci(c), "diff": mean_ci(d),
                                                  "seeds_ours_better": int((d > 0).sum()), "n_seeds": len(d)}
                    res[f"{key}_claude_retry"] = float(np.mean([r["choices"][0] == r["choices"][-1]
                                                                for s in have for r in claude[s, m, key]]))
                res["cost_usd"] = round(sum(r["cost"] for k, rs in claude.items() if k[1] == m for r in rs), 2)
                out[f"{tag}:{cond}@m{m}"] = res
    crit = [out.get(f"claude:{c}@m32", {}).get("L2_ood_within2") for c in ("noise0.2", "varied")]
    out["verdict"] = None if None in crit else (
        "GO" if all(x["diff"][0] >= 0.15 and x["seeds_ours_better"] >= 8 for x in crit) else "NO-GO")
    return out


def expected(r):
    """Claude's expected success (first attempt, within two) from the validity of its two choices."""
    q = 1.0 - r["noise"]
    first = r["valid"][0] * q
    return first, first + (1 - first) * r["valid"][1] * q


def f(x):
    return "–" if x is None else f"{x[0]:.2f} ± {x[1]:.2f}" if isinstance(x, (tuple, list)) else f"{x:.2f}"


def markdown(summary):
    """Tables for results/RESULTS.md (mean ± 95% CI over seeds; expected success in new OOD situations)."""
    S, out = summary, []
    c = S.get("base")
    if c:
        out += ["### Base (hook back, π = 1): learning curve", "",
                "| m | L2 1st attempt | L2 within two | L1 1st | invention 1st | MAP = true essence | queries | trials |",
                "|---|---|---|---|---|---|---|---|"]
        for m in sorted(c["curve"]["L2_ood_0shot"], key=int):
            st = c["structure"][m]
            out.append(f"| {m} | {f(c['curve']['L2_ood_0shot'][m])} | {f(c['curve']['L2_ood_within2'][m])} | "
                       f"{f(c['curve'].get('L1_0shot', {}).get(m))} | {f(c['curve']['invent_0shot'][m])} | "
                       f"{f(st.get('exact_gt'))} | {f(st.get('queries'))} | {st.get('trials', 0):.0f} |")
        b = c["baseline"]
        out += ["", f"Baselines on the same tests: random {f(b['random']['L2_ood_0shot'])} / "
                f"{f(b['random']['L2_ood_within2'])}, VLM prior only {f(b['prior']['L2_ood_0shot'])} / "
                f"{f(b['prior']['L2_ood_within2'])} (L2 1st / within two); invention 1st: random "
                f"{f(b['random']['invent_0shot'])}, VLM prior {f(b['prior']['invent_0shot'])}.", ""]
    rows = [k for k in ("base", "noise0.1", "noise0.2", "unrepresentable", "pi0.75", "varied", "no_name", "informed",
                        "misleading", "hang", "hang_nosplit") if k in S]
    out += ["### All conditions at the last checkpoint", "",
            "| condition | m | L1 1st | L2 1st | L2 within two | invention 1st | structure agrees with truth | queries | |D| | VLM-prior L2 1st |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k in rows:
        cur, st = S[k]["curve"], S[k]["structure"]
        for m in sorted({"32", max(cur["L2_ood_0shot"], key=int)}, key=int):
            if m not in cur["L2_ood_0shot"]:
                continue
            out.append(f"| {k} | {m} | {f(cur.get('L1_0shot', {}).get(m))} | {f(cur['L2_ood_0shot'][m])} | "
                       f"{f(cur['L2_ood_within2'][m])} | {f(cur.get('invent_0shot', {}).get(m))} | "
                       f"{f(st[m].get('map_accuracy'))} | {f(st[m].get('queries'))} | {f(st[m].get('n_cands'))} | "
                       f"{f(S[k]['baseline'].get('prior', {}).get('L2_ood_0shot'))} |")
    arms = [k for k in ("budget_32x1", "budget_16x2", "budget_8x4", "budget_4x8") if k in S]
    if arms:
        out += ["", "### Breadth vs depth (32 random configurations)", "",
                "| arm | L2 1st | L2 within two | trials incl. repeats | structure agrees with truth |", "|---|---|---|---|---|"]
        for k in arms:
            cur, st = S[k]["curve"], S[k]["structure"]
            m = max(cur["L2_ood_0shot"], key=int)
            out.append(f"| {k.split('_')[1]} | {f(cur['L2_ood_0shot'][m])} | {f(cur['L2_ood_within2'][m])} | "
                       f"{st[m]['trials']:.0f} | {f(st[m].get('map_accuracy'))} |")
    g = S.get("gonogo", {})
    keys = [k for k in g if ":" in k]
    if keys:
        out += ["", f"### Go/no-go against Claude (identical inputs). Verdict: **{g.get('verdict')}**", "",
                "| baseline: condition@m | test | ours | Claude | ours − Claude | seeds ours better | Claude retries | cost |",
                "|---|---|---|---|---|---|---|---|"]
        for k in keys:
            for key in ("L2_ood", "invent"):
                for metric in ("within2", "0shot"):
                    r = g[k].get(f"{key}_{metric}")
                    if r:
                        out.append(f"| {k} | {key} {metric} | {f(r['ours'])} | {f(r['claude'])} | {f(r['diff'])} | "
                                   f"{r['seeds_ours_better']}/{r['n_seeds']} | {f(g[k].get(key + '_claude_retry'))} | "
                                   f"${g[k]['cost_usd']} |")
    return "\n".join(out) + "\n"


def main():
    conds = sorted(p.stem for p in RAW.glob("*.jsonl") if not p.stem.startswith("claude"))
    summary = {c: curves(load(c)) for c in conds}
    summary["gonogo"] = gonogo()
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (ROOT / "results" / "RESULTS.md").write_text(markdown(json.loads(json.dumps(summary, default=float))))
    print((ROOT / "results" / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()

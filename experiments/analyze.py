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
    """Paired comparison with the Claude baseline at the m values it was run for."""
    out = {}
    for cond in ("base", "noise0.2", "varied"):
        ours = {(r["seed"], r["m"]): r for r in load(cond)}
        claude = defaultdict(list)
        for r in load(f"claude_{cond}"):
            claude[r["seed"], r["m"], r["set"]].append(r)
        for m in sorted({k[1] for k in claude}):
            seeds = sorted({s for s, mm, _ in claude if mm == m and (s, m) in ours})
            res = {}
            for metric, f in (("within2", any), ("0shot", lambda o: o[0])):
                for key in ("L2_ood", "invent"):
                    c = [np.mean([f(r["outcomes"]) for r in claude[s, m, key]]) for s in seeds if claude[s, m, key]]
                    o = [ours[s, m][f"{key}_{metric}"] for s in seeds if claude[s, m, key]]
                    if c:
                        d = np.array(o) - np.array(c)
                        res[f"{key}_{metric}"] = {"ours": mean_ci(o), "claude": mean_ci(c), "diff": mean_ci(d),
                                                  "seeds_ours_better": int((d > 0).sum()), "n_seeds": len(d)}
            cost = sum(r["cost"] for k, rs in claude.items() if k[1] == m for r in rs)
            res["cost_usd"] = round(cost, 2)
            out[f"{cond}@m{m}"] = res
    crit = [out.get(f"{c}@m32", {}).get("L2_ood_within2") for c in ("noise0.2", "varied")]
    out["verdict"] = None if None in crit else (
        "GO" if all(x["diff"][0] >= 0.15 and x["seeds_ours_better"] >= 8 for x in crit) else "NO-GO")
    return out


def main():
    conds = sorted(p.stem for p in RAW.glob("*.jsonl") if not p.stem.startswith("claude_"))
    summary = {c: curves(load(c)) for c in conds}
    summary["gonogo"] = gonogo()
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for c in conds:
        cur = summary[c]["curve"]
        ms = sorted(cur["L2_ood_within2"])
        print(f"\n== {c}  (m: {ms})")
        for k in ("L2_ood_0shot", "L2_ood_within2", "L1_0shot", "invent_0shot"):
            if k in cur:
                print(f"  {k:16s}", "  ".join(f"{cur[k][m][0]:.2f}±{cur[k][m][1]:.2f}" for m in ms if m in cur[k]),
                      "| random %.2f prior %.2f" % (summary[c]["baseline"].get("random", {}).get(k, (np.nan,))[0],
                                                    summary[c]["baseline"].get("prior", {}).get(k, (np.nan,))[0]))
        st = summary[c]["structure"]
        print("  structure@max m:", {k: (round(v, 2) if v is not None else None) for k, v in st[max(st)].items()})
    print("\n== go/no-go")
    for k, v in summary["gonogo"].items():
        print(" ", k, v)


if __name__ == "__main__":
    main()

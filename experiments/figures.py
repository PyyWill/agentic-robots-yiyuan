"""Static figures: learning curves and tests from results/summary.json, and the model itself from a replayed run.

    python -m experiments.figures
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.run import setup, train_next
from overhyp.proposer import ESSENCE
from overhyp.toolworld import render, succeeds

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (fixed order)
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
GOOD, BAD = "#0ca30c", "#d03b3b"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9.5, "lines.linewidth": 2, "lines.markersize": 6,
                     "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight", "axes.titlesize": 10.5})
MS = [0, 2, 4, 8, 16, 32]


def curve(ax, summ, cond, key, color, label):
    cur = summ[cond]["curve"][key]
    ms = [m for m in MS if str(m) in cur]
    mean, ci = np.array([cur[str(m)][0] for m in ms]), np.array([cur[str(m)][1] for m in ms])
    x = [MS.index(m) for m in ms]
    ax.plot(x, mean, "-o", color=color, label=label, ms=5)
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15, lw=0)
    ax.annotate(label, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=8.5)


def axes_m(ax, title, ylabel, chance=None, prior=None):
    ax.set(xticks=range(len(MS)), xticklabels=MS, xlabel="m  (training situations seen)", ylabel=ylabel,
           ylim=(-0.02, 1.04), title=title, xlim=(-0.2, len(MS) + 0.6))
    if chance is not None:
        ax.axhline(chance, color=MUTED, ls="--", lw=1)
        ax.text(0, chance + 0.02, "chance", color=MUTED, fontsize=8)
    if prior is not None:
        ax.axhline(prior, color=MUTED, ls=":", lw=1.2)
        ax.text(0, prior + 0.02, "VLM prior only", color=MUTED, fontsize=8)


def fig_curves(summ):
    groups = [("H1: L2 rises with m and saturates at a ceiling set by grounding",
               [("base", "π = 1"), ("pi0.75", "π = 0.75"), ("varied", "π = 0.5"), ("unrepresentable", "rigid ∉ D")]),
              ("H2: VLM prior quality shifts the curve, not the ceiling",
               [("informed", "informed prior"), ("base", "flat prior"), ("misleading", "misleading prior")])]
    for i, (title, conds) in enumerate(groups):
        conds = [(c, l) for c, l in conds if c in summ]
        if not conds:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, (key, yl, ch) in zip(axs, [("L2_ood_0shot", "success at 1st attempt", 1 / 6),
                                           ("L2_ood_within2", "success within 2 attempts", 1 / 3)]):
            for (c, lab), col in zip(conds, SERIES):
                curve(ax, summ, c, key, col, lab)
            axes_m(ax, "", yl, ch, summ["base"]["baseline"]["prior"][key][0] if i == 1 else None)
        fig.suptitle(title + "   (new out-of-distribution situations, 10 seeds, 95% CI)", color=INK, fontsize=10.5)
        fig.savefig(FIG / f"{'h1' if i == 0 else 'h2'}_curves.png")
        plt.close(fig)


def fig_smith(summ):
    """Robot version of Smith et al. (2002) Exp 2: L1 (seen situations) vs L2 (new situations)."""
    conds = [("base", "names"), ("no_name", "no name"), ("varied", "varied")]
    if not all(c in summ for c, _ in conds):
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1]})
    smith = {"names": (0.66, 0.65), "no name": (0.62, 0.32), "varied": (0.67, 0.34)}
    for ax, (title, vals, chance) in zip(axs, [
            ("Children, Smith et al. 2002 Exp 2", smith, 1 / 3),
            ("Robot, m = 32, success at 1st attempt", {l: (summ[c]["curve"]["L1_0shot"]["32"][0],
                                                          summ[c]["curve"]["L2_ood_0shot"]["32"][0]) for c, l in conds},
             1 / 6)]):
        x = np.arange(3)
        for j, (lab, col) in enumerate([("L1  (trained category / seen situation)", SERIES[0]),
                                        ("L2  (new category / new situation)", SERIES[1])]):
            v = [vals[l][j] for _, l in conds]
            ax.bar(x + (j - 0.5) * 0.36, v, 0.34, color=col, label=lab)
            for xi, vi in zip(x, v):
                ax.text(xi + (j - 0.5) * 0.36, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=8, color=INK2)
        ax.axhline(chance, color=MUTED, ls="--", lw=1)
        ax.set(xticks=x, xticklabels=[l for _, l in conds], ylim=(0, 1.1), title=title, ylabel="proportion correct")
        ax.grid(axis="x", visible=False)
    axs[0].legend(loc="upper right", fontsize=8)
    fig.savefig(FIG / "h3_smith_exp2.png")
    plt.close(fig)


def fig_budget(summ):
    arms = [c for c in ("budget_32x1", "budget_16x2", "budget_8x4", "budget_4x8") if c in summ]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(arms))
    for j, (key, lab, col) in enumerate([("L2_ood_0shot", "1st attempt", SERIES[0]),
                                         ("L2_ood_within2", "within 2", SERIES[1])]):
        vals = [summ[a]["curve"][key][max(summ[a]["curve"][key], key=int)] for a in arms]
        ax.bar(x + (j - 0.5) * 0.36, [v[0] for v in vals], 0.34, yerr=[v[1] for v in vals], color=col, label=lab,
               error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.set(xticks=x, xticklabels=[a.split("_")[1].replace("x", " sit. × ") + " tries" for a in arms], ylim=(0, 1.05),
           ylabel="L2 success (new OOD situations)", title="H4: same budget of 32 random tries, breadth vs depth")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.savefig(FIG / "h4_breadth_depth.png")
    plt.close(fig)


def fig_gonogo(summ):
    g = summ.get("gonogo", {})
    keys = sorted({k.split(":")[1] for k in g if k.startswith("claude:")}, key=lambda k: (k.split("@")[1], k))
    if not keys:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    who = [("ours", None, SERIES[0], "Overhypothesis"), ("claude", "claude", SERIES[1], "Claude fable-5.1, text memory"),
           ("claude", "claude_anon", SERIES[2], "same, predicate names hidden")]
    for ax, metric in zip(axs, ["L2_ood_within2", "L2_ood_0shot"]):
        x = np.arange(len(keys))
        for j, (field, tag, col, lab) in enumerate(who):
            vals = [g.get(f"{tag or 'claude'}:{k}", {}).get(metric) for k in keys]
            xs = [xi + (j - 1) * 0.27 for xi, v in zip(x, vals) if v]
            v = [v[field] for v in vals if v]
            if v:
                ax.bar(xs, [a[0] for a in v], 0.25, yerr=[a[1] for a in v], color=col, label=lab,
                       error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        ax.set(xticks=x, xticklabels=[k.replace("@m", "\nm = ") for k in keys], ylim=(0, 1.1),
               title="within two attempts (primary)" if "within" in metric else "first attempt")
        ax.grid(axis="x", visible=False)
    axs[0].legend(loc="lower left", fontsize=8)
    axs[0].set_ylabel("expected L2 success, identical inputs")
    fig.suptitle(f"Go/no-go (pre-registered: m = 32, noise0.2 and varied, within two): {g.get('verdict')}", color=INK)
    fig.savefig(FIG / "gonogo.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- the model itself
def replay(cond="base", seed=0, m=32, snapshots=(4, 8, 32)):
    """Replay a run; record after every situation the posterior inclusion of each candidate, and the
    posterior over conjunctions of the unsplit skill at a few snapshots."""
    cfg, agent, rng, sets = setup(cond, seed)
    seen, hist, snaps = [], [], {}
    for k in range(1, m + 1):
        train_next(agent, rng, cfg, seen)
        sk = agent.skills["hook back"]
        hist.append((len(agent.memory), dict(zip(sk.cands, sk.model.inclusion())), sk.model.top[None]))
        if k in snapshots:
            L = sk.model.leaves[None, None]
            snaps[k] = (list(sk.cands), L.H[L.keep], L.w, sk.model.top[None])
    return agent, sets, hist, snaps


def fig_lattice(snaps):
    """Posterior over conjunctions of the four true components (the lattice), at several m."""
    from itertools import combinations
    gt = ESSENCE["hook_back"]
    short = {"fits under gap": "F", "reaches target": "R", "hook holds ring": "H", "rigid": "S"}
    subsets = [c for r in range(5) for c in combinations(gt, r)]
    pos = {}
    for r in range(5):
        level = [c for c in subsets if len(c) == r]
        for i, c in enumerate(level):
            pos[c] = (i - (len(level) - 1) / 2, r)
    fig, axs = plt.subplots(1, len(snaps), figsize=(4.2 * len(snaps), 4.2))
    for ax, (m, (cands, H, w, p_unsplit)) in zip(np.atleast_1d(axs), sorted(snaps.items())):
        mass = {}
        idx = {c: cands.index(c) if c in cands else None for c in gt}
        other = 0.0
        for h, wk in zip(H, w):
            chosen = tuple(c for c in gt if idx[c] is not None and h[idx[c]])
            if h.sum() == len(chosen):  # only true components in the conjunction
                mass[chosen] = mass.get(chosen, 0) + wk
            else:
                other += wk
        for a in subsets:
            for b in subsets:
                if len(b) == len(a) + 1 and set(a) < set(b):
                    ax.plot(*zip(pos[a], pos[b]), color="#e1e0d9", lw=1, zorder=0)
        for c in subsets:
            x, y = pos[c]
            v = mass.get(c, 0.0)
            ax.scatter(x, y, s=40 + 1400 * v, color=SERIES[0] if v > 0.05 else "#c3c2b7", zorder=2,
                       edgecolors=SURF, linewidths=2)
            ax.text(x, y - 0.36 - 0.3 * v, "".join(short[e] for e in c) or "∅", ha="center", fontsize=8, color=INK2)
            if v > 0.05:
                ax.text(x + 0.3 + 0.6 * v, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set(xlim=(-3.6, 3.6), ylim=(-0.7, 4.6), xticks=[], yticks=range(5), ylabel="conjuncts" if m == 4 else "",
               title=f"m = {m}: mass with distractors {other:.2f}" + ("" if "rigid" in cands else ", rigid ∉ D"))
        ax.grid(False)
    fig.suptitle("Posterior over conjunctions of the true components (unsplit skill, seed 0).  F fits under gap, "
                 "R reaches target, H hook holds ring, S rigid (stiff)", color=INK, fontsize=10)
    fig.savefig(FIG / "model_lattice.png")
    plt.close(fig)


def fig_timeline(agent, hist):
    sk = agent.skills["hook back"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    t = [h[0] for h in hist]
    gt = ESSENCE["hook_back"]
    for c, col in zip(gt, SERIES):
        ax.plot(t, [h[1].get(c, np.nan) for h in hist], "-", color=col, label=c)
    for c in sk.cands:
        if c not in gt:
            ax.plot(t, [h[1].get(c, np.nan) for h in hist], "-", color=MUTED, lw=1, alpha=0.6)
    ax.plot([], [], color=MUTED, lw=1, label="distractors (absolute, categorical, correlated, spurious)")
    if "has a hook" in sk.cands:
        ax.annotate("has a hook: implied by hook holds ring,\nso the data cannot move it; Occam keeps it out",
                    (t[-1], hist[-1][1]["has a hook"]), xytext=(-10, 14), textcoords="offset points", ha="right",
                    fontsize=7.5, color=INK2)
    for n, _, new in agent.log:
        ax.axvline(n, color=INK2, ls=":", lw=1)
        ax.text(n + 0.5, 0.5, "D grows: " + ", ".join(new), rotation=90, fontsize=8, color=INK2, va="center")
    ax.set(xlabel="trials", ylabel="P(candidate is in the essence | data)", ylim=(-0.03, 1.03),
           xlim=(0, t[-1] * 1.02), title="hook back: the essence emerges from the robot's own outcomes (seed 0)")
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1))
    fig.savefig(FIG / "model_timeline.png")
    plt.close(fig)


def fig_relational(agent, sets):
    """Absolute vs relational coordinates: tools that satisfy every other component of the essence."""
    sk = agent.skills["hook back"]
    inc = dict(zip(sk.cands, sk.model.inclusion()))
    pts = [(tr.s.gap, tr.t.width, tr.y, "train") for tr in agent.memory
           if tr.t.length >= tr.s.dist and tr.t.hook >= tr.s.ring and tr.t.rigid]
    pts += [(s.gap, t.width, succeeds(s, t), "test") for s, acts in sets["L2_ood"] for t in acts
            if t.length >= s.dist and t.hook >= s.ring and t.rigid]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
    for ph, mk, lab in (("train", "o", "training trials"), ("test", "s", "new OOD situations")):
        for ok, col, mk2 in ((True, GOOD, mk), (False, BAD, "x")):
            P = [(g, w) for g, w, y, p in pts if p == ph and y == ok]
            if P:
                g, w = np.array(P).T
                fc = col if ph == "train" else "none"
                kw = dict(marker=mk2, s=36, color=col) if mk2 == "x" else dict(marker=mk2, s=36, edgecolors=col, facecolors=fc)
                axs[0].scatter(g, w, label=f"{lab}: {'success' if ok else 'failure'}", **kw)
                axs[1].scatter(w / g, g, **kw)
    xs = np.linspace(1.5, 10.5, 2)
    axs[0].plot(xs, xs, color=SERIES[0], lw=1.5)
    axs[0].text(9.8, 9.2, "width = gap", color=SERIES[0], ha="right", fontsize=8)
    axs[0].axhline(4, color=SERIES[1], lw=1.5, ls="--")
    axs[0].text(10.4, 4.2, "width = 4 cm", color=SERIES[1], ha="right", fontsize=8)
    axs[0].axvspan(1.5, 4, color=MUTED, alpha=0.08, lw=0)
    axs[0].text(2.75, 10.3, "test range only", color=MUTED, ha="center", fontsize=8)
    axs[0].set(xlabel="gap under the sofa (cm)", ylabel="tool width (cm)", xlim=(1.5, 10.5),
               title=f"absolute: P(thinner than 4 cm) = {inc.get('thinner than 4 cm', 0):.2f}")
    axs[1].axvline(1, color=SERIES[0], lw=1.5)
    axs[1].set(xlabel="tool width / gap", ylabel="gap (cm)", xlim=(0, 3),
               title=f"relational: P(fits under gap) = {inc.get('fits under gap', 0):.2f}")
    axs[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Why the learned predicate is relational (tools that satisfy the other three components)", color=INK,
                 y=1.03)
    fig.savefig(FIG / "model_relational.png")
    plt.close(fig)


def fig_invention(agent, sets):
    s, acts = sets["invent"][0]
    _, sk = agent.skill(s)
    p = sk.model.predict(sk.X([(s, t) for t in acts]), [s.sid] * len(acts))
    names = ["broom (VLM's pick)", "umbrella", "rubber stick", "ruler", "clip", "shallow clip",
             "ruler + clip", "ruler + shallow clip", "rubber + clip", "rubber + shallow clip"]
    fig = plt.figure(figsize=(12, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], wspace=0.55)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    render(a1, s, acts[0], f"VLM's suggestion: broom, width {acts[0].width:.1f} cm > gap {s.gap:.1f} cm")
    k = int(np.argmax(p))
    render(a2, s, acts[k], f"built: {names[k]} (never built in training)")
    order = np.argsort(p)
    a3.barh(range(len(p)), p[order], color=[SERIES[0] if i == k else MUTED for i in order], height=0.6)
    a3.set(yticks=range(len(p)), yticklabels=[names[i] for i in order], xlim=(0, 1),
           xlabel="posterior predictive P(success)", title="construct-to-spec: every option scored by the essence")
    a3.grid(axis="y", visible=False)
    for i, v in enumerate(p[order]):
        a3.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    fig.savefig(FIG / "model_invention.png")
    plt.close(fig)


def fig_cards(summ_rows):
    """Learned operators (MAP structure) for the three skill families, seed 0, m = 32."""
    fig, axs = plt.subplots(1, len(summ_rows), figsize=(5.2 * len(summ_rows), 3.4))
    for ax, (title, r) in zip(np.atleast_1d(axs), summ_rows):
        ax.axis("off")
        name, s = next(iter(r["skills"].items()))
        lines = [f"skill name: {name}"]
        if s["split"] is None:
            lines += ["precondition:", "  " + "\n  ∧ ".join(s["leaves"]["None"]),
                      f"β(in spec) = {s['beta']['None'][1]:.2f}   β(out of spec) = {s['beta']['None'][0]:.2f}"]
        else:
            lines += [f"split on: {s['split']}", f"  (P(split) = {s['p_split']:.2f})"]
            for side in ("True", "False"):
                lines += [f"[{s['split']} = {side.lower()}]"]
                lines += ["   " + " ∧ ".join(s["leaves"][side][i:i + 2]) for i in range(0, len(s["leaves"][side]), 2)] \
                    if s["leaves"][side] else ["   (no further condition)"]
                lines += [f"   β(in spec) = {s['beta'][side][1]:.2f}  α(in spec) = {s['alpha'][side][1]:.2f}"]
        ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.2, color=INK, transform=ax.transAxes)
        ax.set_title(title, loc="left", color=INK)
    fig.savefig(FIG / "model_operators.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summ = json.loads((ROOT / "results" / "summary.json").read_text())
    fig_curves(summ), fig_smith(summ), fig_budget(summ), fig_gonogo(summ)
    rows = []
    for cond, title in (("base", "hook back (π = 1)"), ("varied", "hook back + push out under one name"),
                        ("hang", "hang (disjunctive)")):
        path = ROOT / "results" / "raw" / f"{cond}.jsonl"
        if path.exists():
            rows += [(title, r) for r in map(json.loads, open(path)) if r["seed"] == 0 and r["m"] == 32]
    if rows:
        fig_cards(rows)
    agent, sets, hist, snaps = replay()
    fig_timeline(agent, hist), fig_relational(agent, sets), fig_invention(agent, sets), fig_lattice(snaps)


if __name__ == "__main__":
    main()

"""Train one agent per (condition, seed) on a stream of situations; evaluate at checkpoints m (situations seen).

    python experiments/run.py --conditions base varied --seeds 10 --workers 2

Writes results/raw/<condition>.jsonl, one line per (seed, checkpoint).
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from overhyp.agent import Agent, PriorAgent, RandomAgent
from overhyp.proposer import OracleProposer
from overhyp.toolworld import actions_of, invention_set, near_miss_set, random_situation, random_tool, succeeds

CHECKPOINTS = (0, 2, 4, 8, 16, 32)
CONDITIONS = {  # proposer knobs, noise, ground-truth skill mixture, protocol
    "base": dict(),
    "varied": dict(pi=0.5),
    "no_name": dict(no_name=True),
    "hang": dict(skill="hang"),
    "noise0.2": dict(noise=0.2),
    "unrepresentable": dict(representable=False),
    "pi0.75": dict(pi=0.75),
    "noise0.1": dict(noise=0.1),
    "hang_nosplit": dict(skill="hang", splits=False),
    "informed": dict(prior="informed"),
    "misleading": dict(prior="misleading"),
    # H4: a fixed budget of 32 random trials, spread over m situations
    "budget_32x1": dict(n_trials=1, checkpoints=(32,)),
    "budget_16x2": dict(n_trials=2, checkpoints=(16,)),
    "budget_8x4": dict(n_trials=4, checkpoints=(8,)),
    "budget_4x8": dict(n_trials=8, checkpoints=(4,)),
}
ROOT = Path(__file__).resolve().parents[1]


def skill_of(rng, cfg):
    if cfg.get("skill") == "hang":
        return "hang"
    return "hook_back" if rng.random() < cfg.get("pi", 1.0) else "push_out"


def test_sets(seed, cfg, n_test=30, n_inv=10):
    """Fixed per seed (shared by all conditions with the same skill mixture): paired comparisons."""
    rng = np.random.default_rng([seed, 1])
    ood = [random_situation(rng, 10_000 + i, skill_of(rng, cfg), ood=True) for i in range(n_test)]
    iid = [random_situation(rng, 20_000 + i, skill_of(rng, cfg)) for i in range(n_test)]
    sets = {"L2_ood": [(s, near_miss_set(rng, s)) for s in ood], "L2_iid": [(s, near_miss_set(rng, s)) for s in iid]}
    if cfg.get("skill") != "hang":
        inv = [random_situation(rng, 30_000 + i, "hook_back", ood=True) for i in range(n_inv)]
        sets["invent"] = [(s, actions_of(*invention_set(rng, s))) for s in inv]
    return sets


def evaluate(solver, sets, rng, l1=()):
    out = {}
    for key, cases in list(sets.items()) + [("L1", l1)]:
        if not cases:
            continue
        res = [solver(s, a, rng) for s, a in cases]
        out[key + "_0shot"] = float(np.mean([r[0][0] for r in res]))
        out[key + "_within2"] = float(np.mean([any(r[0]) for r in res]))
        if key == "invent":  # first choice: a constructed tool? not the broom (item 0, the VLM's suggestion)?
            out["invent_built"] = float(np.mean([len(a[r[1][0]].parts) > 0 for (s, a), r in zip(cases, res)]))
            out["invent_rejects_broom"] = float(np.mean([r[1][0] != 0 for r in res]))
    return out


def structure(agent, sets):
    """Readable MAP structure of each skill and its agreement with ground truth on all test actions."""
    out = {}
    for name, sk in agent.skills.items():
        if sk.model is None or name.startswith("situation"):
            continue
        c, conj = sk.model.map_structure()
        params = sk.model.map_params()
        out[name] = {"split": None if c is None else sk.cands[c], "cands": sk.cands,
                     "leaves": {str(side): [sk.cands[i] for i in idx] for side, idx in conj.items()},
                     "beta": {str(side): [round(params[side][g][1], 3) for g in (0, 1)] for side in conj},
                     "alpha": {str(side): [round(params[side][g][0], 3) for g in (0, 1)] for side in conj},
                     "p_split": float(1 - sk.model.top[None])}
        agree = []
        for key in ("L2_ood", "L2_iid"):
            for s, acts in sets[key]:
                if agent.proposer.name(s) != name:
                    continue
                X = sk.X([(s, t) for t in acts])
                for x, t in zip(X, acts):
                    side = None if c is None else bool(x[c])
                    g = int(all(x[list(conj[side])]))
                    agree.append((params[side][g][1] > 0.5) == succeeds(s, t))
        out[name]["map_accuracy"] = float(np.mean(agree)) if agree else None
    return out


def setup(name, seed):
    """Condition config, a fresh agent, the training-stream RNG and the (seed-fixed) test sets."""
    cfg = CONDITIONS[name]
    proposer = OracleProposer(**{k: cfg[k] for k in ("pi", "no_name", "representable", "prior") if k in cfg})
    agent = Agent(proposer, np.random.default_rng([seed, 2]), noise=cfg.get("noise", 0.0), splits=cfg.get("splits", True))
    return cfg, agent, np.random.default_rng([seed, 0]), test_sets(seed, cfg)


def train_next(agent, rng, cfg, seen):
    """One more training situation: max(6, n_trials) random tools, n_trials random attempts (+ repeats)."""
    s = random_situation(rng, len(seen), skill_of(rng, cfg))
    n = cfg.get("n_trials", 2)
    agent.explore(s, [random_tool(rng) for _ in range(max(6, n))], n)
    seen.append(s)


def run(args):
    name, seed = args
    t0 = time.time()
    cfg, agent, rng, sets = setup(name, seed)
    rows, seen = [], []
    checkpoints = cfg.get("checkpoints", CHECKPOINTS)
    baselines = {"random": RandomAgent(None, agent.noise), "prior": PriorAgent(agent.proposer, None, agent.noise)}

    def solver(policy):
        def f(s, a, r):
            policy.rng = r
            return policy.solve(s, a)
        return f

    for m in range(max(checkpoints) + 1):
        if m in checkpoints:
            l1_rng = np.random.default_rng([seed, 3])
            l1 = [(s, near_miss_set(l1_rng, s)) for s in seen[:10]]
            train_rng, row = agent.rng, {"condition": name, "seed": seed, "m": m, "trials": len(agent.memory)}
            row.update(evaluate(solver(agent), sets, np.random.default_rng([seed, 4, m]), l1))
            agent.rng = train_rng
            row["skills"] = structure(agent, sets)
            row["growth"] = agent.log
            if m == checkpoints[0]:
                for b, pol in baselines.items():
                    row[b] = evaluate(solver(pol), sets, np.random.default_rng([seed, 5]), l1)
            rows.append(row)
        if m < max(checkpoints):
            train_next(agent, rng, cfg, seen)
    rows[-1]["seconds"] = round(time.time() - t0, 1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = ROOT / "results" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s) for c in a.conditions for s in range(a.first_seed, a.first_seed + a.seeds)]
    with Pool(a.workers) as pool:
        for rows in pool.imap_unordered(run, jobs):
            with open(out / f"{rows[0]['condition']}.jsonl", "a") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
            print(rows[0]["condition"], rows[0]["seed"], rows[-1]["seconds"], "s", {k: round(v, 2) for k, v in
                  rows[-1].items() if k.endswith(("0shot", "within2"))}, flush=True)


if __name__ == "__main__":
    main()

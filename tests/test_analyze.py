"""Does the analysis find a planted effect, and stay quiet when there is none?
    python -m tests.test_analyze        prints how often each verdict is reached (power / false-positive check)
    pytest tests                         quick assertions
Synthetic policies: real = a + kappa * S + (planted effect) + policy-level scatter, observed through binomial trials."""
import numpy as np
import pandas as pd

from f3.analyze import MECHANISMS, WORLDS, offsets, summarize, verdict

E_SIM, E_REAL, SD_U = 150, 100, 0.08  # scatter around the line as measured in the previous experiment


def fake(rng, planted, tasks=12, seeds=5, tau=0.02):
    """`planted`: mechanism -> mean offset from the scaling line; it varies across tasks with sd `tau`."""
    rows = []
    for t in range(tasks):
        a, kappa = rng.uniform(-0.05, 0.05), rng.uniform(0.4, 0.9)
        shift = {m: planted.get(m, 0.0) + tau * rng.standard_normal() for m in MECHANISMS} | {"scaling": 0.0}
        for seed in range(seeds):
            S_true = {"scaling": np.clip(np.array([0.05, 0.25, 0.5, 0.75, 0.9, 0.97]) + 0.07 * rng.standard_normal(6), 0.01, 0.99),
                      "dagger": rng.uniform(0.3, 0.8, 5), "self_imitation": rng.uniform(0.2, 0.6, 5),
                      "best_of_k": np.clip(np.array([0.1, 0.3, 0.55, 0.8, 0.92, 0.97]) + 0.07 * rng.standard_normal(6), 0.01, 0.99)}
            for fam, S in S_true.items():
                for step, s in enumerate(S):
                    row = dict(task=f"t{t}", cond="nom", seed=seed, family=fam, step=step, S=rng.binomial(E_SIM, s) / E_SIM)
                    for w in WORLDS:
                        theta = np.clip(a + kappa * s + shift[fam] + SD_U * rng.standard_normal(), 0.0, 1.0)
                        row[w] = rng.binomial(E_REAL, theta) / E_REAL
                    rows.append(row)
    return pd.DataFrame(rows)


def run(planted, reps, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return pd.Series([verdict(summarize(offsets(fake(rng, planted, **kw)), "nom")).split(" - ")[0]
                      for _ in range(reps)]).value_counts(normalize=True)


def test_null_is_not_called_an_effect():
    assert run({}, 12).get("GENERAL", 0) == 0


def test_planted_general_effect_is_found():
    assert run({m: -0.06 for m in MECHANISMS}, 12).get("GENERAL", 0) >= 0.75


def test_single_mechanism_effect_is_called_specific():
    assert run({"dagger": -0.08}, 12).get("SPECIFIC", 0) >= 0.75


if __name__ == "__main__":
    for name, planted, kw in [("no effect", {}, {}),
                              ("all three mechanisms -4 pts", {m: -0.04 for m in MECHANISMS}, {}),
                              ("all three mechanisms -6 pts", {m: -0.06 for m in MECHANISMS}, {}),
                              ("dagger only -6 pts", {"dagger": -0.06}, {}),
                              ("all three -6 pts, but varying a lot across tasks (sd 6)", {m: -0.06 for m in MECHANISMS}, dict(tau=0.06))]:
        print(f"{name:58s}", run(planted, 60, **kw).round(2).to_dict())

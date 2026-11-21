"""Does the analysis find a planted effect, and stay quiet when there is none?
    python -m tests.test_analyze        prints how often each verdict is reached (power / false-positive check)
    pytest tests                         quick assertions
Synthetic policies: real = a + kappa * S^gamma + (planted effect) + seed, run and policy-level scatter, observed
through binomial trials. The scaling curve is bent (gamma != 1), half of the policy scatter is shared by the three
real worlds, and a real world can be 'dead' for a task (nothing transfers there for anyone)."""
import numpy as np
import pandas as pd

from f3.analyze import MECHANISMS, WORLDS, offsets, summarize, verdict

E_SIM, E_REAL, SD_U = 150, 100, 0.08  # scatter around the curve as measured in the previous experiment
RUN_SD = {"scaling": 0.0, "dagger": 0.10, "self_imitation": 0.08, "best_of_k": 0.03}  # how much a whole feedback run
SEED_SD = 0.05  # ... and how much a seed's whole curve lands above or below the task's curve
# (both set so that the seed-to-seed noise of every family, placebo included, matches the pilot)


def fake(rng, planted, tasks=12, seeds=5, tau=0.0, dead=0.0):
    """`planted`: mechanism -> mean offset from the scaling curve; it varies across tasks with sd `tau`.
    `dead`: chance that the `obs` / `all` world of a task is dead."""
    rows = []
    for t in range(tasks):
        a, kappa, gamma = rng.uniform(-0.05, 0.05), rng.uniform(0.4, 0.9), rng.uniform(0.7, 1.5)
        shift = {m: planted.get(m, 0.0) + tau * rng.standard_normal() for m in MECHANISMS} | {"scaling": 0.0}
        live = {w: w == "act" or rng.random() >= dead for w in WORLDS}
        for seed in range(seeds):
            level = SEED_SD * rng.standard_normal()
            S_true = {"scaling": np.clip(np.array([0.05, 0.25, 0.5, 0.75, 0.9, 0.97]) + 0.07 * rng.standard_normal(6), 0.01, 0.99),
                      "dagger": rng.uniform(0.3, 0.8, 5), "self_imitation": rng.uniform(0.2, 0.6, 5),
                      "best_of_k": np.clip(np.array([0.1, 0.3, 0.55, 0.8, 0.92, 0.97]) + 0.07 * rng.standard_normal(6), 0.01, 0.99)}
            for fam, S in S_true.items():
                run = RUN_SD[fam] * rng.standard_normal()
                for step, s in enumerate(S):
                    row = dict(task=f"t{t}", cond="nom", seed=seed, family=fam, step=step, S=rng.binomial(E_SIM, s) / E_SIM)
                    shared = SD_U / np.sqrt(2) * rng.standard_normal()
                    for w in WORLDS:
                        theta = a + kappa * s**gamma + shift[fam] + level + run + shared + SD_U / np.sqrt(2) * rng.standard_normal()
                        row[w] = rng.binomial(E_REAL, np.clip(theta, 0.0, 1.0) * (1.0 if live[w] else 0.05)) / E_REAL
                    rows.append(row)
    return pd.DataFrame(rows)


def run(planted, reps, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return pd.Series([verdict(summarize(offsets(fake(rng, planted, **kw)), "nom")).split(" - ")[0]
                      for _ in range(reps)]).value_counts(normalize=True)


ALL = lambda x: {m: x for m in MECHANISMS}
SCENARIOS = [("no effect", {}, {}),
             ("no effect; obs / all worlds dead in half of the tasks", {}, dict(dead=0.5)),
             ("all three mechanisms -4 pts", ALL(-0.04), {}),
             ("all three mechanisms -6 pts", ALL(-0.06), {}),
             ("all three mechanisms -6 pts; half of the obs / all worlds dead", ALL(-0.06), dict(dead=0.5)),
             ("dagger only -6 pts", {"dagger": -0.06}, {}),
             ("all three -6 pts on average, varying across tasks (sd 6)", ALL(-0.06), dict(tau=0.06)),
             ("no average effect, direction varies across tasks (sd 10)", {}, dict(tau=0.10)),
             ("all three mechanisms +6 pts (feedback transfers better)", ALL(+0.06), {})]


def test_null_is_not_called_an_effect():
    assert run({}, 12).get("NO-GO", 0) >= 0.6


def test_planted_general_effect_is_found():
    assert run(ALL(-0.08), 12).get("GENERAL", 0) >= 0.75


def test_single_mechanism_effect_is_called_specific():
    assert run({"dagger": -0.08}, 12).get("SPECIFIC", 0) >= 0.75


def test_varying_direction_is_called_mixed():
    assert run({}, 12, tau=0.10).get("MIXED", 0) >= 0.75


if __name__ == "__main__":
    import sys
    for name, planted, kw in SCENARIOS:
        print(f"{name:66s}", run(planted, int(sys.argv[1]) if len(sys.argv) > 1 else 100, **kw).round(2).to_dict(), flush=True)

"""Checks of the Overhypothesis core against closed forms and known facts (Kemp et al. 2007)."""
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

from overhyp.overhypothesis import Kemp, Overhypothesis, conjunctions, holds

K = Kemp()


def log_ml(bags, kemp=K):
    S, F = np.array(bags).T
    return logsumexp(kemp.log_joint(S[None], F[None])[0])


def test_prior():
    p = np.exp(K.log_prior)
    assert abs(p.sum() - 1) < 1e-12
    assert abs(p @ K.beta - 0.5) < 1e-12  # grid symmetric in logit(beta)
    assert abs(p[K.alpha < 1].sum() - (1 - np.exp(-1))) < 2e-3  # Exp(1): truncation + midpoint error


def test_fact1_single_observation_is_uninformative_about_alpha():
    # one trial per bag: P = prod beta_v, independent of alpha -> P(S) = E[beta] = 1/2 exactly
    assert abs(log_ml([(1, 0)]) - np.log(0.5)) < 1e-12
    lj = K.log_joint(np.array([[1, 0, 1]]), np.array([[0, 1, 0]]))[0]
    w = np.exp(lj - logsumexp(lj))
    prior_alpha = np.exp(K.log_prior).reshape(32, 60).sum(1)
    assert np.abs(w.reshape(32, 60).sum(1) - prior_alpha).max() < 1e-12


def test_fact2_agreement_of_two_trials():
    # P(two trials of one bag agree) = E_alpha[(alpha * E[b^2 + (1-b)^2] + 1) / (alpha + 1)], E[...] = 2/3
    exact = quad(lambda a: np.exp(-a) * (a * 2 / 3 + 1) / (a + 1), 0, np.inf)[0]
    grid = np.exp(log_ml([(2, 0)])) + np.exp(log_ml([(0, 2)]))
    assert abs(grid - exact) < 2e-3


def test_sequential_recursion_equals_batch():
    rng = np.random.default_rng(0)
    trials = [(rng.integers(5), rng.random() < 0.7) for _ in range(40)]
    S, F = np.zeros(5, int), np.zeros(5, int)
    seq = 0.0
    for j, y in trials:
        w = np.exp(K.log_joint(S[None], F[None])[0] - log_ml(list(zip(S, F))))
        p = w @ K.next_success(S[j], F[j])
        seq += np.log(p if y else 1 - p)
        S[j] += y
        F[j] += 1 - y
    assert abs(seq - log_ml(list(zip(S, F)))) < 1e-9


def test_quadrature_accuracy():
    fine = Kemp(n_alpha=320, n_beta=600)
    rng = np.random.default_rng(1)
    cases = [[(5, 0)] * 50 + [(0, 5)] * 50, [(3, 2), (1, 4), (2, 2)] * 10, [(1, 1)] * 20, [(40, 1)], [(2, 0)] * 100]
    for _ in range(50):  # up to 100 bags of up to 5 trials, pure or mixed
        n = rng.integers(1, 6, size=rng.integers(1, 101))
        s = rng.binomial(n, rng.choice([0.02, 0.5, 0.98], size=len(n)))
        cases.append(list(zip(s, n - s)))
    for bags in cases:
        assert abs(log_ml(bags) - log_ml(bags, fine)) < 0.02
    # known limit: hundreds of trials with an intermediate rate make the beta posterior narrower than a cell
    bags = [(3, 2), (1, 4), (2, 2)] * 33
    assert abs(log_ml(bags) - log_ml(bags, fine)) < 0.15


def test_conjunction_matrix():
    H = conjunctions(3)
    assert H.shape == (8, 3) and not H[0].any() and H[1].tolist() == [True, False, False]
    X = np.array([[1, 1, 0], [1, 1, 1]], bool)
    assert holds(X, H[[0, 3, 7]]).tolist() == [[True, True, False], [True, True, True]]


def synthetic(rng, m=30, n=6, hidden=None, noise=0.0):
    """Hook-back-like data: essence = c0 & c1 & c2 & c3; c4 correlated with c1; c5 irrelevant."""
    X, y, sit, cfg = [], [], [], []
    for j in range(m):
        for t in range(n):
            c = rng.random(6) < [0.6, 0.6, 0.6, 0.6, 0, 0.5]
            c[4] = c[1] and rng.random() < 0.85
            ok = c[:4].all() and rng.random() >= noise
            if hidden is not None:
                ok = ok and hidden(j, t)
            X.append(c), y.append(ok), sit.append(j), cfg.append((j, t))
    return np.array(X), np.array(y), np.array(sit), cfg


def test_structure_posterior_recovers_essence():
    X, y, sit, _ = synthetic(np.random.default_rng(1))
    oh = Overhypothesis(np.full(6, -1.0)).fit(X, y, sit)
    assert oh.map_structure() == (None, {None: (0, 1, 2, 3)})
    assert oh.leaves[None, None].w.max() > 0.9
    p = oh.predict(np.array([[1, 1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 1]], bool), ["new", "new"])
    assert p[0] > 0.9 and p[1] < 0.1


def test_update_equals_refit():
    X, y, sit, _ = synthetic(np.random.default_rng(3), m=12, n=4)
    for split in (False, True):
        oh, ref = Overhypothesis(np.full(6, -1.0)), Overhypothesis(np.full(6, -1.0))
        oh.allow_split = ref.allow_split = split
        oh.fit(X[:30], y[:30], sit[:30])
        for t in range(30, 40):  # includes trials in situations unseen at the last fit
            oh.update(X[t], sit[t], y[t])
        ref.fit(X[:40], y[:40], sit[:40])
        Xq = np.random.default_rng(4).random((8, 6)) < 0.6
        sq = [sit[39], sit[35], sit[0], "new"] * 2
        assert np.abs(oh.predict(Xq, sq) - ref.predict(Xq, sq)).max() < 1e-6
        assert all(abs(oh.top[c] - ref.top[c]) < 1e-6 for c in ref.top)


def test_criticize_separates_noise_from_missing_factor():
    rng = np.random.default_rng(2)
    # missing tool-level factor: candidate c3 hidden from the learner; repeat each configuration twice
    X, y, sit, cfg = synthetic(rng, m=20, n=4)
    Xr, yr, sr, cr = (np.repeat(a, 2, axis=0) for a in (X, y, sit, np.arange(len(y))))
    oh = Overhypothesis(np.full(5, -1.0)).fit(Xr[:, [0, 1, 2, 4, 5]], yr, sr)
    assert oh.criticize(cr) > 0
    # once the factor is a candidate, the same data are explained: nothing left to criticize
    oh = Overhypothesis(np.full(6, -1.0)).fit(Xr, yr, sr)
    assert oh.criticize(cr) == 0
    # execution noise instead: essence c0..c2 fully visible, 30% random failures
    X, y, sit, cfg = synthetic(rng, m=20, n=4)
    Xr, sr, cr = (np.repeat(a, 2, axis=0) for a in (X, sit, np.arange(len(y))))
    yr = Xr[:, :3].all(1) & (rng.random(len(cr)) > 0.3)
    oh = Overhypothesis(np.full(3, -1.0)).fit(Xr[:, :3], yr, sr)
    assert oh.criticize(cr) <= 0

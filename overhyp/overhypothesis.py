"""Overhypothesis: Kemp, Perfors & Tenenbaum (2007) mapped onto the skills of an agentic robot.

Mapping (映射表（2）), for one skill:
    category i : the trials in one situation j that share the value g = h(trial) of the structure h
    value v    : outcome of a trial, success / failure (K = 2)
    for each g in {0, 1}:   alpha_g ~ Exp(lam),  beta_g ~ Dir(1, 1) = U(0, 1),
                            theta_{j,g} ~ Beta(alpha_g beta_g, alpha_g (1 - beta_g)),  y ~ Bernoulli(theta_{j,g})
    structure h: a conjunction of candidate essences (relational predicates proposed by the VLM).
                 After model criticism also a split on one candidate c into two sub-skills (STRIPS-style):
                 trials with c true / c false, each sub-skill with its own conjunction (over the other
                 candidates) and its own pair of Kemp models.

One principle: P(structure | data) ∝ P(structure) · Kemp marginal likelihood.
P(structure) is the proposer's prior: every element (a conjunct, or a split) costs the prior log-odds
of its predicate. Inference is exact model averaging over all structures; (alpha, beta) are integrated
on a fixed grid in (log alpha, logit beta).
"""
import itertools

import numpy as np
from scipy.special import expit, gammaln, log_expit

MASS = 1e-9  # structures outside the top 1 - MASS of posterior mass are dropped from prediction (error <= MASS)


def lse(x, axis=-1):
    """log-sum-exp along an axis."""
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class Kemp:
    """Kemp model with K = 2 values; (alpha, beta) integrated with the midpoint rule on a uniform grid in
    (log alpha, logit beta). Cell edges in log alpha include 0, so "alpha < 1" is a union of whole cells.
    The 32 x 60 grid is within 0.02 nats of a 320 x 600 grid on log marginal likelihoods of up to 100 bags (tests);
    a class with hundreds of trials at an intermediate rate can be off by ~0.1 nats."""

    def __init__(self, lam=1.0, n_alpha=32, n_beta=60, log_alpha=(-9.0, 7.0), logit_beta=9.0):
        mid = lambda lo, hi, n: (np.linspace(lo, hi, n + 1)[:-1] + np.linspace(lo, hi, n + 1)[1:]) / 2
        U, V = (x.ravel() for x in np.meshgrid(mid(*log_alpha, n_alpha), mid(-logit_beta, logit_beta, n_beta),
                                                 indexing="ij"))
        self.shape = (n_alpha, n_beta)
        self.alpha, self.beta = np.exp(U), expit(V)
        self.a = self.alpha * self.beta  # prior pseudo-count of successes
        self.b = self.alpha * expit(-V)  # prior pseudo-count of failures
        logp = np.log(lam) - lam * self.alpha + U + log_expit(V) + log_expit(-V)  # density in (log a, logit b)
        self.log_prior = logp - lse(logp)
        self._table = {}

    def bag_loglik(self, s, f):
        """log P(one outcome sequence with s successes, f failures | alpha, beta) at every grid node."""
        key = (int(s), int(f))
        if key not in self._table:
            a, b, al = self.a, self.b, self.alpha
            self._table[key] = gammaln(s + a) - gammaln(a) + gammaln(f + b) - gammaln(b) + gammaln(al) - gammaln(s + f + al)
        return self._table[key]

    def log_joint(self, S, F):
        """S, F: int arrays (rows, n_bags) of bag counts -> log p(data, node), shape (rows, n_nodes)."""
        S, F = np.asarray(S, np.int64), np.asarray(F, np.int64)
        code = S * (F.max() + 1) + F
        uniq, inv = np.unique(code, return_inverse=True)
        counts = np.bincount((np.arange(len(S))[:, None] * len(uniq) + inv.reshape(S.shape)).ravel(),
                             minlength=len(S) * len(uniq)).reshape(len(S), len(uniq))
        table = np.stack([self.bag_loglik(u // (F.max() + 1), u % (F.max() + 1)) for u in uniq])
        return counts @ table + self.log_prior

    def next_success(self, s, f):
        """Kemp's prediction rule (y_s + alpha beta) / (n + alpha) at every node, for bags with counts (s, f)."""
        s, f = np.asarray(s, float)[..., None], np.asarray(f, float)[..., None]
        return (s + self.a) / (s + f + self.alpha)


def conjunctions(d):
    """All 2^d conjunctions over d candidates, as a (2^d, d) boolean inclusion matrix (row 0 = empty)."""
    return np.array(list(itertools.product((False, True), repeat=d)), dtype=bool)[:, ::-1]


def holds(X, H):
    """g[t, k] = conjunction H[k] is true on trial t (all its candidates are true)."""
    return ((~X).astype(np.int32) @ H.T.astype(np.int32)) == 0


class Leaf:
    """Posterior over the conjunctions H of one (sub-)skill, from the trials `rows` of the skill.

    Conjunctions that induce the same values g on these trials have the same bag counts and evidence; they form a
    group. Only per-group bag counts are stored; grid posteriors are recomputed from them when needed."""

    def __init__(self, kemp, X, y, J, m, rows, H, log_prior):
        self.kemp, self.rows, self.H = kemp, rows, H
        G = holds(X[rows], H)  # (n, K)
        _, first, inv = np.unique(np.packbits(G, axis=0).T, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        S, N = self._counts(G[:, first], y[rows], J[rows], m)
        log_post = log_prior + sum(lse(kemp.log_joint(S[g], N[g] - S[g])) for g in (0, 1))[inv]
        w = softmax(log_post)
        order = np.argsort(-w)
        keep = order[: np.searchsorted(np.cumsum(w[order]), 1 - MASS) + 1]
        used, grp = np.unique(inv[keep], return_inverse=True)
        self.log_ev = lse(log_post)
        self.keep, self.w, self.grp = keep, w[keep] / w[keep].sum(), grp.ravel()
        self.S, self.N = {g: S[g][used] for g in (0, 1)}, {g: N[g][used] for g in (0, 1)}

    @staticmethod
    def _counts(G, y, J, m):
        """Bag counts per conjunction (column of G) and situation: successes S[g] and trials N[g], (K, m)."""
        M = np.eye(m, dtype=np.int32)[J]  # one-hot situation of each trial
        N1, S1 = G.T.astype(np.int32) @ M, (G & y[:, None]).T.astype(np.int32) @ M
        return {1: S1, 0: (M * y[:, None]).sum(0) - S1}, {1: N1, 0: M.sum(0) - N1}

    def grid(self, g):
        """Grid posterior over (alpha_g, beta_g) for every stored group, (U, nodes)."""
        return softmax(self.kemp.log_joint(self.S[g], self.N[g] - self.S[g]))

    def predict(self, Xq, jq):
        """P(success) of query trials, averaged over the conjunctions (and their (alpha, beta) posteriors)."""
        gq = holds(Xq, self.H[self.keep])  # (q, K')
        W = {g: self.grid(g) for g in (0, 1)}
        out = np.zeros(len(Xq))
        for i, j in enumerate(jq):
            for g in (0, 1):
                wu = np.bincount(self.grp, weights=self.w * (gq[i] == g), minlength=len(W[g]))
                u = np.flatnonzero(wu)
                s, n = (self.S[g][u, j], self.N[g][u, j]) if j >= 0 else (np.zeros(len(u)), np.zeros(len(u)))
                out[i] += wu[u] @ np.einsum("un,un->u", W[g][u], self.kemp.next_success(s, n - s))
        return out

    def update(self, x, j, y):
        """Kemp's sequential recursion for one trial in column j; returns log P(y | data) under this leaf."""
        g = holds(x[None], self.H[self.keep])[0].astype(int)
        new, self.grp = np.unique(self.grp * 2 + g, return_inverse=True)  # groups split by the new trial's g
        self.grp = self.grp.ravel()
        old, cls = new // 2, new % 2
        W = {gv: self.grid(gv) for gv in (0, 1)}
        p = np.empty(len(new))
        for gv in (0, 1):
            u = np.flatnonzero(cls == gv)
            s, n = self.S[gv][old[u], j], self.N[gv][old[u], j]
            rule = np.einsum("un,un->u", W[gv][old[u]], self.kemp.next_success(s, n - s))
            p[u] = rule if y else 1 - rule
        self.S, self.N = {gv: self.S[gv][old].copy() for gv in (0, 1)}, {gv: self.N[gv][old].copy() for gv in (0, 1)}
        for gv in (0, 1):
            u = np.flatnonzero(cls == gv)
            self.S[gv][u, j] += int(y)
            self.N[gv][u, j] += 1
        pk = p[self.grp]
        pred = self.w @ pk
        self.w = self.w * pk / pred
        self.log_ev += np.log(pred)
        return np.log(pred)

    def add_situation(self):
        for d in (self.S, self.N):
            for g in (0, 1):
                d[g] = np.pad(d[g], ((0, 0), (0, 1)))

    def best(self):
        """MAP conjunction (candidate indices) and posterior means of (alpha_g, beta_g) under it."""
        k = int(np.argmax(self.w))
        u = self.grp[k]
        params = {}
        for g in (0, 1):
            W = softmax(self.kemp.log_joint(self.S[g][u:u + 1], self.N[g][u:u + 1] - self.S[g][u:u + 1]))[0]
            params[g] = (W @ self.kemp.alpha, W @ self.kemp.beta)
        return tuple(np.flatnonzero(self.H[self.keep[k]])), params


class Overhypothesis:
    """Posterior over structures of one skill, given trials (X, y, situation)."""

    def __init__(self, log_odds, kemp=None):
        self.log_odds = np.asarray(log_odds, float)  # proposer's prior log-odds that each candidate is essential
        self.kemp = kemp or Kemp()
        self.allow_split = False  # switched on by model criticism (see criticize)

    # ---------------------------------------------------------------- inference
    def fit(self, X, y, sit):
        """X: (n, d) candidate values per trial; y: (n,) outcomes; sit: (n,) situation ids."""
        self.X, self.y = np.asarray(X, bool).reshape(len(y), -1), np.asarray(y, bool)
        self.sits, self.J = np.unique(np.asarray(sit), return_inverse=True)
        self.J = self.J.ravel()
        d = self.X.shape[1]
        H = conjunctions(d)
        log1m = -np.logaddexp(0, self.log_odds)  # log(1 - rho_c)
        self.leaves, self.log_top = {}, {}
        for c in [None] + (list(range(d)) if self.allow_split else []):
            Hc = H if c is None else H[~H[:, c]]  # a sub-skill's conjunction is over the other candidates
            prior = Hc @ self.log_odds + log1m.sum() - (0.0 if c is None else log1m[c])
            sides = [None] if c is None else [True, False]
            for side in sides:
                rows = np.ones(len(self.y), bool) if c is None else self.X[:, c] == side
                self.leaves[c, side] = Leaf(self.kemp, self.X, self.y, self.J, len(self.sits), rows, Hc, prior)
            self.log_top[c] = (0.0 if c is None else self.log_odds[c]) + sum(self.leaves[c, s].log_ev for s in sides)
        self._normalise()
        return self

    def _normalise(self):
        keys = list(self.log_top)
        self.top = dict(zip(keys, softmax(np.array([self.log_top[k] for k in keys]))))

    def update(self, x, sit, y):
        """Add one trial by Kemp's sequential recursion: each structure's weight is multiplied by its predictive
        probability of the outcome; bag counts are incremented. Equal to a batch fit on the structures kept at
        the last fit (dropped posterior mass <= MASS per leaf)."""
        x = np.asarray(x, bool)
        if sit not in self.sits:
            self.sits = np.append(self.sits, sit)
            for L in self.leaves.values():
                L.add_situation()
        j = int(np.flatnonzero(self.sits == sit)[0])
        self.X, self.y, self.J = np.vstack([self.X, x]), np.append(self.y, y), np.append(self.J, j)
        for (c, side), L in self.leaves.items():
            inside = c is None or x[c] == side
            L.rows = np.append(L.rows, inside)
            if inside:
                self.log_top[c] += L.update(x, j, y)
        self._normalise()
        return self

    # ---------------------------------------------------------------- prediction
    def predict(self, Xq, sitq):
        """Posterior predictive P(success) for query trials (candidate actions) Xq (q, d) in situations sitq (q,)."""
        Xq = np.asarray(Xq, bool).reshape(len(sitq), -1)
        jq = np.array([np.flatnonzero(self.sits == s)[0] if s in self.sits else -1 for s in sitq])
        p = np.zeros(len(Xq))
        for c, wc in self.top.items():
            if wc < MASS:
                continue
            for side in ([None] if c is None else [True, False]):
                q = np.ones(len(Xq), bool) if c is None else Xq[:, c] == side
                if q.any():
                    p[q] += wc * self.leaves[c, side].predict(Xq[q], jq[q])
        return p

    # ---------------------------------------------------------------- read-out
    def map_structure(self):
        """MAP structure: (split candidate or None, {side: tuple of candidate indices in the conjunction})."""
        c = max(self.top, key=self.top.get)
        return c, {s: self.leaves[c, s].best()[0] for s in ([None] if c is None else [True, False])}

    def inclusion(self):
        """Posterior P(candidate is a conjunct) in the unsplit skill."""
        L = self.leaves[None, None]
        return L.w @ L.H[L.keep]

    def map_params(self):
        """Posterior means of (alpha_g, beta_g) for each leaf of the MAP structure: {side: {g: (alpha, beta)}}."""
        c, conj = self.map_structure()
        return {side: self.leaves[c, side].best()[1] for side in conj}

    # ---------------------------------------------------------------- model criticism
    def criticize(self, X, y, config):
        """Are the outcomes the current skill structure mispredicts reproducible?

        X, y, config: all trials of the skill, including repeats (the model itself is fitted on first trials).
        Within each class (leaf, g) of the MAP structure, bags = configurations (same situation, same tool).
        Every configuration contributes its first trial (this fixes the class rate beta; Fact 1: no information
        about alpha); a configuration whose first outcome the structure mispredicts (opposite to its expected
        outcome, beta_g > 1/2) also contributes its repeats. Returns the largest log Bayes factor over classes of
        "reproducible" (alpha < 1: two trials of one configuration agree more than halfway between chance and
        certainty, Fact 2) versus "noise" (alpha >= 1); -inf if no mispredicted configuration was repeated."""
        X, y, config = np.asarray(X, bool).reshape(len(y), -1), np.asarray(y, bool), np.asarray(config)
        c, conj = self.map_structure()
        small = self.kemp.alpha < 1
        prior_odds = lse(self.kemp.log_prior[small]) - lse(self.kemp.log_prior[~small])
        params = self.map_params()
        best = -np.inf
        for side, idx in conj.items():
            rows = np.flatnonzero(np.ones(len(y), bool) if c is None else X[:, c] == side)
            g = np.all(X[rows][:, list(idx)], axis=1)
            for gv in (False, True):
                cls = rows[g == gv]
                if not len(cls):
                    continue
                expected = params[side][int(gv)][1] > 0.5
                _, first, inv = np.unique(config[cls], return_index=True, return_inverse=True)
                inv = inv.ravel()
                use = (y[cls][first] != expected)[inv] | np.isin(np.arange(len(cls)), first)
                n = np.bincount(inv[use], minlength=len(first))
                if not (n >= 2).any():
                    continue
                s = np.bincount(inv[use], weights=y[cls][use], minlength=len(first)).astype(int)
                lj = self.kemp.log_joint(s[None], (n - s)[None])[0]
                best = max(best, lse(lj[small]) - lse(lj[~small]) - prior_odds)
        return best

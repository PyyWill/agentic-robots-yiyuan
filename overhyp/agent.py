"""Meta-controller and experience memory.

The meta-controller decides *what* to do: which tool to try or build, when to repeat a configuration, when to ask
the proposer for new candidate essences. How a tool is moved is the environment's skill-level action.
Experience memory keeps raw trials (situation, tool, outcome), so a newly proposed candidate is evaluated
retroactively on all past trials.
"""
import copy
from dataclasses import dataclass

import numpy as np

from .overhypothesis import Kemp, Overhypothesis
from .proposer import LIBRARY
from .toolworld import Situation, Tool, attempt


@dataclass(frozen=True)
class Trial:
    s: Situation
    t: Tool
    y: bool
    name: str  # skill name given by the proposer


class Skill:
    """A named skill: candidate essences D (with prior log-odds) and its Overhypothesis."""

    def __init__(self, cands, log_odds, kemp):
        self.cands, self.log_odds, self.kemp = list(cands), list(log_odds), kemp
        self.allow_split = False
        self.asked = 0  # repeated configurations seen at the last query to the proposer
        self.model = None

    def X(self, pairs):
        return np.array([[LIBRARY[c](s, t) for c in self.cands] for s, t in pairs], bool).reshape(len(pairs), -1)

    def fit(self, trials):
        self.model = Overhypothesis(self.log_odds, self.kemp)
        self.model.allow_split = self.allow_split
        self.model.fit(self.X([(tr.s, tr.t) for tr in trials]), [tr.y for tr in trials], [tr.s.sid for tr in trials])
        return self.model


class Agent:
    def __init__(self, proposer, rng, noise=0.0, kemp=None, splits=True):
        self.proposer, self.rng, self.noise, self.splits = proposer, rng, noise, splits
        self.kemp = kemp or Kemp()
        self.memory = []  # experience memory
        self.skills = {}
        self.log = []  # (trial count, skill name, candidates added) for every D growth

    def skill(self, s):
        name = self.proposer.name(s)
        if name not in self.skills:
            cands = self.proposer.initial(s)
            self.skills[name] = Skill(cands, [self.proposer.log_odds(c, s) for c in cands], self.kemp)
        return name, self.skills[name]

    def trials(self, name):
        return [tr for tr in self.memory if tr.name == name]

    def firsts(self, name):
        """First trial of every configuration (situation, tool): the observations (category members) of the Kemp
        model. A repeat re-measures the same member and is kept for model criticism only."""
        seen, out = set(), []
        for tr in self.trials(name):
            if (tr.s.sid, tr.t) not in seen:
                seen.add((tr.s.sid, tr.t))
                out.append(tr)
        return out

    @staticmethod
    def _predict(sk, model, s, tools):
        if model is None:
            return np.full(len(tools), 0.5)
        return model.predict(sk.X([(s, t) for t in tools]), [s.sid] * len(tools))

    @staticmethod
    def _add(sk, model, s, t, y):
        """Sequential (Kemp) update of a model with one trial; returns the model."""
        if model is None:
            model = Overhypothesis(sk.log_odds, sk.kemp)
            model.allow_split = sk.allow_split
            return model.fit(sk.X([(s, t)]), [y], [s.sid])
        return model.update(sk.X([(s, t)])[0], s.sid, y)

    # ------------------------------------------------------------------ training: explore, repeat, criticize
    def explore(self, s, items, n_trials=2):
        """Try random tools. Each outcome updates the posterior; an outcome the skill mispredicts is repeated once
        (same situation, same tool) and the repeat goes to model criticism only. Batch refit after the situation."""
        name, sk = self.skill(s)
        for k in self.rng.permutation(len(items))[:n_trials]:
            p = self._predict(sk, sk.model, s, [items[k]])[0]
            y = attempt(s, items[k], self.rng, self.noise)
            self.memory.append(Trial(s, items[k], y, name))
            sk.model = self._add(sk, sk.model, s, items[k], y)
            if p != 0.5 and (p > 0.5) != y:
                self.memory.append(Trial(s, items[k], attempt(s, items[k], self.rng, self.noise), name))
        sk.fit(self.firsts(name))
        self.criticize(s)

    def criticize(self, s):
        """If mispredicted outcomes are reproducible, ask the proposer for new candidates and allow splits."""
        name, sk = self.skill(s)
        trials = self.trials(name)
        configs = [(tr.s.sid, tr.t) for tr in trials]
        ids = {c: i for i, c in enumerate(dict.fromkeys(configs))}
        repeated = len(configs) - len(ids)
        if repeated <= sk.asked:
            return
        X = sk.X([(tr.s, tr.t) for tr in trials])
        if sk.model.criticize(X, [tr.y for tr in trials], [ids[c] for c in configs]) <= 0:
            return
        sk.asked = repeated
        new = self.proposer.propose([tr.s for tr in trials], sk.cands, self.rng)
        sk.cands += new
        sk.log_odds += [self.proposer.log_odds(c, s) for c in new]
        sk.allow_split = self.splits
        sk.fit(self.firsts(name))
        self.log.append((len(self.memory), name, list(new)))

    # ------------------------------------------------------------------ testing: act on the posterior predictive
    def solve(self, s, actions):
        """First choice c1 (highest posterior predictive), and the choice c2 after observing that c1 failed (the
        failure is added to the situation's bag, not memorised). c2 may be c1 again: a retry is chosen when the
        model attributes the failure to noise. Returns (c1, c2)."""
        _, sk = self.skill(s)
        p = self._predict(sk, sk.model, s, actions)
        c1 = int(self.rng.choice(np.flatnonzero(p == p.max())))
        model = self._add(sk, copy.deepcopy(sk.model), s, actions[c1], False)
        p = self._predict(sk, model, s, actions)
        return c1, int(self.rng.choice(np.flatnonzero(p == p.max())))


class RandomAgent:
    """Two different random actions."""

    def __init__(self, rng):
        self.rng = rng

    def solve(self, s, actions):
        c1, c2 = self.rng.permutation(len(actions))[:2]
        return int(c1), int(c2)


class PriorAgent(RandomAgent):
    """VLM prior only, no experience: P(success | a) = P(a satisfies the essence) = prod over the candidates c
    that a violates of (1 - rho_c), under the proposer's prior over conjunctions. Its two best actions."""

    def __init__(self, proposer, rng):
        super().__init__(rng)
        self.proposer = proposer

    def solve(self, s, actions):
        cands = self.proposer.initial(s)
        log1m = np.array([-np.logaddexp(0, self.proposer.log_odds(c, s)) for c in cands])
        score = (~np.array([[LIBRARY[c](s, t) for c in cands] for t in actions], bool)).astype(float) @ log1m
        c1 = int(self.rng.choice(np.flatnonzero(score == score.max())))
        score[c1] = -np.inf
        return c1, int(self.rng.choice(np.flatnonzero(score == score.max())))

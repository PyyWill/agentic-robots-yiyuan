"""Four ways to obtain a family of policies under one data condition.

All four use the same learner, the same simulators and the same starting data. They differ only in how
simulator feedback enters the improvement:

  scaling         more teacher demonstrations                              no feedback (reference)
  dagger          teacher labels on the states the policy visits in sim    feedback decides WHERE data is collected
  self_imitation  the policy's own solved sim episodes are added           feedback decides WHICH EXPERIENCE is kept
  best_of_k       the best of K retrainings by sim score is kept           feedback decides WHICH POLICY is kept

Each function yields (step, policy).
"""
import numpy as np

from .learner import Policy, episode, successes
from .worlds import training_gap


def sim_episode(world, teacher, cond, rng, policy=None, explore=0.0):
    """One episode in a fresh simulator of data condition `cond`: new random gap under 'dr', new initial state."""
    world.set_gap(training_gap(cond, rng))
    world.reseed(rng.integers(2**31))
    return episode(world, teacher, policy, explore)


def demonstrations(world, teacher, cond, n, rng):
    """n teacher episodes -> list of (seen observations, teacher actions)."""
    return [sim_episode(world, teacher, cond, rng)[:2] for _ in range(n)]


def scaling(demos, seed, cfg):
    for m in cfg.scales:
        yield m, Policy(demos[:m], seed)


def dagger(world, teacher, cond, demos, start, seed, cfg, rng):
    data, policy = list(demos[:cfg.m0]), start
    for r in range(1, cfg.rounds + 1):
        for _ in range(cfg.k_dagger):
            X, Y_teacher, _, _ = sim_episode(world, teacher, cond, rng, policy)
            data.append((X, Y_teacher))  # teacher labels on the states the policy reached in sim
        policy = Policy(data, seed + r)
        yield r, policy


def self_imitation(world, teacher, cond, demos, start, seed, cfg, rng):
    data, policy = list(demos[:cfg.m0]), start
    for r in range(1, cfg.rounds + 1):
        for _ in range(cfg.k_self):
            X, _, Y_done, solved = sim_episode(world, teacher, cond, rng, policy, cfg.explore)
            if solved:  # keep what worked in sim, labelled with what the policy itself did
                data.append((X, Y_done))
        policy = Policy(data, seed + r)
        yield r, policy


def best_of_k(world, cond, demos, seed, cfg, rng):
    for m in cfg.scales:
        candidates = [Policy(demos[:m], seed + 1000 * k) for k in range(cfg.k_best)]  # k = 0 is the scaling policy
        tests = [(training_gap(cond, rng), rng.integers(2**31)) for _ in range(cfg.n_select)]  # one sim test set for all
        scores = [sum(successes(world, c, gap, 1, s) for gap, s in tests) for c in candidates]
        yield m, candidates[int(np.argmax(scores))]

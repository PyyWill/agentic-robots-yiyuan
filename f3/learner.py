"""The learner (plain behaviour cloning with a scikit-learn MLP) and the two rollout primitives.
Every family in `families.py` is built from exactly these three pieces."""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .worlds import T_MAX

GRAD_STEPS, MIN_EPOCHS, BATCH = 2000, 30, 256  # same optimisation budget for every policy in every family


class Policy:
    """Maps the observation the robot sees to a 4-d command in [-1, 1]. `data` is a list of (observations, actions)."""

    def __init__(self, data, seed):
        X, Y = (np.concatenate(part) for part in zip(*data))
        batch = min(BATCH, len(X))
        epochs = max(MIN_EPOCHS, int(np.ceil(GRAD_STEPS / np.ceil(len(X) / batch))))
        self.scaler = StandardScaler().fit(X)
        self.net = MLPRegressor(hidden_layer_sizes=(128, 128), batch_size=batch, max_iter=epochs, tol=0.0, n_iter_no_change=epochs,
                                random_state=int(seed)).fit(self.scaler.transform(X), Y)

    def act(self, obs):
        """Same numbers as `self.net.predict`, without scikit-learn's per-call overhead."""
        h = (obs - self.scaler.mean_) / self.scaler.scale_
        *hidden, (W_out, b_out) = zip(self.net.coefs_, self.net.intercepts_)
        for W, b in hidden:
            h = np.maximum(h @ W + b, 0.0)
        return np.clip(h @ W_out + b_out, -1.0, 1.0)


def episode(world, teacher, policy=None, explore=0.0):
    """One episode in the world's current simulator. The actor is `policy` if given, otherwise the teacher.
    -> (seen observations, teacher's action at each visited state, executed commands, solved?)"""
    obs, seen = world.reset()
    X, Y_teacher, Y_done, solved = [], [], [], False
    for _ in range(T_MAX):
        label = np.clip(teacher.get_action(obs), -1, 1)
        cmd = label if policy is None else policy.act(seen)
        cmd = np.clip(cmd + explore * world.rng.standard_normal(4) * [1, 1, 1, 0], -1, 1)
        X.append(seen), Y_teacher.append(label), Y_done.append(cmd)
        obs, seen, solved = world.step(cmd)
        if solved:
            break
    return np.array(X), np.array(Y_teacher), np.array(Y_done), solved


def successes(world, policy, gap, episodes, seed):
    """Number of solved episodes out of `episodes` fresh ones, in the world defined by `gap`."""
    world.set_gap(gap)
    world.reseed(seed)
    count = 0
    for _ in range(episodes):
        _, seen = world.reset()
        for _ in range(T_MAX):
            _, seen, solved = world.step(policy.act(seen))
            if solved:
                count += 1
                break
    return count

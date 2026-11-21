"""Sim and 'real' worlds.

A world is one Meta-World task whose physics, actuation and perception can be perturbed by a `Gap`.
  sim  = the nominal task; the sim score S is always measured here
  real = the same task under a fixed gap that no learner ever sees (stand-in for the real robot)
Nothing in this file knows about families, transfer or statistics.
"""
import warnings
from dataclasses import dataclass

import mujoco
import numpy as np

warnings.filterwarnings("ignore")
from metaworld.env_dict import ALL_V3_ENVIRONMENTS  # noqa: E402
from metaworld.policies import ENV_POLICY_MAP  # noqa: E402

T_MAX = 120  # control steps per episode
ALL_TASKS = sorted(set(ALL_V3_ENVIRONMENTS) & set(ENV_POLICY_MAP))


@dataclass(frozen=True)
class Gap:
    """What differs from the nominal task. `Gap()` is the nominal sim."""

    fric: float = 1.0  # x sliding friction of every geom
    mass: float = 1.0  # x mass and inertia of every movable non-robot body
    damp: float = 1.0  # x joint damping of every non-robot joint
    act_scale: float = 1.0  # executed displacement = act_scale * commanded
    act_delay: int = 0  # control steps of pure delay
    act_noise: float = 0.0  # sd of noise added to the executed command
    obs_noise: float = 0.0  # sd (m) of per-step noise on the observed object position
    obs_bias: tuple = (0.0, 0.0, 0.0)  # constant offset (m) of the observed object position


SIM = Gap()
REAL = {  # the three 'real' worlds, unchanged from the previous experiment
    "act": Gap(act_scale=0.7, act_delay=3, act_noise=0.2),
    "obs": Gap(obs_noise=0.008, obs_bias=(0.015, -0.015, 0.0)),
    "all": Gap(fric=0.5, mass=2.0, damp=2.0, act_scale=0.75, act_delay=2, act_noise=0.15,
               obs_noise=0.006, obs_bias=(0.012, -0.012, 0.0)),
}


def training_gap(cond, rng):
    """The simulator a learner may touch under data condition `cond`.
    'nom' = the nominal sim only.  'dr' = domain randomisation wide enough to contain every REAL world."""
    if cond == "nom":
        return SIM
    return Gap(fric=rng.uniform(0.4, 1.6), mass=rng.uniform(0.5, 2.5), damp=rng.uniform(0.5, 2.5),
               act_scale=rng.uniform(0.6, 1.1), act_delay=int(rng.integers(0, 4)), act_noise=rng.uniform(0, 0.25),
               obs_noise=rng.uniform(0, 0.010), obs_bias=(*rng.uniform(-0.02, 0.02, 2), 0.0))


def teacher(task):
    """Meta-World's own scripted controller for the task (acts on the true state)."""
    return ENV_POLICY_MAP[task]()


class World:
    """One task instance. `set_gap` switches between sim and real worlds; object and goal are re-drawn every reset."""

    def __init__(self, task, seed=0):
        env = ALL_V3_ENVIRONMENTS[task]()
        env._partially_observable = False  # goal is part of the observation ...
        env.__dict__.pop("sawyer_observation_space", None)  # ... so rebuild the cached observation bounds
        env._freeze_rand_vec, env._set_task_called, env.seeded_rand_vec = False, True, True
        self.env, m = env, env.model
        self.nominal = dict(fric=m.geom_friction.copy(), mass=m.body_mass.copy(),
                            inertia=m.body_inertia.copy(), damp=m.dof_damping.copy())
        name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        robot_roots = {b for b in range(m.nbody) if name(b) in ("base", "mocap")}
        movable_roots = {m.body_rootid[b] for b in range(m.nbody) if m.body_dofnum[b] > 0} - robot_roots
        self.obj_bodies = [b for b in range(1, m.nbody) if m.body_rootid[b] in movable_roots]
        self.obj_dofs = [d for b in self.obj_bodies for d in range(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b])]
        self.set_gap(SIM)
        self.reseed(seed)

    def reseed(self, seed):
        self.env.seed(int(seed))
        self.rng = np.random.default_rng(int(seed))

    def set_gap(self, gap):
        m, nom = self.env.model, self.nominal
        m.geom_friction[:] = nom["fric"]
        m.geom_friction[:, 0] *= gap.fric
        m.body_mass[:], m.body_inertia[:], m.dof_damping[:] = nom["mass"], nom["inertia"], nom["damp"]
        m.body_mass[self.obj_bodies] *= gap.mass
        m.body_inertia[self.obj_bodies] *= gap.mass
        m.dof_damping[self.obj_dofs] *= gap.damp
        self.gap = gap

    def reset(self):
        """-> (true observation, observation the robot sees)"""
        obs, _ = self.env.reset()
        self._pending = [np.zeros(4)] * self.gap.act_delay
        return obs, self._see(obs)

    def step(self, action):
        """-> (true observation, observation the robot sees, task solved?)"""
        action = np.clip(action, -1, 1)
        if self.gap.act_delay:
            self._pending.append(action)
            action = self._pending.pop(0)
        executed = action.copy()
        executed[:3] = self.gap.act_scale * executed[:3] + self.gap.act_noise * self.rng.standard_normal(3)
        obs, _, _, _, info = self.env.step(np.clip(executed, -1, 1))
        return obs, self._see(obs), info["success"] > 0.5

    def _see(self, obs):
        if self.gap.obs_noise == 0.0 and not any(self.gap.obs_bias):
            return obs
        seen = obs.copy()
        shift = np.asarray(self.gap.obs_bias) + self.gap.obs_noise * self.rng.standard_normal(3)
        seen[4:7] += shift  # object position, current frame
        seen[22:25] += shift  # object position, previous frame
        return seen

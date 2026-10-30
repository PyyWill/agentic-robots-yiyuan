"""Environment and protocol checks: test sets have exactly one valid option, constructions follow geometry."""
import numpy as np

from overhyp.agent import Agent
from overhyp.proposer import ESSENCE, LIBRARY, OracleProposer
from overhyp.toolworld import (actions_of, compose, invention_set, near_miss_set, random_situation, random_tool,
                               succeeds)


def test_ground_truth_is_the_listed_essence():
    rng = np.random.default_rng(0)
    for skill in ("hook_back", "push_out"):
        for _ in range(300):
            s, t = random_situation(rng, 0, skill, ood=rng.random() < 0.5), random_tool(rng)
            assert succeeds(s, t) == all(LIBRARY[c](s, t) for c in ESSENCE[skill])


def test_test_sets_have_one_valid_option():
    rng = np.random.default_rng(1)
    for skill in ("hook_back", "push_out", "hang"):
        for ood in (False, True):
            s = random_situation(rng, 0, skill, ood)
            assert sum(succeeds(s, t) for t in near_miss_set(rng, s)) == 1
    for _ in range(20):
        s = random_situation(rng, 0, "hook_back", ood=True)
        acts, labels = invention_set(rng, s)
        valid = [a for a in acts if succeeds(s, a)]
        assert len(valid) == 1 and valid[0].parts  # only a constructed tool works
        assert labels[acts.index(valid[0])] == "ruler + clip"


def test_compose():
    h, tip = random_tool(np.random.default_rng(2)), random_tool(np.random.default_rng(3))
    t = compose(h, tip)
    assert t.length == h.length + tip.length and t.width == max(h.width, tip.width) and t.hook == tip.hook
    assert t.rigid == (h.rigid and tip.rigid)


def test_agent_runs_and_grows_d():
    rng = np.random.default_rng(4)
    agent = Agent(OracleProposer(), np.random.default_rng(5))
    for j in range(12):
        agent.explore(random_situation(rng, j, "hook_back"), [random_tool(rng) for _ in range(6)])
    assert any("rigid" in new for _, _, new in agent.log)
    s = random_situation(rng, 99, "hook_back", ood=True)
    acts = near_miss_set(rng, s)
    c1, c2 = agent.solve(s, acts)
    assert 0 <= c1 < len(acts) and 0 <= c2 < len(acts)

"""Hypothesis proposer: the stand-in for the VLM. It names situations (which ones count as the same skill) and
proposes candidate essences (predicates computable from perception) with prior odds. The knobs set the quality
of grounding that the VLM provides: name consistency (pi, no-name), representability of the true essence,
prior quality, and how many spurious candidates come with every proposal."""
from dataclasses import dataclass

import numpy as np

# Candidate essences: relational predicates over (situation s, tool t), plus non-relational distractors.
LIBRARY = {
    "fits under gap": lambda s, t: t.width < s.gap,
    "reaches target": lambda s, t: t.length >= s.dist,
    "hook holds ring": lambda s, t: t.hook >= s.ring,
    "rigid": lambda s, t: t.rigid,
    "flat tip": lambda s, t: t.hook == 0,
    "loop fits fixture": lambda s, t: t.loop > s.fixture,
    "clip fits fixture": lambda s, t: t.clip >= s.fixture,
    # distractors: absolute thresholds, categorical proxies, correlated and irrelevant attributes
    "thinner than 4 cm": lambda s, t: t.width < 4,
    "longer than 40 cm": lambda s, t: t.length >= 40,
    "has a hook": lambda s, t: t.hook > 0,
    "has a loop": lambda s, t: t.loop > 0,
    "has a clip": lambda s, t: t.clip > 0,
    "loop wider than 3 cm": lambda s, t: t.loop > 3,
    "metal": lambda s, t: t.material == "metal",
    "red": lambda s, t: t.color == "red",
    "blue": lambda s, t: t.color == "blue",
    "plastic": lambda s, t: t.material == "plastic",
    "wider than 2 cm": lambda s, t: t.width > 2,
}
# What the VLM proposes up front for a scene, and the true components it can supply when asked (D growth).
INITIAL = {
    "sofa": ["fits under gap", "reaches target", "hook holds ring", "has a hook", "thinner than 4 cm",
             "longer than 40 cm", "metal", "red"],
    "rack": ["loop fits fixture", "clip fits fixture", "has a loop", "has a clip", "loop wider than 3 cm", "metal",
             "red", "rigid"],
}
ESSENCE = {
    "hook_back": ["fits under gap", "reaches target", "hook holds ring", "rigid"],
    "push_out": ["fits under gap", "reaches target", "rigid", "flat tip"],
    "hang": ["loop fits fixture", "clip fits fixture"],
}
SPURIOUS = ["blue", "plastic", "wider than 2 cm"]
SCENE = {"hook_back": "sofa", "push_out": "sofa", "hang": "rack"}


@dataclass
class OracleProposer:
    pi: float = 1.0  # name consistency: P(a "hook back"-named situation is truly hook_back); the rest are push_out
    no_name: bool = False  # no skill names: every situation is its own category, nothing is pooled
    representable: bool = True  # whether the missing true component ("rigid") can ever be proposed
    prior: str = "flat"  # flat | informed | misleading
    spurious: int = 1  # irrelevant candidates added to every proposal made on request

    def name(self, s):
        if self.no_name:
            return f"situation {s.sid}"
        return "hang" if s.skill == "hang" else "hook back"

    def initial(self, s):
        """Candidate essences proposed when a skill is first named (scene-level vocabulary)."""
        return list(INITIAL[SCENE[s.skill]])

    def propose(self, situations, have, rng):
        """Asked after reproducible mispredictions in these situations: the oracle looks at the scenes (it knows
        their ground-truth skills) and returns one true component not yet in D (if representable), plus
        `spurious` irrelevant candidates."""
        skills = sorted({s.skill for s in situations})
        true = [c for k in skills for c in ESSENCE[k] if c not in have and (self.representable or c != "rigid")]
        junk = [c for c in SPURIOUS if c not in have]
        out = list(dict.fromkeys(true))[:1]
        return out + [str(c) for c in rng.permutation(junk)[: self.spurious]]

    def log_odds(self, cand, s):
        """Prior log-odds that a candidate is part of the essence (the VLM's ranking)."""
        essential = any(cand in ESSENCE[k] for k in ESSENCE if SCENE[k] == SCENE[s.skill])
        rho = {"flat": 0.25, "informed": 0.5 if essential else 0.125, "misleading": 0.125 if essential else 0.5}
        p = rho[self.prior]
        return float(np.log(p / (1 - p)))

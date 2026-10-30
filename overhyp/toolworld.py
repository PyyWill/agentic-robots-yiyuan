"""ToolWorld-2D: a side-view tool-use world with object-centric state, deterministic geometry and skill-level actions.

Ground-truth skills (hidden from the agent; the agent only sees candidate predicates and outcomes):
    hook_back : pull a ring out from under a sofa   width < gap  and length >= dist and hook >= ring and rigid
    push_out  : push the ring out from behind        width < gap  and length >= dist and rigid and hook == 0
    hang      : hang a bag on a fixture               loop > fixture  or  clip >= fixture
Execution noise turns a success into a failure with probability `noise`.
"""
from dataclasses import dataclass, replace

import numpy as np

COLORS = ("red", "blue", "green", "yellow")


@dataclass(frozen=True)
class Tool:
    width: float  # cm, shaft cross-section (height in the side view)
    length: float  # cm
    hook: float  # cm, hook depth at the tip (0 = flat tip)
    rigid: bool
    color: str
    material: str
    loop: float = 0.0  # cm, opening of a loop (hang)
    clip: float = 0.0  # cm, opening of a clip (hang)
    parts: tuple = ()  # the two parts, if the tool was constructed


@dataclass(frozen=True)
class Situation:
    sid: int
    skill: str  # ground truth, hidden from the agent
    gap: float = 0.0
    dist: float = 0.0
    ring: float = 0.0
    fixture: float = 0.0


def succeeds(s: Situation, t: Tool) -> bool:
    """Deterministic ground-truth geometry."""
    if s.skill == "hook_back":
        return t.width < s.gap and t.length >= s.dist and t.hook >= s.ring and t.rigid
    if s.skill == "push_out":
        return t.width < s.gap and t.length >= s.dist and t.rigid and t.hook == 0
    if s.skill == "hang":
        return t.loop > s.fixture or t.clip >= s.fixture
    raise ValueError(s.skill)


def attempt(s: Situation, t: Tool, rng, noise=0.0) -> bool:
    return succeeds(s, t) and rng.random() >= noise


def compose(handle: Tool, tip: Tool) -> Tool:
    """Construct a tool from a handle and a tip (e.g. a ruler and a binder clip -> an L-hook)."""
    return Tool(width=max(handle.width, tip.width), length=handle.length + tip.length, hook=tip.hook,
                rigid=handle.rigid and tip.rigid, color=handle.color, material=handle.material, parts=(handle, tip))


# ------------------------------------------------------------------------------------------- sampling
def random_tool(rng) -> Tool:
    rigid = bool(rng.random() < 0.65)
    material = rng.choice(["metal", "wood", "plastic"], p=[0.5, 0.3, 0.2]) if rigid else \
        rng.choice(["metal", "rubber", "plastic"], p=[0.3, 0.4, 0.3])  # soft metal = wire
    hook = 0.0 if rng.random() < 0.35 else rng.uniform(0.3, 6)
    kind = rng.choice(["loop", "clip", "none"], p=[0.45, 0.45, 0.1])
    return Tool(width=rng.uniform(0.5, 8), length=rng.uniform(10, 90), hook=hook, rigid=rigid,
                color=str(rng.choice(COLORS)), material=str(material),
                loop=rng.uniform(0.5, 7) if kind == "loop" else 0.0, clip=rng.uniform(0.5, 6) if kind == "clip" else 0.0)


def random_situation(rng, sid, skill, ood=False) -> Situation:
    """Training range; ood=True draws new relational values (narrower gaps, farther targets, larger rings/fixtures)."""
    if skill == "hang":
        return Situation(sid, skill, fixture=rng.uniform(4, 6) if ood else rng.uniform(0.5, 4))
    if ood:
        return Situation(sid, skill, gap=rng.uniform(2, 4), dist=rng.uniform(50, 80), ring=rng.uniform(3, 5))
    return Situation(sid, skill, gap=rng.uniform(4, 10), dist=rng.uniform(15, 50), ring=rng.uniform(0.5, 3))


def near_miss_set(rng, s: Situation):
    """Exactly one valid tool; every other tool violates exactly one ground-truth component (chance = 1 / len).

    Sofa scenes hold the same six tool types whatever the hidden skill: a hooked tool and a flat tool that are both
    thin, long and rigid (one is right for hook_back, the other for push_out), and the right tool made too thick,
    too short or soft, plus a hook too small for the ring."""
    t = random_tool(rng)
    if s.skill == "hang":
        t = replace(t, loop=s.fixture + rng.uniform(0.2, 2), clip=0.0) if rng.random() < 0.5 else \
            replace(t, clip=s.fixture + rng.uniform(0, 2), loop=0.0)
        misses = [replace(t, loop=rng.uniform(0.2, 0.95) * s.fixture, clip=0.0),
                  replace(t, clip=rng.uniform(0.2, 0.95) * s.fixture, loop=0.0), replace(t, loop=0.0, clip=0.0)]
    else:
        t = replace(t, width=rng.uniform(0.3, s.gap - 0.1), length=s.dist + rng.uniform(0, 15), rigid=True,
                    material="wood" if t.material == "rubber" else t.material)
        hooked, flat = replace(t, hook=s.ring + rng.uniform(0, 2)), replace(t, hook=0.0)
        t, other = (flat, hooked) if s.skill == "push_out" else (hooked, flat)
        misses = [other,
                  replace(t, width=s.gap + rng.uniform(0, 1.5)),  # still "thin" in absolute terms when gap < 4
                  replace(t, length=s.dist - rng.uniform(1, 15)),  # still "long" in absolute terms when dist > 50
                  replace(t, rigid=False, material=str(rng.choice(["metal", "rubber"]))),  # soft (maybe wire)
                  replace(t, hook=rng.uniform(0.3, 0.9) * s.ring)]  # a hook, but too small for the ring
    misses = [replace(m, color=str(rng.choice(COLORS))) for m in misses]
    items = [t] + misses
    rng.shuffle(items)
    assert sum(succeeds(s, x) for x in items) == 1
    return items


def invention_set(rng, s: Situation):
    """No single item works; exactly one handle + tip construction does.
    Returns the action space (items, tips, constructions) in random order and a label for each action."""
    assert s.skill == "hook_back"
    good_tip = Tool(1.0, 4.0, s.ring + rng.uniform(0.3, 1.5), True, "blue", "metal")  # binder-clip hook
    shallow_tip = Tool(1.0, 4.0, rng.uniform(0.2, 0.8) * s.ring, True, "blue", "metal")
    good_handle = Tool(rng.uniform(0.3, min(s.gap, 3.0) - 0.15), s.dist + rng.uniform(-3.5, 6), 0.0, True,
                       str(rng.choice(COLORS)), "wood")  # ruler: thin, rigid, long enough with the tip
    soft_handle = replace(good_handle, rigid=False, material="rubber", length=s.dist + 8)
    broom = Tool(s.gap + rng.uniform(1, 4), s.dist + 20, s.ring + 2, True, "red", "metal")  # the VLM's suggestion
    umbrella = Tool(rng.uniform(0.5, s.gap - 0.1), s.dist - rng.uniform(5, 20), s.ring + 2, True, "red", "metal")
    actions = actions_of([broom, umbrella, soft_handle, good_handle], [good_handle, soft_handle], [good_tip, shallow_tip])
    labels = ["broom (VLM's pick)", "umbrella", "rubber stick", "ruler", "clip", "shallow clip", "ruler + clip",
              "ruler + shallow clip", "rubber + clip", "rubber + shallow clip"]
    assert [succeeds(s, a) for a in actions].count(True) == 1 and succeeds(s, actions[6])
    order = rng.permutation(len(actions))
    return [actions[i] for i in order], [labels[i] for i in order]


def actions_of(items, handles=(), tips=()):
    """Skill-level action space: use an item, or construct a tool from a handle and a tip."""
    return list(items) + list(tips) + [compose(h, t) for h in handles for t in tips]


def render(ax, s: Situation, t: Tool, title=""):
    """Side view of a sofa scene with a tool slid under it (cm). Red = the tool does not fit, is too short, or bends."""
    import matplotlib.patches as P
    top = s.gap + 12
    ax.add_patch(P.Rectangle((0, s.gap), s.dist + 25, 12, color="#b9a58f"))  # sofa body above the gap
    ax.add_patch(P.Rectangle((s.dist + 25, 0), 3, top, color="#8a7560"))  # back of the sofa
    ax.add_patch(P.Circle((s.dist, s.ring / 2), s.ring / 2, fill=False, lw=2, color="#d4a017"))  # ring target
    ax.plot([-15, s.dist + 35], [0, 0], color="k", lw=1)
    col = "#2e7d32" if succeeds(s, t) else "#c62828"
    ax.add_patch(P.Rectangle((0, 0.2), t.length, t.width, color=col, alpha=0.8 if t.rigid else 0.35))  # held at x=0
    if t.hook > 0:
        ax.add_patch(P.Rectangle((t.length - 0.8, 0.2), 0.8, max(t.hook, t.width), color=col))
    if t.parts:  # mark the joint of a constructed tool
        ax.plot([t.parts[0].length] * 2, [0, t.width + 0.6], color="k", lw=1.5)
    ax.set(xlim=(-15, s.dist + 35), ylim=(-1, top + 2), aspect="equal", xticks=[], yticks=[], title=title)

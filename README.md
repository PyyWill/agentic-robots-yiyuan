# Grounded generalization for agentic robots — an Overhypothesis prototype

A robot learns the **essence** of each skill from its own successes and failures, the hierarchical Bayesian
way (Kemp, Perfors & Tenenbaum 2007), and uses it for second-order (L2) generalization (Smith et al. 2002):
0/1-shot success in new situations, and building tools it has never built.

```
generalization = f( grounding( VLM names & candidate essences , experience ) )
```

The VLM proposes (skill names, candidate predicates, prior odds); the robot's outcomes decide (posterior over
structures); generalization is read out (posterior predictive). Nothing is trained by gradient descent; the only
learned object is the Overhypothesis posterior.

<!-- RESULTS -->

## 1. The model

**Types.**

| symbol | type | meaning |
|---|---|---|
| s | name | skill, as named by the proposer (VLM) |
| j | situation | one task instance of skill s (a scene with its target) |
| a | tool | an item, or a tool constructed from two parts |
| t = (j, a, y) | trial | y ∈ {0, 1} success |
| c ∈ D | (situation, tool) → {0, 1} | candidate essence: a relational predicate proposed by the VLM, e.g. *tool width < gap* |
| h ⊆ D | structure | conjunction; g_h(t) = ∧_{c∈h} c(t) ∈ {0, 1} |
| (c; h₊, h₋) | structure | split into two sub-skills (STRIPS-style) by candidate c, each with its own conjunction |
| ρ_c | (0, 1) | proposer's prior probability that c is essential |

**Mapping (映射表（2）).** For each skill and each structure, each class g ∈ {0, 1} (and each sub-skill after a
split) is one Kemp model; its categories are the situations:

```
alpha_g ~ Exp(1),  beta_g ~ U(0,1) = Dir(1,1),  theta_{j,g} ~ Beta(alpha_g beta_g, alpha_g (1 - beta_g)),
y_t ~ Bernoulli(theta_{j(t), g_h(t)})
```

β_g is the typical success rate of class g across situations, α_g how close the situations' rates θ_{j,g} are
to 0 or 1 (α small: each situation all-or-none).

**One principle: the structure posterior.**

```
P(h | Y) ∝ P(h) · Π_g ∫∫ Π_j B(S_jg + α_g β_g, F_jg + α_g(1-β_g)) / B(α_g β_g, α_g(1-β_g)) dP(α_g) dP(β_g)
P(h) = Π_c ρ_c^[c∈h] (1-ρ_c)^[c∉h];   split on c: odds(ρ_c) · P(h₊) · P(h₋)
```

S_jg, F_jg are successes and failures in situation j and class g. Necessity, sufficiency and minimality of the
essence all come out of this one score: a missing component mixes failures into the in-spec class, an extra one
pushes successes out of it, and a redundant one only costs its prior.

**Prediction and decision.** Kemp's prediction rule, averaged over structures and (α, β):

```
P(y = 1 | j, a, Y) = Σ_h P(h | Y) · E[(S_{j,g} + α_g β_g) / (n_{j,g} + α_g) | Y, h],   g = g_h(j, a)
```

In a new situation the bag is empty and this is E[β_g] (0-shot); after one failure it is updated within the
situation (1-shot; α decides how much one failure says about the rest of the situation). The robot takes the
action with the highest predictive success among items **and constructions** (construct-to-spec).

**Model criticism → D growth and splits.** Training tries random tools; an outcome the skill mispredicts is
repeated once. After each situation, configuration-level Kemp models are fitted inside each class of the MAP
structure (bags = same situation and same tool; only mispredicted configurations contribute their repeats). If the
Bayes factor favours *reproducible* (α < 1: repeats agree more than halfway between chance and certainty, Fact 2)
over *noise* (α ≥ 1), the factor behind the misprediction is missing: the robot asks the proposer for new
candidates and allows splits. Execution noise does not trigger this; a missing component (e.g. rigidity) does.
New candidates are evaluated retroactively on the experience memory.

**Inference.** Exact model averaging over all 2^|D| conjunctions (and all splits, once allowed); (log α, logit β)
on a 32 × 60 midpoint grid (≤ 0.02 nats from a 320 × 600 grid on up to 100 bags). Within a situation the posterior
is updated by Kemp's sequential recursion (tested equal to a batch refit), and refitted in batch after each
situation. Conjunctions that induce the same class labels on the data share evidence and are stored once.

## 2. Code map (the four components of the framework diagram)

| component | file | role |
|---|---|---|
| hypothesis proposer (VLM) | `overhyp/proposer.py` | names situations, proposes candidate predicates with prior odds; knobs set grounding quality |
| experience memory | `overhyp/agent.py` (`Agent.memory`) | raw trials; new candidates are evaluated retroactively |
| **Overhypothesis** | `overhyp/overhypothesis.py` | the only learned part: Kemp model, structure posterior, prediction, criticism |
| meta-controller | `overhyp/agent.py` | *what* to do: explore / repeat / ask the proposer / choose or build |
| environment | `overhyp/toolworld.py` | ToolWorld-2D: object-centric state, deterministic geometry, skill-level actions |

`tests/` checks the math against closed forms (Fact 1, Fact 2, recursion = batch, quadrature) and the environment.

## 3. Test world: ToolWorld-2D

Side-view scenes (KinDER-style: object-centric state, deterministic geometry, skill-level actions; `toolworld.render`).
Ground truth, hidden from the learner:

| skill | success iff |
|---|---|
| hook_back | width < gap ∧ length ≥ dist ∧ hook ≥ ring ∧ rigid |
| push_out | width < gap ∧ length ≥ dist ∧ rigid ∧ flat tip |
| hang | loop > fixture ∨ clip ≥ fixture |

The proposer's initial candidates for a sofa scene: 3 true relational components, plus *has a hook* (categorical
proxy), *thinner than 4 cm*, *longer than 40 cm* (absolute thresholds), *metal* (correlated with rigid), *red*
(irrelevant). *rigid* is **not** proposed up front; it can only enter through D growth. Every proposal made on
request also carries one spurious candidate.

Training: random situations from the training range, 6 random tools, 2 random tries (+ repeats). Tests use new
situations **out of the training range** (narrower gaps 2–4 cm vs 4–10, farther targets 50–80 vs 15–50 cm, larger
rings 3–5 vs 0.5–3 cm), so an absolute or categorical proxy fails where the relational predicate holds. Each test
offers 6 tools with exactly one valid; the others each violate one component (a flat tool that is right for
push_out, too thick, too short, soft, a hook too small). Chance: 1/6 at the first attempt, 1/3 within two.
Invention: no single item works (the VLM's pick, a broom, is too thick); exactly one of 4 handle + tip
constructions does.

Knobs (`experiments/run.py: CONDITIONS`): name consistency π (`pi0.75`, `varied` = π 0.5: hook_back and push_out
under one name), `no_name` (each situation its own category), `unrepresentable` (rigid never proposed), prior
quality (`informed`, `misleading`), execution noise (`noise0.1`, `noise0.2`), budget arms for breadth vs depth,
`hang` and its no-split ablation.

## Deviations from the discussed framework, approximations, limits

- **Repeats after any misprediction**, not only unexpected failures: a disjunction shows up as unexpected
  *successes*, so they must be replicated too.
- **Criticism counts only mispredicted repeats.** With all repeats, a class whose outcomes are all expected (e.g.
  out-of-spec failures) gets a spurious α < 1 vote from the uniform β prior (bounded, ≈ log 2 nats).
- **Splits are one candidate deep** (two sub-skills), allowed only after the first positive criticism. Kemp's kinds
  over arbitrary partitions are not implemented: a partition must be definable by a candidate.
- **Split prior** = the prior odds of its predicate. A single split structure is penalised (a second conjunction
  must be specified), but the total prior mass on "some split" grows with |D|.
- **The proposer is an oracle**: when asked, it returns the true missing component (if representable) plus a spurious
  one. Proposing is the VLM's job; this prototype tests what experience does with the proposals. A Claude proposer
  on rendered scenes is the next step.
- **Level 3** (λ shared across skills) is not implemented; λ = 1 as in Kemp et al.
- **Numerics**: quadrature error ≤ 0.02 nats on up to 100 bags (≈ 0.1 for a class with hundreds of trials at an
  intermediate rate); structures outside the top 1 − 10⁻⁹ posterior mass of a leaf are dropped from prediction;
  within a situation the posterior is updated on the kept structures and refitted in batch afterwards.
- **Exploration is random**; the tests are diagnostic sets, not natural scenes; noise only turns successes into
  failures.

## Reproduce

```
pip install -e .[dev,baseline]
python -m pytest -q tests/                                     # math + environment checks (~2 s)
python experiments/run.py --seeds 10 --workers 2                # all conditions -> results/raw/*.jsonl
OPENROUTER_API_KEY=... python -m experiments.claude_baseline --m 32 --seeds 10
python -m experiments.analyze && python -m experiments.figures  # results/summary.json, results/figures/*.png
```

# Go/no-go criterion (fixed before running the Claude baseline on the noise and varied conditions)

**Question.** Given *identical* experience, does the Overhypothesis model generalize to new situations (L2) better
than a frontier LLM that reads the same experience as text?

**Baseline.** Claude text-memory: `anthropic/claude-fable-5.1` (the strongest Claude on OpenRouter, Oct 2026),
prompted with the same skill name, the same candidate predicates (including those added by D growth), the same
predicate values and outcomes of every training trial, and the same test options; before its second choice it is
told which option failed. Script: `experiments/claude_baseline.py`.

**Setting.** Training stream replayed with the same seeds; m ∈ {8, 32} training situations; 10 seeds; 30 out-of-
distribution test situations per seed (6 options each: chance 1/6 at the first attempt, 1/3 within two).

**Conditions.** `base` (clean), `noise0.2` (an attempt that should succeed fails with probability 0.2, in training
and test), `varied` (π = 0.5: the name "hook back" covers hook_back and push_out situations equally often).

**Primary metric.** L2 success within two attempts (the proposal's "0/1-shot success in new situations");
secondary: success at the first attempt.

**GO** iff, at m = 32, in *both* `noise0.2` and `varied`: mean over seeds of (ours − Claude) ≥ 0.15 on the primary
metric, and ours > Claude in at least 8 of 10 seeds. Otherwise **NO-GO**. `base` and m = 8 are reported without a
criterion.

**Disclosure.** Before fixing this, we had seen our own pilot (1 seed: varied within-two 1.0 at m = 32, 0.77 at
m = 16) and the Claude pilot on 4 clean cases (4/4 correct). No Claude result on `noise0.2` or `varied` had been seen.

## Amendment 1 (protocol, before any re-run; the criterion above is unchanged)

The first protocol excluded the failed option from the second choice. Under execution noise this is an ad-hoc rule
that caps "within two attempts" for every agent (retrying the right tool is the rational move), and it is not the
specified policy (act on the posterior predictive). New protocol, identical for all agents: the first choice c1;
then the choice c2 after observing that c1 failed, which **may repeat c1** (Claude is told it may retry). Metrics
are computed in expectation over execution noise instead of by sampling it (Rao-Blackwellised): with v_i = 1 if
c_i is valid and q = 1 − ε, first attempt = v1·q, within two = v1·q + (1 − v1·q)·v2·q. In noise-free conditions
this equals the sampled outcome.

Seen before this amendment: our results under the old protocol (base, varied, no_name, hang, and part of noise0.2
and unrepresentable), and Claude under the old protocol on base (m = 8, 10 seeds) and noise0.2 (m = 8, 2 seeds).
No Claude result at m = 32 and none on varied had been seen. All runs are repeated under the new protocol.

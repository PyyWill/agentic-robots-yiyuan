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

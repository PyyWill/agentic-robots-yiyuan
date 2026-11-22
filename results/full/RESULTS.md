policies: 2640   tasks: 12   seeds per task: 5
(task, real world) pairs analysed (some family reaches 20% real success): {'dr': 36, 'nom': 32} of 36 per condition

### data condition: nom   (points of real success at equal sim score; negative = worse than scaling)

| | effect | conf. interval | prediction interval (new task) | between-task sd | p (same in every task) | tasks with effect < 0 |
|---|---|---|---|---|---|---|
| average | +0.3 | [-2.9, +3.6] | [-9.0, +9.7] | 3.9 | 0.015 | 7/12 |
| dagger | -0.2 | [-8.1, +7.6] | [-20.2, +19.7] | 8.5 | 0.000 | 9/12 |
| self_imitation | +1.4 | [-4.3, +7.2] | [-11.5, +14.3] | 5.4 | 0.018 | 3/12 |
| best_of_k | +0.0 | [-2.7, +2.7] | [-4.0, +4.0] | 1.5 | 0.269 | 6/12 |
| placebo | -0.0 | [-0.9, +0.9] | [-0.7, +0.6] | 0.0 | 1.000 | 5/12 |

**VERDICT: MIXED - how a sim score was obtained matters, but not as a deficit: below the curve = none, above = none, differs between tasks = ['dagger']**

robustness (straight line instead of the curve): average -0.2 [-3.2, +2.7]  ->  MIXED

robustness (no floor rule): average +0.6 [-2.3, +3.4]  ->  MIXED

### data condition: dr   (points of real success at equal sim score; negative = worse than scaling)

| | effect | conf. interval | prediction interval (new task) | between-task sd | p (same in every task) | tasks with effect < 0 |
|---|---|---|---|---|---|---|
| average | -0.3 | [-1.7, +1.0] | [-4.6, +3.9] | 1.8 | 0.002 | 7/12 |
| dagger | -1.8 | [-5.8, +2.2] | [-12.3, +8.6] | 4.5 | 0.000 | 9/12 |
| self_imitation | +0.0 | [-2.1, +2.1] | [-4.2, +4.3] | 1.8 | 0.058 | 4/12 |
| best_of_k | +0.8 | [-0.2, +1.9] | [-0.0, +1.7] | 0.0 | 0.610 | 3/12 |
| placebo | +0.2 | [-0.7, +1.2] | [-0.5, +0.9] | 0.0 | 0.980 | 6/12 |

### nominal condition, by real world (mean over the analysed tasks, points)

| world   |   dagger |   self_imitation |   best_of_k |   placebo |   tasks |
|:--------|---------:|-----------------:|------------:|----------:|--------:|
| act     |      3.1 |             -2.3 |         0.9 |      -0.3 |      12 |
| all     |     -3.5 |              2.2 |         0   |       0.2 |       9 |
| obs     |      0   |              7.5 |        -0.2 |       0.3 |      11 |

### nominal condition, by task (points)

| task                 |   dagger |   self_imitation |   best_of_k |   placebo |
|:---------------------|---------:|-----------------:|------------:|----------:|
| button-press-wall-v3 |     -7.7 |            -15.7 |        -3.6 |      -1.2 |
| coffee-pull-v3       |     27.8 |              3.2 |        -0.5 |       0.2 |
| coffee-push-v3       |     -5.7 |              1.8 |        -4   |       0.7 |
| dial-turn-v3         |     -2.9 |              1   |        -5.5 |      -2.6 |
| hand-insert-v3       |     -6.6 |              3.5 |         1.3 |       0   |
| handle-press-side-v3 |     -3.4 |             -2.5 |        -1.1 |      -0   |
| pick-place-v3        |     -2.2 |              0.2 |        -1.5 |      -0.3 |
| push-v3              |     -5.2 |             -4.5 |         2   |       0.9 |
| push-wall-v3         |     -4.9 |              4.4 |         0.9 |       0.7 |
| reach-v3             |      6.6 |              5.3 |         2.4 |      -0.3 |
| stick-push-v3        |     -0.2 |             13.2 |         5.3 |       1   |
| sweep-into-v3        |      1.7 |              7.1 |         4.2 |       0.5 |

### where the families sit (mean success rates)

|                           |    S |   act |   obs |   all |
|:--------------------------|-----:|------:|------:|------:|
| ('dr', 'best_of_k')       | 0.53 |  0.42 |  0.46 |  0.38 |
| ('dr', 'dagger')          | 0.43 |  0.35 |  0.36 |  0.3  |
| ('dr', 'scaling')         | 0.48 |  0.38 |  0.42 |  0.34 |
| ('dr', 'self_imitation')  | 0.33 |  0.27 |  0.32 |  0.25 |
| ('nom', 'best_of_k')      | 0.71 |  0.53 |  0.32 |  0.29 |
| ('nom', 'dagger')         | 0.64 |  0.5  |  0.29 |  0.22 |
| ('nom', 'scaling')        | 0.64 |  0.47 |  0.3  |  0.26 |
| ('nom', 'self_imitation') | 0.53 |  0.36 |  0.3  |  0.23 |

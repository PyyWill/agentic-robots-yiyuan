policies: 264   tasks: 4   seeds per task: 3
(task, real world) pairs analysed (some family reaches 20% real success): {'nom': 12} of 12 per condition

### data condition: nom   (points of real success at equal sim score; negative = worse than scaling)

| | effect | conf. interval | prediction interval (new task) | between-task sd | p (same in every task) | tasks with effect < 0 |
|---|---|---|---|---|---|---|
| average | +2.1 | [-4.7, +9.0] | [-11.7, +16.0] | 2.4 | 0.303 | 2/4 |
| dagger | +5.6 | [-17.3, +28.4] | [-30.1, +41.2] | 6.8 | 0.180 | 1/4 |
| self_imitation | +0.9 | [-13.2, +15.0] | [-17.8, +19.6] | 3.2 | 0.299 | 1/4 |
| best_of_k | -0.0 | [-11.0, +11.0] | [-17.5, +17.4] | 3.4 | 0.163 | 2/4 |
| placebo | +0.0 | [-3.0, +3.0] | [-2.2, +2.2] | 0.0 | 0.993 | 2/4 |

**VERDICT: NOT ESTABLISHED - no effect detected, but an effect of practical size cannot be excluded**

robustness (straight line instead of the curve): average +2.6 [-3.1, +8.2]  ->  NOT ESTABLISHED

robustness (no floor rule): average +2.1 [-4.7, +9.0]  ->  NOT ESTABLISHED

### nominal condition, by real world (mean over the analysed tasks, points)

| world   |   dagger |   self_imitation |   best_of_k |   placebo |   tasks |
|:--------|---------:|-----------------:|------------:|----------:|--------:|
| act     |      6.4 |             -3.3 |         2.2 |       0.2 |       4 |
| all     |      3.1 |             -0.6 |        -2.1 |       0.1 |       4 |
| obs     |      7.1 |              6.7 |        -0.3 |      -0.2 |       4 |

### nominal condition, by task (points)

| task                 |   dagger |   self_imitation |   best_of_k |   placebo |
|:---------------------|---------:|-----------------:|------------:|----------:|
| button-press-wall-v3 |     -3.4 |             -7   |         2.1 |       0.9 |
| coffee-pull-v3       |     18.5 |              0.3 |        -0.3 |      -0.8 |
| coffee-push-v3       |      6   |              6   |         4.2 |      -1   |
| dial-turn-v3         |      1.1 |              4.4 |        -6.2 |       0.9 |

### where the families sit (mean success rates)

|                           |    S |   act |   obs |   all |
|:--------------------------|-----:|------:|------:|------:|
| ('nom', 'best_of_k')      | 0.72 |  0.55 |  0.24 |  0.09 |
| ('nom', 'dagger')         | 0.75 |  0.59 |  0.31 |  0.12 |
| ('nom', 'scaling')        | 0.62 |  0.47 |  0.2  |  0.08 |
| ('nom', 'self_imitation') | 0.53 |  0.38 |  0.24 |  0.09 |

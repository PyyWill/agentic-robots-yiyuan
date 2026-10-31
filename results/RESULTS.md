### Base (hook back, π = 1): learning curve

| m | L2 1st attempt | L2 within two | L1 1st | invention 1st | MAP = true essence | queries | trials |
|---|---|---|---|---|---|---|---|
| 0 | 0.18 ± 0.04 | 0.24 ± 0.05 | – | 0.08 ± 0.07 | – | 0.00 | 0 |
| 2 | 0.36 ± 0.07 | 0.45 ± 0.08 | 0.55 ± 0.26 | 0.24 ± 0.20 | 0.00 | 0.00 | 5 |
| 4 | 0.49 ± 0.17 | 0.71 ± 0.16 | 0.47 ± 0.23 | 0.22 ± 0.14 | 0.00 | 0.40 | 10 |
| 8 | 0.88 ± 0.15 | 0.99 ± 0.03 | 0.88 ± 0.13 | 0.65 ± 0.33 | 0.00 | 1.10 | 19 |
| 16 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.50 | 1.10 | 36 |
| 32 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 | 1.10 | 69 |

Baselines on the same tests: random 0.14 ± 0.05 / 0.36 ± 0.06, VLM prior only 0.41 ± 0.06 / 0.86 ± 0.04 (L2 1st / within two); invention 1st: random 0.08 ± 0.06, VLM prior 0.11 ± 0.07.

### All conditions at the last checkpoint

| condition | m | L1 1st | L2 1st | L2 within two | invention 1st | structure agrees with truth | queries | |D| | VLM-prior L2 1st |
|---|---|---|---|---|---|---|---|---|---|
| base | 32 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 | 1.10 | 10.10 | 0.41 ± 0.06 |
| noise0.2 | 32 | 0.80 ± 0.00 | 0.80 ± 0.00 | 0.96 ± 0.01 | 0.80 ± 0.00 | 0.99 | 1.70 | 10.50 | 0.33 ± 0.05 |
| varied | 32 | 0.58 ± 0.18 | 0.47 ± 0.11 | 0.76 ± 0.16 | 0.83 ± 0.38 | 0.74 | 7.67 | 13.00 | 0.23 ± 0.09 |
| varied | 64 | 0.68 ± 0.10 | 0.49 ± 0.07 | 0.96 ± 0.11 | 0.92 ± 0.21 | 0.80 | 18.00 | 13.00 | 0.23 ± 0.09 |

### Go/no-go against Claude (identical inputs). Verdict: **NO-GO**

| baseline: condition@m | test | ours | Claude | ours − Claude | seeds ours better | Claude retries | cost |
|---|---|---|---|---|---|---|---|
| claude:base@m32 | L2_ood within2 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0/10 | 1.00 | $15.8 |
| claude:base@m32 | L2_ood 0shot | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0/10 | 1.00 | $15.8 |
| claude:base@m32 | invent within2 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0/10 | 1.00 | $15.8 |
| claude:base@m32 | invent 0shot | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0/10 | 1.00 | $15.8 |
| claude:noise0.2@m32 | L2_ood within2 | 0.96 ± 0.01 | 0.96 ± 0.00 | -0.00 ± 0.01 | 1/10 | 1.00 | $45.78 |
| claude:noise0.2@m32 | L2_ood 0shot | 0.80 ± 0.00 | 0.80 ± 0.00 | 0.00 ± 0.00 | 0/10 | 1.00 | $45.78 |
| claude:noise0.2@m32 | invent within2 | 0.96 ± 0.00 | 0.96 ± 0.00 | 0.00 ± 0.00 | 0/10 | 1.00 | $45.78 |
| claude:noise0.2@m32 | invent 0shot | 0.80 ± 0.00 | 0.80 ± 0.00 | 0.00 ± 0.00 | 0/10 | 1.00 | $45.78 |
| claude:varied@m32 | L2_ood within2 | 0.65 ± 1.06 | 0.95 ± 0.21 | -0.30 ± 1.27 | 0/2 | 0.48 | $15.99 |
| claude:varied@m32 | L2_ood 0shot | 0.37 ± 0.85 | 0.48 ± 0.21 | -0.12 ± 1.06 | 0/2 | 0.48 | $15.99 |
| claude:varied@m32 | invent within2 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0/2 | 1.00 | $15.99 |
| claude:varied@m32 | invent 0shot | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0/2 | 1.00 | $15.99 |

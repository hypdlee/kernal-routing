# ResNet-18 routing controls: pre-registered analysis

- log check: all control values match their train.log line
- GPU recorded in result files: NVIDIA GeForce RTX 4070 Ti

## Predictions (PREREGISTRATION_CONTROLS_R18.md)

| prediction | statement | outcome |
|---|---|---|
| R1 | Tiny-ImageNet w32: shuffled and frozen_random each lower than gap | pending: cell incomplete or not run |
| R2 | CIFAR-100 w16 and w32: shuffled lower than gap | pending: cell incomplete or not run |
| R3 | Tiny-ImageNet w32: frozen_random higher than uniform | pending: cell incomplete or not run |

## Paired comparisons

| comparison | mean diff (pp) | 95% CI | p | TOST p (0.5 pp) | verdict |
|---|---|---|---|---|---|
| **ResNet-18, cifar100 width 16** (seeds finished of 6: {'shuffled': 6, 'frozen_random': 6}) | | | | | |
| shuffled - static | +0.14 | [-0.20, +0.49] | 0.3361 | 0.0228 | equivalent |
| shuffled - uniform | +0.37 | [-0.04, +0.77] | 0.0678 | 0.2184 | inconclusive |
| shuffled - gap | -0.39 | [-0.66, -0.11] | 0.0150 | 0.1719 | lower |
| frozen_random - static | +0.28 | [-0.03, +0.59] | 0.0694 | 0.0630 | inconclusive |
| frozen_random - uniform | +0.50 | [+0.18, +0.82] | 0.0096 | 0.5051 | higher |
| frozen_random - gap | -0.25 | [-0.52, +0.01] | 0.0566 | 0.0307 | equivalent |
| **ResNet-18, tinyimagenet width 32** (seeds finished of 6: {'shuffled': 4, 'frozen_random': 2}) | | | | | |
| shuffled - static | +0.10 | [-0.51, +0.71] | 0.6472 | 0.0637 | inconclusive (n=4) |
| shuffled - uniform | +0.08 | [-0.45, +0.62] | 0.6559 | 0.0441 | equivalent (n=4) |
| shuffled - gap | -2.14 | [-3.16, -1.12] | 0.0068 | 0.9928 | lower (n=4) |
| frozen_random - static | +1.01 | [-4.71, +6.73] | 0.2668 | 0.7699 | inconclusive (n=2) |
| frozen_random - uniform | +1.14 | [-0.96, +3.23] | 0.0919 | 0.9191 | inconclusive (n=2) |
| frozen_random - gap | -0.85 | [-3.26, +1.56] | 0.1400 | 0.8417 | inconclusive (n=2) |
| **ResNet-18, cifar100 width 64** (seeds finished of 6: {'shuffled': 6, 'frozen_random': 6}) | | | | | |
| shuffled - static | -0.07 | [-0.60, +0.46] | 0.7361 | 0.0464 | equivalent |
| shuffled - uniform | -0.12 | [-0.49, +0.25] | 0.4371 | 0.0234 | equivalent |
| shuffled - gap | -0.15 | [-0.64, +0.33] | 0.4507 | 0.0619 | inconclusive |
| frozen_random - static | +0.18 | [-0.12, +0.47] | 0.1874 | 0.0191 | equivalent |
| frozen_random - uniform | +0.13 | [-0.13, +0.39] | 0.2605 | 0.0072 | equivalent |
| frozen_random - gap | +0.10 | [-0.20, +0.40] | 0.4433 | 0.0089 | equivalent |

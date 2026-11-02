# ResNet-18 routing controls: pre-registered analysis

- log check: all control values match their train.log line
- GPU recorded in result files: NVIDIA GeForce RTX 4070 Ti

## Predictions (PREREGISTRATION_CONTROLS_R18.md)

| prediction | statement | outcome |
|---|---|---|
| R1 | Tiny-ImageNet w32: shuffled and frozen_random each lower than gap | **held** |
| R2 | CIFAR-100 w16 and w32: shuffled lower than gap | **held** |
| R3 | Tiny-ImageNet w32: frozen_random higher than uniform | **held** |

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
| **ResNet-18, cifar100 width 32** (seeds finished of 6: {'shuffled': 6, 'frozen_random': 6}) | | | | | |
| shuffled - static | +0.16 | [-0.21, +0.54] | 0.3124 | 0.0357 | equivalent |
| shuffled - uniform | -0.01 | [-0.52, +0.50] | 0.9614 | 0.0276 | equivalent |
| shuffled - gap | -0.99 | [-1.24, -0.73] | 0.0002 | 0.9978 | lower |
| frozen_random - static | +0.74 | [+0.31, +1.17] | 0.0069 | 0.8951 | higher |
| frozen_random - uniform | +0.57 | [+0.33, +0.81] | 0.0017 | 0.7468 | higher |
| frozen_random - gap | -0.41 | [-0.84, +0.02] | 0.0563 | 0.3092 | inconclusive |
| **ResNet-18, tinyimagenet width 32** (seeds finished of 6: {'shuffled': 6, 'frozen_random': 6}) | | | | | |
| shuffled - static | +0.06 | [-0.25, +0.38] | 0.6301 | 0.0083 | equivalent |
| shuffled - uniform | -0.04 | [-0.44, +0.36] | 0.7916 | 0.0162 | equivalent |
| shuffled - gap | -2.12 | [-2.64, -1.60] | 0.0001 | 0.9998 | lower |
| frozen_random - static | +1.31 | [+0.80, +1.82] | 0.0012 | 0.9952 | higher |
| frozen_random - uniform | +1.20 | [+1.01, +1.40] | 0.0000 | 0.9999 | higher |
| frozen_random - gap | -0.88 | [-1.25, -0.50] | 0.0018 | 0.9752 | lower |
| **ResNet-18, cifar100 width 64** (seeds finished of 6: {'shuffled': 6, 'frozen_random': 6}) | | | | | |
| shuffled - static | -0.07 | [-0.60, +0.46] | 0.7361 | 0.0464 | equivalent |
| shuffled - uniform | -0.12 | [-0.49, +0.25] | 0.4371 | 0.0234 | equivalent |
| shuffled - gap | -0.15 | [-0.64, +0.33] | 0.4507 | 0.0619 | inconclusive |
| frozen_random - static | +0.18 | [-0.12, +0.47] | 0.1874 | 0.0191 | equivalent |
| frozen_random - uniform | +0.13 | [-0.13, +0.39] | 0.2605 | 0.0072 | equivalent |
| frozen_random - gap | +0.10 | [-0.20, +0.40] | 0.4433 | 0.0089 | equivalent |

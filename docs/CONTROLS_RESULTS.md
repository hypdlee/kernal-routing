# Routing controls: pre-registered analysis

- mobilenetv2: 126 runs loaded (6 routing-control cells)
- vgg16: 138 runs loaded (8 routing-control cells)
- log check: all values match their train.log line

## Predictions (PREREGISTRATION_CONTROLS.md, 2026-10-22)

| prediction | statement | sub-results | outcome |
|---|---|---|---|
| C1 | MobileNetV2: both new frozen draws higher than uniform at CIFAR-100 w64 | [np.True_, np.True_] | **held** |
| C2 | VGG-16: neither new frozen draw higher than uniform at CIFAR-100 w64 | [False, False] | **held** |
| C3 | MobileNetV2 w16: frozen > static, frozen not distinguishable from gap, shuffled < gap | [np.True_, True, np.True_] | **held** |
| C4 | VGG-16 w16 and Tiny-ImageNet w32: frozen < gap and shuffled < gap (4 comparisons) | [False, np.True_, np.True_, np.True_] | **failed** |

## Paired comparisons

Paired t-test over six seeds, TOST margin 0.5 pp. Differences are first minus second.

| comparison | mean diff (pp) | 95% CI | p | TOST p (0.5 pp) | verdict |
|---|---|---|---|---|---|
| **mobilenetv2, CIFAR-100 w64 (part A)** | | | | | |
| frozen d2 - uniform | +1.74 | [+1.21, +2.26] | 0.0004 | 0.9991 | higher |
| frozen d3 - uniform | +1.41 | [+0.66, +2.16] | 0.0047 | 0.9869 | higher |
| frozen d1 - uniform (main program) | +1.63 | [+1.05, +2.20] | 0.0008 | 0.9980 | higher |
| frozen d2 - gap | +0.26 | [-0.26, +0.79] | 0.2572 | 0.1484 | inconclusive |
| frozen d3 - gap | -0.06 | [-0.90, +0.77] | 0.8531 | 0.1183 | inconclusive |
| frozen d2 - static | +1.62 | [+1.11, +2.13] | 0.0004 | 0.9988 | higher |
| frozen d3 - static | +1.30 | [+0.68, +1.91] | 0.0029 | 0.9896 | higher |
| **vgg16, CIFAR-100 w64 (part A)** | | | | | |
| frozen d2 - uniform | +0.13 | [-0.05, +0.32] | 0.1145 | 0.0018 | equivalent |
| frozen d3 - uniform | +0.06 | [-0.28, +0.40] | 0.6754 | 0.0100 | equivalent |
| frozen d1 - uniform (main program) | +0.09 | [-0.12, +0.31] | 0.3111 | 0.0022 | equivalent |
| frozen d2 - gap | -0.24 | [-0.43, -0.05] | 0.0236 | 0.0084 | lower |
| frozen d3 - gap | -0.32 | [-0.56, -0.07] | 0.0219 | 0.0559 | lower |
| frozen d2 - static | -0.05 | [-0.27, +0.16] | 0.5568 | 0.0016 | equivalent |
| frozen d3 - static | -0.13 | [-0.44, +0.18] | 0.3237 | 0.0132 | equivalent |
| **mobilenetv2, CIFAR-100 w16 (part B)** | | | | | |
| frozen - static | +1.70 | [+0.56, +2.84] | 0.0122 | 0.9788 | higher |
| frozen - gap | -0.22 | [-1.15, +0.71] | 0.5747 | 0.2341 | inconclusive |
| shuffled - gap | -1.74 | [-2.92, -0.55] | 0.0130 | 0.9783 | lower |
| **mobilenetv2, CIFAR-100 w16, context** | | | | | |
| frozen - uniform | +1.10 | [+0.43, +1.78] | 0.0083 | 0.9658 | higher |
| shuffled - static | +0.18 | [-0.60, +0.96] | 0.5854 | 0.1675 | inconclusive |
| gap - static | +1.91 | [+0.40, +3.43] | 0.0225 | 0.9694 | higher |
| gap - uniform | +1.32 | [+0.82, +1.82] | 0.0011 | 0.9958 | higher |
| **vgg16, CIFAR-100 w16 (part B)** | | | | | |
| frozen - gap | -0.39 | [-0.85, +0.07] | 0.0811 | 0.2864 | inconclusive |
| shuffled - gap | -0.95 | [-1.51, -0.38] | 0.0075 | 0.9518 | lower |
| **vgg16, CIFAR-100 w16 (part B), context** | | | | | |
| frozen - uniform | +0.33 | [-0.14, +0.80] | 0.1333 | 0.1928 | inconclusive |
| frozen - static | +0.47 | [+0.21, +0.74] | 0.0061 | 0.3975 | higher |
| shuffled - static | -0.08 | [-0.64, +0.47] | 0.7146 | 0.0554 | inconclusive |
| gap - uniform | +0.72 | [+0.06, +1.38] | 0.0375 | 0.7840 | higher |
| gap - static | +0.86 | [+0.49, +1.24] | 0.0020 | 0.9720 | higher |
| **vgg16, Tiny-ImageNet w32 (part C)** | | | | | |
| frozen - gap | -1.04 | [-1.38, -0.71] | 0.0005 | 0.9958 | lower |
| shuffled - gap | -1.60 | [-2.03, -1.17] | 0.0002 | 0.9994 | lower |
| **vgg16, Tiny-ImageNet w32 (part C), context** | | | | | |
| frozen - uniform | +0.59 | [+0.29, +0.90] | 0.0041 | 0.7636 | higher |
| frozen - static | +0.72 | [+0.29, +1.15] | 0.0075 | 0.8786 | higher |
| shuffled - static | +0.17 | [-0.29, +0.62] | 0.3916 | 0.0599 | inconclusive |
| gap - uniform | +1.64 | [+1.26, +2.02] | 0.0001 | 0.9997 | higher |
| gap - static | +1.77 | [+1.37, +2.16] | 0.0001 | 0.9998 | higher |

## Cell means

| backbone | condition | arm | mean | sd | n |
|---|---|---|---|---|---|
| mobilenetv2 | cifar100 w16 | frozen_random | 68.05 | 0.72 | 6 |
| mobilenetv2 | cifar100 w16 | gap | 68.27 | 0.53 | 6 |
| mobilenetv2 | cifar100 w16 | shuffled | 66.53 | 0.80 | 6 |
| mobilenetv2 | cifar100 w16 | static | 66.36 | 0.97 | 6 |
| mobilenetv2 | cifar100 w16 | uniform | 66.95 | 0.24 | 6 |
| mobilenetv2 | cifar100 w32 | gap | 72.08 | 0.36 | 6 |
| mobilenetv2 | cifar100 w32 | static | 70.77 | 0.22 | 6 |
| mobilenetv2 | cifar100 w32 | uniform | 70.67 | 0.21 | 6 |
| mobilenetv2 | cifar100 w64 | frozen_random | 73.91 | 0.34 | 6 |
| mobilenetv2 | cifar100 w64 | frozen_random d2 | 74.02 | 0.33 | 6 |
| mobilenetv2 | cifar100 w64 | frozen_random d3 | 73.69 | 0.56 | 6 |
| mobilenetv2 | cifar100 w64 | gap | 73.76 | 0.33 | 6 |
| mobilenetv2 | cifar100 w64 | shuffled | 72.60 | 0.20 | 6 |
| mobilenetv2 | cifar100 w64 | static | 72.39 | 0.25 | 6 |
| mobilenetv2 | cifar100 w64 | uniform | 72.28 | 0.31 | 6 |
| mobilenetv2 | tinyimagenet w32 | gap | 58.66 | 0.57 | 6 |
| mobilenetv2 | tinyimagenet w32 | static | 56.97 | 0.43 | 6 |
| mobilenetv2 | tinyimagenet w32 | uniform | 56.77 | 0.41 | 6 |
| mobilenetv2 | tinyimagenet w64 | gap | 60.45 | 0.41 | 6 |
| mobilenetv2 | tinyimagenet w64 | static | 58.82 | 0.22 | 6 |
| mobilenetv2 | tinyimagenet w64 | uniform | 58.64 | 0.31 | 6 |
| vgg16 | cifar100 w16 | frozen_random | 65.69 | 0.34 | 6 |
| vgg16 | cifar100 w16 | gap | 66.08 | 0.28 | 6 |
| vgg16 | cifar100 w16 | shuffled | 65.14 | 0.41 | 6 |
| vgg16 | cifar100 w16 | static | 65.22 | 0.18 | 6 |
| vgg16 | cifar100 w16 | uniform | 65.36 | 0.45 | 6 |
| vgg16 | cifar100 w32 | gap | 71.80 | 0.30 | 6 |
| vgg16 | cifar100 w32 | static | 71.32 | 0.23 | 6 |
| vgg16 | cifar100 w32 | uniform | 71.17 | 0.36 | 6 |
| vgg16 | cifar100 w64 | frozen_random | 74.71 | 0.20 | 6 |
| vgg16 | cifar100 w64 | frozen_random d2 | 74.75 | 0.12 | 6 |
| vgg16 | cifar100 w64 | frozen_random d3 | 74.68 | 0.17 | 6 |
| vgg16 | cifar100 w64 | gap | 74.99 | 0.22 | 6 |
| vgg16 | cifar100 w64 | shuffled | 74.57 | 0.26 | 6 |
| vgg16 | cifar100 w64 | static | 74.81 | 0.22 | 6 |
| vgg16 | cifar100 w64 | uniform | 74.62 | 0.23 | 6 |
| vgg16 | tinyimagenet w32 | frozen_random | 59.77 | 0.25 | 6 |
| vgg16 | tinyimagenet w32 | gap | 60.81 | 0.28 | 6 |
| vgg16 | tinyimagenet w32 | shuffled | 59.21 | 0.32 | 6 |
| vgg16 | tinyimagenet w32 | static | 59.05 | 0.20 | 6 |
| vgg16 | tinyimagenet w32 | uniform | 59.18 | 0.15 | 6 |
| vgg16 | tinyimagenet w64 | gap | 64.78 | 0.48 | 6 |
| vgg16 | tinyimagenet w64 | static | 63.30 | 0.38 | 6 |
| vgg16 | tinyimagenet w64 | uniform | 63.55 | 0.38 | 6 |

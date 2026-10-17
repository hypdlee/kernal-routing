# Pre-registration: the 132-run program on four more backbones

Recorded 30 September 2026, before any run of MobileNetV2, VGG-16 (BN),
ResNet-50 or ConvNeXt-Tiny. Commit this file unmodified before the first
run finishes. Any later change is dated and reported in the paper with its
reason.

These predictions are made after the ResNet-18 study, and are derived from
its results. They test whether its findings transfer; they are not
independent of it.

## Question

Do the ResNet-18 findings hold on other architectures?

1. The kernel bank alone (the `uniform` arm, which is exactly a static
   network with mean kernels, Proposition 1) is worth nothing.
2. Routing (`gap` over `uniform`) is where any gain comes from.
3. A plain wider network beats routing for the same stored parameters.

## Design: exactly the ResNet-18 program, per backbone

Defined once in `src/program.py`, checked against an independently written
copy of the ResNet-18 program in `tests/test_runner.py` (set-equal, 132 jobs).

| group | dataset | width | K | tau | arms | runs |
|---|---|---|---|---|---|---|
| 1 | CIFAR-100 | 64 | 4 | 1 | static, uniform, gap, shuffled, frozen_random | 30 |
| 2 | CIFAR-100 | 32 | 4 | 1 | static, uniform, gap | 18 |
| 3 | CIFAR-100 | 16 | 4 | 1 | static, uniform, gap | 18 |
| 4 | CIFAR-100 | 32 | 2 | 1 | gap | 6 |
| 5 | CIFAR-100 | 32 | 8 | 1 | gap | 6 |
| 6 | Tiny-ImageNet | 32 | 4 | 1 | static, uniform, gap | 18 |
| 7 | CIFAR-100 | 64 | 4 | 30 -> 1 | gap | 6 |
| 8 | CIFAR-100 | 32 | 4 | 30 -> 1 | gap | 6 |
| 9 | CIFAR-100 | 16 | 4 | 30 -> 1 | gap | 6 |
| 10 | Tiny-ImageNet | 64 | 4 | 1 | static, uniform, gap (seed-major) | 18 |

132 runs per backbone, 528 in all. Seeds 0 to 5.

**Recipe, identical to ResNet-18 and not tuned per architecture:** SGD,
Nesterov momentum 0.9, lr 0.1, weight decay 5e-4 on every parameter
(including normalisation and layer-scale parameters, as ResNet-18 applied it
to BatchNorm), batch 128, 5 warm-up epochs then cosine, 150 epochs, random
crop (4 px padding at 32x32, 8 px at 64x64) and horizontal flip, bf16
autocast, `blend` implementation. Tau annealing: linear 30 -> 1 over 10
epochs. No label smoothing, mixup, cutout, drop path or dropout.

**Width.** Width 64 is each backbone's standard channel count. Widths 32
and 16 multiply every channel count by 1/2 and 1/4 (rounded to a multiple
of 8 where the architecture requires it: MobileNetV2).

**Adaptation to small inputs.** ResNet-50: CIFAR stem (3x3, stride 1, no
max-pool), bottleneck v1.5, [3, 4, 6, 3]. VGG-16: configuration D with
BatchNorm, global average pool, one linear classifier. MobileNetV2: stem
stride 1 and the second stage stride 1, as in the common CIFAR adaptation.
ConvNeXt-Tiny: depths (3, 3, 9, 3), dims (96, 192, 384, 768) at width 64,
stem 1x1 patch-1 convolution with LayerNorm (so the 32x32 input keeps a 4x4
map in the last stage), layer scale 1e-6, no drop path. The same
architecture serves both input sizes.

## Dynamic layers: the ResNet-18 rule, applied literally

Every k x k convolution with k > 1 inside the blocks is dynamic. The stem,
every 1x1 convolution, shortcuts, downsampling layers and the classifier
stay static. Counts, checked by `tests/test_models.py placement`:

| backbone | dynamic layers | which | share of static params in dynamic layers (CIFAR-100 w64) |
|---|---|---|---|
| ResNet-18 | 16 | both 3x3 in every basic block | 98 % |
| ResNet-50 | 16 | the 3x3 in every bottleneck | 48 % |
| VGG-16 | 12 | every 3x3 after the first | 100 % |
| MobileNetV2 | 17 | the 3x3 depthwise in every block | 3 % |
| ConvNeXt-Tiny | 18 | the 7x7 depthwise in every block | 1 % |

The alternative DY-Conv rule (every convolution except the stem) is
implemented (`--placement all`: 52 / 12 / 50 / 57 layers) and is NOT run.
Known consequence, stated before any result: in MobileNetV2 and
ConvNeXt-Tiny almost all parameters stay static, so the bank and routing
act on a small part of the network, and a null there may say more about
placement than about routing.

## Initialisation

New backbones: one scheme for every static convolution and every bank
kernel. Kaiming-normal with fan-out for ResNet-50, VGG-16 and MobileNetV2;
truncated normal, std 0.02, for ConvNeXt-Tiny. Normalisation weights 1,
biases 0; ResNet-50 zero-initialises the last BatchNorm of each block.

ResNet-18 (disclosed here and in the paper's threats section): in the 132
runs, static convolutions kept PyTorch's default initialisation
(Kaiming-uniform, a = sqrt 5) while bank kernels were Kaiming-normal with
fan-out. Its bank term therefore compares two initialisations as well as a
bank against a single kernel. Its routing term (gap - uniform) does not:
both arms have identically initialised banks. The ResNet-18 runs are not
rerun; the mismatch is reported.

## Parameter budgets: known before any run

A K = 4 bank and a doubling of width both multiply a dynamic layer's
parameters by four, so in ResNet-18 `gap` at width c and `static` at width
2c store almost the same number of parameters (ratio 0.99). That does not
transfer. Counted from the model definitions (`PARAMS_EXPECTED.md`):

| backbone | gap(w16)/static(w32) | gap(w32)/static(w64) | TIN gap(w32)/static(w64) |
|---|---|---|---|
| ResNet-18 | 0.99 | 0.99 | 0.99 |
| VGG-16 | 1.00 | 1.00 | 1.00 |
| MobileNetV2 | 0.60 | 0.51 | 0.54 |
| ResNet-50 | 0.61 | 0.61 | 0.61 |
| ConvNeXt-Tiny | 0.30 | 0.28 | 0.28 |

A pair is called matched only within 10 percent. Every other pair is
reported with its ratio, and beside it the comparison against the static
accuracy curve interpolated in log-parameters at the gap network's count
(an interpolation, not a run, no confidence interval).

## Router size: known before any run

The router is the ResNet-18 design unchanged: a two-layer MLP on the global
descriptor with hidden width C_in/4 (at least 16). Its parameters grow with
the square of a layer's input channels; a depthwise kernel grows linearly.
Router parameters of the `gap` network, CIFAR-100 width 64, against the
static network (counted):

| backbone | router | static network | router / static | bank adds |
|---|---|---|---|---|
| ResNet-18 | 0.29 M | 11.22 M | 2.6 % | 32.96 M |
| VGG-16 | 0.39 M | 14.77 M | 2.7 % | 44.13 M |
| ResNet-50 | 0.32 M | 23.71 M | 1.3 % | 33.95 M |
| MobileNetV2 | 1.14 M | 2.35 M | 48.4 % | 0.19 M |
| ConvNeXt-Tiny | 0.82 M | 27.89 M | 2.9 % | 0.99 M |

In MobileNetV2 the router is six times the bank it routes and half the
static network, so its routing term (gap - uniform) also compares networks
of very different size. `frozen_random` (the same router, untrained, held
as buffers) is the control for this at CIFAR-100 width 64. The router is
not changed per backbone, because the program replicates the ResNet-18 one.

## Primary metric and statistics

Final-epoch top-1 test accuracy. Paired by seed, n = 6. Paired t-test and
two-sided Wilcoxon signed-rank (floor 0.031 at n = 6). Equivalence by TOST
against +/-0.5 pp, alpha 0.05. Verdicts: equivalent (TOST p < 0.05 and
difference p >= 0.05), higher or lower (difference p < 0.05), otherwise
inconclusive. Same code as the ResNet-18 paper (`report/data.py`, checked
to reproduce every table, macro and figure of that paper byte for byte).

## Predictions, per new backbone

Conditions: CIFAR-100 widths 16, 32, 64; Tiny-ImageNet widths 32, 64.

- **P1 (bank):** the bank term, `uniform - static`, is equivalent to zero
  in each of the five conditions. Held if 5 of 5.
- **P2 (width beats routing):** `static` at width 2c is more accurate than
  `gap` at width c in each of the three pairs (paired t, p < 0.05,
  positive). Held if 3 of 3. For backbones where the pair is not matched
  the plain network is also the larger one, so a hold there is weaker
  evidence and is reported with the ratio; the interpolated comparison
  carries the equal-parameter claim.
- **P3 (routing carries the effect):** the routing term, `gap - uniform`,
  is positive with p < 0.05 in at least 3 of the 5 conditions. Held if
  >= 3 of 5.

Not predicted: the width-64 CIFAR-100 null (unexplained in ResNet-18), the
sign of the temperature effect, the relation between router entropy and
benefit. They are measured and reported.

## Gate, per backbone

The first run of each backbone is `static`, CIFAR-100, width 64, seed 0.
If its final accuracy is below the floor, the shared recipe does not train
that backbone, its other 131 runs are not spent, and the failure is
reported (`GATE_FAILED.json`, a row in the cross-model summary).

| backbone | floor (CIFAR-100 top-1) |
|---|---|
| MobileNetV2 | 65 % |
| VGG-16 (BN) | 65 % |
| ResNet-50 | 70 % |
| ConvNeXt-Tiny | 60 % |

The floors are set well below published CIFAR-100 results for each
architecture. They test that the recipe trains the network, not how well.

## Operational rules, fixed now

- **Order:** MobileNetV2, VGG-16, ResNet-50, ConvNeXt-Tiny. Within a
  backbone: the gate run, then groups 1 to 10 as in the table, arm-major,
  except Tiny-ImageNet width 64 which is seed-major.
- **Micro-batch:** before each group the runner probes the group's most
  memory-hungry arm at batch 128, then 64 x 2, 32 x 4, 16 x 8 (gradient
  accumulation). The first that fits is used for every arm of that group,
  so paired comparisons share it. Logical batch, optimiser steps and
  learning-rate schedule are unchanged; BatchNorm statistics are per
  micro-batch. Every such group is listed in `DEVIATIONS.md` and in the
  report.
- **Divergence:** a run whose training loss becomes non-finite is recorded
  as diverged (a `.DIVERGED` marker), not rerun, and reported. Paired tests
  use the seeds both arms have and state n.
- **Interruption:** a run resumes from its per-epoch checkpoint; the result
  records `resumed_at_epoch`, and the report lists resumed runs.
- **No run is repeated or replaced** to change a result. Any rerun for a
  technical failure is listed.

## Compute estimate

Scaled from the logged ResNet-18 minutes by counted multiply-accumulates
(`COMPUTE_ESTIMATE.md`): MobileNetV2 35, VGG-16 121, ResNet-50 467,
ConvNeXt-Tiny 518 GPU-hours on the ResNet-18 GPU; about 1,140 in all. This
is an estimate and certainly low for the depthwise backbones (memory
bound); the runner's probe measures the real rate before each group.

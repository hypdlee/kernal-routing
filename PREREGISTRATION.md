# Stage 1 pre-registration

Commit this file, unmodified, before the first run finishes. If you change
the decision rule after seeing results, say so in the paper and date the
change. The point of writing it down is that a null result stays
interpretable, and a null result is the likely outcome here.

## Question

Does the routing signal in dynamic convolution carry information the
network uses, or does the reported benefit come from the extra parameters
the kernel bank adds?

## Hypotheses

- **H1 (adaptivity)** A trained GAP router beats a constant `uniform`
  router by more than the equivalence margin.
- **H0 (capacity)** It does not. The two are equivalent within the margin,
  and the gain attributed to input-adaptive kernel selection is capacity.

## Design

CIFAR-100, ResNet-18 (CIFAR stem), all 3x3 block convs replaced by
`DynamicConv2d` with K = 4. Stem and 1x1 downsample shortcuts stay static.

Four arms, one shared recipe, differing only in the router:

| arm | routing | input-dependent | trained | purpose |
|---|---|---|---|---|
| `uniform` | z = 1/K | no | no | capacity control |
| `gap` | GAP -> MLP | yes | yes | standard baseline |
| `shuffled` | GAP of a permuted batch | yes, wrongly paired | yes | isolates pairing from distribution |
| `frozen_random` | random MLP on GAP | yes | no | isolates variation from usefulness |

`uniform` and `frozen_random` have identical parameter counts to each
other. `gap` and `shuffled` add the router MLP (~0.29M of ~44M, 0.6%).
That 0.6% is itself a confound and is why `frozen_random` is in the grid.

Recipe, fixed in advance and identical across arms: SGD, momentum 0.9,
nesterov, lr 0.1, cosine schedule, 5 warmup epochs, weight decay 5e-4,
batch 128, 150 epochs, random crop with 4px padding, horizontal flip.
No label smoothing, no mixup, no cutout. Six seeds: 0 through 5.

## Primary metric

Final-epoch top-1 test accuracy. Not best-epoch. Best-epoch maximises over
a noisy sequence and inflates whichever arm happens to have a lucky epoch,
which is exactly the kind of freedom this file exists to remove.

## Decision rule

Fixed before running. Margin = 0.5 accuracy points, alpha = 0.05, paired by
seed.

- **Difference:** paired t-test on `gap - uniform`. Wilcoxon signed-rank is
  reported alongside it. Note that two-sided Wilcoxon has a floor of 0.25
  at n=3 and 0.0625 at n=5, so it cannot reach alpha below 6 seeds. Six
  seeds is the default for this reason.
- **Equivalence:** TOST against +/-0.5pp. This is the test that licenses
  the null claim. A large p on the difference test is not evidence of no
  effect; a small p on TOST is.

Outcomes:

| difference | equivalence | conclusion |
|---|---|---|
| p < 0.05 | not equivalent | routing helps. Proceed to Stage 0 and build the relaxed descriptor |
| not significant | p_TOST < 0.05 | routing is worth less than 0.5pp. Write the null |
| p < 0.05 | p_TOST < 0.05 | real but smaller than the margin. Report the effect size, not the p-value |
| neither | neither | underpowered. Add seeds. Do not report either way |

Why 0.5pp: published dynamic-convolution gains on CIFAR-scale benchmarks
sit in the 1 to 3pp range. A margin at 0.5pp is tight enough that
equivalence inside it contradicts those claims, and loose enough to be
reachable with 6 seeds at the seed-to-seed spread ResNet-18 shows on
CIFAR-100.

## Secondary, pre-specified

Normalised routing entropy per layer, mean over the test set, from one
final eval pass. Entropy near 1.0 means the softmax is near-uniform and the
router is not discriminating between kernels, which is the mechanism DCD
(ICLR 2021) predicts. Reported for all arms. `uniform` sits at exactly 1.0
by construction and acts as the reference.

This is descriptive. No hypothesis test is attached to it, because the
entropy-of-mean statistic is foolable and per-layer entropy has not been
validated as a routing-usefulness measure. That validation is its own
piece of work.

## What would falsify the whole program

If `uniform` matches `gap`, then no descriptor swap can matter, and LTD,
DCT, GAP+std+max and everything else in `routers.py` is answering a
question the architecture does not ask. Stage 3 would be pointless and the
paper becomes the null plus the diagnostic.

## Deviations log

Record any departure from the above here, dated, before running the
affected experiment.

- (none yet)

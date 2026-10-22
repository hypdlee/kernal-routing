# Pre-registration: routing controls beyond CIFAR-100 width 64

**Status: DRAFT. Edit the predictions if you disagree, fill in the date,
and commit this file (and upload it to OSF Registries or AsPredicted) before
the first run of `run_controls.py` finishes.**

Date recorded: 2026-10-22 (before any run of `run_controls.py`)

## Why these runs

In the main program the routing controls ran on CIFAR-100 at width 64 only,
with one random draw of the frozen router shared by all seeds. On
MobileNetV2 that draw matched the trained router; on VGG-16 it was
equivalent to the constant mixture, and in that condition no VGG-16 arm beat
the plain network. These runs test whether those two results depend on the
draw and on the condition.

## Runs (fixed now)

* Part A: `frozen_random`, CIFAR-100 width 64, draws d2 (`--frozen-seed 1`)
  and d3 (`--frozen-seed 2`), MobileNetV2 and VGG-16, seeds 0 to 5.
* Part B: `shuffled` and `frozen_random` (draw d1), CIFAR-100 width 16,
  MobileNetV2 and VGG-16, seeds 0 to 5.
* Part C: `shuffled` and `frozen_random` (draw d1), Tiny-ImageNet width 32,
  VGG-16, seeds 0 to 5.
* Part D (optional): as part C for MobileNetV2.

Recipe, code, seeds and micro-batch rules as in the main program. New runs
are paired by seed with the main program's `static`, `uniform` and `gap`
runs of the same backbone and condition.

## Tests (fixed now)

Paired t-test, n = 6, two-sided, alpha 0.05; equivalence by two one-sided
tests at a margin of 0.5 pp. "Higher" or "lower" means p < 0.05 in that
direction; "not distinguishable" means p >= 0.05.

## Predictions

* **C1 (MobileNetV2, part A).** Each of the two new frozen-router draws is
  higher than `uniform` at CIFAR-100 width 64. Holds if both are higher.
* **C2 (VGG-16, part A).** Neither new draw is higher than `uniform` at
  CIFAR-100 width 64. Holds if neither is higher.
* **C3 (MobileNetV2, part B).** At CIFAR-100 width 16, `frozen_random` is
  higher than `static` and not distinguishable from `gap`; `shuffled` is
  lower than `gap`. Holds if all three hold.
* **C4 (VGG-16, parts B and C).** At CIFAR-100 width 16 and on Tiny-ImageNet
  width 32, `frozen_random` is lower than `gap` and `shuffled` is lower than
  `gap`. Holds if all four comparisons hold.

Every prediction is reported as held or failed, whatever the result, as in
the main program.

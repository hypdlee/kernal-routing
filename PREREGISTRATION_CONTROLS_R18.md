# Pre-registration addendum: ResNet-18 routing controls

**Status: DRAFT. Edit the predictions if you disagree, fill in the date and
time, commit this file (better: also upload it to OSF Registries or
AsPredicted) BEFORE the first ResNet-18 control run starts.**

Date and time recorded: 2026-10-27 14:56 (local time, Malta) (before any run of
`run_controls.py --models resnet18`)

## What is known when this is written

These predictions are written after the MobileNetV2 and VGG-16 control
results (PREREGISTRATION_CONTROLS.md, outcomes in docs/CONTROLS_RESULTS.md)
and after all 132 ResNet-18 runs of the main program. They are therefore not
blind to those results. They are recorded before any ResNet-18 control run,
and every prediction is reported as held or failed, whatever the result.

Known ResNet-18 facts (from the main program, six seeds): `gap` minus
`static` is positive at CIFAR-100 widths 16 and 32 and on Tiny-ImageNet width
32; at CIFAR-100 width 64 every ResNet-18 arm, including the controls with
frozen-router draw 1, is equivalent to the plain network.

## Runs (fixed now)

Same recipe, code path (default model and placement of the original 132
runs), seeds 0 to 5 and micro-batch rules as the main program. Each new run
is paired by seed with the main program's `static`, `uniform` and `gap` runs
of the same condition.

* Part B: `shuffled` and `frozen_random` (draw 1), CIFAR-100 width 16.
* Part E: `shuffled` and `frozen_random` (draw 1), CIFAR-100 width 32.
* Part C: `shuffled` and `frozen_random` (draw 1), Tiny-ImageNet width 32.
* Part A (optional, run last or not at all): `frozen_random` draws 2 and 3
  (`--frozen-seed 1`, `2`), CIFAR-100 width 64, where no ResNet-18 arm
  differs from the plain network, so this part cannot show a departure from
  it.

Command: `python run_controls.py --models resnet18 --parts B C E`
(36 runs, about 74 GPU-hours on the RTX 4070 Ti; on another GPU the time
differs and the GPU model is recorded in each result file).

## Tests (fixed now)

Paired t-test, n = 6, two-sided, alpha 0.05; equivalence by two one-sided
tests at a margin of 0.5 pp. "Higher" or "lower" means p < 0.05 in that
direction.

## Predictions

* **R1 (Tiny-ImageNet width 32).** `shuffled` and `frozen_random` are each
  lower than `gap`. Holds if both are.
* **R2 (CIFAR-100 widths 16 and 32).** `shuffled` is lower than `gap` at both
  widths. Holds if both are.
* **R3 (Tiny-ImageNet width 32).** `frozen_random` is higher than `uniform`.

If the hardware differs from the main program's RTX 4070 Ti, that is reported
next to the results.

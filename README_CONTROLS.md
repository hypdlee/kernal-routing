# Routing controls beyond CIFAR-100 width 64 (Windows PowerShell)

The main program already ran `shuffled` and `frozen_random` for MobileNetV2
and VGG-16, on CIFAR-100 at width 64 (paper, Section 7, Table 5). What it did
not run, and what reviewers can attack (reviewer-risk items 8 and 16):

* only **one random draw** of the frozen router, shared by all six seeds;
* the controls in **one condition only**, and on VGG-16 that condition has no
  effect over the plain network to explain.

`run_controls.py` adds exactly those runs. Nothing else changes: same recipe,
code path, seeds 0-5, micro-batch rules and result format, so every new run
pairs by seed with the `static`, `uniform` and `gap` runs already in
`results_models\`.

| part | what | models | runs | time (from the logged minutes of the same groups) |
|---|---|---|---|---|
| A | frozen router, draws d2 and d3 (`--frozen-seed 1`, `2`), CIFAR-100 w64 | MobileNetV2, VGG-16 | 24 | about 26 h |
| B | `shuffled` and `frozen_random`, CIFAR-100 w16 | MobileNetV2, VGG-16 | 24 | about 15 h |
| C | `shuffled` and `frozen_random`, Tiny-ImageNet w32 | VGG-16 | 12 | about 25 h |
| D | `shuffled` and `frozen_random`, Tiny-ImageNet w32 (optional) | MobileNetV2 | 12 | about 74 h |

Parts A to C: 60 runs, about 66 GPU-hours on the RTX 4070 Ti. Run them after
ResNet-50 and ConvNeXt-Tiny finish (one GPU). Part D only if there is time.

## 0. Before the first run

1. Pull the repository (this adds `run_controls.py`, `tests/test_controls.py`
   and the `--frozen-seed` flag in `src/train.py`; without the flag every
   earlier command behaves exactly as before).
2. Read `PREREGISTRATION_CONTROLS.md`, change any prediction you disagree
   with, put today's date in it, and commit it (better: also upload it to
   OSF Registries or AsPredicted) **before the first run finishes**.

## 1. Check (about 1 minute)

```powershell
python tests\test_controls.py
python run_controls.py --dry-run
```

The dry run prints 60 commands; 24 of them carry `--frozen-seed`.

## 2. Run

```powershell
python run_controls.py                        # parts A, B, C, both models
python run_controls.py --status               # progress
python run_controls.py --parts D --models mobilenetv2    # optional, about 74 h
```

The same command resumes after an interruption. Results go to
`results_controls\mobilenetv2\` and `results_controls\vgg16\`, with a
`train.log` and `runner.log` in each.

## 3. Send back

Zip the whole `results_controls` folder (result files, `train.log`,
`runner.log`, `plan.json`, `probes`). The analysis pairs every new run with
the existing runs of the same seed, checks every result file against its log,
and adds the controls to Section 7 and Table 5 of the paper.

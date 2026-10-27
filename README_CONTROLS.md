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

## ResNet-18 controls (opt-in; run on another PC)

ResNet-18 has `shuffled` and `frozen_random` only at CIFAR-100 width 64 with
one draw. The missing cells are runs of parts B (width 16), E (width 32) and
C (Tiny-ImageNet width 32), 36 runs, about 74 GPU-hours on the RTX 4070 Ti.
Part A for ResNet-18 (frozen draws 2 and 3 at width 64) is optional.

Before the first run, fill in the date in `PREREGISTRATION_CONTROLS_R18.md`,
edit the predictions if you disagree, commit and push it.

Setup on a second PC (PowerShell), then run:

```powershell
git clone https://github.com/hypdlee/kernal-routing.git
cd kernal-routing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
python tests\test_controls.py
python download_tinyimagenet.py --root .\data        # for part C
python run_controls.py --models resnet18 --parts B C E --dry-run   # 36 commands
python run_controls.py --models resnet18 --parts B C E
python run_controls.py --models resnet18 --status
```

The same command resumes after an interruption. Results go to
`results_controls\resnet18\` with `runner.log` and `train.log`. Push them
back (checkpoints are not pushed):

```powershell
robocopy results_controls artifacts\controls /E /XD ckpt /XF *.pt
git add artifacts\controls
git commit -m "ResNet-18 control results (in progress)"
git pull --rebase
git push
```

Do not run these on both PCs for the same model, and do not run the
MobileNetV2 or VGG-16 controls again. Keep the PC awake (Settings, Power:
sleep Never) and the window open.

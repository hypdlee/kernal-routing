# Running the 132-run program on four more backbones (Windows PowerShell)

The ResNet-18 study, replicated exactly on MobileNetV2, VGG-16 (BN),
ResNet-50 and ConvNeXt-Tiny: same 10 groups, 22 arms, six seeds, recipe,
margin and dynamic-layer rule. 528 runs. Read `PREREGISTRATION_MODELS.md`
first and commit it before the first run finishes.

Everything below is PowerShell, from the repository folder.

## 1. Environment (once)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
Reports need a LaTeX install with `pdflatex` on PATH (MiKTeX works) plus
`pip install matplotlib scipy`.

## 2. Data

CIFAR-100 downloads itself on the first run. Tiny-ImageNet, if it is not
already in `.\data` from the ResNet-18 study:

```powershell
python download_tinyimagenet.py --root .\data
```

The runner stops, with this message, before the first Tiny-ImageNet job if
the data is missing; rerun the same command afterwards and it continues.

## 3. Check the code before spending GPU time (about 15 minutes on CPU)

```powershell
python tests\test_paths.py
python tests\test_models.py all       # builds, placement, Proposition 1, shuffle, ResNet-18 identity
python tests\test_train.py all        # old commands unchanged, accumulation, probe, resume, 10 end-to-end runs
python tests\test_runner.py all       # 4 x 132 jobs, order, resume, gate, micro-batch, divergence
python tests\test_analysis.py all     # reproduces the ResNet-18 paper byte for byte
python tests\test_report.py all       # 5 per-model PDFs, the rebuilt paper, the cross-model PDF
```

## 4. See what will run

```powershell
python run_all_models.py --dry-run --models mobilenetv2   # every command, nothing executed
python tools\flops.py                                      # compute estimate (writes COMPUTE_ESTIMATE.md)
```

## 5. Run

```powershell
python run_all_models.py
```

That runs all four backbones in order (MobileNetV2, VGG-16, ResNet-50,
ConvNeXt-Tiny) into `.\results_models\<model>\`. Or one at a time:

```powershell
python run_all_models.py --models mobilenetv2
```

What happens, per backbone:

1. **Probe.** Before each group, a few steps of its heaviest arm at batch
   128; on out-of-memory it tries 64 x 2, 32 x 4, 16 x 8 and keeps the
   first that fits for every arm of the group. Saved in `plan.json`; any
   accumulation is written to `DEVIATIONS.md`.
2. **Gate.** The first run is static, CIFAR-100, width 64, seed 0. Below
   its floor (MobileNetV2 65, VGG-16 65, ResNet-50 70, ConvNeXt-Tiny 60)
   the backbone stops, `GATE_FAILED.json` is written, and the runner moves
   to the next backbone. Report it; do not lower the floor after the fact.
3. **The other 131 runs.** Each writes one JSON. A run whose loss goes
   non-finite leaves a `.DIVERGED` marker and the program continues.

**Stopping and resuming.** Ctrl+C at any time. Run the same command again:
finished runs are skipped, an interrupted run resumes from its last epoch
(`ckpt\` folder). Keep the machine awake (Settings, System, Power: Sleep
never) for long stretches.

**Progress.**

```powershell
python run_all_models.py --status
Get-Content .\results_models\mobilenetv2\runner.log -Tail 20 -Wait
```

**Another GPU in the same PC.** Run backbones in parallel, one per GPU:

```powershell
$env:CUDA_VISIBLE_DEVICES=0; python run_all_models.py --models mobilenetv2 vgg16
# second PowerShell window
$env:CUDA_VISIBLE_DEVICES=1; python run_all_models.py --models resnet50 convnext_tiny
```

## 6. Reports

Per backbone (as soon as its 132 runs exist; partial results work too and
the coverage table says what is missing):

```powershell
python report\build_model_report.py --model mobilenetv2 --results .\results_models\mobilenetv2
```

→ `report\out\mobilenetv2\report_mobilenetv2.pdf`: 10 figures and 13
tables, the same as the ResNet-18 paper, plus the coverage and the
equal-parameter tables.

ResNet-18 from your own result files (the 132 JSONs of the original study):

```powershell
python report\build_model_report.py --model resnet18 --results .\results
python report\build_paper.py --results .\results        # the paper, with the corrections
```

All backbones side by side:

```powershell
python report\cross.py --results resnet18=.\results mobilenetv2=.\results_models\mobilenetv2 vgg16=.\results_models\vgg16 resnet50=.\results_models\resnet50 convnext_tiny=.\results_models\convnext_tiny
```

→ `report\out\cross\cross_models.pdf`.

The analysis refuses to run, and says why, if two files claim the same
cell, if the recipe differs inside one backbone, or if a run's parameter
count does not match `PARAMS_EXPECTED.md` (wrong width, K or arm).

## Files

| file | what |
|---|---|
| `src\models\__init__.py` | the five backbones, placement rule, registry |
| `src\program.py` | the 132-run program, defined once |
| `src\train.py` | one run; new flags `--model --placement --arm-name --accum --max-steps --ckpt` |
| `run_all_models.py` | the runner (probe, gate, resume, divergence) |
| `report\` | data loading, figures, tables, per-model, paper and cross-model builds |
| `PARAMS_EXPECTED.md` | counted parameters for every cell |
| `COMPUTE_ESTIMATE.md` | FLOPs-scaled estimate (an estimate) |
| `GATES_MODELS.md` | every check behind this package, with its evidence |

# Kernal routing (KernelRoute)

Where does the gain in dynamic convolution come from? A controlled study of
routing, capacity and width: ResNet-18 (132 runs, complete), MobileNetV2
(132 runs, complete), and the same 132-run program on VGG-16, ResNet-50 and
ConvNeXt-Tiny (pre-registered, running).

Private research repository of Ankit Kumar.

## Where things are

| path | what |
|---|---|
| `paper/two_backbones/` | current paper draft (ResNet-18 + MobileNetV2): `main.tex`, `fig/`, PDF |
| `paper/resnet18_revised/` | the ResNet-18 paper, revised 30 Sep 2026 (corrected parameter counts, initialisation disclosure) |
| `paper/reports/` | earlier technical reports v1 to v3, the first final report, per-model report format, synthetic cross-model preview |
| `src/`, `run_all_models.py`, `tools/`, `tests/` | training code: five backbones, the 132-run program, runner with probe, gate, resume |
| `report/` | analysis and report generators (per-model, cross-model, paper builds) |
| `PREREGISTRATION.md`, `PREREGISTRATION_MODELS.md` | pre-registrations (ResNet-18 stage 1; five backbones, 30 Sep 2026) |
| `GATES_*.md` | check ledgers with recorded evidence |
| `README_MODELS.md`, `docs/KernelRoute_Run_Guide.pdf` | how to run the five-backbone program (PowerShell) |
| `README_RESNET18.md` | the original ResNet-18 study README |
| `artifacts/resnet18/` | 24 stage-1 result JSONs and the training logs shared during the study |
| `artifacts/mobilenetv2/` | MobileNetV2 `train.log` (all 132 runs) and the per-seed CSV parsed from it |
| `docs/` | research ideology documents, day-1 notes, run guide source |
| `history/` | every earlier code and paper package, as delivered, for provenance |

## Rebuild

```powershell
pip install -r requirements.txt
python tests\test_analysis.py all
python report\build_paper2.py --r18-results <ResNet-18 results dir> --mb-log artifacts\mobilenetv2\train.log
```

`--r18-fixture` instead of `--r18-results` rebuilds ResNet-18 from the per-seed
values printed in the paper. The full ResNet-18 and MobileNetV2 result JSONs live
on the training machine (`results/`, `results_models/`) and are not committed.

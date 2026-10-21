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
| `artifacts/resnet18/` | 24 stage-1 result JSONs (CIFAR-100 width 64) and the training logs shared during the study. The other 108 ResNet-18 runs exist here only as per-run lines in the logs; the routing entropies of CIFAR-100 widths 16 and 32 and of the annealed groups have no source file in this archive |
| `tools/audit_paper2.py` | independent audit of the paper: ResNet-18 accuracies against the logs and result files, MobileNetV2 headline statistics recomputed with scipy, method claims against the code, banned phrases. `python tools/audit_paper2.py` |
| `artifacts/mobilenetv2/` | MobileNetV2 `train.log` and `runner.log` (all 132 runs), the per-seed CSV parsed from the log, and `results/`: the 132 result JSONs (accuracy, history, per-layer routing statistics), the 10 memory probes and the runner plan |
| `docs/` | research ideology documents, day-1 notes, run guide source |
| `artifacts/session_uploads/` | every file shared during the project: logs, crash traces, screenshots, the stage-1 results zip |
| `artifacts/mobilenetv2/analysis_first_pass/` | the first, independent MobileNetV2 parser and analysis, used to cross-check the paper |
| `paper/per_model_examples/` | per-model report format: ResNet-18 from the paper's values, the other four on SYNTHETIC test data (watermarked) |
| `history/` | every earlier code and paper package as delivered, loose scripts, the dated fixes, the original report generators |

## Rebuild

```powershell
pip install -r requirements.txt
python tests\test_analysis.py all
python report\build_paper2.py --r18-fixture --mb-log artifacts\mobilenetv2\train.log --mb-results artifacts\mobilenetv2\results
python report\build_docx.py      # Word version from the same build: paper\two_backbones\KernelRoute_paper.docx
python tools\audit_paper2.py      # independent checks; must print no FAIL
```

`--r18-fixture` rebuilds ResNet-18 from the per-seed values printed in the
paper; `--r18-results <dir>` reads ResNet-18 result JSONs instead. MobileNetV2
accuracies are read from the log, routing entropies from the result files, and
the build stops if any final accuracy in the result files differs from the log.
The full ResNet-18 result JSONs live on the training machine (`results/`); only
the 24 stage-1 files in `artifacts/resnet18/` are committed.

Routing entropies, on every backbone, describe the last test batch only (16
images, the same in every run on a dataset); see the threats section of the
paper and `README_RESNET18.md`.

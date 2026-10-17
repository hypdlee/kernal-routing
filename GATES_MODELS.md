# Gates: replicate the 132-run ResNet-18 program on four more backbones

Scope agreed 2026-09-30: all 528 runs (132 x 4), model by model, order
MobileNetV2 -> VGG-16-BN -> ResNet-50 -> ConvNeXt-Tiny. Dynamic placement
copies the ResNet-18 rule literally. ConvNeXt uses the same SGD recipe,
behind a gate.

Nothing below is checked until its EVIDENCE line holds real output.

## A. Models

- [x] A1 five models build through one registry
  CHECK: python -c "from src.models import MODELS; print(sorted(MODELS))"
  EXPECT: ['convnext_tiny', 'mobilenetv2', 'resnet18', 'resnet50', 'vgg16']
  EVIDENCE: ['convnext_tiny', 'mobilenetv2', 'resnet18', 'resnet50', 'vgg16'] (2026-09-30)

- [x] A2 every model x dataset x width x K x router builds and runs forward
  CHECK: python tests/test_models.py build
  EXPECT: 390 configurations ok, 0 failed (5 models x 2 datasets x 3 widths x (1 static + 3 K x 4 routers))
  EVIDENCE: 390 configurations ok, 0 failed (expected 390) in 121s

- [x] A3 literal placement rule gives the documented dynamic-layer counts
  CHECK: python tests/test_models.py placement
  EXPECT: resnet18 16, resnet50 16, vgg16 12, mobilenetv2 17, convnext_tiny 18
  EVIDENCE: spatial placement: resnet18 16, resnet50 16, vgg16 12, mobilenetv2 17, convnext_tiny 18; alt rule 52/12/50/57

- [x] A4 Proposition 1 holds at model level for every model (uniform arm
  equals a static network whose kernels are the bank means, float64)
  CHECK: python tests/test_models.py collapse
  EXPECT: 5 models, max abs diff < 1e-9
  EVIDENCE: 5 models, max abs diff 0.00e+00; negative control routed arm differs 3.1e-04 to 4.7e-02

- [x] A5 shuffled router still permutes in eval, every model
  CHECK: python tests/test_models.py shuffle
  EXPECT: 5 models ok
  EVIDENCE: 5 models ok; relative diff 4.08e-03 (mobilenetv2) to 6.67e-01 (vgg16)

- [x] A6 ResNet-18 through the registry is identical to the original class
  CHECK: python tests/test_models.py resnet18_identity
  EXPECT: parameter counts 11220132 / 44176548 / 44464500 at w64 and state-dict keys identical
  EVIDENCE: params 11220132 / 44176548 / 44464500, state-dict keys identical, ok

- [x] A7 expected parameter counts tabulated for every model/dataset/width/K/arm
  CHECK: python tools/param_table.py --check
  EXPECT: PARAMS_EXPECTED.md written, 0 missing rows
  EVIDENCE: PARAMS_EXPECTED.md written: 95 rows, 0 missing; resnet18 vs training logs 11 of 11 match at 2 dp. FINDING: 6 of 19 cells of the final paper's parameter table differ from the count in the third decimal (e.g. w16 static printed 0.708, counted 0.7131); logs agree with the count, so the paper table is corrected in the regenerated report.

## B. Training script

- [x] B1 --model flag, default resnet18, old commands unchanged
  CHECK: python tests/test_train.py backcompat
  EXPECT: ok
  EVIDENCE: 4 old command lines (gap, uniform, --scope none, anneal) run through the frozen v1 script and the new one: same file name, identical per-epoch losses, final_acc, params and routing

- [x] B2 gradient accumulation keeps the optimiser-step count and LR schedule
  of batch 128
  CHECK: python tests/test_train.py accum
  EXPECT: steps and final lr identical for (128,1) and (64,2); loss scaled by 1/accum
  EVIDENCE: one step at accum 1/2/4 (float64, BN eval): max param diff 3.5e-18, same step count and lr; script (128,1) vs (64,2): identical lr per epoch, record micro_batch 64 accum 2

- [x] B3 probe mode reports peak memory and seconds per step and writes JSON
  CHECK: python tests/test_train.py probe
  EXPECT: ok
  EVIDENCE: probe JSON: mobilenetv2 w16 micro 64 accum 2, oom false, sec_per_step 1.59, steps_per_epoch 4, no result file written

- [x] B4 result JSON records model, placement, micro-batch, accum
  CHECK: python tests/test_train.py config
  EXPECT: ok
  EVIDENCE: vgg16__cifar100__w16__K4__const_static_seed0.json: model=vgg16, placement=spatial, arm=static, micro_batch=32, accum=4, resumed_at_epoch None. Extra: checkpoint resume test resumed_at_epoch=1, epochs [0,1,2], checkpoint removed

- [x] B5 one fake-data epoch end to end for every model on both datasets
  CHECK: python tests/test_train.py e2e
  EXPECT: 10 runs ok, each JSON has routing report for dynamic arms
  EVIDENCE: 10 runs ok (5 models x cifar100, tinyimagenet; gap, width 16, 1 fake epoch); routing layers 16/16/12/17/18 as expected. TIN at micro-batch 32: batch 128 needs ~7 GB fp32 on the CPU container

## C. Runner

- [x] C1 exactly the ResNet-18 program per model: 132 jobs, same
  (dataset, width, K, recipe, arm, seed) set
  CHECK: python tests/test_runner.py grid
  EXPECT: 4 models x 132 jobs, set-equal to the ResNet-18 program
  EVIDENCE: 4 models x 132 jobs ok, each set-equal to the independently written ResNet-18 program; scope/router/placement/batch/epochs/anneal flags asserted per job; 528 result files

- [x] C2 order: gate run first, Tiny-ImageNet width 64 seed-major
  CHECK: python tests/test_runner.py order
  EXPECT: ok
  EVIDENCE: gate first; TIN w64 seed-major (static, uniform, gap per seed); other groups arm-major in program order

- [x] C3 resume skips finished runs
  CHECK: python tests/test_runner.py resume
  EXPECT: ok
  EVIDENCE: crash at run 41 -> 40 done; rerun runs the other 92 (no repeats); third call runs 0

- [x] C4 gate below floor stops that model, logs it, moves to the next
  CHECK: python tests/test_runner.py gate
  EXPECT: ok
  EVIDENCE: mobilenetv2 at 40.0 < 65: 1 run spent, GATE_FAILED.json written and logged, vgg16 continued; rerun skips; 65.0 at floor 65 passes

- [x] C5 per-group probe picks micro-batch, persists the choice, logs any
  deviation from batch 128
  CHECK: python tests/test_runner.py microbatch
  EXPECT: ok
  EVIDENCE: OOM above micro 32: 10 groups probed on heaviest arm (shuffled at w64), accum 4 persisted in plan.json, 10 DEVIATIONS.md lines, applied to all 132 runs; restart re-probes nothing; fits at 128 -> no deviation. Extra: diverge test, 2 DIVERGED markers, status 130+2 of 132, not retried

- [x] C6 full 132-job program completes end to end on fake data for one model
  CHECK: manual (long)
  EVIDENCE: real runner + real src/train.py, fake data, 1 epoch, mobilenetv2: 132 of 132 results, 0 diverged; probe ladder fired for real (TIN w32 -> micro 64, TIN w64 -> micro 32, both in DEVIATIONS.md); two memory kills mid-run and resume from the same command at job 91; exposed and fixed: evaluation batch 256 did not shrink with accumulation. The 132 files load in the analysis with 0 errors (params match PARAMS_EXPECTED).

## D. Analysis

- [x] D1 report/data.py + generators reproduce the published ResNet-18 numbers
  CHECK: python tests/test_analysis.py resnet18
  EXPECT: all reference values match to 3 dp
  EVIDENCE: 13 of 13 tables identical, 130 of 130 macros identical, 10 of 10 figures byte-identical (creation date stripped) vs the paper's own generators on the paper's data (counted params both sides); runs 132, GPU-h 204, +2.282, matched +4.345/+2.313/+2.520, 14x. Real files: 24 accuracies and 24 per-seed entropies from results.zip equal the paper's.

- [x] D2 check_grid treats model as critical config
  CHECK: python tests/test_analysis.py checkgrid
  EXPECT: ok
  EVIDENCE: resnet18 (legacy) vs vgg16: CRITICAL MISMATCH model; legacy vs explicit resnet18: consistent. Loader refuses duplicate cells, recipe drift and wrong parameter counts.

- [x] D3 analysis runs on each new model's (synthetic) results without error
  CHECK: python tests/test_analysis.py synthetic
  EXPECT: 4 models ok
  EVIDENCE: 4 models: 13 tables, 10 figures each on synthetic data with 1 missing + 1 diverged run each, shown as --

## E. Report

- [x] E1 per-model report: 10 figures + 13 tables, 0 LaTeX errors, for all 5 models
  CHECK: python tests/test_report.py per_model
  EXPECT: 5 PDFs, 50 figures, 65 tables, 0 errors, 0 undefined refs
  EVIDENCE: 5 PDFs (resnet18 from paper fixture, 4 synthetic watermarked), 50 figures, 65 tables (+ coverage and interpolation per model), 0 errors, 0 undefined refs, 12 pages each

- [x] E2 ResNet-18 report numbers equal the final paper's
  CHECK: python tests/test_report.py resnet18_values
  EXPECT: ok
  EVIDENCE: rebuilt paper: all 695 decimal numbers of the delivered final PDF present except 44.47 x2 (gap/shuffled CIFAR w64, counted 44.4645M -> 44.46); added ratio column 0.99 x3 and the correction note; 0 errors, 0 undefined

- [x] E3 cross-model report compiles with its figures and tables
  CHECK: python tests/test_report.py cross
  EXPECT: 0 errors
  EVIDENCE: cross_models.pdf: 5 backbones, 6 figures, 8 tables, 0 errors, 0 undefined; a failed gate is listed as an absent row with its accuracy, not dropped

- [x] E4 every page of every report looked at
  CHECK: manual
  EVIDENCE: every page viewed after the final layout change: 5 per-model reports x 9 pages, rebuilt paper 14 pages, cross-model 6 pages (78 pages). Fixed on the way: float pile-ups (placeins + !htbp), X3 ratio labels colliding with tick labels

- [x] E5 matched budgets: pairs labelled with the real parameter ratio; the
  word "matched" only within 10 percent
  CHECK: python tests/test_report.py matched
  EXPECT: ok
  EVIDENCE: per-model tables: 6 matched, 9 not matched, every label consistent with its ratio; verdict sentences and cross table consistent

## F. Pre-registration, operations, delivery

- [x] F1 PREREGISTRATION_MODELS.md: dated predictions, placement, init
  convention, gate floors, micro-batch policy
  CHECK: manual
  EVIDENCE: PREREGISTRATION_MODELS.md dated 30 September 2026: design table (132 per backbone), recipe, adaptation, placement with counts and dynamic share, init convention, parameter ratios known in advance, statistics, P1-P3 with hold rules, gate floors, micro-batch/divergence/interruption rules, compute estimate

- [x] F2 README_MODELS.md with PowerShell commands, order and resume
  CHECK: manual
  EVIDENCE: README_MODELS.md: PowerShell setup, data, 6 test commands, dry run (528 commands verified), run/resume/status/2-GPU, per-model, paper and cross-model report commands

- [x] F3 compute estimate per model from counted FLOPs, labelled an estimate
  CHECK: python tools/flops.py
  EXPECT: table for 5 models
  EVIDENCE: COMPUTE_ESTIMATE.md: resnet18 204 GPU-h (equals the paper's logged total, sanity check), mobilenetv2 35, vgg16 121, resnet50 467, convnext_tiny 518; labelled ESTIMATE, depthwise models flagged as underestimated.

- [x] F4 ResNet-18 init mismatch (static default init vs Kaiming-normal bank)
  disclosed in the pre-registration and in the paper's threats section
  CHECK: manual
  EVIDENCE: PREREGISTRATION_MODELS.md section Initialisation; paper threats paragraph 'Initialisation' present in rebuilt report/out/paper/main.tex (verified in src/resnet.py: only BN is initialised explicitly, convs keep PyTorch default)

- [x] F5 whole test suite passes from a fresh extraction of the delivered zip
  CHECK: manual
  EVIDENCE: fresh unzip of kernelroute-models.zip (66 files): test_paths rc 0; test_models all rc 0 (390 configs, 5 models); test_train all rc 0 (backcompat identical, 10 e2e runs); test_runner all rc 0 (7 tests); test_analysis all rc 0 (13/13 tables, 130/130 macros, 10/10 figures identical); test_report all rc 0 (5 PDFs, 50 figures, 65 tables; paper numbers); param_table --check rc 0

## G. Findings raised while building (must reach the user and the report)

- [x] G1 matched budgets do not transfer: parameter ratio gap(w)/static(2w) is
  0.99 for ResNet-18 and 1.00 for VGG-16, but 0.51-0.60 for MobileNetV2, 0.61
  for ResNet-50 and 0.28-0.30 for ConvNeXt-Tiny. Report the real ratio, call a
  pair matched only within 10 percent, and add a clearly labelled
  interpolated-static comparison at equal parameters.
  CHECK: python tests/test_report.py matched
  EXPECT: ok
  EVIDENCE: per-model MATCHED table and verdicts carry the real ratio and label matched only within 10 percent; tab_INTERP gives the interpolated equal-parameter comparison; cross X3 and XMATCHED do the same (E5 test ok)

- [x] G2 router size does not transfer: the ResNet-18 router (MLP, hidden
  C_in/4) is 1.3-2.9 percent of the static network for four backbones but
  48 percent for MobileNetV2 (6x the bank). Disclosed in the
  pre-registration, the per-model setup text and the cross-model summary
  and threats.
  CHECK: grep router share in built reports
  EVIDENCE: prereg table (router/static 2.6, 2.7, 1.3, 48.4, 2.9 percent; bank adds 0.19M for MobileNetV2); per-model setup text prints router share (verdict macro prouterpct); cross summary column 'router (% of static)' and a Router size threats paragraph

- [x] C7 out of memory mid-run (probe fitted): group steps down one rung from
  that run on, logged in DEVIATIONS.md and plan.json, run redone, earlier
  runs keep their micro-batch and the report lists it per arm; probe also
  checks the evaluation pass
  CHECK: python tests/test_runner.py oom_midrun ; python tests/test_train.py probe
  EXPECT: ok
  EVIDENCE: oom_midrun: OOM in gap seed 2 of CIFAR w32 -> 14 earlier runs kept micro 128, that run and the 3 after used micro 64, plan.json raised_after + DEVIATIONS.md line, 132 results; probe test ok with the evaluation pass included. Also exercised for real in C6: a run killed for memory stopped the runner, the same command resumed at job 91 with the saved plan

ABANDON lines go here if a gate becomes impossible, with the reason.

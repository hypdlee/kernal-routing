# Audit of the two-backbone paper, 19 October 2026

Trigger: Section 3.3 said "Every run uses the right side" although `uniform`
takes a third code path. The user asked for the whole paper to be verified
against the code and data before delivery. Method: every hand-typed factual
claim in `report/paper2_template.tex` was read against `src/`, the run
scripts, the 132 MobileNetV2 result files, the ResNet-18 logs and result
files, and (for related work) the cited papers. Reproduce with
`python tools/audit_paper2.py` (43 checks, 0 failures).

## Errors found and fixed

| # | where | wrong | now |
|---|---|---|---|
| 1 | 3.3 | "Every run uses the right side" (blend); `uniform` uses a constant-kernel path; heading said two paths | Three execution paths; which arm uses which; `combine` never used for a result |
| 2 | 3.3 | "1.2 GB per stage-4 layer" | per 512-to-512 stage-4 convolution, fp32 (the first stage-4 conv is 0.6 GB) |
| 3 | 3.4 | "five warm-up epochs then cosine decay over 150 epochs" | 150 epochs total: 5 warm-up, cosine over the remaining 145 (`train.py`) |
| 4 | 3.4 | "every channel count" scaled at widths 32 and 16 | the final 1x1 convolution keeps 1280 channels at every width (`make_divisible(1280*max(1,m))`); the pre-registration text had said "every channel count" |
| 5 | 3.4 | router hidden width "C_in/4" | max(16, C_in/4) |
| 6 | 3.1 | frozen router "frozen at its random initialisation" | one fixed seed (20260908) independent of the run seed, so the six seeds share one random router; Gaussian/sqrt(fan-in), zero biases. Added to Section 7 and Threats |
| 7 | 3.1 | bank-size groups "at CIFAR-100 widths" | K=2 and K=8 at width 32 only |
| 8 | 3.4 | "quadruples the cost of the deeper stages" | every stage |
| 9 | 3.4 | ResNet-18 zero-init of last BN per block not stated | stated |
| 10 | 3.6 | "stopped by a GPU fault at epoch 125" | CUDA illegal-memory-access error shortly after epoch 120, resumed at epoch 125 (train.log) |
| 11 | 4 | margin "an order of magnitude below" 3.38 to 5.61 pp | seven to eleven times smaller |
| 12 | 5 | "the largest [R18 bank term] is +0.175" | largest in magnitude is -0.223 (width 16) |
| 13 | 5 | MobileNetV2 bank point estimates "lie inside the margin" (+0.593 does not) | +0.593 lies just outside; CIs extend beyond the margin, so neither equivalence nor a difference is shown |
| 14 | 7 | "trained router's entropy is below the controls, so it does vary with the input" (does not follow: a constant non-uniform mixture also has low entropy) | claim removed |
| 15 | 9, abstract, conclusion | "at equal parameters" used for MobileNetV2 where the plain network has 1.7 to 2.0 times the parameters | wording separates the two |
| 16 | 9.1 | ratio range written backwards (0.60 to 0.51) | 0.51 to 0.60 |
| 17 | conclusion | "a few percent more compute" | "at most 8 %" (generated) |
| 18 | 10 | "starved from the first step" (the router starts near uniform) | "can move towards one-hot early in training" |
| 19 | 11 | entropy is "the statistic this literature reports" (no citation) | "a natural statistic" |
| 20 | 11, abstract, conclusion | "within ResNet-18 entropy does not order the benefit" from rho = 0.60, p = 0.21, n = 6, while 0.83 is read as "does" | "did not order it reliably"; the counterexample is the evidence |
| 21 | 10, abstract, conclusion | "annealing never raised accuracy" (R18 width 16 is +0.115, n.s.) | "never detectably"; the count is of significant differences |
| 22 | 2 | FDConv "kernels of DY-Conv" and "buy little adaptability" | it states this for the four parallel kernels of ODConv and says "limited diversity" |
| 23 | 13 | VGG-16 "keeps every parameter" in dynamic convs | "nearly every" (first conv and classifier are static) |
| 24 | 14 | "every result file ... accompany this report" | only 132 MobileNetV2 and 24 ResNet-18 width-64 files are in the archive |
| 25 | 14 | no statement about which ResNet-18 numbers can be re-derived | added; see below |
| 26 | layout | `tab:vsstatic` 14 pt overfull (the earlier gate said none) | wrapped |
| 27 | title | draft date 17 October | 19 October |

## Verified without change

Recipe constants, SGD Nesterov, augmentation and padding, 16 dynamic convs in
ResNet-18, 17 depthwise in MobileNetV2, stem strides, router MLP shape,
`uniform` = mean kernel, 16-image last batch (10000 mod 256), TOST formula,
Wilcoxon floor 2^(1-n), entropy expansion, 1/K +- 0.022 arithmetic, 3.38 /
4.01 / 5.61 (KernelWarehouse Table 2, 300-epoch recipe), SD-Conv (50 % pruned,
-0.5 to -0.8 on CIFAR-100), bibliography volumes and pages, ResNet-50 about
half of parameters in 3x3 convs (48 %), batch 128 and no accumulation in all
132 MobileNetV2 runs, resumed run, MobileNetV2 headline statistics recomputed
independently (bank 2 of 5 equivalent, routing 5 of 5, frozen-gap +0.152,
shuffled-gap -1.155, K8-K2 +0.112, annealed rows, equal-parameter pairs).

## Not verifiable from this archive

The ResNet-18 routing entropies for CIFAR-100 width 16, width 32 and the
three annealed groups (30 per-seed values) have no source file here. They
feed the entropy section (Spearman 0.60, the width-16 counterexample, the
width trend). The ResNet-18 result JSONs for those groups are needed to close
this. The width-64 entropies (gap, shuffled, frozen_random) match the 24
result files.

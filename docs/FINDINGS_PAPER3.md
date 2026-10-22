# Findings ledger: three-backbone paper (ResNet-18, MobileNetV2, VGG-16)

Fixed wording. Abstract, introduction, result sections and conclusion state
these findings with this scope and these numbers. Values are generated
macros (value at 21 Oct 2026 in brackets; pooled and Holm values added 22 Oct 2026). 15 backbone and condition pairs =
3 backbones x 5 conditions (CIFAR-100 w16/32/64, Tiny-ImageNet w32/64).

Terminology unchanged: routing = mixing weights depend on the input image;
trained router = router whose weights are optimised; bank term = uniform -
static; routing term = gap - uniform.

| id | finding | scope | numbers |
|---|---|---|---|
| F1 | The kernel bank on its own adds nothing measurable. | all three | tested once, pooled (added 22 Oct, after the runs): random-effects bank term over 15 pairs -0.007 [95% -0.098, +0.084; 90% -0.083, +0.070], equivalent at +/-0.1 pp, heterogeneity p 0.146; fixed-effect per backbone R18 +0.005, MBv2 -0.098, VGG +0.019, each 90% interval inside +/-0.25. Per pair (descriptive): never significant in 15 of 15 (allbanksig 0); uncorrected equivalent in 9 of 15, 0 after Holm (0/9/14/14 uncorrected and 0/0/9/14 Holm at 0.25/0.5/0.75/1 pp) (R18 5, MBv2 2, VGG 2); the other 6 inconclusive (MBv2 3, VGG 3). VGG and MBv2 share one initialisation for static and bank kernels. |
| F2 | Dynamic convolution beats the plain network in 13 of 15 pairs; the bank term is never significant, and the routing term is significant in 12 of those 13. | all three; where the null falls is backbone-dependent | alleff 13 (12 after Holm; MBv2 C100 w16 drops), alleffrout 12; routing term positive in 13, 12 after Holm (VGG C100 w16 drops); pooled routing +1.234 [0.827, 1.640], heterogeneity p < 1e-4 (the other: VGG C100 w32, routing +0.627, p 0.0569, total +0.478). Nulls (allnull 2): CIFAR-100 w64 on R18 (+0.080) and VGG (+0.185, equivalent); MBv2 +1.360 there. VGG C100 w64: routing term +0.373 significant although total is not (uniform -0.188, equivalent). Null unexplained; not saturation (TIN w64 effect on both R18 and VGG at low static loss); not temperature (survives annealing on both); does not need residual connections (VGG has none). |
| F3 | Whether the router must be trained depends on the backbone. | backbone-dependent (CIFAR-100 w64 only) | MBv2: frozen-gap +0.152 (p 0.52), frozen-uniform +1.625 higher. VGG: gap-uniform +0.373 higher, frozen-uniform +0.093 equivalent, frozen-gap -0.280 inconclusive (p 0.074); no arm beats static on VGG. R18: uninformative (every arm equivalent to static). shuffled-gap lower on MBv2 (-1.155) and VGG (-0.418); inconclusive on R18. Frozen router = one random draw shared by all seeds. |
| F4 | On CIFAR-100 at width 32, banks larger than K=2 are not distinguishable from K=2. | all three (CIFAR-100 w32 only) | K8-K2: R18 +0.098 equivalent; MBv2 +0.112 inconclusive; VGG +0.197 inconclusive. On VGG K=2 is not distinguishable from static (+0.258, p 0.30); K=4 and K=8 beat it. |
| F5 | Whether routing is worth its cost depends on the cost held fixed. | backbone-dependent | twice the width: equal parameters on R18 (0.99) and VGG (1.00): plain wins 6 of 6 (+2.3 to +5.2); MBv2 (plain 1.7-2.0x params) wins 1 of 3. Equal width, at most 8 / 2 / 19 % more MACs (R18 / MBv2 / VGG): routing beats plain in 4 / 5 / 4 of 5. |
| F6 | On CIFAR-100, annealing the temperature never detectably improved accuracy. | all three (CIFAR-100 only) | significantly lower in 7 of 9 (MBv2 3/3, VGG 3/3, R18 w32); never significantly higher; R18 w16 +0.115 n.s. |
| F7 | Routing entropy is not a reliable measure of whether routing helps. | all three backbones (VGG-16 added 22 Oct from its 132 result files) | within backbones: R18 rho 0.60 (p 0.208) with the w16 counterexample (annealed 14x flatter, larger benefit +0.87 vs +0.76); MBv2 rho 0.83 (p 0.042), no counterexample (7.5x flatter, smaller +0.16 vs +1.32); VGG rho 0.77 (p 0.072), no counterexample (6.7x flatter, smaller +0.11 vs +0.72). Across backbones at C100 w64: 1-H 0.0115 / 0.0132 / 0.0133 with gap-uniform +0.03 (equivalent) / +1.47 / +0.37 (both higher). Width trend (const): R18 flattens, MBv2 sharpens, VGG flattens (0.0215, 0.0181, 0.0133). After annealing: MBv2 flat in 15-16 of 17 layers, VGG 10-12 of 12 (non-flat only layers 1 and 2); annealed w64: MBv2 1-H 0.00069 with +0.55 higher, VGG 0.00014 with +0.17 higher, R18 0.00018 equivalent. Router outputs at C100 w64: trained router furthest from uniform on all three; frozen 4.5x (MBv2) and 6.6x (VGG) closer; shuffled near-uniform on R18 (0.0004) and MBv2 (0.00005) but not on VGG (0.0020, same as frozen). Table 7 entropy rows: counterexample backbone-dependent (R18 vs MBv2 and VGG); width trend backbone-dependent (R18 and VGG vs MBv2); same-entropy row across backbones. Counts unchanged: ncontra 10, Hold 4, BD 5, X 1. |
| P | Seven predictions recorded before the runs that test them; four failed. | | R18 TIN w64 saturation (failed); MBv2 P1 failed (2/5), P2 failed (1/3), P3 held (5/5); VGG P1 failed (2/5), P2 held (3/3), P3 held (4/5). Gates: MBv2 72.66, VGG 74.68 (floors 65). |

Provenance: VGG-16 values parsed from its train.log; every final equals the
runner log's; parameter counts in the log equal PARAMS_EXPECTED.md; batch
128 without accumulation in all 132 runs; one run resumed (shuffled, seed
0, CIFAR-100 w64, at epoch 58); without it shuffled-static -0.254
(inconclusive) and shuffled-gap -0.474 (lower), same verdicts.

ResNet-18 result files (22 Oct 2026): all 132 present; every per-seed final
accuracy identical to the transcription; every transcribed routing entropy
(including the 30 that had no source file) equals the file value rounded to
the four decimals transcribed. The build now reads ResNet-18 from the files:
training losses move in the fourth decimal (transcribed means were of 3-dp
log prints), entropies in the fifth decimal, Tiny-ImageNet w64 gap minutes
687.2 -> 687.4; no accuracy, p-value or verdict changes.

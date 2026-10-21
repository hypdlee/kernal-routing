# Findings ledger: three-backbone paper (ResNet-18, MobileNetV2, VGG-16)

Fixed wording. Abstract, introduction, result sections and conclusion state
these findings with this scope and these numbers. Values are generated
macros (value at 21 Oct 2026 in brackets). 15 backbone and condition pairs =
3 backbones x 5 conditions (CIFAR-100 w16/32/64, Tiny-ImageNet w32/64).

Terminology unchanged: routing = mixing weights depend on the input image;
trained router = router whose weights are optimised; bank term = uniform -
static; routing term = gap - uniform.

| id | finding | scope | numbers |
|---|---|---|---|
| F1 | The kernel bank on its own adds nothing measurable. | all three | bank term never significant in 15 of 15 (allbanksig 0); equivalent in 9 of 15 (R18 5, MBv2 2, VGG 2); the other 6 inconclusive (MBv2 3, VGG 3). VGG and MBv2 share one initialisation for static and bank kernels. |
| F2 | Dynamic convolution beats the plain network in 13 of 15 pairs; the bank term is never significant, and the routing term is significant in 12 of those 13. | all three; where the null falls is backbone-dependent | alleff 13, alleffrout 12 (the other: VGG C100 w32, routing +0.627, p 0.0569, total +0.478). Nulls (allnull 2): CIFAR-100 w64 on R18 (+0.080) and VGG (+0.185, equivalent); MBv2 +1.360 there. VGG C100 w64: routing term +0.373 significant although total is not (uniform -0.188, equivalent). Null unexplained; not saturation (TIN w64 effect on both R18 and VGG at low static loss); not temperature (survives annealing on both); does not need residual connections (VGG has none). |
| F3 | Whether the router must be trained depends on the backbone. | backbone-dependent (CIFAR-100 w64 only) | MBv2: frozen-gap +0.152 (p 0.52), frozen-uniform +1.625 higher. VGG: gap-uniform +0.373 higher, frozen-uniform +0.093 equivalent, frozen-gap -0.280 inconclusive (p 0.074); no arm beats static on VGG. R18: uninformative (every arm equivalent to static). shuffled-gap lower on MBv2 (-1.155) and VGG (-0.418); inconclusive on R18. Frozen router = one random draw shared by all seeds. |
| F4 | On CIFAR-100 at width 32, banks larger than K=2 are not distinguishable from K=2. | all three (CIFAR-100 w32 only) | K8-K2: R18 +0.098 equivalent; MBv2 +0.112 inconclusive; VGG +0.197 inconclusive. On VGG K=2 is not distinguishable from static (+0.258, p 0.30); K=4 and K=8 beat it. |
| F5 | Whether routing is worth its cost depends on the cost held fixed. | backbone-dependent | twice the width: equal parameters on R18 (0.99) and VGG (1.00): plain wins 6 of 6 (+2.3 to +5.2); MBv2 (plain 1.7-2.0x params) wins 1 of 3. Equal width, at most 8 / 2 / 19 % more MACs (R18 / MBv2 / VGG): routing beats plain in 4 / 5 / 4 of 5. |
| F6 | On CIFAR-100, annealing the temperature never detectably improved accuracy. | all three (CIFAR-100 only) | significantly lower in 7 of 9 (MBv2 3/3, VGG 3/3, R18 w32); never significantly higher; R18 w16 +0.115 n.s. |
| F7 | Routing entropy is not a reliable measure of whether routing helps. | ResNet-18 and MobileNetV2 only (VGG-16 result files, which hold the entropies, not available) | unchanged from the two-backbone ledger (Spearman 0.60 / 0.83; 0.0115 vs 0.0132 with +0.03 vs +1.47; annealed MBv2 flat in 15-16 of 17 layers). |
| P | Seven predictions recorded before the runs that test them; four failed. | | R18 TIN w64 saturation (failed); MBv2 P1 failed (2/5), P2 failed (1/3), P3 held (5/5); VGG P1 failed (2/5), P2 held (3/3), P3 held (4/5). Gates: MBv2 72.66, VGG 74.68 (floors 65). |

Provenance: VGG-16 values parsed from its train.log; every final equals the
runner log's; parameter counts in the log equal PARAMS_EXPECTED.md; batch
128 without accumulation in all 132 runs; one run resumed (shuffled, seed
0, CIFAR-100 w64, at epoch 58); without it shuffled-static -0.254
(inconclusive) and shuffled-gap -0.474 (lower), same verdicts.

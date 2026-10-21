# Findings ledger: two-backbone paper (ResNet-18, MobileNetV2)

Fixed wording. Abstract, introduction, result sections and conclusion all
state these findings with this scope and these numbers. Numbers come from
the generated macros (value at 17 Oct 2026 in brackets).

Terminology: **routing** means mixing weights that depend on the input
image. A **trained router** is one whose weights are optimised. "Routing
term" = gap - uniform. "Bank term" = uniform - static.

| id | finding | scope | numbers |
|---|---|---|---|
| F1 | The kernel bank on its own adds nothing measurable. | both | bank term never significant in 10 of 10 conditions; equivalent within 0.5 pp in 7 of 10 (R18 5/5, MBv2 2/5); the other 3 (all MBv2) inconclusive: estimates small (+0.59, -0.19, -0.18; the first just outside the margin), CIs extend beyond it. This is why MBv2 prediction P1 (equivalence in all 5) failed. |
| F2 | Wherever dynamic convolution beats the plain network, routing carries the effect. Where that happens depends on the backbone. | both / backbone-dependent | routing term significant in 9 of 10 (R18 4/5, MBv2 5/5). The exception is R18 CIFAR-100 w64: gap-static +0.080, equivalent; a well-powered null that is unexplained (recorded saturation prediction failed: TIN w64 +2.282; not caused by temperature). On MBv2 the same condition gives +1.360. |
| F3 | Routing need not be learned on MobileNetV2: a frozen random router does as well as a trained one; routing by another image's descriptor removes the benefit. | MobileNetV2 only (R18 uninformative, no arm differs from static) | frozen-gap +0.152 (p 0.52, not equivalent either); frozen-uniform +1.625; shuffled-static +0.205 (n.s.); shuffled-gap -1.155. CIFAR-100 w64 only. |
| F4 | On CIFAR-100 at width 32, bank sizes 4 and 8 are not distinguishable from K=2. | both (CIFAR-100 w32 only) | K8-K2: R18 +0.098 equivalent; MBv2 +0.112 inconclusive. |
| F5 | Whether routing is worth its cost depends on the cost held fixed and on the backbone. | backbone-dependent | equal parameters: R18 plain at 2c beats routed at c in 3/3 (+2.3 to +4.3); MBv2 plain with 1.7 to 2.0x the parameters wins 1/3. Equal width (at most 8% / 2% more MACs): routing beats plain in R18 4/5, MBv2 5/5. |
| F6 | On CIFAR-100, annealing the temperature (30 to 1) never detectably improved accuracy (significantly lower in 4 of 6; the R18 w16 point estimate +0.115 is not significant). | both | lower in 4 of 6 conditions (all 3 MBv2 widths, -0.92 to -1.28; R18 w32 -0.68); no detectable change in R18 w16 and w64. The R18 w64 null persists under annealing. |
| F7 | Routing entropy is not a reliable measure of whether routing helps. After annealing, routers end close to uniform. | both / across backbones | within R18 it did not order the benefit reliably (Spearman 0.60, p 0.21; w16 counterexample: 14x flatter, +0.870 vs +0.755); within MBv2 it does (0.83; counterexample absent: 7.5x flatter, +0.16 vs +1.32); across backbones at CIFAR-100 w64 nearly equal entropy (0.0115 vs 0.0132) with +0.03 vs +1.47 over uniform. Near-uniform after annealing: R18 w64 1-H 0.00018 and equivalent to uniform; MBv2 w64 1-H 0.00069 and still +0.55 over uniform; MBv2 annealed routers flat in 15-16 of 17 layers. Measured once, at the end of training, on the last 16 test images. |
| P | Four predictions recorded before the runs that test them; three failed. | | R18 TIN w64 saturation (failed); MBv2 P1 (failed, 2/5), P2 (failed, 1/3), P3 (held, 5/5). |

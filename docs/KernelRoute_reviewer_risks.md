---
title: "KernelRoute: how the paper could be rejected, and how to answer"
subtitle: "Reviewer risk analysis for the three-backbone draft of 21 October 2026"
---

# How to use this document

Each section below is one objection a reviewer is likely to raise. For each
one you get: how serious it is, the honest answer, the evidence behind the
answer (with numbers taken from the current build or recomputed for this
document), and what to change in the paper or run before submission.

Severity means how likely the objection is to sink the paper on its own:

* **High**: a reviewer can reject on this alone. Fix before submitting.
* **Medium**: weakens the paper; a reviewer will ask, and a weak answer
  costs a score point.
* **Low**: worth one sentence in the paper or the rebuttal.

Two rules for every answer. First, never claim more than the runs show:
reviewers forgive a limited, honest scope and punish overclaiming. Second,
where a cheap experiment closes the gap, run it before submission instead
of arguing in the rebuttal.

# Summary

| # | Objection | Severity | What closes it | Cost |
|---|---|---|---|---|
| 1 | "Not new: DY-Conv already ablated the attention, and CondConv already argued a constant mixture is one kernel" | High | Cite both explicitly and explain the difference (trained-from-scratch controls vs. test-time substitution) | Writing only |
| 2 | "CIFAR and Tiny-ImageNet only; no ImageNet" | High | Scope statement, cost numbers, venue choice; optional ImageNet-100 or 64-pixel ImageNet subset | Writing; optional about 5 GPU-days |
| 3 | "Your dynamic convolution is weak; published gains are 3 to 5 pp" | High | Show that your plain baselines are much stronger than the ones in the literature, and that the gains are in line once that is accounted for | Writing only |
| 4 | "Many tests, no multiplicity correction" | Medium-High | Pooled (meta-analytic) bank term and Holm-corrected counts, computed below | Writing only |
| 5 | "The backbones disagree; what is the take-away?" | Medium | One practical recommendation per finding; the disagreements are the result | Writing only |
| 6 | "The CIFAR-100 width-64 null is unexplained" | Medium | One cheap experiment that isolates resolution | About 13 GPU-h (ResNet-18) |
| 7 | "The entropy evidence is thin (16 images, missing files, no VGG-16)" | Medium | Upload the result files; rerun two groups with full-test-set entropy; or move the finding to the appendix | 0 to 25 GPU-h |
| 8 | "One random draw of the frozen router, in one condition" | Medium | Two more frozen-router draws on MobileNetV2 and VGG-16 | About 26 GPU-h |
| 9 | "Your MobileNetV2 is not DY-MobileNetV2 (only depthwise layers dynamic)" | Medium | State the pre-registered reason; optionally run the "all layers" placement in one condition | Writing; optional about 20 GPU-h |
| 10 | "ResNet-18 static and bank kernels were initialised differently" | Low-Medium | Rerun the ResNet-18 static arms with the bank's initialisation | About 22 GPU-h |
| 11 | "The pre-registration cannot be verified" | Medium | Register the ResNet-50 and ConvNeXt-Tiny predictions on OSF now, before those results exist | One hour |
| 12 | "The ±0.5 pp margin is arbitrary" | Low-Medium | Margin-sensitivity table, computed below | Writing only |
| 13 | "One recipe for all backbones; MobileNetV2 underfits" | Low-Medium | Deliberate control; baselines are reported and reasonable | Writing only |
| 14 | "Equal compute is counted, not measured" | Low-Medium | Measure latency of the per-sample implementation | About 1 hour |
| 15 | "Your annealing result contradicts DY-Conv" | Low | DY-Conv's own numbers: single runs, +0.5 pp, and τ = 1 below static | Writing only |
| 16 | "Ablations only on CIFAR-100" | Low-Medium | Already scoped in the paper; optional Tiny-ImageNet controls | Optional about 125 GPU-h |
| 17 | "Results transcribed from logs; result files missing" | Medium | Upload the ResNet-18 and VGG-16 result folders | None (files exist on your PC) |
| 18 | "Too long, single author, report-style" | Medium | Choose TMLR; restructure to a main text of about 12 pages plus appendix | Writing only |
| 19 | "Six seeds is small" | Low | Power analysis already in the paper | Writing only |
| 20 | "CNNs only; no transformers, no detection" | Low-Medium | Scope statement; the layer studied is a convolution | Writing only |

The objections in rows 1, 3 and 4 need no GPU time and are the most
dangerous. Fix those first.

# 1. Novelty: "DY-Conv and CondConv already did this"

**Severity: High.** This is the objection most likely to end the paper,
because a reviewer who knows the DY-Conv paper will find it in minutes, and
the current draft does not mention it.

**What the reviewer will point to.**

* DY-Conv (Chen et al., CVPR 2020), Section 5.2, Table 3, on
  DY-MobileNetV2 ×0.5 on ImageNet: full attention 69.4 % top-1; *average
  kernels* (no attention) 36.0 %; attention taken from another image
  27.3 %; attention shuffled per image 14.8 %. The authors conclude that "it
  is crucial for each image to use its own attention".
* CondConv (Yang et al., NeurIPS 2019), Section 3: if the routing function
  is constant for all examples, the layer has the same capacity as a static
  convolution. That is our Proposition 1.

**The answer.** These are different experiments from ours, and the
difference is the paper's contribution.

* DY-Conv's variants change the aggregation of a network that was trained
  *with* attention. The paper does not say explicitly that the network was
  not retrained, but an average-kernel network at 36.0 % is far below the
  static network (65.4 % in the same paper), which is hard to explain unless
  the network was not retrained with a constant mixture: a constant mixture
  trained from scratch is a single kernel and can reach the static
  network's accuracy (our Proposition 1, and our `uniform` arm, which is
  equivalent to `static` in 9 of 15 pairs and never significantly
  different). Their ablation shows that a network trained with attention
  depends on its attention. It does not show that attention is why the
  dynamic network beats a plain one, because the plain network and the
  bank are never compared on equal terms.
* Our `uniform`, `shuffled` and `frozen_random` arms are each trained from
  scratch, six seeds each, and compared paired by seed with a plain network
  and with the routed network. That is the controlled comparison the
  earlier ablations do not make.
* CondConv states the constant-routing argument as theory and does not test
  it. We test it: a trained constant mixture behaves like the plain network
  (pooled bank term −0.007 pp, Section 4 of this document).

**Change in the paper (required).**

* In Related work, add a paragraph on DY-Conv Table 3 and CondConv
  Section 3 with the distinction above.
* Under Proposition 1, write "as noted by Yang et al. (2019)". Do not
  present it as new.
* Cite the survey of dynamic networks (Han et al., *Dynamic Neural
  Networks: A Survey*, IEEE TPAMI 2021), which reviewers expect.

**Rebuttal text.**

> DY-Conv's Table 3 replaces the attention of a network trained with
> attention; an average-kernel network at 36.0 % against a static network
> at 65.4 % shows the trained network depends on its attention, not that
> attention is what makes it better than a plain network. Our controls are
> trained from scratch with six paired seeds, and the constant mixture
> trained this way is indistinguishable from the plain network (pooled
> difference −0.007 pp, 95 % CI [−0.10, +0.08] over 15 backbone and
> condition pairs). We now discuss both DY-Conv's ablation and CondConv's
> Section 3 explicitly.

# 2. Scale: "No ImageNet"

**Severity: High** at CVPR, ICCV, NeurIPS or ICLR main tracks; **Medium**
at TMLR.

**The honest answer.** The study is a controlled comparison with six
paired seeds per arm and equivalence tests, which needs many runs per
condition (396 so far, 564 GPU-hours on one RTX 4070 Ti). One ResNet-18
`gap` run on Tiny-ImageNet at width 64 already takes about 11.5 hours; an
ImageNet run at 224 pixels is out of reach at six seeds per arm. The
claims are stated for the scale tested, and the paper says so (Threats:
Scale).

**What makes the answer stronger.**

* Show the trade-off: six seeds and equivalence tests at small scale
  answer a question that one seed at ImageNet scale cannot, because the
  effects in question are 0.1 to 2 pp and seed noise is 0.2 to 1.0 pp.
* Optional experiment: ImageNet-100 (100 classes, about 130,000 training
  images) at 64 pixels, ResNet-18 width 64, `static`, `uniform`, `gap`,
  six seeds. Scaled from the Tiny-ImageNet width-64 minutes, about 120 GPU
  hours (about 5 days). It adds a third dataset with more images per class
  and tests the bank term once more; it does not reach full ImageNet.

**Rebuttal text.**

> The effects in question are 0.1 to 2 pp, the seed-to-seed standard
> deviation is 0.2 to 1.0 pp, and the central claims are null claims, so
> each needs paired seeds and an equivalence test. This is what limits the
> study to CIFAR-100 and Tiny-ImageNet on our hardware; we state the
> claims at that scale.

# 3. "Your dynamic convolution is weak"

**Severity: High.** A reviewer who sees +0.08 pp for ResNet-18 on
CIFAR-100 at width 64, against +2.3 pp for DY-ResNet-18 on ImageNet in the
DY-Conv paper, may decide the implementation is broken.

**The answer has three parts.**

1. *Your plain baselines are much stronger than the ones in the
   literature.* Plain ResNet-18 on CIFAR-100 at width 64 reaches 78.34 %
   here. SD-Conv (He et al., WACV 2023, Table 1) reports 67.6 % for the
   plain ResNet-18 on CIFAR-100 and 72.4 % for DY-Conv, a gain of 4.8 pp
   over a baseline 10.7 pp below ours. A larger gain over a weaker
   baseline is expected and is not evidence that our layer is weaker.
2. *The layer matches DY-Conv where it matters.* GAP descriptor,
   two-layer perceptron with hidden width C/4, softmax over K = 4 kernels,
   optional 30-to-1 temperature annealing as in DY-Conv. Where it has an
   effect, it beats the plain network by up to +2.3 pp (ResNet-18,
   Tiny-ImageNet width 64). A broken layer would not.
3. *The tests are in the repository.* The two execution paths agree to
   10⁻¹⁰, and the constant-mixture path equals the mean-kernel convolution.

**Change in the paper (recommended).** Add one table: our plain baselines
next to published plain baselines for the same backbone and dataset (with
the source), and our dynamic gain next to theirs. Only use numbers you can
cite.

# 4. Multiple comparisons

**Severity: Medium-High.** The paper runs well over a hundred tests and
corrects none. A statistician reviewer will ask what survives correction.

**What survives (computed for this document from the same per-seed
values).**

| Family (15 backbone and condition pairs) | Significant, uncorrected | After Holm correction |
|---|---|---|
| Routing term > 0 (gap − uniform) | 13 | 12 (VGG-16 CIFAR-100 width 16 drops) |
| Dynamic beats plain (gap − static) | 13 | 12 (MobileNetV2 CIFAR-100 width 16 drops) |
| Bank term equivalent at ±0.5 pp (TOST) | 9 | **0** |
| Bank term equivalent at ±0.75 pp | 14 | 9 |
| Bank term equivalent at ±1.0 pp | 14 | 14 |

The routing findings survive correction almost intact. **The bank
equivalences at ±0.5 pp do not survive a per-pair Holm correction.** This
has to be handled before a reviewer finds it.

**The answer: test the bank claim once, pooled, instead of fifteen times.**
Finding 1 is a single claim ("the bank adds nothing"), so it should rest
on one test. A random-effects pooled estimate over the 15 pairs gives:

| Bank term (uniform − static) | Estimate | 95 % CI | 90 % CI |
|---|---|---|---|
| Pooled over all 15 pairs | −0.007 pp | [−0.098, +0.084] | [−0.083, +0.070] |
| ResNet-18 (5 conditions, fixed effect) | +0.005 pp | | [−0.092, +0.102] |
| MobileNetV2 (5 conditions) | −0.098 pp | | [−0.242, +0.046] |
| VGG-16 (5 conditions) | +0.019 pp | | [−0.077, +0.114] |

Between-condition heterogeneity is not significant (Q test p = 0.15). The
pooled bank term is equivalent to zero at a margin of 0.1 pp, five times
tighter than the pre-registered 0.5 pp, and each backbone separately is
inside ±0.25 pp. The routing term, pooled the same way, is +1.23 pp
[0.83, 1.64] with strong heterogeneity (p < 0.001), which is exactly the
"depends on the backbone and condition" message of finding 2.

**Change in the paper (required).** Add the pooled analysis to Section 5,
add a "Multiplicity" paragraph to the statistical protocol (primary claims
tested once on pooled estimates; per-condition tests are descriptive;
Holm-corrected counts reported), and state the Holm counts above.

**Rebuttal text.**

> Finding 1 is one claim and we now test it once: the random-effects
> pooled bank term over the 15 backbone and condition pairs is −0.007 pp
> (95 % CI [−0.10, +0.08]), equivalent to zero at a 0.1 pp margin. Per-pair
> equivalence at ±0.5 pp is reported as descriptive. The routing results
> survive Holm correction in 12 of the 13 pairs that are significant
> before correction.

# 5. "The backbones disagree. What should a practitioner do?"

**Severity: Medium.** Ten results, five backbone-dependent, can read as
"no conclusion".

**The answer: the agreements are strong and the disagreements are
informative.** Give the reader one line per result:

* **Do not expect the bank to help on its own.** Pooled −0.007 pp over
  three backbones.
* **If you use dynamic convolution, the gain comes from routing.** Routing
  term pooled +1.23 pp; 12 of 13 survive correction.
* **K = 2 is enough.** Larger banks are not distinguishable from K = 2 on
  any backbone; on ResNet-18 K = 8 costs 3.9 times the parameters and 2.5
  times the training time for nothing detectable.
* **Do not anneal the temperature by default.** Never significantly
  better, lower in 7 of 9 conditions.
* **At equal parameters, widen the network instead** on backbones whose
  convolutions hold most of the parameters (ResNet-18, VGG-16: plain wins
  6 of 6). **At equal compute, routing is worth it** (13 of 15).
* **Do not use routing entropy to decide whether routing works.**

**Change in the paper (recommended).** Add a short "Practical guidance"
paragraph with these lines before the conclusion.

# 6. "The CIFAR-100 width-64 null is unexplained"

**Severity: Medium.** Reviewers dislike anomalies the authors cannot
explain, but the paper already tests and rules out two explanations,
replicates the null on VGG-16, and shows it is a positive equivalence.

**What the reviewer will ask.** Is it resolution, feature-map size, the
number of classes, or the dataset itself?

**One cheap experiment answers most of it.** Train ResNet-18 at width 64
on Tiny-ImageNet *downsampled to 32 × 32* (`static`, `uniform`, `gap`, six
seeds; about 13 GPU-hours, since it costs like CIFAR-100 width 64). This
keeps the 200 Tiny-ImageNet classes and images but uses the CIFAR input
size and final feature map.

* If the null appears, input resolution or feature-map size is the
  likely cause.
* If the effect stays (about +2 pp), the cause is the dataset or the
  number of classes, not the resolution.

Either outcome turns "unexplained" into "narrowed to X". Record the
prediction (dated) before running it, as before. VGG-16 can be added for
about 8 more GPU-hours.

**Rebuttal text if you do not run it.**

> The null is a positive equivalence on two backbones, survives a change
> of temperature schedule, does not need residual connections, and is not
> explained by training loss (a recorded prediction failed). We state the
> remaining candidates (resolution, feature-map size, classes) and leave
> them untested.

# 7. The entropy evidence is thin

**Status (22 Oct 2026): partly addressed.** VGG-16 entropies are now in the
paper (three backbones). The 16-image sample remains; option 2 below still
needs GPU time.

**Severity: Medium.** Finding 7 rests on entropy measured once, on the
last 16 test images; 30 of the ResNet-18 entropy values have no source
file in the archive; VGG-16 has no entropy at all in this draft.

**Options, best first.**

1. **Upload the result folders** (`results\` for ResNet-18,
   `results_models\vgg16\` for VGG-16) from the training PC. That closes
   the provenance gap at no GPU cost and adds VGG-16 to finding 7.
2. **Rerun two groups with full-test-set entropy.** Change the recording
   to accumulate over all test batches (a small code change), then rerun
   ResNet-18 CIFAR-100 width 16 `gap` with constant and annealed
   temperature (12 runs, about 8 GPU-hours; this is the counterexample)
   and MobileNetV2 width 64 `gap` constant and annealed (12 runs, about 17
   GPU-hours). If the 16-image values agree with full-test-set values,
   the objection is gone.
3. **Demote finding 7 to the appendix** and keep only the cross-backbone
   comparison in the main text. Do this if neither 1 nor 2 is possible.

**Rebuttal text (with option 2 done).**

> On the two groups that carry the argument we reran training with
> entropy accumulated over the full test set; the 16-image values agree
> (report the numbers), so the conclusions do not depend on the sample.

# 8. One random draw of the frozen router

**Status (27 October 2026): done for MobileNetV2 and VGG-16 at CIFAR-100 width 64** (draws 2 and 3, 24 runs, pre-registered 22 October; paper Section 7.1). MobileNetV2: both draws beat `uniform`, none distinguishable from `gap`. VGG-16: both equivalent to `uniform`, both lower than `gap`. Draws beyond three and other conditions remain one-draw.

**Severity: Medium.** The frozen router is one fixed random draw, shared
by all seeds, in one condition. "A random router is enough" (MobileNetV2)
and "only the trained router departs from the constant mixture" (VGG-16)
could both be properties of that draw.

**Experiment.** Two more draws (change `FROZEN_SEED`), six seeds each, on
MobileNetV2 CIFAR-100 width 64 (12 runs, about 16 GPU-hours) and VGG-16
CIFAR-100 width 64 (12 runs, about 10 GPU-hours). Report the three draws
side by side. This also turns finding 3 from "one draw" into "three
draws", which the paper currently has to concede.

# 9. "Your MobileNetV2 is not DY-MobileNetV2"

**Severity: Medium.** DY-Conv makes every convolution of MobileNetV2
dynamic, including the 1 × 1 expansions and projections. Ours makes only
the 17 depthwise layers dynamic (3 % of the parameters), so a reviewer can
say the MobileNetV2 results are about a different layer placement.

**The answer.** The placement rule was fixed in the pre-registration
before any MobileNetV2 run, to apply the ResNet-18 rule unchanged
("every k × k convolution with k > 1"), and its consequence was stated in
advance. The paper says so in Threats. The "all layers" placement is
implemented (`--placement all`).

**Optional experiment.** MobileNetV2 CIFAR-100 width 64 with
`--placement all`: `static` is unchanged, so `uniform` and `gap` only, six
seeds (12 runs; time unknown because all 52 layers become dynamic,
estimate 20 GPU-hours, measure with the runner's probe). If the bank term
stays at zero with 52 dynamic layers, the objection is answered.

# 10. ResNet-18 initialisation mismatch

**Severity: Low-Medium.** The paper already discloses that ResNet-18
static convolutions used PyTorch's default initialisation and the bank
used Kaiming-normal, so the ResNet-18 bank term mixes two effects.

**The answer.** The routing term is unaffected (both arms have the same
bank initialisation), and MobileNetV2 and VGG-16, which share one
initialisation, give the same bank result.

**Experiment that removes it.** Rerun the five ResNet-18 `static` groups
with Kaiming-normal (30 runs, about 22 GPU-hours). Report both versions.

# 11. "The pre-registration cannot be verified"

**Severity: Medium.** The predictions are dated documents in your own
repository. A reviewer cannot tell they were written before the runs, and
the repository's commit history starts on 17 October.

**What to do now.** ResNet-50 and ConvNeXt-Tiny are still running. Upload
their predictions (P1 to P3 and the gates, copied from
`PREREGISTRATION_MODELS.md`) to OSF Registries or AsPredicted today,
before their results exist. That gives two backbones with independently
time-stamped predictions. For the three finished backbones, say plainly
that the predictions were recorded in dated documents, not in a public
registry.

**Rebuttal text.**

> For ResNet-18, MobileNetV2 and VGG-16 the predictions were recorded in
> dated documents in the project repository; for ResNet-50 and
> ConvNeXt-Tiny they were registered publicly (OSF, link) before any of
> their runs finished.

# 12. "The ±0.5 pp margin is arbitrary"

**Severity: Low-Medium.** The paper admits the pre-registration gave no
source for 0.5 pp.

**The answer.** Report the margin sensitivity (computed for this
document): per-pair bank equivalences are 0, 9, 14 and 14 of 15 at
margins of 0.25, 0.5, 0.75 and 1.0 pp. The pooled estimate is equivalent
at 0.1 pp. No conclusion depends on the choice of 0.5 pp in the direction
of favouring the claim: a looser margin only strengthens finding 1.

# 13. "One recipe for all backbones; MobileNetV2 underfits"

**Severity: Low-Medium.**

**The answer.** The recipe is fixed on purpose so that a difference
between backbones cannot come from tuning. The plain baselines are in a
normal range for CIFAR-100: ResNet-18 78.34 %, VGG-16 74.81 %, MobileNetV2
72.40 % at width 64. The paper already says a MobileNetV2-specific recipe
could change where routing helps.

# 14. "Equal compute is counted, not measured"

**Severity: Low-Medium.** The +8 / +2 / +19 % MACs are counted for an
implementation that mixes the kernel per sample; the training code uses a
different, slower implementation.

**What to do.** Measure inference latency of the per-sample (`combine`)
path against the plain network for one width per backbone, batch 1 and
batch 128, on the RTX 4070 Ti. About one hour. Report it beside the MAC
ratio. DY-Conv reports MAdds, so counted MACs are the accepted measure,
but a measured number pre-empts the objection.

# 15. "Your annealing result contradicts DY-Conv"

**Severity: Low.**

**The answer.** DY-Conv's own Table 6: annealing improves 69.4 % to
69.9 % (+0.5 pp) in a single run, with no seed variance reported. In the
same table a constant τ = 1 is *below* the static network (64.8 % against
65.4 %), whereas our τ = 1 routed networks beat the plain network in 13 of
15 pairs. Our annealing result is six paired seeds per condition on nine
conditions. The paper already says the settings differ (backbone, dataset,
bank size, schedule length) and does not claim DY-Conv is wrong.

# 16. "The ablations were run on CIFAR-100 only"

**Status (27 October 2026): partly addressed.** Both controls now exist at CIFAR-100 width 16 (MobileNetV2, VGG-16) and on Tiny-ImageNet width 32 (VGG-16); MobileNetV2 on Tiny-ImageNet (about 74 GPU-h) and ResNet-18 are not run. Paper Section 7.1; prediction C4 failed (one of four comparisons).

**Severity: Low-Medium.** Already addressed in the paper (Threats: Scope
of the ablations, with measured cost ratios of 4 to 10 times).

**Optional experiment.** If one is affordable, the most valuable is the
routing controls (`shuffled`, `frozen_random`) on Tiny-ImageNet width 32
for ResNet-18, where routing has its largest effect (+2.2 pp): 12 runs,
about 57 GPU-hours. It makes finding 3 testable on ResNet-18, where it is
currently uninformative.

# 17. "Results were transcribed from logs; result files are missing"

**Status (22 Oct 2026): closed.** All 132 ResNet-18 and 132 VGG-16 result
files are in artifacts/. Every transcribed ResNet-18 accuracy is identical to
its file and every transcribed entropy (the 30 without a source included)
equals its file to four decimals; the build reads ResNet-18 from the files
and stops on any mismatch.

**Severity: Medium.** The archive has the result files for MobileNetV2
(132) and ResNet-18 CIFAR-100 width 64 (24) only. The other 108 ResNet-18
runs and all 132 VGG-16 runs are represented by their logs.

**What to do.** Copy the full `results\` (ResNet-18) and
`results_models\vgg16\` folders from the training PC into the archive.
The build already checks every result file against the logs and stops on
any mismatch, so this costs nothing and removes the objection entirely.

# 18. Format, length and venue

**Severity: Medium.** The draft is 30 pages with 37 tables, written as a
report. A conference reviewer will see that as unfocused.

**Venue.** Transactions on Machine Learning Research (TMLR) fits this
paper best. Its acceptance criteria are whether the claims are supported
by accurate and convincing evidence and whether some of its audience would
be interested; novelty of the method is explicitly not required, and
papers are not rejected for lacking "significant" results. A careful
controlled study with negative and mixed results is exactly what that
policy protects. Main-track computer-vision conferences are the hardest
fit because of the scale objection.

**Restructure.** Main text of about 12 to 15 pages: introduction with the
seven findings, method, one section per finding with one figure and at
most one table each, practical guidance, threats. Move per-backbone
tables, the run inventories and the provenance history to the appendix.

# 19. "Six seeds is small"

**Severity: Low.** The paper already reports a power analysis: minimum
detectable routing term 0.27 pp at n = 6; power 96 % at 0.5 pp. Six is
also the smallest n at which the Wilcoxon test can reach p < 0.05.

# 20. "CNNs only"

**Severity: Low-Medium.** Dynamic convolution is a convolutional layer,
and the backbones cover residual, plain (VGG-16), depthwise (MobileNetV2)
and, when finished, bottleneck (ResNet-50) and ConvNeXt designs. State
that dynamic attention in transformers and downstream tasks (detection,
segmentation) are out of scope.

# Action plan before submission

| Priority | Action | GPU time | Closes |
|---|---|---|---|
| 1 | Add DY-Conv Table 3 and CondConv Section 3 to Related work; cite CondConv for Proposition 1; cite the Han et al. survey | 0 | 1 |
| 2 | Add the pooled bank and routing estimates, the multiplicity paragraph and the Holm counts | 0 | 4, 12 |
| 3 | Add the plain-baseline comparison table and the practical-guidance paragraph | 0 | 3, 5 |
| 4 | Upload the ResNet-18 and VGG-16 result folders | 0 | 7, 17 |
| 5 | Register the ResNet-50 and ConvNeXt-Tiny predictions on OSF now | 0 | 11 |
| 6 | Tiny-ImageNet at 32 × 32, ResNet-18 width 64 (dated prediction first) | about 13 h | 6 |
| 7 | Two more frozen-router draws, MobileNetV2 and VGG-16 | about 26 h | 8 |
| 8 | Full-test-set entropy for the two key groups | about 25 h | 7 |
| 9 | ResNet-18 static arms with Kaiming-normal | about 22 h | 10 |
| 10 | Latency measurement of the per-sample path | about 1 h | 14 |
| Optional | `--placement all` on MobileNetV2 | about 20 h | 9 |
| Optional | Routing controls on Tiny-ImageNet width 32, ResNet-18 | about 57 h | 16 |
| Optional | ImageNet-100 at 64 pixels, ResNet-18 | about 120 h | 2 |

Items 1 to 5 need no GPU time and address the three objections most
likely to reject the paper. Items 6 to 10 together need about 87
GPU-hours, under four days on the RTX 4070 Ti, and can run after
ResNet-50 and ConvNeXt-Tiny finish. GPU times are scaled from the logged
minutes of the same groups and are estimates.

# Where the numbers in this document come from

* Holm-corrected counts, margin sensitivity and pooled estimates were
  computed on 22 October 2026 from the same per-seed final accuracies the
  paper is built on (ResNet-18 from the transcribed values, MobileNetV2
  and VGG-16 from their training logs), with paired t-tests, TOST at the
  stated margin, Holm's step-down procedure over the 15 pairs of each
  family, and a DerSimonian-Laird random-effects pool of the 15 per-pair
  mean differences.
* DY-Conv numbers: Chen et al., *Dynamic Convolution: Attention over
  Convolution Kernels*, CVPR 2020, Tables 3, 6 and 8
  (arxiv.org/abs/1912.03458).
* CondConv: Yang et al., *CondConv: Conditionally Parameterized
  Convolutions for Efficient Inference*, NeurIPS 2019, Section 3
  (arxiv.org/abs/1904.04971).
* SD-Conv numbers: He et al., *SD-Conv: Towards the Parameter-Efficiency
  of Dynamic Convolution*, WACV 2023, Table 1 (arxiv.org/abs/2204.02227).
* TMLR acceptance criteria: jmlr.org/tmlr/acceptance-criteria.html.

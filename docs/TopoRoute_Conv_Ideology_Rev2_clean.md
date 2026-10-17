# TopoRoute-Conv / KernelRoute

## Research ideology, revision 2

**Topological descriptors as a routing signal in dynamic convolution: a falsification-first study**

University of Agder, Grimstad, Norway
Revision date: September 2026 (supersedes April 2026 draft)
Status: confidential, pre-publication

---

## 0. What changed from revision 1

Revision 1 rested on a novelty claim that does not survive a current literature check, and on a complexity claim that is wrong. Both are corrected here, and the research program is reordered around what remains.

Four corrections.

The novelty claim has been overtaken. Persistent-homology descriptors were used as a gating signal in a mixture-of-experts in January 2026 (MI-MoE, molecular property prediction), three months before the April draft was written. What survives is narrower: nobody has done it for per-layer convolution-kernel selection in image CNNs.

"All dynamic convolution routes via GAP" is false. DRConv (CVPR 2021), FDConv (CVPR 2025) and KDConv (Pattern Recognition 2025) all route on spatially resolved signals. FcaNet (ICCV 2021) proved GAP is the lowest DCT coefficient and generalised past it, and CBAM and SRM already augment GAP with max and standard-deviation pooling.

"Full persistent homology is O(n³) for n pixels" does not hold for images. Cubical persistence on 2D images runs in essentially linear time in pixel count via union-find and duality. What makes approximation worthwhile is GPU batchability, kernel-launch overhead, and differentiability inside a training loop, so the argument has to be made on those grounds.

The descriptor as specified has no gradient. Three of the four LTD components are counts, so they are piecewise constant in the feature values and their derivative vanishes almost everywhere. Section 4 deals with this.

---

## 1. Executive summary

Dynamic convolution selects or blends K kernels per input. Almost every variant derives the selection weights from a heavily compressed summary of the feature map, and the field has skipped a prior question: how much does the choice of routing signal matter?

We answer it with a falsification-first protocol. Topological descriptors are one probe among several, tested against dimension-matched controls that include a frozen random projection. We report the answer whichever way it comes out.

A credible negative would say that routing-signal identity is largely irrelevant, and that gains attributed to input-dependent kernel selection come from added capacity instead of adaptivity. That result would be worth more to this literature than another operator with a 0.4pp delta.

Second contribution: an energy and latency Pareto characterisation of dynamic convolution families on Jetson-class hardware, where FLOP-based accounting is known to mispredict real cost.

---

## 2. Problem statement, restated

### 2.1 What is true about routing signals

Dynamic convolution computes W_dynamic = Σ z_k · W_k, where z = f(summary(F)). The dominant choice for `summary` is global average pooling, and that much of revision 1 was fair. The framing that GAP is an unquestioned default nobody has challenged is not available to us:

- FcaNet (ICCV 2021) proved GAP equals the lowest-frequency DCT component up to a constant, and generalised the squeeze step to multi-spectral coefficients. The "GAP discards information" argument is theirs, published and already generalised.
- CBAM and SRM add global max pooling and global standard-deviation pooling for the same reason.
- DRConv (CVPR 2021) partitions space with a learnable guided mask produced by a convolution over the input, then assigns a distinct filter per region. That is spatially resolved routing.
- FDConv (CVPR 2025) applies Kernel Spatial Modulation using local-global feature fusion, adjusting filter response at element level.
- KDConv (2025) lets each spatial position select a filter.

So the gap is narrow. Global routing in the CondConv, DY-Conv, ODConv and KernelWarehouse line is GAP-based, and no work has put a topological summary in that slot.

### 2.2 The question that is open

Whether anyone has tried descriptor X is the weak version of the question. The stronger one is that the field has never systematically ablated the routing signal itself.

DCD (ICLR 2021) found that jointly optimising dynamic attention with static kernels is the binding difficulty, and that attention values in the high-dimensional latent space become small enough to suppress learning of the static kernels. That result implies something nobody has chased down. If attention collapses toward near-uniform, the routing input barely matters, and every paper crediting its gains to input-adaptive kernel selection may be measuring parameter capacity.

One control settles it: a router whose weights are frozen at random initialisation, or fed a shuffled input. If that matches a trained GAP router, the premise of the subfield needs revisiting.

The control is cheap. It is the first experiment in this program.

---

## 3. Prior art, corrected

| Work | What it does | Relation to us |
|---|---|---|
| **MI-MoE (Jan 2026)** | Topology-aware gating network routing to experts using filtration-based descriptors including PH features, on 3D molecular graphs | **Closest prior art. Establishes the principle we claimed.** Our difference is modality (image CNN), granularity (per-layer conv kernel, not expert subnetwork), and constraint (inference-time budget on edge hardware) |
| GraphMoRE (2025), GeoMoE (2026) | Topology- and curvature-aware gating for node-wise expert routing on graphs | Same principle, graph domain. GeoMoE reports expert collapse under topology-agnostic gating, relevant to our diversity-loss design |
| FcaNet (ICCV 2021) | Proves GAP is lowest DCT coefficient; multi-spectral squeeze | **Owns the "GAP loses information" argument.** Must be a baseline, not a citation |
| CBAM, SRM | GAP + max pooling, GAP + std pooling for channel attention | Cheap richer-statistic baselines. If they match LTD, we have nothing |
| DRConv (CVPR 2021) | Spatial region assignment via learnable guided mask | Falsifies "all routing is GAP" |
| FDConv (CVPR 2025) | Fourier-disjoint weights plus kernel spatial modulation | Current strong baseline, non-GAP modulation |
| DCD (ICLR 2021) | Matrix-decomposition view; attention over channel groups is the optimisation bottleneck | **Theoretical reason to expect our null result.** Must be engaged |
| TopologyLayer (AISTATS 2020), PLLay (NeurIPS 2020) | Differentiable PH layers, level-set filtrations | Differentiable in-network PH is solved prior art. We are not first to it |
| DECT (ICLR 2024) | Differentiable Euler Characteristic Transform via sigmoid-relaxed indicators, as a shape-descriptor layer | **The route to making our descriptor differentiable.** Adopt rather than reinvent |
| Persistent Homology Convolutions (2025/26) | Local PH under a sliding window as a convolution-like operator | Nearest CNN-side neighbour. Uses PH to produce features, not to route |
| PHCA (2025) | PH features fused with channel attention for image classification | PH plus attention in image classification already exists as fusion |
| PHG-Net (WACV 2024) | PH features concatenated at final layer | Fusion rather than routing (revision 1 had this right) |
| DSConv (ICCV 2023), Clough et al. (IEEE TMI 2020) | PH-based topological loss for segmentation | Loss rather than routing (correct in revision 1) |
| FMDConv (Pattern Recognition 2026), CoE (2021) | Speed-accuracy framing; CoE reports CondConv-EfficientNet-B0 1.7× slower at parity FLOPs on CPU | **Weakens "first-ever real-hardware benchmark."** The FLOPs-mispredict-latency point is published |

Revised novelty statement: applying topological descriptors as the routing signal for convolution-kernel selection in image CNNs appears unpublished. That is a gap of the "nobody has combined these two things" kind, and gaps of that kind do not carry an ECCV or NeurIPS paper alone. They carry one when paired with a strong result or a strong methodological contribution. We are pursuing the second.

---

## 4. Method, with the flaws fixed

### 4.1 Differentiability

Revision 1's descriptor was a count of local maxima (LEC), a ratio of saddles to critical points (SPR), the alternating sum maxima − saddles + minima (ECA), and an entropy over pseudo-persistence values (PE).

The first three are counts and ratios of counts. Perturb a pixel infinitesimally and nothing moves until a critical point flips, at which point the value jumps. The gradient is zero almost everywhere and undefined at the jumps. Only PE carries a usable gradient through the height differences, and even that goes discontinuous when the critical-point set changes membership.

Ship it as written and the router MLP trains on the descriptor while no gradient reaches the backbone. The backbone never learns to produce routable topology, and any measured effect is the router fitting a fixed arbitrary statistic.

The repair is to replace hard indicators with temperature-relaxed ones, as DECT does for the Euler Characteristic Transform and as the soft Euler characteristic construction does for pixel grids. A soft local-max indicator σ((F_ij − max_neighbours) / τ) replaces the boolean test; the V/E/F formula on continuous pixel probabilities replaces the hard Euler count; τ is annealed on a schedule and reported as an ablation. Straight-through estimation is the fallback if the relaxation smooths away the discriminative signal.

Gate before any accuracy experiment: finite-difference check that ∂LTD/∂F is non-zero on at least 90% of sampled feature maps.

### 4.2 Post-ReLU plateaus

After ReLU a large fraction of feature-map entries are exactly zero. Strict 3×3 local-maximum tests on flat zero regions are degenerate, ties break on implementation detail, and the resulting maxima count tracks activation sparsity instead of topology.

So correlate LEC against the L0 norm of the feature map across a validation batch before proceeding. If |r| > 0.9, the descriptor is a sparsity meter with a topological name and the hypothesis is dead. Report the number either way.

### 4.3 Dimension mismatch

LTD sits in R²⁰. GAP sits in R^C with C between 64 and 512. In revision 1's hybrid concat, twenty topological dimensions ride alongside hundreds of GAP dimensions, so a hybrid-beats-LTD-only result cannot be attributed to topology rather than to extra channel-mean capacity. It is uninterpretable by construction.

Every comparison therefore runs dimension-matched routers:

| Router | Dim | Purpose |
|---|---|---|
| GAP | C | Standard baseline |
| GAP + std + max | 3C | SRM / CBAM-style richer statistics |
| DCT top-k (FcaNet) | k·C | Frequency-domain compression |
| **Random frozen projection** | 20 | **Does the signal matter at all?** |
| Shuffled-input GAP | C | Does input-dependence matter at all? |
| LTD (relaxed) | 20 | Our probe |
| GAP projected to 20 dims | 20 | Dimension-matched GAP |

If LTD cannot beat a frozen random projection at 20 dims, there is no topological signal. If GAP cannot beat it either, we have a different and more interesting result.

### 4.4 Resolution and placement

At the depths where dynamic convolution normally goes in a ResNet on CIFAR-100 or Tiny-ImageNet, feature maps are 8×8 or 4×4. β₀ and β₁ on a 4×4 grid have almost no dynamic range and the descriptor degenerates. This lines up with our own result that global routing beat patch routing by 1 to 2pp (Wilcoxon p = 0.0039): whatever the router uses, it operates above the scale of fine spatial structure.

Revised placement hypothesis: if topological routing helps anywhere, it helps at the stem and early stages, 32×32 and above. Treat placement as an experimental factor rather than replacing every Conv2D uniformly.

### 4.5 Diversity loss

Revision 1's L_topo_div = −Σ_{i≠j} ‖E[t | z_i > z_j] − E[t | z_j > z_i]‖ is unbounded below. Maximising an unnormalised norm difference will diverge or be dominated by descriptor scale. Normalise t per-dimension over the batch and switch to a margin-hinged or cosine-dissimilarity form with an explicit ceiling. Report λ sensitivity including λ = 0.

### 4.6 Latency

Revision 1 estimated under 0.1 ms per layer on Orin Nano from a FLOP count. Critical-point detection is memory-bandwidth-bound, and the top-k-across-channels step in the revision 1 aggregation is the expensive part, which does not vectorise the way the O(HWC) argument implies.

Measure every latency number on-device with tegrastats at fixed clocks before it appears in a claim. Consider dropping top-k in favour of mean and std aggregation.

---

## 5. Experimental plan, falsification-ordered

Order matters here. Each stage is a gate: if it fails we stop and write up the negative result.

**Stage 0, gradient sanity (1 day).** Finite-difference verification that the relaxed descriptor passes gradient. If it fails, fix the relaxation before anything else runs.

**Stage 1, the null control (3 days, CIFAR-100, ResNet-18, 3 seeds).** Frozen-random-projection router and shuffled-input router against a trained GAP router. If GAP does not significantly beat frozen-random, then the routing signal carries no usable information and we write that up.

**Stage 2, redundancy check (2 days).** LEC against L0 correlation; LTD against GAP + std + max mutual information. If LTD is predictable from second-order statistics with R² > 0.9, topology adds nothing and we say so.

**Stage 3, dimension-matched comparison (CIFAR-100 and Tiny-ImageNet, 3 seeds).** Full control set from 4.3. Wilcoxon signed-rank across seeds, effect sizes with confidence intervals, no cherry-picked best runs.

**Stage 4, placement ablation.** Stem-only, early-only, all-layers.

**Stage 5, edge Pareto (Jetson Orin Nano).** Measured latency at batch 1 and 16, measured power via tegrastats at fixed clocks, energy per image. Families compared: static, DY-Conv, CondConv, ODConv, FDConv, KernelWarehouse, LTD-routed. Framed as an energy Pareto across dynamic convolution families.

**Stage 6, robustness.** CIFAR-100-C. This is the one place a topological prior has a theoretical reason to help, since persistence is stable under bounded perturbation. If topology wins anywhere it wins here, and this may be the strongest positive result available even when clean-accuracy gains come back null.

### Evidence already collected

The 24-run Tiny-ImageNet benchmark on H100 across 3 seeds, global against patch routing, gave global a 1 to 2pp margin at Wilcoxon p = 0.0039.

Seam probes at 32×32 and 64×64 with matched controls came back negative on both. Report them. They constrain the hypothesis space, and they are the kind of result the field does not publish and should.

The entropy-of-mean routing diagnostic turned out to be foolable and was replaced with per-patch entropy plus a synthetic self-test. This transfers beyond our own work, so it belongs in the body of the paper rather than an appendix: every dynamic-convolution paper reporting routing entropy may be reporting a foolable number.

Swin-Tiny collapsed under the shared recipe and needs its own. Either fix it or exclude it with the reason stated, since a collapsed run gives us nothing to compare against.

---

## 6. Paper narrative

**Title:** *Does the routing signal matter? A falsifiable diagnostic for dynamic convolution, with topological probes*

**Pitch:** Dynamic convolution is credited with input-adaptive kernel selection, but the routing signal has never been systematically ablated and the standard routing-entropy diagnostic is foolable. We introduce a falsifiable diagnostic protocol, apply it across seven routing signals including a topological descriptor and a frozen random control, and report what survives.

Why this beats revision 1's narrative: revision 1 committed to a result before running the experiment ("achieving X% higher accuracy"). Its risk table put "LTD does not improve accuracy over GAP" at Medium probability, with a pivot to an empirical benchmarking contribution as the mitigation. Given DCD's finding on attention collapse and our own negative seam probes, that outcome is the modal one, not a medium-probability risk. When the mitigation is the likely outcome, it should be the plan.

---

## 7. Venue realignment

ECCV or NeurIPS on a combination gap with a likely-null result is not realistic. Better fits, in order:

- TMLR: accepts rigorous work regardless of outcome direction, which makes it the best home for a diagnostic paper with a negative result.
- BMVC or WACV 2027: practical CV contributions, higher acceptance, receptive to careful empirical work.
- TAG-ML and NeurIPS workshops: the right audience for the topological-probe angle specifically.
- IEEE TNNLS: viable if a positive operator result materialises and the edge story holds up.
- IROS or ICRA: only if stage 5 produces a deployment-relevant energy result.

---

## 8. Bibliographic corrections

CondConv is Yang, Bender, Le and Ngiam, NeurIPS 2019. Revision 1 attributes it to "Chen et al., 2020," conflating it with DY-Conv (Chen et al., "Dynamic Convolution: Attention over Convolution Kernels," CVPR 2020). Reference [1] is DY-Conv but the text uses it as CondConv. Two separate papers.

Clough et al. appears as 2019 in the table and 2020 in the references. It is IEEE TMI 2020, arXiv 2019.

KernelWarehouse's "108 kernel cells" needs verification against the paper. The value of n is a configuration choice.

FcaNet is in the reference list but absent from the novelty table, despite owning the core motivating argument. Add it to both.

Add: MI-MoE, DRConv, FDConv, DCD, DECT, PLLay, TopologyLayer, PHCA, Persistent Homology Convolutions, FMDConv, CoE.

---

## 9. Risk analysis, recalibrated

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| LTD does not beat GAP | **High** (was: Medium) | **Low**, given the reframe | The null is the result. Stages 1 and 2 are built to establish it rigorously |
| Descriptor has no usable gradient | **High if unfixed** | Fatal | Stage 0 gate. Relaxed indicators per DECT |
| LEC is a sparsity proxy | Medium | High | Stage 2 gate, correlation test, reported either way |
| Reviewer finds MI-MoE | **Certain** | High if unaddressed, low if pre-empted | Cite prominently, state the delta precisely, claim no principle-level novelty |
| Reviewer challenges the O(n³) claim | **Certain** if retained | High | Removed. Justification restated as batchability and differentiability |
| Attention collapses regardless of signal | Medium-High | Low | A finding rather than a failure. It is stage 1's hypothesis |
| Edge gains do not materialise | Medium | Medium | Report the Pareto as measured; a negative energy result still characterises the family |

---

## 10. Conclusion

The April framing asked whether a new routing signal could win. The literature and our own negative results both point at a more useful question: whether the routing signal matters at all, and how anyone would know, given that the standard diagnostic can be fooled.

That question is answerable on the compute already available, it yields a publishable result in either direction, and the diagnostic protocol transfers to every paper in the dynamic-convolution literature. The topological descriptor stays in the program as the sharpest probe we have for structural, non-GAP information, and as a plausible win on corruption robustness. It is no longer the claim the paper rests on.

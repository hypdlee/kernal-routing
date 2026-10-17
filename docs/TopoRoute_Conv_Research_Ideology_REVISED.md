# TopoRoute-Conv / KernelRoute

## Research Ideology, Revision 2

**Topological Descriptors as a Routing Signal in Dynamic Convolution: A Falsification-First Study**

University of Agder, Grimstad, Norway
Revision date: September 2026 (supersedes April 2026 draft)
Status: CONFIDENTIAL, pre-publication

---

## 0. What changed from Revision 1, and why

Revision 1 was built on a novelty claim that does not survive a current literature check, and on a complexity claim that is factually wrong. Both are corrected below. The research program is reframed accordingly: from "we propose a new operator that will win" to "we determine whether the routing signal in dynamic convolution carries usable input-dependent information at all, using topology as the sharpest available probe."

Four specific corrections:

1. **The novelty claim is now falsified at the level of principle.** Persistent-homology descriptors have been used as a gating/routing signal in a mixture-of-experts, published January 2026 (MI-MoE, molecular property prediction). The gap that remains is the narrower one: nobody has done it for per-layer convolution-kernel selection in image CNNs.
2. **"All dynamic convolution routes via GAP" is false.** DRConv (CVPR 2021), FDConv (CVPR 2025) and KDConv (Pattern Recognition 2025) all route on spatially resolved signals. FcaNet (ICCV 2021) already proved GAP is the lowest DCT coefficient and generalised past it, and CBAM and SRM already augment GAP with max and standard-deviation pooling.
3. **"Full persistent homology is O(n³) for n pixels" is wrong for images.** Cubical persistence on 2D images is computed in essentially linear time in pixel count via union-find and duality. The justification for approximating PH is not asymptotic cost. It is GPU batchability, kernel-launch overhead, and differentiability inside a training loop.
4. **The descriptor as specified has no gradient.** Three of the four LTD components are counts, hence piecewise constant in the feature values, hence zero-gradient almost everywhere. This is fatal as written and is addressed in Section 4.

---

## 1. Executive summary

Dynamic convolution selects or blends K kernels per input. Almost all variants derive the selection weights from a heavily compressed summary of the feature map. This work asks a prior question that the literature has largely skipped: **how much does the choice of routing signal actually matter?**

We answer it with a falsification-first protocol in which topological descriptors of the feature map are one probe among several, tested against dimension-matched controls including a frozen random projection. We report the answer whichever way it comes out. A credible negative (routing signal identity is largely irrelevant; the gains attributed to "input-dependent kernel selection" come from added capacity rather than from adaptivity) is a more useful contribution to this literature than another operator with a 0.4pp delta.

Secondary contribution: an energy-and-latency Pareto characterisation of dynamic convolution families on Jetson-class hardware, where the accepted FLOP-based accounting is known to mispredict real cost.

---

## 2. Problem statement, restated honestly

### 2.1 What is actually true about routing signals

Dynamic convolution computes W_dynamic = Σ z_k · W_k, where z = f(summary(F)). The literature's dominant choice for `summary` is global average pooling. That much is fair. But the framing "GAP is an unquestioned default nobody has challenged" is not available to us:

- **FcaNet (ICCV 2021)** proved GAP equals the lowest-frequency DCT component up to a constant and generalised the squeeze step to multi-spectral coefficients. The "GAP discards information" argument is theirs, already made, already published, and already generalised.
- **CBAM** and **SRM** add global max pooling and global standard-deviation pooling for exactly this reason.
- **DRConv (CVPR 2021)** partitions space via a learnable guided mask produced by a standard convolution over the input, and assigns a distinct filter per region. That is spatially resolved routing, not GAP.
- **FDConv (CVPR 2025)** applies Kernel Spatial Modulation using local-global feature fusion, adjusting filter response at element level.
- **KDConv (2025)** lets each spatial position select a filter.

So the honest statement of the gap is narrow: *global* routing in the CondConv / DY-Conv / ODConv / KernelWarehouse line is GAP-based, and no work has used a topological summary in that slot.

### 2.2 The question that is actually open

The interesting gap is not "has anyone tried descriptor X." It is that **the field has never systematically ablated the routing signal itself.** DCD (ICLR 2021) found that the joint optimisation of dynamic attention with static kernels is the binding difficulty, and that attention values in the high-dimensional latent space become small enough to suppress learning of the static kernels. That result implies something the community has not chased down: if attention collapses toward near-uniform, then the routing input barely matters, and every paper attributing gains to "input-adaptive kernel selection" may be measuring added parameter capacity instead.

Nobody has run the control that settles this: **a router whose weights are frozen at random initialisation, or fed a shuffled input.** If that matches a trained GAP router, the premise of the entire subfield needs revisiting.

That control is cheap, and it is the first experiment in this program.

---

## 3. Prior art: corrected table

| Work | What it does | Relation to us |
|---|---|---|
| **MI-MoE (Jan 2026)** | Topology-aware gating network routing to experts using filtration-based descriptors including PH features, on 3D molecular graphs | **Closest prior art. Establishes the principle we claimed.** Our difference is modality (image CNN), granularity (per-layer conv kernel, not expert subnetwork), and constraint (inference-time budget on edge hardware) |
| GraphMoRE (2025), GeoMoE (2026) | Topology- and curvature-aware gating for node-wise expert routing on graphs | Same principle, graph domain. GeoMoE additionally reports expert collapse under topology-agnostic gating, which is directly relevant to our diversity-loss design |
| FcaNet (ICCV 2021) | Proves GAP is lowest DCT coefficient; multi-spectral squeeze | **Owns the "GAP loses information" argument.** Must be a baseline, not a citation |
| CBAM, SRM | GAP + max pooling, GAP + std pooling for channel attention | Cheap richer-statistic baselines. If they match LTD, we have nothing |
| DRConv (CVPR 2021) | Spatial region assignment via learnable guided mask | Falsifies "all routing is GAP" |
| FDConv (CVPR 2025) | Fourier-disjoint weights + kernel spatial modulation | Current strong baseline, non-GAP modulation |
| DCD (ICLR 2021) | Matrix-decomposition view; attention over channel groups is the optimisation bottleneck | **Theoretical reason to expect our null result.** Must be engaged, not ignored |
| TopologyLayer (AISTATS 2020), PLLay (NeurIPS 2020) | Differentiable PH layers, level-set filtrations | Differentiable PH in-network is solved prior art. We are not first to that |
| DECT (ICLR 2024) | Differentiable Euler Characteristic Transform via sigmoid-relaxed indicators, as a shape-descriptor layer | **The right way to make our descriptor differentiable.** Adopt, do not reinvent |
| Persistent Homology Convolutions (2025/26) | Local PH computed under a sliding window as a convolution-like operator | Nearest CNN-side neighbour. Uses PH to *produce* features, not to *route* |
| PHCA (2025) | PH features fused with channel attention for image classification | PH + attention in image classification already exists as fusion |
| PHG-Net (WACV 2024) | PH features concatenated at final layer | Fusion, not routing (as Rev 1 stated, this part was correct) |
| DSConv (ICCV 2023), Clough et al. (IEEE TMI 2020) | PH-based topological loss for segmentation | Loss, not routing (correct in Rev 1) |
| FMDConv (Pattern Recognition 2026), CoE (2021) | Speed-accuracy trade-off framing; CoE reports CondConv-EfficientNet-B0 1.7× slower at parity FLOPs on CPU | **Weakens "first-ever real-hardware benchmark."** The FLOPs-mispredict-latency point is published |

**Revised novelty statement.** Applying topological descriptors as the routing signal for convolution-kernel selection in image CNNs appears unpublished. This is a combination-gap, not a principle-gap. Combination-gaps do not carry an ECCV or NeurIPS paper on their own. They carry a paper when paired with either a strong result or a strong methodological contribution. We are pursuing the second.

---

## 4. Method, with the fatal flaws fixed

### 4.1 The differentiability problem

Rev 1's descriptor was:

- LEC: count of local maxima
- SPR: ratio of saddles to critical points
- ECA: maxima − saddles + minima
- PE: entropy over pseudo-persistence values

LEC, SPR and ECA are counts and ratios of counts. They are piecewise constant in the feature-map values: perturb a pixel infinitesimally and the count does not move until a critical point flips, at which moment it jumps. **The gradient is zero almost everywhere and undefined at the jumps.** Only PE carries a usable gradient, through the height differences, and even that is discontinuous when the critical-point set changes membership.

Consequence if shipped as written: the router MLP trains on the descriptor, but no gradient flows back through the descriptor to the backbone. The backbone never learns to produce routable topology. Any measured effect would be the router fitting a fixed, arbitrary statistic.

**Fix.** Replace hard indicators with temperature-relaxed ones, as DECT does for the Euler Characteristic Transform and as the soft Euler characteristic construction does for pixel grids:

- soft local-max indicator: σ((F_ij − max_neighbours) / τ) rather than a boolean test
- soft Euler characteristic via the V/E/F formula on continuous pixel probabilities
- τ annealed on a schedule, reported as an ablation

Straight-through estimation is the fallback if the relaxation proves too smooth to be discriminative. Either way, **gradient flow through the descriptor must be verified numerically before any accuracy experiment is run.** Gate: finite-difference check that ∂LTD/∂F is non-zero for at least 90% of sampled feature maps.

### 4.2 The post-ReLU plateau problem

After ReLU, a large fraction of feature-map entries are exactly zero. Strict 3×3 local-maximum tests on flat zero regions are degenerate: ties are broken arbitrarily by implementation detail, and the resulting "maxima count" tracks activation sparsity rather than topology.

**Gate before proceeding:** correlate LEC against the L0 norm of the feature map across a validation batch. If |r| > 0.9, the descriptor is a sparsity meter with a topological name, and the whole hypothesis is dead. Report this either way.

### 4.3 The dimension-mismatch problem

LTD ∈ R²⁰. GAP ∈ R^C with C from 64 to 512. In the Rev 1 hybrid concat, twenty topological dimensions sit alongside hundreds of GAP dimensions. **Any hybrid-beats-LTD-only result would be uninterpretable**, because it cannot be attributed to topology rather than to the extra channel-mean capacity.

**Fix.** Every comparison uses dimension-matched routers. The control set is:

| Router | Dim | Purpose |
|---|---|---|
| GAP | C | Standard baseline |
| GAP + std + max | 3C | SRM / CBAM-style richer statistics |
| DCT top-k (FcaNet) | k·C | Frequency-domain compression |
| **Random frozen projection** | 20 | **Does the signal matter at all?** |
| Shuffled-input GAP | C | Does input-dependence matter at all? |
| LTD (relaxed) | 20 | Our probe |
| GAP projected to 20 dims | 20 | Dimension-matched GAP |

If LTD does not beat "random frozen projection at 20 dims," there is no topological signal. If *GAP* does not beat it either, that is the paper.

### 4.4 Resolution and placement

At the depths where dynamic convolution is normally inserted in a ResNet on CIFAR-100 or Tiny-ImageNet, feature maps are 8×8 or 4×4. β₀ and β₁ on a 4×4 grid have almost no dynamic range, and the descriptor degenerates. This is consistent with our own finding that global routing outperformed patch routing by 1 to 2pp (Wilcoxon p = 0.0039): whatever the router uses, it is not fine spatial structure.

**Revised placement hypothesis:** if topological routing helps anywhere, it helps at the stem and early stages (32×32 and above), not deep. Test placement explicitly as a factor rather than replacing all Conv2D uniformly.

### 4.5 Diversity loss, corrected

Rev 1's L_topo_div = −Σ_{i≠j} ‖E[t | z_i > z_j] − E[t | z_j > z_i]‖ is unbounded below. Maximising an unnormalised norm difference will diverge or be dominated by descriptor scale. Replace with a bounded surrogate: normalise t per-dimension over the batch, and use a margin-hinged or cosine-dissimilarity form with an explicit ceiling. Report λ sensitivity including λ = 0.

### 4.6 Latency: measured, not estimated

Rev 1 estimated "<0.1 ms per layer on Orin Nano" from a FLOP count. Critical-point detection is memory-bandwidth-bound, not FLOP-bound, and the top-k-across-channels step in the Rev 1 aggregation is the expensive part and does not vectorise the way the O(HWC) argument implies. **All latency numbers must be measured on-device with tegrastats, at fixed clocks, before they appear in any claim.** Consider dropping top-k in favour of mean and std aggregation only.

---

## 5. Experimental plan, falsification-ordered

The order matters. Each stage is a gate: if it fails, we stop and write up the negative result rather than proceeding.

**Stage 0, gradient sanity (1 day).** Finite-difference verification that the relaxed descriptor passes gradient. Fail → fix relaxation before anything else.

**Stage 1, the null control (3 days, CIFAR-100, ResNet-18, 3 seeds).** Frozen-random-projection router and shuffled-input router versus trained GAP router. If GAP does not significantly beat frozen-random, **the finding is that dynamic convolution's gains are capacity, not adaptivity**, and that is the paper. Write it up.

**Stage 2, redundancy check (2 days).** LEC versus L0 correlation; LTD versus GAP+std+max mutual information. If LTD is predictable from second-order statistics with R² > 0.9, topology adds nothing new and we say so.

**Stage 3, dimension-matched comparison (CIFAR-100 and Tiny-ImageNet, 3 seeds).** Full control set from 4.3. Wilcoxon signed-rank across seeds, effect sizes reported with confidence intervals, no cherry-picked best runs.

**Stage 4, placement ablation.** Stem-only, early-only, all-layers.

**Stage 5, edge Pareto (Jetson Orin Nano).** Measured latency at batch 1 and 16, measured power via tegrastats at fixed clocks, energy per image. Families compared: static, DY-Conv, CondConv, ODConv, FDConv, KernelWarehouse, LTD-routed. Framed as "energy Pareto across dynamic convolution families," not "first-ever benchmark."

**Stage 6, robustness.** CIFAR-100-C. This is the one place a topological prior has a genuine theoretical reason to help, since persistence is stable under bounded perturbation. If topology wins anywhere, it wins here, and this may be the strongest positive result available even if clean-accuracy gains are null.

### Already-collected evidence to fold in

- 24-run Tiny-ImageNet benchmark, H100, 3 seeds, global versus patch routing. Global wins by 1 to 2pp, Wilcoxon p = 0.0039.
- Seam probes at 32×32 and 64×64 with matched controls: both negative. **Report these.** They constrain the hypothesis space and they are the kind of result the field does not publish and should.
- The entropy-of-mean routing diagnostic was proven foolable and replaced with per-patch entropy plus a synthetic self-test. **This is the most transferable contribution in the whole program** and should be foregrounded, not buried in an appendix. Every dynamic-convolution paper that reports routing entropy is potentially reporting a foolable metric.
- Swin-Tiny collapsed under the shared recipe and needs a separate one. Either fix it or exclude it explicitly with the reason stated. Do not report a collapsed baseline as a comparison.

---

## 6. Paper narrative, revised

**Title:** *Does the Routing Signal Matter? A Falsifiable Diagnostic for Dynamic Convolution, with Topological Probes*

**Pitch:** Dynamic convolution is credited with input-adaptive kernel selection, but the routing signal has never been systematically ablated and the standard routing-entropy diagnostic is foolable. We introduce a falsifiable diagnostic protocol, apply it across seven routing signals including a topological descriptor and a frozen random control, and report what survives.

**Why this is stronger than Rev 1's narrative.** Rev 1 committed to a result before running the experiment ("achieving X% higher accuracy"). Its risk table listed "LTD does not improve accuracy over GAP" as Medium probability, with a pivot to "empirical benchmarking contribution" as mitigation. Given DCD's finding on attention collapse and our own negative seam probes, that outcome is not medium probability, it is the modal outcome. A design whose backup plan is the likely outcome should make the backup plan the primary plan. This revision does that.

---

## 7. Venue realignment

ECCV or NeurIPS on a combination-gap with a likely-null result is not realistic. Better fits:

- **TMLR.** Explicitly accepts rigorous work regardless of outcome direction. Best home for a diagnostic-plus-negative-result paper.
- **BMVC / WACV 2027.** Practical CV contributions, higher acceptance, receptive to careful empirical work.
- **TAG-ML workshop / NeurIPS workshops.** Right audience for the topological-probe angle specifically.
- **IEEE TNNLS.** Still viable if a positive operator result materialises and the edge story is strong.
- **IROS / ICRA.** Only if Stage 5 produces a strong, deployment-relevant energy result.

---

## 8. Bibliographic corrections needed in Rev 1

- **CondConv is Yang, Bender, Le and Ngiam, NeurIPS 2019**, not "Chen et al., 2020." Rev 1 conflated it with DY-Conv (Chen et al., "Dynamic Convolution: Attention over Convolution Kernels," CVPR 2020). Reference [1] is DY-Conv but is used in the text as CondConv. Two distinct papers.
- Clough et al. appears as 2019 in the table and 2020 in the references. It is IEEE TMI, 2020 (arXiv 2019).
- KernelWarehouse's "108 kernel cells" should be verified against the paper rather than asserted; n is a configuration choice, not a fixed property.
- FcaNet appears in the reference list but is absent from the novelty table, despite owning the core motivating argument. Add it to both.
- Add: MI-MoE, DRConv, FDConv, DCD, DECT, PLLay, TopologyLayer, PHCA, Persistent Homology Convolutions, FMDConv, CoE.

---

## 9. Risk analysis, recalibrated

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| LTD does not beat GAP | **High** (was: Medium) | **Low**, given the reframe | The null is the paper. Stage 1 and 2 are designed to establish it rigorously |
| Descriptor has no usable gradient | **High if unfixed** | Fatal | Stage 0 gate. Relaxed indicators per DECT |
| LEC is a sparsity proxy | Medium | High | Stage 2 gate, correlation test, reported either way |
| Reviewer finds MI-MoE | **Certain** | High if unaddressed, low if pre-empted | Cite prominently, state the delta precisely, do not claim principle-novelty |
| Reviewer challenges the O(n³) claim | **Certain** if retained | High | Removed. Justification restated as batchability and differentiability |
| Attention collapses regardless of signal | Medium-High | Low | This is a finding, not a failure. It is Stage 1's hypothesis |
| Edge gains do not materialise | Medium | Medium | Report the Pareto honestly; a negative energy result is still a useful characterisation |

---

## 10. Conclusion

The April framing asked whether a new routing signal could win. The literature and our own negative results both suggest the more valuable question is whether the routing signal matters at all, and how anyone would know, given that the standard diagnostic is foolable.

That question is answerable with the compute already available, it produces a publishable result in either direction, and the diagnostic protocol transfers to every paper in the dynamic-convolution literature. The topological descriptor remains in the program as the sharpest available probe of structural, non-GAP information, and as a possible win on corruption robustness, but it is no longer the load-bearing claim.

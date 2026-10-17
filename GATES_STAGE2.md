# Stage 2 gates

Two independent lines. 2a is cheap and hardens the weakest existing
claim. 2b is expensive and answers the biggest external objection.

---

## Stage 2a: K sweep (12 runs, ~8 h)

The width-32 peak was measured at K=4 only. The bank size interacts with
capacity by construction, so the peak location is currently a single
observation.

- [ ] K1 test suite passes
  CHECK: `python tests/test_paths.py`
  EXPECT: 9 assertions, "all tests passed"
  EVIDENCE: pending

- [ ] K2 K recorded correctly in every result
  CHECK: read config.num_kernels in any stage2k JSON
  EXPECT: 2 or 8, matching the tag
  EVIDENCE: pending

- [ ] K3 parameter counts scale with K
  EXPECT at width 32: K=2 about 5.6M, K=4 11.13M (already on disk),
  K=8 about 22.2M. If K=2 and K=8 report the same count, --num-kernels
  did not take effect.
  EVIDENCE: pending

- [ ] K4 both K values complete, 6 seeds each
  CHECK: `python check_grid.py --dir results --tag stage2k_K2 --routers gap --seeds 0 1 2 3 4 5`
  then the same for stage2k_K8
  EVIDENCE: pending

- [ ] K5 trend read from per-K rows
  CHECK: `python analyze_ksweep.py`
  EXPECT: three K values against the same static reference. The script
  states whether the effect is monotone in K and refuses trend language
  when it is not.
  EVIDENCE: pending

---

## Stage 2b: Tiny-ImageNet (12 runs, 3 to 4 days)

- [ ] T1 dataset prepared and verified
  CHECK: `python download_tinyimagenet.py --root ./data`
  EXPECT: 200 train classes / 100000 images, 200 val classes / 10000
  images, "READY". The script refuses to report ready on any mismatch.
  EVIDENCE: pending

- [ ] T2 loader reads it and the label spaces agree
  CHECK: first training run prints "[tinyimagenet ...]" and an epoch-0
  loss near 5.3 (= ln 200)
  EXPECT: a loss near 4.6 would mean 100 classes and the wrong head
  EVIDENCE: pending

- [ ] T3 epoch count fixed BEFORE the first run
  DECISION REQUIRED: 150 matches CIFAR exactly; 100 saves about a third
  of three days. Runs at different counts cannot be paired, so changing
  this halfway discards what is finished.
  EVIDENCE: pending

- [ ] T4 both arms complete, 6 seeds each
  CHECK: `python check_grid.py --dir results --tag tin_w32 --routers static gap --seeds 0 1 2 3 4 5`
  EVIDENCE: pending

- [ ] T5 replication verdict stated either way
  EXPECT: gap - static at width 32 on Tiny-ImageNet, with CI and p,
  set beside the CIFAR-100 value of +1.153pp. A failure to replicate is
  a result and goes in the paper.
  EVIDENCE: pending

---

## Known scope decisions, taken deliberately

- uniform is NOT repeated on Tiny-ImageNet. It was equivalent to static
  at all three CIFAR widths; repeating it costs six runs and answers
  nothing. Pass --arms static uniform gap to reverse this.
- The stem is NOT changed for 64x64 inputs. Keeping 3x3 stride 1 leaves
  the final map at 8x8 instead of 4x4 and costs roughly 4x in the deeper
  stages, but changing it would make the two datasets non-comparable.
  This is the main reason Stage 2b is expensive.
- Tiny-ImageNet is run at width 32 only, where the CIFAR effect peaked.

ABANDON lines go here if a gate becomes impossible, with the reason.

# Stage 2 gates

Two independent lines. 2a is cheap and hardens the weakest existing
claim. 2b is expensive and answers the biggest external objection.

---

## Stage 2a: K sweep COMPLETE (evidence recorded 2026-09-14)

- [x] K1 test suite passes
  EVIDENCE: 9 assertions, "all tests passed".

- [x] K2 K recorded correctly
  EVIDENCE: run headers read "K2" and "K8"; config.num_kernels matches
  the tag in every record.

- [x] K3 parameter counts scale with K
  EVIDENCE: K=2 5.64M, K=4 11.13M, K=8 22.12M at width 32. Router
  parameters stay near constant (73.6K / 74.6K / 76.5K), as expected:
  only the output layer of the router widens with K.

- [x] K4 both K values complete, 6 seeds each
  EVIDENCE: 12 runs, 584.3 min total.

- [x] K5 trend read from pairwise tests
  EVIDENCE: vs static, K=2 +1.078pp (p=0.00034), K=4 +1.153pp
  (p=0.00003), K=8 +1.177pp (p=0.00087). PAIRWISE: K4-K2 +0.075pp
  (p=0.288, TOST=0.0005), K8-K4 +0.023pp (p=0.854, TOST=0.0053),
  K8-K2 +0.098pp (p=0.409, TOST=0.0072). All EQUIVALENT.
  (Required a fix to analyze_ksweep.py: it reported "monotone in K" from
  the ordering of the point estimates alone, which is an over-read when
  no pair differs. It now tests adjacent pairs and refuses trend
  language when they are equivalent.)

Ledger: stage2a 5 of 5 checked.

### Finding

The routing benefit is saturated by K=2. Going from K=2 to K=8 costs
3.9x the parameters (5.64M to 22.12M) and 2.6x the training time
(27.9 to 69.3 min/run) for +0.098pp, a difference indistinguishable from
zero. The width-32 peak is therefore not an artefact of choosing K=4.

Cost at width 32, per run: static 8.6 min, K=2 27.9 min (3.2x),
K=4 39.2 min (4.6x), K=8 69.3 min (8.1x).

Note for the paper: static at width 64 stores 11.22M and reaches 78.338,
while gap at width 32 with K=8 stores 22.12M and reaches 76.048. Twice
the parameters, 2.29pp worse. This is a third instance of the
spend-it-on-width result.

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

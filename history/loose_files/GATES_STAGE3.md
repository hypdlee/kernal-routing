# Stage 3 gates

Two gaps that would justify a rejection on their own. Neither is about
more data; both are about the claims already made.

---

## Part 2 first: temperature annealing (12 runs, ~9 h)

Every run in the program so far used a fixed softmax temperature tau = 1.
DY-Conv (Chen et al., CVPR 2020) anneals tau from 30 to 1 over the first
ten epochs and reports it as necessary: at tau = 1 the mixture is close
to one-hot from the first step, so most of the bank receives almost no
gradient.

This is a fidelity gap, not a tuning preference. A reviewer familiar with
DY-Conv will ask whether the width-64 null is a property of dynamic
convolution or of a bank that was never trained properly. Until this
runs, the honest answer is that we do not know.

Run this BEFORE the Tiny-ImageNet arm. It is cheaper, and if annealing
changes the width-64 result then the paper's central claim changes and
the Tiny-ImageNet work would need redoing under the new recipe.

- [ ] P1 annealing implemented and the default path unchanged
  CHECK: `python -m src.train --router gap --width 16 --seed 0 --epochs 2 --batch-size 64 --warmup 1 --fake-data --amp off --workers 0 --out results/smoke --tag c`
  EXPECT: no tau in the progress line; config.temp_start is null. Existing
  90 runs must remain reproducible.
  EVIDENCE: pending

- [ ] P2 annealing actually anneals
  CHECK: add `--temp-start 30 --temp-epochs 3` to the same command
  EXPECT: "tau 30.00" at epoch 0 falling to "tau 1.00" by epoch 3
  EVIDENCE: pending

- [ ] P3 12 runs complete (gap, annealed, widths 64 and 32, 6 seeds)
  CHECK: `python run_stage3.py --part temp`
  EVIDENCE: pending

- [ ] P4 verdict at width 64
  CHECK: `python analyze_stage3.py`
  DECISION POINT. If annealed gap is still equivalent to static, the null
  holds under the published recipe and the objection is closed. If it
  beats static, the width-64 null was an artefact of tau = 1, and the
  abstract, Section 4 and the conclusion must be rewritten. Do not
  proceed to Part 1 until this is known.
  EVIDENCE: pending

---

## Part 1: uniform on Tiny-ImageNet (6 runs, ~30 h)

The decomposition into bank and routing exists only on CIFAR-100, because
Tiny-ImageNet was run with static and gap only. The paper's central claim,
that the bank contributes nothing and routing does all the work, is
therefore a single-dataset claim presented as a general one.

Six runs fix it. This is the highest-value compute remaining.

- [ ] U1 6 runs complete
  CHECK: `python check_grid.py --dir results --tag tin_w32 --routers static uniform gap --seeds 0 1 2 3 4 5`
  EXPECT: 18/18 cells, AUDIT PASSED, uniform at 11.06M parameters
  EVIDENCE: pending

- [ ] U2 decomposition reported for both datasets
  CHECK: `python analyze_stage3.py`
  EXPECT: bank and routing rows for CIFAR-100 and Tiny-ImageNet side by
  side. CIFAR gives bank +0.175pp (equivalent) and routing +0.978pp.
  EVIDENCE: pending

- [ ] U3 claim scoped to what was measured
  If the bank is equivalent to static on Tiny-ImageNet too, the claim
  becomes cross-dataset and the abstract can say so. If not, the paper
  must report the disagreement rather than lead with the CIFAR result.
  EVIDENCE: pending

---

## What Stage 3 does NOT fix

Naming these is part of the job. None is closable with this hardware or
with more runs.

1. No bibliography and no related work section. This is the largest
   remaining reason to reject and no experiment addresses it.
2. The implementation is a CondConv/DY-Conv-style layer, not a
   reproduction of any specific published method. No ODConv, FDConv or
   KernelWarehouse baseline was run. The paper should say "a dynamic
   convolution of this form" rather than imply the family.
3. One backbone. ResNet-18 only.
4. Both datasets are low-resolution and single-GPU. ImageNet-scale
   behaviour is untested and untestable here.
5. The capacity sweep across widths remains CIFAR-100 only even after
   Part 1; Tiny-ImageNet has one width.

ABANDON lines go here if a gate becomes impossible, with the reason.

# Stage 4 gates: Tiny-ImageNet at width 64

Closes the last coverage gap. The inverted-U across widths, and the
saturation account of the width-64 null, are currently CIFAR-100 results.
This tests both on a second dataset.

The mechanism makes a clear prediction here. At width 64 on
Tiny-ImageNet the network should be close to memorising the training set,
so the routing effect should shrink toward nothing, as it did on
CIFAR-100 (+1.153pp at w32 falling to +0.080pp at w64). Predicting the
answer before running is the point; recording the prediction is what
makes the test meaningful.

PREDICTION, recorded 2026-09-19 before any run:
  gap - static at w64 on Tiny-ImageNet will be under +0.5pp and
  statistically equivalent to static by TOST. Final training loss of the
  static arm will be below 0.05, against 0.1194 at width 32.
  If either fails, the saturation account is CIFAR-specific.

---

- [x] S1 probe passes
  EVIDENCE: params=44.52M, dyn_layers=16, epoch 0 at 289s, no OOM.
  Batch 128 held, so width-64 Tiny-ImageNet is directly comparable to
  width 32.

- [x] S2 probe artefacts removed
  EVIDENCE: results/probe deleted before the real run.

- [x] S3 18 runs complete
  EVIDENCE: done in 95.8 h (estimate was 87 h). Params static 11.27M,
  uniform 44.23M, gap 44.52M.

- [x] S4 saturation measured
  EVIDENCE: final train loss static 0.021, uniform 0.022, gap 0.019.

- [x] S5 verdict against the recorded prediction
  PREDICTION (a): gap - static under +0.5pp, EQUIVALENT by TOST.
  MEASURED: +2.282pp, 95% CI [+1.952, +2.611], p=1.0e-05, TOST=1.0000.
  -> FAILED.
  PREDICTION (b): static final train loss below 0.05.
  MEASURED: 0.021. -> held.
  Both were stated as required. THE PREDICTION FAILED.

  The saturation condition was met and the effect did not vanish. The
  account "routing helps only away from memorisation, operationalised as
  train loss" does not survive its second out-of-sample test. A threshold
  near 0.01 would fit all five conditions, but it would be chosen after
  seeing the data, so it is a rescue and not evidence.

  No inverted-U on Tiny-ImageNet: +2.185pp at w32, +2.282pp at w64.
  The width-64 null is CIFAR-100 specific.

  What survived: bank (uniform - static) +0.053pp, TOST=0.0157,
  EQUIVALENT (fifth condition). Routing carries 97.6% of the effect.
  Width beats routing at matched parameters: static@w64 11.27M over
  gap@w32 11.16M by +2.520pp, p=6.6e-07 (third budget, second dataset).

- [ ] S6 paper updated either way
  Abstract, Section 5.1, Section 8 and the conclusion present saturation
  as a mechanism with a confirmed prediction. That is no longer true and
  must be rewritten, not softened.
  EVIDENCE: pending

Ledger: 5 of 6 checked.

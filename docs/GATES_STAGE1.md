# Stage 1 gates

## COMPLETE (evidence recorded 2026-09-10)

- [x] G1 scipy installed
  EVIDENCE: analyze.py ran to completion, no ImportError.

- [x] G2 grid complete, 24/24 cells
  EVIDENCE: check_grid.py -> "24/24 cells present", no MISSING line.

- [x] G3 grid uniformly configured
  EVIDENCE: check_grid.py -> "AUDIT PASSED"; epochs=150 batch_size=128
  lr=0.1 weight_decay=0.0005 warmup=5 num_kernels=4 scope=all
  temperature=1.0 identical across all 24 runs.

- [x] G4 analysis produced a verdict
  EVIDENCE: uniform vs gap, mean diff -0.032pp, TOST p=0.0033 -> EQUIVALENT.
  frozen_random vs gap, +0.097pp, TOST p=0.0089 -> EQUIVALENT.
  shuffled vs gap, -0.153pp, TOST p=0.0619 -> INCONCLUSIVE.

- [x] G5 verdict read against the pre-registration
  EVIDENCE: decision table row 2, "not significant + TOST p<0.05 ->
  routing is worth less than 0.5pp, write the null". t-test p=0.7752,
  TOST p=0.0033. Row applies.

- [x] G6 routing entropy recorded for all four arms
  EVIDENCE: uniform 1.0000, gap 0.9885 (min 0.9289), shuffled 0.9996,
  frozen_random 0.9971. 96 layer-observations per arm.

- [x] G7 deviations logged
  EVIDENCE: none required; the run matched the pre-registration exactly.

Ledger: 7 of 7 checked.

---

# Stage 1b gates

## COMPLETE (evidence recorded 2026-09-10)

- [x] S1 static arm complete, 6/6 seeds at --scope none
  EVIDENCE: check_grid.py --tag stage1b -> "6/6 cells present".

- [x] S2 static arm matches the Stage 1 recipe apart from scope
  EVIDENCE: AUDIT PASSED; epochs=150 batch_size=128 lr=0.1
  weight_decay=0.0005 warmup=5 num_kernels=4 temperature=1.0, scope=none.

- [x] S3 static parameter count is 11.22M
  EVIDENCE: params.total = 11220132, dyn_layers = 0, printed on every run.

- [x] S4 comparison against all four Stage 1 arms
  EVIDENCE: static 78.338 +/- 0.284. Paired differences vs static:
  uniform +0.048 (p=0.757), gap +0.080 (p=0.616),
  shuffled -0.073 (p=0.736), frozen_random +0.177 (p=0.187).
  All four EQUIVALENT to static by TOST at +/-0.5pp
  (p = 0.0142, 0.0188, 0.0464, 0.0191).
  Omnibus one-way F(4,25) = 0.954, p = 0.450, eta-sq = 0.132.

- [x] S5 the re-parameterisation reading stated explicitly
  EVIDENCE: uniform does NOT beat static (+0.048pp, p=0.757), so the
  linear over-parameterisation effect is absent too. The K=4 bank buys
  nothing by any of the three mechanisms considered: not routing, not
  representational capacity, not optimisation via re-parameterisation.
  Cost: gap trains 4.57x slower than static for a difference
  indistinguishable from zero.

Ledger: 5 of 5 checked.

# Stage 1c gates (optional, decide deliberately)

- [ ] C1 shuffled resolved
  Currently TOST p=0.0619, just outside. Projected to clear 0.05 at n=7
  (p=0.0464) on the observed sd of 0.459. One extra seed.
  DECISION REQUIRED: extending after an inconclusive pre-registered
  result is legitimate and must be logged in PREREGISTRATION.md with a
  date. Extending after a promising result would be optional stopping.
  This is the former. Log it if you do it.
  EVIDENCE: pending

- [ ] C2 low-capacity replication
  Final train loss is ~0.008 in every arm: the models memorise the
  training set completely. The null may be an artifact of the
  overparameterised regime. Re-run uniform and gap at width 32 or K=2 to
  check the null survives. Two arms, six seeds, cheaper per run.
  EVIDENCE: pending

ABANDON lines go here if a gate becomes impossible, with the reason.

---

# Stage 1c gates (capacity sweep)

## width 32 COMPLETE (evidence recorded 2026-09-10)

- [x] C1 width recorded in every stage1c result
  EVIDENCE: config.width = 32 in all 18 records; run headers read
  "[gap w32 seed=N]".

- [x] C2 parameter counts match the expected sweep
  EVIDENCE: static 2.82M, uniform 11.06M, gap 11.13M. As predicted.

- [x] C3 width 32 complete, 18 runs
  EVIDENCE: check_grid.py --tag stage1c_w32 -> 18/18, AUDIT PASSED.
  (Required a fix to check_grid.py: 'scope' was in CRITICAL and flagged
  the static arm's scope=none as recipe drift. Scope is now checked
  per-arm instead. stage1 and stage1b re-audited, both still pass.)

- [x] C5 trend read from per-width tests
  EVIDENCE: w32 gap - static = +1.153pp, p=0.00003, CI [+0.940, +1.367].
  w64 gap - static = +0.080pp, p=0.616, EQUIVALENT. Decomposition at w32:
  bank alone +0.175pp (p=0.221, equivalent), routing alone +0.978pp
  (p=0.00104). At w64 both components are null.
  Parameter-matched: static@w64 (11.22M) beats gap@w32 (11.13M) by
  +2.313pp, p=0.00004.

- [ ] C4 saturation actually broken at the small widths
  STATUS: NOT MET at w32. Final train loss 0.0170 (static/uniform) and
  0.0132 (gap), against 0.0080 at w64. Lower, but still near-total
  memorisation. The w32 effect is therefore real but was measured inside
  the saturated regime, and the objection is not yet answered.
  EVIDENCE: pending at w16.

## width 16 COMPLETE (evidence recorded 2026-09-11)

- [x] C6 width 16 complete, 18 runs
  EVIDENCE: params static 0.71M, uniform 2.77M, gap 2.79M as predicted.

- [x] C8 saturation broken
  EVIDENCE: w16 final train loss 0.164 (static and uniform) against 0.008
  at w64. Genuinely outside the memorisation regime. Gate C4, which was
  not met at w32 (0.017), is now met at w16. The reviewer objection is
  answered.

- [x] C7 three-point trend read
  EVIDENCE: gap - static = +0.532pp (w16, p=0.0023), +1.153pp (w32,
  p=0.00003), +0.080pp (w64, p=0.616). NON-MONOTONE, peak at w32.
  Routing isolated (gap - uniform): +0.755 (w16), +0.978 (w32),
  +0.032 (w64). uniform - static equivalent at all three widths.
  (Required a second fix to analyze_capacity.py: its verdict said "real
  at the smallest width, gone at the largest", which is true here but
  implies a monotone trend that does not exist. It now detects an
  interior peak and refuses the trend language.)

Ledger: stage1c 8 of 8 checked. Total across all stages: 20 of 20.

---

# Where the evidence stands (90 runs)

Effect of dynamic convolution over a plain ResNet-18, by width:

    w16   +0.532pp   p=0.0023     train loss 0.164 -> not saturated
    w32   +1.153pp   p=0.00003    train loss 0.017 -> near-saturated
    w64   +0.080pp   p=0.616      train loss 0.008 -> saturated

The kernel bank alone contributes nothing at any width. The routing does
the work, and only outside saturation.

Parameter-matched, replicated at two budgets:

    static@w32 (2.82M)  beats gap@w16 (2.79M)  by +4.345pp
    static@w64 (11.22M) beats gap@w32 (11.13M) by +2.313pp

Training cost of the bank: 3.5x to 4.6x static at every width.

# Open decisions

- [ ] D1 second dataset
  One dataset carries the whole claim. This is the first thing a reviewer
  will ask for. Tiny-ImageNet at w32 and w64, three arms, would settle it.
  DECISION REQUIRED: run it, or scope the paper to CIFAR-100 explicitly
  in the title and abstract.

- [ ] D2 whether to add w128
  Would confirm the null persists as capacity grows rather than only at
  w64. Expensive (roughly 4x the w64 runs) and arguably redundant given
  the saturation measurements already explain the w64 null.

- [ ] D3 K sweep at w32
  The peak sits at w32 with K=4. Whether the peak moves with K is
  unknown and is the obvious follow-up question.

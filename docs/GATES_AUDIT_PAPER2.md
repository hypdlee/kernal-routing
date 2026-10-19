# Gates: audit of the two-backbone paper (2026-10-19)

- [x] A1 Every hand-typed claim in the template read against code, data, logs or the cited paper.
  EVIDENCE: docs/AUDIT_PAPER2.md lists 27 errors fixed and what was verified unchanged.
- [x] A2 Independent audit script passes.
  CHECK: python3 tools/audit_paper2.py
  EXPECT: 0 FAIL
  EVIDENCE: 43 ok, 0 FAIL (96 ResNet-18 per-run accuracies vs logs, 24 result files, MobileNetV2 recomputed with scipy, 15 code claims, 12 banned phrases).
- [x] A3 No reported number changed by the audit.
  EVIDENCE: macros.tex and macros_mb.tex byte-identical to the pre-audit build; every table identical except tab_CONTRA status text ("never significantly higher"); every figure identical.
- [x] A4 Build clean, no overfull box.
  EVIDENCE: errors 0, undefined refs 0, undefined macros [], Overfull 0, 26 pages; Overleaf copy compiles standalone with 0 errors.
- [x] A5 Pages rendered and read; no em-dash.
  EVIDENCE: text of all 26 pages read; pages 3 and 4 viewed as images; template grep for the dash characters: 0.
- [ ] A6 ResNet-18 routing entropies for CIFAR-100 w16, w32 and the three annealed groups re-derived from result files.
  ABANDON: A6 the result files for those groups are not in the archive; disclosed in the paper (Provenance paragraph), README and docs/AUDIT_PAPER2.md. Needs the user's results folder.

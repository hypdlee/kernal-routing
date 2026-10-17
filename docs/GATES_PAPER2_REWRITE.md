# Gates: two-backbone paper rewritten as one consistent story (2026-10-17)

User request: the paper says one thing at the start and something else at
the end; rewrite the whole thing from start to end as what was found, so
it no longer confuses the reader or co-authors. Do not modify results.

Ledger of findings (fixed wording) is FINDINGS_PAPER2.md. Template:
report/paper2_template.tex. Build: report/build_paper2.py.

## Contradictions found in the 17 Oct draft (each must be gone)

- [x] C1 Structure: Sections 5-12 conclude on ResNet-18 alone (width wins, annealing neutral, entropy counterexample), Section 13 then overturns them. Every result section must report both backbones together.
  CHECK: grep -c '\\section{ResNet-18\|\\section{A second backbone' report/paper2_template.tex
  EXPECT: 0
  EVIDENCE: grep count 0. New order: 5 bank+routing (both), 6 the one null, 7 router training, 8 bank size, 9 cost, 10 temperature, 11 entropy, 12 predictions, 13 result by result; every result section reports both backbones.
- [x] C2 "Routing" used for two things: input-dependent mixing vs a trained router. Abstract says "routing carries the effect", MBv2 says the router need not be trained. Define the term once in the intro and use it consistently.
  EVIDENCE: Intro paragraph 'Terms.' defines routing (input-dependent z, trained or not) and trained router; abstract (2)-(3) and conclusion use 'input-dependent mixing' and 'does not have to be learned'.
- [x] C3 Bank: abstract "adds nothing measurable" vs P1 "failed". The distinction (never significant vs equivalence demanded) must be stated in abstract/intro and in the bank section.
  EVIDENCE: Intro 'Predictions.': 'The first failed because three bank terms are inconclusive, not because any is significant, so it does not contradict finding 1.' Repeated in Section 12.
- [x] C4 Old Section 5: "a router whose weights were never trained matches a trained one" on ResNet-18 as if a finding, while the intro calls the ResNet-18 test uninformative. Must say uninformative everywhere.
  EVIDENCE: Section 7 'ResNet-18.': 'With no benefit to explain, this says nothing about whether routing needs to be trained. The finding is therefore MobileNetV2 only'. Intro item 3 and abstract say the same.
- [x] C5 Old Section 11: "the annealed router went flat during the high-tau phase", but entropies were only recorded at the end of training. Must not claim when it went flat.
  CHECK: grep -c 'during the high' report/paper2_template.tex
  EXPECT: 0
  EVIDENCE: grep count 0; Section 10: 'entropy was recorded only at the end of training, so we cannot say when any router became flat'.
- [x] C6 Old Section 11: "This follows from Proposition 1" asserted a mechanism that was never tested. Must be labelled an untested interpretation.
  CHECK: grep -c 'This follows from' report/paper2_template.tex
  EXPECT: 0
  EVIDENCE: grep count 0; Section 10: 'One reading follows from Proposition 1 ... We did not test this'.
- [x] C7 Old Section 10: "This result depends on almost every parameter of ResNet-18 sitting in a dynamic layer" asserted a cause; old Section 14 says no property was isolated. Remove the causal claim.
  CHECK: grep -c 'depends on almost every parameter' report/paper2_template.tex
  EXPECT: 0
  EVIDENCE: grep count 0; Section 13 'Why the backbones may differ': 'none of the backbone-dependent results can be attributed to one of them'.
- [x] C8 Intro listed "whether the router has to be trained" as backbone-dependent; abstract calls it a MobileNetV2-only result. One label: MobileNetV2 only (uninformative on ResNet-18).
  EVIDENCE: Intro item 3 labelled '(MobileNetV2 only)'; table status 'MobileNetV2 only'; abstract (3) 'On MobileNetV2'.
- [x] C9 Temperature framed as "differs" between backbones, although annealing never helped on either. One statement: never higher; lower in 4 of 6 conditions.
  EVIDENCE: Abstract (6), intro item 6, Section 10, conclusion: never improved, lower in 4 of 6 (macro nannlower=4, nannhigher=0); table status 'holds on both (never higher)'.
- [x] C10 Entropy: abstract "does not measure the benefit" vs MBv2 "entropy and benefit move together". One statement covering within-ResNet-18, within-MobileNetV2 and across backbones.
  EVIDENCE: Abstract (7), intro item 7, Section 11 'Conclusion', paper Conclusion all give: does not order within ResNet-18, does within MobileNetV2, same entropy +0.03 vs +1.47 across backbones.
- [x] C11 Old Section 12: "entropy is not meaningless: near-uniform router at R18 w64 collapsed" vs MBv2 annealed w64 near-uniform yet +0.55 pp. Must be reported together.
  EVIDENCE: Section 11 'After annealing...': R18 w64 annealed 1-H 0.00018 equivalent to uniform; MBv2 w64 annealed 1-H 0.00069 still +0.55 pp; 'accompanied no benefit in one case and a significant benefit in the other'.
- [x] C12 Bank size: intro "banks larger than K=2 add nothing further" vs MBv2 K8-K2 inconclusive. Use "not distinguishable" everywhere; equivalent on R18, inconclusive on MBv2.
  CHECK: grep -c 'add nothing further' report/paper2_template.tex
  EXPECT: 0
  EVIDENCE: grep count 0; intro item 4 and Section 8: 'not distinguishable'; equivalent on R18, inconclusive on MBv2.
- [x] C13 Old Section 14 interpretation "a schedule that keeps it flat early removes most of the benefit" ignores R18 width 16 where annealing kept the benefit. Must state the counter-case.
  EVIDENCE: Section 13 'An interpretation.': 'ResNet-18 does not support this reading everywhere: at width 16 the annealed router kept the full benefit (+0.870 against +0.755 pp)'.
- [x] C14 Cost: abstract reported only equal parameters; conclusion adds equal compute. Abstract, intro, section and conclusion must give both views.
  EVIDENCE: Abstract (5), intro item 5, Section 9 (9.1 equal parameters, 9.2 equal compute, Summary) and conclusion all give both views.
- [x] C15 Proposition 1 text says "a plain ResNet-18" in a two-backbone paper.
  CHECK: grep -c 'therefore a plain ResNet-18' report/paper2_template.tex
  EXPECT: 0
  EVIDENCE: grep count 0; Section 3.2 now 'At inference uniform is therefore the plain network. On ResNet-18 at width 64 ...'.
- [x] C16 Contradiction table status labels must use the same vocabulary as the text (holds on both / backbone-dependent / MobileNetV2 only / across backbones), and the count quoted in the threats must be generated, not typed.
  EVIDENCE: tab_CONTRA status column: holds on both x4, backbone-dependent x4, MobileNetV2 only x1, across backbones x1; threats text uses generated macros ncontraBD=4 of ncontra=10.

## Consistency of the finished paper

- [x] K1 Each finding F1-F7 appears in abstract, introduction list, its section and the conclusion with the same scope label and the same numbers. Verified by reading the extracted PDF text with a per-finding table.
  EVIDENCE: Read extracted prose (528 lines) start to end. F1: abstract (1), intro 1, Sec 5, conclusion: never significant, equivalent 7/10, 3 inconclusive MBv2. F2: 9/10 beats plain, routing term 9/10, R18 C100 w64 null unexplained. F3: MBv2 only. F4: not distinguishable from K=2. F5: both cost views, 2.3-4.3 pp, 1.7-2.0x, 4/5 and 5/5. F6: never higher, lower 4/6. F7: within R18 no, within MBv2 yes, +0.03 vs +1.47. Predictions 3 of 4 failed in abstract, intro, Sec 12, conclusion.
- [x] K2 Every table and figure of the old paper is still in the new paper and referenced in the text.
  CHECK: python3 tools_check_floats.py
  EXPECT: all old labels present and referenced
  EVIDENCE: assemble.py: 'all 46 old floats present'; unreferenced-label check: [] (after adding refs to fig:mbdecomp, fig:tin and appendix floats).
- [x] K3 No previously reported number changed: R18 macros identical, every old MBv2 macro identical, every table identical except tab_CONTRA status column.
  EVIDENCE: R18 macros.tex identical; 206 old MBv2 macros: changed []; added 13 counts (bothbanksig, botheff, bothrout, mbpinvhi, mbpinvlo, nannhigher, nannlower, ncontra, ncontraBD, ncontraHold, rbnbanksig, rbnrout); every table byte-identical except tab_CONTRA, whose first three columns are identical (status only); every old figure identical.
- [x] K4 Build clean.
  CHECK: python3 report/build_paper2.py --r18-fixture --mb-log /tmp/claude-0/mbjson/x/mobilenetv2/train.log --mb-results /tmp/claude-0/mbjson/x/mobilenetv2
  EXPECT: errors 0, undefined refs 0, undefined macros []
  EVIDENCE: report/out/paper2/main.pdf: errors 0, undefined refs 0, undefined macros []
- [x] K5 No em-dashes (unicode or ---) in the template.
  CHECK: grep -c -- '---\|—' report/paper2_template.tex
  EXPECT: 0
  EVIDENCE: grep count 0
- [x] K6 Every page rendered and looked at; no float in references, no overfull table.
  EVIDENCE: All 25 pages rendered (pair-01..25.png) and inspected: no float after references, appendix captions name their backbone ('Every run in the study' corrected to 'Every ResNet-18 run').
- [x] K7 Report tests still pass (test_report, test_analysis).
  EVIDENCE: test_report: ok; test_analysis: 4 models ok.
- [x] K8 Overleaf zip compiles standalone; repo rebuild identical; committed and pushed.
  EVIDENCE: Overleaf zip standalone: errors 0, undefined 0, 25 pages; repo rebuild main.tex identical; commit and push: see git log.

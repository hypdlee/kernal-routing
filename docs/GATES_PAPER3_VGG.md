# Gates: add VGG-16 to the paper (2026-10-21)

User request: VGG-16 train.log and runner.log delivered; write the whole
document with this update, carefully, no statement at the start that the
end contradicts. Deliverable: Word document (built from the same source as
the PDF). No result may be changed or invented.

## Data
- [x] D1 132 VGG-16 runs parsed from train.log, one final line each, every runner "final" equal to the log's final.
  EVIDENCE: mbv2_log.parse(model="vgg16") 132 runs; runner log 132 "finished" lines, 0 final mismatches; configuration line checked against each final line.
- [x] D2 Parameter counts printed in the log agree with PARAMS_EXPECTED.md (2 decimals) for every VGG-16 group.
  EVIDENCE: 0 mismatches between log params/router counts and PARAMS_EXPECTED.md (the build stops otherwise).
- [x] D3 Gate, micro-batch, resumed runs and divergences established from the logs.
  EVIDENCE: gate 74.68 (floor 65); micro-batch 128, accum 1 in all 132; one resumed run (shuffled seed 0, C100 w64, epoch 58); no divergence, no traceback.
- [x] D4 VGG-16 headline statistics recomputed independently (scipy, audit script) and equal to the generated macros.
  EVIDENCE: tools/audit_paper2.py section 6 parses the logs with its own regex, recomputes with scipy and compares to report/out/paper2 macros: all pass.

## Findings ledger
- [x] L1 docs/FINDINGS_PAPER3.md written before the text: every finding with scope over three backbones, numbers from the build.
  EVIDENCE: docs/FINDINGS_PAPER3.md written before the template edits.

## Paper
- [x] P1 Every result section reports all three backbones; VGG-16 entropy marked as not available (no result files) everywhere entropy is discussed.
  EVIDENCE: sections 5-12 and 13-15 each name VGG-16; entropy section, Section 7 outputs, Section 10, threats and abstract state VGG-16 entropies not available.
- [x] P2 Abstract, introduction, each section and conclusion state each finding with the same scope and numbers (checked by reading the full text against the ledger).
  EVIDENCE: full PDF text read start to end; fixes made: "well-powered" limited to ResNet-18, stem sentence scoped to ResNet-18, interrupted-run verdict wording, controls row not counted as an R18/VGG agreement, sentences no longer start with a numeral.
- [x] P3 Previously reported ResNet-18 and MobileNetV2 numbers unchanged (macro diff against the pre-VGG build).
  EVIDENCE: macro diff vs pre-VGG build: only totruns 264->396, totgpuh 458->564, ncontraBD 4->5 changed; tables identical except CONTRA, COST (3 VGG rows added) and MBW64 (VGG columns and a gap-uniform row added, old values identical); figures g1,g2,g3,g5,g6 redrawn with VGG-16.
- [x] P4 No hand-typed number that is not checked by the audit script.
  EVIDENCE: hand-typed counts covered by audit (ncontraX=1, VGG p<1e-4 in each twice-width pair, 4-10x cost ratio, ledger counts).
- [x] P5 Build clean (0 errors, 0 undefined, 0 overfull); Word file built from it; every page of the Word file rendered and looked at.
  EVIDENCE: errors 0, undefined 0, Overfull 0, 30 pages PDF; Word file 32 pages rendered and viewed (all six page montages, Table 23 at full size).
- [x] P6 Audit script passes, extended with VGG-16 checks.
  EVIDENCE: 84 ok, 0 FAIL; tests/test_report.py ok, tests/test_analysis.py 4 models ok.
- [x] P7 Committed and pushed; Word file delivered.
  EVIDENCE: see git log; Word file sent.

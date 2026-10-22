"""Report checks (gates E1-E3, E5, G1).

    python tests/test_report.py per_model | resnet18_values | cross | matched | all

per_model builds five per-model PDFs: ResNet-18 from the fixture of the
paper's per-seed values, the four new backbones from synthetic data
(watermarked). Real results replace both with the same commands.
"""
import sys, pathlib, re, tempfile, json, subprocess, collections
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "report")); sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

OUT = pathlib.Path(tempfile.gettempdir()) / "kr_report_test"
OLD_PAPER = pathlib.Path("/mnt/attach/outputs/KernelRoute_Report_final.pdf")
NEW = ["mobilenetv2", "vgg16", "resnet50", "convnext_tiny"]


def per_model_dirs():
    import make_r18_jsons
    from synth import make
    d = {"resnet18": OUT / "json" / "resnet18"}
    if not d["resnet18"].exists():
        make_r18_jsons.write(d["resnet18"])
    for i, m in enumerate(NEW):
        d[m] = OUT / "json" / m
        if not d[m].exists():
            make(m, d[m], seed=10 + i, drop=1, diverge=1)
    return d


def test_per_model():
    from build_model_report import build
    dirs = per_model_dirs()
    tot_f = tot_t = 0; pdfs = []
    for m, d in dirs.items():
        r = build(m, [d], OUT / "pm" / m, synthetic=(m != "resnet18"),
                  banner="Built from the per-seed values printed in the ResNet-18 paper (test fixture)."
                  if m == "resnet18" else "")
        tex = (OUT / "pm" / m / f"report_{m}.tex").read_text()
        nfig = len(set(re.findall(r"\\includegraphics\[[^]]*\]\{(fig/[^}]+)\}", tex)))
        ntab = len(re.findall(r"\\begin\{tabular\}", tex)) - 2           # minus coverage and interpolation
        assert r["pdf"].exists() and not r["errors"] and not r["undefined"], (m, r["errors"], r["undefined"])
        assert nfig == 10 and ntab == 13, (m, nfig, ntab)
        assert all((OUT / "pm" / m / f).exists() for f in re.findall(r"\{(fig/[^}]+)\}", tex))
        pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(r["pdf"])], capture_output=True,
                                                                  text=True).stdout).group(1))
        tot_f += nfig; tot_t += ntab; pdfs.append(r["pdf"])
        print(f"  {m:<14} {r['pdf'].name}: {pages} pages, {nfig} figures, {ntab} tables (+2), "
              f"{r['runs']} runs, 0 errors, 0 undefined")
    assert len(pdfs) == 5 and tot_f == 50 and tot_t == 65
    print(f"5 PDFs, {tot_f} figures, {tot_t} tables, 0 errors, 0 undefined refs")


def test_resnet18_values():
    """The rebuilt ResNet-18 paper against the delivered final PDF: every
    decimal number in the old PDF text is in the new one, except the two
    cells whose unmeasured third decimal changed the rounding."""
    from build_paper import build
    import make_r18_jsons
    with tempfile.TemporaryDirectory() as t:
        make_r18_jsons.write(pathlib.Path(t) / "j")
        pdf, errs, und = build([pathlib.Path(t) / "j"], OUT / "paper")
    assert not errs and not und, (errs, und)
    if not OLD_PAPER.exists():
        print("  delivered PDF not present; built only"); return
    txt = {}
    for k, p in (("old", OLD_PAPER), ("new", pdf)):
        s = subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout
        txt[k] = collections.Counter(re.findall(r"[+\-]?\d+\.\d+", s.replace("\u2212", "-")))
    lost = sorted((txt["old"] - txt["new"]).elements())
    # axis tick labels of the old figures that the plain-matplotlib restyle of
    # 22 Oct 2026 no longer prints (not results): 0.5 (twice) and 1.50, 1.75
    ticks = collections.Counter({"0.5": 2, "1.50": 1, "1.75": 1})
    lost = sorted((collections.Counter(lost) - ticks).elements())
    assert lost == ["44.47", "44.47"], lost
    added = sorted((txt["new"] - txt["old"]).elements())
    print(f"  {sum(txt['old'].values())} decimal numbers in the delivered paper; all present in the rebuild except "
          f"{lost} (gap and shuffled, CIFAR-100 w64: counted 44.4645M rounds to 44.46)")
    print(f"  added: {added} (param-ratio column 0.99 x3; correction note)")
    print("ok")


def test_cross():
    from cross import build
    dirs = {m: [d] for m, d in per_model_dirs().items()}
    r = build(dirs, OUT / "cross", synthetic=True)
    assert not r["errors"] and not r["undefined"] and r["figures"] == 6 and r["tables"] == 8, r
    assert r["models"] == ["resnet18", "mobilenetv2", "vgg16", "resnet50", "convnext_tiny"]
    # a failed gate is listed, not dropped
    with tempfile.TemporaryDirectory() as t:
        g = pathlib.Path(t) / "convnext_tiny"; g.mkdir()
        (g / "GATE_FAILED.json").write_text(json.dumps({"final_acc": 41.2, "floor": 60.0}))
        dirs2 = dict(dirs); dirs2["convnext_tiny"] = [g]
        r2 = build(dirs2, OUT / "cross_gate", synthetic=True)
        assert not r2["errors"] and [m for m, _ in r2["absent"]] == ["convnext_tiny"]
        row = [l for l in (OUT / "cross_gate" / "tab_XSUMMARY.tex").read_text().splitlines() if "ConvNeXt" in l][0]
        assert "gate failed" in row and "41.20" in row, row
    print(f"  cross_models.pdf: 5 backbones, {r['figures']} figures, {r['tables']} tables, 0 errors; "
          f"with a failed gate: ConvNeXt-T listed as '{row.split('&')[1].strip()[:60]}...'")
    print("ok")


def test_matched():
    """'matched' only within 10 percent, and the real ratio always shown."""
    n_m = n_nm = 0
    for m in ["resnet18"] + NEW:
        d = OUT / "pm" / m
        if not d.exists():
            raise SystemExit("run per_model first")
        for line in (d / "tab_MATCHED.tex").read_text().splitlines():
            ratio, lab = re.search(r"& ([0-9.]+) \(((?:not )?matched)\) \\\\$", line).groups()
            within = abs(float(ratio) - 1) <= 0.10
            assert (lab == "matched") == within, (m, line)
            n_m += within; n_nm += not within
        v = (d / "verdicts.tex").read_text()
        ml = re.search(r"vmatchlist\\endcsname\{(.*)\}$", v, re.M).group(1)
        for part in ml.split("; "):
            r_ = float(re.search(r"parameter ratio ([0-9.]+)", part).group(1))
            assert ("(matched)" in part) == (abs(r_ - 1) <= 0.10), part
    for line in (OUT / "cross" / "tab_XMATCHED.tex").read_text().splitlines():
        cells = [c.strip() for c in line.rstrip(" \\").split("&")]
        assert (cells[6] == "matched") == (abs(float(cells[5]) - 1) <= 0.10), line
    print(f"  per-model tables: {n_m} matched, {n_nm} not matched, every label consistent with its ratio; "
          f"verdict sentences and cross table consistent")
    print("ok")


TESTS = dict(per_model=test_per_model, resnet18_values=test_resnet18_values, cross=test_cross, matched=test_matched)
if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for w in (TESTS if which == ["all"] else which):
        print(f"== {w}")
        TESTS[w]()

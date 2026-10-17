"""
Per-model report: generators -> tables substituted -> pdflatex x2.

    python report/build_model_report.py --model vgg16 --results results_models/vgg16
    python report/build_model_report.py --model resnet18 --results results
    python report/build_model_report.py --model vgg16 --synthetic      # pipeline test

Output: report/out/<model>/report_<model>.pdf (and the .tex, figures and
tables beside it, so the folder uploads to Overleaf as is).
"""
import argparse, pathlib, re, shutil, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen import generate

TABLES = ["ALLRUNS", "CONFIG", "VSSTATIC", "DECOMP", "MATCHED", "ENTW64", "KSWEEP", "KPAIRS",
          "PREDICT", "SATUR", "TEMP", "ENTBEN", "INVENT"]
EXTRA = ["INTERP", "COVER"]


def latex(out, stem):
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], cwd=out, capture_output=True)
    log = (out / f"{stem}.log").read_text(errors="ignore")
    errs = [l for l in log.splitlines() if l.startswith("! ")]
    und = [l for l in log.splitlines() if "undefined" in l.lower() and "Warning" in l]
    return errs, und


def build(model, results, out, synthetic=False, banner=""):
    out = pathlib.Path(out)
    D, _ = generate(model, results, out, synthetic=synthetic, banner=banner)
    src = (HERE / "model_template.tex").read_text()
    for k in TABLES + EXTRA:
        src = src.replace(f"%%{k}%%", (out / f"tab_{k}.tex").read_text().rstrip())
    left = re.findall(r"%%[A-Z0-9]+%%", src)
    if left:
        raise SystemExit(f"unsubstituted: {left}")
    stem = f"report_{model}"
    (out / f"{stem}.tex").write_text(src)
    errs, und = latex(out, stem)
    figs = len(re.findall(r"\\includegraphics", src))
    tabs = len([k for k in TABLES if f"%%{k}%%" in (HERE / "model_template.tex").read_text()])
    return dict(pdf=out / f"{stem}.pdf", errors=errs, undefined=und, figures=figs, tables=tabs,
                runs=D.n_runs())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out or HERE / "out" / (a.model + ("_SYNTHETIC" if a.synthetic else "")))
    res = a.results
    if a.synthetic:
        from synth import make
        tmp = pathlib.Path(tempfile.mkdtemp()); make(a.model, tmp, drop=1, diverge=1); res = [tmp]
    r = build(a.model, res, out, a.synthetic)
    print(f"{r['pdf']}: {r['figures']} figures, {r['tables']} of 13 tables (+ interpolation, coverage), "
          f"{r['runs']} runs, errors {len(r['errors'])}, undefined {len(r['undefined'])}")
    for e in r["errors"][:10]: print("  ", e)
    for u in r["undefined"][:10]: print("  ", u)

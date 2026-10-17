"""
The two-backbone paper: ResNet-18 sections from the ResNet-18 results,
MobileNetV2 sections from the MobileNetV2 training log (or, later, results).

    python report/build_paper2.py --r18-results results --mb-log results_models/mobilenetv2/train.log
    python report/build_paper2.py --r18-fixture --mb-log path/to/train.log

Output: report/out/paper2/main.pdf with a self-contained main.tex and fig/.
"""
import argparse, os, pathlib, re, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "tests" / "fixtures"))
from gen import generate


def build(r18_results, mb_log, out):
    out = pathlib.Path(out).resolve()
    generate("resnet18", r18_results, out)
    cwd = os.getcwd()
    try:
        os.chdir(out)
        import mbv2_section
        mbv2_section.main(str(pathlib.Path(mb_log).resolve()))
    finally:
        os.chdir(cwd)
    src = (HERE / "paper2_template.tex").read_text()
    for t in sorted(out.glob("tab_*.tex")):
        src = src.replace(f"%%{t.stem[4:]}%%", t.read_text().rstrip())
    left = re.findall(r"%%[A-Z0-9]+%%", src)
    if left:
        raise SystemExit(f"unsubstituted: {left}")
    src = src.replace("\\input{macros.tex}", (out / "macros.tex").read_text().rstrip())
    src = src.replace("\\input{macros_mb.tex}", (out / "macros_mb.tex").read_text().rstrip())
    (out / "main.tex").write_text(src)
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=out, capture_output=True)
    log = (out / "main.log").read_text(errors="ignore")
    errs = [l for l in log.splitlines() if l.startswith("! ")]
    und = [l for l in log.splitlines() if ("undefined" in l.lower() and "Warning" in l)]
    # every \V{...} used must be defined
    used = set(re.findall(r"\\V\{(\w+)\}", (HERE / "paper2_template.tex").read_text()))
    defined = set(re.findall(r"\\csname v(\w+)\\endcsname", src))
    return out / "main.pdf", errs, und, sorted(used - defined)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r18-results", nargs="+")
    ap.add_argument("--r18-fixture", action="store_true")
    ap.add_argument("--mb-log", required=True)
    ap.add_argument("--out", default=str(HERE / "out" / "paper2"))
    a = ap.parse_args()
    res = a.r18_results
    if a.r18_fixture:
        import make_r18_jsons
        tmp = pathlib.Path(tempfile.mkdtemp()); make_r18_jsons.write(tmp); res = [tmp]
    pdf, errs, und, missing = build(res, a.mb_log, a.out)
    print(f"{pdf}: errors {len(errs)}, undefined refs {len(und)}, undefined macros {missing}")
    for e in errs[:10]: print("  ", e)
    for u in und[:10]: print("  ", u)

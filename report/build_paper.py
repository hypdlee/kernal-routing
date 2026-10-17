"""
The ResNet-18 paper, rebuilt from result files.

    python report/build_paper.py --results results            # the user's 132 JSONs
    python report/build_paper.py --fixture                     # from the paper's own per-seed values

Tables are substituted (not \\input: booktabs alignment breaks inside a
tabular), macros inlined, pdflatex x2. Output report/out/paper/main.pdf plus
the self-contained main.tex, fig/ and references for Overleaf.
"""
import argparse, pathlib, re, shutil, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "tests" / "fixtures"))
from gen import generate


def build(results, out):
    out = pathlib.Path(out)
    generate("resnet18", results, out)
    src = (HERE / "main_template.tex").read_text()
    for f in sorted(out.glob("tab_*.tex")):
        src = src.replace(f"%%{f.stem[4:]}%%", f.read_text().rstrip())
    left = re.findall(r"%%[A-Z0-9]+%%", src)
    if left:
        raise SystemExit(f"unsubstituted: {left}")
    src = src.replace("\\input{macros.tex}", (out / "macros.tex").read_text().rstrip())
    (out / "main.tex").write_text(src)
    for bib in HERE.glob("*.bib"):
        shutil.copy(bib, out)
    for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
        arg = "main" if cmd == ["bibtex"] else "main.tex"
        if cmd == ["bibtex"] and not list(out.glob("*.bib")):
            continue
        subprocess.run(cmd + (["-interaction=nonstopmode"] if cmd != ["bibtex"] else []) + [arg], cwd=out,
                       capture_output=True)
    log = (out / "main.log").read_text(errors="ignore")
    errs = [l for l in log.splitlines() if l.startswith("! ")]
    und = [l for l in log.splitlines() if "undefined" in l.lower() and "Warning" in l]
    return out / "main.pdf", errs, und


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--out", default=str(HERE / "out" / "paper"))
    a = ap.parse_args()
    res = a.results
    if a.fixture:
        import make_r18_jsons
        tmp = pathlib.Path(tempfile.mkdtemp()); make_r18_jsons.write(tmp); res = [tmp]
    pdf, errs, und = build(res, a.out)
    print(f"{pdf}: errors {len(errs)}, undefined {len(und)}")
    for e in errs[:8]: print("  ", e)
    for u in und[:8]: print("  ", u)

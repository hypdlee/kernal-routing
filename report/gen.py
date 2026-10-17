"""
Run the figure, table and macro generators for one model.

    python report/gen.py --model vgg16 --results results_models/vgg16 --out report/out/vgg16

Writes fig/f1..f10 (PDF), tab_*.tex (13 tables) and macros.tex into --out.
--paper-reference runs the paper's own generators on the paper's own data
module instead (tests only).
"""
import argparse, os, pathlib, runpy, sys, io, contextlib

HERE = pathlib.Path(__file__).resolve().parent


def generate(model, results, out, synthetic=False, strict=True, paper_reference=None, quiet=True, patch=None, banner=""):
    out = pathlib.Path(out).resolve(); (out / "fig").mkdir(parents=True, exist_ok=True)
    results = [str(pathlib.Path(r).resolve()) for r in results]
    cwd = os.getcwd()
    # reproducible figures: matplotlib stamps PDFs with this date instead of now
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    src = pathlib.Path(paper_reference).resolve() if paper_reference else HERE
    sys.path.insert(0, str(src))
    for m in ("data", "figs", "figs2", "tables", "macros"):
        sys.modules.pop(m, None)
    try:
        os.chdir(out)
        import data
        errs = [] if paper_reference else data.load(model, results, strict=strict, synthetic=synthetic)
        data.BANNER = banner
        if patch:
            patch(data)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf if quiet else sys.stdout):
            for g in ("figs", "figs2", "tables", "macros") + (() if paper_reference else ("verdicts",)):
                runpy.run_path(str(src / f"{g}.py"), run_name="__main__")
        return data, errs
    finally:
        import matplotlib.pyplot as plt
        if hasattr(plt, "_orig_savefig"):
            plt.savefig = plt._orig_savefig        # the generators' watermark hook stays local
        os.chdir(cwd)
        sys.path.remove(str(src))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    D, errs = generate(a.model, a.results, a.out, a.synthetic, quiet=False)
    print(f"{a.model}: {D.n_runs()} runs, {len(D.MISSING)} missing, {len(D.DIVERGED)} diverged -> {a.out}")

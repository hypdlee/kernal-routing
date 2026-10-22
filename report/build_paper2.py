"""
The two-backbone paper: ResNet-18 sections from the ResNet-18 results,
MobileNetV2 sections from the MobileNetV2 training log and result files.
Accuracies and losses are read from the log; routing entropies from the
result files, after every final accuracy in them has been checked against
the log (the build stops on any mismatch).

    python report/build_paper2.py --r18-results artifacts/resnet18/results \
        --mb-log artifacts/mobilenetv2/train.log --mb-results artifacts/mobilenetv2/results \
        --vgg-log artifacts/vgg16/train.log --vgg-results artifacts/vgg16/results

The ResNet-18 values are read from its 132 result files and checked against
the per-seed values transcribed for the earlier drafts (paper_r18_data.py);
--r18-fixture builds from the transcription instead (tests only).

Output: report/out/paper2/main.pdf with a self-contained main.tex and fig/.
"""
import argparse, os, pathlib, re, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "tests" / "fixtures"))
from gen import generate


FIXTURE = False


def check_r18_transcription(D):
    """The per-seed values the earlier drafts were built on
    (paper_r18_data.py) against the result files: every final accuracy
    identical, every routing entropy equal to the file's value rounded to
    the four decimals that were transcribed."""
    import paper_r18_data as P
    import numpy as np
    bad = []
    for k, arms in P.ACC.items():
        for a, v in arms.items():
            if not np.array_equal(np.asarray(v, float), np.asarray(D.ACC[k][a], float)):
                bad.append(f"accuracy {k} {a}")
    for k, v in P.ENT.items():
        if k[0] == "cifar100" and np.max(np.abs(np.asarray(v) - np.round(np.asarray(D.ENT[k], float), 4))) > 1e-12:
            bad.append(f"entropy {k}")
    for a, v in P.ENT_W64.items():
        if np.max(np.abs(np.asarray(v) - np.round(np.asarray(D.ENT_W64[a], float), 4))) > 1e-12:
            bad.append(f"entropy w64 {a}")
    if bad or D.n_runs() != 132 or D.MISSING:
        raise SystemExit(f"ResNet-18 result files and transcription disagree: {bad}, runs {D.n_runs()}, missing {D.MISSING}")


def build(r18_results, mb_log, mb_results, vgg_log, out, vgg_results=None):
    # resolve every input before changing directory into the output folder
    out = pathlib.Path(out).resolve()
    mb_log, mb_results = pathlib.Path(mb_log).resolve(), pathlib.Path(mb_results).resolve()
    vgg_log = pathlib.Path(vgg_log).resolve()
    r18_results = [pathlib.Path(r).resolve() for r in r18_results]
    D, _ = generate("resnet18", r18_results, out)
    if not FIXTURE:
        check_r18_transcription(D)
    vgg_results = pathlib.Path(vgg_results).resolve() if vgg_results else None
    cwd = os.getcwd()
    try:
        os.chdir(out)
        import mbv2_section
        mbv2_section.main(str(mb_log), str(mb_results), str(vgg_log), vgg_results, None if FIXTURE else D)
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
    ap.add_argument("--mb-results", required=True, help="directory with the 132 MobileNetV2 result files")
    ap.add_argument("--vgg-log", required=True, help="VGG-16 train.log (all 132 runs)")
    ap.add_argument("--vgg-results", help="directory with the 132 VGG-16 result files (routing entropies)")
    ap.add_argument("--out", default=str(HERE / "out" / "paper2"))
    a = ap.parse_args()
    FIXTURE = bool(a.r18_fixture)
    res = a.r18_results
    if a.r18_fixture:
        import make_r18_jsons
        tmp = pathlib.Path(tempfile.mkdtemp()); make_r18_jsons.write(tmp); res = [tmp]
    pdf, errs, und, missing = build(res, a.mb_log, a.mb_results, a.vgg_log, a.out, a.vgg_results)
    print(f"{pdf}: errors {len(errs)}, undefined refs {len(und)}, undefined macros {missing}")
    for e in errs[:10]: print("  ", e)
    for u in und[:10]: print("  ", u)

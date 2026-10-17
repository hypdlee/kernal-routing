"""Analysis checks (gates D1-D3).

    python tests/test_analysis.py resnet18 | real | checkgrid | loader | synthetic | all

resnet18   the JSON-driven generators, fed the 132 ResNet-18 results as
           result files, reproduce the final paper's generators on the
           paper's data: every table, every macro and every figure, byte for
           byte (figures compared without their creation date). The only
           permitted differences: parameter counts (the paper carried six
           unmeasured third decimals; the counted values are used on both
           sides) and the matched-budget table's new ratio column.
real       the 24 real ResNet-18 result files the user uploaded
           (results.zip, width-64 dynamic arms) load to the paper's values.
"""
import sys, pathlib, json, tempfile, shutil, subprocess, re, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "report")); sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import numpy as np
from gen import generate
import make_r18_jsons

PAPER_GEN = ROOT / "tests" / "fixtures" / "paper_generators"
REAL_ZIP = pathlib.Path("/mnt/user-data/uploads/results.zip")


def counted_patch(D):
    """Paper data module with parameter counts replaced by the counted ones."""
    C = make_r18_jsons.counted()
    for key, arms in D.PARAMS.items():
        for a in arms:
            arms[a] = C[(key[0], key[1], key[2], a)][0] / 1e6
    for (ds, w) in list(D.ROUTER_K):
        D.ROUTER_K[(ds, w)] = C[(ds, w, 4, "gap")][1] / 1e3


def fresh_data():
    """The report's data module (not the paper's, which the reference run
    may have left in sys.modules under the same name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("report_data", ROOT / "report" / "data.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def strip_pdf(b):
    return re.sub(rb"/CreationDate \([^)]*\)", b"", b)


def test_resnet18():
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        make_r18_jsons.write(t / "json")
        generate("resnet18", [t / "json"], t / "new")
        generate("resnet18", [], t / "ref", paper_reference=PAPER_GEN, patch=counted_patch)
        tabs = sorted(p.name for p in (t / "ref").glob("tab_*.tex"))
        assert len(tabs) == 13, tabs
        same = 0
        for name in tabs:
            a = (t / "ref" / name).read_text(); b = (t / "new" / name).read_text()
            if name == "tab_MATCHED.tex":
                b = re.sub(r" & [0-9.]+ \((?:not )?matched\) \\\\", r" \\\\", b)
            if a != b:
                for la, lb in zip(a.splitlines(), b.splitlines()):
                    if la != lb:
                        print("  REF", la); print("  NEW", lb); break
                raise AssertionError(name)
            same += 1
        ma = (t / "ref" / "macros.tex").read_text(); mb = (t / "new" / "macros.tex").read_text()
        assert ma == mb, "macros differ"
        nmac = ma.count("\\expandafter")
        figs = sorted(p.name for p in (t / "ref" / "fig").glob("*.pdf"))
        assert len(figs) == 10
        fsame = 0
        for fn in figs:
            ref = (t / "ref" / "fig" / fn).read_bytes(); new = (t / "new" / "fig" / fn).read_bytes()
            # the reference generators leave the creation date in; strip it on both sides
            if strip_pdf(ref) != strip_pdf(new):
                raise AssertionError(f"figure {fn} differs")
            fsame += 1
        # spot values the paper states in its text
        m = dict(re.findall(r"\\csname v(\w+)\\endcsname\{([^}]*)\}", mb))
        for k, v in dict(nruns="132", gpuh="204", effTB="+2.282", mMA="+4.345", mMB="+2.313",
                         mMC="+2.520", flA="14").items():
            assert m[k] == v, (k, m[k], v)
        print(f"  {same} of 13 tables identical, {nmac} of {nmac} macros identical, {fsame} of 10 figures identical")
        print(f"  paper values: runs {m['nruns']}, GPU-h {m['gpuh']}, TIN w64 gap-static {m['effTB']}, "
              f"matched {m['mMA']} / {m['mMB']} / {m['mMC']}, entropy counterexample {m['flA']}x")
    print("ok")


def test_real():
    if not REAL_ZIP.exists():
        print("  results.zip not present; skipped"); return
    D = fresh_data()
    import paper_r18_data as P
    with tempfile.TemporaryDirectory() as t:
        zipfile.ZipFile(REAL_ZIP).extractall(t)
        errs = D.load("resnet18", [t], strict=False)
        assert not [e for e in errs if "mismatch" in e or "two results" in e or "expected" in e], errs
        key = ("cifar100", 64, 4, "const")
        n = 0
        for arm in ("uniform", "gap", "shuffled", "frozen_random"):
            got = np.asarray(D.ACC[key][arm]); ref = P.ACC[key][arm]
            assert np.allclose(got, ref, atol=5e-4), (arm, got, ref)
            eg = np.asarray(D.ENT_W64[arm]); er = P.ENT_W64[arm]
            assert np.allclose(eg, er, atol=6e-5), (arm, eg, er)
            n += 12
        assert abs(D.PARAMS[key]["gap"] - 44.4645) < 1e-4 and abs(D.ROUTER_K[("cifar100", 64)] - 287.952) < 1e-6
        print(f"  24 real files: {n // 2} accuracies and {n // 2} per-seed routing entropies equal the paper's; "
              f"gap params {D.PARAMS[key]['gap']:.4f}M, router {D.ROUTER_K[('cifar100', 64)]:.3f}K counted")
    print("ok")


def test_checkgrid():
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        base = {"tag": "t", "router": "gap", "final_acc": 70.0, "config": {
            "epochs": 150, "batch_size": 128, "lr": 0.1, "weight_decay": 5e-4, "warmup": 5,
            "num_kernels": 4, "temperature": 1.0, "scope": "all"}}
        a = json.loads(json.dumps(base)); a["seed"] = 0                      # legacy: no model key
        b = json.loads(json.dumps(base)); b["seed"] = 1; b["config"]["model"] = "vgg16"
        (t / "a.json").write_text(json.dumps(a)); (t / "b.json").write_text(json.dumps(b))
        r = subprocess.run([sys.executable, str(ROOT / "check_grid.py"), "--dir", str(t), "--tag", "t",
                            "--seeds", "0", "1", "--routers", "gap"], capture_output=True, text=True)
        assert "CRITICAL MISMATCH  model" in r.stdout, r.stdout[-1500:]
        # same model, one legacy file and one explicit resnet18: no mismatch
        b["config"]["model"] = "resnet18"; (t / "b.json").write_text(json.dumps(b))
        r = subprocess.run([sys.executable, str(ROOT / "check_grid.py"), "--dir", str(t), "--tag", "t",
                            "--seeds", "0", "1", "--routers", "gap"], capture_output=True, text=True)
        assert "CRITICAL MISMATCH" not in r.stdout, r.stdout[-1500:]
        print("  resnet18 (legacy) vs vgg16: CRITICAL MISMATCH model; legacy vs explicit resnet18: consistent")
    print("ok")


def test_loader():
    """The analysis refuses: duplicate cells, recipe drift, wrong parameter
    counts. It counts missing and diverged runs instead of hiding them."""
    D = fresh_data()
    from synth import make
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        make("vgg16", t / "a", drop=2, diverge=1)
        errs = D.load("vgg16", [t / "a"])
        assert not errs and len(D.MISSING) == 3 and len(D.DIVERGED) == 1 and D.n_runs() == 129
        # duplicate: the same cell under another file name
        f = sorted((t / "a").glob("*_seed0.json"))[0]
        shutil.copy(f, t / "a" / ("copy_" + f.name))
        assert any("two results" in e for e in D.load("vgg16", [t / "a"], strict=False))
        (t / "a" / ("copy_" + f.name)).unlink()
        # recipe drift
        r = json.loads(f.read_text()); r["config"]["lr"] = 0.05; f.write_text(json.dumps(r))
        assert any("recipe mismatch in lr" in e for e in D.load("vgg16", [t / "a"], strict=False))
        # a 1-epoch probe of the same cell is ignored, not a duplicate
        r["config"]["lr"] = 0.1; f.write_text(json.dumps(r))
        p1 = json.loads(json.dumps(r)); p1["config"]["epochs"] = 1
        (t / "a" / ("probe_" + f.name)).write_text(json.dumps(p1))
        assert not D.load("vgg16", [t / "a"], strict=False) and any("epochs 1" in x for x in D.IGNORED)
        (t / "a" / ("probe_" + f.name)).unlink()
        r["params"]["total"] += 1; f.write_text(json.dumps(r))
        assert any("expected" in e for e in D.load("vgg16", [t / "a"], strict=False))
        try:
            D.load("vgg16", [t / "a"]); raise AssertionError("strict load accepted a bad file")
        except SystemExit:
            pass
        print("  missing 2 + diverged 1 counted (129 runs); duplicate, recipe drift, wrong params all refused; a 1-epoch probe of a cell is ignored and listed")
    print("ok")


def test_synthetic():
    from synth import make
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        for m in ("mobilenetv2", "vgg16", "resnet50", "convnext_tiny"):
            make(m, t / "json" / m, seed=1, drop=1, diverge=1)
            D, errs = generate(m, [t / "json" / m], t / "out" / m, synthetic=True)
            names = {p.stem[4:] for p in (t / "out" / m).glob("tab_*.tex")}
            nt = len(names - {"INTERP", "COVER"}); nf = len(list((t / "out" / m / "fig").glob("*.pdf")))
            # the paper's 13 tables, plus the per-model coverage and equal-parameter tables
            assert nt == 13 and {"INTERP", "COVER"} <= names and nf == 10 and not errs, (m, names, nf, errs)
            txt = (t / "out" / m / "tab_ALLRUNS.tex").read_text()
            assert "--" in txt           # the missing run shows as --, not as a number
            print(f"  {m:<14} 13 tables (+ coverage, equal-parameter), 10 figures, {D.n_runs()} runs, missing {len(D.MISSING)}, diverged {len(D.DIVERGED)}")
    print("4 models ok")


TESTS = dict(resnet18=test_resnet18, real=test_real, checkgrid=test_checkgrid, loader=test_loader,
             synthetic=test_synthetic)
if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for w in (TESTS if which == ["all"] else which):
        print(f"== {w}")
        TESTS[w]()

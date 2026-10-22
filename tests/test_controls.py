"""Checks for the routing-control extension (run_controls.py, --frozen-seed).

    python tests/test_controls.py

1. Without --frozen-seed the frozen router is exactly the one every earlier
   run used (weights drawn from seed 20260908), so no old result changes.
2. --frozen-seed gives a different router and is recorded in the result.
3. The runner lists the intended jobs: part A two new draws at CIFAR-100
   width 64, part B both controls at width 16, part C (VGG-16) and part D
   (MobileNetV2) both controls on Tiny-ImageNet width 32; six seeds each;
   no two jobs share a result file.
"""
import math, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import src.routers as R
import run_controls as RC


def test_default_router_unchanged():
    assert R.FROZEN_SEED == 20260908
    r = R.FrozenRandomRouter(64, 4)
    g = torch.Generator().manual_seed(20260908)
    w1 = torch.randn(64, 16, generator=g) / math.sqrt(64)
    assert torch.equal(r.w1, w1)
    old = R.FROZEN_SEED
    try:
        R.FROZEN_SEED = 1
        assert not torch.equal(R.FrozenRandomRouter(64, 4).w1, w1)
    finally:
        R.FROZEN_SEED = old
    src = (ROOT / "src" / "train.py").read_text()
    assert "if args.frozen_seed is not None:" in src and '"--frozen-seed", type=int, default=None' in src
    print("ok   default frozen router identical to earlier runs; --frozen-seed changes it")


def test_jobs():
    exp = {("mobilenetv2", "A"): 12, ("mobilenetv2", "B"): 12, ("mobilenetv2", "C"): 0, ("mobilenetv2", "D"): 12,
           ("vgg16", "A"): 12, ("vgg16", "B"): 12, ("vgg16", "C"): 12, ("vgg16", "D"): 0}
    for (m, p), n in exp.items():
        assert len(RC.jobs(m, [p])) == n, (m, p)
    for m in RC.MODELS:
        J = RC.jobs(m, ["A", "B", "C", "D"])
        names = [RC.result_path(pathlib.Path("x"), m, g, arm, d, s).name for _, g, arm, _, d, s in J]
        assert len(names) == len(set(names)), "two jobs share a result file"
        for p, g, arm, fs, d, s in J:
            assert (fs is None) == (p != "A") and (d == "d1") == (p != "A")
            if p == "A":
                assert (g.dataset, g.width, arm) == ("cifar100", 64, "frozen_random") and fs in (1, 2)
    print("ok   jobs: A 12+12, B 12+12, C 12 (VGG-16), D 12 (MobileNetV2), unique result files")


def test_dry_run(tmp="results_controls_test_dry"):
    out = subprocess.run([sys.executable, "run_controls.py", "--dry-run", "--parts", "A", "B", "C", "--out",
                          str(ROOT / tmp)], cwd=ROOT, capture_output=True, text=True).stdout
    cmds = [l for l in out.splitlines() if "src.train" in l]
    assert len(cmds) == 60, len(cmds)
    assert sum("--frozen-seed" in c for c in cmds) == 24
    assert all("--epochs 150" in c and "--batch-size 128" in c and "--amp bf16" in c for c in cmds)
    import shutil; shutil.rmtree(ROOT / tmp, ignore_errors=True)
    print("ok   dry run: 60 commands for parts A-C, 24 with --frozen-seed, recipe unchanged")


if __name__ == "__main__":
    test_default_router_unchanged(); test_jobs(); test_dry_run()

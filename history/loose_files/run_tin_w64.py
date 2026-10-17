"""
Stage 4: Tiny-ImageNet at width 64.

Closes the last coverage gap. The inverted-U across widths, and with it
the saturation account of the width-64 null, is currently a CIFAR-100
result. Tiny-ImageNet has only been run at width 32. This adds the three
arms at width 64 on Tiny-ImageNet, which is where the mechanism makes its
clearest prediction: at width 64 the network should be close to
memorising Tiny-ImageNet, and the routing effect should shrink toward
nothing, as it did on CIFAR-100.

A clean null here confirms the mechanism on a second dataset. A surviving
effect means the saturation account is CIFAR-specific, and that is the
more interesting outcome and must be reported as such.

COST. Estimated from measured times, scaling the Tiny-ImageNet width-32
runs by the measured CIFAR width-32 to width-64 ratio of 2.19:

    static   ~2.0 h/run   x6 = 12 h
    uniform  ~2.1 h/run   x6 = 12 h
    gap     ~10.4 h/run   x6 = 62 h
    TOTAL   ~87 h = 3.6 days

MEMORY. At 64x64 the blend path stacks K outputs at 64x64 in stage 1:
268 MB per dynamic layer against 67 MB on CIFAR, and autograd holds them.
Roughly 2 GB of stacked outputs across the network before ordinary
activations. It should fit in 12 GB but there is less headroom than
anywhere else in this program. Run --probe first.

ORDER. Seed-major, unlike every earlier stage. If this is interrupted you
are left with complete seed triplets that still support a paired analysis
at reduced n, rather than one finished arm with nothing to pair against.

    python run_tin_w64.py --probe     # 1 epoch, gap, seed 0: memory + timing
    python run_tin_w64.py --dry-run
    python run_tin_w64.py
"""

from __future__ import annotations

import argparse, subprocess, sys, time
from pathlib import Path

ARMS = [("static", "none"), ("uniform", "all"), ("gap", "all")]
TAG = "tin_w64"


def build(arm, scope, seed, a, epochs=None, tag=None):
    return [sys.executable, "-m", "src.train",
            "--dataset", "tinyimagenet", "--data-root", a.data_root,
            "--router", arm, "--scope", scope, "--width", "64",
            "--num-kernels", "4", "--seed", str(seed),
            "--epochs", str(epochs or a.epochs),
            "--batch-size", str(a.batch_size), "--impl", "blend",
            "--amp", "bf16", "--workers", str(a.workers),
            "--out", a.out, "--tag", tag or TAG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="one epoch of the most expensive arm, to check memory "
                         "and measure the real per-epoch time before committing")
    a = ap.parse_args()

    base = Path(a.data_root) / "tiny-imagenet-200"
    if not (base / "train").is_dir() or (base / "val" / "images").is_dir():
        print(f"Tiny-ImageNet not prepared at {base}. Run download_tinyimagenet.py first.")
        return 1

    if a.probe:
        print("PROBE: gap, width 64, 1 epoch. Watch for OOM and note the epoch time.")
        print("Multiply the epoch time by 150 for one gap run, then by 6.\n")
        rc = subprocess.call(build("gap", "all", 0, a, epochs=1, tag="probe_tin_w64")
                             + ["--out", "results/probe"])
        if rc != 0:
            print("\nPROBE FAILED. If this was an out-of-memory error, drop to")
            print("--batch-size 64 AND halve the learning rate, and apply that to all")
            print("three arms. Note in PREREGISTRATION.md that the width-64")
            print("Tiny-ImageNet arms use a different batch size from width 32, so")
            print("the two widths are not directly comparable on that dataset.")
        else:
            print("\nPROBE OK. Delete results/probe, then run without --probe.")
        return rc

    out = Path(a.out)
    jobs = [(arm, scope, s) for s in a.seeds for arm, scope in ARMS
            if not (out / f"{TAG}_{arm}_seed{s}.json").exists()]
    total = len(a.seeds) * len(ARMS)
    print(f"{total} runs, {total-len(jobs)} already done, {len(jobs)} to go.")
    print("Estimated ~87 h total (static ~2.0 h, uniform ~2.1 h, gap ~10.4 h per run).")
    print("Seed-major order: each seed completes all three arms before the next.")
    if not jobs:
        print("nothing to do. run: python analyze_capacity.py --dataset tinyimagenet")
        return 0

    t0 = time.time()
    for i, (arm, scope, seed) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {arm} seed {seed}   "
              f"(elapsed {(time.time()-t0)/3600:.1f} h)", flush=True)
        if a.dry_run:
            continue
        rc = subprocess.call(build(arm, scope, seed, a))
        if rc != 0:
            print(f"run failed (rc={rc}); stopping.")
            return rc

    print(f"\ndone in {(time.time()-t0)/3600:.1f} h")
    print(f"\nnow run:  python check_grid.py --dir results --tag {TAG} "
          f"--routers static uniform gap --seeds 0 1 2 3 4 5")
    print( "     then: python analyze_capacity.py --dataset tinyimagenet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

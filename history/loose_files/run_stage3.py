"""
Stage 3: close the two gaps a reviewer would reject on.

GAP 1 -- the decomposition is CIFAR-100 only.
Tiny-ImageNet was run with static and gap only, so the +2.185pp there is a
total effect that cannot be split. The paper's central claim (the bank
contributes nothing, routing does all of it) therefore rests on one
dataset. Adding uniform at Tiny-ImageNet width 32 fixes that with six
runs. This is the single highest-value compute left.

GAP 2 -- fidelity to the published method.
Every run so far used a fixed softmax temperature tau = 1. DY-Conv (Chen
et al., CVPR 2020) anneals tau from 30 to 1 over the first ten epochs and
reports this as necessary, not optional: at tau = 1 the mixture is close
to one-hot from the first step and most of the bank is starved of
gradient. A reviewer who knows that paper will ask whether the null at
width 64 is a property of dynamic convolution or of an under-trained
bank. The ablation settles it.

    python run_stage3.py --part uniform    # gap 1, 6 runs, ~30 h
    python run_stage3.py --part temp       # gap 2, 12 runs, ~9 h
    python run_stage3.py                   # both

Run --part temp first. It is cheaper and, if annealing changes the
width-64 result, it changes what the paper says.
"""

from __future__ import annotations

import argparse, subprocess, sys, time
from pathlib import Path


def tin_uniform(a):
    base = Path(a.data_root) / "tiny-imagenet-200"
    if not (base / "train").is_dir() or (base / "val" / "images").is_dir():
        print(f"Tiny-ImageNet not prepared at {base}. Run download_tinyimagenet.py first.")
        return None
    jobs = []
    for s in a.seeds:
        if not (Path(a.out) / f"tin_w32_uniform_seed{s}.json").exists():
            jobs.append(["--dataset", "tinyimagenet", "--data-root", a.data_root,
                         "--router", "uniform", "--scope", "all", "--width", "32",
                         "--num-kernels", "4", "--seed", str(s), "--epochs", str(a.epochs),
                         "--tag", "tin_w32"])
    return jobs


def temp_ablation(a):
    """gap with annealed tau, at the two widths that decide the paper:
    64 (where everything was null) and 32 (where the effect peaks)."""
    jobs = []
    for w in [64, 32]:
        for s in a.seeds:
            tag = f"stage3temp_w{w}"
            if not (Path(a.out) / f"{tag}_gap_seed{s}.json").exists():
                jobs.append(["--dataset", "cifar100", "--router", "gap", "--scope", "all",
                             "--width", str(w), "--num-kernels", "4", "--seed", str(s),
                             "--epochs", str(a.epochs), "--temp-start", "30",
                             "--temp-end", "1", "--temp-epochs", "10", "--tag", tag])
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=["both", "uniform", "temp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs = []
    if a.part in ("both", "temp"):
        jobs += temp_ablation(a)
    if a.part in ("both", "uniform"):
        j = tin_uniform(a)
        if j is None:
            return 1
        jobs += j

    if not jobs:
        print("nothing to do. run: python analyze_stage3.py")
        return 0
    print(f"{len(jobs)} runs to go.")
    print("Estimate: CIFAR w64 ~86 min, CIFAR w32 ~39 min, Tiny-ImageNet w32 ~300 min per run.")

    t0 = time.time()
    for i, extra in enumerate(jobs, 1):
        cmd = [sys.executable, "-m", "src.train", "--batch-size", "128",
               "--impl", "blend", "--amp", "bf16", "--workers", str(a.workers),
               "--out", a.out] + extra
        label = " ".join(extra[extra.index("--router") + 1:])
        print(f"\n=== [{i}/{len(jobs)}] {label}", flush=True)
        if a.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"run failed (rc={rc}); stopping.")
            return rc

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print("\nnow run:  python analyze_stage3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Stage 2a: does the width-32 peak depend on K?

The CIFAR-100 sweep found the routing effect peaking at width 32 with
K=4. That peak is a single observation, and the bank size interacts with
capacity by construction, so "does it move with K" is the first thing a
careful reader asks.

Runs gap at K in {2, 8} at width 32. K=4 and the static reference are
already on disk from stage1c_w32 and are reused, not re-run.

Each result is tagged stage2k_K<k> so check_grid.py's config test stays
valid: num_kernels is in its CRITICAL list and varies here by design.

Cost: 12 runs. K=2 is cheaper than K=4 and K=8 dearer, so expect
somewhere near 8 hours in total on one RTX 4070 Ti.

    python run_ksweep.py
    python run_ksweep.py --ks 2          # half of it
"""

from __future__ import annotations

import argparse, subprocess, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--router", default="gap")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    jobs = []
    for k in args.ks:
        tag = f"stage2k_K{k}"
        for s in args.seeds:
            if not (out / f"{tag}_{args.router}_seed{s}.json").exists():
                jobs.append((k, tag, s))

    total = len(args.ks) * len(args.seeds)
    print(f"{total} runs, {total-len(jobs)} already done, {len(jobs)} to go.")
    print(f"K=4 and static at width {args.width} are reused from stage1c_w{args.width}.")
    if not jobs:
        print("nothing to do. run: python analyze_ksweep.py")
        return 0

    t0 = time.time()
    for i, (k, tag, seed) in enumerate(jobs, 1):
        cmd = [sys.executable, "-m", "src.train",
               "--router", args.router, "--scope", "all",
               "--width", str(args.width), "--num-kernels", str(k),
               "--seed", str(seed), "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--impl", "blend",
               "--amp", "bf16", "--workers", str(args.workers),
               "--out", args.out, "--tag", tag]
        print(f"\n=== [{i}/{len(jobs)}] K={k} seed {seed}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"run failed (rc={rc}); stopping so the sweep stays interpretable.")
            return rc

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print("\nnow run:  python analyze_ksweep.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

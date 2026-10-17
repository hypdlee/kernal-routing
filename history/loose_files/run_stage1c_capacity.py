"""
Stage 1c: does the null survive outside the saturated regime?

Stage 1 and 1b showed that at width 64 every arm, including a plain
ResNet-18, lands within a quarter of a point of every other. Every run
also finished at a training loss near 0.008, meaning all five models
memorised CIFAR-100 completely. A reviewer will ask whether the null is a
property of dynamic convolution or of the saturated regime, and the
question is fair.

This replicates three arms at reduced width. Combined with the width-64
results you already have, that gives a three-point capacity sweep:

    width 16   static 0.71M   uniform 2.77M   gap 2.79M
    width 32   static 2.82M   uniform 11.06M  gap 11.13M
    width 64   static 11.22M  uniform 44.18M  gap 44.46M   (already run)

Two readings, both publishable:

  the gap stays at zero across all three widths
      the null is about dynamic convolution, not about capacity. The
      strongest version of the Stage 1 claim.

  the gap opens as width shrinks
      dynamic convolution helps only when the network is starved. That is
      a sharper and more useful finding than either "it works" or "it does
      not", and it explains why the published gains exist at all.

Three arms rather than five: shuffled and frozen_random already did their
job at width 64, which was to locate which component of routing was inert.
Repeating them here would add runs without adding an answer.

Note the accidental control this creates: gap at width 32 stores 11.13M
parameters and static at width 64 stores 11.22M. Those two are matched on
stored parameters to within one percent.

    python run_stage1c_capacity.py              # widths 16 and 32
    python run_stage1c_capacity.py --widths 32  # just the cheaper half
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ARMS = [("static", "none"), ("uniform", "all"), ("gap", "all")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-kernels", type=int, default=4)
    ap.add_argument("--impl", default="blend")
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    # One tag per width. Keeps check_grid.py's config-consistency test valid,
    # since width is exactly the thing that varies here and it would
    # otherwise be flagged as a critical mismatch.
    jobs = []
    for w in args.widths:
        tag = f"stage1c_w{w}"
        for router, scope in ARMS:
            for s in args.seeds:
                if not (out / f"{tag}_{router}_seed{s}.json").exists():
                    jobs.append((w, tag, router, scope, s))

    total = len(args.widths) * len(ARMS) * len(args.seeds)
    print(f"{total} runs in the sweep, {total - len(jobs)} already done, {len(jobs)} to go.")
    if not jobs:
        print("nothing to do. run: python analyze_capacity.py")
        return 0

    t0 = time.time()
    for i, (w, tag, router, scope, seed) in enumerate(jobs, 1):
        cmd = [sys.executable, "-m", "src.train",
               "--router", router, "--scope", scope, "--width", str(w),
               "--seed", str(seed), "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size),
               "--num-kernels", str(args.num_kernels), "--impl", args.impl,
               "--amp", args.amp, "--workers", str(args.workers),
               "--out", args.out, "--tag", tag]
        print(f"\n=== [{i}/{len(jobs)}] w{w} {router} seed {seed}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"run failed (rc={rc}); stopping so the sweep stays interpretable.")
            return rc

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print("\nnow run:  python analyze_capacity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

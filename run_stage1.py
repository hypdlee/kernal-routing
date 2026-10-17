"""
Stage 1 orchestrator: every (router, seed) pair, sequentially, resumable.

Skips any run whose JSON already exists, so killing this and restarting is
safe. Ctrl-C between runs loses nothing.

    python run_stage1.py                      # full grid, 6 seeds
    python run_stage1.py --seeds 0 1 2        # fewer seeds (see the
                                              # Wilcoxon floor warning)
    python run_stage1.py --routers uniform gap --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ROUTERS = ["uniform", "gap", "shuffled", "frozen_random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", nargs="+", default=DEFAULT_ROUTERS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-kernels", type=int, default=4)
    ap.add_argument("--scope", default="all")
    ap.add_argument("--impl", default="blend")
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake-data", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    jobs = [(r, s) for r in args.routers for s in args.seeds]
    todo = [(r, s) for r, s in jobs
            if not (outdir / f"{args.tag}_{r}_seed{s}.json").exists()]

    print(f"{len(jobs)} runs in the grid, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to go.")
    if len(args.seeds) < 6:
        print(f"note: {len(args.seeds)} seeds. Two-sided Wilcoxon cannot reach "
              f"p<0.05 below 6 seeds; the t-test and TOST still work.")

    t0 = time.time()
    for i, (router, seed) in enumerate(todo, 1):
        cmd = [sys.executable, "-m", "src.train",
               "--router", router, "--seed", str(seed),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--num-kernels", str(args.num_kernels), "--scope", args.scope,
               "--impl", args.impl, "--amp", args.amp,
               "--workers", str(args.workers), "--out", args.out, "--tag", args.tag]
        if args.fake_data:
            cmd.append("--fake-data")
        print(f"\n=== [{i}/{len(todo)}] {' '.join(cmd[2:])}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"run failed (rc={rc}); stopping so the grid stays interpretable.")
            return rc

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    if not args.dry_run:
        print("\nnow run:  python -m src.analyze --dir %s --tag %s" % (args.out, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

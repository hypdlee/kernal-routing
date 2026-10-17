"""
Stage 2b: does the CIFAR-100 finding replicate on a second dataset?

Width 32 only, because that is where the routing effect peaked on
CIFAR-100, so a failure to replicate there is the most informative
failure available.

Two arms, not three. static and gap. uniform is skipped deliberately: it
was equivalent to static at all three CIFAR widths, so repeating it on a
second dataset costs six runs and answers nothing. If you want it back,
pass --arms static uniform gap.

COST WARNING. Tiny-ImageNet is 100k images at 64x64 against CIFAR's 50k
at 32x32, so roughly 8x the pixels per epoch. The stem is deliberately
unchanged, which leaves the final feature map at 8x8 rather than 4x4 and
costs about 4x more again in the deeper stages. Expect on the order of
5 hours per gap run and 1.5 per static run on one RTX 4070 Ti, so the
12-run default is three to four days of wall clock.

Decide --epochs BEFORE starting. Runs at different epoch counts cannot be
paired, so changing it halfway means discarding what is finished.

    python download_tinyimagenet.py --root ./data     # once
    python run_tinyimagenet.py --dry-run              # check the plan
    python run_tinyimagenet.py
    python run_tinyimagenet.py --epochs 100           # cheaper, still valid
"""

from __future__ import annotations

import argparse, subprocess, sys, time
from pathlib import Path

SCOPE = {"static": "none", "uniform": "all", "gap": "all"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["static", "gap"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-kernels", type=int, default=4)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = Path(args.data_root) / "tiny-imagenet-200"
    if not (base / "train").is_dir() or (base / "val" / "images").is_dir():
        print(f"Tiny-ImageNet is not prepared at {base}.")
        print(f"Run:  python download_tinyimagenet.py --root {args.data_root}")
        return 1

    tag = f"tin_w{args.width}"
    out = Path(args.out)
    jobs = [(a, s) for a in args.arms for s in args.seeds
            if not (out / f"{tag}_{a}_seed{s}.json").exists()]
    total = len(args.arms) * len(args.seeds)
    print(f"{total} runs, {total-len(jobs)} already done, {len(jobs)} to go.")
    print(f"arms={args.arms}  width={args.width}  epochs={args.epochs}  tag={tag}")
    print("Estimate: ~5 h per gap run, ~1.5 h per static run on a 4070 Ti.")
    if not jobs:
        print("nothing to do.")
        return 0

    t0 = time.time()
    for i, (arm, seed) in enumerate(jobs, 1):
        router = "static" if arm == "static" else arm
        cmd = [sys.executable, "-m", "src.train",
               "--dataset", "tinyimagenet", "--data-root", args.data_root,
               "--router", router, "--scope", SCOPE[arm],
               "--width", str(args.width), "--num-kernels", str(args.num_kernels),
               "--seed", str(seed), "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--impl", "blend",
               "--amp", "bf16", "--workers", str(args.workers),
               "--out", args.out, "--tag", tag]
        print(f"\n=== [{i}/{len(jobs)}] {arm} seed {seed}", flush=True)
        if args.dry_run:
            print("    " + " ".join(cmd[2:]))
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"run failed (rc={rc}); stopping.")
            return rc

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print(f"\nnow run:  python check_grid.py --dir results --tag {tag} "
          f"--routers {' '.join(args.arms)} --seeds {' '.join(map(str,args.seeds))}")
    print(f"     then: python analyze_capacity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

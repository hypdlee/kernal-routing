"""
Stage 1b: the static baseline the original grid is missing.

Stage 1 established that uniform, gap, shuffled and frozen_random are all
within a quarter of a point of each other. That says routing does not
matter. It does NOT say what the kernel bank is doing, because every arm
in the grid has one.

This adds the missing reference: a plain ResNet-18 with no dynamic
convolution anywhere (--scope none, 11.22M parameters).

It matters more than it looks. The uniform arm routes with z = 1/K for
every input, so its four kernels collapse to their mean and it computes
exactly what a single static kernel computes. Stored 44.18M, effective
11.22M. Whatever separates uniform from static is therefore an
optimisation effect of linear over-parameterisation, not extra
representational capacity.

Three outcomes, three different papers:

  static ~= uniform      the kernel bank does nothing at all. Dynamic
                         convolution on this benchmark is inert.
  static <  uniform      the gain is the re-parameterisation, and it is
                         reproducible with a static network at inference
                         cost of 11.22M. This is the strongest version.
  static >  uniform      the bank actively hurts. Report it.

Six runs, cheaper than the others since there is no bank to blend.

    python run_stage1b_static.py
"""

import subprocess, sys, time
from pathlib import Path

SEEDS = [0, 1, 2, 3, 4, 5]
TAG = "stage1b"

def main():
    out = Path("results")
    todo = [s for s in SEEDS if not (out / f"{TAG}_static_seed{s}.json").exists()]
    print(f"{len(SEEDS)} runs, {len(SEEDS)-len(todo)} already done, {len(todo)} to go.")
    t0 = time.time()
    for i, seed in enumerate(todo, 1):
        # --router is ignored when --scope none (no dynamic layers exist),
        # but train.py records it, so name it 'static' for clean grouping.
        cmd = [sys.executable, "-m", "src.train",
               "--router", "gap", "--scope", "none", "--seed", str(seed),
               "--epochs", "150", "--batch-size", "128", "--num-kernels", "4",
               "--impl", "blend", "--amp", "bf16", "--workers", "4",
               "--out", "results", "--tag", TAG]
        print(f"\n=== [{i}/{len(todo)}] static seed {seed}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"run failed (rc={rc}); stopping.")
            return rc
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print("\nnow run:  python compare_static.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

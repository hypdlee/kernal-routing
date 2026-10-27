"""
Routing controls beyond the main program, for MobileNetV2 and VGG-16.

The main program ran shuffled and frozen_random on CIFAR-100 at width 64
only (six seeds, one frozen-router draw). This runner adds:

  A  two more frozen-router draws at CIFAR-100 width 64     (frozen_random)
  B  both controls at CIFAR-100 width 16                    (shuffled, frozen_random)
  C  both controls on Tiny-ImageNet at width 32, VGG-16     (shuffled, frozen_random)
  D  both controls on Tiny-ImageNet at width 32, MobileNetV2 (optional; about 74 h)
  E  both controls at CIFAR-100 width 32, ResNet-18 only

ResNet-18 (the original study, recorded under the older file names) is opt-in:
parts A, B, C and E apply to it, and a plain run does not start it:

    python run_controls.py --models resnet18 --parts B C E      # about 74 GPU-h on the RTX 4070 Ti

Every new run uses the recipe, code path, seeds 0-5 and micro-batch rules of
run_all_models.py, so it pairs by seed with the static, uniform and gap runs
already in results_models/. Draw d1 is the frozen router every earlier run
used (seed 20260908); draws d2 and d3 use --frozen-seed 1 and 2.

    python run_controls.py --dry-run                  # every job, nothing executed
    python run_controls.py                            # parts A, B, C for both models
    python run_controls.py --parts A                  # one part
    python run_controls.py --parts D                  # the expensive optional part
    python run_controls.py --status

Results go to results_controls/<model>/. A finished result is skipped, so the
same command resumes; an interrupted run resumes from its checkpoint.
Write and commit PREREGISTRATION_CONTROLS.md before the first run finishes.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

from src.program import Group, SEEDS
from run_all_models import (arm_flags, choose_accum, default_exec, log, tin_ready)

MODELS = ["mobilenetv2", "vgg16", "resnet18"]
DEFAULT_MODELS = ["mobilenetv2", "vgg16"]   # ResNet-18 is opt-in (--models resnet18)
DRAWS = {"d2": 1, "d3": 2}           # d1 is the original router, FROZEN_SEED = 20260908


def parts_for(model):
    """(part, group, arm, frozen seed or None, draw label) in run order."""
    P = {
        "A": [(Group("cifar100", 64, 4, "const", ("frozen_random",)), "frozen_random", fs, d) for d, fs in DRAWS.items()],
        "B": [(Group("cifar100", 16, 4, "const", ("shuffled", "frozen_random")), a, None, "d1")
              for a in ("shuffled", "frozen_random")],
        "C": [(Group("tinyimagenet", 32, 4, "const", ("shuffled", "frozen_random")), a, None, "d1")
              for a in ("shuffled", "frozen_random")] if model in ("vgg16", "resnet18") else [],
        "D": [(Group("tinyimagenet", 32, 4, "const", ("shuffled", "frozen_random")), a, None, "d1")
              for a in ("shuffled", "frozen_random")] if model == "mobilenetv2" else [],
        "E": [(Group("cifar100", 32, 4, "const", ("shuffled", "frozen_random")), a, None, "d1")
              for a in ("shuffled", "frozen_random")] if model == "resnet18" else [],
    }
    return P


def ctag(model, g, draw):
    t = f"{model}__{g.dataset}__w{g.width}__K{g.K}__{g.recipe}"
    return t if draw == "d1" else f"{t}__{draw}"


def jobs(model, parts, seeds=SEEDS):
    """Seed-major inside each part, so an interruption leaves complete seed sets."""
    out = []
    P = parts_for(model)
    for p in parts:
        cells = P[p]
        for s in seeds:
            for g, arm, fs, draw in cells:
                out.append((p, g, arm, fs, draw, s))
    return out


def result_path(d: Path, model, g, arm, draw, seed):
    return d / f"{ctag(model, g, draw)}_{arm}_seed{seed}.json"


def train_cmd(a, model, g, arm, fs, draw, seed, accum, d: Path):
    fl = arm_flags(arm)
    res = result_path(d, model, g, arm, draw, seed)
    cmd = [sys.executable, "-m", "src.train",
           "--model", model, "--placement", "spatial",
           "--dataset", g.dataset, "--data-root", a.data_root,
           "--scope", fl["scope"], "--router", fl["router"], "--arm-name", arm,
           "--width", str(g.width), "--num-kernels", str(g.K),
           "--seed", str(seed), "--epochs", str(a.epochs),
           "--batch-size", "128", "--accum", str(accum),
           "--impl", "blend", "--amp", "bf16", "--workers", str(a.workers),
           "--out", str(d), "--tag", ctag(model, g, draw),
           "--ckpt", str(d / "ckpt" / res.name.replace(".json", ".pt"))]
    if fs is not None:
        cmd += ["--frozen-seed", str(fs)]
    if a.fake_data:
        cmd += ["--fake-data"]
    return cmd


def probing(execute):
    """Wrap an executor so a probe that fails for a reason other than running
    out of memory (exit 5) stops with that fact, instead of climbing the
    micro-batch ladder and ending in a false 'out of memory'. Typical cause on
    Windows: WinError 1114 while importing torch (page file / commit memory
    exhausted, or a broken torch install)."""
    def run(cmd, logfile=None):
        rc = execute(cmd, logfile)
        if logfile is None and rc not in (0, 5):
            try:
                out = Path(cmd[cmd.index("--probe-out") + 1])
            except (ValueError, IndexError):
                out = None
            if out is None or not out.exists():
                raise SystemExit(
                    f"The memory probe crashed (exit code {rc}) before it could measure anything. "
                    "This is not a GPU out-of-memory. Check that  python -c \"import torch; "
                    "print(torch.cuda.is_available())\"  works in this window. On WinError 1114 "
                    "(c10.dll) raise the Windows page file or close other torch jobs, then rerun. "
                    "To skip the probe use --skip-probe (batch 128 is then assumed to fit).")
        return rc
    return run


def run_model(a, model, execute=default_exec):
    execute = probing(execute)
    d = Path(a.out) / model
    d.mkdir(parents=True, exist_ok=True)
    plan = json.loads((d / "plan.json").read_text()) if (d / "plan.json").exists() else {}
    J = jobs(model, a.parts, a.seeds)
    if not J:
        return "nothing"
    for i, (p, g, arm, fs, draw, seed) in enumerate(J, 1):
        res = result_path(d, model, g, arm, draw, seed)
        marker = Path(str(res) + ".DIVERGED")
        if res.exists() or marker.exists():
            continue
        if g.dataset == "tinyimagenet" and not a.fake_data and not a.dry_run and not tin_ready(a.data_root):
            log(d, f"Tiny-ImageNet not prepared at {a.data_root}. Run: python download_tinyimagenet.py "
                   f"--root {a.data_root}   then rerun this command; it resumes here.")
            return "tin_missing"
        accum = choose_accum(a, model, g, d, execute, plan)
        cmd = train_cmd(a, model, g, arm, fs, draw, seed, accum, d)
        log(d, f"[{i}/{len(J)}] part {p} {ctag(model, g, draw)} {arm} seed {seed} (micro-batch {128 // accum})")
        if a.dry_run:
            print("   ", " ".join(cmd[1:]))
            continue
        t0 = time.time()
        rc = execute(cmd, d / "train.log")
        if rc == 3:
            marker.write_text(json.dumps({"diverged": True, "cmd": cmd, "time": time.ctime()}))
            log(d, f"DIVERGED {res.name}: non-finite training loss. Marker written; continuing.")
            continue
        if rc != 0 or not res.exists():
            log(d, f"run failed (rc={rc}); stopping. Rerun the same command to resume "
                   f"(out of memory: rerun; the group's micro-batch is in plan.json).")
            return "stopped"
        log(d, f"   finished in {(time.time() - t0) / 60:.1f} min, final {json.loads(res.read_text())['final_acc']:.2f}")
    if a.dry_run:
        print(f"DRY RUN {model}: {len(J)} commands listed, nothing executed")
        return "dry_run"
    return "done"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=MODELS)
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                    help="only these seeds (to split the work over two processes or two PCs)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="results_controls")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe-steps", type=int, default=20)
    ap.add_argument("--skip-probe", action="store_true", help="assume batch 128 fits (logged)")
    ap.add_argument("--fake-data", action="store_true", help="pipeline test only; results are meaningless")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        for m in a.models:
            d = Path(a.out) / m
            J = jobs(m, a.parts, a.seeds)
            done = sum(result_path(d, m, g, arm, draw, s).exists() for _, g, arm, _, draw, s in J)
            print(f"{m:<13} {done:>3} of {len(J)} done (parts {' '.join(a.parts)})")
        return 0
    out = {}
    for m in a.models:
        out[m] = run_model(a, m)
        if out[m] in ("stopped", "tin_missing"):
            break
    print(", ".join(f"{m}: {o}" for m, o in out.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

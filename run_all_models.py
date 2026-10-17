"""
The ResNet-18 program, 132 runs, replicated on four more backbones.

    python run_all_models.py --dry-run                 # print every job, run nothing
    python run_all_models.py                           # all four models, in order
    python run_all_models.py --models mobilenetv2      # one model
    python run_all_models.py --status                  # progress table, run nothing

Order: mobilenetv2 -> vgg16 -> resnet50 -> convnext_tiny (cheapest first).
Per model the 132 jobs come from src/program.py, the same groups, arms and
seeds as the ResNet-18 study, and the same recipe (SGD Nesterov 0.9, lr 0.1,
wd 5e-4, batch 128, 5 warm-up + cosine, 150 epochs, crop + flip, bf16).

GATE. Each model starts with one full static run (CIFAR-100, width 64,
seed 0). If its final accuracy is below the floor recorded in
PREREGISTRATION_MODELS.md the recipe does not train that backbone, its other
131 runs would measure nothing, and the runner stops that model, writes
GATE_FAILED.json and moves to the next model. The gate run is also the first
job of the program, so it costs no extra run.

MICRO-BATCH. Before a group's first job the runner probes it (a few steps of
the group's most memory-hungry arm) at batch 128. On out-of-memory it retries
at 64 x 2, then 32 x 4, then 16 x 8 (gradient accumulation: the same data
order, optimiser steps and learning-rate schedule; only BatchNorm statistics
are per micro-batch). The choice is made once per group, so every arm of a
paired comparison uses the same micro-batch, and it is saved to plan.json and
DEVIATIONS.md, and reported.

RESUME. A job whose result JSON exists is skipped, so the same command
continues where it stopped. A run interrupted mid-way resumes from its
per-epoch checkpoint (recorded in the result as resumed_at_epoch).

DIVERGENCE. A run whose training loss becomes non-finite writes a
DIVERGED marker, not a result, and the program continues. The analysis
counts and reports diverged runs; they are never silently dropped. Any other
failure stops the runner: it is more likely a broken environment than a
property of the model.
"""

from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

from src.program import PROGRAM, SEEDS, ANNEAL_ARGS, GATE_JOB, jobs, tag, result_name, arm_flags

RUN_ORDER = ["mobilenetv2", "vgg16", "resnet50", "convnext_tiny"]

# CIFAR-100 top-1 (%) the static width-64 seed-0 run must reach. These are
# floors for "the shared recipe trains this backbone", set well below
# published CIFAR-100 numbers for each architecture, not performance claims.
# Recorded before any run in PREREGISTRATION_MODELS.md.
GATE_FLOOR = {"mobilenetv2": 65.0, "vgg16": 65.0, "resnet50": 70.0, "convnext_tiny": 60.0,
              "resnet18": 70.0}

ACCUM_LADDER = (1, 2, 4, 8)
# Memory ranking of arms inside a group: the routed arms hold K stacked
# outputs and the router activations; static holds neither.
ARM_WEIGHT = {"shuffled": 4, "gap": 3, "frozen_random": 2, "uniform": 1, "static": 0}


def log(model_dir: Path, msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(model_dir / "runner.log", "a") as f:
        f.write(line + "\n")


def train_cmd(a, model, g, arm, seed, accum, out_dir: Path):
    fl = arm_flags(arm)
    cmd = [sys.executable, "-m", "src.train",
           "--model", model, "--placement", "spatial",
           "--dataset", g.dataset, "--data-root", a.data_root,
           "--scope", fl["scope"], "--router", fl["router"], "--arm-name", arm,
           "--width", str(g.width), "--num-kernels", str(g.K),
           "--seed", str(seed), "--epochs", str(a.epochs),
           "--batch-size", "128", "--accum", str(accum),
           "--impl", "blend", "--amp", "bf16", "--workers", str(a.workers),
           "--out", str(out_dir), "--tag", tag(model, g),
           "--ckpt", str(out_dir / "ckpt" / result_name(model, g, arm, seed).replace(".json", ".pt"))]
    if g.recipe == "anneal":
        cmd += ANNEAL_ARGS
    if a.fake_data:
        cmd += ["--fake-data"]
    return cmd


def probe_cmd(a, model, g, arm, accum, out: Path):
    fl = arm_flags(arm)
    cmd = [sys.executable, "-m", "src.train", "--model", model, "--placement", "spatial",
           "--dataset", g.dataset, "--data-root", a.data_root, "--fake-data",
           "--scope", fl["scope"], "--router", fl["router"],
           "--width", str(g.width), "--num-kernels", str(g.K),
           "--batch-size", "128", "--accum", str(accum), "--epochs", str(a.epochs),
           "--impl", "blend", "--amp", "bf16", "--workers", "0",
           "--max-steps", str(a.probe_steps), "--probe-out", str(out)]
    return cmd


PROGRESS = re.compile(r"^\s*\d+(\.\d+)?%\s*$")


def default_exec(cmd, logfile=None):
    if logfile is None:
        return subprocess.call(cmd)
    # UTF-8 with replacement: on Windows the default (cp1252) can crash on a
    # character the child prints. Download progress ("12.3%", redrawn with \r
    # and so split into one line per update by the pipe) is dropped.
    with open(logfile, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            if PROGRESS.match(line):
                continue
            sys.stdout.write(line); sys.stdout.flush(); f.write(line)
        return p.wait()


def choose_accum(a, model, g, model_dir: Path, execute, plan):
    """Micro-batch for a group: the first rung of the ladder whose probe does
    not run out of memory, using the group's most memory-hungry arm. Stored
    in plan.json so a restart reuses the decision instead of re-probing."""
    key = tag(model, g)
    if key in plan:
        return plan[key]["accum"]
    arm = max(g.arms, key=lambda x: ARM_WEIGHT[x])
    if a.dry_run:
        return 1          # nothing is executed or saved in a dry run
    if a.skip_probe:
        plan[key] = {"accum": 1, "probe": "skipped", "arm": arm}
    else:
        for accum in ACCUM_LADDER:
            out = model_dir / "probes" / f"{key}_{arm}_accum{accum}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            rc = execute(probe_cmd(a, model, g, arm, accum, out))
            res = json.loads(out.read_text()) if out.exists() else {"oom": True, "rc": rc}
            if rc == 0 and not res.get("oom"):
                plan[key] = {"accum": accum, "micro_batch": 128 // accum, "arm": arm,
                             "sec_per_step": res.get("sec_per_step"),
                             "peak_mem_gb": res.get("peak_mem_gb"),
                             "est_minutes_per_run": res.get("est_minutes")}
                break
            log(model_dir, f"probe {key} {arm} accum {accum}: out of memory or failed (rc {rc})")
        else:
            raise SystemExit(f"{key}: out of memory even at micro-batch {128 // ACCUM_LADDER[-1]}")
    (model_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    est = plan[key].get("est_minutes_per_run")
    if est:
        n = len(g.arms) * 6
        log(model_dir, f"probe {key}: about {est:.0f} min per run for the heaviest arm ({arm}) at "
                       f"micro-batch {128 // plan[key]['accum']}; this group is {n} runs, at most "
                       f"~{est * n / 60:.1f} h (training steps only, data loading and evaluation add to it)")
    if plan[key]["accum"] > 1:
        with open(model_dir / "DEVIATIONS.md", "a") as f:
            f.write(f"- {key}: micro-batch {128 // plan[key]['accum']} x {plan[key]['accum']} "
                    f"accumulation steps (logical batch 128 unchanged) for every arm of this "
                    f"group; batch 128 ran out of memory in the probe.\n")
        log(model_dir, f"DEVIATION {key}: micro-batch {128 // plan[key]['accum']} x {plan[key]['accum']}")
    return plan[key]["accum"]


def tin_ready(root):
    base = Path(root) / "tiny-imagenet-200"
    return (base / "train").is_dir() and not (base / "val" / "images").is_dir()


def status(a, model):
    d = Path(a.out) / model
    J = jobs(model)
    done = sum((d / result_name(model, g, arm, s)).exists() for g, arm, s in J)
    div = sum((d / (result_name(model, g, arm, s) + ".DIVERGED")).exists() for g, arm, s in J)
    gate = "FAILED" if (d / "GATE_FAILED.json").exists() else ""
    return done, div, len(J), gate


def run_model(a, model, execute=default_exec):
    """Run (or resume) one model's 132 jobs. Returns 'done', 'gate_failed',
    'stopped' or 'tin_missing'."""
    d = Path(a.out) / model
    d.mkdir(parents=True, exist_ok=True)
    if (d / "GATE_FAILED.json").exists() and not a.force_gate:
        log(d, f"{model}: gate failed earlier (GATE_FAILED.json); skipping. --force-gate to override.")
        return "gate_failed"
    plan = json.loads((d / "plan.json").read_text()) if (d / "plan.json").exists() else {}
    J = jobs(model)
    assert len(J) == 132, len(J)
    floor = a.gate_floor if a.gate_floor is not None else GATE_FLOOR[model]
    if a.gate_floor is not None:
        log(d, f"gate floor overridden on the command line: {floor} (default {GATE_FLOOR[model]})")
    t_model = time.time()
    ran = 0
    for i, (g, arm, seed) in enumerate(J, 1):
        res = d / result_name(model, g, arm, seed)
        marker = Path(str(res) + ".DIVERGED")
        is_gate = (g.key + (arm, seed)) == GATE_JOB
        if res.exists() or marker.exists():
            if is_gate and res.exists() and not gate_ok(a, model, d, res, floor):
                return "gate_failed"
            continue
        if g.dataset == "tinyimagenet" and not a.fake_data and not a.dry_run and not tin_ready(a.data_root):
            log(d, f"Tiny-ImageNet not prepared at {a.data_root}. Run: python download_tinyimagenet.py "
                   f"--root {a.data_root}   then rerun this command; it resumes here.")
            return "tin_missing"
        accum = choose_accum(a, model, g, d, execute, plan)
        cmd = train_cmd(a, model, g, arm, seed, accum, d)
        remaining = sum(not (d / result_name(model, *j)).exists() for j in J)
        log(d, f"[{i}/132] {tag(model, g)} {arm} seed {seed} (micro-batch {128 // accum}); "
               f"{remaining} left for {model}")
        if a.dry_run:
            print("   ", " ".join(cmd[1:]))
            continue
        t0 = time.time()
        rc = execute(cmd, d / "train.log")
        ran += 1
        if rc in (5, -9):              # 5: CUDA out of memory; -9: killed by the OS for memory (Linux)
            # out of memory mid-run although the probe fit (fragmentation, a
            # longer eval pass): step down one rung for the rest of the group,
            # say so, and rerun this job from its checkpoint
            key = tag(model, g)
            rung = ACCUM_LADDER.index(plan[key]["accum"]) + 1
            if rung >= len(ACCUM_LADDER):
                log(d, f"out of memory at micro-batch {128 // plan[key]['accum']} mid-run; stopping.")
                return "stopped"
            old = plan[key]["accum"]; plan[key]["accum"] = ACCUM_LADDER[rung]
            plan[key]["micro_batch"] = 128 // ACCUM_LADDER[rung]
            plan[key]["raised_after"] = res.name
            (d / "plan.json").write_text(json.dumps(plan, indent=2))
            why = "out of memory mid-run" if rc == 5 else "killed mid-run (rc -9, memory)"
            msg = (f"- {key}: {why} in {res.name} at micro-batch {128 // old}; "
                   f"this and later runs of the group use micro-batch {128 // ACCUM_LADDER[rung]}. "
                   f"Runs of the group finished before keep micro-batch {128 // old}; the report lists "
                   f"the micro-batch of every arm.\n")
            with open(d / "DEVIATIONS.md", "a") as f:
                f.write(msg)
            log(d, "DEVIATION " + msg.strip()[2:])
            cmd = train_cmd(a, model, g, arm, seed, ACCUM_LADDER[rung], d)
            rc = execute(cmd, d / "train.log")
        if rc == 3:
            marker.write_text(json.dumps({"diverged": True, "cmd": cmd, "time": time.ctime()}))
            log(d, f"DIVERGED {res.name}: non-finite training loss. Marker written; continuing.")
            continue
        if rc != 0 or not res.exists():
            log(d, f"run failed (rc={rc}); stopping. Rerun the same command to resume.")
            return "stopped"
        mins = (time.time() - t0) / 60
        log(d, f"   finished in {mins:.1f} min, final {json.loads(res.read_text())['final_acc']:.2f}")
        if is_gate and not gate_ok(a, model, d, res, floor):
            return "gate_failed"
    if a.dry_run:
        # nothing was run: say so, rather than "all jobs present"
        n_done = sum((d / result_name(model, *j)).exists() for j in J)
        print(f"DRY RUN {model}: {len(J)} commands listed, {n_done} already finished, nothing executed")
        return "dry_run"
    log(d, f"{model}: all 132 jobs present ({ran} run this session, {(time.time()-t_model)/3600:.1f} h)")
    return "done"


def gate_ok(a, model, d, res, floor):
    acc = json.loads(res.read_text())["final_acc"]
    if acc >= floor:
        return True
    (d / "GATE_FAILED.json").write_text(json.dumps(
        {"model": model, "final_acc": acc, "floor": floor, "result": res.name,
         "meaning": "the shared SGD recipe does not train this backbone to a usable accuracy; "
                    "its remaining runs were not spent. Report this, do not hide it."}, indent=2))
    log(d, f"GATE FAILED {model}: static w64 seed 0 reached {acc:.2f} < floor {floor}. "
           f"Skipping its other 131 runs; moving to the next model.")
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=RUN_ORDER, choices=RUN_ORDER + ["resnet18"])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="results_models")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe-steps", type=int, default=20)
    ap.add_argument("--skip-probe", action="store_true", help="assume batch 128 fits (logged)")
    ap.add_argument("--gate-floor", type=float, default=None, help="override every gate floor (logged)")
    ap.add_argument("--force-gate", action="store_true", help="run a model whose gate failed before")
    ap.add_argument("--fake-data", action="store_true", help="pipeline test only; results are meaningless")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)

    if a.status:
        print(f"{'model':<15}{'done':>6}{'diverged':>10}{'of':>5}  gate")
        for m in a.models:
            done, div, n, gate = status(a, m)
            print(f"{m:<15}{done:>6}{div:>10}{n:>5}  {gate}")
        return 0
    outcome = {}
    for m in a.models:
        outcome[m] = run_model(a, m)
        if outcome[m] in ("stopped", "tin_missing"):
            break
    print("\n" + ", ".join(f"{m}: {o}" for m, o in outcome.items()))
    if a.dry_run:
        return 0
    if all(o in ("done", "gate_failed") for o in outcome.values()) and len(outcome) == len(a.models):
        print("next: python report/build_model_report.py --model <m> --results results_models/<m>   and   python report/cross.py --results ...  (README_MODELS.md, section 6)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

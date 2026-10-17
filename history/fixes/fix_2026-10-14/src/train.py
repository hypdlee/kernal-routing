"""
One training run = one (router, seed) pair.

Writes a single JSON record to --out. run_stage1.py loops this and
analyze.py reads the JSON files. Nothing downstream parses stdout.

Usage:
    python -m src.train --router gap --seed 0 --epochs 150
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import build_loaders, num_classes
from .resnet import param_counts
from .models import MODELS, build_model


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def train_step(model, opt, sched, crit, scaler, x, y, device, amp_dtype, accum):
    """One optimiser step on one logical batch, split into `accum` equal
    micro-batches. Each micro-batch loss is divided by `accum`, so the
    accumulated gradient equals the gradient of the mean loss over the whole
    batch (exactly, up to BatchNorm, whose statistics are per micro-batch).
    Returns the logical-batch mean loss, the quantity logged before."""
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for xs, ys in zip(x.chunk(accum), y.chunk(accum)):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = crit(model(xs), ys) / accum
        scaler.scale(loss).backward()
        total += loss.item()
    scaler.step(opt)
    scaler.update()
    sched.step()
    return total


def probe(args, model, opt, sched, crit, scaler, train_ld, device, amp_dtype, micro, pc, test_ld=None):
    """Measure what one configuration costs before committing a run to it.
    Two warm-up steps (cudnn autotune, allocator), then --max-steps timed
    steps. Writes JSON; an out-of-memory error is a result, not a crash."""
    out = {"model": args.model, "dataset": args.dataset, "width": args.width,
           "num_kernels": args.num_kernels, "scope": args.scope, "router": args.router,
           "batch_size": args.batch_size, "micro_batch": micro, "accum": args.accum,
           "params": pc["total"], "device": str(device), "oom": False}
    it = iter(train_ld)

    def batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(train_ld)
            return next(it)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        for _ in range(2):
            train_step(model, opt, sched, crit, scaler, *batch(), device, amp_dtype, args.accum)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(args.max_steps):
            train_step(model, opt, sched, crit, scaler, *batch(), device, amp_dtype, args.accum)
        if device.type == "cuda":
            torch.cuda.synchronize()
        out["sec_per_step"] = (time.time() - t) / args.max_steps
        # The runner probes on fake data (512 images), so count the steps of the
        # real training set, not of the fake loader.
        real = {"cifar100": 50000, "tinyimagenet": 100000}[args.dataset] // args.batch_size
        out["steps_per_epoch"] = real
        out["est_minutes"] = out["sec_per_step"] * real * args.epochs / 60
        # the evaluation pass uses its own, larger batch (max(batch, 256)) and
        # records routing statistics: check it fits too
        if test_ld is not None:
            model.collect_stats(True)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                                 enabled=amp_dtype is not None):
                model.eval(); model(next(iter(test_ld))[0].to(device))
            model.collect_stats(False)
        out["peak_mem_gb"] = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else None
    except torch.OutOfMemoryError:
        out["oom"] = True
    if args.probe_out:
        Path(args.probe_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.probe_out).write_text(json.dumps(out, indent=2))
    print("probe", json.dumps(out), flush=True)
    return 0


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def main():
    try:
        return _main()
    except torch.OutOfMemoryError as e:
        print(f"  out of memory: {str(e).splitlines()[0][:200]} (exit 5)", flush=True)
        return 5


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=sorted(MODELS),
                    help="backbone. resnet18 is the original study and the default, "
                         "so every earlier command runs unchanged.")
    ap.add_argument("--placement", default="spatial", choices=["spatial", "all"],
                    help="which convolutions become dynamic. 'spatial' is the ResNet-18 "
                         "rule (every kxk conv inside blocks, k>1); 'all' is every conv "
                         "except the stem (DY-Conv's rule). The replication uses 'spatial'.")
    ap.add_argument("--router", default="gap")
    ap.add_argument("--arm-name", default=None,
                    help="label for the result file and record. Default: 'static' when "
                         "--scope none, else the router name. Old runs named files by "
                         "--router; pass nothing to keep that behaviour.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="logical batch: one optimiser step per batch of this size")
    ap.add_argument("--accum", type=int, default=1,
                    help="split each logical batch into this many micro-batches and "
                         "accumulate gradients. Same data order, same optimiser-step count, "
                         "same LR schedule as --accum 1. Only BatchNorm statistics differ "
                         "(computed per micro-batch); recorded and reported as a deviation.")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="probe mode: run this many optimiser steps, write peak memory and "
                         "seconds per step to --probe-out, and exit without a result file")
    ap.add_argument("--probe-out", default=None)
    ap.add_argument("--stop-after-epoch", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path, written every epoch and resumed from if present. "
                         "A resumed run is not bit-identical to an uninterrupted one; the "
                         "record says where it resumed.")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--num-kernels", type=int, default=4)
    ap.add_argument("--scope", default="all", choices=["all", "stage34", "none"])
    ap.add_argument("--width", type=int, default=64,
                    help="base channel width. 64 is standard ResNet-18; "
                         "32 and 16 are the reduced-capacity replications.")
    ap.add_argument("--impl", default="blend", choices=["blend", "combine"])
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="constant softmax temperature (used when --temp-start "
                         "is not given)")
    ap.add_argument("--temp-start", type=float, default=None,
                    help="initial temperature for annealing. DY-Conv (Chen et al., "
                         "CVPR 2020) uses 30. Omit for a constant temperature, "
                         "which reproduces every result before this flag existed.")
    ap.add_argument("--temp-end", type=float, default=1.0)
    ap.add_argument("--temp-epochs", type=int, default=10,
                    help="epochs over which to anneal, linearly")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--dataset", default="cifar100",
                    choices=["cifar100", "tinyimagenet"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fake-data", action="store_true", help="smoke test, no download")
    args = ap.parse_args()
    if args.batch_size % args.accum:
        ap.error("--batch-size must be divisible by --accum")
    micro = args.batch_size // args.accum
    arm = args.arm_name or ("static" if args.scope == "none" else args.router)

    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    # With accumulation the evaluation batch shrinks by the same factor
    # (256 at accum 1, exactly as before), so a group that needed smaller
    # micro-batches does not run out of memory in evaluation instead.
    train_ld, test_ld = build_loaders(args.data_root, args.batch_size, args.workers,
                                      fake=args.fake_data, dataset=args.dataset,
                                      eval_batch=None if args.accum == 1 else max(16, 256 // args.accum))

    model = build_model(args.model, num_classes(args.dataset), scope=args.scope,
                        num_kernels=args.num_kernels, router=args.router,
                        temperature=args.temperature, width=args.width,
                        placement=args.placement).to(device)
    for m in model.dynamic_layers():
        m.impl = args.impl

    anneal = args.temp_start is not None

    def set_temperature(epoch):
        """Linear anneal from temp_start to temp_end over temp_epochs, then
        constant. A high initial temperature flattens the softmax so every
        kernel in the bank receives gradient early; without it the mixture is
        close to one-hot from the first step and most of the bank is starved.
        DY-Conv reports this as necessary rather than optional."""
        if not anneal:
            return args.temperature
        f = min(1.0, epoch / max(1, args.temp_epochs))
        tau = args.temp_start + f * (args.temp_end - args.temp_start)
        for m in model.dynamic_layers():
            m.temperature = tau
        return tau
    # No channels_last: the kernel bank is rank 5 and torch rejects it.

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    steps = max(1, len(train_ld))
    warm = args.warmup * steps
    total = args.epochs * steps

    def lr_at(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)
    crit = nn.CrossEntropyLoss()

    pc = param_counts(model)
    if args.max_steps:
        return probe(args, model, opt, sched, crit, scaler, train_ld, device, amp_dtype, micro, pc, test_ld)

    start_epoch, best, history, resumed_at, t_prev = 0, 0.0, [], None, 0.0
    if args.ckpt and Path(args.ckpt).exists():
        # Load to CPU: map_location=cuda would move the saved RNG states to the
        # GPU, and torch.set_rng_state only accepts a CPU ByteTensor (this is
        # what broke resuming on Windows). load_state_dict copies model and
        # optimiser tensors onto the device of the live parameters anyway.
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        t_prev = ck.get("seconds", 0.0)
        torch.set_rng_state(ck["rng_cpu"].cpu().to(torch.uint8))
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in ck["rng_cuda"]])
        resumed_at = start_epoch
        print(f"  resumed from {args.ckpt} at epoch {start_epoch}", flush=True)

    print(f"[{args.model} {args.dataset} {arm} w{args.width} K{args.num_kernels} "
          f"seed={args.seed}] device={device} "
          f"params={pc['total']/1e6:.2f}M router={pc['router']/1e3:.1f}K "
          f"dyn_layers={len(model.dynamic_layers())} micro_batch={micro} accum={args.accum}",
          flush=True)

    t0 = time.time() - t_prev

    for epoch in range(start_epoch, args.epochs):
        tau = set_temperature(epoch)
        model.train()
        run_loss = 0.0
        for x, y in train_ld:
            run_loss += train_step(model, opt, sched, crit, scaler, x, y, device, amp_dtype, args.accum)
        if not np.isfinite(run_loss):
            print(f"  non-finite training loss at epoch {epoch}; stopping (exit 3)", flush=True)
            return 3

        acc = evaluate(model, test_ld, device, amp_dtype)
        best = max(best, acc)
        history.append({"epoch": epoch, "loss": round(run_loss / steps, 4),
                        "test_acc": round(acc, 3), "lr": round(sched.get_last_lr()[0], 6)})
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            tnote = f"  tau {tau:.2f}" if anneal else ""
            print(f"  ep {epoch:3d}  loss {run_loss/steps:.3f}  acc {acc:.2f}  best {best:.2f}"
                  f"{tnote}  ({time.time()-t0:.0f}s)", flush=True)
        if args.ckpt:
            Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(args.ckpt) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best": best, "history": history,
                        "seconds": time.time() - t0, "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       tmp)
            os.replace(tmp, args.ckpt)
        if args.stop_after_epoch is not None and epoch == args.stop_after_epoch:
            print("  stopping early (test hook)", flush=True)
            return 4

    # Routing collapse evidence from one final eval pass.
    model.collect_stats(True)
    final_acc = evaluate(model, test_ld, device, amp_dtype)
    routing = model.routing_report()
    model.collect_stats(False)

    rec = {
        "tag": args.tag,
        "model": args.model,
        "placement": args.placement,
        "arm": arm,
        "micro_batch": micro,
        "accum": args.accum,
        "resumed_at_epoch": resumed_at,
        "router": args.router,
        "seed": args.seed,
        "final_acc": round(final_acc, 3),
        "best_acc": round(best, 3),
        "params": pc,
        "minutes": round((time.time() - t0) / 60, 2),
        "config": vars(args),
        "env": {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python": platform.python_version(),
            "git": git_sha(),
        },
        "routing": routing,
        "history": history,
    }

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    name = args.arm_name or args.router
    path = outdir / f"{args.tag}_{name}_seed{args.seed}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, path)          # a crash never leaves a half-written result
    if args.ckpt and Path(args.ckpt).exists():
        Path(args.ckpt).unlink()
    print(f"[{args.model} {arm} seed={args.seed}] final={final_acc:.2f} best={best:.2f} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

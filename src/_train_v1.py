# Frozen copy of src/train.py as it was for the 132 ResNet-18 runs, before the
# multi-model change. Used only by tests/test_train.py (backcompat) to show the
# new script reproduces it exactly. Do not edit.
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
from .resnet import ResNet18, param_counts


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="gap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
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

    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    train_ld, test_ld = build_loaders(args.data_root, args.batch_size, args.workers,
                                      fake=args.fake_data, dataset=args.dataset)

    model = ResNet18(num_classes(args.dataset), scope=args.scope, num_kernels=args.num_kernels,
                     router=args.router, temperature=args.temperature,
                     width=args.width).to(device)
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
    print(f"[{args.dataset} {args.router} w{args.width} K{args.num_kernels} "
          f"seed={args.seed}] device={device} "
          f"params={pc['total']/1e6:.2f}M router={pc['router']/1e3:.1f}K "
          f"dyn_layers={len(model.dynamic_layers())}", flush=True)

    best = 0.0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        tau = set_temperature(epoch)
        model.train()
        run_loss = 0.0
        for x, y in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()

        acc = evaluate(model, test_ld, device, amp_dtype)
        best = max(best, acc)
        history.append({"epoch": epoch, "loss": round(run_loss / steps, 4),
                        "test_acc": round(acc, 3), "lr": round(sched.get_last_lr()[0], 6)})
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            tnote = f"  tau {tau:.2f}" if anneal else ""
            print(f"  ep {epoch:3d}  loss {run_loss/steps:.3f}  acc {acc:.2f}  best {best:.2f}"
                  f"{tnote}  ({time.time()-t0:.0f}s)", flush=True)

    # Routing collapse evidence from one final eval pass.
    model.collect_stats(True)
    final_acc = evaluate(model, test_ld, device, amp_dtype)
    routing = model.routing_report()
    model.collect_stats(False)

    rec = {
        "tag": args.tag,
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
    path = outdir / f"{args.tag}_{args.router}_seed{args.seed}.json"
    path.write_text(json.dumps(rec, indent=2))
    print(f"[{args.router} seed={args.seed}] final={final_acc:.2f} best={best:.2f} -> {path}", flush=True)


if __name__ == "__main__":
    main()

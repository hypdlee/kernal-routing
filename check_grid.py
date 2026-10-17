"""
Pre-analysis audit of the Stage 1 grid.

Run this BEFORE analyze.py. It answers three questions that analyze.py
assumes rather than checks:

  1. Is every (router, seed) cell present?
  2. Was every run produced under the SAME recipe? The orchestrator skips
     runs whose JSON already exists, so a grid assembled across sessions
     can silently mix settings. Paired tests on mixed settings are void.
  3. Does anything look wrong before the statistics get involved?

Drop this next to run_stage1.py and run:
    python check_grid.py --dir results --tag stage1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXPECTED_ROUTERS = ["uniform", "gap", "shuffled", "frozen_random"]

# Config keys that MUST match across every run for the pairing to be valid.
# A difference in any of these makes two runs incomparable.
# 'scope' is deliberately NOT here. In the capacity sweep the static arm
# runs at scope=none while the dynamic arms run at scope=all, which is the
# design rather than recipe drift. Scope is checked per-arm instead, below.
CRITICAL = ["epochs", "batch_size", "lr", "weight_decay", "warmup",
            "num_kernels", "temperature", "width", "dataset", "model", "placement"]

# Values a run recorded before the key existed. Every run before the
# multi-model change was ResNet-18 with the spatial placement, CIFAR-100
# unless stated, width 64 unless stated.
LEGACY_DEFAULTS = {"model": "resnet18", "placement": "spatial", "width": 64, "dataset": "cifar100"}

# Differences here change speed or numerics slightly but not the comparison.
ADVISORY = ["impl", "amp", "workers", "deterministic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--routers", nargs="+", default=EXPECTED_ROUTERS)
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        raise SystemExit(f"FAIL: no such directory: {root}")

    recs = {}
    bad = []
    for f in sorted(root.rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            bad.append((f.name, str(e)))
            continue
        if d.get("tag") != args.tag:
            continue
        recs[(d["router"], d["seed"])] = (f, d)

    problems = []
    print(f"\n=== Grid audit: dir={root}  tag={args.tag} ===\n")

    if bad:
        print("Unreadable files:")
        for n, e in bad:
            print(f"  {n}: {e}")
        problems.append(f"{len(bad)} unreadable JSON file(s)")

    # --- 1. completeness --------------------------------------------------
    print("1. Cell completeness")
    missing = []
    header = "        " + "".join(f"{s:>6}" for s in args.seeds)
    print(header)
    for r in args.routers:
        row = f"  {r:<14}"
        for s in args.seeds:
            if (r, s) in recs:
                row += f"{recs[(r, s)][1]['final_acc']:>6.1f}"
            else:
                row += "     ."
                missing.append((r, s))
        print(row)
    n_expected = len(args.routers) * len(args.seeds)
    print(f"\n   {len(recs)}/{n_expected} cells present")
    if missing:
        print(f"   MISSING: {missing}")
        problems.append(f"{len(missing)} missing cell(s)")

    extra = [k for k in recs if k[0] not in args.routers or k[1] not in args.seeds]
    if extra:
        print(f"   unexpected cells (ignored by the grid): {extra}")

    # --- 2. recipe consistency -------------------------------------------
    print("\n2. Recipe consistency")
    vals = defaultdict(lambda: defaultdict(list))
    for (r, s), (f, d) in recs.items():
        cfg = d.get("config", {})
        for k in CRITICAL + ADVISORY:
            vals[k][json.dumps(cfg.get(k, LEGACY_DEFAULTS.get(k)))].append(f"{r}/s{s}")

    ok = True
    for k in CRITICAL:
        if len(vals[k]) > 1:
            ok = False
            print(f"   CRITICAL MISMATCH  {k}:")
            for v, who in vals[k].items():
                print(f"       {v:<14} <- {len(who)} run(s): {who[:6]}{' ...' if len(who) > 6 else ''}")
            problems.append(f"critical config mismatch on '{k}'")
    for k in ADVISORY:
        if len(vals[k]) > 1:
            print(f"   advisory difference  {k}: "
                  + ", ".join(f"{v}({len(w)})" for v, w in vals[k].items()))
    if ok:
        shown = {k: json.loads(next(iter(vals[k]))) for k in CRITICAL if vals[k]}
        print("   all critical settings identical across runs:")
        print("     " + "  ".join(f"{k}={v}" for k, v in shown.items()))

    # --- 3. sanity --------------------------------------------------------
    print("\n3. Sanity")
    for (r, s), (f, d) in sorted(recs.items()):
        acc = d["final_acc"]
        hist = d.get("history", [])
        if acc < 60:
            print(f"   LOW    {r}/s{s}: final {acc:.2f} (expected 70+). Possible failed run.")
            problems.append(f"{r}/s{s} suspiciously low ({acc:.2f})")
        if hist:
            last = hist[-1]
            if last.get("loss", 1) > 0.5:
                print(f"   NOTE   {r}/s{s}: final train loss {last['loss']:.3f}, "
                      f"model may not have converged")
        if abs(d.get("best_acc", acc) - acc) > 2.0:
            print(f"   NOTE   {r}/s{s}: best {d['best_acc']:.2f} vs final {acc:.2f}, "
                  f"unstable tail")

    # scope must be constant WITHIN an arm, even though it varies between
    # arms by design.
    scopes = defaultdict(set)
    for (r, s), (f, d) in recs.items():
        scopes[r].add(d.get("config", {}).get("scope"))
    for r, ss in sorted(scopes.items()):
        if len(ss) > 1:
            print(f"   SCOPE MISMATCH within arm '{r}': {sorted(ss)}")
            problems.append(f"arm '{r}' has inconsistent scope")

    # parameter counts: uniform and frozen_random must match exactly
    pc = {r: recs[(r, s)][1]["params"]["total"]
          for (r, s) in recs if (r, s) in recs for r_ in [r]}
    pc = {}
    for (r, s), (f, d) in recs.items():
        pc.setdefault(r, set()).add(d["params"]["total"])
    print("\n   parameter counts by arm:")
    for r in args.routers:
        if r in pc:
            vs = sorted(pc[r])
            print(f"     {r:<14} {', '.join(f'{v/1e6:.2f}M' for v in vs)}")
    if "uniform" in pc and "frozen_random" in pc and pc["uniform"] != pc["frozen_random"]:
        print("   WARNING: uniform and frozen_random should have identical param counts")
        problems.append("uniform/frozen_random param count mismatch")

    # --- verdict ----------------------------------------------------------
    print("\n" + "=" * 60)
    if problems:
        print("AUDIT FAILED. Do not run analyze.py until these are resolved:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("AUDIT PASSED. The grid is complete and internally consistent.")
    print("Next: python -m src.analyze --dir %s --tag %s --baseline gap --margin 0.5"
          % (args.dir, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Stage 1 analysis. Reads the JSON records and applies the pre-registered
decision rule.

Two tests, doing different jobs:

  Difference test  paired t-test, plus Wilcoxon signed-rank when there are
                   enough seeds for it to mean anything. Answers "is there
                   an effect?" A large p here is NOT evidence of no effect.

  Equivalence test TOST (two one-sided tests) against the pre-registered
                   margin. Answers "is any effect smaller than the margin?"
                   This is the test that licenses a null claim, and it is
                   the one Stage 1 exists to run.

Seed count matters. The two-sided Wilcoxon signed-rank test has a floor on
its attainable p-value at small n:

    n=3 -> 0.25    n=4 -> 0.125    n=5 -> 0.0625    n=6 -> 0.031

So at 3 seeds you cannot reach p < 0.05 with Wilcoxon no matter how large
the effect is. Run 6 seeds if you want Wilcoxon to carry the difference
claim; 5 is workable if the t-test and TOST carry it instead. The script
warns when n is too small for the test being reported.

Usage:
    python -m src.analyze --dir results --baseline gap --margin 0.5
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

WILCOXON_FLOOR = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125, 5: 0.0625, 6: 0.03125, 7: 0.015625}


def load(dirpath: Path, tag: str | None):
    runs = defaultdict(dict)
    for f in sorted(dirpath.rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "router" not in d or "final_acc" not in d:
            continue
        if tag and d.get("tag") != tag:
            continue
        runs[d["router"]][d["seed"]] = d
    return runs


def paired(runs, a: str, b: str):
    seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
    xa = np.array([runs[a][s]["final_acc"] for s in seeds], dtype=float)
    xb = np.array([runs[b][s]["final_acc"] for s in seeds], dtype=float)
    return seeds, xa, xb


def tost(diff: np.ndarray, margin: float):
    """Two one-sided tests for equivalence within +/- margin.

    Returns the larger of the two one-sided p-values. p < 0.05 means the
    true mean difference is inside the margin, which is a positive claim
    of equivalence rather than a failure to reject."""
    n = len(diff)
    if n < 2:
        return float("nan")
    se = diff.std(ddof=1) / math.sqrt(n)
    if se == 0:
        # No spread at all. Report NaN rather than a confident 0, since a
        # zero-variance sample carries no information about the population.
        return float("nan")
    df = n - 1
    p_lower = stats.t.sf((diff.mean() + margin) / se, df)   # H0: mu <= -margin
    p_upper = stats.t.cdf((diff.mean() - margin) / se, df)  # H0: mu >= +margin
    return float(max(p_lower, p_upper))


def compare(runs, treat: str, ref: str, margin: float):
    seeds, xt, xr = paired(runs, treat, ref)
    n = len(seeds)
    if n == 0:
        return None
    d = xt - xr

    row = {
        "treatment": treat, "reference": ref, "n_seeds": n, "seeds": seeds,
        "mean_treat": float(xt.mean()), "mean_ref": float(xr.mean()),
        "mean_diff": float(d.mean()),
        "sd_diff": float(d.std(ddof=1)) if n > 1 else float("nan"),
    }

    if n > 1 and d.std(ddof=1) > 0:
        t_stat, p_t = stats.ttest_rel(xt, xr)
        se = d.std(ddof=1) / math.sqrt(n)
        crit = stats.t.ppf(0.975, n - 1)
        row["p_ttest"] = float(p_t)
        row["ci95"] = (float(d.mean() - crit * se), float(d.mean() + crit * se))
        row["cohens_dz"] = float(d.mean() / d.std(ddof=1))
    else:
        row["p_ttest"] = float("nan")
        row["ci95"] = (float("nan"), float("nan"))
        row["cohens_dz"] = float("nan")

    if n >= 3 and np.any(d != 0):
        try:
            row["p_wilcoxon"] = float(stats.wilcoxon(xt, xr).pvalue)
        except ValueError:
            row["p_wilcoxon"] = float("nan")
    else:
        row["p_wilcoxon"] = float("nan")

    row["wilcoxon_floor"] = WILCOXON_FLOOR.get(n, 0.0)
    row["p_tost"] = tost(d, margin)
    return row


def verdict(row, margin: float, alpha: float = 0.05) -> str:
    if row["n_seeds"] < 2:
        return "INSUFFICIENT (need >= 2 paired seeds)"
    if math.isnan(row["p_ttest"]) and row["sd_diff"] == 0:
        # Every seed produced an identical accuracy. Real on a coarse test
        # set, but it means the variance estimate is degenerate and the
        # equivalence p-value below is not trustworthy.
        return "DEGENERATE (zero variance across seeds; check the eval set size)"
    sig = (not math.isnan(row["p_ttest"])) and row["p_ttest"] < alpha
    equiv = (not math.isnan(row["p_tost"])) and row["p_tost"] < alpha
    if sig and not equiv:
        return f"EFFECT ({row['mean_diff']:+.2f}pp, outside +/-{margin}pp)"
    if equiv and not sig:
        return f"EQUIVALENT (within +/-{margin}pp)"
    if sig and equiv:
        return f"SMALL BUT REAL ({row['mean_diff']:+.2f}pp, detectable yet inside the margin)"
    return "INCONCLUSIVE (underpowered: more seeds needed)"


def routing_summary(runs, router: str):
    """Mean normalised routing entropy across layers and seeds. 1.0 means
    uniform mixing, i.e. the router is not discriminating between kernels."""
    ents, degenerate = [], 0
    for rec in runs.get(router, {}).values():
        for layer in rec.get("routing", []):
            ents.append(layer["z_entropy_norm"])
            counts = layer.get("argmax_counts") or []
            if counts and max(counts) == sum(counts):
                degenerate += 1
    if not ents:
        return None
    return {"mean_entropy_norm": float(np.mean(ents)), "min": float(np.min(ents)),
            "layers": len(ents), "single_branch_layers": degenerate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--baseline", default="gap", help="the router everything is compared against")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="pre-registered equivalence margin in accuracy points")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    runs = load(Path(args.dir), args.tag or None)
    if not runs:
        raise SystemExit(f"no runs found in {args.dir} (tag={args.tag!r})")

    print(f"\nStage 1 analysis   dir={args.dir}  tag={args.tag}  "
          f"baseline={args.baseline}  margin=+/-{args.margin}pp\n")

    print("Per-variant accuracy")
    print(f"  {'router':<16}{'n':>3}  {'mean':>7} {'sd':>6}   seeds")
    for r in sorted(runs):
        accs = np.array([v["final_acc"] for v in runs[r].values()], float)
        sd = accs.std(ddof=1) if len(accs) > 1 else float("nan")
        print(f"  {r:<16}{len(accs):>3}  {accs.mean():>7.2f} {sd:>6.2f}   "
              f"{sorted(runs[r])}")

    rows = []
    print(f"\nPaired comparisons against '{args.baseline}'")
    for r in sorted(runs):
        if r == args.baseline:
            continue
        row = compare(runs, r, args.baseline, args.margin)
        if row is None:
            continue
        rows.append(row)
        v = verdict(row, args.margin, args.alpha)
        ci = row["ci95"]
        print(f"\n  {r} - {args.baseline}   n={row['n_seeds']}")
        print(f"    mean diff   {row['mean_diff']:+.3f} pp   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        print(f"    t-test      p = {row['p_ttest']:.4f}    dz = {row['cohens_dz']:+.2f}")
        wf = row["wilcoxon_floor"]
        wnote = "" if wf < args.alpha else f"  <-- floor at n={row['n_seeds']} is {wf}, cannot reach {args.alpha}"
        print(f"    Wilcoxon    p = {row['p_wilcoxon']:.4f}{wnote}")
        print(f"    TOST        p = {row['p_tost']:.4f}  (equivalence within +/-{args.margin}pp)")
        print(f"    verdict     {v}")

    print("\nRouting collapse (normalised entropy; 1.0 = uniform mixing)")
    for r in sorted(runs):
        s = routing_summary(runs, r)
        if s:
            print(f"  {r:<16} mean={s['mean_entropy_norm']:.4f}  min={s['min']:.4f}  "
                  f"single-branch layers={s['single_branch_layers']}/{s['layers']}")

    print("\nPre-registered read of the result:")
    for row in rows:
        if row["treatment"] == "uniform":
            v = verdict(row, args.margin, args.alpha)
            if v.startswith("EQUIVALENT"):
                print("  uniform matches the trained GAP router. On this benchmark the")
                print("  gain from dynamic convolution is capacity, not routing. Write it up.")
            elif v.startswith("EFFECT"):
                print("  Routing beats the capacity control. The premise holds and Stage 0")
                print("  (descriptor differentiability) is now worth building.")
            else:
                print(f"  uniform vs {args.baseline} is {v.lower()}. Add seeds before concluding.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"config": vars(args), "comparisons": rows,
             "routing": {r: routing_summary(runs, r) for r in runs}}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

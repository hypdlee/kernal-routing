"""Read the K sweep at fixed width. Pulls K=4 and static from stage1c."""
from __future__ import annotations
import glob, json, math
import numpy as np
from scipy import stats


def load(width=32, dataset="cifar100"):
    """Filter on dataset as well as width. Grouping on width alone merges
    runs from different datasets into one arm, which silently produces a
    static reference averaged across both."""
    arms = {}
    for f in glob.glob("results/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "router" not in d:
            continue
        c = d["config"]
        if c.get("width", 64) != width:
            continue
        if c.get("dataset", "cifar100") != dataset:
            continue
        if c.get("scope") == "none":
            arms.setdefault("static", {})[d["seed"]] = d["final_acc"]
        elif d["router"] == "gap":
            arms.setdefault(c.get("num_kernels", 4), {})[d["seed"]] = d["final_acc"]
    return arms


def main(width=32, margin=0.5, dataset="cifar100"):
    arms = load(width, dataset)
    if "static" not in arms:
        raise SystemExit(f"no static arm at width {width} for dataset={dataset}")
    ks = sorted(k for k in arms if k != "static")
    if not ks:
        raise SystemExit("no gap arms found; run run_ksweep.py first")

    print(f"\nK sweep  |  dataset={dataset}  width={width}   (static reference, n={len(arms['static'])})\n")
    print(f"  {'K':>3} {'n':>3}  {'mean':>7} {'sd':>6}   {'vs static':>10}  "
          f"{'95% CI':>20}  {'p':>9}")
    base = arms["static"]
    rows = []
    for k in ks:
        seeds = sorted(set(arms[k]) & set(base))
        x = np.array([arms[k][s] for s in seeds])
        y = np.array([base[s] for s in seeds])
        d = x - y; n = len(d)
        if n < 2:
            print(f"  {k:>3} {n:>3}   (need >= 2 paired seeds)"); continue
        t, p = stats.ttest_rel(x, y)
        se = d.std(ddof=1)/math.sqrt(n); crit = stats.t.ppf(0.975, n-1)
        print(f"  {k:>3} {n:>3}  {x.mean():7.3f} {x.std(ddof=1):6.3f}   "
              f"{d.mean():+10.3f}  [{d.mean()-crit*se:+7.3f},{d.mean()+crit*se:+7.3f}]  "
              f"{p:9.5f}")
        rows.append((k, d.mean(), p))

    print(f"\n  static: {np.mean(list(base.values())):.3f}\n")

    # Pairwise between consecutive K values. Reporting only that the point
    # estimates happen to be ordered would be an over-read: the question is
    # whether any pair actually differs, and that needs a test per pair.
    print("  Pairwise between K values (margin +/- 0.5pp):")
    equivalent = []
    for a, b in zip(ks, ks[1:]):
        seeds = sorted(set(arms[a]) & set(arms[b]))
        if len(seeds) < 2:
            continue
        x = np.array([arms[b][s] for s in seeds])
        y = np.array([arms[a][s] for s in seeds])
        d = x - y; n = len(d)
        t, p = stats.ttest_rel(x, y)
        se = d.std(ddof=1)/math.sqrt(n)
        pl = stats.t.sf((d.mean()+margin)/se, n-1)
        pu = stats.t.cdf((d.mean()-margin)/se, n-1)
        tost = max(pl, pu)
        eq = tost < 0.05 and p >= 0.05
        equivalent.append(eq)
        print(f"    K={b} vs K={a}: {d.mean():+.3f}pp  p={p:.4f}  "
              f"TOST={tost:.4f}  {'EQUIVALENT' if eq else 'differs' if p<0.05 else 'inconclusive'}")

    print()
    if equivalent and all(equivalent):
        print(f"  -> No pair of K values differs. The routing benefit is saturated by")
        print(f"     K={ks[0]}; larger banks cost parameters and time for nothing.")
        print("     Do not describe the ordering of the means as a trend in K.")
    elif any(equivalent):
        print("  -> Mixed: some adjacent pairs are equivalent, some are not.")
        print("     Read the pairwise rows; no single summary fits.")
    else:
        best = max(rows, key=lambda r: r[1])
        print(f"  -> Largest effect at K={best[0]} ({best[1]:+.3f}pp), and adjacent")
        print("     K values do differ, so K genuinely matters here.")
    print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dataset", default="cifar100")
    a = ap.parse_args()
    main(a.width, dataset=a.dataset)


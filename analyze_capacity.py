"""
Read the capacity sweep: does the dynamic-convolution advantage over a
plain ResNet-18 depend on width?

Pulls width 64 from the existing stage1 / stage1b results and widths 16
and 32 from stage1c. Reports, per width, each arm's accuracy, the paired
difference against static, and whether the models were still saturating.
"""

from __future__ import annotations

import glob
import json
import math

import numpy as np
from scipy import stats

ARMS = ["static", "uniform", "gap"]


def load_all(dataset="cifar100", kernels=4):
    out = {}
    for f in glob.glob("results/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cfg = d.get("config", {})
        if "final_acc" not in d:
            continue
        # The grouping key must include the dataset AND the bank size.
        # Grouping on width alone silently merges runs that differ in
        # either, and the last file read wins. That produced a comparison
        # between one dataset's uniform arm and another's static arm.
        # Runs predating the --dataset flag are CIFAR-100 by construction.
        w = cfg.get("width", 64)
        ds = cfg.get("dataset", "cifar100")
        k = cfg.get("num_kernels", 4)
        arm = "static" if cfg.get("scope") == "none" else d["router"]
        if arm not in ARMS:
            continue
        if k != kernels:            # keep one bank size per sweep
            continue
        if ds != dataset:
            continue
        out.setdefault(w, {}).setdefault(arm, {})[d["seed"]] = d
    return out


def tost(d, margin=0.5):
    n = len(d)
    if n < 2:
        return float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return float("nan")
    return max(stats.t.sf((d.mean() + margin) / se, n - 1),
               stats.t.cdf((d.mean() - margin) / se, n - 1))


def datasets_present():
    seen = set()
    for f in glob.glob("results/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "config" in d and "final_acc" in d:
            seen.add(d["config"].get("dataset", "cifar100"))
    return sorted(seen)


def main(dataset=None, kernels=4):
    if dataset is None:
        present = datasets_present()
        if len(present) > 1:
            print(f"\nResults contain more than one dataset: {present}")
            print("Reporting each separately. Pass --dataset to select one.\n")
            for ds in present:
                main(ds, kernels)
            return
        dataset = present[0] if present else "cifar100"

    data = load_all(dataset, kernels)
    if not data:
        raise SystemExit(f"no results for dataset={dataset}, K={kernels}")

    print("\n" + "=" * 74)
    print(f"CAPACITY SWEEP  |  dataset={dataset}  K={kernels}")
    print("=" * 74)

    trend = {}
    for w in sorted(data):
        arms = data[w]
        have = [a for a in ARMS if a in arms and len(arms[a]) >= 2]
        if "static" not in have:
            print(f"\nwidth {w}: no static arm, skipping")
            continue

        print(f"\n--- width {w} " + "-" * 58)
        acc = {a: np.array([arms[a][s]["final_acc"] for s in sorted(arms[a])]) for a in have}
        seedsets = {a: sorted(arms[a]) for a in have}

        for a in have:
            v = acc[a]
            p = arms[a][seedsets[a][0]]["params"]["total"] / 1e6
            loss = np.mean([arms[a][s]["history"][-1]["loss"] for s in seedsets[a]
                            if arms[a][s].get("history")])
            print(f"  {a:<9} n={len(v)}  {v.mean():6.3f} +/- {v.std(ddof=1):.3f}   "
                  f"{p:6.2f}M   final train loss {loss:.4f}")

        print()
        for a in have:
            if a == "static":
                continue
            common = sorted(set(seedsets[a]) & set(seedsets["static"]))
            if len(common) < 2:
                continue
            x = np.array([arms[a][s]["final_acc"] for s in common])
            y = np.array([arms["static"][s]["final_acc"] for s in common])
            d = x - y
            n = len(d)
            t, p = stats.ttest_rel(x, y)
            se = d.std(ddof=1) / math.sqrt(n)
            crit = stats.t.ppf(0.975, n - 1)
            pt = tost(d)
            verdict = "EQUIVALENT to static" if pt < 0.05 else (
                "BEATS static" if p < 0.05 and d.mean() > 0 else "inconclusive")
            print(f"  {a} - static: {d.mean():+6.3f}pp  "
                  f"95% CI [{d.mean()-crit*se:+.3f}, {d.mean()+crit*se:+.3f}]  "
                  f"p={p:.4f}  TOST={pt:.4f}  -> {verdict}")
            trend.setdefault(a, []).append((w, d.mean(), p, pt))

    print("\n" + "=" * 74)
    print("TREND: does the advantage grow as capacity shrinks?")
    print("=" * 74)
    for a, pts in trend.items():
        pts.sort()
        print(f"\n  {a} minus static, by width:")
        for w, m, p, pt in pts:
            bar = ("+" if m >= 0 else "-") * min(40, int(abs(m) * 20))
            print(f"    w{w:<3} {m:+6.3f}pp  p={p:.4f}  {bar}")
        if len(pts) < 3:
            print(f"    only {len(pts)} width(s) present; run the rest before reading a trend.")
            continue

        ws = np.array([np.log2(q[0]) for q in pts])
        ms = np.array([q[1] for q in pts])
        r = stats.pearsonr(ws, ms)[0]
        # A correlation across three widths has one degree of freedom and
        # essentially no power, so its p-value is not reported: it would
        # call a near-perfect monotone trend "not significant" and that
        # would be the same error as reading a Wilcoxon p at n=3. The
        # per-width tests below carry the claim instead; each of those has
        # six paired seeds behind it.
        print(f"    Pearson r with log2(width) = {r:+.3f} "
              f"(3 points, df=1; no p-value reported, see note in source)")

        smallest, largest = pts[0], pts[-1]
        small_wins = smallest[2] < 0.05 and smallest[1] > 0
        large_equiv = (not math.isnan(largest[3])) and largest[3] < 0.05

        # Monotonicity is checked explicitly. A verdict of the form "real at
        # the small width, gone at the large one" is true but misleading when
        # the middle width is the peak, because it implies a trend that is
        # not there.
        effects = [q[1] for q in pts]
        monotone = all(effects[i] >= effects[i + 1] for i in range(len(effects) - 1))
        peak = max(range(len(pts)), key=lambda i: effects[i])
        interior_peak = 0 < peak < len(pts) - 1

        # An interior peak is only meaningful if there is an effect to have
        # a shape. When every width is equivalent to static, the ordering of
        # the point estimates is seed noise and must not be described as a
        # peak. Equivalence is therefore checked first.
        all_equiv = all(q[3] < 0.05 for q in pts if not math.isnan(q[3]))
        any_sig = any(q[2] < 0.05 for q in pts)

        if all_equiv and not any_sig:
            print("    -> equivalent to static at every width tested. No effect at any")
            print("       capacity, so the ordering of the point estimates is noise.")
        elif interior_peak and not monotone and any_sig:
            print(f"    -> NON-MONOTONE. The effect peaks at w{pts[peak][0]} "
                  f"({effects[peak]:+.3f}pp), not at the smallest width.")
            print(f"       w{smallest[0]} {effects[0]:+.3f}pp, w{pts[peak][0]} {effects[peak]:+.3f}pp, "
                  f"w{largest[0]} {effects[-1]:+.3f}pp.")
            print("       Do not describe this as a monotone capacity trend. The shape")
            print("       is an inverted U and the peak width is itself a finding.")
        elif small_wins and large_equiv and monotone:
            print(f"    -> at w{smallest[0]} the advantage is real ({smallest[1]:+.3f}pp, "
                  f"p={smallest[2]:.4f}) and at w{largest[0]} it is gone, monotonically.")
            print("       The large-width null is a saturation effect.")
        elif all_equiv:
            print("    -> equivalent to static at every width tested, though at least")
            print("       one width also reached significance. Read the rows.")
        else:
            print("    -> mixed. Read the per-width rows above; no single summary fits.")

    print("\nSaturation check: a final train loss near 0.008 means the model")
    print("memorised the training set. A width where that no longer happens is")
    print("the one that answers the reviewer's objection.\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="cifar100 or tinyimagenet; default reports each separately")
    ap.add_argument("--kernels", type=int, default=4,
                    help="bank size to report; other K values are excluded")
    a = ap.parse_args()
    main(a.dataset, a.kernels)

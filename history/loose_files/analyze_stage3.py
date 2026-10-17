"""
Read Stage 3. Two independent questions, reported separately.

1. Does the bank-vs-routing decomposition replicate on Tiny-ImageNet?
2. Does temperature annealing change the CIFAR-100 conclusions?
"""
from __future__ import annotations
import glob, json, math
import numpy as np
from scipy import stats

MARGIN = 0.5


def load():
    R = {}
    for f in glob.glob("results/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "router" not in d or "config" not in d:
            continue
        c = d["config"]
        arm = "static" if c.get("scope") == "none" else d["router"]
        key = (c.get("dataset", "cifar100"), c.get("width", 64),
               c.get("num_kernels", 4), c.get("temp_start") is not None)
        R.setdefault(key, {}).setdefault(arm, {})[d["seed"]] = d["final_acc"]
    return R


def pair(x, y):
    d = x - y; n = len(d)
    t, p = stats.ttest_rel(x, y)
    se = d.std(ddof=1) / math.sqrt(n); crit = stats.t.ppf(0.975, n - 1)
    pl = stats.t.sf((d.mean() + MARGIN) / se, n - 1)
    pu = stats.t.cdf((d.mean() - MARGIN) / se, n - 1)
    return dict(m=d.mean(), lo=d.mean() - crit * se, hi=d.mean() + crit * se,
                p=float(p), tost=float(max(pl, pu)), n=n)


def arr(R, key, arm):
    if key not in R or arm not in R[key]:
        return None
    seeds = sorted(R[key][arm])
    return np.array([R[key][arm][s] for s in seeds]), seeds


def show(q, label):
    v = "EQUIVALENT" if q["tost"] < 0.05 and q["p"] >= 0.05 else (
        "BEATS" if q["p"] < 0.05 and q["m"] > 0 else
        "WORSE" if q["p"] < 0.05 else "inconclusive")
    print(f"    {label:<34} {q['m']:+7.3f}pp  [{q['lo']:+.3f},{q['hi']:+.3f}]  "
          f"p={q['p']:.5f}  TOST={q['tost']:.4f}  {v}")


def main():
    R = load()

    print("\n" + "=" * 82)
    print("PART 1  Does the decomposition replicate on Tiny-ImageNet?")
    print("=" * 82)
    for ds, w in [("cifar100", 32), ("tinyimagenet", 32)]:
        key = (ds, w, 4, False)
        got = {a: arr(R, key, a) for a in ["static", "uniform", "gap"]}
        if got["uniform"] is None:
            print(f"\n  {ds} w{w}: no uniform arm yet. "
                  f"Run: python run_stage3.py --part uniform")
            continue
        print(f"\n  {ds}, width {w}")
        for a in ["static", "uniform", "gap"]:
            v = got[a][0]
            print(f"    {a:<9} n={len(v)}  {v.mean():7.3f} +/- {v.std(ddof=1):.3f}")
        print()
        show(pair(got["uniform"][0], got["static"][0]), "bank    (uniform - static)")
        show(pair(got["gap"][0], got["uniform"][0]), "routing (gap - uniform)")
        show(pair(got["gap"][0], got["static"][0]), "total   (gap - static)")

    print("\n" + "=" * 82)
    print("PART 2  Does temperature annealing (tau 30 -> 1) change anything?")
    print("=" * 82)
    print("  Without annealing the mixture is near one-hot from step 0 and most of")
    print("  the bank is starved. DY-Conv reports annealing as necessary.\n")
    any_found = False
    for w in [64, 32]:
        const = arr(R, ("cifar100", w, 4, False), "gap")
        anneal = arr(R, ("cifar100", w, 4, True), "gap")
        stat = arr(R, ("cifar100", w, 4, False), "static")
        if anneal is None:
            print(f"  width {w}: no annealed runs yet. "
                  f"Run: python run_stage3.py --part temp")
            continue
        any_found = True
        print(f"  width {w}")
        print(f"    gap, tau=1 constant : {const[0].mean():7.3f} +/- {const[0].std(ddof=1):.3f}")
        print(f"    gap, tau 30 -> 1    : {anneal[0].mean():7.3f} +/- {anneal[0].std(ddof=1):.3f}")
        seeds = sorted(set(const[1]) & set(anneal[1]))
        ca = np.array([R[("cifar100", w, 4, False)]["gap"][s] for s in seeds])
        aa = np.array([R[("cifar100", w, 4, True)]["gap"][s] for s in seeds])
        show(pair(aa, ca), "annealed - constant")
        if stat is not None:
            sa = np.array([R[("cifar100", w, 4, False)]["static"][s] for s in seeds])
            show(pair(aa, sa), "annealed gap - static")
        print()

    if any_found:
        print("  Read: if annealed gap still equals static at width 64, the null holds")
        print("  under the published training recipe and the objection is closed. If it")
        print("  beats static, the width-64 null was an artefact of tau=1 and the paper")
        print("  must say so.")
    print()


if __name__ == "__main__":
    main()

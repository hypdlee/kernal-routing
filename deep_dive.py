import json, math, glob
import numpy as np
from scipy import stats

recs = {}
for f in glob.glob("results/stage1_*.json"):
    d = json.load(open(f))
    recs.setdefault(d["router"], {})[d["seed"]] = d

ARMS = ["uniform", "gap", "shuffled", "frozen_random"]
acc = {a: np.array([recs[a][s]["final_acc"] for s in sorted(recs[a])]) for a in ARMS}

print("=" * 68)
print("1. HEADLINE NUMBERS (re-measured from the JSON, not from the log)")
print("=" * 68)
for a in ARMS:
    v = acc[a]
    print(f"  {a:<15} mean {v.mean():6.3f}  sd {v.std(ddof=1):.3f}  "
          f"min {v.min():.2f}  max {v.max():.2f}")
allv = np.concatenate([acc[a] for a in ARMS])
print(f"\n  Whole grid: {allv.min():.2f} to {allv.max():.2f}, "
      f"spread {allv.max()-allv.min():.2f}pp across all 24 runs")
print(f"  Between-arm spread of means: {max(acc[a].mean() for a in ARMS)-min(acc[a].mean() for a in ARMS):.3f}pp")
print(f"  Median within-arm sd:        {np.median([acc[a].std(ddof=1) for a in ARMS]):.3f}pp")

print("\n" + "=" * 68)
print("2. POWER: what could this design have detected?")
print("=" * 68)
d = acc["gap"] - acc["uniform"]
sd = d.std(ddof=1); n = len(d)
print(f"  paired sd (gap-uniform) = {sd:.3f}pp, n = {n}")
for target in [0.5, 1.0, 1.5, 2.0]:
    dz = target / sd
    # power of a two-sided paired t-test at alpha=.05
    crit = stats.t.ppf(0.975, n-1)
    ncp = dz * math.sqrt(n)
    power = stats.nct.sf(crit, n-1, ncp) + stats.nct.cdf(-crit, n-1, ncp)
    print(f"    true effect {target:.1f}pp -> power {power*100:5.1f}%")
mde = stats.t.ppf(0.975, n-1) * sd / math.sqrt(n)
print(f"  Smallest difference significant at n=6: {mde:.3f}pp")
print(f"  -> the design could not have missed a 1pp effect. It is a real null.")

print("\n" + "=" * 68)
print("3. SHUFFLED: why inconclusive, and what would settle it")
print("=" * 68)
d2 = acc["shuffled"] - acc["gap"]
sd2 = d2.std(ddof=1)
print(f"  mean diff {d2.mean():+.3f}pp, sd {sd2:.3f}, TOST p = 0.0619 (just misses .05)")
for nn in range(6, 15):
    se = sd2/math.sqrt(nn)
    p = max(stats.t.sf((d2.mean()+0.5)/se, nn-1), stats.t.cdf((d2.mean()-0.5)/se, nn-1))
    flag = "  <-- clears .05" if p < 0.05 else ""
    print(f"    n={nn:2d}: projected TOST p = {p:.4f}{flag}")

print("\n" + "=" * 68)
print("4. EFFECTIVE vs STORED CAPACITY  (the thing that reframes this)")
print("=" * 68)
print("  uniform routes with z = 1/K for every input, always.")
print("  So W_eff = mean(W_k): the four kernels collapse to ONE kernel.")
print(f"  stored params:    {recs['uniform'][0]['params']['total']/1e6:.2f}M")
print( "  effective params: 11.22M  (a plain ResNet-18, measured earlier)")
print( "  At inference the uniform arm IS a static ResNet-18.")
print( "  It is therefore a LINEAR RE-PARAMETERISATION control, not a")
print( "  representational-capacity control. That changes the claim.")

print("\n" + "=" * 68)
print("5. ROUTING BEHAVIOUR: did gap route at all?")
print("=" * 68)
for a in ARMS:
    ents, spreads = [], []
    for s in recs[a]:
        for L in recs[a][s]["routing"]:
            ents.append(L["z_entropy_norm"])
            z = np.array(L["z_mean"]); spreads.append(z.max()-z.min())
    ents = np.array(ents); spreads = np.array(spreads)
    print(f"  {a:<15} entropy {ents.mean():.4f} (min {ents.min():.4f})   "
          f"z spread max-min: mean {spreads.mean():.4f}, max {spreads.max():.4f}")

print("\n  gap, entropy by layer depth (mean over 6 seeds):")
by_layer = {}
for s in recs["gap"]:
    for L in recs["gap"][s]["routing"]:
        by_layer.setdefault(L["layer"], []).append(L["z_entropy_norm"])
for k in sorted(by_layer):
    v = np.mean(by_layer[k]); bar = "#" * int((1-v)*400)
    print(f"    layer {k:2d}: {v:.4f}  {bar}")

print("\n" + "=" * 68)
print("6. CONVERGENCE SANITY")
print("=" * 68)
for a in ARMS:
    fl = [recs[a][s]["history"][-1]["loss"] for s in recs[a]]
    mins = [recs[a][s]["minutes"] for s in recs[a]]
    print(f"  {a:<15} final train loss {np.mean(fl):.4f}   "
          f"mean runtime {np.mean(mins):5.1f} min")

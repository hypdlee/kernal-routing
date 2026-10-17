"""Compare the Stage 1b static baseline against the Stage 1 arms."""
import glob, json, math
import numpy as np
from scipy import stats

def load(tag):
    out = {}
    for f in glob.glob(f"results/{tag}_*.json"):
        d = json.load(open(f))
        if d.get("tag") != tag: continue
        key = "static" if d["config"].get("scope") == "none" else d["router"]
        out.setdefault(key, {})[d["seed"]] = d["final_acc"]
    return out

a = load("stage1"); b = load("stage1b")
if not b:
    raise SystemExit("no stage1b results yet. run run_stage1b_static.py first.")
a.update(b)

print("\nAccuracy by arm")
for k in sorted(a, key=lambda k: -np.mean(list(a[k].values()))):
    v = np.array(list(a[k].values()))
    sd = v.std(ddof=1) if len(v) > 1 else float('nan')
    print(f"  {k:<15} n={len(v)}  {v.mean():6.3f} +/- {sd:.3f}")

if "static" not in a:
    raise SystemExit("no static arm found")

print("\nEverything vs static (paired by seed)")
for k in sorted(a):
    if k == "static": continue
    seeds = sorted(set(a[k]) & set(a["static"]))
    if len(seeds) < 2: continue
    x = np.array([a[k][s] for s in seeds]); y = np.array([a["static"][s] for s in seeds])
    d = x - y; n = len(d); sd = d.std(ddof=1)
    t, p = stats.ttest_rel(x, y)
    crit = stats.t.ppf(0.975, n-1); se = sd/math.sqrt(n)
    print(f"  {k:<15} {d.mean():+6.3f}pp  95% CI [{d.mean()-crit*se:+.3f},"
          f" {d.mean()+crit*se:+.3f}]  p={p:.4f}  n={n}")

print("\nRead: if every arm sits significantly above static by a similar")
print("margin, the kernel bank helps and routing does not. If every arm sits")
print("level with static, the bank is inert and the whole family is.")

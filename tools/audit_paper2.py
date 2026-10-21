"""Independent audit of the two-backbone paper. Run from the repository root:

    python tools/audit_paper2.py

It does not use the report pipeline's statistics code. It checks:
  1. every ResNet-18 per-seed accuracy against the per-run lines of the
     training logs or the result files, and the two Tiny-ImageNet width-32
     groups against the printed mean and sd;
  2. ResNet-18 width-64 routing entropies against the result files;
  3. the MobileNetV2 headline comparisons, recomputed with scipy from the 132
     result files (paired t, TOST at 0.5 pp);
  4. the method claims in the paper against the code (recipe, schedule, init,
     router shape, dynamic-layer count, last-batch size);
  5. the template for phrases that were found wrong and must not return.
Exit status 1 on any failure.
"""
import collections, glob, json, pathlib, re, sys
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report")); sys.path.insert(0, str(ROOT))
import paper_r18_data as P

fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


# 1. ResNet-18 accuracies ----------------------------------------------------
txt = ""
for f in glob.glob(str(ROOT / "artifacts/session_uploads/*.txt")) + glob.glob(str(ROOT / "artifacts/resnet18/logs/*.txt")):
    txt += open(f, errors="ignore").read() + "\n"
rows = re.findall(r"^\[(\w+) seed=(\d)\] final=([\d.]+) best=[\d.]+ -> results\\(?:[\w]+\\)?(\w+?)_seed\d\.json", txt, re.M)
D = {}
for arm, s, fin, tag in rows:
    if tag.startswith(("probe_", "smoketest")):
        continue
    k = (tag, arm, int(s)); v = float(fin)
    assert D.get(k, v) == v, f"duplicate logs disagree for {k}"
    D[k] = v
tagmap = {}
for w in (16, 32):
    for a in ("static", "uniform", "gap"):
        tagmap[(f"stage1c_w{w}_{a}", a)] = (("cifar100", w, 4, "const"), a)
tagmap[("stage1b_static", "static")] = (("cifar100", 64, 4, "const"), "static")
for K in (2, 8):
    tagmap[(f"stage2k_K{K}_gap", "gap")] = (("cifar100", 32, K, "const"), "gap")
for w in (16, 32, 64):
    tagmap[(f"stage3temp_w{w}_gap", "gap")] = (("cifar100", w, 4, "anneal"), "gap")
for a in ("static", "uniform", "gap"):
    tagmap[(f"tin_w64_{a}", a)] = (("tinyimagenet", 64, 4, "const"), a)
tagmap[("tin_w32_uniform", "uniform")] = (("tinyimagenet", 32, 4, "const"), "uniform")
bad = [(k, v, P.ACC[tagmap[(k[0], k[1])][0]][tagmap[(k[0], k[1])][1]][k[2]]) for k, v in D.items()
       if abs(v - P.ACC[tagmap[(k[0], k[1])][0]][tagmap[(k[0], k[1])][1]][k[2]]) > 1e-9]
check(f"R18 accuracies vs per-run log lines ({len(D)} runs)", not bad, str(bad[:3]))
for a, (m, sd) in {"static": (61.560, 0.217), "gap": (63.745, 0.260)}.items():
    v = np.array(P.ACC[("tinyimagenet", 32, 4, "const")][a])
    check(f"R18 TIN w32 {a} mean/sd vs printed summary", abs(v.mean() - m) < 5e-4 and abs(v.std(ddof=1) - sd) < 5e-4)
key = ("cifar100", 64, 4, "const")
res = ROOT / "artifacts/resnet18/stage1_results"
bad = [(a, s) for a in ("uniform", "gap", "shuffled", "frozen_random") for s in range(6)
       if abs(json.load(open(res / f"stage1_{a}_seed{s}.json"))["final_acc"] - P.ACC[key][a][s]) > 1e-9]
check("R18 CIFAR-100 w64 accuracies vs 24 result files", not bad, str(bad[:3]))

# 2. entropies ---------------------------------------------------------------
for a in ("gap", "shuffled", "frozen_random"):
    vals = [np.mean([l["z_entropy_norm"] for l in json.load(open(res / f"stage1_{a}_seed{s}.json"))["routing"]]) for s in range(6)]
    check(f"R18 w64 {a} entropy vs result files", np.allclose(vals, P.ENT_W64[a], atol=6e-5))
print("note  R18 entropies for CIFAR-100 w16, w32 and the three annealed groups (30 values) have no source file in the archive")

# 3. MobileNetV2 independent recompute ----------------------------------------
R = collections.defaultdict(dict)
for f in glob.glob(str(ROOT / "artifacts/mobilenetv2/results/mobilenetv2__*.json")):
    r = json.load(open(f)); c = r["config"]
    R[(c["dataset"], c["width"], c["num_kernels"], r["arm"], "ann" if c.get("temp_start") else "const")][r["seed"]] = r["final_acc"]
check("MobileNetV2: 132 result files", sum(len(v) for v in R.values()) == 132, str(sum(len(v) for v in R.values())))


def pair(a, b):
    ss = sorted(set(R[a]) & set(R[b])); d = np.array([R[a][s] - R[b][s] for s in ss]); n = len(d)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    return m, 2 * stats.t.sf(abs(m / se), n - 1), max(stats.t.sf((m + .5) / se, n - 1), stats.t.cdf((m - .5) / se, n - 1))


K = lambda ds, w, arm, k=4, t="const": (ds, w, k, arm, t)
exp = [  # (label, a, b, mean, p-lo, p-hi) from the paper's tables
    ("frozen-gap w64", K("cifar100", 64, "frozen_random"), K("cifar100", 64, "gap"), 0.152, 0.51, 0.53),
    ("shuffled-gap w64", K("cifar100", 64, "shuffled"), K("cifar100", 64, "gap"), -1.155, 0.001, 0.002),
    ("K8-K2 w32", K("cifar100", 32, "gap", 8), K("cifar100", 32, "gap", 2), 0.112, 0.59, 0.60),
    ("gap-uniform w64", K("cifar100", 64, "gap"), K("cifar100", 64, "uniform"), 1.473, 0, 0.0005),
    ("ann-const w32", K("cifar100", 32, "gap", 4, "ann"), K("cifar100", 32, "gap"), -1.277, 0.001, 0.002),
    ("ann-uniform w64", K("cifar100", 64, "gap", 4, "ann"), K("cifar100", 64, "uniform"), 0.552, 0.013, 0.014),
    ("uniform-static w32 (bank)", K("cifar100", 32, "uniform"), K("cifar100", 32, "static"), -0.102, 0.54, 0.56),
]
for lab, a, b, m0, plo, phi in exp:
    m, p, t = pair(a, b)
    check(f"MBv2 {lab}: mean {m:+.3f}, p {p:.4f}", abs(m - m0) < 6e-4 and plo <= p <= phi)
nb = sum(pair(K(ds, w, "uniform"), K(ds, w, "static"))[2] < .05 for ds, w in [("cifar100", 16), ("cifar100", 32), ("cifar100", 64), ("tinyimagenet", 32), ("tinyimagenet", 64)])
nr = sum(pair(K(ds, w, "gap"), K(ds, w, "uniform"))[1] < .05 for ds, w in [("cifar100", 16), ("cifar100", 32), ("cifar100", 64), ("tinyimagenet", 32), ("tinyimagenet", 64)])
check("MBv2 bank term equivalent in 2 of 5", nb == 2, str(nb))
check("MBv2 routing term significant in 5 of 5", nr == 5, str(nr))

# 4. method claims vs code ------------------------------------------------------
tr = (ROOT / "src/train.py").read_text(); rt = (ROOT / "src/routers.py").read_text()
md = (ROOT / "src/models/__init__.py").read_text(); rn = (ROOT / "src/resnet.py").read_text()
dc = (ROOT / "src/dynconv.py").read_text(); da = (ROOT / "src/data.py").read_text()
check("warm-up 5, epochs 150, lr 0.1, wd 5e-4", all(s in tr for s in ('"--warmup", type=int, default=5', '"--epochs", type=int, default=150', '"--lr", type=float, default=0.1', 'default=5e-4')))
check("cosine runs from the end of warm-up to the last step", "(step - warm) / max(1, total - warm)" in tr)
check("temperature 30 to 1 over 10 epochs, then constant", "min(1.0, epoch / max(1, args.temp_epochs))" in tr and '"--temp-epochs", type=int, default=10' in tr)
check("SGD Nesterov momentum 0.9", "momentum=0.9" in tr and "nesterov=True" in tr)
check("router hidden width max(16, C/4), ReLU", "hidden_ratio: float = 0.25, min_hidden: int = 16" in rt and "nn.ReLU" in rt)
check("frozen router fixed seed independent of run seed", "FROZEN_SEED = 20260908" in rt and "manual_seed(FROZEN_SEED)" in rt)
check("uniform uses the constant path", "is_constant = True" in rt and "if self.router.is_constant" in dc)
check("all training scripts pass --impl blend", all('"--impl", "blend"' in (ROOT / f).read_text() or '"--impl", args.impl' in (ROOT / f).read_text()
      for f in ("run_all_models.py", "run_ksweep.py", "run_stage3.py", "run_tin_w64.py", "run_tinyimagenet.py")))
check("R18 stem 3x3 stride 1, last BN of each block zero-initialised", "nn.Conv2d(3, width, 3, stride=1, padding=1" in rn and "nn.init.constant_(m.bn2.weight, 0.0)" in rn)
check("MobileNetV2 final 1x1 keeps 1280 at widths <= 64", "make_divisible(1280 * max(1.0, m))" in md)
check("crop padding 4 / 8, flip, eval batch >= 256 unshuffled", "pad=8" in da and "RandomHorizontalFlip" in da and "shuffle=False" in da and "max(batch_size, 256)" in da)
check("test set 10000 images -> last batch of 256 has 16", 10000 - (10000 // 256) * 256 == 16)
mb_r = [json.load(open(f)) for f in glob.glob(str(ROOT / "artifacts/mobilenetv2/results/mobilenetv2__cifar100__w64__K4__const_gap_seed*.json"))]
check("MBv2 gap w64 has 17 dynamic layers recorded", all(len(r["routing"]) == 17 for r in mb_r))
check("every MBv2 run: batch 128, no accumulation, 150 epochs, bf16", all(
    (r["accum"], r["config"]["batch_size"], r["config"]["epochs"], r["config"]["amp"]) == (1, 128, 150, "bf16")
    for f in glob.glob(str(ROOT / "artifacts/mobilenetv2/results/mobilenetv2__*.json")) for r in [json.load(open(f))]))

# 4b. cost numbers in the scope paragraph -------------------------------------
mins = collections.defaultdict(list)
for f in glob.glob(str(ROOT / "artifacts/mobilenetv2/results/mobilenetv2__*.json")):
    r = json.load(open(f)); c = r["config"]
    mins[(c["dataset"], c["width"], c["num_kernels"], "ann" if c.get("temp_start") else "const", r["arm"])].append(r["minutes"])
mm = lambda *k: float(np.mean(mins[k]))
M = P.MINS
ratios = [M[("tinyimagenet", 32, 4, "const")]["gap"] / M[("cifar100", 32, 4, "const")]["gap"],
          M[("tinyimagenet", 64, 4, "const")]["gap"] / M[("cifar100", 64, 4, "const")]["gap"],
          mm("tinyimagenet", 32, 4, "const", "gap") / mm("cifar100", 32, 4, "const", "gap"),
          mm("tinyimagenet", 64, 4, "const", "gap") / mm("cifar100", 64, 4, "const", "gap")]
check(f"Tiny-ImageNet/CIFAR-100 gap run cost 7 to 10x ({min(ratios):.2f} to {max(ratios):.2f})", 6.5 <= min(ratios) and max(ratios) <= 10.5)
# CIFAR-only groups repeated on Tiny-ImageNet, scaled by the width's gap ratio
def extra(rb_c, rb_t32, rb_t64, c32, c64, k2, k8, a32, a64, fr, sh):
    return 6 * (k2 * rb_t32 / c32 + k8 * rb_t32 / c32 + rb_t32 + rb_t64 + (fr + sh) * rb_t64 / c64) / 60
rb = extra(None, M[("tinyimagenet", 32, 4, "const")]["gap"], M[("tinyimagenet", 64, 4, "const")]["gap"],
           M[("cifar100", 32, 4, "const")]["gap"], M[("cifar100", 64, 4, "const")]["gap"],
           M[("cifar100", 32, 2, "const")]["gap"], M[("cifar100", 32, 8, "const")]["gap"], 0, 0,
           M[("cifar100", 64, 4, "const")]["frozen_random"], M[("cifar100", 64, 4, "const")]["shuffled"])
mb = extra(None, mm("tinyimagenet", 32, 4, "const", "gap"), mm("tinyimagenet", 64, 4, "const", "gap"),
           mm("cifar100", 32, 4, "const", "gap"), mm("cifar100", 64, 4, "const", "gap"),
           mm("cifar100", 32, 2, "const", "gap"), mm("cifar100", 32, 8, "const", "gap"), 0, 0,
           mm("cifar100", 64, 4, "const", "frozen_random"), mm("cifar100", 64, 4, "const", "shuffled"))
total = sum(sum(v) for v in mins.values()) / 60 + sum(6 * x for g in M.values() for x in g.values()) / 60
check(f"repeating the ablations on Tiny-ImageNet ~{rb + mb:.0f} GPU-h vs {total:.0f} for the study: 1.3 to 1.7x", 1.3 <= (rb + mb) / total <= 1.7)

# 5. phrases that must not return -------------------------------------------------
tex = (ROOT / "report/paper2_template.tex").read_text()
banned = ["Every run uses the right", "order of magnitude below", "GPU fault", "deeper stages",
          "so it does vary with", "every channel count by", "a few percent more compute",
          "five warm-up epochs then cosine decay over 150", "this literature reports", "close to one-hot from the first step", "At equal parameters, width wins",
          "—", "---"]
for b in banned:
    check(f"template does not contain {b!r}", b not in tex)
sys.exit(1 if fails else 0)

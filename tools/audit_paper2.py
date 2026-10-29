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
res = ROOT / "artifacts/resnet18/results"
bad = [(a, s) for a in ("uniform", "gap", "shuffled", "frozen_random") for s in range(6)
       if abs(json.load(open(res / f"stage1_{a}_seed{s}.json"))["final_acc"] - P.ACC[key][a][s]) > 1e-9]
check("R18 CIFAR-100 w64 accuracies vs 24 result files", not bad, str(bad[:3]))

# 2. all 132 ResNet-18 result files against the transcription ----------------
rfiles = sorted((ROOT / "artifacts/resnet18/results").glob("*_seed[0-5].json"))
check("R18: 132 result files", len(rfiles) == 132)
R18F = {}
fname = re.compile(r"^(stage1|stage1b|stage1c_w(16|32)|stage2k_K(2|8)|stage3temp_w(16|32|64)|tin_w(32|64))_(static|uniform|gap|shuffled|frozen_random)_seed(\d)\.json$")
for fp in rfiles:
    m = fname.match(fp.name); j = json.load(open(fp))
    tag, arm, s = m.group(1), m.group(6), int(m.group(7))
    if tag in ("stage1", "stage1b"): key = ("cifar100", 64, 4, "const")
    elif tag.startswith("stage1c"): key = ("cifar100", int(m.group(2)), 4, "const")
    elif tag.startswith("stage2k"): key = ("cifar100", 32, int(m.group(3)), "const")
    elif tag.startswith("stage3temp"): key = ("cifar100", int(m.group(4)), 4, "anneal")
    else: key = ("tinyimagenet", int(m.group(5)), 4, "const")
    ent = np.mean([l["z_entropy_norm"] for l in j["routing"]]) if j["routing"] else None
    R18F[(key, arm, s)] = (j["final_acc"], ent)
check("R18: every program cell has exactly one result file", len(R18F) == 132 and
      all((k, a, s) in R18F for k, arms in P.ACC.items() for a in arms for s in range(6)))
check("R18: every transcribed final accuracy equals its result file",
      all(R18F[(k, a, s)][0] == P.ACC[k][a][s] for k, arms in P.ACC.items() for a in arms for s in range(6)))
ok = all(round(R18F[(("cifar100", w, 4, rec), "gap", s)][1], 4) == P.ENT[("cifar100", w, rec)][s]
         for w in (16, 32, 64) for rec in ("const", "anneal") for s in range(6))
ok = ok and all(round(R18F[(("cifar100", 64, 4, "const"), a, s)][1], 4) == P.ENT_W64[a][s]
                for a in ("uniform", "gap", "shuffled", "frozen_random") for s in range(6))
check("R18: every transcribed routing entropy (the 30 earlier without a source included) equals its file to 4 decimals", ok)

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
check(f"ResNet-18/MobileNetV2 Tiny-ImageNet/CIFAR-100 gap run cost 7 to 10x ({min(ratios):.2f} to {max(ratios):.2f})", 6.5 <= min(ratios) and max(ratios) <= 10.5)
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
check(f"repeating the ablations on Tiny-ImageNet ~{rb + mb:.0f} GPU-h vs {total:.0f} for ResNet-18 and MobileNetV2: 1.3 to 1.7x", 1.3 <= (rb + mb) / total <= 1.7)

# 6. VGG-16: an independent parse of its logs, recomputed with scipy, against the built macros
import re as _re
def macros(path):
    tx = pathlib.Path(path).read_text(); out = {}
    for m in _re.finditer(r"\\expandafter\\def\\csname v(\w+)\\endcsname\{", tx):
        i = m.end(); d = 1; j = i
        while d:
            d += {"{": 1, "}": -1}.get(tx[j], 0); j += 1
        out[m.group(1)] = tx[i:j - 1]
    return out
built = ROOT / "report/out/paper2"
if not (built / "macros_mb.tex").exists():
    check("built macros present (run report/build_paper2.py first)", False)
else:
    MAC = {**macros(built / "macros.tex"), **macros(built / "macros_mb.tex")}
    V = collections.defaultdict(dict)
    log = (ROOT / "artifacts/vgg16/train.log").read_text(errors="replace")
    for m in _re.finditer(r"final=([\d.]+) best=[\d.]+ -> \S*?vgg16__(\w+?)__w(\d+)__K(\d+)__(const|anneal)_(\w+?)_seed(\d)\.json", log):
        V[(m.group(2), int(m.group(3)), int(m.group(4)), m.group(5), m.group(6))][int(m.group(7))] = float(m.group(1))
    check("VGG-16: 132 final lines in train.log", sum(len(v) for v in V.values()) == 132)
    run = (ROOT / "artifacts/vgg16/runner.log").read_text()
    fins = [float(x) for x in _re.findall(r"finished in [\d.]+ min, final ([\d.]+)", run)]
    check("VGG-16: runner log has 132 finished runs with the same finals as train.log",
          sorted(fins) == sorted(x for v in V.values() for x in v.values()))
    def vp(a, b):
        ss = sorted(set(V[a]) & set(V[b])); d = np.array([V[a][s] - V[b][s] for s in ss]); n = len(d)
        m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
        return m, 2 * stats.t.sf(abs(m / se), n - 1), max(stats.t.sf((m + .5) / se, n - 1), stats.t.cdf((m - .5) / se, n - 1))
    G = lambda ds, w, arm, k=4, r="const": (ds, w, k, r, arm)
    num = lambda s: float(s.replace("$<10^{-4}$", "0").replace("+", ""))
    tags = {("cifar100", 16): "CA", ("cifar100", 32): "CB", ("cifar100", 64): "CC", ("tinyimagenet", 32): "TA", ("tinyimagenet", 64): "TB"}
    for (ds, w), tg in tags.items():
        mb_, pb, tb = vp(G(ds, w, "uniform"), G(ds, w, "static"))
        mr, pr, _ = vp(G(ds, w, "gap"), G(ds, w, "uniform"))
        me, pe, _ = vp(G(ds, w, "gap"), G(ds, w, "static"))
        ok = (abs(mb_ - num(MAC[f"vgbank{tg}"])) < 6e-4 and abs(tb - num(MAC[f"vgbank{tg}T"])) < 6e-5
              and abs(mr - num(MAC[f"vgrout{tg}"])) < 6e-4 and abs(me - num(MAC[f"vgeff{tg}"])) < 6e-4
              and (MAC[f"vgrout{tg}p"] == "$<10^{-4}$" and pr < 1e-4 or abs(pr - num(MAC[f"vgrout{tg}p"])) < 6e-5))
        check(f"VGG-16 {ds} w{w}: bank {mb_:+.3f}, routing {mr:+.3f}, total {me:+.3f} match the macros", ok)
    for a, b, key in [("frozen_random", "gap", "frozenrandomgap"), ("frozen_random", "uniform", "frozenrandomuniform"),
                      ("gap", "uniform", "gapuniform"), ("shuffled", "gap", "shuffledgap"), ("shuffled", "static", "shuffledstatic")]:
        m_, p_, t_ = vp(G("cifar100", 64, a), G("cifar100", 64, b))
        check(f"VGG-16 w64 {a}-{b} {m_:+.3f} (p {p_:.4f}) matches", abs(m_ - num(MAC[f"vgw{key}"])) < 6e-4 and abs(p_ - num(MAC[f"vgw{key}p"])) < 6e-5)
    eff = []
    for w in (16, 32, 64):
        m_, p_, _ = vp(G("cifar100", w, "gap", 4, "anneal"), G("cifar100", w, "gap"))
        eff.append(p_ < .05 and m_ < 0)
    check("VGG-16 annealing lower at all three widths", all(eff) and MAC["vgannlower"] == "3")
    pp = [vp(G(ds, ws, "static"), G(ds, wg, "gap")) for ds, wg, ws in [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]]
    check("VGG-16 plain 2c beats gap c in 3 of 3, each p < 1e-4 (text says 'in each')", all(q[0] > 0 and q[1] < 1e-4 for q in pp))
    m_, p_, _ = vp(G("cifar100", 32, "gap", 8), G("cifar100", 32, "gap", 2))
    check(f"VGG-16 K8-K2 {m_:+.3f} inconclusive", abs(m_ - num(MAC["vgk82"])) < 6e-4 and MAC["vgk82v"] == "inconclusive")
    m_, p_, _ = vp(G("cifar100", 32, "gap", 2), G("cifar100", 32, "static"))
    check(f"VGG-16 K2-static {m_:+.3f} not significant (p {p_:.3f})", p_ >= .05)
    check("VGG-16 gate >= 65", V[G("cifar100", 64, "static")][0] >= 65)
    # cost ratio for the scope paragraph, VGG-16 from the runner log minutes
    vmins = collections.defaultdict(list); cur = None
    for l in run.splitlines():
        mm_ = _re.search(r"\[\d+/132\] vgg16__(\w+?)__w(\d+)__K(\d+)__(const|anneal) (\w+) seed", l)
        if mm_: cur = (mm_.group(1), int(mm_.group(2)), int(mm_.group(3)), mm_.group(4), mm_.group(5))
        mm_ = _re.search(r"finished in ([\d.]+) min", l)
        if mm_: vmins[cur].append(float(mm_.group(1)))
    vrat = [np.mean(vmins[("tinyimagenet", w, 4, "const", "gap")]) / np.mean(vmins[("cifar100", w, 4, "const", "gap")]) for w in (32, 64)]
    allrat = ratios + vrat
    check(f"Tiny-ImageNet/CIFAR-100 gap run cost over three backbones 4 to 10x ({min(allrat):.2f} to {max(allrat):.2f})", 3.5 <= min(allrat) and max(allrat) <= 10.5)
    for k_, v_ in [("allbankeq", "9"), ("allbanksig", "0"), ("alleff", "13"), ("alleffrout", "12"), ("allannlower", "7"),
                   ("allannhigher", "0"), ("allpred", "7"), ("allpredfail", "4"), ("totruns", "396"), ("ncontraX", "1"),
                   ("ncontra", "10"), ("ncontraHold", "4"), ("ncontraBD", "5"), ("vgP1", "2"), ("vgP2", "3"), ("vgP3", "4")]:
        check(f"macro {k_} = {v_} (ledger docs/FINDINGS_PAPER3.md)", MAC[k_] == v_)

# 7. pooled estimates and Holm counts: tools/reviewer_stats.py against the built macros
import runpy, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    RS = runpy.run_path(str(ROOT / "tools/reviewer_stats.py"))
if (built / "macros_mb.tex").exists():
    close = lambda a, key: abs(a - num(MAC[key])) < 6e-4
    for nm, ds_, pre in [("bank", RS["bank"], "poolbank"), ("routing", RS["rout"], "poolrout")]:
        re_, c95, c90, q = RS["pool"](ds_)
        ok = close(re_, pre) and close(c95[0], pre + "lo") and close(c95[1], pre + "hi") and close(c90[0], pre + "lonine") and close(c90[1], pre + "hinine")
        ok = ok and (MAC[pre + "Qp"] == "$<10^{-4}$" and q < 1e-4 or abs(q - num(MAC[pre + "Qp"])) < 6e-4)
        check(f"pooled {nm} term {re_:+.3f} [{c95[0]:+.3f}, {c95[1]:+.3f}], heterogeneity p {q:.4f} matches the macros", ok)
    for nm, ds_, key in [("routing", RS["rout"], "multrout"), ("total", RS["tot"], "multtot")]:
        ps = np.array([RS["pt"](d) for d in ds_]); pos = np.array([d.mean() > 0 for d in ds_])
        raw, h = int(((ps < .05) & pos).sum()), int((RS["holm"](ps) & pos).sum())
        check(f"Holm {nm}: {raw} uncorrected, {h} corrected, exactly one lost (text says 'the one lost')",
              MAC[key + "raw"] == str(raw) and MAC[key + "holm"] == str(h) and raw - h == 1)
    for m_, tg in [(.25, "A"), (.5, "B"), (.75, "C"), (1.0, "D")]:
        ps = np.array([RS["tost"](d, m_) for d in RS["bank"]])
        check(f"bank equivalences at +/-{m_}: {(ps < .05).sum()} / {RS['holm'](ps).sum()} match",
              MAC[f"multbank{tg}raw"] == str((ps < .05).sum()) and MAC[f"multbank{tg}holm"] == str(RS["holm"](ps).sum()))
    check("routing positive count in Section 5 (allrout) equals the uncorrected Holm-table count", MAC["allrout"] == MAC["multroutraw"])
    re_, c95, c90, q = RS["pool"](RS["bank"]); mg = num(MAC["poolbankmargin"])
    check(f"pooled bank 90% interval inside +/-{mg} and not inside +/-{mg - .05:.2f}", -mg < c90[0] and c90[1] < mg and not (-(mg - .05) < c90[0] and c90[1] < mg - .05))
    check("pooled bank margin is five times tighter than 0.5 pp (text says 'five times')", abs(0.5 / mg - 5) < 1e-9)
    worst = num(MAC["poolbankworst"])
    for i, b in enumerate(RS["BB"]):
        ds_ = RS["bank"][5 * i:5 * i + 5]
        m = np.array([d.mean() for d in ds_]); w = 1 / np.array([d.var(ddof=1) / len(d) for d in ds_])
        est = (w * m).sum() / w.sum(); se = np.sqrt(1 / w.sum()); pre = {"ResNet-18": "rb", "MobileNetV2": "mb", "VGG-16": "vg"}[b]
        check(f"{b} fixed-effect bank {est:+.3f}, 90% [{est - 1.645 * se:+.3f}, {est + 1.645 * se:+.3f}] matches and lies inside +/-{worst}",
              close(est, pre + "poolbank") and close(est - 1.645 * se, pre + "poolbanklonine") and close(est + 1.645 * se, pre + "poolbankhinine")
              and -worst < est - 1.645 * se and est + 1.645 * se < worst)
    check(f"plain ResNet-18 CIFAR-100 w64 {MAC['rbstaticCC']} is more than ten points above SD-Conv's 67.6 (text)", num(MAC["rbstaticCC"]) - 67.6 > 10)
    check("Tiny-ImageNet w64 ResNet-18 effect within 0.5 pp of DY-Conv's 2.3 ('of the published size')", abs(num(MAC["effTB"]) - 2.3) < 0.5)
    for k_ in ["rbstaticCC", "rbgapCC", "vgstaticCC", "vggapCC", "mbstaticCC", "mbgapCC", "rbstaticTB", "rbgapTB"]:
        check(f"macro {k_} present", k_ in MAC)
    six = [RS["get"](b, ds, ws, "static") - RS["get"](b, ds, wg, "gap") for b in ("ResNet-18", "VGG-16")
           for ds, wg, ws in [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]]
    check("plain 2c beats gap c in all six ResNet-18 and VGG-16 pairs ('six of six pairs')", all(d.mean() > 0 and RS["pt"](d) < .05 for d in six))

# 8. VGG-16 routing entropies: its result files, parsed here, against the log and the built macros
vfiles = sorted((ROOT / "artifacts/vgg16/results").glob("vgg16__*_seed[0-5].json"))
check("VGG-16: 132 result files", len(vfiles) == 132)
VF = {}
for fp in vfiles:
    m = _re.match(r"vgg16__(\w+?)__w(\d+)__K(\d)__(const|anneal)_(\w+)_seed(\d)\.json", fp.name); j = json.load(open(fp))
    VF[(m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5), int(m.group(6)))] = j
check("VGG-16: every result file's final accuracy equals train.log",
      all(VF[k]["final_acc"] == V[k[:5]][k[5]] for k in VF) and len(VF) == 132)
check("VGG-16: every routing record counts the same 16 test images",
      all(sum(l["argmax_counts"]) == 16 for j in VF.values() for l in j["routing"]))
VEn = lambda ds, w, a="gap", rec="const": np.array([np.mean([l["z_entropy_norm"] for l in VF[(ds, w, 4, rec, a, s)]["routing"]]) for s in range(6)])
VHh = lambda ds, w, a="gap", rec="const": 1 - VEn(ds, w, a, rec).mean()
if (built / "macros_mb.tex").exists():
    nm = lambda k: float(MAC[k])
    ok = all(abs(VHh("cifar100", w) - nm(f"vghc{t_}")) < 6e-5 and abs(VHh("cifar100", w, rec="anneal") - nm(f"vgha{t_}")) < 6e-6
             for w, t_ in [(16, "A"), (32, "B"), (64, "C")])
    check("VGG-16 constant and annealed 1-H at widths 16, 32, 64 match the macros", ok)
    check("VGG-16 w64 frozen and shuffled 1-H match the macros",
          abs(VHh("cifar100", 64, "frozen_random") - nm("vgeFrz")) < 6e-5 and abs(VHh("cifar100", 64, "shuffled") - nm("vgeShuf")) < 6e-5)
    six = [(VHh("cifar100", w, rec=r), np.mean([V[("cifar100", w, 4, r, "gap")][s] - V[("cifar100", w, 4, "const", "uniform")][s] for s in range(6)]))
           for w in (16, 32, 64) for r in ("const", "anneal")]
    rho = stats.spearmanr(*zip(*six))
    check(f"VGG-16 rank correlation {rho.statistic:.2f} (p {rho.pvalue:.3f}) matches, not significant (text)",
          abs(rho.statistic - nm("vgrho")) < 6e-3 and abs(rho.pvalue - nm("vgrhop")) < 6e-4 and rho.pvalue >= .05)
    ann_p = [vp(("cifar100", w, 4, "anneal", "gap"), ("cifar100", w, 4, "const", "uniform"))[1] for w in (16, 32, 64)]
    check("VGG-16: annealed router closest to uniform (w64) is the only annealed one with a significant benefit (text)",
          VHh("cifar100", 64, rec="anneal") < min(VHh("cifar100", w, rec="anneal") for w in (16, 32)) and ann_p[2] < .05 and min(ann_p[:2]) >= .05)
    b16c = vp(("cifar100", 16, 4, "const", "gap"), ("cifar100", 16, 4, "const", "uniform"))[0]
    b16a = vp(("cifar100", 16, 4, "anneal", "gap"), ("cifar100", 16, 4, "const", "uniform"))[0]
    check("VGG-16 w16: annealed router flatter and smaller benefit, so no counterexample (text)",
          VHh("cifar100", 16, rec="anneal") < VHh("cifar100", 16) and b16a < b16c)
    check("VGG-16 constant-tau routers move towards uniform as width grows (text: like ResNet-18)",
          VHh("cifar100", 16) > VHh("cifar100", 32) > VHh("cifar100", 64))
    flat = lambda w, rec: [int(sum(l["z_entropy_norm"] >= 0.9999 for l in VF[("cifar100", w, 4, rec, "gap", s)]["routing"])) for s in range(6)]
    fa = [x for w in (16, 32, 64) for x in flat(w, "anneal")]; fc = [x for w in (16, 32, 64) for x in flat(w, "const")]
    check(f"VGG-16 flat layers annealed {min(fa)}-{max(fa)}, constant {min(fc)}-{max(fc)} match",
          (str(min(fa)), str(max(fa)), str(min(fc)), str(max(fc))) == (MAC["vgflatAmin"], MAC["vgflatAmax"], MAC["vgflatCmin"], MAC["vgflatCmax"]))
    # cross-backbone claims of Section 7 and 11
    mbfile = lambda a: np.mean([np.mean([l["z_entropy_norm"] for l in json.load(open(ROOT / f"artifacts/mobilenetv2/results/mobilenetv2__cifar100__w64__K4__const_{a}_seed{s}.json"))["routing"]]) for s in range(6)])
    r18e = lambda a: np.mean([R18F[(("cifar100", 64, 4, "const"), a, s)][1] for s in range(6)])
    ok = all(1 - f_("gap") > max(1 - f_("frozen_random"), 1 - f_("shuffled"), 1 - f_("uniform")) for f_ in (mbfile, r18e))
    ok = ok and VHh("cifar100", 64) > max(VHh("cifar100", 64, a) for a in ("frozen_random", "shuffled", "uniform"))
    check("trained router furthest from uniform at C100 w64 on all three backbones (text)", ok)
    check("shuffled router near-uniform (1-H < 0.001) on ResNet-18 and MobileNetV2, not on VGG-16 (text)",
          1 - r18e("shuffled") < 1e-3 and 1 - mbfile("shuffled") < 1e-3 and VHh("cifar100", 64, "shuffled") > 1e-3)
    check("VGG-16 shuffled within 5% of frozen ('as far from uniform as the frozen router')",
          abs(VHh("cifar100", 64, "shuffled") / VHh("cifar100", 64, "frozen_random") - 1) < .05)
    hs = [1 - r18e("gap"), 1 - mbfile("gap"), VHh("cifar100", 64)]
    check(f"C100 w64 trained-router 1-H on the three backbones within a factor 1.2 ('almost the same'): {[round(h, 4) for h in hs]}",
          max(hs) / min(hs) < 1.2)
    check("ResNet-18 annealed w64 router equivalent to uniform; MobileNetV2 and VGG-16 annealed w64 routers higher (text)",
          MAC["vgbavC"] == "higher" and MAC["mbbavC"] == "higher")

# 5. phrases that must not return -------------------------------------------------
tex = (ROOT / "report/paper2_template.tex").read_text()
banned = ["Every run uses the right", "order of magnitude below", "GPU fault", "GPU\nfault", "on both backbones", "of ten conditions", "holds on both", "deeper stages",
          "so it does vary with", "every channel count by", "a few percent more compute",
          "five warm-up epochs then cosine decay over 150", "this literature reports", "close to one-hot from the first step", "At equal parameters, width wins",
          "not available (no result files)", "were not available for this", "two of them measured", "—", "---", "Moreover", "Furthermore", "Additionally", "Notably", "It is worth noting", "worth noting", "it is important to note"]
ai_words = ["robust", "comprehensive", "crucial", "delve", "leverag", "pivotal", "seamless", "meticulous", "underscore", "showcas",
            "landscape", "tapestry", "realm", "paradigm", "testament", "intricate", "holistic", "utiliz", "in order to", "genuinely", "truly"]
for b in ai_words:
    check(f"template avoids {b!r} (avoid-ai-writing pass)", b not in tex.lower())
for b in banned:
    check(f"template does not contain {b!r}", b not in tex)

# 9. routing controls (PREREGISTRATION_CONTROLS.md): files, logs, outcomes, dates
if (built / "macros_mb.tex").exists():
    import json as _j, subprocess as _sp, re as _r2
    sys.path.insert(0, str(ROOT / "tools"))
    import controls_analysis as _CA
    for m_, n_ in (("mobilenetv2", 24), ("vgg16", 36)):
        fs = sorted((ROOT / "artifacts" / "controls" / m_).glob("*_seed*.json"))
        check(f"{m_}: {n_} control result files", len(fs) == n_, str(len(fs)))
        seeds = {}
        for f_ in fs:
            r_ = _j.loads(f_.read_text()); seeds.setdefault((r_["tag"], r_["arm"]), set()).add(r_["seed"])
        check(f"{m_}: every control cell has seeds 0 to 5", all(v == set(range(6)) for v in seeds.values()), str(len(seeds)))
        _, probs = _CA.load(m_)
        check(f"{m_}: every final accuracy equals its train.log line", not probs, str(probs[:2]))
    check("macros: 60 control runs (24 + 36)", MAC["ctruns"] == "60" and MAC["ctmbruns"] == "24" and MAC["ctvgruns"] == "36")
    check("control predictions match docs/CONTROLS_RESULTS.md (tools/controls_analysis.py): C1 held, C2 held, C3 held, C4 failed",
          (MAC["ctC1o"], MAC["ctC2o"], MAC["ctC3o"], MAC["ctC4o"]) == ("held", "held", "held", "failed"))
    tpl_ = (ROOT / "report" / "paper2_template.tex").read_text()
    check("text says exactly one control prediction failed", "\\V{ctnfail}" in tpl_ and MAC["ctnfail"] == "1")
    log_ = _sp.run(["git", "log", "--format=%ad", "--date=iso", "--", "PREREGISTRATION_CONTROLS.md"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    first = _r2.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+\[1/\d+\] part A",
                       (ROOT / "artifacts" / "controls" / "mobilenetv2" / "runner.log").read_text())
    dated = [l for l in log_ if l.startswith("2026-10-22 19:15")]
    check("pre-registration commit 22 Oct 19:15 (+0200) precedes the first control run in runner.log", bool(dated) and bool(first) and first.group(1) > "2026-10-22 19:15:59", f"{dated[:1]} {first.group(1) if first else None}")
    check("no em-dash or double hyphen in the control text", "—" not in tpl_)

# 10. ResNet-18 controls reported with seed counts (PREREGISTRATION_CONTROLS_R18.md)
import json as _j2, subprocess as _sp2, re as _r3
sys.path.insert(0, str(ROOT / "tools"))
_spec = __import__("importlib.util").util.spec_from_file_location("r18c", ROOT / "tools" / "controls_r18_analysis.py")
_R18 = __import__("importlib.util").util.module_from_spec(_spec); _spec.loader.exec_module(_R18)
_A, _probs = _R18.load(ROOT)
check("ResNet-18 controls: every final accuracy equals its train.log line", not _probs, str(_probs[:2]))
_fs = [f for d in ("results_controls_a", "results_controls_b") for f in sorted((ROOT / d / "resnet18").glob("*_seed*.json"))]
check("macros: ResNet-18 control run count equals the result files", MAC["pdr18runs"] == str(len(_fs)), f"{MAC['pdr18runs']} vs {len(_fs)}")
check("ResNet-18 control files are unique runs", len({(_j2.loads(f.read_text())["tag"], _j2.loads(f.read_text())["arm"], _j2.loads(f.read_text())["seed"]) for f in _fs}) == len(_fs))
_n = {}
for f in _fs:
    r = _j2.loads(f.read_text()); _n[(r["tag"].split("__")[1], r["arm"])] = _n.get((r["tag"].split("__")[1], r["arm"]), 0) + 1
check("macros: seed counts shown for the incomplete cells match the files",
      MAC["pdr18TINshufn"] == str(_n.get(("tinyimagenet", "shuffled"), 0)) and MAC["pdr18TINfrzn"] == str(_n.get(("tinyimagenet", "frozen_random"), 0))
      and MAC["pdr18W16shufn"] == str(_n.get(("cifar100", "shuffled"), 0)) and MAC["pdr18W16frzn"] == str(_n.get(("cifar100", "frozen_random"), 0)), str(_n))
_rl = ((ROOT / "results_controls_a/resnet18/runner.log").read_text() + (ROOT / "results_controls_b/resnet18/runner.log").read_text())
_ts = sorted(_r3.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+\[\d+/\d+\] part", _rl))
_pc = _sp2.run(["git", "log", "-1", "--format=%ad", "--date=iso", "dbbacb8"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
check("ResNet-18 pre-registration commit (dbbacb8) precedes the first ResNet-18 control run", bool(_ts) and bool(_pc) and _pc[:19] < _ts[0], f"{_pc} vs {_ts[:1]}")
for nm in ("PDR18", "PDALLRUNS"):
    check(f"table tab_{nm}.tex exists and is non-empty", (built / f"tab_{nm}.tex").exists() and (built / f"tab_{nm}.tex").stat().st_size > 0)
_tb = (built / "tab_PDR18.tex").read_text().splitlines()
_bad = []; _cur = 6
for l in _tb:
    m = _r3.search(r"shuffled\} (\d) of 6 seeds, .*?frozen.*?\} (\d) of 6 seeds", l)
    if m: _cur = (int(m.group(1)), int(m.group(2))); continue
    if "equivalent" in l or "inconclusive" in l or "lower" in l or "higher" in l:
        n_ = _cur[1] if "frozen" in l.split("$-$")[0] else _cur[0]
        if n_ != 6 and f"(n={n_})" not in l: _bad.append(l[:50])
check("every ResNet-18 comparison with n<6 carries its n in the verdict", not _bad, str(_bad[:2]))
check("built paper has no em-dash", "\u2014" not in (built / "main.tex").read_text())
check("template mentions the n shown in every partial cell (sec:partial)", "sec:partial" in tpl_ and "app:partial" in tpl_)

sys.exit(1 if fails else 0)

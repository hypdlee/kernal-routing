"""Multiplicity and pooled analyses quoted in docs/KernelRoute_reviewer_risks.md.

    python tools/reviewer_stats.py

Holm-corrected counts over the 15 backbone x condition pairs, bank-term
equivalence at several margins, and DerSimonian-Laird random-effects pools of
the bank term and the routing term. Same per-seed values as the paper."""
import pathlib, sys
import numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))
import paper_r18_data as P, mbv2_log as L
MB = L.parse(ROOT / "artifacts/mobilenetv2/train.log"); VG = L.parse(ROOT / "artifacts/vgg16/train.log", "vgg16")
C = [("cifar100", 16), ("cifar100", 32), ("cifar100", 64), ("tinyimagenet", 32), ("tinyimagenet", 64)]


def get(b, ds, w, arm):
    if b == "ResNet-18":
        return np.array(P.ACC[(ds, w, 4, "const")][arm], float)
    R = MB if b == "MobileNetV2" else VG
    return np.array([R[(ds, w, 4, "const", arm, s)]["final"] for s in range(6)])


def tost(d, m):
    n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    return max(stats.t.sf((d.mean() + m) / se, n - 1), stats.t.cdf((d.mean() - m) / se, n - 1))


def pt(d):
    return 2 * stats.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d) - 1)


def holm(ps):
    o = np.argsort(ps); r = np.zeros(len(ps), bool)
    for i, j in enumerate(o):
        if ps[j] < .05 / (len(ps) - i): r[j] = True
        else: break
    return r


def pool(ds_):
    m = np.array([d.mean() for d in ds_]); v = np.array([d.var(ddof=1) / len(d) for d in ds_]); w = 1 / v
    fe = (w * m).sum() / w.sum(); Q = (w * (m - fe) ** 2).sum(); k = len(m)
    t2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); w2 = 1 / (v + t2)
    re = (w2 * m).sum() / w2.sum(); se = np.sqrt(1 / w2.sum())
    return re, (re - 1.96 * se, re + 1.96 * se), (re - 1.645 * se, re + 1.645 * se), stats.chi2.sf(Q, k - 1)


BB = ["ResNet-18", "MobileNetV2", "VGG-16"]
bank = [get(b, ds, w, "uniform") - get(b, ds, w, "static") for b in BB for ds, w in C]
rout = [get(b, ds, w, "gap") - get(b, ds, w, "uniform") for b in BB for ds, w in C]
tot = [get(b, ds, w, "gap") - get(b, ds, w, "static") for b in BB for ds, w in C]
for name, ds_ in [("routing term > 0", rout), ("gap - static > 0", tot)]:
    ps = np.array([pt(d) for d in ds_]); print(f"{name}: {(ps < .05).sum()} uncorrected, {holm(ps).sum()} after Holm")
for m in (.25, .5, .75, 1.0):
    ps = np.array([tost(d, m) for d in bank]); print(f"bank equivalent at +/-{m}: {(ps < .05).sum()} uncorrected, {holm(ps).sum()} after Holm")
for name, ds_ in [("bank", bank), ("routing", rout)]:
    re, c95, c90, q = pool(ds_)
    print(f"{name} pooled {re:+.3f}  95% [{c95[0]:+.3f}, {c95[1]:+.3f}]  90% [{c90[0]:+.3f}, {c90[1]:+.3f}]  heterogeneity p {q:.4f}")

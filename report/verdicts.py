"""
Per-model factual statements, computed, never typed: written to verdicts.tex
as \\V{...} macros, plus tab_INTERP.tex (equal-parameter comparison against
an interpolated static curve) and tab_COVER.tex (coverage and deviations).

Every sentence here states what a test returned. None interprets. The
interpretation belongs to the cross-model report and to the authors.
"""
import math, pathlib, sys, json
import numpy as np
from scipy import stats
from data import *
import data as _D

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_all_models import GATE_FLOOR

M = {}
put = lambda k, v: M.__setitem__(k, v)
f = lambda x, n=3: "--" if isinstance(x, float) and math.isnan(x) else f"{x:.{n}f}"
sg = lambda x, n=3: "--" if isinstance(x, float) and math.isnan(x) else f"{x:+.{n}f}"
tt = lambda a: "\\texttt{" + a.replace("_", "\\_") + "}"
DSN = {"cifar100": "CIFAR-100", "tinyimagenet": "Tiny-ImageNet"}
NAME = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "vgg16": "VGG-16 (BN)",
        "mobilenetv2": "MobileNetV2", "convnext_tiny": "ConvNeXt-Tiny"}
PLACE = {"resnet18": "the two $3\\times3$ convolutions of every basic block",
         "resnet50": "the $3\\times3$ convolution of every bottleneck block",
         "vgg16": "every $3\\times3$ convolution after the first",
         "mobilenetv2": "the $3\\times3$ depthwise convolution of every inverted-residual block",
         "convnext_tiny": "the $7\\times7$ depthwise convolution of every block"}
NDYN = {"resnet18": 16, "resnet50": 16, "vgg16": 12, "mobilenetv2": 17, "convnext_tiny": 18}
INIT = {"resnet18": "static convolutions keep PyTorch's default initialisation while every bank kernel is "
                    "Kaiming-normal (fan-out), exactly as in the 132 original runs; this mismatch is disclosed "
                    "in the ResNet-18 paper",
        "resnet50": "Kaiming-normal (fan-out) for every static convolution and every bank kernel",
        "vgg16": "Kaiming-normal (fan-out) for every static convolution and every bank kernel",
        "mobilenetv2": "Kaiming-normal (fan-out) for every static convolution and every bank kernel",
        "convnext_tiny": "truncated normal, std 0.02, for every static convolution and every bank kernel"}
m = _D.MODEL
put("preregsource", "Recorded in writing, dated, before the Tiny-ImageNet width-64 runs of the ResNet-18 study."
    if m == "resnet18" else "Recorded in \\texttt{PREREGISTRATION\\_MODELS.md} on 30 September 2026, before any run of this backbone.")
put("modelname", NAME[m]); put("placement", PLACE[m]); put("ndyn", str(NDYN[m])); put("initdesc", INIT[m])
put("synthbanner", "\\fbox{\\parbox{0.95\\textwidth}{\\centering\\color{red}\\bfseries SYNTHETIC TEST DATA. "
    "Every number in this document is generated to test the pipeline. None is a result.}}"
    if _D.SYNTHETIC else ("\\fbox{\\parbox{0.95\\textwidth}{\\centering\\bfseries " + _D.BANNER + "}}" if _D.BANNER else ""))
cond = [("cifar100", 16), ("cifar100", 32), ("cifar100", 64), ("tinyimagenet", 32), ("tinyimagenet", 64)]
clab = lambda ds, w: f"{DSN[ds]} width {w}"

# setup and gate
key64 = ("cifar100", 64, 4, "const")
put("pstatic", f(PARAMS[key64]["static"], 2)); put("pgap", f(PARAMS[key64]["gap"], 2))
put("prouter", f(ROUTER_K.get(("cifar100", 64), float("nan")), 1))
put("prouterpct", f(100 * ROUTER_K.get(("cifar100", 64), float("nan")) / 1e3 / PARAMS[key64]["static"], 1))
put("pbankadd", f(PARAMS[key64]["uniform"] - PARAMS[key64]["static"], 2))
g0 = float(np.asarray(ACC[key64]["static"])[0])
floor = GATE_FLOOR.get(m, float("nan"))
put("gateacc", f(g0, 2)); put("gatefloor", f(floor, 1))
put("gateverdict", "passed" if g0 >= floor else ("failed" if not math.isnan(g0) else "not run"))

# coverage
n = n_runs(); put("nruns", str(n)); put("nmissing", str(132 - n)); put("ndiverged", str(len(_D.DIVERGED)))
dev = sorted({(k, tuple(v)) for (k, a), v in _D.ACCUM.items() if v and v != [1]})
put("ndevgroups", str(len({k for k, _ in dev})))
rows = []
for (key, arm, s) in _D.MISSING:
    name = f"{m}__{key[0]}__w{key[1]}__K{key[2]}__{key[3]}_{arm}_seed{s}.json"
    why = "diverged (non-finite loss)" if name in _D.DIVERGED else "not run"
    rows.append(f"{DSN[key[0]]} & {key[1]} & {key[2]} & {key[3]} & {tt(arm)} & {s} & {why} \\\\")
groups = {}
for (k, a), v in _D.ACCUM.items():
    if v:
        groups.setdefault(k, {})[a] = v
for key in sorted(groups):
    arms_v = groups[key]
    if all(v == [1] for v in arms_v.values()):
        continue
    mb = lambda v: "/".join(str(128 // x) for x in v)
    if len({tuple(v) for v in arms_v.values()}) == 1:
        v = next(iter(arms_v.values()))
        rows.append(f"{DSN[key[0]]} & {key[1]} & {key[2]} & {key[3]} & all arms & -- & micro-batch "
                    f"{mb(v)} (accumulation {'/'.join(map(str, v))}) \\\\")
    else:                       # raised mid-group after an out-of-memory: list every arm
        for a, v in sorted(arms_v.items()):
            rows.append(f"{DSN[key[0]]} & {key[1]} & {key[2]} & {key[3]} & {tt(a)} & -- & micro-batch {mb(v)} \\\\")
for r in _D.RESUMED:
    rows.append(f"\\multicolumn{{6}}{{l}}{{{r}}} & resumed from checkpoint \\\\")
if _D.IGNORED:
    rows.append(f"\\multicolumn{{6}}{{l}}{{{len(_D.IGNORED)} result file(s) outside the program}} & ignored, not averaged \\\\")
if not rows:
    rows = ["\\multicolumn{7}{l}{none: all 132 runs present, batch 128 throughout, no resumed run} \\\\"]
pathlib.Path("tab_COVER.tex").write_text("\n".join(rows) + "\n")


def vs(ds, w, a, b, k=4, rec="const", kb=4, recb="const"):
    return paired(acc(ds, w, a, k=k, rec=rec), acc(ds, w, b, k=kb, rec=recb))


# width 64
w64 = [(a, vs("cifar100", 64, a, "static")) for a in ["uniform", "gap", "shuffled", "frozen_random"]]
put("wsixfour", "; ".join(f"{tt(a)} {sg(q['mean'])}\\,pp ({verdict(q)})" for a, q in w64))
put("wsixfourneq", str(sum(verdict(q).startswith("equivalent") for _, q in w64)))

# decomposition
bank = [vs(ds, w, "uniform", "static") for ds, w in cond]
rout = [vs(ds, w, "gap", "uniform") for ds, w in cond]
put("nbankeq", str(sum(verdict(q).startswith("equivalent") for q in bank)))
put("bankother", "; ".join(f"{clab(*c)}: {verdict(q)} ({sg(q['mean'])}\\,pp)" for c, q in zip(cond, bank)
                           if not verdict(q).startswith("equivalent")) or "none")
put("nroutpos", str(sum(q["p"] < 0.05 and q["mean"] > 0 for q in rout)))
put("routlist", "; ".join(f"{clab(*c)} {sg(q['mean'])}\\,pp ($p={f(q['p'], 4)}$)" for c, q in zip(cond, rout)))

# capacity sweep (gap - static by width, CIFAR)
cap = [vs("cifar100", w, "gap", "static") for w in (16, 32, 64)]
put("caplist", ", ".join(f"{sg(q['mean'])}\\,pp at width {w} ({verdict(q)})" for w, q in zip((16, 32, 64), cap)))

# K sweep
q = paired(acc("cifar100", 32, "gap", k=8), acc("cifar100", 32, "gap", k=2))
put("keight", sg(q["mean"])); put("keightv", verdict(q))
put("kparratio", f(PARAMS[("cifar100", 32, 8, "const")]["gap"] / PARAMS[("cifar100", 32, 2, "const")]["gap"], 2))

# Tiny-ImageNet
for w in (32, 64):
    q = vs("tinyimagenet", w, "gap", "static")
    put(f"tin{w}", sg(q["mean"])); put(f"tin{w}v", verdict(q))

# saturation: does static training loss order the effect? (Spearman over 5 conditions)
Ls = [LOSS[(ds, w, 4, "const")]["static"] for ds, w in cond]
Es = [vs(ds, w, "gap", "static")["mean"] for ds, w in cond]
ok = [i for i in range(5) if not (math.isnan(Ls[i]) or math.isnan(Es[i]))]
rho = stats.spearmanr([Ls[i] for i in ok], [Es[i] for i in ok]).statistic if len(ok) >= 3 else float("nan")
put("satrho", sg(rho, 2))

# matched budgets and the interpolated comparison
pairs = [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]
mt, it_rows = [], []
for ds, wg, ws in pairs:
    q = paired(acc(ds, ws, "static"), acc(ds, wg, "gap"))
    ratio = PARAMS[(ds, wg, 4, "const")]["gap"] / PARAMS[(ds, ws, 4, "const")]["static"]
    mt.append((ds, wg, ws, q, ratio))
    # static accuracy interpolated linearly in log(params) through the static
    # widths of the same dataset, evaluated at the gap network's parameter count
    wl = [16, 32, 64] if ds == "cifar100" else [32, 64]
    xs = np.log([PARAMS[(ds, w, 4, "const")]["static"] for w in wl])
    ys = [acc(ds, w, "static").mean() for w in wl]
    xg = math.log(PARAMS[(ds, wg, 4, "const")]["gap"])
    inside = xs[0] <= xg <= xs[-1]
    si = float(np.interp(xg, xs, ys)) if inside else float("nan")
    gm = acc(ds, wg, "gap").mean()
    it_rows.append(f"{DSN[ds]} & {tt('gap')} w{wg} & {f(PARAMS[(ds, wg, 4, 'const')]['gap'], 2)} & "
                   f"{f(gm)} & {f(si) if inside else 'outside the static range'} & "
                   f"{sg(si - gm) if inside else '--'} \\\\")
put("nmatchedpairs", str(sum(abs(r - 1) <= 0.10 for *_, r in mt)))
put("matchlist", "; ".join(
    f"{DSN[ds]} {tt('gap')} w{wg} vs {tt('static')} w{ws}: {sg(q['mean'])}\\,pp, parameter ratio {r:.2f}"
    f" ({'matched' if abs(r - 1) <= 0.10 else 'not matched'})" for ds, wg, ws, q, r in mt))
pathlib.Path("tab_INTERP.tex").write_text("\n".join(it_rows) + "\n")

# temperature
tl = []
for w in (16, 32, 64):
    q = paired(acc("cifar100", w, "gap", rec="anneal"), acc("cifar100", w, "gap"))
    tl.append(f"width {w} {sg(q['mean'])}\\,pp ({verdict(q)})")
put("templist", "; ".join(tl))

# entropy vs benefit (6 points: 3 widths x 2 recipes)
xs, ys = [], []
for w in (16, 32, 64):
    for rec in ("const", "anneal"):
        e = ENT.get(("cifar100", w, rec))
        if e is None:
            continue
        xs.append(1 - e.mean()); ys.append(paired(acc("cifar100", w, "gap", rec=rec), acc("cifar100", w, "uniform"))["mean"])
rho = stats.spearmanr(xs, ys).statistic if len(xs) >= 3 else float("nan")
put("entrho", sg(rho, 2)); put("entn", str(len(xs)))
e16 = [1 - ENT[("cifar100", 16, r)].mean() for r in ("const", "anneal")]
b16 = [paired(acc("cifar100", 16, "gap", rec=r), acc("cifar100", 16, "uniform"))["mean"] for r in ("const", "anneal")]
put("flatratio", f(e16[0] / e16[1], 1)); put("bsixc", sg(b16[0])); put("bsixa", sg(b16[1]))

out = [f"\\expandafter\\def\\csname v{k}\\endcsname{{{v}}}" for k, v in M.items()]
with open("verdicts.tex", "w") as fh:
    fh.write("\n".join(out) + "\n")
print(len(M), "verdict macros; tab_INTERP, tab_COVER written")

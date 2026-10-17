"""
Cross-model report: the five backbones side by side.

    python report/cross.py --results resnet18=results mobilenetv2=results_models/mobilenetv2 ...
    python report/cross.py --synthetic          # pipeline test, watermarked

Figures X1-X6 and tables XT1-XT9 are computed from the same data module as
the per-model reports, so a number here always equals the number there.
Models whose gate failed or whose results are absent are listed as such, not
dropped.
"""
import argparse, importlib.util, json, math, pathlib, re, subprocess, sys, tempfile
import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["resnet18", "mobilenetv2", "vgg16", "resnet50", "convnext_tiny"]
NAME = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "vgg16": "VGG-16", "mobilenetv2": "MobileNetV2",
        "convnext_tiny": "ConvNeXt-T"}
COL = {"resnet18": "#444444", "mobilenetv2": "#1f77b4", "vgg16": "#2ca02c", "resnet50": "#d62728",
       "convnext_tiny": "#9467bd"}
COND = [("cifar100", 16), ("cifar100", 32), ("cifar100", 64), ("tinyimagenet", 32), ("tinyimagenet", 64)]
CL = {("cifar100", 16): "C16", ("cifar100", 32): "C32", ("cifar100", 64): "C64",
      ("tinyimagenet", 32): "T32", ("tinyimagenet", 64): "T64"}
PAIRS = [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]
plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "legend.frameon": False})
f = lambda x, n=3: "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
sg = lambda x, n=3: "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{n}f}"
pv = lambda p: "--" if p is None or math.isnan(p) else ("$<10^{-4}$" if p < 1e-4 else f"{p:.4f}")
EQ = "$\\equiv$"
tt = lambda a: "\\texttt{" + a.replace("_", "\\_") + "}"


def fresh_data():
    spec = importlib.util.spec_from_file_location("xdata", HERE / "data.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def summarise(model, dirs):
    """Everything the cross-model report needs from one model, as plain
    numbers. Returns None if the model has no results."""
    D = fresh_data()
    D.load(model, dirs)
    if D.n_runs() == 0:
        return None
    a = D.acc; P = D.paired
    S = {"model": model, "runs": D.n_runs(), "diverged": len(D.DIVERGED), "gpuh": D.gpu_hours(),
         "dev_groups": len({k for (k, _), v in D.ACCUM.items() if v and v != [1]})}
    k64 = ("cifar100", 64, 4, "const")
    S["static64"] = a("cifar100", 64, "static").mean()
    S["gate"] = float(np.asarray(a("cifar100", 64, "static"))[0])
    S["p_static"] = D.PARAMS[k64]["static"]; S["p_gap"] = D.PARAMS[k64]["gap"]
    S["router"] = D.ROUTER_K[("cifar100", 64)] / 1e3
    # share of the static network's parameters that sit in dynamic layers:
    # a K=4 bank adds 3x each dynamic layer's weights
    S["dyn_share"] = (D.PARAMS[k64]["uniform"] - D.PARAMS[k64]["static"]) / (3 * D.PARAMS[k64]["static"])
    S["bank"] = {c: P(a(*c, "uniform"), a(*c, "static")) for c in COND}
    S["rout"] = {c: P(a(*c, "gap"), a(*c, "uniform")) for c in COND}
    S["total"] = {c: P(a(*c, "gap"), a(*c, "static")) for c in COND}
    S["verdict"] = D.verdict
    S["pairs"] = {}
    for ds, wg, ws in PAIRS:
        q = P(a(ds, ws, "static"), a(ds, wg, "gap"))
        pg = D.PARAMS[(ds, wg, 4, "const")]["gap"]; ps = D.PARAMS[(ds, ws, 4, "const")]["static"]
        wl = [16, 32, 64] if ds == "cifar100" else [32, 64]
        xs = np.log([D.PARAMS[(ds, w, 4, "const")]["static"] for w in wl])
        ys = [a(ds, w, "static").mean() for w in wl]
        inside = xs[0] <= math.log(pg) <= xs[-1]
        interp = float(np.interp(math.log(pg), xs, ys)) - a(ds, wg, "gap").mean() if inside else float("nan")
        S["pairs"][(ds, wg, ws)] = dict(q=q, ratio=pg / ps, interp=interp)
    S["temp"] = {w: P(a("cifar100", w, "gap", rec="anneal"), a("cifar100", w, "gap")) for w in (16, 32, 64)}
    S["k82"] = P(a("cifar100", 32, "gap", k=8), a("cifar100", 32, "gap", k=2))
    S["ent"] = []
    for w in (16, 32, 64):
        for rec in ("const", "anneal"):
            e = D.ENT.get(("cifar100", w, rec))
            if e is not None:
                S["ent"].append((w, rec, 1 - e.mean(),
                                 P(a("cifar100", w, "gap", rec=rec), a("cifar100", w, "uniform"))["mean"]))
    xs = [e[2] for e in S["ent"]]; ys = [e[3] for e in S["ent"]]
    S["entrho"] = stats.spearmanr(xs, ys).statistic if len(xs) >= 3 else float("nan")
    S["curve"] = {arm: [(D.PARAMS[("cifar100", w, 4, "const")][arm], a("cifar100", w, arm).mean())
                        for w in (16, 32, 64)] for arm in ("static", "uniform", "gap")}
    nb = sum(q["tost"] < 0.05 for q in S["bank"].values())
    npr = sum(v["q"]["p"] < 0.05 and v["q"]["mean"] > 0 for v in S["pairs"].values())
    nr = sum(q["p"] < 0.05 and q["mean"] > 0 for q in S["rout"].values())
    S["prereg"] = (nb, npr, nr)
    return S


def forest(ax, Ss, key, title, margin=True):
    y = 0; ticks, labels = [], []
    for S in Ss:
        for c in COND:
            q = S[key][c]
            ax.errorbar(q["mean"], y, xerr=[[q["mean"] - q["lo"]], [q["hi"] - q["mean"]]], fmt="o", ms=3.5,
                        capsize=2, color=COL[S["model"]])
            ticks.append(y); labels.append(f"{NAME[S['model']]} {CL[c]}"); y -= 1
        y -= 0.8
    ax.axvline(0, color="k", lw=.8)
    if margin:
        ax.axvspan(-.5, .5, color="grey", alpha=.13, label="$\\pm0.5$ pp margin")
        ax.legend(fontsize=7, loc="lower right")
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.3); ax.set_title(title, fontsize=9)
    ax.set_xlabel("$\\Delta$ accuracy (pp), 95% CI")


def mark(fig, synthetic):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC TEST DATA", fontsize=28, color="red", alpha=0.25, ha="center",
                 va="center", rotation=20, transform=fig.transFigure)


def figures(Ss, out, synthetic):
    (out / "fig").mkdir(parents=True, exist_ok=True)
    h = 0.13 * 6 * len(Ss) + 0.6
    fig, ax = plt.subplots(1, 2, figsize=(7.6, h), sharey=True)
    forest(ax[0], Ss, "bank", "(a) bank term: uniform $-$ static")
    forest(ax[1], Ss, "rout", "(b) routing term: gap $-$ uniform", margin=False)
    mark(fig, synthetic); plt.savefig(out / "fig/x1_decomp.pdf", metadata={"CreationDate": None}); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for S in Ss:
        for k, (ds, ws) in enumerate([("cifar100", [16, 32, 64]), ("tinyimagenet", [32, 64])]):
            q = [S["total"][(ds, w)] for w in ws]
            ax[k].errorbar(ws, [e["mean"] for e in q], yerr=[[e["mean"] - e["lo"] for e in q], [e["hi"] - e["mean"] for e in q]],
                           marker="o", ms=4, capsize=3, lw=1.3, color=COL[S["model"]], label=NAME[S["model"]])
    for k, t in enumerate(["(a) CIFAR-100", "(b) Tiny-ImageNet"]):
        ax[k].axhline(0, color="k", lw=.8); ax[k].axhspan(-.5, .5, color="grey", alpha=.13)
        ax[k].set_xscale("log", base=2); ax[k].set_xticks([16, 32, 64]); ax[k].set_xticklabels([16, 32, 64])
        ax[k].set_xlabel("base width $c$"); ax[k].set_title(t, fontsize=9)
    ax[0].set_ylabel("gap $-$ static (pp)"); ax[0].legend(fontsize=7)
    mark(fig, synthetic); plt.savefig(out / "fig/x2_total.pdf", metadata={"CreationDate": None}); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.2))
    xs = np.arange(len(PAIRS)); bw = 0.8 / len(Ss)
    for j, S in enumerate(Ss):
        q = [S["pairs"][p]["q"] for p in PAIRS]
        pos = xs + (j - (len(Ss) - 1) / 2) * bw
        ax[0].bar(pos, [e["mean"] for e in q], bw, color=COL[S["model"]], label=NAME[S["model"]],
                  yerr=[[e["mean"] - e["lo"] for e in q], [e["hi"] - e["mean"] for e in q]], capsize=2, error_kw=dict(lw=.8))
        for x_, p, e in zip(pos, PAIRS, q):
            r = S["pairs"][p]["ratio"]
            ax[0].annotate(f"{r:.2f}", (x_, max(e["hi"], 0)), xytext=(0, 2), textcoords="offset points",
                           ha="center", va="bottom", fontsize=5, rotation=90,
                           color="k" if abs(r - 1) <= .1 else "#b00")
        ax[1].scatter(pos, [S["pairs"][p]["interp"] for p in PAIRS], color=COL[S["model"]], s=18, zorder=3)
    lab = ["C-100\n16 vs 32", "C-100\n32 vs 64", "TIN\n32 vs 64"]
    for k in (0, 1):
        ax[k].axhline(0, color="k", lw=.8); ax[k].set_xticks(xs); ax[k].set_xticklabels(lab, fontsize=6.5)
    ax[0].set_ylabel("static(2c) $-$ gap(c) (pp)"); ax[0].legend(fontsize=6, ncol=2, loc="upper right")
    ax[0].margins(y=0.25)
    ax[0].set_title("(a) width doubling; numbers: param ratio (red: not matched)", fontsize=8)
    ax[1].set_ylabel("static(equal params) $-$ gap (pp)")
    ax[1].set_title("(b) interpolated static at equal parameters", fontsize=8)
    mark(fig, synthetic); plt.savefig(out / "fig/x3_matched.pdf", metadata={"CreationDate": None}); plt.close()

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for S in Ss:
        for w, rec, x, y in S["ent"]:
            ax.scatter(x, y, color=COL[S["model"]], marker="o" if rec == "const" else "^", s=26, zorder=3)
        ax.plot([], [], "o", color=COL[S["model"]], label=f"{NAME[S['model']]} ($\\rho={S['entrho']:+.2f}$)")
    ax.plot([], [], "ko", mfc="none", label="$\\tau=1$"); ax.plot([], [], "k^", mfc="none", label="$\\tau:30\\to1$")
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("$1-\\tilde H$ (log)"); ax.set_ylabel("gap $-$ uniform (pp)"); ax.legend(fontsize=6.3)
    mark(fig, synthetic); plt.savefig(out / "fig/x4_entropy.pdf", metadata={"CreationDate": None}); plt.close()

    n = len(Ss); fig, ax = plt.subplots(1, n, figsize=(1.6 * n + 0.6, 2.4))
    ax = np.atleast_1d(ax)
    for k, S in enumerate(Ss):
        for arm, c in [("static", "#444444"), ("uniform", "#1f77b4"), ("gap", "#d62728")]:
            p, m = zip(*S["curve"][arm]); ax[k].plot(p, m, marker="o", ms=3, lw=1.2, color=c, label=arm)
        ax[k].set_xscale("log"); ax[k].set_title(NAME[S["model"]], fontsize=8); ax[k].tick_params(labelsize=6)
        ax[k].set_xlabel("params (M)", fontsize=7)
    ax[0].set_ylabel("CIFAR-100 top-1 (%)"); ax[0].legend(fontsize=6)
    mark(fig, synthetic); plt.savefig(out / "fig/x5_curves.pdf", metadata={"CreationDate": None}); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for j, S in enumerate(Ss):
        q = [S["temp"][w] for w in (16, 32, 64)]
        ax[0].errorbar(np.array([16, 32, 64]) * (1 + 0.04 * (j - 2)), [e["mean"] for e in q],
                       yerr=[[e["mean"] - e["lo"] for e in q], [e["hi"] - e["mean"] for e in q]],
                       marker="o", ms=3.5, capsize=2, lw=1.1, color=COL[S["model"]], label=NAME[S["model"]])
        k = S["k82"]
        ax[1].errorbar(j, k["mean"], yerr=[[k["mean"] - k["lo"]], [k["hi"] - k["mean"]]], fmt="o", capsize=3,
                       color=COL[S["model"]])
    ax[0].set_xscale("log", base=2); ax[0].set_xticks([16, 32, 64]); ax[0].set_xticklabels([16, 32, 64])
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_xlabel("width"); ax[0].set_ylabel("annealed $-$ constant (pp)")
    ax[0].set_title("(a) temperature schedule, gap, CIFAR-100", fontsize=8.5); ax[0].legend(fontsize=6.3)
    ax[1].axhline(0, color="k", lw=.8); ax[1].axhspan(-.5, .5, color="grey", alpha=.13)
    ax[1].set_xticks(range(len(Ss))); ax[1].set_xticklabels([NAME[S["model"]] for S in Ss], fontsize=6.5, rotation=20)
    ax[1].set_ylabel("$K=8$ $-$ $K=2$ (pp)"); ax[1].set_title("(b) bank size, CIFAR-100 width 32", fontsize=8.5)
    mark(fig, synthetic); plt.savefig(out / "fig/x6_temp_k.pdf", metadata={"CreationDate": None}); plt.close()
    return 6


def tables(Ss, absent, out):
    T = {}
    r = []
    for S in Ss:
        r.append(f"{NAME[S['model']]} & {f(S['p_static'], 2)} & {f(S['p_gap'], 2)} & {100 * S['dyn_share']:.0f} & "
                 f"{f(100 * S['router'] / S['p_static'], 1)} & "
                 f"{f(S['gate'], 2)} & {f(S['static64'], 2)} & {S['runs']} & {S['diverged']} & {S['dev_groups']} & {f(S['gpuh'], 0)} \\\\")
    for m, why in absent:
        r.append(f"{NAME[m]} & \\multicolumn{{10}}{{l}}{{{why}}} \\\\")
    T["XSUMMARY"] = r
    for key, nm in (("bank", "XBANK"), ("rout", "XROUT")):
        r = []
        for S in Ss:
            cells = []
            for c in COND:
                q = S[key][c]
                tagm = EQ if key == "bank" and q["tost"] < 0.05 else ("*" if key == "rout" and q["p"] < 0.05 else "")
                cells.append(f"{sg(q['mean'])} {tagm}")
            r.append(f"{NAME[S['model']]} & " + " & ".join(cells) + " \\\\")
        T[nm] = r
    r = []
    for S in Ss:
        for p in PAIRS:
            v = S["pairs"][p]; q = v["q"]
            r.append(f"{NAME[S['model']]} & {'CIFAR-100' if p[0] == 'cifar100' else 'Tiny-ImageNet'} w{p[1]}/w{p[2]} & "
                     f"{sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & {v['ratio']:.2f} & "
                     f"{'matched' if abs(v['ratio'] - 1) <= .1 else 'not matched'} & {sg(v['interp'])} \\\\")
    T["XMATCHED"] = r
    r = []
    for S in Ss:
        nb, npr, nr = S["prereg"]
        note = " (post hoc)" if S["model"] == "resnet18" else ""
        r.append(f"{NAME[S['model']]}{note} & {nb} of 5 & {'held' if nb == 5 else 'failed'} & {npr} of 3 & "
                 f"{'held' if npr == 3 else 'failed'} & {nr} of 5 & {'held' if nr >= 3 else 'failed'} \\\\")
    T["XPREREG"] = r
    r = []
    for S in Ss:
        r.append(f"{NAME[S['model']]} & " + " & ".join(f"{sg(S['temp'][w]['mean'])} ({pv(S['temp'][w]['p'])})" for w in (16, 32, 64))
                 + f" & {sg(S['k82']['mean'])} & {S['verdict'](S['k82'])} \\\\")
    T["XTEMPK"] = r
    r = []
    for S in Ss:
        e16 = {rec: x for w, rec, x, _ in S["ent"] if w == 16}
        ratio = e16.get("const", float("nan")) / e16.get("anneal", float("nan"))
        r.append(f"{NAME[S['model']]} & {len(S['ent'])} & {sg(S['entrho'], 2)} & {f(ratio, 1)} \\\\")
    T["XENT"] = r
    r = []
    for S in Ss:
        r.append(f"{NAME[S['model']]} & " + " & ".join(
            f"{sg(S['total'][c]['mean'])} ({S['verdict'](S['total'][c])})" for c in COND) + " \\\\")
    T["XTOTAL"] = r
    for k, v in T.items():
        (out / f"tab_{k}.tex").write_text("\n".join(v) + "\n")
    return len(T)


def macros(Ss, out, synthetic):
    M = {}
    new = [S for S in Ss if S["model"] != "resnet18"]
    M["nmodels"] = str(len(Ss)); M["nnew"] = str(len(new))
    M["nruns"] = str(sum(S["runs"] for S in Ss)); M["gpuh"] = f"{sum(S['gpuh'] for S in Ss):.0f}"
    M["bankeq"] = str(sum(q["tost"] < 0.05 for S in Ss for q in S["bank"].values()))
    M["bankn"] = str(5 * len(Ss))
    M["routpos"] = str(sum(q["p"] < 0.05 and q["mean"] > 0 for S in Ss for q in S["rout"].values()))
    M["nmatched"] = str(sum(abs(v["ratio"] - 1) <= .1 for S in Ss for v in S["pairs"].values()))
    M["npairs"] = str(3 * len(Ss))
    M["p1held"] = str(sum(S["prereg"][0] == 5 for S in new)); M["p2held"] = str(sum(S["prereg"][1] == 3 for S in new))
    M["p3held"] = str(sum(S["prereg"][2] >= 3 for S in new))
    M["synthbanner"] = ("\\fbox{\\parbox{0.95\\textwidth}{\\centering\\color{red}\\bfseries SYNTHETIC TEST DATA. "
                        "Every number in this document is generated to test the pipeline. None is a result.}}"
                        if synthetic else "")
    (out / "macros.tex").write_text("\\newcommand{\\V}[1]{\\csname v#1\\endcsname}\n" + "\n".join(
        f"\\expandafter\\def\\csname v{k}\\endcsname{{{v}}}" for k, v in M.items()) + "\n")
    return M


def build(model_dirs, out, synthetic=False, absent_reasons=None):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    Ss, absent = [], []
    for m in ORDER:
        if m not in model_dirs:
            continue
        gf = [pathlib.Path(d) / "GATE_FAILED.json" for d in model_dirs[m]]
        gf = [g for g in gf if g.exists()]
        if gf:
            g = json.loads(gf[0].read_text())
            absent.append((m, f"gate failed: static w64 seed 0 reached {g['final_acc']:.2f}\\,\\% "
                              f"(floor {g['floor']:.0f}\\,\\%); no further runs"))
            continue
        S = summarise(m, model_dirs[m])
        if S is None:
            absent.append((m, "no results yet")); continue
        Ss.append(S)
    for m, why in (absent_reasons or {}).items():
        absent.append((m, why))
    nf = figures(Ss, out, synthetic); nt = tables(Ss, absent, out); macros(Ss, out, synthetic)
    src = (HERE / "cross_template.tex").read_text()
    for k in re.findall(r"%%([A-Z0-9]+)%%", src):
        src = src.replace(f"%%{k}%%", (out / f"tab_{k}.tex").read_text().rstrip())
    (out / "cross_models.tex").write_text(src)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "cross_models.tex"], cwd=out, capture_output=True)
    log = (out / "cross_models.log").read_text(errors="ignore")
    errs = [l for l in log.splitlines() if l.startswith("! ")]
    und = [l for l in log.splitlines() if "undefined" in l.lower() and "Warning" in l]
    return dict(pdf=out / "cross_models.pdf", figures=nf, tables=nt, errors=errs, undefined=und,
                models=[S["model"] for S in Ss], absent=absent)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=[], help="model=dir [model=dir ...]")
    ap.add_argument("--out", default=str(HERE / "out" / "cross"))
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    dirs = {}
    for kv in a.results:
        m, d = kv.split("=", 1); dirs.setdefault(m, []).append(d)
    if a.synthetic:
        from synth import make
        tmp = pathlib.Path(tempfile.mkdtemp())
        for i, m in enumerate(ORDER):
            make(m, tmp / m, seed=i, drop=1, diverge=1); dirs[m] = [tmp / m]
        a.out = a.out + "_SYNTHETIC"
    r = build(dirs, a.out, a.synthetic)
    print(f"{r['pdf']}: models {r['models']}, absent {r['absent']}, {r['figures']} figures, "
          f"{r['tables']} tables, errors {len(r['errors'])}, undefined {len(r['undefined'])}")
    for e in r["errors"][:10]: print("  ", e)

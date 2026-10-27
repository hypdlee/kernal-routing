"""
Numbers, tables and the figure for the routing-control extension
(PREREGISTRATION_CONTROLS.md, dated 22 October 2026).

Every number comes from the result files in artifacts/controls/<model>/ and
artifacts/<model>/results/, after each final accuracy has been checked
against its train.log line (tools/controls_analysis.load). Called from
mbv2_section.main with its `put`; run with the output directory as cwd.
"""
import importlib.util, json, math, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import plotstyle as PS
_s = importlib.util.spec_from_file_location("kr_stats", HERE / "data.py")
S = importlib.util.module_from_spec(_s); _s.loader.exec_module(S)
_c = importlib.util.spec_from_file_location("controls_analysis", ROOT / "tools" / "controls_analysis.py")
CA = importlib.util.module_from_spec(_c); _c.loader.exec_module(CA)

MARGIN = 0.5
f = lambda x, n=3: "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
sg = lambda x, n=2: f"{x:+.{n}f}"
pv = lambda p: "$<10^{-4}$" if p < 1e-4 else f"{p:.4f}"
tt = lambda a: "\\texttt{" + a.replace("_", "\\_") + "}"
CELLS = {"W64": ("cifar100", 64), "W16": ("cifar100", 16), "TIN": ("tinyimagenet", 32)}
CELLNAME = {"W64": "CIFAR-100, width 64", "W16": "CIFAR-100, width 16", "TIN": "Tiny-ImageNet, width 32"}
MODELS = {"mb": "mobilenetv2", "vg": "vgg16"}
MODELNAME = {"mb": "MobileNetV2", "vg": "VGG-16"}
ARM = {"uni": ("uniform", "d1"), "sta": ("static", "d1"), "gap": ("gap", "d1"), "frzA": ("frozen_random", "d1"),
       "frzB": ("frozen_random", "d2"), "frzC": ("frozen_random", "d3"), "shuf": ("shuffled", "d1")}
LABEL = {"uni": "uniform", "sta": "static", "gap": "gap", "frzA": "frozen_random (draw 1)", "frzB": "frozen_random (draw 2)",
         "frzC": "frozen_random (draw 3)", "shuf": "shuffled"}
SHORT = {"uni": "uniform", "sta": "static", "gap": "gap", "frzA": "frozen_random", "frzB": "frozen d2",
         "frozen d3": "x", "frzC": "frozen d3", "shuf": "shuffled"}


def main(put, root=ROOT):
    A, problems = {}, []
    for k, m in MODELS.items():
        A[k], p = CA.load(m, root); problems += p
    if problems:
        raise SystemExit("control result files disagree with their logs: " + "; ".join(problems[:5]))

    def vals(b, cell, x):
        arm, draw = ARM[x]
        ds, w = CELLS[cell]
        return np.asarray(A[b][(ds, w, arm, draw)], float)

    def q(b, cell, x, y):
        return S.paired(vals(b, cell, x), vals(b, cell, y), margin=MARGIN)

    def has(b, cell, x):
        arm, draw = ARM[x]; ds, w = CELLS[cell]
        return (ds, w, arm, draw) in A[b] and not np.isnan(A[b][(ds, w, arm, draw)]).any()

    # ---- run counts and cost of the extension
    n = 0; mins = 0.0
    for k, m in MODELS.items():
        for fp in sorted((root / "artifacts" / "controls" / m).glob("*_seed*.json")):
            r = json.loads(fp.read_text()); n += 1; mins += r["minutes"]
    put("ctruns", str(n)); put("ctgpuh", f"{mins / 60:.0f}")
    put("ctmbruns", str(len(list((root / "artifacts" / "controls" / "mobilenetv2").glob("*_seed*.json")))))
    put("ctvgruns", str(len(list((root / "artifacts" / "controls" / "vgg16").glob("*_seed*.json")))))

    # ---- every comparison as macros
    for b in MODELS:
        for cell in CELLS:
            xs = [x for x in ARM if has(b, cell, x)]
            for x in xs:
                for y in xs:
                    if x == y:
                        continue
                    qq = q(b, cell, x, y); key = f"ct{b}{cell}{x}{y}"
                    put(key, sg(qq["mean"])); put(key + "lo", sg(qq["lo"])); put(key + "hi", sg(qq["hi"]))
                    put(key + "p", pv(qq["p"])); put(key + "T", f(qq["tost"], 4)); put(key + "v", S.verdict(qq))
                    put(key + "a", f(abs(qq["mean"]), 2))
            for x in xs:
                v = vals(b, cell, x)
                put(f"ct{b}{cell}{x}m", f(v.mean(), 2))

    # ---- main table of the new comparisons
    rows = []
    spec = [("mb", "W64", [("frzB", "uni"), ("frzC", "uni"), ("frzB", "gap"), ("frzC", "gap")]),
            ("vg", "W64", [("frzB", "uni"), ("frzC", "uni"), ("frzB", "gap"), ("frzC", "gap")])]
    for b in ("mb", "vg"):
        spec.append((b, "W16", [("frzA", "sta"), ("frzA", "uni"), ("frzA", "gap"), ("shuf", "sta"), ("shuf", "gap")]))
    spec.append(("vg", "TIN", [("frzA", "sta"), ("frzA", "uni"), ("frzA", "gap"), ("shuf", "sta"), ("shuf", "gap")]))
    for b, cell, comps in spec:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{MODELNAME[b]}, {CELLNAME[cell]}}}}} \\\\")
        for x, y in comps:
            qq = q(b, cell, x, y)
            rows.append(f"{tt(SHORT[x])} $-$ {tt(SHORT[y])} & {sg(qq['mean'])} & [{sg(qq['lo'])}, {sg(qq['hi'])}] & "
                        f"{pv(qq['p'])} & {S.verdict(qq)} \\\\")
    pathlib.Path("tab_CTRL.tex").write_text("\n".join(rows) + "\n")

    # ---- the four pre-registered predictions
    c1 = [q("mb", "W64", x, "uni") for x in ("frzB", "frzC")]
    c1n = sum(z["p"] < 0.05 and z["mean"] > 0 for z in c1)
    c2 = [q("vg", "W64", x, "uni") for x in ("frzB", "frzC")]
    c2n = sum(z["p"] < 0.05 and z["mean"] > 0 for z in c2)
    a, bq, c = q("mb", "W16", "frzA", "sta"), q("mb", "W16", "frzA", "gap"), q("mb", "W16", "shuf", "gap")
    c3p = [a["p"] < 0.05 and a["mean"] > 0, bq["p"] >= 0.05, c["p"] < 0.05 and c["mean"] < 0]
    c4q = [q("vg", "W16", "frzA", "gap"), q("vg", "W16", "shuf", "gap"), q("vg", "TIN", "frzA", "gap"), q("vg", "TIN", "shuf", "gap")]
    c4p = [z["p"] < 0.05 and z["mean"] < 0 for z in c4q]
    out = {"C1": c1n == 2, "C2": c2n == 0, "C3": all(c3p), "C4": all(c4p)}
    put("ctC1n", str(c1n)); put("ctC2n", str(c2n)); put("ctC3n", str(sum(c3p))); put("ctC4n", str(sum(c4p)))
    for k, v in out.items():
        put(f"ct{k}o", "held" if v else "failed")
    put("ctnheld", str(sum(out.values()))); put("ctnfail", str(sum(not v for v in out.values())))
    failing = [lab for lab, ok in zip(["\\texttt{frozen\\_random} at CIFAR-100 width 16", "\\texttt{shuffled} at CIFAR-100 width 16",
                                       "\\texttt{frozen\\_random} on Tiny-ImageNet", "\\texttt{shuffled} on Tiny-ImageNet"], c4p) if not ok]
    put("ctC4fail", ", ".join(failing) if failing else "none")
    pathlib.Path("tab_CTPREREG.tex").write_text(
        "C1 & MobileNetV2, CIFAR-100 width 64: each of the two new frozen-router draws is higher than \\texttt{uniform} & "
        f"{c1n} of 2 & {'held' if out['C1'] else 'failed'} \\\\\n"
        "C2 & VGG-16, CIFAR-100 width 64: neither new draw is higher than \\texttt{uniform} & "
        f"{c2n} of 2 higher & {'held' if out['C2'] else 'failed'} \\\\\n"
        "C3 & MobileNetV2, CIFAR-100 width 16: \\texttt{frozen\\_random} is higher than \\texttt{static} and not distinguishable "
        f"from \\texttt{{gap}}; \\texttt{{shuffled}} is lower than \\texttt{{gap}} & {sum(c3p)} of 3 & {'held' if out['C3'] else 'failed'} \\\\\n"
        "C4 & VGG-16, CIFAR-100 width 16 and Tiny-ImageNet width 32: \\texttt{frozen\\_random} and \\texttt{shuffled} are each lower "
        f"than \\texttt{{gap}} & {sum(c4p)} of 4 & {'held' if out['C4'] else 'failed'} \\\\\n")

    gu = q("vg", "TIN", "frzA", "uni")["mean"] / q("vg", "TIN", "gap", "uni")["mean"]
    put("ctvgTINshare", f"{100 * gu:.0f}")
    # every comparison behind a held prediction must survive Bonferroni over its own prediction
    bonf = []
    if out["C1"]: bonf += [(z["p"], 2) for z in c1]
    if out["C3"]: bonf += [(a["p"], 3), (c["p"], 3)]
    if out["C4"]: bonf += [(z["p"], 4) for z in c4q]
    if not all(pp * k < 0.05 for pp, k in bonf):
        raise SystemExit("a comparison behind a held control prediction fails Bonferroni; fix the text")
    put("ctC4pFail", pv(max(z["p"] for z, ok in zip(c4q, c4p) if not ok)) if not all(c4p) else "--")
    put("ctC4dFail", sg(min((z["mean"] for z, ok in zip(c4q, c4p) if not ok), default=0)))
    # frozen draws, w64: mean over the three draws per seed and its comparison with uniform and gap
    for b in MODELS:
        if all(has(b, "W64", x) for x in ("frzA", "frzB", "frzC")):
            fm = np.mean([vals(b, "W64", x) for x in ("frzA", "frzB", "frzC")], axis=0)
            for y in ("uni", "gap", "sta"):
                qq = S.paired(fm, vals(b, "W64", y), margin=MARGIN); key = f"ct{b}W64frz3{y}"
                put(key, sg(qq["mean"])); put(key + "lo", sg(qq["lo"])); put(key + "hi", sg(qq["hi"]))
                put(key + "p", pv(qq["p"])); put(key + "T", f(qq["tost"], 4)); put(key + "v", S.verdict(qq))
            put(f"ct{b}W64frzsd", f(np.std([vals(b, "W64", x).mean() for x in ("frzA", "frzB", "frzC")], ddof=1), 2))

    # ---- per-seed appendix table
    rows = []
    for b in ("mb", "vg"):
        for cell in ("W64", "W16", "TIN"):
            if cell == "TIN" and b == "mb":
                continue
            ds, w = CELLS[cell]
            xs = ["frzB", "frzC"] if cell == "W64" else ["shuf", "frzA"]
            for x in xs:
                v = vals(b, cell, x)
                rows.append(f"{MODELNAME[b]} & {'CIFAR-100' if ds == 'cifar100' else 'Tiny-ImageNet'} & {w} & {tt(ARM[x][0])}"
                            f"{' (' + ARM[x][1] + ')' if cell == 'W64' else ''} & " + " & ".join(f"{z:.2f}" for z in v)
                            + f" & {f(v.mean(), 2)} & {f(v.std(ddof=1), 2)} \\\\")
    pathlib.Path("tab_CTALLRUNS.tex").write_text("\n".join(rows) + "\n")

    # ---- figure: difference from the constant mixture, all control cells
    panels = [("mb", "W64"), ("vg", "W64"), ("mb", "W16"), ("vg", "W16"), ("vg", "TIN")]
    fig, axs = plt.subplots(1, 5, figsize=(8.2, 2.9), sharey=True)
    colors = {"gap": "#d62728", "frzA": "#9467bd", "frzB": "#b39ad1", "frzC": "#8a6fb5", "shuf": "#2ca02c", "sta": "#444444"}
    for ax, (b, cell) in zip(axs, panels):
        xs = [x for x in ("sta", "frzA", "frzB", "frzC", "shuf", "gap") if has(b, cell, x)]
        for i, x in enumerate(xs):
            z = q(b, cell, x, "uni")
            ax.errorbar([i], [z["mean"]], yerr=[[z["mean"] - z["lo"]], [z["hi"] - z["mean"]]], fmt="o", ms=4, capsize=2.5,
                        color=colors[x], lw=1.2)
        ax.axhline(0, color="k", lw=.8); ax.axhspan(-.5, .5, color="grey", alpha=.13)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([{"sta": "static", "frzA": "frz d1" if cell == "W64" else "frozen", "frzB": "frz d2", "frzC": "frz d3", "shuf": "shuf", "gap": "gap"}[x] for x in xs], rotation=45, ha="right", fontsize=6.5)
        ax.set_title(f"{MODELNAME[b]}\n{CELLNAME[cell].replace('CIFAR-100, ', 'C100 ').replace('Tiny-ImageNet, ', 'TIN ')}", fontsize=7.5)
        ax.set_xlim(-.6, 5.6 if len(xs) > 5 else len(xs) - .4)
    axs[0].set_ylabel("difference from uniform (pp)")
    plt.tight_layout(w_pad=.6)
    PS.save("fig/g4_controls.pdf"); plt.close()
    return out

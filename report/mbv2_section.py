"""
Figures, tables and quoted numbers for the MobileNetV2 replication and the
two-backbone comparisons. Run with the output directory as the working
directory (build_paper2.py does this).

Inputs: the MobileNetV2 records parsed from train.log (mbv2_log.py), the
132 MobileNetV2 result files (for routing entropies; every final accuracy
in them is checked against the log and the build stops on any mismatch),
the ResNet-18 per-seed values the paper is built on (paper_r18_data.py),
and counted parameter totals (PARAMS_EXPECTED.md). Every number in the
MobileNetV2 section is computed here; none is typed.
"""
import importlib.util, math, pathlib, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import paper_r18_data as P
import mbv2_log

_spec = importlib.util.spec_from_file_location("kr_stats", HERE / "data.py")
S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S)
paired, verdict = S.paired, S.verdict

plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "legend.frameon": False})
C = {"static": "#444444", "uniform": "#1f77b4", "gap": "#d62728", "shuffled": "#2ca02c",
     "frozen_random": "#9467bd", "ann": "#ff7f0e", "r18": "#7f7f7f", "mb": "#1f6fb4"}
DSN = {"cifar100": "CIFAR-100", "tinyimagenet": "Tiny-ImageNet"}
COND = [("cifar100", 16), ("cifar100", 32), ("cifar100", 64), ("tinyimagenet", 32), ("tinyimagenet", 64)]
f = lambda x, n=3: "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
sg = lambda x, n=3: "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{n}f}"
pv = lambda p: "--" if math.isnan(p) else ("$<10^{-4}$" if p < 1e-4 else f"{p:.4f}")
tt = lambda a: "\\texttt{" + a.replace("_", "\\_") + "}"


def counted(model):
    out, cur = {}, None
    for l in (ROOT / "PARAMS_EXPECTED.md").read_text().splitlines():
        if l.startswith("## "):
            cur = l[3:].strip()
        x = [c.strip() for c in l.split("|")[1:-1]]
        if cur == model and len(x) == 8 and x[0] in ("cifar100", "tinyimagenet"):
            out[(x[0], int(x[1]), int(x[2]), x[3])] = (int(x[4]) / 1e6, int(x[6]) / 1e3)
    return out


def main(log_path, results_dir):
    R = mbv2_log.parse(log_path)
    assert len(R) == 132, len(R)
    PM = counted("mobilenetv2")

    def mb(ds, w, arm, k=4, rec="const", field="final"):
        return np.array([R[(ds, w, k, rec, arm, s)][field] for s in range(6)], dtype=float)

    r18 = lambda ds, w, arm, k=4, rec="const": P.ACC[(ds, w, k, rec)][arm]
    M = {}
    put = M.__setitem__
    pathlib.Path("fig").mkdir(exist_ok=True)

    # ------------------------------------------------------------ headline numbers
    put("mbruns", str(len(R)))
    put("mbgpuh", f"{sum(r['minutes'] for r in R.values()) / 60:.0f}")
    put("totruns", str(len(R) + P.n_runs()))
    put("totgpuh", f"{sum(r['minutes'] for r in R.values()) / 60 + P.gpu_hours():.0f}")
    put("mbgate", f(R[("cifar100", 64, 4, "const", "static", 0)]["final"], 2))
    put("mbstatic", f(mb("cifar100", 64, "static").mean(), 2))
    put("mbpstatic", f(PM[("cifar100", 64, 4, "static")][0], 2))
    put("mbpgap", f(PM[("cifar100", 64, 4, "gap")][0], 2))
    put("mbrouterK", f(PM[("cifar100", 64, 4, "gap")][1], 0))
    put("mbrouterpct", f(100 * PM[("cifar100", 64, 4, "gap")][1] / 1e3 / PM[("cifar100", 64, 4, "static")][0], 0))
    put("mbbankadd", f(PM[("cifar100", 64, 4, "uniform")][0] - PM[("cifar100", 64, 4, "static")][0], 2))
    sh = lambda m: (PM[("cifar100", 64, 4, "uniform")][0] - PM[("cifar100", 64, 4, "static")][0]) / (3 * PM[("cifar100", 64, 4, "static")][0])
    put("mbdynshare", f"{100 * sh(None):.0f}")

    # ------------------------------------------------------------ decomposition
    dec = {}
    rows = []
    for ds, w in COND:
        b = paired(mb(ds, w, "uniform"), mb(ds, w, "static"))
        r = paired(mb(ds, w, "gap"), mb(ds, w, "uniform"))
        t = paired(mb(ds, w, "gap"), mb(ds, w, "static"))
        dec[(ds, w)] = (b, r, t)
        rows.append(f"{DSN[ds]} & {w} & {sg(b['mean'])} & {pv(b['tost'])} & {verdict(b)} & {sg(r['mean'])} & "
                    f"{pv(r['p'])} & {sg(t['mean'])} & {pv(t['p'])} \\\\")
    tag = {("cifar100", 16): "CA", ("cifar100", 32): "CB", ("cifar100", 64): "CC",
           ("tinyimagenet", 32): "TA", ("tinyimagenet", 64): "TB"}
    for c, (b, r, t) in dec.items():
        k = tag[c]
        put(f"mbbank{k}", sg(b["mean"])); put(f"mbbank{k}T", f(b["tost"], 4))
        put(f"mbrout{k}", sg(r["mean"])); put(f"mbrout{k}p", pv(r["p"]))
        put(f"mbeff{k}", sg(t["mean"])); put(f"mbeff{k}lo", sg(t["lo"])); put(f"mbeff{k}hi", sg(t["hi"]))
        put(f"mbeff{k}p", pv(t["p"]))
    put("mbnbankeq", str(sum(verdict(b) == "equivalent" for b, _, _ in dec.values())))
    put("mbnbanksig", str(sum(b["p"] < 0.05 for b, _, _ in dec.values())))
    put("mbroutmin", f(min(r["mean"] for _, r, _ in dec.values()), 2))
    put("mbroutmax", f(max(r["mean"] for _, r, _ in dec.values()), 2))
    put("mbtotmin", f(min(t["mean"] for _, _, t in dec.values()), 2))
    put("mbtotmax", f(max(t["mean"] for _, _, t in dec.values()), 2))
    pathlib.Path("tab_MBDECOMP.tex").write_text("\n".join(rows) + "\n")
    # ResNet-18 bank-equivalence count, for the cross-backbone sentence
    r18bank = [paired(r18(ds, w, "uniform"), r18(ds, w, "static")) for ds, w in COND]
    put("rbnbankeq", str(sum(verdict(b) == "equivalent" for b in r18bank)))
    put("bothbankeq", str(sum(verdict(b) == "equivalent" for b in r18bank) +
                          sum(verdict(b) == "equivalent" for b, _, _ in dec.values())))

    # sensitivity: the resumed run
    res = [k for k, v in R.items() if v["resumed"]]
    put("mbnresumed", str(len(res)))
    k = res[0]
    keep = [s for s in range(6) if s != k[5]]
    q = paired(mb(k[0], k[1], "gap")[keep], mb(k[0], k[1], "static")[keep])
    put("mbsensTB", sg(q["mean"])); put("mbsensTBp", pv(q["p"]))
    put("mbresumedacc", f(R[k]["final"], 2))

    # ------------------------------------------------------------ width 64 controls
    arms = ["static", "uniform", "gap", "shuffled", "frozen_random"]
    comps = [("uniform", "static"), ("gap", "static"), ("shuffled", "static"), ("frozen_random", "static"),
             ("frozen_random", "gap"), ("shuffled", "gap"), ("frozen_random", "uniform")]
    rows = []
    for a, b_ in comps:
        qm = paired(mb("cifar100", 64, a), mb("cifar100", 64, b_))
        qr = paired(r18("cifar100", 64, a), r18("cifar100", 64, b_))
        rows.append(f"{tt(a)} $-$ {tt(b_)} & {sg(qr['mean'])} & {verdict(qr)} & {sg(qm['mean'])} & "
                    f"[{sg(qm['lo'])}, {sg(qm['hi'])}] & {pv(qm['p'])} & {pv(qm['tost'])} & {verdict(qm)} \\\\")
        key = (a + b_).replace("_", "")
        put(f"mbw{key}", sg(qm["mean"])); put(f"mbw{key}lo", sg(qm["lo"])); put(f"mbw{key}hi", sg(qm["hi"]))
        put(f"mbw{key}p", pv(qm["p"])); put(f"mbw{key}T", f(qm["tost"], 4))
    pathlib.Path("tab_MBW64.tex").write_text("\n".join(rows) + "\n")
    F_, p_ = stats.f_oneway(*[mb("cifar100", 64, a) for a in arms])
    put("mbanovaF", f(F_, 1)); put("mbanovaP", "<10^{-4}" if p_ < 1e-4 else "=" + f(p_, 4))

    # ------------------------------------------------------------ matched budgets
    rows = []
    for ds, wg, ws, kk in [("cifar100", 16, 32, "MA"), ("cifar100", 32, 64, "MB"), ("tinyimagenet", 32, 64, "MC")]:
        q = paired(mb(ds, ws, "static"), mb(ds, wg, "gap"))
        pg, ps = PM[(ds, wg, 4, "gap")][0], PM[(ds, ws, 4, "static")][0]
        ratio = pg / ps
        wl = [16, 32, 64] if ds == "cifar100" else [32, 64]
        xs = np.log([PM[(ds, w, 4, "static")][0] for w in wl]); ys = [mb(ds, w, "static").mean() for w in wl]
        inside = xs[0] <= math.log(pg) <= xs[-1]
        interp = float(np.interp(math.log(pg), xs, ys)) - mb(ds, wg, "gap").mean() if inside else float("nan")
        rows.append(f"{DSN[ds]} & {tt('gap')} w{wg} ({f(pg, 2)}M) & {tt('static')} w{ws} ({f(ps, 2)}M) & "
                    f"{f(mb(ds, wg, 'gap').mean())} & {f(mb(ds, ws, 'static').mean())} & {sg(q['mean'])} & "
                    f"[{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & {verdict(q)} & {ratio:.2f} & {sg(interp)} \\\\")
        put(f"mbm{kk}", sg(q["mean"])); put(f"mbm{kk}p", pv(q["p"])); put(f"mbm{kk}v", verdict(q))
        put(f"mbm{kk}r", f"{ratio:.2f}"); put(f"mbm{kk}i", sg(interp))
    pathlib.Path("tab_MBMATCHED.tex").write_text("\n".join(rows) + "\n")

    # ------------------------------------------------------------ bank size
    rows = []
    for kk in (2, 4, 8):
        v = mb("cifar100", 32, "gap", k=kk); q = paired(v, mb("cifar100", 32, "static"))
        rows.append(f"{kk} & {f(v.mean())} & {f(v.std(ddof=1))} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & "
                    f"{pv(q['p'])} & {f(PM[('cifar100', 32, kk, 'gap')][0], 2)} & {f(mb('cifar100', 32, 'gap', k=kk, field='minutes').mean(), 1)} \\\\")
    v = mb("cifar100", 32, "static")
    rows.append(f"\\midrule {tt('static')} & {f(v.mean())} & {f(v.std(ddof=1))} & -- & -- & -- & "
                f"{f(PM[('cifar100', 32, 4, 'static')][0], 2)} & {f(mb('cifar100', 32, 'static', field='minutes').mean(), 1)} \\\\")
    pathlib.Path("tab_MBK.tex").write_text("\n".join(rows) + "\n")
    q = paired(mb("cifar100", 32, "gap", k=8), mb("cifar100", 32, "gap", k=2))
    put("mbk82", sg(q["mean"])); put("mbk82p", pv(q["p"])); put("mbk82v", verdict(q))

    # ------------------------------------------------------------ temperature
    rows = []
    for w, kk in [(16, "A"), (32, "B"), (64, "C")]:
        c = mb("cifar100", w, "gap"); an = mb("cifar100", w, "gap", rec="anneal")
        a1 = paired(an, c); s1 = paired(c, mb("cifar100", w, "static")); s2 = paired(an, mb("cifar100", w, "static"))
        u2 = paired(an, mb("cifar100", w, "uniform"))
        rows.append(f"{w} & {f(c.mean())} & {f(an.mean())} & {sg(a1['mean'])} & {pv(a1['p'])} & {sg(s1['mean'])} & "
                    f"{sg(s2['mean'])} & {pv(s2['p'])} & {sg(u2['mean'])} & {verdict(u2)} \\\\")
        put(f"mban{kk}", sg(a1["mean"])); put(f"mban{kk}p", pv(a1["p"]))
        put(f"mbanu{kk}", sg(u2["mean"])); put(f"mbanu{kk}v", verdict(u2))
        put(f"mbla{kk}", f(mb("cifar100", w, "gap", rec="anneal", field="loss").mean(), 3))
        put(f"mblc{kk}", f(mb("cifar100", w, "gap", field="loss").mean(), 3))
    pathlib.Path("tab_MBTEMP.tex").write_text("\n".join(rows) + "\n")

    # ------------------------------------------------------------ pre-registered predictions
    nb = sum(b["tost"] < 0.05 for b, _, _ in dec.values())
    pairs = [paired(mb(ds, ws, "static"), mb(ds, wg, "gap")) for ds, wg, ws in
             [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]]
    np2 = sum(q["p"] < 0.05 and q["mean"] > 0 for q in pairs)
    nr = sum(r["p"] < 0.05 and r["mean"] > 0 for _, r, _ in dec.values())
    put("mbP1", str(nb)); put("mbP2", str(np2)); put("mbP3", str(nr))
    pathlib.Path("tab_MBPREREG.tex").write_text(
        f"P1 & The bank term (\\texttt{{uniform}}$-$\\texttt{{static}}) is equivalent to zero (TOST, $\\pm$\\SI{{0.5}}{{pp}}) in each of the five conditions & {nb} of 5 & {'held' if nb == 5 else 'failed'} \\\\\n"
        f"P2 & \\texttt{{static}} at width $2c$ is more accurate than \\texttt{{gap}} at width $c$ in each of the three pairs & {np2} of 3 & {'held' if np2 == 3 else 'failed'} \\\\\n"
        f"P3 & The routing term is positive with $p<0.05$ in at least three of the five conditions & {nr} of 5 & {'held' if nr >= 3 else 'failed'} \\\\\n")

    # ------------------------------------------------------------ the contradiction table
    rbank = sum(verdict(b) == "equivalent" for b in r18bank)
    r18_w64 = paired(r18("cifar100", 64, "gap"), r18("cifar100", 64, "static"))
    mb_w64 = dec[("cifar100", 64)][2]
    r18_pairs = [paired(r18(ds, ws, "static"), r18(ds, wg, "gap")) for ds, wg, ws in
                 [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]]
    r18_ann = [paired(r18("cifar100", w, "gap", rec="anneal"), r18("cifar100", w, "gap")) for w in (16, 32, 64)]
    mb_ann = [paired(mb("cifar100", w, "gap", rec="anneal"), mb("cifar100", w, "gap")) for w in (16, 32, 64)]
    r18_frz = paired(r18("cifar100", 64, "frozen_random"), r18("cifar100", 64, "gap"))
    mb_frz = paired(mb("cifar100", 64, "frozen_random"), mb("cifar100", 64, "gap"))
    mb_frzu = paired(mb("cifar100", 64, "frozen_random"), mb("cifar100", 64, "uniform"))
    r18_k = paired(r18("cifar100", 32, "gap", k=8), r18("cifar100", 32, "gap", k=2))
    nsig_r18 = sum(q["p"] < 0.05 and q["mean"] > 0 for q in r18_pairs)
    lst = lambda qs: ", ".join(sg(q["mean"], 2) for q in qs)
    contra = [
        ("Bank term ($\\texttt{uniform}-\\texttt{static}$)", f"equivalent in {rbank} of 5", f"equivalent in {nb} of 5; never significant (largest {max((b['mean'] for b,_,_ in dec.values()), key=abs):+.2f})", "holds, weaker"),
        ("Routing term significant", f"{sum(paired(r18(ds,w,'gap'),r18(ds,w,'uniform'))['p']<0.05 for ds,w in COND)} of 5", f"{nr} of 5", "holds"),
        ("$\\texttt{gap}-\\texttt{static}$, CIFAR-100 w64", f"{sg(r18_w64['mean'],2)} ({verdict(r18_w64)})", f"{sg(mb_w64['mean'],2)} ({verdict(mb_w64)})", "\\textbf{contradicts}"),
        ("Plain $2c$ beats \\texttt{gap} $c$ (3 pairs)", f"{nsig_r18} of 3 ({lst(r18_pairs)})", f"{np2} of 3 ({lst(pairs)})", "\\textbf{contradicts}"),
        ("Annealed $-$ constant $\\tau$ (w16, 32, 64)", lst(r18_ann), lst(mb_ann), "\\textbf{differs}"),
        ("$\\texttt{frozen\\_random}-\\texttt{gap}$, w64", f"{sg(r18_frz['mean'],2)} (both null)", f"{sg(mb_frz['mean'],2)}; vs \\texttt{{uniform}} {sg(mb_frzu['mean'],2)}", "new"),
        ("$K=8$ vs $K=2$, CIFAR-100 w32", f"{sg(r18_k['mean'],2)} ({verdict(r18_k)})", f"{sg(q['mean'],2) if False else sg(paired(mb('cifar100',32,'gap',k=8),mb('cifar100',32,'gap',k=2))['mean'],2)} ({verdict(paired(mb('cifar100',32,'gap',k=8),mb('cifar100',32,'gap',k=2)))})", "holds"),
    ]
    pathlib.Path("tab_CONTRA.tex").write_text("\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in contra) + "\n")
    put("rbmpairs", lst(r18_pairs)); put("mbmpairs", lst(pairs))
    put("rbanlist", lst(r18_ann)); put("mbanlist", lst(mb_ann))

    # ------------------------------------------------------------ loss vs effect (both backbones)
    pts = []
    for ds, w in COND:
        pts.append(("ResNet-18", ds, w, P.LOSS[(ds, w, 4, "const")]["static"], paired(r18(ds, w, "gap"), r18(ds, w, "static"))))
        pts.append(("MobileNetV2", ds, w, mb(ds, w, "static", field="loss").mean(), dec[(ds, w)][2]))
    put("mblossCC", f(mb("cifar100", 64, "static", field="loss").mean(), 3))
    put("mblossmin", f(min(p[3] for p in pts if p[0] == "MobileNetV2"), 3))
    put("rblossCC", f(P.LOSS[("cifar100", 64, 4, "const")]["static"], 4))

    # ------------------------------------------------------------ appendix tables
    rows = []
    for ds, ws in [("cifar100", [16, 32, 64]), ("tinyimagenet", [32, 64])]:
        for w in ws:
            for a in (arms if (ds, w) == ("cifar100", 64) else ["static", "uniform", "gap"]):
                v = mb(ds, w, a)
                mark = ["$^\\dagger$" if R[(ds, w, 4, "const", a, s)]["resumed"] else "" for s in range(6)]
                rows.append(f"{DSN[ds]} & {w} & {tt(a)} & " + " & ".join(f"{x:.2f}{m}" for x, m in zip(v, mark)) +
                            f" & {f(v.mean())} & {f(v.std(ddof=1))} \\\\")
    pathlib.Path("tab_MBALLRUNS.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for ds, w in COND:
        for a in (["uniform", "gap", "shuffled", "frozen_random"] if (ds, w) == ("cifar100", 64) else ["uniform", "gap"]):
            q = paired(mb(ds, w, a), mb(ds, w, "static"))
            rows.append(f"{DSN[ds]} & {w} & {tt(a)} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & "
                        f"{pv(q['wil'])} & {sg(q['dz'], 2)} & {pv(q['tost'])} & {verdict(q)} \\\\")
    pathlib.Path("tab_MBVSSTATIC.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for ds, ws in [("cifar100", [16, 32, 64]), ("tinyimagenet", [32, 64])]:
        for w in ws:
            for a in (arms if (ds, w) == ("cifar100", 64) else ["static", "uniform", "gap"]):
                p_, rk = PM[(ds, w, 4, a)]
                rows.append(f"{DSN[ds]} & {w} & {tt(a)} & {f(p_, 2)} & {f(rk, 1)} & {f(mb(ds, w, a, field='loss').mean(), 3)} & "
                            f"{f(mb(ds, w, a, field='minutes').mean(), 1)} \\\\")
    pathlib.Path("tab_MBCONFIG.tex").write_text("\n".join(rows) + "\n")
    rows, tot, hrs = [], 0, 0.0
    groups = sorted({k[:4] for k in R}, key=lambda g: S.DISPLAY_ORDER.index(g))
    for g in groups:
        ks = [k for k in R if k[:4] == g]
        armset = []
        for a in arms:
            if any(k[4] == a for k in ks):
                armset.append(a)
        h = sum(R[k]["minutes"] for k in ks) / 60
        tot += len(ks); hrs += h
        rows.append(f"{DSN[g[0]]} & {g[1]} & {g[2]} & {'constant' if g[3] == 'const' else 'annealed'} & "
                    f"{', '.join(tt(a) for a in armset)} & {len(ks)} & {f(h, 1)} \\\\")
    rows.append(f"\\midrule \\multicolumn{{5}}{{l}}{{total}} & {tot} & {f(hrs, 1)} \\\\")
    pathlib.Path("tab_MBINVENT.tex").write_text("\n".join(rows) + "\n")

    # ================================================================ figures
    def eb(ax, x, qs, **k):
        ax.errorbar(x, [q["mean"] for q in qs], yerr=[[q["mean"] - q["lo"] for q in qs], [q["hi"] - q["mean"] for q in qs]],
                    capsize=3, **k)

    # G1: decomposition, both backbones
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True)
    xs = np.arange(5); bw = .36
    labels = ["C100\nw16", "C100\nw32", "C100\nw64", "TIN\nw32", "TIN\nw64"]
    for ax, name, getter in [(axs[0], "(a) ResNet-18", r18), (axs[1], "(b) MobileNetV2", mb)]:
        for j, (lbl, fn, col) in enumerate([("bank (uniform $-$ static)", lambda d, w: paired(getter(d, w, "uniform"), getter(d, w, "static")), C["uniform"]),
                                            ("routing (gap $-$ uniform)", lambda d, w: paired(getter(d, w, "gap"), getter(d, w, "uniform")), C["gap"])]):
            qs = [fn(d, w) for d, w in COND]
            ax.bar(xs + (j - .5) * bw, [q["mean"] for q in qs], bw, color=col, label=lbl,
                   yerr=[[q["mean"] - q["lo"] for q in qs], [q["hi"] - q["mean"] for q in qs]], capsize=2.5, error_kw=dict(lw=.9))
        ax.axhline(0, color="k", lw=.8); ax.axhspan(-.5, .5, color="grey", alpha=.10)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7); ax.set_title(name, fontsize=9)
    axs[0].set_ylabel("$\\Delta$ accuracy (pp)"); axs[0].legend(fontsize=7, loc="upper left")
    plt.savefig("fig/g1_decomp_both.pdf"); plt.close()

    # G2: MobileNetV2 width 64, five arms; and controls on both backbones
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for i, a in enumerate(arms):
        v = mb("cifar100", 64, a)
        ax[0].scatter([i] * 6, v, s=22, color=C[a], alpha=.85, zorder=3)
        ax[0].hlines(v.mean(), i - .3, i + .3, color="k", lw=1.5, zorder=4)
    s0 = mb("cifar100", 64, "static").mean(); ax[0].axhspan(s0 - .5, s0 + .5, color="grey", alpha=.13)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(arms, rotation=35, ha="right", fontsize=7)
    ax[0].set_ylabel("top-1 accuracy (%)"); ax[0].set_title("(a) MobileNetV2, CIFAR-100, width 64", fontsize=9)
    cl = [("gap", "static"), ("frozen_random", "static"), ("shuffled", "static"), ("frozen_random", "gap"), ("shuffled", "gap")]
    for j, (name, getter, col) in enumerate([("ResNet-18", r18, C["r18"]), ("MobileNetV2", mb, C["mb"])]):
        qs = [paired(getter("cifar100", 64, a), getter("cifar100", 64, b_)) for a, b_ in cl]
        y = np.arange(len(cl)) + (j - .5) * .3
        ax[1].errorbar([q["mean"] for q in qs], y, xerr=[[q["mean"] - q["lo"] for q in qs], [q["hi"] - q["mean"] for q in qs]],
                       fmt="o", ms=4, capsize=2.5, color=col, label=name)
    ax[1].axvline(0, color="k", lw=.8); ax[1].axvspan(-.5, .5, color="grey", alpha=.13)
    ax[1].set_yticks(range(len(cl))); ax[1].set_yticklabels([f"{a} $-$ {b_}" for a, b_ in cl], fontsize=7)
    ax[1].invert_yaxis(); ax[1].set_xlabel("$\\Delta$ (pp), 95% CI"); ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_title("(b) routing controls, width 64", fontsize=9)
    plt.savefig("fig/g2_mb_w64.pdf"); plt.close()

    # G3: gap - static by width, both backbones, both datasets
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for k_, (ds, ws, ttl) in enumerate([("cifar100", [16, 32, 64], "(a) CIFAR-100"), ("tinyimagenet", [32, 64], "(b) Tiny-ImageNet")]):
        for name, getter, col, mk in [("ResNet-18", r18, C["r18"], "s"), ("MobileNetV2", mb, C["mb"], "o")]:
            eb(ax[k_], ws, [paired(getter(ds, w, "gap"), getter(ds, w, "static")) for w in ws], marker=mk, ms=4.5, lw=1.4,
               color=col, label=name)
        ax[k_].axhline(0, color="k", lw=.8); ax[k_].axhspan(-.5, .5, color="grey", alpha=.13)
        ax[k_].set_xscale("log", base=2); ax[k_].set_xticks([16, 32, 64]); ax[k_].set_xticklabels([16, 32, 64])
        ax[k_].set_xlabel("base width $c$"); ax[k_].set_title(ttl, fontsize=9)
    ax[0].set_ylabel("gap $-$ static (pp)"); ax[0].legend(fontsize=7)
    plt.savefig("fig/g3_width_both.pdf"); plt.close()

    # G4: matched budgets, MobileNetV2
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for k_, (ds, ws, ttl) in enumerate([("cifar100", [16, 32, 64], "(a) CIFAR-100"), ("tinyimagenet", [32, 64], "(b) Tiny-ImageNet")]):
        for a in ["static", "uniform", "gap"]:
            p_ = [PM[(ds, w, 4, a)][0] for w in ws]; m_ = [mb(ds, w, a).mean() for w in ws]
            ax[k_].plot(p_, m_, marker="o", ms=4, lw=1.3, color=C[a], label=a)
            for w, pp, mm in zip(ws, p_, m_):
                ax[k_].annotate(f"w{w}", (pp, mm), fontsize=6, xytext=(3, -8), textcoords="offset points")
        for wg, wsx in ([(16, 32), (32, 64)] if ds == "cifar100" else [(32, 64)]):
            xg, yg = PM[(ds, wg, 4, "gap")][0], mb(ds, wg, "gap").mean()
            xs_, ys_ = PM[(ds, wsx, 4, "static")][0], mb(ds, wsx, "static").mean()
            ax[k_].annotate("", xy=(xs_, ys_), xytext=(xg, yg), arrowprops=dict(arrowstyle="->", lw=1, ls=":"))
            ax[k_].annotate(f"${ys_ - yg:+.2f}$ (params $\\times{xg / xs_:.2f}$)", ((xg + xs_) / 2, (yg + ys_) / 2), fontsize=6.3,
                            ha="center", bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
        ax[k_].set_xscale("log"); ax[k_].set_xlabel("stored parameters (M, log)"); ax[k_].set_title(ttl, fontsize=9)
        ax[k_].legend(fontsize=7)
    ax[0].set_ylabel("top-1 accuracy (%)")
    plt.savefig("fig/g4_mb_matched.pdf"); plt.close()

    # G5: temperature and bank size, both backbones
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6))
    for name, getter, col, mk, off in [("ResNet-18", r18, C["r18"], "s", .96), ("MobileNetV2", mb, C["mb"], "o", 1.04)]:
        eb(ax[0], np.array([16, 32, 64]) * off, [paired(getter("cifar100", w, "gap", rec="anneal"), getter("cifar100", w, "gap")) for w in (16, 32, 64)],
           marker=mk, ms=4.5, lw=1.3, color=col, label=name)
        eb(ax[1], np.array([2, 4, 8]) * off, [paired(getter("cifar100", 32, "gap", k=kk), getter("cifar100", 32, "static")) for kk in (2, 4, 8)],
           marker=mk, ms=4.5, lw=1.3, color=col, label=name)
    for a_, xl, tl in [(ax[0], "base width $c$", "(a) annealed $-$ constant $\\tau$, gap"), (ax[1], "bank size $K$", "(b) gap $-$ static, CIFAR-100 w32")]:
        a_.axhline(0, color="k", lw=.8); a_.set_xscale("log", base=2); a_.set_xlabel(xl); a_.set_title(tl, fontsize=9)
    ax[0].set_xticks([16, 32, 64]); ax[0].set_xticklabels([16, 32, 64]); ax[1].set_xticks([2, 4, 8]); ax[1].set_xticklabels([2, 4, 8])
    ax[0].set_ylabel("$\\Delta$ (pp)"); ax[0].legend(fontsize=7)
    plt.savefig("fig/g5_temp_k.pdf"); plt.close()

    # G6: training loss and the saturation scatter, both backbones
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    EP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 149]
    for (ds, w, a), ls in [(("cifar100", 16, "static"), ":"), (("cifar100", 32, "static"), "--"), (("cifar100", 64, "static"), "-"),
                           (("tinyimagenet", 64, "static"), "-."), (("tinyimagenet", 64, "gap"), (0, (1, 1)))]:
        cv = R[(ds, w, 4, "const", a, 0)]["curve"]
        ax[0].plot(EP, [cv[e]["loss"] for e in EP], ls=ls, color=C["mb"] if ds == "cifar100" else "#e377c2", lw=1.3,
                   label=f"{'CIFAR' if ds == 'cifar100' else 'TIN'} w{w} {a}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss (log)")
    ax[0].legend(fontsize=6.3); ax[0].set_title("(a) MobileNetV2 training loss, seed 0", fontsize=9)
    for name, col, mk in [("ResNet-18", C["r18"], "s"), ("MobileNetV2", C["mb"], "o")]:
        sel = [p for p in pts if p[0] == name]
        ax[1].errorbar([p[3] for p in sel], [p[4]["mean"] for p in sel],
                       yerr=[[p[4]["mean"] - p[4]["lo"] for p in sel], [p[4]["hi"] - p[4]["mean"] for p in sel]],
                       fmt=mk, ms=4.5, capsize=3, color=col, label=name)
        for p in sel:
            ax[1].annotate(f"{'T' if p[1] == 'tinyimagenet' else 'C'}{p[2]}", (p[3], p[4]["mean"]), xytext=(4, -3),
                           textcoords="offset points", fontsize=6)
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xscale("log"); ax[1].legend(fontsize=7, loc="upper left")
    ax[1].set_xlabel("static arm final training loss (log)"); ax[1].set_ylabel("gap $-$ static (pp)")
    ax[1].set_title("(b) effect against training loss, 10 conditions", fontsize=9)
    plt.savefig("fig/g6_loss_effect.pdf"); plt.close()

    # G7 (appendix): every MobileNetV2 run at K=4, tau=1
    fig, axs = plt.subplots(1, 5, figsize=(7.8, 2.5), gridspec_kw=dict(width_ratios=[3, 3, 5, 3, 3]))
    for k_, (ds, w) in enumerate(COND):
        aa = arms if (ds, w) == ("cifar100", 64) else ["static", "uniform", "gap"]
        for i, a in enumerate(aa):
            v = mb(ds, w, a); axs[k_].scatter([i] * 6, v, s=14, color=C[a], alpha=.85, zorder=3)
            axs[k_].hlines(v.mean(), i - .3, i + .3, color="k", lw=1.3, zorder=4)
        axs[k_].set_xticks(range(len(aa))); axs[k_].set_xticklabels(aa, rotation=45, ha="right", fontsize=6)
        axs[k_].set_title(f"{'C100' if ds == 'cifar100' else 'TIN'} w{w}", fontsize=8); axs[k_].tick_params(labelsize=6)
    axs[0].set_ylabel("top-1 accuracy (%)")
    plt.tight_layout(); plt.savefig("fig/g7_mb_allruns.pdf"); plt.close()

    # ------------------------------------------------------------ two cost views
    # MACs per image counted from the model definitions (tools/flops.py,
    # COMPUTE_ESTIMATE.md): convolutions, linears, and the per-sample mixing
    # of K kernels. This is the cost of an implementation that aggregates the
    # kernel per sample; the blend implementation used for training runs K
    # convolutions per dynamic layer and costs more.
    macs = {}
    for l in (ROOT / "COMPUTE_ESTIMATE.md").read_text().splitlines():
        x = [c.strip() for c in l.split("|")[1:-1]]
        if len(x) == 6 and x[0] in ("resnet18", "mobilenetv2"):
            macs[(x[0], x[1], int(x[2]), int(x[3]), x[4])] = float(x[5])
    PR = counted("resnet18")
    rows = []
    for name, model, getter, PP in [("ResNet-18", "resnet18", r18, PR), ("MobileNetV2", "mobilenetv2", mb, PM)]:
        for ds, wg, ws in [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]:
            q2 = paired(getter(ds, ws, "static"), getter(ds, wg, "gap"))
            q1 = paired(getter(ds, wg, "gap"), getter(ds, wg, "static"))
            pr_ = PP[(ds, wg, 4, "gap")][0] / PP[(ds, ws, 4, "static")][0]
            mr2 = macs[(model, ds, ws, 4, "static")] / macs[(model, ds, wg, 4, "gap")]
            mr1 = macs[(model, ds, wg, 4, "gap")] / macs[(model, ds, wg, 4, "static")]
            rows.append(f"{name} & {DSN[ds]} w{wg} & {pr_:.2f} & {mr2:.1f}$\\times$ & {sg(q2['mean'], 2)} ({verdict(q2)}) & "
                        f"{mr1:.2f}$\\times$ & {sg(q1['mean'], 2)} ({verdict(q1)}) \\\\")
    pathlib.Path("tab_COST.tex").write_text("\n".join(rows) + "\n")
    r18_eq = [paired(r18(ds, w, "gap"), r18(ds, w, "static")) for ds, w in COND]
    put("rbeqmac", str(sum(q["p"] < 0.05 and q["mean"] > 0 for q in r18_eq)))
    put("mbeqmac", str(sum(t["p"] < 0.05 and t["mean"] > 0 for _, _, t in dec.values())))
    put("rbmacextra", f"{100 * max(macs[('resnet18', ds, w, 4, 'gap')] / macs[('resnet18', ds, w, 4, 'static')] - 1 for ds, w in COND):.0f}")
    put("mbmacextra", f"{100 * max(macs[('mobilenetv2', ds, w, 4, 'gap')] / macs[('mobilenetv2', ds, w, 4, 'static')] - 1 for ds, w in COND):.0f}")

    entropy(results_dir, R, mb, r18, put)

    out = [f"\\expandafter\\def\\csname v{k}\\endcsname{{{v}}}" for k, v in M.items()]
    pathlib.Path("macros_mb.tex").write_text("\n".join(out) + "\n")
    return M


FLAT = 0.9999   # a layer is "flat" when its normalised entropy is at least this


def entropy(results_dir, R, mb, r18, put):
    """Routing-entropy analyses for MobileNetV2, from the result files.

    The result files are loaded with the strict loader (duplicates, recipe
    drift and parameter counts are checked there), then every run's final
    accuracy is compared with the value parsed from the log: they must be
    identical, and the final training loss must agree to the log's three
    decimals. Only then are the entropies used."""
    import json
    errs = S.load("mobilenetv2", [results_dir])
    if errs or S.n_runs() != 132 or S.MISSING:
        raise SystemExit(f"result files: errors {errs}, runs {S.n_runs()}, missing {len(S.MISSING)}")
    files = sorted(pathlib.Path(results_dir).glob("mobilenetv2__*_seed[0-9].json"))
    layers = {}
    for fp in files:
        j = json.loads(fp.read_text())
        _, key, arm, seed = S.cell_of(j)
        k = key + (arm, seed)
        if k not in R:
            raise SystemExit(f"{fp.name}: not in the log")
        if j["final_acc"] != R[k]["final"] or abs(j["history"][-1]["loss"] - R[k]["loss"]) > 5e-4 + 1e-9:
            raise SystemExit(f"{fp.name}: result file and log disagree")
        if j.get("routing"):
            layers[k] = np.array([l["z_entropy_norm"] for l in j["routing"]])
    if len(files) != 132:
        raise SystemExit(f"{len(files)} result files")
    put("mbjsonn", str(len(files)))
    nl = len(layers[("cifar100", 64, 4, "const", "gap", 0)])
    put("mbnlayers", str(nl))

    E = lambda ds, w, rec="const": S.ENT[(ds, w, rec)]
    H = lambda ds, w, rec="const": 1 - E(ds, w, rec).mean()
    HR = lambda w, rec: 1 - P.ENT[("cifar100", w, rec)].mean()
    rmsdev = lambda d: math.sqrt(d / (4 / (2 * math.log(4))) / 4)
    nflat = lambda ds, w, arm, rec="const", k=4: np.array([(layers[(ds, w, k, rec, arm, s)] >= FLAT).sum() for s in range(6)])
    rng = lambda v: f"{v.min()}" if v.min() == v.max() else f"{v.min()}--{v.max()}"

    # ---------------- table: entropy by arm at CIFAR-100 width 64, both backbones
    rows = []
    for a in ["uniform", "gap", "frozen_random", "shuffled"]:
        e, er = S.ENT_W64[a], P.ENT_W64[a]
        nf = np.array([(layers[("cifar100", 64, 4, "const", a, s)] >= FLAT).sum() for s in range(6)])
        rows.append(f"{tt(a)} & {f(er.mean(), 4)} & {f(1 - er.mean(), 5)} & {f(e.mean(), 5)} & {f(e.min(), 5)} & "
                    f"{f(e.max(), 5)} & {f(1 - e.mean(), 5)} & {rng(nf)} \\\\")
    pathlib.Path("tab_MBENTW64.tex").write_text("\n".join(rows) + "\n")
    put("mbeGap", f(H("cifar100", 64), 4)); put("mbeFrz", f(1 - S.ENT_W64["frozen_random"].mean(), 4))
    put("mbeShuf", f(1 - S.ENT_W64["shuffled"].mean(), 5))
    put("mbeGapMin", f(S.ENT_W64["gap"].min(), 4)); put("mbeGapMax", f(S.ENT_W64["gap"].max(), 4))
    put("mbeFrzMin", f(S.ENT_W64["frozen_random"].min(), 4)); put("mbeFrzMax", f(S.ENT_W64["frozen_random"].max(), 4))
    put("mbeFrzRatio", f"{H('cifar100', 64) / (1 - S.ENT_W64['frozen_random'].mean()):.1f}")
    put("mbeShufFlat", rng(nflat("cifar100", 64, "shuffled"))); put("mbeFrzFlat", rng(nflat("cifar100", 64, "frozen_random")))
    put("mbeGapFlat", rng(nflat("cifar100", 64, "gap")))
    put("rbeShuf", f(1 - P.ENT_W64["shuffled"].mean(), 4)); put("rbeFrz", f(1 - P.ENT_W64["frozen_random"].mean(), 4))

    # ---------------- table: entropy against benefit, MobileNetV2, every gap group with a uniform arm
    rows, sixmb, sixr18 = [], [], []
    conds = [("cifar100", 16, "const"), ("cifar100", 16, "anneal"), ("cifar100", 32, "const"), ("cifar100", 32, "anneal"),
             ("cifar100", 64, "const"), ("cifar100", 64, "anneal"), ("tinyimagenet", 32, "const"), ("tinyimagenet", 64, "const")]
    BEN = {}
    for ds, w, rec in conds:
        d = H(ds, w, rec)
        q = paired(mb(ds, w, "gap", rec=rec), mb(ds, w, "uniform")); BEN[(ds, w, rec)] = q
        nf = nflat(ds, w, "gap", rec)
        taul = "$1$" if rec == "const" else "$30\\!\\to\\!1$"
        rows.append(f"{DSN[ds]} & {w} & {taul} & {f(E(ds, w, rec).mean(), 5)} & "
                    f"{f(d, 5)} & {f(rmsdev(d), 4)} & {rng(nf)} & {sg(q['mean'])} & {pv(q['p'])} & {verdict(q)} \\\\")
        if ds == "cifar100":
            sixmb.append((d, q["mean"]))
            sixr18.append((HR(w, rec), paired(r18(ds, w, "gap", rec=rec), r18(ds, w, "uniform"))["mean"]))
    pathlib.Path("tab_MBENTBEN.tex").write_text("\n".join(rows) + "\n")
    rho_mb = stats.spearmanr(*zip(*sixmb)); rho_r = stats.spearmanr(*zip(*sixr18))
    put("mbrho", f"{rho_mb.statistic:.2f}"); put("mbrhop", f"{rho_mb.pvalue:.3f}")
    put("rbrho", f"{rho_r.statistic:.2f}"); put("rbrhop", f"{rho_r.pvalue:.3f}")
    for w, tag in [(16, "A"), (32, "B"), (64, "C")]:
        put(f"mbhc{tag}", f(H("cifar100", w), 4)); put(f"mbha{tag}", f(H("cifar100", w, "anneal"), 5))
        put(f"mbfl{tag}", f"{H('cifar100', w) / H('cifar100', w, 'anneal'):.1f}")
        put(f"mbbc{tag}", sg(BEN[("cifar100", w, "const")]["mean"], 2)); put(f"mbba{tag}", sg(BEN[("cifar100", w, "anneal")]["mean"], 2))
        put(f"mbbap{tag}", pv(BEN[("cifar100", w, "anneal")]["p"]))
        put(f"mbbav{tag}", verdict(BEN[("cifar100", w, "anneal")]))
        put(f"rbhc{tag}", f(HR(w, "const"), 4))
    put("mbflatAmin", str(min(nflat("cifar100", w, "gap", "anneal").min() for w in (16, 32, 64))))
    put("mbflatAmax", str(max(nflat("cifar100", w, "gap", "anneal").max() for w in (16, 32, 64))))
    put("mbflatCmin", str(min(nflat("cifar100", w, "gap").min() for w in (16, 32, 64))))
    put("mbflatCmax", str(max(nflat("cifar100", w, "gap").max() for w in (16, 32, 64))))
    # which layers carry the annealed routers' remaining departure from uniform
    ann = np.array([layers[("cifar100", w, 4, "anneal", "gap", s)] for w in (16, 32, 64) for s in range(6)])
    nonflat = sorted({int(i) for row in ann for i in np.where(row < FLAT)[0]})
    names = [str(i + 1) for i in nonflat]
    put("mbannlayers", names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1])
    r18w64 = paired(r18("cifar100", 64, "gap"), r18("cifar100", 64, "uniform"))
    put("rbguC", sg(r18w64["mean"], 2)); put("rbguCv", verdict(r18w64))
    put("mbguC", sg(BEN[("cifar100", 64, "const")]["mean"], 2)); put("mbguCp", pv(BEN[("cifar100", 64, "const")]["p"]))
    put("rbguA", sg(paired(r18("cifar100", 16, "gap"), r18("cifar100", 16, "uniform"))["mean"], 2))
    put("rbguAa", sg(paired(r18("cifar100", 16, "gap", rec="anneal"), r18("cifar100", 16, "uniform"))["mean"], 2))
    put("rbflA", f"{HR(16, 'const') / HR(16, 'anneal'):.0f}")

    # ---------------- table (appendix): per-layer departure from uniform, x 10^3
    cols = [("cifar100", 64, "const", "gap"), ("cifar100", 64, "anneal", "gap"), ("cifar100", 64, "const", "frozen_random"),
            ("cifar100", 64, "const", "shuffled"), ("cifar100", 16, "const", "gap"), ("cifar100", 16, "anneal", "gap"),
            ("tinyimagenet", 64, "const", "gap")]
    mat = {c: np.mean([1 - layers[(c[0], c[1], 4, c[2], c[3], s)] for s in range(6)], axis=0) for c in cols}
    rows = []
    for i in range(nl):
        rows.append(f"{i + 1} & " + " & ".join(f"{1e3 * mat[c][i]:.2f}" for c in cols) + " \\\\")
    rows.append("\\midrule mean & " + " & ".join(f"{1e3 * mat[c].mean():.2f}" for c in cols) + " \\\\")
    pathlib.Path("tab_MBLAYER.tex").write_text("\n".join(rows) + "\n")

    # ---------------- contradiction rows for the transfer table
    extra = [
        ("Entropy counterexample, CIFAR-100 w16",
         f"annealed {P_fl(HR)}$\\times$ flatter, larger benefit ({sg(paired(r18('cifar100',16,'gap',rec='anneal'),r18('cifar100',16,'uniform'))['mean'],2)} vs "
         f"{sg(paired(r18('cifar100',16,'gap'),r18('cifar100',16,'uniform'))['mean'],2)})",
         f"annealed {H('cifar100',16)/H('cifar100',16,'anneal'):.1f}$\\times$ flatter, smaller benefit "
         f"({sg(BEN[('cifar100',16,'anneal')]['mean'],2)} vs {sg(BEN[('cifar100',16,'const')]['mean'],2)})", "\\textbf{does not replicate}"),
        ("Constant-$\\tau$ $1-\\tilde H$ as width grows (w16, 32, 64)",
         ", ".join(f(HR(w, "const"), 4) for w in (16, 32, 64)) + " (flattens)",
         ", ".join(f(H("cifar100", w), 4) for w in (16, 32, 64)) + " (sharpens)", "\\textbf{differs}"),
        ("Same $1-\\tilde H$, CIFAR-100 w64, \\texttt{gap}$-$\\texttt{uniform}",
         f"{f(HR(64,'const'),4)}: {sg(r18w64['mean'],2)} ({verdict(r18w64)})",
         f"{f(H('cifar100',64),4)}: {sg(BEN[('cifar100',64,'const')]['mean'],2)} ({verdict(BEN[('cifar100',64,'const')])})", "new"),
    ]
    with open("tab_CONTRA.tex", "a") as fh:
        fh.write("\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in extra) + "\n")

    # ---------------- figure G8
    fig, ax = plt.subplots(1, 3, figsize=(7.9, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15, 1.2]))
    W = [16, 32, 64]
    for name, hf, col, mk in [("ResNet-18", lambda w, rec: HR(w, rec), C["r18"], "s"), ("MobileNetV2", lambda w, rec: H("cifar100", w, rec), C["mb"], "o")]:
        ax[0].plot(W, [hf(w, "const") for w in W], marker=mk, ms=4.5, lw=1.3, color=col, label=f"{name}, $\\tau=1$")
        ax[0].plot(W, [hf(w, "anneal") for w in W], marker=mk, ms=4.5, lw=1.3, ls="--", color=col, mfc="white",
                   label=f"{name}, $\\tau:30\\to1$")
    ax[0].set_yscale("log"); ax[0].set_xscale("log", base=2); ax[0].set_xticks(W); ax[0].set_xticklabels(W)
    ax[0].set_xlabel("base width $c$"); ax[0].set_ylabel("$1-\\tilde H$ (log)"); ax[0].legend(fontsize=5.6, loc="upper right", handlelength=2.2)
    ax[0].set_ylim(top=ax[0].get_ylim()[1] * 6)
    ax[0].set_title("(a) router flatness, CIFAR-100", fontsize=9)
    # label offsets (points) chosen so that labels of neighbouring markers do not overlap
    OFF = {("MobileNetV2", 16, "const"): (-5, -2), ("MobileNetV2", 32, "const"): (-5, 1), ("MobileNetV2", 64, "const"): (4, 2),
           ("MobileNetV2", 16, "anneal"): (4, 1), ("MobileNetV2", 32, "anneal"): (-5, -5), ("MobileNetV2", 64, "anneal"): (-5, 2),
           ("ResNet-18", 64, "anneal"): (4, 3), ("ResNet-18", 64, "const"): (4, 3)}
    for name, pts_, col, mk in [("ResNet-18", sixr18, C["r18"], "s"), ("MobileNetV2", sixmb, C["mb"], "o")]:
        for (x, y), (w, rec) in zip(pts_, [(w, r) for w in W for r in ("const", "anneal")]):
            ax[1].scatter(x, y, marker=mk, s=30, color=col if rec == "const" else "white", edgecolor=col, zorder=3,
                          label=name if (w, rec) == (16, "const") else None)
            off = OFF.get((name, w, rec), (4, -2))
            ax[1].annotate(f"w{w}{'' if rec == 'const' else 'a'}", (x, y), xytext=off, textcoords="offset points",
                           fontsize=5.8, ha="right" if off[0] < 0 else "left")
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xscale("log"); ax[1].set_xlabel("$1-\\tilde H$ (log)")
    ax[1].set_ylabel("gap $-$ uniform (pp)"); ax[1].legend(fontsize=6.5, loc="upper left")
    ax[1].set_title("(b) flatness vs benefit (a = annealed)", fontsize=9)
    x = np.arange(1, nl + 1)
    for (c, lbl, col, ls) in [(("cifar100", 64, "const", "gap"), "gap, $\\tau=1$", C["gap"], "-"),
                              (("cifar100", 64, "anneal", "gap"), "gap, annealed", C["ann"], "--"),
                              (("cifar100", 64, "const", "frozen_random"), "frozen_random", C["frozen_random"], "-."),
                              (("cifar100", 64, "const", "shuffled"), "shuffled", C["shuffled"], ":")]:
        ax[2].plot(x, 1e3 * mat[c], marker="o", ms=2.5, lw=1.2, color=col, ls=ls, label=lbl)
    ax[2].set_xlabel("dynamic layer (input to output)"); ax[2].set_ylabel("$10^3\\,(1-\\tilde H)$")
    ax[2].set_xticks([1, 5, 9, 13, 17]); ax[2].legend(fontsize=6.3, loc="upper left")
    ax[2].set_title("(c) MobileNetV2 by layer, C100 w64", fontsize=9)
    plt.tight_layout(); plt.savefig("fig/g8_entropy.pdf"); plt.close()


def P_fl(HR):
    return f"{HR(16, 'const') / HR(16, 'anneal'):.0f}"


if __name__ == "__main__":
    M = main(sys.argv[1], sys.argv[2])
    for k, v in M.items():
        print(f"{k:16} {v}")

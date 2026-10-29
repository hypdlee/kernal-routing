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

import plotstyle as PS
PS.apply()
C = {"static": "#444444", "uniform": "#1f77b4", "gap": "#d62728", "shuffled": "#2ca02c",
     "frozen_random": "#9467bd", "ann": "#ff7f0e", "r18": "#7f7f7f", "mb": "#1f6fb4", "vg": "#b8860b"}
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


def main(log_path, results_dir, vgg_log, vgg_results=None, r18_data=None):
    """r18_data: the report's data module after loading the 132 ResNet-18
    result files; when given it replaces the transcribed values of
    paper_r18_data.py, which the build has already checked against it."""
    global P
    if r18_data is not None:
        P = r18_data
    R = mbv2_log.parse(log_path)
    assert len(R) == 132, len(R)
    PM = counted("mobilenetv2")
    RV = mbv2_log.parse(vgg_log, "vgg16")
    assert len(RV) == 132, len(RV)

    def vg(ds, w, arm, k=4, rec="const", field="final"):
        return np.array([RV[(ds, w, k, rec, arm, s)][field] for s in range(6)], dtype=float)

    def mb(ds, w, arm, k=4, rec="const", field="final"):
        return np.array([R[(ds, w, k, rec, arm, s)][field] for s in range(6)], dtype=float)

    r18 = lambda ds, w, arm, k=4, rec="const": P.ACC[(ds, w, k, rec)][arm]
    M = {}
    put = M.__setitem__
    pathlib.Path("fig").mkdir(exist_ok=True)

    # ------------------------------------------------------------ headline numbers
    put("mbruns", str(len(R)))
    put("mbgpuh", f"{sum(r['minutes'] for r in R.values()) / 60:.0f}")
    put("totruns", str(len(R) + len(RV) + P.n_runs()))
    put("totgpuh", f"{sum(r['minutes'] for r in R.values()) / 60 + sum(r['minutes'] for r in RV.values()) / 60 + P.gpu_hours():.0f}")
    put("twogpuh", f"{sum(r['minutes'] for r in R.values()) / 60 + P.gpu_hours():.0f}")
    vdec, vpairs = vgg_part(RV, vg, put)
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
             ("frozen_random", "gap"), ("shuffled", "gap"), ("frozen_random", "uniform"), ("gap", "uniform")]
    rows = []
    for a, b_ in comps:
        qm = paired(mb("cifar100", 64, a), mb("cifar100", 64, b_))
        qr = paired(r18("cifar100", 64, a), r18("cifar100", 64, b_))
        qv = paired(vg("cifar100", 64, a), vg("cifar100", 64, b_))
        rows.append(f"{tt(a)} $-$ {tt(b_)} & {sg(qr['mean'])} & {pv(qr['p'])} & {verdict(qr)} & {sg(qm['mean'])} & "
                    f"{pv(qm['p'])} & {verdict(qm)} & {sg(qv['mean'])} & {pv(qv['p'])} & {verdict(qv)} \\\\")
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
    inv = [PM[(ds, ws, 4, "static")][0] / PM[(ds, wg, 4, "gap")][0]
           for ds, wg, ws in [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]]
    put("mbpinvlo", f"{min(inv):.1f}"); put("mbpinvhi", f"{max(inv):.1f}")

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

    # ------------------------------------------------------------ the result-by-result table (three backbones)
    rbank = sum(verdict(b) == "equivalent" for b in r18bank)
    r18_w64 = paired(r18("cifar100", 64, "gap"), r18("cifar100", 64, "static"))
    mb_w64 = dec[("cifar100", 64)][2]
    vg_w64 = vdec[("cifar100", 64)][2]
    r18_pairs = [paired(r18(ds, ws, "static"), r18(ds, wg, "gap")) for ds, wg, ws in
                 [("cifar100", 16, 32), ("cifar100", 32, 64), ("tinyimagenet", 32, 64)]]
    r18_ann = [paired(r18("cifar100", w, "gap", rec="anneal"), r18("cifar100", w, "gap")) for w in (16, 32, 64)]
    mb_ann = [paired(mb("cifar100", w, "gap", rec="anneal"), mb("cifar100", w, "gap")) for w in (16, 32, 64)]
    vg_ann = [paired(vg("cifar100", w, "gap", rec="anneal"), vg("cifar100", w, "gap")) for w in (16, 32, 64)]
    W = lambda g, a, b_: paired(g("cifar100", 64, a), g("cifar100", 64, b_))
    k82 = lambda g: paired(g("cifar100", 32, "gap", k=8), g("cifar100", 32, "gap", k=2))
    nsig_r18 = sum(q["p"] < 0.05 and q["mean"] > 0 for q in r18_pairs)
    nsig_vg = sum(q["p"] < 0.05 and q["mean"] > 0 for q in vpairs)
    lst = lambda qs: ", ".join(sg(q["mean"], 2) for q in qs)
    vnb = sum(verdict(b) == "equivalent" for b, _, _ in vdec.values())
    vnr = sum(r["p"] < 0.05 and r["mean"] > 0 for _, r, _ in vdec.values())
    largest = lambda d: f"{max((b['mean'] for b, _, _ in d.values()), key=abs):+.2f}"
    ctl = lambda g: f"{sg(W(g, 'gap', 'uniform')['mean'], 2)} / {sg(W(g, 'frozen_random', 'uniform')['mean'], 2)} ({verdict(W(g, 'gap', 'uniform'))} / {verdict(W(g, 'frozen_random', 'uniform'))})"
    contra = [
        ("Bank term ($\\texttt{uniform}-\\texttt{static}$)", f"equivalent in {rbank} of 5",
         f"equivalent in {nb} of 5; never significant (largest {largest(dec)})",
         f"equivalent in {vnb} of 5; never significant (largest {largest(vdec)})", "holds on all three (pooled, equivalent to zero; per pair weaker on MobileNetV2 and VGG-16)"),
        ("Routing term significant", f"{sum(paired(r18(ds,w,'gap'),r18(ds,w,'uniform'))['p']<0.05 for ds,w in COND)} of 5", f"{nr} of 5",
         f"{vnr} of 5", "holds on all three"),
        ("$\\texttt{gap}-\\texttt{static}$, CIFAR-100 w64", f"{sg(r18_w64['mean'],2)} ({verdict(r18_w64)})", f"{sg(mb_w64['mean'],2)} ({verdict(mb_w64)})",
         f"{sg(vg_w64['mean'],2)} ({verdict(vg_w64)})", "\\textbf{backbone-dependent}"),
        ("Plain $2c$ beats \\texttt{gap} $c$ (3 pairs)", f"{nsig_r18} of 3 ({lst(r18_pairs)})", f"{np2} of 3 ({lst(pairs)})",
         f"{nsig_vg} of 3 ({lst(vpairs)})", "\\textbf{backbone-dependent}"),
        ("Annealed $-$ constant $\\tau$ (w16, 32, 64)", lst(r18_ann), lst(mb_ann), lst(vg_ann), "holds on all three (never significantly higher)"),
        ("\\texttt{gap} / \\texttt{frozen\\_random} $-$ \\texttt{uniform}, w64", ctl(r18), ctl(mb), ctl(vg), "\\textbf{backbone-dependent}"),
        ("$K=8$ vs $K=2$, CIFAR-100 w32", f"{sg(k82(r18)['mean'],2)} ({verdict(k82(r18))})", f"{sg(k82(mb)['mean'],2)} ({verdict(k82(mb))})",
         f"{sg(k82(vg)['mean'],2)} ({verdict(k82(vg))})", "holds on all three"),
    ]
    pathlib.Path("tab_CONTRA.tex").write_text("\n".join(f"{a} & {b} & {c} & {d} & {e} \\\\" for a, b, c, d, e in contra) + "\n")
    put("rbmpairs", lst(r18_pairs)); put("mbmpairs", lst(pairs)); put("vgmpairs", lst(vpairs))
    put("rbanlist", lst(r18_ann)); put("mbanlist", lst(mb_ann))
    put("nannlower", str(sum(q["p"] < 0.05 and q["mean"] < 0 for q in r18_ann + mb_ann)))
    put("nannhigher", str(sum(q["p"] < 0.05 and q["mean"] > 0 for q in r18_ann + mb_ann)))
    put("allannlower", str(sum(q["p"] < 0.05 and q["mean"] < 0 for q in r18_ann + mb_ann + vg_ann)))
    put("allannhigher", str(sum(q["p"] < 0.05 and q["mean"] > 0 for q in r18_ann + mb_ann + vg_ann)))
    put("vgannlower", str(sum(q["p"] < 0.05 and q["mean"] < 0 for q in vg_ann)))
    rbn = sum(paired(r18(ds, w, 'gap'), r18(ds, w, 'uniform'))['p'] < 0.05 for ds, w in COND)
    put("rbnrout", str(rbn)); put("bothrout", str(rbn + nr)); put("allrout", str(rbn + nr + vnr))
    put("rbnbanksig", str(sum(b["p"] < 0.05 for b in r18bank)))
    put("bothbanksig", str(sum(b["p"] < 0.05 for b in r18bank) + sum(b["p"] < 0.05 for b, _, _ in dec.values())))
    put("allbanksig", str(sum(b["p"] < 0.05 for b in r18bank) + sum(b["p"] < 0.05 for b, _, _ in dec.values())
                          + sum(b["p"] < 0.05 for b, _, _ in vdec.values())))
    put("allbankeq", str(rbank + nb + vnb))
    put("allP2", str(nsig_r18 + nsig_vg))
    # pairs with an effect over static, and whether the routing term is significant in them
    effpairs = ([(paired(r18(ds, w, "gap"), r18(ds, w, "static")), paired(r18(ds, w, "gap"), r18(ds, w, "uniform"))) for ds, w in COND]
                + [(t_, r_) for _, r_, t_ in dec.values()] + [(t_, r_) for _, r_, t_ in vdec.values()])
    eff = [(t_, r_) for t_, r_ in effpairs if t_["p"] < 0.05 and t_["mean"] > 0]
    put("alleff", str(len(eff))); put("alleffrout", str(sum(r_["p"] < 0.05 and r_["mean"] > 0 for _, r_ in eff)))
    put("allnull", str(len(effpairs) - len(eff)))
    eqp = [q["mean"] for q in r18_pairs + vpairs]
    put("eqpmin", f(min(eqp), 1)); put("eqpmax", f(max(eqp), 1))
    q = W(r18, "shuffled", "gap"); put("rbwshuffledgap", sg(q["mean"])); put("rbwshuffledgapv", verdict(q))
    # predictions recorded before the runs that test them
    r18tb = paired(r18("tinyimagenet", 64, "gap"), r18("tinyimagenet", 64, "static"))
    outcomes = ([verdict(r18tb) == "equivalent"] + [nb == 5, np2 == 3, nr >= 3]
                + [vnb == 5, nsig_vg == 3, vnr >= 3])
    put("allpred", str(len(outcomes))); put("allpredfail", str(sum(not o for o in outcomes)))
    put("allpredheld", str(sum(outcomes)))

    # ------------------------------------------------------------ loss vs effect (both backbones)
    pts = []
    for ds, w in COND:
        pts.append(("ResNet-18", ds, w, P.LOSS[(ds, w, 4, "const")]["static"], paired(r18(ds, w, "gap"), r18(ds, w, "static"))))
        pts.append(("MobileNetV2", ds, w, mb(ds, w, "static", field="loss").mean(), dec[(ds, w)][2]))
        pts.append(("VGG-16", ds, w, vg(ds, w, "static", field="loss").mean(), vdec[(ds, w)][2]))
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
    fig, axs = plt.subplots(1, 3, figsize=(7.9, 2.8), sharey=True)
    xs = np.arange(5); bw = .36
    labels = ["C100\nw16", "C100\nw32", "C100\nw64", "TIN\nw32", "TIN\nw64"]
    for ax, name, getter in [(axs[0], "(a) ResNet-18", r18), (axs[1], "(b) MobileNetV2", mb), (axs[2], "(c) VGG-16", vg)]:
        for j, (lbl, fn, col) in enumerate([("bank (uniform $-$ static)", lambda d, w: paired(getter(d, w, "uniform"), getter(d, w, "static")), C["uniform"]),
                                            ("routing (gap $-$ uniform)", lambda d, w: paired(getter(d, w, "gap"), getter(d, w, "uniform")), C["gap"])]):
            qs = [fn(d, w) for d, w in COND]
            ax.bar(xs + (j - .5) * bw, [q["mean"] for q in qs], bw, color=col, label=lbl,
                   yerr=[[q["mean"] - q["lo"] for q in qs], [q["hi"] - q["mean"] for q in qs]], capsize=2.5, error_kw=dict(lw=.9))
        ax.axhline(0, color="k", lw=.8); ax.axhspan(-.5, .5, color="grey", alpha=.10)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.5); ax.set_title(name)
    axs[0].set_ylabel("$\\Delta$ accuracy (pp)")
    plt.tight_layout(); PS.place_legend(axs[0])
    PS.save("fig/g1_decomp_both.pdf"); plt.close()

    # G2: MobileNetV2 width 64, five arms; and controls on both backbones
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for i, a in enumerate(arms):
        v = mb("cifar100", 64, a)
        ax[0].scatter([i] * 6, v, s=22, color=C[a], alpha=.85, zorder=3)
        ax[0].hlines(v.mean(), i - .3, i + .3, color="k", lw=1.5, zorder=4)
    s0 = mb("cifar100", 64, "static").mean(); ax[0].axhspan(s0 - .5, s0 + .5, color="grey", alpha=.13)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(arms, rotation=35, ha="right", fontsize=7)
    ax[0].set_ylabel("top-1 accuracy (%)"); ax[0].set_title("(a) MobileNetV2, CIFAR-100, width 64")
    cl = [("gap", "static"), ("frozen_random", "static"), ("shuffled", "static"), ("gap", "uniform"), ("frozen_random", "uniform"),
          ("frozen_random", "gap"), ("shuffled", "gap")]
    for j, (name, getter, col) in enumerate([("ResNet-18", r18, C["r18"]), ("MobileNetV2", mb, C["mb"]), ("VGG-16", vg, C["vg"])]):
        qs = [paired(getter("cifar100", 64, a), getter("cifar100", 64, b_)) for a, b_ in cl]
        y = np.arange(len(cl)) + (j - 1) * .25
        ax[1].errorbar([q["mean"] for q in qs], y, xerr=[[q["mean"] - q["lo"] for q in qs], [q["hi"] - q["mean"] for q in qs]],
                       fmt="o", ms=4, capsize=2.5, color=col, label=name)
    ax[1].axvline(0, color="k", lw=.8); ax[1].axvspan(-.5, .5, color="grey", alpha=.13)
    ax[1].set_yticks(range(len(cl))); ax[1].set_yticklabels([f"{a} $-$ {b_}" for a, b_ in cl], fontsize=7)
    ax[1].invert_yaxis(); ax[1].set_xlabel("$\\Delta$ (pp), 95% CI")
    ax[1].set_title("(b) routing controls, three backbones")
    plt.tight_layout(w_pad=1.0); PS.place_legend(ax[1])
    PS.save("fig/g2_mb_w64.pdf"); plt.close()

    # G3: gap - static by width, both backbones, both datasets
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for k_, (ds, ws, ttl) in enumerate([("cifar100", [16, 32, 64], "(a) CIFAR-100"), ("tinyimagenet", [32, 64], "(b) Tiny-ImageNet")]):
        for name, getter, col, mk, off in [("ResNet-18", r18, C["r18"], "s", .96), ("MobileNetV2", mb, C["mb"], "o", 1.0),
                                           ("VGG-16", vg, C["vg"], "^", 1.04)]:
            eb(ax[k_], np.array(ws) * off, [paired(getter(ds, w, "gap"), getter(ds, w, "static")) for w in ws], marker=mk, ms=4.5, lw=1.4,
               color=col, label=name)
        ax[k_].axhline(0, color="k", lw=.8); ax[k_].axhspan(-.5, .5, color="grey", alpha=.13)
        ax[k_].set_xscale("log", base=2); ax[k_].set_xticks([16, 32, 64]); ax[k_].set_xticklabels([16, 32, 64])
        ax[k_].set_xlabel("base width $c$"); ax[k_].set_title(ttl)
    ax[0].set_ylabel("gap $-$ static (pp)")
    plt.tight_layout(); PS.place_legend(ax[0])
    PS.save("fig/g3_width_both.pdf"); plt.close()

    # G4: matched budgets, MobileNetV2
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    lab4 = [[], []]
    for k_, (ds, ws, ttl) in enumerate([("cifar100", [16, 32, 64], "(a) CIFAR-100"), ("tinyimagenet", [32, 64], "(b) Tiny-ImageNet")]):
        for a in ["static", "uniform", "gap"]:
            p_ = [PM[(ds, w, 4, a)][0] for w in ws]; m_ = [mb(ds, w, a).mean() for w in ws]
            ax[k_].plot(p_, m_, marker="o", ms=4, lw=1.3, color=C[a], label=a)
            lab4[k_] += [(pp, mm, f"w{w}") for w, pp, mm in zip(ws, p_, m_)]
        for wg, wsx in ([(16, 32), (32, 64)] if ds == "cifar100" else [(32, 64)]):
            xg, yg = PM[(ds, wg, 4, "gap")][0], mb(ds, wg, "gap").mean()
            xs_, ys_ = PM[(ds, wsx, 4, "static")][0], mb(ds, wsx, "static").mean()
            ax[k_].plot([xg, xs_], [yg, ys_], ls=":", color="k", lw=1)
            lab4[k_].append((math.sqrt(xg * xs_), (yg + ys_) / 2, f"${ys_ - yg:+.2f}$\n(params $\\times{xg / xs_:.2f}$)"))
        ax[k_].set_xscale("log"); ax[k_].set_xlabel("stored parameters (M, log)"); ax[k_].set_title(ttl)
    ax[0].set_ylabel("top-1 accuracy (%)")
    plt.tight_layout()
    for k_ in range(2):
        PS.place_legend(ax[k_]); PS.place_labels(ax[k_], lab4[k_])
    PS.save("fig/g4_mb_matched.pdf"); plt.close()

    # G5: temperature and bank size, both backbones
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for name, getter, col, mk, off in [("ResNet-18", r18, C["r18"], "s", .95), ("MobileNetV2", mb, C["mb"], "o", 1.0),
                                       ("VGG-16", vg, C["vg"], "^", 1.05)]:
        eb(ax[0], np.array([16, 32, 64]) * off, [paired(getter("cifar100", w, "gap", rec="anneal"), getter("cifar100", w, "gap")) for w in (16, 32, 64)],
           marker=mk, ms=4.5, lw=1.3, color=col, label=name)
        eb(ax[1], np.array([2, 4, 8]) * off, [paired(getter("cifar100", 32, "gap", k=kk), getter("cifar100", 32, "static")) for kk in (2, 4, 8)],
           marker=mk, ms=4.5, lw=1.3, color=col, label=name)
    for a_, xl, tl in [(ax[0], "base width $c$", "(a) annealed $-$ constant $\\tau$, gap"), (ax[1], "bank size $K$", "(b) gap $-$ static, CIFAR-100 w32")]:
        a_.axhline(0, color="k", lw=.8); a_.set_xscale("log", base=2); a_.set_xlabel(xl); a_.set_title(tl)
    ax[0].set_xticks([16, 32, 64]); ax[0].set_xticklabels([16, 32, 64]); ax[1].set_xticks([2, 4, 8]); ax[1].set_xticklabels([2, 4, 8])
    ax[0].set_ylabel("$\\Delta$ (pp)")
    plt.tight_layout(); PS.place_legend(ax[0])
    PS.save("fig/g5_temp_k.pdf"); plt.close()

    # G6: training loss and the saturation scatter, both backbones
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    EP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 149]
    for (ds, w, a), ls in [(("cifar100", 16, "static"), ":"), (("cifar100", 32, "static"), "--"), (("cifar100", 64, "static"), "-"),
                           (("tinyimagenet", 64, "static"), "-."), (("tinyimagenet", 64, "gap"), (0, (1, 1)))]:
        cv = R[(ds, w, 4, "const", a, 0)]["curve"]
        ax[0].plot(EP, [cv[e]["loss"] for e in EP], ls=ls, color=C["mb"] if ds == "cifar100" else "#e377c2", lw=1.3,
                   label=f"{'CIFAR' if ds == 'cifar100' else 'TIN'} w{w} {a}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss (log)")
    ax[0].set_title("(a) MobileNetV2 training loss, seed 0")
    lab6 = []
    for name, col, mk in [("ResNet-18", C["r18"], "s"), ("MobileNetV2", C["mb"], "o"), ("VGG-16", C["vg"], "^")]:
        sel = [p for p in pts if p[0] == name]
        ax[1].errorbar([p[3] for p in sel], [p[4]["mean"] for p in sel],
                       yerr=[[p[4]["mean"] - p[4]["lo"] for p in sel], [p[4]["hi"] - p[4]["mean"] for p in sel]],
                       fmt=mk, ms=4.5, capsize=3, color=col, label=name)
        lab6 += [(p[3], p[4]["mean"], f"{'T' if p[1] == 'tinyimagenet' else 'C'}{p[2]}") for p in sel]
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xscale("log")
    ax[1].set_xlabel("static arm final training loss (log)"); ax[1].set_ylabel("gap $-$ static (pp)")
    ax[1].set_title("(b) effect against training loss, 15 conditions")
    plt.tight_layout(); PS.place_legend(ax[0]); PS.place_legend(ax[1]); PS.place_labels(ax[1], lab6, fontsize=5.8)
    PS.save("fig/g6_loss_effect.pdf"); plt.close()

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
    plt.tight_layout(); PS.save("fig/g7_mb_allruns.pdf"); plt.close()

    # ------------------------------------------------------------ two cost views
    # MACs per image counted from the model definitions (tools/flops.py,
    # COMPUTE_ESTIMATE.md): convolutions, linears, and the per-sample mixing
    # of K kernels. This is the cost of an implementation that aggregates the
    # kernel per sample; the blend implementation used for training runs K
    # convolutions per dynamic layer and costs more.
    macs = {}
    for l in (ROOT / "COMPUTE_ESTIMATE.md").read_text().splitlines():
        x = [c.strip() for c in l.split("|")[1:-1]]
        if len(x) == 6 and x[0] in ("resnet18", "mobilenetv2", "vgg16"):
            macs[(x[0], x[1], int(x[2]), int(x[3]), x[4])] = float(x[5])
    PR = counted("resnet18")
    rows = []
    for name, model, getter, PP in [("ResNet-18", "resnet18", r18, PR), ("MobileNetV2", "mobilenetv2", mb, PM),
                                    ("VGG-16", "vgg16", vg, counted("vgg16"))]:
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
    put("botheff", str(sum(q["p"] < 0.05 and q["mean"] > 0 for q in r18_eq)
                       + sum(t["p"] < 0.05 and t["mean"] > 0 for _, _, t in dec.values())))
    put("mbeqmac", str(sum(t["p"] < 0.05 and t["mean"] > 0 for _, _, t in dec.values())))
    put("rbmacextra", f"{100 * max(macs[('resnet18', ds, w, 4, 'gap')] / macs[('resnet18', ds, w, 4, 'static')] - 1 for ds, w in COND):.0f}")
    put("vgmacextra", f"{100 * max(macs[('vgg16', ds, w, 4, 'gap')] / macs[('vgg16', ds, w, 4, 'static')] - 1 for ds, w in COND):.0f}")
    put("vgeqmac", str(sum(t["p"] < 0.05 and t["mean"] > 0 for _, _, t in vdec.values())))
    allmac = [macs[(m_, ds, w, 4, 'gap')] / macs[(m_, ds, w, 4, 'static')] - 1 for m_ in ("resnet18", "mobilenetv2", "vgg16") for ds, w in COND]
    put("allmacextra", f"{100 * max(allmac):.0f}")
    put("mbmacextra", f"{100 * max(macs[('mobilenetv2', ds, w, 4, 'gap')] / macs[('mobilenetv2', ds, w, 4, 'static')] - 1 for ds, w in COND):.0f}")

    pooled_part(r18, mb, vg, put)
    import controls_section
    controls_section.main(put)
    import partial_section
    partial_section.main(put)
    if vgg_results is None:
        raise SystemExit("--vgg-results is required: the VGG-16 routing entropies are in its result files")
    entropy(results_dir, R, mb, r18, put, vgg_results, RV, vg)

    out = [f"\\expandafter\\def\\csname v{k}\\endcsname{{{v}}}" for k, v in M.items()]
    pathlib.Path("macros_mb.tex").write_text("\n".join(out) + "\n")
    return M


FLAT = 0.9999   # a layer is "flat" when its normalised entropy is at least this


def vgg_layers(vgg_results, RV):
    """The 132 VGG-16 result files, loaded with the strict loader (one recipe,
    no duplicate, parameter counts as counted from the model definition),
    then checked run by run against the training log: final accuracy
    identical, final training loss equal to the log's three decimals,
    parameter count equal to the log's two decimals, and every routing record
    taken on the same 16 test images. Returns the per-layer entropies and the
    loader holding the per-seed entropy arrays."""
    import json
    spec = importlib.util.spec_from_file_location("kr_vgg", HERE / "data.py")
    SV = importlib.util.module_from_spec(spec); spec.loader.exec_module(SV)
    errs = SV.load("vgg16", [vgg_results])
    if errs or SV.n_runs() != 132 or SV.MISSING:
        raise SystemExit(f"VGG-16 result files: errors {errs}, runs {SV.n_runs()}, missing {len(SV.MISSING)}")
    files = sorted(pathlib.Path(vgg_results).glob("vgg16__*_seed[0-9].json"))
    if len(files) != 132:
        raise SystemExit(f"{len(files)} VGG-16 result files")
    layers = {}
    for fp in files:
        j = json.loads(fp.read_text())
        _, key, arm, seed = SV.cell_of(j)
        k = key + (arm, seed)
        if k not in RV:
            raise SystemExit(f"{fp.name}: not in the VGG-16 log")
        if (j["final_acc"] != RV[k]["final"] or abs(j["history"][-1]["loss"] - RV[k]["loss"]) > 5e-4 + 1e-9
                or round(j["params"]["total"] / 1e6, 2) != RV[k]["params"]):
            raise SystemExit(f"{fp.name}: result file and log disagree")
        if j.get("routing"):
            if any(sum(l["argmax_counts"]) != 16 for l in j["routing"]):
                raise SystemExit(f"{fp.name}: routing not recorded on 16 images")
            layers[k] = np.array([l["z_entropy_norm"] for l in j["routing"]])
    return layers, SV, len(files)


def trend(Hf):
    v = [Hf("cifar100", w) for w in (16, 32, 64)]
    return " (flattens)" if v[0] > v[1] > v[2] else " (sharpens)" if v[0] < v[1] < v[2] else " (no monotone trend)"


def entropy(results_dir, R, mb, r18, put, vgg_results, RV, vg):
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

    # ---------------- VGG-16, from its result files
    VL, SV, nvj = vgg_layers(vgg_results, RV)
    put("vgjsonn", str(nvj))
    vnl = len(VL[("cifar100", 64, 4, "const", "gap", 0)])
    if vnl != counted_layers("vgg16")[("cifar100", 64, 4, "gap")]:
        raise SystemExit("VGG-16 routing records and dynamic-layer count disagree")
    VE = lambda ds, w, rec="const": SV.ENT[(ds, w, rec)]
    VH = lambda ds, w, rec="const": 1 - VE(ds, w, rec).mean()
    vflat = lambda ds, w, arm, rec="const", k=4: np.array([(VL[(ds, w, k, rec, arm, s)] >= FLAT).sum() for s in range(6)])
    rows = []
    for a in ["uniform", "gap", "frozen_random", "shuffled"]:
        e = SV.ENT_W64[a]
        rows.append(f"{tt(a)} & {f(e.mean(), 5)} & {f(e.min(), 5)} & {f(e.max(), 5)} & {f(1 - e.mean(), 5)} & "
                    f"{rng(vflat('cifar100', 64, a))} \\\\")
    pathlib.Path("tab_VGENTW64.tex").write_text("\n".join(rows) + "\n")
    put("vgeGap", f(VH("cifar100", 64), 4)); put("vgeFrz", f(1 - SV.ENT_W64["frozen_random"].mean(), 4))
    put("vgeShuf", f(1 - SV.ENT_W64["shuffled"].mean(), 4))
    put("vgeFrzRatio", f"{VH('cifar100', 64) / (1 - SV.ENT_W64['frozen_random'].mean()):.1f}")
    put("vgeShufRatio", f"{VH('cifar100', 64) / (1 - SV.ENT_W64['shuffled'].mean()):.1f}")
    put("vgeShufFlat", rng(vflat("cifar100", 64, "shuffled"))); put("vgeFrzFlat", rng(vflat("cifar100", 64, "frozen_random")))
    put("vgeGapFlat", rng(vflat("cifar100", 64, "gap")))
    put("vgeShufOverMb", f"{(1 - SV.ENT_W64['shuffled'].mean()) / (1 - S.ENT_W64['shuffled'].mean()):.0f}")
    put("vgeShufOverRb", f"{(1 - SV.ENT_W64['shuffled'].mean()) / (1 - P.ENT_W64['shuffled'].mean()):.0f}")
    rows, sixvg, VBEN = [], [], {}
    for ds, w, rec in conds:
        d = VH(ds, w, rec)
        q = paired(vg(ds, w, "gap", rec=rec), vg(ds, w, "uniform")); VBEN[(ds, w, rec)] = q
        taul = "$1$" if rec == "const" else "$30\\!\\to\\!1$"
        rows.append(f"{DSN[ds]} & {w} & {taul} & {f(VE(ds, w, rec).mean(), 5)} & "
                    f"{f(d, 5)} & {f(rmsdev(d), 4)} & {rng(vflat(ds, w, 'gap', rec))} & {sg(q['mean'])} & {pv(q['p'])} & {verdict(q)} \\\\")
        if ds == "cifar100":
            sixvg.append((d, q["mean"]))
    pathlib.Path("tab_VGENTBEN.tex").write_text("\n".join(rows) + "\n")
    rho_v = stats.spearmanr(*zip(*sixvg))
    put("vgrho", f"{rho_v.statistic:.2f}"); put("vgrhop", f"{rho_v.pvalue:.3f}")
    for w, tag in [(16, "A"), (32, "B"), (64, "C")]:
        put(f"vghc{tag}", f(VH("cifar100", w), 4)); put(f"vgha{tag}", f(VH("cifar100", w, "anneal"), 5))
        put(f"vgfl{tag}", f"{VH('cifar100', w) / VH('cifar100', w, 'anneal'):.1f}")
        put(f"vgbc{tag}", sg(VBEN[("cifar100", w, "const")]["mean"], 2)); put(f"vgba{tag}", sg(VBEN[("cifar100", w, "anneal")]["mean"], 2))
        put(f"vgbcp{tag}", pv(VBEN[("cifar100", w, "const")]["p"])); put(f"vgbap{tag}", pv(VBEN[("cifar100", w, "anneal")]["p"]))
        put(f"vgbav{tag}", verdict(VBEN[("cifar100", w, "anneal")]))
    put("vgflatAmin", str(min(vflat("cifar100", w, "gap", "anneal").min() for w in (16, 32, 64))))
    put("vgflatAmax", str(max(vflat("cifar100", w, "gap", "anneal").max() for w in (16, 32, 64))))
    put("vgflatCmin", str(min(vflat("cifar100", w, "gap").min() for w in (16, 32, 64))))
    put("vgflatCmax", str(max(vflat("cifar100", w, "gap").max() for w in (16, 32, 64))))
    vann = np.array([VL[("cifar100", w, 4, "anneal", "gap", s)] for w in (16, 32, 64) for s in range(6)])
    vnonflat = sorted({int(i) for row in vann for i in np.where(row < FLAT)[0]})
    vnames = [str(i + 1) for i in vnonflat]
    put("vgannlayers", vnames[0] if len(vnames) == 1 else ", ".join(vnames[:-1]) + " and " + vnames[-1])
    vcols = [("cifar100", 64, "const", "gap"), ("cifar100", 64, "anneal", "gap"), ("cifar100", 64, "const", "frozen_random"),
             ("cifar100", 64, "const", "shuffled"), ("cifar100", 16, "const", "gap"), ("cifar100", 16, "anneal", "gap"),
             ("tinyimagenet", 64, "const", "gap")]
    vmat = {c: np.mean([1 - VL[(c[0], c[1], 4, c[2], c[3], s)] for s in range(6)], axis=0) for c in vcols}
    rows = [f"{i + 1} & " + " & ".join(f"{1e3 * vmat[c][i]:.2f}" for c in vcols) + " \\\\" for i in range(vnl)]
    rows.append("\\midrule mean & " + " & ".join(f"{1e3 * vmat[c].mean():.2f}" for c in vcols) + " \\\\")
    pathlib.Path("tab_VGLAYER.tex").write_text("\n".join(rows) + "\n")
    vgw64 = VBEN[("cifar100", 64, "const")]
    put("vgguC", sg(vgw64["mean"], 2)); put("vgguCp", pv(vgw64["p"])); put("vgguCv", verdict(vgw64))

    # ---------------- contradiction rows for the transfer table
    extra = [
        ("Entropy counterexample, CIFAR-100 w16",
         f"annealed {P_fl(HR)}$\\times$ flatter, larger benefit ({sg(paired(r18('cifar100',16,'gap',rec='anneal'),r18('cifar100',16,'uniform'))['mean'],2)} vs "
         f"{sg(paired(r18('cifar100',16,'gap'),r18('cifar100',16,'uniform'))['mean'],2)})",
         f"annealed {H('cifar100',16)/H('cifar100',16,'anneal'):.1f}$\\times$ flatter, smaller benefit "
         f"({sg(BEN[('cifar100',16,'anneal')]['mean'],2)} vs {sg(BEN[('cifar100',16,'const')]['mean'],2)})",
         f"annealed {VH('cifar100',16)/VH('cifar100',16,'anneal'):.1f}$\\times$ flatter, "
         f"{'smaller' if VBEN[('cifar100',16,'anneal')]['mean'] < VBEN[('cifar100',16,'const')]['mean'] else 'larger'} benefit "
         f"({sg(VBEN[('cifar100',16,'anneal')]['mean'],2)} vs {sg(VBEN[('cifar100',16,'const')]['mean'],2)})", "\\textbf{backbone-dependent}"),
        ("Constant-$\\tau$ $1-\\tilde H$ as width grows (w16, 32, 64)",
         ", ".join(f(HR(w, "const"), 4) for w in (16, 32, 64)) + trend(lambda ds, w: HR(w, "const")),
         ", ".join(f(H("cifar100", w), 4) for w in (16, 32, 64)) + trend(H),
         ", ".join(f(VH("cifar100", w), 4) for w in (16, 32, 64)) + trend(VH), "\\textbf{backbone-dependent}"),
        ("Same $1-\\tilde H$, CIFAR-100 w64, \\texttt{gap}$-$\\texttt{uniform}",
         f"{f(HR(64,'const'),4)}: {sg(r18w64['mean'],2)} ({verdict(r18w64)})",
         f"{f(H('cifar100',64),4)}: {sg(BEN[('cifar100',64,'const')]['mean'],2)} ({verdict(BEN[('cifar100',64,'const')])})",
         f"{f(VH('cifar100',64),4)}: {sg(vgw64['mean'],2)} ({verdict(vgw64)})", "across backbones"),
    ]
    with open("tab_CONTRA.tex", "a") as fh:
        fh.write("\n".join(f"{a} & {b} & {c} & {d} & {e} \\\\" for a, b, c, d, e in extra) + "\n")
    status = [l.rsplit("&", 1)[1] for l in pathlib.Path("tab_CONTRA.tex").read_text().splitlines() if l.strip()]
    put("ncontra", str(len(status)))
    put("ncontraBD", str(sum("backbone-dependent" in x for x in status)))
    put("ncontraHold", str(sum("holds on all three" in x for x in status)))
    put("ncontraMB", str(sum("MobileNetV2 only" in x for x in status)))
    put("ncontraX", str(sum("across backbones" in x for x in status)))

    # ---------------- figure G8
    fig, ax = plt.subplots(1, 3, figsize=(7.9, 3.3), gridspec_kw=dict(width_ratios=[1, 1.15, 1.2]))
    W = [16, 32, 64]
    for name, hf, col, mk in [("ResNet-18", lambda w, rec: HR(w, rec), C["r18"], "s"), ("MobileNetV2", lambda w, rec: H("cifar100", w, rec), C["mb"], "o"),
                              ("VGG-16", lambda w, rec: VH("cifar100", w, rec), C["vg"], "^")]:
        ax[0].plot(W, [hf(w, "const") for w in W], marker=mk, ms=4.5, lw=1.3, color=col, label=f"{name}, $\\tau=1$")
        ax[0].plot(W, [hf(w, "anneal") for w in W], marker=mk, ms=4.5, lw=1.3, ls="--", color=col, mfc="white",
                   label=f"{name}, $\\tau:30\\to1$")
    ax[0].set_yscale("log"); ax[0].set_xscale("log", base=2); ax[0].set_xticks(W); ax[0].set_xticklabels(W)
    ax[0].set_xlabel("base width $c$"); ax[0].set_ylabel("$1-\\tilde H$ (log)")
    ax[0].set_title("(a) router flatness, CIFAR-100")
    lab8 = []
    for name, pts_, col, mk in [("ResNet-18", sixr18, C["r18"], "s"), ("MobileNetV2", sixmb, C["mb"], "o"), ("VGG-16", sixvg, C["vg"], "^")]:
        for (x, y), (w, rec) in zip(pts_, [(w, r) for w in W for r in ("const", "anneal")]):
            ax[1].scatter(x, y, marker=mk, s=30, color=col if rec == "const" else "white", edgecolor=col, zorder=3,
                          label=name if (w, rec) == (16, "const") else None)
            lab8.append((x, y, f"w{w}{'' if rec == 'const' else 'a'}"))
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xscale("log"); ax[1].set_xlabel("$1-\\tilde H$ (log)")
    ax[1].set_ylabel("gap $-$ uniform (pp)")
    ax[1].set_title("(b) flatness vs benefit (a = annealed)")
    x = np.arange(1, nl + 1)
    for (c, lbl, col, ls) in [(("cifar100", 64, "const", "gap"), "gap, $\\tau=1$", C["gap"], "-"),
                              (("cifar100", 64, "anneal", "gap"), "gap, annealed", C["ann"], "--"),
                              (("cifar100", 64, "const", "frozen_random"), "frozen_random", C["frozen_random"], "-."),
                              (("cifar100", 64, "const", "shuffled"), "shuffled", C["shuffled"], ":")]:
        ax[2].plot(x, 1e3 * mat[c], marker="o", ms=2.5, lw=1.2, color=col, ls=ls, label=lbl)
    ax[2].set_xlabel("dynamic layer (input to output)"); ax[2].set_ylabel("$10^3\\,(1-\\tilde H)$")
    ax[2].set_xticks([1, 5, 9, 13, 17])
    ax[2].set_title("(c) MobileNetV2 by layer, C100 w64")
    plt.tight_layout(w_pad=1.0)
    for a_ in ax: PS.place_legend(a_)
    PS.place_labels(ax[1], lab8, fontsize=5.8)
    PS.save("fig/g8_entropy.pdf"); plt.close()


def P_fl(HR):
    return f"{HR(16, 'const') / HR(16, 'anneal'):.0f}"


if __name__ == "__main__":
    M = main(sys.argv[1], sys.argv[2])
    for k, v in M.items():
        print(f"{k:16} {v}")


# =========================================================================== VGG-16
def counted_layers(model):
    out, cur = {}, None
    for l in (ROOT / "PARAMS_EXPECTED.md").read_text().splitlines():
        if l.startswith("## "):
            cur = l[3:].strip()
        x = [c.strip() for c in l.split("|")[1:-1]]
        if cur == model and len(x) == 8 and x[0] in ("cifar100", "tinyimagenet"):
            out[(x[0], int(x[1]), int(x[2]), x[3])] = int(x[7])
    return out


def vgg_part(RV, vg, put):
    """Every VGG-16 number, table and appendix figure. Records come from the
    VGG-16 train.log (mbv2_log.parse with model='vgg16'); parameter counts
    from PARAMS_EXPECTED.md, checked against the counts the log prints."""
    PV = counted("vgg16")
    for k, r in RV.items():
        pc, rk = PV[(k[0], k[1], k[2], k[4])]
        if abs(r["params"] - round(pc, 2)) > 1e-9 or abs(r["router_k"] - round(rk, 1)) > 1e-9:
            raise SystemExit(f"VGG-16 parameter count in the log differs from PARAMS_EXPECTED.md for {k}")
    arms = ["static", "uniform", "gap", "shuffled", "frozen_random"]
    put("vgruns", str(len(RV)))
    put("vggpuh", f"{sum(r['minutes'] for r in RV.values()) / 60:.0f}")
    put("vggate", f(RV[("cifar100", 64, 4, "const", "static", 0)]["final"], 2))
    put("vgpstatic", f(PV[("cifar100", 64, 4, "static")][0], 2))
    put("vgrouterK", f(PV[("cifar100", 64, 4, "gap")][1], 0))
    put("vgrouterpct", f(100 * PV[("cifar100", 64, 4, "gap")][1] / 1e3 / PV[("cifar100", 64, 4, "static")][0], 1))
    put("vgbankadd", f(PV[("cifar100", 64, 4, "uniform")][0] - PV[("cifar100", 64, 4, "static")][0], 2))
    put("vgdynshare", f"{100 * (PV[('cifar100', 64, 4, 'uniform')][0] - PV[('cifar100', 64, 4, 'static')][0]) / (3 * PV[('cifar100', 64, 4, 'static')][0]):.0f}")
    put("vgnlayers", str(counted_layers("vgg16")[("cifar100", 64, 4, "gap")]))
    put("vgmicro", ", ".join(sorted({str(r["micro"]) for r in RV.values()})))

    # decomposition
    dec, rows = {}, []
    tag = {("cifar100", 16): "CA", ("cifar100", 32): "CB", ("cifar100", 64): "CC",
           ("tinyimagenet", 32): "TA", ("tinyimagenet", 64): "TB"}
    for ds, w in COND:
        b = paired(vg(ds, w, "uniform"), vg(ds, w, "static"))
        r = paired(vg(ds, w, "gap"), vg(ds, w, "uniform"))
        t = paired(vg(ds, w, "gap"), vg(ds, w, "static"))
        dec[(ds, w)] = (b, r, t)
        rows.append(f"{DSN[ds]} & {w} & {sg(b['mean'])} & {pv(b['tost'])} & {verdict(b)} & {sg(r['mean'])} & "
                    f"{pv(r['p'])} & {sg(t['mean'])} & {pv(t['p'])} \\\\")
        k = tag[(ds, w)]
        put(f"vgbank{k}", sg(b["mean"])); put(f"vgbank{k}T", f(b["tost"], 4)); put(f"vgbank{k}v", verdict(b))
        put(f"vgrout{k}", sg(r["mean"])); put(f"vgrout{k}p", pv(r["p"])); put(f"vgrout{k}v", verdict(r))
        put(f"vgeff{k}", sg(t["mean"])); put(f"vgeff{k}lo", sg(t["lo"])); put(f"vgeff{k}hi", sg(t["hi"]))
        put(f"vgeff{k}p", pv(t["p"])); put(f"vgeff{k}T", f(t["tost"], 4)); put(f"vgeff{k}v", verdict(t))
    pathlib.Path("tab_VGDECOMP.tex").write_text("\n".join(rows) + "\n")
    put("vgnbankeq", str(sum(verdict(b) == "equivalent" for b, _, _ in dec.values())))
    put("vgnbanksig", str(sum(b["p"] < 0.05 for b, _, _ in dec.values())))
    put("vgnrout", str(sum(r["p"] < 0.05 and r["mean"] > 0 for _, r, _ in dec.values())))
    put("vgneff", str(sum(t["p"] < 0.05 and t["mean"] > 0 for _, _, t in dec.values())))
    put("vgbankmax", sg(max((b["mean"] for b, _, _ in dec.values()), key=abs), 2))

    # resumed run
    res = [k for k, v in RV.items() if v["resumed"]]
    put("vgnresumed", str(len(res)))
    assert len(res) == 1 and res[0][:5] == ("cifar100", 64, 4, "const", "shuffled"), res
    k = res[0]
    keep = [s for s in range(6) if s != k[5]]
    put("vgresumedseed", str(k[5])); put("vgresumedacc", f(RV[k]["final"], 2))
    for other, nm in [("static", "S"), ("gap", "G")]:
        q = paired(vg("cifar100", 64, "shuffled")[keep], vg("cifar100", 64, other)[keep])
        put(f"vgsens{nm}", sg(q["mean"])); put(f"vgsens{nm}p", pv(q["p"])); put(f"vgsens{nm}v", verdict(q).split(" (")[0])

    # width-64 controls
    comps = [("uniform", "static"), ("gap", "static"), ("shuffled", "static"), ("frozen_random", "static"),
             ("frozen_random", "gap"), ("shuffled", "gap"), ("frozen_random", "uniform"), ("gap", "uniform")]
    for a, b_ in comps:
        q = paired(vg("cifar100", 64, a), vg("cifar100", 64, b_))
        key = (a + b_).replace("_", "")
        put(f"vgw{key}", sg(q["mean"])); put(f"vgw{key}lo", sg(q["lo"])); put(f"vgw{key}hi", sg(q["hi"]))
        put(f"vgw{key}p", pv(q["p"])); put(f"vgw{key}T", f(q["tost"], 4)); put(f"vgw{key}v", verdict(q))
    F_, p_ = stats.f_oneway(*[vg("cifar100", 64, a) for a in arms])
    put("vganovaF", f(F_, 2)); put("vganovaP", f(p_, 3))
    means = [vg("cifar100", 64, a).mean() for a in arms]
    put("vgspread64", f(max(means) - min(means), 3))

    # twice the width
    rows, inv, pairs = [], [], []
    for ds, wg, ws, kk in [("cifar100", 16, 32, "MA"), ("cifar100", 32, 64, "MB"), ("tinyimagenet", 32, 64, "MC")]:
        q = paired(vg(ds, ws, "static"), vg(ds, wg, "gap")); pairs.append(q)
        pg, ps = PV[(ds, wg, 4, "gap")][0], PV[(ds, ws, 4, "static")][0]
        inv.append(ps / pg)
        rows.append(f"{DSN[ds]} & {tt('gap')} w{wg} ({f(pg, 2)}M) & {tt('static')} w{ws} ({f(ps, 2)}M) & "
                    f"{f(vg(ds, wg, 'gap').mean())} & {f(vg(ds, ws, 'static').mean())} & {sg(q['mean'])} & "
                    f"[{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & {verdict(q)} & {pg / ps:.2f} \\\\")
        put(f"vgm{kk}", sg(q["mean"])); put(f"vgm{kk}p", pv(q["p"])); put(f"vgm{kk}r", f"{pg / ps:.2f}")
    pathlib.Path("tab_VGMATCHED.tex").write_text("\n".join(rows) + "\n")
    put("vgmmin", f(min(q["mean"] for q in pairs), 1)); put("vgmmax", f(max(q["mean"] for q in pairs), 1))
    put("vgrlo", f(min(1 / x for x in inv), 2)); put("vgrhi", f(max(1 / x for x in inv), 2))

    # bank size
    rows = []
    nk = 0
    for kk in (2, 4, 8):
        v = vg("cifar100", 32, "gap", k=kk); q = paired(v, vg("cifar100", 32, "static"))
        nk += q["p"] < 0.05 and q["mean"] > 0
        rows.append(f"{kk} & {f(v.mean())} & {f(v.std(ddof=1))} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & "
                    f"{pv(q['p'])} & {f(PV[('cifar100', 32, kk, 'gap')][0], 2)} & {f(vg('cifar100', 32, 'gap', k=kk, field='minutes').mean(), 1)} \\\\")
        put(f"vgk{kk}s", sg(q["mean"])); put(f"vgk{kk}sp", pv(q["p"]))
    v = vg("cifar100", 32, "static")
    rows.append(f"\\midrule {tt('static')} & {f(v.mean())} & {f(v.std(ddof=1))} & -- & -- & -- & "
                f"{f(PV[('cifar100', 32, 4, 'static')][0], 2)} & {f(vg('cifar100', 32, 'static', field='minutes').mean(), 1)} \\\\")
    pathlib.Path("tab_VGK.tex").write_text("\n".join(rows) + "\n")
    put("vgnkbeat", str(nk))
    q = paired(vg("cifar100", 32, "gap", k=8), vg("cifar100", 32, "gap", k=2))
    put("vgk82", sg(q["mean"])); put("vgk82p", pv(q["p"])); put("vgk82v", verdict(q)); put("vgk82T", f(q["tost"], 4))

    # temperature
    rows = []
    for w, kk in [(16, "A"), (32, "B"), (64, "C")]:
        c = vg("cifar100", w, "gap"); an = vg("cifar100", w, "gap", rec="anneal")
        a1 = paired(an, c); s1 = paired(c, vg("cifar100", w, "static")); s2 = paired(an, vg("cifar100", w, "static"))
        u2 = paired(an, vg("cifar100", w, "uniform"))
        rows.append(f"{w} & {f(c.mean())} & {f(an.mean())} & {sg(a1['mean'])} & {pv(a1['p'])} & {sg(s1['mean'])} & "
                    f"{sg(s2['mean'])} & {pv(s2['p'])} & {sg(u2['mean'])} & {verdict(u2)} \\\\")
        put(f"vgan{kk}", sg(a1["mean"])); put(f"vgan{kk}p", pv(a1["p"]))
        put(f"vganu{kk}", sg(u2["mean"])); put(f"vganu{kk}v", verdict(u2)); put(f"vganu{kk}p", pv(u2["p"]))
        put(f"vgans{kk}", sg(s2["mean"])); put(f"vgans{kk}v", verdict(s2)); put(f"vgans{kk}T", f(s2["tost"], 4))
        put(f"vgla{kk}", f(vg("cifar100", w, "gap", rec="anneal", field="loss").mean(), 4))
        put(f"vglc{kk}", f(vg("cifar100", w, "gap", field="loss").mean(), 4))
    pathlib.Path("tab_VGTEMP.tex").write_text("\n".join(rows) + "\n")

    # predictions
    nb = sum(b["tost"] < 0.05 for b, _, _ in dec.values())
    np2 = sum(q["p"] < 0.05 and q["mean"] > 0 for q in pairs)
    nr = sum(r["p"] < 0.05 and r["mean"] > 0 for _, r, _ in dec.values())
    put("vgP1", str(nb)); put("vgP2", str(np2)); put("vgP3", str(nr))
    put("vgP1o", "held" if nb == 5 else "failed"); put("vgP2o", "held" if np2 == 3 else "failed"); put("vgP3o", "held" if nr >= 3 else "failed")
    pathlib.Path("tab_VGPREREG.tex").write_text(
        f"P1 & The bank term (\\texttt{{uniform}}$-$\\texttt{{static}}) is equivalent to zero (TOST, $\\pm$\\SI{{0.5}}{{pp}}) in each of the five conditions & {nb} of 5 & {'held' if nb == 5 else 'failed'} \\\\\n"
        f"P2 & \\texttt{{static}} at width $2c$ is more accurate than \\texttt{{gap}} at width $c$ in each of the three pairs & {np2} of 3 & {'held' if np2 == 3 else 'failed'} \\\\\n"
        f"P3 & The routing term is positive with $p<0.05$ in at least three of the five conditions & {nr} of 5 & {'held' if nr >= 3 else 'failed'} \\\\\n")

    # losses (saturation)
    for ds, w in COND:
        put(f"vgloss{tag[(ds, w)]}", f(vg(ds, w, "static", field="loss").mean(), 4))

    # appendix tables
    rows = []
    for ds, ws in [("cifar100", [16, 32, 64]), ("tinyimagenet", [32, 64])]:
        for w in ws:
            for a in (arms if (ds, w) == ("cifar100", 64) else ["static", "uniform", "gap"]):
                v = vg(ds, w, a)
                mark = ["$^\\dagger$" if RV[(ds, w, 4, "const", a, s)]["resumed"] else "" for s in range(6)]
                rows.append(f"{DSN[ds]} & {w} & {tt(a)} & " + " & ".join(f"{x:.2f}{m}" for x, m in zip(v, mark)) +
                            f" & {f(v.mean())} & {f(v.std(ddof=1))} \\\\")
    pathlib.Path("tab_VGALLRUNS.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for ds, w in COND:
        for a in (["uniform", "gap", "shuffled", "frozen_random"] if (ds, w) == ("cifar100", 64) else ["uniform", "gap"]):
            q = paired(vg(ds, w, a), vg(ds, w, "static"))
            rows.append(f"{DSN[ds]} & {w} & {tt(a)} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & "
                        f"{pv(q['wil'])} & {sg(q['dz'], 2)} & {pv(q['tost'])} & {verdict(q)} \\\\")
    pathlib.Path("tab_VGVSSTATIC.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for ds, ws in [("cifar100", [16, 32, 64]), ("tinyimagenet", [32, 64])]:
        for w in ws:
            for a in (arms if (ds, w) == ("cifar100", 64) else ["static", "uniform", "gap"]):
                p_, rk = PV[(ds, w, 4, a)]
                rows.append(f"{DSN[ds]} & {w} & {tt(a)} & {f(p_, 2)} & {f(rk, 1)} & {f(vg(ds, w, a, field='loss').mean(), 4)} & "
                            f"{f(vg(ds, w, a, field='minutes').mean(), 1)} \\\\")
    pathlib.Path("tab_VGCONFIG.tex").write_text("\n".join(rows) + "\n")
    rows, tot, hrs = [], 0, 0.0
    groups = sorted({k[:4] for k in RV}, key=lambda g: S.DISPLAY_ORDER.index(g))
    for g in groups:
        ks = [k for k in RV if k[:4] == g]
        armset = [a for a in arms if any(k[4] == a for k in ks)]
        h = sum(RV[k]["minutes"] for k in ks) / 60
        tot += len(ks); hrs += h
        rows.append(f"{DSN[g[0]]} & {g[1]} & {g[2]} & {'constant' if g[3] == 'const' else 'annealed'} & "
                    f"{', '.join(tt(a) for a in armset)} & {len(ks)} & {f(h, 1)} \\\\")
    rows.append(f"\\midrule \\multicolumn{{5}}{{l}}{{total}} & {tot} & {f(hrs, 1)} \\\\")
    pathlib.Path("tab_VGINVENT.tex").write_text("\n".join(rows) + "\n")

    # appendix figure: every VGG-16 run at K=4, tau=1
    fig, axs = plt.subplots(1, 5, figsize=(7.8, 2.5), gridspec_kw=dict(width_ratios=[3, 3, 5, 3, 3]))
    for k_, (ds, w) in enumerate(COND):
        aa = arms if (ds, w) == ("cifar100", 64) else ["static", "uniform", "gap"]
        for i, a in enumerate(aa):
            v = vg(ds, w, a); axs[k_].scatter([i] * 6, v, s=14, color=C[a], alpha=.85, zorder=3)
            axs[k_].hlines(v.mean(), i - .3, i + .3, color="k", lw=1.3, zorder=4)
        axs[k_].set_xticks(range(len(aa))); axs[k_].set_xticklabels(aa, rotation=45, ha="right", fontsize=6)
        axs[k_].set_title(f"{'C100' if ds == 'cifar100' else 'TIN'} w{w}", fontsize=8); axs[k_].tick_params(labelsize=6)
    axs[0].set_ylabel("top-1 accuracy (%)")
    plt.tight_layout(); PS.save("fig/g9_vg_allruns.pdf"); plt.close()
    return dec, pairs


# =========================================================================== pooled estimates and multiplicity
def _holm(ps, alpha=0.05):
    o = np.argsort(ps); r = np.zeros(len(ps), bool)
    for i, j in enumerate(o):
        if ps[j] < alpha / (len(ps) - i):
            r[j] = True
        else:
            break
    return r


def _pool(diffs):
    """DerSimonian-Laird random-effects pool of per-pair mean differences."""
    m = np.array([d.mean() for d in diffs]); v = np.array([d.var(ddof=1) / len(d) for d in diffs]); w = 1 / v
    fe = (w * m).sum() / w.sum(); Q = (w * (m - fe) ** 2).sum(); k = len(m)
    t2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); w2 = 1 / (v + t2)
    re = (w2 * m).sum() / w2.sum(); se = math.sqrt(1 / w2.sum())
    return dict(est=re, lo=re - 1.96 * se, hi=re + 1.96 * se, lo90=re - 1.645 * se, hi90=re + 1.645 * se,
                Qp=float(stats.chi2.sf(Q, k - 1)), tau=math.sqrt(t2))


def _fixed(diffs):
    m = np.array([d.mean() for d in diffs]); v = np.array([d.var(ddof=1) / len(d) for d in diffs]); w = 1 / v
    est = (w * m).sum() / w.sum(); se = math.sqrt(1 / w.sum())
    return dict(est=est, lo90=est - 1.645 * se, hi90=est + 1.645 * se, lo=est - 1.96 * se, hi=est + 1.96 * se)


def pooled_part(r18, mb, vg, put):
    BB = [("ResNet-18", r18, "rb"), ("MobileNetV2", mb, "mb"), ("VGG-16", vg, "vg")]
    names = {("cifar100", 16): "CIFAR-100 at width 16", ("cifar100", 32): "CIFAR-100 at width 32",
             ("cifar100", 64): "CIFAR-100 at width 64", ("tinyimagenet", 32): "Tiny-ImageNet at width 32",
             ("tinyimagenet", 64): "Tiny-ImageNet at width 64"}
    labels, bank, rout, tot = [], [], [], []
    for name, g, _ in BB:
        for ds, w in COND:
            labels.append(f"{name} on {names[(ds, w)]}")
            bank.append(np.asarray(g(ds, w, "uniform"), float) - np.asarray(g(ds, w, "static"), float))
            rout.append(np.asarray(g(ds, w, "gap"), float) - np.asarray(g(ds, w, "uniform"), float))
            tot.append(np.asarray(g(ds, w, "gap"), float) - np.asarray(g(ds, w, "static"), float))
    # multiplicity: Holm over the 15 pairs of each family
    rows = []
    for key, txt, diffs in [("rout", "routing term positive ($\\texttt{gap}-\\texttt{uniform}$)", rout),
                            ("tot", "dynamic beats plain ($\\texttt{gap}-\\texttt{static}$)", tot)]:
        q = [paired(d + 0, np.zeros_like(d)) for d in diffs]
        ps = np.array([x["p"] for x in q]); pos = np.array([x["mean"] > 0 for x in q])
        raw = (ps < 0.05) & pos; h = _holm(ps) & pos
        drop = [labels[i] for i in range(len(ps)) if raw[i] and not h[i]]
        put(f"mult{key}raw", str(raw.sum())); put(f"mult{key}holm", str(h.sum()))
        put(f"mult{key}drop", " and ".join(drop) if drop else "none")
        rows.append(f"{txt} & {raw.sum()} & {h.sum()} \\\\")
    for m_, tag in [(0.25, "A"), (0.5, "B"), (0.75, "C"), (1.0, "D")]:
        ps = np.array([paired(d + 0, np.zeros_like(d), margin=m_)["tost"] for d in bank])
        raw = ps < 0.05; h = _holm(ps)
        put(f"multbank{tag}raw", str(raw.sum())); put(f"multbank{tag}holm", str(h.sum()))
        rows.append(f"bank term equivalent at $\\pm${m_:g}\\,pp (TOST) & {raw.sum()} & {h.sum()} \\\\")
    pathlib.Path("tab_MULT.tex").write_text("\n".join(rows) + "\n")
    # pooled estimates
    pb, pr = _pool(bank), _pool(rout)
    for k, p_ in [("poolbank", pb), ("poolrout", pr)]:
        put(k, sg(p_["est"])); put(k + "lo", sg(p_["lo"])); put(k + "hi", sg(p_["hi"]))
        put(k + "lonine", sg(p_["lo90"])); put(k + "hinine", sg(p_["hi90"]))
        put(k + "Qp", "$<10^{-4}$" if p_["Qp"] < 1e-4 else f(p_["Qp"], 3)); put(k + "tau", f(p_["tau"], 2))
    marg = next(m_ for m_ in np.arange(0.05, 1.01, 0.05) if pb["lo90"] > -m_ and pb["hi90"] < m_)
    put("poolbankmargin", f"{marg:.2f}".rstrip("0").rstrip("."))
    rows = [f"all three, 15 pairs (random effects) & {sg(pb['est'])} & [{sg(pb['lo'])}, {sg(pb['hi'])}] & "
            f"[{sg(pb['lo90'])}, {sg(pb['hi90'])}] & {sg(pr['est'])} & [{sg(pr['lo'])}, {sg(pr['hi'])}] \\\\"]
    worst = 0.0
    for i, (name, g, pre) in enumerate(BB):
        fb = _fixed(bank[5 * i:5 * i + 5]); fr = _fixed(rout[5 * i:5 * i + 5])
        worst = max(worst, abs(fb["lo90"]), abs(fb["hi90"]))
        put(f"{pre}poolbank", sg(fb["est"])); put(f"{pre}poolbanklonine", sg(fb["lo90"])); put(f"{pre}poolbankhinine", sg(fb["hi90"]))
        rows.append(f"{name}, 5 conditions (fixed effect) & {sg(fb['est'])} & [{sg(fb['lo'])}, {sg(fb['hi'])}] & "
                    f"[{sg(fb['lo90'])}, {sg(fb['hi90'])}] & {sg(fr['est'])} & [{sg(fr['lo'])}, {sg(fr['hi'])}] \\\\")
    # smallest margin on a 0.05 pp grid that contains every backbone's 90 % interval
    wm = next(m_ for m_ in np.arange(0.05, 1.01, 0.05) if worst < m_)
    put("poolbankworst", f"{wm:.2f}".rstrip("0").rstrip("."))
    put("rbstaticCC", f(np.mean(r18("cifar100", 64, "static")), 2)); put("rbgapCC", f(np.mean(r18("cifar100", 64, "gap")), 2))
    put("vgstaticCC", f(np.mean(vg("cifar100", 64, "static")), 2)); put("mbstaticCC", f(np.mean(mb("cifar100", 64, "static")), 2))
    put("vggapCC", f(np.mean(vg("cifar100", 64, "gap")), 2)); put("mbgapCC", f(np.mean(mb("cifar100", 64, "gap")), 2))
    put("rbstaticTB", f(np.mean(r18("tinyimagenet", 64, "static")), 2)); put("rbgapTB", f(np.mean(r18("tinyimagenet", 64, "gap")), 2))
    pathlib.Path("tab_POOL.tex").write_text("\n".join(rows) + "\n")

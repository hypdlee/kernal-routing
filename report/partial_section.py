"""
Numbers and tables for control runs that were still incomplete when this draft
was built: the ResNet-18 controls (PREREGISTRATION_CONTROLS_R18.md, committed
27 October 2026) and, when its log is present, the MobileNetV2 Tiny-ImageNet
controls (part D, which has no recorded prediction).

Every final accuracy is checked against its train.log line. A comparison uses
the seeds both arms have finished; the seed count of every cell is written
next to it. Called from mbv2_section.main with its `put`; run with the output
directory as cwd.
"""
import importlib.util, json, math, pathlib, re, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
_s = importlib.util.spec_from_file_location("kr_stats", HERE / "data.py")
S = importlib.util.module_from_spec(_s); _s.loader.exec_module(S)
_r = importlib.util.spec_from_file_location("r18", ROOT / "tools" / "controls_r18_analysis.py")
R18 = importlib.util.module_from_spec(_r); _r.loader.exec_module(R18)
import controls_analysis as CA

f = lambda x, n=2: "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
sg = lambda x, n=2: f"{x:+.{n}f}"
pv = lambda p: "$<10^{-4}$" if p < 1e-4 else f"{p:.4f}"
tt = lambda a: "\\texttt{" + a.replace("_", "\\_") + "}"
CELLS = {"W16": ("cifar100", 16), "W32": ("cifar100", 32), "TIN": ("tinyimagenet", 32)}
CELLNAME = {"W16": "CIFAR-100, width 16", "W32": "CIFAR-100, width 32", "TIN": "Tiny-ImageNet, width 32"}
ARMS = {"shuf": "shuffled", "frz": "frozen_random"}
REFS = {"sta": "static", "uni": "uniform", "gap": "gap"}
MARGIN = 0.5


def nseeds(v):
    return int((~np.isnan(np.asarray(v, float))).sum())


def load_mb_partD(root):
    """MobileNetV2 Tiny-ImageNet width 32 controls from results_controls/mobilenetv2, checked against its log."""
    d = root / "results_controls" / "mobilenetv2"
    log = d / "train.log"
    if not log.exists():
        return None, []
    lv = CA.log_values(log)
    acc, problems, runs = {}, [], []
    for fp in sorted(d.glob("mobilenetv2__tinyimagenet__w32__K4__const_*_seed*.json")):
        r = json.loads(fp.read_text())
        if fp.name not in lv:
            problems.append(f"{fp.name}: no log line")
        elif abs(lv[fp.name] - r["final_acc"]) > 0.005:
            problems.append(f"{fp.name}: json {r['final_acc']} != log {lv[fp.name]}")
        acc.setdefault(r["arm"], {})[r["seed"]] = r["final_acc"]
        runs.append(r)
    return {a: np.array([v.get(s, np.nan) for s in range(6)], float) for a, v in acc.items()}, (problems, runs)


def main(put, root=ROOT):
    # ---------------- ResNet-18 ----------------
    A, problems = R18.load(root)
    if problems:
        raise SystemExit("ResNet-18 control files disagree with their logs: " + "; ".join(problems[:5]))
    files = []
    for d in ("results_controls_a", "results_controls_b", "results_controls"):
        files += sorted((root / d / "resnet18").glob("*_seed*.json"))
    runs = [json.loads(p.read_text()) for p in files]
    put("pdr18runs", str(len(runs)))
    put("pdr18hours", f"{sum(r['minutes'] for r in runs) / 60:.0f}")
    put("pdr18gpu", ", ".join(R18.load.gpus))

    def vals(cell, arm):
        ds, w = CELLS[cell]
        return np.asarray(A[(ds, w, arm, "d1")], float)

    rows, allrows = [], []
    for cell in ("W16", "W32", "TIN"):
        ns = {x: nseeds(vals(cell, a)) for x, a in ARMS.items()}
        for x, n in ns.items():
            put(f"pdr18{cell}{x}n", str(n))
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{ResNet-18, {CELLNAME[cell]}: {tt('shuffled')} {ns['shuf']} of 6 seeds, "
                    f"{tt('frozen_random')} {ns['frz']} of 6 seeds}}}} \\\\")
        for x, a in ARMS.items():
            for y, b in REFS.items():
                if ns[x] < 2:
                    continue
                q = S.paired(vals(cell, a), vals(cell, b), margin=MARGIN)
                key = f"pdr18{cell}{x}{y}"
                put(key, sg(q["mean"])); put(key + "lo", sg(q["lo"])); put(key + "hi", sg(q["hi"]))
                put(key + "p", pv(q["p"])); put(key + "T", f(q["tost"], 4)); put(key + "v", S.verdict(q))
                put(key + "peq", "p<10^{-4}" if q["p"] < 1e-4 else f"p={q['p']:.4f}")
                put(key + "a", f(abs(q["mean"])))
                rows.append(f"{tt(a)} $-$ {tt(b)} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & "
                            f"{pv(q['p'])} & {S.verdict(q)} \\\\")
        for y, b_ in (("uni", "uniform"), ("sta", "static")):
            q = S.paired(vals(cell, "gap"), vals(cell, b_), margin=MARGIN); key = f"pdr18{cell}gap{y}"
            put(key, sg(q["mean"])); put(key + "p", pv(q["p"])); put(key + "v", S.verdict(q))
        for a in (list(ARMS.values()) + ["static", "uniform", "gap"]):
            v = vals(cell, a)
            allrows.append(f"ResNet-18 & {CELLNAME[cell].split(',')[0]} & {CELLS[cell][1]} & {tt(a)} & " +
                           " & ".join("--" if np.isnan(z) else f"{z:.2f}" for z in v) +
                           f" & {f(np.nanmean(v))} & {nseeds(v)} \\\\")

    # ---- the three recorded predictions (PREREGISTRATION_CONTROLS_R18.md), scored only on complete cells
    complete = all(nseeds(vals(c, a)) == 6 for c in CELLS for a in ARMS.values())
    put("pdr18complete", "yes" if complete else "no")
    if complete:
        Q = lambda c, a, b: S.paired(vals(c, a), vals(c, b), margin=MARGIN)
        lower = lambda z: z["p"] < 0.05 and z["mean"] < 0
        r1q = [Q("TIN", "shuffled", "gap"), Q("TIN", "frozen_random", "gap")]
        r2q = [Q("W16", "shuffled", "gap"), Q("W32", "shuffled", "gap")]
        r3q = [Q("TIN", "frozen_random", "uniform")]
        res = {"R1": all(lower(z) for z in r1q), "R2": all(lower(z) for z in r2q),
               "R3": r3q[0]["p"] < 0.05 and r3q[0]["mean"] > 0}
        for k, v in res.items():
            put(f"pdr18{k}o", "held" if v else "failed")
        put("pdr18nfail", str(sum(not v for v in res.values())))
        for k, qs in (("R1", r1q), ("R2", r2q), ("R3", r3q)):
            if res[k] and not all(z["p"] * len(qs) < 0.05 for z in qs):
                raise SystemExit(f"{k} held but a comparison fails Bonferroni over its own prediction; fix the text")
        put("pdr18TINfrzshare", f(100 * Q("TIN", "frozen_random", "uniform")["mean"] / Q("TIN", "gap", "uniform")["mean"], 0))
        put("pdr18W32frzshare", f(100 * Q("W32", "frozen_random", "uniform")["mean"] / Q("W32", "gap", "uniform")["mean"], 0))
        put("pdr18W16frzshare", f(100 * Q("W16", "frozen_random", "uniform")["mean"] / Q("W16", "gap", "uniform")["mean"], 0))

    # verdict column
    out = []
    for r_ in rows:
        out.append(r_)
    pathlib.Path("tab_PDR18.tex").write_text("\n".join(out) + "\n")

    # ---------------- MobileNetV2 part D ----------------
    got, extra = load_mb_partD(root)
    if got is not None:
        mp, mruns = extra
        if mp:
            raise SystemExit("MobileNetV2 part D files disagree with their log: " + "; ".join(mp[:5]))
        Amb, _ = CA.load("mobilenetv2", root)
        base = {a: np.asarray(Amb[("tinyimagenet", 32, a, "d1")], float) for a in ("static", "uniform", "gap")}
        put("pdmbruns", str(len(mruns)))
        put("pdmbhours", f"{sum(r['minutes'] for r in mruns) / 60:.0f}")
        ns = {x: nseeds(got.get(a, np.full(6, np.nan))) for x, a in ARMS.items()}
        for x, n in ns.items():
            put(f"pdmb{x}n", str(n))
        mrows = [f"\\multicolumn{{5}}{{l}}{{\\emph{{MobileNetV2, Tiny-ImageNet, width 32: {tt('shuffled')} {ns['shuf']} of 6 seeds, "
                 f"{tt('frozen_random')} {ns['frz']} of 6 seeds}}}} \\\\"]
        for x, a in ARMS.items():
            for y, b in REFS.items():
                if ns[x] < 2:
                    continue
                q = S.paired(got[a], base[b], margin=MARGIN)
                key = f"pdmb{x}{y}"
                put(key, sg(q["mean"])); put(key + "lo", sg(q["lo"])); put(key + "hi", sg(q["hi"]))
                put(key + "p", pv(q["p"])); put(key + "T", f(q["tost"], 4)); put(key + "v", S.verdict(q))
                put(key + "a", f(abs(q["mean"])))
                mrows.append(f"{tt(a)} $-$ {tt(b)} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & {S.verdict(q)} \\\\")
        pathlib.Path("tab_PDMB.tex").write_text("\n".join(mrows) + "\n")
        for a in list(ARMS.values()):
            v = got[a]
            allrows.append(f"MobileNetV2 & Tiny-ImageNet & 32 & {tt(a)} & " +
                           " & ".join("--" if np.isnan(z) else f"{z:.2f}" for z in v) + f" & {f(np.nanmean(v))} & {nseeds(v)} \\\\")
        for a in ("static", "uniform", "gap"):
            v = base[a]
            allrows.append(f"MobileNetV2 & Tiny-ImageNet & 32 & {tt(a)} & " + " & ".join(f"{z:.2f}" for z in v) +
                           f" & {f(np.nanmean(v))} & {nseeds(v)} \\\\")
    else:
        put("pdmbruns", "0")
    pathlib.Path("tab_PDALLRUNS.tex").write_text("\n".join(allrows) + "\n")
    return got is not None

"""
Pre-registered analysis of the routing controls (PREREGISTRATION_CONTROLS.md,
dated 2026-10-22).

    python tools/controls_analysis.py [--out docs/CONTROLS_RESULTS.md]

Reads artifacts/controls/<model>/ (new runs) and artifacts/<model>/results/
(main program), checks every final accuracy against its train.log line, pairs
by seed, and evaluates predictions C1 to C4 exactly as written there:
paired t-test, n = 6, two-sided, alpha 0.05; equivalence by TOST at 0.5 pp.
Nothing is typed; every number is computed from the result files.
"""
import argparse, json, math, pathlib, re, sys
import importlib.util
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kr_stats", ROOT / "report" / "data.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
MARGIN = 0.5
SEEDS = range(6)
LINE = re.compile(r"final=([\d.]+) best=([\d.]+) -> (?:.*[\\/])?([^\\/\s]+\.json)")


def log_values(path):
    out = {}
    for l in pathlib.Path(path).read_text(errors="replace").splitlines():
        m = LINE.search(l)
        if m:
            out[m.group(3)] = float(m.group(1))
    return out


def load(model):
    """{(dataset, width, arm, draw): [acc by seed]} with every value checked against its log."""
    acc, problems = {}, []
    main = ROOT / "artifacts" / model
    ctl = ROOT / "artifacts" / "controls" / model
    for d, logp, is_ctl in ((main / "results", main / "train.log", False), (ctl, ctl / "train.log", True)):
        lv = log_values(logp)
        for f in sorted(d.glob("*_seed*.json")):
            r = json.loads(f.read_text())
            tag = r["tag"]
            m = re.match(rf"{model}__(\w+?)__w(\d+)__K(\d+)__(\w+?)(?:__(d\d))?$", tag)
            if not m:
                continue
            ds, w, K, rec, draw = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5) or "d1"
            if K != 4 or rec != "const":
                continue
            if f.name not in lv:
                problems.append(f"{model}: {f.name} has no log line")
            elif abs(lv[f.name] - r["final_acc"]) > 0.005:
                problems.append(f"{model}: {f.name} json {r['final_acc']} != log {lv[f.name]}")
            acc.setdefault((ds, w, r["arm"], draw), {})[r["seed"]] = r["final_acc"]
    out = {}
    for k, v in acc.items():
        out[k] = [v.get(s, float("nan")) for s in SEEDS]
    return out, problems


def cmp(A, a, b):
    return S.paired(A[a], A[b], margin=MARGIN)


def row(name, q):
    v = S.verdict(q)
    return (f"| {name} | {q['mean']:+.2f} | [{q['lo']:+.2f}, {q['hi']:+.2f}] | {q['p']:.4f} | {q['tost']:.4f} | {v} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "CONTROLS_RESULTS.md"))
    a = ap.parse_args()
    L, T, V = [], [], {}
    A, problems = {}, []
    for m in ("mobilenetv2", "vgg16"):
        A[m], p = load(m); problems += p
    for m in A:
        n = sum(1 for k in A[m] if k[2] in ("shuffled", "frozen_random"))
        cnt = sum(int(np.sum(~np.isnan(v))) for k, v in A[m].items())
        L.append(f"{m}: {cnt} runs loaded ({n} routing-control cells)")
    L.append("log check: " + ("all values match their train.log line" if not problems else f"{len(problems)} PROBLEMS"))
    for p in problems:
        L.append("  " + p)

    def g(m, ds, w, arm, draw="d1"):
        return A[m][(ds, w, arm, draw)]

    def grp(m, ds, w, extra=()):
        d = {x: g(m, ds, w, x) for x in ("static", "uniform", "gap")}
        for arm, draw, key in extra:
            d[key] = g(m, ds, w, arm, draw)
        return d

    T.append("| comparison | mean diff (pp) | 95% CI | p | TOST p (0.5 pp) | verdict |\n|---|---|---|---|---|---|")
    pred = {}

    def block(title, rows):
        T.append(f"| **{title}** | | | | | |")
        out = []
        for name, q in rows:
            T.append(row(name, q)); out.append(q)
        return out

    # C1 / C2: part A
    for m, key in (("mobilenetv2", "C1"), ("vgg16", "C2")):
        G = grp(m, "cifar100", 64, [("frozen_random", "d1", "frozen_d1"), ("frozen_random", "d2", "frozen_d2"),
                                    ("frozen_random", "d3", "frozen_d3"), ("shuffled", "d1", "shuffled")])
        qs = block(f"{m}, CIFAR-100 w64 (part A)",
                   [(f"frozen d2 - uniform", cmp(G, "frozen_d2", "uniform")), (f"frozen d3 - uniform", cmp(G, "frozen_d3", "uniform")),
                    (f"frozen d1 - uniform (main program)", cmp(G, "frozen_d1", "uniform")),
                    (f"frozen d2 - gap", cmp(G, "frozen_d2", "gap")), (f"frozen d3 - gap", cmp(G, "frozen_d3", "gap")),
                    (f"frozen d2 - static", cmp(G, "frozen_d2", "static")), (f"frozen d3 - static", cmp(G, "frozen_d3", "static"))])
        hi = [q["p"] < 0.05 and q["mean"] > 0 for q in qs[:2]]
        pred[key] = all(hi) if key == "C1" else not any(hi)
        V[key + "_detail"] = hi

    # C3: MobileNetV2 w16
    m = "mobilenetv2"
    G = grp(m, "cifar100", 16, [("frozen_random", "d1", "frozen"), ("shuffled", "d1", "shuffled")])
    q1, q2, q3 = block("mobilenetv2, CIFAR-100 w16 (part B)",
                       [("frozen - static", cmp(G, "frozen", "static")), ("frozen - gap", cmp(G, "frozen", "gap")),
                        ("shuffled - gap", cmp(G, "shuffled", "gap"))])
    block("mobilenetv2, CIFAR-100 w16, context", [("frozen - uniform", cmp(G, "frozen", "uniform")),
                                                   ("shuffled - static", cmp(G, "shuffled", "static")),
                                                   ("gap - static", cmp(G, "gap", "static")),
                                                   ("gap - uniform", cmp(G, "gap", "uniform"))])
    c3 = [q1["p"] < 0.05 and q1["mean"] > 0, q2["p"] >= 0.05, q3["p"] < 0.05 and q3["mean"] < 0]
    pred["C3"] = all(c3); V["C3_detail"] = c3

    # C4: VGG-16 w16 and TIN w32
    m = "vgg16"; c4 = []
    for ds, w, lab in (("cifar100", 16, "CIFAR-100 w16 (part B)"), ("tinyimagenet", 32, "Tiny-ImageNet w32 (part C)")):
        G = grp(m, ds, w, [("frozen_random", "d1", "frozen"), ("shuffled", "d1", "shuffled")])
        q1, q2 = block(f"vgg16, {lab}", [("frozen - gap", cmp(G, "frozen", "gap")), ("shuffled - gap", cmp(G, "shuffled", "gap"))])
        block(f"vgg16, {lab}, context", [("frozen - uniform", cmp(G, "frozen", "uniform")), ("frozen - static", cmp(G, "frozen", "static")),
                                         ("shuffled - static", cmp(G, "shuffled", "static")), ("gap - uniform", cmp(G, "gap", "uniform")),
                                         ("gap - static", cmp(G, "gap", "static"))])
        c4 += [q1["p"] < 0.05 and q1["mean"] < 0, q2["p"] < 0.05 and q2["mean"] < 0]
    pred["C4"] = all(c4); V["C4_detail"] = c4

    desc = {"C1": "MobileNetV2: both new frozen draws higher than uniform at CIFAR-100 w64",
            "C2": "VGG-16: neither new frozen draw higher than uniform at CIFAR-100 w64",
            "C3": "MobileNetV2 w16: frozen > static, frozen not distinguishable from gap, shuffled < gap",
            "C4": "VGG-16 w16 and Tiny-ImageNet w32: frozen < gap and shuffled < gap (4 comparisons)"}
    P = ["| prediction | statement | sub-results | outcome |\n|---|---|---|---|"]
    for k in ("C1", "C2", "C3", "C4"):
        P.append(f"| {k} | {desc[k]} | {V[k + '_detail']} | **{'held' if pred[k] else 'failed'}** |")

    # per-cell means
    M = ["| backbone | condition | arm | mean | sd | n |\n|---|---|---|---|---|---|"]
    for m in A:
        for (ds, w, arm, draw), v in sorted(A[m].items()):
            if arm in ("static", "uniform", "gap", "shuffled", "frozen_random"):
                v = np.array(v); M.append(f"| {m} | {ds} w{w} | {arm}{'' if draw=='d1' else ' '+draw} | {np.nanmean(v):.2f} | {np.nanstd(v, ddof=1):.2f} | {int(np.sum(~np.isnan(v)))} |")

    txt = ("# Routing controls: pre-registered analysis\n\n" + "\n".join("- " + l for l in L) +
           "\n\n## Predictions (PREREGISTRATION_CONTROLS.md, 2026-10-22)\n\n" + "\n".join(P) +
           "\n\n## Paired comparisons\n\nPaired t-test over six seeds, TOST margin 0.5 pp. Differences are first minus second.\n\n" + "\n".join(T) +
           "\n\n## Cell means\n\n" + "\n".join(M) + "\n")
    pathlib.Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()

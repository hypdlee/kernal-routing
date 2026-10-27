"""
Analysis of the ResNet-18 routing controls (PREREGISTRATION_CONTROLS_R18.md).

    python tools/controls_r18_analysis.py [--root .] [--out docs/CONTROLS_R18_RESULTS.md]

ResNet-18's 132 original runs are recorded under the older file names
(stage1_*, stage1b_static_*, stage1c_w16_*, stage1c_w32_*, tin_w32_*); the new
control runs under artifacts/controls/resnet18/ use the common tag format.
Every control final accuracy is checked against its train.log line. Pairing
by seed, paired t-test, two-sided, alpha 0.05; TOST margin 0.5 pp.
"""
import argparse, json, pathlib, re, sys
import numpy as np
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kr_stats", ROOT / "report" / "data.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
sys.path.insert(0, str(ROOT / "tools"))
import controls_analysis as CA

OLD = {"stage1": ("cifar100", 64), "stage1b": ("cifar100", 64), "stage1c_w16": ("cifar100", 16),
       "stage1c_w32": ("cifar100", 32), "tin_w32": ("tinyimagenet", 32), "tin_w64": ("tinyimagenet", 64)}
CELL = {"W16": ("cifar100", 16), "W32": ("cifar100", 32), "TIN": ("tinyimagenet", 32), "W64": ("cifar100", 64)}


def load(root):
    acc, problems = {}, []
    for f in sorted((root / "artifacts" / "resnet18" / "results").glob("*_seed*.json")):
        r = json.loads(f.read_text())
        if r["tag"] not in OLD:
            continue
        ds, w = OLD[r["tag"]]
        arm = r["router"]
        acc.setdefault((ds, w, arm, "d1"), {})[r["seed"]] = r["final_acc"]
    ctl = root / "artifacts" / "controls" / "resnet18"
    lv = CA.log_values(ctl / "train.log") if (ctl / "train.log").exists() else {}
    for f in sorted(ctl.glob("*_seed*.json")):
        r = json.loads(f.read_text())
        m = re.match(r"resnet18__(\w+?)__w(\d+)__K(\d+)__(\w+?)(?:__(d\d))?$", r["tag"])
        if not m or int(m.group(3)) != 4 or m.group(4) != "const":
            continue
        if f.name not in lv:
            problems.append(f"{f.name}: no log line")
        elif abs(lv[f.name] - r["final_acc"]) > 0.005:
            problems.append(f"{f.name}: json {r['final_acc']} != log {lv[f.name]}")
        acc.setdefault((m.group(1), int(m.group(2)), r["arm"], m.group(5) or "d1"), {})[r["seed"]] = r["final_acc"]
    return {k: np.array([v.get(s, np.nan) for s in range(6)], float) for k, v in acc.items()}, problems


def q(A, cell, x, y):
    ds, w = CELL[cell]
    return S.paired(A[(ds, w, x[0], x[1])], A[(ds, w, y[0], y[1])], margin=0.5)


def have(A, cell, arm, draw="d1"):
    ds, w = CELL[cell]
    k = (ds, w, arm, draw)
    return k in A and not np.isnan(A[k]).any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=str(ROOT / "docs" / "CONTROLS_R18_RESULTS.md"))
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    A, problems = load(root)
    L = ["log check: " + ("all control values match their train.log line" if not problems else f"{len(problems)} PROBLEMS")] + problems
    rows = ["| comparison | mean diff (pp) | 95% CI | p | TOST p (0.5 pp) | verdict |\n|---|---|---|---|---|---|"]
    res, outcome = {}, {}
    for cell in ("W16", "W32", "TIN", "W64"):
        arms = [(x, d) for x, d in (("static", "d1"), ("uniform", "d1"), ("gap", "d1"), ("shuffled", "d1"), ("frozen_random", "d1"),
                                    ("frozen_random", "d2"), ("frozen_random", "d3")) if have(A, cell, x, d)]
        new = [t for t in arms if t[0] in ("shuffled", "frozen_random") and (cell != "W64" or t[1] != "d1" or True)]
        if not any(t[0] in ("shuffled", "frozen_random") for t in arms) or len(arms) < 4:
            continue
        rows.append(f"| **ResNet-18, {CELL[cell][0]} width {CELL[cell][1]}** | | | | | |")
        for t in new:
            for ref in (("static", "d1"), ("uniform", "d1"), ("gap", "d1")):
                qq = q(A, cell, t, ref); res[(cell, t, ref)] = qq
                rows.append(f"| {t[0]}{'' if t[1]=='d1' else ' '+t[1]} - {ref[0]} | {qq['mean']:+.2f} | [{qq['lo']:+.2f}, {qq['hi']:+.2f}] | "
                            f"{qq['p']:.4f} | {qq['tost']:.4f} | {S.verdict(qq)} |")
    sh, fz, gp, un = ("shuffled", "d1"), ("frozen_random", "d1"), ("gap", "d1"), ("uniform", "d1")
    lo = lambda c, t: (c, t, gp) in res and res[(c, t, gp)]["p"] < 0.05 and res[(c, t, gp)]["mean"] < 0
    hi = lambda c, t, r: (c, t, r) in res and res[(c, t, r)]["p"] < 0.05 and res[(c, t, r)]["mean"] > 0
    P = ["| prediction | statement | outcome |\n|---|---|---|"]
    if all((("TIN", t, gp) in res) for t in (sh, fz)):
        outcome["R1"] = lo("TIN", sh) and lo("TIN", fz); P.append(f"| R1 | Tiny-ImageNet w32: shuffled and frozen_random each lower than gap | **{'held' if outcome['R1'] else 'failed'}** |")
    if all((("W16", sh, gp) in res, ("W32", sh, gp) in res)):
        outcome["R2"] = lo("W16", sh) and lo("W32", sh); P.append(f"| R2 | CIFAR-100 w16 and w32: shuffled lower than gap | **{'held' if outcome['R2'] else 'failed'}** |")
    if ("TIN", fz, un) in res:
        outcome["R3"] = hi("TIN", fz, un); P.append(f"| R3 | Tiny-ImageNet w32: frozen_random higher than uniform | **{'held' if outcome['R3'] else 'failed'}** |")
    txt = ("# ResNet-18 routing controls: pre-registered analysis\n\n" + "\n".join("- " + l for l in L) +
           "\n\n## Predictions (PREREGISTRATION_CONTROLS_R18.md)\n\n" + ("\n".join(P) if len(P) > 1 else "no control results yet") +
           "\n\n## Paired comparisons\n\n" + "\n".join(rows) + "\n")
    pathlib.Path(a.out).write_text(txt); print(txt)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

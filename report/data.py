"""
Result JSONs -> the arrays every figure, table and macro is computed from.

This replaces the hand-transcribed data.py of the ResNet-18 report. It
exposes the same names (ACC, LOSS, PARAMS, ROUTER_K, MINS, ENT, ENT_W64, EP,
CURVE, paired, verdict, acc, n_runs, gpu_hours) so the generators are the
paper's generators, unchanged in their arithmetic.

Call load(model, result_dirs) once, before importing figs/tables/macros.

    ACC[(dataset, width, K, recipe)][arm] -> 6 seeds, NaN where a run is
    missing or diverged. Means and standard deviations ignore NaN; paired
    tests use the seeds both arms have and report n.

Every JSON is classified by what it recorded, not by its file name. Files
from before a key existed get the value every such run had: model
resnet18, placement spatial, width 64, dataset cifar100, constant
temperature. A cell with two result files is an error, as is a recipe
mismatch inside one model or a parameter count that differs from
PARAMS_EXPECTED (tools/param_table.py).
"""
import json, math, pathlib, sys
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.program import PROGRAM, SEEDS as _SEEDS

SEEDS = 6
MARGIN = 0.5
EP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 149]
LEGACY = {"model": "resnet18", "placement": "spatial", "width": 64, "dataset": "cifar100"}
RECIPE_KEYS = ["epochs", "batch_size", "lr", "weight_decay", "warmup", "temperature",
               "impl", "amp", "placement"]


# Tables list groups in the paper's order (by width within each block), not
# in run order.
DISPLAY_ORDER = [("cifar100", 16, 4, "const"), ("cifar100", 32, 4, "const"), ("cifar100", 64, 4, "const"),
                 ("cifar100", 32, 2, "const"), ("cifar100", 32, 8, "const"), ("cifar100", 16, 4, "anneal"),
                 ("cifar100", 32, 4, "anneal"), ("cifar100", 64, 4, "anneal"),
                 ("tinyimagenet", 32, 4, "const"), ("tinyimagenet", 64, 4, "const")]


class NA(np.ndarray):
    """Seed array whose summaries ignore missing runs (NaN)."""
    def mean(self, *a, **k): return float(np.nanmean(np.asarray(self), *a, **k))
    def std(self, *a, **k): return float(np.nanstd(np.asarray(self), *a, **k))
    def min(self, *a, **k): return float(np.nanmin(np.asarray(self), *a, **k))
    def max(self, *a, **k): return float(np.nanmax(np.asarray(self), *a, **k))


def _na(v):
    return np.asarray(v, dtype=float).view(NA)


ACC, LOSS, PARAMS, ROUTER_K, MINS, ENT, ENT_W64, CURVE = {}, {}, {}, {}, {}, {}, {}, {}
ACCUM, DIVERGED, MISSING, IGNORED, RESUMED = {}, [], [], [], []
MODEL = None
SYNTHETIC = False
BANNER = ""       # a note printed at the top of the report (e.g. for a fixture build)


def cell_of(d):
    cfg = d.get("config", {})
    model = d.get("model", cfg.get("model", LEGACY["model"]))
    key = (cfg.get("dataset", LEGACY["dataset"]), int(cfg.get("width", LEGACY["width"])),
           int(cfg.get("num_kernels", 4)), "anneal" if cfg.get("temp_start") is not None else "const")
    arm = d.get("arm") or ("static" if cfg.get("scope") == "none" else d["router"])
    return model, key, arm, int(d["seed"])


def _expected_params(model):
    p = ROOT / "PARAMS_EXPECTED.md"
    out, cur = {}, None
    if not p.exists():
        return out
    for l in p.read_text().splitlines():
        if l.startswith("## "):
            cur = l[3:].strip()
        x = [c.strip() for c in l.split("|")[1:-1]]
        if cur == model and len(x) == 8 and x[0] in ("cifar100", "tinyimagenet"):
            out[(x[0], int(x[1]), int(x[2]), x[3])] = int(x[4])
    return out


def load(model, dirs, strict=True, synthetic=False, epochs=150):
    """epochs: runs trained for any other length (a 1-epoch memory probe,
    a smoke test) are not part of the program; they are listed in IGNORED,
    never averaged in, and never counted as duplicates."""
    global MODEL, SYNTHETIC
    MODEL, SYNTHETIC = model, synthetic
    for D in (ACC, LOSS, PARAMS, ROUTER_K, MINS, ENT, ENT_W64, CURVE, ACCUM):
        D.clear()
    for L in (DIVERGED, MISSING, IGNORED, RESUMED):
        L.clear()
    want = {(g.key, a, s) for g in PROGRAM for a in g.arms for s in _SEEDS}
    recs, errors = {}, []
    for d in dirs:
        for f in sorted(pathlib.Path(d).rglob("*.json")):
            if f.name.startswith("plan") or "probes" in f.parts:
                continue
            try:
                r = json.loads(f.read_text())
            except Exception as e:
                errors.append(f"unreadable {f}: {e}")
                continue
            if "final_acc" not in r:
                continue
            m, key, arm, seed = cell_of(r)
            if m != model:
                continue
            if epochs is not None and r.get("config", {}).get("epochs", epochs) != epochs:
                IGNORED.append(f"{f.name} (epochs {r['config'].get('epochs')})")
                continue
            if (key, arm, seed) not in want:
                IGNORED.append(f.name)
                continue
            if (key, arm, seed) in recs:
                errors.append(f"two results for {key} {arm} seed {seed}: {recs[(key, arm, seed)][0]} and {f}")
            recs[(key, arm, seed)] = (f, r)
        for f in sorted(pathlib.Path(d).rglob("*.DIVERGED")):
            name = f.name[:-len(".DIVERGED")]
            if name.startswith(model + "__"):
                DIVERGED.append(name)
    # one recipe per model
    for k in RECIPE_KEYS:
        vals = {json.dumps(r["config"].get(k, LEGACY.get(k))) for _, r in recs.values()}
        if len(vals) > 1:
            errors.append(f"recipe mismatch in {k}: {sorted(vals)}")
    exp = _expected_params(model)
    for g in sorted(PROGRAM, key=lambda g: DISPLAY_ORDER.index(g.key)):
        ACC[g.key], LOSS[g.key], PARAMS[g.key], MINS[g.key] = {}, {}, {}, {}
        for arm in g.arms:
            got = [recs.get((g.key, arm, s)) for s in _SEEDS]
            for s, x in zip(_SEEDS, got):
                if x is None:
                    MISSING.append((g.key, arm, s))
            have = [x[1] for x in got if x]
            ACC[g.key][arm] = _na([x[1]["final_acc"] if x else np.nan for x in got])
            LOSS[g.key][arm] = float(np.mean([r["history"][-1]["loss"] for r in have])) if have else np.nan
            MINS[g.key][arm] = float(np.mean([r["minutes"] for r in have])) if have else np.nan
            ps = {r["params"]["total"] for r in have}
            if len(ps) > 1:
                errors.append(f"{g.key} {arm}: parameter counts differ across seeds {ps}")
            PARAMS[g.key][arm] = (ps.pop() / 1e6) if ps else np.nan
            e = exp.get((g.dataset, g.width, g.K, arm))
            if have and e is not None and have[0]["params"]["total"] != e:
                errors.append(f"{g.key} {arm}: {have[0]['params']['total']} params, expected {e} "
                              f"(PARAMS_EXPECTED.md): wrong width, K or arm")
            ACCUM[(g.key, arm)] = sorted({r.get("accum", 1) for r in have})
            RESUMED.extend(f"{g.key} {arm} seed {r['seed']}" for r in have if r.get("resumed_at_epoch"))
            if arm == "gap" and g.K == 4 and g.recipe == "const" and have:
                ROUTER_K[(g.dataset, g.width)] = have[0]["params"]["router"] / 1e3
            if arm == "gap" and g.K == 4:
                ENT[(g.dataset, g.width, g.recipe)] = _na([_ent(x[1]) if x else np.nan for x in got])
            if g.key == ("cifar100", 64, 4, "const") and arm != "static":
                ENT_W64[arm] = _na([_ent(x[1]) if x else np.nan for x in got])
            if g.recipe == "const" and g.K == 4 and got[0] is not None:
                h = got[0][1]["history"]
                if len(h) >= 150:
                    CURVE[(g.dataset, g.width, arm)] = [h[e]["loss"] for e in EP]
    if errors and strict:
        raise SystemExit("analysis refused:\n  " + "\n  ".join(errors))
    return errors


def _ent(r):
    v = [l["z_entropy_norm"] for l in r.get("routing", [])]
    return float(np.mean(v)) if v else np.nan


def paired(x, y, margin=MARGIN):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 2:
        nan = float("nan")
        return dict(mean=nan, sd=nan, lo=nan, hi=nan, p=nan, wil=nan, dz=nan, tost=nan, n=n)
    d = x - y; sd = d.std(ddof=1)
    t, p = stats.ttest_rel(x, y)
    se = sd / math.sqrt(n); crit = stats.t.ppf(0.975, n - 1)
    try:    w = float(stats.wilcoxon(x, y).pvalue)
    except ValueError: w = float("nan")
    tost = max(stats.t.sf((d.mean()+margin)/se, n-1), stats.t.cdf((d.mean()-margin)/se, n-1))
    return dict(mean=d.mean(), sd=sd, lo=d.mean()-crit*se, hi=d.mean()+crit*se,
                p=float(p), wil=w, dz=d.mean()/sd, tost=float(tost), n=n)


def verdict(q):
    if q["n"] < 2 or math.isnan(q["p"]): return "no data"
    v = ("equivalent" if q["tost"] < 0.05 and q["p"] >= 0.05 else
         ("higher" if q["mean"] > 0 else "lower") if q["p"] < 0.05 else "inconclusive")
    return v if q["n"] == SEEDS else f"{v} (n={q['n']})"


def acc(ds, w, arm, k=4, rec="const"): return ACC[(ds, w, k, rec)][arm]


def n_runs(): return int(sum(np.sum(~np.isnan(np.asarray(v))) for g in ACC.values() for v in g.values()))


def gpu_hours():
    return sum(MINS[k][a] * np.sum(~np.isnan(np.asarray(ACC[k][a])))
               for k in ACC for a in ACC[k] if not math.isnan(MINS[k][a])) / 60

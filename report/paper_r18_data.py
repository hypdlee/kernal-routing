"""
Single source of truth for the report. Every per-seed final-epoch top-1
accuracy below was transcribed from the training logs (final_acc is printed
verbatim at the end of each run). Nothing downstream is typed by hand:
figures and tables are computed from these arrays.
"""
import math
import numpy as np
from scipy import stats

SEEDS = 6
A = lambda *v: np.array(v, dtype=float)

# ---- accuracy, (dataset, width, K, recipe) -> arm -> 6 seeds -------------
ACC = {
 ("cifar100", 16, 4, "const"): dict(
    static =A(69.91,69.75,69.86,70.46,70.00,69.99),
    uniform=A(69.68,69.32,69.84,69.91,69.72,70.16),
    gap    =A(70.35,70.51,70.41,70.63,70.79,70.47)),
 ("cifar100", 32, 4, "const"): dict(
    static =A(74.81,74.65,74.53,74.92,75.11,75.21),
    uniform=A(75.29,74.86,74.75,75.30,74.71,75.37),
    gap    =A(76.13,75.88,75.91,75.90,76.27,76.06)),
 ("cifar100", 64, 4, "const"): dict(
    static       =A(78.05,78.67,78.63,78.27,78.00,78.41),
    uniform      =A(78.64,78.59,78.34,78.30,78.36,78.09),
    gap          =A(78.59,78.81,78.11,78.61,78.07,78.32),
    shuffled     =A(78.42,77.83,78.18,78.41,78.47,78.28),
    frozen_random=A(78.47,78.49,78.48,78.63,78.45,78.57)),
 ("cifar100", 32, 2, "const"): dict(gap=A(75.94,75.80,76.14,75.78,76.10,75.94)),
 ("cifar100", 32, 8, "const"): dict(gap=A(76.13,75.83,76.19,76.34,76.11,75.69)),
 ("cifar100", 16, 4, "anneal"): dict(gap=A(71.07,70.62,70.17,70.69,70.69,70.61)),
 ("cifar100", 32, 4, "anneal"): dict(gap=A(75.60,75.01,75.61,75.25,75.18,75.41)),
 ("cifar100", 64, 4, "anneal"): dict(gap=A(78.44,78.65,78.61,78.10,78.24,78.18)),
 ("tinyimagenet", 32, 4, "const"): dict(
    static =A(61.93,61.61,61.57,61.46,61.27,61.52),
    uniform=A(61.52,61.53,61.55,61.62,61.66,62.12),
    gap    =A(63.53,64.20,63.61,63.58,63.91,63.64)),
 ("tinyimagenet", 64, 4, "const"): dict(
    static =A(66.29,66.59,65.88,65.97,66.63,66.23),
    uniform=A(66.27,66.27,66.41,66.21,66.22,66.53),
    gap    =A(68.73,68.35,68.42,68.55,68.71,68.52)),
}

# ---- final-epoch training loss, mean over seeds -------------------------
LOSS = {
 ("cifar100",16,4,"const"): dict(static=.1643, uniform=.1653, gap=.0638),
 ("cifar100",32,4,"const"): dict(static=.0170, uniform=.0170, gap=.0132),
 ("cifar100",64,4,"const"): dict(static=.0080, uniform=.0081, gap=.0074,
                                 shuffled=.0083, frozen_random=.0077),
 ("cifar100",32,2,"const"): dict(gap=.0137),
 ("cifar100",32,8,"const"): dict(gap=.0133),
 ("cifar100",16,4,"anneal"): dict(gap=.1540),
 ("cifar100",32,4,"anneal"): dict(gap=.0173),
 ("cifar100",64,4,"anneal"): dict(gap=.0080),
 ("tinyimagenet",32,4,"const"): dict(static=.1194, uniform=.1200, gap=.0420),
 ("tinyimagenet",64,4,"const"): dict(static=.0210, uniform=.0220, gap=.0190),
}

# ---- stored parameters (millions) and router parameters -----------------
PARAMS = {
 ("cifar100",16,4,"const"): dict(static=.708, uniform=2.770, gap=2.791),
 ("cifar100",32,4,"const"): dict(static=2.818, uniform=11.056, gap=11.134),
 ("cifar100",64,4,"const"): dict(static=11.220, uniform=44.177, gap=44.465,
                                 shuffled=44.465, frozen_random=44.177),
 ("cifar100",32,2,"const"): dict(gap=5.643),
 ("cifar100",32,8,"const"): dict(gap=22.116),
 ("cifar100",16,4,"anneal"): dict(gap=2.791),
 ("cifar100",32,4,"anneal"): dict(gap=11.134),
 ("cifar100",64,4,"anneal"): dict(gap=44.465),
 ("tinyimagenet",32,4,"const"): dict(static=2.846, uniform=11.09, gap=11.160),
 ("tinyimagenet",64,4,"const"): dict(static=11.27, uniform=44.23, gap=44.52),
}
ROUTER_K = {("cifar100",16):21.3, ("cifar100",32):74.6, ("cifar100",64):288.0,
            ("tinyimagenet",32):74.6, ("tinyimagenet",64):288.0}

# ---- minutes per run, mean over seeds (from the logged wall clock) ------
MINS = {
 ("cifar100",16,4,"const"): dict(static=9.6, uniform=11.4, gap=33.7),
 ("cifar100",32,4,"const"): dict(static=8.6, uniform=12.9, gap=39.2),
 ("cifar100",64,4,"const"): dict(static=18.8, uniform=25.2, gap=86.0,
                                 shuffled=92.3, frozen_random=85.4),
 ("cifar100",32,2,"const"): dict(gap=27.9),
 ("cifar100",32,8,"const"): dict(gap=69.3),
 ("cifar100",16,4,"anneal"): dict(gap=44.1),
 ("cifar100",32,4,"anneal"): dict(gap=38.4),
 ("cifar100",64,4,"anneal"): dict(gap=86.0),
 ("tinyimagenet",32,4,"const"): dict(static=54.2, uniform=56.3, gap=284.2),
 ("tinyimagenet",64,4,"const"): dict(static=129.0, uniform=141.0, gap=687.2),
}

# ---- normalised routing entropy, per seed: mean over layers and over the
# ---- last test batch (16 images, the same in every run; see README) ------
ENT = {
 ("cifar100",16,"const"):  A(0.9613,0.9566,0.9662,0.9550,0.9549,0.9670),
 ("cifar100",32,"const"):  A(0.9757,0.9770,0.9782,0.9795,0.9856,0.9761),
 ("cifar100",64,"const"):  A(0.9879,0.9912,0.9880,0.9879,0.9895,0.9867),
 ("cifar100",16,"anneal"): A(0.9986,0.9983,0.9973,0.9948,0.9979,0.9962),
 ("cifar100",32,"anneal"): A(0.9994,0.9996,0.9993,0.9998,0.9988,0.9987),
 ("cifar100",64,"anneal"): A(0.9999,1.0000,0.9999,0.9999,0.9999,0.9993),
}
ENT_W64 = dict(uniform=A(1,1,1,1,1,1),
               gap=ENT[("cifar100",64,"const")],
               shuffled=A(0.9996,0.9995,0.9995,0.9996,0.9997,0.9997),
               frozen_random=A(0.9974,0.9967,0.9967,0.9975,0.9977,0.9968))

# ---- seed-0 training-loss curves, sampled every 10 epochs, from the logs
EP = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,149]
CURVE = {
 ("cifar100",16,"static"):[4.393,1.808,1.538,1.436,1.365,1.309,1.235,1.150,1.062,0.940,0.812,0.646,0.459,0.282,0.181,0.160],
 ("cifar100",32,"static"):[4.319,1.586,1.297,1.193,1.110,1.017,0.933,0.820,0.715,0.570,0.413,0.230,0.088,0.028,0.019,0.017],
 ("cifar100",64,"static"):[4.238,1.450,1.137,1.012,0.924,0.834,0.741,0.627,0.506,0.379,0.222,0.090,0.016,0.009,0.008,0.008],
 ("tinyimagenet",64,"static"):[5.000,2.295,2.004,1.874,1.773,1.677,1.553,1.406,1.234,1.007,0.718,0.396,0.116,0.031,0.022,0.021],
 ("tinyimagenet",64,"gap"):[5.002,2.275,1.965,1.824,1.714,1.604,1.470,1.310,1.105,0.848,0.545,0.239,0.047,0.026,0.020,0.019],
}

MARGIN = 0.5

def paired(x, y, margin=MARGIN):
    d = x - y; n = len(d); sd = d.std(ddof=1)
    t, p = stats.ttest_rel(x, y)
    se = sd / math.sqrt(n); crit = stats.t.ppf(0.975, n - 1)
    try:    w = float(stats.wilcoxon(x, y).pvalue)
    except ValueError: w = float("nan")
    tost = max(stats.t.sf((d.mean()+margin)/se, n-1), stats.t.cdf((d.mean()-margin)/se, n-1))
    return dict(mean=d.mean(), sd=sd, lo=d.mean()-crit*se, hi=d.mean()+crit*se,
                p=float(p), wil=w, dz=d.mean()/sd, tost=float(tost))

def verdict(q):
    if q["tost"] < 0.05 and q["p"] >= 0.05: return "equivalent"
    if q["p"] < 0.05: return "higher" if q["mean"] > 0 else "lower"
    return "inconclusive"

def acc(ds, w, arm, k=4, rec="const"): return ACC[(ds, w, k, rec)][arm]

def n_runs(): return sum(len(v) for g in ACC.values() for v in g.values())

def gpu_hours():
    return sum(m * SEEDS for g in MINS.values() for m in g.values()) / 60

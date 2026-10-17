"""
Synthetic result JSONs for one model, in the exact format src/train.py
writes. For testing the analysis and report pipeline ONLY: every figure
built from these carries a SYNTHETIC TEST DATA watermark and every report a
banner. Parameter counts are the real counted ones, so the analysis's
parameter check is exercised.

    python report/synth.py --model vgg16 --out /tmp/synth/vgg16 [--drop 2] [--diverge 1]
"""
import argparse, json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.program import PROGRAM, tag, result_name

BASE = {"cifar100": {16: 66.0, 32: 72.0, 64: 76.0}, "tinyimagenet": {32: 57.0, 64: 62.0}}
OFFSET = {"resnet18": 2.0, "mobilenetv2": -2.0, "vgg16": -3.0, "resnet50": 3.0, "convnext_tiny": -8.0}
EFFECT = {"static": 0.0, "uniform": 0.05, "gap": 0.9, "shuffled": 0.1, "frozen_random": 0.15}
NDYN = {"resnet18": 16, "resnet50": 16, "vgg16": 12, "mobilenetv2": 17, "convnext_tiny": 18}


def counted(model):
    out, cur = {}, None
    for l in (ROOT / "PARAMS_EXPECTED.md").read_text().splitlines():
        if l.startswith("## "): cur = l[3:].strip()
        x = [c.strip() for c in l.split("|")[1:-1]]
        if cur == model and len(x) == 8 and x[0] in ("cifar100", "tinyimagenet"):
            out[(x[0], int(x[1]), int(x[2]), x[3])] = (int(x[4]), int(x[6]))
    return out


def make(model, out, seed=0, drop=0, diverge=0):
    rng = np.random.default_rng(seed)
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    C = counted(model); files = []
    for g in PROGRAM:
        for arm in g.arms:
            for s in range(6):
                acc = BASE[g.dataset][g.width] + OFFSET[model] + EFFECT[arm] * (1 + 0.1 * np.log2(g.K / 4)) \
                      - (0.3 if g.recipe == "anneal" else 0) + rng.normal(0, 0.2)
                loss_end = 0.2 * 2 ** (-(g.width / 16)) * (1.5 if g.dataset == "tinyimagenet" else 1) * (0.7 if arm == "gap" else 1)
                hist = [{"epoch": e, "loss": round(float(4.6 * np.exp(-e / 25) + loss_end), 4),
                         "test_acc": 0.0, "lr": 0.0} for e in range(150)]
                ent = {"static": None, "uniform": 1.0, "shuffled": 0.9996, "frozen_random": 0.997}.get(arm)
                if arm == "gap":
                    ent = 0.9999 if g.recipe == "anneal" else 0.98 + 0.005 * np.log2(g.width / 16)
                routing = [] if ent is None else [
                    {"layer": i, "router": arm, "z_mean": [0.25] * g.K,
                     "z_entropy_norm": round(float(min(1.0, ent + rng.normal(0, 2e-4) * (arm == "gap"))), 5),
                     "argmax_counts": [1] * g.K} for i in range(NDYN[model])]
                tot, rt = C[(g.dataset, g.width, g.K, arm)]
                cfg = {"model": model, "placement": "spatial", "router": "gap" if arm == "static" else arm,
                       "seed": s, "epochs": 150, "batch_size": 128, "accum": 1, "lr": 0.1, "weight_decay": 5e-4,
                       "warmup": 5, "num_kernels": g.K, "scope": "none" if arm == "static" else "all",
                       "width": g.width, "impl": "blend", "temperature": 1.0,
                       "temp_start": 30.0 if g.recipe == "anneal" else None, "dataset": g.dataset,
                       "amp": "bf16", "tag": tag(model, g), "arm_name": arm}
                rec = {"tag": cfg["tag"], "model": model, "placement": "spatial", "arm": arm, "micro_batch": 128,
                       "accum": 1, "resumed_at_epoch": None, "router": cfg["router"], "seed": s,
                       "final_acc": round(float(acc), 3), "best_acc": round(float(acc) + 0.1, 3),
                       "params": {"total": tot, "trainable": tot, "router": rt},
                       "minutes": round(float(10 * (g.width / 16) * (4 if arm != "static" else 1)), 2),
                       "config": cfg, "routing": routing, "history": hist}
                f = out / result_name(model, g, arm, s)
                f.write_text(json.dumps(rec)); files.append(f)
    # knock out some runs: missing (deleted) and diverged (marker, no result)
    pick = rng.choice(len(files), drop + diverge, replace=False)
    for i, j in enumerate(pick):
        files[j].unlink()
        if i >= drop:
            pathlib.Path(str(files[j]) + ".DIVERGED").write_text('{"diverged": true}')
    return len(files) - drop - diverge


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", type=int, default=0); ap.add_argument("--diverge", type=int, default=0)
    a = ap.parse_args()
    print(make(a.model, a.out, a.seed, a.drop, a.diverge), "result files")

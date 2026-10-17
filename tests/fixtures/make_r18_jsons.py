"""
Write the 132 ResNet-18 results as result JSONs in the format the study's
train.py produced (no 'model' or 'arm' keys, stage tags, --router in the
file name), from the values the final paper was built on
(report/paper_r18_data.py). Parameter counts are the counted ones
(PARAMS_EXPECTED.md), as real JSONs record them.

Used by tests/test_analysis.py to show the JSON-driven pipeline reproduces
the paper's numbers. It is a fixture, not data: the user's real JSONs
replace it (python report/build_model_report.py --model resnet18 --results <dir>).
"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "report")); sys.path.insert(0, str(ROOT))
import paper_r18_data as P
from src.program import PROGRAM


def counted():
    out, cur = {}, None
    for l in (ROOT / "PARAMS_EXPECTED.md").read_text().splitlines():
        if l.startswith("## "): cur = l[3:].strip()
        x = [c.strip() for c in l.split("|")[1:-1]]
        if cur == "resnet18" and len(x) == 8 and x[0] in ("cifar100", "tinyimagenet"):
            out[(x[0], int(x[1]), int(x[2]), x[3])] = (int(x[4]), int(x[6]))
    return out


def write(outdir):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    C = counted(); n = 0
    for g in PROGRAM:
        for arm in g.arms:
            accs = P.ACC[g.key][arm]
            L = P.LOSS[g.key][arm]
            curve = P.CURVE.get((g.dataset, g.width, arm)) if g.recipe == "const" and g.K == 4 else None
            for s in range(6):
                # seed 0 follows the logged curve; the others make the mean equal the paper's
                if curve:
                    last = curve[-1] if s == 0 else (6 * L - curve[-1]) / 5
                else:
                    last = L
                hist = [{"epoch": e, "loss": last, "test_acc": 0.0, "lr": 0.0} for e in range(150)]
                if curve and s == 0:
                    for e, v in zip(P.EP, curve):
                        hist[e]["loss"] = v
                if arm == "static":
                    ent = None
                elif arm == "uniform":
                    ent = 1.0
                elif arm == "gap" and g.K == 4 and (g.dataset, g.width, g.recipe) in P.ENT:
                    ent = float(P.ENT[(g.dataset, g.width, g.recipe)][s])
                elif g.key == ("cifar100", 64, 4, "const") and arm in P.ENT_W64:
                    ent = float(P.ENT_W64[arm][s])
                else:
                    ent = 0.99
                tot, rt = C[(g.dataset, g.width, g.K, arm)]
                cfg = {"router": "gap" if arm == "static" else arm, "seed": s, "epochs": 150,
                       "batch_size": 128, "lr": 0.1, "weight_decay": 5e-4, "warmup": 5,
                       "num_kernels": g.K, "scope": "none" if arm == "static" else "all",
                       "impl": "blend", "temperature": 1.0, "amp": "bf16", "tag": f"stage_{g.dataset}_{g.width}"}
                # older runs predate --width / --dataset / --temp-start: omit the key, as they did
                if g.width != 64: cfg["width"] = g.width
                if g.dataset != "cifar100": cfg["dataset"] = g.dataset
                if g.recipe == "anneal":
                    cfg.update(temp_start=30.0, temp_end=1.0, temp_epochs=10)
                rec = {"tag": cfg["tag"], "router": cfg["router"], "seed": s, "final_acc": float(accs[s]),
                       "best_acc": float(accs[s]), "params": {"total": tot, "trainable": tot, "router": rt},
                       "minutes": P.MINS[g.key][arm], "config": cfg,
                       "routing": [] if ent is None else [{"layer": 0, "router": arm, "z_entropy_norm": ent}],
                       "history": hist}
                name = f"{cfg['tag']}_K{g.K}_{g.recipe}_{cfg['router'] if arm != 'static' else 'static'}_seed{s}.json"
                (outdir / name).write_text(json.dumps(rec))
                n += 1
    return n


if __name__ == "__main__":
    print(write(sys.argv[1]), "files")

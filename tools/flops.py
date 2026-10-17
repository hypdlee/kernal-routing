"""
Counted multiply-accumulates per image, and a labelled compute ESTIMATE per
model for the full 132-run program.

    python tools/flops.py [--out COMPUTE_ESTIMATE.md]

How the estimate is made. The ResNet-18 study logged wall-clock minutes for
every (dataset, width, K, recipe, arm) cell on the user's GPU. For another
model the same cell is estimated as

    minutes(model) = minutes(resnet18) * MACs(model) / MACs(resnet18)

with MACs counted for the same cell (same arm, so the per-sample kernel
mixing of dynamic arms is included). This is an ESTIMATE. It assumes time
scales with arithmetic, which is false for depthwise convolutions (memory
bound): MobileNetV2 and ConvNeXt-Tiny will run slower than this predicts,
probably by a factor of 1.5-3. The per-group probe in run_all_models.py
measures real seconds per step before any group starts and replaces this.
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from src.models import MODELS, RUN_ORDER, build_model
from src.dynconv import DynamicConv2d
from src.data import num_classes
from src.program import PROGRAM, SEEDS, arm_flags

RES = {"cifar100": 32, "tinyimagenet": 64}

# ResNet-18 minutes per run, mean over seeds, from the training logs
# (report generator data.py MINS).
R18_MINS = {
    ("cifar100", 16, 4, "const"): dict(static=9.6, uniform=11.4, gap=33.7),
    ("cifar100", 32, 4, "const"): dict(static=8.6, uniform=12.9, gap=39.2),
    ("cifar100", 64, 4, "const"): dict(static=18.8, uniform=25.2, gap=86.0, shuffled=92.3, frozen_random=85.4),
    ("cifar100", 32, 2, "const"): dict(gap=27.9),
    ("cifar100", 32, 8, "const"): dict(gap=69.3),
    ("cifar100", 16, 4, "anneal"): dict(gap=44.1),
    ("cifar100", 32, 4, "anneal"): dict(gap=38.4),
    ("cifar100", 64, 4, "anneal"): dict(gap=86.0),
    ("tinyimagenet", 32, 4, "const"): dict(static=54.2, uniform=56.3, gap=284.2),
    ("tinyimagenet", 64, 4, "const"): dict(static=129.0, uniform=141.0, gap=687.2),
}


def macs(model, res):
    """MACs per image for one forward pass. Convolutions and linears are
    counted; a dynamic layer adds the per-sample mixing of its K kernels
    (K * |W| MACs). BN, activations and pooling are ignored (small)."""
    total = [0]

    def conv_hook(m, inp, out):
        cin = m.in_channels
        k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
        hw = out.shape[-1] * out.shape[-2]
        total[0] += hw * out.shape[1] * (cin // m.groups) * k * k
        if isinstance(m, DynamicConv2d):
            total[0] += m.num_kernels * m.weight[0].numel()

    def lin_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    hs = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, DynamicConv2d)):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, res, res))
    for h in hs:
        h.remove()
    return total[0]


def table():
    out = {}
    for name in MODELS:
        for g in PROGRAM:
            for arm in g.arms:
                key = (name, g.dataset, g.width, g.K, arm)
                if key not in out:
                    m = build_model(name, num_classes(g.dataset), num_kernels=g.K, width=g.width, **arm_flags(arm))
                    out[key] = macs(m, RES[g.dataset])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="COMPUTE_ESTIMATE.md")
    a = ap.parse_args()
    M = table()
    lines = ["# Compute estimate (ESTIMATE, not a measurement)", "",
             "MACs are counted per image, forward pass. Hours scale the ResNet-18 study's logged",
             "minutes by the MAC ratio for the same cell. Depthwise-heavy models (MobileNetV2,",
             "ConvNeXt-Tiny) are memory bound and will be slower than this. The runner's probe",
             "measures the real rate before each group and prints a measured ETA.", "",
             "| model | CIFAR-100 w64 static GMACs | est. GPU-h, 132 runs | vs ResNet-18 |", "|---|---|---|---|"]
    per = {}
    for name in ["resnet18"] + RUN_ORDER:
        hrs = 0.0
        for g in PROGRAM:
            for arm in g.arms:
                r18 = M[("resnet18", g.dataset, g.width, g.K, arm)]
                hrs += R18_MINS[(g.dataset, g.width, g.K, g.recipe)][arm] * len(SEEDS) / 60 \
                       * M[(name, g.dataset, g.width, g.K, arm)] / r18
        per[name] = hrs
    for name, hrs in per.items():
        lines.append(f"| {name} | {M[(name, 'cifar100', 64, 4, 'static')]/1e9:.3f} | {hrs:.0f} | {hrs/per['resnet18']:.2f}x |")
    lines += ["", f"Four new models together: {sum(v for k, v in per.items() if k != 'resnet18'):.0f} GPU-h (estimate).", "",
              "## MACs per image, every cell", "", "| model | dataset | width | K | arm | MMACs |", "|---|---|---|---|---|---|"]
    for (n, ds, w, k, arm), v in sorted(M.items()):
        lines.append(f"| {n} | {ds} | {w} | {k} | {arm} | {v/1e6:.1f} |")
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
    for l in lines[6:6 + 2 + len(per) + 2]:
        print(l)
    print(f"{a.out} written, {len(M)} cells")


if __name__ == "__main__":
    main()

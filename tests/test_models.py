"""Model-level checks for the cross-architecture replication (gates A2-A6).

    python tests/test_models.py build | placement | collapse | shuffle | resnet18_identity | all
"""
import sys, pathlib, itertools, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from src.models import MODELS, build_model, dynamic_layers, collapse_to_static
from src.resnet import ResNet18, param_counts
from src.data import num_classes

DATASETS = {"cifar100": 32, "tinyimagenet": 64}
WIDTHS = (16, 32, 64)
KS = (2, 4, 8)
ROUTERS = ("uniform", "gap", "shuffled", "frozen_random")
EXPECTED_DYN = {"resnet18": 16, "resnet50": 16, "vgg16": 12, "mobilenetv2": 17, "convnext_tiny": 18}


def configs(model):
    for ds, w in itertools.product(DATASETS, WIDTHS):
        yield ds, w, "none", 4, "gap"
        for k, r in itertools.product(KS, ROUTERS):
            yield ds, w, "all", k, r


def _sharpen_routers(model, scale=200.0):
    """Freshly initialised routers emit near-uniform weights, which makes any
    routing-dependent difference vanishingly small. Scale each router's last
    layer so routing is decisive, then compare."""
    for m in dynamic_layers(model):
        last = [l for l in m.router.modules() if isinstance(l, torch.nn.Linear)]
        if last:
            last[-1].weight.mul_(scale)


def test_build():
    ok = fail = 0
    t0 = time.time()
    for name in MODELS:
        for ds, w, scope, k, r in configs(name):
            try:
                m = build_model(name, num_classes(ds), scope=scope, num_kernels=k, router=r, width=w).eval()
                with torch.no_grad():
                    y = m(torch.randn(2, 3, DATASETS[ds], DATASETS[ds]))
                assert y.shape == (2, num_classes(ds)), y.shape
                assert torch.isfinite(y).all()
                nd = len(dynamic_layers(m))
                assert nd == (0 if scope == "none" else EXPECTED_DYN[name]), nd
                ok += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {name} {ds} w{w} {scope} K{k} {r}: {e}")
    n_expected = len(MODELS) * len(DATASETS) * len(WIDTHS) * (1 + len(KS) * len(ROUTERS))
    print(f"{ok} configurations ok, {fail} failed (expected {n_expected}) in {time.time()-t0:.0f}s")
    assert fail == 0 and ok == n_expected


def test_placement():
    counts = {}
    for name in MODELS:
        m = build_model(name, 100, scope="all", width=16)
        counts[name] = len(dynamic_layers(m))
        assert len(dynamic_layers(build_model(name, 100, scope="none", width=16))) == 0
    print("spatial placement:", ", ".join(f"{k} {v}" for k, v in counts.items()))
    assert counts == EXPECTED_DYN, counts
    # the alternative rule is available for the four new models only
    alt = {n: len(dynamic_layers(build_model(n, 100, scope="all", width=16, placement="all")))
           for n in MODELS if n != "resnet18"}
    print("all-but-stem placement (not run):", ", ".join(f"{k} {v}" for k, v in alt.items()))
    # VGG has only 3x3 convolutions, so both rules coincide there.
    assert alt == {"resnet50": 52, "vgg16": 12, "mobilenetv2": 50, "convnext_tiny": 57}, alt


def test_collapse():
    """Proposition 1 per model: the uniform arm computes exactly what a static
    network with each bank replaced by its mean computes."""
    worst = 0.0
    for name in MODELS:
        torch.manual_seed(0)
        m = build_model(name, 100, scope="all", router="uniform", width=16).double().eval()
        with torch.no_grad():
            for p in m.parameters():          # move off the zero/ones init so BN etc. matter
                p.add_(0.01 * torch.randn_like(p))
            s = collapse_to_static(m).eval()
            assert len(dynamic_layers(s)) == 0
            x = torch.randn(3, 3, 32, 32, dtype=torch.float64)
            d = (m(x) - s(x)).abs().max().item()
        worst = max(worst, d)
        # Negative control: a routed network is NOT a mean-kernel network, so
        # the same collapse must fail for it. Without this the check could
        # pass vacuously.
        torch.manual_seed(0)
        g = build_model(name, 100, scope="all", router="gap", width=16).double().eval()
        with torch.no_grad():
            for p in g.parameters():
                p.add_(0.01 * torch.randn_like(p))
            _sharpen_routers(g)
            dg = (g(x) - collapse_to_static(g).eval()(x)).abs().max().item()
        assert dg > 1e-6, (name, dg)
        print(f"  {name:<14} max |uniform - collapsed static| = {d:.2e}   "
              f"(control: routed arm differs by {dg:.2e})")
    print(f"{len(MODELS)} models, max abs diff {worst:.2e}")
    assert worst < 1e-9


def test_shuffle():
    for name in MODELS:
        torch.manual_seed(0)
        g = build_model(name, 100, scope="all", router="gap", width=16).eval()
        torch.manual_seed(0)
        s = build_model(name, 100, scope="all", router="shuffled", width=16).eval()
        s.load_state_dict(g.state_dict())
        with torch.no_grad():
            for p in g.parameters():
                p.add_(0.05 * torch.randn_like(p))
            _sharpen_routers(g)
            s.load_state_dict(g.state_dict())
            # Per-sample shift and scale. With iid noise every image has almost
            # the same channel means, so all GAP descriptors coincide and a
            # shuffle swaps near-identical routing inputs (diagnosed: median
            # across-sample spread of z 7.7e-4 vs 1.4e-2 with this input).
            x = torch.randn(16, 3, 32, 32) * (3 * torch.rand(16, 1, 1, 1)) + torch.randn(16, 3, 1, 1)
            torch.manual_seed(1)
            yg, ys = g(x), s(x)
            # Relative to output scale: MobileNetV2 and VGG initialise their
            # classifier at N(0, 0.01), so absolute logit differences are tiny.
            diff = ((yg - ys).abs().max() / yg.abs().max()).item()
        assert diff > 1e-3, (name, diff)
        print(f"  {name:<14} shuffled differs from gap in eval: max relative diff {diff:.3e}")
    print(f"{len(MODELS)} models ok")


def test_resnet18_identity():
    for scope, router in [("none", "gap"), ("all", "uniform"), ("all", "gap")]:
        a = build_model("resnet18", 100, scope=scope, router=router, width=64)
        b = ResNet18(100, scope=scope, router=router, width=64)
        pa, pb = param_counts(a)["total"], param_counts(b)["total"]
        assert pa == pb, (pa, pb)
        assert list(a.state_dict()) == list(b.state_dict())
        print(f"  resnet18 {scope:<4} {router:<8} params {pa}")
    print("ok")


TESTS = {"build": test_build, "placement": test_placement, "collapse": test_collapse,
         "shuffle": test_shuffle, "resnet18_identity": test_resnet18_identity}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for w in (TESTS if which == ["all"] else which):
        print(f"== {w}")
        TESTS[w]()

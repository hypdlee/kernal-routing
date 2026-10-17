"""Equality tests for the three DynamicConv2d execution paths.

Run with:  python -m pytest tests -q       (or just: python tests/test_paths.py)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.dynconv import DynamicConv2d
from src.routers import ROUTERS


def _sync(a, b):
    b.weight.data.copy_(a.weight.data)
    if a.bias is not None:
        b.bias.data.copy_(a.bias.data)
    b.router.load_state_dict(a.router.state_dict())


def test_blend_equals_combine():
    torch.manual_seed(0)
    x = torch.randn(6, 16, 12, 12, dtype=torch.float64)
    for router in ["gap", "frozen_random", "gap_std_max", "dct_topk"]:
        a = DynamicConv2d(16, 24, 3, padding=1, bias=True, num_kernels=4,
                          router=router, impl="blend").double()
        b = DynamicConv2d(16, 24, 3, padding=1, bias=True, num_kernels=4,
                          router=router, impl="combine").double()
        _sync(a, b)
        torch.manual_seed(7); ya = a(x)
        torch.manual_seed(7); yb = b(x)
        assert torch.allclose(ya, yb, atol=1e-10), router
    print("ok  blend == combine")


def test_uniform_equals_mean_kernel():
    torch.manual_seed(0)
    x = torch.randn(5, 16, 12, 12, dtype=torch.float64)
    m = DynamicConv2d(16, 24, 3, stride=2, padding=1, bias=True,
                      num_kernels=4, router="uniform").double()
    ref = F.conv2d(x, m.weight.mean(0), m.bias.mean(0), 2, 1)
    assert torch.allclose(m(x), ref, atol=1e-10)
    print("ok  uniform == mean-kernel conv")


def test_uniform_has_no_router_params():
    m = DynamicConv2d(16, 24, 3, padding=1, num_kernels=4, router="uniform")
    assert sum(p.numel() for p in m.router.parameters()) == 0
    m2 = DynamicConv2d(16, 24, 3, padding=1, num_kernels=4, router="frozen_random")
    assert sum(p.numel() for p in m2.router.parameters() if p.requires_grad) == 0
    print("ok  uniform and frozen_random contribute no trainable router params")


def test_shuffle_active_in_eval():
    """The shuffled router must keep permuting at eval, or it silently
    degenerates into plain GAP and the control is void."""
    torch.manual_seed(0)
    x = torch.randn(32, 16, 8, 8)
    m = DynamicConv2d(16, 24, 3, padding=1, num_kernels=4, router="shuffled").eval()
    g = DynamicConv2d(16, 24, 3, padding=1, num_kernels=4, router="gap").eval()
    _sync(m, g)
    with torch.no_grad():
        assert not torch.allclose(m(x), g(x), atol=1e-6)
    print("ok  shuffled router still permutes in eval mode")


def test_all_routers_backprop():
    x = torch.randn(4, 16, 8, 8)
    for name in ROUTERS:
        m = DynamicConv2d(16, 24, 3, padding=1, num_kernels=4, router=name)
        m(x).sum().backward()
        assert m.weight.grad is not None and torch.isfinite(m.weight.grad).all(), name
    print(f"ok  gradients flow for all {len(ROUTERS)} routers")


# --- added for stage 2 -------------------------------------------------

def test_dataset_specs():
    from src.data import SPECS, num_classes
    assert num_classes("cifar100") == 100
    assert num_classes("tinyimagenet") == 200
    assert SPECS["cifar100"]["size"] == 32 and SPECS["cifar100"]["pad"] == 4
    assert SPECS["tinyimagenet"]["size"] == 64 and SPECS["tinyimagenet"]["pad"] == 8
    print("ok  dataset specs: 100/200 classes, 32/64 px, pad 4/8")


def test_fake_loaders_both_datasets():
    from src.data import build_loaders
    for ds, size, nc in [("cifar100", 32, 100), ("tinyimagenet", 64, 200)]:
        tr, te = build_loaders(fake=True, batch_size=8, dataset=ds)
        x, y = next(iter(tr))
        assert x.shape[1:] == (3, size, size), ds
        assert int(y.max()) < nc, ds
    print("ok  fake loaders produce correct shapes for both datasets")


def test_unknown_dataset_rejected():
    from src.data import build_loaders
    try:
        build_loaders(fake=True, dataset="imagenet21k")
    except KeyError:
        print("ok  unknown dataset name is rejected")
        return
    raise AssertionError("expected KeyError for unknown dataset")


def test_model_head_matches_dataset():
    import torch
    from src.resnet import ResNet18
    from src.data import num_classes
    for ds in ["cifar100", "tinyimagenet"]:
        m = ResNet18(num_classes(ds), scope="all", width=16)
        size = 32 if ds == "cifar100" else 64
        out = m(torch.randn(2, 3, size, size))
        assert out.shape == (2, num_classes(ds)), ds
    print("ok  classifier head width matches each dataset")


if __name__ == "__main__":
    test_blend_equals_combine()
    test_uniform_equals_mean_kernel()
    test_uniform_has_no_router_params()
    test_shuffle_active_in_eval()
    test_all_routers_backprop()
    test_dataset_specs()
    test_fake_loaders_both_datasets()
    test_unknown_dataset_rejected()
    test_model_head_matches_dataset()
    print("\nall tests passed")

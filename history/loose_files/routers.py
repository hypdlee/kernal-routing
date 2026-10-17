"""
Routing signals for dynamic convolution.

Every router maps a feature map (B, C, H, W) to K logits (B, K). The
DynamicConv2d layer softmaxes those into kernel-mixing weights. Routers
never see the kernel bank and never apply the softmax themselves.

The point of Stage 1 is that swapping the routing signal is a config
string, not a code fork. If you find yourself copying a class to make a
variant, add a flag to an existing one instead.

Stage 1 variants (capacity / adaptivity controls):
    uniform         constant logits, so z = 1/K. No routing at all.
    gap             trained GAP -> MLP. The standard baseline.
    shuffled        trained GAP router fed a permuted batch.
    frozen_random   input-dependent, never trained.

Stage 3 variants (dimension-matched comparison, not run in Stage 1):
    gap_std_max     SRM / CBAM-style richer statistics, 3C dims.
    gap_proj20      GAP compressed to 20 dims by a trained linear map.
    random_proj20   GAP compressed to 20 dims by a frozen random map.
    dct_topk        FcaNet-style low-frequency 2D DCT coefficients.

Adding LTD later means one class here plus one registry line. Nothing
else in the codebase should need to know it exists.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Every frozen-random component in the codebase draws from this seed, so
# "the random router" is the same random router in every run and every
# variant. Do not derive it from the run seed.
FROZEN_SEED = 20260908


class Router(nn.Module):
    """
    Contract:
      forward(x) -> (B, K) logits.
      is_constant  True when the logits do not depend on x. The conv layer
                   uses this to aggregate the kernel bank once instead of
                   once per sample. Speed only; the maths is identical.
      signal_dim   width of the descriptor handed to the MLP, recorded so
                   dimension-matched comparisons stay auditable.
    """

    is_constant: bool = False

    def __init__(self, in_channels: int, num_kernels: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_kernels = num_kernels
        self.signal_dim = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


def _mlp(in_dim: int, out_dim: int, hidden_ratio: float = 0.25, min_hidden: int = 16) -> nn.Sequential:
    hidden = max(min_hidden, int(in_dim * hidden_ratio))
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


def _gap(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------


class UniformRouter(Router):
    """
    The capacity control.

    Identical logits for every input, so softmax gives exactly 1/K and the
    K kernels collapse to their unweighted mean. All K kernels still exist
    and still receive gradient through that mean, so parameter count is
    unchanged. What is removed is routing.

    If this matches the trained GAP router, then what dynamic convolution
    buys is capacity, and the input-adaptive story needs revisiting.
    """

    is_constant = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[0], self.num_kernels)


class GAPRouter(Router):
    """The standard baseline: GAP -> MLP."""

    def __init__(self, in_channels: int, num_kernels: int):
        super().__init__(in_channels, num_kernels)
        self.signal_dim = in_channels
        self.mlp = _mlp(in_channels, num_kernels)

    def descriptor(self, x: torch.Tensor) -> torch.Tensor:
        return _gap(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.descriptor(x))


class ShuffledGAPRouter(GAPRouter):
    """
    Adaptivity broken, distribution preserved.

    Sample i is routed using sample perm(i)'s descriptor. The marginal
    distribution of z matches the trained GAP router; only the pairing
    between image and kernel is destroyed.

    The permutation stays on in eval mode. Without that, the variant
    silently becomes plain GAP at test time and the comparison means
    nothing.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.descriptor(x)
        b = d.shape[0]
        if b > 1:
            perm = torch.randperm(b, device=d.device)
            if torch.equal(perm, torch.arange(b, device=d.device)):
                perm = torch.roll(perm, 1)
            d = d[perm]
        return self.mlp(d)


class FrozenRandomRouter(Router):
    """
    Input-dependent, never trained.

    Same shape as GAPRouter but every weight is a buffer frozen at
    initialisation, so the optimiser never sees it. z varies with the
    input and was never optimised to be useful.
    """

    def __init__(self, in_channels: int, num_kernels: int):
        super().__init__(in_channels, num_kernels)
        self.signal_dim = in_channels
        g = torch.Generator().manual_seed(FROZEN_SEED)
        hidden = max(16, in_channels // 4)
        self.register_buffer("w1", torch.randn(in_channels, hidden, generator=g) / math.sqrt(in_channels))
        self.register_buffer("b1", torch.zeros(hidden))
        self.register_buffer("w2", torch.randn(hidden, num_kernels, generator=g) / math.sqrt(hidden))
        self.register_buffer("b2", torch.zeros(num_kernels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(_gap(x) @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


# --------------------------------------------------------------------------
# Stage 3 (registered now so the abstraction is exercised by more than
# the four cases it was designed around; not run by run_stage1.py)
# --------------------------------------------------------------------------


class GAPStdMaxRouter(GAPRouter):
    """SRM / CBAM-style: mean, standard deviation and max per channel."""

    def __init__(self, in_channels: int, num_kernels: int):
        Router.__init__(self, in_channels, num_kernels)
        self.signal_dim = 3 * in_channels
        self.mlp = _mlp(3 * in_channels, num_kernels)

    def descriptor(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        return torch.cat([flat.mean(-1), flat.std(-1), flat.amax(-1)], dim=1)


class GAPProj20Router(GAPRouter):
    """GAP compressed to 20 dims by a trained linear map. The dimension-
    matched partner for any 20-dim descriptor such as LTD."""

    def __init__(self, in_channels: int, num_kernels: int, proj_dim: int = 20):
        Router.__init__(self, in_channels, num_kernels)
        self.signal_dim = proj_dim
        self.proj = nn.Linear(in_channels, proj_dim, bias=False)
        self.mlp = _mlp(proj_dim, num_kernels)

    def descriptor(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(_gap(x))


class RandomProj20Router(GAPRouter):
    """GAP compressed to 20 dims by a frozen random map, then a trained
    MLP. Isolates 'is 20 dims enough' from 'is topology informative'."""

    def __init__(self, in_channels: int, num_kernels: int, proj_dim: int = 20):
        Router.__init__(self, in_channels, num_kernels)
        self.signal_dim = proj_dim
        g = torch.Generator().manual_seed(FROZEN_SEED)
        self.register_buffer("proj", torch.randn(in_channels, proj_dim, generator=g) / math.sqrt(in_channels))
        self.mlp = _mlp(proj_dim, num_kernels)

    def descriptor(self, x: torch.Tensor) -> torch.Tensor:
        return _gap(x) @ self.proj


class DCTTopKRouter(GAPRouter):
    """
    FcaNet-style multi-spectral squeeze.

    Channels are split into groups; each group is reduced by a different
    low-frequency 2D DCT basis instead of by the mean. The (0, 0) basis
    is GAP up to a constant, so this strictly generalises GAPRouter.

    The basis is rebuilt whenever the spatial size changes and cached, so
    a layer that always sees 8x8 pays for it once.
    """

    # (u, v) ordered by u + v, the standard low-frequency-first ordering
    FREQS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1),
             (2, 2), (0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]

    def __init__(self, in_channels: int, num_kernels: int, num_freq: int = 4):
        Router.__init__(self, in_channels, num_kernels)
        self.signal_dim = in_channels
        self.num_freq = min(num_freq, len(self.FREQS))
        self.mlp = _mlp(in_channels, num_kernels)
        self._cache_hw: tuple[int, int] | None = None

    def _build_basis(self, h: int, w: int, device, dtype) -> torch.Tensor:
        # (num_freq, h, w) separable DCT-II basis, unnormalised
        ii = torch.arange(h, device=device, dtype=torch.float32)
        jj = torch.arange(w, device=device, dtype=torch.float32)
        planes = []
        for u, v in self.FREQS[: self.num_freq]:
            bu = torch.cos(math.pi * u * (ii + 0.5) / h)
            bv = torch.cos(math.pi * v * (jj + 0.5) / w)
            planes.append(torch.outer(bu, bv))
        return torch.stack(planes).to(dtype)

    def descriptor(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self._cache_hw != (h, w) or getattr(self, "basis", None) is None:
            self.register_buffer("basis", self._build_basis(h, w, x.device, x.dtype), persistent=False)
            self._cache_hw = (h, w)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        groups = torch.chunk(x, self.num_freq, dim=1)
        out = [(g * basis[i]).flatten(2).mean(-1) for i, g in enumerate(groups)]
        return torch.cat(out, dim=1)


ROUTERS: dict[str, type[Router]] = {
    "uniform": UniformRouter,
    "gap": GAPRouter,
    "shuffled": ShuffledGAPRouter,
    "frozen_random": FrozenRandomRouter,
    "gap_std_max": GAPStdMaxRouter,
    "gap_proj20": GAPProj20Router,
    "random_proj20": RandomProj20Router,
    "dct_topk": DCTTopKRouter,
}

# The four that Stage 1 actually runs, in reporting order.
STAGE1 = ["uniform", "gap", "shuffled", "frozen_random"]


def build_router(name: str, in_channels: int, num_kernels: int) -> Router:
    if name not in ROUTERS:
        raise KeyError(f"unknown router '{name}'. available: {sorted(ROUTERS)}")
    return ROUTERS[name](in_channels, num_kernels)

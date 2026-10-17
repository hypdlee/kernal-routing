"""
Dynamic convolution with a pluggable routing signal.

W_dynamic(x) = sum_k z_k(x) * W_k,  z = softmax(router(x) / temperature)

Three execution paths, all mathematically identical, because convolution
is linear in its weights:

    conv(x, sum_k z_k W_k) == sum_k z_k conv(x, W_k)

  constant router  z is the same for every sample, so aggregate the bank
                   once into one (out, in, kh, kw) kernel and run an
                   ordinary conv2d. This is what makes the uniform control
                   cheap enough to be a free baseline.

  blend            run K ordinary convs and mix the outputs. Costs K times
                   the FLOPs but only K times the activation memory. This
                   is the default, because it is what fits on a 12 GB card
                   at batch 128 with scope=all.

  combine          aggregate per sample, then one grouped conv with the
                   batch folded into the groups. This is the CondConv
                   formulation and it is faster per step, but it holds
                   B * out * in * kh * kw floats for the aggregated
                   weights. Layer 4 of a ResNet-18 at batch 128 is roughly
                   1.2 GB per conv before activations, so reach for it only
                   on a big card or a small batch.

Equality of the two is asserted in tests/test_paths.py. If you change one,
that test is what tells you the other drifted.

Routing statistics are recorded on the layer during eval so that
train.py can dump collapse evidence without threading a logger through
the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routers import build_router


class DynamicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        num_kernels: int = 4,
        router: str = "gap",
        temperature: float = 1.0,
        impl: str = "blend",
    ):
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels
        self.temperature = temperature
        self.router_name = router
        if impl not in ("blend", "combine"):
            raise ValueError("impl must be 'blend' or 'combine'")
        self.impl = impl

        self.weight = nn.Parameter(
            torch.empty(num_kernels, out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(num_kernels, out_channels)) if bias else None
        self.reset_parameters()

        self.router = build_router(router, in_channels, num_kernels)

        # Filled during eval by _record. Consumed and cleared by train.py.
        self.last_z_mean: torch.Tensor | None = None
        self.last_z_entropy: float | None = None
        self.last_argmax_counts: torch.Tensor | None = None
        self._collect = False

    def reset_parameters(self) -> None:
        # Initialise each of the K kernels the way a single conv would be,
        # so the bank is not systematically smaller or larger than the
        # static baseline it is compared against.
        for k in range(self.num_kernels):
            nn.init.kaiming_normal_(self.weight[k], mode="fan_out", nonlinearity="relu")

    def collect_stats(self, enabled: bool = True) -> None:
        self._collect = enabled

    def _record(self, z: torch.Tensor) -> None:
        if not self._collect:
            return
        p = z.detach().float().clamp_min(1e-12)
        ent = -(p * p.log()).sum(1).mean() / math.log(self.num_kernels)
        self.last_z_mean = p.mean(0).cpu()
        self.last_z_entropy = float(ent.cpu())
        self.last_argmax_counts = torch.bincount(p.argmax(1), minlength=self.num_kernels).cpu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        logits = self.router(x)
        z = F.softmax(logits / self.temperature, dim=1)
        self._record(z)

        if self.router.is_constant:
            # Every row of z is identical; aggregate once.
            w = torch.einsum("k,koihw->oihw", z[0].to(self.weight.dtype), self.weight)
            bias = None if self.bias is None else (z[0].to(self.bias.dtype) @ self.bias)
            return F.conv2d(x, w, bias, self.stride, self.padding, self.dilation, self.groups)

        if self.impl == "blend":
            # sum_k z_k * conv(x, W_k). K full convolutions, no per-sample
            # weight tensor. Equivalent to the combine path exactly.
            outs = []
            for k in range(self.num_kernels):
                bk = None if self.bias is None else self.bias[k]
                outs.append(F.conv2d(x, self.weight[k], bk, self.stride,
                                     self.padding, self.dilation, self.groups))
            stacked = torch.stack(outs, dim=1)                 # (B, K, O, H, W)
            zz = z.to(stacked.dtype).view(b, self.num_kernels, 1, 1, 1)
            return (stacked * zz).sum(1)

        w = torch.einsum("bk,koihw->boihw", z.to(self.weight.dtype), self.weight)
        w = w.reshape(b * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        bias = None if self.bias is None else (z.to(self.bias.dtype) @ self.bias).reshape(-1)

        out = F.conv2d(
            x.reshape(1, b * self.in_channels, *x.shape[2:]),
            w,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            groups=self.groups * b,
        )
        return out.reshape(b, self.out_channels, *out.shape[2:])

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, k={self.kernel_size}, "
            f"stride={self.stride}, K={self.num_kernels}, router={self.router_name}, "
            f"T={self.temperature}"
        )

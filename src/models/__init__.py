"""
Backbone registry for the cross-architecture replication.

Every model here exposes the same helpers the ResNet-18 study relied on
(dynamic_layers, collect_stats, routing_report) and is built with

    build_model(name, num_classes, scope, num_kernels, router, temperature,
                width, placement)

Width
    `width` keeps the ResNet-18 convention: 64 is the standard network, 32
    and 16 are uniform slimmings. For every other backbone it is converted to
    a channel multiplier width/64, so width 64 means the published channel
    counts and width 16 means one quarter of them.

Scope
    "all"  dynamic layers placed according to `placement`
    "none" plain network, no dynamic layer anywhere
    ("stage34" exists only for ResNet-18, as before.)

Placement
    "spatial" (default, the ResNet-18 rule applied literally): every
        convolution with a k x k kernel, k > 1, that sits inside a block
        becomes dynamic. Stems, 1 x 1 convolutions, shortcut projections,
        downsampling layers between stages and classifiers stay static.
    "all": every convolution except the stem becomes dynamic (the rule
        DY-Conv states for its ResNets). Not used by the planned runs;
        provided so the alternative can be tested without new code.

Initialisation
    ResNet-18 is built by its original class, unchanged, so the existing
    132 runs stay reproducible. That class leaves static convolutions at
    PyTorch's default initialisation while each dynamic kernel is
    Kaiming-normal (fan_out); see PREREGISTRATION_MODELS.md.
    The four new backbones apply ONE scheme, their standard one, to static
    convolutions and to every kernel in every bank alike, so static and
    dynamic arms start from the same distribution.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dynconv import DynamicConv2d
from ..resnet import ResNet18

PLACEMENTS = ("spatial", "all")


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------

class Placement:
    """Decides, per convolution role, whether it is dynamic.

    Roles: stem, block_spatial, block_pointwise, shortcut, downsample, head.
    """

    def __init__(self, scope: str, placement: str):
        if scope not in ("all", "none"):
            raise ValueError(f"scope must be 'all' or 'none' for this model, got {scope!r}")
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
        self.scope, self.placement = scope, placement

    def dynamic(self, role: str) -> bool:
        if self.scope == "none" or role in ("stem", "head"):
            return False
        if self.placement == "spatial":
            return role == "block_spatial"
        return True


class ConvFactory:
    def __init__(self, place: Placement, num_kernels: int, router: str, temperature: float):
        self.place, self.K, self.router, self.T = place, num_kernels, router, temperature

    def __call__(self, role, cin, cout, k, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = k // 2
        if self.place.dynamic(role):
            return DynamicConv2d(cin, cout, k, stride=stride, padding=padding, groups=groups,
                                 bias=bias, num_kernels=self.K, router=self.router,
                                 temperature=self.T)
        return nn.Conv2d(cin, cout, k, stride=stride, padding=padding, groups=groups, bias=bias)


class DynNet(nn.Module):
    """Helpers identical in behaviour to the ones on ResNet18."""

    def dynamic_layers(self):
        return [m for m in self.modules() if isinstance(m, DynamicConv2d)]

    def collect_stats(self, enabled=True):
        for m in self.dynamic_layers():
            m.collect_stats(enabled)

    def routing_report(self):
        out = []
        for i, m in enumerate(self.dynamic_layers()):
            if m.last_z_entropy is None:
                continue
            out.append({
                "layer": i,
                "router": m.router_name,
                "z_mean": [round(v, 5) for v in m.last_z_mean.tolist()],
                "z_entropy_norm": round(m.last_z_entropy, 5),
                "argmax_counts": m.last_argmax_counts.tolist(),
            })
        return out

    def init_conv_weights(self, fn):
        """Apply `fn(tensor)` to every static conv weight and to every kernel
        of every dynamic bank, so both arms share one initialisation."""
        for m in self.modules():
            if isinstance(m, DynamicConv2d):
                for k in range(m.num_kernels):
                    fn(m.weight.data[k])
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fn(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def _kaiming_fan_out(t):
    nn.init.kaiming_normal_(t, mode="fan_out", nonlinearity="relu")


def _trunc02(t):
    nn.init.trunc_normal_(t, std=0.02)


def _norm_init(model, zero_attr=None):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, LayerNorm2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    if zero_attr:
        for m in model.modules():
            bn = getattr(m, zero_attr, None)
            if isinstance(bn, nn.BatchNorm2d) and hasattr(m, "is_residual_block"):
                nn.init.zeros_(bn.weight)


# --------------------------------------------------------------------------
# ResNet-50 (CIFAR stem, v1.5 bottleneck: stride on the 3x3)
# --------------------------------------------------------------------------

class Bottleneck(nn.Module):
    expansion = 4
    is_residual_block = True

    def __init__(self, conv, cin, planes, stride):
        super().__init__()
        out = planes * self.expansion
        self.conv1 = conv("block_pointwise", cin, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv("block_spatial", planes, planes, 3, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv("block_pointwise", planes, out, 1)
        self.bn3 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != out:
            self.downsample = nn.Sequential(conv("shortcut", cin, out, 1, stride=stride),
                                            nn.BatchNorm2d(out))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + idt)


class ResNet50(DynNet):
    def __init__(self, num_classes, conv, width=64):
        super().__init__()
        c = width
        self.stem = nn.Sequential(conv("stem", 3, c, 3), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        layers, cin = [], c
        for i, (planes, blocks, stride) in enumerate([(c, 3, 1), (2 * c, 4, 2),
                                                      (4 * c, 6, 2), (8 * c, 3, 2)]):
            for b in range(blocks):
                layers.append(Bottleneck(conv, cin, planes, stride if b == 0 else 1))
                cin = planes * Bottleneck.expansion
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)
        self.init_conv_weights(_kaiming_fan_out)
        _norm_init(self, zero_attr="bn3")

    def forward(self, x):
        return self.fc(self.pool(self.layers(self.stem(x))).flatten(1))


# --------------------------------------------------------------------------
# VGG-16 with batch norm (CIFAR form: 5 pools, global pool, one linear)
# --------------------------------------------------------------------------

VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG16(DynNet):
    def __init__(self, num_classes, conv, width=64):
        super().__init__()
        m, feats, cin, first = width / 64.0, [], 3, True
        for v in VGG16_CFG:
            if v == "M":
                feats.append(nn.MaxPool2d(2, 2))
                continue
            cout = max(1, int(round(v * m)))
            feats += [conv("stem" if first else "block_spatial", cin, cout, 3),
                      nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            cin, first = cout, False
        self.features = nn.Sequential(*feats)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)
        self.init_conv_weights(_kaiming_fan_out)
        _norm_init(self)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


# --------------------------------------------------------------------------
# MobileNetV2 (CIFAR form: stride-1 stem, second stage stride 1)
# --------------------------------------------------------------------------

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, conv, cin, cout, stride, t):
        super().__init__()
        hidden = cin * t
        self.use_res = stride == 1 and cin == cout
        layers = []
        if t != 1:
            layers += [conv("block_pointwise", cin, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [conv("block_spatial", hidden, hidden, 3, stride=stride, groups=hidden),
                   nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
                   conv("block_pointwise", hidden, cout, 1), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


MBV2_CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


class MobileNetV2(DynNet):
    def __init__(self, num_classes, conv, width=64):
        super().__init__()
        m = width / 64.0
        cin = make_divisible(32 * m)
        last = make_divisible(1280 * max(1.0, m))
        self.stem = nn.Sequential(conv("stem", 3, cin, 3), nn.BatchNorm2d(cin), nn.ReLU6(inplace=True))
        blocks = []
        for t, c, n, s in MBV2_CFG:
            cout = make_divisible(c * m)
            for i in range(n):
                blocks.append(InvertedResidual(conv, cin, cout, s if i == 0 else 1, t))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(conv("head", cin, last, 1), nn.BatchNorm2d(last), nn.ReLU6(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last, num_classes)
        self.init_conv_weights(_kaiming_fan_out)
        _norm_init(self)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(self.pool(self.head(self.blocks(self.stem(x)))).flatten(1))


# --------------------------------------------------------------------------
# ConvNeXt-Tiny (CIFAR form: patch size 1 so the final map is 4x4 at 32 px)
# --------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, conv, dim, layer_scale=1e-6):
        super().__init__()
        self.dw = conv("block_spatial", dim, dim, 7, groups=dim, bias=True)
        self.norm = LayerNorm2d(dim)
        self.pw1 = conv("block_pointwise", dim, 4 * dim, 1, bias=True)
        self.act = nn.GELU()
        self.pw2 = conv("block_pointwise", 4 * dim, dim, 1, bias=True)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):
        y = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + self.gamma[:, None, None] * y


class ConvNeXtTiny(DynNet):
    DEPTHS, DIMS = (3, 3, 9, 3), (96, 192, 384, 768)

    def __init__(self, num_classes, conv, width=64):
        super().__init__()
        m = width / 64.0
        dims = [max(8, int(round(d * m))) for d in self.DIMS]
        self.stem = nn.Sequential(conv("stem", 3, dims[0], 1, bias=True), LayerNorm2d(dims[0]))
        stages = []
        for i, (d, n) in enumerate(zip(dims, self.DEPTHS)):
            if i > 0:
                stages.append(nn.Sequential(LayerNorm2d(dims[i - 1]),
                                            conv("downsample", dims[i - 1], d, 2, stride=2,
                                                 padding=0, bias=True)))
            stages += [ConvNeXtBlock(conv, d) for _ in range(n)]
        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.fc = nn.Linear(dims[-1], num_classes)
        self.init_conv_weights(_trunc02)
        _norm_init(self)
        _trunc02(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.fc(self.norm(x.mean((2, 3))))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def _build_resnet18(num_classes, scope, num_kernels, router, temperature, width, placement):
    if placement != "spatial":
        raise ValueError("resnet18 is built by its original class, which only supports the "
                         "'spatial' placement used in the existing 132 runs")
    return ResNet18(num_classes, scope=scope, num_kernels=num_kernels, router=router,
                    temperature=temperature, width=width)


def _factory(cls):
    def build(num_classes, scope, num_kernels, router, temperature, width, placement):
        conv = ConvFactory(Placement(scope, placement), num_kernels, router, temperature)
        return cls(num_classes, conv, width=width)
    return build


MODELS = {
    "resnet18": _build_resnet18,
    "resnet50": _factory(ResNet50),
    "vgg16": _factory(VGG16),
    "mobilenetv2": _factory(MobileNetV2),
    "convnext_tiny": _factory(ConvNeXtTiny),
}

# Order the replication runs in (cheapest and most informative first).
RUN_ORDER = ["mobilenetv2", "vgg16", "resnet50", "convnext_tiny"]


def build_model(name, num_classes=100, scope="all", num_kernels=4, router="gap",
                temperature=1.0, width=64, placement="spatial"):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](num_classes, scope, num_kernels, router, temperature, width, placement)


def dynamic_layers(model):
    return [m for m in model.modules() if isinstance(m, DynamicConv2d)]


def collapse_to_static(model):
    """Deep copy in which every DynamicConv2d is replaced by an ordinary
    convolution holding the mean of its bank. For the uniform arm this is
    exactly Proposition 1; the tests use it to check that claim per model."""
    m = copy.deepcopy(model)

    def swap(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, DynamicConv2d):
                conv = nn.Conv2d(child.in_channels, child.out_channels, child.kernel_size,
                                 stride=child.stride, padding=child.padding,
                                 dilation=child.dilation, groups=child.groups,
                                 bias=child.bias is not None)
                conv = conv.to(child.weight.dtype)
                conv.weight.data.copy_(child.weight.data.mean(0))
                if child.bias is not None:
                    conv.bias.data.copy_(child.bias.data.mean(0))
                setattr(parent, name, conv)
            else:
                swap(child)
    swap(m)
    return m

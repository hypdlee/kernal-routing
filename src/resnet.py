"""
CIFAR-style ResNet-18 with dynamic convolutions.

Standard CIFAR modifications: 3x3 stem, stride 1, no max-pool.

The stem and the 1x1 downsample shortcuts stay static. Replacing the stem
would change the input representation itself, which confounds a study
about routing, and the 1x1 shortcuts carry too little computation to be
worth the memory of a per-sample kernel.

Scopes:
    all       every 3x3 conv inside every BasicBlock
    stage34   3x3 convs in the last two stages only (cheapest, and where
              dynamic convolution papers usually report the largest gains)
    none      plain ResNet-18, no dynamic convolution anywhere
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dynconv import DynamicConv2d

SCOPES = ("all", "stage34", "none")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dynamic=False, num_kernels=4,
                 router="gap", temperature=1.0):
        super().__init__()

        def conv3x3(cin, cout, s):
            if dynamic:
                return DynamicConv2d(cin, cout, 3, stride=s, padding=1, bias=False,
                                     num_kernels=num_kernels, router=router,
                                     temperature=temperature)
            return nn.Conv2d(cin, cout, 3, stride=s, padding=1, bias=False)

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes=100, scope="all", num_kernels=4, router="gap",
                 temperature=1.0, width=64):
        super().__init__()
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        self.scope = scope
        self.in_planes = width

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        cfg = [(width, 1), (width * 2, 2), (width * 4, 2), (width * 8, 2)]
        stages = []
        for idx, (planes, stride) in enumerate(cfg, start=1):
            dyn = scope == "all" or (scope == "stage34" and idx >= 3)
            stages.append(self._make_stage(planes, 2, stride, dyn, num_kernels, router, temperature))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        # Zero-init the last BN in each block so residual branches start as
        # identity. Standard, and it keeps the four variants comparable at
        # step zero.
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0.0)

    def _make_stage(self, planes, blocks, stride, dynamic, num_kernels, router, temperature):
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride if i == 0 else 1,
                                     dynamic, num_kernels, router, temperature))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.pool(x).flatten(1))

    # -- helpers used by train.py -------------------------------------

    def dynamic_layers(self):
        return [m for m in self.modules() if isinstance(m, DynamicConv2d)]

    def collect_stats(self, enabled=True):
        for m in self.dynamic_layers():
            m.collect_stats(enabled)

    def routing_report(self):
        """Per-layer routing collapse evidence, gathered after an eval pass."""
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


def param_counts(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    router = sum(p.numel() for m in model.modules() if isinstance(m, DynamicConv2d)
                 for p in m.router.parameters())
    return {"total": total, "trainable": trainable, "router": router}

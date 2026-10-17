"""
Dataset loaders. Two datasets, one code path, so the two are comparable.

CIFAR-100      32x32, 100 classes, 50k train / 10k test. torchvision.
Tiny-ImageNet  64x64, 200 classes, 100k train / 10k val. Manual download,
               see download_tinyimagenet.py.

The stem is NOT changed between datasets. Keeping a 3x3 stride-1 stem at
64x64 leaves the last feature map at 8x8 instead of 4x4, which costs
roughly 4x the compute, but changing the stem would change the network
between datasets and the whole point is that they are comparable.

Crop padding scales with resolution (4 at 32px, 8 at 64px), which is the
standard convention for each.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

SPECS = {
    "cifar100": dict(
        num_classes=100, size=32, pad=4,
        mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762),
    ),
    "tinyimagenet": dict(
        num_classes=200, size=64, pad=8,
        mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821),
    ),
}


def num_classes(dataset: str) -> int:
    return SPECS[dataset]["num_classes"]


class FakeSet(Dataset):
    """Random tensors of the right shape, for smoke tests without a
    download. Accuracy on this is meaningless by construction."""

    def __init__(self, n, size, classes):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, size, size, generator=g)
        self.y = torch.randint(0, classes, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _transforms(spec):
    train = transforms.Compose([
        transforms.RandomCrop(spec["size"], padding=spec["pad"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])
    return train, test


def build_loaders(root="./data", batch_size=128, workers=4, fake=False,
                  download=True, dataset="cifar100", eval_batch=None):
    if dataset not in SPECS:
        raise KeyError(f"unknown dataset '{dataset}'. available: {sorted(SPECS)}")
    spec = SPECS[dataset]

    if fake:
        train_ds = FakeSet(512, spec["size"], spec["num_classes"])
        test_ds = FakeSet(256, spec["size"], spec["num_classes"])
        workers = 0
    else:
        train_tf, test_tf = _transforms(spec)
        if dataset == "cifar100":
            train_ds = datasets.CIFAR100(root, train=True, download=download, transform=train_tf)
            test_ds = datasets.CIFAR100(root, train=False, download=download, transform=test_tf)
        else:
            base = Path(root) / "tiny-imagenet-200"
            tr, va = base / "train", base / "val"
            if not tr.is_dir():
                raise FileNotFoundError(
                    f"{tr} not found. Run:  python download_tinyimagenet.py --root {root}")
            # Structural check, not a check for one specific class id: val
            # is ready when the flat images/ folder is gone and per-class
            # folders exist.
            if (va / "images").is_dir() or not any(p.is_dir() for p in va.iterdir()):
                raise FileNotFoundError(
                    f"{va} is not restructured into class folders. Run:  "
                    f"python download_tinyimagenet.py --root {root}")
            train_ds = datasets.ImageFolder(str(tr), transform=train_tf)
            test_ds = datasets.ImageFolder(str(va), transform=test_tf)
            # ImageFolder sorts class names, and both splits contain the same
            # 200 WordNet ids, so the label mappings agree. Assert it rather
            # than trust it: a silent mismatch would look like a bad model.
            if train_ds.classes != test_ds.classes:
                raise RuntimeError("train and val class orderings differ; "
                                   "re-run download_tinyimagenet.py")
            if len(train_ds.classes) != spec["num_classes"]:
                raise RuntimeError(f"expected {spec['num_classes']} classes, "
                                   f"found {len(train_ds.classes)}")

    common = dict(num_workers=workers, pin_memory=not fake,
                  persistent_workers=workers > 0)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, **common)
    # Evaluation batch size does not change any number (BatchNorm in eval
    # mode, per-sample routing); it only sets memory. Default as before.
    test_ld = DataLoader(test_ds, batch_size=eval_batch or max(batch_size, 256),
                         shuffle=False, **common)
    return train_ld, test_ld

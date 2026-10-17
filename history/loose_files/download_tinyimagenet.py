"""
Fetch and prepare Tiny-ImageNet-200.

The archive ships in a layout torchvision's ImageFolder cannot read:

    tiny-imagenet-200/
      train/n01443537/images/*.JPEG        <- images one level too deep
      val/images/*.JPEG                    <- all 10k in one folder
      val/val_annotations.txt              <- labels live here instead
      wnids.txt, words.txt

This script flattens train/<wnid>/images/ up one level and rebuilds val/
into per-class folders using val_annotations.txt, after which both splits
are plain ImageFolder trees with the same 200 class names.

    python download_tinyimagenet.py --root ./data

Idempotent: re-running on a prepared directory verifies and exits. If the
download is blocked on your network, fetch the zip by hand and pass
--zip path/to/tiny-imagenet-200.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
N_CLASSES, N_TRAIN, N_VAL = 200, 100_000, 10_000


def _progress(done, block, total):
    if total <= 0:
        return
    pct = min(100.0, done * block * 100.0 / total)
    sys.stdout.write(f"\r  downloading... {pct:5.1f}%")
    sys.stdout.flush()


def download(dest: Path) -> Path:
    zp = dest / "tiny-imagenet-200.zip"
    if zp.exists():
        print(f"  archive already present: {zp} ({zp.stat().st_size/1e6:.0f} MB)")
        return zp
    print(f"  fetching {URL}")
    print("  (about 240 MB; if your network blocks this, download it manually")
    print("   and re-run with --zip <path>)")
    try:
        urllib.request.urlretrieve(URL, zp, _progress)
        print()
    except Exception as e:
        zp.unlink(missing_ok=True)
        raise SystemExit(f"\ndownload failed: {e}\n"
                         f"Fetch {URL} by hand, then re-run with --zip <path>.")
    return zp


def flatten_train(train: Path) -> int:
    """train/<wnid>/images/*.JPEG -> train/<wnid>/*.JPEG"""
    moved = 0
    for wnid in sorted(p for p in train.iterdir() if p.is_dir()):
        imgs = wnid / "images"
        if not imgs.is_dir():
            continue
        for f in imgs.iterdir():
            if f.is_file():
                f.rename(wnid / f.name)
                moved += 1
        imgs.rmdir()
        # The per-class *_boxes.txt is bounding-box metadata ImageFolder
        # would try to open as an image. Remove it.
        for extra in wnid.glob("*_boxes.txt"):
            extra.unlink()
    return moved


def restructure_val(val: Path) -> int:
    """val/images/*.JPEG + val_annotations.txt -> val/<wnid>/*.JPEG"""
    ann = val / "val_annotations.txt"
    imgs = val / "images"
    if not ann.exists():
        raise SystemExit(f"missing {ann}")
    if not imgs.is_dir():
        raise SystemExit(f"missing {imgs}")

    mapping = {}
    for line in ann.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise SystemExit(f"malformed annotation line: {line!r}")
        mapping[parts[0]] = parts[1]

    moved = 0
    for fname, wnid in mapping.items():
        src = imgs / fname
        if not src.exists():
            continue
        out = val / wnid
        out.mkdir(exist_ok=True)
        src.rename(out / fname)
        moved += 1

    if any(imgs.iterdir()):
        leftover = len(list(imgs.iterdir()))
        raise SystemExit(f"{leftover} images in {imgs} had no annotation entry")
    imgs.rmdir()
    ann.unlink()
    return moved


def verify(base: Path) -> bool:
    tr, va = base / "train", base / "val"
    tc = sorted(p.name for p in tr.iterdir() if p.is_dir())
    vc = sorted(p.name for p in va.iterdir() if p.is_dir())
    ntr = sum(1 for _ in tr.rglob("*.JPEG"))
    nva = sum(1 for _ in va.rglob("*.JPEG"))
    ok = True
    print(f"  train classes {len(tc):>4}   images {ntr:>7}")
    print(f"  val   classes {len(vc):>4}   images {nva:>7}")
    for label, got, want in [("train classes", len(tc), N_CLASSES),
                             ("val classes", len(vc), N_CLASSES),
                             ("train images", ntr, N_TRAIN),
                             ("val images", nva, N_VAL)]:
        if got != want:
            print(f"  MISMATCH {label}: got {got}, expected {want}")
            ok = False
    if tc != vc:
        print("  MISMATCH: train and val class lists differ")
        ok = False
    if (tr / tc[0] / "images").exists() if tc else False:
        print("  MISMATCH: train still has nested images/ folders")
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--zip", default="", help="path to an already-downloaded zip")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    base = root / "tiny-imagenet-200"

    if base.is_dir() and (base / "val").is_dir() and not (base / "val" / "images").is_dir():
        print(f"Already prepared at {base}. Verifying:")
        raise SystemExit(0 if verify(base) else 1)

    if not base.is_dir():
        zp = Path(args.zip) if args.zip else download(root)
        if not zp.exists():
            raise SystemExit(f"no such file: {zp}")
        print(f"  extracting {zp} -> {root}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
        if not args.keep_zip and not args.zip:
            zp.unlink()

    print("  flattening train/")
    n = flatten_train(base / "train")
    print(f"    moved {n} files")

    if (base / "val" / "images").is_dir():
        print("  restructuring val/")
        n = restructure_val(base / "val")
        print(f"    moved {n} files")

    print("Verifying:")
    ok = verify(base)
    print("\n" + ("READY. Next: python run_tinyimagenet.py" if ok
                  else "PREPARATION FAILED. Do not train on this."))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

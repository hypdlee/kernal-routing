# KernelRoute Stage 1: does the routing signal matter?

> **Cross-architecture replication (30 September 2026).** The full 132-run
> program is now set up for MobileNetV2, VGG-16 (BN), ResNet-50 and
> ConvNeXt-Tiny. Start with `README_MODELS.md` (PowerShell) and
> `PREREGISTRATION_MODELS.md`. Everything below still applies to ResNet-18.

One experiment. Four routers, same backbone, same recipe, same seeds. If a
constant `uniform` router matches a trained GAP router, then what dynamic
convolution buys on this benchmark is capacity rather than adaptivity, and
that result is worth more than another operator with a 0.4pp delta.

Read `PREREGISTRATION.md` before running anything. The decision rule is
fixed in advance on purpose.

## Setup (WSL2 Ubuntu, RTX 4070 Ti)

This is PyTorch, not TensorFlow, so it runs natively on Windows too. WSL2
is still the smoother path if that is where your CUDA toolchain already
lives.

```bash
cd ~/kernelroute-stage1
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# torch first, matched to your CUDA build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

In VS Code: install the WSL extension, `code .` from inside WSL, then
Ctrl+Shift+P, "Python: Select Interpreter", pick `.venv/bin/python`.

CIFAR-100 downloads itself on the first run into `./data` (about 170 MB).

## Verify before you trust it

```bash
python tests/test_paths.py
```

Five assertions. The important ones are that the `blend` and `combine`
execution paths agree to 1e-10, and that the `shuffled` router still
permutes in eval mode. If that second one ever breaks, `shuffled` silently
degenerates into plain `gap` and the control is void without anything
failing loudly.

## Run it

```bash
# smoke test: no download, tiny, finishes in a couple of minutes
python run_stage1.py --routers uniform gap --seeds 0 --epochs 1 \
    --scope stage34 --fake-data --tag smoke --out results/smoke

# the real grid: 4 routers x 6 seeds x 150 epochs
python run_stage1.py

# analysis
python -m src.analyze --dir results --tag stage1 --baseline gap --margin 0.5
```

`run_stage1.py` skips any run whose JSON already exists, so Ctrl-C and
restart is safe. Nothing downstream parses stdout; `analyze.py` reads only
the JSON records.

## Runtime and memory

ResNet-18 with K=4 at `--scope all` is roughly four times the convolution
FLOPs of the static model. Expect something in the region of 2 to 4 minutes
per epoch on a 4070 Ti at batch 128, so a 150-epoch run lands around 5 to
10 hours and the full 24-run grid is a few days of wall clock. Measure one
run before committing to the grid.

If that is too long, in order of preference:

1. `--scope stage34` replaces only the last two stages. Roughly half the
   cost, and it is where dynamic-convolution papers usually report their
   largest gains anyway. State the choice in the paper.
2. `--epochs 100`. Absolute accuracy drops a point or two; the paired
   comparison is unaffected because every arm gets the same budget.
3. `--num-kernels 2`. Weakens the capacity confound, which cuts both ways.

Two implementations of the same mathematics, since convolution is linear in
its weights:

- `--impl blend` (default) runs K convolutions and mixes the outputs. K
  times the FLOPs, K times the activation memory. This is what fits in
  12 GB at batch 128.
- `--impl combine` aggregates per-sample kernels and runs one grouped
  convolution. Fewer FLOPs, but it holds `B x out x in x kh x kw` floats.
  Layer 4 at batch 128 is about 1.2 GB per conv before activations, so it
  will OOM on a 4070 Ti at `--scope all`. Use it at batch 32 or on a
  bigger card.

On OOM: drop `--batch-size` to 64 and halve `--lr` to 0.05, or switch to
`--scope stage34`. Change it for every arm, not just the one that crashed.

## What the code is

```
src/routers.py    the routing signals. Eight of them, one registry.
src/dynconv.py    DynamicConv2d, three execution paths, provably equal.
src/resnet.py     CIFAR ResNet-18 with a --scope switch.
src/data.py       CIFAR-100 loaders plus a fake-data mode for smoke tests.
src/train.py      one (router, seed) run -> one JSON record.
src/analyze.py    paired t-test, Wilcoxon, TOST equivalence, verdict.
run_stage1.py     the grid, resumable.
tests/            path-equivalence assertions.
```

The router abstraction is the part that matters for everything after Stage
1. A router takes a feature map and returns K logits. That is the entire
contract. Adding LTD means one class in `routers.py` and one registry line;
nothing else in the codebase needs to know it exists. Four more routers for
Stage 3 (`gap_std_max`, `gap_proj20`, `random_proj20`, `dct_topk`) are
already registered and tested, unused by Stage 1.

## Statistics, and why six seeds

The two-sided Wilcoxon signed-rank test has a hard floor on its attainable
p-value at small n:

```
n=3 -> 0.25    n=4 -> 0.125    n=5 -> 0.0625    n=6 -> 0.031
```

At three seeds you cannot reach p < 0.05 with Wilcoxon regardless of effect
size. `analyze.py` prints the floor next to the p-value so this never gets
misread.

More importantly, a large p on a difference test is not evidence of no
effect. Establishing a null needs an equivalence test, so `analyze.py` runs
TOST against the pre-registered margin and reports both. The verdict line
combines them.

## Reading the routing entropy

Every run ends with one eval pass that records, per dynamic layer, the mean
softmax weights, normalised entropy, and how many samples each branch won.
The statistics are overwritten at every test batch, so what is stored
describes the last batch of the unshuffled test set only: 16 images (10000
test images, eval batch 256), the same images in every run on a dataset.
The stored values are not test-set averages, and since the final weights
are not kept they cannot be recomputed.
Entropy near 1.0 means near-uniform mixing: the router is not choosing.
`uniform` sits at exactly 1.0 by construction and is the reference.

This is descriptive only. The entropy-of-mean statistic is foolable, which
is why per-sample entropy is recorded here and why no hypothesis test is
attached to it in the pre-registration.

## Known behaviour worth not being surprised by

`shuffled` re-draws its permutation on every forward pass, including at
eval, so its test accuracy carries extra run-to-run variance. That is
intended. Turning the permutation off at eval would make it identical to
`gap` and destroy the control.

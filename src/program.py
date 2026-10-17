"""
The 132-run program, defined once.

Every run of the ResNet-18 study belongs to one of ten groups keyed by
(dataset, width, bank size K, temperature recipe). This module lists them in
the order the replication runs them, and is the single source for the
runner, the expected-parameter table and the analysis. If a group or arm is
ever added, it is added here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

SEEDS = (0, 1, 2, 3, 4, 5)
ANNEAL_ARGS = ["--temp-start", "30", "--temp-end", "1", "--temp-epochs", "10"]


@dataclass(frozen=True)
class Group:
    dataset: str
    width: int
    K: int
    recipe: str            # "const" or "anneal"
    arms: tuple
    seed_major: bool = False

    @property
    def key(self):
        return (self.dataset, self.width, self.K, self.recipe)


# Order follows the ResNet-18 study's stages: width 64 first, then the
# capacity sweep, the bank-size sweep, Tiny-ImageNet width 32, the
# temperature ablation, and Tiny-ImageNet width 64 (seed-major, as in Stage 4,
# so an interruption leaves complete seed triplets).
PROGRAM = (
    Group("cifar100", 64, 4, "const", ("static", "uniform", "gap", "shuffled", "frozen_random")),
    Group("cifar100", 32, 4, "const", ("static", "uniform", "gap")),
    Group("cifar100", 16, 4, "const", ("static", "uniform", "gap")),
    Group("cifar100", 32, 2, "const", ("gap",)),
    Group("cifar100", 32, 8, "const", ("gap",)),
    Group("tinyimagenet", 32, 4, "const", ("static", "uniform", "gap")),
    Group("cifar100", 64, 4, "anneal", ("gap",)),
    Group("cifar100", 32, 4, "anneal", ("gap",)),
    Group("cifar100", 16, 4, "anneal", ("gap",)),
    Group("tinyimagenet", 64, 4, "const", ("static", "uniform", "gap"), seed_major=True),
)

# The first run of every model: a full static run whose accuracy must clear
# a floor before the remaining 131 are spent. It is also the first job of
# group 0, so it is not an extra run.
GATE_JOB = ("cifar100", 64, 4, "const", "static", 0)


def arm_flags(arm: str) -> dict:
    """How an arm maps onto the model's (scope, router). The static arm is
    scope 'none' with the router ignored, exactly as run_stage1b_static.py
    did for ResNet-18; train.py names its result file by --router, so the
    runner passes --arm-name to keep the 'static' label."""
    if arm == "static":
        return {"scope": "none", "router": "gap"}
    return {"scope": "all", "router": arm}


def tag(model: str, g: Group) -> str:
    return f"{model}__{g.dataset}__w{g.width}__K{g.K}__{g.recipe}"


def jobs(model: str):
    """Every (group, arm, seed) for one model, gate first, in run order."""
    out = []
    for g in PROGRAM:
        if g.seed_major:
            cells = [(a, s) for s in SEEDS for a in g.arms]
        else:
            cells = [(a, s) for a in g.arms for s in SEEDS]
        out += [(g, a, s) for a, s in cells]
    gate = [j for j in out if (j[0].key + (j[1], j[2])) == GATE_JOB]
    assert len(gate) == 1
    return gate + [j for j in out if j is not gate[0]]


def result_name(model: str, g: Group, arm: str, seed: int) -> str:
    return f"{tag(model, g)}_{arm}_seed{seed}.json"


def cells():
    """Set of (dataset, width, K, recipe, arm, seed) for the whole program."""
    return {(g.dataset, g.width, g.K, g.recipe, a, s) for g in PROGRAM for a in g.arms for s in SEEDS}

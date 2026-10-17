"""Runner checks (gates C1-C5), with a fake executor: no training happens.

    python tests/test_runner.py grid | order | resume | gate | microbatch | diverge | all
"""
import sys, pathlib, json, tempfile, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_all_models as R
from src.program import PROGRAM, SEEDS, jobs, cells, result_name, GATE_JOB

# The ResNet-18 program as run, written out independently of src/program.py
# from the stage scripts and the paper's inventory table, so the two can be
# compared rather than one checked against itself.
R18 = set()
for ds, w, K, rec, arms in [
        ("cifar100", 64, 4, "const", ["static", "uniform", "gap", "shuffled", "frozen_random"]),
        ("cifar100", 32, 4, "const", ["static", "uniform", "gap"]),
        ("cifar100", 16, 4, "const", ["static", "uniform", "gap"]),
        ("cifar100", 32, 2, "const", ["gap"]), ("cifar100", 32, 8, "const", ["gap"]),
        ("tinyimagenet", 32, 4, "const", ["static", "uniform", "gap"]),
        ("tinyimagenet", 64, 4, "const", ["static", "uniform", "gap"]),
        ("cifar100", 64, 4, "anneal", ["gap"]), ("cifar100", 32, 4, "anneal", ["gap"]),
        ("cifar100", 16, 4, "anneal", ["gap"])]:
    for a in arms:
        for s in range(6):
            R18.add((ds, w, K, rec, a, s))
assert len(R18) == 132


def args(tmp, **kw):
    a = R.main.__wrapped__ if hasattr(R.main, "__wrapped__") else None
    ns = dict(models=R.RUN_ORDER, data_root=str(tmp / "data"), out=str(tmp / "out"), epochs=150,
              workers=0, probe_steps=2, skip_probe=False, gate_floor=None, force_gate=False,
              fake_data=True, dry_run=False, status=False)
    ns.update(kw)
    return type("A", (), ns)()


class FakeExec:
    """Stands in for subprocess: writes a result JSON (or probe JSON) the
    way src/train.py would. acc(model, cmd) decides the accuracy; oom_above
    makes probes fail for micro-batches larger than a threshold."""

    def __init__(self, acc=lambda m, c: 75.0, oom_above=None, diverge=lambda c: False):
        self.acc, self.oom_above, self.diverge = acc, oom_above, diverge
        self.train_cmds, self.probe_cmds = [], []

    @staticmethod
    def val(cmd, flag):
        return cmd[cmd.index(flag) + 1]

    def __call__(self, cmd, logfile=None):
        if "--max-steps" in cmd:
            self.probe_cmds.append(cmd)
            micro = 128 // int(self.val(cmd, "--accum"))
            oom = self.oom_above is not None and micro > self.oom_above
            pathlib.Path(self.val(cmd, "--probe-out")).write_text(json.dumps(
                {"oom": oom, "sec_per_step": 0.1, "peak_mem_gb": 1.0, "est_minutes": 10}))
            return 0
        self.train_cmds.append(cmd)
        if self.diverge(cmd):
            return 3
        name = f"{self.val(cmd, '--tag')}_{self.val(cmd, '--arm-name')}_seed{self.val(cmd, '--seed')}.json"
        model = self.val(cmd, "--model")
        pathlib.Path(self.val(cmd, "--out"), name).write_text(json.dumps(
            {"final_acc": self.acc(model, cmd), "model": model, "accum": int(self.val(cmd, "--accum"))}))
        return 0


def key_of(cmd):
    v = FakeExec.val
    rec = "anneal" if "--temp-start" in cmd else "const"
    return (v(cmd, "--dataset"), int(v(cmd, "--width")), int(v(cmd, "--num-kernels")), rec,
            v(cmd, "--arm-name"), int(v(cmd, "--seed")))


def test_grid():
    assert cells() == R18, (cells() ^ R18)
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t); ex = FakeExec()
        a = args(t, skip_probe=True)
        for m in R.RUN_ORDER:
            assert R.run_model(a, m, ex) == "done"
        by = {}
        for c in ex.train_cmds:
            by.setdefault(FakeExec.val(c, "--model"), []).append(c)
        for m in R.RUN_ORDER:
            ks = [key_of(c) for c in by[m]]
            assert len(ks) == 132 and set(ks) == R18, m
            # flags that define an arm
            for c in by[m]:
                k = key_of(c)
                assert FakeExec.val(c, "--scope") == ("none" if k[4] == "static" else "all")
                if k[4] != "static":
                    assert FakeExec.val(c, "--router") == k[4]
                assert FakeExec.val(c, "--placement") == "spatial"
                assert FakeExec.val(c, "--batch-size") == "128" and FakeExec.val(c, "--epochs") == "150"
                if k[3] == "anneal":
                    assert c[c.index("--temp-start"):c.index("--temp-start") + 6] == \
                        ["--temp-start", "30", "--temp-end", "1", "--temp-epochs", "10"]
            print(f"  {m:<14} 132 jobs, set-equal to the ResNet-18 program, arm flags correct")
        assert len(list((t / "out").rglob("*_seed*.json"))) == 528
    print("4 models x 132 jobs ok")


def test_order():
    J = jobs("vgg16")
    assert (J[0][0].key + (J[0][1], J[0][2])) == GATE_JOB, "gate job first"
    tin64 = [(arm, s) for g, arm, s in J if g.key == ("tinyimagenet", 64, 4, "const")]
    assert tin64 == [(a, s) for s in SEEDS for a in ("static", "uniform", "gap")], tin64[:6]
    # the other groups are arm-major, in the program's group order
    c64 = [(arm, s) for g, arm, s in J[1:] if g.key == ("cifar100", 64, 4, "const")]
    assert c64[:5] == [("static", 1), ("static", 2), ("static", 3), ("static", 4), ("static", 5)]
    order = []
    for g, _, _ in J[1:]:
        if not order or order[-1] != g.key:
            order.append(g.key)
    assert order == [g.key for g in PROGRAM], order
    print("  gate first; TIN w64 seed-major (static, uniform, gap per seed); other groups arm-major in program order")
    print("ok")


def test_resume():
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        a = args(t, skip_probe=True, models=["mobilenetv2"])
        calls = {"n": 0}

        class Crash(FakeExec):
            def __call__(self, cmd, logfile=None):
                if "--max-steps" not in cmd:
                    calls["n"] += 1
                    if calls["n"] == 41:
                        return 1                     # a crash on the 41st run
                return super().__call__(cmd, logfile)
        ex1 = Crash()
        assert R.run_model(a, "mobilenetv2", ex1) == "stopped"
        assert len(ex1.train_cmds) == 40
        ex2 = FakeExec()
        assert R.run_model(a, "mobilenetv2", ex2) == "done"
        assert len(ex2.train_cmds) == 92, len(ex2.train_cmds)
        first = {key_of(c) for c in ex1.train_cmds}; second = {key_of(c) for c in ex2.train_cmds}
        assert not first & second and len(first | second) == 132
        ex3 = FakeExec()
        assert R.run_model(a, "mobilenetv2", ex3) == "done" and not ex3.train_cmds
        print("  crash at run 41 -> 40 done; rerun runs the other 92 (no repeats); third call runs 0")
    print("ok")


def test_gate():
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        a = args(t, skip_probe=True, models=["mobilenetv2", "vgg16"])
        ex = FakeExec(acc=lambda m, c: 40.0 if m == "mobilenetv2" else 75.0)
        out = {m: R.run_model(a, m, ex) for m in a.models}
        assert out == {"mobilenetv2": "gate_failed", "vgg16": "done"}, out
        mb = [c for c in ex.train_cmds if FakeExec.val(c, "--model") == "mobilenetv2"]
        assert len(mb) == 1, "only the gate run was spent"
        g = json.loads((t / "out" / "mobilenetv2" / "GATE_FAILED.json").read_text())
        assert g["final_acc"] == 40.0 and g["floor"] == 65.0
        assert "GATE FAILED" in (t / "out" / "mobilenetv2" / "runner.log").read_text()
        # rerun: skipped without spending anything
        ex2 = FakeExec()
        assert R.run_model(a, "mobilenetv2", ex2) == "gate_failed" and not ex2.train_cmds
        # at the floor passes (>=)
        a2 = args(t / "b", skip_probe=True)
        assert R.run_model(a2, "vgg16", FakeExec(acc=lambda m, c: 65.0)) == "done"
        print("  mobilenetv2 at 40.0 < 65: 1 run spent, GATE_FAILED.json written, logged; vgg16 continued; "
              "rerun skips; 65.0 at floor 65 passes")
    print("ok")


def test_microbatch():
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        a = args(t, models=["resnet50"])
        ex = FakeExec(oom_above=32)                    # only micro-batch <= 32 fits
        assert R.run_model(a, "resnet50", ex) == "done"
        d = t / "out" / "resnet50"
        plan = json.loads((d / "plan.json").read_text())
        assert len(plan) == 10 and all(v["accum"] == 4 for v in plan.values())
        # probes: 3 rungs per group (128 oom, 64 oom, 32 ok), on the heaviest arm
        assert len(ex.probe_cmds) == 30, len(ex.probe_cmds)
        heavy = {FakeExec.val(c, "--router") for c in ex.probe_cmds
                 if FakeExec.val(c, "--width") == "64" and FakeExec.val(c, "--dataset") == "cifar100"}
        assert "shuffled" in heavy
        # every run of every group used the group's choice
        assert all(FakeExec.val(c, "--accum") == "4" for c in ex.train_cmds)
        dev = (d / "DEVIATIONS.md").read_text()
        assert dev.count("micro-batch 32 x 4") == 10
        # a restart reuses plan.json, no new probes
        ex2 = FakeExec(oom_above=32)
        for f in d.glob("*_seed*.json"):
            f.unlink()
        R.run_model(a, "resnet50", ex2)
        assert not ex2.probe_cmds
        # fits at 128 -> accum 1, no deviation written
        a3 = args(t / "c", models=["vgg16"])
        R.run_model(a3, "vgg16", FakeExec())
        assert not (t / "c" / "out" / "vgg16" / "DEVIATIONS.md").exists()
        print("  OOM above 32: 10 groups probed on heaviest arm, accum 4 chosen, persisted, logged in "
              "DEVIATIONS.md, applied to all 132 runs; restart re-probes nothing; fits at 128 -> no deviation")
    print("ok")


def test_diverge():
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        a = args(t, skip_probe=True, models=["convnext_tiny"])
        bad = lambda c: FakeExec.val(c, "--arm-name") == "gap" and FakeExec.val(c, "--seed") == "3" \
            and FakeExec.val(c, "--width") == "16"
        ex = FakeExec(diverge=bad)
        assert R.run_model(a, "convnext_tiny", ex) == "done"
        d = t / "out" / "convnext_tiny"
        marks = sorted(p.name for p in d.glob("*.DIVERGED"))
        assert len(marks) == 2, marks                  # w16 const and w16 anneal gap seed 3
        done, div, n, _ = R.status(a, "convnext_tiny")
        assert (done, div, n) == (130, 2, 132)
        ex2 = FakeExec()
        R.run_model(a, "convnext_tiny", ex2)
        assert not ex2.train_cmds, "diverged runs are not silently retried"
        print(f"  2 diverged runs -> markers {marks}; status 130 done + 2 diverged of 132; not retried")
    print("ok")


def test_oom_midrun():
    """Probe fits, a real run then runs out of memory (exit 5): the group
    steps down one rung from that run on, the change is logged, the run is
    redone, and earlier runs keep their micro-batch."""
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        a = args(t, models=["resnet50"])
        hit = {"n": 0}

        class OOM(FakeExec):
            def __call__(self, cmd, logfile=None):
                if "--max-steps" not in cmd and FakeExec.val(cmd, "--arm-name") == "gap" \
                        and FakeExec.val(cmd, "--width") == "32" and FakeExec.val(cmd, "--dataset") == "cifar100" \
                        and FakeExec.val(cmd, "--num-kernels") == "4" and "--temp-start" not in cmd \
                        and FakeExec.val(cmd, "--seed") == "2" and hit["n"] == 0:
                    hit["n"] = 1; self.train_cmds.append(cmd); return 5
                return super().__call__(cmd, logfile)
        ex = OOM()
        assert R.run_model(a, "resnet50", ex) == "done"
        d = t / "out" / "resnet50"
        g = [c for c in ex.train_cmds if FakeExec.val(c, "--width") == "32" and FakeExec.val(c, "--dataset") == "cifar100"
             and FakeExec.val(c, "--num-kernels") == "4" and "--temp-start" not in c]
        acc = [(FakeExec.val(c, "--arm-name"), FakeExec.val(c, "--seed"), FakeExec.val(c, "--accum")) for c in g]
        assert acc.count(("gap", "2", "1")) == 1 and ("gap", "2", "2") in acc
        assert all(x[2] == "1" for x in acc[:acc.index(("gap", "2", "1"))])
        assert all(x[2] == "2" for x in acc[acc.index(("gap", "2", "2")):])
        plan = json.loads((d / "plan.json").read_text())["resnet50__cifar100__w32__K4__const"]
        assert plan["accum"] == 2 and plan["raised_after"].endswith("gap_seed2.json")
        assert "out of memory mid-run" in (d / "DEVIATIONS.md").read_text()
        assert len(list(d.glob("*_seed*.json"))) == 132
        print(f"  OOM in gap seed 2 of CIFAR w32: {acc.index(('gap', '2', '1'))} earlier runs kept micro 128, "
              f"that run and the {len(acc) - acc.index(('gap', '2', '2')) - 1} after it used micro 64; logged; 132 results")
    print("ok")


TESTS = dict(oom_midrun=test_oom_midrun, grid=test_grid, order=test_order, resume=test_resume, gate=test_gate,
             microbatch=test_microbatch, diverge=test_diverge)
if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for w in (TESTS if which == ["all"] else which):
        print(f"== {w}")
        TESTS[w]()

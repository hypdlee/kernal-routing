"""Training-script checks (gates B1-B5).

    python tests/test_train.py backcompat | accum | probe | config | resume | e2e | all
"""
import sys, pathlib, json, subprocess, tempfile, copy
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

PY = sys.executable


def run(mod, *args, check=True):
    r = subprocess.run([PY, "-m", mod, "--fake-data", "--workers", "0", *args],
                       cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise AssertionError(f"{mod} {args} rc={r.returncode}")
    return r


def load(d):
    fs = list(pathlib.Path(d).glob("*.json"))
    assert len(fs) == 1, fs
    return fs[0], json.loads(fs[0].read_text())


def test_backcompat():
    """An old command line, unchanged, gives the same run under the new
    script as under the frozen copy used for the 132 ResNet-18 runs."""
    for extra in (["--router", "gap"], ["--router", "uniform"], ["--scope", "none"],
                  ["--router", "gap", "--temp-start", "30", "--temp-epochs", "1"]):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            common = ["--epochs", "2", "--width", "16", "--tag", "bc", *extra]
            run("src._train_v1", "--out", a, *common)
            run("src.train", "--out", b, *common)
            (fa, ra), (fb, rb) = load(a), load(b)
            assert fa.name == fb.name, (fa.name, fb.name)          # same file name
            la = [h["loss"] for h in ra["history"]]; lb = [h["loss"] for h in rb["history"]]
            assert la == lb and ra["final_acc"] == rb["final_acc"], (la, lb)
            assert ra["params"] == rb["params"]
            assert ra["routing"] == rb["routing"]
            print(f"  {' '.join(extra):<45} file {fb.name}, losses {lb} identical")
    print("ok")


def test_accum():
    """(128,1) and (64,2): same optimiser steps and LR; loss scaled by 1/accum
    so gradients agree. BN put in eval mode to isolate the accumulation
    arithmetic from the per-micro-batch BN statistics (the one documented
    difference)."""
    from src.train import train_step
    from src.models import build_model
    torch.manual_seed(0)
    base = build_model("resnet18", 100, width=16).double()
    x = torch.randn(128, 3, 32, 32, dtype=torch.float64); y = torch.randint(0, 100, (128,))
    params = {}
    for accum in (1, 2, 4):
        m = copy.deepcopy(base)
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / 10)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        loss = train_step(m, opt, sched, nn.CrossEntropyLoss(), scaler, x, y,
                          torch.device("cpu"), None, accum)
        params[accum] = (torch.cat([p.detach().flatten() for p in m.parameters()]),
                         sched.last_epoch, opt.param_groups[0]["lr"], loss)
    for a in (2, 4):
        d = (params[a][0] - params[1][0]).abs().max().item()
        assert d < 1e-12, (a, d)
        assert params[a][1:3] == params[1][1:3]
        assert abs(params[a][3] - params[1][3]) < 1e-12
        print(f"  accum {a}: max param diff after one step {d:.1e}, steps {params[a][1]}, lr {params[a][2]:.3f}")
    # the real script: identical step count and final lr for (128,1) and (64,2)
    recs = {}
    for acc in ("1", "2"):
        with tempfile.TemporaryDirectory() as d:
            run("src.train", "--out", d, "--epochs", "2", "--width", "16", "--accum", acc, "--tag", "a")
            recs[acc] = load(d)[1]
    assert [h["lr"] for h in recs["1"]["history"]] == [h["lr"] for h in recs["2"]["history"]]
    assert recs["2"]["micro_batch"] == 64 and recs["2"]["accum"] == 2
    print(f"  script: lr per epoch identical {[h['lr'] for h in recs['2']['history']]}")
    print("ok")


def test_probe():
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "p.json"
        run("src.train", "--model", "mobilenetv2", "--width", "16", "--max-steps", "2",
            "--probe-out", str(out), "--accum", "2", "--out", d)
        p = json.loads(out.read_text())
        assert p["oom"] is False and p["sec_per_step"] > 0 and p["micro_batch"] == 64
        assert p["steps_per_epoch"] == 390 and p["est_minutes"] > 0   # real CIFAR-100 steps, not the fake loader's 4
        assert not list(pathlib.Path(d).glob("*_seed*.json"))     # probe writes no result
        print(f"  {json.dumps(p)}")
    print("ok")


def test_config():
    with tempfile.TemporaryDirectory() as d:
        run("src.train", "--model", "vgg16", "--width", "16", "--epochs", "1", "--scope", "none",
            "--arm-name", "static", "--accum", "4", "--tag", "vgg16__cifar100__w16__K4__const", "--out", d)
        f, r = load(d)
        assert f.name == "vgg16__cifar100__w16__K4__const_static_seed0.json", f.name
        for k, v in dict(model="vgg16", placement="spatial", arm="static", micro_batch=32, accum=4,
                         resumed_at_epoch=None).items():
            assert r[k] == v, (k, r[k])
        assert r["config"]["model"] == "vgg16" and r["config"]["width"] == 16
        print(f"  {f.name}: " + ", ".join(f"{k}={r[k]}" for k in ("model", "placement", "arm", "micro_batch", "accum")))
    print("ok")


def test_resume():
    """Crash after epoch 0, rerun the same command: it resumes at epoch 1,
    finishes with a full history and deletes the checkpoint."""
    with tempfile.TemporaryDirectory() as d:
        ck = str(pathlib.Path(d) / "ck.pt")
        args = ["--model", "resnet50", "--width", "16", "--epochs", "3", "--ckpt", ck, "--out", d, "--tag", "r"]
        r = run("src.train", *args, "--stop-after-epoch", "0", check=False)
        assert r.returncode == 4 and pathlib.Path(ck).exists()
        run("src.train", *args)
        f, rec = load(d)
        assert rec["resumed_at_epoch"] == 1 and [h["epoch"] for h in rec["history"]] == [0, 1, 2]
        assert not pathlib.Path(ck).exists()
        print(f"  resumed_at_epoch={rec['resumed_at_epoch']}, epochs {[h['epoch'] for h in rec['history']]}, checkpoint removed")
    print("ok")


def test_e2e():
    """One fake-data epoch, every model, both datasets, gap arm, width 16
    (CPU budget). Checks the result JSON and its routing report."""
    from src.models import MODELS
    n = 0
    for name in MODELS:
        for ds in ("cifar100", "tinyimagenet"):
            with tempfile.TemporaryDirectory() as d:
                # micro-batch 32 on Tiny-ImageNet: a 64x64 batch of 128 through the
                # dynamic depthwise layers needs ~7 GB fp32 on CPU, the whole
                # test container. (On the GPU the runner's probe decides.)
                run("src.train", "--model", name, "--dataset", ds, "--width", "16", "--epochs", "1",
                    "--router", "gap", "--out", d, "--tag", f"{name}__{ds}",
                    "--accum", "4" if ds == "tinyimagenet" else "1")
                f, r = load(d)
                exp = {"resnet18": 16, "resnet50": 16, "vgg16": 12, "mobilenetv2": 17, "convnext_tiny": 18}[name]
                assert len(r["routing"]) == exp, (name, len(r["routing"]))
                assert all(0 <= l["z_entropy_norm"] <= 1 for l in r["routing"])
                assert r["model"] == name and r["config"]["dataset"] == ds
                n += 1
                print(f"  {name:<14} {ds:<13} loss {r['history'][0]['loss']:.3f}  routing layers {len(r['routing'])}")
    print(f"{n} runs ok")
    assert n == 10


TESTS = dict(backcompat=test_backcompat, accum=test_accum, probe=test_probe, config=test_config,
             resume=test_resume, e2e=test_e2e)
if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for w in (TESTS if which == ["all"] else which):
        print(f"== {w}")
        TESTS[w]()

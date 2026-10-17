"""
MobileNetV2 results recovered from the runner's train.log.

The training log prints, for every run, its configuration line, the loss
and accuracy every ten epochs, and a final line with the final and best
accuracy and the result file name. This module turns that log into per-run
records keyed exactly as the result files are, so the paper can be built
before the result JSONs are copied off the training machine.

What the log does not contain: routing entropies (they are only in the JSON
files). Nothing here invents them; mbv2_section.py reads them from the
result files after checking every final accuracy there against this log.

    python report/mbv2_log.py path/to/train.log out.csv
"""
import csv, re, sys
from pathlib import Path

EP_RE = re.compile(r"\s+ep\s+(\d+)\s+loss ([\d.]+)\s+acc ([\d.]+)\s+best ([\d.]+).*\((\d+)s\)")
FINAL_RE = re.compile(r"final=([\d.]+) best=([\d.]+) -> .*?(mobilenetv2__(\w+?)__w(\d+)__K(\d+)__(const|anneal)_(\w+?)_seed(\d))\.json")
HEAD_RE = re.compile(r"^\[mobilenetv2 (\w+) (\w+) w(\d+) K(\d+) seed=(\d)\] device=(\w+) .*micro_batch=(\d+) accum=(\d+)")


def parse(path):
    """Returns {(dataset, width, K, recipe, arm, seed): record}. A run that
    crashed and was resumed appears with the epochs from both attempts; the
    record keeps the last value seen for each epoch, which is what the
    resumed run continued from."""
    recs, curve, head, resumed = {}, {}, None, False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        h = HEAD_RE.match(line)
        if h:
            head = h
            continue
        if "resumed from" in line:
            resumed = True
            continue
        m = EP_RE.match(line)
        if m:
            curve[int(m.group(1))] = dict(loss=float(m.group(2)), acc=float(m.group(3)),
                                          secs=int(m.group(5)))
            continue
        m = FINAL_RE.search(line)
        if m:
            key = (m.group(4), int(m.group(5)), int(m.group(6)), m.group(7), m.group(8), int(m.group(9)))
            if key in recs:
                raise SystemExit(f"two final lines for {key}")
            last = curve[max(curve)]
            recs[key] = dict(final=float(m.group(1)), best=float(m.group(2)), loss=last["loss"],
                             minutes=last["secs"] / 60, resumed=resumed, curve=dict(curve),
                             micro=int(head.group(7)) if head else None)
            curve, resumed = {}, False
    return recs


def write_csv(recs, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "width", "K", "recipe", "arm", "seed", "final_acc", "best_acc",
                    "final_train_loss", "minutes", "resumed", "micro_batch"])
        for k in sorted(recs):
            r = recs[k]
            w.writerow(list(k) + [r["final"], r["best"], r["loss"], round(r["minutes"], 1),
                                  int(r["resumed"]), r["micro"]])


if __name__ == "__main__":
    R = parse(sys.argv[1])
    print(len(R), "runs")
    if len(sys.argv) > 2:
        write_csv(R, sys.argv[2])

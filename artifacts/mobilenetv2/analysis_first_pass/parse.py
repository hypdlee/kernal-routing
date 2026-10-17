import re, sys, json, collections
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
recs = {}; last_ep = {}
cur_loss = None; cur_secs = None; resumed = False
for l in lines:
    m = re.match(r"\s+ep\s+(\d+)\s+loss ([\d.]+)\s+acc ([\d.]+)\s+best ([\d.]+).*\((\d+)s\)", l)
    if m:
        cur_ep = int(m.group(1)); cur_loss = float(m.group(2)); cur_secs = int(m.group(5)); continue
    if "resumed from" in l: resumed = True
    m = re.search(r"final=([\d.]+) best=([\d.]+) -> .*?(mobilenetv2__(\w+?)__w(\d+)__K(\d+)__(const|anneal)_(\w+?)_seed(\d))\.json", l)
    if m:
        key = (m.group(4), int(m.group(5)), int(m.group(6)), m.group(7), m.group(8), int(m.group(9)))
        if key in recs: print("DUPLICATE", key)
        recs[key] = dict(final=float(m.group(1)), best=float(m.group(2)), loss=cur_loss, secs=cur_secs, resumed=resumed)
        resumed = False
print(len(recs), "runs")
json.dump({"|".join(map(str, k)): v for k, v in recs.items()}, open(sys.argv[2], "w"))

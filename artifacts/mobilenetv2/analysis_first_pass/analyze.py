import json, sys, math
import numpy as np
from scipy import stats
sys.path.insert(0, "/home/claude/repo/report"); sys.path.insert(0, "/home/claude/repo")
from data import paired, verdict, MARGIN
R = {tuple(k.split("|")): v for k, v in json.load(open("/tmp/claude-0/mbs/runs.json")).items()}
def A(ds, w, arm, K=4, rec="const", f="final"):
    return np.array([R[(ds, str(w), str(K), rec, arm, str(s))][f] for s in range(6)])
DSN={"cifar100":"C100","tinyimagenet":"TIN"}
print("== every cell: mean +- sd (final), mean final train loss, mean minutes")
for ds,w,K,rec,arms in [("cifar100",64,4,"const",["static","uniform","gap","shuffled","frozen_random"]),("cifar100",32,4,"const",["static","uniform","gap"]),("cifar100",16,4,"const",["static","uniform","gap"]),("cifar100",32,2,"const",["gap"]),("cifar100",32,8,"const",["gap"]),("cifar100",64,4,"anneal",["gap"]),("cifar100",32,4,"anneal",["gap"]),("cifar100",16,4,"anneal",["gap"]),("tinyimagenet",32,4,"const",["static","uniform","gap"]),("tinyimagenet",64,4,"const",["static","uniform","gap"])]:
    for a in arms:
        v=A(ds,w,a,K,rec); L=A(ds,w,a,K,rec,"loss"); S=A(ds,w,a,K,rec,"secs")/60
        print(f"  {DSN[ds]} w{w} K{K} {rec:6} {a:14} {v.mean():6.2f} +- {v.std(ddof=1):.2f}  [{' '.join(f'{x:.2f}' for x in v)}]  loss {L.mean():.3f}  min {S.mean():6.1f}")
def show(lbl,q): print(f"  {lbl:42} {q['mean']:+.3f} [{q['lo']:+.3f},{q['hi']:+.3f}] p={q['p']:.4f} tost={q['tost']:.4f} W={q['wil']:.3f} -> {verdict(q)}")
cond=[("cifar100",16),("cifar100",32),("cifar100",64),("tinyimagenet",32),("tinyimagenet",64)]
print("\n== vs static, K=4, tau=1")
for ds,w in cond:
    for a in (["uniform","gap","shuffled","frozen_random"] if (ds,w)==("cifar100",64) else ["uniform","gap"]):
        show(f"{DSN[ds]} w{w} {a}-static", paired(A(ds,w,a),A(ds,w,"static")))
print("\n== decomposition")
for ds,w in cond:
    b=paired(A(ds,w,"uniform"),A(ds,w,"static")); r=paired(A(ds,w,"gap"),A(ds,w,"uniform")); t=paired(A(ds,w,"gap"),A(ds,w,"static"))
    print(f"  {DSN[ds]} w{w}: bank {b['mean']:+.3f} (tost {b['tost']:.4f}, {verdict(b)}) | routing {r['mean']:+.3f} (p {r['p']:.4f}) | total {t['mean']:+.3f} (p {t['p']:.4f})")
print("\n== w64 controls")
show("frozen_random - gap", paired(A("cifar100",64,"frozen_random"),A("cifar100",64,"gap")))
show("shuffled - gap", paired(A("cifar100",64,"shuffled"),A("cifar100",64,"gap")))
show("frozen_random - uniform", paired(A("cifar100",64,"frozen_random"),A("cifar100",64,"uniform")))
arms=["static","uniform","gap","shuffled","frozen_random"]
F,p=stats.f_oneway(*[A("cifar100",64,a) for a in arms]); print(f"  ANOVA 5 arms w64: F={F:.2f} p={p:.4g}")
print("\n== matched budgets (params M from PARAMS_EXPECTED: gap w16 0.49, static w32 0.82, gap w32 1.20, static w64 2.35; TIN gap w32 1.33, static w64 2.48)")
for ds,wg,ws,r in [("cifar100",16,32,0.60),("cifar100",32,64,0.51),("tinyimagenet",32,64,0.54)]:
    show(f"{DSN[ds]} static w{ws} - gap w{wg} (ratio {r})", paired(A(ds,ws,"static"),A(ds,wg,"gap")))
print("\n== K sweep (C100 w32 gap)")
for k in (2,4,8): show(f"K{k} - static", paired(A("cifar100",32,"gap",K=k),A("cifar100",32,"static")))
show("K8 - K2", paired(A("cifar100",32,"gap",K=8),A("cifar100",32,"gap",K=2)))
show("K4 - K2", paired(A("cifar100",32,"gap",K=4),A("cifar100",32,"gap",K=2)))
print("\n== temperature")
for w in (16,32,64):
    show(f"w{w} anneal - const (gap)", paired(A("cifar100",w,"gap",rec="anneal"),A("cifar100",w,"gap")))
    show(f"w{w} anneal - uniform", paired(A("cifar100",w,"gap",rec="anneal"),A("cifar100",w,"uniform")))
print("\n== prereg: P1 bank equiv in 5/5, P2 static(2c)>gap(c) 3/3, P3 routing>0 p<.05 >=3/5")
nb=sum(paired(A(ds,w,"uniform"),A(ds,w,"static"))["tost"]<0.05 for ds,w in cond)
np2=sum((lambda q:q["p"]<0.05 and q["mean"]>0)(paired(A(ds,ws,"static"),A(ds,wg,"gap"))) for ds,wg,ws in [("cifar100",16,32),("cifar100",32,64),("tinyimagenet",32,64)])
nr=sum((lambda q:q["p"]<0.05 and q["mean"]>0)(paired(A(ds,w,"gap"),A(ds,w,"uniform"))) for ds,w in cond)
print(f"  P1 {nb}/5  P2 {np2}/3  P3 {nr}/5")
print("\n== gate", R[("cifar100","64","4","const","static","0")])
print("== total GPU hours (sum of final elapsed seconds)", sum(v["secs"] for v in R.values())/3600)
print("== resumed runs", [k for k,v in R.items() if v["resumed"]])

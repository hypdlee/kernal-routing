import math
import numpy as np
from data import *
f=lambda x,n=3: f"{x:.{n}f}"
sg=lambda x,n=3: f"{x:+.{n}f}"
def pv(p):
    if isinstance(p,float) and math.isnan(p): return "--"
    return "$<10^{-4}$" if p<1e-4 else f"{p:.4f}"
tt=lambda a: "\\texttt{"+a.replace("_","\\_")+"}"
T={}
DSN={"cifar100":"CIFAR-100","tinyimagenet":"Tiny-ImageNet"}

# A1 every run, both datasets (K=4, tau=1)
r=[]
for ds,ws in [("cifar100",[16,32,64]),("tinyimagenet",[32,64])]:
    for w in ws:
        for a in ACC[(ds,w,4,"const")]:
            v=acc(ds,w,a)
            r.append(f"{DSN[ds]} & {w} & {tt(a)} & "+" & ".join(f(x,2) for x in v)+f" & {f(v.mean())} & {f(v.std(ddof=1))} \\\\")
T["ALLRUNS"]="\n".join(r)

# A2 configuration and cost
r=[]
for key in [("cifar100",16,4,"const"),("cifar100",32,4,"const"),("cifar100",64,4,"const"),
            ("tinyimagenet",32,4,"const"),("tinyimagenet",64,4,"const")]:
    ds,w=key[0],key[1]
    for a in ACC[key]:
        rp=ROUTER_K[(ds,w)] if a in ("gap","shuffled") else 0
        r.append(f"{DSN[ds]} & {w} & {tt(a)} & {f(PARAMS[key][a],2)} & {f(rp,1)} & {f(LOSS[key][a],4)} & {f(MINS[key][a],1)} \\\\")
T["CONFIG"]="\n".join(r)

# vs static, all five conditions
r=[]
for ds,ws in [("cifar100",[16,32,64]),("tinyimagenet",[32,64])]:
    for w in ws:
        for a in [x for x in ACC[(ds,w,4,"const")] if x!="static"]:
            q=paired(acc(ds,w,a),acc(ds,w,"static"))
            r.append(f"{DSN[ds]} & {w} & {tt(a)} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & "
                     f"{pv(q['p'])} & {pv(q['wil'])} & {sg(q['dz'],2)} & {pv(q['tost'])} & {verdict(q)} \\\\")
T["VSSTATIC"]="\n".join(r)

# decomposition
r=[]
for ds,w in [("cifar100",16),("cifar100",32),("cifar100",64),("tinyimagenet",32),("tinyimagenet",64)]:
    b=paired(acc(ds,w,"uniform"),acc(ds,w,"static")); g=paired(acc(ds,w,"gap"),acc(ds,w,"uniform"))
    t=paired(acc(ds,w,"gap"),acc(ds,w,"static"))
    # No "routing share" column. Where the bank term is negative (w16) the
    # routing term exceeds the total and a share reads as >100%, which is
    # meaningless. The bank is equivalent to zero everywhere, so the total
    # effect is the routing effect to within the margin.
    r.append(f"{DSN[ds]} & {w} & {sg(b['mean'])} & {pv(b['tost'])} & {verdict(b)} & {sg(g['mean'])} & {pv(g['p'])} & {sg(t['mean'])} & {pv(t['p'])} \\\\")
T["DECOMP"]="\n".join(r)

# matched budgets
r=[]
for ds,wg,ws in [("cifar100",16,32),("cifar100",32,64),("tinyimagenet",32,64)]:
    q=paired(acc(ds,ws,"static"),acc(ds,wg,"gap"))
    r.append(f"{DSN[ds]} & {tt('gap')} w{wg} ({f(PARAMS[(ds,wg,4,'const')]['gap'],2)}M) & "
             f"{tt('static')} w{ws} ({f(PARAMS[(ds,ws,4,'const')]['static'],2)}M) & "
             f"{f(acc(ds,wg,'gap').mean())} & {f(acc(ds,ws,'static').mean())} & {sg(q['mean'])} & "
             f"[{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & "
             f"{f(MINS[(ds,wg,4,'const')]['gap'],1)} / {f(MINS[(ds,ws,4,'const')]['static'],1)} \\\\")
T["MATCHED"]="\n".join(r)

# width-64 routing entropy
r=[]
for a in ["uniform","gap","shuffled","frozen_random"]:
    e=ENT_W64[a]; r.append(f"{tt(a)} & {f(e.mean(),4)} & {f(e.min(),4)} & {f(e.max(),4)} & {f(1-e.mean(),5)} \\\\")
T["ENTW64"]="\n".join(r)

# K sweep
r=[]
for k in [2,4,8]:
    v=acc("cifar100",32,"gap",k=k); q=paired(v,acc("cifar100",32,"static"))
    r.append(f"{k} & {f(v.mean())} & {f(v.std(ddof=1))} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & "
             f"{f(PARAMS[('cifar100',32,k,'const')]['gap'],2)} & {f(MINS[('cifar100',32,k,'const')]['gap'],1)} \\\\")
r.append(f"\\midrule {tt('static')} & {f(acc('cifar100',32,'static').mean())} & {f(acc('cifar100',32,'static').std(ddof=1))} & -- & -- & -- & "
         f"{f(PARAMS[('cifar100',32,4,'const')]['static'],2)} & {f(MINS[('cifar100',32,4,'const')]['static'],1)} \\\\")
T["KSWEEP"]="\n".join(r)
r=[]
for a,b in [(2,4),(4,8),(2,8)]:
    q=paired(acc("cifar100",32,"gap",k=b),acc("cifar100",32,"gap",k=a))
    r.append(f"$K={b}$ vs $K={a}$ & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & {pv(q['p'])} & {pv(q['tost'])} & {verdict(q)} \\\\")
T["KPAIRS"]="\n".join(r)

# pre-registered prediction
q=paired(acc("tinyimagenet",64,"gap"),acc("tinyimagenet",64,"static"))
L=LOSS[("tinyimagenet",64,4,"const")]["static"]
T["PREDICT"]=(f"(a) & \\texttt{{gap}} $-$ \\texttt{{static}} below \\SI{{0.5}}{{pp}}, equivalent by TOST & "
              f"{sg(q['mean'])} pp, $p={q['p']*1e5:.1f}\\times10^{{-5}}$, $p_{{\\text{{TOST}}}}={q['tost']:.4f}$ & failed \\\\\n"
              f"(b) & static final training loss below 0.05 & {f(L,3)} & held \\\\")

# saturation
r=[]
for ds,w in sorted([("cifar100",16),("cifar100",32),("cifar100",64),("tinyimagenet",32),("tinyimagenet",64)],
                   key=lambda z:LOSS[(z[0],z[1],4,"const")]["static"]):
    q=paired(acc(ds,w,"gap"),acc(ds,w,"static"))
    r.append(f"{DSN[ds]} & {w} & {f(LOSS[(ds,w,4,'const')]['static'],4)} & {sg(q['mean'])} & [{sg(q['lo'])}, {sg(q['hi'])}] & {verdict(q)} \\\\")
T["SATUR"]="\n".join(r)

# temperature
r=[]
for w in [16,32,64]:
    c=acc("cifar100",w,"gap"); an=acc("cifar100",w,"gap",rec="anneal")
    a1=paired(an,c); s1=paired(c,acc("cifar100",w,"static")); s2=paired(an,acc("cifar100",w,"static"))
    u2=paired(an,acc("cifar100",w,"uniform"))
    r.append(f"{w} & {f(c.mean())} & {f(an.mean())} & {sg(a1['mean'])} & {pv(a1['p'])} & {sg(s1['mean'])} & "
             f"{sg(s2['mean'])} & {pv(s2['tost'])} & {sg(u2['mean'])} & {verdict(u2)} \\\\")
T["TEMP"]="\n".join(r)

# entropy vs benefit
r=[]
for w in [16,32,64]:
    for lbl,rec in [("$1$","const"),("$30\\!\\to\\!1$","anneal")]:
        e=ENT[("cifar100",w,rec)]; d=1-e.mean()
        rms=math.sqrt(d/(4/(2*math.log(4)))/4)
        q=paired(acc("cifar100",w,"gap",rec=rec),acc("cifar100",w,"uniform"))
        r.append(f"{w} & {lbl} & {f(e.mean(),5)} & {f(d,5)} & {f(rms,4)} & {sg(q['mean'])} & {pv(q['p'])} \\\\")
T["ENTBEN"]="\n".join(r)

# inventory
r=[];tot=0;hrs=0
for key,g in ACC.items():
    ds,w,k,rec=key; n=sum(len(v) for v in g.values()); tot+=n
    h=sum(MINS[key][a]*6 for a in g)/60; hrs+=h
    r.append(f"{DSN[ds]} & {w} & {k} & {'constant' if rec=='const' else 'annealed'} & {', '.join(tt(a) for a in g)} & {n} & {f(h,1)} \\\\")
r.append(f"\\midrule \\multicolumn{{5}}{{l}}{{total}} & {tot} & {f(hrs,1)} \\\\")
T["INVENT"]="\n".join(r)

for k,v in T.items(): open(f"tab_{k}.tex","w").write(v.rstrip()+"\n")
print(f"{len(T)} tables written; runs={n_runs()}, GPU h={gpu_hours():.1f}")

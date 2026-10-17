"""Every number quoted in running text is emitted here as \\V{key}."""
import math
import numpy as np
from scipy import stats
from data import *
M={}
def put(k,v): M[k]=v
f=lambda x,n=3: "--" if isinstance(x,float) and math.isnan(x) else f"{x:.{n}f}"
sg=lambda x,n=3: "--" if isinstance(x,float) and math.isnan(x) else f"{x:+.{n}f}"

put("nruns", str(n_runs())); put("gpuh", f"{gpu_hours():.0f}")

# width-64 CIFAR five-arm picture
arms=["static","uniform","gap","shuffled","frozen_random"]
means=[acc("cifar100",64,a).mean() for a in arms]
put("spread64", f(max(means)-min(means)))
put("seedsd64", f(np.median([acc("cifar100",64,a).std(ddof=1) for a in arms])))
_ok=lambda v: np.asarray(v,dtype=float)[~np.isnan(np.asarray(v,dtype=float))]
F,p=stats.f_oneway(*[_ok(acc("cifar100",64,a)) for a in arms])
allv=np.concatenate([_ok(acc("cifar100",64,a)) for a in arms]); g=allv.mean()
ssb=sum(len(_ok(acc("cifar100",64,a)))*(m-g)**2 for a,m in zip(arms,means)); ssw=sum(((_ok(acc("cifar100",64,a))-_ok(acc("cifar100",64,a)).mean())**2).sum() for a in arms)
put("anovaF",f(F)); put("anovaP",f(p)); put("anovaEta",f(ssb/(ssb+ssw)))
for a in ["uniform","gap","shuffled","frozen_random"]:
    q=paired(acc("cifar100",64,a),acc("cifar100",64,"static")); k=a.replace("_","")
    put(f"w64{k}d",sg(q["mean"])); put(f"w64{k}tost",f(q["tost"],4))
q=paired(acc("cifar100",64,"frozen_random"),acc("cifar100",64,"gap"))
put("frzVgap",sg(q["mean"])); put("frzVgapP",f(q["p"])); put("frzVgapT",f(q["tost"],4))
put("gapEntMin", f(ENT_W64["gap"].min(),4)); put("gapEntMean", f(ENT_W64["gap"].mean(),4))

# power at width 64, gap - uniform
d=_ok(acc("cifar100",64,"gap")-acc("cifar100",64,"uniform")); sd=d.std(ddof=1); _n=len(d); crit=stats.t.ppf(.975,_n-1)
put("powsd",f(sd)); put("mde",f(crit*sd/math.sqrt(_n)))
rng=np.random.default_rng(0)
for tgt,key in [(0.25,"pwA"),(0.5,"pwB"),(1.0,"pwC")]:
    s=rng.normal(tgt,sd,(200000,_n)); t=s.mean(1)/(s.std(1,ddof=1)/math.sqrt(_n))
    put(key, f"{100*(np.abs(t)>crit).mean():.1f}")

# gap - static by condition
for ds,w,tag in [("cifar100",16,"CA"),("cifar100",32,"CB"),("cifar100",64,"CC"),("tinyimagenet",32,"TA"),("tinyimagenet",64,"TB")]:
    q=paired(acc(ds,w,"gap"),acc(ds,w,"static"))
    put(f"eff{tag}",sg(q["mean"])); put(f"eff{tag}lo",sg(q["lo"])); put(f"eff{tag}hi",sg(q["hi"]))
    put(f"eff{tag}p", "<10^{-4}" if q["p"]<1e-4 else "="+f(q["p"],4))
    b=paired(acc(ds,w,"uniform"),acc(ds,w,"static")); r=paired(acc(ds,w,"gap"),acc(ds,w,"uniform"))
    put(f"bank{tag}",sg(b["mean"])); put(f"bank{tag}T",f(b["tost"],4))
    put(f"rout{tag}",sg(r["mean"]))
    put(f"loss{tag}",f(LOSS[(ds,w,4,"const")]["static"],4))

# matched budgets
for ds,wg,ws,tag in [("cifar100",16,32,"MA"),("cifar100",32,64,"MB"),("tinyimagenet",32,64,"MC")]:
    q=paired(acc(ds,ws,"static"),acc(ds,wg,"gap")); put(f"m{tag}",sg(q["mean"]))
    put(f"m{tag}lo",sg(q["lo"])); put(f"m{tag}hi",sg(q["hi"]))
    put(f"t{tag}g",f(MINS[(ds,wg,4,"const")]["gap"],1)); put(f"t{tag}s",f(MINS[(ds,ws,4,"const")]["static"],1))
put("k8vs64", f(acc("cifar100",64,"static").mean()-acc("cifar100",32,"gap",k=8).mean(),2))

# K sweep
q=paired(acc("cifar100",32,"gap",k=8),acc("cifar100",32,"gap",k=2)); put("k82",sg(q["mean"])); put("k82p",f(q["p"]))
put("kpar",f(PARAMS[("cifar100",32,8,"const")]["gap"]/PARAMS[("cifar100",32,2,"const")]["gap"],1))
put("ktime",f(MINS[("cifar100",32,8,"const")]["gap"]/MINS[("cifar100",32,2,"const")]["gap"],1))

# prediction
q=paired(acc("tinyimagenet",64,"gap"),acc("tinyimagenet",64,"static"))
put("predP", f"{q['p']*1e5:.1f}\\times10^{{-5}}")
put("tin64r", f(100*paired(acc("tinyimagenet",64,"gap"),acc("tinyimagenet",64,"uniform"))["mean"]/q["mean"],1))

# temperature
for w,tag in [(16,"A"),(32,"B"),(64,"C")]:
    an=acc("cifar100",w,"gap",rec="anneal"); c=acc("cifar100",w,"gap")
    a1=paired(an,c); s2=paired(an,acc("cifar100",w,"static")); u2=paired(an,acc("cifar100",w,"uniform"))
    put(f"an{tag}",sg(a1["mean"])); put(f"an{tag}p",f(a1["p"],4))
    put(f"ans{tag}",sg(s2["mean"])); put(f"ans{tag}T",f(s2["tost"],4))
    put(f"anu{tag}",sg(u2["mean"])); put(f"anu{tag}p",f(u2["p"],4)); put(f"anu{tag}T",f(u2["tost"],4))
    put(f"hc{tag}",f(1-ENT[("cifar100",w,"const")].mean(),5)); put(f"ha{tag}",f(1-ENT[("cifar100",w,"anneal")].mean(),5))
    put(f"fl{tag}",f"{(1-ENT[('cifar100',w,'const')].mean())/(1-ENT[('cifar100',w,'anneal')].mean()):.0f}")
    put(f"la{tag}",f(LOSS[("cifar100",w,4,"anneal")]["gap"],4)); put(f"lc{tag}",f(LOSS[("cifar100",w,4,"const")]["gap"],4))
    bc=paired(acc("cifar100",w,"gap"),acc("cifar100",w,"uniform")); put(f"bc{tag}",sg(bc["mean"]))
q=paired(acc("cifar100",64,"gap"),acc("cifar100",64,"static")); put("constT64",f(q["tost"],4))
# weight span implied by entropy at w16 annealed
dA=1-ENT[("cifar100",16,"anneal")].mean(); rms=math.sqrt(dA/(4/(2*math.log(4)))/4)
put("spanlo",f(0.25-rms,3)); put("spanhi",f(0.25+rms,3)); put("spanpct",f"{100*rms/0.25:.0f}")

out=["\\newcommand{\\V}[1]{\\csname v#1\\endcsname}"]
for k,v in M.items(): out.append(f"\\expandafter\\def\\csname v{k}\\endcsname{{{v}}}")
open("macros.tex","w").write("\n".join(out)+"\n")
print(len(M),"macros"); 
for k in ["nruns","gpuh","spread64","anovaF","anovaP","mde","effTB","predP","tin64r","mMC","k82","flA","spanlo","spanhi","spanpct","anuCT","constT64","ansCT"]:
    print(f"  {k:<10} {M[k]}")

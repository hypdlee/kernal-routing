import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, math
from data import *
import plotstyle as PS
R18 = True   # the paper's generators were written for ResNet-18 only
if not hasattr(plt, "_orig_savefig"): plt._orig_savefig = plt.savefig
_savefig = plt._orig_savefig
def _save(path, **k):
    plt.gcf().tight_layout(); PS.check(plt.gcf(), path); _savefig(path, **k)
plt.savefig = _save
PS.apply()
C={"static":"#444444","uniform":"#1f77b4","gap":"#d62728","shuffled":"#2ca02c",
   "frozen_random":"#9467bd","ann":"#ff7f0e","tin":"#e377c2"}
W=[16,32,64]
def eb(ax,x,q,**k):
    ax.errorbar(x,[e["mean"] for e in q],yerr=[[e["mean"]-e["lo"] for e in q],[e["hi"]-e["mean"] for e in q]],capsize=3,**k)

# F1 CIFAR capacity sweep
f,ax=plt.subplots(1,2,figsize=(7.4,2.8))
for a in ["static","uniform","gap"]:
    ax[0].errorbar(W,[acc("cifar100",w,a).mean() for w in W],yerr=[acc("cifar100",w,a).std(ddof=1) for w in W],
                   marker="o",ms=4,capsize=3,lw=1.4,color=C[a],label=a)
for x in ax: x.set_xscale("log",base=2); x.set_xticks(W); x.set_xticklabels(W); x.set_xlabel("base width $c$")
ax[0].set_ylabel("top-1 accuracy (%)"); ax[0].set_title("(a) accuracy")
for a,m in [("uniform","s"),("gap","o")]:
    eb(ax[1],W,[paired(acc("cifar100",w,a),acc("cifar100",w,"static")) for w in W],marker=m,ms=4,lw=1.4,color=C[a],label=f"{a} $-$ static")
ax[1].axhline(0,color="k",lw=.8); ax[1].axhspan(-.5,.5,color="grey",alpha=.13,label="$\\pm0.5$ pp margin")
ax[1].set_ylabel("$\\Delta$ vs static (pp)"); ax[1].set_title("(b) paired difference, 95% CI")
plt.tight_layout(); PS.place_legend(ax[0]); PS.place_legend(ax[1])
plt.savefig("fig/f1_sweep.pdf"); plt.close()

# F2 every CIFAR run
f,axs=plt.subplots(1,3,figsize=(7.4,2.7))
for k,w in enumerate(W):
    arms=["static","uniform","gap"]+(["shuffled","frozen_random"] if w==64 else [])
    for i,a in enumerate(arms):
        v=acc("cifar100",w,a); axs[k].scatter([i]*6,v,s=24,color=C[a],alpha=.85,zorder=3)
        axs[k].hlines(v.mean(),i-.3,i+.3,color="k",lw=1.5,zorder=4)
    axs[k].set_xticks(range(len(arms))); axs[k].set_xticklabels(arms,rotation=35,ha="right",fontsize=7)
    axs[k].set_title(f"width {w}")
axs[0].set_ylabel("top-1 accuracy (%)")
plt.tight_layout(); plt.savefig("fig/f2_allruns.pdf"); plt.close()

# F3 width-64 five arms + entropy per seed
f,ax=plt.subplots(1,2,figsize=(7.4,2.7))
arms=["static","uniform","gap","shuffled","frozen_random"]
for i,a in enumerate(arms):
    v=acc("cifar100",64,a); ax[0].scatter([i]*6,v,s=24,color=C[a],alpha=.85,zorder=3)
    ax[0].hlines(v.mean(),i-.3,i+.3,color="k",lw=1.5,zorder=4)
s=acc("cifar100",64,"static").mean(); ax[0].axhspan(s-.5,s+.5,color="grey",alpha=.13)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(arms,rotation=35,ha="right",fontsize=7)
ax[0].set_ylabel("top-1 accuracy (%)"); ax[0].set_title("(a) CIFAR-100, width 64")
for i,a in enumerate(["uniform","gap","shuffled","frozen_random"]):
    ax[1].scatter([i]*6,1-ENT_W64[a]+1e-6,s=24,color=C[a],zorder=3)
ax[1].set_yscale("log"); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["uniform","gap","shuffled","frozen_random"],rotation=35,ha="right",fontsize=7)
ax[1].set_ylabel("$1-\\tilde H$ per seed (log)"); ax[1].set_title("(b) routing flatness by arm")
plt.tight_layout(); plt.savefig("fig/f3_w64.pdf"); plt.close()

# F4 decomposition, five conditions
cond=[("C16","cifar100",16),("C32","cifar100",32),("C64","cifar100",64),("T32","tinyimagenet",32),("T64","tinyimagenet",64)]
f,ax=plt.subplots(figsize=(6.4,2.8)); x=np.arange(5); bw=.36
for j,(lbl,fn,col) in enumerate([("bank (uniform $-$ static)",lambda d,w:paired(acc(d,w,"uniform"),acc(d,w,"static")),C["uniform"]),
                                 ("routing (gap $-$ uniform)",lambda d,w:paired(acc(d,w,"gap"),acc(d,w,"uniform")),C["gap"])]):
    q=[fn(d,w) for _,d,w in cond]
    ax.bar(x+(j-.5)*bw,[e["mean"] for e in q],bw,color=col,label=lbl,
           yerr=[[e["mean"]-e["lo"] for e in q],[e["hi"]-e["mean"] for e in q]],capsize=3,error_kw=dict(lw=1))
ax.axhline(0,color="k",lw=.8); ax.set_xticks(x)
ax.set_xticklabels(["CIFAR-100\nw16","CIFAR-100\nw32","CIFAR-100\nw64","Tiny-ImageNet\nw32","Tiny-ImageNet\nw64"],fontsize=7.5)
ax.set_ylabel("$\\Delta$ accuracy (pp)"); plt.tight_layout(); PS.place_legend(ax)
plt.savefig("fig/f4_decomp.pdf"); plt.close()

# F5 matched budgets
f,ax=plt.subplots(1,2,figsize=(7.4,3.0))
lab5=[[],[]]
for k,(ds,ws,ttl) in enumerate([("cifar100",W,"(a) CIFAR-100"),("tinyimagenet",[32,64],"(b) Tiny-ImageNet")]):
    for a in ["static","uniform","gap"]:
        p=[PARAMS[(ds,w,4,"const")][a] for w in ws]; m=[acc(ds,w,a).mean() for w in ws]
        ax[k].plot(p,m,marker="o",ms=4,lw=1.3,color=C[a],label=a)
        lab5[k]+= [(pp,mm,f"w{w}") for w,pp,mm in zip(ws,p,m)]
    pairs=[(16,32),(32,64)] if ds=="cifar100" else [(32,64)]
    for wg,wsx in pairs:
        xg,yg=PARAMS[(ds,wg,4,"const")]["gap"],acc(ds,wg,"gap").mean()
        xs,ys=PARAMS[(ds,wsx,4,"const")]["static"],acc(ds,wsx,"static").mean()
        ax[k].plot([xg,xs],[yg,ys],ls=":",color="k",lw=1)
        ratio=xg/xs
        lab=f"${ys-yg:+.2f}$" if abs(ratio-1)<=0.10 else f"${ys-yg:+.2f}$\n(params $\\times{ratio:.2f}$)"
        lab5[k].append((math.sqrt(xg*xs),(yg+ys)/2,lab))
    ax[k].set_xscale("log"); ax[k].set_xlabel("stored parameters (M, log)"); ax[k].set_title(ttl)
ax[0].set_ylabel("top-1 accuracy (%)")
plt.tight_layout()
for k in range(2):
    PS.place_legend(ax[k]); PS.place_labels(ax[k],lab5[k])
plt.savefig("fig/f5_matched.pdf"); plt.close()

# F6 K sweep
f,ax=plt.subplots(1,2,figsize=(7.2,2.6)); ks=[2,4,8]
eb(ax[0],ks,[paired(acc("cifar100",32,"gap",k=k),acc("cifar100",32,"static")) for k in ks],marker="o",ms=5,lw=1.4,color=C["gap"])
ax[0].axhline(0,color="k",lw=.8)
if R18: ax[0].set_ylim(0,1.8)
ax2=ax[1].twinx()
ax[1].plot(ks,[PARAMS[("cifar100",32,k,"const")]["gap"] for k in ks],marker="s",color=C["uniform"])
ax2.plot(ks,[MINS[("cifar100",32,k,"const")]["gap"] for k in ks],marker="^",ls="--",color=C["shuffled"]); ax2.grid(False)
for x in ax: x.set_xscale("log",base=2); x.set_xticks(ks); x.set_xticklabels(ks); x.set_xlabel("bank size $K$")
ax[0].set_ylabel("gap $-$ static (pp)"); ax[0].set_title("(a) benefit")
ax[1].set_ylabel("stored parameters (M)",color=C["uniform"]); ax2.set_ylabel("minutes per run",color=C["shuffled"])
ax[1].set_title("(b) cost")
plt.tight_layout(); plt.savefig("fig/f6_k.pdf"); plt.close()
print("F1-F6 written")

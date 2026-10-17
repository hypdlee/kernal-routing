import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, math
from data import *
plt.rcParams.update({"font.size":8.5,"axes.grid":True,"grid.alpha":0.25,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200,
    "savefig.bbox":"tight","font.family":"DejaVu Sans","legend.frameon":False})
C={"static":"#444444","uniform":"#1f77b4","gap":"#d62728","ann":"#ff7f0e","tin":"#e377c2","cif":"#7f7f7f"}
W=[16,32,64]

# F7 Tiny-ImageNet: every run, and effect vs CIFAR
f,ax=plt.subplots(1,2,figsize=(7.4,2.8))
for k,w in enumerate([32,64]):
    for i,a in enumerate(["static","uniform","gap"]):
        v=acc("tinyimagenet",w,a); xs=np.full(6,i+4*k)
        ax[0].scatter(xs,v,s=22,color=C[a],alpha=.85,zorder=3,label=a if k==0 else None)
        ax[0].hlines(v.mean(),i+4*k-.3,i+4*k+.3,color="k",lw=1.4,zorder=4)
ax[0].set_xticks([1,5]); ax[0].set_xticklabels(["width 32","width 64"])
ax[0].set_ylabel("top-1 accuracy (%)"); ax[0].legend(fontsize=7); ax[0].set_title("(a) all 36 Tiny-ImageNet runs",fontsize=9)
for ds,col,lbl,ws in [("cifar100",C["cif"],"CIFAR-100",W),("tinyimagenet",C["tin"],"Tiny-ImageNet",[32,64])]:
    q=[paired(acc(ds,w,"gap"),acc(ds,w,"static")) for w in ws]
    ax[1].errorbar(ws,[e["mean"] for e in q],yerr=[[e["mean"]-e["lo"] for e in q],[e["hi"]-e["mean"] for e in q]],
                   marker="o",ms=5,capsize=3,lw=1.4,color=col,label=lbl)
ax[1].axhline(0,color="k",lw=.8); ax[1].axhspan(-.5,.5,color="grey",alpha=.13)
ax[1].set_xscale("log",base=2); ax[1].set_xticks(W); ax[1].set_xticklabels(W)
ax[1].set_xlabel("base width $c$"); ax[1].set_ylabel("gap $-$ static (pp)")
ax[1].legend(fontsize=7.5); ax[1].set_title("(b) the width-64 null does not replicate",fontsize=9)
plt.savefig("fig/f7_tin.pdf"); plt.close()

# F8 the failed prediction: curves + loss vs effect
f,ax=plt.subplots(1,2,figsize=(7.4,2.8))
for (ds,w,a),ls,col in [(("cifar100",16,"static"),":",C["cif"]),(("cifar100",32,"static"),"--",C["cif"]),
                        (("cifar100",64,"static"),"-",C["cif"]),(("tinyimagenet",64,"static"),"-",C["tin"]),
                        (("tinyimagenet",64,"gap"),"--",C["tin"])]:
    ax[0].plot(EP,CURVE[(ds,w,a)],ls,color=col,lw=1.3,label=f"{'CIFAR' if ds=='cifar100' else 'TIN'} w{w} {a}")
ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss (log)")
ax[0].legend(fontsize=6.3); ax[0].set_title("(a) training loss, seed 0",fontsize=9)
pts=[("cifar100",16),("cifar100",32),("cifar100",64),("tinyimagenet",32),("tinyimagenet",64)]
for ds,w in pts:
    q=paired(acc(ds,w,"gap"),acc(ds,w,"static")); L=LOSS[(ds,w,4,"const")]["static"]
    col=C["tin"] if ds=="tinyimagenet" else C["cif"]
    ax[1].errorbar(L,q["mean"],yerr=[[q["mean"]-q["lo"]],[q["hi"]-q["mean"]]],marker="o",ms=5,capsize=3,color=col)
    ax[1].annotate(f"{'TIN' if ds=='tinyimagenet' else 'C'} w{w}",(L,q["mean"]),xytext=(5,-3),textcoords="offset points",fontsize=6.5)
ax[1].axvspan(0.004,0.05,color="#ffe08a",alpha=.35,label="predicted null region")
ax[1].axhline(0,color="k",lw=.8); ax[1].set_xscale("log")
ax[1].set_xlabel("static arm final training loss (log)"); ax[1].set_ylabel("gap $-$ static (pp)")
ax[1].legend(fontsize=7,loc="upper left"); ax[1].set_title("(b) loss does not order the effect",fontsize=9)
plt.savefig("fig/f8_prediction.pdf"); plt.close()

# F9 temperature
f,ax=plt.subplots(1,3,figsize=(7.8,2.7)); x=np.arange(3); bw=.36
for j,(lbl,rec,col) in enumerate([("$\\tau=1$","const",C["gap"]),("$\\tau:30\\to1$","anneal",C["ann"])]):
    q=[paired(acc("cifar100",w,"gap",rec=rec),acc("cifar100",w,"static")) for w in W]
    ax[0].bar(x+(j-.5)*bw,[e["mean"] for e in q],bw,color=col,label=lbl,
              yerr=[[e["mean"]-e["lo"] for e in q],[e["hi"]-e["mean"] for e in q]],capsize=3,error_kw=dict(lw=1))
    ax[1].plot(W,[1-ENT[("cifar100",w,rec)].mean() for w in W],marker="o",color=col,label=lbl)
    xs=[1-ENT[("cifar100",w,rec)].mean() for w in W]
    ys=[(acc("cifar100",w,"gap",rec=rec)-acc("cifar100",w,"uniform")).mean() for w in W]
    ax[2].scatter(xs,ys,s=36,color=col,label=lbl,zorder=3)
    for w,xx,yy in zip(W,xs,ys): ax[2].annotate(f"w{w}",(xx,yy),xytext=(4,-8),textcoords="offset points",fontsize=6.5)
ax[0].axhline(0,color="k",lw=.8); ax[0].axhspan(-.5,.5,color="grey",alpha=.13)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"w{w}" for w in W]); ax[0].set_ylabel("gap $-$ static (pp)")
ax[0].legend(fontsize=7); ax[0].set_title("(a) effect, both recipes",fontsize=9)
ax[1].set_yscale("log"); ax[1].set_xscale("log",base=2); ax[1].set_xticks(W); ax[1].set_xticklabels(W)
ax[1].set_xlabel("width"); ax[1].set_ylabel("$1-\\tilde H$ (log)"); ax[1].legend(fontsize=7)
ax[1].set_title("(b) router flatness",fontsize=9)
ax[2].axhline(0,color="k",lw=.8); ax[2].set_xscale("log"); ax[2].set_xlabel("$1-\\tilde H$ (log)")
ax[2].set_ylabel("gap $-$ uniform (pp)"); ax[2].legend(fontsize=7); ax[2].set_title("(c) flatness vs benefit",fontsize=9)
plt.savefig("fig/f9_temp.pdf"); plt.close()

# F10 entropy counterexample
f,ax=plt.subplots(1,2,figsize=(6.8,2.8)); lab=["$\\tau=1$","$\\tau:30\\to1$"]
e=[1-ENT[("cifar100",16,r)].mean() for r in ["const","anneal"]]
b=[(acc("cifar100",16,"gap",rec=r)-acc("cifar100",16,"uniform")).mean() for r in ["const","anneal"]]
r0=ax[0].bar(lab,e,color=[C["gap"],C["ann"]],width=.55)
ax[0].set_yscale("log"); ax[0].set_ylim(1e-3,2e-1)
for rr,v in zip(r0,e): ax[0].annotate(f"{v:.5f}",(rr.get_x()+rr.get_width()/2,v*1.25),ha="center",fontsize=7)
ax[0].set_ylabel("$1-\\tilde H$  (log; lower = flatter)")
ax[0].set_title(f"(a) flatness: annealed is {e[0]/e[1]:.0f}x flatter",fontsize=8.5)
r1=ax[1].bar(lab,b,color=[C["gap"],C["ann"]],width=.55); ax[1].set_ylim(0,1.15)
for rr,v in zip(r1,b): ax[1].annotate(f"+{v:.3f} pp",(rr.get_x()+rr.get_width()/2,v+.03),ha="center",fontsize=7.5)
ax[1].set_ylabel("gap $-$ uniform (pp)"); ax[1].set_title("(b) benefit over uniform",fontsize=8.5)
plt.tight_layout(); plt.savefig("fig/f10_entropy.pdf"); plt.close()
print("F7-F10 written")

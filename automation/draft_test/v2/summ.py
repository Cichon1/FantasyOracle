import sys, pandas as pd, numpy as np
for f in sys.argv[1:]:
    R=pd.read_csv(f); S=R.groupby(["season","cfg"])[["pf","po"]].mean().reset_index()
    b=S[S.cfg=="c_adp|None"].set_index("season"); out=[]
    for c,g in S.groupby("cfg"):
        g=g.set_index("season"); d=(g[["pf","po"]]-b[["pf","po"]])*100
        out.append(dict(cfg=c, d_pf=d.pf.mean(), se_pf=d.pf.std()/np.sqrt(len(d)), d_po=d.po.mean(), se_po=d.po.std()/np.sqrt(len(d)), seasons_better=int((d.pf>0).sum()), n=len(d)))
    print(f); print(pd.DataFrame(out).round(2).to_string(index=False))

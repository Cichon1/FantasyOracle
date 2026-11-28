"""Does the test detect a deliberately bad drafter? Buggy waiver rule (as run 1) vs fixed rule."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "" + __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))) + "")
import tournament as T

def add_ros(D):
    pts, P = D["pts"], D["P"]; played = D["proj"] >= 0
    cum_pts = np.vstack([np.zeros(len(P)), np.cumsum(pts*played,0)[:-1]])
    cum_g = np.vstack([np.zeros(len(P)), np.cumsum(played,0)[:-1]])
    ppg = (cum_pts + T.SHRINK_K*P["exp_ppg"].values)/(cum_g+T.SHRINK_K)
    # consecutive weeks missed BEFORE week w (known at waiver time)
    out = np.zeros_like(ppg)
    for w in range(1, ppg.shape[0]):
        out[w] = np.where(~played[w-1], out[w-1]+1, 0)
    D["ros"] = np.where(out >= 3, 0.0, ppg)      # 3+ straight missed games: treat as droppable
    return D

def run_season_fixed(D, rosters, avail, sched):
    """same as T.run_season but add/drop judged on rest-of-season value, not this week's (bye = 0)"""
    orig = D["proj"]; ros = D["ros"]
    rosters=[list(r) for r in rosters]; avail=avail.copy(); posi=D["pos"]
    wins,pf=np.zeros(12),np.zeros(12)
    for w in range(17):
        if w>=1:
            for t in np.lexsort((pf,wins)):
                fa=np.where(avail)[0]; best,move=1.0,None
                for p in range(4):
                    f=fa[posi[fa]==p]; mine=[i for i in rosters[t] if posi[i]==p]
                    if not len(f) or not mine: continue
                    c=f[np.argmax(ros[w][f])]; worst=min(mine,key=lambda i: ros[w][i])
                    g=ros[w][c]-ros[w][worst]
                    if g>best: best,move=g,(c,worst)
                if move:
                    a,d=move; rosters[t].remove(d); rosters[t].append(a); avail[a],avail[d]=False,True
        sc=np.array([T.lineup_score(rosters[t],w,D) for t in range(12)])
        if w<14:
            for a,b in sched[w]:
                wins[a]+=sc[a]>sc[b]; wins[b]+=sc[b]>sc[a]
            pf+=sc
    # playoffs reuse the original logic by re-running only the bracket: approximate with seeds + weeks 15-17
    seeds=list(np.lexsort((pf,wins))[::-1]); n=T.PLAYOFF[D["lg"]]
    return wins,pf,set(seeds[:n])

if __name__ == "__main__":
    CFGS=[("c_adp",None),("c_noisy",None),("bpa","adp"),("zero_rb","adp"),("robust_rb","adp"),("late_qb","adp")]
    rows=[]; t0=time.time()
    seasons=[int(a) for a in sys.argv[1:]] or list(range(2014,2026))
    for y in seasons:
        D=add_ros(T.season_data(y,"bdsm"))
        for slot in range(0,12,1):
            for s in range(3):
                rng=np.random.default_rng([y,slot,s,7]); z=rng.standard_normal(len(D["P"])); zb=rng.standard_normal(len(D["P"]))
                sched=T.schedule(rng)
                for cfg in CFGS:
                    r,av=T.run_draft(D,slot,cfg,z,zb,{})
                    w1,pf1,po1,_=T.run_season(D,r,av,sched)
                    w2,pf2,po2=run_season_fixed(D,r,av,sched)
                    rows.append(dict(season=y,slot=slot,sim=s,cfg=cfg[0],
                        pf_bug=pf1[slot]/pf1.mean(), po_bug=float(slot in po1),
                        pf_fix=pf2[slot]/pf2.mean(), po_fix=float(slot in po2)))
        print(y, f"{time.time()-t0:.0f}s", flush=True)
    R=pd.DataFrame(rows); R.to_csv(__import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/results/sensitivity.csv",index=False)
    S=R.groupby(["season","cfg"])[["pf_bug","po_bug","pf_fix","po_fix"]].mean().reset_index()
    base=S[S.cfg=="c_adp"].set_index("season")
    out=[]
    for c,g in S.groupby("cfg"):
        g=g.set_index("season")
        d={k:(g[k]-base[k]) for k in ["pf_bug","po_bug","pf_fix","po_fix"]}
        out.append(dict(cfg=c, **{f"d_{k}":round(v.mean()*100,2) for k,v in d.items()},
                        **{f"se_{k}":round(v.std(ddof=1)/np.sqrt(len(v))*100,2) for k,v in d.items()}))
    print(pd.DataFrame(out).to_string(index=False))

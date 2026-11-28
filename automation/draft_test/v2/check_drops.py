import sys, numpy as np, pandas as pd
sys.path.insert(0, "" + __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))) + "")
import tournament as T

def run_season_logged(D, rosters, avail, sched, log):
    rosters=[list(r) for r in rosters]; avail=avail.copy()
    posi, proj = D["pos"], D["proj"]; P=D["P"]
    wins,pf=np.zeros(12),np.zeros(12)
    for w in range(1,14):
        for t in np.lexsort((pf,wins)):
            fa=np.where(avail)[0]; best_gain,move=1.0,None
            for p in range(4):
                f=fa[posi[fa]==p]; mine=[i for i in rosters[t] if posi[i]==p]
                if not len(f) or not mine: continue
                cand=f[np.argmax(proj[w][f])]; worst=min(mine,key=lambda i: proj[w][i])
                g=proj[w][cand]-max(proj[w][worst],0)
                if g>best_gain: best_gain,move=g,(cand,worst)
            if move:
                a,d=move
                # how good is the dropped player for the REST of the season?
                ros_actual = D["pts"][w:14, d].sum(); ros_add = D["pts"][w:14, a].sum()
                log.append(dict(week=w+1, dropped=P["player"].iat[d], drop_adp=P["adp_ffc"].iat[d],
                                drop_inactive=proj[w][d] < 0, drop_ros=ros_actual, add_ros=ros_add))
                rosters[t].remove(d); rosters[t].append(a); avail[a],avail[d]=False,True
        sc=np.array([T.lineup_score(rosters[t],w,D) for t in range(12)]); pf+=sc
    return rosters

allrows=[]
for y in [2019, 2023]:
    D=T.season_data(y,"bdsm")
    for s in range(3):
        rng=np.random.default_rng([y,0,s,7]); z=rng.standard_normal(len(D["P"]))
        sched=T.schedule(rng)
        rosters,avail=T.run_draft(D,0,("c_adp",None),z,z,{})
        log=[]; run_season_logged(D,rosters,avail,sched,log)
        for r in log: r["season"]=y
        allrows+=log
L=pd.DataFrame(allrows)
print("moves:",len(L)," per league-season:",len(L)/6)
print("share of drops where the dropped player was inactive that week (bye/injury):", round(L.drop_inactive.mean(),3))
top=L[L.drop_adp<=60]
print("drops of top-60 ADP players:",len(top)," per league-season:",round(len(top)/6,1))
print("dropped player ROS pts vs added player ROS pts (mean):", round(L.drop_ros.mean(),1), round(L.add_ros.mean(),1))
print(top.sort_values("drop_adp").head(15).to_string(index=False))

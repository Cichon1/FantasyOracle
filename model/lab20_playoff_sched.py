"""Lab 20 (Nov 28, 2026): how much of a fantasy playoff schedule (Weeks 15-17) carries into points.

Setup: at a cutoff after Week 12, rate each defense by points allowed to a position (skill) or points scored against
it by DEFs / kickers, shrunk toward average with PRIOR games; predict each player's / unit's Weeks 15-17 points as
baseline x (1 + b x (ratio - 1)). Baseline: Weeks 1-12 points per game (players with 6+ games, 6+ PPG) or the unit's
exponential average (half-life 6). b fit on 2019-2022 (K / DEF: 2021-2022), checked on 2023-2025.

Results (prior 8; used by automation/playoff_sched.py):
  skill pooled b 0.56 (train); test RMSE by position, b=0 -> b=0.5: QB 8.58 -> 8.52, RB 8.15 -> 8.14,
    WR 7.85 -> 7.85, TE 6.91 -> 6.97 (per-position b noisy: train QB .29 RB 1.05 WR .27 TE .79)
  DEF b 0.87 train / 1.26 test; RMSE 6.26 -> 6.09 (b .5) -> 6.00 (b 1): the schedule matters most for defenses
  K   b 0.43 train / 0.14 test; RMSE 4.78 -> 4.79: no gain, kicker schedules are noise

  python lab20_playoff_sched.py <data dir with week_YYYY.parquet, games.parquet, pbp/>
"""
import sys
import pandas as pd, numpy as np
D=sys.argv[1].rstrip("/")+"/"
CUT=12
def season(y, prior):
    w=pd.read_parquet(f"{D}week_{y}.parquet")
    w=w[(w.season_type=="REG")&w.position.isin(["QB","RB","WR","TE"])]
    pre=w[w.week<=CUT]
    a=pre.groupby(["opponent_team","week","position"]).fantasy_points_ppr.sum().reset_index()
    lg=a.groupby("position").fantasy_points_ppr.mean()
    st=a.groupby(["opponent_team","position"]).fantasy_points_ppr.agg(["sum","count"]).reset_index()
    st["r"]=((st["sum"]+prior*st.position.map(lg))/(st["count"]+prior))/st.position.map(lg)
    R=st.set_index(["opponent_team","position"]).r
    base=pre.groupby(["player_id","position"]).agg(ppg=("fantasy_points_ppr","mean"),n=("week","count")).reset_index()
    base=base[(base.n>=6)&(base.ppg>=6)]
    post=w[w.week.between(15,17)][["player_id","week","opponent_team","fantasy_points_ppr"]]
    x=post.merge(base,on="player_id")
    x["r"]=[R.get((o,p),1.0) for o,p in zip(x.opponent_team,x.position)]
    x["season"]=y
    return x
def evaluate(x,b):
    pred=x.ppg*(1+b*(x.r-1))
    return np.mean(np.abs(x.fantasy_points_ppr-pred))
for prior in [4,8,16]:
    tr=pd.concat([season(y,prior) for y in [2019,2020,2021,2022]]); te=pd.concat([season(y,prior) for y in [2023,2024,2025]])
    bs=np.round(np.arange(0,1.51,0.1),2)
    m=[evaluate(tr,b) for b in bs]; bb=bs[int(np.argmin(m))]
    # also slope fit
    tr2=tr.assign(y=tr.fantasy_points_ppr/tr.ppg-1,xx=tr.r-1)
    slope=(tr2.y*tr2.xx).sum()/(tr2.xx**2).sum()
    print(f"prior {prior}: best b train {bb} (ols {slope:.2f}) | test MAE base {evaluate(te,0):.3f} b {evaluate(te,bb):.3f} b1 {evaluate(te,1):.3f} | n test {len(te)} | sd r {te.r.std():.3f}")
    for pos in ["QB","RB","WR","TE"]:
        t=te[te.position==pos]; print("   ",pos, round(evaluate(t,0),3), round(evaluate(t,bb),3))
print("--- per position, prior 8")
tr=pd.concat([season(y,8) for y in [2019,2020,2021,2022]]); te=pd.concat([season(y,8) for y in [2023,2024,2025]])
for pos in ["QB","RB","WR","TE"]:
    t=tr[tr.position==pos]; yy=t.fantasy_points_ppr-t.ppg; xx=t.ppg*(t.r-1); b=(yy*xx).sum()/(xx**2).sum()
    e=te[te.position==pos]; yy2=e.fantasy_points_ppr-e.ppg; xx2=e.ppg*(e.r-1); b2=(yy2*xx2).sum()/(xx2**2).sum()
    rmse=lambda bb: np.sqrt(np.mean((e.fantasy_points_ppr-e.ppg*(1+bb*(e.r-1)))**2))
    print(pos,"train b",round(b,2),"test b",round(b2,2),"test rmse base",round(rmse(0),3),"trainb",round(rmse(b),3), "b=0.5",round(rmse(.5),3))

# ---- K / DEF
sys.path.insert(0,"/home/claude/fantasyoracle/model")
import pandas as pd, numpy as np, lab17_kdef as K
K.DATA=sys.argv[1]
g=pd.read_parquet(sys.argv[1]+"/games.parquet")
CUT=12
def kseason(S,prior):
    u=K.season_units(S,g); u=u[u.played] if "played" in u else u
    pre=u[u.week<=CUT].copy(); post=u[u.week.between(15,17)].copy()
    out=[]
    for col in ["def_pts","k_pts"]:
        m=pre[col].mean(); x=pre.groupby("opp")[col].agg(["sum","count"])
        R=((x["sum"]+prior*m)/(x["count"]+prior))/m
        base=pre.sort_values("week").groupby("team")[col].apply(lambda s: s.ewm(halflife=6).mean().iloc[-1])
        p=post[["team","opp",col]].copy(); p["base"]=p.team.map(base); p["r"]=p.opp.map(R); p["unit"]=col; p["y"]=p[col]
        out.append(p[["unit","base","r","y"]])
    return pd.concat(out)
for prior in [4,8,16]:
    tr=pd.concat([kseason(S,prior) for S in [2021,2022]]); te=pd.concat([kseason(S,prior) for S in [2023,2024,2025]])
    for unit in ["def_pts","k_pts"]:
        t=tr[tr.unit==unit]; e=te[te.unit==unit]
        rm=lambda d,b: np.sqrt(np.mean((d.y-d.base*(1+b*(d.r-1)))**2))
        fit=lambda d: ((d.y-d.base)*(d.base*(d.r-1))).sum()/((d.base*(d.r-1))**2).sum()
        from scipy.stats import spearmanr
        print(prior,unit,"b train",round(fit(t),2),"b test",round(fit(e),2),"test rmse base",round(rm(e,0),3),"b=.5",round(rm(e,.5),3),"b=1",round(rm(e,1),3), "sd r",round(e.r.std(),3))

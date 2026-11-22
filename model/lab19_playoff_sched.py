"""Playoff-schedule check (Nov 22, 2026): Weeks 15-17 matchups for Jackson's rosters.

Skill players: opponent's 2026 PPR allowed to the position per game (Weeks 1-10), shrunk toward the league average
(8 games of prior), as a ratio to average. DEF / K: points team defenses / kickers scored against that opponent
(lab17 units, Yahoo default), shrunk the same way. Ratios > 1 = softer matchup.
"""
import sys
import pandas as pd, numpy as np
import lab17_kdef as K

S, LAST = 2026, 10
w = pd.read_parquet("data/week_2026.parquet")
w = w[(w.week <= LAST) & w.position.isin(["QB", "RB", "WR", "TE"])]
# per defense-game, total PPR allowed to each position
a = w.groupby(["opponent_team", "week", "position"])["fantasy_points_ppr"].sum().reset_index()
lg = a.groupby("position")["fantasy_points_ppr"].mean()
st = a.groupby(["opponent_team", "position"])["fantasy_points_ppr"].agg(["sum", "count"]).reset_index()
PRIOR = 8
st["adj"] = (st["sum"] + PRIOR * st["position"].map(lg)) / (st["count"] + PRIOR)
st["ratio"] = st["adj"] / st["position"].map(lg)
allowed = st.set_index(["opponent_team", "position"])["ratio"]

u = K.build(S, "data")
u = u[(u.season == S) & (u.week <= LAST)]
kd = {}
for col in ["def_pts", "k_pts"]:
    x = u.groupby("opp")[col].agg(["sum", "count"])
    m = u[col].mean()
    kd[col] = ((x["sum"] + PRIOR * m) / (x["count"] + PRIOR)) / m
# against an offense: def_pts scored by DEFs facing team T = rows where opp == T  -> groupby opp above is "the DEF's
# opponent" = the offense; for kickers we want the kicker's opponent defense: k_pts grouped by opp = points kickers
# scored against defense `opp`. Both keyed by the opponent of the unit, which is what we need.

g = pd.read_parquet("data/games.parquet")
g = g[(g.season == S) & g.week.between(15, 17)]
sched = pd.concat([g.assign(team=g.home_team, opp=g.away_team, home=1), g.assign(team=g.away_team, opp=g.home_team, home=0)])
sched = sched[["week", "team", "opp", "home", "roof", "gameday"]]

r = pd.read_csv("../docs/data/league_rosters_v5.csv")
lgs = {200436: "The League", 727092: "Ezekiel 25.17", 126912: "BDSM League"}
out = []
for lid, name in lgs.items():
    mine = r[(r.league_id == lid) & (r.is_me == 1)]
    for _, p in mine.iterrows():
        if p.kind not in ("skill", "K", "DEF"):
            continue
        for wk in (15, 16, 17):
            m = sched[(sched.team == p.nfl) & (sched.week == wk)]
            if not len(m):
                out.append(dict(league=name, player=p.player, pos=p.pos, slot=p.slot, week=wk, opp="BYE", ratio=np.nan))
                continue
            m = m.iloc[0]
            if p.kind == "skill":
                ratio = allowed.get((m.opp, p.pos), np.nan)
            elif p.kind == "DEF":
                ratio = kd["def_pts"].get(m.opp, np.nan)
            else:
                ratio = kd["k_pts"].get(m.opp, np.nan)
            out.append(dict(league=name, player=p.player, pos=p.pos, slot=p.slot, week=wk,
                            opp=("vs " if m.home else "@ ") + m.opp, ratio=ratio, roof=m.roof))
o = pd.DataFrame(out)
piv = o.pivot_table(index=["league", "pos", "player", "slot"], columns="week", values="ratio", aggfunc="first")
opp = o.pivot_table(index=["league", "pos", "player", "slot"], columns="week", values="opp", aggfunc="first")
piv["avg"] = piv[[15, 16, 17]].mean(axis=1)
res = piv.join(opp, rsuffix="_opp")
pd.set_option("display.width", 250)
for name in lgs.values():
    x = res.loc[name].sort_values("avg")
    print("\n==", name)
    print(x.round(2).to_string())

# streamable DEFs for the playoffs: every DEF unit's average matchup over Weeks 15-17 and who's free in The League / BDSM
rows = []
for t in sorted(sched.team.unique()):
    m = sched[sched.team == t].sort_values("week")
    rows.append(dict(team=t, **{f"w{int(x.week)}": ("vs " if x.home else "@ ") + x.opp for x in m.itertuples()},
                     d15=kd["def_pts"].get(m[m.week == 15].opp.iloc[0], np.nan) if (m.week == 15).any() else np.nan,
                     d16=kd["def_pts"].get(m[m.week == 16].opp.iloc[0], np.nan) if (m.week == 16).any() else np.nan,
                     d17=kd["def_pts"].get(m[m.week == 17].opp.iloc[0], np.nan) if (m.week == 17).any() else np.nan))
d = pd.DataFrame(rows)
d["avg"] = d[["d15", "d16", "d17"]].mean(axis=1)
own = r[r.kind == "DEF"]
for lid in (200436, 126912):
    held = set(own[own.league_id == lid].nfl)
    d[f"free_{lid}"] = ~d.team.isin(held)
# DEF quality itself: season DEF points per game
dq = u.groupby("team")["def_pts"].mean()
d["def_ppg"] = d.team.map(dq)
d["score"] = d["avg"] * d["def_ppg"]
print("\n== DEF playoff matchups (opp softness avg x own DEF ppg)")
print(d.sort_values("score", ascending=False).head(16).round(2).to_string(index=False))
o.to_csv("lab/playoff_sched_2026.csv", index=False)
d.to_csv("lab/playoff_def_2026.csv", index=False)

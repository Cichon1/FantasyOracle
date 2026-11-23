"""Hold or sell a short-term injured star? (Nov 23, 2026, Jackson: "if Puka Nacua is hurt for 2-3 weeks ... I would not
trade him for Drake London").

Event: snapshot week W (6..12), a fantasy starter (12+ PPR per game in 3+ games played that season before W) who didn't
play his team's last game, isn't on IR / reserve (nflverse weekly roster status RES) at W, and is still ranked in
FantasyPros' rest-of-season PPR rankings (snapshot before W's games). Comparison: healthy players (played the team's
last game) at the same position whom that snapshot ranked within 3 spots of him - what the market would hand you for him.
Outcome: PPR points after W through the fantasy regular season, in the fantasy playoffs (Weeks 15-17; 14-16 in 2020)
and over both. Seasons 2020 (from Week 7, first ROS snapshots) - 2025.

  python hold_lab.py data fpdata
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
FP = sys.argv[2] if len(sys.argv) > 2 else "fpdata"
POS = ["QB", "RB", "WR", "TE"]


def main():
    st = pd.concat([pd.read_parquet(f, columns=["player_id", "player_display_name", "position", "season", "week",
                                                "season_type", "team", "fantasy_points_ppr"])
                    for f in sorted(glob.glob(os.path.join(DATA, "week_20*.parquet")))], ignore_index=True)
    st = st[(st["season_type"] == "REG") & (st["season"].between(2020, 2025)) & st["position"].isin(POS)]
    st["season"], st["week"] = st["season"].astype(int), st["week"].astype(int)
    g = pd.read_parquet(os.path.join(DATA, "games.parquet"))
    g = g[(g["game_type"] == "REG") & g["season"].between(2020, 2025)]
    tw = pd.concat([g[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
                    g[["season", "week", "away_team"]].rename(columns={"away_team": "team"})])
    team_weeks = {(s, t): sorted(x["week"]) for (s, t), x in tw.groupby(["season", "team"])}
    pts = st.set_index(["player_id", "season", "week"])["fantasy_points_ppr"]
    ro = pd.concat([pd.read_parquet(f, columns=["season", "week", "gsis_id", "status"])
                    for f in sorted(glob.glob(os.path.join(DATA, "roster", "roster_weekly_20*.parquet")))], ignore_index=True)
    ro = ro.dropna(subset=["gsis_id"]).drop_duplicates(["season", "week", "gsis_id"], keep="last").set_index(["season", "week", "gsis_id"])["status"]
    # FantasyPros ROS positional ranks, latest snapshot before each week's first game
    e = pd.read_parquet(os.path.join(FP, "db_fpecr.parquet"))
    ids = pd.read_csv(os.path.join(FP, "db_playerids.csv"), usecols=["fantasypros_id", "gsis_id"], low_memory=False).dropna()
    ids["fantasypros_id"] = ids["fantasypros_id"].astype(int).astype(str)
    e = e[(e["ecr_type"] == "rp") & e["fp_page"].str.contains("ros-ppr-(?:rb|wr|te)|ros-qb", regex=True)]
    e = e.merge(ids.drop_duplicates("fantasypros_id"), left_on="id", right_on="fantasypros_id")
    e["pos"] = e["pos"].str.extract(r"^([A-Z]+)")[0]
    e["date"] = pd.to_datetime(e["scrape_date"])
    e["pos_rank"] = e.groupby(["fp_page", "scrape_date"])["ecr"].rank(method="first")
    first = g.groupby(["season", "week"])["gameday"].min().map(pd.to_datetime)
    rows = []
    for S in range(2020, 2026):
        po = [15, 16, 17] if S >= 2021 else [14, 15, 16]
        fs = st[st["season"] == S]
        for W in range(6, 13):
            if (S, W) not in first.index:
                continue
            snap = e[(e["date"] < first[(S, W)]) & (e["date"] >= first[(S, W)] - pd.Timedelta(days=8))]
            if not len(snap):
                continue
            snap = snap[snap["date"] == snap["date"].max()].drop_duplicates("gsis_id").set_index("gsis_id")
            before = fs[fs["week"] < W]
            ppg = before.groupby("player_id")["fantasy_points_ppr"].agg(["mean", "size"])
            last_team = before.sort_values("week").groupby("player_id")["team"].last()
            for pid in snap.index:
                pos = snap.at[pid, "pos"]
                if pos not in POS or pid not in last_team.index:
                    continue
                t = last_team[pid]
                prev = [w for w in team_weeks.get((S, t), []) if w < W]
                if not prev:
                    continue
                healthy = (pid, S, prev[-1]) in pts.index
                after = [w for w in team_weeks.get((S, t), []) if W <= w <= po[-1]]
                rest = sum(pts.get((pid, S, w), 0.0) for w in after if w < po[0])
                play = sum(pts.get((pid, S, w), 0.0) for w in after if w in po)
                rows.append(dict(season=S, week=W, player_id=pid, pos=pos, rank=snap.at[pid, "pos_rank"], healthy=healthy,
                                 res=ro.get((S, W, pid)) == "RES", star=bool(ppg["mean"].get(pid, 0) >= 12 and ppg["size"].get(pid, 0) >= 3),
                                 ppg=ppg["mean"].get(pid, np.nan), rest_reg=rest, playoffs=play))
    d = pd.DataFrame(rows)
    ev = d[d["star"] & ~d["healthy"] & ~d["res"]]
    out = []
    for _, x in ev.iterrows():
        c = d[(d.season == x.season) & (d.week == x.week) & (d.pos == x.pos) & d.healthy & ((d["rank"] - x["rank"]).abs() <= 3)]
        if not len(c):
            continue
        out.append(dict(season=x.season, week=x.week, player_id=x.player_id, pos=x.pos, rank=x["rank"], ppg=x.ppg, n_cmp=len(c),
                        star_reg=x.rest_reg, cmp_reg=c.rest_reg.mean(), star_po=x.playoffs, cmp_po=c.playoffs.mean(),
                        cmp_ppg=c.ppg.mean()))
    o = pd.DataFrame(out).drop_duplicates(["season", "player_id"])      # first week of each absence only
    o["d_reg"], o["d_po"] = o.star_reg - o.cmp_reg, o.star_po - o.cmp_po
    o["d_all"] = o.d_reg + o.d_po

    def show(x, lab):
        se = lambda v: v.std() / np.sqrt(len(v))
        print(f"{lab}: n={len(x)}  rest of regular season {x.d_reg.mean():+.1f} +/- {se(x.d_reg):.1f} PPR (star better in "
              f"{(x.d_reg > 0).mean():.0%})  |  playoff weeks {x.d_po.mean():+.1f} +/- {se(x.d_po):.1f} (star better in "
              f"{(x.d_po > 0).mean():.0%})  |  both {x.d_all.mean():+.1f} +/- {se(x.d_all):.1f}")
    print("Healthy-before-injury PPR/g: injured stars", round(o.ppg.mean(), 1), "vs comparison", round(o.cmp_ppg.mean(), 1))
    show(o, "All short-term injured stars (not IR) vs healthy players FantasyPros ranked within 3 spots")
    for p, x in o.groupby("pos"):
        show(x, "  " + p)
    show(o[o.week <= 9], "  injured Weeks 6-9")
    show(o[o.week >= 10], "  injured Weeks 10-12")
    show(o[o["rank"] <= 12], "  ranked top 12 at the position")
    return o


if __name__ == "__main__":
    main()

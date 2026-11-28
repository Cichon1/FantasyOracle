"""Fantasy playoff schedules, Weeks 15-17 (Nov 28, 2026): docs/data/playoff_sched_v5.csv.

All three of Jackson's leagues play their fantasy playoffs in Weeks 15, 16 and 17. For every NFL team and fantasy
position (QB / RB / WR / TE / K / DEF) and each playoff week: the opponent and how soft that matchup is.

- Skill positions: the opponent's PPR points allowed to the position per game this season (games played so far),
  shrunk toward the league average with 8 games of prior, as a ratio to average (> 1 = softer).
- DEF: fantasy points team defenses scored against that offense; K: points kickers scored against that defense.
  Same shrinkage. Yahoo default scoring and BDSM League's own K / DEF rules (lab17_kdef units).
- adj = 1 + b * (ratio - 1): how much of the matchup carries into a player's points. b fit on 2019-2022 (cutoff after
  Week 12, predicting Weeks 15-17), checked on 2023-2025 (model/lab20_playoff_sched.py):
    skill b 0.55 (test RMSE 8.58/8.15/7.85/6.91 -> small gains for QB/RB/WR, none for TE);
    DEF b 0.85 (test RMSE 6.26 -> 6.09; the defense matchup matters most); K b 0.4 (no test gain: treat as noise).
  Bye in a playoff week = adj 0.
- K / DEF rows also carry the unit's recent scoring (exponential average, half-life 6 games) for the page's points.

  python automation/playoff_sched.py --data model/data --season 2026 --out docs/data
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import lab17_kdef as K  # noqa: E402

WEEKS = (15, 16, 17)
PRIOR = 8
B = {"QB": 0.55, "RB": 0.55, "WR": 0.55, "TE": 0.55, "DEF": 0.85, "K": 0.4}


def skill_ratios(w):
    w = w[(w["season_type"] == "REG") & w["position"].isin(["QB", "RB", "WR", "TE"])]
    a = w.groupby(["opponent_team", "week", "position"])["fantasy_points_ppr"].sum().reset_index()
    lg = a.groupby("position")["fantasy_points_ppr"].mean()
    st = a.groupby(["opponent_team", "position"])["fantasy_points_ppr"].agg(["sum", "count"]).reset_index()
    st["ratio"] = ((st["sum"] + PRIOR * st["position"].map(lg)) / (st["count"] + PRIOR)) / st["position"].map(lg)
    return st.set_index(["opponent_team", "position"])["ratio"], st.groupby("opponent_team")["count"].max()


def unit_tables(u):
    """ratio by opponent and each team's recent average for DEF and K."""
    u = u[u["played"]].sort_values("week")
    out = {}
    for unit, col in [("DEF", "def_pts"), ("K", "k_pts")]:
        m = u[col].mean()
        x = u.groupby("opp")[col].agg(["sum", "count"])
        ratio = ((x["sum"] + PRIOR * m) / (x["count"] + PRIOR)) / m
        avg = u.groupby("team")[col].apply(lambda s: s.ewm(halflife=6).mean().iloc[-1])
        out[unit] = (ratio, avg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "model", "data"))
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data"))
    a = ap.parse_args()
    K.DATA = a.data
    games = pd.read_parquet(os.path.join(a.data, "games.parquet"))
    g = games[(games["season"] == a.season) & (games["game_type"] == "REG")]
    played = g[g["home_score"].notna()]
    thru = int(played["week"].max()) if len(played) else 0
    sched = pd.concat([g.assign(team=g["home_team"], opp=g["away_team"], home=1),
                       g.assign(team=g["away_team"], opp=g["home_team"], home=0)])[["week", "team", "opp", "home"]]
    teams = sorted(set(sched["team"]))

    w = pd.read_parquet(os.path.join(a.data, f"week_{a.season}.parquet"))
    sk, _ = skill_ratios(w[w["week"] <= thru])
    units = {lab: unit_tables(K.season_units(a.season, games, scoring))
             for scoring, lab in [("yahoo", "default"), ("bdsm", "126912")]}

    rows = []
    for t in teams:
        for wk in WEEKS:
            m = sched[(sched["team"] == t) & (sched["week"] == wk)]
            opp, home = (m.iloc[0]["opp"], int(m.iloc[0]["home"])) if len(m) else ("BYE", None)
            for pos in ["QB", "RB", "WR", "TE"]:
                r = np.nan if opp == "BYE" else sk.get((opp, pos), 1.0)
                rows.append(dict(team=t, pos=pos, scoring="", week=wk, opp=opp, home=home, ratio=r, recent_avg=np.nan))
            for lab, tab in units.items():
                for unit in ["DEF", "K"]:
                    ratio, avg = tab[unit]
                    r = np.nan if opp == "BYE" else ratio.get(opp, 1.0)
                    rows.append(dict(team=t, pos=unit, scoring=lab, week=wk, opp=opp, home=home, ratio=r,
                                     recent_avg=avg.get(t, np.nan)))
    out = pd.DataFrame(rows)
    out["adj"] = np.where(out["opp"] == "BYE", 0.0, 1 + out["pos"].map(B) * (out["ratio"] - 1))
    out["thru_week"] = thru
    for c in ["ratio", "adj", "recent_avg"]:
        out[c] = out[c].round(3)
    out.to_csv(os.path.join(a.out, "playoff_sched_v5.csv"), index=False)
    cat_p = os.path.join(a.out, "datasets.json")
    if os.path.exists(cat_p):
        cat = json.load(open(cat_p))
        cat["datasets"]["playoff_sched"] = {
            "file": "playoff_sched_v5.csv", "title": f"Playoff schedules, Weeks 15-17 (through Week {thru})",
            "description": "Per NFL team, fantasy position and playoff week (15-17): opponent, matchup ratio (opponent's "
                           "points allowed to the position vs average, shrunk with 8 games of prior; > 1 softer) and adj "
                           "(share of the matchup that carries into points: skill 0.55, DEF 0.85, K 0.4, fit on "
                           "2019-22, checked 2023-25). K / DEF by scoring (Yahoo default or a league id) with the "
                           "unit's recent average.",
            "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        json.dump(cat, open(cat_p, "w"), indent=1)
    print(f"playoff_sched: {len(out)} rows, ratings through Week {thru}")


if __name__ == "__main__":
    main()

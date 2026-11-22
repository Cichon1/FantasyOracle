"""Nov 19, 2026: kicker and team-defense fantasy points for the league sims (Yahoo default scoring), 2018-2026.

Kicker (team unit - whoever kicked for the team): FG made 0-39 yds 3, 40-49 4, 50+ 5; PAT made 1.
Defense / special teams (team unit): sack 1, interception 2, fumble recovery 2, defensive or return TD 6, safety 2,
blocked kick 2, points allowed 0: 10, 1-6: 7, 7-13: 4, 14-20: 1, 21-27: 0, 28-34: -1, 35+: -4.
(BDSM League's own K / DEF settings haven't been read from Yahoo yet; these are Yahoo's defaults.)

Pre-game projections (walk-forward: season S fit on seasons 2018..S-1, linear):
  K   ~ team implied points (Vegas total / spread), dome, recent K average
  DEF ~ opponent implied points, spread, home, recent DEF average, opponent's recent points scored

  python lab17_kdef.py        -> lab/kdef_weekly.parquet
"""
import os

import numpy as np
import pandas as pd

DATA = "data"


def points_allowed_pts(pa):
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34], [10, 7, 4, 1, 0, -1], -4)


def season_units(S, games):
    p = pd.read_parquet(f"{DATA}/pbp/pbp_{S}.parquet", columns=[
        "season_type", "week", "game_id", "posteam", "defteam", "field_goal_result", "kick_distance", "extra_point_result",
        "td_team", "interception", "fumble_lost", "sack", "safety", "punt_blocked", "play_type"])
    p = p[p["season_type"] == "REG"]
    fg = p[p["field_goal_result"] == "made"]
    fgp = np.select([fg["kick_distance"] >= 50, fg["kick_distance"] >= 40], [5, 4], 3)
    k = pd.DataFrame({"week": fg["week"], "team": fg["posteam"], "pts": fgp})
    xp = p[p["extra_point_result"] == "good"]
    k = pd.concat([k, pd.DataFrame({"week": xp["week"], "team": xp["posteam"], "pts": 1})])
    k = k.groupby(["week", "team"])["pts"].sum()
    d = p.copy()
    d["dp"] = (d["sack"].fillna(0) * 1 + d["interception"].fillna(0) * 2 + d["fumble_lost"].fillna(0) * 2
               + (d["field_goal_result"] == "blocked") * 2 + d["punt_blocked"].fillna(0) * 2
               + (d["extra_point_result"] == "blocked") * 2)
    # safeties are scored by the defense (defteam); TDs by the team that isn't on offense (returns, defensive TDs)
    d["dp"] += d["safety"].fillna(0) * 2
    dt = d["td_team"].notna() & (d["td_team"] == d["defteam"])
    d.loc[dt, "dp"] += 6
    dfe = d.groupby(["week", "defteam"])["dp"].sum()
    # kickoff / punt return TDs: on those plays posteam is the kicking team, defteam the returning team (counted above)
    g = games[(games["season"] == S) & (games["game_type"] == "REG")]
    rows = []
    for _, r in g.iterrows():
        for team, opp, pf, pa, home in [(r["home_team"], r["away_team"], r["home_score"], r["away_score"], 1),
                                        (r["away_team"], r["home_team"], r["away_score"], r["home_score"], 0)]:
            spread = r["spread_line"] if home else -r["spread_line"]        # positive = team favored
            implied = (r["total_line"] + spread) / 2 if pd.notna(r["total_line"]) else np.nan
            opp_implied = (r["total_line"] - spread) / 2 if pd.notna(r["total_line"]) else np.nan
            played = pd.notna(pa)
            rows.append({"season": S, "week": int(r["week"]), "team": team, "opp": opp, "home": home,
                         "spread": spread, "implied": implied, "opp_implied": opp_implied,
                         "dome": int(str(r.get("roof", "")) in ("dome", "closed")), "played": played,
                         "k_pts": float(k.get((r["week"], team), 0.0)) if played else np.nan,
                         "def_pts": (float(dfe.get((r["week"], team), 0.0)) + float(points_allowed_pts(np.array([pa]))[0]))
                         if played else np.nan,
                         "pf": pf, "pa": pa})
    return pd.DataFrame(rows)


def build(last_season=2026, data=None):
    """All team-weeks 2018..last_season with K / DEF points (played games) and walk-forward projections."""
    global DATA
    if data:
        DATA = data
    games = pd.read_parquet(f"{DATA}/games.parquet")
    u = pd.concat([season_units(S, games) for S in range(2018, last_season + 1)], ignore_index=True)
    u = u.sort_values(["team", "season", "week"])
    for col, src in [("k_avg", "k_pts"), ("def_avg", "def_pts"), ("pf_avg", "pf")]:
        u[col] = u.groupby("team")[src].transform(lambda x: x.shift(1).ewm(halflife=6, min_periods=1).mean())
    opp_pf = u[["season", "week", "team", "pf_avg"]].rename(columns={"team": "opp", "pf_avg": "opp_pf_avg"})
    u = u.merge(opp_pf, on=["season", "week", "opp"], how="left")
    XK = ["implied", "dome", "k_avg"]
    XD = ["opp_implied", "spread", "home", "def_avg", "opp_pf_avg"]
    u["k_proj"], u["def_proj"] = np.nan, np.nan
    for S in range(2020, last_season + 1):
        tr = u[(u["season"] < S) & u["played"]].dropna(subset=XK + XD + ["k_pts", "def_pts"])
        te = u["season"] == S
        for X, y, out in [(XK, "k_pts", "k_proj"), (XD, "def_pts", "def_proj")]:
            A = np.c_[np.ones(len(tr)), tr[X].values]
            beta = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
            Xt = u.loc[te, X].fillna(u.loc[te, X].mean())
            u.loc[te, out] = np.c_[np.ones(te.sum()), Xt.values] @ beta
    return u


def main():
    u = build()
    u.to_parquet(os.path.join("lab", "kdef_weekly.parquet"))
    v = u[(u["season"] >= 2023) & u["played"]]
    from scipy.stats import spearmanr
    for unit in ["k", "def"]:
        cs = [spearmanr(g[f"{unit}_proj"], g[f"{unit}_pts"]).correlation for _, g in v.groupby(["season", "week"])]
        ca = [spearmanr(g[f"{unit}_avg"], g[f"{unit}_pts"], nan_policy="omit").correlation for _, g in v.groupby(["season", "week"])]
        print(unit, "weekly rank corr: projection", round(np.nanmean(cs), 3), "| recent average", round(np.nanmean(ca), 3),
              "| mean pts", round(v[f"{unit}_pts"].mean(), 2), "sd", round(v[f"{unit}_pts"].std(), 2))


if __name__ == "__main__":
    main()

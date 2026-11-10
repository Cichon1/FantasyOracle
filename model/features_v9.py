"""v9 candidate inputs (Nov 9, 2026): a team's pass-first identity this season vs its head coach's history, and how
broken its run game is (the Gemini review's "forced to pass" idea; the Green Bay / Christian Watson debate).

From play-by-play (data/plays.parquet), neutral situations only (win probability 20-80%, not the last two minutes of
a half) on 1st and 2nd down:
  v9_proe_cur        the team's pass rate over expected so far this season
  v9_proe_n          its neutral early-down plays so far this season
  v9_proe_coach3     the head coach's neutral early-down PROE over his previous three seasons (any team)
  v9_eb_proe         this season shrunk toward the coach's history by sample size: n / (n + 180)
  v9_proe_break      (this season - coach's history) x sqrt(n) / 10: how far, and how surely, the team left its norm
  v9_rush_epa_cur    neutral-situation EPA per designed run so far this season
  v9_forced_pass     this season's PROE x how far its run EPA sits below zero
All attached from games before the row's week (attach_prior).
"""
import os

import numpy as np
import pandas as pd

import advanced_features as af

COLS = ["v9_proe_cur", "v9_proe_n", "v9_proe_coach3", "v9_eb_proe", "v9_proe_break", "v9_rush_epa_cur",
        "v9_forced_pass"]
K = 180.0


def team_identity(data_dir, df):
    p = pd.read_parquet(os.path.join(data_dir, "plays.parquet"),
                        columns=["game_id", "season", "week", "posteam", "home_team", "home_coach", "away_coach",
                                 "pass_oe", "neutral", "down", "rush", "qb_dropback", "epa", "designed_run"])
    p = p[p["posteam"].notna() & (p["neutral"] == 1)]
    p["coach"] = np.where(p["posteam"] == p["home_team"], p["home_coach"], p["away_coach"])
    ed = p[p["down"].isin([1, 2]) & p["pass_oe"].notna()]
    g = ed.groupby(["game_id", "season", "week", "posteam", "coach"], as_index=False).agg(
        proe_sum=("pass_oe", "sum"), proe_n=("pass_oe", "size"))
    run = p[(p["designed_run"] == 1) if "designed_run" in p else ((p["rush"] == 1) & (p["qb_dropback"] != 1))]
    r = run.groupby(["game_id", "posteam"], as_index=False).agg(rush_epa_sum=("epa", "sum"), rush_n=("epa", "size"))
    g = g.merge(r, on=["game_id", "posteam"], how="left").fillna({"rush_epa_sum": 0, "rush_n": 0})
    g = g.sort_values(["posteam", "season", "week"])
    # season to date, through each game (attach_prior then gives the state before the row's week)
    grp = g.groupby(["posteam", "season"])
    for c in ["proe_sum", "proe_n", "rush_epa_sum", "rush_n"]:
        g[c + "_cum"] = grp[c].cumsum()
    g["v9_proe_cur"] = g["proe_sum_cum"] / g["proe_n_cum"]
    g["v9_proe_n"] = g["proe_n_cum"]
    g["v9_rush_epa_cur"] = g["rush_epa_sum_cum"] / g["rush_n_cum"].where(g["rush_n_cum"] > 0)
    # coach's previous three seasons (whatever team)
    cs = g.groupby(["coach", "season"]).agg(s=("proe_sum", "sum"), n=("proe_n", "sum")).reset_index()
    prior = []
    for _, row in cs.iterrows():
        h = cs[(cs["coach"] == row["coach"]) & (cs["season"] < row["season"]) & (cs["season"] >= row["season"] - 3)]
        prior.append(h["s"].sum() / h["n"].sum() if h["n"].sum() > 0 else np.nan)
    cs["v9_proe_coach3"] = prior
    g = g.merge(cs[["coach", "season", "v9_proe_coach3"]], on=["coach", "season"], how="left")
    c3 = g["v9_proe_coach3"].fillna(0.0)
    n = g["v9_proe_n"]
    g["v9_eb_proe"] = n / (n + K) * g["v9_proe_cur"] + K / (n + K) * c3
    g["v9_proe_break"] = (g["v9_proe_cur"] - c3) * np.sqrt(n) / 10
    g["v9_forced_pass"] = g["v9_proe_cur"] * np.maximum(0, -g["v9_rush_epa_cur"].fillna(0))
    tab = g.rename(columns={"posteam": "team"})[["team", "season", "week"] + COLS]
    # week 0 of each season: the coach's prior with no current-season plays (for each team's first game)
    first = tab.sort_values("week").drop_duplicates(["team", "season"]).copy()
    first["week"] = 0
    for c in ["v9_proe_cur", "v9_rush_epa_cur", "v9_forced_pass", "v9_proe_break"]:
        first[c] = np.nan
    first["v9_proe_n"] = 0.0
    first["v9_eb_proe"] = first["v9_proe_coach3"].fillna(0.0)
    tab = pd.concat([tab, first], ignore_index=True)
    return af.attach_prior(df, tab, "team", "team", COLS)

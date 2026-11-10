"""Schedule-neutral weekly inputs for rest-of-season projections (Nov 9, 2026; idea from the Gemini review).

ROS points use 0.8 x this week's projection for every later game, so this week's matchup (Vegas line, opponent,
venue, weather, short week) leaks into all of them. neutralize() returns copies of weekly-model rows with those
single-game inputs replaced by the team's typical values, staying inside what the model saw in training:
  Vegas: the team's average implied points and its opponents' average over its games so far this season (this game
         when it's the first), total and spread rebuilt from them;
  opponent's points allowed to the position: league average (relative 1.0, the position's average level);
  venue / schedule: half home, no division game, normal rest, no short week / Thursday / prime time / travel /
         international / altitude; weather: the season's median wind and temperature, no rain or snow.
Opponent scheme inputs (coverage, blitz, box rates) are left as they are.
"""
import numpy as np
import pandas as pd

CTX_ZERO = ["div_game", "ctx_short_week", "ctx_thursday", "ctx_prime_time", "ctx_tz_travel", "ctx_international",
            "ctx_altitude"]


def team_vegas_avg(df):
    """Per (team, season, week): average implied points and opponent implied points over the team's games this season
    up to and including that week (regular season and playoffs)."""
    g = df[["team", "season", "week", "game_id", "implied_pts", "opp_implied_pts"]].dropna(subset=["implied_pts"])
    g = g.drop_duplicates(["team", "game_id"]).sort_values(["team", "season", "week"])
    grp = g.groupby(["team", "season"])
    g["avg_implied"] = grp["implied_pts"].transform(lambda s: s.expanding().mean())
    g["avg_opp_implied"] = grp["opp_implied_pts"].transform(lambda s: s.expanding().mean())
    return g.set_index(["team", "season", "week"])[["avg_implied", "avg_opp_implied"]]


def neutralize(rows, df):
    """rows: weekly-model rows (frame columns); df: the frame (for team Vegas averages and medians)."""
    out = rows.copy()
    tv = team_vegas_avg(df)
    k = pd.MultiIndex.from_arrays([out["team"], out["season"], out["week"]])
    ai = pd.Series(tv["avg_implied"].reindex(k).values, index=out.index)
    ao = pd.Series(tv["avg_opp_implied"].reindex(k).values, index=out.index)
    if "implied_pts" in out:
        out["implied_pts"] = ai.fillna(out["implied_pts"])
        out["opp_implied_pts"] = ao.fillna(out["opp_implied_pts"])
        out["total_line"] = out["implied_pts"] + out["opp_implied_pts"]
        out["team_spread"] = out["implied_pts"] - out["opp_implied_pts"]
    if "def_allowed_rel" in out:
        base = df.groupby(["season", "position"])["def_allowed_ewm"].mean()
        kk = pd.MultiIndex.from_arrays([out["season"], out["position"]])
        out["def_allowed_ewm"] = pd.Series(base.reindex(kk).values, index=out.index).fillna(out["def_allowed_ewm"])
        out["def_allowed_rel"] = 1.0
    if "is_home" in out:
        out["is_home"] = 0.5
    if "rest" in out:
        out["rest"] = 7
    for c in CTX_ZERO:
        if c in out:
            out[c] = 0.0
    outdoor = out["wx_dome"].fillna(0).eq(0) if "wx_dome" in out else pd.Series(True, index=out.index)
    for c in ["wx_wind", "wx_temp"]:
        if c in out:
            src = df[df["wx_dome"].fillna(0).eq(0)] if "wx_dome" in df else df
            med = src.groupby("season")[c].median()
            out.loc[outdoor, c] = out.loc[outdoor, "season"].map(med).fillna(out.loc[outdoor, c])
    for c in ["wx_rain", "wx_snow"]:
        if c in out:
            out[c] = 0.0
    return out

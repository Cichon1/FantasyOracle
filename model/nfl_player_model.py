"""
NFL weekly player-stat prediction model  (v2)
=============================================

Predicts next-game stats for QBs, RBs, WRs and TEs:
  - PPR fantasy points (with a 10th-90th percentile range)
  - passing yards (QB), rushing yards (QB/RB), receiving yards and receptions (RB/WR/TE)

v2 adds scheme, coaching, pace, weather, snap-share and injury features
(see advanced_features.py) and keeps a feature group only for the stats where it
improves accuracy on the 2024 validation season.

Run:  python nfl_player_model.py --data data --out out --season 2026 --week 5
"""
import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import fastops as fo

import advanced_features as af

POSITIONS = ["QB", "RB", "WR", "TE"]
FORM_STATS = [
    "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
    "passing_epa", "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
    "receiving_yards", "receiving_tds", "receiving_air_yards", "target_share",
    "air_yards_share", "wopr", "fantasy_points_ppr", "carry_share",
]
TRAIN_SEASONS = range(2018, 2024)
VALID_SEASON = 2024
TEST_SEASONS = [2025, 2026]
DEF_SCHEME = ["blitz", "man", "zone", "cover0", "cover1", "cover2", "cover3", "cover4", "two_high",
              "single_high", "pressure", "base", "nickel", "dime", "light_box", "heavy_box"]


# --------------------------------------------------------------------------- data
def _align_teams(df, games, team_col="team", opp_col="opponent_team"):
    """Use the schedule's team codes for each game (weekly stats say LV for the 2018-19 Raiders, the schedule
    and play-by-play say OAK)."""
    g = games.set_index("game_id")[["home_team", "away_team"]]
    home, away = df["game_id"].map(g["home_team"]), df["game_id"].map(g["away_team"])
    ok = home.notna()
    bad = ok & (df[team_col] != home) & (df[team_col] != away)
    is_home = (df[opp_col] == away) | (df[team_col].replace({"LV": "OAK", "LAC": "SD", "LA": "STL"}) == home)
    df.loc[bad, team_col] = np.where(is_home[bad], home[bad], away[bad])
    df.loc[ok, opp_col] = np.where(df.loc[ok, team_col] == home[ok], away[ok], home[ok])
    return df


def snap_only_rows(data_dir, stats, games, players):
    """Player-games with offensive snaps but no recorded stat (a TE who only blocked, a WR with no target).
    They are real games worth zero points; weekly stats leave them out."""
    sn = pd.concat([pd.read_parquet(f, columns=["game_id", "season", "game_type", "week", "pfr_player_id", "team",
                                                "opponent", "offense_snaps"])
                    for f in sorted(glob.glob(os.path.join(data_dir, "snaps", "*.parquet")))], ignore_index=True)
    sn = sn[sn["offense_snaps"] > 0]
    pl = players[["pfr_id", "gsis_id", "position", "position_group", "display_name", "short_name",
                  "headshot"]].dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
    sn = sn.merge(pl, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    sn = sn[sn["position"].isin(POSITIONS)]
    have = pd.MultiIndex.from_frame(stats[["player_id", "game_id"]])
    sn = sn[~pd.MultiIndex.from_arrays([sn["gsis_id"], sn["game_id"]]).isin(have)]
    sn = sn[sn["game_id"].isin(set(games["game_id"]))].drop_duplicates(["gsis_id", "game_id"])
    out = pd.DataFrame({
        "player_id": sn["gsis_id"].values, "player_name": sn["short_name"].values,
        "player_display_name": sn["display_name"].values, "position": sn["position"].values,
        "position_group": sn["position_group"].values, "headshot_url": sn["headshot"].values,
        "season": sn["season"].values, "week": sn["week"].values,
        "season_type": np.where(sn["game_type"].values == "REG", "REG", "POST"), "game_id": sn["game_id"].values,
        "team": sn["team"].values, "opponent_team": sn["opponent"].values})
    out = _align_teams(out, games)
    num = [c for c in stats.columns if c not in out.columns and pd.api.types.is_numeric_dtype(stats[c])]
    nan_cols = {"passing_epa", "passing_cpoe", "pacr", "rushing_epa", "receiving_epa", "racr", "fg_pct", "pat_pct"}
    out = pd.concat([out, pd.DataFrame({c: (np.nan if c in nan_cols else 0.0) for c in num}, index=out.index)],
                    axis=1)
    out["snap_only"] = True
    return out


def load(data_dir):
    stats = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(data_dir, "week_*.parquet")))],
                      ignore_index=True)
    stats = stats[stats["season_type"].isin(["REG", "POST"])]
    team_carries = stats.groupby(["game_id", "team"])["carries"].transform("sum")
    stats["carry_share"] = (stats["carries"] / team_carries).where(team_carries > 0)
    stats = stats[stats["position"].isin(POSITIONS)].copy()
    games = pd.read_parquet(os.path.join(data_dir, "games.parquet"))
    players = pd.read_parquet(os.path.join(data_dir, "players.parquet"))
    stats = _align_teams(stats, games)
    stats["snap_only"] = False
    stats = pd.concat([stats, snap_only_rows(data_dir, stats, games, players)], ignore_index=True)
    return stats, games, players


def team_game_table(games):
    g = games.copy()
    g["home_implied"] = (g["total_line"] + g["spread_line"]) / 2
    g["away_implied"] = (g["total_line"] - g["spread_line"]) / 2
    cols = ["game_id", "season", "week", "total_line", "div_game"]
    home = g[cols + ["home_team", "away_team", "home_implied", "away_implied", "spread_line", "home_rest",
                     "home_coach"]].rename(columns={"home_team": "team", "away_team": "opponent_team",
                                                    "home_implied": "implied_pts", "away_implied": "opp_implied_pts",
                                                    "home_rest": "rest", "home_coach": "coach"})
    home["is_home"] = 1
    home["team_spread"] = home["spread_line"]
    away = g[cols + ["away_team", "home_team", "away_implied", "home_implied", "spread_line", "away_rest",
                     "away_coach"]].rename(columns={"away_team": "team", "home_team": "opponent_team",
                                                    "away_implied": "implied_pts", "home_implied": "opp_implied_pts",
                                                    "away_rest": "rest", "away_coach": "coach"})
    away["is_home"] = 0
    away["team_spread"] = -away["spread_line"]
    tg = pd.concat([home, away], ignore_index=True).drop(columns="spread_line")
    neutral = set(g.loc[g["location"] == "Neutral", "game_id"])
    tg.loc[tg["game_id"].isin(neutral), "is_home"] = 0
    # games the current head coach has coached for this team before this one (schedule-known)
    tg = tg.sort_values(["team", "season", "week"])
    tg["coach_games_with_team"] = tg.groupby(["team", "coach"]).cumcount()
    return tg


# ----------------------------------------------------------------------- features
def add_player_form(df):
    """Recent-form features from the player's previous *played* games. Rows for games the player didn't play
    (expected to play but sat, or upcoming) get the state after his last played game, same as played rows."""
    df = df.copy()
    played = ~df["upcoming"] & ~df["phantom"]
    p = df[played].sort_values(["player_id", "season", "week"])
    key = p["player_id"]
    st = p[["player_id", "season", "week"]].copy()
    for s in FORM_STATS:
        st[f"{s}_ewm"] = fo.gewm(p[s], key, 4)
        st[f"{s}_l3"] = fo.groll_mean(p[s], key, 3, min_periods=1)
        st[f"{s}_ewm_long"] = fo.gewm(p[s], key, 12)
    st["fpts_ppr_l8_std"] = fo.groll_std(p["fantasy_points_ppr"], key, 8, min_periods=2)
    st["last_fpts"] = p["fantasy_points_ppr"]
    st["games_hist"] = key.groupby(key, sort=False).cumcount() + 1
    st["_last_abs"] = p["season"] * 25 + p["week"]
    st["_last_team"] = p["team"]
    cols = [c for c in st.columns if c not in ("player_id", "season", "week")]
    att = af.attach_prior(df, st, "player_id", "player_id", cols)
    att["games_hist"] = att["games_hist"].fillna(0)
    att["weeks_since_last"] = df["season"] * 25 + df["week"] - att["_last_abs"]
    att["same_team_as_last"] = (df["team"] == att["_last_team"]).astype(float)
    return pd.concat([df, att.drop(columns=["_last_abs", "_last_team"])], axis=1)


def defense_allowed(stats, tg):
    allowed = (stats.groupby(["opponent_team", "position", "game_id", "season", "week"], as_index=False)
               ["fantasy_points_ppr"].sum()
               .rename(columns={"opponent_team": "def_team", "fantasy_points_ppr": "pts_allowed"}))
    dg = tg[["team", "game_id", "season", "week"]].rename(columns={"team": "def_team"})
    dg = dg[dg["game_id"].isin(set(stats["game_id"])) | dg["season"].ge(2026)]
    full = dg.merge(pd.DataFrame({"position": POSITIONS}), how="cross")
    full = full.merge(allowed, on=["def_team", "position", "game_id", "season", "week"], how="left")
    played = full["game_id"].isin(set(stats["game_id"]))
    full.loc[played, "pts_allowed"] = full.loc[played, "pts_allowed"].fillna(0)
    full = full.sort_values(["def_team", "position", "season", "week"])
    g = full.groupby(["def_team", "position"])["pts_allowed"]
    full["def_allowed_ewm"] = fo.gewm(full["pts_allowed"], [full["def_team"], full["position"]], 6, shift=1)
    base = full[full["season"].isin(TRAIN_SEASONS)].groupby("position")["pts_allowed"].mean()
    full["def_allowed_rel"] = full["def_allowed_ewm"] / full["position"].map(base)
    return full[["def_team", "position", "game_id", "def_allowed_ewm", "def_allowed_rel"]]


STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 2}


def depth_team_table(data_dir, games):
    """Pre-game depth chart per team-week: weekly charts through 2024, and from 2025 the latest snapshot
    taken before each game (within 10 days). Columns: player_id, season, week, team, pos, order."""
    g = games[["game_id", "season", "week", "gameday", "home_team", "away_team"]].copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    tg = pd.concat([g.rename(columns={"home_team": "team"}).drop(columns="away_team"),
                    g.rename(columns={"away_team": "team"}).drop(columns="home_team")])
    rows = []
    for f in sorted(glob.glob(os.path.join(data_dir, "depth", "depth_*.parquet"))):
        d = pd.read_parquet(f)
        if "dt" not in d.columns:
            d = d[(d["formation"] == "Offense") & d["position"].isin(POSITIONS)].dropna(subset=["week", "gsis_id"])
            d = d.assign(order=pd.to_numeric(d["depth_team"], errors="coerce"), team=d["club_code"],
                         week=d["week"].astype(int), pos=d["position"])
            rows.append(d.groupby(["gsis_id", "season", "week", "team", "pos"], as_index=False)["order"].min())
        else:
            d = d[d["pos_abb"].isin(POSITIONS)].copy()
            d["dt"] = pd.to_datetime(d["dt"]).dt.tz_localize(None)
            snaps = d.groupby(["team", "dt"]).size().reset_index()[["team", "dt"]].sort_values("dt")
            m = pd.merge_asof(tg.sort_values("gameday"), snaps.rename(columns={"dt": "snap"}), left_on="gameday",
                              right_on="snap", by="team", direction="backward", allow_exact_matches=False)
            m = m.dropna(subset=["snap"])
            m = m[(m["gameday"] - m["snap"]).dt.days <= 10]
            dd = d.rename(columns={"dt": "snap", "pos_abb": "pos", "pos_rank": "order"}).merge(
                m[["team", "snap", "season", "week"]], on=["team", "snap"])
            rows.append(dd.groupby(["gsis_id", "season", "week", "team", "pos"], as_index=False)["order"].min())
    dt = pd.concat(rows, ignore_index=True).rename(columns={"gsis_id": "player_id"})
    return dt.sort_values("order").drop_duplicates(["player_id", "season", "week", "team"])


def roster_table(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "roster", "roster_weekly_*.parquet")))
    if not files:
        return None
    r = pd.concat([pd.read_parquet(f, columns=["season", "week", "team", "gsis_id", "status"]) for f in files],
                  ignore_index=True).dropna(subset=["gsis_id", "week"])
    r["week"] = r["week"].astype(int)
    r["team"] = r["team"].replace({"LAR": "LA"})
    return r.rename(columns={"gsis_id": "player_id"})


def candidate_rows(stats, tg, inj, dtab, players, games_sel, roster=None, keep_played=False):
    """Players expected to play in each game of games_sel (rows of tg): on the team's pre-game depth chart
    (when one exists) and either a depth-chart starter (QB1, RB1-2, WR1-3, TE1-2 after removing players
    ruled Out) or played for the team in one of its previous two games; never a player ruled Out.
    Returns the candidates who have no played row in that game (the caller marks them as zero-point games
    for history, or as upcoming)."""
    order = tg.sort_values(["team", "season", "week"])[["team", "game_id", "season", "week"]].copy()
    order["prev1"] = order.groupby("team")["game_id"].shift(1)
    order["prev2"] = order.groupby("team")["game_id"].shift(2)
    sel = order[order["game_id"].isin(set(games_sel["game_id"]))]
    played = stats[["player_id", "game_id", "team"]]
    recent = []
    for c in ("prev1", "prev2"):
        r = sel[["team", "game_id", "season", "week", c]].merge(
            played.rename(columns={"game_id": c}), on=[c, "team"], how="inner")
        recent.append(r[["player_id", "team", "game_id", "season", "week"]])
    recent = pd.concat(recent).drop_duplicates(["player_id", "game_id"])
    outs = inj.loc[inj["report_status"] == "Out", ["player_id", "season", "week"]].drop_duplicates()
    outs["_out"] = True
    ch = dtab.merge(sel[["team", "season", "week", "game_id"]], on=["team", "season", "week"], how="inner")
    ch = ch.merge(outs, on=["player_id", "season", "week"], how="left")
    ch = ch[ch["_out"] != True].drop(columns="_out")
    ch = ch.sort_values(["game_id", "pos", "order"])
    ch["_r"] = ch.groupby(["game_id", "pos"]).cumcount() + 1
    starters = ch[ch["_r"] <= ch["pos"].map(STARTERS)][["player_id", "team", "game_id", "season", "week"]]
    has_chart = set(ch["game_id"].astype(str) + "|" + ch["team"])
    on_chart = set(ch["player_id"] + "|" + ch["game_id"] + "|" + ch["team"])
    rk = recent["game_id"] + "|" + recent["team"]
    keep = ~rk.isin(has_chart) | (recent["player_id"] + "|" + rk).isin(on_chart)
    cand = pd.concat([starters, recent[keep]]).drop_duplicates(["player_id", "game_id"])
    cand = cand.merge(outs, on=["player_id", "season", "week"], how="left")
    cand = cand[cand["_out"] != True].drop(columns="_out")
    if roster is not None:
        # weekly roster (set during the week): drop players on IR / reserve, cut, practice squad, or listed
        # with another team. Game-day inactives (INA) are not used.
        have = set(zip(roster["season"], roster["week"]))
        r = roster[roster["status"].isin(["ACT", "INA"])][["player_id", "season", "week", "team"]].drop_duplicates()
        r["_ok"] = True
        cand = cand.merge(r, on=["player_id", "season", "week", "team"], how="left")
        covered = pd.Series([(a, b) in have for a, b in zip(cand["season"], cand["week"])], index=cand.index)
        cand = cand[(cand["_ok"] == True) | ~covered].drop(columns="_ok")
    if keep_played:   # the rule's full list, played or not (to flag played games the live rule would have projected)
        return cand[["player_id", "game_id"]].drop_duplicates()
    # not if he played in this game, or for anyone else that week
    pk = set(stats["player_id"] + "|" + stats["game_id"])
    pw = set(stats["player_id"] + "|" + stats["season"].astype(str) + "|" + stats["week"].astype(str))
    cand = cand[~(cand["player_id"] + "|" + cand["game_id"]).isin(pk)]
    cand = cand[~(cand["player_id"] + "|" + cand["season"].astype(str) + "|" + cand["week"].astype(str)).isin(pw)]
    # player info: latest stats row, else the players table
    info = stats.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")[
        ["player_id", "player_name", "player_display_name", "position", "position_group", "headshot_url"]]
    pl = players[["gsis_id", "short_name", "display_name", "position", "position_group", "headshot"]].rename(
        columns={"gsis_id": "player_id", "short_name": "player_name", "display_name": "player_display_name",
                 "headshot": "headshot_url"}).drop_duplicates("player_id")
    pl = pl[~pl["player_id"].isin(set(info["player_id"]))]
    info = pd.concat([info, pl], ignore_index=True)
    cand = cand.merge(info, on="player_id", how="inner")
    cand = cand[cand["position"].isin(POSITIONS)]
    cand = cand.merge(tg[["game_id", "team", "opponent_team"]], on=["game_id", "team"], how="left")
    cand["season_type"] = np.where(cand["game_id"].map(games_sel.drop_duplicates("game_id").set_index("game_id")
                                                       ["game_type"]) == "REG", "REG", "POST")
    return cand.reset_index(drop=True)


def injury_features(df, stats, inj):
    """Own practice status this week + teammates' vacated opportunity."""
    own = inj[["player_id", "season", "week", "practice_dnp", "practice_limited", "on_report"]]
    df = df.merge(own, on=["player_id", "season", "week"], how="left")
    for c in ["practice_dnp", "practice_limited", "on_report"]:
        df[c] = df[c].fillna(0)
    # each player's opportunity going into the week (state after his last game)
    st = stats.sort_values(["player_id", "season", "week"]).copy()
    for c in ["target_share", "carry_share", "attempts"]:
        st[f"{c}_state"] = fo.gewm(st[c], st["player_id"], 4)
    dnp = inj[inj["practice_dnp"] == 1][["player_id", "team", "season", "week"]].copy()
    prior = af.attach_prior(dnp, st, "player_id", "player_id",
                            ["target_share_state", "carry_share_state", "attempts_state"])
    dnp = pd.concat([dnp, prior], axis=1).fillna({"target_share_state": 0, "carry_share_state": 0,
                                                  "attempts_state": 0})
    dnp["qb_starter_dnp"] = (dnp["attempts_state"] >= 15).astype(int)
    vac = dnp.groupby(["team", "season", "week"], as_index=False).agg(
        vacated_target_share=("target_share_state", "sum"), vacated_carry_share=("carry_share_state", "sum"),
        starting_qb_dnp=("qb_starter_dnp", "max"))
    df = df.merge(vac, on=["team", "season", "week"], how="left")
    for c in ["vacated_target_share", "vacated_carry_share", "starting_qb_dnp"]:
        df[c] = df[c].fillna(0)
    # don't count the player's own missed share as "vacated" for himself
    own_state = af.attach_prior(df, st, "player_id", "player_id", ["target_share_state", "carry_share_state",
                                                                   "attempts_state"])
    is_dnp = df["practice_dnp"] == 1
    df["vacated_target_share"] -= own_state["target_share_state"].fillna(0).where(is_dnp, 0)
    df["vacated_carry_share"] -= own_state["carry_share_state"].fillna(0).where(is_dnp, 0)
    df.loc[is_dnp & (own_state["attempts_state"].fillna(0) >= 15), "starting_qb_dnp"] = 0
    df["vacated_target_share"] = df["vacated_target_share"].clip(lower=0)
    df["vacated_carry_share"] = df["vacated_carry_share"].clip(lower=0)
    return df


def build_frame(data_dir, season=None, week=None):
    stats, games, players = load(data_dir)
    tg = team_game_table(games)
    inj = af.injury_table(data_dir)
    if season is not None:
        # before the final report (Friday), practice status is midweek: training only sees each week's final
        # practice status, so a midweek DNP counts as limited and doesn't vacate targets / carries yet
        import features_v4 as _f4
        prov = _f4.provisional_teams(data_dir, stats, season, week)   # final report not out yet
        cur = (inj["season"] == season) & (inj["week"] == week) & inj["report_status"].isna() & inj["team"].isin(prov)
        inj.loc[cur, "practice_limited"] = inj.loc[cur, ["practice_limited", "practice_dnp"]].max(axis=1)
        inj.loc[cur, "practice_dnp"] = 0
    depth = load_depth(data_dir, season)
    dtab = depth_team_table(data_dir, games)
    df = stats.copy()
    df["phantom"], df["upcoming"] = False, False
    # expected to play but didn't (questionable and sat, surprise inactive): zero-point games, 2018+
    hist_games = games[(games["season"] >= 2018) & games["game_id"].isin(set(stats["game_id"]))]
    roster = roster_table(data_dir)
    ph = candidate_rows(stats, tg, inj, dtab, players, hist_games, roster)
    ph["phantom"], ph["upcoming"] = True, False
    parts = [df, ph]
    if season is not None:
        up_games = games[(games["season"] == season) & (games["week"] == week)
                         & ~games["game_id"].isin(set(stats["game_id"]))]  # skip games already played
        up = candidate_rows(stats, tg, inj, dtab, players, up_games, roster)
        up["phantom"], up["upcoming"] = False, True
        parts.append(up)
    df = pd.concat(parts, ignore_index=True)
    for c in ["phantom", "upcoming", "snap_only"]:
        df[c] = df[c].eq(True)
    # live-like scoring pool: rows the expected-to-play rule picks (played or not) - a played game by a player the
    # rule wouldn't have projected is still used for training, but not for scoring
    allc = candidate_rows(stats, tg, inj, dtab, players, hist_games, roster, keep_played=True)
    key = df["player_id"].astype(str) + "|" + df["game_id"].astype(str)
    df["expected"] = df["phantom"] | df["upcoming"] | key.isin(set(allc["player_id"] + "|" + allc["game_id"]))
    df = add_player_form(df)
    df = df.merge(tg.drop(columns=["opponent_team", "season", "week"]), on=["game_id", "team"], how="left")
    da = defense_allowed(stats, tg)
    df = df.merge(da, left_on=["opponent_team", "position", "game_id"],
                  right_on=["def_team", "position", "game_id"], how="left").drop(columns="def_team")
    df["pos_code"] = df["position"].map({p: i for i, p in enumerate(POSITIONS)})
    df["playoff"] = (df["season_type"] == "POST").astype(int)
    df = df.reset_index(drop=True)

    # ---- advanced groups
    plays = af.build_plays(data_dir)
    off, de, lg, def_cols = af.team_features(plays)
    pace_cols = ["pace_plays_pg", "pace_sec_per_play", "pace_no_huddle_rate", "team_proe"]
    df = df.join(af.attach_prior(df, off, "team", "team", pace_cols))
    df = df.join(af.attach_prior(df, off, "opponent_team", "team", pace_cols, prefix="opp_"))
    coach_cols = ["coach_proe", "coach_proe_light_box", "coach_proe_mid_box", "coach_proe_heavy_box",
                  "coach_proe_vs_base", "coach_proe_vs_sub", "coach_games"]
    # as-available: participation-based columns only know last season's charting during a season
    off_c = af.carry_prior_season(off, "coach", ["coach_proe_vs_base", "coach_proe_vs_sub"],
                                  ["coach_proe_light_box", "coach_proe_mid_box", "coach_proe_heavy_box"])
    df = df.join(af.attach_prior(df, off_c, "coach", "coach", coach_cols))
    df["coach_games"] = df["coach_games"].fillna(0)
    looks_part, looks_box = ["man", "zone", "two_high", "single_high"], ["blitz", "noblitz"]
    de_c = af.carry_prior_season(de, "team", [f"def_{c}_rate" for c in af.PART_RATES] + ["def_avg_lb"]
                                 + [f"def_epa_in_{l}" for l in looks_part],
                                 [f"def_{c}_rate" for c in af.BOX_RATES] + [f"def_epa_in_{l}" for l in looks_box])
    df = df.join(af.attach_prior(df, de_c, "opponent_team", "team", def_cols))
    df["_lg"] = "NFL"
    lg_cols = [f"lg_{c}_rate" for c in DEF_SCHEME]
    lg_c = af.carry_prior_season(lg, "lg", [f"lg_{c}_rate" for c in af.PART_RATES],
                                 [f"lg_{c}_rate" for c in af.BOX_RATES])
    lgv = af.attach_prior(df, lg_c, "_lg", "lg", lg_cols)
    for c in DEF_SCHEME:
        df[f"def_{c}_rel"] = df[f"def_{c}_rate"] - lgv[f"lg_{c}_rate"]
    for t in af.player_cov_features(plays):
        cols = [c for c in t.columns if c not in ("game_id", "season", "week", "player_id")]
        part = [c for c in cols if any(c.endswith(l) for l in looks_part)]
        box = [c for c in cols if c not in part]
        t = af.carry_prior_season(t, "player_id", part, box)
        df = df.join(af.attach_prior(df, t, "player_id", "player_id", cols))

    # matchups: coach tendencies x opponent looks; player splits x opponent rates
    mid = (1 - df["def_light_box_rate"] - df["def_heavy_box_rate"]).clip(lower=0)
    df["coach_proe_matchup_box"] = (df["coach_proe_light_box"] * df["def_light_box_rate"]
                                    + df["coach_proe_mid_box"] * mid
                                    + df["coach_proe_heavy_box"] * df["def_heavy_box_rate"])
    df["coach_proe_matchup_pers"] = (df["coach_proe_vs_base"] * df["def_base_rate"]
                                     + df["coach_proe_vs_sub"] * (1 - df["def_base_rate"]))
    mz = (df["def_man_rate"] + df["def_zone_rate"]).replace(0, np.nan)
    man_w, zone_w = df["def_man_rate"] / mz, df["def_zone_rate"] / mz
    sh = (df["def_two_high_rate"] + df["def_single_high_rate"]).replace(0, np.nan)
    th_w, sh_w = df["def_two_high_rate"] / sh, df["def_single_high_rate"] / sh
    df["pc_tgt_share_matchup_mz"] = df["pc_tgt_share_vs_man"] * man_w + df["pc_tgt_share_vs_zone"] * zone_w
    df["pc_tgt_share_matchup_shell"] = (df["pc_tgt_share_vs_two_high"] * th_w
                                        + df["pc_tgt_share_vs_single_high"] * sh_w)
    df["pc_yds_matchup_mz"] = df["pc_yds_per_db_vs_man"] * man_w + df["pc_yds_per_db_vs_zone"] * zone_w
    df["pc_yds_matchup_shell"] = (df["pc_yds_per_db_vs_two_high"] * th_w
                                  + df["pc_yds_per_db_vs_single_high"] * sh_w)
    df["pc_man_edge"] = (df["pc_tgt_share_vs_man"] - df["pc_tgt_share_vs_zone"]) * df["def_man_rel"]
    df["pc_qb_epa_matchup_blitz"] = (df["pc_qb_epa_vs_blitz"] * df["def_blitz_rate"]
                                     + df["pc_qb_epa_vs_noblitz"] * (1 - df["def_blitz_rate"]))
    df["pc_qb_epa_matchup_mz"] = df["pc_qb_epa_vs_man"] * man_w + df["pc_qb_epa_vs_zone"] * zone_w
    df["pc_qb_epa_matchup_shell"] = (df["pc_qb_epa_vs_two_high"] * th_w
                                     + df["pc_qb_epa_vs_single_high"] * sh_w)
    df["pc_ypc_matchup_box"] = (df["pc_ypc_vs_light_box"] * df["def_light_box_rate"]
                                + df["pc_ypc_vs_mid_box"] * mid + df["pc_ypc_vs_heavy_box"] * df["def_heavy_box_rate"])
    df["pace_game_sec_per_play"] = df["pace_sec_per_play"] + df["opp_pace_sec_per_play"]
    df["pace_exp_plays"] = (df["pace_plays_pg"] + df["def_plays_allowed_pg"]) / 2


    # offensive style / run quality / red zone / contracts (video-inspired groups)
    soff, sde, soff_cols, sde_cols = af.style_features(plays)
    soff_c = af.carry_prior_season(soff, "team", ["style_heavy_pers_rate", "style_fb_rate", "off_pa_fb_epa"])
    sde_c = af.carry_prior_season(sde, "team", ["def_base_vs_heavy_rate", "def_rush_epa_vs_heavy",
                                                "def_heavy_faced_rate"], ["def_rush_epa_light_box"])
    df = df.join(af.attach_prior(df, soff_c, "team", "team", soff_cols))
    df = df.join(af.attach_prior(df, sde_c, "opponent_team", "team", sde_cols))
    pu, pu_cols = af.player_usage_features(plays, stats)
    df = df.join(af.attach_prior(df, pu, "player_id", "player_id", pu_cols))
    df["ru_explosive_x_two_high"] = df["ru_explosive_rate"] * df["def_two_high_rate"]
    df["ru_explosive_x_light_box"] = df["ru_explosive_rate"] * df["def_light_box_rate"]
    df["ru_explosive_vs_def"] = df["ru_explosive_rate"] * df["def_explosive_run_rate"]
    df["ru_ypc_formation_wtd"] = (df["ru_ypc_under_center"] * df["style_under_center_rate"]
                                  + df["ru_ypc_shotgun"] * (1 - df["style_under_center_rate"]))
    df["style_heavy_x_def_rush_epa"] = df["style_heavy_pers_rate"] * df["def_rush_epa_vs_heavy"]
    df["qb_pa_epa_x_pa_rate"] = df["qb_pa_epa"] * df["style_play_action_rate"]
    df["off_pa_edge"] = (df["off_pa_epa"] - df["def_pa_epa_allowed"]) * df["style_play_action_rate"]
    con = af.contract_features(data_dir)
    cs = con.sort_values(["player_id", "year_signed", "apy_cap_pct"]).drop_duplicates(
        ["player_id", "year_signed"], keep="last")[["player_id", "year_signed", "apy_cap_pct"]].dropna()
    cs["year_signed"] = cs["year_signed"].astype("int64")
    left = df[["player_id", "season"]].reset_index().rename(columns={"season": "year_signed"})
    left["year_signed"] = left["year_signed"].astype("int64")
    ct = pd.merge_asof(left.dropna(subset=["player_id"]).sort_values("year_signed"), cs.sort_values("year_signed"),
                       on="year_signed", by="player_id", direction="backward", suffixes=("", "_c"))
    # merge_asof keeps the left year; recover the signing year from the matched contract
    sign = pd.merge_asof(left.dropna(subset=["player_id"]).sort_values("year_signed"),
                         cs.assign(signed=cs["year_signed"]).sort_values("year_signed"),
                         on="year_signed", by="player_id", direction="backward")
    ct = ct.set_index("index"); sign = sign.set_index("index")
    df["inv_cap_pct"] = ct["apy_cap_pct"].reindex(df.index)
    df["inv_contract_year"] = df["season"] - sign["signed"].reindex(df.index)
    dr = con.drop_duplicates("player_id").set_index("player_id")
    df["inv_draft_overall"] = df["player_id"].map(dr["draft_overall"]).fillna(300)
    df["inv_years_since_draft"] = df["season"] - df["player_id"].map(dr["draft_year"])
    bd = pd.to_datetime(df["player_id"].map(players.drop_duplicates("gsis_id").set_index("gsis_id")["birth_date"]),
                        errors="coerce")
    df["inv_age"] = (pd.to_datetime(df["season"].astype(str) + "-09-01") - bd).dt.days / 365.25

    # ---- v4 groups
    import features_v4 as f4
    _dbg = os.environ.get("NFL_DEBUG_STAGE")
    if _dbg:
        import pickle
        with open(_dbg, "wb") as fh:
            pickle.dump(dict(df=df, stats=stats, games=games, players=players, inj=inj, depth=depth, off=off,
                             soff=soff, con=con, season=season, week=week), fh)
    df = add_v4_features(df, data_dir, stats, games, players, inj, depth, off, soff, con, season, week)

    # weather
    wx = af.game_weather(plays, games)
    df = df.merge(wx, on="game_id", how="left")

    # snap share
    sn = af.snap_features(data_dir, players).sort_values(["player_id", "season", "week"])
    sn["snap_pct_ewm"] = fo.gewm(sn["offense_pct"], sn["player_id"], 3)
    sn["snap_pct_last"] = sn["offense_pct"]
    df = df.join(af.attach_prior(df, sn, "player_id", "player_id", ["snap_pct_ewm", "snap_pct_last"]))

    # injuries
    df = injury_features(df, stats, inj)
    # targets for expected-to-play games the player sat out: zero
    for c in ["fantasy_points_ppr", "passing_yards", "rushing_yards", "receiving_yards", "receptions"]:
        df.loc[df["phantom"], c] = 0.0
    return df


def add_v4_features(df, data_dir, stats, games, players, inj, depth, off, soff, con, season, week):
    """v4 groups (see features_v4.py)."""
    import features_v4 as f4
    S, W = (season, week) if season is not None else (None, None)
    rc, rc_cols = f4.receiving_features(data_dir, stats)
    df = df.join(af.attach_prior(df, rc, "player_id", "player_id", rc_cols))
    df = df.join(f4.career_features(df, con.rename(columns={})))
    ecr_dir = os.environ.get("FP_DATA", "/home/claude/fpdata")
    df = df.join(f4.attach_consensus(df, os.path.join(ecr_dir, "db_fpecr.parquet"),
                                     os.path.join(ecr_dir, "db_playerids.csv"), games))
    for t, cols in [(x, [c for c in x.columns if c.startswith("ngs_")]) for x in f4.ngs_features(data_dir)[0]]:
        df = df.join(af.attach_prior(df, t, "player_id", "player_id", cols))
    xf, xf_cols = f4.xfp_features(data_dir, stats)
    df = df.join(af.attach_prior(df, xf, "player_id", "player_id", xf_cols))
    df = df.join(f4.attach_depth(df, data_dir, games))
    df = df.join(f4.attach_injury_final(df, data_dir, players, stats, (S, W) if S else None))
    wk = inj[(inj["season"] == S) & (inj["week"] == W)] if S else inj.iloc[:0]
    out_ids = set(wk.loc[wk["report_status"] == "Out", "player_id"])
    pq, _ = f4.protection_and_qb(data_dir, df, games, depth, out_ids, (S, W) if S else None)
    df = df.join(pq)
    df = df.join(f4.context_features(df, games, data_dir))
    off_sums = off[["game_id", "team", "season", "week", "plays", "proe_sum", "proe_cnt", "proe_light_box_sum",
                    "cnt_light_box", "proe_heavy_box_sum", "cnt_heavy_box", "proe_base_sum", "cnt_base",
                    "proe_sub_sum", "cnt_sub"]]
    df = df.join(f4.playcaller_features(df, off_sums, soff, games, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "ref", "playcallers.csv")))
    mid_pc = (1 - df["def_light_box_rate"] - df["def_heavy_box_rate"]).clip(lower=0)
    df["call_proe_matchup_box"] = (df["call_proe_light_box"] * df["def_light_box_rate"]
                                   + df["call_proe"] * mid_pc + df["call_proe_heavy_box"] * df["def_heavy_box_rate"])
    df["call_proe_matchup_pers"] = (df["call_proe_vs_base"] * df["def_base_rate"]
                                    + df["call_proe_vs_sub"] * (1 - df["def_base_rate"]))
    df["rc_tprr_x_route_part"] = df["rc_tprr"] * df["rc_route_participation"]

    return df


def load_depth(data_dir, season):
    path = os.path.join(data_dir, f"depth_{season}.parquet")
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
    return d


# ---------------------------------------------------------------- feature groups
BASE = ([f"{s}_{k}" for s in FORM_STATS for k in ("ewm", "l3", "ewm_long")]
        + ["fpts_ppr_l8_std", "last_fpts", "weeks_since_last", "same_team_as_last", "games_hist",
           "implied_pts", "opp_implied_pts", "team_spread", "total_line", "is_home", "rest", "div_game",
           "def_allowed_ewm", "def_allowed_rel", "pos_code", "playoff", "week"])
GROUPS = {
    "pace": ["pace_plays_pg", "pace_sec_per_play", "pace_no_huddle_rate", "team_proe", "opp_pace_plays_pg",
             "opp_pace_sec_per_play", "opp_pace_no_huddle_rate", "def_plays_allowed_pg",
             "def_sec_per_play_allowed", "pace_game_sec_per_play", "pace_exp_plays"],
    "coach": ["coach_proe", "coach_proe_light_box", "coach_proe_mid_box", "coach_proe_heavy_box",
              "coach_proe_vs_base", "coach_proe_vs_sub", "coach_proe_matchup_box", "coach_proe_matchup_pers",
              "coach_games", "coach_games_with_team"],
    "scheme": [f"def_{c}_rate" for c in DEF_SCHEME] + [f"def_{c}_rel" for c in DEF_SCHEME]
              + ["def_avg_lb", "def_pass_rate_faced"],
    "scheme_eff": ["def_epa_per_dropback", "def_epa_per_rush"] + [f"def_epa_in_{l}" for l in
                                                                  ["man", "zone", "two_high", "single_high",
                                                                   "blitz", "noblitz"]],
    "player_cov": [f"pc_tgt_share_vs_{l}" for l in ["man", "zone", "two_high", "single_high"]]
                  + [f"pc_yds_per_db_vs_{l}" for l in ["man", "zone", "two_high", "single_high"]]
                  + [f"pc_qb_epa_vs_{l}" for l in ["blitz", "noblitz", "man", "zone", "two_high", "single_high"]]
                  + [f"pc_ypc_vs_{l}_box" for l in ["light", "mid", "heavy"]]
                  + ["pc_tgt_share_matchup_mz", "pc_tgt_share_matchup_shell", "pc_yds_matchup_mz",
                     "pc_yds_matchup_shell", "pc_man_edge", "pc_qb_epa_matchup_blitz", "pc_qb_epa_matchup_mz",
                     "pc_qb_epa_matchup_shell", "pc_ypc_matchup_box"],
    "off_style": ["style_under_center_rate", "style_play_action_rate", "style_motion_rate", "style_two_back_rate",
                  "style_heavy_pers_rate", "style_fb_rate", "off_run_epa", "off_run_success", "off_dropback_epa",
                  "off_pa_epa", "off_nopa_epa", "off_pa_fb_epa", "off_uc_run_epa", "off_sg_run_epa",
                  "def_pa_epa_allowed", "def_base_vs_heavy_rate", "def_rush_epa_vs_heavy", "def_heavy_faced_rate",
                  "style_heavy_x_def_rush_epa", "qb_pa_epa", "qb_pa_epa_x_pa_rate", "off_pa_edge"],
    "run_quality": ["ru_ypc", "ru_stuff_rate", "ru_explosive_rate", "ru_gump_rate", "ru_epa_per_carry",
                    "ru_success_rate", "ru_ypc_under_center", "ru_ypc_shotgun", "ru_ypc_formation_wtd",
                    "off_stuff_rate", "off_explosive_run_rate", "off_gump_rate", "def_stuff_rate",
                    "def_explosive_run_rate", "def_gump_rate", "def_run_success_allowed", "def_rush_epa_light_box",
                    "ru_explosive_x_two_high", "ru_explosive_x_light_box", "ru_explosive_vs_def"],
    "red_zone": ["rz_goal_line_carry_share", "rz_target_share", "rz_inside5_carries_pg", "rz_inside10_targets_pg",
                 "sy_carries_pg", "sy_success_rate", "off_short_yd_success", "off_short_yd_pg",
                 "def_short_yd_success_allowed"],
    "investment": ["inv_cap_pct", "inv_contract_year", "inv_draft_overall", "inv_years_since_draft", "inv_age"],
    "receiving": ["rc_adot", "rc_route_participation", "rc_tprr", "rc_yprr", "rc_first_read_share",
                  "rc_deep_target_share", "rc_endzone_targets_pg", "rc_catchable_rate", "rc_contested_rate",
                  "rc_drop_rate", "rc_tprr_long", "rc_yprr_long", "rc_adot_long", "rc_tprr_x_route_part"],
    "career": None,  # filled below from the cr_ columns
    "consensus": ["ecr_week_ecr", "ecr_week_pos_rank", "ecr_week_sd", "ecr_ros_pos_rank", "ecr_preseason_pos_rank",
                  "ecr_week_available"],
    "ngs": None,
    "xfp": ["xfp_ewm", "xfp_rec_ewm", "xfp_rush_ewm", "xfp_pass_ewm", "xfp_fpoe_ewm", "xfp_td_exp_ewm",
            "xfp_td_luck_ewm", "xfp_team_share", "xfp_rec_yds_exp_ewm", "xfp_rush_yds_exp_ewm",
            "xfp_receptions_exp_ewm", "xfp_ewm_long", "xfp_fpoe_long", "xfp_td_luck_long"],
    "depth": ["dc_depth_rank", "dc_is_starter", "dc_depth_change", "dc_promoted", "dc_demoted"],
    "injury_final": ["ij_vacated_tgt", "ij_vacated_car", "ij_ol_starters_out", "ij_opp_db_starters_out",
                     "ij_opp_front_starters_out", "ij_questionable", "ij_doubtful"],
    "protection": ["pp_sack_rate_allowed", "pp_hit_rate_allowed", "pp_pressure_allowed", "pp_def_sack_rate",
                   "pp_def_hit_rate", "pp_def_pressure_rate", "pp_sack_matchup", "pp_pressure_matchup"],
    "qb_context": ["qb_changed", "qb_starter_epa", "qb_starter_db_pg", "qb_starter_starts", "qb_is_projected_starter",
                   "qb_tgt_share_with_qb", "qb_games_with_qb"],
    "context": ["ctx_short_week", "ctx_thursday", "ctx_prime_time", "ctx_tz_travel", "ctx_international", "ctx_turf",
                "ctx_altitude", "ctx_ref_penalties_pg"],
    "playcaller": ["call_proe", "call_proe_light_box", "call_proe_heavy_box", "call_proe_vs_base", "call_proe_vs_sub",
                   "call_under_center_rate", "call_play_action_rate", "call_heavy_rate", "call_plays_pg", "call_games",
                   "call_changed", "call_new_this_season", "call_games_with_team", "call_low_confidence",
                   "call_proe_matchup_box", "call_proe_matchup_pers"],
    "weather": ["wx_temp", "wx_wind", "wx_rain", "wx_snow", "wx_dome"],
    "usage": ["snap_pct_ewm", "snap_pct_last"],
    "injury": ["practice_dnp", "practice_limited", "on_report", "vacated_target_share",
               "vacated_carry_share", "starting_qb_dnp"],
}

import features_v4 as _f4
GROUPS["career"] = None
del GROUPS["career"]
GROUPS["career_decay"] = [f"cr_{s}_hl{h}" for s in _f4.CAREER_STATS for h in (2, 8, 16, 32)]
GROUPS["career_seasons"] = ([f"cr_{s}_{k}" for s in _f4.CAREER_STATS for k in
                             ("season_to_date", "last_season", "two_seasons_ago", "career_wtd", "shrunk", "rookie_prior")]
                            + ["cr_last_season_games", "cr_career_games_wtd", "cr_season_games", "cr_is_rookie",
                               "cr_new_team_this_season"])
GROUPS["ngs"] = (["ngs_rush_" + c for c in ["rush_yards_over_expected_per_att", "efficiency",
                                            "percent_attempts_gte_eight_defenders", "avg_time_to_los"]]
                 + ["ngs_rece_" + c for c in ["avg_separation", "avg_cushion", "avg_yac_above_expectation",
                                              "percent_share_of_intended_air_yards", "catch_percentage",
                                              "avg_intended_air_yards"]]
                 + ["ngs_pass_" + c for c in ["avg_time_to_throw", "aggressiveness",
                                              "completion_percentage_above_expectation", "avg_air_yards_differential",
                                              "avg_intended_air_yards"]])
# the v3 model's kept groups per model (for the like-for-like comparison)
V3_KEPT = {
    "fantasy_points_ppr|QB": ["off_style", "investment", "weather", "usage"],
    "fantasy_points_ppr|RB": ["scheme", "run_quality", "investment", "usage", "injury"],
    "fantasy_points_ppr|WR/TE": ["scheme", "scheme_eff", "run_quality", "usage", "injury"],
    "passing_yards|QB": ["pace", "coach", "scheme_eff", "player_cov", "off_style", "run_quality", "red_zone",
                         "investment", "weather", "usage", "injury"],
    "rushing_yards|QB/RB": ["coach", "scheme", "scheme_eff", "off_style", "run_quality", "red_zone", "investment",
                            "weather", "usage", "injury"],
    "receiving_yards|RB/WR/TE": ["pace", "scheme", "player_cov", "run_quality", "weather", "usage", "injury"],
    "receptions|RB/WR/TE": ["scheme", "investment", "usage", "injury"],
}
NEW_V4 = ["receiving", "career_decay", "career_seasons", "consensus", "ngs", "xfp", "depth", "injury_final",
          "protection", "qb_context", "context", "playcaller"]

PARAMS = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7, num_threads=2)
SEL_PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=31, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=2, max_bin=63)
MODELS = [
    ("fantasy_points_ppr", ["QB"]), ("fantasy_points_ppr", ["RB"]), ("fantasy_points_ppr", ["WR", "TE"]),
    ("passing_yards", ["QB"]), ("rushing_yards", ["QB", "RB"]),
    ("receiving_yards", ["RB", "WR", "TE"]), ("receptions", ["RB", "WR", "TE"]),
]
FOLDS = [2023, 2024, 2025, 2026]
SEL_SEEDS = (7, 23)


def all_features():
    return BASE + [c for g in GROUPS.values() for c in g]


def fit(train, valid, target, X, objective="regression", alpha=None, rounds=None, seed=7):
    p = dict(PARAMS, objective=objective, seed=seed)
    if alpha is not None:
        p["alpha"] = alpha
    dtr = lgb.Dataset(train[X], train[target])
    if rounds is None:
        dva = lgb.Dataset(valid[X], valid[target])
        m = lgb.train(p, dtr, 4000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
        return m, m.best_iteration
    return lgb.train(p, dtr, rounds), rounds


def metrics(y, pred):
    err = pred - y
    return dict(n=int(len(y)), mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))),
                r2=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
                corr=float(np.corrcoef(y, pred)[0, 1]))


def select_groups(tr, va, target, pos):
    """Drop-one test on the validation season (2 seeds). The training data is binned once and
    groups are switched off with LightGBM's per-feature weights (feature_contri = 0)."""
    trp, vap = tr[tr["position"].isin(pos)], va[va["position"].isin(pos)]
    X = all_features()
    dtr = lgb.Dataset(trp[X].astype(np.float32), trp[target], free_raw_data=False, params={"max_bin": 63})
    dva = lgb.Dataset(vap[X].astype(np.float32), vap[target], reference=dtr, free_raw_data=False)
    idx = {c: i for i, c in enumerate(X)}

    def score(off_cols):
        contri = [1.0] * len(X)
        for c in off_cols:
            contri[idx[c]] = 0.0
        out = []
        for s in SEL_SEEDS:
            p = dict(SEL_PARAMS, seed=s, feature_contri=contri)
            mod = lgb.train(p, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
            out.append(float(np.sqrt(np.mean((mod.predict(vap[X].astype(np.float32),
                                                          num_iteration=mod.best_iteration) - vap[target]) ** 2))))
        return float(np.mean(out))
    every = [c for g in GROUPS.values() for c in g]
    full = score([])
    res = {"all_rmse": full, "base_rmse": score(every), "groups": {}}
    keep = []
    for g, cols in GROUPS.items():
        r = score(cols)
        res["groups"][g] = {"rmse_without": r, "gain": r - full}
        if r > full:
            keep.append(g)
    res["kept"] = keep
    return res


def apply_forecast(up, season, week):
    """Fill this week's weather from ref/weather_<season>_wk<week>.csv (keyed by home team), if present.
    Rain is the forecast chance of rain (0-1); wind in mph."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", f"weather_{season}_wk{week}.csv")
    if not os.path.exists(path) or len(up) == 0:
        return up
    wx = pd.read_csv(path).set_index("home_team")
    games = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "games.parquet"))
    home = up["game_id"].map(games.set_index("game_id")["home_team"])
    for c in ["wx_temp", "wx_wind", "wx_rain", "wx_snow", "wx_dome"]:
        up[c] = home.map(wx[c]).where(home.isin(wx.index), up[c])
    return up


def frame_cached(data_dir, season, week, path):
    if path and os.path.exists(path):
        return pd.read_parquet(path)
    df = build_frame(data_dir, season, week)
    df["upcoming"] = df["upcoming"].eq(True)
    if path:
        df.to_parquet(path)
    return df


def run(data_dir, out_dir, season, week, ablation=True, frame_path=None):
    os.makedirs(out_dir, exist_ok=True)
    df = frame_cached(data_dir, season, week, frame_path)
    hist = df[~df["upcoming"] & (df["games_hist"] >= 1) & (df["season"] >= 2018)]
    up = apply_forecast(df[df["upcoming"]].copy(), season, week)
    report = {"folds": {}, "prediction_week": {"season": season, "week": week}}
    preds = []
    production = {}
    for S in FOLDS:
        tr, va, te = hist[hist["season"] <= S - 2], hist[hist["season"] == S - 1], hist[hist["season"] == S]
        fold = {}
        tp = te[["player_id", "player_display_name", "position", "team", "season", "week", "game_id"]].copy()
        for target, pos in MODELS:
            key = f"{target}|{'/'.join(pos)}"
            sel = select_groups(tr, va, target, pos) if ablation else {"kept": list(GROUPS)}
            X = BASE + [c for g in sel["kept"] for c in GROUPS[g]]
            X3 = BASE + [c for g in V3_KEPT[key] for c in GROUPS[g]]
            trp, vap, tep = (d[d["position"].isin(pos)] for d in (tr, va, te))
            m, n_rounds = fit(trp, vap, target, X)
            m3, _ = fit(trp, vap, target, X3)
            mb, _ = fit(trp, vap, target, BASE)
            tp.loc[tep.index, f"pred_{target}"] = m.predict(tep[X])
            tp.loc[tep.index, f"v3_{target}"] = m3.predict(tep[X3])
            tp.loc[tep.index, f"v1_{target}"] = mb.predict(tep[BASE])
            tp.loc[tep.index, f"actual_{target}"] = tep[target]
            tp.loc[tep.index, f"ewm_{target}"] = tep[f"{target}_ewm"].fillna(0)
            if target == "fantasy_points_ppr":
                for a in (0.1, 0.9):
                    qm, qr = fit(trp, vap, target, X, objective="quantile", alpha=a)
                    tp.loc[tep.index, f"q{int(a*100)}_{target}"] = qm.predict(tep[X])
                    if S == FOLDS[-1]:
                        production.setdefault(key, {})[f"q{int(a*100)}_rounds"] = qr
            imp = pd.Series(m.feature_importance("gain"), index=X)
            gimp = {"form & Vegas (v1)": float(imp[BASE].sum() / imp.sum())}
            for g in sel["kept"]:
                gimp[g] = float(imp[GROUPS[g]].sum() / imp.sum())
            fold[key] = {"selection": sel, "rounds": n_rounds, "group_importance": gimp,
                         "top_features": (imp / imp.sum()).sort_values(ascending=False).head(20).round(4).to_dict()}
            if S == FOLDS[-1]:
                production.setdefault(key, {}).update({"kept": sel["kept"], "rounds": n_rounds})
            print(f"fold {S} {key:28s} kept={sel['kept']}", flush=True)
        report["folds"][str(S)] = fold
        preds.append(tp)
        pd.concat(preds).to_csv(os.path.join(out_dir, "walkforward_predictions.csv"), index=False)
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=float)
    wf = pd.concat(preds)
    report["summary"] = summarize(wf)

    # production models: groups chosen on the latest fold, trained on every season through last week;
    # a second variant without the weekly consensus rank for weeks where it isn't posted yet
    for target, pos in MODELS:
        key = f"{target}|{'/'.join(pos)}"
        kept = production[key]["kept"]
        X = BASE + [c for g in kept for c in GROUPS[g]]
        Xn = [c for c in X if not c.startswith("ecr_week_")]
        allp = hist[hist["position"].isin(pos)]
        upp = up[up["position"].isin(pos)]
        if len(upp) == 0:
            continue
        final, _ = fit(allp, None, target, X, rounds=production[key]["rounds"])
        final_n, _ = fit(allp, None, target, Xn, rounds=production[key]["rounds"])
        has = upp["ecr_week_ecr"].notna()
        pred = np.where(has, final.predict(upp[X]), final_n.predict(upp[Xn]))
        up.loc[upp.index, f"pred_{target}"] = pred
        if target == "fantasy_points_ppr":
            for a in (0.1, 0.9):
                qf, _ = fit(allp, None, target, Xn, objective="quantile", alpha=a,
                            rounds=production[key][f"q{int(a*100)}_rounds"])
                up.loc[upp.index, f"pred_fpts_q{int(a*100)}"] = qf.predict(upp[Xn])
    report["production"] = production
    report["weekly_consensus_available_for_prediction_week"] = bool(up["ecr_week_ecr"].notna().any())
    keep = [c for c in ["player_id", "player_display_name", "position", "team", "opponent_team", "is_home",
                        "implied_pts", "pace_exp_plays", "fantasy_points_ppr_ewm", "games_hist", "practice_dnp",
                        "ij_questionable", "ij_doubtful", "rc_tprr", "rc_yprr", "rc_adot", "rc_route_participation",
                        "rc_first_read_share", "xfp_ewm", "xfp_fpoe_ewm", "xfp_td_luck_ewm", "dc_depth_rank",
                        "dc_depth_change", "qb_changed", "ngs_rush_rush_yards_over_expected_per_att",
                        "ngs_rece_avg_separation", "ecr_ros_pos_rank", "ecr_preseason_pos_rank",
                        "cr_fantasy_points_ppr_career_wtd", "cr_fantasy_points_ppr_shrunk", "coach"]
            if c in up.columns] + [c for c in up.columns if c.startswith("pred_")]
    up = up[keep].sort_values("pred_fantasy_points_ppr", ascending=False)
    up.to_csv(os.path.join(out_dir, f"predictions_{season}_wk{week}.csv"), index=False)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    return report, up


def summarize(wf):
    out = {}
    for target in dict(MODELS):
        d = wf.dropna(subset=[f"pred_{target}"])
        res = {}
        for S, g in list(d.groupby("season")) + [("all", d)]:
            y = g[f"actual_{target}"].values
            r = {k: metrics(y, g[f"{k}_{target}"].values) for k in ["pred", "v3", "v1", "ewm"]}
            if target == "fantasy_points_ppr":
                r["q10_coverage"] = float(np.mean(y <= g["q10_fantasy_points_ppr"].values))
                r["q90_coverage"] = float(np.mean(y <= g["q90_fantasy_points_ppr"].values))
                r["by_position"] = {p: {k: metrics(gg[f"actual_{target}"].values, gg[f"{k}_{target}"].values)
                                        for k in ["pred", "v3", "v1"]} for p, gg in g.groupby("position")}
            res[str(S)] = r
        out[target] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=5)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--frame", default=None, help="cache the feature frame here (parquet)")
    a = ap.parse_args()
    rep, up = run(a.data, a.out, a.season, a.week, ablation=not a.no_ablation, frame_path=a.frame)
    print(up.head(15)[["player_display_name", "position", "team", "opponent_team",
                       "pred_fantasy_points_ppr"]].to_string())

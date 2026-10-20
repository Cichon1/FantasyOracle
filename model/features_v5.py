"""
v5 candidate features (each tested on its own in the honest walk-forward before it's kept).

  trend      short-term vs long-term form: last-3 and 4-game-weighted averages minus the 12-game average
             (targets, target share, carries, carry share, PPR points, air-yards share, expected points,
             snap share)
  half       second-half role: the player's share of team targets / carries after halftime of his last
             game, its change vs the first half, and a fast-decaying average (catches mid-game promotions
             and benchings before multi-week averages move)
  slot       alignment proxy from where targets go: share of targets over the middle, short (<5 air
             yards) and behind the line; the defense's receiving PPR allowed per game on middle vs outside
             targets; matchup = player's middle share x defense middle allowed + outside share x outside
  teamfirst  team-first volume: expected team dropbacks and rushes (expected plays x team pass rate) and
             the player's expected targets / carries from his share, with vacated targets and carries from
             teammates ruled Out/Doubtful handed out in proportion to the remaining shares
  backupqb   QB change x receiver depth of target, short-target share and target share; new starter's
             EPA x the receiver's share
  injtype    soft-tissue (hamstring, calf, groin, quad) and other lower-body injury on this week's report,
             x questionable / limited; first game back after missing 2+ games
  rookie     combine measurables (40, speed score, height, weight, vertical, broad, cone, shuttle), draft
             slot, age at draft, x rookie flag

All features are known before kickoff (post-game state attached to later games, or this week's report).
"""
import glob
import os

import numpy as np
import pandas as pd

import advanced_features as af
import features_v4 as f4

SOFT = r"hamstring|calf|groin|quad|thigh|achilles"
LOWER = r"ankle|knee|foot|toe|hip|heel|shin|leg|ACL|MCL"


def trend_features(df):
    out = pd.DataFrame(index=df.index)
    for s in ["targets", "target_share", "carries", "carry_share", "fantasy_points_ppr", "air_yards_share",
              "receptions", "wopr"]:
        out[f"tr_{s}_l3_vs_long"] = df[f"{s}_l3"] - df[f"{s}_ewm_long"]
        out[f"tr_{s}_ewm_vs_long"] = df[f"{s}_ewm"] - df[f"{s}_ewm_long"]
    out["tr_xfp_vs_long"] = df["xfp_ewm"] - df["xfp_ewm_long"]
    out["tr_snap_last_vs_ewm"] = df["snap_pct_last"] - df["snap_pct_ewm"]
    out["tr_tprr_vs_long"] = df["rc_tprr"] - df["rc_tprr_long"]
    return out


def _pbp(data_dir):
    return f4.load_pbp(data_dir, ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
                                  "qtr", "qb_dropback", "qb_spike", "two_point_attempt", "receiver_player_id",
                                  "rusher_player_id", "pass_location", "air_yards", "receiving_yards",
                                  "complete_pass", "pass_touchdown", "rush", "qb_scramble", "play_type"])


def half_and_slot_tables(data_dir, stats):
    p = _pbp(data_dir)
    p = p[(p["two_point_attempt"] != 1) & (p["qb_spike"] != 1)]
    tgt = p[(p["qb_dropback"] == 1) & p["receiver_player_id"].notna()].copy()
    car = p[(p["rush"] == 1) & p["rusher_player_id"].notna() & (p["qb_scramble"] != 1)].copy()
    tgt["h2"] = (tgt["qtr"] >= 3).astype(float)
    car["h2"] = (car["qtr"] >= 3).astype(float)
    # player-game counts by half
    pt = tgt.groupby(["game_id", "receiver_player_id"]).agg(t_all=("h2", "size"), t_h2=("h2", "sum")).reset_index() \
        .rename(columns={"receiver_player_id": "player_id"})
    pc = car.groupby(["game_id", "rusher_player_id"]).agg(c_all=("h2", "size"), c_h2=("h2", "sum")).reset_index() \
        .rename(columns={"rusher_player_id": "player_id"})
    tt = tgt.groupby(["game_id", "posteam"]).agg(tt_all=("h2", "size"), tt_h2=("h2", "sum")).reset_index() \
        .rename(columns={"posteam": "team"})
    tc = car.groupby(["game_id", "posteam"]).agg(tc_all=("h2", "size"), tc_h2=("h2", "sum")).reset_index() \
        .rename(columns={"posteam": "team"})
    # alignment proxy: where the player's targets go
    tgt["mid"] = (tgt["pass_location"] == "middle").astype(float)
    tgt["loc_known"] = tgt["pass_location"].notna().astype(float)
    tgt["short"] = (tgt["air_yards"] < 5).astype(float).where(tgt["air_yards"].notna())
    tgt["behind"] = (tgt["air_yards"] <= 0).astype(float).where(tgt["air_yards"].notna())
    tgt["ay_known"] = tgt["air_yards"].notna().astype(float)
    ps = tgt.groupby(["game_id", "receiver_player_id"])[["mid", "loc_known", "short", "behind", "ay_known"]].sum() \
        .reset_index().rename(columns={"receiver_player_id": "player_id"})
    bb = stats[["player_id", "game_id", "team", "season", "week"]].drop_duplicates(["player_id", "game_id"])
    bb = bb[bb["season"] >= 2016]
    bb = bb.merge(pt, on=["game_id", "player_id"], how="left").merge(pc, on=["game_id", "player_id"], how="left") \
        .merge(tt, on=["game_id", "team"], how="left").merge(tc, on=["game_id", "team"], how="left") \
        .merge(ps, on=["game_id", "player_id"], how="left")
    z = ["t_all", "t_h2", "c_all", "c_h2", "mid", "loc_known", "short", "behind", "ay_known"]
    bb[z] = bb[z].fillna(0)
    bb["t_h1"], bb["c_h1"] = bb["t_all"] - bb["t_h2"], bb["c_all"] - bb["c_h2"]
    bb["tt_h1"], bb["tc_h1"] = bb["tt_all"] - bb["tt_h2"], bb["tc_all"] - bb["tc_h2"]
    bb = bb.sort_values(["player_id", "season", "week"])
    bb["hf_tgt_share_h2_last"] = bb["t_h2"] / bb["tt_h2"].replace(0, np.nan)
    bb["hf_tgt_share_h1_last"] = bb["t_h1"] / bb["tt_h1"].replace(0, np.nan)
    bb["hf_car_share_h2_last"] = bb["c_h2"] / bb["tc_h2"].replace(0, np.nan)
    bb["hf_car_share_h1_last"] = bb["c_h1"] / bb["tc_h1"].replace(0, np.nan)
    bb["hf_tgt_h2_minus_h1_last"] = bb["hf_tgt_share_h2_last"] - bb["hf_tgt_share_h1_last"]
    bb["hf_car_h2_minus_h1_last"] = bb["hf_car_share_h2_last"] - bb["hf_car_share_h1_last"]
    half = af.roll(bb, "player_id", ["season", "week"], {"hf_tgt_share_h2_ewm": ("t_h2", "tt_h2"),
                                                          "hf_car_share_h2_ewm": ("c_h2", "tc_h2")}, halflife=2)
    half_cols = ["hf_tgt_share_h2_last", "hf_car_share_h2_last", "hf_tgt_h2_minus_h1_last", "hf_car_h2_minus_h1_last",
                 "hf_tgt_share_h2_ewm", "hf_car_share_h2_ewm"]
    slot = af.roll(bb, "player_id", ["season", "week"], {"sl_mid_share": ("mid", "loc_known"),
                                                          "sl_short_share": ("short", "ay_known"),
                                                          "sl_behind_los_share": ("behind", "ay_known")}, halflife=8)
    slot_cols = ["sl_mid_share", "sl_short_share", "sl_behind_los_share"]
    # defense: receiving PPR allowed per game on middle vs outside targets
    tgt["ppr"] = (tgt["receiving_yards"].fillna(0) * 0.1 + tgt["complete_pass"].fillna(0)
                  + 6 * tgt["pass_touchdown"].fillna(0))
    tgt["ppr_mid"] = tgt["ppr"].where(tgt["mid"] == 1, 0)
    tgt["ppr_out"] = tgt["ppr"].where((tgt["mid"] == 0) & (tgt["loc_known"] == 1), 0)
    dg = tgt.groupby(["game_id", "season", "week", "defteam"], as_index=False)[["ppr_mid", "ppr_out"]].sum() \
        .rename(columns={"defteam": "team"})
    dg = af.roll(dg, "team", ["season", "week"], {"sl_def_mid_ppr_pg": ("ppr_mid", None),
                                                  "sl_def_out_ppr_pg": ("ppr_out", None)}, halflife=8)
    return (half[["player_id", "season", "week"] + half_cols], half_cols,
            slot[["player_id", "season", "week"] + slot_cols], slot_cols,
            dg[["team", "season", "week", "sl_def_mid_ppr_pg", "sl_def_out_ppr_pg"]])


def team_pass_rate_table(data_dir):
    p = f4.load_pbp(data_dir, ["game_id", "play_id", "season", "week", "season_type", "posteam", "qb_dropback",
                               "rush", "qb_spike", "two_point_attempt", "play_type"])
    p = p[((p["qb_dropback"] == 1) | (p["rush"] == 1)) & (p["qb_spike"] != 1) & (p["two_point_attempt"] != 1)]
    t = p.groupby(["game_id", "season", "week", "posteam"], as_index=False).agg(
        db=("qb_dropback", "sum"), plays=("qb_dropback", "size")).rename(columns={"posteam": "team"})
    return af.roll(t, "team", ["season", "week"], {"tf_team_pass_rate": ("db", "plays")}, halflife=6)


def injury_type_features(df, data_dir):
    inj = pd.concat([pd.read_parquet(f, columns=["season", "week", "gsis_id", "report_primary_injury",
                                                  "practice_primary_injury", "report_status", "practice_status"])
                     for f in sorted(glob.glob(os.path.join(data_dir, "inj", "*.parquet")))], ignore_index=True)
    inj = inj.rename(columns={"gsis_id": "player_id"}).drop_duplicates(["player_id", "season", "week"], keep="last")
    txt = inj["report_primary_injury"].fillna("") + " " + inj["practice_primary_injury"].fillna("")
    inj["it_soft_tissue"] = txt.str.contains(SOFT, case=False).astype(float)
    inj["it_lower_body"] = txt.str.contains(LOWER, case=False).astype(float)
    x = df[["player_id", "season", "week"]].merge(inj[["player_id", "season", "week", "it_soft_tissue", "it_lower_body"]],
                                                  on=["player_id", "season", "week"], how="left")
    out = pd.DataFrame(index=df.index)
    out["it_soft_tissue"] = x["it_soft_tissue"].fillna(0).values
    out["it_lower_body"] = x["it_lower_body"].fillna(0).values
    q = df["ij_questionable"].fillna(0) + df["practice_limited"].fillna(0) + df["practice_dnp"].fillna(0)
    out["it_soft_x_flagged"] = out["it_soft_tissue"] * (q > 0)
    out["it_soft_x_wr"] = out["it_soft_tissue"] * df["position"].isin(["WR", "TE"]).astype(float)
    out["it_first_game_back"] = ((df["weeks_since_last"] >= 3) & (df["weeks_since_last"] < 20)).astype(float)
    return out


def rookie_features(df, data_dir, players):
    path = os.path.join(data_dir, "combine.parquet")
    out = pd.DataFrame(index=df.index)
    if not os.path.exists(path):
        return out
    c = pd.read_parquet(path)
    ids = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    c = c.merge(ids, on="pfr_id", how="inner").drop_duplicates("gsis_id")

    def inches(h):
        try:
            a, b = str(h).split("-")
            return int(a) * 12 + int(b)
        except Exception:
            return np.nan
    c["ht_in"] = c["ht"].map(inches)
    c["speed_score"] = c["wt"] * 200 / c["forty"] ** 4
    c = c.set_index("gsis_id")
    for col, name in [("forty", "rk_forty"), ("speed_score", "rk_speed_score"), ("ht_in", "rk_height"), ("wt", "rk_weight"),
                      ("vertical", "rk_vertical"), ("broad_jump", "rk_broad"), ("cone", "rk_cone"), ("shuttle", "rk_shuttle")]:
        out[name] = df["player_id"].map(c[col]).astype(float)
    bd = pd.to_datetime(df["player_id"].map(players.drop_duplicates("gsis_id").set_index("gsis_id")["birth_date"]),
                        errors="coerce")
    dy = df["player_id"].map(players.drop_duplicates("gsis_id").set_index("gsis_id")["draft_year"])
    out["rk_draft_age"] = (pd.to_datetime(dy.astype("Int64").astype(str) + "-04-25", errors="coerce") - bd).dt.days / 365.25
    rook = df["cr_is_rookie"].fillna(0)
    out["rk_rookie_x_draft"] = rook * df["inv_draft_overall"]
    out["rk_rookie_x_speed"] = rook * out["rk_speed_score"]
    out["rk_rookie_x_games"] = rook * df["cr_season_games"].fillna(0)
    return out


def build_v5(df, data_dir, stats, players):
    """All candidate groups, aligned to df.index. Returns (features frame, {group: cols})."""
    groups = {}
    tr = trend_features(df)
    groups["trend"] = list(tr.columns)
    half, half_cols, slot, slot_cols, dslot = half_and_slot_tables(data_dir, stats)
    hf = af.attach_prior(df, half, "player_id", "player_id", half_cols)
    groups["half"] = half_cols
    sl = af.attach_prior(df, slot, "player_id", "player_id", slot_cols)
    sl = sl.join(af.attach_prior(df, dslot, "opponent_team", "team", ["sl_def_mid_ppr_pg", "sl_def_out_ppr_pg"]))
    sl["sl_matchup_ppr"] = (sl["sl_mid_share"] * sl["sl_def_mid_ppr_pg"]
                            + (1 - sl["sl_mid_share"]) * sl["sl_def_out_ppr_pg"])
    groups["slot"] = list(sl.columns)
    tp = team_pass_rate_table(data_dir)
    tf = af.attach_prior(df, tp, "team", "team", ["tf_team_pass_rate"])
    plays = df["pace_exp_plays"]
    tf["tf_exp_dropbacks"] = plays * tf["tf_team_pass_rate"]
    tf["tf_exp_rushes"] = plays * (1 - tf["tf_team_pass_rate"])
    ts, cs = df["target_share_ewm"].fillna(0), df["carry_share_ewm"].fillna(0)
    vt, vc = df["ij_vacated_tgt"].fillna(0).clip(0, 0.9), df["ij_vacated_car"].fillna(0).clip(0, 0.9)
    # hand vacated volume to the remaining players in proportion to their shares (never more than the
    # vacated share itself, never above 100%)
    tf["tf_tgt_share_adj"] = np.minimum(ts / (1 - vt), ts + vt).clip(upper=1)
    tf["tf_car_share_adj"] = np.minimum(cs / (1 - vc), cs + vc).clip(upper=1)
    tf["tf_exp_targets"] = tf["tf_tgt_share_adj"] * tf["tf_exp_dropbacks"] * 0.88
    tf["tf_exp_carries"] = tf["tf_car_share_adj"] * tf["tf_exp_rushes"]
    tf["tf_exp_touch_pts"] = tf["tf_exp_targets"] * 1.6 + tf["tf_exp_carries"] * 0.6
    tf["tf_exp_tds"] = df["implied_pts"] / 7 * (df["rz_target_share"].fillna(0) * 0.5
                                                + df["rz_goal_line_carry_share"].fillna(0) * 0.5)
    groups["teamfirst"] = list(tf.columns)
    bq = pd.DataFrame(index=df.index)
    chg = df["qb_changed"].fillna(0)
    bq["bq_x_adot"] = chg * df["rc_adot"]
    bq["bq_x_short_share"] = chg * sl["sl_short_share"]
    bq["bq_x_tgt_share"] = chg * df["target_share_ewm"]
    bq["bq_new_starter_inexp"] = chg * (df["qb_starter_starts"].fillna(0) < 8)
    bq["bq_qb_epa_x_share"] = df["qb_starter_epa"] * df["target_share_ewm"]
    groups["backupqb"] = list(bq.columns)
    it = injury_type_features(df, data_dir)
    groups["injtype"] = list(it.columns)
    rk = rookie_features(df, data_dir, players)
    groups["rookie"] = list(rk.columns)
    feats = pd.concat([tr, hf, sl, tf, bq, it, rk], axis=1).replace([np.inf, -np.inf], np.nan)
    return feats, groups

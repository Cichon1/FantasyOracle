"""
Advanced feature groups for the NFL player model.

Builds one play-level table (nflverse play-by-play + participation coverage /
personnel data + FTN charting), then rolls it up into features that are known
*before* each game:

  pace        – offensive plays per game, neutral-situation seconds per play,
                no-huddle rate, and the same allowed by the opponent
  coach       – head coach's neutral pass rate over expected (PROE), overall and
                split by the defensive look faced (light / heavy box, base vs
                nickel/dime), plus matchup versions weighted by the opponent's
                tendencies. Follows the coach across teams.
  scheme      – defense's blitz, man, zone, Cover 0/1/2/3/4, two-high vs
                single-high, base/nickel/dime, light/heavy box, pressure rates
  scheme_eff  – defense's EPA allowed per dropback in man, zone, two-high,
                single-high, blitz and non-blitz looks, and per rush
  player_cov  – each player's production vs man, zone, two-high, single-high,
                blitz (QB) and box counts (RB), and matchup versions weighted
                by the opponent's rates
  weather     – temperature, wind, rain, snow, dome
  usage       – offensive snap share (PFR snap counts)
  injury      – player's own final practice status, and the target / carry share
                of teammates who did not practice (vacated opportunity), plus a
                starting-QB-missing flag

Every rolling feature is shifted so a game only sees games before it.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

import fastops as fo

TWO_HIGH = {"COVER_2", "2_MAN", "COVER_4", "COVER_6"}
SINGLE_HIGH = {"COVER_0", "COVER_1", "COVER_3"}
SCHEME_RATES = ["blitz", "man", "zone", "cover0", "cover1", "cover2", "cover3", "cover4", "two_high",
                "single_high", "pressure", "base", "nickel", "dime", "light_box", "heavy_box"]
DB_TOKENS = {"DB", "CB", "FS", "SS", "S", "SAF"}
LB_TOKENS = {"LB", "ILB", "OLB", "MLB"}

PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "home_team",
            "away_team", "play_type", "pass", "rush", "qb_dropback", "xpass", "pass_oe", "epa", "success",
            "wp", "qtr", "half_seconds_remaining", "game_seconds_remaining", "no_huddle", "down",
            "fixed_drive", "receiver_player_id", "rusher_player_id", "passer_player_id", "yards_gained",
            "complete_pass", "receiving_yards", "rushing_yards", "roof", "temp", "wind", "weather",
            "home_coach", "away_coach", "qb_kneel", "qb_spike", "sack", "qb_scramble", "two_point_attempt",
            "shotgun", "yardline_100", "ydstogo"]


# --------------------------------------------------------------------- play table
def _parse_personnel(s):
    if not isinstance(s, str) or not s:
        return np.nan, np.nan
    db = lb = 0
    for n, tok in re.findall(r"(\d+)\s+([A-Z]+)", s):
        if tok in DB_TOKENS:
            db += int(n)
        elif tok in LB_TOKENS:
            lb += int(n)
    return db, lb


def _parse_off_personnel(s):
    """(backs incl. fullbacks, tight ends) from strings like '1 RB, 2 TE, 2 WR' or '... 1 FB, 1 RB ...'."""
    if not isinstance(s, str) or not s:
        return np.nan, np.nan
    backs = te = 0
    for n, tok in re.findall(r"(\d+)\s+([A-Z]+)", s):
        if tok in ("RB", "FB", "HB"):
            backs += int(n)
        elif tok == "TE":
            te += int(n)
    return backs, te


def build_plays(data_dir, cache=True):
    path = os.path.join(data_dir, "plays.parquet")
    if cache and os.path.exists(path):
        return pd.read_parquet(path)
    pbp = pd.concat([pd.read_parquet(f, columns=PBP_COLS)
                     for f in sorted(glob.glob(os.path.join(data_dir, "pbp", "pbp_*.parquet")))], ignore_index=True)
    pbp = pbp[pbp["season_type"].isin(["REG", "POST"])]
    pbp["play_id"] = pbp["play_id"].astype(int)

    part = pd.concat([pd.read_parquet(f, columns=["nflverse_game_id", "play_id", "defense_personnel",
                                                   "defenders_in_box", "number_of_pass_rushers",
                                                   "defense_man_zone_type", "defense_coverage_type",
                                                   "was_pressure", "offense_personnel"])
                      for f in sorted(glob.glob(os.path.join(data_dir, "part", "part_*.parquet")))],
                     ignore_index=True)
    part = part.rename(columns={"nflverse_game_id": "game_id"})
    part["play_id"] = part["play_id"].astype(int)
    uniq = part["defense_personnel"].dropna().unique()
    pmap = {s: _parse_personnel(s) for s in uniq}
    part["def_db"] = part["defense_personnel"].map(lambda s: pmap.get(s, (np.nan, np.nan))[0])
    part["def_lb"] = part["defense_personnel"].map(lambda s: pmap.get(s, (np.nan, np.nan))[1])
    part = part.drop(columns="defense_personnel")
    uo = part["offense_personnel"].dropna().unique()
    omap = {x: _parse_off_personnel(x) for x in uo}
    part["off_backs"] = part["offense_personnel"].map(lambda x: omap.get(x, (np.nan, np.nan))[0])
    part["off_te"] = part["offense_personnel"].map(lambda x: omap.get(x, (np.nan, np.nan))[1])
    part = part.drop(columns="offense_personnel")

    ftn = pd.concat([pd.read_parquet(f, columns=["nflverse_game_id", "nflverse_play_id", "n_blitzers",
                                                  "n_pass_rushers", "n_defense_box", "is_play_action",
                                                  "n_offense_backfield", "is_motion"])
                     for f in sorted(glob.glob(os.path.join(data_dir, "ftn", "ftn_*.parquet")))],
                    ignore_index=True)
    ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})

    df = pbp.merge(part, on=["game_id", "play_id"], how="left").merge(ftn, on=["game_id", "play_id"], how="left")
    df = df[(df["pass"] == 1) | (df["rush"] == 1)]
    df = df[(df["qb_kneel"] != 1) & (df["qb_spike"] != 1) & (df["two_point_attempt"] != 1)]

    # unified box / rushers: FTN charting where it exists (2022+), else participation
    box = df["n_defense_box"].where(df["n_defense_box"] > 0, df["defenders_in_box"])
    df["box"] = box.where(box > 0)
    rushers = df["n_pass_rushers"].where(df["n_pass_rushers"] > 0, df["number_of_pass_rushers"])
    df["rushers"] = rushers.where((rushers > 0) & (df["qb_dropback"] == 1))
    df["blitz"] = (df["rushers"] >= 5).astype(float).where(df["rushers"].notna())
    cov = df["defense_coverage_type"].where(df["defense_coverage_type"].isin(TWO_HIGH | SINGLE_HIGH))
    df["cov_known"] = cov.notna().astype(float)
    for name, s in {"cover0": {"COVER_0"}, "cover1": {"COVER_1"}, "cover2": {"COVER_2", "2_MAN"},
                    "cover3": {"COVER_3"}, "cover4": {"COVER_4", "COVER_6"}}.items():
        df[name] = cov.isin(s).astype(float).where(cov.notna())
    df["two_high"] = cov.isin(TWO_HIGH).astype(float).where(cov.notna())
    df["single_high"] = cov.isin(SINGLE_HIGH).astype(float).where(cov.notna())
    mz = df["defense_man_zone_type"].where(df["defense_man_zone_type"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"]))
    df["man"] = (mz == "MAN_COVERAGE").astype(float).where(mz.notna())
    df["zone"] = (mz == "ZONE_COVERAGE").astype(float).where(mz.notna())
    df["pressure"] = df["was_pressure"].astype(float).where(df["qb_dropback"] == 1)
    df["base"] = (df["def_db"] <= 4).astype(float).where(df["def_db"] > 0)
    df["nickel"] = (df["def_db"] == 5).astype(float).where(df["def_db"] > 0)
    df["dime"] = (df["def_db"] >= 6).astype(float).where(df["def_db"] > 0)
    df["def_lb"] = df["def_lb"].where(df["def_db"] > 0)
    df["light_box"] = (df["box"] <= 6).astype(float).where(df["box"].notna())
    df["heavy_box"] = (df["box"] >= 8).astype(float).where(df["box"].notna())
    # offensive style (the "old school" signals: under center, play-action, heavy sets, fullbacks)
    ftn_known = df["n_blitzers"].notna()
    df["under_center"] = (1 - df["shotgun"]).astype(float)
    df["play_action"] = df["is_play_action"].astype(float).where(ftn_known & (df["qb_dropback"] == 1))
    df["motion"] = df["is_motion"].astype(float).where(ftn_known)
    df["two_back"] = (df["n_offense_backfield"] >= 2).astype(float).where(ftn_known)
    known_pers = (df["off_backs"] + df["off_te"]) > 0
    df["heavy_pers"] = ((df["off_backs"] + df["off_te"]) >= 3).astype(float).where(known_pers)
    df["fb_on_field"] = (df["off_backs"] >= 2).astype(float).where(known_pers)
    designed_run = (df["rush"] == 1) & (df["qb_scramble"] != 1) & (df["qb_dropback"] != 1)
    df["designed_run"] = designed_run.astype(float)
    ry = df["rushing_yards"].fillna(df["yards_gained"])
    df["explosive_run"] = (ry >= 10).astype(float).where(designed_run)
    df["inside10"] = (df["yardline_100"] <= 10).astype(float)
    df["inside20"] = (df["yardline_100"] <= 20).astype(float)
    df["short_yardage"] = ((df["ydstogo"] <= 2) & (df["down"] >= 3)) | (df["yardline_100"] <= 2)
    df["neutral"] = ((df["wp"].between(0.2, 0.8)) & (df["half_seconds_remaining"] > 120)
                     & (df["qtr"] <= 4)).astype(int)
    df = df.sort_values(["game_id", "fixed_drive", "game_seconds_remaining"], ascending=[True, True, False])
    nxt = df.groupby(["game_id", "fixed_drive"])["game_seconds_remaining"].shift(-1)
    sec = df["game_seconds_remaining"] - nxt
    df["sec_to_next"] = sec.where((sec > 0) & (sec <= 60) & (df["neutral"] == 1))
    df["coach"] = np.where(df["posteam"] == df["home_team"], df["home_coach"], df["away_coach"])
    keep = [c for c in df.columns if c not in ("defense_man_zone_type", "defense_coverage_type", "was_pressure",
                                               "n_defense_box", "defenders_in_box", "n_pass_rushers",
                                               "number_of_pass_rushers", "n_blitzers", "is_play_action", "is_motion",
                                               "n_offense_backfield")]
    df = df[keep].reset_index(drop=True)
    if cache:
        df.to_parquet(path)
    return df


# ------------------------------------------------------------------ rolling helper
def roll(frame, key, order, ratios, halflife):
    """ratios: {feature_name: (numerator_col, denominator_col or None)}.

    Returns the frame with each feature = EW mean of numerator / EW mean of denominator
    *including* the row's own game (state after the game). attach_prior() then hands each
    later game the state from the last game strictly before it, so nothing leaks.
    """
    frame = frame.sort_values([key] + order).copy()
    sm = {}
    cols = {c for pair in ratios.values() for c in pair if c}
    for c in cols:
        sm[c] = fo.gewm(frame[c], frame[key], halflife)
    for name, (n, d) in ratios.items():
        frame[name] = sm[n] / sm[d] if d else sm[n]
    return frame


# ------------------------------------------------------------ as-available (honest backtest)
# Participation charting (coverage, personnel, pressure, routes) for a season is published only after
# the season ends, so during season S the newest known value of any feature built from it is the state at
# the end of season S-1. FTN charting (box count, pass rushers, play-action, motion) is published in-season
# from 2022; before that, box / rushers came from participation too.
AS_AVAILABLE = os.environ.get("NFL_AS_AVAILABLE", "1") == "1"
FTN_FIRST_SEASON = 2022


PART_RATES = ["man", "zone", "cover0", "cover1", "cover2", "cover3", "cover4", "two_high", "single_high", "pressure",
              "base", "nickel", "dime"]          # participation only
BOX_RATES = ["blitz", "light_box", "heavy_box"]  # participation before 2022, FTN (in-season) from 2022


def carry_prior_season(tb, key, part_cols=(), box_cols=()):
    """Return a copy of an attach table where the participation-based columns hold, for every row of season S,
    the key's state at the end of its latest earlier season, plus a synthetic week-0 row per key-season that
    carries that state into the first game of S. part_cols: carried in every season; box_cols: carried only
    before FTN charting existed (FTN_FIRST_SEASON). Use the result only with attach_prior()."""
    part_cols = [c for c in part_cols if c in tb.columns]
    box_cols = [c for c in box_cols if c in tb.columns]
    cols = part_cols + box_cols
    if not AS_AVAILABLE or not cols:
        return tb
    tb = tb.dropna(subset=[key]).sort_values([key, "season", "week"]).reset_index(drop=True)
    tail = tb.groupby([key, "season"], sort=False).tail(1).set_index([key, "season"])
    last_nn = tb.groupby([key, "season"], sort=False)[cols].last()   # last non-missing value in the season
    syn = tail.copy()
    syn[cols] = last_nn[cols]
    syn = syn.reset_index()
    syn["season"] = syn["season"] + 1
    syn["week"] = 0
    syn["_syn"] = True
    tb["_syn"] = False
    out = pd.concat([tb, syn], ignore_index=True)
    out = out.sort_values([key, "season", "week"], kind="mergesort").reset_index(drop=True)
    real = ~out["_syn"]
    for c in cols:
        mask = real if c in part_cols else real & (out["season"] < FTN_FIRST_SEASON)
        carried = out[c].where(out["_syn"]).groupby(out[key]).ffill()
        out.loc[mask, c] = carried[mask]
    return out.drop(columns="_syn")


def attach_prior(target, table, key_target, key_table, cols, prefix=""):
    """For each target row, take `cols` from the latest table row with the same key and an
    earlier (season, week). Works for history and for upcoming games alike."""
    t = target[[key_target, "season", "week"]].copy()
    t["_t"] = (t["season"] * 100 + t["week"]).astype("int64")
    t["_i"] = np.arange(len(t))
    tb = table[[key_table, "season", "week"] + cols].copy()
    tb["_t"] = (tb["season"] * 100 + tb["week"]).astype("int64")
    tb = tb.drop(columns=["season", "week"]).rename(columns={key_table: key_target})
    t = t.dropna(subset=[key_target]).sort_values("_t")
    tb = tb.dropna(subset=[key_target]).sort_values("_t")
    m = pd.merge_asof(t, tb, on="_t", by=key_target, allow_exact_matches=False)
    m = m.set_index("_i").reindex(np.arange(len(target)))
    res = m[cols].copy()
    res.columns = [prefix + c for c in cols]
    res.index = target.index
    return res


# ------------------------------------------------------------- team-level features
def team_game_aggregates(plays):
    p = plays
    p = p.assign(
        one=1.0,
        db=p["qb_dropback"].astype(float),
        rsh=((p["rush"] == 1) & (p["qb_dropback"] != 1)).astype(float),
        n_neutral=p["neutral"].astype(float),
        proe_n=p["pass_oe"].where(p["neutral"] == 1),
        nh=p["no_huddle"].astype(float),
    )
    p["proe_cnt"] = p["proe_n"].notna().astype(float)
    for look in ["light_box", "heavy_box", "base"]:
        flag = p[look] == 1
        p[f"proe_{look}"] = p["proe_n"].where(flag)
        p[f"cnt_{look}"] = p[f"proe_{look}"].notna().astype(float)
    sub = (p["nickel"] == 1) | (p["dime"] == 1)
    p["proe_sub"] = p["proe_n"].where(sub)
    p["cnt_sub"] = p["proe_sub"].notna().astype(float)
    midbox = p["box"].between(7, 7)
    p["proe_mid_box"] = p["proe_n"].where(midbox)
    p["cnt_mid_box"] = p["proe_mid_box"].notna().astype(float)
    p["sec_cnt"] = p["sec_to_next"].notna().astype(float)
    # defense-side numerators restricted to dropbacks / rushes
    dbk = p["qb_dropback"] == 1
    for col in ["blitz", "man", "zone", "cover0", "cover1", "cover2", "cover3", "cover4", "two_high",
                "single_high", "pressure"]:
        p[f"{col}_k"] = p[col].notna().astype(float).where(dbk, 0)
        p[col] = p[col].where(dbk)
    for col in ["base", "nickel", "dime", "light_box", "heavy_box"]:
        p[f"{col}_k"] = p[col].notna().astype(float)
    p["lb_k"] = p["def_lb"].notna().astype(float)
    p["epa_db"] = p["epa"].where(dbk)
    p["epa_rush"] = p["epa"].where(p["rsh"] == 1)
    for look in ["man", "zone", "two_high", "single_high", "blitz"]:
        p[f"epa_{look}"] = p["epa"].where(dbk & (p[look] == 1))
        p[f"n_{look}"] = (dbk & (p[look] == 1)).astype(float)
    p["epa_noblitz"] = p["epa"].where(dbk & (p["blitz"] == 0))
    p["n_noblitz"] = (dbk & (p["blitz"] == 0)).astype(float)

    keys = ["game_id", "season", "week"]
    off = p.groupby(keys + ["posteam", "coach"], as_index=False).agg(
        plays=("one", "sum"), neutral_plays=("n_neutral", "sum"), sec_sum=("sec_to_next", "sum"),
        sec_cnt=("sec_cnt", "sum"), nh_sum=("nh", "sum"),
        proe_sum=("proe_n", "sum"), proe_cnt=("proe_cnt", "sum"),
        **{f"proe_{l}_sum": (f"proe_{l}", "sum") for l in ["light_box", "heavy_box", "mid_box", "base", "sub"]},
        **{f"cnt_{l}": (f"cnt_{l}", "sum") for l in ["light_box", "heavy_box", "mid_box", "base", "sub"]},
    ).rename(columns={"posteam": "team"})
    agg = dict(plays_allowed=("one", "sum"), dropbacks=("db", "sum"), rushes=("rsh", "sum"),
               sec_sum_allowed=("sec_to_next", "sum"), sec_cnt_allowed=("sec_cnt", "sum"),
               epa_db_sum=("epa_db", "sum"), epa_rush_sum=("epa_rush", "sum"), lb_sum=("def_lb", "sum"),
               lb_k=("lb_k", "sum"))
    for col in ["blitz", "man", "zone", "cover0", "cover1", "cover2", "cover3", "cover4", "two_high",
                "single_high", "pressure", "base", "nickel", "dime", "light_box", "heavy_box"]:
        agg[f"{col}_sum"] = (col, "sum")
        agg[f"{col}_k"] = (f"{col}_k", "sum")
    for look in ["man", "zone", "two_high", "single_high", "blitz", "noblitz"]:
        agg[f"epa_{look}_sum"] = (f"epa_{look}", "sum")
        agg[f"n_{look}"] = (f"n_{look}", "sum")
    de = p.groupby(keys + ["defteam"], as_index=False).agg(**agg).rename(columns={"defteam": "team"})
    return off, de


def team_features(plays):
    off, de = team_game_aggregates(plays)
    order = ["season", "week"]
    # pace + offense tendencies by team (recent) --------------------------------
    off = roll(off, "team", order, {
        "pace_plays_pg": ("plays", None),
        "pace_sec_per_play": ("sec_sum", "sec_cnt"),
        "pace_no_huddle_rate": ("nh_sum", "plays"),
        "team_proe": ("proe_sum", "proe_cnt"),
    }, halflife=6)
    # coach tendencies (follow the coach across teams, longer memory) ------------
    off = roll(off, "coach", order, {
        "coach_proe": ("proe_sum", "proe_cnt"),
        "coach_proe_light_box": ("proe_light_box_sum", "cnt_light_box"),
        "coach_proe_mid_box": ("proe_mid_box_sum", "cnt_mid_box"),
        "coach_proe_heavy_box": ("proe_heavy_box_sum", "cnt_heavy_box"),
        "coach_proe_vs_base": ("proe_base_sum", "cnt_base"),
        "coach_proe_vs_sub": ("proe_sub_sum", "cnt_sub"),
    }, halflife=16)
    off = off.sort_values(["coach", "season", "week"])
    off["coach_games"] = off.groupby("coach").cumcount() + 1  # games coached through this one

    rates = {f"def_{c}_rate": (f"{c}_sum", f"{c}_k") for c in SCHEME_RATES}
    rates["def_avg_lb"] = ("lb_sum", "lb_k")
    rates["def_pass_rate_faced"] = ("dropbacks", "plays_allowed")
    rates["def_plays_allowed_pg"] = ("plays_allowed", None)
    rates["def_sec_per_play_allowed"] = ("sec_sum_allowed", "sec_cnt_allowed")
    rates["def_epa_per_dropback"] = ("epa_db_sum", "dropbacks")
    rates["def_epa_per_rush"] = ("epa_rush_sum", "rushes")
    for look in ["man", "zone", "two_high", "single_high", "blitz", "noblitz"]:
        rates[f"def_epa_in_{look}"] = (f"epa_{look}_sum", f"n_{look}")
    de = roll(de, "team", order, rates, halflife=8)

    # league-wide rates by week (to express each defense relative to the league at the time;
    # also absorbs charting-source changes between seasons)
    sum_cols = [f"{c}_sum" for c in SCHEME_RATES] + [f"{c}_k" for c in SCHEME_RATES]
    lg = de.groupby(["season", "week"], as_index=False)[sum_cols].sum()
    lg["lg"] = "NFL"
    lg = roll(lg, "lg", order, {f"lg_{c}_rate": (f"{c}_sum", f"{c}_k") for c in SCHEME_RATES}, halflife=4)
    return off, de, lg, list(rates)


# ------------------------------------------------------------ player-level splits
def player_cov_features(plays):
    """Player production split by the coverage / pressure / box look, prior games only."""
    p = plays[(plays["qb_dropback"] == 1) | (plays["rush"] == 1)].copy()
    dbk = p["qb_dropback"] == 1
    out = []
    # receivers: target share of team dropbacks + yards per team dropback, by look
    looks = ["man", "zone", "two_high", "single_high"]
    pdb = p[dbk]
    team_db = (pdb[looks] == 1).groupby([pdb["game_id"], pdb["posteam"]]).sum()
    team_db.columns = [f"tdb_{l}" for l in looks]
    team_db["tdb_all"] = p[dbk].groupby(["game_id", "posteam"]).size()
    tg = p[dbk & p["receiver_player_id"].notna()].copy()
    tg["rec_yds"] = tg["receiving_yards"].fillna(0)
    rows = {}
    for l in looks:
        m = tg[l] == 1
        rows[f"tgt_{l}"] = m.astype(float)
        rows[f"yds_{l}"] = tg["rec_yds"].where(m, 0)
    tg = tg.assign(**rows, tgt_all=1.0)
    rec = tg.groupby(["game_id", "season", "week", "posteam", "receiver_player_id"], as_index=False)[
        [f"tgt_{l}" for l in looks] + [f"yds_{l}" for l in looks] + ["tgt_all"]].sum()
    rec = rec.merge(team_db.reset_index(), on=["game_id", "posteam"], how="left").rename(
        columns={"receiver_player_id": "player_id"})
    ratios = {}
    for l in looks:
        ratios[f"pc_tgt_share_vs_{l}"] = (f"tgt_{l}", f"tdb_{l}")
        ratios[f"pc_yds_per_db_vs_{l}"] = (f"yds_{l}", f"tdb_{l}")
    rec = roll(rec, "player_id", ["season", "week"], ratios, halflife=8)
    out.append(rec[["game_id", "season", "week", "player_id"] + list(ratios)])

    # QBs: EPA per dropback vs blitz / no blitz / man / zone / pressure
    q = p[dbk & p["passer_player_id"].notna()].copy()
    rows = {}
    for l, cond in {"blitz": q["blitz"] == 1, "noblitz": q["blitz"] == 0, "man": q["man"] == 1,
                    "zone": q["zone"] == 1, "two_high": q["two_high"] == 1,
                    "single_high": q["single_high"] == 1}.items():
        rows[f"qepa_{l}"] = q["epa"].where(cond, 0).fillna(0)
        rows[f"qn_{l}"] = cond.astype(float)
    q = q.assign(**rows)
    qa = q.groupby(["game_id", "season", "week", "passer_player_id"], as_index=False)[list(rows)].sum().rename(
        columns={"passer_player_id": "player_id"})
    qr = {f"pc_qb_epa_vs_{l}": (f"qepa_{l}", f"qn_{l}") for l in
          ["blitz", "noblitz", "man", "zone", "two_high", "single_high"]}
    qa = roll(qa, "player_id", ["season", "week"], qr, halflife=10)
    out.append(qa[["game_id", "season", "week", "player_id"] + list(qr)])

    # rushers: yards per carry vs light / normal / heavy box
    r = p[(p["rush"] == 1) & p["rusher_player_id"].notna() & (p["qb_scramble"] != 1)].copy()
    r["ry"] = r["rushing_yards"].fillna(r["yards_gained"]).fillna(0)
    rows = {}
    for l, cond in {"light": r["box"] <= 6, "mid": r["box"] == 7, "heavy": r["box"] >= 8}.items():
        rows[f"ry_{l}"] = r["ry"].where(cond, 0)
        rows[f"rn_{l}"] = cond.astype(float)
    r = r.assign(**rows)
    ra = r.groupby(["game_id", "season", "week", "rusher_player_id"], as_index=False)[list(rows)].sum().rename(
        columns={"rusher_player_id": "player_id"})
    rr = {f"pc_ypc_vs_{l}_box": (f"ry_{l}", f"rn_{l}") for l in ["light", "mid", "heavy"]}
    ra = roll(ra, "player_id", ["season", "week"], rr, halflife=10)
    out.append(ra[["game_id", "season", "week", "player_id"] + list(rr)])
    return out


# --------------------------------------------------------------------- weather
WEATHER_PRECIP = re.compile(r"rain|shower|drizzle|storm|thunder", re.I)
WEATHER_SNOW = re.compile(r"snow|flurr|sleet", re.I)


def game_weather(plays, games):
    w = plays.groupby("game_id", as_index=False).agg(roof=("roof", "first"), temp=("temp", "first"),
                                                     wind=("wind", "first"), weather=("weather", "first"))
    txt = w["weather"].fillna("")
    # pull temp / wind out of the text when the numeric columns are empty
    t_txt = txt.str.extract(r"Temp:\s*(-?\d+)", expand=False).astype(float)
    w_txt = txt.str.extract(r"Wind:\s*\w*\s*(\d+)\s*mph", expand=False).astype(float)
    calm = txt.str.contains(r"Wind:\s*calm", case=False)
    w["temp"] = w["temp"].fillna(t_txt)
    w["wind"] = w["wind"].fillna(w_txt).where(~calm, w["wind"].fillna(0))
    w["dome"] = w["roof"].isin(["dome", "closed"]).astype(int)
    cond = txt.str.split("Temp:").str[0]
    w["snow"] = cond.str.contains(WEATHER_SNOW).astype(int)
    w["rain"] = (cond.str.contains(WEATHER_PRECIP) & (w["snow"] == 0)).astype(int)
    indoor = w["dome"] == 1
    w.loc[indoor, ["wind", "rain", "snow"]] = 0
    w.loc[indoor, "temp"] = 70
    w = w.rename(columns={"temp": "wx_temp", "wind": "wx_wind", "rain": "wx_rain", "snow": "wx_snow",
                          "dome": "wx_dome"})
    # upcoming games: roof from schedule, conditions unknown unless indoors
    g = games[["game_id", "roof"]]
    up = g[~g["game_id"].isin(w["game_id"])].copy()
    up["wx_dome"] = up["roof"].isin(["dome", "closed"]).astype(int)
    for c in ["wx_temp", "wx_wind", "wx_rain", "wx_snow"]:
        up[c] = np.nan
    ind = up["wx_dome"] == 1
    up.loc[ind, ["wx_wind", "wx_rain", "wx_snow"]] = 0
    up.loc[ind, "wx_temp"] = 70
    cols = ["game_id", "wx_temp", "wx_wind", "wx_rain", "wx_snow", "wx_dome"]
    return pd.concat([w[cols], up[cols]], ignore_index=True)


# ------------------------------------------------------------------ snaps / injury
def snap_features(data_dir, players):
    snaps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(data_dir, "snaps", "*.parquet")))],
                      ignore_index=True)
    snaps = snaps.merge(players[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id"),
                        left_on="pfr_player_id", right_on="pfr_id", how="inner")
    snaps = snaps.rename(columns={"gsis_id": "player_id"})[
        ["player_id", "season", "week", "game_type", "offense_pct", "offense_snaps"]]
    snaps = snaps.groupby(["player_id", "season", "week"], as_index=False)[["offense_pct", "offense_snaps"]].max()
    return snaps


def injury_table(data_dir):
    inj = pd.concat([pd.read_parquet(f, columns=["season", "week", "team", "gsis_id", "report_status",
                                                  "practice_status"])
                     for f in sorted(glob.glob(os.path.join(data_dir, "inj", "*.parquet")))], ignore_index=True)
    inj = inj.rename(columns={"gsis_id": "player_id"}).drop_duplicates(["player_id", "season", "week"], keep="last")
    ps = inj["practice_status"].fillna("")
    inj["practice_dnp"] = ps.str.startswith("Did Not").astype(int)
    inj["practice_limited"] = ps.str.startswith("Limited").astype(int)
    inj["on_report"] = 1
    return inj


# ------------------------------------------------- offensive style / run game / red zone
def style_features(plays):
    """Team offensive style and run-game efficiency, plus what each defense allows against it.
    Returns (offense table keyed by team, defense table keyed by team), post-game EW state."""
    p = plays.copy()
    dbk = p["qb_dropback"] == 1
    run = p["designed_run"] == 1
    ry = p["rushing_yards"].fillna(p["yards_gained"]).fillna(0)
    p["one"] = 1.0
    p["run_n"] = run.astype(float)
    p["db_n"] = dbk.astype(float)
    p["stuff"] = (run & (ry <= 0)).astype(float)
    p["gump"] = (run & (ry >= 20)).astype(float)
    p["expl"] = (run & (ry >= 10)).astype(float)
    p["run_epa"] = p["epa"].where(run)
    p["run_succ"] = p["success"].where(run)
    p["db_epa"] = p["epa"].where(dbk)
    for c in ["under_center", "motion", "two_back", "heavy_pers", "fb_on_field"]:
        p[f"{c}_k"] = p[c].notna().astype(float)
    p["pa_k"] = p["play_action"].notna().astype(float)
    pa = dbk & (p["play_action"] == 1)
    p["pa_epa"] = p["epa"].where(pa)
    p["pa_n"] = pa.astype(float)
    p["nopa_epa"] = p["epa"].where(dbk & (p["play_action"] == 0))
    p["nopa_n"] = (dbk & (p["play_action"] == 0)).astype(float)
    pa_fb = pa & (p["fb_on_field"] == 1)
    p["pafb_epa"] = p["epa"].where(pa_fb)
    p["pafb_n"] = pa_fb.astype(float)
    uc_run = run & (p["under_center"] == 1)
    sg_run = run & (p["under_center"] == 0)
    p["uc_run_epa"], p["uc_run_n"] = p["epa"].where(uc_run), uc_run.astype(float)
    p["sg_run_epa"], p["sg_run_n"] = p["epa"].where(sg_run), sg_run.astype(float)
    sy = p["short_yardage"] & ((p["ydstogo"] <= 1) | (p["yardline_100"] <= 1))
    sneak = sy & run & (p["rusher_player_id"] == p["passer_player_id"]) | (sy & run & p["rusher_player_id"].notna()
                                                                          & p["rusher_player_id"].isin([]))
    p["sy_n"] = sy.astype(float)
    p["sy_succ"] = p["success"].where(sy)
    p["heavy_faced"] = (p["heavy_pers"] == 1)
    p["base_vs_heavy"] = p["base"].where(p["heavy_faced"])
    p["base_vs_heavy_k"] = p["base_vs_heavy"].notna().astype(float)
    p["rush_epa_vs_heavy"] = p["epa"].where(run & p["heavy_faced"])
    p["rush_n_vs_heavy"] = (run & p["heavy_faced"]).astype(float)
    p["rush_epa_light"] = p["epa"].where(run & (p["light_box"] == 1))
    p["rush_n_light"] = (run & (p["light_box"] == 1)).astype(float)
    keys = ["game_id", "season", "week"]
    sums = ["one", "run_n", "db_n", "stuff", "gump", "expl", "run_epa", "run_succ", "db_epa", "pa_epa", "pa_n",
            "nopa_epa", "nopa_n", "pafb_epa", "pafb_n", "uc_run_epa", "uc_run_n", "sg_run_epa", "sg_run_n",
            "sy_n", "sy_succ", "pa_k", "play_action", "under_center", "motion", "two_back", "heavy_pers",
            "fb_on_field", "under_center_k", "motion_k", "two_back_k", "heavy_pers_k", "fb_on_field_k",
            "base_vs_heavy", "base_vs_heavy_k", "rush_epa_vs_heavy", "rush_n_vs_heavy", "rush_epa_light",
            "rush_n_light"]
    off = p.groupby(keys + ["posteam"], as_index=False)[sums].sum(min_count=0).rename(columns={"posteam": "team"})
    de = p.groupby(keys + ["defteam"], as_index=False)[sums].sum(min_count=0).rename(columns={"defteam": "team"})
    order = ["season", "week"]
    off_r = {
        "style_under_center_rate": ("under_center", "under_center_k"),
        "style_play_action_rate": ("play_action", "pa_k"),
        "style_motion_rate": ("motion", "motion_k"),
        "style_two_back_rate": ("two_back", "two_back_k"),
        "style_heavy_pers_rate": ("heavy_pers", "heavy_pers_k"),
        "style_fb_rate": ("fb_on_field", "fb_on_field_k"),
        "off_run_epa": ("run_epa", "run_n"),
        "off_run_success": ("run_succ", "run_n"),
        "off_dropback_epa": ("db_epa", "db_n"),
        "off_pa_epa": ("pa_epa", "pa_n"),
        "off_nopa_epa": ("nopa_epa", "nopa_n"),
        "off_pa_fb_epa": ("pafb_epa", "pafb_n"),
        "off_uc_run_epa": ("uc_run_epa", "uc_run_n"),
        "off_sg_run_epa": ("sg_run_epa", "sg_run_n"),
        "off_stuff_rate": ("stuff", "run_n"),
        "off_explosive_run_rate": ("expl", "run_n"),
        "off_gump_rate": ("gump", "run_n"),
        "off_short_yd_success": ("sy_succ", "sy_n"),
        "off_short_yd_pg": ("sy_n", None),
    }
    off = roll(off, "team", order, off_r, halflife=6)
    de_r = {
        "def_stuff_rate": ("stuff", "run_n"),
        "def_explosive_run_rate": ("expl", "run_n"),
        "def_gump_rate": ("gump", "run_n"),
        "def_run_success_allowed": ("run_succ", "run_n"),
        "def_pa_epa_allowed": ("pa_epa", "pa_n"),
        "def_base_vs_heavy_rate": ("base_vs_heavy", "base_vs_heavy_k"),
        "def_rush_epa_vs_heavy": ("rush_epa_vs_heavy", "rush_n_vs_heavy"),
        "def_rush_epa_light_box": ("rush_epa_light", "rush_n_light"),
        "def_short_yd_success_allowed": ("sy_succ", "sy_n"),
        "def_heavy_faced_rate": ("heavy_pers", "heavy_pers_k"),
    }
    de = roll(de, "team", order, de_r, halflife=8)
    return off, de, list(off_r), list(de_r)


def player_usage_features(plays, stats):
    """Per-player rushing quality and red-zone / goal-line role, on a player-game backbone
    (so games with zero such touches count as zeros)."""
    p = plays
    run = p["designed_run"] == 1
    ry = p["rushing_yards"].fillna(p["yards_gained"]).fillna(0)
    r = p[run & p["rusher_player_id"].notna()].assign(
        car=1.0, stuff=(ry <= 0).astype(float), expl=(ry >= 10).astype(float), gump=(ry >= 20).astype(float),
        ry=ry, repa=p["epa"], rsucc=p["success"],
        uc_car=p["under_center"], uc_yds=ry.where(p["under_center"] == 1, 0),
        sg_car=1 - p["under_center"], sg_yds=ry.where(p["under_center"] == 0, 0),
        i10=p["inside10"], i5=(p["yardline_100"] <= 5).astype(float),
        sy=p["short_yardage"].astype(float), sy_succ=p["success"].where(p["short_yardage"], 0))
    rg = r.groupby(["game_id", "rusher_player_id"], as_index=False)[
        ["car", "stuff", "expl", "gump", "ry", "repa", "rsucc", "uc_car", "uc_yds", "sg_car", "sg_yds", "i10", "i5",
         "sy", "sy_succ"]].sum().rename(columns={"rusher_player_id": "player_id"})
    tgt = p[(p["qb_dropback"] == 1) & p["receiver_player_id"].notna()].assign(
        rz_tgt=p["inside20"], i10_tgt=p["inside10"])
    tg = tgt.groupby(["game_id", "receiver_player_id"], as_index=False)[["rz_tgt", "i10_tgt"]].sum().rename(
        columns={"receiver_player_id": "player_id"})
    team = p.assign(t_i10_car=(run & (p["inside10"] == 1)).astype(float),
                    t_rz_tgt=((p["qb_dropback"] == 1) & p["receiver_player_id"].notna() & (p["inside20"] == 1)).astype(float))
    team = team.groupby(["game_id", "posteam"], as_index=False)[["t_i10_car", "t_rz_tgt"]].sum().rename(
        columns={"posteam": "team"})
    qb = p[(p["qb_dropback"] == 1) & p["passer_player_id"].notna() & p["play_action"].notna()].assign(
        q_pa_epa=p["epa"].where(p["play_action"] == 1, 0).fillna(0), q_pa_n=(p["play_action"] == 1).astype(float))
    qg = qb.groupby(["game_id", "passer_player_id"], as_index=False)[["q_pa_epa", "q_pa_n"]].sum().rename(
        columns={"passer_player_id": "player_id"})
    bb = stats[["player_id", "game_id", "team", "season", "week"]].drop_duplicates(["player_id", "game_id"])
    bb = bb.merge(rg, on=["game_id", "player_id"], how="left").merge(tg, on=["game_id", "player_id"], how="left") \
        .merge(team, on=["game_id", "team"], how="left").merge(qg, on=["game_id", "player_id"], how="left")
    zero = ["car", "stuff", "expl", "gump", "ry", "uc_car", "uc_yds", "sg_car", "sg_yds", "i10", "i5", "sy",
            "sy_succ", "rz_tgt", "i10_tgt", "t_i10_car", "t_rz_tgt"]
    bb[zero] = bb[zero].fillna(0)
    ratios = {
        "ru_ypc": ("ry", "car"), "ru_stuff_rate": ("stuff", "car"), "ru_explosive_rate": ("expl", "car"),
        "ru_gump_rate": ("gump", "car"), "ru_epa_per_carry": ("repa", "car"), "ru_success_rate": ("rsucc", "car"),
        "ru_ypc_under_center": ("uc_yds", "uc_car"), "ru_ypc_shotgun": ("sg_yds", "sg_car"),
        "rz_goal_line_carry_share": ("i10", "t_i10_car"), "rz_target_share": ("rz_tgt", "t_rz_tgt"),
        "rz_inside5_carries_pg": ("i5", None), "rz_inside10_targets_pg": ("i10_tgt", None),
        "sy_carries_pg": ("sy", None), "sy_success_rate": ("sy_succ", "sy"),
        "qb_pa_epa": ("q_pa_epa", "q_pa_n"),
    }
    bb = roll(bb, "player_id", ["season", "week"], ratios, halflife=8)
    return bb[["player_id", "season", "week"] + list(ratios)], list(ratios)


def contract_features(data_dir):
    """Salary-cap share of the player's current contract and draft capital."""
    c = pd.read_parquet(os.path.join(data_dir, "contracts.parquet"),
                        columns=["gsis_id", "year_signed", "apy_cap_pct", "years", "draft_overall", "draft_year"])
    c = c.dropna(subset=["gsis_id"]).rename(columns={"gsis_id": "player_id"})
    return c

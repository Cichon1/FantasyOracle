"""
v6 candidate features, computed on the finished feature frame (rows sorted by anything; upcoming rows included).

  ecr_prev   last weekly FantasyPros positional rank / ECR posted before this week (same season) -- known on
             Thursday, before this week's ranking
  qbs        QB form over previous starts only (15+ pass attempts), so a backup's relief appearances don't drag
             down his projection when he becomes the starter: points / passing yards / rushing yards / attempts
             (exponential average, half-life 6 starts), mean of the last 8 starts, career and season starts
  tq         for RB / WR / TE: the projected starting QB's per-start form and its change from the team's previous
             game (catches a QB change in either direction)
Each uses only games before the row's game.
"""
import numpy as np
import pandas as pd

import fastops as fo

TARGET = "fantasy_points_ppr"
ECR_PREV = ["ecr_prev_pos_rank", "ecr_prev_ecr"]
QBS = ["qbs_pts_ewm", "qbs_pts_l8", "qbs_pass_yds_ewm", "qbs_rush_yds_ewm", "qbs_att_ewm", "qbs_n_starts",
       "qbs_season_starts"]
TQ = ["tq_pts_ewm", "tq_n_starts", "tq_pts_delta"]


def _last8_mean(x):
    out = np.full(len(x), np.nan)
    vals = []
    for i, v in enumerate(x):
        if vals:
            out[i] = np.mean(vals[-8:])
        if not np.isnan(v):
            vals.append(v)
    return out


def build_v6(df):
    """-> DataFrame (same index as df) with the v6 candidate columns."""
    out = pd.DataFrame(index=df.index)
    o = df.sort_values(["player_id", "season", "week"])
    pid = o["player_id"]
    for c, n in (("ecr_week_pos_rank", "ecr_prev_pos_rank"), ("ecr_week_ecr", "ecr_prev_ecr")):
        prev = o.groupby(["player_id", "season"])[c].shift(1)
        out[n] = prev.groupby([pid, o["season"]]).ffill().reindex(df.index)
    played = ~o["phantom"].eq(True) & ~o["upcoming"].eq(True)
    start = (o["attempts"].fillna(0) >= 15) & played
    for col, n in ((TARGET, "qbs_pts"), ("passing_yards", "qbs_pass_yds"), ("rushing_yards", "qbs_rush_yds"),
                   ("attempts", "qbs_att")):
        v = o[col].where(start)
        out[n + "_ewm"] = fo.gewm(v, pid, 6, shift=1)
    v = o[TARGET].where(start)
    out["qbs_pts_l8"] = pd.Series(np.concatenate([_last8_mean(g.values) for _, g in v.groupby(pid, sort=False)]),
                                  index=np.concatenate([g.index.values for _, g in v.groupby(pid, sort=False)]))
    s = start.astype(float)
    out["qbs_n_starts"] = fo.gcumsum_prior(s, pid).fillna(0)
    out["qbs_season_starts"] = fo.gcumsum_prior(s, [pid, o["season"]]).fillna(0)
    out = out.reindex(df.index)
    for c in [c for c in out.columns if c.startswith("qbs_")]:
        out[c] = out[c].where(df["position"] == "QB")
    # team's projected starting QB (qb_is_projected_starter marks him among the team's QB rows for the game)
    qb = df[(df["position"] == "QB") & (df["qb_is_projected_starter"] == 1)]
    qb = pd.DataFrame({"game_id": qb["game_id"], "team": qb["team"], "season": qb["season"], "week": qb["week"],
                       "tq_pts_ewm": out.loc[qb.index, "qbs_pts_ewm"], "tq_n_starts": out.loc[qb.index, "qbs_n_starts"]})
    qb = qb.drop_duplicates(["game_id", "team"]).sort_values(["team", "season", "week"])
    qb["tq_pts_delta"] = qb["tq_pts_ewm"] - qb.groupby("team")["tq_pts_ewm"].shift(1)
    m = df[["game_id", "team"]].merge(qb[["game_id", "team"] + TQ], on=["game_id", "team"], how="left")
    m.index = df.index
    for c in TQ:
        out[c] = m[c].where(df["position"] != "QB")
    return out


# ------------------------------------------------------------------------------------- offensive line (free data)
OLINE = ["ol_run_block", "ol_run_block_adj", "opp_run_stop_adj", "ol_run_matchup",
         "ol_pass_block_adj", "opp_pass_rush_adj", "ol_pass_matchup",
         "ol_continuity", "ol_last5_out", "ol_cap_avail", "ol_cap_out", "ol_starts_avail"]
OL_POS = {"T", "G", "C", "OL", "OT", "OG", "LT", "RT", "LG", "RG"}


def _ewm_state(t, key, col, hl):
    """EWM over each key's games in order, state after the game (attach_prior then gives the pre-game value)."""
    return fo.gewm(t[col], t[key] if isinstance(key, str) else [t[k] for k in key], hl)


def build_oline(df, data_dir):
    """Offensive-line features for every frame row (team's line; opponent's front), from games before the row's game
    plus that week's final injury report. Columns: OLINE.

    run blocking   Football Outsiders-style line yards on designed runs (losses x1.2, first 4 yards full, yards
                   5-10 half, beyond 10 none), per game; opponent-adjusted = residual vs the defense's average
                   allowed going into the game; exponential average (half-life 8 games)
    pass blocking  sacks + QB hits per dropback, residual vs the opponent's average generated going in
    personnel      from snap counts: the five linemen with the most snaps are that game's starting line.
                   continuity = how many of them also started the team's previous game (avg, half-life 3 games);
                   for this game: last game's five who are ruled Out / Doubtful, the contract cap % (APY share of the
                   cap at signing) of the rest and of those out, and the available five's starts over this season and the two before
    """
    import glob
    import os
    import advanced_features as af
    plays = pd.read_parquet(os.path.join(data_dir, "plays.parquet"),
                            columns=["game_id", "play_id", "season", "week", "posteam", "defteam", "designed_run",
                                     "qb_dropback", "sack", "yards_gained", "qb_kneel"])
    hits = pd.concat([pd.read_parquet(f, columns=["game_id", "play_id", "qb_hit"])
                      for f in sorted(glob.glob(os.path.join(data_dir, "pbp", "pbp_*.parquet")))])
    p = plays.merge(hits, on=["game_id", "play_id"], how="left")
    keys = ["game_id", "season", "week"]
    # --- run blocking
    r = p[(p["designed_run"] == 1) & (p["qb_kneel"] != 1)].copy()
    y = r["yards_gained"].astype(float)
    r["ly"] = np.where(y < 0, 1.2 * y, np.where(y <= 4, y, np.where(y <= 10, 4 + 0.5 * (y - 4), 7.0)))
    rg = r.groupby(keys + ["posteam", "defteam"], as_index=False)["ly"].mean()
    # --- pass blocking
    d = p[p["qb_dropback"] == 1].copy()
    d["pr"] = ((d["sack"] == 1) | (d["qb_hit"] == 1)).astype(float)
    pg = d.groupby(keys + ["posteam", "defteam"], as_index=False)["pr"].mean()
    g = rg.merge(pg, on=keys + ["posteam", "defteam"], how="outer").sort_values(["season", "week"])
    g["_t"] = g["season"] * 100 + g["week"]
    for side, k in (("off", "posteam"), ("def", "defteam")):
        g = g.sort_values([k, "_t"])
        for c in ("ly", "pr"):
            st = _ewm_state(g, k, c, 8)
            g[f"{side}_{c}_prior"] = st.groupby(g[k]).shift(1)      # going into this game
            g[f"{side}_{c}_state"] = st
    g["ly_res_off"] = g["ly"] - g["def_ly_prior"]                   # line yards above what the defense allows
    g["ly_res_def"] = g["ly"] - g["off_ly_prior"]                   # allowed above what the offense usually gets
    g["pr_res_off"] = g["pr"] - g["def_pr_prior"]                   # pressure allowed above the rush's usual
    g["pr_res_def"] = g["pr"] - g["off_pr_prior"]                   # pressure generated above the line's usual
    g = g.sort_values(["posteam", "_t"])
    g["ol_run_block_adj"] = _ewm_state(g, "posteam", "ly_res_off", 8)
    g["ol_pass_block_adj"] = _ewm_state(g, "posteam", "pr_res_off", 8)
    g["ol_run_block"] = g["off_ly_state"]
    g = g.sort_values(["defteam", "_t"])
    g["opp_run_stop_adj"] = _ewm_state(g, "defteam", "ly_res_def", 8)
    g["opp_pass_rush_adj"] = _ewm_state(g, "defteam", "pr_res_def", 8)
    off = g[["posteam", "season", "week", "ol_run_block", "ol_run_block_adj", "ol_pass_block_adj"]].rename(
        columns={"posteam": "team"})
    de = g[["defteam", "season", "week", "opp_run_stop_adj", "opp_pass_rush_adj"]].rename(columns={"defteam": "team"})
    out = pd.DataFrame(index=df.index)
    out = out.join(af.attach_prior(df, off, "team", "team", ["ol_run_block", "ol_run_block_adj", "ol_pass_block_adj"]))
    out = out.join(af.attach_prior(df, de, "opponent_team", "team", ["opp_run_stop_adj", "opp_pass_rush_adj"]))
    out["ol_run_matchup"] = out["ol_run_block_adj"] + out["opp_run_stop_adj"]
    out["ol_pass_matchup"] = out["ol_pass_block_adj"] + out["opp_pass_rush_adj"]

    # --- personnel: starting five per team-game from snap counts
    sn = pd.concat([pd.read_parquet(f, columns=["game_id", "season", "week", "game_type", "team", "pfr_player_id",
                                                "position", "offense_snaps"])
                    for f in sorted(glob.glob(os.path.join(data_dir, "snaps", "*.parquet")))], ignore_index=True)
    sn = sn[sn["position"].isin(OL_POS)].copy()
    sn["team"] = sn["team"].replace({"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA"})
    pl = pd.read_parquet(os.path.join(data_dir, "players.parquet"), columns=["pfr_id", "gsis_id"]).dropna()
    sn = sn.merge(pl.drop_duplicates("pfr_id"), left_on="pfr_player_id", right_on="pfr_id", how="left")
    sn["pid"] = sn["gsis_id"].fillna(sn["pfr_player_id"])
    sn = sn.sort_values(["game_id", "team", "offense_snaps"], ascending=[True, True, False])
    st5 = sn.groupby(["game_id", "team"]).head(5)
    lineup = st5.groupby(["game_id", "team", "season", "week"])["pid"].apply(frozenset).reset_index()
    lineup = lineup.sort_values(["team", "season", "week"])
    lineup["prev"] = lineup.groupby("team")["pid"].shift(1)
    lineup["kept"] = [len(a & b) if isinstance(b, frozenset) else np.nan for a, b in zip(lineup["pid"], lineup["prev"])]
    lineup["ol_continuity"] = _ewm_state(lineup, "team", "kept", 3)
    out = out.join(af.attach_prior(df, lineup[["team", "season", "week", "ol_continuity"]], "team", "team",
                                   ["ol_continuity"]))
    # starts (games in a team's starting five) before each game
    st5 = st5.sort_values(["pid", "season", "week"])
    st5["starts_before"] = st5.groupby("pid").cumcount()
    # last game's five for each frame row (strictly before), then this week's injury report and contracts
    lu = lineup[["team", "season", "week", "pid"]].copy()
    lu["_t"] = (lu["season"] * 100 + lu["week"]).astype("int64")
    tw = df[["team", "season", "week"]].drop_duplicates().copy()
    tw["_t"] = (tw["season"] * 100 + tw["week"]).astype("int64")
    m = pd.merge_asof(tw.sort_values("_t"), lu[["team", "_t", "pid"]].sort_values("_t"), on="_t", by="team",
                      allow_exact_matches=False)
    inj = pd.concat([pd.read_parquet(f, columns=["season", "week", "gsis_id", "report_status"])
                     for f in sorted(glob.glob(os.path.join(data_dir, "inj", "*.parquet")))], ignore_index=True)
    inj = inj.drop_duplicates(["gsis_id", "season", "week"], keep="last")
    outs = set(map(tuple, inj.loc[inj["report_status"].isin(["Out", "Doubtful"]), ["season", "week", "gsis_id"]]
                   .values.tolist()))
    con = af.contract_features(data_dir)
    cs = con.dropna(subset=["apy_cap_pct"]).sort_values(["player_id", "year_signed"])
    cap = {}
    for pid_, grp in cs.groupby("player_id"):
        cap[pid_] = list(zip(grp["year_signed"].astype(int), grp["apy_cap_pct"].astype(float)))

    def cap_at(pid_, season):
        v = [c for yr, c in cap.get(pid_, []) if yr <= season]
        return v[-1] if v else 0.0
    starts = st5.set_index(["pid", "season", "week"])["starts_before"]
    starts_d = {}
    for (pid_, s_, w_), v in starts.items():
        starts_d.setdefault(pid_, []).append((s_ * 100 + w_, v))
    res = []
    m = m.rename(columns={"_t": "tkey"})
    for row in m.itertuples(index=False):
        five = row.pid if isinstance(row.pid, frozenset) else frozenset()
        o = [x for x in five if (row.season, row.week, x) in outs]
        a = [x for x in five if x not in o]
        st_ = 0
        for x in a:
            # starts in this season and the two before (a fixed window, so the count doesn't grow with data history)
            st_ += sum(1 for t_, v in starts_d.get(x, []) if row.tkey - 300 < t_ < row.tkey)
        res.append((row.team, row.season, row.week, len(o) if five else np.nan,
                    sum(cap_at(x, row.season) for x in a) if five else np.nan,
                    sum(cap_at(x, row.season) for x in o) if five else np.nan, st_ if five else np.nan))
    res = pd.DataFrame(res, columns=["team", "season", "week", "ol_last5_out", "ol_cap_avail", "ol_cap_out",
                                     "ol_starts_avail"])
    x = df[["team", "season", "week"]].merge(res, on=["team", "season", "week"], how="left")
    for c in ["ol_last5_out", "ol_cap_avail", "ol_cap_out", "ol_starts_avail"]:
        out[c] = x[c].values
    return out[OLINE]


# ------------------------------------------------------------------------------------- market structure
MARKET = ["mkt_share_ewm", "mkt_pts", "mkt_qb_start_share"]


def build_market(df, data_dir):
    """Player's recent share of his team's skill-position fantasy points (exponential average over his previous games,
    half-life 4) and Vegas implied team points x that share (a "market only" projection)."""
    import glob
    import os
    import advanced_features as af
    st = pd.concat([pd.read_parquet(f, columns=["player_id", "season", "week", "team", "position", "fantasy_points_ppr"])
                    for f in sorted(glob.glob(os.path.join(data_dir, "week_*.parquet")))])
    st = st[st["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    st["team"] = st["team"].replace({"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV"})
    tot = st.groupby(["team", "season", "week"])["fantasy_points_ppr"].transform("sum")
    st["share"] = st["fantasy_points_ppr"] / tot.replace(0, np.nan)
    st = st.sort_values(["player_id", "season", "week"])
    st["mkt_share_ewm"] = fo.gewm(st["share"], st["player_id"], 4)
    # QBs: share over games he started (schedule's listed starter), so relief appearances don't set a starter's share
    g = pd.read_parquet(os.path.join(data_dir, "games.parquet"), columns=["season", "week", "home_team", "away_team",
                                                                          "home_qb_id", "away_qb_id"])
    ls = pd.concat([g[["season", "week", "home_qb_id"]].rename(columns={"home_qb_id": "player_id"}),
                    g[["season", "week", "away_qb_id"]].rename(columns={"away_qb_id": "player_id"})]).dropna()
    ls["started"] = True
    qs = st[st["position"] == "QB"].merge(ls.drop_duplicates(["player_id", "season", "week"]),
                                          on=["player_id", "season", "week"], how="inner")
    qs = qs.sort_values(["player_id", "season", "week"])
    qs["mkt_qb_start_share"] = fo.gewm(qs["share"], qs["player_id"], 4)
    out = af.attach_prior(df, st[["player_id", "season", "week", "mkt_share_ewm"]], "player_id", "player_id",
                          ["mkt_share_ewm"])
    out = out.join(af.attach_prior(df, qs[["player_id", "season", "week", "mkt_qb_start_share"]], "player_id",
                                   "player_id", ["mkt_qb_start_share"]))
    out["mkt_pts"] = df["implied_pts"] * out["mkt_share_ewm"]
    return out[MARKET]


def qb_prior_starts(df, data_dir):
    """Number of games each player started at QB (the schedule's listed starting QBs, 1999 on) before the row's game."""
    import os
    g = pd.read_parquet(os.path.join(data_dir, "games.parquet"),
                        columns=["game_id", "gameday", "home_qb_id", "away_qb_id", "home_score"])
    gd = g.set_index("game_id")["gameday"]
    done = g[g["home_score"].notna()]
    s = pd.concat([done[["gameday", "home_qb_id"]].rename(columns={"home_qb_id": "player_id"}),
                   done[["gameday", "away_qb_id"]].rename(columns={"away_qb_id": "player_id"})]).dropna()
    s["_t"] = pd.to_datetime(s["gameday"])
    s = s.sort_values("_t")
    s["n"] = s.groupby("player_id").cumcount() + 1
    k = pd.DataFrame({"player_id": df["player_id"].values, "_t": pd.to_datetime(df["game_id"].map(gd)).values,
                      "_i": np.arange(len(df))}).dropna(subset=["_t"])
    m = pd.merge_asof(k.sort_values("_t"), s[["player_id", "_t", "n"]], on="_t", by="player_id",
                      allow_exact_matches=False)
    out = pd.Series(np.nan, index=np.arange(len(df)))
    out.loc[m["_i"].values] = m["n"].fillna(0).values
    out.index = df.index
    return out.rename("qb_prior_starts")


def qb_market_adjust(rows, w_model, offsets, group=("season", "week"), default_share=0.18):
    """v5.1 QB step, for the QBs the depth chart projects to start (backups are left as the model has them).
    Ranking score = w_model * model + (1 - w_model) * market projection, where the market projection is the team's
    Vegas implied points x the QB's recent share of team skill-position fantasy points in games he started
    (default_share if he has never started), plus, for starters with fewer than 4 prior starts, the out-of-sample
    under-projection for their bucket (0, 1, 2-3 prior starts). The market projection is on a smaller scale than PPR
    points, so the model's own starter point totals are handed out in the order of that score within each week (points
    stay on the model's scale). rows needs pred_ppr_model, position, implied_pts, mkt_qb_start_share,
    qb_is_projected_starter, qb_prior_starts."""
    out = rows["pred_ppr_model"].copy()
    r = rows[(rows["position"] == "QB") & (rows["qb_is_projected_starter"] == 1)]
    if not len(r):
        return out
    mkt = r["implied_pts"] * r["mkt_qb_start_share"].fillna(default_share)
    score = w_model * r["pred_ppr_model"] + (1 - w_model) * mkt.fillna(r["pred_ppr_model"])
    cold = r["qb_prior_starts"] < 4
    b = r.loc[cold, "qb_prior_starts"].map(lambda v: "0" if v < 1 else ("1" if v < 2 else "2-3"))
    score[cold] = score[cold] + b.map(offsets).fillna(0).values
    keys = [c for c in group if c in r.columns]
    for _, g in (r.groupby(keys) if keys else [(None, r)]):
        order = score.loc[g.index].sort_values(ascending=False, kind="mergesort").index
        out.loc[order] = np.sort(g["pred_ppr_model"].values)[::-1]
    return out
    r = rows[q]
    starter = r["qb_is_projected_starter"] == 1
    mkt = r["mkt_pts"].fillna(r["implied_pts"] * np.where(starter, 0.33, 0.03))
    score = w_model * r["pred_ppr_model"] + (1 - w_model) * mkt.fillna(r["pred_ppr_model"])
    cold = starter & (r["qb_prior_starts"] < 4)
    b = r.loc[cold, "qb_prior_starts"].map(lambda v: "0" if v < 1 else ("1" if v < 2 else "2-3"))
    score[cold] = score[cold] + b.map(offsets).fillna(0).values
    keys = [c for c in group if c in r.columns]
    for _, g in r.groupby(keys) if keys else [(None, r)]:
        order = score.loc[g.index].sort_values(ascending=False, kind="mergesort").index
        out.loc[order] = np.sort(g["pred_ppr_model"].values)[::-1]
    return out

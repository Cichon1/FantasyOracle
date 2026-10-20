"""
Rest-of-season (ROS) PPR projections and trade values, built on the weekly model (v5.1).

ROS points for a player at snapshot week W (the earliest week with an unplayed game) =
    the weekly model's official projection for his week-W game, if it hasn't been played (Thursday / Sunday runs)
  + ROS rate x his team's remaining games after week W, through the last fantasy week (17; 16 before 2021).
The ROS rate is a separate model with the weekly model's inputs (minus the week-specific expert rank) plus the
remaining schedule: target = PPR points over weeks W+1..last / team games in that span, zeros included for games
missed, so availability is priced in. Trained on every (season, week 2..last-2) snapshot of past seasons with the
season complete. Players who are out this week but still on the roster (or ranked in the FantasyPros ROS consensus)
are projected from their latest row, flagged as out.

Official ROS = the model's rank averaged with the FantasyPros ROS rank within the position (weight chosen on
validation seasons), with the model's own point totals handed out in that order - the same rule as the weekly
official projection.

Trade value (PPR) = ROS points above replacement, where replacement is the first player at the position left out
of a league's starting lineups (QB, 2 RB, 2 WR, TE, FLEX, plus a superflex in SF leagues), per remaining game.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import features_v4 as f4
import lab5 as L
import modelcache as MC
import nfl_player_model as m

POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
POS_MODELS = [("QB",), ("RB",), ("WR", "TE")]
EXTRA = ["ros_games_after", "ros_bye_after", "ros_opp_def_rel", "ros_out_now", "ros_bye_now", "ros_weeks_since_row",
         "ros_res", "ros_ecr_pos_rank"]
PARAMS = dict(L.LAB_PARAMS, extra_trees=True, min_data_in_leaf=40)


def last_week(season):
    """Last fantasy week: most leagues end the week before the final regular-season week."""
    return 17 if season >= 2021 else 16


# ------------------------------------------------------------------------------------------- context tables
def sched_table(games):
    g = games[games["game_type"] == "REG"]
    cols = ["season", "week", "game_id", "gameday", "home_score"]
    h = g[cols + ["home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opp"})
    a = g[cols + ["away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opp"})
    s = pd.concat([h, a], ignore_index=True)
    s["played"] = s["home_score"].notna()
    return s.drop(columns="home_score")


def def_table(stats, games):
    """Opponent PPR points allowed to each position, relative to league average (exponential average of games
    before each game, half-life 6), with the game's season and week."""
    tg = m.team_game_table(games)
    da = m.defense_allowed(stats, tg)
    wk = tg.drop_duplicates("game_id").set_index("game_id")[["season", "week"]]
    return da.join(wk, on="game_id").dropna(subset=["season", "week"])


def ecr_ros_table(fp_dir, games):
    _, ros, _, _ = f4.consensus_features(os.path.join(fp_dir, "db_fpecr.parquet"),
                                         os.path.join(fp_dir, "db_playerids.csv"), games)
    return ros[["player_id", "scrape_date", "pos", "ecr", "pos_rank"]].sort_values("scrape_date")


def week_cutoff(games, season, week):
    """Sunday of the week (the experts' ROS snapshot is the latest one before it)."""
    g = games[(games["season"] == season) & (games["week"] == week) & (games["game_type"] == "REG")]
    d = pd.to_datetime(g["gameday"])
    sun = d[d.dt.weekday == 6]
    return (sun.min() if len(sun) else d.max()).normalize()


def ecr_snapshot(ros_tab, cutoff, max_age_days=10):
    s = ros_tab[(ros_tab["scrape_date"] < cutoff) & (ros_tab["scrape_date"] >= cutoff - pd.Timedelta(days=max_age_days))]
    if not len(s):
        return pd.DataFrame(columns=["player_id", "ros_ecr_pos_rank", "ros_ecr_date"])
    s = s[s["scrape_date"] == s["scrape_date"].max()]
    s = s.sort_values("pos_rank").drop_duplicates("player_id")
    return s.rename(columns={"pos_rank": "ros_ecr_pos_rank", "scrape_date": "ros_ecr_date"})[
        ["player_id", "ros_ecr_pos_rank", "ros_ecr_date"]]


def roster_asof(roster, season, week):
    if roster is None:
        return None
    r = roster[(roster["season"] == season) & (roster["week"] <= week)]
    if not len(r):
        return None
    r = r[r["week"] == r["week"].max()]
    return r.drop_duplicates("player_id", keep="last").set_index("player_id")


class Context:
    """Everything a snapshot needs besides the feature frame."""

    def __init__(self, data_dir, fp_dir, stats=None, games=None):
        self.data_dir = data_dir
        if stats is None:
            stats, games, _ = m.load(data_dir)
        self.stats, self.games = stats, games
        self.sched = sched_table(games)
        self.dtab = def_table(stats, games)
        self.ros_ecr = ecr_ros_table(fp_dir, games)
        self.roster = m.roster_table(data_dir)
        reg = stats[stats["season_type"] == "REG"]
        self.pts = reg.groupby(["player_id", "season", "week"])["fantasy_points_ppr"].sum()


# ------------------------------------------------------------------------------------------------ snapshots
def schedule_features(ctx, season, week, live=False):
    """Per team and position: games in week W still to play (all of them for a past-season snapshot, the unplayed ones
    live), games after week W through the last fantasy week, byes in that span, and the average PPR points allowed to
    the position (relative to league average, as of week W) by the remaining opponents."""
    lw = last_week(season)
    s = ctx.sched[(ctx.sched["season"] == season) & (ctx.sched["week"] >= week) & (ctx.sched["week"] <= lw)]
    cur = s[(s["week"] == week) & (~s["played"] if live else True)]
    aft = s[s["week"] > week]
    d = ctx.dtab[(ctx.dtab["season"] == season) & (ctx.dtab["week"] >= week)].sort_values("week")
    asof = d.drop_duplicates(["def_team", "position"])[["def_team", "position", "def_allowed_rel"]]
    teams = sorted(set(ctx.sched.loc[ctx.sched["season"] == season, "team"]))
    # per-team slices computed once (the loop below used to filter the schedule 4 x 32 times per snapshot)
    aft_by = {t: g for t, g in aft.groupby("team", sort=False)}
    cur_n = cur["team"].value_counts()
    empty = aft.iloc[:0]
    out = []
    for pos in m.POSITIONS:
        a = asof[asof["position"] == pos].set_index("def_team")["def_allowed_rel"]
        for t in teams:
            ga = aft_by.get(t, empty)
            out.append({"team": t, "position": pos, "ros_games_cur": int(cur_n.get(t, 0)),
                        "ros_games_after": len(ga), "ros_bye_after": max(0, lw - week) - len(ga),
                        "ros_opp_def_rel": a.reindex(ga["opp"]).mean() if len(ga) else np.nan})
    return pd.DataFrame(out)


def season_parts(df):
    """Pieces of the frame every snapshot needs, computed once for many snapshots (build_history): regular-season
    rows by season, all rows by season, and each player's latest regular-season position."""
    reg = df[df["season_type"] == "REG"]
    return {"reg": {S: g for S, g in reg.groupby("season", sort=False)},
            "all": {S: g for S, g in df.groupby("season", sort=False)},
            "pos_last": reg[["player_id", "position"]].drop_duplicates("player_id", keep="last")}


def _rows_for(parts, key, seasons, like):
    if len(seasons) > 1 and not like.index.is_monotonic_increasing:   # keep the frame's own row order
        return like[like["season"].isin(seasons)]
    ck = (key,) + tuple(seasons)
    if ck in parts.setdefault("_combined", {}):
        return parts["_combined"][ck]
    got = [parts[key][S] for S in seasons if S in parts[key]]
    if not got:
        res = like.iloc[:0]
    else:
        res = got[0] if len(got) == 1 else pd.concat(got).sort_index()
    cache = parts["_combined"]
    cache[ck] = res
    while len(cache) > 4:   # snapshots come season by season: keep only the latest few
        cache.pop(next(iter(cache)))
    return res


def snapshot(df, ctx, season, week, live=False, parts=None):
    """One row per player for the snapshot: everyone the weekly rule projects for week W (played or upcoming),
    plus players out this week who were expected in one of the team's previous three weeks and are still on its
    roster, or who are in the experts' ROS top pool. Out players carry their latest row's inputs."""
    lw = last_week(season)
    if parts is None:
        reg = df[df["season_type"] == "REG"]
        pos_last = reg[["player_id", "position"]].drop_duplicates("player_id", keep="last")
        near = df
    else:   # same rows, without filtering the whole frame for every snapshot
        reg = _rows_for(parts, "reg", [season], df)
        pos_last = parts["pos_last"]
        near = _rows_for(parts, "all", [season - 1, season], df)
    cur = reg[(reg["season"] == season) & (reg["week"] == week) & reg["expected"]]
    cur = cur.sort_values("phantom").drop_duplicates("player_id").copy()
    cur["ros_out_now"], cur["ros_weeks_since_row"], cur["ros_bye_now"] = 0, 0, 0
    # a past-season snapshot is taken before the week; live, games already played this week are done
    cur["ros_cur_played"] = (~cur["upcoming"]) if live else False
    ecr = ecr_snapshot(ctx.ros_ecr, week_cutoff(ctx.games, season, week))
    rost = roster_asof(ctx.roster, season, week)
    # out this week: expected in one of the previous three weeks, or in the experts' ROS top pool
    prev = reg[(reg["season"] == season) & (reg["week"] < week) & (reg["week"] >= week - 3) & reg["expected"]]
    cand = set(prev["player_id"])
    if len(ecr):
        e = ecr.merge(pos_last, on="player_id")
        e = e[e["ros_ecr_pos_rank"] <= e["position"].map(POOL) * 1.5]
        cand |= set(e["player_id"])
    cand -= set(cur["player_id"])
    before = near[near["player_id"].isin(cand) & (((near["season"] == season) & (near["week"] < week))
                                                   | (near["season"] == season - 1))]
    before = before[before["season_type"] == "REG"].sort_values(["season", "week", "phantom"])
    out = before.drop_duplicates("player_id", keep="last").copy()
    if rost is not None and len(out):
        st = out["player_id"].map(rost["status"])
        tm = out["player_id"].map(rost["team"])
        keep = st.isin(["ACT", "INA", "RES"]) | st.isna()
        out = out[keep]
        tm = tm[keep]
        out["team"] = tm.where(tm.notna(), out["team"])
    if len(out):
        gap = (season * 25 + week) - (out["season"] * 25 + out["week"])
        out["ros_weeks_since_row"] = gap
        out["weeks_since_last"] = np.where(out["phantom"], out["weeks_since_last"] + gap, gap)
        # team on bye this week: players expected in the team's previous game aren't out, they just don't play
        ts = ctx.sched[(ctx.sched["season"] == season) & (ctx.sched["week"] <= week)]
        bye = ~out["team"].isin(set(ts.loc[ts["week"] == week, "team"]))
        prev_wk = out["team"].map(ts[(ts["week"] < week) & ts["played"]].groupby("team")["week"].max())  # last played game (a mid-week run can have an unplayed game in the previous week)
        on_bye = bye & (out["season"] == season) & (out["week"] == prev_wk) & out["expected"].astype(bool)
        out["ros_out_now"] = np.where(on_bye, 0, 1)
        out["ros_bye_now"] = on_bye.astype(int)
        out["ros_cur_played"] = False
        out["season"], out["week"] = season, week
        out["game_id"] = np.nan
    rows = pd.concat([cur, out], ignore_index=False)
    rows["ros_row_index"] = rows.index
    rows = rows.reset_index(drop=True)
    if rost is not None:
        rows["ros_res"] = (rows["player_id"].map(rost["status"]) == "RES").astype(float)
    else:
        rows["ros_res"] = np.nan
    rows = rows.merge(ecr[["player_id", "ros_ecr_pos_rank"]], on="player_id", how="left")
    sf = schedule_features(ctx, season, week, live)
    rows = rows.drop(columns=[c for c in sf.columns if c not in ("team", "position") and c in rows.columns])
    rows = rows.merge(sf, on=["team", "position"], how="left")
    rows["ros_snap_week"], rows["ros_last_week"] = week, lw
    return rows


def add_targets(rows, ctx):
    """Actual PPR points over weeks W..last and W+1..last (missed games count as zero)."""
    p = ctx.pts.reset_index()
    out = []
    for (S, W), g in rows.groupby(["season", "week"]):
        lw = last_week(int(S))
        q = p[(p["season"] == S) & (p["week"] >= W) & (p["week"] <= lw)]
        tot = q.groupby("player_id")["fantasy_points_ppr"].sum()
        aft = q[q["week"] > W].groupby("player_id")["fantasy_points_ppr"].sum()
        out.append(pd.DataFrame({"ros_pts": g["player_id"].map(tot).fillna(0).values,
                                 "ros_pts_after": g["player_id"].map(aft).fillna(0).values}, index=g.index))
    t = pd.concat(out)
    rows = rows.join(t)
    rows["ros_rate_after"] = rows["ros_pts_after"] / rows["ros_games_after"].where(rows["ros_games_after"] > 0)
    return rows


def build_history(df, ctx, seasons, weeks=None):
    parts = []
    pre = season_parts(df)
    for S in seasons:
        lw = last_week(S)
        for W in (weeks or range(2, lw - 1)):
            if W > lw - 2:
                continue
            parts.append(snapshot(df, ctx, S, W, parts=pre))
    rows = pd.concat(parts, ignore_index=True)
    return add_targets(rows, ctx)


# -------------------------------------------------------------------------------------------------- models
def feature_list(rows):
    X = [c for c in L.feature_list(rows) if not c.startswith("ecr_week_")]
    return X + [c for c in EXTRA if c in rows.columns]


def fit(trn, X, target, rounds, seeds=(7, 23, 41), params=None, objective=None, alpha=None, weight="ros_games_after"):
    p = dict(params or PARAMS)
    if objective:
        p["objective"], p["alpha"] = objective, alpha
    Xtr = trn[X].astype(np.float32)
    w = trn[weight] if weight else None
    dkey = MC.data_key(Xtr.values, trn[target].values, None if w is None else w.values, X) if MC.ON else None
    ds = lgb.Dataset(Xtr, trn[target], weight=w, free_raw_data=False)
    return [MC.lgb_train(lgb, dict(p, seed=s), ds, int(rounds), dkey) for s in seeds]


def predict(models, rows, X):
    return np.mean([b.predict(rows[X].astype(np.float32)) for b in models], axis=0)


def early_stop_rounds(trn, va, X, target, params=None, weight="ros_games_after"):
    p = dict(params or PARAMS, seed=7)

    def run():
        dtr = lgb.Dataset(trn[X].astype(np.float32), trn[target], weight=trn[weight])
        dva = lgb.Dataset(va[X].astype(np.float32), va[target], weight=va[weight], reference=dtr)
        b = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        return int(b.best_iteration)
    if not MC.ON:
        return run()
    dkey = MC.data_key(np.concatenate([trn[X].to_numpy(np.float32), va[X].to_numpy(np.float32)]),
                       np.concatenate([trn[target].values, va[target].values]),
                       np.concatenate([trn[weight].values, va[weight].values]), [len(trn)] + list(X))
    return MC.cached_value("ros_early_stop", dkey, [lgb.__version__, p], run)


def cur_open(rows):
    """1 where the player's week-W game is still to play and he isn't out for it."""
    return (rows["ros_games_cur"].clip(upper=1) * (~rows["ros_cur_played"].astype(bool)) * (1 - rows["ros_out_now"]))


def combine(rows, rate, weekly=None):
    """ROS points = weekly projection for his unplayed week-W game (when given) + rate x team games after week W.
    Without a weekly projection, rate x all remaining games (week W included)."""
    rate = pd.Series(np.asarray(rate, dtype=float), index=rows.index)
    if weekly is None:
        return rate * (rows["ros_games_after"] + cur_open(rows))
    w = pd.Series(np.asarray(weekly, dtype=float), index=rows.index).fillna(0) * cur_open(rows)
    return rate * rows["ros_games_after"] + w


def blend_with_experts(rows, score_col, w_model, group=("season", "week", "position")):
    """Official ROS: weighted rank average of the model and the FantasyPros ROS rank within the position (players the
    experts don't rank count as ranked behind every ranked player, in the model's order); the model's point totals
    are handed out in that order."""
    out = rows[score_col].copy()
    for _, g in rows.groupby(list(group)):
        mr = g[score_col].rank(ascending=False, method="first")
        er = g["ros_ecr_pos_rank"].rank(method="first")
        er = er.fillna(mr + len(g))   # unranked by the experts: behind every ranked player, in the model's order
        sc = w_model * mr + (1 - w_model) * er + mr * 1e-6
        order = sc.sort_values(kind="mergesort").index
        out.loc[order] = np.sort(g[score_col].values)[::-1]
    return out


# --------------------------------------------------------------------------------------------- trade values
LEAGUES = {"10_1qb": (10, False), "12_1qb": (12, False), "14_1qb": (14, False),
           "10_sf": (10, True), "12_sf": (12, True), "14_sf": (14, True)}


def replacement_rates(r, teams, superflex, pts="ros_ppr", games="ros_games"):
    """Fill every team's starting lineup (QB, RB x2, WR x2, TE, FLEX from RB/WR/TE, and a superflex from any position
    in SF leagues) with the best ROS players; replacement at each position = the per-game ROS rate of the best
    player left out."""
    d = r[r[games] > 0].copy()
    d["rate"] = d[pts] / d[games]
    d = d.sort_values(pts, ascending=False)
    used = set()
    need = {"QB": teams, "RB": 2 * teams, "WR": 2 * teams, "TE": teams}
    for p, n in need.items():
        used |= set(d[d["position"] == p].head(n).index)
    used |= set(d[d["position"].isin(["RB", "WR", "TE"]) & ~d.index.isin(used)].head(teams).index)
    if superflex:
        used |= set(d[~d.index.isin(used)].head(teams).index)
    left = d[~d.index.isin(used)]
    return {p: float(left.loc[left["position"] == p, "rate"].iloc[0]) if (left["position"] == p).any() else 0.0
            for p in need}


# players a team typically rosters at each position (about 13-14 skill players per team)
ROSTER_DEPTH = {False: {"QB": 2, "RB": 5, "WR": 5, "TE": 1.5}, True: {"QB": 3, "RB": 4.5, "WR": 4.5, "TE": 1.5}}


def waiver_rates(r, teams, superflex, pts="ros_ppr", games="ros_games"):
    """Waiver-wire level at each position: the per-game ROS rate of the best player beyond the league's typical
    rostered depth (ROSTER_DEPTH x teams)."""
    d = r[r[games] > 0].copy()
    d["rate"] = d[pts] / d[games]
    out = {}
    for p, per in ROSTER_DEPTH[superflex].items():
        s = d[d["position"] == p].sort_values(pts, ascending=False)
        n = int(round(per * teams))
        out[p] = float(s["rate"].iloc[n]) if len(s) > n else 0.0
    return out


def ros_samples(r, n=199):
    """Quantiles of each player's ROS points: a two-piece normal through the 10th percentile (ros_low), the projection
    (ros_ppr) and the 90th percentile (ros_high), floored at 0."""
    from scipy.stats import norm
    z = norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))
    lo = ((r["ros_ppr"] - r["ros_low"]).clip(lower=0.5) / 1.2816).values[:, None]
    hi = ((r["ros_high"] - r["ros_ppr"]).clip(lower=0.5) / 1.2816).values[:, None]
    m = r["ros_ppr"].values[:, None]
    return np.clip(np.where(z[None, :] < 0, m + z[None, :] * lo, m + z[None, :] * hi), 0, None)


def trade_values(r, pts="ros_ppr"):
    """PPR trade value for each league setting: the average of two expected values over the player's ROS range -
    points above a waiver-wire pickup and points above the league's last starter at his position, each over the
    games he has left and counted only when positive (so a bench player keeps value for his upside, and a star's edge
    in a starting lineup counts in full)."""
    out = pd.DataFrame(index=r.index)
    g = r["ros_games"].values[:, None]   # his remaining games (one fewer when he's out this week)
    X = ros_samples(r)
    for name, (teams, sf) in LEAGUES.items():
        rep = replacement_rates(r, teams, sf, pts)
        wav = waiver_rates(r, teams, sf, pts)
        over_w = np.maximum(X - r["position"].map(wav).values[:, None] * g, 0).mean(axis=1)
        over_s = np.maximum(X - r["position"].map(rep).values[:, None] * g, 0).mean(axis=1)
        out[f"val_{name}"] = np.round(0.5 * (over_w + over_s), 1)
        out[f"rep_{name}"] = r["position"].map(rep).round(2)
        out[f"wav_{name}"] = r["position"].map(wav).round(2)
    return out


# ------------------------------------------------------------------------------------------------ live run
def train_rate_models(hist_rows, season, cfg):
    """ROS rate models (mean, 10th and 90th percentile) per position group, trained on every past-season snapshot:
    tree count early-stopped on season-1 (trained through season-2), then refit through season-1 with 1.1x."""
    params = dict(PARAMS, min_data_in_leaf=cfg.get("min_data_in_leaf", 40))
    X = feature_list(hist_rows)
    ok = hist_rows["ros_games_after"] >= 1
    models = {}
    for pos in POS_MODELS:
        p = hist_rows["position"].isin(pos) & ok
        tr, va = hist_rows[p & (hist_rows["season"] <= season - 2)], hist_rows[p & (hist_rows["season"] == season - 1)]
        r = max(50, int(early_stop_rounds(tr, va, X, "ros_rate_after", params) * 1.1))
        full = hist_rows[p & (hist_rows["season"] <= season - 1)]
        models[pos] = {"rounds": r,
                       "mean": fit(full, X, "ros_rate_after", r, cfg.get("seeds", (7, 23, 41)), params),
                       "q10": fit(full, X, "ros_rate_after", r, cfg.get("q_seeds", (7,)), params, "quantile", 0.1),
                       "q90": fit(full, X, "ros_rate_after", r, cfg.get("q_seeds", (7,)), params, "quantile", 0.9)}
    return models, X


def weekly_proxy(df, rows, season, week, cfg_all):
    """Model-only weekly projection for the snapshot week when no weekly run has projected it yet (the Tuesday
    refresh): the extra-trees member of the weekly model without the weekly expert-ranking inputs, trained on every
    game before the week, plus the v5.1 QB market step - the setting the ROS test used for its no-ranking runs."""
    import features_v6 as f6
    import weekly_v5 as W5
    hist = L.hist_rows(df)
    trn = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))]
    X = [c for c in L.feature_list(hist) if not c.startswith("ecr_week_")]
    src = df.loc[rows["ros_row_index"]].copy()
    src = m.apply_forecast(src, season, week)
    pred = pd.Series(np.nan, index=src.index)
    for pos in L.POS_MODELS:
        key = "/".join(pos)
        tp, rp = trn[trn["position"].isin(pos)], src[src["position"].isin(pos)]
        if len(rp):
            pred.loc[rp.index] = W5.train_predict(tp, rp, X, L.TARGET, dict(L.LAB_PARAMS, extra_trees=True,
                                                                             min_data_in_leaf=10),
                                                  cfg_all["rounds_noweek"]["et10"][key] * 1.1, [7, 23, 41])
    src["pred_ppr_model"] = pred
    if cfg_all.get("qb_market") and "mkt_qb_start_share" in src:
        qm = cfg_all["qb_market"]
        src["pred_ppr_model"] = f6.qb_market_adjust(src, qm["w_model"], qm["cold_offsets"])
    return src["pred_ppr_model"].values


def project_live(df, ctx, season, week, cfg_all, store=None, log=print):
    """ROS projections for the snapshot week. store: the dashboard's weekly projections (week rows used when they
    exist for this week); otherwise a model-only weekly projection is made here."""
    cfg = cfg_all["ros"]
    hist_rows = build_history(df, ctx, list(range(2018, season)))
    log(f"ROS training snapshots: {len(hist_rows)}")
    models, X = train_rate_models(hist_rows, season, cfg)
    snap = snapshot(df, ctx, season, week, live=True)
    for pos in POS_MODELS:
        sp = snap["position"].isin(pos)
        if sp.any():
            for k, col in [("mean", "rate"), ("q10", "rate_q10"), ("q90", "rate_q90")]:
                snap.loc[sp, col] = predict(models[pos][k], snap[sp], X)
    snap["rate"] = snap["rate"].clip(lower=0)
    # this week's projection: the dashboard's official weekly projection when this week has been projected
    snap["wk"], snap["wk_low"], snap["wk_high"], snap["wk_status"], snap["injury"] = np.nan, np.nan, np.nan, None, None
    wk_src = "none"
    if store is not None and (store["week"] == week).any():
        s = store[store["week"] == week].drop_duplicates("player_id").set_index("player_id")
        snap["wk"] = snap["player_id"].map(s["proj_ppr"])
        snap["wk_low"], snap["wk_high"] = snap["player_id"].map(s["ppr_low"]), snap["player_id"].map(s["ppr_high"])
        snap["wk_status"], snap["injury"] = snap["player_id"].map(s["status"]), snap["player_id"].map(s["injury"])
        wk_src = "official"
    else:
        inn = snap["ros_out_now"].eq(0) & snap["ros_bye_now"].eq(0) & ~snap["ros_cur_played"].astype(bool)
        if inn.any():
            snap.loc[inn, "wk"] = weekly_proxy(df, snap[inn], season, week, cfg_all)
            snap.loc[inn, "wk_status"] = "Upcoming"
        wk_src = "model-only (no weekly run yet)"
    # ruled inactive / didn't play this week: out for this week's game
    gone = snap["wk_status"].isin(["Inactive", "Did not play"])
    snap.loc[gone, "ros_out_now"] = 1
    snap.loc[gone, "ros_cur_played"] = True
    snap.loc[snap["wk_status"].eq("Final"), "ros_cur_played"] = True
    a = cfg["a_weekly"]
    # players with a projection for this week (made before kickoff, also when the game has since been played)
    has = snap["wk"].notna() & snap["ros_out_now"].eq(0) & (snap["ros_games_cur"].gt(0) | snap["wk_status"].eq("Final"))
    o = cur_open(snap)
    snap["ros_games"] = snap["ros_games_after"] + o
    fut = np.where(has, a * snap["wk"].fillna(0) + (1 - a) * snap["rate"], snap["rate"])
    cur = np.where(has, snap["wk"].fillna(0), snap["rate"]) * o
    snap["this_week"] = np.where(o > 0, cur, np.nan)
    snap["ros_model"] = cur + fut * snap["ros_games_after"]
    snap["season"], snap["week"] = season, week
    snap["ros_ppr"] = blend_with_experts(snap, "ros_model", cfg["w_model"], group=("position",))
    k = cfg["range_k"]
    lo_s = (snap["rate"] - snap["rate_q10"]).clip(lower=0) * snap["ros_games_after"]
    hi_s = (snap["rate_q90"] - snap["rate"]).clip(lower=0) * snap["ros_games_after"]
    snap["ros_low"] = (snap["ros_ppr"] - snap["position"].map({p: v[0] for p, v in k.items()}) * lo_s).clip(lower=0)
    snap["ros_high"] = snap["ros_ppr"] + snap["position"].map({p: v[1] for p, v in k.items()}) * hi_s
    info = {"season": season, "week": week, "last_week": last_week(season), "players": int(len(snap)),
            "out_this_week": int(snap["ros_out_now"].sum()), "on_bye": int(snap["ros_bye_now"].sum()),
            "this_week_source": wk_src, "training_snapshots": int(len(hist_rows)),
            "rounds": {"/".join(p): v["rounds"] for p, v in models.items()},
            "experts_ros_date": str(ecr_snapshot(ctx.ros_ecr, week_cutoff(ctx.games, season, week))["ros_ecr_date"].max())}
    return snap, info


def status_label(r):
    if r["ros_bye_now"] == 1:
        return "Bye this week"
    if r["wk_status"] == "Inactive":
        return "Inactive this week"
    if r["ros_out_now"] == 1:
        return "IR / reserve" if r.get("ros_res") == 1 else "Not projected this week"
    if r["wk_status"] in ("Final", "Did not play"):
        return "Played this week"
    return "Plays this week"


def injury_labels(ctx, season, week, ids):
    """Latest injury-report entry this season (this week's or last week's): status and injury, with the week when it
    is last week's (this week's report isn't out until Thursday)."""
    f = os.path.join(ctx.data_dir, "inj", f"inj_{season}.parquet")
    if not os.path.exists(f):
        return pd.Series("", index=ids.index)
    inj = pd.read_parquet(f)
    inj = inj[(inj["week"] <= week) & (inj["week"] >= week - 1)].sort_values("week")
    inj = inj.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")

    def lab(pid):
        if pid not in inj.index:
            return ""
        r = inj.loc[pid]
        st = r["report_status"] if isinstance(r["report_status"], str) else ""
        ps = r["practice_status"] if isinstance(r.get("practice_status"), str) else ""
        if not st:
            if int(r["week"]) < week or not ps:
                return ""
            st = "DNP (practice)" if ps.startswith("Did Not") else "Limited (practice)" if ps.startswith("Limited") else ""
            if not st:
                return ""
        what = r["report_primary_injury"] if isinstance(r.get("report_primary_injury"), str) else (
            r["practice_primary_injury"] if isinstance(r.get("practice_primary_injury"), str) else "")
        return st + (f" (Wk {int(r['week'])})" if int(r["week"]) < week else "") + (f" · {what}" if what else "")
    return ids.map(lab)


def depth_inputs(s, ctx, season, week, cfg):
    """Inputs for simulating weekly lineups (trade analyzer): chance he plays each remaining game, points when he plays,
    the team's remaining bye week, fumbles lost per game, and chances of big yardage games (for league bonuses)."""
    av = cfg["avail"]
    cat = np.where(s["ros_out_now"].eq(1) & s["ros_res"].eq(1), "res", np.where(s["ros_out_now"].eq(1), "out", "active"))
    a = pd.Series([av[c].get(p, 0.85) for c, p in zip(cat, s["position"])], index=s.index)
    G = (s["ros_games_after"] + s["ros_games_cur"]).clip(lower=1)
    h = s["ros_ppr"] / (a * G)
    # players out now: when they come back they score like their healthy selves (long-run points per game played),
    # so a low ROS total means fewer games, not weaker games; expected ROS points stay the same
    outp = s["ros_out_now"].eq(1)
    healthy = 0.9 * s["fantasy_points_ppr_ewm_long"].fillna(0)
    h = h.where(~outp, np.maximum(h, healthy))
    a = a.where(~outp, (s["ros_ppr"] / (h * G).where(h * G > 0)).fillna(0).clip(0, 1))
    out = pd.DataFrame(index=s.index)
    out["play_rate"] = a.round(3)
    out["pts_when_playing"] = h.round(2)
    # out this week (or ruled inactive): no game this week; later weeks carry the rest of his expected games
    now_out = s["ros_out_now"].eq(1) | s["wk_status"].isin(["Inactive", "Did not play"]) | s["ros_bye_now"].eq(1) | s["ros_cur_played"].astype(bool)
    has_now = s["ros_games_cur"].gt(0) & ~s["ros_cur_played"].astype(bool)
    out["play_now"] = np.where(now_out | ~has_now, 0.0, a).round(3)
    later = np.where(s["ros_out_now"].eq(1) & has_now & (G > 1), (a * G / (G - 1).clip(lower=1)).clip(upper=1), a)
    out["play_later"] = np.round(later, 3)
    lw = last_week(season)
    sc = ctx.sched[(ctx.sched["season"] == season) & (ctx.sched["week"] > week) & (ctx.sched["week"] <= lw)]
    byes = {t: [w for w in range(week + 1, lw + 1) if w not in set(g["week"])] for t, g in sc.groupby("team")}
    out["bye_week"] = s["team"].map(lambda t: (byes.get(t) or [np.nan])[0])
    st = ctx.stats[(ctx.stats["season"] >= season - 1) & (ctx.stats["season_type"] == "REG")]
    fl = st.groupby("player_id").agg(n=("week", "size"), f=("fumbles_lost_total", "sum")) if "fumbles_lost_total" in st else None
    if fl is not None:
        base = (fl["f"].sum() / max(fl["n"].sum(), 1))
        out["fum_pg"] = s["player_id"].map((fl["f"] + 8 * base) / (fl["n"] + 8)).fillna(base).round(3)
    bg = cfg["big_games"]
    def chance(col, b):
        v = s[col].clip(lower=1)
        return (1 / (1 + np.exp(-(b[0] + b[1] * np.log(v))))).where(s[col].notna(), 0).round(4)
    out["p_pass400"] = chance("passing_yards_ewm_long", bg["pass400"]).where(s["position"] == "QB", 0)
    out["p_rush150"] = chance("rushing_yards_ewm_long", bg["rush150"]).where(s["position"].isin(["QB", "RB"]), 0)
    out["p_rec175"] = chance("receiving_yards_ewm_long", bg["rec175"]).where(s["position"] != "QB", 0)
    return out


def market_value(s, vcfg=None):
    """Undervalued / at value / overvalued against the market (the FantasyPros ROS expert consensus): the gap is our
    model's ROS points minus the points our model gives the player ranked at his market rank in his position, per game
    left. Undervalued when the gap is at least +threshold points per game, overvalued at -threshold or below; only for
    players inside the tested market pool (top QB32 / RB60 / WR80 / TE32). Threshold chosen on 2020-2022 walk-forward
    snapshots, tested on 2023-2025 (value_lab.py)."""
    vcfg = vcfg or {}
    T = vcfg.get("threshold_pg", 1.0)
    pool = vcfg.get("pool", {"QB": 32, "RB": 60, "WR": 80, "TE": 32})
    gap = pd.Series(np.nan, index=s.index)
    mr = pd.Series(np.nan, index=s.index)
    for p, g in s[s["ros_ecr_pos_rank"].notna()].groupby("position"):
        r = g["ros_ecr_pos_rank"].rank(method="first").astype(int)
        sm = np.sort(g["ros_model"].values)[::-1]
        gap[g.index] = (g["ros_model"].values - sm[r.values - 1]) / g["ros_games"].clip(lower=1).values
        mr[g.index] = r.values
    ok = mr.notna() & (mr <= s["position"].map(pool))
    gap = gap.where(ok)
    label = pd.Series(np.where(gap >= T, "Undervalued", np.where(gap <= -T, "Overvalued", "At value")), index=s.index)
    return gap.round(2), label.where(ok, "")


def dash_rows(snap, ctx, season, week, cfg=None):
    """ROS rankings and trade values for the dashboard."""
    p = ctx.pts.reset_index()
    p = p[(p["season"] == season) & (p["week"] < week)]
    gp = p.groupby("player_id")["fantasy_points_ppr"].agg(["sum", "count"])
    s = snap.copy()
    s["key"] = s["player_display_name"] + " (" + s["team"] + ")"
    s["ros_rank"] = s.groupby("position")["ros_ppr"].rank(ascending=False, method="first")
    s["model_rank"] = s.groupby("position")["ros_model"].rank(ascending=False, method="first")
    s["season_ppg"] = s["player_id"].map(gp["sum"] / gp["count"])
    s["season_games"] = s["player_id"].map(gp["count"]).fillna(0)
    s["status"] = [status_label(r) for r in s.to_dict("records")]
    lab = injury_labels(ctx, season, week, s["player_id"])
    store_lab = s["injury"].where(s["injury"].notna() & ~s["injury"].isin(["Not listed", "Full (practice)"]), "")
    s["injury"] = np.where(s["injury"].notna(), store_lab, lab)
    s["ros_ppg"] = s["ros_ppr"] / s["ros_games"].where(s["ros_games"] > 0)
    s["value_gap_pg"], s["value"] = market_value(s, (cfg or {}).get("value"))
    out = pd.DataFrame({
        "key": s["key"], "player": s["player_display_name"], "player_id": s["player_id"], "pos": s["position"],
        "team": s["team"], "status": s["status"], "injury": pd.Series(s["injury"]).fillna("").values,
        "ros_rank": s["ros_rank"].astype(int), "ros_ppr": s["ros_ppr"].round(1), "ros_low": s["ros_low"].round(1),
        "ros_high": s["ros_high"].round(1), "ros_model": s["ros_model"].round(1), "model_rank": s["model_rank"].astype(int),
        "expert_rank": s["ros_ecr_pos_rank"].round(0), "value": s["value"], "value_gap_pg": s["value_gap_pg"],
        "games_left": s["ros_games"].astype(int),
        "ros_ppg": s["ros_ppg"].round(2), "this_week": s["this_week"].round(1),
        "season_ppg": s["season_ppg"].round(2), "season_games": s["season_games"].astype(int),
        "opp_allowed": s["ros_opp_def_rel"].round(3), "from_week": week, "through_week": last_week(season),
        # passing rates (per game, long-run weighted average) so a league's own QB scoring can be applied on the page
        "pass_yds_pg": s["passing_yards_ewm_long"].where(s["position"] == "QB").round(1),
        "pass_td_pg": s["passing_tds_ewm_long"].where(s["position"] == "QB").round(3),
        "int_pg": s["passing_interceptions_ewm_long"].where(s["position"] == "QB").round(3),
        "rec_pg": s["receptions_ewm_long"].round(2),
        "avail": (s["ros_ppg"] / s["fantasy_points_ppr_ewm_long"].where(s["fantasy_points_ppr_ewm_long"] > 1)).clip(0, 1).round(3)})
    if cfg is not None and "avail" in cfg:
        out = pd.concat([out, depth_inputs(s, ctx, season, week, cfg)], axis=1)
    out = out.sort_values(["pos", "ros_rank"]).drop_duplicates("key")
    tv = trade_values(s.rename(columns={"position": "position"}))
    t = pd.concat([s[["key", "player_display_name", "position", "team", "status", "ros_ppr", "ros_games"]], tv], axis=1)
    t = t.rename(columns={"player_display_name": "player", "position": "pos", "ros_games": "games_left"})
    t["ros_ppr"] = t["ros_ppr"].round(1)
    t = t.drop_duplicates("key").sort_values("val_12_1qb", ascending=False)
    return out, t

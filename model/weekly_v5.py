"""
v5 weekly run: rebuild features with the latest data, refit on every game before the target week, and
project the whole week, including games already played (shown as Final with actual stats next to the
projection, which was made without that game's result).

  FP_DATA=/home/claude/fpdata python weekly_v5.py --season 2026 --week 5 --config v5_config.json --out out_v5
  ... --backfill 1 4      also re-create weeks 1-4 the same way (each trained only on games before it)

Official PPR projection
  model      equal-weight average of standard LightGBM, extra-trees LightGBM and CatBoost (several seeds
             each), trained on every game from 2018 through the week before. Players
             without this week's FantasyPros ranking (Thursday run, or a game before the weekly snapshot) use
             the same model trained without the weekly-ranking inputs.
  blend      where the week's FantasyPros ranking exists: average of the model's rank and the experts' rank
             within the position; the model's own point totals are handed out in that blended order (points
             stay on the model's scale, so flex comparisons still work)
  range      10th / 90th percentile models
Other stats (passing, rushing, receiving yards, receptions): one LightGBM model each.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import features_v5 as f5
import features_v6 as f6
import lab5 as L
import modelcache as MC
import nfl_player_model as m

STAT_MODELS = [("passing_yards", ["QB"]), ("rushing_yards", ["QB", "RB"]),
               ("receiving_yards", ["RB", "WR", "TE"]), ("receptions", ["RB", "WR", "TE"])]
ACTUAL = ["fantasy_points_ppr", "passing_yards", "passing_tds", "passing_interceptions", "rushing_yards",
          "rushing_tds", "carries", "targets", "receptions", "receiving_yards", "receiving_tds"]


def train_predict(trn, rows, X, target, params, rounds, seeds, objective=None, alpha=None, kind="lgb"):
    """Fit one ensemble member (LightGBM with `params`, or CatBoost) per seed on trn; average predictions."""
    out = np.zeros(len(rows))
    Xtr = trn[X].astype(np.float32)
    Xrows = rows[X].astype(np.float32)
    dkey = MC.data_key(Xtr.values, trn[target].values, cols=X) if MC.ON else None
    if kind == "cat":
        from catboost_lab import CAT_PARAMS, cat_fit
        for sd in seeds:
            cm = MC.cat_model(dict(CAT_PARAMS, random_seed=sd), dkey, int(rounds),
                              lambda: cat_fit(trn, None, X, target, sd, iters=int(rounds))[0].m)
            out += cm.predict(Xrows.values) / len(seeds)
        return out
    p = dict(params)
    if objective:
        p["objective"] = objective
    if alpha is not None:
        p["alpha"] = alpha
    ds = lgb.Dataset(Xtr, trn[target], free_raw_data=False)
    for sd in seeds:
        b = MC.lgb_train(lgb, dict(p, seed=sd), ds, int(rounds), dkey)
        out += b.predict(Xrows) / len(seeds)
    return out


def ensemble_predict(cfg, trn, rows, X, target, rounds_key, scale):
    """Equal-weight average of the configured members (e.g. standard LightGBM, extra-trees LightGBM, CatBoost)."""
    preds = []
    for name, mem in cfg["members"].items():
        params = dict(L.LAB_PARAMS, **mem.get("params", {}))
        preds.append(train_predict(trn, rows, X, target, params, cfg[rounds_key][name] * scale, mem["seeds"],
                                   kind=mem.get("kind", "lgb")))
    return np.mean(preds, axis=0)


def blend_with_experts(up):
    """Rank average with the experts within each position-week (rows with that week's ranking); the model's
    own point projections are handed out in the blended order."""
    up = up.copy()
    up["pred_ppr_official"] = up["pred_ppr_model"]
    up["blend_rank"] = np.nan
    has = up["ecr_week_ecr"].notna()
    for _, g in up[has].groupby([c for c in ["season", "week", "position"] if c in up.columns]):
        mr = g["pred_ppr_model"].rank(ascending=False, method="first")
        er = g["ecr_week_ecr"].rank(method="first")
        order = ((mr + er) / 2 + mr * 1e-6).sort_values(kind="mergesort").index
        up.loc[order, "pred_ppr_official"] = np.sort(g["pred_ppr_model"].values)[::-1]
        up.loc[order, "blend_rank"] = np.arange(1, len(order) + 1)
    return up


LIVE_OTHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "automation", "history", "other_proj", "live")
SITE_MIN_COVER = 0.6   # share of the week's upcoming QB/RB/WR/TE rows (model >= 3 pts) a site must project


def _nk(x):
    import re
    import unicodedata
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    x = re.sub(r"[.'\-]", "", x)
    x = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", x)
    return re.sub(r"\s+", " ", x).strip()


def attach_sites(up, season, week, ids_path=None):
    """Adds `sleeper` / `espn` columns (their live PPR projection for this week; NaN when unmatched) from
    automation/other_proj_live files. Matching: the site's id through DynastyProcess's id table, then name + position
    for anyone left (unique names only). Returns (up, coverage per site)."""
    ids_path = ids_path or os.path.join(os.environ.get("FP_DATA", "fpdata"), "db_playerids.csv")
    cover = {}
    for src, col in [("sleeper", "sleeper_id"), ("espn", "espn_id")]:
        up[src] = np.nan
        f = os.path.join(LIVE_OTHER, f"{src}_{season}_wk{week}.csv")
        if not os.path.exists(f):
            cover[src] = 0.0
            continue
        sl = pd.read_csv(f, dtype={"source_id": str})
        sl["source_id"] = sl["source_id"].str.replace(r"\.0$", "", regex=True)
        try:
            ids = pd.read_csv(ids_path, usecols=[col, "gsis_id"], low_memory=False).dropna()
            ids[col] = ids[col].astype(str).str.replace(r"\.0$", "", regex=True)
            ids = ids.drop_duplicates(col).set_index(col)["gsis_id"]
            sl["player_id"] = sl["source_id"].map(ids)
        except Exception:
            sl["player_id"] = np.nan
        by_id = sl.dropna(subset=["player_id"]).drop_duplicates("player_id").set_index("player_id")["proj_ppr"]
        up[src] = up["player_id"].map(by_id)
        # name + position fallback for players the id table misses (rookies, recent signings)
        sl["nk"] = sl["player"].map(_nk) + "|" + sl["pos"].astype(str)
        uniq = sl[~sl["nk"].duplicated(keep=False)].set_index("nk")["proj_ppr"]
        upk = up["player_display_name"].map(_nk) + "|" + up["position"].astype(str)
        dup = upk.duplicated(keep=False)
        miss = up[src].isna() & ~dup
        up.loc[miss, src] = upk[miss].map(uniq)
        rel = (up["status"] == "Upcoming") & (up["pred_ppr_model"] >= 3)
        cover[src] = float(up.loc[rel, src].notna().mean()) if rel.any() else 0.0
    return up, cover


def blend_sites(up):
    """Official weekly = model + Sleeper + ESPN: within each position-week, the three sources' ranks averaged and their
    average point curve handed out in that order. A player a site doesn't list takes the model's number for it."""
    up = up.copy()
    up["pred_ppr_official"] = up["pred_ppr_model"]
    up["blend_rank"] = np.nan
    members = ["pred_ppr_model", "sleeper", "espn"]
    vals = up[members].copy()
    for c in ["sleeper", "espn"]:
        vals[c] = vals[c].fillna(vals["pred_ppr_model"])
    grp = [c for c in ["season", "week", "position"] if c in up.columns]
    for _, g in up.groupby(grp):
        v = vals.loc[g.index]
        rk = sum(v[m].rank(ascending=False, method="first") for m in members) / len(members)
        rk = rk + v["pred_ppr_model"].rank(ascending=False, method="first") * 1e-6
        pool = np.mean([np.sort(v[m].values)[::-1] for m in members], axis=0)
        order = rk.rank(method="first").astype(int).values - 1
        up.loc[g.index, "pred_ppr_official"] = pool[order]
        up.loc[g.index, "blend_rank"] = order + 1
    return up


def official_blend(rows, cfg, season, week):
    """cfg["weekly_blend"] == "sites": model + Sleeper + ESPN when both sites cover this week; otherwise (or when set to
    "experts") the FantasyPros rank blend. Keeps the FantasyPros blend as pred_ppr_fp_blend for comparison."""
    fp = blend_with_experts(rows)
    fp["pred_ppr_fp_blend"] = fp["pred_ppr_official"]
    fp["blend_source"] = "model+FantasyPros"
    if cfg.get("weekly_blend") != "sites":
        return fp, {"blend": "model+FantasyPros", "reason": "config"}
    try:
        up, cover = attach_sites(fp, season, week)
    except Exception as e:      # noqa: BLE001
        return fp, {"blend": "model+FantasyPros", "reason": f"site files: {e}"[:200]}
    if min(cover.values()) < SITE_MIN_COVER:
        return fp, {"blend": "model+FantasyPros", "reason": "site coverage", "coverage": cover}
    out = blend_sites(up)
    out["blend_source"] = "model+Sleeper+ESPN"
    return out, {"blend": "model+Sleeper+ESPN", "coverage": {k: round(v, 3) for k, v in cover.items()}}


def features_for(cfg, hist, key):
    extra = list(cfg.get("extra", [])) + cfg.get("extra_by_pos", {}).get(key, [])
    return L.feature_list(hist, cfg.get("drop_groups", []), extra, cfg.get("drop_cols", []))


def project(hist, rows, cfg, season, week):
    """Train on every row before (season, week) and project `rows`."""
    trn_all = hist[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))]
    base = dict(L.LAB_PARAMS)
    scale = cfg.get("round_scale", 1.1)
    rows = rows.copy()
    # weekly-ranking model for every game whose FantasyPros weekly snapshot exists (players it leaves unranked are
    # deep players, as in training); the no-weekly-ranking model only for games before the snapshot (Thursday)
    has = rows.groupby("game_id")["ecr_week_ecr"].transform(lambda s: s.notna().any())
    for pos in L.POS_MODELS:
        key = "/".join(pos)
        X = features_for(cfg, hist, key)
        Xn = [c for c in X if not c.startswith("ecr_week_")]
        trn, rp = trn_all[trn_all["position"].isin(pos)], rows[rows["position"].isin(pos)]
        if len(rp) == 0:
            continue
        sub = {k: {m: v[key] for m, v in cfg[k].items()} for k in ("rounds", "rounds_noweek")}
        c2 = dict(cfg, rounds=sub["rounds"], rounds_noweek=sub["rounds_noweek"])
        hr = has.loc[rp.index].values
        pred = np.full(len(rp), np.nan)
        if (~hr).any():
            pred[~hr] = ensemble_predict(c2, trn, rp[~hr], Xn, L.TARGET, "rounds_noweek", scale)
        if hr.any():
            pred[hr] = ensemble_predict(c2, trn, rp[hr], X, L.TARGET, "rounds", scale)
        rows.loc[rp.index, "pred_ppr_model"] = pred
        for q in (0.1, 0.9):
            rows.loc[rp.index, f"pred_ppr_q{int(q * 100)}"] = train_predict(
                trn, rp, Xn, L.TARGET, base, cfg["q_rounds"][key][str(q)] * scale, [7], objective="quantile", alpha=q)
    Xs = L.feature_list(hist, [], [], [])
    Xsn = [c for c in Xs if not c.startswith("ecr_week_")]
    for target, pos in STAT_MODELS:
        trn, rp = trn_all[trn_all["position"].isin(pos)], rows[rows["position"].isin(pos)]
        if len(rp):
            rows.loc[rp.index, f"pred_{target}"] = train_predict(trn, rp, Xsn, target, base,
                                                                 cfg["stat_rounds"][target] * scale, [7, 23])
    if cfg.get("qb_market"):   # v5.1: QBs anchored to the market projection, cold-start starters corrected
        qm = cfg["qb_market"]
        rows["pred_ppr_model_raw"] = rows["pred_ppr_model"]
        rows["pred_ppr_model"] = f6.qb_market_adjust(rows, qm["w_model"], qm["cold_offsets"])
    rows, binfo = official_blend(rows, cfg, season, week)
    print("weekly blend:", json.dumps(binfo), flush=True)
    try:
        os.makedirs("out_v5", exist_ok=True)
        json.dump(binfo, open(os.path.join("out_v5", f"blend_info_{season}_wk{week}.json"), "w"), indent=1)
    except Exception:
        pass
    rows["pred_ppr_q10"] = np.minimum(rows["pred_ppr_q10"].clip(lower=0), rows["pred_ppr_official"])
    rows["pred_ppr_q90"] = np.maximum(rows["pred_ppr_q90"], rows["pred_ppr_official"])
    # range calibration (walk-forward 2023-2026: the raw 10th/90th percentile models left ~12% of games outside each
    # end): widen the high end by k around the official projection, lower the low end by c points
    for pos, (k, c) in cfg.get("range_adjust", {}).items():
        p = rows["position"] == pos
        o = rows.loc[p, "pred_ppr_official"]
        rows.loc[p, "pred_ppr_q90"] = o + k * (rows.loc[p, "pred_ppr_q90"] - o)
        rows.loc[p, "pred_ppr_q10"] = np.minimum((rows.loc[p, "pred_ppr_q10"] - c).clip(lower=0), o)
    return rows


KEEP = ["player_id", "player_display_name", "position", "team", "opponent_team", "game_id", "season", "week",
        "is_home", "status", "implied_pts", "pace_exp_plays", "fantasy_points_ppr_ewm", "games_hist",
        "practice_dnp", "practice_limited", "ij_questionable", "ij_doubtful", "rc_tprr", "rc_yprr", "rc_adot",
        "rc_route_participation", "rc_first_read_share", "xfp_ewm", "xfp_fpoe_ewm", "xfp_td_luck_ewm",
        "dc_depth_rank", "dc_depth_change", "qb_changed", "ecr_week_ecr", "ecr_week_pos_rank", "ecr_ros_pos_rank",
        "wx_temp", "wx_wind", "wx_rain", "wx_dome", "coach", "blend_rank", "blend_source", "sleeper", "espn", "carries_ewm", "ru_ypc", "ru_explosive_rate",
        "ru_gump_rate", "ru_stuff_rate", "ru_epa_per_carry", "rz_goal_line_carry_share", "def_two_high_rate",
        "def_explosive_run_rate", "mkt_pts", "mkt_share_ewm", "mkt_qb_start_share", "qb_prior_starts", "qb_is_projected_starter"]


def week_rows(df, hist, season, week):
    """Every expected player for the week: games already played (from history) and upcoming games."""
    played = df[~df["upcoming"] & (df["season"] == season) & (df["week"] == week)].copy()
    played["status"] = np.where(played["phantom"], "Final (did not play)", "Final")
    up = df[df["upcoming"] & (df["season"] == season) & (df["week"] == week)].copy()
    up["status"] = "Upcoming"
    up = m.apply_forecast(up, season, week)
    return pd.concat([played, up])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--config", default="v5_config.json")
    ap.add_argument("--out", default="out_v5")
    ap.add_argument("--frame", default=None, help="reuse / save the feature frame (parquet)")
    ap.add_argument("--backfill", type=int, nargs=2, default=None, help="also re-create weeks A..B")
    ap.add_argument("--upcoming-only", action="store_true", help="project only games not yet played")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(a.config))
    t0 = time.time()
    if a.frame and os.path.exists(a.frame):
        df = pd.read_parquet(a.frame)
    else:
        df = m.build_frame(a.data, a.season, a.week)
        if a.frame:
            df.to_parquet(a.frame)
    stats, games, players = m.load(a.data)
    v5, _ = f5.build_v5(df, a.data, stats, players)
    df = df.join(v5[[c for c in v5.columns if c not in df.columns]])
    if cfg.get("qb_market"):
        df = df.join(f6.build_market(df, a.data))
        df["qb_prior_starts"] = f6.qb_prior_starts(df, a.data)
    for c in ["phantom", "snap_only", "upcoming"]:
        df[c] = df[c].eq(True)
    hist = L.hist_rows(df)
    t_feat = time.time() - t0
    weeks = [a.week] + (list(range(a.backfill[0], a.backfill[1] + 1)) if a.backfill else [])
    for w in weeks:
        rows = week_rows(df, hist, a.season, w)
        if a.upcoming_only and w == a.week:
            rows = rows[rows["status"] == "Upcoming"]
        if not len(rows):
            print(f"week {w}: nothing to project", flush=True)
            continue
        t1 = time.time()
        res = project(hist, rows, cfg, a.season, w)
        t_train = time.time() - t1
        act = res[ACTUAL].where(res["status"] != "Upcoming")
        act.columns = ["actual_" + c for c in ACTUAL]
        res = pd.concat([res[[c for c in KEEP if c in res.columns] + [c for c in res.columns if c.startswith("pred_")]],
                         act], axis=1).sort_values("pred_ppr_official", ascending=False)
        res.to_csv(os.path.join(a.out, f"predictions_{a.season}_wk{w}.csv"), index=False)
        info = {"season": a.season, "week": w, "players": int(len(res)),
                "final": int((res["status"] != "Upcoming").sum()), "upcoming": int((res["status"] == "Upcoming").sum()),
                "weekly_consensus_for": int(res["ecr_week_ecr"].notna().sum()), "config": a.config,
                "kind": "live" if w == a.week else "backfill (trained only on games before the week)",
                "seconds": {"features": round(t_feat), "models": round(t_train)},
                "model_cache": dict(MC.stats, on=MC.ON)}
        json.dump(info, open(os.path.join(a.out, f"run_info_{a.season}_wk{w}.json"), "w"), indent=2)
        print(info, flush=True)
        print(res.head(12)[["player_display_name", "position", "team", "status", "pred_ppr_model",
                            "pred_ppr_official", "actual_fantasy_points_ppr"]].to_string(), flush=True)


if __name__ == "__main__":
    main()

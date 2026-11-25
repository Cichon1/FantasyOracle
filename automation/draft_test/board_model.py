"""Phase 2 (PREREG.md): our preseason draft board, walk-forward.

For season Y the board is fit only on seasons < Y (asserted) and uses only information that existed before Y's draft:
  market     every ADP available that year (FFC 12-team PPR + its half / standard / 10 / 14-team boards; Yahoo and
             MFL join in the GitHub run where the encrypted files can be read), their spread and disagreement
  history    prior seasons' weekly stats (games, points per game in the league's scoring, targets / carries / pass
             attempts per game, target share, WOPR), two seasons back, career
  player     age on Sept 1, years pro, NFL draft round / pick (rookies: the April draft), team change since last season
  contract   the deal he was on going into the season (board_data.contract_features: pre-draft deals only)
Target: actual season points (Weeks 1-17) in that league's scoring; players who never played count 0.

Board value = predicted points minus the position's replacement level, both from seasons < Y. Replacement ranks for 12
teams: QB 12, TE 13, RB 29, WR 30 (BDSM) / 41 (The League, 3 WR).

Scoring (PREREG section 6): on the same players for every board - each position's top-N by FFC ADP (QB 24, RB 48,
WR 60, TE 24) - the Spearman correlation of each board's order with actual points, by position and season; and the
overall top-150 order by board value vs actual value above replacement.

Tuning (features, model settings) uses 2013-2019 only and is then frozen in CONFIG; `test` runs 2020-2025 once.

  python automation/draft_test/board_model.py tune      # 2013-2019 only
  python automation/draft_test/board_model.py test      # frozen CONFIG, 2020-2025
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import board_data as B  # noqa: E402
import sources as S  # noqa: E402
from audit_draft_data import norm  # noqa: E402

OUT = B.OUT
SEASONS = B.SEASONS
TUNE = [y for y in SEASONS if 2013 <= y <= 2019]
TEST = [2020, 2021, 2022, 2023, 2024, 2025]
TOPN = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
REPL = {"bdsm": {"QB": 12, "RB": 29, "WR": 30, "TE": 13}, "league": {"QB": 12, "RB": 29, "WR": 41, "TE": 13}}
TEAMFIX = {"JAC": "JAX", "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS", "GBP": "GB", "KCC": "KC",
           "NEP": "NE", "NOS": "NO", "SFO": "SF", "TBB": "TB", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}

# frozen after tuning on 2013-2019 (see tune()); changed only through PREREG's Changes log
# Nov 24 tuning (2013-2019, tune_log_round1-3.json): history + player features help, bagging helps, the residual
# target doesn't add to bagging, contracts and the other ADP boards don't help, ridge is worse.
CONFIG = {"model": "gbm", "max_iter": 300, "learning_rate": 0.04, "max_depth": 3, "min_samples_leaf": 20,
          "l2": 1.0, "features": "adp+hist+player", "bag": 8}


# ---------------------------------------------------------------- features
def season_stats(wk_all):
    w = wk_all
    g = w.groupby(["season", "gsis_id"])
    s = g.agg(games=("week", "nunique"), pts_bdsm=("pts_bdsm", "sum"), pts_league=("pts_league", "sum"),
              tgt=("targets", "sum"), car=("carries", "sum"), att=("attempts", "sum"), recyd=("receiving_yards", "sum"),
              rushyd=("rushing_yards", "sum"), passyd=("passing_yards", "sum"), tds=("tds", "sum"),
              tshare=("target_share", "mean"), wopr=("wopr", "mean"),
              team=("team", lambda x: x.value_counts().index[0])).reset_index()
    for c in ["pts_bdsm", "pts_league", "tgt", "car", "att", "recyd", "rushyd", "passyd", "tds"]:
        s[c + "_g"] = s[c] / s["games"].clip(lower=1)
    return s


def weekly_full():
    rows = []
    for y in SEASONS + [2011]:
        p = os.path.join(B.DATA, f"week_{y}.parquet")
        w = pd.read_parquet(p if os.path.exists(p) else f"{B.NFLV}/stats_player/stats_player_week_{y}.parquet")
        w = w[(w["season_type"] == "REG") & (w["week"] <= 17)]
        rows.append(w[["season", "week", "player_id", "team", "targets", "carries", "attempts", "receiving_yards",
                       "rushing_yards", "passing_yards", "passing_tds", "rushing_tds", "receiving_tds", "target_share", "wopr"]])
    w = pd.concat(rows, ignore_index=True).rename(columns={"player_id": "gsis_id"})
    w["tds"] = w[["passing_tds", "rushing_tds", "receiving_tds"]].fillna(0).sum(axis=1)
    pts = pd.concat([B.weekly(y)[["season", "week", "gsis_id", "pts_bdsm", "pts_league"]] for y in SEASONS + [2011]])
    return w.merge(pts, on=["season", "week", "gsis_id"], how="left")


def extra_adps(y):
    """other FFC boards (half / standard PPR, 10 / 14 teams) + Yahoo / MFL where readable (encrypted: GitHub run)."""
    out = {}
    for fmt, teams in [("half-ppr", 12), ("standard", 12), ("ppr", 10), ("ppr", 14)]:
        try:
            d, _ = S.ffc(fmt, teams, y)
            out[f"adp_ffc_{fmt}_{teams}"] = d.assign(key=d["name"].map(norm)).drop_duplicates("key").set_index("key")["adp"]
        except (Exception, SystemExit):
            pass
    for name, fn in [("yahoo", lambda: S.yahoo(y)), ("mfl", lambda: S.mfl(y, "ppr"))]:
        try:
            d = fn()
            d = d[0] if isinstance(d, tuple) else d
            nm = "name" if "name" in d else "player"
            out[f"adp_{name}"] = d.assign(key=d[nm].map(norm)).drop_duplicates("key").set_index("key")["adp"]
        except (Exception, SystemExit):
            pass
    return out


def fp_ecr(y):
    """FantasyPros preseason expert consensus rank (redraft PPR overall): the last snapshot between June 1 and 3 days
    before kickoff (DynastyProcess's public archive, model/fpdata)."""
    p = os.path.join(ROOT, "model", "fpdata", "db_fpecr.parquet")
    if not os.path.exists(p):
        return None
    e = pd.read_parquet(p, columns=["page_type", "ecr_type", "player", "pos", "ecr", "scrape_date"])
    e = e[(e["page_type"] == "redraft-overall") & (e["ecr_type"] == "ro")].copy()   # "ro" = redraft overall (the only type archived)
    e["d"] = pd.to_datetime(e["scrape_date"])
    cut = S.kickoffs()[y] - pd.Timedelta(days=3)
    e = e[(e["d"] <= cut) & (e["d"] >= pd.Timestamp(f"{y}-06-01"))]
    if e.empty:
        return None
    e = e[e["d"] == e["d"].max()]
    return e.assign(key=e["player"].map(norm)).drop_duplicates("key").set_index("key")["ecr"]


def build(y, ss, pl_birth):
    pool = pd.read_parquet(os.path.join(OUT, f"pool_{y}.parquet"))
    pool = pool[pool["matched"]].drop_duplicates("gsis_id").copy()
    pool["key"] = pool["player"].map(norm)
    for c, ser in extra_adps(y).items():
        pool[c] = pool["key"].map(ser)
    ecr = fp_ecr(y)
    pool["fp_ecr"] = pool["key"].map(ecr) if ecr is not None else np.nan
    # FFC boards only, so the frozen board is identical wherever it runs (Yahoo / MFL are comparison boards)
    adps = [c for c in pool if c.startswith("adp_ffc") and c not in ("adp_ffc_sd", "adp_ffc_high", "adp_ffc_low")]
    pool["adp_log"] = np.log(pool["adp_ffc"])
    pool["adp_pos_rank"] = pool.groupby("pos")["adp_ffc"].rank()
    pool["adp_min"] = pool[adps].min(axis=1)
    pool["adp_spread_src"] = pool[adps].max(axis=1) - pool[adps].min(axis=1)
    pool["adp_n_src"] = pool[adps].notna().sum(axis=1)
    for c in adps + [c for c in pool if c in ("adp_yahoo", "adp_mfl")]:
        if c != "adp_ffc":
            pool[c + "_diff"] = np.log(pool[c]) - pool["adp_log"]
    # history (seasons < y only)
    prev = ss[ss["season"] < y]
    p1 = prev[prev["season"] == y - 1].set_index("gsis_id")
    p2 = prev[prev["season"] == y - 2].set_index("gsis_id")
    if y - 1 == 2011 and p1.empty:
        p1 = prev[prev["season"] == y - 1].set_index("gsis_id")
    for c in ["games", "pts_bdsm_g", "pts_league_g", "tgt_g", "car_g", "att_g", "recyd_g", "rushyd_g", "passyd_g",
              "tds_g", "tshare", "wopr"]:
        pool["p1_" + c] = pool["gsis_id"].map(p1[c])
    for c in ["games", "pts_bdsm_g", "pts_league_g"]:
        pool["p2_" + c] = pool["gsis_id"].map(p2[c])
    car = prev.groupby("gsis_id").agg(n_seasons=("season", "nunique"), car_games=("games", "sum"),
                                      car_pts=("pts_bdsm", "sum"), best_ppg=("pts_bdsm_g", "max"))
    pool["career_seasons"] = pool["gsis_id"].map(car["n_seasons"]).fillna(0)
    pool["career_ppg"] = pool["gsis_id"].map(car["car_pts"] / car["car_games"].clip(lower=1))
    pool["career_best_ppg"] = pool["gsis_id"].map(car["best_ppg"])
    t_now = pool["team"].astype(str).str.upper().replace(TEAMFIX)
    t_prev = pool["gsis_id"].map(p1["team"]).astype(str).str.upper().replace(TEAMFIX)
    pool["team_changed"] = np.where(pool["gsis_id"].map(p1["team"]).isna(), np.nan, (t_now != t_prev).astype(float))
    bd = pd.to_datetime(pool["gsis_id"].map(pl_birth), errors="coerce")
    pool["age"] = (pd.Timestamp(f"{y}-09-01") - bd).dt.days / 365.25
    pool["rookie"] = (pool["years_pro"] == 0).astype(float)
    return pool


FEATS_BASE = ["adp_ffc", "adp_log", "adp_pos_rank", "adp_ffc_sd", "adp_min", "adp_spread_src", "adp_n_src"]


def feature_list(df, which):
    hist = [c for c in df if c.startswith(("p1_", "p2_", "career_"))] + ["team_changed"]
    player = ["age", "years_pro", "rookie", "draft_round", "draft_overall"]
    contract = ["c_apy", "c_gtd_share", "c_years_left", "c_contract_year", "c_new_deal", "c_rookie_deal", "c_apy_pos_rank"]
    other_adp = [c for c in df if c.startswith("adp_") and c.endswith("_diff")]
    sets = {"adp": FEATS_BASE, "adp+hist": FEATS_BASE + hist, "adp+hist+player": FEATS_BASE + hist + player,
            "adp+hist+player+contract": FEATS_BASE + hist + player + contract,
            "adp+hist+player+otheradp": FEATS_BASE + hist + player + other_adp,
            "all": FEATS_BASE + other_adp + hist + player + contract}
    return [c for c in sets[which] if c in df] + ["pos_code"]


# ---------------------------------------------------------------- model
def fit_predict(train, test, feats, target, cfg):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    assert train["season"].max() < test["season"].min(), "walk-forward violated"
    feats = [c for c in feats if c == "pos_code" or train[c].nunique() >= 2]     # e.g. a source not yet existing
    if cfg["model"] == "gbm":
        y_tr, base_te = train[target].values, np.zeros(len(test))
        if cfg.get("target") == "resid":           # learn what the market misses: actual - ADP-implied points
            from sklearn.isotonic import IsotonicRegression
            base_tr = np.zeros(len(train))
            for p in train["pos"].unique():
                iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
                    train.loc[train["pos"] == p, "adp_ffc"], train.loc[train["pos"] == p, target])
                base_tr[(train["pos"] == p).values] = iso.predict(train.loc[train["pos"] == p, "adp_ffc"])
                if (test["pos"] == p).any():
                    base_te[(test["pos"] == p).values] = iso.predict(test.loc[test["pos"] == p, "adp_ffc"])
            y_tr = y_tr - base_tr
        preds = []
        rng = np.random.RandomState(0)
        for b in range(cfg.get("bag", 1)):
            idx = np.arange(len(train)) if b == 0 and cfg.get("bag", 1) == 1 else rng.randint(0, len(train), len(train))
            m = HistGradientBoostingRegressor(max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
                                              max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
                                              l2_regularization=cfg["l2"], categorical_features=[feats.index("pos_code")],
                                              random_state=b)
            m.fit(train[feats].iloc[idx], y_tr[idx])
            preds.append(m.predict(test[feats]))
        return base_te + np.mean(preds, axis=0)
    # ridge per position on median-imputed features
    pred = np.zeros(len(test))
    for p in test["pos"].unique():
        tr, te = train[train["pos"] == p], test["pos"] == p
        X = tr[feats].drop(columns="pos_code")
        med = X.median()
        mu, sd = X.fillna(med).mean(), X.fillna(med).std().replace(0, 1)
        r = Ridge(alpha=cfg.get("alpha", 10)).fit(((X.fillna(med) - mu) / sd).fillna(0), tr[target])
        Xt = test.loc[te, feats].drop(columns="pos_code")
        pred[te.values] = r.predict(((Xt.fillna(med) - mu) / sd).fillna(0))
    return pred


def replacement(prev_pool, lg):
    """position replacement level in points: the Nth-best actual season at the position, mean over prior seasons."""
    r = {}
    for p, n in REPL[lg].items():
        v = [g.nlargest(n, f"pts_{lg}")[f"pts_{lg}"].iloc[-1] for _, g in prev_pool[prev_pool["pos"] == p].groupby("season")
             if (g["pos"] == p).sum() >= n]
        r[p] = float(np.mean(v)) if v else 0.0
    return r


def score(df, col, lg, ascending, repl):
    """rank correlations of a board column with actual points, on the fixed player sets."""
    out = {}
    for p, n in TOPN.items():
        g = df[df["pos"] == p].nsmallest(n, "adp_ffc")
        out[p] = float(g[col].rank(ascending=ascending).corr(g[f"pts_{lg}"].rank(ascending=False)))
    top = df.nsmallest(150, "adp_ffc").copy()
    act = top[f"pts_{lg}"] - top["pos"].map(repl)
    if col == "adp_ffc":
        val = -top["adp_ffc"]
    else:
        val = top[col] - top["pos"].map(repl) if col.startswith("pred") else -top[col]
    out["overall"] = float(val.rank().corr(act.rank()))
    out["pos_mean"] = float(np.mean([out[p] for p in TOPN]))
    return out


def run(seasons, cfg, lg="bdsm", verbose=True, data=None):
    allp = data
    res = {}
    target = f"pts_{lg}"
    for y in seasons:
        train = allp[allp["season"] < y]
        test = allp[allp["season"] == y].copy()
        feats = feature_list(allp, cfg["features"])
        test["pred"] = fit_predict(train, test, feats, target, cfg)
        repl = replacement(train, lg)
        r = {"ours": score(test, "pred", lg, False, repl), "ffc": score(test, "adp_ffc", lg, True, repl)}
        for c in ["adp_yahoo", "adp_mfl", "fp_ecr"]:
            if c in test and test[c].notna().sum() > 100:
                t2 = test.assign(**{c: test[c].fillna(test[c].max() + 1)})     # unranked = after everyone ranked
                r[c.replace("adp_", "")] = score(t2, c, lg, True, repl)
        res[y] = r
        if verbose:
            print(y, lg, {k: (round(v["pos_mean"], 3), round(v["overall"], 3)) for k, v in r.items()})
    return res


def load_all():
    path = os.path.join(OUT, "board_features.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    wk = weekly_full()
    ss = season_stats(wk)
    pp = os.path.join(B.DATA, "players.parquet")
    pl = pd.read_parquet(pp if os.path.exists(pp) else f"{B.NFLV}/players/players.parquet")
    pl = pl.drop_duplicates("gsis_id").set_index("gsis_id")["birth_date"]
    parts = [build(y, ss, pl) for y in SEASONS]
    d = pd.concat(parts, ignore_index=True)
    d["pos_code"] = d["pos"].map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})
    d.to_parquet(path, index=False)          # autosave (rebuilt only when deleted)
    return d


def summarize(res, label):
    keys = sorted({k for r in res.values() for k in r})
    s = {}
    for k in keys:
        pm = [r[k]["pos_mean"] for r in res.values() if k in r]
        ov = [r[k]["overall"] for r in res.values() if k in r]
        dpm = [r["ours"]["pos_mean"] - r[k]["pos_mean"] for r in res.values() if k in r]
        s[k] = {"pos_mean": round(np.mean(pm), 3), "overall": round(np.mean(ov), 3), "seasons": len(pm),
                "ours_minus_this_pos": round(np.mean(dpm), 3),
                "se": round(np.std(dpm, ddof=1) / np.sqrt(len(dpm)), 3) if len(dpm) > 1 else None,
                "ours_better_in": int(sum(x > 0 for x in dpm))}
    print(label, json.dumps(s))
    return s


def tune():
    d = load_all()
    assert d["season"].max() >= max(TUNE)
    # round 1 (Nov 24): feature sets, depth / rate / leaf, ridge -> adp+hist+player and all best, ridge worse
    # round 2: residual target (actual - ADP-implied), bagging
    # round 3: ablation of contracts and the other FFC ADP boards on top of the round-2 winner (bag 8)
    grid = [dict(CONFIG, features=f, bag=8) for f in ["adp+hist+player+contract", "adp+hist+player+otheradp"]]
    grid += [dict(CONFIG, features="adp+hist+player", bag=16)]
    for f in []:
        grid += [dict(CONFIG, features=f, target="resid"), dict(CONFIG, features=f, bag=8),
                 dict(CONFIG, features=f, target="resid", bag=8), dict(CONFIG, features=f, target="resid", min_samples_leaf=40, bag=8)]
    log = []
    for cfg in grid:
        for lg in ["bdsm", "league"]:
            res = run(TUNE, cfg, lg, verbose=False, data=d)
            s = summarize(res, f"{lg} {cfg}")
            log.append({"cfg": cfg, "league": lg, "summary": s, "by_season": res})
            with open(os.path.join(OUT, "tune_log.json"), "w") as fh:            # autosave per config
                json.dump(log, fh, indent=1)


def test():
    d = load_all()
    out = {}
    for lg in ["bdsm", "league"]:
        res = run(TEST, CONFIG, lg, data=d)
        out[lg] = {"summary": summarize(res, lg), "by_season": res, "config": CONFIG}
        json.dump(out, open(os.path.join(OUT, "board_test.json"), "w"), indent=1)


if __name__ == "__main__":
    {"tune": tune, "test": test}[sys.argv[1]]()

"""
v5 test harness: honest walk-forward for PPR points, scored against the FantasyPros expert consensus.

For each test season S (2023-2026) and position model (QB, RB, WR/TE):
  train on seasons <= S-2, early-stop on S-1 (validation predictions are kept for fitting blends),
  then refit on every season <= S-1 with 1.1x the rounds and predict S.
The production model is refit every week on everything through last week, so this is, if anything,
slightly harder than live use.

Scoring pools
  expected   every player expected to play (played games + zero-point games for players who sat)
  played     games actually played (incl. on-field games with no stats)
  experts    per position-week, the consensus top QB24 / RB48 / WR60 / TE24 among expected players who were
             ranked in a snapshot taken before kickoff (Thursday games have none); the experts' rank is
             turned into points with a rank->points curve fit on the previous season.
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import nfl_player_model as m

FOLDS = [2023, 2024, 2025, 2026]
POS_MODELS = [("QB",), ("RB",), ("WR", "TE")]
POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
TOPK = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
TARGET = "fantasy_points_ppr"
LAB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=2, max_bin=127)
LAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab")
os.makedirs(LAB_DIR, exist_ok=True)


def hist_rows(df):
    for c in ["phantom", "snap_only"]:
        if c not in df.columns:
            df[c] = False
    if "expected" not in df.columns:
        df["expected"] = True
    return df[~df["upcoming"] & (df["games_hist"] >= 1) & (df["season"] >= 2018)].copy()


def feature_list(df, drop_groups=(), extra=(), drop_cols=()):
    X = m.BASE + [c for g, cols in m.GROUPS.items() if g not in drop_groups for c in cols] + list(extra)
    X = [c for c in dict.fromkeys(X) if c in df.columns and c not in drop_cols]
    return X


def _fit(tr, va, X, params, weight=None, rounds=None, wva=None):
    dtr = lgb.Dataset(tr[X].astype(np.float32), tr[TARGET], weight=weight, free_raw_data=True)
    if rounds is None:
        dva = lgb.Dataset(va[X].astype(np.float32), va[TARGET], weight=wva, reference=dtr)
        b = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        return b, b.best_iteration
    return lgb.train(params, dtr, rounds), rounds


def run_walkforward(hist, X_all, name, params=None, seeds=(7,), weight_fn=None, pos_models=POS_MODELS,
                    learner="lgb", save=True):
    """Returns (test predictions, validation predictions) for every fold; saves both under lab/."""
    params = dict(LAB_PARAMS, **(params or {}))
    tests, vals, info = [], [], {}
    t0 = time.time()
    for S in FOLDS:
        tr, va, te = hist[hist["season"] <= S - 2], hist[hist["season"] == S - 1], hist[hist["season"] == S]
        for pos in pos_models:
            Xp = X_all[ "/".join(pos)] if isinstance(X_all, dict) else X_all
            X = Xp
            trp, vap, tep = (d[d["position"].isin(pos)] for d in (tr, va, te))
            trva = pd.concat([trp, vap])
            pv, pt = np.zeros(len(vap)), np.zeros(len(tep))
            for sd in seeds:
                p = dict(params, seed=sd)
                if learner == "cat":
                    from catboost_lab import cat_fit
                    bv, pvs, r = cat_fit(trp, vap, X, TARGET, sd, valid_pred=True)
                    pv += pvs / len(seeds)
                    bt, _, _ = cat_fit(trva, None, X, TARGET, sd, iters=int(r * 1.1))
                    pt += bt.predict(tep[X].astype(np.float32)) / len(seeds)
                    continue
                w = weight_fn(trp, S - 1) if weight_fn else None
                wv = weight_fn(vap, S - 1) if weight_fn else None
                b, r = _fit(trp, vap, X, p, weight=w, wva=wv)
                pv += b.predict(vap[X].astype(np.float32), num_iteration=r) / len(seeds)
                w2 = weight_fn(trva, S) if weight_fn else None
                b2, _ = _fit(trva, None, X, p, weight=w2, rounds=max(50, int(r * 1.1)))
                pt += b2.predict(tep[X].astype(np.float32)) / len(seeds)
                info[f"{S}|{'/'.join(pos)}|{sd}"] = r
            vals.append(pd.DataFrame({"idx": vap.index, "fold": S, "pred": pv}))
            tests.append(pd.DataFrame({"idx": tep.index, "fold": S, "pred": pt}))
    test = pd.concat(tests).set_index("idx")
    val = pd.concat(vals).set_index("idx")
    if save:
        test.to_parquet(os.path.join(LAB_DIR, f"{name}_test.parquet"))
        val.to_parquet(os.path.join(LAB_DIR, f"{name}_val.parquet"))
        with open(os.path.join(LAB_DIR, f"{name}_info.json"), "w") as f:
            json.dump({"rounds": info, "features": X_all, "minutes": (time.time() - t0) / 60}, f)
    return test, val


# ------------------------------------------------------------------------------- scoring
def r2(y, p):
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def regression_scores(hist, test, pool="expected"):
    d = hist.loc[test.index, ["season", "position", "phantom", "expected", TARGET]].join(test["pred"])
    if pool == "played":
        d = d[~d["phantom"]]
    elif pool == "expected":   # live-like: only players the expected-to-play rule would have projected
        d = d[d["expected"]]
    out = {"all": r2(d[TARGET].values, d["pred"].values)}
    for S, g in d.groupby("season"):
        out[int(S)] = r2(g[TARGET].values, g["pred"].values)
    for p, g in d.groupby("position"):
        out[p] = r2(g[TARGET].values, g["pred"].values)
    out["mae"] = float(np.mean(np.abs(d["pred"] - d[TARGET])))
    out["n"] = int(len(d))
    return out


def expert_pool(hist):
    """Expected-to-play rows with a weekly consensus rank (snapshot before kickoff), regular season,
    consensus top-N per position-week."""
    d = hist[hist["ecr_week_ecr"].notna() & (hist["season_type"] == "REG") & hist["expected"]][
        ["season", "week", "position", "player_id", "ecr_week_ecr", TARGET, "phantom"]].copy()
    d["ecr_rank"] = d.groupby(["season", "week", "position"])["ecr_week_ecr"].rank(method="first")
    d = d[d["ecr_rank"] <= d["position"].map(POOL)]
    return d


def ecr_curves(pool):
    curves = {}
    for S, g in pool.groupby("season"):
        c = g.groupby(["position", "ecr_rank"])[TARGET].mean().rename("pts").reset_index()
        c["pts"] = c.groupby("position")["pts"].transform(
            lambda s: s.rolling(7, center=True, min_periods=1).mean().cummin())
        curves[int(S)] = c
    return curves


def add_ecr_points(d, curves):
    d = d.copy()
    d["ecr_pts"] = np.nan
    for S in d["season"].unique():
        if int(S) - 1 in curves:
            c = curves[int(S) - 1]
            idx = d["season"] == S
            d.loc[idx, "ecr_pts"] = d.loc[idx, ["position", "ecr_rank"]].merge(
                c, on=["position", "ecr_rank"], how="left")["pts"].values
    return d


def pairwise(score, actual):
    s, a = np.asarray(score), np.asarray(actual)
    i, j = np.triu_indices(len(s), 1)
    ok = a[i] != a[j]
    return float(np.mean(np.sign(s[i] - s[j])[ok] == np.sign(a[i] - a[j])[ok]))


def topk(score, actual, k):
    return len(set(np.argsort(-np.asarray(score))[:k]) & set(np.argsort(-np.asarray(actual))[:k])) / k


def expert_scores(hist, test, val=None, extra_sources=None, blend_weights=None):
    """Head-to-head on the experts' pool. extra_sources: {name: Series of scores aligned to hist index}.
    blend_weights: {(season, pos): w} for the points blend w*model + (1-w)*expert points."""
    pool = expert_pool(hist)
    curves = ecr_curves(pool)
    d = add_ecr_points(pool[pool["season"].isin(FOLDS)], curves)
    d = d.join(test["pred"], how="inner")
    d["rank_blend"] = -(d.groupby(["season", "week", "position"])["pred"].rank(ascending=False) + d["ecr_rank"])
    if blend_weights is not None:
        w = pd.Series([blend_weights.get((int(s), p), 0.5) for s, p in zip(d["season"], d["position"])], index=d.index)
        d["pts_blend"] = w * d["pred"] + (1 - w) * d["ecr_pts"].fillna(d["pred"])
    srcs = {"model": "pred", "experts": None, "rank_blend": "rank_blend"}
    if "pts_blend" in d:
        srcs["pts_blend"] = "pts_blend"
    for k, s in (extra_sources or {}).items():
        d[k] = s.reindex(d.index)
        srcs[k] = k
    rows = []
    for (S, w, p), g in d.groupby(["season", "week", "position"]):
        if len(g) < 10:
            continue
        y = g[TARGET].values
        for name, col in srcs.items():
            sc = -g["ecr_rank"].values if col is None else g[col].values
            r = {"season": S, "week": w, "pos": p, "source": name, "n": len(g),
                 "spearman": spearmanr(sc, y).correlation, "pairwise": pairwise(sc, y), "topk": topk(sc, y, TOPK[p])}
            pts = {"model": "pred", "experts": "ecr_pts"}.get(name)
            if pts and g[pts].notna().all():
                r["mae"] = float(np.mean(np.abs(g[pts].values - y)))
            rows.append(r)
    res = pd.DataFrame(rows)
    piv = res.pivot_table(index=["season", "week", "pos"], columns="source", values="spearman")
    summ = {"position_weeks": int(len(piv))}
    for name in srcs:
        summ[name] = {k: float(res.loc[res["source"] == name, k].mean()) for k in ["spearman", "pairwise", "topk"]}
        if "mae" in res and res.loc[res["source"] == name, "mae"].notna().any():
            summ[name]["mae"] = float(res.loc[res["source"] == name, "mae"].mean())
        if name != "experts":
            diff = piv[name] - piv["experts"]
            summ[name]["vs_experts"] = float(diff.mean())
            summ[name]["vs_experts_se"] = float(diff.std() / np.sqrt(len(diff)))
            summ[name]["win_share"] = float((diff > 0).mean())
    by_pos = res.groupby(["pos", "source"])["spearman"].mean().unstack().round(4).to_dict(orient="index")
    by_season = res.groupby(["season", "source"])["spearman"].mean().unstack().round(4)
    summ["by_pos"] = by_pos
    summ["by_season"] = {int(k): v for k, v in by_season.to_dict(orient="index").items()}
    return summ, res


def fit_blend_weights(hist, val, grid=np.round(np.arange(0, 1.0001, 0.05), 2)):
    """Per fold and position: weight on the model in w*model + (1-w)*expert points that minimises squared
    error on the validation season (model trained without that season)."""
    pool = expert_pool(hist)
    curves = ecr_curves(pool)
    out = {}
    for S in FOLDS:
        v = val[val["fold"] == S]
        d = add_ecr_points(pool[pool["season"] == S - 1], curves).join(v["pred"], how="inner").dropna(
            subset=["ecr_pts"])
        for p, g in d.groupby("position"):
            errs = [np.mean((w * g["pred"] + (1 - w) * g["ecr_pts"] - g[TARGET]) ** 2) for w in grid]
            out[(S, p)] = float(grid[int(np.argmin(errs))])
    return out


def fit_rank_blend_weights(hist, val, grid=np.round(np.arange(0, 1.0001, 0.1), 2)):
    """Per fold and position: weight w on the model's rank in w*model_rank + (1-w)*expert_rank that maximises
    the mean weekly rank correlation on the validation season (model trained without that season)."""
    pool = expert_pool(hist)
    out = {}
    for S in FOLDS:
        v = val[val["fold"] == S]
        d = pool[pool["season"] == S - 1].join(v["pred"], how="inner")
        d["mrank"] = d.groupby(["season", "week", "position"])["pred"].rank(ascending=False)
        for p, g in d.groupby("position"):
            best, bw = -9, 0.5
            for w in grid:
                sc = -(w * g["mrank"] + (1 - w) * g["ecr_rank"])
                rho = np.mean([spearmanr(sc[ix], g.loc[ix, TARGET]).correlation
                               for _, ix in g.groupby("week").groups.items() if len(ix) >= 10])
                if rho > best:
                    best, bw = rho, w
            out[(S, p)] = float(bw)
    return out


def report(name, hist, test, val=None, blend=False, extra_sources=None, note=""):
    reg = regression_scores(hist, test, "expected")
    regp = regression_scores(hist, test, "played")
    bw = fit_blend_weights(hist, val) if (blend and val is not None) else None
    extra_sources = dict(extra_sources or {})
    rw = None
    if blend and val is not None:
        rw = fit_rank_blend_weights(hist, val)
        d = expert_pool(hist).join(test["pred"], how="inner")
        d["mrank"] = d.groupby(["season", "week", "position"])["pred"].rank(ascending=False)
        w = pd.Series([rw.get((int(s), p), 0.5) for s, p in zip(d["season"], d["position"])], index=d.index)
        extra_sources["rank_blend_w"] = -(w * d["mrank"] + (1 - w) * d["ecr_rank"])
    ex, res = expert_scores(hist, test, val, extra_sources, bw)
    out = {"name": name, "note": note, "r2_expected": reg, "r2_played": regp, "experts": ex,
           "blend_weights": {f"{k[0]}|{k[1]}": v for k, v in (bw or {}).items()},
           "rank_blend_weights": {f"{k[0]}|{k[1]}": v for k, v in (rw or {}).items()}}
    with open(os.path.join(LAB_DIR, f"{name}_report.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    res.to_csv(os.path.join(LAB_DIR, f"{name}_experts_weekly.csv"), index=False)
    line = (f"{name:28s} R2exp {reg['all']:.4f} [{' '.join(f'{reg[s]:.3f}' for s in FOLDS)}] "
            f"R2played {regp['all']:.4f} | rho model {ex['model']['spearman']:.4f} experts {ex['experts']['spearman']:.4f} "
            f"diff {ex['model']['vs_experts']:+.4f}±{ex['model']['vs_experts_se']:.4f} "
            f"rankblend {ex['rank_blend']['spearman']:.4f}")
    if "rank_blend_w" in ex:
        line += f" rankblend_w {ex['rank_blend_w']['spearman']:.4f} ({ex['rank_blend_w']['vs_experts']:+.4f})"
    print(line, flush=True)
    with open(os.path.join(LAB_DIR, "results.log"), "a") as f:
        f.write(line + (f"  # {note}" if note else "") + "\n")
    return out

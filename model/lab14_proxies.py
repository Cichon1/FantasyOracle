"""Nov 18, 2026: are the candidate variables we tested and dropped just proxies for inputs the model already has?

For every candidate column from the rounds that were tested and left out:
  v5 groups (trend, 2nd-half role, slot, team-first volume, backup-QB, injury type, rookie profile),
  v6 (last week's expert rank, QB-start form, team QB quality), offensive line, market (QB-only in v5.1),
  v8 eye test (FTN, PFR advanced), v9 team pass identity, same-season charting (routes, coverage, personnel,
  pressure: what a weekly feed would add, from a frame built with NFL_AS_AVAILABLE=0)
three numbers:
  redundancy   out-of-fold R² of the candidate predicted from the model's 437 current inputs (LightGBM; folds =
               odd / even seasons). 0.8+ = a proxy (the model already knows it), under 0.5 = mostly new information
  mirrors      the current inputs it tracks most closely (rank correlation)
  unique_vs_miss  correlation of the part the inputs can't explain (candidate - its prediction) with the model's
               miss (actual PPR - walk-forward projection, 2023-2026, lab11_preds.py), within position. Near 0 = the
               new information doesn't line up with what the model gets wrong
Also: raw correlation of the candidate with the miss.

  NFL_AS_AVAILABLE=0 python build_frame_cache.py 2026 11 FRAME0.parquet      (same-season charting frame)
  python lab14_proxies.py FRAME.parquet [FRAME0.parquet]    -> lab/proxy_test.csv, lab/proxy_groups.csv
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import features_v5 as f5
import features_v6 as f6
import features_v8 as f8
import lab5 as L
import nfl_player_model as m

DATA = "data"
N_SAMPLE = int(os.environ.get("PROXY_SAMPLE", "40000"))
P = dict(objective="regression", learning_rate=0.1, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, max_bin=63)
ROUNDS = 200
rng = np.random.default_rng(18)


def candidates(df, frame0=None):
    stats, games, players = m.load(DATA)
    groups, parts = {}, []
    v5, g5 = f5.build_v5(df, DATA, stats, players)
    for g, cols in g5.items():
        groups[f"v5 {g}"] = cols
    parts.append(v5)
    try:
        v6 = f6.build_v6(df)
        groups["v6 last week's expert rank"] = [c for c in f6.ECR_PREV if c in v6]
        groups["v6 QB-start form"] = [c for c in f6.QBS if c in v6]
        groups["v6 team QB quality"] = [c for c in f6.TQ if c in v6]
        parts.append(v6)
    except Exception as e:
        print("v6 skipped:", e)
    for name, fn in [("offensive line", lambda: f6.build_oline(df, DATA)),
                     ("market (QB step only)", lambda: f6.build_market(df, DATA)),
                     ("v8 FTN eye test", lambda: f8.eye_ftn(DATA, df)),
                     ("v8 PFR advanced", lambda: f8.eye_pfr(DATA, df, stats, players))]:
        try:
            x = fn()
            groups[name] = list(x.columns)
            parts.append(x)
        except Exception as e:
            print(name, "skipped:", str(e)[:200])
    try:
        import features_v9 as f9
        x = f9.team_identity(DATA, df)
        groups["v9 team pass identity"] = [c for c in x.columns if c in f9.COLS]
        parts.append(x[groups["v9 team pass identity"]])
    except Exception as e:
        print("v9 skipped:", str(e)[:200])
    if frame0 is not None:
        f0 = pd.read_parquet(frame0)
        f0 = f0.loc[f0.index.intersection(df.index)]
        cols = [c for c in f0.columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
                and c not in ("season", "week")]
        a, b = df.loc[f0.index, cols].astype(float), f0[cols].astype(float)
        diff = [c for c in cols if ((a[c] - b[c]).abs() > 1e-9).mean() > 0.05]
        ss = b[diff].add_suffix("__same_season")
        groups["same-season charting (weekly feed)"] = list(ss.columns)
        parts.append(ss.reindex(df.index))
    cand = pd.concat(parts, axis=1)
    cand = cand.loc[:, ~cand.columns.duplicated()]
    return cand, groups


def main():
    t0 = time.time()
    df = pd.read_parquet(sys.argv[1])
    for c in ["phantom", "snap_only", "upcoming"]:
        df[c] = df[c].eq(True)
    hist = L.hist_rows(df)
    base = L.feature_list(hist)
    cand, groups = candidates(hist, sys.argv[2] if len(sys.argv) > 2 else None)
    cand = cand.reindex(hist.index).replace([np.inf, -np.inf], np.nan)
    print("candidates", cand.shape, round((time.time() - t0) / 60, 1), "min", flush=True)
    test = pd.read_parquet(os.path.join(L.LAB_DIR, "sim_v5proxy_test.parquet"))
    miss = (hist.loc[test.index, L.TARGET] - test["pred"]).rename("miss")
    Xb = hist[base].astype(np.float32)
    rows = []
    only = os.environ.get("PROXY_GROUPS")
    tag = os.environ.get("PROXY_TAG", "")
    for g, cols in groups.items():
        if only and not any(o in g for o in only.split(",")):
            continue
        for c in cols:
            if c in base or c not in cand.columns:
                continue
            y = cand[c]
            ok = y.notna() & np.isfinite(y)
            if ok.mean() < 0.05 or y[ok].nunique() < 2:
                continue
            idx = y[ok].index
            if len(idx) > N_SAMPLE:
                idx = pd.Index(rng.choice(idx, N_SAMPLE, replace=False))
            idx = idx.union(y[ok].index.intersection(miss.index))      # always predict the scored rows
            par = hist.loc[idx, "season"] % 2
            oof = pd.Series(np.nan, index=idx)
            for k in (0, 1):
                tr, te = idx[par.values != k], idx[par.values == k]
                trs = tr if len(tr) <= N_SAMPLE else pd.Index(rng.choice(tr, N_SAMPLE, replace=False))
                if len(trs) < 500 or len(te) == 0:
                    continue
                b = lgb.train(P, lgb.Dataset(Xb.loc[trs], y.loc[trs].astype(float)), ROUNDS)
                oof.loc[te] = b.predict(Xb.loc[te])
            v = oof.notna()
            yy, pp = y.loc[idx][v].astype(float), oof[v]
            r2 = 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2) if yy.var() > 0 else np.nan
            # inputs it tracks most closely
            samp = idx[:8000] if len(idx) > 8000 else idx
            cors = {}
            for bcol in base:
                xb = hist.loc[samp, bcol]
                if xb.notna().sum() < 500 or xb.nunique() < 3:
                    continue
                cors[bcol] = abs(spearmanr(xb, y.loc[samp], nan_policy="omit").correlation)
            top = sorted(((k2, v2) for k2, v2 in cors.items() if np.isfinite(v2)), key=lambda kv: -kv[1])[:3]
            # unique part vs the model's miss, within position
            sc = idx.intersection(miss.index)
            uq, raw, n = np.nan, np.nan, 0
            if len(sc) > 500:
                u = (y.loc[sc].astype(float) - oof.loc[sc])
                mm = miss.loc[sc]
                pos = hist.loc[sc, "position"]
                # pool within-position correlations (each position standardized)
                z = lambda s: s.groupby(pos).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
                good = u.notna() & mm.notna()
                n = int(good.sum())
                if n > 500:
                    uq = float(np.corrcoef(z(u[good]), z(mm[good]))[0, 1])
                    raw = float(np.corrcoef(z(y.loc[sc][good].astype(float)), z(mm[good]))[0, 1])
            rows.append({"group": g, "variable": c, "coverage": round(float(ok.mean()), 3),
                         "redundancy_r2": round(float(r2), 3), "unique_vs_miss": round(uq, 4) if n else np.nan,
                         "raw_vs_miss": round(raw, 4) if n else np.nan, "n_scored": n,
                         "mirrors": "; ".join(f"{k2} ({v2:.2f})" for k2, v2 in top)})
            print(g, c, rows[-1]["redundancy_r2"], rows[-1]["unique_vs_miss"], round((time.time() - t0) / 60, 1),
                  "min", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(L.LAB_DIR, f"proxy_test{tag}.csv"), index=False)
    se = 1 / np.sqrt(out["n_scored"].clip(lower=1))
    out["signal"] = out["unique_vs_miss"].abs() > 3 * se
    summ = out.groupby("group").agg(variables=("variable", "size"),
                                    median_redundancy=("redundancy_r2", "median"),
                                    proxies_r2_08=("redundancy_r2", lambda s: int((s >= 0.8).sum())),
                                    partly_05_08=("redundancy_r2", lambda s: int(((s >= 0.5) & (s < 0.8)).sum())),
                                    new_info_lt05=("redundancy_r2", lambda s: int((s < 0.5).sum())),
                                    max_abs_unique_vs_miss=("unique_vs_miss", lambda s: s.abs().max()),
                                    with_signal=("signal", "sum")).round(3)
    summ.to_csv(os.path.join(L.LAB_DIR, f"proxy_groups{tag}.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()

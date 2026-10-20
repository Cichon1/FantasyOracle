"""
Oct 20, 2026 round: can the model stand without FantasyPros, and do the "eye test" stats help?

Honest walk-forward (lab5.run_walkforward): each season S predicted by a model trained on seasons <= S-1, tree count
early-stopped on S-1 with a model trained on <= S-2 (whose S-1 predictions are the validation set). Proxy learner:
extra-trees LightGBM, 10 per leaf, 3 seeds (the strongest single v5 member). Decisions on validation seasons
2022-2025; test seasons 2023-2026 for information.

Score: mean Spearman rank correlation per position-week among players who suited up, within the experts' pool
(top QB24 / RB48 / WR60 / TE24 by the weekly consensus) - the same head-to-head as the live scoreboard - plus the
experts' own correlation on the same players. Also R² on everyone expected to play.

  python lab8.py FRAME.parquet          -> lab/experiments_v8.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import features_v5 as f5
import features_v6 as f6
import features_v8 as f8
import lab5 as L
import nfl_player_model as m

DATA = "data"
SEEDS = (7, 23, 41)
PARAMS = dict(extra_trees=True, min_data_in_leaf=10)


def frame(path):
    df = pd.read_parquet(path)
    stats, games, players = m.load(DATA)
    v5, _ = f5.build_v5(df, DATA, stats, players)
    df = df.join(v5[[c for c in v5.columns if c not in df.columns]])
    df = df.join(f6.build_market(df, DATA))
    df["qb_prior_starts"] = f6.qb_prior_starts(df, DATA)
    for c in ["phantom", "snap_only", "upcoming"]:
        df[c] = df[c].eq(True)
    df = df.join(f8.eye_ftn(DATA, df))
    df = df.join(f8.eye_pfr(DATA, df, stats, players))
    return df


def pool(hist):
    d = hist[hist["ecr_week_ecr"].notna() & (hist["season_type"] == "REG") & hist["expected"] & ~hist["phantom"]][
        ["season", "week", "position", "ecr_week_ecr", L.TARGET]].copy()
    d["ecr_rank"] = d.groupby(["season", "week", "position"])["ecr_week_ecr"].rank(method="first")
    return d[d["ecr_rank"] <= d["position"].map(L.POOL)]


def rank_scores(pl, preds):
    d = pl.join(preds.rename("pred"), how="inner")
    rows = []
    for (S, w, p), g in d.groupby(["season", "week", "position"]):
        if len(g) < 10:
            continue
        y = g[L.TARGET].values
        rows.append({"season": S, "pos": p, "model": spearmanr(g["pred"], y).correlation,
                     "experts": spearmanr(-g["ecr_rank"], y).correlation,
                     "blend": spearmanr(-(g["pred"].rank(ascending=False) + g["ecr_rank"]), y).correlation})
    return pd.DataFrame(rows)


def r2(hist, preds):
    d = hist.loc[preds.index]
    d = d[d["expected"]]
    y, p = d[L.TARGET].values, preds.loc[d.index].values
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def main():
    df = frame(sys.argv[1])
    hist = L.hist_rows(df)
    base = L.feature_list(hist)
    ecr = [c for c in base if c.startswith("ecr_")]
    nofp = [c for c in base if c not in ecr]
    variants = {
        "v5 inputs (with FantasyPros)": base,
        "no FantasyPros": nofp,
        "no FantasyPros + FTN eye test": nofp + f8.FTN_COLS,
        "no FantasyPros + PFR advanced": nofp + f8.PFR_COLS,
        "no FantasyPros + both": nofp + f8.FTN_COLS + f8.PFR_COLS,
        "with FantasyPros + both": base + f8.FTN_COLS + f8.PFR_COLS,
    }
    only = sys.argv[2:] or list(variants)
    pl = pool(hist)
    out_path = os.path.join(L.LAB_DIR, "experiments_v8.csv")
    rows = [] if not os.path.exists(out_path) else pd.read_csv(out_path).to_dict("records")
    done = {r["variant"] for r in rows}
    for name in only:
        if name in done:
            continue
        X = [c for c in dict.fromkeys(variants[name]) if c in hist.columns]
        test, val = L.run_walkforward(hist, X, "v8_" + "".join(ch for ch in name if ch.isalnum()), PARAMS, SEEDS)
        rt, rv = rank_scores(pl, test["pred"]), rank_scores(pl, val["pred"])
        rec = {"variant": name, "inputs": len(X),
               "val_model": rv["model"].mean(), "val_experts": rv["experts"].mean(), "val_blend": rv["blend"].mean(),
               "test_model": rt["model"].mean(), "test_experts": rt["experts"].mean(), "test_blend": rt["blend"].mean(),
               "val_r2": r2(hist, val["pred"]), "test_r2": r2(hist, test["pred"]), "position_weeks_val": len(rv),
               "by_pos_val": json.dumps(rv.groupby("pos")["model"].mean().round(4).to_dict()),
               "by_pos_test": json.dumps(rt.groupby("pos")["model"].mean().round(4).to_dict())}
        rows.append(rec)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)


if __name__ == "__main__":
    main()

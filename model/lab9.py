"""
Nov 9, 2026 round: does a team's pass-first identity this season (vs its coach's history, and a broken run game) help
the weekly model? (features_v9.py; the Gemini review's idea #3, Jackson's Green Bay / Christian Watson question.)

Same honest setup as lab8.py: walk-forward (lab5.run_walkforward), proxy learner extra-trees LightGBM 10 per leaf,
3 seeds; decisions on validation seasons 2022-2025, test seasons 2023-2026 for information. Score: mean Spearman rank
correlation per position-week among players who suited up in the experts' pool, paired against the baseline run
(difference +/- standard error over season x position cells). Also the mean error on players whose team passes 4+
points more (or less) than its coach's history this season, where the change should show if anywhere.

  python lab9.py FRAME.parquet          -> lab/experiments_v9.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import features_v5 as f5
import features_v6 as f6
import features_v9 as f9
import lab5 as L
import lab8 as L8
import nfl_player_model as m

DATA = "data"


def frame(path):
    df = pd.read_parquet(path)
    if "mkt_qb_start_share" not in df:
        stats, games, players = m.load(DATA)
        v5, _ = f5.build_v5(df, DATA, stats, players)
        df = df.join(v5[[c for c in v5.columns if c not in df.columns]])
        df = df.join(f6.build_market(df, DATA))
        df["qb_prior_starts"] = f6.qb_prior_starts(df, DATA)
    for c in ["phantom", "snap_only", "upcoming"]:
        df[c] = df[c].eq(True)
    return df.join(f9.team_identity(DATA, df))


def main():
    df = frame(sys.argv[1])
    hist = L.hist_rows(df)
    base = L.feature_list(hist)
    variants = {"baseline (v5 inputs)": base, "+ team pass identity (v9)": base + f9.COLS}
    pl = L8.pool(hist)
    out_path = os.path.join(L.LAB_DIR, "experiments_v9.csv")
    rows, preds = [], {}
    for name, X in variants.items():
        X = [c for c in dict.fromkeys(X) if c in hist.columns]
        test, val = L.run_walkforward(hist, X, "v9_" + "".join(ch for ch in name if ch.isalnum()), L8.PARAMS, L8.SEEDS)
        preds[name] = (test, val)
    (bt, bv) = preds["baseline (v5 inputs)"]
    div = (hist["v9_proe_cur"] - hist["v9_proe_coach3"]).abs() >= 4
    for name, (test, val) in preds.items():
        rt, rv = L8.rank_scores(pl, test["pred"]), L8.rank_scores(pl, val["pred"])
        brt, brv = L8.rank_scores(pl, bt["pred"]), L8.rank_scores(pl, bv["pred"])
        rec = {"variant": name, "inputs": len(variants[name])}
        for ph, r, b in (("val", rv, brv), ("test", rt, brt)):
            dd = r.assign(d=r["model"].values - b["model"].values)
            cell = dd.groupby(["season", "pos"])["d"].mean()
            rec.update({f"{ph}_model": r["model"].mean(), f"{ph}_experts": r["experts"].mean(), f"{ph}_blend": r["blend"].mean(),
                        f"{ph}_vs_base": dd["d"].mean(), f"{ph}_se": cell.std() / np.sqrt(len(cell)),
                        f"{ph}_by_pos": json.dumps(dd.groupby("pos")["d"].mean().round(4).to_dict())})
            p = (val if ph == "val" else test)["pred"]
            h = hist.loc[p.index]
            sel = h["expected"] & div.loc[p.index] & h["position"].isin(["WR", "TE", "QB"])
            rec[f"{ph}_bias_divergent_pass_teams"] = float((h.loc[sel, L.TARGET] - p[sel]).mean())
            rec[f"{ph}_r2"] = L8.r2(hist, p)
        rows.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

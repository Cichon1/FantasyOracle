"""Out-of-sample weekly projections for past seasons (season-level walk-forward), used to test ROS methods that start
from the weekly projection. Proxy of the v5.1 weekly model: extra-trees LightGBM (10 per leaf, 3 seeds, v5 tree
counts), QB market step, rank blend with the FantasyPros weekly consensus. Each season S is predicted by models
trained only on seasons before S (harder than live, which also trains on S's earlier weeks)."""
import json, sys, time
import numpy as np, pandas as pd
import lab5 as L, weekly_v5 as W, features_v6 as f6, ros_neutral as RN

frame, out = sys.argv[1], sys.argv[2]
seasons = [int(s) for s in sys.argv[3].split(",")]
cfg = json.load(open("v5_config.json"))
df = pd.read_parquet(frame)
hist = L.hist_rows(df)
X = L.feature_list(hist)
res = []
t0 = time.time()
NEUTRAL = "--neutral" in sys.argv   # Nov 9, 2026: also a schedule-neutral projection (ros_neutral.py)
for S in seasons:
    trn = hist[hist["season"] < S]
    rows = hist[(hist["season"] == S) & (hist["season_type"] == "REG")].copy()
    nrows = RN.neutralize(rows, df) if NEUTRAL else None
    for pos in L.POS_MODELS:
        key = "/".join(pos)
        tp, rp = trn[trn["position"].isin(pos)], rows[rows["position"].isin(pos)]
        params = dict(L.LAB_PARAMS, extra_trees=True, min_data_in_leaf=10)
        if NEUTRAL:   # same models (cached), predicted on both the real and the neutral inputs
            both = pd.concat([rp, nrows.loc[rp.index].set_axis([("n", i) for i in rp.index])])
            pb = W.train_predict(tp, both, X, L.TARGET, params, cfg["rounds"]["et10"][key] * 1.1, [7, 23, 41])
            rows.loc[rp.index, "pred_ppr_model"] = pb[:len(rp)]
            rows.loc[rp.index, "pred_ppr_neutral"] = pb[len(rp):]
        else:
            rows.loc[rp.index, "pred_ppr_model"] = W.train_predict(tp, rp, X, L.TARGET, params,
                                                                   cfg["rounds"]["et10"][key] * 1.1, [7, 23, 41])
        # without the weekly expert-ranking inputs (the Thursday run, or a Tuesday ROS run for the next week)
        Xn = [c for c in X if not c.startswith("ecr_week_")]
        rows.loc[rp.index, "pred_ppr_noweek"] = W.train_predict(tp, rp, Xn, L.TARGET, params,
                                                                cfg["rounds_noweek"]["et10"][key] * 1.1, [7, 23, 41])
    qm = cfg["qb_market"]
    nw = rows.assign(pred_ppr_model=rows["pred_ppr_noweek"])
    rows["pred_ppr_noweek"] = f6.qb_market_adjust(nw, qm["w_model"], qm["cold_offsets"])
    rows["pred_ppr_model"] = f6.qb_market_adjust(rows, qm["w_model"], qm["cold_offsets"])
    if NEUTRAL:
        nn = nrows.assign(pred_ppr_model=rows["pred_ppr_neutral"])
        rows["pred_ppr_neutral"] = f6.qb_market_adjust(nn, qm["w_model"], qm["cold_offsets"])
    rows = W.blend_with_experts(rows)
    res.append(rows[[c for c in ["season", "week", "player_id", "game_id", "position", "pred_ppr_model", "pred_ppr_official",
                                 "pred_ppr_noweek", "pred_ppr_neutral"] if c in rows.columns]])
    print(S, len(rows), round((time.time() - t0) / 60, 1), "min", flush=True)
pd.concat(res).to_parquet(out)
print("saved", out, flush=True)

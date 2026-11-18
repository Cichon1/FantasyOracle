"""Nov 17, 2026 league sim, step 1: honest walk-forward projections for every player-week, 2023-2026.

Same harness and learner as lab8 / lab10 (extra-trees LightGBM proxy, 10 per leaf, 3 seeds; season S predicted by
models trained on seasons <= S-1), v5 inputs with FantasyPros. Writes lab/sim_rows.parquet with, per player-week:
actual PPR points, the model's projection, the weekly and rest-of-season FantasyPros ranks and the flags the sim
needs (expected to play, sat out, Thursday game).

  NFL_AS_AVAILABLE=1 python build_frame_cache.py 2026 11 FRAME.parquet
  python lab11_preds.py FRAME.parquet
"""
import os
import sys

import pandas as pd

import lab5 as L

PARAMS = dict(extra_trees=True, min_data_in_leaf=10)
SEEDS = (7, 23, 41)
KEEP = ["player_id", "player_display_name", "season", "week", "season_type", "position", "team", "opponent_team",
        "expected", "phantom", "ctx_thursday", "ecr_week_ecr", "ecr_week_pos_rank", "ecr_ros_pos_rank",
        "ecr_preseason_pos_rank", L.TARGET]


def main():
    df = pd.read_parquet(sys.argv[1])
    for c in ["phantom", "snap_only", "upcoming"]:
        df[c] = df[c].eq(True)
    hist = L.hist_rows(df)
    X = L.feature_list(hist)
    test, val = L.run_walkforward(hist, X, "sim_v5proxy", PARAMS, SEEDS)
    keep = [c for c in KEEP if c in hist.columns]
    out = hist.loc[test.index, keep].join(test[["pred"]])
    out.to_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet"))
    print(out.shape, out.groupby("season").size().to_dict(), flush=True)


if __name__ == "__main__":
    main()

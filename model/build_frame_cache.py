"""Build and cache the full feature frame (history + the target week's upcoming rows) exactly as weekly_v5 does."""
import sys, time, json
import pandas as pd
import features_v5 as f5, features_v6 as f6, nfl_player_model as m
season, week, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
cfg = json.load(open("v5_config.json"))
t0 = time.time()
df = m.build_frame("data", season, week)
print("frame", df.shape, round((time.time()-t0)/60, 1), "min", flush=True)
stats, games, players = m.load("data")
v5, _ = f5.build_v5(df, "data", stats, players)
df = df.join(v5[[c for c in v5.columns if c not in df.columns]])
if cfg.get("qb_market"):
    df = df.join(f6.build_market(df, "data"))
    df["qb_prior_starts"] = f6.qb_prior_starts(df, "data")
for c in ["phantom", "snap_only", "upcoming"]:
    df[c] = df[c].eq(True)
df.to_parquet(out)
print("saved", df.shape, round((time.time()-t0)/60, 1), "min", flush=True)

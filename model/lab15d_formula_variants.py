import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, '.')
import lab5 as L, lab11_sim as S11, lab12_season as G, lab15_trade as T15
frame = pd.read_parquet('/home/claude/sim/frame.parquet', columns=["season","week","season_type","position","expected","ecr_week_pos_rank",L.TARGET])
cv = S11.curves(frame); d = S11.prepare(pd.read_parquet('lab/sim_rows.parquet'), cv)
games = pd.read_parquet("data/games.parquet", columns=["season","week","game_type","gameday","home_team","away_team"])
ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id","gsis_id"], low_memory=False)
runs = pd.read_parquet('lab/trade_sim_runs.parquet')
out = []
for S in [2023, 2024, 2025]:
    board, _ = G.draft_board(S, games, ids); se = G.Season(d, S, board, games, 17)
    nm = {n: i for i, n in enumerate(se.name)}
    for W in T15.WEEKS:
        v = se.ros(W, cv[S-1])["official"]
        cw = np.zeros_like(v)
        for p in range(4):
            sel = se.pos == p; srt = np.sort(v[sel])[::-1]; cw[sel] = srt[T15.WAIVER_RANK[p]-1]
        f = {"starter_clip": np.maximum(v, 0), "waiver_lin": v - cw, "hub_lin": v - cw / 2}
        r = runs[(runs.season == S) & (runs.week == W)].copy()
        xi = r.player.map(nm).values; di = r.dropped.values
        for k, a in f.items():
            r["official_" + k] = a[xi] - np.where(di >= 0, a[np.clip(di, 0, None)], 0)
        out.append(r)
r = pd.concat(out)
fs = ['official_hub', 'official_starter', 'official_starter_clip', 'official_waiver_lin', 'official_hub_lin']
for t in ['d_pts', 'd_wins', 'd_playoffs']:
    cells = {f: [] for f in fs}
    for _, g in r.groupby(['season', 'week', 'league']):
        if g[t].std() == 0: continue
        for f in fs: cells[f].append(spearmanr(g[f], g[t]).correlation)
    base = np.array(cells['official_hub'])
    print(t, {f: (round(np.nanmean(v), 4), round(np.nanmean(np.array(v) - base), 4), round(np.nanstd(np.array(v) - base) / np.sqrt(len(v)), 4)) for f, v in cells.items()})

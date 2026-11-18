import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '.')
import lab5 as L, lab8 as L8, lab14_proxies as P
df = pd.read_parquet('/home/claude/sim/frame.parquet')
for c in ["phantom", "snap_only", "upcoming"]: df[c] = df[c].eq(True)
hist = L.hist_rows(df)
base = L.feature_list(hist)
cand, groups = P.candidates(hist, '/home/claude/sim/frame0.parquet')
d = pd.read_csv('lab/proxy_test.csv'); d['thr'] = 3 / np.sqrt(d.n_scored.clip(lower=1))
sig = list(d[d.unique_vs_miss.abs() > d.thr].variable)
# same-season columns are not available live: test only those a live model could have
live = [c for c in sig if not c.endswith('__same_season')]
print('live signal vars', live, flush=True)
h2 = hist.join(cand[[c for c in live if c not in hist.columns]].reindex(hist.index))
test, val = L.run_walkforward(h2, base + live, 'proxy_sig', dict(extra_trees=True, min_data_in_leaf=10), (7, 23, 41))
basev = pd.read_parquet('lab/sim_v5proxy_val.parquet'); baset = pd.read_parquet('lab/sim_v5proxy_test.parquet')
pl = L8.pool(hist)
out = {}
for nm, (a, b) in {'val': (val, basev), 'test': (test, baset)}.items():
    ra, rb = L8.rank_scores(pl, a['pred']), L8.rank_scores(pl, b['pred'])
    m = ra.merge(rb, on=['season', 'pos'], suffixes=('', '_b'), how='inner') if False else None
    out[nm] = {'with': round(ra['model'].mean(), 4), 'base': round(rb['model'].mean(), 4),
               'diff': round(ra['model'].mean() - rb['model'].mean(), 4),
               'se': round((ra['model'].values - rb['model'].values).std(ddof=1) / np.sqrt(len(ra)), 4) if len(ra) == len(rb) else None}
print(json.dumps(out))

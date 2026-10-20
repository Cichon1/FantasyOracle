"""Value model v2 test: can we predict 'beats his market price' better than the raw gap?
Rows: ros_lab/value_rows.parquet (value_lab.py), 2020-2025 snapshots weeks 3..last-3, experts' pool.
Fit on 2020-2022 only, score on 2023-2025 (and the reverse-free walk-forward 2020-21->22 etc. for stability)."""
import numpy as np, pandas as pd, lightgbm as lgb, json
from scipy.stats import spearmanr
import ros as R, ros_lab as L
d = pd.read_parquet('ros_lab/value_rows.parquet')
rows = pd.read_parquet('ros_lab/rows.parquet')
extra = ['inv_age', 'xfp_ewm', 'xfp_ewm_long', 'xfp_td_luck_ewm', 'fantasy_points_ppr_ewm_long', 'snap_pct_ewm',
         'cr_fantasy_points_ppr_last_season', 'cr_fantasy_points_ppr_season_to_date', 'dc_depth_rank', 'rc_route_participation',
         'implied_pts', 'weeks_since_last', 'inv_draft_overall', 'cr_is_rookie', 'ros_opp_def_rel', 'ros_res']
extra = [c for c in extra if c in rows.columns]
k = rows[['season', 'week', 'player_id'] + extra].drop_duplicates(['season', 'week', 'player_id'])
d = d.merge(k, on=['season', 'week', 'player_id'], how='left', suffixes=('', '_x'))
# alternative worth: the pure model (weekly model before the expert blend)
d['model_pure'] = L.ros_mix(d, 0.8, 'wk_model')
out = []
for (S, W, p), g in d.groupby(['season', 'week', 'position']):
    sm = np.sort(g['model_pure'].values)[::-1]
    out.append(pd.Series(sm[g['mr'].values.clip(max=len(sm)) - 1], index=g.index))
d['mkt_model_pure'] = pd.concat(out)
d['gap_pure_pg'] = (d['model_pure'] - d['mkt_model_pure']) / d['G']
d['pos_code'] = d['position'].map({'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3})
d['model_pg'] = d['model'] / d['G']
d['form_gap'] = d['fantasy_points_ppr_ewm'] - d['model_pg']          # recent form vs our per-game view (market chases form?)
d['luck'] = d['fantasy_points_ppr_ewm'] - d.get('xfp_ewm', np.nan)  # scoring above expected (TD luck)
d['rate_vs_wk'] = d['rate'] - d['wk_official']
d['mr_frac'] = d['mr'] / d['position'].map({'QB': 32, 'RB': 60, 'WR': 80, 'TE': 32})
F = ['gap_pg', 'gap_pct', 'gap_pure_pg', 'mr', 'mr_frac', 'pos_code', 'week', 'G', 'ros_out_now', 'games_hist', 'model_pg',
     'form_gap', 'luck', 'rate_vs_wk'] + extra
F = [f for f in dict.fromkeys(F) if f in d.columns]
VAL, TEST = [2020, 2021, 2022], [2023, 2024, 2025]
def sp(a, b): return spearmanr(a, b).correlation
def cellwise(v, col):   # mean Spearman within season-week-position (how a single week's labels rank players)
    return np.mean([sp(g[col], g['outc_pg']) for _, g in v.groupby(['season', 'week', 'position']) if len(g) >= 10])
res = {}
for nm, col in [('gap (current)', 'gap_pg'), ('gap, pure model', 'gap_pure_pg')]:
    res[nm] = {ph: (round(sp(d.loc[d.season.isin(ss), col], d.loc[d.season.isin(ss), 'outc_pg']), 4),
                    round(cellwise(d[d.season.isin(ss)], col), 4)) for ph, ss in (('val', VAL), ('test', TEST))}
P = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=2)
# inner check on validation: train 2020-21, score 2022 to pick rounds
tr, va = d[d.season.isin([2020, 2021])], d[d.season == 2022]
b = lgb.train(dict(P, seed=7), lgb.Dataset(tr[F], tr['outc_pg']), 2000, valid_sets=[lgb.Dataset(va[F], va['outc_pg'])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
rounds = max(50, b.best_iteration)
va = va.assign(pred=b.predict(va[F], num_iteration=rounds))
res['learned (val 2022 only)'] = {'val2022': (round(sp(va['pred'], va['outc_pg']), 4), round(cellwise(va, 'pred'), 4)),
                                  'gap on 2022': (round(sp(va['gap_pg'], va['outc_pg']), 4), round(cellwise(va, 'gap_pg'), 4))}
# final: train on 2020-2022, score 2023-2025
trv, te = d[d.season.isin(VAL)], d[d.season.isin(TEST)].copy()
preds = np.mean([lgb.train(dict(P, seed=s), lgb.Dataset(trv[F], trv['outc_pg']), int(rounds * 1.1)).predict(te[F]) for s in (7, 23, 41)], axis=0)
te['pred'] = preds
res['learned (train 2020-22)'] = {'test': (round(sp(te['pred'], te['outc_pg']), 4), round(cellwise(te, 'pred'), 4))}
# labels on test: top / bottom 27% like the current label shares
for nm, col in [('gap', 'gap_pg'), ('learned', 'pred')]:
    q = te[col].rank(pct=True)
    lab = np.where(q >= 0.73, 'Undervalued', np.where(q <= 0.29, 'Overvalued', 'At value'))
    t = te.assign(lab=lab).groupby('lab').agg(share=('beat', 'size'), beat=('beat', 'mean'), vs_mkt=('outc_pg', 'mean'))
    t['share'] = (t['share'] / len(te)).round(3)
    res[f'labels test ({nm})'] = t.round(3).to_dict()
imp = pd.Series(b.feature_importance('gain'), index=F).sort_values(ascending=False)
res['top features'] = (imp / imp.sum()).round(3).head(10).to_dict()
res['rounds'] = rounds
print(json.dumps(res, indent=1))

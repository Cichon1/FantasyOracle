import numpy as np, pandas as pd, lightgbm as lgb, json
from scipy.stats import spearmanr
exec(open('value_lab2.py').read().split("VAL, TEST =")[0])
VAL, TEST = [2020, 2021, 2022], [2023, 2024, 2025]
d['act_pg'] = d['ros_pts'] / d['G']
# price = what players at this market rank (same position) typically score per game, from the OTHER seasons
# (leave-one-season-out), smoothed over +-3 ranks
def price(d):
    out = pd.Series(np.nan, index=d.index)
    for S in d.season.unique():
        o = d[d.season != S].groupby(['position', 'mr'])['act_pg'].mean().reset_index()
        o['sm'] = o.groupby('position')['act_pg'].transform(lambda x: x.rolling(7, center=True, min_periods=1).mean())
        m = d[d.season == S][['position', 'mr']].merge(o[['position', 'mr', 'sm']], on=['position', 'mr'], how='left')
        out[d.index[d.season == S]] = m['sm'].values
    return out
d['price_pg'] = price(d)
d['outc2'] = d['act_pg'] - d['price_pg']
d['beat2'] = (d['outc2'] > 0).astype(float)
def sp(a, b): return spearmanr(a, b).correlation
def cellwise(v, col, y): return np.mean([sp(g[col], g[y]) for _, g in v.groupby(['season', 'week', 'position']) if len(g) >= 10])
Fn = [f for f in F if f not in ('mr', 'mr_frac')]
P = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=2)
tr, va = d[d.season.isin([2020, 2021])], d[d.season == 2022]
b = lgb.train(dict(P, seed=7), lgb.Dataset(tr[Fn], tr['outc2']), 2000, valid_sets=[lgb.Dataset(va[Fn], va['outc2'])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
r = max(50, b.best_iteration)
va = va.assign(pred=b.predict(va[Fn], num_iteration=r))
trv, te = d[d.season.isin(VAL)], d[d.season.isin(TEST)].copy()
te['pred'] = np.mean([lgb.train(dict(P, seed=s), lgb.Dataset(trv[Fn], trv['outc2']), int(r * 1.1)).predict(te[Fn]) for s in (7, 23, 41)], axis=0)
res = {'rounds': r, 'mean outc2': round(d['outc2'].mean(), 3),
       'val2022': {k: round(cellwise(va, k, 'outc2'), 4) for k in ['pred', 'gap_pg', 'gap_pure_pg']},
       'test': {k: round(cellwise(te, k, 'outc2'), 4) for k in ['pred', 'gap_pg', 'gap_pure_pg']}}
for nm, col in [('v1 gap +-1.0', None), ('v2', 'pred')]:
    if col is None:
        lab = np.where(te['gap_pg'] >= 1, 'U', np.where(te['gap_pg'] <= -1, 'O', 'A'))
    else:
        q = te[col].rank(pct=True); lab = np.where(q >= 0.73, 'U', np.where(q <= 0.29, 'O', 'A'))
    t = te.assign(lab=lab).groupby('lab').agg(share=('beat2', 'size'), beat=('beat2', 'mean'), vs_price=('outc2', 'mean'))
    t['share'] = t['share'] / len(te)
    res[nm] = t.round(3).to_dict()
# are top market ranks systematically labeled overvalued?
te['top5'] = te['mr'] <= 5
res['top-5 market rank share labeled overvalued (v2)'] = round(float((te.loc[te.top5, 'pred'].rank(pct=True) <= 0.29).mean()), 3)
res['top-5 avg outc2'] = round(float(te.loc[te.top5, 'outc2'].mean()), 3)
print(json.dumps(res, indent=1))
d[['season','week','position','player_id','price_pg','outc2']].to_parquet('ros_lab/value_outc2.parquet')

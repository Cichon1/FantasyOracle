import numpy as np, pandas as pd, lightgbm as lgb, json
from scipy.stats import spearmanr
exec(open('value_lab2.py').read().split("VAL, TEST =")[0])   # same rows and features
VAL, TEST = [2020, 2021, 2022], [2023, 2024, 2025]
# rank-based outcome: how many spots he beat his market rank by, as a share of the pool (no order-statistic scale)
d['act_rank'] = d.groupby(['season', 'week', 'position'])['ros_pts'].rank(ascending=False, method='average')
d['n_pool'] = d.groupby(['season', 'week', 'position'])['ros_pts'].transform('size')
d['outc_rank'] = (d['mr'] - d['act_rank']) / d['n_pool']
def sp(a, b): return spearmanr(a, b).correlation
def cellwise(v, col, y): return np.mean([sp(g[col], g[y]) for _, g in v.groupby(['season', 'week', 'position']) if len(g) >= 10])
Fn = [f for f in F if f not in ('mr', 'mr_frac')]
P = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=2)
res = {}
for y in ['outc_pg', 'outc_rank']:
    for fname, FF in [('all', F), ('no market-rank inputs', Fn)]:
        tr, va = d[d.season.isin([2020, 2021])], d[d.season == 2022]
        b = lgb.train(dict(P, seed=7), lgb.Dataset(tr[FF], tr[y]), 2000, valid_sets=[lgb.Dataset(va[FF], va[y])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        r = max(50, b.best_iteration)
        trv, te = d[d.season.isin(VAL)], d[d.season.isin(TEST)].copy()
        te['pred'] = np.mean([lgb.train(dict(P, seed=s), lgb.Dataset(trv[FF], trv[y]), int(r * 1.1)).predict(te[FF]) for s in (7, 23, 41)], axis=0)
        va2 = va.assign(pred=b.predict(va[FF], num_iteration=r))
        res[f'{y} | {fname}'] = {'val2022 learned': round(cellwise(va2, 'pred', y), 4), 'val2022 gap': round(cellwise(va2, 'gap_pg', y), 4),
                                 'val2022 pure gap': round(cellwise(va2, 'gap_pure_pg', y), 4),
                                 'test learned': round(cellwise(te, 'pred', y), 4), 'test gap': round(cellwise(te, 'gap_pg', y), 4),
                                 'test pure gap': round(cellwise(te, 'gap_pure_pg', y), 4), 'rounds': r}
        if y == 'outc_rank' and fname != 'all':
            imp = pd.Series(b.feature_importance('gain'), index=FF).sort_values(ascending=False)
            res['top features (rank outcome, no mr)'] = (imp / imp.sum()).round(3).head(8).to_dict()
print(json.dumps(res, indent=1))

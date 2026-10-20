import numpy as np, pandas as pd, json
from scipy.stats import spearmanr
exec(open('value_lab2.py').read().split("VAL, TEST =")[0])
o = pd.read_parquet('ros_lab/value_outc2.parquet')
d = d.merge(o, on=['season', 'week', 'position', 'player_id'], how='left')
VAL, TEST = [2020, 2021, 2022], [2023, 2024, 2025]
def sp(a, b): return spearmanr(a, b).correlation
def cellwise(v, col): return np.mean([sp(g[col], g['outc2']) for _, g in v.groupby(['season', 'week', 'position']) if len(g) >= 10 and g[col].notna().sum() >= 10])
d['luck'] = d['luck'].fillna(0); d['form_gap'] = d['form_gap'].fillna(0)
cands = {'gap_pg': d['gap_pg'], 'gap_pct': d['gap_pct'], 'gap_pure_pg': d['gap_pure_pg'],
         'gap - 0.15*luck': d['gap_pg'] - 0.15 * d['luck'], 'gap - 0.3*luck': d['gap_pg'] - 0.3 * d['luck'],
         'gap + 0.1*form': d['gap_pg'] + 0.1 * d['form_gap'], 'gap - 0.1*form': d['gap_pg'] - 0.1 * d['form_gap'],
         'avg(gap, pure)': (d['gap_pg'] + d['gap_pure_pg']) / 2,
         'gap, out players x0.5': d['gap_pg'] * np.where(d['ros_out_now'] == 1, 0.5, 1)}
res = {}
for k, v in cands.items():
    d['_s'] = v
    res[k] = {'val': round(cellwise(d[d.season.isin(VAL)], '_s'), 4), 'test': round(cellwise(d[d.season.isin(TEST)], '_s'), 4)}
print(json.dumps(res, indent=1))
v = d[d.season.isin(VAL)]
up = (v['gap_pg'] >= 1).mean(); dn = (v['gap_pg'] <= -1).mean()
hi = float(v['gap_pct'].quantile(1 - up)); lo = float(v['gap_pct'].quantile(dn))
print('pct thresholds from validation', round(hi, 3), round(lo, 3))
te = d[d.season.isin(TEST)]
for nm, lab in [('v1 points gap +-1.0', np.where(te['gap_pg'] >= 1, 'U', np.where(te['gap_pg'] <= -1, 'O', 'A'))),
                ('percent gap', np.where(te['gap_pct'] >= hi, 'U', np.where(te['gap_pct'] <= lo, 'O', 'A')))]:
    t = te.assign(lab=lab).groupby('lab').agg(n=('outc2', 'size'), beat=('outc2', lambda x: (x > 0).mean()), vs_price_pg=('outc2', 'mean'))
    t['n'] = (t['n'] / len(te)).round(3)
    print(nm); print(t.round(3).to_string())

"""Undervalued / at value / overvalued: our ROS projection vs the FantasyPros ROS consensus (market), walk-forward.
gap = our ROS points - the points our model gives the player ranked at his market rank (same position, same week);
outcome = his actual ROS points - the actual ROS points of the player who finished at his market rank."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import ros as R
import ros_lab as L
d = L.attach_weekly(pd.read_parquet('ros_lab/base_md40.parquet'), 'ros_lab/weekly_oos.parquet')
lw = d['season'].map(R.last_week)
d = d[(d['week'] >= 3) & (d['week'] <= lw - 3) & d['ros_ecr_pos_rank'].notna()].copy()
d['model'] = L.ros_mix(d, 0.8, 'wk_official')
d['G'] = (d['ros_games_after'] + R.cur_open(d)).clip(lower=1)
out = []
for (S, W, p), g in d.groupby(['season', 'week', 'position']):
    g = g.copy()
    g['mr'] = g['ros_ecr_pos_rank'].rank(method='first').astype(int)        # market rank among ranked players
    sm = np.sort(g['model'].values)[::-1]; sa = np.sort(g['ros_pts'].values)[::-1]
    g['mkt_model'] = sm[g['mr'].values - 1]; g['mkt_actual'] = sa[g['mr'].values - 1]
    g['our_rank'] = g['model'].rank(ascending=False, method='first')
    out.append(g)
d = pd.concat(out)
d = d[d['mr'] <= d['position'].map({'QB': 32, 'RB': 60, 'WR': 80, 'TE': 32})]
d['gap'] = d['model'] - d['mkt_model']; d['gap_pg'] = d['gap'] / d['G']
d['gap_pct'] = d['model'] / d['mkt_model'].clip(lower=1) - 1
d['outc'] = d['ros_pts'] - d['mkt_actual']; d['outc_pg'] = d['outc'] / d['G']
d['beat'] = (d['outc'] > 0).astype(float)
d.to_parquet('ros_lab/value_rows.parquet')
VAL, TEST = [2020, 2021, 2022], [2023, 2024, 2025]
for nm, ss in (('VAL', VAL), ('TEST', TEST)):
    v = d[d['season'].isin(ss)]
    print(nm, len(v), 'spearman gap_pg~outc_pg', round(v[['gap_pg', 'outc_pg']].corr('spearman').iloc[0, 1], 3),
          'gap_pct~outc_pg', round(v[['gap_pct', 'outc_pg']].corr('spearman').iloc[0, 1], 3))
    v = v.copy(); v['dec'] = pd.qcut(v['gap_pg'], 10, labels=False)
    print(v.groupby('dec').agg(gap_pg=('gap_pg', 'mean'), outc_pg=('outc_pg', 'mean'), beat=('beat', 'mean'), n=('beat', 'size')).round(3).to_string())

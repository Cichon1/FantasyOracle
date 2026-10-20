"""Value labels in each league's scoring, walk-forward (2020-2022 validation, 2023-2025 test).
worth = our model's ROS points converted to the league's scoring; price = the league-scoring model points of the player at
his market rank (FantasyPros ROS consensus); gap per game left; outcome = actual ROS points in the league's scoring minus
the actual league points of whoever finished at his market rank."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '.')
import ros as R, ros_lab as L
cfg = json.load(open('v5_config.json'))['ros']
LG = pd.read_csv('dash_out/leagues_v5.csv')
VAL, TEST = [2020, 2021, 2022], [2023, 2024, 2025]
POOLV = {'QB': 32, 'RB': 60, 'WR': 80, 'TE': 32}
d = L.attach_weekly(pd.read_parquet('ros_lab/base_md40.parquet'), 'ros_lab/weekly_oos.parquet')
rates = ['passing_tds_ewm_long', 'passing_interceptions_ewm_long', 'passing_yards_ewm_long', 'receptions_ewm_long',
         'rushing_yards_ewm_long', 'receiving_yards_ewm_long', 'ros_res']
d = d.merge(pd.read_parquet('ros_lab/ros_rows.parquet', columns=['season', 'week', 'player_id'] + rates), on=['season', 'week', 'player_id'])
lw = d['season'].map(R.last_week)
d = d[(d['week'] >= 3) & (d['week'] <= lw - 3)].copy()
d['model'] = L.ros_mix(d, 0.8, 'wk_official')
o = R.cur_open(d)
d['G'] = (d['ros_games_after'] + o).clip(lower=1)
d['first_wk'] = np.where(o > 0, d['week'], d['week'] + 1)
# weekly stats: per-game league deltas and fumbles history
W = pd.concat([pd.read_parquet(f'data/week_{s}.parquet') for s in range(2019, 2026)])
W = W[W['season_type'] == 'REG'][['season', 'week', 'player_id', 'passing_tds', 'passing_interceptions', 'passing_yards',
                                   'receptions', 'rushing_yards', 'receiving_yards', 'fumbles_lost_total', 'fantasy_points_ppr']].fillna(0)
# fumbles lost per game known before the snapshot (last season + this season so far), shrunk to the average
def fum_rate(rows):
    out = pd.Series(np.nan, index=rows.index)
    for (S, Wk), g in rows.groupby(['season', 'week']):
        h = W[((W.season == S - 1)) | ((W.season == S) & (W.week < Wk))]
        f = h.groupby('player_id')['fumbles_lost_total'].agg(['sum', 'size'])
        base = f['sum'].sum() / max(f['size'].sum(), 1)
        out[g.index] = g['player_id'].map((f['sum'] + 8 * base) / (f['size'] + 8)).fillna(base).values
    return out
d['fum_pg'] = fum_rate(d)
bg = cfg['big_games']
chance = lambda v, b: (1 / (1 + np.exp(-(b[0] + b[1] * np.log(v.clip(lower=1)))))).where(v.notna(), 0)
d['p400'] = chance(d['passing_yards_ewm_long'], bg['pass400']).where(d.position == 'QB', 0)
d['p150'] = chance(d['rushing_yards_ewm_long'], bg['rush150']).where(d.position.isin(['QB', 'RB']), 0)
d['p175'] = chance(d['receiving_yards_ewm_long'], bg['rec175']).where(d.position != 'QB', 0)
av = cfg['avail']
cat = np.where(d['ros_out_now'].eq(1) & d['ros_res'].eq(1), 'res', np.where(d['ros_out_now'].eq(1), 'out', 'active'))
d['a'] = [av[c].get(p, 0.85) for c, p in zip(cat, d['position'])]

def adj_pg(x, lg, qb):
    a = (lg.fum_lost + 2) * x['fum_pg'] + (lg.rec_pt - 1) * x['receptions_ewm_long'].fillna(0) \
        + lg.bonus_rush150 * x['p150'] + lg.bonus_rec175 * x['p175']
    q = (lg.pass_td - 4) * x['passing_tds_ewm_long'].fillna(0) + (lg.pass_int + 2) * x['passing_interceptions_ewm_long'].fillna(0) \
        + (1 / lg.pass_yds_per_pt - 1 / 25) * x['passing_yards_ewm_long'].fillna(0) + lg.bonus_pass400 * x['p400']
    return a + np.where(qb, q, 0)

def game_delta(lg):
    w = W
    q = (lg.pass_td - 4) * w.passing_tds + (lg.pass_int + 2) * w.passing_interceptions + (1 / lg.pass_yds_per_pt - 1 / 25) * w.passing_yards \
        + lg.bonus_pass400 * (w.passing_yards >= 400)
    return q + (lg.fum_lost + 2) * w.fumbles_lost_total + (lg.rec_pt - 1) * w.receptions \
        + lg.bonus_rush150 * (w.rushing_yards >= 150) + lg.bonus_rec175 * (w.receiving_yards >= 175)

def label(gap, T=1.0): return np.where(gap >= T, 'U', np.where(gap <= -T, 'O', 'A'))
rows = []
for _, lg in LG.iterrows():
    x = d.copy()
    # actual ROS points in league scoring = PPR ROS points + the league's per-game deltas over the same games
    gd = W[['season', 'week', 'player_id']].assign(delta=game_delta(lg).values)
    m = x[['season', 'week', 'player_id', 'first_wk']].reset_index().merge(gd.rename(columns={'week': 'gw'}), on=['season', 'player_id'])
    lwm = m['season'].map(R.last_week)
    m = m[(m.gw >= m.first_wk) & (m.gw <= lwm)]
    x['act_L'] = x['ros_pts'] + m.groupby('index')['delta'].sum().reindex(x.index).fillna(0)
    x['mod_L'] = x['model'] + adj_pg(x, lg, x.position == 'QB') * x['a'] * x['G']
    out = []
    for k, g in x.groupby(['season', 'week', 'position']):
        sm_ppr = np.sort(g['model'].values)[::-1]
        r = g[g['ros_ecr_pos_rank'].notna()].copy()
        if r.empty: continue
        r['mr'] = r['ros_ecr_pos_rank'].rank(method='first').astype(int)
        smL = np.sort(r['mod_L'].values)[::-1]; saL = np.sort(r['act_L'].values)[::-1]; sP = np.sort(r['model'].values)[::-1]
        r['gap_L'] = (r['mod_L'] - smL[r['mr'].values - 1]) / r['G']
        r['gap_P'] = (r['model'] - sP[r['mr'].values - 1]) / r['G']
        r['outc_L'] = (r['act_L'] - saL[r['mr'].values - 1]) / r['G']
        out.append(r)
    x = pd.concat(out)
    x = x[x['mr'] <= x['position'].map(POOLV)]
    for nm, ss in (('val', VAL), ('test', TEST)):
        g = x[x.season.isin(ss)]
        for meth in ('gap_P', 'gap_L'):
            lab = pd.Series(label(g[meth]), index=g.index)
            rho = pd.Series(np.select([lab == 'U', lab == 'O'], [1, -1], 0), index=g.index).corr(g['outc_L'], method='spearman')
            st = {k: (round(float((g.loc[lab == k, 'outc_L'] > 0).mean()), 3), round(float(g.loc[lab == k, 'outc_L'].mean()), 2), int((lab == k).sum())) for k in 'UAO'}
            rows.append({'league': lg.league, 'phase': nm, 'labels': 'PPR' if meth == 'gap_P' else 'league', 'rho': round(float(rho), 4),
                         'U beat/vs/n': st['U'], 'A': st['A'], 'O': st['O'],
                         'changed vs PPR %': round(float((label(g['gap_L']) != label(g['gap_P'])).mean() * 100), 1)})
    if lg.league == 'Ezekiel 25.17':
        x.to_parquet('ros_lab/league_value_rows_ez.parquet')
res = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(res.to_string(index=False))
res.to_csv('ros_lab/league_value_report.csv', index=False)

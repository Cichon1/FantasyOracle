"""Nov 24, 2026: how accurate are the K and DEF streaming ranks (automation/kdef_week.py -> docs/data/kdef_v5.csv)?
Weekly rank correlation with actual K / DEF points, 2023 - 2026 Week 11, vs FantasyPros weekly ECR (Thursday scrape,
fpdata/db_fpecr.parquet), Sleeper and ESPN projections (automation/history/other_proj), Vegas line alone and recent
average; same teams per week for every source; top-5 / top-12 = their actual points above the week average. Yahoo default
and BDSM League scoring. Run from model/:  python lab19_kdef_eval.py
Result (Yahoo): DEF ours 0.324, FP 0.296, Sleeper 0.304, ESPN 0.295, Vegas-only 0.328 (ours - FP +0.028 +/- 0.010,
39 of 57 weeks); K ours 0.137, FP 0.142, ESPN 0.141, Sleeper 0.099 (all near random). Blends add nothing.
Caveat: backtest uses closing Vegas lines, FP ranks are from Thursday."""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, '.')
import lab17_kdef as K, lab18_evo as EV
games = pd.read_parquet('data/games.parquet'); g = games[games.game_type == 'REG']
FPMAP = {'LAR': 'LA', 'NEP': 'NE', 'KCC': 'KC', 'SFO': 'SF', 'GBP': 'GB', 'TBB': 'TB', 'JAC': 'JAX', 'NOS': 'NO', 'LVR': 'LV'}
SLMAP = {'LAR': 'LA'}
# week dates per season
wk = g.assign(d=pd.to_datetime(g.gameday)).groupby(['season', 'week']).d.agg(['min', 'max']).reset_index()
def week_of(season_dates):
    pass
e = pd.read_parquet('fpdata/db_fpecr.parquet', columns=['page_type', 'tm', 'ecr', 'scrape_date'])
e = e[e.page_type.isin(['weekly-dst', 'weekly-k'])].copy()
e['d'] = pd.to_datetime(e.scrape_date); e['season'] = np.where(e.d.dt.month <= 2, e.d.dt.year - 1, e.d.dt.year)
def map_week(r):
    w = wk[(wk.season == r.season) & (wk['max'] >= r.d)]
    return w.week.min() if len(w) else np.nan
dd = e[['season', 'd']].drop_duplicates(); dd['week'] = dd.apply(map_week, axis=1)
e = e.merge(dd, on=['season', 'd']); e['team'] = e.tm.map(lambda t: FPMAP.get(t, t))
e = e.sort_values('d').drop_duplicates(['season', 'week', 'page_type', 'team'], keep='last')
fp = {u: e[e.page_type == pt].set_index(['season', 'week', 'team'])['ecr'] for u, pt in [('def', 'weekly-dst'), ('k', 'weekly-k')]}
other = EV.load_other_kdef()
oth = {}
for src, df in other.items():
    if df is None: continue
    df = df.copy(); df['team'] = df.team.map(lambda t: SLMAP.get(t, t)) if src == 'sleeper' else df.team
    for u, pos in [('def', 'DEF'), ('k', 'K')]:
        x = df[df.pos == pos].groupby(['season', 'week', 'team']).proj_ppr.max()
        oth[(src, u)] = x
for scoring in ['yahoo', 'bdsm']:
    U = K.build(2026, 'data', scoring)
    v = U[(U.season >= 2023) & U.played].copy()
    print(f'\n===== scoring: {scoring} =====')
    for u in ['def', 'k']:
        v['vegas'] = -v.opp_implied if u == 'def' else v.implied
        idx = pd.MultiIndex.from_arrays([v.season, v.week, v.team])
        v['fp'] = -pd.Series(fp[u].reindex(idx).values, index=v.index)
        for src in ['sleeper', 'espn']:
            v[src] = oth[(src, u)].reindex(idx).values if (src, u) in oth else np.nan
        srcs = {'Our model': f'{u}_proj', 'FantasyPros experts': 'fp', 'Sleeper': 'sleeper', 'ESPN': 'espn', 'Vegas line only': 'vegas', 'Recent average': f'{u}_avg'}
        res = []
        for (S, W), gw in v.groupby(['season', 'week']):
            row = {'season': S, 'week': W}
            ok = gw[list(srcs.values())].notna().all(axis=1)     # same teams for every source
            gg = gw[ok]
            if len(gg) < 20: continue
            mean = gg[f'{u}_pts'].mean()
            for name, c in srcs.items():
                row[name] = spearmanr(gg[c], gg[f'{u}_pts']).correlation
                row[name + ' top5'] = gg.nlargest(5, c)[f'{u}_pts'].mean() - mean
                row[name + ' top12'] = gg.nlargest(12, c)[f'{u}_pts'].mean() - mean
            res.append(row)
        r = pd.DataFrame(res)
        print(f'--- {u.upper()} ({len(r)} weeks with every source; per week ~{int(v.groupby(["season","week"]).size().mean())} teams)')
        tab = []
        for name in srcs:
            tab.append({'source': name, 'rank corr': r[name].mean(), 'se': r[name].std() / np.sqrt(len(r)),
                        **{f'{S}': r[r.season == S][name].mean() for S in sorted(r.season.unique())},
                        'top-5 pts above avg': r[name + ' top5'].mean(), 'top-12 above avg': r[name + ' top12'].mean()})
        print(pd.DataFrame(tab).round(3).to_string(index=False))
        d = r['Our model'] - r['FantasyPros experts']; print('ours - FP: %.3f +/- %.3f, weeks better %d / %d' % (d.mean(), d.std() / np.sqrt(len(d)), (d > 0).sum(), len(d)))
        r26 = r[r.season == 2026]
        print('2026 weeks:', len(r26), {n: round(r26[n].mean(), 3) for n in srcs}, '| top5:', {n: round(r26[n + ' top5'].mean(), 2) for n in ['Our model', 'FantasyPros experts', 'Sleeper', 'ESPN']})
        print('pts sd per team-week %.2f; average weekly corr between our model and FP %.2f' % (v[f'{u}_pts'].std(), np.nanmean([spearmanr(x[f'{u}_proj'], x['fp'], nan_policy='omit').correlation for _, x in v.groupby(['season','week'])])))

print('\n===== blends (yahoo scoring) =====')
U = K.build(2026, 'data', 'yahoo'); v = U[(U.season >= 2023) & U.played].copy()
for u in ['def', 'k']:
    idx = pd.MultiIndex.from_arrays([v.season, v.week, v.team])
    v['fp'] = -pd.Series(fp[u].reindex(idx).values, index=v.index)
    for src in ['sleeper', 'espn']: v[src] = oth[(src, u)].reindex(idx).values
    cols = [f'{u}_proj', 'fp', 'sleeper', 'espn']
    out = {}
    for (S, W), gw in v.groupby(['season', 'week']):
        gg = gw.dropna(subset=cols)
        if len(gg) < 20: continue
        z = {c: (gg[c] - gg[c].mean()) / gg[c].std() for c in cols}
        combos = {'ours': z[f'{u}_proj'], 'ours+FP': z[f'{u}_proj'] + z['fp'], 'all four': sum(z.values()), 'FP': z['fp'], 'FP+Sleeper+ESPN': z['fp'] + z['sleeper'] + z['espn']}
        for n, s in combos.items():
            out.setdefault(n, []).append((S, spearmanr(s, gg[f'{u}_pts']).correlation, gg.loc[s.nlargest(5).index, f'{u}_pts'].mean() - gg[f'{u}_pts'].mean()))
    print(u.upper(), {n: '%.3f (top5 %+.2f; 2026 %.3f)' % (np.mean([x[1] for x in l]), np.mean([x[2] for x in l]), np.mean([x[1] for x in l if x[0] == 2026])) for n, l in out.items()})
    b = np.array([x[1] for x in out['ours+FP']]) - np.array([x[1] for x in out['ours']]); print('  ours+FP - ours %.3f +/- %.3f' % (b.mean(), b.std() / np.sqrt(len(b))))
    b = np.array([x[1] for x in out['all four']]) - np.array([x[1] for x in out['ours']]); print('  all four - ours %.3f +/- %.3f' % (b.mean(), b.std() / np.sqrt(len(b))))

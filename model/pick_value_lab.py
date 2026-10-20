"""Next-season draft pick values for The League (Yahoo 200436): 12 teams, snake, 16 rounds, 1 keeper per team (keepers
take each team's round-16 pick), lineup QB, 2 RB, 3 WR, TE, W/R/T, K, DEF; 6-pt pass TD, -0.5 INT, 25 pass yds/pt,
1 per catch, +10 at 400 pass / 150 rush / 175 rec yards, -2 fumble lost.

Value of overall pick n = expected points above replacement over the next season of the player taken there, from
2020-2025: the preseason FantasyPros PPR draft consensus (last snapshot before September, a stand-in for draft
position) with the 12 keepers taken out of the pool (the pool loses the consensus ranks this league's 2026 keepers held:
1, 2, 3, 4, 7, 9, 10, 12, 23, 26, 28, 38 - each team keeps its best player, so next year's keepers should look alike), and each player's actual
season points in this league's scoring minus the replacement level at his position (best player outside the league's
starting lineups: QB13, RB/WR/TE after 24 RB, 36 WR, 12 TE and 12 flex). Kickers and defenses count as 0.
Check: curves from 2020-2022 vs 2023-2025."""
import json, numpy as np, pandas as pd

LG = dict(pass_td=6, pass_int=-0.5, pass_yds_per_pt=25, fum_lost=-2, rec_pt=1, bonus_pass400=10, bonus_rush150=10, bonus_rec175=10)
KEEPERS, TEAMS, ROUNDS = 12, 12, 16
KEEPER_RANKS = {1, 2, 3, 4, 7, 9, 10, 12, 23, 26, 28, 38}   # consensus ranks of The League's 2026 keepers
SEASONS = list(range(2020, 2026))


def season_points():
    W = pd.concat([pd.read_parquet(f'data/week_{s}.parquet') for s in SEASONS])
    W = W[(W.season_type == 'REG') & (W.week <= 17)].fillna(0)
    lg = LG
    delta = ((lg['pass_td'] - 4) * W.passing_tds + (lg['pass_int'] + 2) * W.passing_interceptions
             + (1 / lg['pass_yds_per_pt'] - 1 / 25) * W.passing_yards + lg['bonus_pass400'] * (W.passing_yards >= 400)
             + (lg['fum_lost'] + 2) * W.fumbles_lost_total + (lg['rec_pt'] - 1) * W.receptions
             + lg['bonus_rush150'] * (W.rushing_yards >= 150) + lg['bonus_rec175'] * (W.receiving_yards >= 175))
    W['pts'] = W.fantasy_points_ppr + delta
    return W.groupby(['season', 'player_id']).agg(pts=('pts', 'sum'), pos=('position', 'last')).reset_index()


def baselines(sp):
    """Replacement level per season and position: the best player left out of 12 teams' starting lineups."""
    out = {}
    for s, g in sp.groupby('season'):
        g = g[g.pos.isin(['QB', 'RB', 'WR', 'TE'])].sort_values('pts', ascending=False)
        start = {'QB': 12, 'RB': 24, 'WR': 36, 'TE': 12}
        taken = pd.concat([g[g.pos == p].head(n) for p, n in start.items()])
        rest = g.drop(taken.index)
        flex = rest[rest.pos.isin(['RB', 'WR', 'TE'])].head(12)
        rest = rest.drop(flex.index)
        out[s] = {p: float(rest[rest.pos == p].pts.max()) for p in start}
    return out


def preseason_ranks():
    d = pd.read_parquet('fpdata/db_fpecr.parquet', columns=['fp_page', 'ecr_type', 'scrape_date', 'player', 'id', 'pos', 'ecr'])
    d = d[d.fp_page.str.contains('ppr-cheatsheets', na=False)].copy()
    d['scrape_date'] = pd.to_datetime(d.scrape_date)
    ids = pd.read_csv('fpdata/db_playerids.csv', usecols=['fantasypros_id', 'gsis_id']).dropna()
    ids['fantasypros_id'] = ids.fantasypros_id.astype(int).astype(str)
    out = []
    for s in SEASONS + [2026]:
        x = d[(d.scrape_date.dt.year == s) & (d.scrape_date <= f'{s}-09-01')]
        x = x[x.scrape_date == x.scrape_date.max()].copy()
        x['pos'] = x.pos.str.replace(r'\d+', '', regex=True)
        x = x.sort_values('ecr').drop_duplicates('id')
        x['rank'] = np.arange(1, len(x) + 1)
        x['season'] = s
        x['id'] = x['id'].astype(str)
        out.append(x.merge(ids, left_on='id', right_on='fantasypros_id', how='left'))
    return pd.concat(out)


def curve(rows, n_picks=ROUNDS * TEAMS):
    """Average points above replacement at each consensus rank (pooled +-3 ranks, across seasons), forced to fall."""
    by = rows.groupby('rank')['var'].agg(['mean', 'size'])
    r = np.arange(1, n_picks + KEEPERS + 4)
    m = np.array([rows.loc[(rows['rank'] - k).abs() <= 3, 'var'].mean() for k in r])
    # isotonic (non-increasing) fit
    v = m.copy(); w = np.ones_like(v); blocks = [[v[0], w[0], 1]]
    for i in range(1, len(v)):
        blocks.append([v[i], w[i], 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            a, b = blocks.pop(), blocks.pop()
            blocks.append([(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]])
    iso = np.concatenate([[b[0]] * b[2] for b in blocks])
    return pd.Series(iso, index=r)


def main():
    sp = season_points()
    base = baselines(sp)
    pr = preseason_ranks()
    keep26 = ['Jahmyr Gibbs', 'Bijan Robinson', "Ja'Marr Chase", 'Jonathan Taylor', 'Justin Jefferson', 'CeeDee Lamb', 'Puka Nacua',
              'Derrick Henry', 'Ashton Jeanty', 'Malik Nabers', 'A.J. Brown', 'Josh Allen']
    k26 = pr[(pr.season == 2026) & pr.player.isin(keep26)][['player', 'rank']].sort_values('rank')
    rows = pr[pr.season.isin(SEASONS)].merge(sp, left_on=['season', 'gsis_id'], right_on=['season', 'player_id'], how='left', suffixes=('', '_s'))
    rows['pts'] = rows['pts'].fillna(0)
    rows['var'] = [max(0.0, p - base[s].get(pos, 1e9)) if pos in ('QB', 'RB', 'WR', 'TE') else 0.0
                   for p, s, pos in zip(rows.pts, rows.season, rows.pos)]
    full = curve(rows)
    a = curve(rows[rows.season <= 2022]); b = curve(rows[rows.season >= 2023])
    avail = [k for k in range(1, ROUNDS * TEAMS + KEEPERS + 4) if k not in KEEPER_RANKS]
    picks = []
    for n in range(1, ROUNDS * TEAMS + 1):
        rnd = (n - 1) // TEAMS + 1
        r = avail[n - 1]
        win = rows.loc[(rows['rank'] - r).abs() <= 3, 'var']
        picks.append(dict(overall=n, round=rnd, pick_in_round=(n - 1) % TEAMS + 1, consensus_rank=r,
                          value=round(float(full[r]), 1) if rnd < ROUNDS else 0.0,
                          low=round(float(win.quantile(0.1)), 1) if rnd < ROUNDS else 0.0,
                          high=round(float(win.quantile(0.9)), 1) if rnd < ROUNDS else 0.0,
                          hit_rate=round(float((win > 0).mean()), 3) if rnd < ROUNDS else 0.0,
                          value_2020_22=round(float(a[r]), 1), value_2023_25=round(float(b[r]), 1)))
    P = pd.DataFrame(picks)
    P.loc[P['round'] == ROUNDS, 'note'] = 'keeper round (each team keeps 1 player with its round-16 pick)'
    P.to_csv('dash_out/pick_values_v5.csv', index=False)
    rd = P[P['round'] < ROUNDS].groupby('round')[['value', 'low', 'high', 'hit_rate', 'value_2020_22', 'value_2023_25']].mean().round(1)
    print(rd.to_string())
    agree = P[P['round'] < ROUNDS]
    print('2026 keepers consensus ranks:', k26.values.tolist())
    print('baselines', {s: {p: round(v) for p, v in b.items()} for s, b in base.items()})
    print('half-curves mean abs diff (pts):', round(float((agree.value_2020_22 - agree.value_2023_25).abs().mean()), 1),
          'round-1 avg', round(agree[agree['round'] == 1].value_2020_22.mean(), 1), round(agree[agree['round'] == 1].value_2023_25.mean(), 1))
    print('matched season points:', round(float(rows.pts.gt(0).mean()), 3), 'players ranked', len(rows))


if __name__ == '__main__':
    main()

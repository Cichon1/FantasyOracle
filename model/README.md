# NFL player projection model (v5)

Projects each week's PPR points (with a 10th–90th percentile range), passing yards, rushing yards,
receiving yards and receptions for QBs, RBs, WRs and TEs, and keeps games already played on display with
actual stats next to the projection.

## Run it each week

```bash
pip install pandas pyarrow lightgbm catboost scipy
bash download_data.sh 2026        # nflverse (incl. weekly rosters, combine), expected points, FantasyPros history
# put this week's forecast in ref/weather_2026_wk6.csv (and update ref/playcallers.csv if a team changed play-callers)
FP_DATA=fpdata python weekly_v5.py --season 2026 --week 6 --config v5_config.json --out out_v5
python dash5_build.py 2026 6 <frame.parquet>     # dashboard data files
```

`weekly_v5.py` rebuilds every feature from the latest data, trains on every game before the target week
(about 10 minutes on 2 cores) and writes `out_v5/predictions_<season>_wk<week>.csv`: upcoming games projected,
games already played marked Final with actual stats. `--backfill 1 4` re-creates earlier weeks the same way
(each trained only on games before it).

Runs: **Thursday** before the Thursday game (no weekly expert ranking yet, midweek practice reports),
**Sunday morning** (final injury statuses, weekly expert ranking, roster moves), **Sunday after each kickoff
window's inactives post** (about 90 minutes before kickoff), **Tuesday** (final results).

### Game-day inactives (`inactives.py`, `cloud_run.py project --late`)

The scheduled Sunday runs read the FantasyPros inactives page into `ref/inactives_<season>_wk<week>.csv`
(columns `player,team,pos`). `cloud_run.py` matches the names to nflverse ids (exact name, first initial + last
name, then a unique same-team / same-position last name for nicknames), adds players nflverse's weekly roster
already lists as INA, and records them as ruled Out for the week. The projection run then drops them and
re-projects their teammates; on the dashboard they stay listed as **Inactive** at 0. `--late` re-projects only
games not yet played and keeps the stored pre-kickoff projections for games under way.

Backtest value (2023–2026 walk-forward, locked v5 predictions, inactive players set to 0): official weekly rank
correlation 0.432 → 0.450, R² 0.496 → 0.510 — larger than any modelling change tested.

The live scoreboard leaves out players ruled inactive on game day: the experts' consensus snapshot is taken
mid-week and can't know them, while the late runs do.

## Speed (Oct 19, 2026): same projections, faster runs

Measured on the 2-core cloud machine, Week 7 2026, Sunday-style run (`cloud_run.py project`, then `ros`):

| step | before | after | after, repeat run in the same folder |
|---|---|---|---|
| data download | 45 s | 20 s | 20 s |
| weekly projection (`project`) | 336 s | 261 s | 46 s |
| rest of season (`ros`) | 82 s | 51 s | 24 s |
| **total** | **~7.7 min** | **~5.5 min** | **~1.5 min** |

Every output file (projections, scoreboard, receivers, RB profiles, ROS, trade values) is byte-for-byte the same as
before the change (checked column by column; `check_fastops.py` compares feature frames).

- **Feature build 4-6x faster** (`fastops.py`): rolling / exponential averages per player were computed with a Python
  call per player (`groupby().transform(lambda ...)`, ~10,000 players x 400+ columns); pandas' grouped windows run the
  same arithmetic in one pass and give identical numbers. Also vectorised: each player's main team per season, team
  dropback counts by coverage look.
- **ROS snapshots 2x faster** (`ros.py`): the 110 training snapshots no longer re-filter the whole feature frame and the
  schedule for every snapshot.
- **Model cache** (`modelcache.py`): a trained model is saved under `cache/models/` keyed by a fingerprint of its exact
  training data, settings and seed, and reloaded when the same fit comes up again. The Thursday, Sunday-morning and
  Sunday-inactives runs of a week train on the same games, and the ROS rate models train on past seasons only (the same
  all season), so a run in a folder that already has the cache skips nearly all training. Anything different in the
  data means a new fingerprint and a fresh fit, so a cached model is never used for different data. `NFL_MODEL_CACHE=off`
  turns it off; the folder is trimmed to 600 MB (`NFL_MODEL_CACHE_MB`). Run info now reports seconds per stage and cache
  hits (`out_v5/run_info_*.json`, `dash_out/ros_info.json`).
- **Downloads** run 8 at a time (`download_data.sh`), same files.

What still takes the time on a fresh machine: fitting the models themselves (~3.5 of the 5.5 minutes; the 6-seed
ensemble, CatBoost and the range / yardage models). Making that faster would change the model (fewer seeds or rounds),
so it was left alone.

## v5.1 (Oct 12, 2026): QB market anchor

For projected starting QBs: ranking score = 30% model + 70% market projection (Vegas implied team points x the QB's
recent share of team skill-position fantasy points in his starts, `features_v6.build_market`), plus, for starters with
fewer than 4 prior starts, the out-of-sample under-projection for their bucket; the model's own QB point totals are
handed out in that order (`features_v6.qb_market_adjust`, `qb_market` in `v5_config.json`). Honest test: QB official
rank correlation +0.009 (validation) / +0.011 (test); all positions +0.002 / +0.003. Details: `lab6/coldstart.py`,
`lab6/market.py`, `lab6/reconcile.py`, `lab/experiments_v6.csv`.

## Rest of season (ROS) rankings and trade values (Oct 12, 2026; `ros.py`, `cloud_run.py ros`)

ROS PPR points from the earliest week W with an unplayed game through the last fantasy week (17):
**this week's official projection** (for a game not yet played) **+ games after week W x a per-game rate**, where the
rate is 0.8 x this week's projection + 0.2 x a separate **ROS rate model**; players out this week, on bye, or ruled
inactive use the ROS rate model alone. The ROS rate model has the weekly model's inputs (without the week-specific
expert rank) plus the remaining schedule (games and byes left, opponents' PPR allowed to the position as of W), the
experts' ROS rank and out/IR flags; target = PPR points over weeks W+1..17 per team game, zeros included for games
missed, so availability is priced in. Trained on every week-2..15 snapshot of 2018 through last season (extra-trees
LightGBM, 40 per leaf, 3 seeds; tree count early-stopped on last season). **Official ROS** = 50/50 rank average with
the FantasyPros ROS consensus within the position, model points handed out in that order. **Range**: 10th / 90th
percentile rate models, scaled per position on validation seasons (`ros.range_k` in `v5_config.json`).

Tuesday runs have no projection for the coming week yet, so `ros.weekly_proxy` makes a model-only one (extra-trees
member, no weekly expert rank, QB market step) - the setting the test calls "model-only".

Honest test (`ros_weekly_oos.py`, `ros_lab.py`): each of 2020-2025 projected at every week 3-14 by models trained only on
earlier seasons, weekly projections out of sample too; scored on the experts' ROS top QB24 / RB48 / WR60 / TE24 with
actual ROS points. Settings chosen on 2020-2022; 2023-2025 untouched:

| mean weekly rank corr. | validation 2020-22 | test 2023-25 |
|---|---|---|
| FantasyPros ROS consensus | 0.474 | 0.459 |
| model (weekly + ROS rate) | 0.478 | 0.467 |
| **official ROS** | **0.508** (+0.034) | **0.483** (+0.024 +/- 0.007; 11/12 season-positions ahead) |
| official ROS, model-only weekly projection | 0.509 | 0.489 |

Typical miss of official ROS points (all projected players, weeks 3-14, test): 22.0 PPR; 10% of players finish below
the low end and 12% above the high end. The experts' snapshot is their Friday ranking of week W (after Thursday's
game), so if anything they hold slightly more information.

**Trade values (PPR)** (revised Oct 12, 2026): for 10 / 12 / 14-team leagues, 1QB or superflex, value = the
average of two expected values over the player's ROS range (a two-piece normal through ros_low, ros_ppr, ros_high),
each counted only when positive: points above a **waiver-wire** pickup (per-game rate of the best player beyond typical
rostered depth: per team QB 2, RB 5, WR 5, TE 1.5; superflex QB 3, RB 4.5, WR 4.5, TE 1.5) and points above the
league's **last starter** (QB, 2 RB, 2 WR, TE, FLEX, plus SUPERFLEX filled with the best ROS players), both over his
remaining games. The first version (points above the last starter only) left most players at 0; this one gives every
rosterable player a value (12-team 1QB: first zero at QB34, RB88, WR142, TE42) while keeping a star premium.

Outputs: `dash_out/ros_v5.csv` (rankings), `dash_out/trade_values_v5.csv`, `dash_out/ros_info.json`.

The trade-values table is still written (datasets/trade_values) but the page no longer shows it: the trade analyzer
below replaced it.

## League trade analyzer (Oct 12, 2026; page scripts `taEngine`, `league_players`; `yahoo_league.py`)

Trades are judged inside each of Jackson's Yahoo leagues (BDSM League, The League, Ezekiel 25.17) the way FantasyPros
does it: each team's best lineup in the league's lineup slots (fixed slots first, then FLEX), filled from its players'
rest-of-season points per week, before and after the trade; the verdict is each side's % change and power-rank move.

**Points per week in league scoring**: official ROS points per game left (expected missed games count as zero) plus the
league's scoring changes vs standard PPR (4-pt pass TD, -2 INT, 25 pass yds/pt, -2 fumble lost, 1 per catch) from his
per-game rates (passing TDs / INTs / yards, fumbles lost, catches, chances of 400 / 150 / 175-yard games for bonuses)
x his chance to play a game (`ros.depth_inputs`: `play_rate`, `pts_when_playing`, `fum_pg`, `p_pass400` ...).
Return yards and other unmodeled settings are listed on the page. A bench-depth simulation and forced cut / IR / free
agent moves were tried the same evening and removed at Jackson's request (confusing).

**Value (undervalued / at value / overvalued)**, computed on the page per league (calc `league_players`): worth = our
model's ROS points (before the 50/50 blend with the experts) converted to the league's scoring; price = the same
league-scoring model points of the player at his market rank (same position), per game left. Undervalued at +1.0 or
more, overvalued at -1.0 or less, for the top QB32 / RB60 / WR80 / TE32. Two markets:
- **Experts** (FantasyPros ROS consensus rank) - the label shown. Tested walk-forward (`value_lab.py` in PPR,
  `league_value_lab.py` in each league's scoring; outcome = actual ROS points minus the actual points of whoever
  finished at his market rank):

| PPR labels | validation 2020-22: share | beat market rank | vs market (PPR/g) | test 2023-25: share | beat market rank | vs market (PPR/g) |
|---|---|---|---|---|---|---|
| Undervalued | 28% | 53% | +0.44 | 26% | 56% | +0.60 |
| At value | 41% | 41% | -0.70 | 46% | 41% | -0.78 |
| Overvalued | 31% | 28% | -2.72 | 29% | 30% | -2.42 |

  League-scoring labels change 2.5-5% of labels; label-vs-outcome correlation in league scoring: validation
  0.244-0.246 (PPR labels) vs 0.251-0.253 (league labels), test 0.236-0.238 vs 0.233-0.236 - no real difference, kept
  league labels (chosen on validation, consistent with league points everywhere else on the page). Within a position
  most of a league's scoring change cancels out between the player and the price.
- **Trade market** (FantasyPros Trade Market Values: redraft PPR value from real fantasy trades over the last 10 days;
  players with 20+ trades) - shown beside the label as "both agree" or what it says instead. Not tested yet: its
  history is saved with each pull (datasets/trade_market file + store collection market_history) so it can be.
  FantasyPros can't be reached from the cloud runs, so it's pulled in Chrome (`fp_trade_market.js`, no sign-in needed)
  by a daily scheduled task on Jackson's computer and whenever he asks; `trade_market.py` checks a pull.

**Draft picks (The League only; `pick_value_lab.py`, `yahoo_draft.py`, page script `taDraft`)**: next season's picks
can be traded (pick-for-pick only). Rules from Jackson: order follows this season's finish (last place first), snake,
16 rounds, 1 keeper per team using the round-16 pick. Pick value = expected points above replacement next season (this
league's scoring) of the player taken at that overall pick, 2020-2025: preseason FantasyPros PPR consensus (last
snapshot before September) as the draft order, pool minus keepers (the consensus ranks this league's 2026 keepers held:
1-4, 7, 9, 10, 12, 23, 26, 28, 38), replacement = best player outside 12 teams' starting lineups (QB13; after 24 RB,
36 WR, 12 TE, 12 flex), averaged over +-3 ranks, forced non-increasing. 1st round ~80-116 points, 2nd ~65, 9th ~13,
round 16 (keeper round) 0; curves from 2020-22 vs 2023-25 differ by 5.7 points per pick on average. The page simulates
the rest of the season 4,000 times (remaining Yahoo matchups; weekly score = team's best-lineup points per week + 22-pt
weekly noise + a 7-pt season-long strength error; the week under way from Yahoo's live projection; 6-team playoff, 1-2
byes) for each team's finish distribution -> each pick's slot and value, before and after a trade (same random draws),
plus playoff and title odds. This season = change in lineup points per week x weeks left; next season = change in the
draft value of picks owned; net = this season + weight x next season (weight chosen on the page, default 50%).

**Refresh**: projections and value labels refresh on the schedule; rosters and settings only when Jackson asks
("update my teams"): `yahoo_rosters.js` in his signed-in Chrome on Yahoo, then `python yahoo_league.py
yahoo/rosters_raw.txt dash_out/ros_v5.csv dash_out` -> `league_rosters_v5.csv`, `leagues_v5.csv`
(datasets/league_rosters, datasets/leagues).

## The model

- **Inputs**: the v4 variable set (≈437 inputs in 24 groups: form & Vegas, pace, coaching, defensive scheme
  and effectiveness, player vs coverage, offensive style, run quality, red zone, contract/draft/age, receiving
  usage, career history to 2010, expert consensus, Next Gen Stats, expected fantasy points, depth chart, final
  injury reports, pass protection, QB context, game context, play-caller, weather, snap share, injuries).
- **Models**: one per position group (QB, RB, WR/TE). Each is the equal-weight average of four members:
  extra-trees LightGBM with 10 and with 25 games per leaf (6 seeds each), standard LightGBM (6 seeds) and
  CatBoost (3 seeds). Tree counts come from early stopping on the latest complete season (`make_config.py`).
- **Official projection**: where this week's FantasyPros ranking exists, the model's rank and the experts'
  rank are averaged within the position, and the model's own point totals are handed out in that order
  (points stay on the model's scale, so flex comparisons work). Before the ranking posts (Thursday), the model
  alone, trained without the weekly-ranking inputs.
- **Range**: 10th / 90th percentile LightGBM models, calibrated (Oct 11): the high end is widened around the
  official projection (×1.16 QB/RB, ×1.14 WR, ×1.07 TE) and the QB low end lowered 0.3 points, which brings the
  walk-forward share of games beyond each end from about 12% to about 10% (`range_adjust` in `v5_config.json`).
  **Other stats**: one LightGBM model each.
- **Which model per game**: the weekly-ranking model for every game whose FantasyPros weekly snapshot exists
  (players it leaves unranked are deep players, as in training); the no-weekly-ranking model only for games before
  the snapshot (the Thursday run).

## Honest testing

`lab5.py` is the test harness: each of 2023–2026 is predicted by models trained only on earlier seasons
(early-stopped on the season before, then refit through it), scored on R² and head-to-head with the
FantasyPros consensus on the same players and weeks.

What makes the backtest honest (v5):

1. **As-available data**: participation charting (coverage shells, man/zone, personnel, pressure, routes run)
   is published only after a season, so during season S every feature built from it holds the end of season
   S-1 (`advanced_features.carry_prior_season`). FTN charting (box count, blitzers, play-action) is in-season.
2. **Expected-to-play pool**: games where a player was on the field but recorded no stat are zero-point games
   (snap counts), and players expected to play who sat (on the depth chart and a starter or played in one of
   the team's previous two games, not ruled Out, not on IR/reserve in that week's roster) count as zeros.
   The same rule picks the players projected each week.
3. **Settings chosen on validation seasons only**; a same-model/different-seeds run measures the noise floor
   (±0.001 R²) every change had to beat.
4. **Forward test**: v5 was locked on Oct 9, 2026 (before Week 5's Sunday games); every week from here on is a
   true out-of-sample test (scoreboard on the dashboard).

Remaining known limits: Vegas lines and weather in history are closing / actual values (removing Vegas lines
entirely costs only 0.004 R², so the effect is small); historical injury reports are final (Friday) reports,
so Thursday runs map a midweek DNP/limited to Questionable until the final report posts.

## Results (walk-forward 2023–2026, expected-to-play pool)

See `lab/results.log`, `lab/experiments_v5.csv` and `lab/final_summary.json`.

| | R² | Rank corr. vs actual (experts' pool) |
|---|---|---|
| v4 variables, honest test (3 seeds) | 0.497 | 0.412 |
| **v5** | **0.502** | **0.419** model, **0.421** official blend |
| FantasyPros experts | | 0.406 |

Same games, old vs honest test: v4 0.449 → 0.443 R² (2026, which had no participation data, unchanged).

What was tested and dropped (inside the noise floor or worse): removing the expert rank from the model with a
learned blend, trend features, second-half role, slot proxies, team-first volume, backup-QB interactions,
injury type, rookie combine profile, recency weighting, other tree sizes / learning rates.
What was kept: extra-trees LightGBM, more seeds, CatBoost in the average.

## Files

- `nfl_player_model.py` — data loading (incl. on-field zero-stat games), expected-to-play rows, the feature frame.
- `advanced_features.py`, `features_v4.py` — feature groups; `features_v5.py` — the v5 candidate groups (tested, not used).
- `lab5.py`, `exp_queue.py`, `ens_eval.py`, `val_scores.py`, `analyze_groups.py` — test harness and experiments.
- `make_config.py` → `v5_config.json`; `weekly_v5.py` — weekly projections; `final_eval.py` — locked-model test;
  `dash5_build.py` — dashboard data.
- `fastops.py` — fast grouped rolling / exponential averages; `modelcache.py` — trained-model cache;
  `check_fastops.py` — compares two feature frames column by column.
- `inactives.py` — game-day inactives; `cloud_run.py` — scheduled runs (project / results / ros, `--late`).
- `ros.py` — rest-of-season projections and trade values; `ros_weekly_oos.py`, `ros_lab.py` — their honest test;
  `build_frame_cache.py` — build and save the full feature frame once.
- `yahoo_rosters.js` (run in the Yahoo page) and `yahoo_league.py` — league rosters and settings for the trade analyzer.
- `value_lab.py` — walk-forward test of the undervalued / at value / overvalued labels.
- `pick_value_lab.py` — next-season draft pick values; `yahoo_picks.js` / `yahoo_draft.py` — The League standings, schedule and pick ownership.
- `league_value_lab.py` — the same test in each league's scoring; `fp_trade_market.js` / `trade_market.py` — FantasyPros trade market pull.
- `lab6/` — v6 experiments: `lab6.py` (walk-forward with any target / row filter / baseline, validation-first
  scoring incl. the players-who-suited-up pool), `inactive_value.py`, `segments.py`.
- `ref/playcallers.csv`, `ref/weather_<season>_wk<week>.csv`, `ref/inactives_<season>_wk<week>.csv`.

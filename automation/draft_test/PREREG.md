# Draft strategy test: pre-registered rules and bias audit (Nov 24, 2026)

These rules are written before any data is pulled or any result is seen. Changing one later has to be logged in
the "Changes" section at the bottom, with the reason and the date, and has to state whether results had been seen.

Scope: BDSM League (126912) and The League (200436). Ezekiel (IDP, 727092) is a later, separate test.

## 1. Hindsight (look-ahead) bias

| Risk | Rule | Automated check (`audit_draft_data.py`) |
|---|---|---|
| ADP collected after the season started (archives updated later; FFC's 2023 8-team page shows April 2024 drafts) | Use an ADP snapshot only if its collection window ends on or before that season's first regular-season game (kickoff dates from `games.parquet`). Otherwise drop that source-year and log it | FFC `meta.end_date` ≤ kickoff; any FP ADP page date found ≤ kickoff; unknown-date sources flagged |
| Expert rankings from after the draft | Use the last FantasyPros preseason consensus snapshot dated ≤ kickoff − 3 days (a typical late draft). Also run an "early draft" version using the last snapshot ≤ kickoff − 14 days | Snapshot `scrape_date` ≤ cutoff for every season |
| The players file shows current teams/positions | A player's team and role come from the Week 1 weekly roster and the Week 1 depth chart of that season, never from `players.parquet` | No feature reads current team; diff report vs Week 1 roster |
| Contracts are a current snapshot (later extensions and restructures) | Only contracts with `year_signed` < season, plus that season's signings marked free agent / rookie. In-season extensions count as unknown. Cap numbers from a later restructure aren't used | Every contract feature tagged with its year_signed; any `year_signed` ≥ season (except allowed types) fails |
| Trades during the season | Only trades dated before kickoff move a player's preseason team | Trade dates ≤ kickoff |
| Injuries during the preseason | Allowed: ADP and expert ranks already price them. Not allowed: any injury report from Week 1 on | Injury features only from data dated ≤ kickoff |
| Spotrac's past-year incentive pages also show how each player finished against his thresholds (e.g. "2024 Off Snaps: 99.3%") | Only the clause (stat, threshold, amount, contract year) is parsed for the board; the progress / result lines are dropped by the parser | Parser output has no result columns; checked |
| Incentive articles pick players who ended up near a threshold | The draft board may only use incentive clauses known at signing (Spotrac year pages, May free-agent lists). The late-season tracker articles are used only for the separate "incentive chasing" in-season test | Board features list has no tracker-derived column |
| Model training uses the test season | Walk-forward: the board for season S is trained only on seasons < S; tuning choices (features, settings) are fixed on 2012-2019 and frozen before the 2020-2025 test | Training-season max < S asserted on every fit |
| Replacement levels / pick values computed from the test season's actual results | Replacement levels and positional baselines for season S come from seasons < S | Asserted |
| Rookies | Draft position (NFL draft, April) and college/combine stats only; no rookie-season data | Asserted by date |
| The NFL draft-picks file carries each player's later career (games, starts, Pro Bowls, AV, career stats) | Only draft year, round, pick, team, position, college and age are allowed from it | Column allow-list enforced when loaded |
| PFR advanced stats are season totals | Season S's board uses only seasons < S | Asserted |

## 2. Selection and survivorship bias
- **Player pool:** every player with ADP, an expert rank, or a projection in any source that season, not only players who
  played. Players who never played that season score 0 (busts count).
- **Seasons:** all seasons with the needed data, decided before results: board 2016-2025 where ADP exists (FP ADP 2017+,
  FFC 2010+); expert comparison 2020-2025; strategy test 2018-2025 plus 2026 Weeks 1-11 shown separately.
- **Strategies:** the list is fixed now (section 5). Strategies added later are reported as added later.
- **Source availability changes by year** (FP's ADP sources differ: 2017 CBS/FFC/NFL/ESPN/Draft Wizard; 2022 ESPN/Yahoo/NFL/RTSports/Sleeper). The board's "consensus ADP" uses whatever existed that year, and each platform is also tested alone only in years it existed.

## 3. Measurement and scoring
- **League scoring exactly as each league:** BDSM (6-pt pass TD, -0.5 INT, -2 fumble lost, K 0.1 per FG yard, DEF return yards) and The League (+10 bonuses for 400 pass / 150 rush / 175 rec yards, 3 WR). Bonuses use weekly actuals, not season totals.
- **17-game seasons from 2021:** values are per team game and per counted week, so 16- and 17-game seasons compare.
- **Fantasy playoff weeks:** regular season Weeks 1-14, playoffs 15-17 for both leagues (BDSM 8 teams, The League 6 with two byes), as now. Earlier seasons are mapped to the same week numbers.
- **K and DEF** use the same scoring engine as the in-season tests (lab17), BDSM's own rules for BDSM.

## 4. Opponent and league simulation (the biggest source of skew)
- **Opponents draft like real Yahoo managers:** Yahoo ADP (FP's Yahoo column) plus noise drawn from each player's real
  ADP spread (FFC standard deviation, scaled to Yahoo), plus roster-need rules (no 3rd QB before round 12, K and DEF last
  three rounds, max roster counts). Calibrated before testing: simulated draft positions must match the real ADP spread
  by round (within 10%), and position counts by round must match FFC's real mock drafts.
- **No strategy drafts against copies of itself only:** each league seats 1 test strategy, 11 realistic opponents. A
  second design puts 2 test seats (different strategies) to check for interaction.
- **Same leagues for every strategy** (common random numbers): same opponents, same draft noise, same season draws. Only the test seat's strategy changes.
- **Draft slot:** The League tests every slot 1-12 (order follows finish). BDSM averages over random slots but reports each slot.
- **Keepers (The League):** 12 keepers removed before the draft, one per team, sampled so the ADP ranks of kept players match the league's real 2026 pattern (ranks 1-4, 7, 9, 10, 12, 23, 26, 28, 38). Each team's round-16 pick is the keeper's. A separate test compares keeping each candidate vs. the value of the round-16 pick.
- **In-season play is identical across strategies:** honest weekly lineups from projections available at the time (our walk-forward weekly model where it exists, FP weekly ranks otherwise), the same typical waiver and trade behavior for every seat. A second pass gives the test seat the tournament's best in-season strategy (testing whether the draft changes what in-season moves can do).
- **Waiver pool fairness:** undrafted players and in-season breakouts are available to every team by the same rules; free-agent quality isn't tuned per strategy.
- **Information timing in-season** (found in the Nov 23 audit): every manager uses the same freshness of information.

## 5. Fixed strategy list
Controls: (C1) Yahoo ADP drafter, (C2) FantasyPros expert-rank drafter, (C3) ADP with random noise (a "bad drafter").
Strategies, each run with every board (ours, FP ECR, Yahoo ADP, ESPN ADP, Sleeper ADP, FFC ADP, consensus ADP):
1. Best available by board value (VBD, points above replacement)
2. Zero RB (no RB in rounds 1-4/5)
3. Hero / Anchor RB (one RB in rounds 1-2, then none until round 6)
4. Robust RB (RBs in 3 of the first 4 rounds)
5. WR-heavy early (3+ WR in rounds 1-4)
6. Early elite QB (QB in rounds 1-3)
7. Late-round QB (no QB before round 8)
8. Elite TE (TE in rounds 1-3)
9. Late TE (no TE before round 10)
10. Tier-based: take the last player of a positional tier when the next tier drop is bigger than at other positions
11. Upside-late: from round 8, pick by 80th-percentile projection instead of mean
12. Handcuff: from round 10, draft your RB1's backup
13. Our board's "value vs. Yahoo ADP" picks: take players our board likes who are likely gone before your next pick, otherwise wait
Situations reported for each: draft slot, round, roster so far, tiers left by position, and position runs.

## 6. Statistics and over-fitting
- **The unit of independent evidence is the season** (8 full seasons), not the simulated league. Report season-level
  means and standard errors, and say how many seasons a result held in. Simulated leagues inside a season share that
  season's actual outcomes, so they aren't independent.
- **Many strategies x boards x slots = many comparisons.** Headline claims use a held-out check: rank strategies on
  2018-2021, confirm on 2022-2025. Anything that only wins in one half is called unproven.
- **Our board's accuracy** is compared with every other board on the same players and seasons, with paired differences and season-level SEs. "Better than any other board" is claimed only if it beats each one by 2+ SE and in most seasons.
- **No tuning on the test seasons:** board design and strategy parameters are frozen before the 2020-2025 test runs. A rerun after changes is labelled as such.

## 7. Data quality
- **Player ID matching** across FP, FFC, nflverse, Spotrac, contracts: match rates logged by source and season; unmatched top-200 ADP players must be under 2%, and each is listed.
- **Duplicate names and team changes** (e.g. two "Mike Williams"): match on name + position + team, then ID tables (`db_playerids`, nflverse players).
- **DEF and K naming** normalized to team abbreviations (the same maps as the K/DEF accuracy test).

## 8. Autosaves (so a lost run doesn't lose data)
- **Phase 1 pulls:** every unit (source x year x format) is written to its own file as soon as it's fetched (write to a temp file, then rename, so a crash never leaves half a file). A rerun skips units already saved. The workflow commits after each source and again in an `always()` step, so partial results are saved even if a step fails or times out.
- **Later phases** (board fits, simulations): results are saved per (league, season, strategy batch) as each finishes, with a progress file. A restart resumes from the last saved batch. GitHub Actions runs commit every few minutes and on failure.
- Licensed data (FantasyPros pages, Spotrac pages) is stored encrypted, like the existing FantasyPros rankings, because the repository is public. FFC data (free with attribution) and nflverse data are stored as is.

## Changes
- Nov 24 (before any results): FantasyPros' archived ADP pages return "report not available" to scripts, so per-platform ADP comes from the platforms themselves (Yahoo's API draft analysis, ESPN's public API) plus FFC; FP ADP dropped. Spotrac result lines excluded (row added above).
- Nov 24 (before any results): the hindsight audit's late-update test flagged ESPN's archived ADP. In 2020-2024 it
  ranks each season's actual points 0.11-0.24 better than FFC's dated preseason mock-draft ADP on the same players, and
  agrees with FFC only 0.70-0.86 (Yahoo: 0.86-0.94). ESPN's archive kept updating after the drafts, so **ESPN ADP is
  dropped for every year**. Yahoo's draft analysis passes (within +/-0.04 of FFC in 13 of 14 checked years) and is
  kept, except **Yahoo 2012** (+0.093, over the 0.05 flag line), dropped by the rule. Yahoo 2025 has no ADP values in the
  API response: to be checked before the board uses 2025.
- Multi-platform ADP for the test is therefore FFC (dated mock drafts; PPR / half / standard; 10 / 12 / 14 teams) +
  Yahoo (real Yahoo drafts, the leagues' own platform). Candidate third source with dated windows: MyFantasyLeague's
  public ADP export (to be pulled and audited the same way before use).
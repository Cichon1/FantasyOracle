"""Nov 18, 2026: how much is a gain of points worth to a team, by its record when the trade happens?

Clean version of the team-context question (the trade runs in lab15_trade.py are too noisy for it): replay leagues
with every team on FantasyPros (lab12_season.py), then for each team and trade week W add +K points to that team's
score in every game from week W on (a stand-in for a trade that adds K points per week) and recompute the standings,
playoffs (8 of 12) and title. The change per +1 point per week, by the team's record before week W.

  python lab15c_context.py FRAME.parquet        (CTX_LEAGUES default 300)  -> lab/trade_context.csv
"""
import os, sys, time
import numpy as np, pandas as pd
import lab5 as L, lab11_sim as S11, lab12_season as G

N = int(os.environ.get("CTX_LEAGUES", "300"))
K = 2.0
WEEKS = [4, 7, 10, 12]


def outcomes(wp, sched):
    wins, pf = np.zeros(G.TEAMS), np.zeros(G.TEAMS)
    for w in range(1, G.REG_WEEKS + 1):
        for a, b in sched[w - 1]:
            pf[a] += wp[a, w]; pf[b] += wp[b, w]
            if wp[a, w] > wp[b, w]: wins[a] += 1
            elif wp[b, w] > wp[a, w]: wins[b] += 1
            else: wins[a] += .5; wins[b] += .5
    seeds = list(np.lexsort((-pf, -wins))[:G.PO_TEAMS])
    alive = list(seeds)
    for w in range(G.REG_WEEKS + 1, G.LAST + 1):
        k = len(alive)
        alive = [alive[i] if wp[alive[i], w] >= wp[alive[k - 1 - i], w] else alive[k - 1 - i] for i in range(k // 2)]
    made = np.zeros(G.TEAMS); made[seeds] = 1
    title = np.zeros(G.TEAMS); title[alive[0]] = 1
    last = np.zeros(G.TEAMS); last[np.lexsort((pf, wins))[0]] = 1
    return wins, made, title, last


def main():
    frame = pd.read_parquet(sys.argv[1], columns=["season", "week", "season_type", "position", "expected",
                                                   "ecr_week_pos_rank", L.TARGET])
    cv = S11.curves(frame)
    d = S11.prepare(pd.read_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet")), cv)
    games = pd.read_parquet("data/games.parquet", columns=["season", "week", "game_type", "gameday", "home_team", "away_team"])
    ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], low_memory=False)
    recs = []
    for S in [2023, 2024, 2025]:
        board, _ = G.draft_board(S, games, ids)
        se = G.Season(d, S, board, games, 17)
        rng = np.random.default_rng(9000 + S)
        for lg in range(N):
            rb = np.random.default_rng(rng.integers(1 << 31))
            rosters, sched = G.draft(se, rb), G.schedule(rb)
            wp = G.play_season(se, cv[S - 1], rosters, sched, [], "control")["weekly_pts"]
            base = outcomes(wp, sched)
            for W in WEEKS:
                for T in range(G.TEAMS):
                    wb = sum(1.0 if wp[T, w] > wp[o, w] else (0.5 if wp[T, w] == wp[o, w] else 0.0)
                             for w in range(1, W) for a, b in sched[w - 1] if T in (a, b) for o in [b if a == T else a])
                    w2 = wp.copy(); w2[T, W:] += K
                    new = outcomes(w2, sched)
                    recs.append({"season": S, "league": lg, "week": W, "team": T, "wins_before": wb,
                                 "games_before": W - 1, "ppw_before": wp[T, 1:W].mean(),
                                 **{f"d_{n}": (new[i][T] - base[i][T]) / K for i, n in enumerate(["wins", "playoffs", "title", "last"])},
                                 **{f"base_{n}": base[i][T] for i, n in enumerate(["wins", "playoffs", "title", "last"])}})
        print(S, len(recs), flush=True)
    df = pd.DataFrame(recs)
    df["losses_before"] = df["games_before"] - df["wins_before"]
    df["record"] = df["wins_before"].astype(int).astype(str) + "-" + df["losses_before"].round().astype(int).astype(str)
    t = df.groupby(["week", "record"]).agg(teams=("team", "size"), final_wins=("base_wins", "mean"), playoff_odds=("base_playoffs", "mean"),
                                          title_odds=("base_title", "mean"), last_odds=("base_last", "mean"),
                                          wins_per_ppw=("d_wins", "mean"), playoff_per_ppw=("d_playoffs", "mean"),
                                          title_per_ppw=("d_title", "mean"), last_per_ppw=("d_last", "mean")).reset_index()
    t = t[t["teams"] >= 100]
    t.to_csv(os.path.join(L.LAB_DIR, "trade_context.csv"), index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Nov 18, 2026: 2-for-1 trades - does a value table that adds players up price consolidation right?

Same league sim as lab15_trade.py. At trade week W, team T gets one star X (one of the 36 most valuable players by
the official hub-style value who isn't on T) for two of its own players A + B: the pair whose combined official
hub-style value is closest to X's (each worth less than X). T fills the open roster spot with its best legal free
agent; X's team gets A and B (and cuts its least valuable legal player). Replayed vs the same league without the
trade.

If the formulas priced 2-for-1 deals right, trades they call even would gain T nothing on average. A positive average
= a consolidation premium the formula misses (the star is worth more than the sum). Compared with 1-for-1 trades
from lab15_trade.py (lab/trade_sim_runs.parquet) through the same regression: realized points gained ~ formula
difference; the intercept is the formula's bias for that kind of trade.

  python lab15b_2for1.py FRAME.parquet       (TRADE_LEAGUES default 20) -> lab/trade_2for1_runs.parquet, summary json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import lab5 as L
import lab11_sim as S11
import lab12_season as G
import lab15_trade as T15

N_LEAGUES = int(os.environ.get("TRADE_LEAGUES", "20"))
N_STARS = 36
NO_FILL = os.environ.get("TRADE_NO_FILL", "0") == "1"   # leave the freed roster spot empty
TAG = "_nofill" if NO_FILL else ""


def main():
    t0 = time.time()
    frame = pd.read_parquet(sys.argv[1], columns=["season", "week", "season_type", "position", "expected",
                                                   "ecr_week_pos_rank", L.TARGET])
    rows = pd.read_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet"))
    cv = S11.curves(frame)
    d = S11.prepare(rows, cv)
    games = pd.read_parquet("data/games.parquet", columns=["season", "week", "game_type", "gameday", "home_team",
                                                           "away_team"])
    ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], low_memory=False)
    recs = []
    for S in [2023, 2024, 2025]:
        board, _ = G.draft_board(S, games, ids)
        se = G.Season(d, S, board, games, 17)
        curve = cv[S - 1]
        rng = np.random.default_rng(7000 + S)
        F = {W: T15.formulas(se, W, curve) for W in T15.WEEKS}
        for lg in range(N_LEAGUES):
            rb = np.random.default_rng(rng.integers(1 << 31))
            rosters = G.draft(se, rb)
            sched = G.schedule(rb)
            snaps = {W: None for W in T15.WEEKS}
            base = G.play_season(se, curve, rosters, sched, [], "control", snap=snaps)
            for W in T15.WEEKS:
                T = int(rb.integers(G.TEAMS))
                mine = snaps[W][T]
                hub = F[W]["official_hub"]
                others = np.array([j for j in range(len(se.pid)) if j not in mine and se.pos[j] >= 0
                                   and se.games_left[j, W] > 0])
                stars = others[np.argsort(-hub[others])][:N_STARS]
                for x in stars:
                    pool = [j for j in mine if 0 < hub[j] < hub[x]]
                    best, pair = np.inf, None
                    for i in range(len(pool)):
                        for k in range(i + 1, len(pool)):
                            gap = abs(hub[pool[i]] + hub[pool[k]] - hub[x])
                            if gap < best:
                                best, pair = gap, (pool[i], pool[k])
                    if pair is None:
                        continue
                    tr = {"week": W, "team": T, "add": [int(x)], "drop": [int(pair[0]), int(pair[1])], "no_fill": NO_FILL}
                    r = G.play_season(se, curve, rosters, sched, [], "control", trade=tr)
                    rec = {"season": S, "league": lg, "week": W, "team": T, "star": se.name[x],
                           "gave": f"{se.name[pair[0]]} + {se.name[pair[1]]}", "kind": "2-for-1"}
                    for k in ["wins", "playoffs", "title", "last"]:
                        rec[f"d_{k}"] = float(r[k][T]) - float(base[k][T])
                    rec["d_pts"] = float(r["weekly_pts"][T, W:].sum() - base["weekly_pts"][T, W:].sum())
                    for f, v in F[W].items():
                        rec[f] = float(v[x]) - float(v[pair[0]]) - float(v[pair[1]])
                    recs.append(rec)
            print(S, lg, len(recs), round((time.time() - t0) / 60, 1), "min", flush=True)
    df = pd.DataFrame(recs)
    df.to_parquet(os.path.join(L.LAB_DIR, f"trade_2for1_runs{TAG}.parquet"))
    one = pd.read_parquet(os.path.join(L.LAB_DIR, "trade_sim_runs.parquet")).assign(kind="1-for-1")
    out = {"runs_2for1": len(df), "runs_1for1": len(one), "bias": {}}
    for f in ["official_hub", "official_starter", "official_waiver", "fp_hub", "fp_starter", "model_hub",
              "cons_free_hub", "cons_free_starter"]:
        r = {}
        for kind, g in [("1-for-1", one), ("2-for-1", df)]:
            b, a = np.polyfit(g[f], g["d_pts"], 1)
            # points the trade gains when the formula calls it even (formula difference 0), and per formula point
            near = g[g[f].abs() <= 10]
            r[kind] = {"slope": round(float(b), 3), "pts_gained_at_even": round(float(a), 2),
                       "near_even_trades": int(len(near)),
                       "near_even_mean_pts": round(float(near["d_pts"].mean()), 2) if len(near) else None,
                       "near_even_mean_wins": round(float(near["d_wins"].mean()), 3) if len(near) else None,
                       "near_even_mean_playoffs": round(float(near["d_playoffs"].mean()), 4) if len(near) else None}
        out["bias"][f] = r
    json.dump(out, open(os.path.join(L.LAB_DIR, f"trade_2for1_summary{TAG}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

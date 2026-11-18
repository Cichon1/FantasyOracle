"""Nov 17, 2026 league sim, draft stage, step 1: the model's preseason projections, walk-forward.

For each season S (2023-2026), a season-long PPR model trained only on seasons 2012..S-1 projects every candidate's
weeks 1-17 PPR total (zeros for games missed, so availability is priced in). Inputs, all known before the season:
the player's last two seasons (points, games, per-game rates, target / carry shares, receptions, pass attempts), a
recency-weighted career line, age, years in the league, draft pick (rookies: their draft pick), position and whether
he changed teams. No FantasyPros inputs: this is the model's own board. (FantasyPros' preseason ranks only go back
to 2021, too short to train on.)

Candidates: players with 20+ PPR points the season before, or rookies drafted in the first 200 picks.

  python lab13_preseason.py        -> lab/preseason_proj.parquet, prints rank correlation vs FantasyPros' board
"""
import glob
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lab5 as L

DATA = "data"
POS = ["QB", "RB", "WR", "TE"]
SEEDS = (7, 23, 41)
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
ROUNDS = 600


def seasons_table():
    w = pd.concat([pd.read_parquet(f, columns=["player_id", "player_display_name", "position", "season", "week",
                                               "season_type", "team", "fantasy_points_ppr", "targets", "carries",
                                               "receptions", "attempts", "receiving_yards", "rushing_yards"])
                   for f in sorted(glob.glob(f"{DATA}/week_*.parquet"))])
    w = w[(w["season_type"] == "REG") & (w["week"] <= 17)]
    tm = w.groupby(["season", "team"])[["targets", "carries"]].sum().rename(columns=lambda c: "team_" + c)
    w = w.join(tm, on=["season", "team"])
    g = w.groupby(["player_id", "season"])
    s = g.agg(name=("player_display_name", "last"), pos=("position", "last"), team=("team", "last"),
              first_team=("team", "first"), ppr=("fantasy_points_ppr", "sum"), games=("week", "nunique"),
              tgt=("targets", "sum"), car=("carries", "sum"), rec=("receptions", "sum"), att=("attempts", "sum"),
              team_tgt=("team_targets", "mean"), team_car=("team_carries", "mean"))
    s["tgt_share"] = s["tgt"] / s["games"] / (s["team_tgt"] / 17)
    s["car_share"] = s["car"] / s["games"] / (s["team_car"] / 17)
    return s.reset_index()


def features(s, players, S):
    """Rows for season S: candidates and their pre-season inputs."""
    prev = s[s["season"] == S - 1].set_index("player_id")
    prev2 = s[s["season"] == S - 2].set_index("player_id")
    hist = s[s["season"] < S]
    wts = 0.6 ** (S - 1 - hist["season"])
    car = hist.assign(w=wts, wppr=hist["ppr"] * wts, wg=hist["games"] * wts).groupby("player_id")[["w", "wppr", "wg"]].sum()
    pl = players.set_index("gsis_id")
    rook = pl[(pl["rookie_season"] == S) & pl["position"].isin(POS) & (pl["draft_pick"].fillna(999) <= 200)].index
    cand = list(dict.fromkeys(list(prev[prev["ppr"] >= 20].index) + list(rook)))
    cur = s[s["season"] == S].set_index("player_id")
    r = pd.DataFrame(index=pd.Index(cand, name="player_id"))
    r["season"] = S
    r["pos"] = [prev["pos"].get(p) if p in prev.index else pl["position"].get(p) for p in cand]
    r = r[r["pos"].isin(POS)]
    for k in ["ppr", "games", "tgt_share", "car_share", "rec", "att", "tgt", "car"]:
        r[f"l1_{k}"] = prev[k].reindex(r.index)
        r[f"l2_{k}"] = prev2[k].reindex(r.index)
    r["l1_ppg"] = r["l1_ppr"] / r["l1_games"]
    r["l2_ppg"] = r["l2_ppr"] / r["l2_games"]
    c = car.reindex(r.index)
    r["career_ppg_wtd"] = c["wppr"] / c["wg"]
    r["career_games_wtd"] = c["wg"]
    bd = pd.to_datetime(pl["birth_date"].reindex(r.index), errors="coerce")
    r["age"] = (pd.Timestamp(f"{S}-09-01") - bd).dt.days / 365.25
    r["yrs"] = S - pl["rookie_season"].reindex(r.index)
    r["draft_pick"] = pl["draft_pick"].reindex(r.index).fillna(300)
    r["is_rookie"] = (r["yrs"] == 0).astype(float)
    # team for S: the team of his first game in S if he played (known after free agency / the draft), else last team
    team_s = cur["first_team"].reindex(r.index)
    r["changed_team"] = (team_s.notna() & prev["team"].reindex(r.index).notna() &
                         (team_s != prev["team"].reindex(r.index))).astype(float)
    r["pos_code"] = r["pos"].map({p: i for i, p in enumerate(POS)})
    r["target"] = cur["ppr"].reindex(r.index).fillna(0.0)
    r["name"] = [prev["name"].get(p) if p in prev.index else pl["display_name"].get(p) for p in r.index]
    return r.reset_index()


X = ["pos_code", "age", "yrs", "draft_pick", "is_rookie", "changed_team", "career_ppg_wtd", "career_games_wtd",
     "l1_ppg", "l2_ppg"] + [f"l{i}_{k}" for i in (1, 2) for k in ["ppr", "games", "tgt_share", "car_share", "rec", "att",
                                                                  "tgt", "car"]]


def main():
    s = seasons_table()
    players = pd.read_parquet(f"{DATA}/players.parquet")
    allrows = pd.concat([features(s, players, S) for S in range(2012, 2027)], ignore_index=True)
    out = []
    for S in [2023, 2024, 2025, 2026]:
        tr, te = allrows[allrows["season"] < S], allrows[allrows["season"] == S].copy()
        pred = np.zeros(len(te))
        for sd in SEEDS:
            b = lgb.train(dict(PARAMS, seed=sd), lgb.Dataset(tr[X].astype(np.float32), tr["target"]), ROUNDS)
            pred += b.predict(te[X].astype(np.float32)) / len(SEEDS)
        te["model_total"] = pred
        out.append(te)
    o = pd.concat(out, ignore_index=True)
    o.to_parquet(os.path.join(L.LAB_DIR, "preseason_proj.parquet"))
    # how good is the board? rank correlation with the actual season, within position, vs FantasyPros' draft ranks
    import lab12_season as G
    games = pd.read_parquet(f"{DATA}/games.parquet", columns=["season", "week", "game_type", "gameday", "home_team",
                                                              "away_team"])
    ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], low_memory=False)
    top = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
    for S in [2023, 2024, 2025]:
        b, _ = G.draft_board(S, games, ids)
        d = o[o["season"] == S].merge(b[["player_id", "ecr"]], on="player_id", how="inner")
        res = {}
        for p, g in d.groupby("pos"):
            g = g.nsmallest(top[p], "ecr")
            res[p] = (round(spearmanr(g["model_total"], g["target"]).correlation, 3),
                      round(spearmanr(-g["ecr"], g["target"]).correlation, 3))
        print(S, "rank corr with actual season (model, FantasyPros) on FP's top players:", res, flush=True)


if __name__ == "__main__":
    main()

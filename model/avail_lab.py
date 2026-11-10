"""Availability lab (Nov 9, 2026): does a player-specific chance to play beat the position x weeks-ahead curve?

Sample: every (season, snapshot week W = 3..14) for players who played 50%+ of their team's offensive snaps (QB 80%+)
in its last game before W; target = played (1+ offensive snap) in each of his team's games from W through Week 17.
Inputs (all known before week W): position, games ahead k, age, share of team games missed this season and over the
two seasons before, his snap share in the last game, and his injury-report status for week W (final report) and the
week before. Train 2021-2023, test 2024-2025 (log loss and Brier score per game; also on games 1+ weeks out only).

  python avail_lab.py data
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from avail import design

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
POS = ["QB", "RB", "WR", "TE"]

sn = pd.concat([pd.read_parquet(f, columns=["season", "week", "game_type", "team", "pfr_player_id", "position",
                                            "offense_snaps", "offense_pct"])
                for f in sorted(glob.glob(os.path.join(DATA, "snaps", "snaps_*.parquet")))], ignore_index=True)
sn = sn[(sn["game_type"] == "REG") & sn["position"].isin(POS)]
pl = pd.read_parquet(os.path.join(DATA, "players.parquet"), columns=["pfr_id", "gsis_id", "birth_date"]).dropna(
    subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
inj = pd.concat([pd.read_parquet(f, columns=["season", "week", "gsis_id", "report_status", "practice_status"])
                 for f in sorted(glob.glob(os.path.join(DATA, "inj", "inj_*.parquet")))], ignore_index=True)
inj = inj.drop_duplicates(["season", "week", "gsis_id"], keep="last").set_index(["season", "week", "gsis_id"])
tg = sn[["season", "week", "team"]].drop_duplicates()
played = sn[sn["offense_snaps"] > 0]
pset = set(zip(played["pfr_player_id"], played["season"], played["week"]))


def lw(season):
    return 17 if season >= 2021 else 16


rows = []
for S in range(2021, int(sn["season"].max()) + 1):
    tws = {t: sorted(g["week"]) for t, g in tg[tg["season"] == S].groupby("team")}
    ps = played[played["season"] == S]
    prev2 = played[played["season"].isin([S - 1, S - 2])].groupby("pfr_player_id")["week"].size()
    team_g2 = tg[tg["season"].isin([S - 1, S - 2])].groupby("team").size()
    for W in range(3, 15):
        before = ps[ps["week"] < W].sort_values("week")
        last = before.groupby("pfr_player_id").tail(1)
        for _, r in last.iterrows():
            t, pid, pos = r["team"], r["pfr_player_id"], r["position"]
            tw = [w for w in tws.get(t, []) if w < W]
            if not tw or r["week"] != tw[-1]:
                continue
            if r["offense_pct"] < (0.8 if pos == "QB" else 0.5):
                continue
            mine = before[(before["pfr_player_id"] == pid) & (before["team"] == t)]
            first = mine["week"].min()
            span = [w for w in tw if w >= first]
            cur_miss = 1 - len(mine) / max(len(span), 1)
            prior = prev2.get(pid, 0) / max(team_g2.mean(), 1)
            gs = pl["gsis_id"].get(pid) if pid in pl.index else None
            bd = pl["birth_date"].get(pid) if pid in pl.index else None
            age = (pd.Timestamp(f"{S}-09-01") - pd.to_datetime(bd)).days / 365.25 if bd is not None and pd.notna(bd) else np.nan
            rep = inj["report_status"].get((S, W, gs)) if gs else None
            prac = inj["practice_status"].get((S, W, gs)) if gs else None
            rep1 = inj["report_status"].get((S, W - 1, gs)) if gs else None
            rem = [w for w in tws[t] if W <= w <= lw(S)]
            for k, w in enumerate(rem):
                rows.append(dict(season=S, W=W, pos=pos, k=k, y=int((pid, S, w) in pset), age=age, cur_miss=cur_miss,
                                 prior_games=prior, snap_last=r["offense_pct"],
                                 q_now=int(rep == "Questionable"), d_now=int(rep in ("Doubtful", "Out")),
                                 lim_now=int(isinstance(prac, str) and prac.startswith("Limited")),
                                 dnp_now=int(isinstance(prac, str) and prac.startswith("Did Not")),
                                 rep_prev=int(isinstance(rep1, str) and rep1 != "")))
d = pd.DataFrame(rows)
print("rows", len(d), "players-snapshots", len(d[d.k == 0]))
d["age"] = d["age"].fillna(d["age"].median())
d["prior_games"] = d["prior_games"].clip(0, 1.2)

trn, tst = d[d["season"] <= 2023], d[d["season"] >= 2024]
# baseline: the position x games-ahead curve from the training seasons
curve = trn.groupby(["pos", "k"])["y"].mean()
base = tst.apply(lambda r: curve.get((r["pos"], min(r["k"], trn["k"].max())), trn["y"].mean()), axis=1).clip(0.01, 0.99)


trn = trn.assign(curve=trn.apply(lambda r: curve.get((r["pos"], r["k"]), 0.85), axis=1))
tst = tst.assign(curve=base.values)
lr = LogisticRegression(C=1.0, max_iter=2000).fit(design(trn), trn["y"])
p = lr.predict_proba(design(tst))[:, 1]
coef = pd.Series(lr.coef_[0], index=design(trn).columns).round(3)
print(coef.to_string())
for name, mask in [("all games", tst["k"] >= 0), ("next game", tst["k"] == 0), ("1+ weeks out", tst["k"] >= 1),
                   ("4+ weeks out", tst["k"] >= 4)]:
    y = tst.loc[mask, "y"]
    print(f"{name:13s} n={mask.sum():6d}  log loss curve {log_loss(y, base[mask]):.4f}  player {log_loss(y, p[mask.values]):.4f}"
          f"   Brier curve {brier_score_loss(y, base[mask]):.4f}  player {brier_score_loss(y, p[mask.values]):.4f}")
tst = tst.assign(p=p)
s = tst[tst["k"] >= 1].groupby(["season", "W", "pos"]).size()
spread = tst[tst["k"] >= 1].groupby("pos")["p"].describe(percentiles=[0.1, 0.5, 0.9])[["10%", "50%", "90%"]].round(3)
print("player-specific chance, games 1+ weeks out (test):\n", spread.to_string())
# calibration of the player model by decile
tst["bin"] = pd.qcut(tst["p"], 10, duplicates="drop")
print(tst.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).to_string())

# production fit: every season (2021-2025), curve from the same seasons; saved to lab/avail_coef.json for v5_config
import json
full_curve = d.groupby(["pos", "k"])["y"].mean()
d = d.assign(curve=d.apply(lambda r: full_curve.get((r["pos"], r["k"]), 0.85), axis=1))
lr2 = LogisticRegression(C=1.0, max_iter=2000).fit(design(d), d["y"])
coef = {"intercept": round(float(lr2.intercept_[0]), 4),
        "coef": {c: round(float(v), 4) for c, v in zip(design(d).columns, lr2.coef_[0])},
        "curve": {p: [round(float(full_curve.get((p, k), np.nan)), 3) for k in range(0, 14)] for p in POS}}
json.dump(coef, open(os.path.join("lab", "avail_coef.json"), "w"), indent=1)
print(json.dumps(coef))

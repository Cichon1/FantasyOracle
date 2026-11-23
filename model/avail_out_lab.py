"""Return-timing lab (Nov 23, 2026): when does a player who is out come back?

Jackson's ask: a star out for 2-3 weeks (Puka Nacua, Mike Evans) shouldn't look like a flat ~60% chance every week;
he's mostly out now and mostly back later, which matters for the fantasy playoffs. The ROS engine shapes an out
player's remaining games with one position curve (v5_config ros.avail_curve.out: everyone out, short and long
absences mixed, so it's nearly flat) and scales it to his ROS total.

Sample: every (season, snapshot week W = 3..14) for players who played 50%+ of their team's offensive snaps (QB 80%+)
in one of its 6 games before W but not in its last one (the 'out' definition of avail_curve), not on IR / reserve
(nflverse weekly roster status RES) at W. Inputs known at W: injury-report status and practice participation for W,
how many straight team games he has missed, the body part, position, games ahead k. Target: played (1+ offensive snap)
in each of his team's games after W, through Week 17 (k = 1..6). Logistic regression with k-specific base rates and the
inputs' effects split by games ahead (k = 1, k = 2, k >= 3). Train 2018-2021, validate 2022-2023, test 2024-2025.

  python avail_out_lab.py data            -> prints the comparison, writes lab/avail_out_coef.json (fit on 2018-2025)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
from avail_out import BODY, FEATS, K, POS, body_group, design, features, predict


def build(data):
    sn = pd.concat([pd.read_parquet(f, columns=["season", "week", "game_type", "team", "pfr_player_id", "position",
                                                "offense_snaps", "offense_pct"])
                    for f in sorted(glob.glob(os.path.join(data, "snaps", "snaps_*.parquet")))], ignore_index=True)
    sn = sn[(sn["game_type"] == "REG") & sn["position"].isin(POS) & (sn["season"] >= 2018)]
    pl = pd.read_parquet(os.path.join(data, "players.parquet"), columns=["pfr_id", "gsis_id"]).dropna().drop_duplicates("pfr_id")
    p2g = pl.set_index("pfr_id")["gsis_id"]
    inj = pd.concat([pd.read_parquet(f, columns=["season", "week", "gsis_id", "report_status", "practice_status",
                                                 "report_primary_injury", "practice_primary_injury"])
                     for f in sorted(glob.glob(os.path.join(data, "inj", "inj_*.parquet")))], ignore_index=True)
    inj["season"] = inj["season"].astype(int); inj["week"] = inj["week"].astype(int)
    inj = inj.drop_duplicates(["season", "week", "gsis_id"], keep="last").set_index(["season", "week", "gsis_id"])
    ro = pd.concat([pd.read_parquet(f, columns=["season", "week", "gsis_id", "status"])
                    for f in sorted(glob.glob(os.path.join(data, "roster", "roster_weekly_*.parquet")))], ignore_index=True)
    ro = ro.dropna(subset=["gsis_id"]).drop_duplicates(["season", "week", "gsis_id"], keep="last").set_index(["season", "week", "gsis_id"])["status"]
    tg = sn[["season", "week", "team"]].drop_duplicates()
    team_weeks = {(s, t): sorted(g["week"]) for (s, t), g in tg.groupby(["season", "team"])}
    played = sn[sn["offense_snaps"] > 0]
    pset = set(zip(played["pfr_player_id"], played["season"], played["week"]))
    big = sn[sn["offense_pct"] >= np.where(sn["position"] == "QB", 0.8, 0.5)]
    rows = []
    for (S, t), wks in team_weeks.items():
        lw = 17 if S >= 2021 else 16
        bt = big[(big["season"] == S) & (big["team"] == t)]
        for W in range(3, 15):
            prev = [w for w in wks if w < W][-6:]
            if len(prev) < 3:
                continue
            after = [w for w in wks if W < w <= lw][:K]
            if not after:
                continue
            cands = bt[bt["week"].isin(prev)][["pfr_player_id", "position"]].drop_duplicates("pfr_player_id")
            for pid, pos in zip(cands["pfr_player_id"], cands["position"]):
                if (pid, S, prev[-1]) in pset:
                    continue          # played the team's last game: not out
                g = p2g.get(pid)
                if g is None or ro.get((S, W, g)) == "RES":
                    continue          # IR / reserve: the 'res' category
                streak = 0
                for w in reversed([w for w in wks if w < W]):
                    if (pid, S, w) in pset:
                        break
                    streak += 1
                r = inj.loc[(S, W, g)] if (S, W, g) in inj.index else None
                rep = str(r["report_status"]) if r is not None and isinstance(r["report_status"], str) else ""
                pr = str(r["practice_status"]) if r is not None and isinstance(r["practice_status"], str) else ""
                what = (r["report_primary_injury"] if r is not None and isinstance(r["report_primary_injury"], str)
                        else (r["practice_primary_injury"] if r is not None and isinstance(r["practice_primary_injury"], str) else ""))
                if not what:            # last report that named the injury, this absence
                    for w in range(W - 1, max(W - 6, 0), -1):
                        if (S, w, g) in inj.index:
                            q = inj.loc[(S, w, g)]
                            what = q["report_primary_injury"] if isinstance(q["report_primary_injury"], str) else (
                                q["practice_primary_injury"] if isinstance(q["practice_primary_injury"], str) else "")
                            if what:
                                break
                base = dict(season=S, week=W, pid=pid, gsis=g, position=pos, streak=streak, rep=rep, prac=pr,
                            body=body_group(what))
                for k, w in enumerate(after, start=1):
                    rows.append({**base, "k": k, "played": int((pid, S, w) in pset)})
    d = pd.DataFrame(rows)
    return features(d)


def fit(d, C=1.0):
    X = design(d)
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(X, d["played"])
    return {"intercept": 0.0, "coef": dict(zip(X.columns, np.round(m.coef_[0], 4)))}


def main():
    cache = os.path.join("lab", "avail_out_rows.parquet")
    d = pd.read_parquet(cache) if os.path.exists(cache) else build(DATA)
    os.makedirs("lab", exist_ok=True); d.to_parquet(cache)
    cfg = json.load(open("v5_config.json"))["ros"]["avail_curve"]["out"]
    d["pop"] = [cfg[p][min(k - 1, len(cfg[p]) - 1)] for p, k in zip(d["position"], d["k"])]
    print(f"rows {len(d):,}, snapshots {d.drop_duplicates(['season','week','pid']).shape[0]:,}; played share by k:",
          d.groupby("k")["played"].mean().round(3).to_dict())
    tr, va, te = d[d.season <= 2021], d[d.season.between(2022, 2023)], d[d.season >= 2024]
    best = None
    for C in [0.1, 0.3, 1.0, 3.0]:
        cf = fit(tr, C)
        ll = log_loss(va["played"], predict(va, cf))
        print(f"C={C}: validation log loss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, C)
    C = best[1]
    cf = fit(pd.concat([tr, va]), C)
    for name, part in [("validation 2022-23 (fit 2018-21)", (fit(tr, C), va)), ("test 2024-25 (fit 2018-23)", (cf, te))]:
        c, x = part
        p_new, p_pop = predict(x, c), x["pop"]
        p_cal = p_pop * x["played"].mean() / p_pop.mean()     # population curve rescaled to the right average (shape only)
        print(name, f"n={len(x):,}")
        for lab, p in [("population curve (now)", p_pop), ("population curve, rescaled", p_cal), ("return model", p_new)]:
            print(f"  {lab:30s} log loss {log_loss(x['played'], np.clip(p, 1e-4, 1 - 1e-4)):.4f}  Brier {brier_score_loss(x['played'], np.clip(p, 0, 1)):.4f}")
        t = x.assign(new=p_new, pop=p_pop).groupby("k")[["played", "new", "pop"]].mean().round(3)
        print("  by games ahead (actual / model / population):"); print(t.T.to_string())
        sx = x.assign(new=p_new)
        for lab, m in [("Out, missed 1 game", (sx.st_out == 1) & (sx.streak == 1)), ("Questionable, missed 1", (sx.st_q == 1) & (sx.streak == 1)),
                       ("missed 3-4 games", sx.streak.between(3, 4)), ("missed 5+", sx.streak >= 5),
                       ("no report status, missed 1", (sx.rep == "") & (sx.streak == 1))]:
            g = sx[m].groupby("k")[["played", "new", "pop"]].mean().round(2)
            print(f"  {lab} (n={m.sum()}):", "actual", g["played"].tolist(), "model", g["new"].tolist(), "pop", g["pop"].tolist())
    final = fit(d, C)
    final["note"] = ("Return-timing model for players out (didn't play the team's last game, not on IR); avail_out_lab.py, "
                     f"C={C}, fit on 2018-2025 snapshots weeks 3-14; games ahead k=1..{K} after the snapshot week.")
    final["k_max"] = K
    json.dump(final, open(os.path.join("lab", "avail_out_coef.json"), "w"), indent=1)
    print("wrote lab/avail_out_coef.json")


if __name__ == "__main__":
    main()

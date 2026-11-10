"""Player-specific chance to play each remaining game (Nov 9, 2026).

The position x games-ahead curve (v5_config ros.avail_curve) adjusted for the player: his injury-report status this
week and last, age, share of his team's games he has missed this season and over the two seasons before, and his snap
share in the team's last game. Logistic regression on the curve's log-odds, fit on 2021-2025 starters (avail_lab.py:
tested on 2024-2025 with models fit on 2021-2023, log loss on games 1+ weeks out 0.416 -> 0.405, Brier 0.126 -> 0.122).
"""
import glob
import os

import numpy as np
import pandas as pd

POS = ["QB", "RB", "WR", "TE"]
FLAGS = ["q_now", "d_now", "lim_now", "dnp_now", "rep_prev"]


def design(x):
    """Model inputs; x has curve (the position curve's chance for that game), k (games ahead: 0 = this week) and the
    player columns from player_inputs."""
    out = pd.DataFrame(index=x.index)
    c = np.clip(x["curve"].astype(float), 0.01, 0.99)
    out["curve_logit"] = np.log(c / (1 - c))
    kk = np.minimum(x["k"].astype(float), 13)
    near = np.exp(-kk / 2.0)   # report status matters most for the next game or two
    for f in FLAGS:
        out[f] = x[f] * near
        out[f + "_far"] = x[f] * (1 - near)
    out["age30"] = np.maximum(x["age"] - 30, 0) * kk
    out["age"] = (x["age"] - 26) * kk
    out["cur_miss"] = x["cur_miss"] * kk
    out["prior_short"] = (1 - x["prior_games"].clip(0, 1)) * kk
    out["snap_last"] = (x["snap_last"] - 0.8) * kk
    return out


def predict(rows, coef):
    """Chance to play for design rows, from the saved coefficients (dict: intercept, coef by column)."""
    X = design(rows)
    z = coef["intercept"] + sum(X[c] * coef["coef"][c] for c in X.columns)
    return 1 / (1 + np.exp(-z))


def player_inputs(data_dir, season, week, player_ids):
    """Inputs for each gsis player id at snapshot (season, week): age, cur_miss, prior_games, snap_last and the injury
    report flags (week's report as posted so far, and last week's)."""
    sn = []
    for S in (season - 2, season - 1, season):
        f = os.path.join(data_dir, "snaps", f"snaps_{S}.parquet")
        if os.path.exists(f):
            sn.append(pd.read_parquet(f, columns=["season", "week", "game_type", "team", "pfr_player_id",
                                                  "offense_snaps", "offense_pct"]))
    pl = pd.read_parquet(os.path.join(data_dir, "players.parquet"), columns=["pfr_id", "gsis_id", "birth_date"])
    out = pd.DataFrame(index=pd.Index(player_ids, name="player_id"))
    bd = pl.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["birth_date"]
    out["age"] = ((pd.Timestamp(f"{season}-09-01") - pd.to_datetime(out.index.map(bd), errors="coerce")).days / 365.25)
    for f in ["cur_miss", "prior_games", "snap_last"] + FLAGS:
        out[f] = np.nan
    if sn:
        sn = pd.concat(sn, ignore_index=True)
        sn = sn[sn["game_type"] == "REG"]
        g2p = pl.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("gsis_id").set_index("gsis_id")["pfr_id"]
        pid = out.index.map(g2p)
        tg = sn[["season", "week", "team"]].drop_duplicates()
        played = sn[sn["offense_snaps"] > 0]
        cur = played[(played["season"] == season) & (played["week"] < week)].sort_values("week")
        tw = tg[(tg["season"] == season) & (tg["week"] < week)]
        lastp = cur.groupby("pfr_player_id").tail(1).set_index("pfr_player_id")
        firstw = cur.groupby("pfr_player_id")["week"].min()
        n_pl = cur.groupby("pfr_player_id").size()
        team_weeks = {t: sorted(g["week"]) for t, g in tw.groupby("team")}
        prev = played[played["season"].isin([season - 1, season - 2])].groupby("pfr_player_id").size()
        team_g2 = tg[tg["season"].isin([season - 1, season - 2])].groupby("team").size().mean()
        cm, sl = [], []
        for p in pid:
            if p is None or (isinstance(p, float) and np.isnan(p)) or p not in lastp.index:
                cm.append(np.nan); sl.append(np.nan); continue
            t = lastp.at[p, "team"]
            span = [w for w in team_weeks.get(t, []) if w >= firstw[p]]
            cm.append(1 - n_pl[p] / max(len(span), 1))
            sl.append(lastp.at[p, "offense_pct"])
        out["cur_miss"], out["snap_last"] = cm, sl
        out["prior_games"] = [prev.get(p, 0) / max(team_g2, 1) if isinstance(p, str) else np.nan for p in pid]
    f = os.path.join(data_dir, "inj", f"inj_{season}.parquet")
    if os.path.exists(f):
        inj = pd.read_parquet(f, columns=["week", "gsis_id", "report_status", "practice_status"])
        now = inj[inj["week"] == week].drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
        prv = inj[inj["week"] == week - 1].drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
        rep = out.index.map(now["report_status"])
        prac = pd.Series(out.index.map(now["practice_status"]), index=out.index).fillna("").astype(str)
        rep = pd.Series(rep, index=out.index)
        out["q_now"] = (rep == "Questionable").astype(int)
        out["d_now"] = rep.isin(["Doubtful", "Out"]).astype(int)
        out["lim_now"] = prac.str.startswith("Limited").astype(int)
        out["dnp_now"] = prac.str.startswith("Did Not").astype(int)
        r1 = pd.Series(out.index.map(prv["report_status"]), index=out.index)
        out["rep_prev"] = (r1.notna() & (r1.astype(str) != "")).astype(int)
    return out


def fill_defaults(x):
    x = x.copy()
    x["age"] = x["age"].fillna(26.0)
    x["cur_miss"] = x["cur_miss"].fillna(0.0)
    x["prior_games"] = x["prior_games"].fillna(0.6).clip(0, 1.2)
    x["snap_last"] = x["snap_last"].fillna(0.6).clip(0.3, 1.0)
    for f in FLAGS:
        x[f] = x[f].fillna(0).astype(int)
    return x

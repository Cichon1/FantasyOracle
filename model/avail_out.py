"""When an out player comes back (Nov 23, 2026): his chance to play each of his team's games after this week.

For players who didn't play their team's last game and aren't on IR / reserve. Inputs: this week's injury-report status
and practice participation (last week's report early in the week, before this week's is out), how many straight team
games he has missed, the body part, position, games ahead. Logistic model from avail_out_lab.py (2018-2025; tested on
2024-2025 fit on 2018-2023: log loss 0.634 -> 0.583 vs the position 'out' curve, Brier 0.221 -> 0.200).

The ROS engine keeps each out player's expected games from his ROS total and uses this only for their timing (ros.py
depth_inputs): a player listed Out after missing one game plays ~40% of next week's games, ~60% of the week after and
~70% from then on, instead of a flat ~55-65% every week.
"""
import os

import numpy as np
import pandas as pd

POS = ["QB", "RB", "WR", "TE"]
K = 6
BODY = {"soft": "hamstring|calf|groin|quad|thigh|pectoral|abdom|oblique",
        "ankle_foot": "ankle|foot|toe|heel|achilles",
        "knee": "knee|acl|mcl|meniscus",
        "head": "concussion|head",
        "upper": "shoulder|rib|chest|hand|wrist|elbow|thumb|finger|arm|clavicle|collarbone",
        "core": "back|neck|hip|glute|spine",
        "illness": "illness|personal|not injury|rest|covid"}
STATUS = ["st_out", "st_doubt", "st_q", "pr_dnp", "pr_lim", "pr_full"]
STREAK = ["sk2", "sk3", "sk5"]
FEATS = STATUS + STREAK + ["b_" + b for b in BODY] + ["p_RB", "p_WR", "p_TE"]


def body_group(s):
    s = str(s or "").lower()
    for k, pat in BODY.items():
        if any(p in s for p in pat.split("|")):
            return k
    return "other" if s else ""


def features(d):
    """d: rows with rep, prac, streak, body, position (and k). Adds the 0/1 inputs."""
    d = d.copy()
    d["rep"], d["prac"] = d["rep"].fillna("").astype(str), d["prac"].fillna("").astype(str)
    d["st_out"] = (d["rep"] == "Out").astype(int)
    d["st_doubt"] = (d["rep"] == "Doubtful").astype(int)
    d["st_q"] = (d["rep"] == "Questionable").astype(int)
    d["pr_dnp"] = d["prac"].str.startswith("Did Not").astype(int)
    d["pr_lim"] = d["prac"].str.startswith("Limited").astype(int)
    d["pr_full"] = d["prac"].str.startswith("Full").astype(int)
    d["sk2"] = (d["streak"] == 2).astype(int)
    d["sk3"] = d["streak"].between(3, 4).astype(int)
    d["sk5"] = (d["streak"] >= 5).astype(int)
    for b in BODY:
        d["b_" + b] = (d["body"] == b).astype(int)
    for p in ["RB", "WR", "TE"]:
        d["p_" + p] = (d["position"] == p).astype(int)
    return d


def design(x):
    """Model matrix: a base rate per games ahead (k = 1..K, later games use K) and each input's effect for k = 1, k = 2
    and k >= 3."""
    out = pd.DataFrame(index=x.index)
    kk = x["k"].clip(upper=K)
    for k in range(1, K + 1):
        out[f"k{k}"] = (kk == k).astype(float)
    k1, k2 = (x["k"] == 1).astype(float), (x["k"] == 2).astype(float)
    for f in FEATS:
        out[f + "_1"] = x[f] * k1
        out[f + "_2"] = x[f] * k2
        out[f + "_f"] = x[f] * (1 - k1 - k2)
    return out


def predict(x, coef):
    X = design(x)
    z = sum(X[c] * coef["coef"].get(c, 0.0) for c in X.columns) + coef.get("intercept", 0.0)
    return 1 / (1 + np.exp(-z))


def player_inputs(data_dir, season, week, ids):
    """For gsis ids: straight team games missed before `week` (this season), this week's report / practice status (last
    week's while this week's report isn't out for his team) and the body part (latest report naming it, 6 weeks)."""
    out = pd.DataFrame(index=pd.Index(list(ids), name="player_id"))
    out["streak"], out["rep"], out["prac"], out["body"] = 1, "", "", ""
    pl = pd.read_parquet(os.path.join(data_dir, "players.parquet"), columns=["pfr_id", "gsis_id"]).dropna().drop_duplicates("gsis_id")
    g2p = pl.set_index("gsis_id")["pfr_id"]
    f = os.path.join(data_dir, "snaps", f"snaps_{season}.parquet")
    if os.path.exists(f):
        sn = pd.read_parquet(f, columns=["season", "week", "game_type", "team", "pfr_player_id", "offense_snaps"])
        sn = sn[(sn["game_type"] == "REG") & (sn["week"] < week)]
        team_weeks = {t: sorted(g["week"].unique()) for t, g in sn.groupby("team")}
        pset = set(zip(sn.loc[sn["offense_snaps"] > 0, "pfr_player_id"], sn.loc[sn["offense_snaps"] > 0, "week"]))
        last_team = sn.sort_values("week").groupby("pfr_player_id")["team"].last()
        st = []
        for g in out.index:
            p = g2p.get(g)
            t = last_team.get(p) if p is not None else None
            n = 0
            for w in reversed(team_weeks.get(t, [])):
                if (p, w) in pset:
                    break
                n += 1
            st.append(max(n, 1))
        out["streak"] = st
    f = os.path.join(data_dir, "inj", f"inj_{season}.parquet")
    if os.path.exists(f):
        inj = pd.read_parquet(f, columns=["week", "gsis_id", "team", "report_status", "practice_status",
                                          "report_primary_injury", "practice_primary_injury"])
        inj = inj[inj["week"] <= week]
        teams_now = set(inj.loc[inj["week"] == week, "team"])
        by = {g: x.sort_values("week") for g, x in inj[inj["gsis_id"].isin(out.index)].groupby("gsis_id")}
        for g in out.index:
            x = by.get(g)
            if x is None or not len(x):
                continue
            team = x["team"].iloc[-1]
            use = week if team in teams_now else week - 1
            r = x[x["week"] == use]
            if len(r):
                r = r.iloc[-1]
                out.at[g, "rep"] = r["report_status"] if isinstance(r["report_status"], str) else ""
                out.at[g, "prac"] = r["practice_status"] if isinstance(r["practice_status"], str) else ""
            named = x[x["week"] >= week - 6]
            for _, q in named[::-1].iterrows():
                what = q["report_primary_injury"] if isinstance(q["report_primary_injury"], str) else (
                    q["practice_primary_injury"] if isinstance(q["practice_primary_injury"], str) else "")
                if what:
                    out.at[g, "body"] = body_group(what)
                    break
    return out


def curves(data_dir, season, week, players, n_after, coef):
    """players: DataFrame with player_id (gsis) and position; n_after: games after this week per row (same index).
    Returns {row index: array of chances for games 1..n}."""
    if not len(players):
        return {}
    inp = player_inputs(data_dir, season, week, players["player_id"].unique())
    rows = []
    for i, pid, pos, n in zip(players.index, players["player_id"], players["position"], n_after):
        if pid in inp.index:
            for k in range(1, int(n) + 1):
                rows.append({"idx": i, "k": k, "position": pos, **inp.loc[pid].to_dict()})
    if not rows:
        return {}
    d = features(pd.DataFrame(rows))
    d["p"] = predict(d, coef).values
    return {i: g.sort_values("k")["p"].values for i, g in d.groupby("idx")}

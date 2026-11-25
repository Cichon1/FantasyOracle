"""Phase 4 analysis (PREREG section 6): strategy rankings by season, rolling walk-forward 2021-2025 with four look-backs,
position-value trends, draft-slot breakdown.

Seat result vs the ADP control in the same simulated league (paired): d_title, d_playoffs, d_pf (points for vs league
mean, in % of the league mean). Season = unit of evidence: per-season means, then mean and SE across seasons.

  python automation/draft_test/analyze.py   -> automation/draft_test/results/tournament_*.json / .csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_data as B  # noqa: E402

T = os.path.join(B.OUT, "tourney")
RES = os.path.join(HERE, "results")
LOOKBACKS = {"all prior": None, "last 5": 5, "last 3": 3, "recency (half-life 3)": "hl3"}
REPL = {"bdsm": {"QB": 12, "RB": 29, "WR": 30, "TE": 13}, "league": {"QB": 12, "RB": 29, "WR": 41, "TE": 13}}


def load():
    fs = [os.path.join(T, f) for f in os.listdir(T) if f.endswith(".parquet")]
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["seat"] = d["strategy"] + np.where(d["board"] == "-", "", " | " + d["board"])
    base = d[d["strategy"] == "c_adp"].set_index(["league", "season", "slot", "sim"])[["title", "playoffs", "pf_rel"]]
    d = d.join(base, on=["league", "season", "slot", "sim"], rsuffix="_ctl")
    d["d_title"] = d["title"] - d["title_ctl"]
    d["d_playoffs"] = d["playoffs"] - d["playoffs_ctl"]
    d["d_pf"] = 100 * (d["pf_rel"] - d["pf_rel_ctl"])
    return d


def by_season(d):
    return d.groupby(["league", "seat", "season"])[["title", "playoffs", "pf_rel", "d_title", "d_playoffs", "d_pf"]].mean().reset_index()


def overall(s, seasons):
    x = s[s["season"].isin(seasons)]
    g = x.groupby(["league", "seat"])
    o = g[["title", "playoffs", "d_title", "d_playoffs", "d_pf"]].mean()
    o["se_title"] = g["d_title"].std(ddof=1) / np.sqrt(g.size())
    o["se_pf"] = g["d_pf"].std(ddof=1) / np.sqrt(g.size())
    o["seasons"] = g.size()
    o["beat_ctl_title_in"] = g["d_title"].apply(lambda v: int((v > 0).sum()))
    return o.reset_index().sort_values(["league", "d_title"], ascending=[True, False])


def weights(seasons, y, lb):
    prior = [s for s in seasons if s < y]
    if lb is None:
        return {s: 1.0 for s in prior}
    if lb == "hl3":
        return {s: 0.5 ** ((y - 1 - s) / 3) for s in prior}
    return {s: 1.0 for s in prior[-lb:]}


def walk_forward(s, metric="d_title"):
    """for each test season Y 2021-2025 and look-back: pick the best seat on seasons < Y, score it in Y."""
    out = []
    for lg in s["league"].unique():
        x = s[s["league"] == lg]
        seasons = sorted(x["season"].unique())
        for y in [2021, 2022, 2023, 2024, 2025]:
            if y not in seasons:
                continue
            for name, lb in LOOKBACKS.items():
                w = weights(seasons, y, lb)
                h = x[x["season"].isin(w)].assign(w=lambda t: t["season"].map(w))
                # a seat must have existed in every look-back season (ECR boards start 2021)
                cnt = h.groupby("seat")["season"].nunique()
                ok = cnt[cnt == len(w)].index
                h = h[h["seat"].isin(ok) & (h["seat"] != "c_adp")]
                score = h.groupby("seat").apply(lambda t: np.average(t[metric], weights=t["w"]))
                pick = score.idxmax()
                r = x[(x["season"] == y) & (x["seat"] == pick)].iloc[0]
                out.append({"league": lg, "test_season": y, "lookback": name, "picked": pick,
                            "history_score": round(float(score.max()), 4), "test_d_title": float(r["d_title"]),
                            "test_d_playoffs": float(r["d_playoffs"]), "test_d_pf": float(r["d_pf"])})
    return pd.DataFrame(out)


def position_value_trends():
    """points above replacement (each league's scoring) of the players drafted in each round, by position and season."""
    rows = []
    for y in range(2013, 2026):
        p = pd.read_parquet(os.path.join(B.OUT, f"pool_{y}.parquet"))
        p = p[p["matched"]]
        for lg in ["bdsm", "league"]:
            for pos, n in REPL[lg].items():
                g = p[p["pos"] == pos]
                repl = g.nlargest(n, f"pts_{lg}")[f"pts_{lg}"].iloc[-1] if len(g) >= n else 0
                g = g.assign(rnd=np.ceil(g["adp_ffc"] / 12).clip(upper=14))
                for r, gg in g.groupby("rnd"):
                    rows.append({"season": y, "league": lg, "pos": pos, "round": int(r), "n": len(gg),
                                 "par": float((gg[f"pts_{lg}"] - repl).mean())})
    t = pd.DataFrame(rows)
    # trend per position x round band: slope over seasons with season-level SE; must hold in both halves
    res = []
    for (lg, pos), g in t.groupby(["league", "pos"]):
        for band, rr in [("rounds 1-2", (1, 2)), ("rounds 3-5", (3, 5)), ("rounds 6-9", (6, 9)), ("rounds 10+", (10, 14))]:
            x = g[(g["round"] >= rr[0]) & (g["round"] <= rr[1])].groupby("season")["par"].mean()
            if len(x) < 6:
                continue
            X = x.index.values - x.index.values.mean()
            b = np.polyfit(X, x.values, 1)[0]
            resid = x.values - np.polyval(np.polyfit(X, x.values, 1), X)
            se = np.sqrt(resid.var(ddof=2) / (X ** 2).sum())
            h1 = x[x.index <= 2019].mean() - x.mean()
            h2 = x[x.index >= 2020].mean() - x.mean()
            res.append({"league": lg, "pos": pos, "band": band, "par_2013_19": round(float(x[x.index <= 2019].mean()), 1),
                        "par_2020_25": round(float(x[x.index >= 2020].mean()), 1), "slope_per_season": round(float(b), 2),
                        "se": round(float(se), 2), "trend": abs(b) >= 2 * se and np.sign(h2) == np.sign(b) and np.sign(h1) == -np.sign(b)})
    return t, pd.DataFrame(res)


def main():
    os.makedirs(RES, exist_ok=True)
    d = load()
    s = by_season(d)
    s.to_csv(os.path.join(RES, "tournament_by_season.csv"), index=False)
    allseasons = sorted(s["season"].unique())
    o_all = overall(s, allseasons)
    o_recent = overall(s, [y for y in allseasons if y >= 2020])
    o_all.to_csv(os.path.join(RES, "tournament_overall_2013_2025.csv"), index=False)
    o_recent.to_csv(os.path.join(RES, "tournament_overall_2020_2025.csv"), index=False)
    wf = walk_forward(s)
    wf.to_csv(os.path.join(RES, "tournament_walk_forward.csv"), index=False)
    slot = d.groupby(["league", "seat", "slot"])[["d_title", "d_pf"]].mean().reset_index()
    slot.to_csv(os.path.join(RES, "tournament_by_slot.csv"), index=False)
    t, tr = position_value_trends()
    t.to_csv(os.path.join(RES, "position_value_by_round.csv"), index=False)
    tr.to_csv(os.path.join(RES, "position_value_trends.csv"), index=False)
    pd.set_option("display.width", 250)
    for lg in ["bdsm", "league"]:
        print(f"\n== {lg}: all seasons", o_all[o_all["league"] == lg].head(45).round(3).to_string(index=False))
        print(f"\n== {lg}: 2020-2025", o_recent[o_recent["league"] == lg].head(15).round(3).to_string(index=False))
    print("\n== walk-forward\n", wf.round(3).to_string(index=False))
    print("\n== walk-forward by look-back\n", wf.groupby(["league", "lookback"])[["test_d_title", "test_d_playoffs", "test_d_pf"]].mean().round(3))
    print("\n== position value trends\n", tr.to_string(index=False))


if __name__ == "__main__":
    main()

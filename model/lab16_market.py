"""Nov 19, 2026: who is right when we disagree with the trade market? A live test of the buy-low / sell-high labels.

The trade analyzer labels players undervalued / overvalued by comparing our rest-of-season points per game with what
the market pays. FantasyPros' Trade Market Values (real redraft PPR trades over 10 days) have been saved daily since
Oct 20 (automation/history/trade_market_<date>.csv), and our ROS snapshots weekly (docs/data/ros_history_v5.csv;
weeks 7-8 re-created with the same code and as-of data, 9-10 live). So for snapshot weeks 7-10 of 2026:

  ours     official ROS points per game (ros_ppg)
  model    the model's own ROS per game (before the 50/50 expert blend)
  experts  FantasyPros ROS consensus rank
  market   trade market value (players with 20+ trades in the 10 days before the snapshot)
  outcome  actual PPR from week W through the last finished week per team game (missed games count as 0, the same
           basis as ROS per game)

Scored within position: rank correlation of each with the outcome, and the label test - "price" = our ROS per game of
the player at the same market rank (as the analyzer does); undervalued if ours - price >= +1.0, overvalued <= -1.0;
did undervalued players beat their price and overvalued ones fall short?

  python lab16_market.py      -> lab/market_test.csv (player rows), lab/market_test.json
"""
import glob
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_MARKET = {7: "2026-10-21", 8: "2026-10-28", 9: "2026-11-04", 10: "2026-11-11"}   # day before each week's Thursday
LAST_WEEK = int(os.environ.get("MARKET_LAST_WEEK", "10"))
MIN_TRADES = 20


def key(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ros = pd.read_csv(os.path.join(ROOT, "docs", "data", "ros_history_v5.csv"))
    w = pd.read_parquet(os.path.join(ROOT, "model", "data", "week_2026.parquet"),
                        columns=["player_id", "week", "season_type", "team", "fantasy_points_ppr"])
    w = w[(w["season_type"] == "REG") & (w["week"] <= LAST_WEEK)]
    g = pd.read_parquet(os.path.join(ROOT, "model", "data", "games.parquet"),
                        columns=["season", "week", "game_type", "home_team", "away_team"])
    g = g[(g["season"] == 2026) & (g["game_type"] == "REG") & (g["week"] <= LAST_WEEK)]   # finished weeks only
    team_weeks = pd.concat([g[["week", "home_team"]].rename(columns={"home_team": "team"}),
                            g[["week", "away_team"]].rename(columns={"away_team": "team"})])
    rows = []
    for W, day in SNAP_MARKET.items():
        if W > LAST_WEEK:
            continue
        mk = pd.read_csv(os.path.join(ROOT, "automation", "history", f"trade_market_{day}.csv"))
        mk = mk[mk["trades_10d"] >= MIN_TRADES]
        r = ros[ros["week"] == W].copy()
        r["name_key"] = r["player"].map(key)
        m = r.merge(mk[["name_key", "pos", "market_value", "trades_10d"]], on=["name_key", "pos"], how="inner")
        # outcome: PPR in weeks W..LAST per team game (team at the snapshot)
        pts = w[w["week"] >= W].groupby("player_id")["fantasy_points_ppr"].sum()
        tg = team_weeks[team_weeks["week"] >= W].groupby("team").size()
        m["team_games"] = m["team"].map(tg)
        m["actual_ppg"] = m["player_id"].map(pts).fillna(0.0) / m["team_games"]
        m["model_ppg"] = m["ros_model"] / m["games_left"].clip(lower=1)
        m["snap_week"] = W
        m["market_day"] = day
        # price = our per-game value of the player at the same market rank (within position), as the analyzer does
        for src, col, asc in [("market", "market_value", False), ("experts", "expert_rank", True)]:
            m[f"price_{src}"] = np.nan
            for p, gp in m.groupby("pos"):
                gp = gp[gp[col].notna()]
                order = gp[col].rank(ascending=asc, method="first").astype(int).values - 1
                m.loc[gp.index, f"price_{src}"] = np.sort(gp["ros_ppg"].values)[::-1][order]
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    d = d[d["team_games"] > 0]
    d.to_csv(os.path.join(ROOT, "model", "lab", "market_test.csv"), index=False)

    out = {"players_by_week": d.groupby("snap_week").size().to_dict(), "rank_corr": {}, "labels": {}}
    cells = []
    for (W, p), gp in d.groupby(["snap_week", "pos"]):
        if len(gp) < 8:
            continue
        cells.append({"week": W, "pos": p, "n": len(gp),
                      "ours": spearmanr(gp["ros_ppg"], gp["actual_ppg"]).correlation,
                      "model": spearmanr(gp["model_ppg"], gp["actual_ppg"]).correlation,
                      "experts": spearmanr(-gp["expert_rank"], gp["actual_ppg"], nan_policy="omit").correlation,
                      "market": spearmanr(gp["market_value"], gp["actual_ppg"]).correlation})
    c = pd.DataFrame(cells)
    out["rank_corr"]["cells"] = c.round(3).to_dict("records")
    for s in ["ours", "model", "experts", "market"]:
        out["rank_corr"][s] = round(float(c[s].mean()), 4)
    for s in ["ours", "model", "experts"]:
        dd = c[s] - c["market"]
        out["rank_corr"][f"{s}_minus_market"] = [round(float(dd.mean()), 4), round(float(dd.std(ddof=1) / np.sqrt(len(dd))), 4),
                                                  int((dd > 0).sum()), int(len(dd))]
    by_pos = c.groupby("pos")[["ours", "experts", "market"]].mean().round(3)
    out["rank_corr"]["by_pos"] = by_pos.to_dict("index")
    for src in ["market", "experts"]:
        gap = d["ros_ppg"] - d[f"price_{src}"]
        lab = np.where(gap >= 1.0, "undervalued", np.where(gap <= -1.0, "overvalued", "at value"))
        res = d.assign(label=lab, beat=d["actual_ppg"] - d[f"price_{src}"], ours_err=d["actual_ppg"] - d["ros_ppg"])
        t = res.groupby("label").agg(players=("player", "size"), beat_price_ppg=("beat", "mean"),
                                     share_beat=("beat", lambda x: (x > 0).mean()),
                                     ours_minus_actual=("ours_err", "mean"))
        # standard error of the mean (player rows; weeks overlap for the same player, so this is optimistic)
        t["se"] = res.groupby("label")["beat"].agg(lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        out["labels"][src] = t.round(3).to_dict("index")
        # does the size of the gap predict beating the price? (within week-position, rank correlation)
        cs = [spearmanr((gp["ros_ppg"] - gp[f"price_{src}"]), gp["actual_ppg"] - gp[f"price_{src}"]).correlation
              for _, gp in res.groupby(["snap_week", "pos"]) if len(gp) >= 8]
        out["labels"][src]["gap_vs_beat_rank_corr"] = round(float(np.nanmean(cs)), 3)
    json.dump(out, open(os.path.join(ROOT, "model", "lab", "market_test.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

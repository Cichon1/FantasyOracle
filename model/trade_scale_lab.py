"""Trade value scale check (Oct 12, 2026): does our trade value order players the way real trades do?

Trade value per game = expected points above the league's waiver level + expected points above its last starter, each
counted only when positive, over a two-piece normal with the ROS projection as its mean and the spreads of the 10th /
90th percentile projections, less what the 3 best free agents at the position offer on the same terms (floored at 0, so
a free agent is worth about nothing). Waiver level = average of the 3 best players on no team in the league (Yahoo rosters);
last starter = best player left once every team's starting lineup is filled from the whole pool.

Compared with FantasyPros Trade Market Values (redraft PPR, players with 20+ trades in 10 days), in full PPR (the
market's scoring), with each of Jackson's leagues' rosters and lineups. The market value is a bounded 0-1000 scale
(the top ~25 players all sit at 780-1000), so it is used for order only: rank correlation overall and within each
position, and how often two players at different positions are ordered the same way (positional balance). The star
premium (last-starter term) is checked against waiver-only values.

Usage: python trade_scale_lab.py ros_v5.csv trade_market_v5.csv league_rosters_v5.csv leagues_v5.csv [out.json]
"""
import json
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

POS = ["QB", "RB", "WR", "TE"]
Z90 = norm.ppf(0.9)


def upside(m, lo, hi, r):
    """E[max(0, X - r)], X two-piece normal with mean m and 10th / 90th percentile spreads (m - lo, hi - m)."""
    sl, sh = max(m - lo, 1e-6) / Z90, max(hi - m, 1e-6) / Z90
    d = m - (sh - sl) * np.sqrt(2 / np.pi) / 2 - r
    up = d + sh * np.sqrt(2 / np.pi) if d >= 0 else 2 * (d * (1 - norm.cdf(-d / sh)) + sh * norm.pdf(-d / sh))
    dn = 0.0 if d <= 0 else 2 * (d * (norm.cdf(d / sl) - 0.5) - sl * (norm.pdf(0) - norm.pdf(d / sl)))
    return 0.5 * up + 0.5 * dn


def nkey(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[.'’\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def levels(ros, rostered, lineup, slot_pos, teams):
    pool = ros.sort_values("ros_ppg", ascending=False)
    fa = pool[~pool.key.isin(rostered)]
    waiver = {p: float(fa[fa.pos == p].ros_ppg.head(3).mean()) for p in POS}
    m = dict(x.split("=") for x in slot_pos.split("; "))
    slots = sorted([m[t].split(" ") for t in lineup.split() if t in m and all(q in POS for q in m[t].split(" "))], key=len)
    used = set()
    for elig in slots:
        for _ in range(teams):
            c = pool[pool.pos.isin(elig) & ~pool.key.isin(used)].head(1)
            if len(c):
                used.add(c.key.iloc[0])
    left = pool[~pool.key.isin(used)]
    starter = {p: float(left[left.pos == p].ros_ppg.iloc[0]) for p in POS}
    return waiver, starter


def pair_agreement(df, col):
    a = df[["pos", col, "market_value"]].to_numpy()
    agree = tot = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i, 0] != a[j, 0] and a[i, 1] != a[j, 1] and a[i, 2] != a[j, 2]:
                tot += 1
                agree += (a[i, 1] > a[j, 1]) == (a[i, 2] > a[j, 2])
    return agree / tot


def main():
    ros, tm, rs, lg = (pd.read_csv(f) for f in sys.argv[1:5])
    ros = ros[ros.pos.isin(POS)].copy()
    ros["name_key"] = ros.player.map(nkey)
    tm = tm[(tm.trades_10d >= 20) & (tm.market_value > 0)]
    out = {}
    for L in lg.itertuples():
        rostered = set(rs[rs.league_id == L.league_id].ros_key.dropna())
        waiver, starter = levels(ros, rostered, L.lineup, L.slot_pos, int(L.teams))
        g = ros.games_left.clip(lower=1)
        ow = [upside(r.ros_ppg, r.ros_low / gg, r.ros_high / gg, waiver[r.pos]) for r, gg in zip(ros.itertuples(), g)]
        os_ = [upside(r.ros_ppg, r.ros_low / gg, r.ros_high / gg, starter[r.pos]) for r, gg in zip(ros.itertuples(), g)]
        x = ros.assign(uw=ow, us=os_, fa=~ros.key.isin(rostered))
        base = {p: x[(x.pos == p) & x.fa].sort_values("ros_ppg", ascending=False).head(3)[["uw", "us"]].mean() for p in POS}
        x["ow"] = [max(0.0, a - base[p].uw) for a, p in zip(x.uw, x.pos)]
        x["os"] = [max(0.0, b - base[p].us) for b, p in zip(x.us, x.pos)]
        x["value"] = x.ow + x.os
        on = x[~x.fa]
        zeros = int((on.value < 0.05).sum()), int(len(on))
        x = x.merge(tm[["name_key", "pos", "market_value"]], on=["name_key", "pos"])
        res = {"waiver_ppr": {p: round(v, 2) for p, v in waiver.items()}, "last_starter_ppr": {p: round(v, 2) for p, v in starter.items()},
               "players": int(len(x)), "rostered_at_zero": zeros}
        for col, lab in [("ow", "waiver_only"), ("value", "trade_value")]:
            res[lab] = {"spearman_all": round(float(spearmanr(x[col], x.market_value).correlation), 3),
                        "spearman_by_pos": {p: round(float(spearmanr(x[x.pos == p][col], x[x.pos == p].market_value).correlation), 3) for p in POS},
                        "cross_position_pairs_agree": round(pair_agreement(x, col), 3)}
        x["ours"] = x.value.rank(ascending=False)
        x["mkt"] = x.market_value.rank(ascending=False)
        x["gap"] = x.mkt - x.ours
        cols = ["player", "pos", "ros_ppg", "value", "market_value", "ours", "mkt"]
        res["we_like_more"] = x.sort_values("gap", ascending=False).head(6)[cols].round(1).to_dict("records")
        res["market_likes_more"] = x.sort_values("gap").head(6)[cols].round(1).to_dict("records")
        out[L.league] = res
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 5:
        json.dump(out, open(sys.argv[5], "w"), indent=1)


if __name__ == "__main__":
    main()

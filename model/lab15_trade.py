"""Nov 18, 2026: what is a player actually worth in a trade? A sim test of trade-value formulas.

In the full-season league sim (lab12_season.py: BDSM settings, persistent teams, FantasyPros-run managers, weekly
waivers, byes and injuries, 14-week season, 8-team playoffs), at trade week W a random team T receives player X
(from whoever rosters him; T sends back its least valuable player that keeps its roster legal, or X comes from free
agency). The rest of the season is replayed and compared with the same league without the trade:
  realized value of the trade for T = change in wins, playoff odds, title odds, last-place odds, points.
Real seasons 2023-2025 and real points, so this is hindsight value - the target a trade-value formula, computed with
what was known before week W, should predict.

Formulas compared (each = value(X) - value(player T gives up), sources FantasyPros / model / model+Sleeper+ESPN /
official):
  starter   rest-of-season points over the last starter (QB13 / RB30 / WR36 / TE13)
  waiver    points over a waiver-wire player (QB25 / RB61 / WR61 / TE19)
  hub       average of the two (the shape of the hub's current trade value)
  playoff   per-game value over the last starter x (games left + extra weight on weeks 15-17 games)

  python lab15_trade.py FRAME.parquet       (TRADE_LEAGUES default 30, TRADE_WEEKS default 4,7,10)
  -> lab/trade_sim_runs.parquet, lab/trade_sim_summary.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lab5 as L
import lab11_sim as S11
import lab12_season as G

N_LEAGUES = int(os.environ.get("TRADE_LEAGUES", "30"))
WEEKS = [int(w) for w in os.environ.get("TRADE_WEEKS", "4,7,10").split(",")]
SAMPLE = {0: 16, 1: 40, 2: 50, 3: 16}           # players tested per position at each trade week (by FP value)
WAIVER_RANK = {0: 25, 1: 61, 2: 61, 3: 19}
SOURCES = ["fp", "model", "cons_free", "official"]


def formulas(se, W, curve):
    """Per player: each formula's value before week W (points)."""
    ros = se.ros(W, curve)          # value over the last starter (lab12: QB13 / RB30 / WR36 / TE13), per source
    out = {}
    po_games = se.plays[:, 15:18].sum(1).astype(float)
    games = np.maximum(se.games_left[:, W], 1)
    for s in SOURCES:
        v = ros[s]
        wav = np.zeros_like(v)
        for p in range(4):
            sel = se.pos == p
            srt = np.sort(v[sel])[::-1]
            k = WAIVER_RANK[p] - 1
            wav[sel] = v[sel] - (srt[k] if len(srt) > k else srt[-1])
        out[f"{s}_starter"] = v
        out[f"{s}_waiver"] = np.maximum(wav, 0)
        out[f"{s}_hub"] = 0.5 * (np.maximum(v, 0) + np.maximum(wav, 0))
        for k in (1, 2):
            out[f"{s}_playoff{k}"] = v / games * (games + k * po_games)
    return out


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
    for S in [int(s) for s in os.environ.get("TRADE_SEASONS", "2023,2024,2025").split(",")]:
        board, _ = G.draft_board(S, games, ids)
        se = G.Season(d, S, board, games, 17)
        curve = cv[S - 1]
        rng = np.random.default_rng(5000 + S)
        F = {W: formulas(se, W, curve) for W in WEEKS}
        for lg in range(N_LEAGUES):
            rb = np.random.default_rng(rng.integers(1 << 31))
            rosters = G.draft(se, rb)
            sched = G.schedule(rb)
            base = G.play_season(se, curve, rosters, sched, [], "control")
            for W in WEEKS:
                T = int(rb.integers(G.TEAMS))
                # T's standing before week W (from the no-trade replay)
                wp = base["weekly_pts"]
                wins_before = 0.0
                for w in range(1, W):
                    for a, b in sched[w - 1]:
                        if T in (a, b):
                            o = b if a == T else a
                            wins_before += 1.0 if wp[T, w] > wp[o, w] else (0.5 if wp[T, w] == wp[o, w] else 0.0)
                ppw_before = wp[T, 1:W].mean()
                fpv = F[W]["fp_starter"]
                cand = []
                for p, n in SAMPLE.items():
                    sel = np.where((se.pos == p) & (se.games_left[:, W] > 0))[0]
                    cand += list(sel[np.argsort(-fpv[sel])][:n])
                for x in cand:
                    sw = {"week": W, "team": T, "add": [int(x)], "drop": None}
                    r = G.play_season(se, curve, rosters, sched, [], "control", trade=sw)
                    drop = sw.get("_dropped")
                    rec = {"season": S, "league": lg, "week": W, "team": T, "player": se.name[x], "pos": G.POS[se.pos[x]],
                           "wins_before": wins_before, "ppw_before": round(float(ppw_before), 2)}
                    for k in ["wins", "playoffs", "title", "last"]:
                        rec[f"d_{k}"] = float(r[k][T]) - float(base[k][T])
                    rec["d_pts"] = float(r["weekly_pts"][T, W:].sum() - base["weekly_pts"][T, W:].sum())
                    rec["dropped"] = int(drop) if drop is not None else -1
                    for f, v in F[W].items():
                        rec[f] = float(v[x]) - (float(v[drop]) if drop is not None else 0.0)
                    recs.append(rec)
            if lg % 5 == 0:
                print(S, lg, len(recs), round((time.time() - t0) / 60, 1), "min", flush=True)
    df = pd.DataFrame(recs)
    df.to_parquet(os.path.join(L.LAB_DIR, "trade_sim_runs.parquet"))
    json.dump(summarize(df), open(os.path.join(L.LAB_DIR, "trade_sim_summary.json"), "w"), indent=1)
    print(json.dumps(summarize(df), indent=1))


def summarize(df):
    fcols = [c for c in df.columns if any(c.startswith(s + "_") for s in SOURCES)]
    out = {"runs": int(len(df)), "baseline_rates": {}}
    res = {}
    for target in ["d_pts", "d_wins", "d_playoffs", "d_title"]:
        r = {}
        for f in fcols:
            # mean rank correlation within season-week (same information set), formula delta vs realized change
            cs = [spearmanr(g[f], g[target]).correlation for _, g in df.groupby(["season", "week"]) if g[target].std() > 0]
            r[f] = round(float(np.mean(cs)), 4)
        res[target] = dict(sorted(r.items(), key=lambda kv: -kv[1]))
    out["rank_corr_formula_vs_realized"] = res
    # context: realized playoff-odds change per 10 points of realized points gained, by T's standing before the trade
    df = df.assign(strength=pd.cut(df["ppw_before"], [0, 90, 95, 100, 105, 200]),
                   rec=pd.cut(df["wins_before"] / (df["week"] - 1), [-0.01, 0.34, 0.5, 0.67, 1.01],
                              labels=["<=.33", ".34-.50", ".51-.67", ">.67"]))
    pos = df[df["d_pts"] > 5]
    ctx = pos.groupby("rec", observed=True).apply(lambda g: pd.Series({
        "playoff_pts_per_10": 100 * 10 * g["d_playoffs"].sum() / g["d_pts"].sum(),
        "title_pts_per_10": 100 * 10 * g["d_title"].sum() / g["d_pts"].sum(),
        "last_pts_per_10": 100 * 10 * g["d_last"].sum() / g["d_pts"].sum(),
        "wins_per_10": 10 * g["d_wins"].sum() / g["d_pts"].sum(), "n": len(g)})).round(3)
    out["context_by_record"] = ctx.reset_index().astype({"rec": str}).to_dict("records")
    return out


if __name__ == "__main__":
    main()

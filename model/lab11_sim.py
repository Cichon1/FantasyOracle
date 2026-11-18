"""Nov 17, 2026 league sim, step 2: does the model set better lineups than FantasyPros? (start/sit only)

Real seasons, real points: 2023-2025 weeks 1-17 and 2026 weeks 1-10, every player's actual PPR points, every
projection the walk-forward model made before that week (lab11_preds.py). Only the leagues are simulated.

League (BDSM League settings): 12 teams, start QB, RB, RB, WR, WR, TE, W/R/T flex (K and DEF left out: the model
doesn't project them), 13 QB/RB/WR/TE per roster (2 QB, 4 RB, 5 WR, 2 TE).

Rosters: each week, N_LEAGUES random leagues. Rosters come from that week's FantasyPros rest-of-season rank (known
before the week; weekly rank, then preseason rank, where missing), turned into points and jittered, then snake-split
within each position - so rosters look like an in-season league where waivers fill holes, and change league to league.

Managers (paired: every team's lineup is set by each source, against the same opponent):
  model     the model's projection (0 for players the model wouldn't project: not expected to play)
  official  the website's blend: model rank and FantasyPros rank averaged within the position, model points
            handed out in that order
  fp        FantasyPros weekly rank, turned into points with last season's rank -> points curve (players not ranked
            that week: 0; Thursday-night players with no rank that week keep their previous rank)
  perfect   hindsight best lineup (the ceiling)
Opponents always set their lineup with FantasyPros (a typical league).

Two settings: "inactives known" (every manager benches players who didn't suit up - Yahoo shows inactives before
lock) and "set midweek" (nobody checks Sunday news).

Scored per team-week: points started, points short of the perfect lineup, win rate vs the opponent. Paired
differences (model - fp, official - fp) with standard errors over weeks (each week's league mean is one observation).

  python lab11_sim.py FRAME.parquet        -> lab/sim_results.json, lab/sim_weekly.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import lab5 as L

N_LEAGUES = int(os.environ.get("SIM_LEAGUES", "500"))
USE_FLEX = os.environ.get("SIM_FLEX", "1") == "1"   # 0: no flex spot (isolates within-position calls)
TAG = os.environ.get("SIM_TAG", "")
TEAMS = 12
QUOTA = {"QB": 2, "RB": 4, "WR": 5, "TE": 2}
START = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX = ("RB", "WR", "TE")
LAST_WEEK = 17
JITTER = 2.0          # points of noise on each player's draft value, per league
SOURCES = ["model", "official", "fp", "perfect"]
rng = np.random.default_rng(20261117)


def curves(frame):
    """Rank -> points per position for each season (mean actual points by weekly pos rank, smoothed, non-increasing)."""
    d = frame[(frame["season_type"] == "REG") & frame["expected"].eq(True) & frame["ecr_week_pos_rank"].notna()
              & (frame["season"] >= 2022)]
    out = {}
    for S, g in d.groupby("season"):
        c = g.groupby(["position", "ecr_week_pos_rank"])[L.TARGET].mean()
        cs = {}
        for p in QUOTA:
            s = c.loc[p].sort_index() if p in c.index.get_level_values(0) else pd.Series(dtype=float)
            s = s.reindex(np.arange(1, 251)).interpolate(limit_direction="both")
            cs[p] = s.rolling(7, center=True, min_periods=1).mean().cummin().values
        out[int(S)] = cs
    return out


def rank_pts(curve, pos, rank):
    r = np.clip(np.nan_to_num(rank, nan=250), 1, 250).astype(int) - 1
    return np.array([curve[p][i] for p, i in zip(pos, r)])


def prepare(rows, cv):
    d = rows[(rows["season_type"] == "REG") & (rows["week"] <= LAST_WEEK) & rows["position"].isin(QUOTA)].copy()
    d = d.sort_values(["player_id", "season", "week"])
    d["expected"] = d["expected"].eq(True)
    d["sat"] = d["phantom"].eq(True)
    # Thursday-night players often have no weekly rank in the snapshot: keep their previous rank that season
    prev = d.groupby(["player_id", "season"])["ecr_week_pos_rank"].shift(1)
    tnf = d["ecr_week_pos_rank"].isna() & d["ctx_thursday"].eq(1)
    d["fp_rank"] = d["ecr_week_pos_rank"].where(~tnf, prev)
    d["model"] = np.where(d["expected"], d["pred"], 0.0)
    d["fp"] = 0.0
    d["value_rank"] = d["ecr_ros_pos_rank"].fillna(d["ecr_week_pos_rank"]).fillna(d["ecr_preseason_pos_rank"])
    d["value"] = np.nan
    for S, g in d.groupby("season"):
        c = cv[int(S) - 1]
        idx = g.index
        has = g["fp_rank"].notna()
        d.loc[idx[has], "fp"] = rank_pts(c, g.loc[has, "position"], g.loc[has, "fp_rank"])
        hv = g["value_rank"].notna()
        d.loc[idx[hv], "value"] = rank_pts(c, g.loc[hv, "position"], g.loc[hv, "value_rank"])
    # official blend: within position-week, among players with both, order by the average of the two ranks and
    # hand out the model's points in that order
    d["official"] = d["model"]
    both = d["fp_rank"].notna() & d["expected"]
    for _, g in d[both].groupby(["season", "week", "position"]):
        mr = g["model"].rank(ascending=False, method="first")
        er = g["fp_rank"].rank(method="first")
        order = (mr + er + mr * 1e-6).rank(method="first").astype(int).values - 1
        pts = np.sort(g["model"].values)[::-1]
        d.loc[g.index, "official"] = pts[order]
    d["perfect"] = d[L.TARGET]
    return d


def lineup_points(pos_idx, score, act):
    """pos_idx: (n,) position code 0..3, score/act: (n,). Greedy QB1 RB2 WR2 TE1 + best remaining RB/WR/TE.
    Returns starters' actual points per position code (QB, RB, WR, TE)."""
    out = np.zeros(4)
    order = np.argsort(-score, kind="stable")
    need = [1, 2, 2, 1]
    flex_left = 1 if USE_FLEX else 0
    for i in order:
        p = pos_idx[i]
        if need[p] > 0:
            need[p] -= 1
            out[p] += act[i]
        elif flex_left and p != 0:
            flex_left -= 1
            out[p] += act[i]
        if flex_left == 0 and not any(need):
            break
    return out


def simulate_week(g, inactives_known):
    g = g[g["value"].notna()].reset_index(drop=True)
    pos_code = g["position"].map({"QB": 0, "RB": 1, "WR": 2, "TE": 3}).values
    act = g[L.TARGET].values
    sat = g["sat"].values
    scores = {s: g[s].values.astype(float).copy() for s in SOURCES}
    if inactives_known:
        for s in SOURCES:
            scores[s] = np.where(sat, -1e9, scores[s])
    by_pos = {p: np.where(g["position"].values == p)[0] for p in QUOTA}
    res = {s: np.zeros((N_LEAGUES, TEAMS, 4)) for s in SOURCES}
    win = {s: np.zeros((N_LEAGUES, TEAMS)) for s in SOURCES}
    for lg in range(N_LEAGUES):
        teams = [[] for _ in range(TEAMS)]
        for p, q in QUOTA.items():
            idx = by_pos[p]
            v = g["value"].values[idx] + rng.normal(0, JITTER, len(idx))
            take = idx[np.argsort(-v)][: TEAMS * q]
            order = rng.permutation(TEAMS)
            snake = np.concatenate([order if r % 2 == 0 else order[::-1] for r in range(q)])
            for t, i in zip(snake[: len(take)], take):
                teams[t].append(i)
        for t in range(TEAMS):
            ix = np.array(teams[t])
            for s in SOURCES:
                res[s][lg, t] = lineup_points(pos_code[ix], scores[s][ix], act[ix])
        opp = rng.permutation(TEAMS).reshape(-1, 2)
        fp_tot = res["fp"][lg].sum(1)
        for a, b in opp:
            for s in SOURCES:
                win[s][lg, a] = (res[s][lg, a].sum() > fp_tot[b]) + 0.5 * (res[s][lg, a].sum() == fp_tot[b])
                win[s][lg, b] = (res[s][lg, b].sum() > fp_tot[a]) + 0.5 * (res[s][lg, b].sum() == fp_tot[a])
    rec = {}
    for s in SOURCES:
        rec[f"pts_{s}"] = res[s].sum(2).mean()
        rec[f"win_{s}"] = win[s].mean()
        for j, p in enumerate(QUOTA):
            rec[f"pos_{p}_{s}"] = res[s][:, :, j].mean()
    # how often the lineup differs from FantasyPros' (share of team-weeks)
    return rec


def summarize(wk, label):
    out = {"setting": label, "weeks": int(len(wk))}
    for s in SOURCES:
        out[f"pts_{s}"] = round(wk[f"pts_{s}"].mean(), 2)
        out[f"win_{s}"] = round(wk[f"win_{s}"].mean(), 4)
    for a in ["model", "official", "perfect"]:
        dp = wk[f"pts_{a}"] - wk["pts_fp"]
        dw = wk[f"win_{a}"] - wk["win_fp"]
        out[f"{a}_minus_fp_pts"] = [round(dp.mean(), 3), round(dp.std(ddof=1) / np.sqrt(len(dp)), 3)]
        out[f"{a}_minus_fp_win"] = [round(dw.mean(), 4), round(dw.std(ddof=1) / np.sqrt(len(dw)), 4)]
        out[f"{a}_weeks_better"] = int((dp > 0).sum())
        out[f"{a}_minus_fp_by_pos"] = {p: round((wk[f"pos_{p}_{a}"] - wk[f"pos_{p}_fp"]).mean(), 3) for p in QUOTA}
    for S, g in wk.groupby("season"):
        out[f"season_{S}"] = {a: round((g[f"pts_{a}"] - g["pts_fp"]).mean(), 3) for a in ["model", "official"]}
        out[f"season_{S}"]["weeks"] = int(len(g))
        out[f"season_{S}"]["win_model_minus_fp"] = round((g["win_model"] - g["win_fp"]).mean(), 4)
    return out


def main():
    frame = pd.read_parquet(sys.argv[1], columns=["season", "week", "season_type", "position", "expected",
                                                   "ecr_week_pos_rank", L.TARGET])
    rows = pd.read_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet"))
    d = prepare(rows, curves(frame))
    results, weekly = [], []
    for known in (True, False):
        label = "inactives known" if known else "set midweek"
        recs = []
        for (S, w), g in d.groupby(["season", "week"]):
            r = simulate_week(g, known)
            r.update(season=int(S), week=int(w), setting=label)
            recs.append(r)
            print(label, S, w, round(r["pts_model"] - r["pts_fp"], 2), flush=True)
        wk = pd.DataFrame(recs)
        weekly.append(wk)
        results.append(summarize(wk, label))
        print(json.dumps(results[-1], indent=1), flush=True)
    pd.concat(weekly).to_csv(os.path.join(L.LAB_DIR, f"sim_weekly{TAG}.csv"), index=False)
    json.dump({"leagues_per_week": N_LEAGUES, "results": results}, open(os.path.join(L.LAB_DIR, f"sim_results{TAG}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()

"""
ROS walk-forward test. Each season S (2020-2025) is projected at every snapshot week by models trained only on
earlier seasons (rounds early-stopped on S-1 with S-2 and earlier, then refit through S-1 with 1.1x the rounds);
the weekly projection used for week W is the out-of-sample weekly proxy (ros_weekly_oos.py).
Validation seasons 2020-2022 decide every setting; 2023-2025 are reported as the test.

  python ros_lab.py build   FRAME OUT_ROWS          snapshot rows for 2018-2025 (+ targets)
  python ros_lab.py run     ROWS WEEKLY_OOS NAME [--md N] [--seeds K] [--nofeat a,b]
  python ros_lab.py score   NAME...                 scores (validation / test), blend weight from validation
  python -c "import ros_lab as RL; RL.score2('NAME')"    weekly + ROS-rate mix by run state (a, w grid)
  python -c "import ros_lab as RL; RL.final_eval()"      the chosen setting as production computes it
Outputs go to $ROS_LAB (default lab_ros/); the weekly projections file is ros_weekly_oos.py's output.

Findings (Oct 12, 2026; experts' ROS pool, snapshot weeks 3-14, mean weekly rank correlation with actual ROS PPR):
  FantasyPros ROS 0.474 val / 0.459 test. ROS rate model alone x games 0.441 / 0.427; weekly projection x games
  0.472 / 0.463; mix (current week = weekly projection, later weeks 0.8 weekly + 0.2 rate; out / bye = rate) 0.478 /
  0.467; official (50/50 rank blend with the experts) 0.508 / 0.483 (+0.024 +/- 0.007, 11/12 season-positions ahead).
  Model-only weekly projection instead of the official one: 0.509 / 0.489. Rate-model variants (min leaf 15 / 100,
  6 seeds, weekly expert rank as input) were within noise on validation; kept min leaf 40, 3 seeds.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ros as R

VAL, TEST = [2020, 2021, 2022], [2023, 2024, 2025]
OUT = os.environ.get("ROS_LAB", "lab_ros")


def build(frame, out):
    t0 = time.time()
    df = pd.read_parquet(frame)
    ctx = R.Context("data", "fpdata")
    rows = R.build_history(df, ctx, list(range(2018, 2026)))
    rows.to_parquet(out)
    print("rows", rows.shape, round((time.time() - t0) / 60, 1), "min", flush=True)
    print(rows.groupby("season").agg(n=("player_id", "size"), out=("ros_out_now", "sum"),
                                      ecr=("ros_ecr_pos_rank", lambda s: s.notna().sum())).to_string())


def run(rows_path, weekly_path, name, md=40, seeds=3, nofeat=(), q=False, ecrweek=False):
    os.makedirs(OUT, exist_ok=True)
    rows = pd.read_parquet(rows_path)
    wk = pd.read_parquet(weekly_path)
    rows["wk_official"] = rows["ros_row_index"].map(wk["pred_ppr_official"])
    rows["wk_model"] = rows["ros_row_index"].map(wk["pred_ppr_model"])
    if "pred_ppr_noweek" in wk:
        rows["wk_noweek"] = rows["ros_row_index"].map(wk["pred_ppr_noweek"])
    for c in ["wk_official", "wk_model", "wk_noweek"]:
        if c in rows:
            rows.loc[(rows["ros_out_now"] == 1) | (rows["ros_bye_now"] == 1), c] = np.nan
    trn_ok = rows["ros_games_after"] >= 1
    X = [c for c in R.feature_list(rows) if c not in nofeat]
    if ecrweek:
        X += [c for c in rows.columns if c.startswith("ecr_week_")]
    params = dict(R.PARAMS, min_data_in_leaf=md)
    sd = (7, 23, 41, 101, 202, 303)[:seeds]
    res, info = [], {}
    t0 = time.time()
    for S in VAL + TEST:
        te = rows[rows["season"] == S].copy()
        for pos in R.POS_MODELS:
            key = "/".join(pos)
            p = rows["position"].isin(pos) & trn_ok
            tr, va = rows[p & (rows["season"] <= S - 2)], rows[p & (rows["season"] == S - 1)]
            r = R.early_stop_rounds(tr, va, X, "ros_rate_after", params)
            mods = R.fit(rows[p & (rows["season"] <= S - 1)], X, "ros_rate_after", max(50, int(r * 1.1)), sd, params)
            tp = te["position"].isin(pos)
            te.loc[tp, "rate"] = R.predict(mods, te[tp], X)
            if q:
                for a in (0.1, 0.9):
                    qm = R.fit(rows[p & (rows["season"] <= S - 1)], X, "ros_rate_after", max(50, int(r * 1.1)), (7,),
                               params, objective="quantile", alpha=a)
                    te.loc[tp, f"rate_q{int(a * 100)}"] = R.predict(qm, te[tp], X)
            info[f"{S}|{key}"] = r
        res.append(te[[c for c in ["season", "week", "player_id", "player_display_name", "position", "team",
                                   "ros_out_now", "ros_bye_now", "ros_games_after", "ros_games_cur", "ros_cur_played",
                                   "ros_ecr_pos_rank", "ros_pts", "ros_pts_after", "wk_official", "wk_model", "wk_noweek", "rate",
                                   "rate_q10", "rate_q90", "fantasy_points_ppr_ewm", "games_hist"] if c in te.columns]])
        print(S, round((time.time() - t0) / 60, 1), "min", {k: v for k, v in info.items() if k.startswith(str(S))},
              flush=True)
    out = pd.concat(res)
    out.to_parquet(f"{OUT}/{name}.parquet")
    json.dump({"rounds": info, "md": md, "seeds": seeds, "nofeat": list(nofeat), "X": len(X)},
              open(f"{OUT}/{name}_info.json", "w"))
    print("saved", name, flush=True)


# ---------------------------------------------------------------------------------------------------- scoring
def methods(d):
    """Score columns (higher = better) for every method."""
    open_ = R.cur_open(d)
    d["m_naive"] = d["wk_official"].fillna(0) * (d["ros_games_after"] + open_)
    d["m_rate"] = d["rate"] * (d["ros_games_after"] + open_)
    d["m_ros"] = d["rate"] * d["ros_games_after"] + d["wk_official"].fillna(0) * open_
    d["m_ros_model"] = d["rate"] * d["ros_games_after"] + d["wk_model"].fillna(0) * open_
    d["experts"] = -d["ros_ecr_pos_rank"]
    return d


def pool(d):
    lw = d["season"].map(R.last_week)
    d = d[(d["week"] >= 3) & (d["week"] <= lw - 3) & d["ros_ecr_pos_rank"].notna()].copy()
    d["_r"] = d.groupby(["season", "week", "position"])["ros_ecr_pos_rank"].rank(method="first")
    return d[d["_r"] <= d["position"].map(R.POOL)]


def blend(d, col, w):
    out = []
    for _, g in d.groupby(["season", "week", "position"]):
        mr = g[col].rank(ascending=False, method="first")
        er = g["ros_ecr_pos_rank"].rank(method="first")
        out.append(-(w * mr + (1 - w) * er))
    return pd.concat(out)


def corr_table(d, cols):
    rows = []
    for (S, W, p), g in d.groupby(["season", "week", "position"]):
        if len(g) < 10:
            continue
        r = {"season": S, "week": W, "pos": p, "n": len(g)}
        for c in cols:
            r[c] = spearmanr(g[c], g["ros_pts"]).correlation
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(t, cols, ref="experts"):
    out = {}
    for c in cols:
        diff = t[c] - t[ref]
        cell = t.assign(_d=diff).groupby(["season", "pos"])["_d"].mean()
        out[c] = {"rho": round(float(t[c].mean()), 4), "vs_experts": round(float(diff.mean()), 4),
                  "se_cells": round(float(cell.std() / np.sqrt(len(cell))), 4),
                  "se_weeks": round(float(diff.std() / np.sqrt(len(diff))), 4),
                  "cells_ahead": f"{int((cell > 0).sum())}/{len(cell)}"}
    return out


def score(names):
    grid = np.round(np.arange(0, 1.0001, 0.1), 2)
    for name in names:
        d = methods(pd.read_parquet(f"{OUT}/{name}.parquet"))
        pl = pool(d)
        base = ["experts", "m_naive", "m_rate", "m_ros", "m_ros_model"]
        # blend weight on validation seasons only
        v = pl[pl["season"].isin(VAL)].copy()
        best = None
        for w in grid:
            v["_b"] = blend(v, "m_ros", w)
            rho = corr_table(v, ["_b"])["_b"].mean()
            if best is None or rho > best[1]:
                best = (float(w), float(rho))
        w = best[0]
        pl["official"] = blend(pl, "m_ros", w)
        pl["blend_50"] = blend(pl, "m_ros", 0.5)
        cols = base + ["official", "blend_50"]
        t = corr_table(pl, cols)
        rep = {"name": name, "blend_w_model": w, "val_grid_best": best}
        for ph, ss in (("validation", VAL), ("test", TEST)):
            rep[ph] = summarize(t[t["season"].isin(ss)], cols)
            rep[ph + "_by_pos"] = t[t["season"].isin(ss)].groupby("pos")[cols].mean().round(4).to_dict(orient="index")
            rep[ph + "_by_season"] = {int(k): v for k, v in t[t["season"].isin(ss)].groupby("season")[cols].mean()
                                      .round(4).to_dict(orient="index").items()}
            # accuracy of ROS totals on the experts' pool (model points)
            g = pl[pl["season"].isin(ss)]
            for c in ["m_naive", "m_rate", "m_ros"]:
                err = g[c] - g["ros_pts"]
                rep[ph][c]["mae"] = round(float(err.abs().mean()), 2)
                rep[ph][c]["r2"] = round(float(1 - (err ** 2).sum() / ((g["ros_pts"] - g["ros_pts"].mean()) ** 2).sum()), 4)
            # all projected players (not only the experts' pool)
            a = d[d["season"].isin(ss) & (d["week"] >= 3) & (d["week"] <= d["season"].map(R.last_week) - 3)]
            err = a["m_ros"] - a["ros_pts"]
            rep[ph]["m_ros"]["mae_all"] = round(float(err.abs().mean()), 2)
            rep[ph]["m_ros"]["r2_all"] = round(float(1 - (err ** 2).sum() / ((a["ros_pts"] - a["ros_pts"].mean()) ** 2).sum()), 4)
            if "rate_q10" in d:
                o = R.cur_open(a)
                lo = a["rate_q10"] * a["ros_games_after"] + a["wk_official"].fillna(0) * o
                hi = a["rate_q90"] * a["ros_games_after"] + a["wk_official"].fillna(0) * o
                rep[ph]["range"] = {"below_low": round(float((a["ros_pts"] < lo).mean()), 3),
                                    "above_high": round(float((a["ros_pts"] > hi).mean()), 3)}
        t.to_csv(f"{OUT}/{name}_weekly_corr.csv", index=False)
        json.dump(rep, open(f"{OUT}/{name}_report.json", "w"), indent=1)
        print(json.dumps({k: rep[k] for k in ["name", "blend_w_model", "validation", "test"]}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3])
    elif cmd == "run":
        a = sys.argv[2:]
        kw = {}
        if "--md" in a:
            kw["md"] = int(a[a.index("--md") + 1])
        if "--seeds" in a:
            kw["seeds"] = int(a[a.index("--seeds") + 1])
        if "--nofeat" in a:
            kw["nofeat"] = a[a.index("--nofeat") + 1].split(",")
        if "--q" in a:
            kw["q"] = True
        if "--ecrweek" in a:
            kw["ecrweek"] = True
        run(a[0], a[1], a[2], **kw)
    elif cmd == "score":
        score(sys.argv[2:])


# ------------------------------------------------------------------------- v2 scoring: weekly + ROS-rate mix
def ros_mix(d, a, wkcol):
    """ROS points with the week-W projection for the current game and, for later games, a per-game rate that mixes
    the week-W projection (share a) with the ROS rate model; players without a week-W projection (out, on bye, or no
    weekly run yet) use the ROS rate alone."""
    o = R.cur_open(d)
    wk = d[wkcol] if wkcol else pd.Series(np.nan, index=d.index)
    has = wk.notna() & (o > 0)
    fut = np.where(has, a * wk.fillna(0) + (1 - a) * d["rate"], d["rate"])
    cur = np.where(has, wk.fillna(0), d["rate"] * o)
    return pd.Series(cur + fut * d["ros_games_after"], index=d.index)


STATES = {"sunday": "wk_official", "noweek": "wk_noweek", "tuesday": None}


WEEKLY = os.path.join(OUT, "weekly_oos.parquet")


def attach_weekly(d, path=WEEKLY):
    """Out-of-sample weekly projections for in-rows (not out, not on bye), by season / week / player."""
    wk = pd.read_parquet(path).drop_duplicates(["season", "week", "player_id"]).set_index(["season", "week", "player_id"])
    k = pd.MultiIndex.from_arrays([d["season"], d["week"], d["player_id"]])
    inn = (d["ros_out_now"] == 0) & (d.get("ros_bye_now", 0) == 0)
    for src, dst in [("pred_ppr_official", "wk_official"), ("pred_ppr_model", "wk_model"), ("pred_ppr_noweek", "wk_noweek")]:
        if src in wk:
            d[dst] = pd.Series(wk[src].reindex(k).values, index=d.index).where(inn)
    return d


def score2(name, a_grid=(0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), w_grid=(0.3, 0.4, 0.5, 0.6, 0.7), states=None):
    d = attach_weekly(pd.read_parquet(f"{OUT}/{name}.parquet"))
    if "ros_bye_now" not in d:
        d["ros_bye_now"] = 0
    d["experts"] = -d["ros_ecr_pos_rank"]
    pl = pool(d)
    rep = {"name": name}
    for st, wkcol in (states or STATES).items():
        grid = [(a, w) for a in (a_grid if wkcol else (0,)) for w in w_grid]
        best, table = None, []
        for a, w in grid:
            pl["_m"] = ros_mix(pl, a, wkcol)
            pl["_b"] = blend(pl, "_m", w)
            t = corr_table(pl, ["_m", "_b", "experts"])
            v, te = t[t["season"].isin(VAL)], t[t["season"].isin(TEST)]
            row = {"a": a, "w": w, "val_model": v["_m"].mean(), "val_off": v["_b"].mean(), "val_exp": v["experts"].mean(),
                   "test_model": te["_m"].mean(), "test_off": te["_b"].mean(), "test_exp": te["experts"].mean()}
            table.append(row)
            if best is None or row["val_off"] > best[0]["val_off"]:
                best = (row, t)
        row, t = best
        cols = ["_m", "_b"]
        rep[st] = {"a": row["a"], "w_model": row["w"]}
        for ph, ss in (("validation", VAL), ("test", TEST)):
            s = summarize(t[t["season"].isin(ss)], cols)
            rep[st][ph] = {"experts": round(float(t[t["season"].isin(ss)]["experts"].mean()), 4),
                           "model": s["_m"], "official": s["_b"],
                           "by_pos": t[t["season"].isin(ss)].groupby("pos")[["experts", "_m", "_b"]].mean().round(4)
                           .rename(columns={"_m": "model", "_b": "official"}).to_dict(orient="index"),
                           "by_season": {int(k): v for k, v in t[t["season"].isin(ss)].groupby("season")[["experts", "_m", "_b"]]
                                         .mean().round(4).rename(columns={"_m": "model", "_b": "official"}).to_dict(orient="index").items()}}
        rep[st]["grid"] = pd.DataFrame(table).round(4).to_dict(orient="records")
        t.to_csv(f"{OUT}/{name}_{st}_weekly_corr.csv", index=False)
    json.dump(rep, open(f"{OUT}/{name}_report2.json", "w"), indent=1, default=float)
    for st in (states or STATES):
        r = rep[st]
        print(f"{name:14s} {st:9s} a={r['a']} w={r['w_model']} | val exp {r['validation']['experts']:.4f} model {r['validation']['model']['rho']:.4f} "
              f"off {r['validation']['official']['rho']:.4f} ({r['validation']['official']['vs_experts']:+.4f}±{r['validation']['official']['se_cells']:.4f}) | "
              f"test exp {r['test']['experts']:.4f} model {r['test']['model']['rho']:.4f} off {r['test']['official']['rho']:.4f} "
              f"({r['test']['official']['vs_experts']:+.4f}±{r['test']['official']['se_cells']:.4f}, {r['test']['official']['cells_ahead']})")
    return rep


# ----------------------------------------------------------------------------------- final configuration test
def final_eval(name="base_md40", a=0.8, w=0.5):
    """The chosen setting, scored as production computes it: official = rank blend over every projected player
    (experts' unranked players behind the ranked ones), model points handed out in that order; ranges calibrated
    per position on validation seasons."""
    d = attach_weekly(pd.read_parquet(f"{OUT}/{name}.parquet"))
    lw = d["season"].map(R.last_week)
    d = d[(d["week"] >= 3) & (d["week"] <= lw - 3)].copy()
    out = {"name": name, "a": a, "w_model": w}
    rows_all = []
    for st, wkcol in [("sunday", "wk_official"), ("noweek", "wk_noweek")]:
        d["model"] = ros_mix(d, a, wkcol)
        d["official"] = R.blend_with_experts(d, "model", w)
        d["experts"] = -d["ros_ecr_pos_rank"]
        # ranges: spread from the rate quantile models over the games after this week, scaled per position
        lo_s = (d["rate"] - d["rate_q10"]).clip(lower=0) * d["ros_games_after"]
        hi_s = (d["rate_q90"] - d["rate"]).clip(lower=0) * d["ros_games_after"]
        ks = {}
        for p in R.POOL:
            v = d[(d["position"] == p) & d["season"].isin(VAL)]
            grid = np.round(np.arange(0.5, 3.01, 0.02), 2)
            klo = min(grid, key=lambda k: abs((v["ros_pts"] < v["official"] - k * lo_s[v.index]).mean() - 0.10))
            khi = min(grid, key=lambda k: abs((v["ros_pts"] > v["official"] + k * hi_s[v.index]).mean() - 0.10))
            ks[p] = (float(klo), float(khi))
        d["low"] = (d["official"] - d["position"].map({p: k[0] for p, k in ks.items()}) * lo_s).clip(lower=0)
        d["high"] = d["official"] + d["position"].map({p: k[1] for p, k in ks.items()}) * hi_s
        pl = pool(d)
        t = corr_table(pl, ["experts", "model", "official"])
        res = {"range_k": ks}
        for ph, ss in (("validation", VAL), ("test", TEST)):
            tt = t[t["season"].isin(ss)]
            s = summarize(tt, ["model", "official"])
            g, ga = pl[pl["season"].isin(ss)], d[d["season"].isin(ss)]
            res[ph] = {"experts": round(float(tt["experts"].mean()), 4), "model": s["model"], "official": s["official"],
                       "mae_pool": round(float((g["official"] - g["ros_pts"]).abs().mean()), 2),
                       "mae_all": round(float((ga["official"] - ga["ros_pts"]).abs().mean()), 2),
                       "r2_all": round(float(1 - ((ga["official"] - ga["ros_pts"]) ** 2).sum()
                                             / ((ga["ros_pts"] - ga["ros_pts"].mean()) ** 2).sum()), 4),
                       "below_low": round(float((ga["ros_pts"] < ga["low"]).mean()), 3),
                       "above_high": round(float((ga["ros_pts"] > ga["high"]).mean()), 3),
                       "position_weeks": int(len(tt))}
            for p, gp in tt.groupby("pos"):
                rows_all.append({"state": st, "phase": ph, "pos": p, "experts": gp["experts"].mean(),
                                 "model": gp["model"].mean(), "official": gp["official"].mean(),
                                 "diff": (gp["official"] - gp["experts"]).mean(), "n": len(gp)})
            rows_all.append({"state": st, "phase": ph, "pos": "All", "experts": tt["experts"].mean(),
                             "model": tt["model"].mean(), "official": tt["official"].mean(),
                             "diff": (tt["official"] - tt["experts"]).mean(), "n": len(tt),
                             "se": s["official"]["se_cells"], "cells": s["official"]["cells_ahead"]})
            for S, gs in tt.groupby("season"):
                rows_all.append({"state": st, "phase": ph, "pos": f"season {S}", "experts": gs["experts"].mean(),
                                 "model": gs["model"].mean(), "official": gs["official"].mean(),
                                 "diff": (gs["official"] - gs["experts"]).mean(), "n": len(gs)})
        out[st] = res
        t.to_csv(f"{OUT}/final_{st}_weekly_corr.csv", index=False)
    pd.DataFrame(rows_all).round(4).to_csv(f"{OUT}/final_summary.csv", index=False)
    json.dump(out, open(f"{OUT}/final_report.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    return out

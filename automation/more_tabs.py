"""Refresh the site's "More" tabs after each model update.

  Matchups          coach_tendencies.csv, defense_profiles.csv, offense_style.csv - this week's teams, from the model's
                    own feature frame (team tendencies as of the last game played, weighted toward recent games).
  Run game & style  league_trends.csv (current-season row), runs_by_box.csv, short_yardage.csv - from play-by-play,
                    FTN charting and participation data (2018 to date).
  Vs experts        experts_v5.csv, experts_season_v5.csv, experts_position_v5.csv - 2023-25 walk-forward backtest
                    (fixed, automation/ref/experts_weeks_2023_25.csv) plus this season's live projections.
  Model             accuracy_v5.csv (current-season row and All), calibration_v5.csv, coverage_v5.csv - 2023-25
                    backtest (fixed, automation/ref/) plus this season's live projections.

Left as they are: weather study, play-action/fullback study (2022-25 charting), ROS backtests, experiment log and the
old-vs-honest test - historical studies that a few more weeks don't change.

  python automation/more_tabs.py --season 2026 --week 8 --frame model/out_v5/frame.parquet --data model/data
"""
import argparse
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "..", "docs", "data")
REF = os.path.join(HERE, "ref")
POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
TOPK = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
SOURCES = [("Model v5", "proj_model"), ("FantasyPros experts", None), ("Official (model + experts)", "proj_ppr")]


# ----------------------------------------------------------------------------------------------- matchups
def matchups(frame, season, week):
    f = pd.read_parquet(frame)
    f = f[(f["season"] == season) & (f["week"] == week)]
    if not len(f):
        return {}
    by_team = f.groupby("team").first()
    by_def = f.groupby("opponent_team").first()
    r1 = lambda s: s.astype(float).round(1)
    r3 = lambda s: s.astype(float).round(3)
    co = pd.DataFrame({"team": by_team.index, "coach": by_team["coach"], "opp": by_team["opponent_team"],
                       "proe": r1(by_team["coach_proe"]), "proe_light_box": r1(by_team["coach_proe_light_box"]),
                       "proe_heavy_box": r1(by_team["coach_proe_heavy_box"]), "proe_vs_base": r1(by_team["coach_proe_vs_base"]),
                       "proe_vs_sub": r1(by_team["coach_proe_vs_sub"]), "proe_matchup": r1(by_team["coach_proe_matchup_pers"]),
                       "plays_pg": r1(by_team["pace_plays_pg"]), "sec_per_play": r1(by_team["pace_sec_per_play"]),
                       "week": week})
    de = pd.DataFrame({"defense": by_def.index, "faces": by_def["team"], "blitz_rate": r3(by_def["def_blitz_rate"]),
                       "man_rate": r3(by_def["def_man_rate"]), "two_high_rate": r3(by_def["def_two_high_rate"]),
                       "cover2_rate": r3(by_def["def_cover2_rate"]), "cover3_rate": r3(by_def["def_cover3_rate"]),
                       "base_rate": r3(by_def["def_base_rate"]), "base_vs_heavy_rate": r3(by_def["def_base_vs_heavy_rate"]),
                       "dime_rate": r3(by_def["def_dime_rate"]), "light_box_rate": r3(by_def["def_light_box_rate"]),
                       "pressure_rate": r3(by_def["def_pressure_rate"]), "epa_in_man": r3(by_def["def_epa_in_man"]),
                       "epa_in_zone": r3(by_def["def_epa_in_zone"]), "epa_per_rush": r3(by_def["def_epa_per_rush"]),
                       "explosive_runs_allowed": r3(by_def["def_explosive_run_rate"]),
                       "stuff_rate": r3(by_def["def_stuff_rate"]), "plays_allowed_pg": r3(by_def["def_plays_allowed_pg"]),
                       "week": week})
    of = pd.DataFrame({"team": by_team.index, "opp": by_team["opponent_team"],
                       "under_center_rate": r3(by_team["style_under_center_rate"]),
                       "play_action_rate": r3(by_team["style_play_action_rate"]),
                       "motion_rate": r3(by_team["style_motion_rate"]), "two_back_rate": r3(by_team["style_two_back_rate"]),
                       "heavy_rate": r3(by_team["style_heavy_pers_rate"]), "fullback_rate": r3(by_team["style_fb_rate"]),
                       "run_epa": r3(by_team["off_run_epa"]), "pa_epa": r3(by_team["off_pa_epa"]),
                       "explosive_run_rate": r3(by_team["off_explosive_run_rate"]), "stuff_rate": r3(by_team["off_stuff_rate"]),
                       "short_yd_success": r3(by_team["off_short_yd_success"]), "week": week})
    return {"coach_tendencies.csv": co, "defense_profiles.csv": de, "offense_style.csv": of}


# ----------------------------------------------------------------------------------------- run game & style
def plays(data, season):
    cols = ["season", "season_type", "game_id", "play_id", "rush", "pass", "two_point_attempt", "qb_kneel", "qb_spike",
            "qb_scramble", "qb_dropback", "epa", "yards_gained", "down", "ydstogo", "first_down", "touchdown",
            "rusher_player_id", "shotgun"]
    out = []
    for s in range(2018, season + 1):
        p = os.path.join(data, "pbp", f"pbp_{s}.parquet")
        if os.path.exists(p):
            out.append(pd.read_parquet(p, columns=cols))
    p = pd.concat(out)
    return p[(p["season_type"] == "REG") & ((p["rush"] == 1) | (p["pass"] == 1)) & (p["two_point_attempt"].fillna(0) == 0)
             & (p["qb_kneel"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0)]


def style(data, season):
    p = plays(data, season)
    run = p[(p["rush"] == 1) & (p["qb_scramble"].fillna(0) == 0)]
    out = {}
    # league trends: refresh the current season's row (personnel / coverage columns wait for that season's charting)
    lt = pd.read_csv(os.path.join(SITE, "league_trends.csv"))
    cur = p[p["season"] == season]
    r = run[run["season"] == season]
    db = cur[cur["qb_dropback"] == 1]
    ftn_p = os.path.join(data, "ftn", f"ftn_{season}.parquet")
    pa = np.nan
    if os.path.exists(ftn_p):
        f = pd.read_parquet(ftn_p, columns=["nflverse_game_id", "nflverse_play_id", "is_play_action"])
        m = db.merge(f, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"])
        pa = m["is_play_action"].astype(float).mean() if len(m) else np.nan
    row = {"season": season, "under_center_rate": round((cur["shotgun"] == 0).mean(), 4),
           "play_action_rate": round(pa, 4), "epa_per_run": round(r["epa"].mean(), 4),
           "epa_per_dropback": round(db["epa"].mean(), 4), "explosive_run_rate": round((r["yards_gained"] >= 10).mean(), 4)}
    lt = lt.set_index("season")
    for k, v in row.items():
        if k != "season":
            lt.loc[season, k] = v
    out["league_trends.csv"] = lt.reset_index()
    # runs by box count (participation through last season, FTN charting this season)
    box = []
    for s in range(2018, season + 1):
        pp = os.path.join(data, "part", f"part_{s}.parquet")
        if os.path.exists(pp):
            box.append(pd.read_parquet(pp, columns=["nflverse_game_id", "play_id", "defenders_in_box"])
                       .rename(columns={"nflverse_game_id": "game_id"}))
        fp = os.path.join(data, "ftn", f"ftn_{s}.parquet")
        if os.path.exists(fp) and not os.path.exists(pp):
            box.append(pd.read_parquet(fp, columns=["nflverse_game_id", "nflverse_play_id", "n_defense_box"])
                       .rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id",
                                        "n_defense_box": "defenders_in_box"}))
    if box:
        b = pd.concat(box).drop_duplicates(["game_id", "play_id"])
        rr = run.merge(b, on=["game_id", "play_id"]).dropna(subset=["defenders_in_box"])
        rr["box_bucket"] = pd.cut(rr["defenders_in_box"], [-1, 5, 6, 7, 8, 99], labels=["5 or fewer", "6", "7", "8", "9+"])
        out["runs_by_box.csv"] = rr.groupby("box_bucket", observed=True).agg(
            runs=("epa", "size"), epa_per_run=("epa", "mean"),
            explosive_rate=("yards_gained", lambda x: (x >= 10).mean()),
            stuff_rate=("yards_gained", lambda x: (x <= 0).mean()), ypc=("yards_gained", "mean")).round(4).reset_index()
    # 3rd / 4th and 1
    pl = pd.read_parquet(os.path.join(data, "players.parquet"), columns=["gsis_id", "position"]).dropna()
    pos = pl.drop_duplicates("gsis_id").set_index("gsis_id")["position"]
    sy = p[p["down"].isin([3, 4]) & (p["ydstogo"] == 1)].copy()
    sy["kind"] = np.where(sy["qb_dropback"] == 1, "Dropback pass",
                          np.where(sy["rusher_player_id"].map(pos) == "QB", "QB sneak / QB run", "RB / other run"))
    sy["conv"] = ((sy["first_down"] == 1) | (sy["touchdown"] == 1)).astype(float)
    out["short_yardage.csv"] = sy.groupby("kind").agg(plays=("conv", "size"), conversion_rate=("conv", "mean"),
                                                       epa_per_play=("epa", "mean")).round(4).reset_index()
    return out


# ------------------------------------------------------------------------------------------- vs experts
def pairwise(score, actual):
    s, a = np.asarray(score, float), np.asarray(actual, float)
    i, j = np.triu_indices(len(s), 1)
    ok = a[i] != a[j]
    return float(np.mean(np.sign(s[i] - s[j])[ok] == np.sign(a[i] - a[j])[ok]))


def topk(score, actual, k):
    return len(set(np.argsort(-np.asarray(score))[:k]) & set(np.argsort(-np.asarray(actual))[:k])) / k


def live_expert_weeks(store, season):
    d = store[(store["status"] != "Upcoming") & store["expert_rank"].notna()]
    rows = []
    for (w, pos), g in d.groupby(["week", "pos"]):
        g = g[g["expert_rank"].rank(method="first") <= POOL[pos]]
        g = g[g["status"] != "Inactive"]
        if len(g) < 10 or g["act_ppr"].isna().any():
            continue
        y = g["act_ppr"].values
        for name, col in SOURCES:
            sc = -g["expert_rank"].values if col is None else g[col].values
            rows.append({"season": season, "week": int(w), "pos": pos, "source": name, "n": len(g),
                         "spearman": spearmanr(sc, y).correlation, "pairwise": pairwise(sc, y), "topk": topk(sc, y, TOPK[pos])})
    return pd.DataFrame(rows)


def experts(store, season):
    ref = pd.read_csv(os.path.join(REF, "experts_weeks_2023_25.csv"))
    res = pd.concat([ref, live_expert_weeks(store, season)], ignore_index=True)
    piv = res.pivot_table(index=["season", "week", "pos"], columns="source", values="spearman")
    rows = []
    for name, _ in SOURCES:
        x = res[res["source"] == name]
        r = {"source": name, "spearman": x["spearman"].mean(), "pairwise": x["pairwise"].mean(), "topk_hit": x["topk"].mean()}
        if name == "FantasyPros experts":
            r.update(vs_experts=0.0, vs_experts_se=0.0, win_share=np.nan)
        else:
            diff = piv[name] - piv["FantasyPros experts"]
            r.update(vs_experts=diff.mean(), vs_experts_se=diff.std() / np.sqrt(len(diff)), win_share=(diff > 0).mean())
        r["weeks"] = len(piv)
        rows.append(r)
    ov = pd.DataFrame(rows)
    ov[["spearman", "pairwise", "topk_hit", "vs_experts", "vs_experts_se"]] = ov[["spearman", "pairwise", "topk_hit", "vs_experts", "vs_experts_se"]].round(4)
    ov["win_share"] = ov["win_share"].round(3)
    se = res.groupby(["season", "source"])[["spearman", "pairwise", "topk"]].mean().round(4).reset_index()
    se["key"] = se["season"].astype(str) + " · " + se["source"]
    po = res.groupby(["pos", "source"])[["spearman", "pairwise", "topk"]].mean().round(4).reset_index()
    po["key"] = po["pos"] + " · " + po["source"]
    return {"experts_v5.csv": ov, "experts_season_v5.csv": se, "experts_position_v5.csv": po}


# ------------------------------------------------------------------------------------------------ model
def model_tables(store, season):
    d = store[(store["status"] != "Upcoming") & (store["status"] != "Inactive") & store["proj_model"].notna()].copy()
    d["y"] = d["act_ppr"].fillna(0)
    out = {}
    acc = pd.read_csv(os.path.join(SITE, "accuracy_v5.csv"))
    acc = acc[~acc["season"].astype(str).isin([str(season), "All"])]
    played = d[d["status"] == "Final"]
    r2 = lambda y, p: 1 - np.mean((y - p) ** 2) / np.var(y)
    cur = {"season": str(season), "games": len(d), "r2_expected": round(r2(d["y"], d["proj_model"]), 4),
           "r2_played": round(r2(played["y"], played["proj_model"]), 4), "mae": round(np.mean(np.abs(d["y"] - d["proj_model"])), 4),
           "official_mae": round(np.mean(np.abs(d["y"] - d["proj_ppr"])), 4),
           "below_q10": round(float((d["y"] < d["ppr_low"]).mean()), 4), "above_q90": round(float((d["y"] > d["ppr_high"]).mean()), 4)}
    acc = pd.concat([acc, pd.DataFrame([cur])], ignore_index=True)
    w = acc["games"].astype(float)
    allr = {"season": "All", "games": int(w.sum())}
    for c in ["r2_expected", "r2_played", "mae", "official_mae", "below_q10", "above_q90"]:
        allr[c] = round(float((acc[c].astype(float) * w).sum() / w.sum()), 4)
    out["accuracy_v5.csv"] = pd.concat([acc, pd.DataFrame([allr])], ignore_index=True)
    # calibration: fixed 2023-25 sums + this season
    cal = pd.read_csv(os.path.join(REF, "calibration_2023_25.csv"))
    bins, labs = [-1, 3, 6, 9, 12, 15, 18, 22, 999], list(cal["bin"])
    d["bin"] = pd.cut(d["proj_ppr"], bins, labels=labs)
    c2 = d.groupby("bin", observed=False).agg(p=("proj_ppr", "sum"), a=("y", "sum"), n=("y", "size")).reset_index()
    m = cal.merge(c2, on="bin")
    out["calibration_v5.csv"] = pd.DataFrame({
        "projected_range": m["bin"], "avg_projected": ((m["sum_projected"] + m["p"]) / (m["games"] + m["n"])).round(2),
        "avg_actual": ((m["sum_actual"] + m["a"]) / (m["games"] + m["n"])).round(2), "games": (m["games"] + m["n"]).astype(int)})
    # range coverage: fixed 2023-25 counts + this season
    cv = json.load(open(os.path.join(REF, "coverage_2023_25.json")))
    n = cv["games"] + len(d)
    below = cv["below"] + int((d["y"] < d["ppr_low"]).sum())
    above = cv["above"] + int((d["y"] > d["ppr_high"]).sum())
    out["coverage_v5.csv"] = pd.DataFrame({"outcome": ["Below low end", "Inside range", "Above high end"],
                                           "share_of_games": [round(below / n, 3), round(1 - (below + above) / n, 3), round(above / n, 3)]})
    return out


# ------------------------------------------------------------------------------------------------- main
DATASET = {  # file -> dataset id on the site
    "coach_tendencies.csv": "coaches", "defense_profiles.csv": "defense", "offense_style.csv": "offense_style",
    "league_trends.csv": "league_trends", "runs_by_box.csv": "runs_by_box", "short_yardage.csv": "short_yardage",
    "experts_v5.csv": "experts", "experts_season_v5.csv": "experts_season", "experts_position_v5.csv": "experts_position",
    "accuracy_v5.csv": "accuracy", "calibration_v5.csv": "calibration", "coverage_v5.csv": "coverage"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "model", "data"))
    a = ap.parse_args()
    store = pd.read_csv(os.path.join(SITE, "projections_v5.csv"))
    tables, errors = {}, []
    for name, fn in [("matchups", lambda: matchups(a.frame, a.season, a.week) if a.frame and os.path.exists(a.frame) else {}),
                     ("style", lambda: style(a.data, a.season)), ("experts", lambda: experts(store, a.season)),
                     ("model", lambda: model_tables(store, a.season))]:
        try:
            tables.update(fn())
        except Exception as e:      # one table failing never blocks the rest or the update
            errors.append(f"{name}: {str(e)[:200]}")
    cat_p = os.path.join(SITE, "datasets.json")
    cat = json.load(open(cat_p))
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    changed = []
    for f, df in tables.items():
        path = os.path.join(SITE, f)
        old = pd.read_csv(path) if os.path.exists(path) else None
        if old is not None and list(old.columns) == list(df.columns) and old.astype(str).equals(df.reset_index(drop=True).astype(str)):
            continue
        df.to_csv(path, index=False)
        ds = cat["datasets"][DATASET[f]]
        ds["file"], ds["updated"] = f, now
        if f in ("coach_tendencies.csv", "defense_profiles.csv", "offense_style.csv"):
            ds["title"] = {"coaches": "Week {w} coach tendencies", "defense": "Week {w} defense schemes",
                           "offense_style": "Week {w} offensive style"}[DATASET[f]].format(w=a.week)
            ds["description"] = re.sub(r"Week \d+ (offense|defense)", f"Week {a.week} \\1", ds.get("description", ""))
            ds["description"] = re.sub(r"as of Week \d+", f"as of Week {a.week - 1}", ds["description"])
        changed.append(f)
    if changed:
        cat["updated"] = now
        json.dump(cat, open(cat_p, "w"), indent=1)
    print(json.dumps({"more_tabs": changed, "errors": errors}))


if __name__ == "__main__":
    main()

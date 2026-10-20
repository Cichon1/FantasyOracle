"""
Scheduled weekly runs for the dashboard (Thursday, Sunday, Tuesday).

  project  python cloud_run.py project  --store store/projections_v5.csv [--season 2026] [--week N]
           Re-projects the week's upcoming games (weekly_v5), keeps the live projections already stored for
           games that have since been played, and writes dash_out/: projections, receivers, RB profiles, scoreboard.
  results  python cloud_run.py results  --store store/projections_v5.csv [--season 2026] [--week N]
           No re-projection: fills actual stats into the stored projections for the finished week and rebuilds
           the scoreboard (Tuesday).
  ros      python cloud_run.py ros      --store dash_out/projections_v5.csv [--frame out_v5/frame.parquet] [--week N]
           Rest-of-season rankings and PPR trade values (ros.py) from the earliest week with an unplayed game:
           writes dash_out/ros_v5.csv, dash_out/trade_values_v5.csv and dash_out/ros_info.json. Uses this week's
           official projections from --store when the week has been projected (Thursday / Sunday runs), else a
           model-only weekly projection (Tuesday). --frame reuses the feature frame the project run just saved.

  Game-day inactives: if ref/inactives_<season>_wk<week>.csv exists (columns player, team, pos; or pass
  --inactives FILE), those players are treated as ruled Out: dropped from the projection run (teammates are
  re-projected without them) and shown as "Inactive" with a projection of 0. --late re-projects only games not
  yet played (the late-Sunday runs after inactives are posted).
  Scoreboard pool: the experts' top 24 QB / 48 RB / 60 WR / 24 TE, minus players ruled inactive on game day
  (neither the experts' Thursday-Friday snapshot nor a fair test knows those in advance).

The week defaults to the current one: the earliest regular-season week with an unplayed game (project), or the
latest week with a played game (results). The store is the dashboard's current projections file.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import inactives as I

FRAME = os.path.join("out_v5", "frame.parquet")

POOL = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}
OUT = "dash_out"
ACT = {"act_ppr": "fantasy_points_ppr", "act_pass_yds": "passing_yards", "act_rush_yds": "rushing_yards",
       "act_rec_yds": "receiving_yards", "act_receptions": "receptions"}
PROJ_COLS = ["injury", "implied_pts", "exp_plays", "xfp", "expert_rank", "proj_ppr", "proj_model", "ppr_low", "ppr_high",
             "pass_yds", "rush_yds", "rec_yds", "receptions"]


def current_week(season, mode):
    g = pd.read_parquet("data/games.parquet")
    g = g[(g["season"] == season) & (g["game_type"] == "REG")]
    if mode == "project":
        return int(g.loc[g["home_score"].isna(), "week"].min())
    return int(g.loc[g["home_score"].notna(), "week"].max())


def injury_label(r):
    if isinstance(r.get("report_status"), str) and r["report_status"]:
        return r["report_status"]
    ps = r.get("practice_status") if isinstance(r.get("practice_status"), str) else ""
    return ("DNP (practice)" if ps.startswith("Did Not") else "Limited (practice)" if ps.startswith("Limited")
            else "Full (practice)" if ps.startswith("Full") else "Not listed")


def to_dash(p):
    """weekly_v5 predictions -> dashboard rows."""
    inj = pd.concat([pd.read_parquet(f, columns=["season", "week", "gsis_id", "report_status", "practice_status"])
                     for f in glob.glob("data/inj/inj_*.parquet")]).rename(columns={"gsis_id": "player_id"})
    inj = inj.drop_duplicates(["player_id", "season", "week"], keep="last")
    p = p.merge(inj, on=["player_id", "season", "week"], how="left")
    o = pd.DataFrame({
        "key": p["player_display_name"] + " (" + p["team"] + ")", "week": p["week"].astype(int),
        "player": p["player_display_name"], "player_id": p["player_id"], "game_id": p["game_id"],
        "pos": p["position"], "team": p["team"], "opp": p["opponent_team"], "venue": p["is_home"].map({1: "Home", 0: "Away"}),
        "status": p["status"].replace({"Final (did not play)": "Did not play"}),
        "injury": [injury_label(r) for r in p[["report_status", "practice_status"]].to_dict("records")],
        "implied_pts": p["implied_pts"].round(1), "exp_plays": p["pace_exp_plays"].round(1), "xfp": p["xfp_ewm"].round(1),
        "expert_rank": p["ecr_week_pos_rank"].round(0),
        "proj_ppr": p["pred_ppr_official"].round(1), "proj_model": p["pred_ppr_model"].round(1),
        "ppr_low": p["pred_ppr_q10"].round(1), "ppr_high": p["pred_ppr_q90"].round(1),
        "pass_yds": p["pred_passing_yards"].where(p["position"] == "QB").round(0),
        "rush_yds": p["pred_rushing_yards"].where(p["position"].isin(["QB", "RB"])).round(0),
        "rec_yds": p["pred_receiving_yards"].where(p["position"] != "QB").round(0),
        "receptions": p["pred_receptions"].where(p["position"] != "QB").round(1)})
    for k, v in ACT.items():
        o[k] = p[f"actual_{v}"].round(1 if k == "act_ppr" else 0)
    o.loc[o["status"] == "Upcoming", list(ACT)] = np.nan
    o.loc[o["pos"] == "QB", ["act_rec_yds", "act_receptions"]] = np.nan
    o.loc[o["pos"].isin(["WR", "TE"]), ["act_pass_yds", "act_rush_yds"]] = np.nan
    o.loc[o["pos"] == "RB", "act_pass_yds"] = np.nan
    o["vs_proj"] = (o["act_ppr"] - o["proj_ppr"]).round(1)
    return o.drop_duplicates(["key", "week"])


def fill_actuals(store, season, week):
    """Actual stats for the week's played games from the latest weekly stats and snap counts."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nfl_player_model as m
    stats, games, players = m.load("data")
    st = stats[(stats["season"] == season) & (stats["week"] == week)].set_index(["player_id", "game_id"])
    ina = I.roster_inactive("data", season, week)
    played_games = set(st.index.get_level_values("game_id"))
    wk = store["week"] == week
    for i in store.index[wk]:
        gid, pid = store.at[i, "game_id"], store.at[i, "player_id"]
        if gid not in played_games:
            continue
        if (pid, gid) in st.index:
            r = st.loc[(pid, gid)]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            for k, v in ACT.items():
                store.at[i, k] = round(float(r[v]), 1 if k == "act_ppr" else 0)
            store.at[i, "status"] = "Final"
        else:
            for k in ACT:
                store.at[i, k] = 0.0
            store.at[i, "status"] = "Inactive" if (store.at[i, "status"] == "Inactive" or pid in ina) else "Did not play"
        pos = store.at[i, "pos"]
        if pos == "QB":
            store.loc[i, ["act_rec_yds", "act_receptions"]] = np.nan
        elif pos in ("WR", "TE"):
            store.loc[i, ["act_pass_yds", "act_rush_yds"]] = np.nan
        else:
            store.at[i, "act_pass_yds"] = np.nan
    store["vs_proj"] = (store["act_ppr"] - store["proj_ppr"]).round(1)
    return store


def relabel_inactive(store, season):
    """Finished-week rows of players listed inactive (INA) on the nflverse weekly roster -> status Inactive."""
    for w in sorted(store.loc[store["status"] == "Did not play", "week"].unique()):
        ina = I.roster_inactive("data", season, int(w))
        hit = (store["week"] == w) & (store["status"] == "Did not play") & store["player_id"].isin(ina)
        store.loc[hit, "status"] = "Inactive"
    return store


def mark_inactive(store, before, ids, week, started):
    """Players ruled inactive for games not yet started: keep their row (from the run before) with status
    Inactive and zero projections."""
    old = before[(before["week"] == week) & before["player_id"].isin(ids) & ~before["game_id"].isin(started)]
    if not len(old):
        return store, 0
    store = store[~((store["week"] == week) & store["player_id"].isin(set(old["player_id"])))]
    old = old.copy()
    old["status"], old["injury"] = "Inactive", "Inactive"
    for c in ["proj_ppr", "proj_model", "ppr_low", "ppr_high", "pass_yds", "rush_yds", "rec_yds", "receptions"]:
        old[c] = old[c].where(old[c].isna(), 0.0)
    return pd.concat([store, old], ignore_index=True), len(old)


def started_games(season, week):
    """Games of the week whose kickoff (schedule time, US Eastern) has passed."""
    g = pd.read_parquet("data/games.parquet")
    g = g[(g["season"] == season) & (g["week"] == week)]
    ko = pd.to_datetime(g["gameday"] + " " + g["gametime"]).dt.tz_localize("America/New_York")
    return set(g.loc[ko <= pd.Timestamp.now(tz="America/New_York"), "game_id"])


def merge_week(store, new, week, keep_games=()):
    """Replace the week's rows with the new run, but keep live projections already stored for games that have
    been played or kicked off since (they were made before kickoff)."""
    if len(keep_games):
        old_rows = store[(store["week"] == week) & store["game_id"].isin(keep_games)]
        new = pd.concat([new[~new["game_id"].isin(keep_games)], old_rows], ignore_index=True)
    keep = store[store["week"] != week]
    old = store[store["week"] == week].set_index(["key", "week"])
    new = new.set_index(["key", "week"])
    played = new.index[new["status"] != "Upcoming"]
    have = played.intersection(old.index)
    new.loc[have, PROJ_COLS] = old.loc[have, PROJ_COLS]
    new["vs_proj"] = (new["act_ppr"] - new["proj_ppr"]).round(1)
    return pd.concat([keep, new.reset_index()], ignore_index=True)


def scoreboard(store):
    rows = []
    d = store[(store["status"] != "Upcoming") & store["expert_rank"].notna()]
    for (w, pos), g in d.groupby(["week", "pos"]):
        g = g[g["expert_rank"].rank(method="first") <= POOL[pos]]
        g = g[g["status"] != "Inactive"]
        if len(g) < 10 or g["act_ppr"].isna().any():
            continue
        y = g["act_ppr"].values
        rows.append({"week": int(w), "pos": pos, "players": len(g), "model": spearmanr(g["proj_model"], y).correlation,
                     "experts": spearmanr(-g["expert_rank"], y).correlation,
                     "official": spearmanr(g["proj_ppr"], y).correlation})
    sc = pd.DataFrame(rows)
    if not len(sc):
        return sc
    agg = {"model": "mean", "experts": "mean", "official": "mean", "players": "sum"}
    allp = sc.groupby("week", as_index=False).agg(agg).assign(pos="All")
    tot = sc.groupby("pos", as_index=False).agg(agg).assign(week=0)
    tot_all = pd.DataFrame([{"week": 0, "pos": "All", "players": int(sc["players"].sum()), "model": sc["model"].mean(),
                             "experts": sc["experts"].mean(), "official": sc["official"].mean()}])
    sc = pd.concat([sc, allp, tot, tot_all], ignore_index=True)
    sc["key"] = np.where(sc["week"] == 0, "Season", "Week " + sc["week"].astype(str)) + " · " + sc["pos"]
    sc["official_vs_experts"] = sc["official"] - sc["experts"]
    return sc[["key", "week", "pos", "players", "model", "experts", "official", "official_vs_experts"]].round(4)


def side_tables(p):
    rc = p[p["position"].isin(["WR", "TE", "RB"]) & p["rc_route_participation"].notna()]
    rec = pd.DataFrame({"key": rc["player_display_name"] + " (" + rc["team"] + ")", "pos": rc["position"],
                        "opp": rc["opponent_team"], "status": rc["status"],
                        "route_participation": rc["rc_route_participation"].round(3), "tprr": rc["rc_tprr"].round(3),
                        "yprr": rc["rc_yprr"].round(2), "adot": rc["rc_adot"].round(1),
                        "first_read_share": rc["rc_first_read_share"].round(3), "xfp": rc["xfp_ewm"].round(1),
                        "td_luck": rc["xfp_td_luck_ewm"].round(2), "proj_rec_yds": rc["pred_receiving_yards"].round(0),
                        "act_rec_yds": rc["actual_receiving_yards"].round(0)})
    rec = rec[(rec["route_participation"] >= 0.2) | (rec["xfp"] >= 6)].drop_duplicates("key")
    rb = p[(p["position"] == "RB") & p["ru_ypc"].notna() & (p["carries_ewm"] >= 5)]
    rbp = pd.DataFrame({"key": rb["player_display_name"] + " (" + rb["team"] + ")", "team": rb["team"],
                        "opp": rb["opponent_team"], "ypc": rb["ru_ypc"].round(2),
                        "explosive_rate": rb["ru_explosive_rate"].round(3), "gump_rate": rb["ru_gump_rate"].round(3),
                        "stuff_rate": rb["ru_stuff_rate"].round(3), "epa_per_carry": rb["ru_epa_per_carry"].round(3),
                        "goal_line_share": rb["rz_goal_line_carry_share"].round(3),
                        "opp_two_high": rb["def_two_high_rate"].round(3),
                        "opp_explosive_allowed": rb["def_explosive_run_rate"].round(3),
                        "proj_rush_yds": rb["pred_rushing_yards"].round(0)}).drop_duplicates("key")
    return rec, rbp


def ros_run(a):
    """Rest-of-season rankings and trade values (ros.py)."""
    import features_v6 as f6
    import nfl_player_model as m
    import ros as R
    cfg = json.load(open("v5_config.json"))
    W = a.week or current_week(a.season, "project")
    import time
    import modelcache as MC
    t0 = time.time()
    info = {"mode": "ros", "season": a.season, "week": W}
    if W > R.last_week(a.season):
        info["skipped"] = f"fantasy season over (last fantasy week {R.last_week(a.season)})"
        json.dump(info, open(os.path.join(OUT, "ros_info.json"), "w"), indent=1)
        print(json.dumps(info))
        return
    df = None
    if a.frame and os.path.exists(a.frame):
        df = pd.read_parquet(a.frame)
        up = sorted(df.loc[df["upcoming"].eq(True), "week"].unique()) if "upcoming" in df else []
        if not (df["season"].max() == a.season and (not up or up == [W])):
            df = None
    if df is None:
        df = m.build_frame("data", a.season, W)
    df = df.drop(columns=[c for c in list(f6.MARKET) + ["qb_prior_starts"] if c in df.columns])
    df = df.join(f6.build_market(df, "data"))
    df["qb_prior_starts"] = f6.qb_prior_starts(df, "data")
    for c in ["phantom", "snap_only", "upcoming"]:
        df[c] = df[c].eq(True)
    stats, games, _ = m.load("data")
    ctx = R.Context("data", os.environ.get("FP_DATA", "fpdata"), stats, games)
    store = pd.read_csv(a.store) if a.store and os.path.exists(a.store) else None
    snap, rinfo = R.project_live(df, ctx, a.season, W, cfg, store)
    ros, tv = R.dash_rows(snap, ctx, a.season, W, cfg["ros"])
    ros.to_csv(os.path.join(OUT, "ros_v5.csv"), index=False)
    tv.to_csv(os.path.join(OUT, "trade_values_v5.csv"), index=False)
    info.update(rinfo)
    info["seconds"] = round(time.time() - t0)
    info["model_cache"] = dict(MC.stats, on=MC.ON)
    info["top"] = {p: g.nsmallest(3, "ros_rank")[["player", "ros_ppr"]].values.tolist()
                   for p, g in ros.groupby("pos")}
    json.dump(info, open(os.path.join(OUT, "ros_info.json"), "w"), indent=1, default=str)
    print(json.dumps(info, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["project", "results", "ros"])
    ap.add_argument("--store", required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--inactives", default=None, help="CSV of game-day inactives (player, team, pos)")
    ap.add_argument("--late", action="store_true", help="re-project only games not yet played")
    ap.add_argument("--frame", default=None, help="ros: feature frame saved by the project run")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "ros":
        return ros_run(a)
    W = a.week or current_week(a.season, a.mode)
    store = pd.read_csv(a.store)
    info = {"mode": a.mode, "season": a.season, "week": W}
    if a.mode == "project":
        env = dict(os.environ, FP_DATA=os.environ.get("FP_DATA", "fpdata"))
        ipath = a.inactives or f"ref/inactives_{a.season}_wk{W}.csv"
        listed = I.roster_inactive_rows("data", a.season, W)
        info["inactives"] = {"nflverse_roster": int(len(listed))}
        if os.path.exists(ipath):
            matched, unmatched = I.match(ipath, "data", a.season, W)
            info["inactives"].update(file=ipath, matched=int(len(matched)), unmatched=unmatched)
            listed = pd.concat([matched, listed]).drop_duplicates("player_id")
        if len(listed):
            I.apply(listed, "data", a.season, W)
        ids = set(listed["player_id"])
        info["inactives"]["total"] = len(ids)
        before = store.copy()
        if os.path.exists(FRAME):   # never reuse a frame from an earlier run
            os.remove(FRAME)
        subprocess.run([sys.executable, "weekly_v5.py", "--season", str(a.season), "--week", str(W),
                        "--config", "v5_config.json", "--out", "out_v5", "--frame", FRAME]
                       + (["--upcoming-only"] if a.late else []), check=True, env=env)
        p = pd.read_csv(f"out_v5/predictions_{a.season}_wk{W}.csv")
        started = started_games(a.season, W)
        if a.late:   # games outside this run (already played) keep their stored rows
            keep = (set(store.loc[store["week"] == W, "game_id"]) - set(p["game_id"])) | started
        else:
            keep = started - set(p.loc[p["status"] != "Upcoming", "game_id"])
        store = merge_week(store, to_dash(p), W, keep)
        store, n_ina = mark_inactive(store, before, ids, W, started)
        info["inactive_rows"] = n_ina
        rec, rbp = side_tables(p)
        rec.to_csv(os.path.join(OUT, "receivers_v5.csv"), index=False)
        rbp.to_csv(os.path.join(OUT, "rb_profiles_v5.csv"), index=False)
        info.update(players=int((store["week"] == W).sum()), upcoming=int(((store["week"] == W) & (store["status"] == "Upcoming")).sum()))
    else:
        store = fill_actuals(store, a.season, W)
        info.update(final=int(((store["week"] == W) & (store["status"] != "Upcoming")).sum()))
    store = relabel_inactive(store, a.season)
    store = store.sort_values(["week", "proj_ppr"], ascending=[True, False])
    store.to_csv(os.path.join(OUT, "projections_v5.csv"), index=False)
    sc = scoreboard(store)
    sc.to_csv(os.path.join(OUT, "scoreboard_v5.csv"), index=False)
    if len(sc):
        s = sc[(sc["week"] == 0) & (sc["pos"] == "All")].iloc[0]
        info["season_scoreboard"] = {k: round(float(s[k]), 4) for k in ["model", "experts", "official"]}
        lw = sc[(sc["pos"] == "All") & (sc["week"] > 0)].sort_values("week").iloc[-1]
        info["latest_week_scoreboard"] = {"week": int(lw["week"]), **{k: round(float(lw[k]), 4) for k in ["model", "experts", "official"]}}
    json.dump(info, open(os.path.join(OUT, "run_info.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()

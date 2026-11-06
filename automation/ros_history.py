"""Rest-of-season ranking history: one snapshot of the ROS rankings per week of the season, so the site (and future
model work) can see how each player's rest-of-season outlook moved as the season went on.

  python automation/ros_history.py append docs/data            # after every model update: this week's ROS -> history
  python automation/ros_history.py backfill 1 8 --season 2026  # once: re-create Weeks 1-8 (run from the repo root)

append: the week's row set is replaced by the latest run, so each week keeps the last ROS ranking made that week
  (the one made before that week's games, refreshed for news through Sunday's inactives).
backfill: runs the ROS model as of each earlier week (model/cloud_run.py ros --week W): models trained only on earlier
  seasons, the expert ROS snapshot of that date and that week's official weekly projection (made before kickoff).
  Rosters and injury labels come from the data as it is now, so these weeks are re-created after the fact; the
  history marks them source="re-created".
Output: docs/data/ros_history_v5.csv (week, source, snapshot time and the main ROS columns) and its datasets.json entry.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "model")
SITE = os.path.join(ROOT, "docs", "data")
FILE = "ros_history_v5.csv"
COLS = ["week", "source", "snapshot_at", "key", "player", "player_id", "pos", "team", "status", "injury", "ros_rank",
        "ros_ppr", "ros_low", "ros_high", "ros_model", "model_rank", "expert_rank", "games_left", "ros_ppg",
        "season_ppg", "season_games", "bye_week", "ros_start_pct", "ros_confidence"]
DESC = ("Rest-of-season rankings as they stood each week of the season: rank within the position, ROS PPR points "
        "and range, model and expert ranks. Weeks from 9 on are the live ranking (the last run of that week); "
        "earlier weeks were re-created after the fact by the same ROS model as of that week (models trained on "
        "earlier seasons, that date's expert ROS ranking and that week's pre-kickoff projection).")


def merge(site, rows, week, source):
    path = os.path.join(site, FILE)
    rows = rows.copy()
    rows["week"], rows["source"] = int(week), source
    rows["snapshot_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for c in COLS:
        if c not in rows:
            rows[c] = None
    rows = rows[COLS]
    if os.path.exists(path):
        old = pd.read_csv(path)
        # a live snapshot replaces whatever the week had; a re-created one never overwrites a live one
        if source != "live" and ((old["week"] == week) & (old["source"] == "live")).any():
            return False
        rows = pd.concat([old[old["week"] != week], rows], ignore_index=True)
    rows.sort_values(["week", "pos", "ros_rank"]).to_csv(path, index=False)
    cat_path = os.path.join(site, "datasets.json")
    cat = json.load(open(cat_path))
    cat["datasets"]["ros_history"] = {"file": FILE, "title": "ROS rankings by week", "description": DESC,
                                      "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    json.dump(cat, open(cat_path, "w"), indent=1)
    return True


def append(site):
    ros = pd.read_csv(os.path.join(site, "ros_v5.csv"))
    if ros.empty:
        return
    week = int(ros["from_week"].iloc[0])
    merge(site, ros, week, "live")
    print(f"ros history: week {week} live snapshot, {len(ros)} players")


def backfill(first, last, season):
    store = os.path.join(SITE, "projections_v5.csv")
    keep = os.path.join(MODEL, "dash_out", "ros_v5.keep.csv")
    cur = os.path.join(MODEL, "dash_out", "ros_v5.csv")
    if os.path.exists(cur):
        shutil.copy(cur, keep)
    env = dict(os.environ, FP_DATA="fpdata")
    for w in range(first, last + 1):
        r = subprocess.run([sys.executable, "cloud_run.py", "ros", "--store", store, "--season", str(season),
                            "--week", str(w)], cwd=MODEL, env=env)
        if r.returncode != 0:
            print(f"week {w}: ROS run failed", flush=True)
            continue
        ros = pd.read_csv(cur)
        # start % for these rows, the same rule as the live site (confidence.py's ROS part)
        try:
            sys.path.insert(0, os.path.join(ROOT, "automation"))
            import confidence as C
            th = {pos: g.nlargest(C.N.get(pos, 12)).min() for pos, g in ros.groupby("pos")["ros_ppr"]}
            rs = C.chance(ros["ros_ppr"], ros["ros_low"], ros["ros_high"], ros["pos"].map(th), floor=2.0)
            ros["ros_start_pct"], ros["ros_confidence"] = (rs * 100).round(), C.label(rs)
        except Exception as e:   # optional
            print("start % skipped:", e)
        ok = merge(SITE, ros, w, "re-created")
        print(f"week {w}: {len(ros)} players {'saved' if ok else 'kept the live snapshot'}", flush=True)
    if os.path.exists(keep):
        shutil.move(keep, cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["append", "backfill"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    if a.mode == "append":
        append(a.args[0] if a.args else SITE)
    else:
        backfill(int(a.args[0]), int(a.args[1]), a.season)


if __name__ == "__main__":
    main()

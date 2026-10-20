"""One automatic update: fresh data -> results for finished games -> this week's projections -> rest of season ->
the website's data files. Run from the repository root (GitHub Actions does this when watch.py sees new data).

  python automation/pipeline.py               # season worked out from today's date
  python automation/pipeline.py --skip-download

Model code lives in model/ (unchanged commands: cloud_run.py results / project / ros). The site's current projections
file (docs/data/projections_v5.csv) is the model's store: games already played keep the projection they had before
kickoff, exactly as on the claude.ai dashboard.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "model")
SITE = os.path.join(ROOT, "docs", "data")
OUT = os.path.join(MODEL, "dash_out")
# model output file -> dataset id on the site
PUBLISH = {"projections_v5.csv": "projections", "scoreboard_v5.csv": "scoreboard", "receivers_v5.csv": "receivers",
           "rb_profiles_v5.csv": "rb_profiles", "ros_v5.csv": "ros", "trade_values_v5.csv": "trade_values"}


def run(cmd, log):
    t0 = time.time()
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=MODEL, env=dict(os.environ, FP_DATA="fpdata"))
    log.append({"step": " ".join(cmd[1:3]) if cmd[0] == sys.executable else cmd[0], "seconds": round(time.time() - t0),
                "ok": r.returncode == 0})
    if r.returncode:
        raise SystemExit(f"step failed: {' '.join(cmd)}")


def weeks(season):
    g = pd.read_parquet(os.path.join(MODEL, "data", "games.parquet"))
    g = g[(g["season"] == season) & (g["game_type"] == "REG")]
    unplayed = g.loc[g["home_score"].isna(), "week"]
    played = g.loc[g["home_score"].notna(), "week"]
    w_proj = int(unplayed.min()) if len(unplayed) else None
    w_res = int(played.max()) if len(played) else None
    started = False
    if w_proj is not None:
        x = g[g["week"] == w_proj]
        ko = pd.to_datetime(x["gameday"] + " " + x["gametime"]).dt.tz_localize("America/New_York")
        started = bool((ko <= pd.Timestamp.now(tz="America/New_York")).any())
    return w_proj, w_res, started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    today = dt.date.today()
    season = a.season or (today.year if today.month >= 3 else today.year - 1)
    log = []
    if not a.skip_download:
        run(["bash", "download_data.sh", str(season)], log)
    w_proj, w_res, started = weeks(season)
    os.makedirs(OUT, exist_ok=True)
    store = os.path.join(MODEL, "store.csv")
    shutil.copy(os.path.join(SITE, "projections_v5.csv"), store)

    # finished games: fill in actual stats (the Tuesday step) whenever the latest played week is on the site
    if w_res is not None and (pd.read_csv(store, usecols=["week"])["week"] == w_res).any():
        run([sys.executable, "cloud_run.py", "results", "--store", store, "--season", str(season), "--week", str(w_res)],
            log)
        shutil.copy(os.path.join(OUT, "projections_v5.csv"), store)

    if w_proj is not None:
        run([sys.executable, os.path.join(ROOT, "automation", "weather.py"), "--data", "data", "--season", str(season),
             "--week", str(w_proj), "--out", "ref"], log)
        run([sys.executable, os.path.join(ROOT, "automation", "injuries_live.py"), "--data", "data", "--season",
             str(season), "--week", str(w_proj)], log)
        run([sys.executable, os.path.join(ROOT, "automation", "inactives_nfl.py"), "--season", str(season),
             "--week", str(w_proj), "--out", "ref"], log)
        run([sys.executable, "cloud_run.py", "project", "--store", store, "--season", str(season), "--week", str(w_proj)]
            + (["--late"] if started else []), log)
        run([sys.executable, "cloud_run.py", "ros", "--store", os.path.join("dash_out", "projections_v5.csv"),
             "--frame", os.path.join("out_v5", "frame.parquet"), "--season", str(season), "--week", str(w_proj)], log)

    # publish to the site
    cat_path = os.path.join(SITE, "datasets.json")
    cat = json.load(open(cat_path))
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for f, did in PUBLISH.items():
        src = os.path.join(OUT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SITE, f))
            cat["datasets"].setdefault(did, {"file": f, "title": did})["updated"] = now
    subprocess.run([sys.executable, os.path.join(ROOT, "automation", "confidence.py"), SITE], check=True)
    cat["updated"] = now
    cat["note"] = f"Week {w_proj} projections" if w_proj else "season complete"
    json.dump(cat, open(cat_path, "w"), indent=1)
    if w_proj is not None:
        json.dump({"week": w_proj}, open(os.path.join(SITE, "params.json"), "w"))
    info = {"updated": now, "season": season, "projection_week": w_proj, "results_week": w_res,
            "games_started": started, "reason": a.reason, "steps": log}
    for f in ["run_info.json", "ros_info.json"]:
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            info[f[:-5]] = json.load(open(p))
    json.dump(info, open(os.path.join(SITE, "run_info.json"), "w"), indent=1, default=str)
    print(json.dumps({k: info[k] for k in ["updated", "projection_week", "results_week", "steps"]}, default=str))


if __name__ == "__main__":
    main()

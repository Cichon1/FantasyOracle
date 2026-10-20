"""Has any data the model uses changed since the last update?

Asks each free source for its last-modified stamp (a HEAD request, about a second in all), compares with
automation/state/sources.json from the last update, and prints which sources changed. Writes GitHub Actions outputs
`changed` (true/false) and `what` (comma-separated names). The new stamps go to automation/state/sources.new.json; the
workflow keeps them only after the update succeeds, so a failed run is retried at the next check.

  python automation/watch.py --season 2026
"""
import argparse
import datetime
import hashlib
import sys
from zoneinfo import ZoneInfo
import json
import os
import urllib.request

NV = "https://github.com/nflverse/nflverse-data/releases/download"
FFO = "https://github.com/ffverse/ffopportunity/releases/download/latest-data"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")


def sources(season):
    s = {
        "player_stats": f"{NV}/stats_player/stats_player_week_{season}.parquet",
        "play_by_play": f"{NV}/pbp/play_by_play_{season}.parquet",
        "schedule_and_lines": f"{NV}/schedules/games.parquet",
        "injury_reports": f"{NV}/injuries/injuries_{season}.parquet",
        "depth_charts": f"{NV}/depth_charts/depth_charts_{season}.parquet",
        "rosters": f"{NV}/weekly_rosters/roster_weekly_{season}.parquet",
        "snap_counts": f"{NV}/snap_counts/snap_counts_{season}.parquet",
        "ftn_charting": f"{NV}/ftn_charting/ftn_charting_{season}.parquet",
        "expected_points": f"{FFO}/ep_weekly_{season}.parquet",
    }
    if os.environ.get("HUB_USE_FANTASYPROS", "1") == "1":   # the model's expert-rank inputs (DynastyProcess copy)
        s["expert_rankings"] = ("https://api.github.com/repos/dynastyprocess/data/commits"
                                "?path=files/db_fpecr.parquet&per_page=1")
    return s


def stamp(url):
    if "api.github.com" in url:
        h = {"Accept": "application/vnd.github+json"}
        if os.environ.get("GITHUB_TOKEN"):
            h["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
            j = json.load(r)
        return j[0]["sha"] if j else None
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:  # follows the redirect
        return r.headers.get("Last-Modified") or r.headers.get("ETag")


def current_week(season):
    """Earliest regular-season week with an unplayed game, from the schedule file (fetched if not downloaded)."""
    import io
    import pandas as pd
    p = os.path.join(os.path.dirname(STATE), "..", "model", "data", "games.parquet")
    if os.path.exists(p):
        g = pd.read_parquet(p)
    else:
        with urllib.request.urlopen(f"{NV}/schedules/games.parquet", timeout=60) as r:
            g = pd.read_parquet(io.BytesIO(r.read()))
    g = g[(g["season"] == season) & (g["game_type"] == "REG") & g["home_score"].isna()]
    return int(g["week"].min())


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(STATE, exist_ok=True)
    path = os.path.join(STATE, "sources.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    new, changed = {}, []
    for name, url in sources(a.season).items():
        try:
            new[name] = stamp(url)
        except Exception as e:                  # a source that doesn't answer: keep its old stamp, try next time
            print(f"{name}: no answer ({e})")
            new[name] = old.get(name)
            continue
        if new[name] != old.get(name):
            changed.append(name)
    # game days (Thursday, Sunday, Monday, US Eastern): NFL.com's inactives articles, as a hash of the list
    now = datetime.datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() in (3, 6, 0):
        try:
            import inactives_nfl as IN
            wk = current_week(a.season)
            rows = []
            for u in IN.article_urls(a.season, wk):
                try:
                    rows += IN.parse(IN.fetch(u))
                except Exception:
                    pass
            stamp_i = hashlib.sha1(json.dumps(sorted({(r["player"], r["team"]) for r in rows})).encode()).hexdigest()[:12] \
                if rows else None
            new["inactives"] = stamp_i or old.get("inactives")
            if stamp_i and stamp_i != old.get("inactives"):
                changed.append("inactives")
        except Exception as e:
            print(f"inactives: no answer ({e})")
    json.dump(new, open(os.path.join(STATE, "sources.new.json"), "w"), indent=1, sort_keys=True)
    go = bool(changed) or a.force
    print("changed:", ", ".join(changed) or "nothing")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"changed={'true' if go else 'false'}\n")
            f.write(f"what={','.join(changed) or ('manual' if a.force else '')}\n")


if __name__ == "__main__":
    main()

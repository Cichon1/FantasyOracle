"""This week's Sleeper and ESPN PPR projections, pulled live for the official weekly blend (Nov 22, 2026).

The league sims (claude/league-sim-2026-11-17.md) found the best weekly ranking was model + Sleeper + ESPN (ranks
averaged within position, the three sources' average point curve handed out in that order): +0.24 wins a season over
FantasyPros, vs +0.20 for model + FantasyPros. The pipeline calls this before the projection run; weekly_v5.py
reads the files and falls back to the FantasyPros blend when a source is missing or too thin.

  python automation/other_live.py --season 2026 --week 12

Writes automation/history/other_proj/live/{sleeper,espn}_<season>_wk<week>.csv (latest pull, overwritten; git keeps
the history) and live/pull_log.json. Never fails the update.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "automation"))
import other_projections as O  # noqa: E402

LIVE = os.path.join(ROOT, "automation", "history", "other_proj", "live")
MIN_ROWS = 150      # fewer QB/RB/WR/TE rows than this = the source isn't serving this week yet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(LIVE, exist_ok=True)
    logp = os.path.join(LIVE, "pull_log.json")
    try:
        log = json.load(open(logp))
    except Exception:
        log = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, fn in [("sleeper", O.sleeper), ("espn", O.espn)]:
        entry = {"at": now, "season": a.season, "week": a.week}
        try:
            rows = fn(a.season, [a.week])
            errs = [r["_err"] for r in rows if "_err" in r]
            df = pd.DataFrame([r for r in rows if "_err" not in r])
            if len(df):
                df = df[df["pos"].isin(O.POS)]
            entry.update(rows=int(len(df)), errors=errs[:3])
            if len(df) >= MIN_ROWS:
                df["pulled_at"] = now
                df.to_csv(os.path.join(LIVE, f"{name}_{a.season}_wk{a.week}.csv"), index=False)
                entry["saved"] = True
            else:
                entry["saved"] = False
        except Exception as e:      # noqa: BLE001 - never fail the update
            entry["error"] = str(e)[:300]
        log[name] = entry
        print(name, entry)
    json.dump(log, open(logp, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Update clock for game days (Nov 22, 2026).

Why: GitHub runs scheduled workflows only when it has spare capacity. Since Nov 19 the 30-minute "Update projections"
schedule has fired 2-6 times a day instead of 48 (before that about 23), and on Sunday Nov 22 no update ran between
10:07 am CT and the evening, so none of that day's game-day inactives reached the projections or start/sit.

What it does: on game days it starts the update workflow itself, every 10 minutes in the 2 hours before each kickoff
(inactives post about 90 minutes before each kickoff window) and every 30 minutes otherwise, from 10 hours before the
first kickoff until the last one. Away from game days it exits at once, so it only holds a
runner near games. A started update still checks whether any source changed (watch.py) and skips the model if
nothing did, exactly as a scheduled run does.

A GitHub job can run at most 6 hours, so the clock stops after --minutes and, if a game window is still ahead or
under way, starts its own next run (workflow_dispatch from GITHUB_TOKEN is allowed to start workflows). Its own
hourly schedule and every update run also try to start it, so a missed hand-off is picked up again; the workflow's
concurrency group keeps it to one clock at a time.

  python automation/clock.py --minutes 345          # in .github/workflows/clock.yml
  python automation/clock.py --kick                 # end of every update run: start the clock if a game is near
  python automation/clock.py --plan                 # print today's windows and what the clock would do now
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

import pandas as pd

GAMES = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
ET = "America/New_York"
LEAD_FAST = pd.Timedelta(minutes=120)      # inactives post ~90 min before kickoff
LEAD_SLOW = pd.Timedelta(hours=8)          # 30-minute updates before that (Sunday-morning injury news, FP ranks)
FAST, SLOW = 10, 30                         # minutes between update starts


def kickoffs(path=None):
    if path and os.path.exists(path):
        g = pd.read_parquet(path)
    else:
        with urllib.request.urlopen(GAMES, timeout=60) as r:
            g = pd.read_parquet(io.BytesIO(r.read()))
    g = g[g["game_type"] == "REG"].dropna(subset=["gameday", "gametime"])
    ko = pd.to_datetime(g["gameday"] + " " + g["gametime"]).dt.tz_localize(ET)
    return sorted(ko.unique())


def windows(kos):
    """One window per game day (ET date): (slow start, first kickoff, last kickoff, that day's kickoffs)."""
    by_day = {}
    for k in kos:
        by_day.setdefault(k.date(), set()).add(k)
    return [(min(v) - LEAD_FAST - LEAD_SLOW, min(v), max(v), sorted(v)) for _, v in sorted(by_day.items())]


def phase(now, wins):
    """'fast' within 2 hours before any kickoff, 'slow' elsewhere in a game day's window, else 'idle'."""
    for slow, first, last, kos in wins:
        if any(k - LEAD_FAST <= now <= k for k in kos):
            return "fast"
        if slow <= now <= last:
            return "slow"
    return "idle"


# workflows sharing the "update" concurrency group: starting an update while one of them waits would cancel the
# waiting run (GitHub keeps one pending run per group), e.g. the Sunday-morning FantasyPros pull
GROUP = {"Update projections", "Expert rankings", "Prop lines", "ROS history backfill", "Other sites' projections",
         "FantasyPros API check"}


def busy(dry=False):
    """True while any run of the update group is queued, waiting or running."""
    if dry:
        return False
    r = subprocess.run(["gh", "run", "list", "--limit", "30", "--json", "workflowName,status"], capture_output=True,
                       text=True)
    if r.returncode:
        return False
    return any(x["workflowName"] in GROUP and x["status"] != "completed" for x in json.loads(r.stdout or "[]"))


def gh(*args, dry=False):
    cmd = ["gh", "workflow", "run", *args]
    print(pd.Timestamp.now(tz=ET).strftime("%a %H:%M"), "$", " ".join(cmd), flush=True)
    if dry:
        return True
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("  failed:", (r.stderr or r.stdout).strip()[:300], flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=345)
    ap.add_argument("--ref", default=os.environ.get("GITHUB_REF_NAME", "main"))
    ap.add_argument("--games", default=None, help="local games.parquet (default: download nflverse's)")
    ap.add_argument("--plan", action="store_true", help="print the windows and the current phase, start nothing")
    ap.add_argument("--dry", action="store_true", help="loop but only print the commands")
    ap.add_argument("--kick", action="store_true", help="start a clock run if a game window is near (update.yml)")
    a = ap.parse_args()
    wins = windows(kickoffs(a.games))
    now = pd.Timestamp.now(tz=ET)
    ph = phase(now, wins)
    if a.plan:
        for slow, first, last, kos in wins:
            if last >= now - pd.Timedelta(days=1) and slow <= now + pd.Timedelta(days=8):
                print(f"{first:%a %b %d}: updates every 30 min from {slow:%H:%M}, every 10 min in the 2 hours before "
                      f"kickoffs at {', '.join(f'{k:%H:%M}' for k in kos)} ET")
        print("now:", now.strftime("%a %H:%M ET"), "->", ph)
        return
    if a.kick:
        if ph != "idle":
            gh("clock.yml", "--ref", a.ref, dry=a.dry)   # one clock at a time (concurrency group); extra starts queue
        return
    if ph == "idle":
        print("no game window in the next", LEAD_SLOW + LEAD_FAST, "- nothing to do")
        return
    stop = now + pd.Timedelta(minutes=a.minutes)
    nxt = now
    while True:
        now = pd.Timestamp.now(tz=ET)
        ph = phase(now, wins)
        if ph == "idle" or now >= stop:
            break
        if now >= nxt and not busy(a.dry):      # busy: try again next tick, right after the running one ends
            gh("update.yml", "--ref", a.ref, "-f", "force=false", dry=a.dry)
            nxt = now + pd.Timedelta(minutes=FAST if ph == "fast" else SLOW)
        time.sleep(30)
    # hand off to the next clock run while a window is still under way or close
    if phase(pd.Timestamp.now(tz=ET) + pd.Timedelta(minutes=1), wins) != "idle":
        gh("clock.yml", "--ref", a.ref, dry=a.dry)
    print("clock done", flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""Input-freshness guard for the rest-of-season (ROS) values (Nov 23, 2026).

Fails (exit 1) when the latest model run projected any player to play from inputs that leave out a game he has
already played - the problem that hid Blake Corum's 20.9 and Kyren Williams' 5.2 (Week 10) for LA's whole Week 11
bye, and every player's latest game from final whistle until the Tuesday rollover. ros.refresh_next_rows prevents
it; ros.freshness counts any player it missed, and this check reads that count from the published run info.

Also fails when the ROS step ran but reported no freshness count (the check itself was removed or broke).
Out players and manual return estimates are reported, not failed: the return-timing model prices their missed games.

  python automation/freshness_check.py docs/data
Run by .github/workflows/update.yml (job "freshness") after every published update; a failure there marks the run
red and GitHub emails the repo owner, without holding back the site update.
"""
import json
import os
import sys


def main(site):
    info = json.load(open(os.path.join(site, "run_info.json")))
    ros = info.get("ros_info")
    if not ros:
        print("freshness: no ROS step in this run - nothing to check")
        return 0
    if ros.get("skipped"):
        print(f"freshness: ROS skipped ({ros['skipped']})")
        return 0
    f = ros.get("freshness")
    if not f:
        print("FRESHNESS CHECK FAILED: the ROS run reported no freshness count (ros.freshness missing)")
        return 1
    print(f"freshness: week {ros.get('week')}, {f['checked']} players projected to play checked, {f['stale']} stale; "
          f"next-game rows {ros.get('next_game_rows')} (frames for weeks {ros.get('next_frames')}); "
          f"not expected next game: {', '.join(f.get('not_expected_next_game', [])) or 'none'}; "
          f"out players on their last row: {f.get('out_last_row')}; manual return estimates: {f.get('manual_out_skipped')}")
    if f["stale"]:
        print("FRESHNESS CHECK FAILED - ROS inputs leave out a game these players already played:")
        for s in f["stale_players"]:
            print("  " + s)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("docs", "data")))

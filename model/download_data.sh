#!/usr/bin/env bash
# Downloads every data source the model uses into ./data (and FantasyPros consensus history into ./fpdata)
# Usage: bash download_data.sh 2026     (current season; model history from 2018, career stats from 2010)
# Oct 19, 2026: files download 8 at a time (was one by one); same files, same places.
set -e
CUR=${1:-2026}
B=https://github.com/nflverse/nflverse-data/releases/download
FFO=https://github.com/ffverse/ffopportunity/releases/download/latest-data
mkdir -p data/pbp data/part data/ftn data/snaps data/inj data/ngs data/ffo data/depth data/roster fpdata
REQ=$(mktemp); OPT=$(mktemp)
for y in $(seq 2010 "$CUR"); do
  echo "data/week_$y.parquet $B/stats_player/stats_player_week_$y.parquet" >> "$REQ"   # career history
done
for y in $(seq 2018 "$CUR"); do
  echo "data/pbp/pbp_$y.parquet $B/pbp/play_by_play_$y.parquet" >> "$REQ"
  echo "data/part/part_$y.parquet $B/pbp_participation/pbp_participation_$y.parquet" >> "$OPT"  # coverage/personnel/routes; lags a season
  echo "data/ftn/ftn_$y.parquet $B/ftn_charting/ftn_charting_$y.parquet" >> "$OPT"               # 2022+
  echo "data/snaps/snaps_$y.parquet $B/snap_counts/snap_counts_$y.parquet" >> "$REQ"
  echo "data/inj/inj_$y.parquet $B/injuries/injuries_$y.parquet" >> "$REQ"
  echo "data/depth/depth_$y.parquet $B/depth_charts/depth_charts_$y.parquet" >> "$REQ"
  echo "data/ffo/ep_weekly_$y.parquet $FFO/ep_weekly_$y.parquet" >> "$OPT"                       # expected fantasy points
  echo "data/roster/roster_weekly_$y.parquet $B/weekly_rosters/roster_weekly_$y.parquet" >> "$OPT" # IR / reserve status
done
for t in passing rushing receiving; do
  echo "data/ngs/ngs_$t.parquet $B/nextgen_stats/ngs_$t.parquet" >> "$REQ"
done
echo "data/games.parquet $B/schedules/games.parquet" >> "$REQ"
echo "data/players.parquet $B/players/players.parquet" >> "$REQ"
echo "data/contracts.parquet $B/contracts/historical_contracts.parquet" >> "$REQ"
echo "data/combine.parquet $B/combine/combine.parquet" >> "$REQ"
# required files: any failure stops the script (as before); optional ones may be missing upstream
xargs -P 8 -L 1 sh -c 'curl -sfL --retry 5 --retry-all-errors --retry-delay 5 -o "$0" "$1"' < "$REQ"
xargs -P 8 -L 1 sh -c 'curl -sfL --retry 5 --retry-all-errors --retry-delay 5 -o "$0" "$1" || true' < "$OPT"
rm -f "$REQ" "$OPT"
cp data/depth/depth_$CUR.parquet data/depth_raw.parquet
python3 - "$CUR" <<'EOF'
import sys, pandas as pd
cur = sys.argv[1]
d = pd.read_parquet("data/depth_raw.parquet")
d = d[d.dt == d.groupby("team").dt.transform("max")]
d = d[d.pos_abb == "QB"][["team", "player_name", "gsis_id", "pos_rank"]]
d.rename(columns={"gsis_id": "player_id", "pos_rank": "depth"}).to_parquet(f"data/depth_{cur}.parquet")
EOF
# FantasyPros expert consensus history (DynastyProcess, updated weekly)
rm -rf /tmp/dp_data && GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/dynastyprocess/data /tmp/dp_data
cp /tmp/dp_data/files/db_fpecr.parquet /tmp/dp_data/files/db_playerids.csv fpdata/
rm -f data/plays.parquet   # force the play table to rebuild with the new week
echo "done. Before running: put this week's forecast in ref/weather_<season>_wk<week>.csv and update"
echo "ref/playcallers.csv if a team changed play-callers. Run with FP_DATA=fpdata."

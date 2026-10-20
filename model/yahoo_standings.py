"""Standings and regular-season matchups for all three Yahoo leagues (power rankings' playoff predictor, trade
analyzer weeks-that-count). Pulled on request with yahoo_standings.js in Jackson's signed-in Chrome (page text saved
as yahoo/standings_<date>.txt), then:
  python yahoo_standings.py yahoo/standings_<date>.txt dash_out/league_rosters_v5.csv dash_out
-> dash_out/league_standings_v5.csv, dash_out/league_schedule_v5.csv (every league; The League's draft picks still
come from yahoo_picks.js / yahoo_draft.py)."""
import sys
from datetime import datetime, timezone

import pandas as pd


def build(raw, rosters_csv, out_dir):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ro = pd.read_csv(rosters_csv, dtype=str)
    names = ro[["league_id", "team_id", "team"]].drop_duplicates().set_index(["league_id", "team_id"])["team"].to_dict()
    st, sc, lid, part = [], [], None, None
    for line in open(raw, encoding="utf-8").read().split("\n"):
        x = line.split()
        if not x:
            continue
        if x[0] == "LEAGUE":
            lid, part = x[1], None
            continue
        if x[0] in ("STANDINGS", "MATCHUPS"):
            part = x[0]
            continue
        if part == "STANDINGS":
            tid, wlt, pf, pa = x[0], x[-3], x[-2], x[-1]
            w, l, t = map(int, wlt.split("-"))
            st.append(dict(league_id=lid, team_id=tid, wins=w, losses=l, ties=t, points_for=float(pf),
                           points_against=float(pa), team=names.get((lid, tid), "")))
        elif part == "MATCHUPS":
            nums = [float(v) for v in x[3:]] + [0, 0, 0, 0]
            sc.append(dict(league_id=lid, week=int(x[0]), team1=x[1], team2=x[2], points1=nums[0], proj1=nums[1],
                           points2=nums[2], proj2=nums[3]))
    st, sc = pd.DataFrame(st), pd.DataFrame(sc)
    st["games"] = st.wins + st.losses + st.ties
    st["pulled_at"] = now
    played = st.groupby("league_id")["games"].max()
    sc["status"] = ["final" if w <= played[l] else ("in progress" if (a > 0 or b > 0) else "upcoming")
                    for l, w, a, b in zip(sc.league_id, sc.week, sc.points1, sc.points2)]
    for l, g in st.groupby("league_id"):
        n = len(g)
        assert n >= 4 and g.team.ne("").all(), f"league {l}: team names missing"
        m = sc[sc.league_id == l]
        assert (m.groupby("week").size() == n // 2).all(), f"league {l}: matchup count per week looks wrong"
        print(l, n, "teams,", int(played[l]), "weeks final,", len(m), "matchups")
    st.to_csv(f"{out_dir}/league_standings_v5.csv", index=False)
    sc.to_csv(f"{out_dir}/league_schedule_v5.csv", index=False)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3])

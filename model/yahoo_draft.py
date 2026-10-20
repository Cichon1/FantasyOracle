"""The League (Yahoo 200436): standings, weekly matchups and next season's draft pick ownership, for the trade
analyzer's draft-pick feature. Pulled on request with yahoo_picks.js in Jackson's signed-in Chrome (text saved as
yahoo/the_league_<date>.txt), then:  python yahoo_draft.py yahoo/the_league_<date>.txt dash_out
-> dash_out/league_picks_v5.csv (standings and schedules for all leagues: yahoo_standings.js / yahoo_standings.py).
Draft rules (Jackson, Oct 12, 2026): next season's order follows this season's finish (last place picks first),
snake, 16 rounds, 1 keeper per team (keepers take the round-16 pick), every pick tradeable but only pick-for-pick."""
import sys
from datetime import datetime, timezone
import pandas as pd

LID, SEASON_NEXT, ROUNDS = "200436", 2027, 16


def build(raw, out_dir):
    txt = open(raw, encoding="utf-8").read()
    st_txt, rest = txt.split("STANDINGS\n")[1].split("MATCHUPS\n")
    m_txt, p_txt = rest.split("PICKS")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    teams = pd.read_csv(f"{out_dir}/league_rosters_v5.csv", dtype=str)
    teams = teams[teams.league_id == LID][["team_id", "team"]].drop_duplicates()
    name_to_id = dict(zip(teams.team, teams.team_id))
    st = []
    for line in st_txt.strip().split("\n"):
        tid, wlt, pf, pa = line.split("\t")[0], *line.split("\t")[-3:]
        w, l, t = map(int, wlt.split("-"))
        st.append(dict(league_id=LID, team_id=tid, wins=w, losses=l, ties=t, points_for=float(pf), points_against=float(pa)))
    st = pd.DataFrame(st)
    st["team"] = st.team_id.map(dict(zip(teams.team_id, teams.team)))
    st["games"] = st.wins + st.losses + st.ties
    st["pulled_at"] = now
    sc = []
    for line in m_txt.strip().split("\n"):
        w, a, b, nums = line.split("\t")
        x = [float(v) for v in nums.split()]
        a1, p1, a2, p2 = (x + [0, 0, 0, 0])[:4]
        sc.append(dict(league_id=LID, week=int(w), team1=a, team2=b, points1=a1, proj1=p1, points2=a2, proj2=p2))
    sc = pd.DataFrame(sc)
    played = int(st.games.max())
    sc["status"] = ["final" if w <= played else ("in progress" if (a1 > 0 or a2 > 0) else "upcoming")
                    for w, a1, a2 in zip(sc.week, sc.points1, sc.points2)]
    pk = []
    p_txt = p_txt.strip()
    if p_txt.startswith("_DEFAULT"):          # no picks traded yet: every team owns its own picks
        for r in range(1, ROUNDS + 1):
            for tid in st.team_id:
                pk.append((r, tid, tid))
    else:
        names = sorted(name_to_id, key=len, reverse=True)
        for line in p_txt.split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            r, owner, owned = int(parts[0]), parts[1].strip(), parts[2].strip()
            owner_id = name_to_id.get(owner, owner)
            while owned:                           # cell text may run several team names together
                hit = next((n for n in names if owned.startswith(n)), None)
                if not hit:
                    break
                pk.append((r, owner_id, name_to_id[hit]))
                owned = owned[len(hit):].strip(" ,")
    pk = pd.DataFrame(pk, columns=["round", "owner_team_id", "orig_team_id"])
    pk.insert(0, "league_id", LID)
    pk.insert(1, "season", SEASON_NEXT)
    pk["key"] = LID + "|" + str(SEASON_NEXT) + "|" + pk["round"].astype(str) + "|" + pk.orig_team_id
    pk["keeper_round"] = (pk["round"] == ROUNDS).astype(int)
    pk["pulled_at"] = now
    # standings / schedule for every league now come from yahoo_standings.py (league_standings_v5.csv,
    # league_schedule_v5.csv); these copies are The League only, kept for checking
    st.to_csv(f"{out_dir}/league_standings_200436_v5.csv", index=False)
    sc.to_csv(f"{out_dir}/league_schedule_200436_v5.csv", index=False)
    pk.to_csv(f"{out_dir}/league_picks_v5.csv", index=False)
    print(len(st), "teams,", played, "weeks final;", len(sc), "matchups;", len(pk), "picks (", (pk.owner_team_id != pk.orig_team_id).sum(), "traded )")
    assert len(pk) == ROUNDS * len(st), "pick count looks wrong"


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])

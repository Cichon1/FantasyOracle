"""
Yahoo league rosters and settings for the trade analyzer (pulled on request, not on a schedule: Yahoo needs
Jackson's sign-in, and the scheduled cloud runs can't reach it).

How a refresh works (from a chat with Claude in Chrome signed in to Yahoo):
  1. On football.fantasysports.yahoo.com run yahoo_rosters.js in the page: it reads each league's Rosters page
     (/f1/<id>/starters, every position incl. K / DEF / IDP and IR slots) and Settings page, and writes
     TEAMS / ROSTERS / SETTINGS text into the page. Read it with get_page_text, then reload the page.
  2. Save that text as yahoo/rosters_raw.txt and run
       python yahoo_league.py yahoo/rosters_raw.txt dash_out/ros_v5.csv dash_out
     -> dash_out/league_rosters_v5.csv (every rostered player; QB/RB/WR/TE matched to the ROS rankings by name and
     position) and dash_out/leagues_v5.csv (lineup slots, bench and IR sizes, scoring that changes player values).
"""
import re
import sys
import unicodedata
from datetime import datetime, timezone

import pandas as pd

MY_TEAM = {"126912": "1", "200436": "3", "727092": "4"}   # Jackson's team in each league
SHORT = {"126912": "BDSM League", "200436": "The League", "727092": "Ezekiel 25.17"}
TEAM_FIX = {"LAR": "LA", "Jax": "JAX", "Wsh": "WAS"}
SKILL = ["QB", "RB", "WR", "TE"]
IDP = {"S", "CB", "DB", "DE", "DT", "DL", "LB"}
# lineup slot -> positions that can fill it
SLOT_POS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "W/R/T": "RB WR TE", "W/R": "RB WR", "W/T": "WR TE",
            "R/W/T": "RB WR TE", "Q/W/R/T": "QB RB WR TE", "K": "K", "DEF": "DEF", "DB": "DB", "DL": "DL", "LB": "LB",
            "D": "IDP", "DP": "IDP"}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def kind(pos_list):
    first = pos_list.split(",")[0]
    if first in SKILL:
        return "skill"
    if first in ("K", "DEF"):
        return first
    return "IDP" if set(pos_list.split(",")) & IDP else "other"


def parse(raw_path):
    txt = open(raw_path, encoding="utf-8").read()
    head, rest = txt.split("ROSTERS\n")
    body, sets = (rest.split("SETTINGS\n") + [""])[:2]
    teams = {}
    for line in head.strip().split("\n")[1:]:
        lid, tid, name = line.split(" ", 2)
        teams[(lid, tid)] = name.strip()
    pat = re.compile(r"^(\d+) (\d+) (\S+) (.+) (\d+) (\S+) (\S+)(?: (\S+))?$")
    rows = [m.groups() for m in (pat.match(l.strip()) for l in body.strip().split("\n")) if m]
    r = pd.DataFrame(rows, columns=["league_id", "team_id", "slot", "player", "yahoo_id", "nfl", "pos_list", "status"])
    r = r[r["league_id"].isin(MY_TEAM)].drop_duplicates(["league_id", "team_id", "player", "slot"]).copy()
    r["kind"] = r["pos_list"].map(kind)
    r["pos"] = [p.split(",")[0] if k == "skill" else (k if k in ("K", "DEF", "IDP") else p.split(",")[0])
                for p, k in zip(r["pos_list"], r["kind"])]
    r["team"] = [teams.get((a, b), "") for a, b in zip(r["league_id"], r["team_id"])]
    st = {}
    for line in sets.strip().split("\n"):
        parts = line.split("\t") if "\t" in line else line.split(" | ")
        if len(parts) >= 3:
            st.setdefault(parts[0], {})[re.sub(r"\s*\(Yahoo Default:[^)]*\)\s*$", "", parts[1].strip())] = parts[2].strip()  # Yahoo appends "(Yahoo Default: x)" to changed settings
    return r, teams, st


def num(v, default):
    m = re.search(r"-?\d*\.?\d+", v or "")
    return float(m.group()) if m else default


def bonus(v, at):
    m = re.search(r"(\d+\.?\d*) points? at (\d+) yards", v or "")
    return float(m.group(1)) if m and int(m.group(2)) == at else 0.0


def league_rows(st, teams):
    out = []
    for lid in MY_TEAM:
        s = st.get(lid, {})
        slots = [x.strip() for x in s.get("Roster Positions", "").split(",") if x.strip()]
        starters = [x for x in slots if x not in ("BN", "IR")]
        pyd = s.get("Passing Yards", "")
        row = dict(league_id=lid, league=SHORT[lid], my_team=MY_TEAM[lid], my_team_name=teams.get((lid, MY_TEAM[lid]), ""),
                   teams=int(num(s.get("Max Teams"), 12)),
                   lineup=" ".join(starters), slot_pos="; ".join(f"{x}={SLOT_POS.get(x, x)}" for x in dict.fromkeys(starters)),
                   bench=slots.count("BN"), ir=slots.count("IR"), active_spots=len(starters) + slots.count("BN"),
                   pass_yds_per_pt=num(pyd, 25), pass_td=num(s.get("Passing Touchdowns"), 4),
                   pass_int=num(s.get("Interceptions"), 0), fum_lost=num(s.get("Fumbles Lost"), 0),
                   rec_pt=num(s.get("Receptions"), 0), rush_yds_per_pt=num(s.get("Rushing Yards"), 10),
                   rec_yds_per_pt=num(s.get("Receiving Yards"), 10), rush_td=num(s.get("Rushing Touchdowns"), 6),
                   rec_td=num(s.get("Receiving Touchdowns"), 6),
                   bonus_pass400=bonus(pyd, 400), bonus_rush150=bonus(s.get("Rushing Yards"), 150),
                   bonus_rec175=bonus(s.get("Receiving Yards"), 175),
                   negative_points=s.get("Negative Points", "Yes"), trade_end=s.get("Trade End Date", ""),
                   playoffs=s.get("Playoffs", ""))
        # settings the projections can't reflect (shown on the page as a note)
        notes = []
        if row["rush_yds_per_pt"] != 10 or row["rec_yds_per_pt"] != 10 or row["rush_td"] != 6 or row["rec_td"] != 6:
            notes.append("non-standard rushing / receiving scoring")
        if "Return Yards" in s:
            notes.append("return yards (" + s["Return Yards"] + ")")
        other_bonus = [k for k, v in s.items() if "points at" in v and k not in ("Passing Yards", "Rushing Yards", "Receiving Yards")]
        notes += [k + " bonus" for k in other_bonus]
        row["not_modeled"] = "; ".join(notes)
        out.append(row)
    return pd.DataFrame(out)


def build(raw_path, ros_path, out_dir):
    r, teams, st = parse(raw_path)
    ros = pd.read_csv(ros_path)
    ros["n"] = ros["player"].map(norm)
    r["n"] = r["player"].map(norm)
    sk = r["kind"] == "skill"
    m = r[sk].merge(ros[["n", "pos", "key"]].drop_duplicates(["n", "pos"]).rename(columns={"key": "ros_key"}),
                    on=["n", "pos"], how="left")
    r = pd.concat([m, r[~sk]], ignore_index=True)
    r["league"] = r["league_id"].map(SHORT)
    r["is_me"] = [int(MY_TEAM[a] == b) for a, b in zip(r["league_id"], r["team_id"])]
    r["nfl"] = r["nfl"].map(lambda t: TEAM_FIX.get(t, t).upper())
    r["status"] = r["status"].fillna("")
    r["key"] = r["league_id"] + "|" + r["team_id"] + "|" + r["yahoo_id"] + "|" + r["player"].map(norm).str.replace(" ", "")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    r["pulled_at"] = now
    cols = ["key", "league_id", "league", "team_id", "team", "is_me", "slot", "player", "pos", "pos_list", "kind", "nfl",
            "status", "yahoo_id", "ros_key", "pulled_at"]
    r = r.sort_values(["league_id", "team_id"], key=lambda s: s.astype(int))[cols]
    r.to_csv(f"{out_dir}/league_rosters_v5.csv", index=False)
    lg = league_rows(st, teams)
    lg["pulled_at"] = now
    lg.to_csv(f"{out_dir}/leagues_v5.csv", index=False)
    s = r[r["kind"] == "skill"]
    print(len(r), "rostered players,", len(s), "QB/RB/WR/TE;", int(s["ros_key"].isna().sum()), "without a ROS projection:",
          ", ".join(sorted(set(s.loc[s["ros_key"].isna(), "player"]))))
    print(lg[["league", "lineup", "bench", "ir", "active_spots", "pass_td", "pass_int", "pass_yds_per_pt", "fum_lost",
              "bonus_pass400", "bonus_rush150", "bonus_rec175", "not_modeled"]].to_string(index=False))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3])

"""Faster injury news for the current week -> the injury file the model reads (model/data/inj/inj_<season>.parquet).

nflverse's injury file updates about once a day, so a practice report or a Friday game status could reach the model
a day late. This reads two free sources as soon as they change and writes the current week's rows into that file:

  NFL.com official injury report (https://www.nfl.com/injuries/): every team's practice participation (Wed-Fri, or
      Mon-Wed before a Thursday game) and Friday game statuses, minutes after the league releases them. Used only when
      the page shows the week being projected. Fills practice_status / report_status / injury; never blanks a value the
      daily file already has.
  ESPN injury feed (site.api.espn.com/.../nfl/injuries, free, no key): breaking news between official reports. Only
      strong statuses are used - injured reserve, PUP, suspension or Out - and only when ESPN dated the entry inside the
      current game week; they become report_status "Out". (Weaker statuses linger from earlier weeks, so they aren't
      trusted.)

The model already treats an early-week practice report as provisional (a midweek DNP counts as limited until the
final report), so earlier data changes nothing about how a status is read - only when it arrives.

  python automation/injuries_live.py --data model/data --season 2026 --week 7          # apply; prints a hash
  python automation/injuries_live.py --data model/data --season 2026 --week 7 --dry    # hash only (the watcher)
"""
import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, HERE)
import inactives as I            # noqa: E402  (model/inactives.py: name normalising and the weekly roster)
from inactives_nfl import NICK, UA  # noqa: E402

PRACTICE = {"Did Not Participate In Practice", "Limited Participation in Practice", "Full Participation in Practice"}
STATUS = {"Out", "Doubtful", "Questionable"}
TITLE = {k.title() if k != "49ERS" else "49ers": v for k, v in NICK.items()}


def fetch(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def clean(x):
    return html.unescape(re.sub(r"<[^>]+>", " ", x)).strip()


def nfl_report(page):
    """-> (week shown, rows of name/team/pos/injury/practice/status)."""
    m = re.search(r"Injuries\s*-\s*WEEK\s*(\d+)", re.sub(r"<[^>]+>", " ", page), re.I)
    week = int(m.group(1)) if m else None
    rows, last = [], 0
    for tm in re.finditer(r"<table.*?</table>", page, re.S):
        before = re.sub(r"\s+", " ", clean(page[max(last, tm.start() - 3000):tm.start()]))
        last = tm.end()
        found = [(before.rfind(n), a) for n, a in TITLE.items() if before.rfind(n) >= 0]
        if not found or "Practice Status" not in tm.group(0):
            continue
        team = max(found)[1]
        for tr in re.findall(r"<tr>(.*?)</tr>", tm.group(0), re.S):
            td = [re.sub(r"\s+", " ", clean(x)) for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(td) >= 5 and td[0]:
                rows.append({"name": td[0], "team": team, "pos": td[1], "injury": td[2], "practice": td[3],
                             "status": td[4]})
    return week, rows


ESPN_STRONG = {"Out": "Out", "Injured Reserve": "Out", "Suspension": "Out", "Physically Unable to Perform": "Out",
               "Non-Football Injury": "Out"}
FULL = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
        "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
        "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}


def espn_news(since):
    """ESPN's league-wide injury feed: players put on IR / PUP / suspended or ruled Out since `since`."""
    d = json.loads(fetch("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries", timeout=120))
    out = []
    for tm in d.get("injuries", []):
        team = FULL.get(tm.get("displayName"))
        for x in tm.get("injuries", []):
            st = ESPN_STRONG.get(x.get("status"))
            when = pd.Timestamp(x.get("date")) if x.get("date") else None
            if not team or not st or when is None or when < since:
                continue
            a = x.get("athlete", {})
            out.append({"name": a.get("displayName", ""), "team": team,
                        "pos": (a.get("position") or {}).get("abbreviation", ""), "injury": "",
                        "status": st, "espn": x.get("status")})
    return out


def week_window(data_dir, season, week):
    """Tuesday 00:00 ET before the week's first game (statuses from then on belong to this week)."""
    g = pd.read_parquet(os.path.join(data_dir, "games.parquet"))
    g = g[(g["season"] == season) & (g["week"] == week)]
    first = pd.to_datetime(g["gameday"]).min()
    tue = first - pd.Timedelta(days=(first.weekday() - 1) % 7)
    return tue.tz_localize("America/New_York")


def match(rows, data_dir, season, week):
    """Official-report names -> gsis ids on the team's weekly roster (exact name, then first initial + last name)."""
    c = I.candidates(data_dir, season, week)
    c["k2"] = c["key"].str.split().map(lambda t: (t[0][:1] + " " + t[-1]) if len(t) > 1 else "")
    exact = c.drop_duplicates(["team", "key"], keep=False).set_index(["team", "key"])["player_id"]
    loose = c.drop_duplicates(["player_id", "k2"]).drop_duplicates(["team", "k2"], keep=False).set_index(["team", "k2"])["player_id"]
    pos = c.drop_duplicates("player_id").set_index("player_id")["position"]
    out, miss = [], []
    for r in rows:
        k = I.norm(r["name"])
        t = k.split()
        pid = exact.get((r["team"], k)) or (loose.get((r["team"], t[0][:1] + " " + t[-1])) if len(t) > 1 else None)
        if pid:
            out.append(dict(r, player_id=pid, position=pos.get(pid, r["pos"])))
        else:
            miss.append(r["name"] + " (" + r["team"] + ")")
    return out, miss


def gather(data_dir, season, week):
    info, official, news = {}, [], []
    try:
        wk, rows = nfl_report(fetch("https://www.nfl.com/injuries/"))
        info["nfl_week"], info["nfl_rows"] = wk, len(rows)
        if wk == week:
            official, miss = match(rows, data_dir, season, week)
            info["nfl_matched"], info["nfl_unmatched"] = len(official), miss[:20]
    except Exception as e:
        info["nfl_error"] = str(e)[:200]
    try:
        news, miss = match(espn_news(week_window(data_dir, season, week)), data_dir, season, week)
        info["espn_out"], info["espn_unmatched"] = len(news), miss[:20]
    except Exception as e:
        info["espn_error"] = str(e)[:200]
    key = sorted((r["player_id"], r.get("practice", ""), r.get("status", "")) for r in official + news)
    info["hash"] = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12] if key else None
    return official, news, info


def apply(official, news, data_dir, season, week):
    f = os.path.join(data_dir, "inj", f"inj_{season}.parquet")
    inj = pd.read_parquet(f)
    cur = inj["week"] == week
    idx = {pid: i for i, pid in zip(inj.index[cur], inj.loc[cur, "gsis_id"])}
    add, n_upd = [], 0
    official_ids = {r["player_id"] for r in official}
    for r in official + [x for x in news if x["player_id"] not in official_ids]:
        prac = r.get("practice") if r.get("practice") in PRACTICE else None
        stat = r.get("status") if r.get("status") in STATUS else None
        if r["player_id"] in idx:
            i = idx[r["player_id"]]
            if prac and inj.at[i, "practice_status"] != prac:
                inj.at[i, "practice_status"] = prac; n_upd += 1
            if stat and inj.at[i, "report_status"] != stat and (pd.isna(inj.at[i, "report_status"]) or r in official):
                inj.at[i, "report_status"] = stat; n_upd += 1
            if r.get("injury") and pd.isna(inj.at[i, "report_primary_injury"]):
                inj.at[i, "report_primary_injury"] = r["injury"]
        elif prac or stat:
            nm = r.get("name", "")
            add.append({"season": season, "season_type": "REG", "game_type": "REG", "team": r["team"], "week": week,
                        "gsis_id": r["player_id"], "position": r.get("position") or r.get("pos"), "full_name": nm,
                        "first_name": nm.split(" ")[0] if nm else None, "last_name": nm.split(" ")[-1] if nm else None,
                        "report_primary_injury": r.get("injury") or None, "report_status": stat,
                        "practice_primary_injury": r.get("injury") or None, "practice_status": prac})
    if add:
        inj = pd.concat([inj, pd.DataFrame(add)[[c for c in inj.columns if c in add[0]]]], ignore_index=True)
    if add or n_upd:
        inj.to_parquet(f)
    return {"rows_added": len(add), "values_updated": n_upd}


def raw_stamp(week):
    """Hash of what the sources say right now (no name matching, no data files): for the 30-minute check."""
    key = []
    try:
        wk, rows = nfl_report(fetch("https://www.nfl.com/injuries/"))
        if wk == week:
            key += sorted((r["team"], r["name"], r["practice"], r["status"]) for r in rows)
    except Exception:
        pass
    try:
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
        key += sorted((r["team"], r["name"], r["status"]) for r in espn_news(since))
    except Exception:
        pass
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12] if key else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="model/data")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--info", default=None, help="also write the summary to this JSON file")
    a = ap.parse_args()
    official, news, info = gather(a.data, a.season, a.week)
    if not a.dry and (official or news):
        info.update(apply(official, news, a.data, a.season, a.week))
    if a.info:
        json.dump(info, open(a.info, "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()

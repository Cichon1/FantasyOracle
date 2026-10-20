"""Player prop betting lines from The Odds API (free plan: 500 credits a month) -> site file + saved history.

Markets: passing yards, rushing yards, receiving yards, anytime touchdown. For each player and market: the median line
across US sportsbooks and the no-vig chance of the over (each book's over/under pair rescaled to 100%, then averaged);
for anytime touchdowns, the average implied chance of "Yes" (one-sided, so it still carries the books' margin).

Cost: the events list is free; each game's odds cost 1 credit per market per region (4 here). A full Sunday slate is
about 52 credits, the Thursday game 4 - roughly 250 a month on the schedule in props.yml. Credits left are recorded.

  ODDS_API_KEY=... python automation/odds_props.py                 # every game in the next 4 days
  ODDS_API_KEY=... python automation/odds_props.py --hours 30      # only games in the next 30 hours (Thursday)

-> docs/data/props.csv and automation/history/props_<date>_<HHMM>.csv
"""
import argparse
import datetime as dt
import json
import os
import statistics
import sys
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
MARKETS = {"player_pass_yds": "pass_yds", "player_rush_yds": "rush_yds", "player_reception_yds": "rec_yds",
           "player_anytime_td": "anytime_td"}
FULL = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
        "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
        "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
credits = {}


def get(path, **params):
    params["apiKey"] = os.environ["ODDS_API_KEY"]
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "FantasyOracle/1.0"}), timeout=60) as r:
        credits["remaining"] = r.headers.get("x-requests-remaining")
        credits["used"] = r.headers.get("x-requests-used")
        return json.load(r)


def prob(price):
    price = float(price)
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def summarize(ev):
    rows = {}
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            m = MARKETS.get(mk.get("key"))
            if not m:
                continue
            by = {}
            for o in mk.get("outcomes", []):
                by.setdefault(o.get("description") or "", []).append(o)
            for player, outs in by.items():
                r = rows.setdefault((player, m), {"lines": [], "p_over": []})
                if m == "anytime_td":
                    yes = [o for o in outs if o.get("name") in ("Yes", "Over")]
                    if yes:
                        r["p_over"].append(prob(yes[0]["price"]))
                    continue
                ov = [o for o in outs if o.get("name") == "Over"]
                un = [o for o in outs if o.get("name") == "Under"]
                if ov and un and ov[0].get("point") is not None:
                    po, pu = prob(ov[0]["price"]), prob(un[0]["price"])
                    r["lines"].append(float(ov[0]["point"]))
                    r["p_over"].append(po / (po + pu))
    out = []
    for (player, m), r in rows.items():
        if not r["p_over"]:
            continue
        out.append({"player": player, "market": m,
                    "line": statistics.median(r["lines"]) if r["lines"] else None,
                    "p_over": round(sum(r["p_over"]) / len(r["p_over"]), 4), "books": len(r["p_over"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=96)
    a = ap.parse_args()
    if not os.environ.get("ODDS_API_KEY"):
        print(json.dumps({"props": "no ODDS_API_KEY secret yet"}))
        return
    now = dt.datetime.now(dt.timezone.utc)
    events = get("events")
    soon = [e for e in events if now < pd.Timestamp(e["commence_time"]).to_pydatetime() < now + dt.timedelta(hours=a.hours)]
    rows = []
    for e in soon:
        try:
            odds = get(f"events/{e['id']}/odds", regions="us", markets=",".join(MARKETS), oddsFormat="american")
        except Exception as ex:
            print(f"{e.get('away_team')} at {e.get('home_team')}: {ex}", file=sys.stderr)
            continue
        for r in summarize(odds):
            rows.append(dict(r, game=f"{FULL.get(e['away_team'], e['away_team'])}@{FULL.get(e['home_team'], e['home_team'])}",
                             kickoff=e["commence_time"], event_id=e["id"]))
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    info = {"props": len(rows), "games": len(soon), "credits_remaining": credits.get("remaining"), "pulled_at": stamp}
    if rows:
        df = pd.DataFrame(rows)
        df["pulled_at"] = stamp
        site = os.path.join(ROOT, "docs", "data", "props.csv")
        if os.path.exists(site):   # keep earlier games' lines (e.g. Thursday's) when a later pull covers fewer games
            old = pd.read_csv(site)
            old = old[~old["event_id"].isin(df["event_id"])]
            old = old[pd.to_datetime(old["kickoff"], utc=True) > pd.Timestamp(now) - pd.Timedelta(days=4)]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(site, index=False)
        hist = os.path.join(HERE, "history")
        os.makedirs(hist, exist_ok=True)
        pd.DataFrame(rows).assign(pulled_at=stamp).to_csv(os.path.join(hist, f"props_{now:%Y-%m-%d_%H%M}.csv"), index=False)
    os.makedirs(os.path.join(HERE, "state"), exist_ok=True)
    json.dump(info, open(os.path.join(HERE, "state", "props_last.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()

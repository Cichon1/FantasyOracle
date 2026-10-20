"""Free signals saved every update, for the site now and for testing as model inputs later.

  Sleeper trending (https://api.sleeper.app/v1/players/nfl/trending/{add,drop}, free, no key): how many Sleeper leagues
  added / dropped each player in the last 24 hours - the crowd reacting to news, often before box scores show it.
  -> docs/data/trending.csv (player key matching the projections, adds_24h, drops_24h)
  -> automation/history/trending_<date>.csv (one file per day, so a season of history builds up for a fair test)

  python automation/snapshots.py docs/data
"""
import datetime as dt
import json
import os
import sys
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "FantasyOracle/1.0 (+https://github.com/Cichon1/FantasyOracle)"}
TEAM = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC"}


def get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.load(r)


def main(site):
    players = get("https://api.sleeper.app/v1/players/nfl", timeout=120)
    rows = {}
    for kind in ("add", "drop"):
        for t in get(f"https://api.sleeper.app/v1/players/nfl/trending/{kind}?lookback_hours=24&limit=100"):
            p = players.get(str(t["player_id"])) or {}
            if p.get("position") not in ("QB", "RB", "WR", "TE"):
                continue
            k = str(t["player_id"])
            r = rows.setdefault(k, {"sleeper_id": k, "gsis_id": str(p.get("gsis_id") or "").strip(),
                                    "player": p.get("full_name"), "pos": p.get("position"),
                                    "team": TEAM.get(p.get("team"), p.get("team")), "adds_24h": 0, "drops_24h": 0})
            r[kind + "s_24h"] = int(t.get("count", 0))
    df = pd.DataFrame(rows.values())
    if not len(df):
        print("trending: nothing")
        return
    proj = pd.read_csv(os.path.join(site, "ros_v5.csv"), usecols=["key", "player_id"])
    df = df.merge(proj.rename(columns={"player_id": "gsis_id"}), on="gsis_id", how="left")
    df["key"] = df["key"].fillna(df["player"] + " (" + df["team"].fillna("FA") + ")")
    df["pulled_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.sort_values("adds_24h", ascending=False)
    df.to_csv(os.path.join(site, "trending.csv"), index=False)
    hist = os.path.join(HERE, "history")
    os.makedirs(hist, exist_ok=True)
    df.to_csv(os.path.join(hist, f"trending_{dt.date.today().isoformat()}.csv"), index=False)
    print(json.dumps({"trending": len(df), "top_add": df.iloc[0]["player"]}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/data")

"""FantasyPros Trade Market Values (redraft PPR, from real fantasy trades over the last 10 days) -> docs/data/trade_market_v5.csv
and automation/history/trade_market_<date>.csv. Public feed, no sign-in; same columns the Chrome script wrote. Runs
once a day (skipped when today's history file exists). Used by the trade analyzer and the trade finder's market value.

  python automation/trade_market_pull.py docs/data
"""
import datetime as dt
import json
import os
import re
import sys
import unicodedata
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://mpbnfl.fantasypros.com/api/tradeMarketValues?type=redraft&scoring=PPR&includeChartData=true"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
      "Origin": "https://www.fantasypros.com", "Referer": "https://www.fantasypros.com/"}


def key(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’\-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def main(site):
    hist = os.path.join(HERE, "history")
    today = dt.date.today().isoformat()
    if os.path.exists(os.path.join(hist, f"trade_market_{today}.csv")):
        print("trade market: already pulled today")
        return
    with urllib.request.urlopen(urllib.request.Request(URL, headers=UA), timeout=60) as r:
        j = json.load(r)
    anchor = dt.date.fromisoformat(j["anchorDate"])
    dates = [(anchor - dt.timedelta(days=i)).isoformat() for i in range(j["windowDays"] - 1, -1, -1)]
    rows = []
    for p in j["players"]:
        if p.get("position") not in ("QB", "RB", "WR", "TE") or not (p.get("marketValue") or 0) > 0:
            continue
        vm = {x["date"]: x["value"] for x in p.get("valueSeries") or []}
        nm = {x["date"]: x["volume"] for x in p.get("volumeSeries") or []}
        vals = [vm.get(d, "") for d in dates]
        vols = [nm.get(d, "") for d in dates]
        first = next((v for v in vals if v not in ("", None)), None)
        rows.append({"as_of": j["anchorDate"], "window_start": dates[0], "fp_id": p.get("playerId"), "player": p["name"],
                     "pos": p["position"], "team": p.get("team") or "", "market_value": p["marketValue"],
                     "market_rank_pos": p.get("marketRankPositional"), "trades_10d": sum(int(v or 0) for v in vols),
                     "value_change_10d": "" if first is None else p["marketValue"] - first,
                     "values_by_day": " ".join(str(v) for v in vals), "trades_by_day": " ".join(str(v) for v in vols),
                     "name_key": key(p["name"])})
    if len(rows) < 100:
        raise SystemExit(f"trade market: only {len(rows)} players - not saved")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(site, "trade_market_v5.csv"), index=False)
    os.makedirs(hist, exist_ok=True)
    df.to_csv(os.path.join(hist, f"trade_market_{today}.csv"), index=False)
    print(json.dumps({"trade_market": len(df), "as_of": j["anchorDate"]}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/data")

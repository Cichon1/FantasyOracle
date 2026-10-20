"""Game-time weather forecast for the week's outdoor games -> model/ref/weather_<season>_wk<week>.csv.

Free, no key: Open-Meteo forecast API (https://open-meteo.com). Columns are what the model's apply_forecast() reads,
keyed by home team: wx_temp (F), wx_wind (mph), wx_rain (chance of rain at kickoff, 0-1), wx_snow (1 if snow is
forecast), wx_dome. Domes are left to the schedule (the model already sets them indoors); retractable roofs are left
unknown (whether the roof is open is decided on game day). Any failure leaves the week without a forecast file, which
is how the model ran before a forecast was added (conditions unknown).

  python automation/weather.py --data model/data --season 2026 --week 7 --out model/ref
"""
import argparse
import json
import os
import sys
import urllib.request

import pandas as pd

# open-air stadiums (stadium_id in nflverse's schedule): latitude, longitude
OUTDOOR = {
    "BAL00": (39.278, -76.623), "BOS00": (42.091, -71.264), "BUF00": (42.774, -78.787), "CAR00": (35.226, -80.853),
    "CHI98": (41.862, -87.617), "CIN00": (39.095, -84.516), "CLE00": (41.506, -81.700), "DEN00": (39.744, -105.020),
    "GNB00": (44.501, -88.062), "JAX00": (30.324, -81.637), "KAN00": (39.049, -94.484), "MIA00": (25.958, -80.239),
    "NAS00": (36.166, -86.771), "NYC01": (40.814, -74.074), "PHI00": (39.901, -75.168), "PIT00": (40.447, -80.016),
    "SEA00": (47.595, -122.332), "SFO01": (37.403, -121.970), "TAM00": (27.976, -82.503), "WAS00": (38.908, -76.864),
    "LON00": (51.556, -0.280), "LON02": (51.604, -0.066), "MEX00": (19.303, -99.150), "MAD01": (40.453, -3.688),
    "RIO00": (-22.912, -43.230),
}


def forecast(lat, lon, day, hour):
    url = ("https://api.open-meteo.com/v1/forecast?latitude=%.3f&longitude=%.3f"
           "&hourly=temperature_2m,wind_speed_10m,precipitation_probability,snowfall"
           "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America%%2FNew_York"
           "&start_date=%s&end_date=%s" % (lat, lon, day, day))
    with urllib.request.urlopen(url, timeout=30) as r:
        h = json.load(r)["hourly"]
    t = pd.DataFrame(h)
    t["hour"] = pd.to_datetime(t["time"]).dt.hour
    w = t[(t["hour"] >= hour) & (t["hour"] <= hour + 3)]   # the game window, kickoff + 3 hours
    if not len(w) or w["temperature_2m"].isna().all():
        return None
    return {"wx_temp": round(float(w["temperature_2m"].mean()), 1),
            "wx_wind": round(float(w["wind_speed_10m"].mean()), 1),
            "wx_rain": round(float(w["precipitation_probability"].max()) / 100, 2),
            "wx_snow": int(float(w["snowfall"].sum()) >= 0.5),
            "wx_dome": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="model/data")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", default="model/ref")
    a = ap.parse_args()
    g = pd.read_parquet(os.path.join(a.data, "games.parquet"))
    g = g[(g["season"] == a.season) & (g["week"] == a.week) & g["home_score"].isna()]
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    rows = []
    for r in g.itertuples():
        if r.stadium_id not in OUTDOOR:
            continue
        day = pd.Timestamp(r.gameday)
        if (day - today).days > 14:          # beyond the forecast range
            continue
        lat, lon = OUTDOOR[r.stadium_id]
        try:
            f = forecast(lat, lon, day.strftime("%Y-%m-%d"), int(str(r.gametime).split(":")[0]))
        except Exception as e:                # network trouble: that game stays unknown
            print(f"weather {r.game_id}: {e}", file=sys.stderr)
            continue
        if f:
            rows.append({"home_team": r.home_team, **f})
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"weather_{a.season}_wk{a.week}.csv")
    if rows:
        pd.DataFrame(rows).drop_duplicates("home_team").to_csv(path, index=False)
    print(json.dumps({"weather_file": path if rows else None, "games": len(rows)}))


if __name__ == "__main__":
    main()

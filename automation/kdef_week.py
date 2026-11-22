"""K and DEF weekly projections for the site (Nov 22, 2026): docs/data/kdef_v5.csv.

Team units (whoever kicks / the team defense), Yahoo default scoring, from model/lab17_kdef.py: linear models fit on
every earlier season (K ~ team implied points, dome, recent K average; DEF ~ opponent implied points, spread, home,
recent DEF average, opponent's recent scoring). In the league sims these DEF projections ranked weekly defenses as well
as Sleeper's and ESPN's (rank correlation ~0.31 vs 0.30 / 0.29; recent average 0.14); kickers are close to random
(~0.14). Rows for the projection week and the two after it.

  python automation/kdef_week.py --data model/data --season 2026 --week 12 --out docs/data
"""
import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import lab17_kdef as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "model", "data"))
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data"))
    a = ap.parse_args()
    u = K.build(a.season, a.data)
    u = u[(u["season"] == a.season) & u["week"].between(a.week, a.week + 2)].copy()
    rows = []
    for unit, col in [("DEF", "def"), ("K", "k")]:
        x = u[["season", "week", "team", "opp", "home", "spread", "implied", "opp_implied", "dome", f"{col}_proj",
               f"{col}_avg"]].rename(columns={f"{col}_proj": "proj", f"{col}_avg": "recent_avg"})
        x["unit"] = unit
        x["rank"] = x.groupby("week")["proj"].rank(ascending=False, method="first").astype(int)
        rows.append(x)
    out = pd.concat(rows, ignore_index=True)
    for c in ["spread", "implied", "opp_implied", "proj", "recent_avg"]:
        out[c] = out[c].round(2)
    out.sort_values(["week", "unit", "rank"]).to_csv(os.path.join(a.out, "kdef_v5.csv"), index=False)
    cat_p = os.path.join(a.out, "datasets.json")
    if os.path.exists(cat_p):
        cat = json.load(open(cat_p))
        cat["datasets"]["kdef"] = {
            "file": "kdef_v5.csv", "title": f"Kicker and defense projections, Weeks {a.week}-{a.week + 2}",
            "description": "Team kicker and team defense projected Yahoo-default fantasy points by week, from Vegas "
                           "lines (implied points, spread), dome, home and recent scoring; weeks without lines yet "
                           "fall back to season averages.",
            "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        json.dump(cat, open(cat_p, "w"), indent=1)
    print(f"kdef: {len(out)} rows, weeks {a.week}-{a.week + 2}")


if __name__ == "__main__":
    main()

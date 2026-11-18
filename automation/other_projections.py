"""Past weekly PPR projections from other sites, for the league sims (Nov 17, 2026).

FantasyPros is the only source whose week-by-week history we had (DynastyProcess archives it). This pulls whatever
Sleeper's and ESPN's public projection endpoints still return for past weeks, so their managers can be added to the
sims. Run on GitHub Actions (the Claude workspace can't reach these sites):

  python automation/other_projections.py 2023 2024 2025 2026

Writes automation/history/other_proj/<source>_<season>.csv (season, week, source_id, player, pos, team, proj_ppr)
and automation/history/other_proj/pull_log.json (what each source returned, per season). A source that returns
nothing for past weeks is logged and skipped; nothing else depends on this.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "automation", "history", "other_proj")
UA = {"User-Agent": "Mozilla/5.0 (FantasyOracle; +https://github.com/Cichon1/FantasyOracle)"}
POS = ["QB", "RB", "WR", "TE"]


def fetch(url, headers=None, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404):
                raise RuntimeError(f"HTTP {e.code}") from None
            time.sleep(3 * (k + 1))
        except Exception:
            time.sleep(3 * (k + 1))
    raise RuntimeError("no response")


# ---------------------------------------------------------------------------------------------------- Sleeper
def sleeper(season, weeks):
    rows = []
    for w in weeks:
        url = (f"https://api.sleeper.app/projections/nfl/{season}/{w}?season_type=regular"
               + "".join(f"&position[]={p}" for p in POS))
        try:
            data = fetch(url)
        except Exception as e:
            rows.append({"_err": f"week {w}: {e}"})
            continue
        for d in data or []:
            st, pl = d.get("stats") or {}, d.get("player") or {}
            pts = st.get("pts_ppr")
            if pts is None:
                continue
            rows.append({"season": season, "week": w, "source_id": str(d.get("player_id")),
                         "player": f"{pl.get('first_name', '')} {pl.get('last_name', '')}".strip(),
                         "pos": pl.get("position"), "team": d.get("team") or pl.get("team"), "proj_ppr": float(pts)})
        time.sleep(0.5)
    return rows


# ------------------------------------------------------------------------------------------------------- ESPN
ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}


def espn_points(s):
    """PPR points from an ESPN projected stat line (stat ids: 3 pass yds, 4 pass TD, 20 INT, 24 rush yds,
    25 rush TD, 53 receptions, 42 rec yds, 43 rec TD, 72 fumbles lost, 19/26/44 two-point conversions)."""
    g = lambda k: float(s.get(str(k), 0) or 0)
    return (g(3) / 25 + 4 * g(4) - 2 * g(20) + g(24) / 10 + 6 * g(25) + g(53) + g(42) / 10 + 6 * g(43)
            - 2 * g(72) + 2 * (g(19) + g(26) + g(44)))


def espn(season, weeks):
    rows = []
    base = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3"
    for w in weeks:
        flt = {"players": {"filterSlotIds": {"value": [0, 2, 4, 6]}, "limit": 600,
                           "filterStatsForTopScoringPeriodIds": {"value": 2, "additionalValue": [f"11{season}{w}",
                                                                                                f"10{season}{w}"]}}}
        try:
            data = fetch(f"{base}?scoringPeriodId={w}&view=kona_player_info",
                         headers={"X-Fantasy-Filter": json.dumps(flt), "X-Fantasy-Platform": "kona-PROD"})
        except Exception as e:
            rows.append({"_err": f"week {w}: {e}"})
            continue
        for p in data.get("players", []):
            pl = p.get("player") or {}
            pos = ESPN_POS.get(pl.get("defaultPositionId"))
            if not pos:
                continue
            for st in pl.get("stats") or []:
                # statSourceId 1 = projection, statSplitTypeId 1 = single scoring period
                if st.get("statSourceId") == 1 and st.get("scoringPeriodId") == w and st.get("seasonId") == season \
                        and st.get("statSplitTypeId") == 1:
                    rows.append({"season": season, "week": w, "source_id": str(pl.get("id")),
                                 "player": pl.get("fullName"), "pos": pos, "team": pl.get("proTeamId"),
                                 "proj_ppr": round(espn_points(st.get("stats") or {}), 2),
                                 "espn_applied": st.get("appliedTotal")})
                    break
        time.sleep(0.5)
    return rows


def main():
    seasons = [int(s) for s in sys.argv[1:]] or [2023, 2024, 2025, 2026]
    os.makedirs(OUT, exist_ok=True)
    logp = os.path.join(OUT, "pull_log.json")
    log = json.load(open(logp)) if os.path.exists(logp) else {}
    for S in seasons:
        weeks = list(range(1, 18 if S < 2026 else 19))
        for name, fn in [("sleeper", sleeper), ("espn", espn)]:
            try:
                rows = fn(S, weeks)
            except Exception as e:
                log[f"{name}_{S}"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": str(e)}
                continue
            errs = [r["_err"] for r in rows if "_err" in r]
            df = pd.DataFrame([r for r in rows if "_err" not in r])
            nz = int((df["proj_ppr"] > 0).sum()) if len(df) else 0
            log[f"{name}_{S}"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": len(df),
                                  "nonzero": nz, "weeks": int(df["week"].nunique()) if len(df) else 0,
                                  "errors": errs[:5]}
            if nz:
                df.to_csv(os.path.join(OUT, f"{name}_{S}.csv"), index=False)
            print(name, S, log[f"{name}_{S}"], flush=True)
    json.dump(log, open(logp, "w"), indent=1)


if __name__ == "__main__":
    main()

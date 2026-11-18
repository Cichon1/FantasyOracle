"""Save Yahoo's weekly player projections, with the time each one was seen (Nov 17, 2026).

Why: Yahoo's projections are a control in the league sim (model vs FantasyPros vs Yahoo managers). Yahoo keeps no
public archive and updates each week's projection until kickoff, so we save them ourselves, every time the
30-minute update runs, and keep a row only when a player's projection changed. That gives the full path of each
projection through the week; the sim uses the last value seen before a time (Thursday run, Sunday morning,
before each kickoff window).

Which API call returns player projections isn't documented, so the first run probes a few candidates, records the
result in automation/state/yahoo_proj_probe.json and uses the first one that returns projections (non-zero points
for players whose game hasn't been played). If none does, nothing is saved and the probe file says why.

Called from yahoo_sync.sync() after the league pull; any failure here is recorded and never blocks the sync.
Output: automation/history/yahoo_proj/proj_<season>_wk<week>.csv
        columns seen_at, season, week, league_id, player_key, yahoo_id, player, nfl, pos, proj_pts, stats, source
        (proj_pts is in the scoring of the league queried: BDSM League through the signed-in API, Yahoo's default
        public-league scoring through the public API; stats is the projected stat line to re-score from)
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "automation", "state")
OUT = os.path.join(ROOT, "automation", "history", "yahoo_proj")
PROBE = os.path.join(STATE, "yahoo_proj_probe.json")
LEAGUE = "126912"          # BDSM League: 12 teams, full PPR, 6-pt passing TDs
POS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}   # pages of 25 free agents per position (sorted by season points)
# (host, path): "api" = the signed-in Fantasy API, "pub" = Yahoo's public read-only API that its own website uses
CANDIDATES = [
    ("api", "league/{lk}/players;player_keys={keys};out=stats;stats.type=projected_week;stats.week={w}"),
    ("pub", "league/{gk}.l.public/players;player_keys={keys};out=stats;stats.type=projected_week;stats.week={w}"),
    ("pub", "league/{gk}.l.public/players;player_keys={keys}/stats;type=projected_week;week={w}"),
    ("api", "league/{lk}/players;player_keys={keys}/stats;type=projected_week;week={w}"),
    ("pub", "league/{gk}.l.public/players;player_keys={keys};out=stats;stats.type=week;stats.week={w}"),
]
PUB = "https://pub-api-ro.fantasysports.yahoo.com/fantasy/v2"


def pub_get(path):
    import urllib.request, urllib.error, xml.etree.ElementTree as ET
    req = urllib.request.Request(f"{PUB}/{path}", headers={"User-Agent": "Mozilla/5.0 (FantasyOracle)",
                                                           "Origin": "https://football.fantasysports.yahoo.com"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            e = ET.fromstring(r.read())
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"HTTP {err.code} {err.read()[:300]!r}") from None
    for x in e.iter():
        if "}" in x.tag:
            x.tag = x.tag.split("}", 1)[1]
    return e


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _players(get, t, path):
    """(player_key, yahoo_id, name, nfl, pos, points, extra projected fields) for each player in a response."""
    root = get(path)
    rows = []
    for p in root.iter("player"):
        extra = {}
        for x in p.iter():
            if "proj" in x.tag.lower() and x.text and x.text.strip():
                extra[x.tag] = x.text.strip()
        tot = t(p, "player_projected_points/total") or t(p, "player_points/total")
        rows.append({"player_key": t(p, "player_key"), "yahoo_id": t(p, "player_id"), "player": t(p, "name/full"),
                     "nfl": t(p, "editorial_team_abbr"), "pos": t(p, "display_position"),
                     "proj_pts": float(tot) if tot not in ("", None) else None,
                     "coverage": t(p, "player_points/coverage_type") or t(p, "player_stats/coverage_type"),
                     # the projected stat line (Yahoo stat_id:value), so points can be re-scored for any league
                     "stats": ";".join(f"{t(x, 'stat_id')}:{t(x, 'value')}" for x in p.findall("player_stats/stats/stat")
                                       if t(x, "value") not in ("", "0", "-")),
                     "extra": extra})
    return rows


def _week(get, t, lk):
    s = get(f"league/{lk}")
    return int(t(s, ".//current_week", "0") or 0)


def _keys(get, t, lk):
    keys = []
    r = get(f"league/{lk}/teams/roster")
    for p in r.iter("player"):
        if t(p, "display_position") in POS or any(e.text in POS for e in p.findall("eligible_positions/position")):
            keys.append(t(p, "player_key"))
    for pos, pages in POS.items():
        for pg in range(pages):
            fa = get(f"league/{lk}/players;status=A;position={pos};sort=AR;sort_type=season;start={25 * pg};count=25")
            keys += [t(p, "player_key") for p in fa.iter("player")]
    return list(dict.fromkeys(k for k in keys if k))


def _probe(get, t, lk, week, keys):
    """Try each candidate on 25 players; a candidate 'works' if most players get a non-zero total."""
    res = {"at": _now(), "week": week, "candidates": {}}
    sample = ",".join(keys[:25])
    gk = lk.split(".l.")[0]
    for host, c in CANDIDATES:
        path = c.format(lk=lk, gk=gk, keys=sample, w=week)
        c = f"{host}|{c}"
        try:
            rows = _players(get if host == "api" else pub_get, t, path)
            vals = [r["proj_pts"] for r in rows if r["proj_pts"] is not None]
            res["candidates"][c] = {"ok": True, "players": len(rows), "nonzero": sum(v != 0 for v in vals),
                                    "coverage": sorted({r["coverage"] for r in rows if r["coverage"]}),
                                    "proj_fields": sorted({k for r in rows for k in r["extra"]}),
                                    "example": [(r["player"], r["proj_pts"]) for r in rows[:5]]}
        except Exception as e:
            msg = str(e)
            res["candidates"][c] = {"ok": False, "error": msg[msg.find("HTTP"):][:400] if "HTTP" in msg else msg[-400:]}
    good = [c for c, v in res["candidates"].items() if v.get("ok") and v["players"] and v["nonzero"] >= v["players"] * 0.5
            and "type=week" not in c]       # the last candidate is actual stats: a syntax check, never used
    res["use"] = good[0] if good else None
    return res


def capture(get, t, game_key, season):
    os.makedirs(STATE, exist_ok=True)
    lk = f"{game_key}.l.{LEAGUE}"
    week = _week(get, t, lk)
    if not week:
        return {"yahoo_proj": "no current week"}
    keys = _keys(get, t, lk)
    probe = json.load(open(PROBE)) if os.path.exists(PROBE) else {}
    if not probe.get("use"):
        # after a failed probe, try again once a day (each probe rewrites, and so commits, the file)
        last = probe.get("at")
        if last and (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() < 86400:
            return {"yahoo_proj": "no projection endpoint found (retrying daily)", "week": week}
        probe = _probe(get, t, lk, week, keys)
        json.dump(probe, open(PROBE, "w"), indent=1)
        if not probe.get("use"):
            return {"yahoo_proj": "no projection endpoint found", "week": week}
    host, cand = probe["use"].split("|", 1)
    gk = lk.split(".l.")[0]
    getter = get if host == "api" else pub_get
    weeks = [week]
    done = os.path.join(OUT, "backfill_done.json")
    if not os.path.exists(done):        # once: whatever Yahoo still returns for earlier weeks (likely the final value)
        weeks = list(range(1, week + 1))
    os.makedirs(OUT, exist_ok=True)
    added = 0
    for w in weeks:
        rows = []
        for i in range(0, len(keys), 25):
            path = cand.format(lk=lk, gk=gk, keys=",".join(keys[i:i + 25]), w=w)
            rows += _players(getter, t, path)
        now = _now()
        df = pd.DataFrame([{"seen_at": now, "season": season, "week": w, "league_id": LEAGUE,
                            "player_key": r["player_key"], "yahoo_id": r["yahoo_id"], "player": r["player"],
                            "nfl": r["nfl"], "pos": r["pos"], "proj_pts": r["proj_pts"], "stats": r["stats"],
                            "source": "live" if w == week else "backfill"} for r in rows if r["proj_pts"] is not None])
        if df.empty:
            continue
        f = os.path.join(OUT, f"proj_{season}_wk{w:02d}.csv")
        if os.path.exists(f):
            old = pd.read_csv(f, dtype={"yahoo_id": str})
            last = old.sort_values("seen_at").groupby("player_key")["proj_pts"].last()
            df = df[df["player_key"].map(last).ne(df["proj_pts"]) | ~df["player_key"].isin(last.index)]
            if df.empty:
                continue
            df = pd.concat([old, df], ignore_index=True)
            added += len(df) - len(old)
        else:
            added += len(df)
        df.to_csv(f, index=False)
    if not os.path.exists(done):
        json.dump({"at": _now(), "weeks": weeks}, open(done, "w"))
    return {"yahoo_proj": "saved", "week": week, "players": len(keys), "rows_added": added}

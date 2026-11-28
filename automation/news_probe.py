"""One-off look at free injury-news feeds (Nov 28, 2026), for an automatic return-estimate step.

Saves the shape of each feed and a few full entries (incl. Colston Loveland's) to automation/ref/news_probe.json.
The cloud sessions can't reach these hosts; GitHub's runners can.
"""
import json
import os
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (FantasyOracle injury probe)"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "news_probe.json")


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def shape(x, depth=0):
    if depth > 6:
        return "..."
    if isinstance(x, dict):
        return {k: shape(v, depth + 1) for k, v in x.items()}
    if isinstance(x, list):
        return [shape(x[0], depth + 1)] if x else []
    return type(x).__name__


rep = {}
try:
    d = get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries")
    rep["espn_injuries_shape"] = shape(d)
    entries = [i for t in d.get("injuries", []) for i in t.get("injuries", [])]
    rep["espn_injuries_n"] = len(entries)
    rep["espn_status_counts"] = {}
    for i in entries:
        s = i.get("status")
        rep["espn_status_counts"][s] = rep["espn_status_counts"].get(s, 0) + 1
    rep["espn_with_return_date"] = sum(1 for i in entries if (i.get("details") or {}).get("returnDate"))
    rep["espn_loveland"] = [i for i in entries if "Loveland" in (i.get("athlete") or {}).get("displayName", "")]
    rep["espn_samples"] = {}
    for i in entries:
        s = i.get("status")
        if len(rep["espn_samples"].get(s, [])) < 3:
            rep["espn_samples"].setdefault(s, []).append(i)
    lov = rep["espn_loveland"]
    aid = None
    if lov:
        link = json.dumps(lov[0])
        import re
        m = re.search(r"/athletes/(\d+)", link) or re.search(r'"id": "(\d+)"', link)
        aid = m.group(1) if m else None
    rep["loveland_espn_id"] = aid
    if aid:
        for name, url in [("espn_athlete_v3", f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{aid}"),
                          ("espn_fantasy_news", f"https://site.api.espn.com/apis/fantasy/v2/games/ffl/news/players?playerId={aid}&limit=5")]:
            try:
                rep[name] = get(url)
            except Exception as e:
                rep[name] = "error: " + str(e)[:200]
except Exception as e:
    rep["espn_error"] = str(e)[:300]

try:
    p = get("https://api.sleeper.app/v1/players/nfl", timeout=120)
    lov = [v for v in p.values() if v.get("full_name") == "Colston Loveland"]
    keys = sorted({k for v in list(p.values())[:50] for k in v if "injur" in k or "status" in k or "news" in k})
    rep["sleeper_injury_keys"] = keys
    rep["sleeper_loveland"] = [{k: v.get(k) for k in keys + ["full_name", "team", "position"]} for v in lov]
    rep["sleeper_injury_status_counts"] = {}
    for v in p.values():
        s = v.get("injury_status")
        if s:
            rep["sleeper_injury_status_counts"][s] = rep["sleeper_injury_status_counts"].get(s, 0) + 1
except Exception as e:
    rep["sleeper_error"] = str(e)[:300]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: (v if k.endswith(("counts", "_n", "date", "keys", "error", "_id")) else "...") for k, v in rep.items()}, default=str)[:3000])

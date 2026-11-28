"""Injury news -> automatic return estimates (Nov 28, 2026; Jackson: "Loveland just got hurt and is likely out for a bit").

The weekly model already reads this week's injury report (injuries_live.py) and the ROS engine prices out players'
return timing from report history (avail_out.py). What it missed: news that a player will miss SEVERAL games before any
official status says so (Colston Loveland, Week 12: carted off Thursday, "week-to-week", IR expected = at least four
games, while his ROS still assumed ~95% to play every game).

This reads ESPN's free league-wide injury feed (site.api.espn.com/.../nfl/injuries; each entry has a status, ESPN's
expected return date and Rotowire's news blurb) for QB / RB / WR / TE and estimates the last week each player misses:

  1. ESPN's return date: his team's games before that date are missed. (For most "Out" players it's simply his next
     game after this one, so it only adds information when it is later than that.)
  2. The news text, in sentences about him (his last name, or starting "He" / "His"):
       season-ending / out for the rest of the season / not expected to return this season  -> out through Week 17
         (unless the text also says the opposite: "not expected to be season-ending", "expected to return this season")
       "at least (the next) four games / weeks", "minimum of four weeks", "4-to-6 weeks", "out three more games"
         -> that many games (the low end of a range) from the news date
     "week-to-week", "multi-week absence", "days to weeks" with no number: shown on the site as news to watch, no
     automatic change (no number to use).
  Estimate = the later of 1 and 2. Only kept when he misses at least one game after this week's.

The estimates go to model/auto_out.csv in manual_out.csv's format; manual_out.py merges them (a manual row for the same
player always wins), so they flow through the same paths as Jackson's own estimates: Out in the weekly projections,
missed games cut from his ROS and from his game-by-game chance to play. docs/data/injury_news_v5.csv lists every
estimate and watch item for the site. A dated copy of the feed's injury entries goes to automation/history/injury_news/
so the estimates can be scored against actual returns later (ESPN keeps no archive, so there is no past-season test).

  python automation/injury_news.py --data model/data --season 2026 --week 13
  python automation/injury_news.py --data model/data --season 2026 --week 13 --feed saved_feed.json --dry
"""
import argparse
import datetime as dt
import gzip
import json
import os
import re
import sys
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEED = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
POS = {"QB", "RB", "WR", "TE"}
LAST_WEEK = 17
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
       "a couple": 2, "a couple of": 2, "several": 3}
N = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
SEASON_END = re.compile(
    r"season-ending|(?:rest|remainder) of the (?:\d{4} )?(?:regular )?season|out for the (?:\d{4} )?season"
    r"|(?:not|n't) (?:expected|anticipate|likely|going)[^.]{0,60}?return[^.]{0,40}?(?:this season|season)"
    r"|won't be activated[^.]{0,60}this season", re.I)
NOT_END = re.compile(r"not (?:expected to be|considered) season-ending|not a season-ending|expected to return (?:later )?this"
                     r" season|could return (?:later|at some point)[^.]{0,40}season|return before the end of the", re.I)
AT_LEAST = re.compile(r"(?:at least|minimum of|a minimum of)\s+(?:the\s+)?(?:next\s+)?" + N
                      + r"\s+(?:more\s+|additional\s+)?(games?|weeks?|contests?)", re.I)
RANGE = re.compile(N + r"\s*(?:-to-|-| to )\s*" + N + r"\s+weeks", re.I)
MORE = re.compile(r"(?:miss|out for|sidelined for)\s+(?:at least\s+)?(?:the\s+)?(?:next\s+)?" + N
                  + r"\s+(?:more\s+)?(games?|weeks?|contests?)", re.I)
VAGUE = re.compile(r"week-to-week|multi-week|multiple weeks|days to weeks|extended absence|significant amount of time", re.I)
UA = {"User-Agent": "Mozilla/5.0 (FantasyOracle injury news)"}


def num(s):
    s = s.lower()
    return int(s) if s.isdigit() else NUM.get(s)


def about(text, last):
    """Sentences about this player: his last name, or starting with He / His."""
    sents = re.split(r"(?<=[.!?])\s+", text or "")
    return [s for s in sents if (last and last.lower() in s.lower()) or re.match(r"(he|his)\b", s.strip(), re.I)]


def parse_text(text, last):
    """-> (season_over, min_games, min_weeks, vague, evidence) from the news blurb, sentences about him only.
    Games count his team's games from the news date; weeks are calendar weeks from the news date."""
    sents = about(text, last)
    body = " ".join(sents)
    season_over = bool(SEASON_END.search(body)) and not NOT_END.search(body)
    g, wk, ev = 0, 0, []
    for rx in (AT_LEAST, MORE):
        for m in rx.finditer(body):
            k = num(m.group(1))
            if k and k <= 17:
                if m.group(2).lower().startswith("week"):
                    wk = max(wk, k)
                else:
                    g = max(g, k)
                ev.append(m.group(0))
    for m in RANGE.finditer(body):
        k = num(m.group(1))
        if k and k <= 17:
            wk = max(wk, k); ev.append(m.group(0))
    if season_over:
        ev.insert(0, SEASON_END.search(body).group(0))
    vague = bool(VAGUE.search(body)) and not season_over and not g and not wk
    if vague:
        ev.append(VAGUE.search(body).group(0))
    return season_over, g, wk, vague, "; ".join(dict.fromkeys(ev))


def fetch():
    req = urllib.request.Request(FEED, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def entries(feed):
    """Flatten the feed: one dict per injury entry with team (abbr), ESPN id, name, position, status, dates, text."""
    sys.path.insert(0, HERE)
    from injuries_live import FULL
    out = []
    for tm in feed.get("injuries", []):
        team = FULL.get(tm.get("displayName"))
        for x in tm.get("injuries", []):
            a = x.get("athlete") or {}
            m = re.search(r"/id/(\d+)/", json.dumps(a.get("links", [])))
            det = x.get("details") or {}
            out.append({"team": team, "espn_id": m.group(1) if m else None, "name": a.get("displayName", ""),
                        "last": a.get("lastName", "") or (a.get("displayName", "").split() or [""])[-1],
                        "pos": (a.get("position") or {}).get("abbreviation", ""), "status": x.get("status"),
                        "date": x.get("date"), "return": det.get("returnDate"), "injury": det.get("type") or "",
                        "short": x.get("shortComment") or "", "long": x.get("longComment") or ""})
    return out


def estimate(e, games, week):
    """games: this team's unplayed regular-season games [(week, gameday Timestamp)] from `week` on, in order.
    -> dict with out_through_week (or None), how, evidence, vague."""
    season_over, g_text, w_text, vague, ev = parse_text(e["short"] + " " + e["long"], e["last"])
    through, how = None, []
    if e.get("return"):
        rd = pd.Timestamp(e["return"][:10])
        miss = [w for w, d in games if d < rd]
        if miss:
            through = max(miss); how.append("ESPN return date " + e["return"][:10])
    if season_over:
        through = LAST_WEEK; how.append("news: season over")
    else:
        since = pd.Timestamp(e["date"][:10]) if e.get("date") else pd.Timestamp.now().normalize()
        cand = []
        if g_text:
            cand += [w for w, d in games if d >= since][:g_text]
            how.append(f"news: {g_text} games")
        if w_text:
            cand += [w for w, d in games if since <= d < since + pd.Timedelta(days=7 * w_text)]
            how.append(f"news: {w_text} weeks")
        if cand and (through is None or max(cand) > through):
            through = max(cand)
    return {"out_through_week": through, "how": ", ".join(how), "evidence": ev, "vague": vague}


def build(feed, data_dir, season, week, today=None):
    today = today or dt.date.today().isoformat()
    pl = pd.read_parquet(os.path.join(data_dir, "players.parquet"), columns=["gsis_id", "espn_id", "display_name",
                                                                              "position", "latest_team"])
    pl = pl.dropna(subset=["espn_id"]).assign(espn_id=lambda d: d["espn_id"].astype(str).str.replace(r"\.0$", "", regex=True))
    by_espn = pl.drop_duplicates("espn_id").set_index("espn_id")
    by_name = pl.drop_duplicates(["display_name", "latest_team"]).set_index(["display_name", "latest_team"])
    g = pd.read_parquet(os.path.join(data_dir, "games.parquet"))
    g = g[(g["season"] == season) & (g["game_type"] == "REG") & (g["week"] >= week) & (g["week"] <= LAST_WEEK)
          & g["home_score"].isna()]
    sched = {}
    for _, r in g.sort_values("week").iterrows():
        for t in (r["home_team"], r["away_team"]):
            sched.setdefault(t, []).append((int(r["week"]), pd.Timestamp(r["gameday"])))
    mp = os.path.join(ROOT, "model", "manual_out.csv")
    manual = {}
    if os.path.exists(mp):
        m = pd.read_csv(mp, dtype={"player_id": str})
        m = m[(m["season"] == season) & (m["out_through_week"] >= week)]
        manual = dict(zip(m["player_id"], m["out_through_week"].astype(int)))
    auto, site = [], []
    for e in entries(feed):
        if e["pos"] not in POS or e["status"] in (None, "Active") or not e["team"]:
            continue
        if e["espn_id"] in by_espn.index:
            pid = by_espn.at[e["espn_id"], "gsis_id"]
        elif (e["name"], e["team"]) in by_name.index:
            pid = by_name.at[(e["name"], e["team"]), "gsis_id"]
        else:
            pid = None
        games = sched.get(e["team"], [])
        est = estimate(e, games, week)
        first_after = min([w for w, _ in games if w > week], default=None)
        multi = est["out_through_week"] is not None and first_after is not None and est["out_through_week"] >= first_after
        if not multi and not est["vague"]:
            continue
        row = {"player_id": pid, "player": e["name"], "team": e["team"], "position": e["pos"], "status": e["status"],
               "espn_return": (e["return"] or "")[:10], "news_date": (e["date"] or "")[:10],
               "out_through_week": est["out_through_week"] if multi else None, "how": est["how"] if multi else "",
               "evidence": est["evidence"], "applied": bool(multi and pid) and pid not in manual,
               "manual_through_week": manual.get(pid), "news": e["short"][:300]}
        site.append(row)
        if multi and pid:
            auto.append({"player_id": pid, "player": e["name"], "team": e["team"], "position": e["pos"],
                         # first unplayed game he misses (never a game already played, e.g. the one he got hurt in)
                         "from_week": min(w for w, _ in games), "out_through_week": int(est["out_through_week"]),
                         "season": season,
                         "added": today, "source": f"ESPN news {row['news_date']} ({est['how']}): "
                                                    + (est["evidence"] or e["short"][:120])})
    return auto, site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "model", "data"))
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--feed", default=None, help="a saved copy of the ESPN feed instead of fetching it")
    ap.add_argument("--auto", default=os.path.join(ROOT, "model", "auto_out.csv"))
    ap.add_argument("--site", default=os.path.join(ROOT, "docs", "data"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    feed = json.load(open(a.feed)) if a.feed else fetch()
    auto, site = build(feed, a.data, a.season, a.week)
    cols = ["player_id", "player", "team", "position", "from_week", "out_through_week", "season", "added", "source"]
    s = pd.DataFrame(site)
    if a.dry:
        pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
        print(s.drop(columns=["news"]).to_string() if len(s) else "no multi-week news")
        return
    pd.DataFrame(auto, columns=cols).to_csv(a.auto, index=False)
    if len(s):
        s.to_csv(os.path.join(a.site, "injury_news_v5.csv"), index=False)
    else:
        pd.DataFrame(columns=["player_id", "player", "team", "position", "status", "espn_return", "news_date",
                              "out_through_week", "how", "evidence", "applied", "manual_through_week", "news"]).to_csv(
            os.path.join(a.site, "injury_news_v5.csv"), index=False)
    cat_p = os.path.join(a.site, "datasets.json")
    if os.path.exists(cat_p):
        cat = json.load(open(cat_p))
        cat["datasets"]["injury_news"] = {
            "file": "injury_news_v5.csv", "title": f"Injury news: multi-week absences (Week {a.week})",
            "description": "QB/RB/WR/TE whose ESPN injury entry points to missing games after this week: ESPN's return "
                           "date and the news text (season-ending, 'at least N games', 'N-to-M weeks'). applied = used "
                           "as an automatic return estimate (a manual estimate for the same player wins); vague news "
                           "('week-to-week') is listed without a change.",
            "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        json.dump(cat, open(cat_p, "w"), indent=1)
    # dated copy of the feed's injury entries (non-active), to score the estimates against actual returns later
    hist = os.path.join(ROOT, "automation", "history", "injury_news")
    os.makedirs(hist, exist_ok=True)
    keep = [e for e in entries(feed) if e["status"] not in (None, "Active")]
    with gzip.open(os.path.join(hist, dt.date.today().isoformat() + ".json.gz"), "wt") as f:
        json.dump(keep, f)
    print(f"injury news: {len(auto)} automatic return estimates, {len(s)} items on the site")


if __name__ == "__main__":
    main()

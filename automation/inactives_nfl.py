"""Game-day inactives from NFL.com's weekly "inactives" articles -> model/ref/inactives_<season>_wk<week>.csv.

NFL.com posts one article per week ("NFL Week N inactives: Players ruled out ...", plus separate ones for Thursday /
Monday games) and updates it as each kickoff window's inactives come out, about 90 minutes before kickoff. Each game
lists the teams by nickname in capitals followed by one "POS Name" line per inactive player. The file this writes is
the one cloud_run.py already reads (columns player, team, pos); inactives.py matches the names to player ids.

  python automation/inactives_nfl.py --season 2026 --week 7 --out model/ref      # prints a hash of the list
  python automation/inactives_nfl.py --parse saved_article.html                   # test the parser on a file
"""
import argparse
import difflib
import hashlib
import html
import json
import os
import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
NICK = {"CARDINALS": "ARI", "FALCONS": "ATL", "RAVENS": "BAL", "BILLS": "BUF", "PANTHERS": "CAR", "BEARS": "CHI",
        "BENGALS": "CIN", "BROWNS": "CLE", "COWBOYS": "DAL", "BRONCOS": "DEN", "LIONS": "DET", "PACKERS": "GB",
        "TEXANS": "HOU", "COLTS": "IND", "JAGUARS": "JAX", "CHIEFS": "KC", "RAMS": "LA", "CHARGERS": "LAC",
        "RAIDERS": "LV", "DOLPHINS": "MIA", "VIKINGS": "MIN", "PATRIOTS": "NE", "SAINTS": "NO", "GIANTS": "NYG",
        "JETS": "NYJ", "EAGLES": "PHI", "STEELERS": "PIT", "SEAHAWKS": "SEA", "49ERS": "SF", "BUCCANEERS": "TB",
        "TITANS": "TEN", "COMMANDERS": "WAS"}
LINE = re.compile(r"^([A-Z]{1,4})\s+([A-Z][^()]+?)\s*(\(.*\))?$")


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def text_lines(page):
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    t = re.sub(r"<(br|p|li|h[1-6]|div)[^>]*>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    return [l.strip() for l in t.split("\n") if l.strip()]


def parse(page):
    rows, team = [], None
    for l in text_lines(page):
        nick = l if l in NICK else (difflib.get_close_matches(l, NICK, n=1, cutoff=0.8) or [None])[0] \
            if re.fullmatch(r"[A-Z0-9]{4,12}", l) else None   # tolerate typos like "EAGLS"
        if nick:
            team = NICK[nick]
            continue
        if team is None:
            continue
        m = LINE.match(l)
        if m and len(m.group(2).split()) >= 2 and len(l) < 60:
            rows.append({"player": m.group(2).strip(), "team": team, "pos": m.group(1)})
        elif len(l) > 60 or l.startswith(("WHERE:", "WHEN:")) or l in ("AT", "VS."):
            team = None if (l.startswith("WHERE:") or len(l) > 60) else team
    return rows


def article_urls(season, week):
    urls = set()
    for hub in ["https://www.nfl.com/injuries/", "https://www.nfl.com/news/"]:
        try:
            page = fetch(hub)
        except Exception:
            continue
        for h in re.findall(r'href="(/news/[^"]*inactives[^"]*)"', page):
            if re.search(rf"week-{week}-", h) and str(season) in h:
                urls.add("https://www.nfl.com" + h)
    urls.add(f"https://www.nfl.com/news/nfl-week-{week}-inactives-players-ruled-out-sunday-{season}-season")
    return sorted(urls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--out", default="model/ref")
    ap.add_argument("--parse", default=None)
    a = ap.parse_args()
    if a.parse:
        rows = parse(open(a.parse, encoding="utf-8", errors="ignore").read())
        print(json.dumps({"players": len(rows), "teams": len({r["team"] for r in rows}), "sample": rows[:5]}))
        return
    rows = []
    for u in article_urls(a.season, a.week):
        try:
            rows += parse(fetch(u))
        except Exception:
            continue
    seen, uniq = set(), []
    for r in rows:
        k = (r["player"], r["team"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    path = os.path.join(a.out, f"inactives_{a.season}_wk{a.week}.csv")
    if uniq:
        os.makedirs(a.out, exist_ok=True)
        with open(path, "w") as f:
            f.write("player,team,pos\n" + "".join(f'"{r["player"]}",{r["team"]},{r["pos"]}\n' for r in uniq))
    h = hashlib.sha1(json.dumps(sorted(seen)).encode()).hexdigest()[:12] if uniq else None
    print(json.dumps({"inactives": len(uniq), "file": path if uniq else None, "hash": h}))


if __name__ == "__main__":
    main()

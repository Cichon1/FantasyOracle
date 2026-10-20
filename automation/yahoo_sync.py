"""Yahoo leagues -> the trade analyzer's data, through Yahoo's official Fantasy Sports API (no browser needed).

What it pulls for BDSM League, The League and Ezekiel 25.17: every team's roster (incl. IR), league settings and
scoring, standings, and every regular-season matchup (scores and Yahoo's projections). It writes them in the same
text format the in-Chrome scripts produced and runs the same builders (model/yahoo_league.py,
model/yahoo_standings.py), so the website reads exactly the files it read before. The League's next-season draft
pick ownership is not in Yahoo's API and stays a manual pull.

Sign-in: a Yahoo developer app (free). Its client ID and secret are GitHub secrets YAHOO_CLIENT_ID and
YAHOO_CLIENT_SECRET. The long-lived refresh token from the one-time connect step is stored encrypted with the
client secret in automation/state/yahoo_token.enc (useless without the secret).

  python automation/yahoo_sync.py connect --code CODE     # one time, with the code from Yahoo's sign-in page
  python automation/yahoo_sync.py sync                    # every update; does nothing until connected
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "automation", "state")
TOKEN = os.path.join(STATE, "yahoo_token.enc")
SITE = os.path.join(ROOT, "docs", "data")
API = "https://fantasysports.yahooapis.com/fantasy/v2"
REDIRECT = os.environ.get("YAHOO_REDIRECT", "https://cichon1.github.io/FantasyOracle/yahoo.html")
LEAGUES = ["126912", "200436", "727092"]
FILES = ["league_rosters_v5.csv", "leagues_v5.csv", "league_standings_v5.csv", "league_schedule_v5.csv"]


# ------------------------------------------------------------------------------------------- token handling
def _fernet():
    from cryptography.fernet import Fernet
    key = base64.urlsafe_b64encode(hashlib.sha256(os.environ["YAHOO_CLIENT_SECRET"].encode()).digest())
    return Fernet(key)


def _token_call(data):
    auth = base64.b64encode(f"{os.environ['YAHOO_CLIENT_ID']}:{os.environ['YAHOO_CLIENT_SECRET']}".encode()).decode()
    req = urllib.request.Request("https://api.login.yahoo.com/oauth2/get_token",
                                 data=urllib.parse.urlencode(data).encode(),
                                 headers={"Authorization": "Basic " + auth,
                                          "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def save_refresh(tok):
    os.makedirs(STATE, exist_ok=True)
    open(TOKEN, "wb").write(_fernet().encrypt(tok.encode()))


def access_token():
    refresh = _fernet().decrypt(open(TOKEN, "rb").read()).decode()
    j = _token_call({"grant_type": "refresh_token", "refresh_token": refresh, "redirect_uri": REDIRECT})
    if j.get("refresh_token") and j["refresh_token"] != refresh:
        save_refresh(j["refresh_token"])
    return j["access_token"]


# ---------------------------------------------------------------------------------------------- API reads
def strip(e):
    for x in e.iter():
        if "}" in x.tag:
            x.tag = x.tag.split("}", 1)[1]
    return e


def get(tok, path):
    req = urllib.request.Request(f"{API}/{path}", headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=60) as r:
        return strip(ET.fromstring(r.read()))


def t(e, path, default=""):
    x = e.find(path)
    return x.text.strip() if x is not None and x.text else default


def pull(tok):
    game_key = t(get(tok, "game/nfl"), ".//game_key")
    teams_lines, roster_lines, settings_lines, stand_lines = ["TEAMS"], [], [], []
    old = pd.read_csv(os.path.join(SITE, "leagues_v5.csv"), dtype=str).set_index("league_id")
    for lid in LEAGUES:
        lk = f"{game_key}.l.{lid}"
        # settings ---------------------------------------------------------------------------------------------
        s = get(tok, f"league/{lk}/settings")
        slots = []
        for rp in s.findall(".//roster_positions/roster_position"):
            slots += [t(rp, "position")] * int(t(rp, "count", "0"))
        names = {t(c, "stat_id"): t(c, "name") for c in s.findall(".//stat_categories/stats/stat")}
        mods = {}
        for m in s.findall(".//stat_modifiers/stats/stat"):
            sid = t(m, "stat_id")
            bon = [(float(t(b, "target", "0")), float(t(b, "points", "0"))) for b in m.findall(".//bonuses/bonus")]
            mods[names.get(sid, sid)] = (float(t(m, "value", "0")), bon)

        def per_pt(name):
            v, bon = mods.get(name, (0, []))
            txt = (f"{1 / v:g} yards per point" if v else "0")
            return txt + "".join(f"; {p:g} points at {int(tg)} yards" for tg, p in bon)

        def val(name):
            return f"{mods[name][0]:g}" if name in mods else ""
        S = {"Roster Positions": ", ".join(slots), "Max Teams": t(s, ".//num_teams"),
             "Passing Yards": per_pt("Passing Yards"), "Passing Touchdowns": val("Passing Touchdowns"),
             "Interceptions": val("Interceptions"), "Fumbles Lost": val("Fumbles Lost"), "Receptions": val("Receptions"),
             "Rushing Yards": per_pt("Rushing Yards"), "Receiving Yards": per_pt("Receiving Yards"),
             "Rushing Touchdowns": val("Rushing Touchdowns"), "Receiving Touchdowns": val("Receiving Touchdowns")}
        if "Return Yards" in mods:
            S["Return Yards"] = per_pt("Return Yards")
        # not in the API's settings: carried from the last pull
        if lid in old.index:
            S["Negative Points"] = str(old.at[lid, "negative_points"])
            S["Trade End Date"] = str(old.at[lid, "trade_end"])
            S["Playoffs"] = str(old.at[lid, "playoffs"])
        settings_lines += [f"{lid} | {k} | {v}" for k, v in S.items() if v != ""]
        # rosters -----------------------------------------------------------------------------------------------
        r = get(tok, f"league/{lk}/teams/roster")
        for tm in r.findall(".//teams/team"):
            tid = t(tm, "team_id")
            teams_lines.append(f"{lid} {tid} {t(tm, 'name')}")
            for p in tm.findall(".//roster/players/player"):
                pos_list = ",".join(e.text for e in p.findall("eligible_positions/position")
                                    if e.text and "/" not in e.text and e.text not in ("BN", "IR", "IL"))
                status = t(p, "status").replace(" ", "")
                roster_lines.append(" ".join(x for x in [lid, tid, t(p, "selected_position/position"),
                                                         t(p, "name/full"), t(p, "player_id"),
                                                         t(p, "editorial_team_abbr", "FA"), pos_list or t(p, "display_position"),
                                                         status] if x != ""))
        # standings and matchups -------------------------------------------------------------------------------
        st = get(tok, f"league/{lk}/standings")
        stand_lines += [f"LEAGUE {lid}", "STANDINGS"]
        for tm in st.findall(".//teams/team"):
            ot = tm.find(".//team_standings/outcome_totals")
            stand_lines.append(f"{t(tm, 'team_id')} {t(ot, 'wins', '0')}-{t(ot, 'losses', '0')}-{t(ot, 'ties', '0')} "
                               f"{t(tm, './/team_standings/points_for', '0')} {t(tm, './/team_standings/points_against', '0')}")
        stand_lines.append("MATCHUPS")
        end_week = int(t(s, ".//end_week", "17") or 17)
        po_start = int(t(s, ".//playoff_start_week", "15") or 15)
        for w in range(1, min(end_week, po_start - 1) + 1):
            sb = get(tok, f"league/{lk}/scoreboard;week={w}")
            for mu in sb.findall(".//matchups/matchup"):
                tt = mu.findall(".//teams/team")
                if len(tt) != 2:
                    continue
                vals = []
                for x in tt:
                    vals += [t(x, "team_id"), t(x, "team_points/total", "0") or "0",
                             t(x, "team_projected_points/total", "0") or "0"]
                stand_lines.append(f"{w} {vals[0]} {vals[3]} {vals[1]} {vals[2]} {vals[4]} {vals[5]}")
    return teams_lines, roster_lines, settings_lines, stand_lines


def sync():
    if not (os.path.exists(TOKEN) and os.environ.get("YAHOO_CLIENT_SECRET")):
        print(json.dumps({"yahoo": "not connected yet"}))
        return False
    sys.path.insert(0, os.path.join(ROOT, "model"))
    import yahoo_league as YL
    import yahoo_standings as YS
    tok = access_token()
    teams, rosters, settings, stand = pull(tok)
    work = os.path.join(ROOT, "model", "yahoo")
    os.makedirs(work, exist_ok=True)
    raw = os.path.join(work, "rosters_raw.txt")
    open(raw, "w", encoding="utf-8").write("\n".join(teams) + "\nROSTERS\n" + "\n".join(rosters)
                                           + "\nSETTINGS\n" + "\n".join(settings) + "\n")
    sraw = os.path.join(work, "standings_raw.txt")
    open(sraw, "w", encoding="utf-8").write("\n".join(stand) + "\n")
    out = os.path.join(work, "out")
    os.makedirs(out, exist_ok=True)
    YL.build(raw, os.path.join(SITE, "ros_v5.csv"), out)
    YS.build(sraw, os.path.join(out, "league_rosters_v5.csv"), out)
    # publish only what changed (ignoring the pull time), so the site isn't republished every half hour
    changed = []
    for f in FILES:
        new = pd.read_csv(os.path.join(out, f), dtype=str)
        cur_p = os.path.join(SITE, f)
        cur = pd.read_csv(cur_p, dtype=str) if os.path.exists(cur_p) else None
        drop = [c for c in ["pulled_at"] if c in new.columns]
        same = cur is not None and list(cur.columns) == list(new.columns) and \
            cur.drop(columns=drop).fillna("").equals(new.drop(columns=drop).fillna(""))
        if not same:
            new.to_csv(cur_p, index=False)
            changed.append(f)
    if changed:
        cat_p = os.path.join(SITE, "datasets.json")
        cat = json.load(open(cat_p))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ids = {"league_rosters_v5.csv": "league_rosters", "leagues_v5.csv": "leagues",
               "league_standings_v5.csv": "league_standings", "league_schedule_v5.csv": "league_schedule"}
        for f in changed:
            cat["datasets"][ids[f]]["updated"] = now
        cat["updated"] = now
        json.dump(cat, open(cat_p, "w"), indent=1)
    print(json.dumps({"yahoo": "synced", "changed": changed, "roster_rows": len(rosters)}))
    return bool(changed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["connect", "sync"])
    ap.add_argument("--code", default=None)
    a = ap.parse_args()
    if a.mode == "connect":
        j = _token_call({"grant_type": "authorization_code", "code": a.code.strip(), "redirect_uri": REDIRECT})
        save_refresh(j["refresh_token"])
        print("connected")
        return
    changed = False
    try:
        changed = sync()
    except Exception as e:      # Yahoo trouble never blocks the projections update
        print(json.dumps({"yahoo": "failed", "error": str(e)[:300]}))
        os.makedirs(STATE, exist_ok=True)
        json.dump({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": str(e)[:1000]},
                  open(os.path.join(STATE, "yahoo_last_error.json"), "w"), indent=1)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        open(out, "a").write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    main()

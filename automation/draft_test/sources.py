"""Readers for the Phase 1 draft-test data (automation/history/draft). Licensed pages are decrypted in memory only."""
import html as H
import json
import os
import re
from html.parser import HTMLParser

import pandas as pd

from draft_data import OUT, load

FP_TEAM = {"JAC": "JAX", "LAR": "LA", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "KCC": "KC",
           "NEP": "NE", "NOS": "NO", "SFO": "SF", "TBB": "TB", "GBP": "GB"}
# allow-list for the NFL draft-picks file: its other columns describe the player's later career (hindsight)
DRAFT_PICK_COLS = ["season", "round", "pick", "team", "gsis_id", "pfr_player_id", "pfr_player_name", "position",
                   "category", "college", "age"]


def kickoffs():
    """First regular-season game date per season (nflverse schedules)."""
    for p in [os.path.join(os.path.dirname(OUT), "..", "..", "model", "data", "games.parquet")]:
        if os.path.exists(p):
            g = pd.read_parquet(p)
            break
    else:
        g = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet")
    g = g[g["game_type"] == "REG"]
    return pd.to_datetime(g.groupby("season")["gameday"].min()).to_dict()


def ffc(fmt, teams, year):
    p = os.path.join(OUT, "ffc", f"ffc_{fmt}_{teams}_{year}.json")
    if not os.path.exists(p):
        return None, None
    j = json.load(open(p))
    d = pd.DataFrame(j.get("players", []))
    if len(d):
        d["team"] = d["team"].replace(FP_TEAM)
    return d, j.get("meta", {})


class _Table(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.cur, self.row, self.cell, self.in_cell = [], None, None, None, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.cur = {"id": a.get("id", ""), "rows": []}
        elif tag == "tr" and self.cur is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell, self.in_cell = [], True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
            self.in_cell = False
        elif tag == "tr" and self.row is not None and self.cur is not None:
            if self.row:
                self.cur["rows"].append(self.row)
            self.row = None
        elif tag == "table" and self.cur is not None:
            self.tables.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)


def fp_adp(fmt, year):
    """FantasyPros ADP table: player, team, pos, one column per platform, AVG."""
    p = os.path.join(OUT, "fp_adp", f"fp_adp_{fmt}_{year}.enc")
    if not os.path.exists(p):
        return None
    t = _Table()
    t.feed(load(p, enc=True))
    tabs = sorted(t.tables, key=lambda x: len(x["rows"]), reverse=True)
    if not tabs:
        return None
    rows = tabs[0]["rows"]
    head = [H.unescape(h) for h in rows[0]]
    body = [r for r in rows[1:] if len(r) == len(head)]
    d = pd.DataFrame(body, columns=head)
    pcol = next((c for c in d.columns if c.lower().startswith("player")), None)
    poscol = next((c for c in d.columns if c.upper() in ("POS", "POSITION")), None)
    m = d[pcol].str.extract(r"^(?P<player>.+?)\s+(?P<team>[A-Z]{2,3})?\s*(?:\((?P<bye>\d+)\))?\s*(?:[A-Z]+)?$")
    d["player"], d["team"] = m["player"].fillna(d[pcol]), m["team"].replace(FP_TEAM)
    if poscol:
        d["pos"] = d[poscol].str.extract(r"^([A-Z]+)")[0].replace({"DST": "DEF"})
    for c in d.columns:
        if c not in (pcol, poscol, "player", "team", "pos"):
            d[c] = pd.to_numeric(d[c].str.replace(",", ""), errors="coerce")
    return d


ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}
ESPN_TEAM = {1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN", 11: "IND",
             12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI",
             22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL",
             34: "HOU"}


def espn(year):
    """ESPN average draft position (ownership.averageDraftPosition) per player; preseason draft rank where present."""
    p = os.path.join(OUT, "espn_adp", f"espn_adp_{year}.enc")
    if not os.path.exists(p):
        return None
    j = json.loads(load(p, enc=True))
    rows = []
    for x in j.get("players", []):
        pl = x.get("player", {})
        own = pl.get("ownership") or {}
        rk = (pl.get("draftRanksByRankType") or {}).get("PPR", {})
        rows.append({"player": pl.get("fullName"), "pos": ESPN_POS.get(pl.get("defaultPositionId")),
                     "team": ESPN_TEAM.get(pl.get("proTeamId")), "adp": own.get("averageDraftPosition"),
                     "espn_rank": rk.get("rank")})
    d = pd.DataFrame(rows)
    d["adp"] = pd.to_numeric(d["adp"], errors="coerce")
    return d[d["adp"].notna() & (d["adp"] > 0)]


def yahoo(year):
    """Yahoo draft analysis: average pick / round / percent drafted from real Yahoo drafts that season."""
    import xml.etree.ElementTree as ET
    p = os.path.join(OUT, "yahoo_adp", f"yahoo_adp_{year}.enc")
    if not os.path.exists(p):
        return None
    j = json.loads(load(p, enc=True))
    rows = {}
    for ch in j["pages"]:
        for pl in ET.fromstring(ch).iter("player"):
            key = pl.findtext("player_key")
            da = pl.find("draft_analysis")
            rows[key] = {"player": pl.findtext("name/full"), "pos": pl.findtext("display_position"),
                         "team": (pl.findtext("editorial_team_abbr") or "").upper(),
                         "adp": da.findtext("average_pick") if da is not None else None,
                         "avg_round": da.findtext("average_round") if da is not None else None,
                         "pct_drafted": da.findtext("percent_drafted") if da is not None else None}
    d = pd.DataFrame(rows.values())
    for c in ["adp", "avg_round", "pct_drafted"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["team"] = d["team"].replace({**FP_TEAM, "JAC": "JAX", "LAR": "LA", "WSH": "WAS"})
    d["pos"] = d["pos"].str.split(",").str[0].replace({"DST": "DEF"})
    return d[d["adp"].notna() & (d["adp"] > 0)]


def mfl(year, fmt="ppr"):
    """MFL ADP (averagePick) named through that season's MFL player list."""
    p = os.path.join(OUT, "mfl_adp", f"mfl_adp_{fmt}_{year}.enc")
    pp = os.path.join(OUT, "mfl_adp", f"mfl_players_{year}.enc")
    if not os.path.exists(p) or not os.path.exists(pp):
        return None
    adp = json.loads(load(p, enc=True)).get("adp", {}).get("player", [])
    pl = {x["id"]: x for x in json.loads(load(pp, enc=True)).get("players", {}).get("player", [])}
    rows = []
    for a in adp:
        x = pl.get(a.get("id"), {})
        nm = x.get("name", "")
        if "," in nm:                     # MFL names are "Last, First"
            last, first = nm.split(",", 1)
            nm = first.strip() + " " + last.strip()
        rows.append({"player": nm, "pos": {"Def": "DEF", "PK": "K"}.get(x.get("position"), x.get("position")),
                     "team": FP_TEAM.get(x.get("team"), x.get("team")), "adp": a.get("averagePick"), "mfl_id": a.get("id")})
    d = pd.DataFrame(rows)
    d["adp"] = pd.to_numeric(d["adp"], errors="coerce")
    return d[d["adp"].notna() & (d["adp"] > 0)]

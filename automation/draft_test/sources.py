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

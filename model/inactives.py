"""
Game-day inactives (posted about 90 minutes before each kickoff).

The scheduled Sunday runs read the FantasyPros inactives page and write ref/inactives_<season>_wk<week>.csv with
columns player, team, pos (one row per inactive player; extra columns are ignored). This module matches those
names to nflverse player ids and records them as ruled Out for the week, so the projection run drops them and
re-projects their teammates as if the player had been ruled out on the final injury report.

In the 2018-2025 backtest, knowing the inactives would have raised the official projection's weekly rank
correlation from 0.432 to 0.450 and R² from 0.496 to 0.510.
"""
import glob
import os
import re
import unicodedata

import pandas as pd

TEAM_FIX = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA", "GBP": "GB", "KCC": "KC",
            "NEP": "NE", "NOS": "NO", "SFO": "SF", "TBB": "TB", "LVR": "LV"}
SKILL = {"QB", "RB", "WR", "TE", "FB"}


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def candidates(data_dir, season, week):
    """Every player on a team for the week (weekly roster, falling back to the latest week), with name variants."""
    r = pd.read_parquet(os.path.join(data_dir, "roster", f"roster_weekly_{season}.parquet"))
    r = r.dropna(subset=["gsis_id"])
    r = r[r["week"] == (week if (r["week"] == week).any() else r["week"].max())]
    out = []
    for c in ["full_name", "football_name"]:
        if c in r:
            out.append(pd.DataFrame({"player_id": r["gsis_id"], "team": r["team"], "position": r["position"],
                                     "key": r[c].map(norm)}))
    if {"football_name", "last_name"} <= set(r.columns):
        out.append(pd.DataFrame({"player_id": r["gsis_id"], "team": r["team"], "position": r["position"],
                                 "key": (r["football_name"].fillna("") + " " + r["last_name"].fillna("")).map(norm)}))
    c = pd.concat(out).drop_duplicates(["player_id", "key"])
    c["team"] = c["team"].replace({"LAR": "LA"})
    return c[c["key"] != ""]


def match(path, data_dir, season, week):
    """-> (matched DataFrame player_id/team/name/position, list of unmatched names) for skill positions.
    Passes: exact name on the team; first initial + last name; nickname (same team, position and last name, unique)."""
    x = pd.read_csv(path)
    x.columns = [c.strip().lower() for c in x.columns]
    name_col = next(c for c in x.columns if c in ("player", "name", "player_name"))
    x = x.rename(columns={name_col: "name"})
    x["team"] = x["team"].astype(str).str.strip().str.upper().replace(TEAM_FIX)
    x["pos"] = x["pos"].astype(str).str.upper().str.strip() if "pos" in x.columns else ""
    x = x[x["pos"].isin(SKILL) | x["pos"].isin(["", "NAN"])].drop_duplicates(["name", "team"]).reset_index(drop=True)
    x["key"] = x["name"].map(norm)
    tok = x["key"].str.split()
    x["k2"] = tok.map(lambda t: (t[0][:1] + " " + t[-1]) if len(t) > 1 else "")
    x["last"] = tok.map(lambda t: t[-1] if t else "")
    c = candidates(data_dir, season, week)
    ct = c["key"].str.split()
    c = c.assign(k2=ct.map(lambda t: (t[0][:1] + " " + t[-1]) if len(t) > 1 else ""), last=ct.map(lambda t: t[-1]))
    x["player_id"], x["position"] = None, None
    passes = [(["team", "key"], c.drop_duplicates(["team", "key"], keep=False)),
              (["team", "k2"], c.drop_duplicates(["player_id", "k2"]).drop_duplicates(["team", "k2"], keep=False)),
              (["team", "pos", "last"], c.assign(pos=c["position"]).drop_duplicates(["player_id", "last"])
               .drop_duplicates(["team", "pos", "last"], keep=False))]
    for keys, cc in passes:
        todo = x["player_id"].isna()
        if not todo.any():
            break
        if "pos" in keys:
            cc = cc.rename(columns={"position": "_p"}).assign(position=lambda d: d["_p"])
        hit = x.loc[todo, keys].merge(cc[keys + ["player_id", "position"]], on=keys, how="left")
        hit.index = x.index[todo]
        x.loc[todo, "player_id"] = hit["player_id"]
        x.loc[todo, "position"] = hit["position"]
    miss = x["player_id"].isna()
    unmatched = (x.loc[miss, "name"] + " (" + x.loc[miss, "team"] + ")").tolist()
    ok = x[~miss][["player_id", "team", "name", "position"]]
    return ok.drop_duplicates("player_id"), unmatched


def apply(matched, data_dir, season, week):
    """Mark the matched players ruled Out for the week in the injury file the model reads."""
    f = os.path.join(data_dir, "inj", f"inj_{season}.parquet")
    inj = pd.read_parquet(f)
    add = pd.DataFrame({"season": season, "season_type": "REG", "game_type": "REG", "team": matched["team"].values,
                        "week": week, "gsis_id": matched["player_id"].values, "position": matched["position"].values,
                        "full_name": matched["name"].values, "report_status": "Out"})
    old = inj[(inj["week"] == week) & inj["gsis_id"].isin(set(matched["player_id"]))]
    for c in ["report_primary_injury", "practice_status", "practice_primary_injury"]:
        if c in inj.columns:
            add[c] = add["gsis_id"].map(old.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")[c])
    inj = pd.concat([inj, add[[c for c in add.columns if c in inj.columns]]], ignore_index=True)
    inj.to_parquet(f)
    return len(add)


def roster_inactive(data_dir, season, week):
    """Players listed inactive (INA) on the week's nflverse roster (filled in after the games)."""
    f = os.path.join(data_dir, "roster", f"roster_weekly_{season}.parquet")
    if not os.path.exists(f):
        return set()
    r = pd.read_parquet(f, columns=["week", "gsis_id", "status"])
    return set(r.loc[(r["week"] == week) & (r["status"] == "INA"), "gsis_id"].dropna())


def roster_inactive_rows(data_dir, season, week):
    """Same shape as match()'s result: skill players listed INA on the week's nflverse roster (nflverse adds
    game-day inactives some hours after they are posted, so this mainly backs up the FantasyPros list)."""
    f = os.path.join(data_dir, "roster", f"roster_weekly_{season}.parquet")
    if not os.path.exists(f):
        return pd.DataFrame(columns=["player_id", "team", "name", "position"])
    r = pd.read_parquet(f, columns=["week", "gsis_id", "status", "team", "full_name", "position"])
    r = r[(r["week"] == week) & (r["status"] == "INA") & r["position"].isin(SKILL)].dropna(subset=["gsis_id"])
    return pd.DataFrame({"player_id": r["gsis_id"], "team": r["team"].replace({"LAR": "LA"}), "name": r["full_name"],
                         "position": r["position"]}).drop_duplicates("player_id")

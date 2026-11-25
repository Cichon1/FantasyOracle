"""Phase 2 foundation (PREREG.md): the player pool each preseason and every player's actual weekly points in each
league's exact scoring - what every draft board is graded against and what the season simulations score.

Pool: every player with an FFC ADP that season (12-team PPR; Yahoo / MFL join in the GitHub run), matched to nflverse
player ids by name + position (aliases, then a position-change fallback). Players who never played score 0 (busts
count; PREREG section 2).

Scoring (weekly, regular-season NFL weeks 1-17 = fantasy weeks 1-14 + playoffs 15-17, the same for every season):
  both leagues: 25 pass yds / pt, pass TD 6, INT -0.5, 10 rush / rec yds per pt, rush / rec TD 6, reception 1,
                fumble lost -2, 2-pt conversion 2, special-teams TD 6
  The League:   +10 for a 400-yard passing game, 150-yard rushing game, 175-yard receiving game
  (BDSM's kick / punt return yards aren't in the weekly player stats: not modeled, as on the site. K and DEF are team
   units scored by lab17_kdef.py.)

Output (model/lab/draft/):
  pool_<season>.parquet          one row per pool player: ids, ADP, actual season points / games for both leagues
  weekly_points.parquet          player-week points for both leagues (seasons 2010-2025)
  adp_accuracy.json              each ADP source's rank correlation with actual points, by season and position

Contracts (nflverse / OverTheCap historical contracts, free): only what was signed before that season's draft
(PREREG row "Contracts are a current snapshot"): year_signed < season, or year_signed == season for offseason-only
types (Drafted, UDFA, UFA, RFA, ERFA, Franchise, Transition). Extensions, street / practice-squad signings in the
season year count as unknown. Outcome fields (status, amount_earned, percent_earned, effective_apy) and season cap
numbers (restructured later) are never read.

  python automation/draft_test/board_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import sources as S  # noqa: E402
from audit_draft_data import norm  # noqa: E402

DATA = os.path.join(ROOT, "model", "data")
OUT = os.path.join(ROOT, "model", "lab", "draft")
NFLV = "https://github.com/nflverse/nflverse-data/releases/download"
SEASONS = [y for y in range(2010, 2026) if y != 2011]          # FFC 2011 dropped by the audit
SKILL = ["QB", "RB", "WR", "TE"]


def weekly(y):
    p = os.path.join(DATA, f"week_{y}.parquet")
    w = pd.read_parquet(p) if os.path.exists(p) else pd.read_parquet(f"{NFLV}/stats_player/stats_player_week_{y}.parquet")
    w = w[(w["season_type"] == "REG") & (w["week"] <= 17)].copy()
    f = lambda c: w[c].fillna(0) if c in w else 0
    base = (f("passing_yards") / 25 + 6 * f("passing_tds") - 0.5 * f("passing_interceptions")
            + f("rushing_yards") / 10 + 6 * f("rushing_tds") + f("receptions") + f("receiving_yards") / 10
            + 6 * f("receiving_tds")
            - 2 * (f("sack_fumbles_lost") + f("rushing_fumbles_lost") + f("receiving_fumbles_lost"))
            + 2 * (f("passing_2pt_conversions") + f("rushing_2pt_conversions") + f("receiving_2pt_conversions"))
            + 6 * f("special_teams_tds"))
    bonus = 10 * ((f("passing_yards") >= 400).astype(int) + (f("rushing_yards") >= 150).astype(int)
                  + (f("receiving_yards") >= 175).astype(int))
    pid = "player_id" if "player_id" in w else "gsis_id"
    return pd.DataFrame({"season": y, "week": w["week"].values, "gsis_id": w[pid].values,
                         "name": w["player_display_name"].values, "pos": w["position"].values,
                         "team": w["team"].values if "team" in w else w["recent_team"].values,
                         "pts_bdsm": base.values, "pts_league": (base + bonus).values, "bonus": bonus.values})


def players_table():
    p = os.path.join(DATA, "players.parquet")
    pl = pd.read_parquet(p if os.path.exists(p) else f"{NFLV}/players/players.parquet")
    idc = "gsis_id" if "gsis_id" in pl else "player_id"
    pl = pl[[idc, "display_name", "position"]].rename(columns={idc: "gsis_id", "display_name": "pname", "position": "ppos"})
    pl["key"] = pl["pname"].map(norm)
    return pl.dropna(subset=["gsis_id"])


OFFSEASON_TYPES = {"Drafted", "UDFA", "UFA", "RFA", "ERFA", "Franchise", "Transition"}
CONTRACT_COLS = ["team", "contract_type", "year_signed", "yrs", "total", "apy", "guarantees"]   # allow-list


def contracts_long():
    p = os.path.join(DATA, "contracts.parquet")
    d = pd.read_parquet(p if os.path.exists(p) else f"{NFLV}/contracts/historical_contracts.parquet")
    rows = [dict({k: x.get(k) for k in CONTRACT_COLS}, gsis_id=g)
            for g, h in zip(d["gsis_id"], d["contract_history"]) if g and h is not None for x in h]
    ch = pd.DataFrame(rows).drop_duplicates()
    ch = ch[ch["contract_type"].isin(OFFSEASON_TYPES | {"Extension", "SFA", "Other"})]
    ch["year_signed"] = ch["year_signed"].astype(float)
    draft = d.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")[["draft_year", "draft_round", "draft_overall"]]
    return ch, draft


def contract_features(ids, pos, y, ch, draft):
    """the contract each player was on going into season y, from deals known before the draft."""
    ok = ch[(ch["year_signed"] < y) | ((ch["year_signed"] == y) & ch["contract_type"].isin(OFFSEASON_TYPES))]
    ok = ok[ok["gsis_id"].isin(set(ids))].sort_values(["year_signed", "apy"])
    cur = ok.groupby("gsis_id").tail(1).set_index("gsis_id")
    f = pd.DataFrame(index=pd.Index(ids, name="gsis_id"))
    f["c_type"] = cur["contract_type"]
    f["c_apy"] = cur["apy"]
    f["c_gtd_share"] = (cur["guarantees"] / cur["total"].where(cur["total"] > 0))
    f["c_years_left"] = cur["year_signed"] + cur["yrs"] - 1 - y        # 0 -> playing out the last year
    f["c_contract_year"] = (f["c_years_left"] <= 0).astype(float).where(f["c_type"].notna())
    f["c_new_deal"] = (cur["year_signed"] == y).astype(float).reindex(f.index)
    f["c_rookie_deal"] = (cur["contract_type"].isin(["Drafted", "UDFA"])).astype(float).reindex(f.index)
    f = f.join(draft, how="left")
    f["years_pro"] = y - f["draft_year"]
    f["pos"] = list(pos)
    f["c_apy_pos_rank"] = f.groupby("pos")["c_apy"].rank(ascending=False)   # among this pool, by position
    return f.drop(columns=["pos"]).reset_index()


def match(pool, pl, wk):
    """name + position first; then name with any position if unique; then name among that season's players."""
    pool = pool.copy()
    pool["key"] = pool["player"].map(norm)
    a = pl.drop_duplicates(["key", "ppos"])
    m = pool.merge(a[["key", "ppos", "gsis_id"]], left_on=["key", "pos"], right_on=["key", "ppos"], how="left")
    uniq = pl.groupby("key").filter(lambda g: g["gsis_id"].nunique() == 1).drop_duplicates("key")
    m["gsis_id"] = m["gsis_id"].fillna(m["key"].map(uniq.set_index("key")["gsis_id"]))
    sw = wk.assign(key=wk["name"].map(norm)).drop_duplicates(["key", "pos"])
    m["gsis_id"] = m["gsis_id"].fillna(m.set_index(["key", "pos"]).index.map(sw.set_index(["key", "pos"])["gsis_id"]).to_series(index=m.index))
    return m.drop(columns=["ppos"])


def main():
    os.makedirs(OUT, exist_ok=True)
    pl = players_table()
    ch, draft = contracts_long()
    weeks, acc = [], {}
    for y in SEASONS:
        wk = weekly(y)
        weeks.append(wk)
        fd, meta = S.ffc("ppr", 12, y)
        pool = fd.rename(columns={"name": "player", "position": "pos", "adp": "adp_ffc", "stdev": "adp_ffc_sd",
                                  "high": "adp_ffc_high", "low": "adp_ffc_low"})
        pool["pos"] = pool["pos"].replace({"PK": "K", "DEF": "DEF"})
        pool = pool[["player", "pos", "team", "adp_ffc", "adp_ffc_sd", "adp_ffc_high", "adp_ffc_low"]]
        sk = pool[pool["pos"].isin(SKILL)]
        sk = match(sk, pl, wk)
        tot = wk.groupby("gsis_id").agg(pts_bdsm=("pts_bdsm", "sum"), pts_league=("pts_league", "sum"),
                                        bonus=("bonus", "sum"), games=("week", "nunique"))
        sk = sk.merge(tot, left_on="gsis_id", right_index=True, how="left")
        for c in ["pts_bdsm", "pts_league", "bonus", "games"]:
            sk[c] = sk[c].fillna(0)        # never played: 0 (busts count)
        sk["season"] = y
        sk["matched"] = sk["gsis_id"].notna()
        cf = contract_features(sk.loc[sk["matched"], "gsis_id"].tolist(), sk.loc[sk["matched"], "pos"], y, ch, draft)
        sk = sk.merge(cf.drop_duplicates("gsis_id"), on="gsis_id", how="left")
        sk.to_parquet(os.path.join(OUT, f"pool_{y}.parquet"), index=False)
        # yardstick: FFC ADP vs actual points, within position and overall (top 150 by ADP)
        r = {"pool": int(len(sk)), "matched": int(sk["matched"].sum()), "ffc_window_end": meta.get("end_date"),
             "with_contract": int(sk["c_type"].notna().sum())}
        top = sk.nsmallest(150, "adp_ffc")
        for lg in ["bdsm", "league"]:
            r[f"overall_{lg}"] = round(float(top["adp_ffc"].rank().corr(top[f"pts_{lg}"].rank(ascending=False))), 3)
        for p in SKILL:
            g = sk[sk["pos"] == p].nsmallest({"QB": 24, "RB": 48, "WR": 60, "TE": 24}[p], "adp_ffc")
            r[p] = round(float(g["adp_ffc"].rank().corr(g["pts_bdsm"].rank(ascending=False))), 3)
        acc[y] = r
        print(y, r)
    pd.concat(weeks, ignore_index=True).to_parquet(os.path.join(OUT, "weekly_points.parquet"), index=False)
    json.dump(acc, open(os.path.join(OUT, "adp_accuracy.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

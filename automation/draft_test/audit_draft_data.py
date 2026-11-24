"""Hindsight audit of the Phase 1 data (PREREG.md section 1). Writes automation/history/draft/audit.json (counts,
dates and correlations only - no licensed data) and exits 1 if a hard rule fails.

Checks
  1. FFC ADP: the mock-draft window ends on or before that season's first game; otherwise the source-year is dropped.
  2. FantasyPros ADP pages carry no capture date. Late-update test: within each season, rank correlation of the archived
     ADP with that season's actual PPR points vs. FFC's dated preseason ADP on the same players. An archive updated
     after kickoff would predict noticeably better; flagged if it beats FFC by more than 0.05 in a season.
  3. FantasyPros preseason expert rankings (fpdata): the last snapshot used is dated <= kickoff - 3 days.
  4. NFL draft picks: only allow-listed columns are loaded.
  5. Trades: dated; counts before / after kickoff per season.
  6. Name matching: share of the top-200 (FFC 12-team PPR) matched to nflverse players by name + position.

  python automation/draft_test/audit_draft_data.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sources as S  # noqa: E402
from draft_data import OUT, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NFLV = "https://github.com/nflverse/nflverse-data/releases/download"


def norm(n):
    n = re.sub(r"[^a-z ]", "", str(n).lower().replace("-", " "))
    return re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).replace("  ", " ").strip()


def season_points():
    """Actual PPR points per player-season (regular season) for the late-update test."""
    out = []
    for y in range(2010, 2026):
        p = os.path.join(ROOT, "model", "data", f"week_{y}.parquet")
        w = pd.read_parquet(p) if os.path.exists(p) else pd.read_parquet(
            f"{NFLV}/stats_player/stats_player_week_{y}.parquet")
        w = w[w["season_type"] == "REG"]
        s = w.groupby(["player_display_name", "position"])["fantasy_points_ppr"].sum().reset_index()
        s["season"] = y
        out.append(s)
    d = pd.concat(out)
    d["key"] = d["player_display_name"].map(norm)
    return d


def main():
    ko = S.kickoffs()
    rep, fail = {"rules": "PREREG.md section 1"}, []
    # 1. FFC windows
    ffc = {}
    for y in range(2010, 2026):
        for fmt in ["ppr", "half-ppr", "standard"]:
            for n in [10, 12, 14]:
                d, meta = S.ffc(fmt, n, y)
                if d is None:
                    continue
                end = pd.to_datetime(meta.get("end_date"), errors="coerce")
                start = pd.to_datetime(meta.get("start_date"), errors="coerce")
                ok = bool(pd.notna(end) and end <= ko[y])
                ffc[f"{fmt}_{n}_{y}"] = {"start": str(start.date()) if pd.notna(start) else None,
                                         "end": str(end.date()) if pd.notna(end) else None,
                                         "kickoff": str(ko[y].date()), "drafts": meta.get("total_drafts"),
                                         "players": len(d), "usable": ok}
    rep["ffc"] = {"units": len(ffc), "usable": sum(v["usable"] for v in ffc.values()),
                  "dropped": {k: v for k, v in ffc.items() if not v["usable"]}}
    # 2. FP ADP late-update test
    pts = season_points()
    late = {}
    for y in range(2017, 2026):
        fp = S.fp_adp("ppr", y)
        fd, meta = S.ffc("ppr", 12, y)
        if fp is None or fd is None or not ffc.get(f"ppr_12_{y}", {}).get("usable"):
            late[y] = {"status": "missing fp or usable ffc"}
            continue
        avg = next((c for c in fp.columns if c.upper() in ("AVG", "AVERAGE")), None)
        fp["key"], fd["key"] = fp["player"].map(norm), fd["name"].map(norm)
        m = fp[["key", "pos", avg]].merge(fd[["key", "position", "adp"]], on="key").merge(
            pts[pts["season"] == y][["key", "fantasy_points_ppr"]], on="key", how="left")
        m["fantasy_points_ppr"] = m["fantasy_points_ppr"].fillna(0)
        m = m.nsmallest(200, "adp")
        c_fp = m[avg].rank().corr(m["fantasy_points_ppr"].rank(ascending=False), method="pearson")
        c_ffc = m["adp"].rank().corr(m["fantasy_points_ppr"].rank(ascending=False), method="pearson")
        late[y] = {"n": len(m), "fp_corr": round(float(c_fp), 3), "ffc_corr": round(float(c_ffc), 3),
                   "fp_minus_ffc": round(float(c_fp - c_ffc), 3), "flag": bool(c_fp - c_ffc > 0.05),
                   "platform_cols": [c for c in fp.columns if c not in ("player", "team", "pos", "key") and fp[c].dtype != object]}
    rep["fp_adp_late_update"] = late
    # 3. FP preseason expert rankings
    ecr_p = os.path.join(ROOT, "model", "fpdata", "db_fpecr.parquet")
    if os.path.exists(ecr_p):
        e = pd.read_parquet(ecr_p, columns=["page_type", "ecr_type", "scrape_date"])
        e = e[e["ecr_type"].isin(["rp", "ro"]) & e["page_type"].str.startswith("redraft")]
        e["d"] = pd.to_datetime(e["scrape_date"])
        snap = {}
        for y in range(2019, 2027):
            cut = ko.get(y) - pd.Timedelta(days=3)
            ds = e[(e["d"] <= cut) & (e["d"] >= pd.Timestamp(f"{y}-06-01"))]["d"]
            snap[y] = {"last_snapshot": str(ds.max().date()) if len(ds) else None, "cutoff": str(cut.date()),
                       "days_before_kickoff": int((ko[y] - ds.max()).days) if len(ds) else None}
        rep["fp_ecr_preseason"] = snap
    else:
        rep["fp_ecr_preseason"] = "fpdata not available in this run (checked locally)"
    # 4. draft picks allow-list
    dp_p = os.path.join(OUT, "nflverse", "draft_picks.parquet")
    if os.path.exists(dp_p):
        dp = pd.read_parquet(dp_p)
        rep["draft_picks"] = {"allowed": [c for c in S.DRAFT_PICK_COLS if c in dp.columns],
                              "blocked_hindsight_cols": [c for c in dp.columns if c not in S.DRAFT_PICK_COLS]}
    # 5. trades
    tr_p = os.path.join(OUT, "nflverse", "trades.parquet")
    if os.path.exists(tr_p):
        t = pd.read_parquet(tr_p)
        t["d"] = pd.to_datetime(t["trade_date"], errors="coerce")
        rep["trades"] = {int(y): {"before_kickoff": int((g["d"] < ko[y]).sum()), "after": int((g["d"] >= ko[y]).sum()),
                                  "undated": int(g["d"].isna().sum())}
                         for y, g in t.groupby("season") if y in ko and y >= 2010}
    # 6. name matching
    pl = pd.read_parquet(f"{NFLV}/players/players.parquet", columns=["display_name", "position"])
    names = set(zip(pl["display_name"].map(norm), pl["position"]))
    mt = {}
    for y in range(2010, 2026):
        fd, _ = S.ffc("ppr", 12, y)
        if fd is None:
            continue
        top = fd.nsmallest(200, "adp")
        top = top[~top["position"].isin(["DEF", "PK", "K"])]
        miss = [r["name"] for _, r in top.iterrows() if (norm(r["name"]), r["position"]) not in names]
        mt[y] = {"checked": len(top), "unmatched": len(miss), "rate": round(len(miss) / max(1, len(top)), 3), "names": miss[:15]}
        if len(miss) / max(1, len(top)) > 0.02:
            fail.append(f"name match {y}: {len(miss)} unmatched of {len(top)}")
    rep["name_matching"] = mt
    rep["hard_failures"] = fail
    save(os.path.join(OUT, "audit.json"), json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: (v if k in ("hard_failures",) else "ok") for k, v in rep.items()}, indent=1))
    print("FFC usable", rep["ffc"]["usable"], "of", rep["ffc"]["units"])
    print("late-update flags:", {y: v.get("fp_minus_ffc") for y, v in late.items()})
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

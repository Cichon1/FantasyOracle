"""Manual return estimates (model/manual_out.csv), added Nov 16, 2026 at Jackson's request.

When news says a player will miss several weeks before any injury report shows it (or the report only covers one week),
list him in manual_out.csv with the first and last weeks he is expected to miss (from_week, out_through_week). Then:
  - weekly projections: automation/injuries_live.py marks him Out for projected weeks from_week..out_through_week
    (never the week he got hurt in, which may already be played);
  - rest of season (ros.py): his team's games through out_through_week count as missed - the future part of his ROS
    points is cut to the games left after his return, the expert blend is skipped for him (the experts may not have
    moved yet), and his game-by-game chance to play (play_curve, used by the trade analyzer) is 0 for those games.
Rows for other seasons, or whose out_through_week has passed, do nothing. Delete a row when he's back (or to undo).

Nov 28, 2026: automatic estimates from injury news (automation/injury_news.py -> model/auto_out.csv, rewritten every
update) are merged in, same format; a manual row for the same player always wins. Rows carry kind = "manual" / "news".
"""
import os

import pandas as pd

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_out.csv")
AUTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_out.csv")


def _read(path, auto=AUTO):
    """manual_out.csv plus the automatic news estimates for players without a manual row."""
    frames = []
    if os.path.exists(path):
        frames.append(pd.read_csv(path, dtype={"player_id": str}).assign(kind="manual"))
    if auto and path == PATH and os.path.exists(auto):
        a = pd.read_csv(auto, dtype={"player_id": str})
        if len(a):
            have = set(frames[0]["player_id"]) if frames else set()
            frames.append(a[~a["player_id"].isin(have)].assign(kind="news"))
    return pd.concat(frames, ignore_index=True) if frames else None


def load(season, week=None, path=PATH, asof=None):
    """{player_id: out_through_week} for this season (and, with week, only rows not yet past). asof (a date): only
    estimates added by then - a re-created past week (ros_history backfill) must not use a later estimate (Nov 23,
    2026: Week 1's re-created ROS had Drake Maye's Week 11 injury taking out 12 of his 15 games)."""
    d = _read(path)
    if d is None:
        return {}
    d = d[d["season"] == season]
    if week is not None:
        d = d[d["out_through_week"] >= week]
    if asof is not None and "added" in d:
        d = d[pd.to_datetime(d["added"]) <= pd.Timestamp(asof)]
    return dict(zip(d["player_id"], d["out_through_week"].astype(int)))


def rows(season, week, path=PATH):
    """The manual_out rows still in force for this week, as dicts."""
    d = _read(path)
    if d is None:
        return []
    fw = d["from_week"] if "from_week" in d else 0
    return d[(d["season"] == season) & (fw <= week) & (d["out_through_week"] >= week)].to_dict("records")

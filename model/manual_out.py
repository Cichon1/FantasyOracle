"""Manual return estimates (model/manual_out.csv), added Nov 16, 2026 at Jackson's request.

When news says a player will miss several weeks before any injury report shows it (or the report only covers one week),
list him in manual_out.csv with the first and last weeks he is expected to miss (from_week, out_through_week). Then:
  - weekly projections: automation/injuries_live.py marks him Out for projected weeks from_week..out_through_week
    (never the week he got hurt in, which may already be played);
  - rest of season (ros.py): his team's games through out_through_week count as missed - the future part of his ROS
    points is cut to the games left after his return, the expert blend is skipped for him (the experts may not have
    moved yet), and his game-by-game chance to play (play_curve, used by the trade analyzer) is 0 for those games.
Rows for other seasons, or whose out_through_week has passed, do nothing. Delete a row when he's back (or to undo).
"""
import os

import pandas as pd

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_out.csv")


def load(season, week=None, path=PATH):
    """{player_id: out_through_week} for this season (and, with week, only rows not yet past)."""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path, dtype={"player_id": str})
    d = d[d["season"] == season]
    if week is not None:
        d = d[d["out_through_week"] >= week]
    return dict(zip(d["player_id"], d["out_through_week"].astype(int)))


def rows(season, week, path=PATH):
    """The manual_out rows still in force for this week, as dicts."""
    if not os.path.exists(path):
        return []
    d = pd.read_csv(path, dtype={"player_id": str})
    fw = d["from_week"] if "from_week" in d else 0
    return d[(d["season"] == season) & (fw <= week) & (d["out_through_week"] >= week)].to_dict("records")

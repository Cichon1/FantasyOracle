"""Confidence scores for the weekly and rest-of-season rankings -> extra columns in the site's data files.

start_pct (weekly): chance the player finishes the week as a fantasy starter - top 12 QB, 24 RB, 36 WR, 12 TE in PPR.
  The projection and its calibrated 10th / 90th percentile range give a two-piece normal; the chance is the share of
  it above the score the Nth-best player at the position typically ends with (2023-2025 average: QB 18.3, RB 11.0,
  WR 10.6, TE 10.7 PPR). Check on the 2026 weeks already played (2,337 player-games): predicted 15% -> 16% finished as
  starters, 25% -> 23%, 35% -> 30%, 44% -> 38%, 55% -> 52%, 65% -> 61%, 75% -> 71%, 83% -> 90%; Brier score 0.123
  vs 0.170 for a flat rate. A recalibration step was tried and made weeks 5-6 worse, so none is applied.
ros_start_pct (rest of season): chance his rest-of-season points finish at or above the Nth-best rest-of-season
  projection at his position (same N), from the ROS projection and its calibrated range.
confidence / ros_confidence: High (65%+), Medium (35-65%), Low (under 35%).

  python automation/confidence.py docs/data
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

N = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
WEEK_T = {"QB": 18.26, "RB": 10.96, "WR": 10.60, "TE": 10.73}
Z90 = 1.2816


def chance(mid, lo, hi, t, floor=0.5):
    mid, lo, hi, t = (np.asarray(v, dtype=float) for v in (mid, lo, hi, t))
    s_lo = np.maximum((mid - lo) / Z90, floor)
    s_hi = np.maximum((hi - mid) / Z90, floor)
    p = np.where(t >= mid, 1 - norm.cdf((t - mid) / s_hi), 1 - norm.cdf((t - mid) / s_lo))
    return np.where(np.isnan(mid), np.nan, p)


def label(p):
    return np.select([p >= 0.65, p >= 0.35, p >= 0], ["High", "Medium", "Low"], default="")


def main(site):
    pp = os.path.join(site, "projections_v5.csv")
    p = pd.read_csv(pp)
    sp = chance(p["proj_ppr"], p["ppr_low"], p["ppr_high"], p["pos"].map(WEEK_T))
    sp = np.where(p["status"].isin(["Inactive", "Did not play"]) & (p["proj_ppr"].fillna(0) == 0), 0.0, sp)
    p["start_pct"] = np.round(sp * 100).astype("float")
    p["confidence"] = label(sp)
    p.to_csv(pp, index=False)
    rp = os.path.join(site, "ros_v5.csv")
    if os.path.exists(rp):
        r = pd.read_csv(rp)
        th = {pos: g.nlargest(N.get(pos, 12)).min() for pos, g in r.groupby("pos")["ros_ppr"]}
        t = r["pos"].map(th)
        rs = chance(r["ros_ppr"], r["ros_low"], r["ros_high"], t, floor=2.0)
        r["ros_start_pct"] = np.round(rs * 100).astype("float")
        r["ros_confidence"] = label(rs)
        r.to_csv(rp, index=False)
    print("confidence added")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/data")

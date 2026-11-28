"""How BDSM League and The League draft compared with FFC 12-team PPR ADP."""
import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, "" + __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))) + "")
from audit_draft_data import norm
R = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "history", "draft")

def ffc(y):
    d = json.load(open(f"{R}/ffc/ffc_ppr_12_{y}.json"))
    f = pd.DataFrame(d["players"]); f["key"] = f["name"].map(norm)
    f["pos"] = f["position"].replace({"PK": "K", "DEF": "DEF"})
    return f.drop_duplicates("key").set_index("key")

def load(lg):
    out = []
    for p in sorted(glob.glob(f"{R}/league_drafts/{lg}_*.csv")):
        d = pd.read_csv(p); y = int(d.season.iat[0])
        f = ffc(y); d["key"] = d["player"].map(norm)
        d["adp"] = d["key"].map(f["adp"]); d["adp_sd"] = d["key"].map(f["stdev"])
        d["overall"] = (d["round"] - 1) * 12 + d["pick"].where(d["pick"] <= 12, ((d["pick"] - 1) % 12) + 1)
        d["overall"] = d["pick"] if d["pick"].max() > 12 else d["overall"]
        d["pos"] = d["pos"].str.split(",").str[0]
        out.append(d)
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    for lg in ["bdsm", "league"]:
        D = load(lg)
        print(f"\n===== {lg}: {D.season.nunique()} drafts, {len(D)} picks; matched to FFC ADP: {D.adp.notna().mean():.0%}")
        nk = D[D.is_keeper == 0]
        m = nk[nk.adp.notna() & (nk.pos.isin(["QB", "RB", "WR", "TE"]))].copy()
        m["diff"] = m["overall"] - m["adp"]          # + = taken later than market, - = reached
        m["logr"] = np.log(m["overall"] / m["adp"])
        print("taken vs market ADP, by position (picks within top 150 ADP); + means LATER than market")
        mm = m[m.adp <= 150]
        print(mm.groupby("pos").agg(n=("diff", "size"), mean_picks=("diff", "mean"), median_picks=("diff", "median"),
              pct_later=("logr", lambda s: (np.exp(s.mean()) - 1) * 100)).round(1).to_string())
        print("by season:"); print(mm.groupby(["season", "pos"])["logr"].mean().apply(lambda s: round((np.exp(s)-1)*100)).unstack().to_string())
        # spread vs market spread
        e = mm[mm.adp <= 100]
        print(f"spread: league sd of (pick - ADP) {e['diff'].std():.1f} picks vs FFC's own sd {e['adp_sd'].mean():.1f} (ADP <= 100)")
        # positions by round
        tab = nk[nk["round"] <= 10].groupby(["round", "pos"]).size().unstack(fill_value=0) / D.season.nunique()
        print("picks per round by position (avg per draft):"); print(tab.reindex(columns=["QB", "RB", "WR", "TE", "K", "DEF"], fill_value=0).round(1).to_string())
        print("first QB / first TE taken (overall pick), by season:",
              nk.groupby("season").apply(lambda g: (int(g[g.pos == "QB"].overall.min()), int(g[g.pos == "TE"].overall.min()))).to_dict())
        print("unmatched examples:", D[D.adp.isna() & D.pos.isin(["QB","RB","WR","TE"])].head(8)[["season","overall","player","pos"]].values.tolist())

def ranked(lg):
    """k-th non-keeper pick vs the player's market rank among players not kept (FFC ADP)."""
    D = load(lg); out = []
    for y, g in D.groupby("season"):
        f = ffc(y); f = f[f.pos.isin(["QB", "RB", "WR", "TE"])].copy()
        kept = set(g.loc[g.is_keeper == 1, "key"])
        f = f[~f.index.isin(kept)].sort_values("adp"); f["mkt_rank"] = np.arange(1, len(f) + 1)
        nk = g[(g.is_keeper == 0)].sort_values("pick").copy()
        nk["live_pick"] = np.arange(1, len(nk) + 1)
        sk = nk[nk.pos.isin(["QB", "RB", "WR", "TE"])].copy()
        sk["mkt_rank"] = sk["key"].map(f["mkt_rank"]); sk["adp_sd"] = sk["key"].map(f["stdev"])
        out.append(sk)
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__" and "--ranked" in sys.argv:
    for lg in ["bdsm", "league"]:
        m = ranked(lg); m = m[m.mkt_rank.notna() & (m.mkt_rank <= 150)].copy()
        # live picks include K / DEF; compare skill-pick order with skill market rank
        m["skill_pick"] = m.groupby("season").live_pick.rank()
        m["diff"] = m.skill_pick - m.mkt_rank; m["logr"] = np.log(m.skill_pick / m.mkt_rank)
        print(f"\n== {lg} (keepers removed, skill picks only) + = later than market")
        print(m.groupby("pos").agg(n=("diff","size"), mean_picks=("diff","mean"), median=("diff","median"),
              pct=("logr", lambda s: (np.exp(s.mean())-1)*100)).round(1).to_string())
        print(m.groupby(["season","pos"]).logr.mean().apply(lambda s: round((np.exp(s)-1)*100)).unstack().to_string())
        for lo, hi in [(1, 36), (37, 84), (85, 150)]:
            e = m[(m.mkt_rank >= lo) & (m.mkt_rank <= hi)]
            print(f"  market rank {lo}-{hi}: league sd of (pick - rank) {e['diff'].std():.1f} vs FFC player sd {e['adp_sd'].mean():.1f}")

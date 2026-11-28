import sys, pandas as pd, numpy as np
for f in sys.argv[1:]:
    R = pd.read_csv(f); S = R.groupby(["season", "cfg"])[["pf", "po", "wins"]].mean().reset_index()
    b = S[S.cfg == "c_adp"].set_index("season"); out = []
    for c, g in S.groupby("cfg"):
        g = g.set_index("season"); d = g[["pf", "po", "wins"]] - b[["pf", "po", "wins"]]
        d[["pf", "po"]] *= 100
        late = d.loc[d.index >= 2021]
        out.append(dict(cfg=c, pf=d.pf.mean(), se_pf=d.pf.std() / np.sqrt(len(d)), playoffs=d.po.mean(),
                        se_po=d.po.std() / np.sqrt(len(d)), wins=d.wins.mean(), better=f"{int((d.pf > 0).sum())}/{len(d)}",
                        pf_2021_25=late.pf.mean()))
    a = R[R.cfg == "c_adp"]
    print(f"\n{f}  (plain ADP itself: {100*(a.pf.mean()-1):+.1f}% pts vs league avg, playoffs {100*a.po.mean():.0f}%)")
    print(pd.DataFrame(out).sort_values("pf", ascending=False).round(2).to_string(index=False))

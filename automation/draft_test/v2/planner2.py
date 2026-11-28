"""Planner v2: smooth value board (+ ADP/ECR blend) and opponents with each league's real draft habits.
Opponent model ("league"): each player's market ADP x exp(position bias) + noise x NOISE_MULT, biases and noise from
BDSM's 2024-26 / The League's 2025-26 Yahoo drafts (league_habits.py, keepers removed). "market" = run 1's model.
The test seat's planner uses either model for its own availability forecast. Season engine: fixed waivers."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "" + __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))) + ""); sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import tournament as T
import board_model as BM
from audit_draft_data import norm
from sensitivity import add_ros, run_season_fixed
import league_habits as LH

POS = T.POS; CAPI = [T.CAP[q] for q in POS]

def fit_habits(lg, shrink_n=20):
    m = LH.ranked(lg); m = m[m.mkt_rank.notna() & (m.mkt_rank <= 150)].copy()
    m["skill_pick"] = m.groupby("season").live_pick.rank(); m["logr"] = np.log(m.skill_pick / m.mkt_rank)
    bias = {}
    for p in POS:
        s = m.loc[m.pos == p, "logr"]; bias[p] = float(s.sum() / (len(s) + shrink_n))
    m["diff"] = m.skill_pick - m.mkt_rank
    mult = []
    for lo, hi in [(1, 36), (37, 84), (85, 150)]:
        e = m[(m.mkt_rank >= lo) & (m.mkt_rank <= hi)]; mult.append(float(e["diff"].std() / e["adp_sd"].mean()))
    return {"bias": bias, "mult": mult}

HAB = {lg: fit_habits(lg) for lg in ("bdsm", "league")}

def smooth_curve(x_tr, y_tr, x, h=0.22):
    lx, lt = np.log(x)[:, None], np.log(x_tr)[None, :]
    w = np.exp(-0.5 * ((lx - lt) / h) ** 2); y = (w * y_tr[None, :]).sum(1) / w.sum(1)
    o = np.argsort(x); ys = y[o]; ys = np.maximum.accumulate(ys[::-1])[::-1]   # non-increasing in ADP
    out = np.empty_like(y); out[o] = ys; return out

def augment(D):
    P, lg, y = D["P"], D["lg"], D["y"]
    prev = pd.concat([pd.read_parquet(f"{T.B.OUT}/pool_{s}.parquet") for s in T.B.SEASONS if s < y])
    prev = prev[prev.matched]
    ecr = BM.fp_ecr(y)
    P["ecr_now"] = P["player"].map(norm).map(ecr) if ecr is not None and "player" in P else np.nan
    for p in POS:
        t = prev[prev.pos == p]; m = (P.pos == p).values
        P.loc[m, "ev_s"] = smooth_curve(t.adp_ffc.values, t[f"pts_{lg}"].values, P.loc[m, "adp_ffc"].values)
        e = P.loc[m, "ecr_now"].values; a = P.loc[m, "adp_ffc"].values
        blend = np.where(np.isfinite(e), np.exp(0.5 * np.log(a) + 0.5 * np.log(np.clip(e, 1, None))), a)
        P.loc[m, "ev_b"] = smooth_curve(t.adp_ffc.values, t[f"pts_{lg}"].values, blend)
    # out-of-pool players keep run 1's low expectation
    P.loc[~P.in_pool, ["ev_s", "ev_b"]] = P.loc[~P.in_pool, ["exp_season"]].values.repeat(2, 1)
    D["has_ecr"] = bool(np.isfinite(P.loc[P.in_pool, "ecr_now"]).mean() > 0.5)
    return D

def opp_params(D, model):
    P = D["P"]; adp = P.adp_ffc.values; sd = P.adp_ffc_sd.values
    if model == "market": return adp, sd
    h = HAB[D["lg"]]
    eff = adp * np.exp(P.pos.map(h["bias"]).values)
    mult = np.where(adp <= 36, h["mult"][0], np.where(adp <= 84, h["mult"][1], h["mult"][2]))
    return eff, sd * mult

def run_draft2(D, seat, cfg, z, keepers, opp_model):
    P, lg = D["P"], D["lg"]; N = len(P)
    avail = np.ones(N, bool); rosters = [[] for _ in range(12)]; counts = [{p: 0 for p in POS} for _ in range(12)]
    for t, k in keepers.items():
        avail[k] = False; rosters[t].append(k); counts[t][P.pos.iat[k]] += 1
    eff, sd = opp_params(D, opp_model); sampled = eff + sd * z
    posv = P.pos.values; order = T.pick_order(); mine = [i for i, t in enumerate(order) if t == seat]
    for k, t in enumerate(order):
        rnd = k // 12 + 1
        if t == seat:
            nxt = [q for q in mine if q > k]
            i = seat_pick2(cfg, D, avail, counts[t], rosters[t], rnd, k + 1, mine)
        else:
            okp = np.array([T.need_ok(counts[t], p, rnd, T.ROUNDS - rnd + 1, lg) for p in POS])
            cand = avail & okp[D["pos"]]; i = int(np.where(cand)[0][np.argmin(sampled[cand])])
        avail[i] = False; rosters[t].append(i); counts[t][posv[i]] += 1
    return rosters, avail

def seat_pick2(cfg, D, avail, counts, roster, rnd, pick_no, mine_all):
    kind, value, belief, te_rule = cfg
    P, lg = D["P"], D["lg"]; posi = D["pos"]; adp = P.adp_ffc.values
    okp = [T.need_ok(counts, q, rnd, T.ROUNDS - rnd + 1, lg) for q in POS]
    if kind == "c_adp":
        cand = np.where(avail & np.array([okp[q] for q in posi]))[0]; return int(cand[np.argmin(adp[cand])])
    ev = P[value].values; eff, sd = opp_params(D, belief); st = T.START[lg]; M = 200
    mine = [k + 1 for k in mine_all if k + 1 >= pick_no]
    ai = np.where(avail)[0]; rng = np.random.default_rng(pick_no * 101 + len(ai))
    ranks = np.argsort(np.argsort(eff[ai][None, :] + sd[ai][None, :] * rng.standard_normal((M, len(ai))), 1), 1)
    opp_before = np.array([(m - pick_no) - j for j, m in enumerate(mine)])
    best = np.zeros((4, len(mine))); second = np.zeros((4, len(mine)))
    for p in range(4):
        cols = np.where(posi[ai] == p)[0]
        if not len(cols): continue
        e = ev[ai][cols]; R = ranks[:, cols]
        for j, c in enumerate(opp_before):
            v = np.where(R >= c, e[None, :], -1e9); top2 = -np.sort(-v, 1)[:, :2]
            if top2.shape[1] < 2: top2 = np.hstack([top2, np.full((M, 1), -1e9)])
            best[p, j] = np.mean(np.maximum(top2[:, 0], 0)); second[p, j] = np.mean(np.maximum(top2[:, 1], 0))
    # survival of each available player to my next pick (for timing within a position)
    surv = (ranks >= (opp_before[1] if len(opp_before) > 1 else 10**6)).mean(0) if len(mine) > 1 else np.zeros(len(ai))
    surv_full = np.zeros(len(P)); surv_full[ai] = surv
    # TE rule: own a top-4 TE (by value) by the end of round 2; take him at the last pick he's likely to survive to
    if te_rule and counts["TE"] == 0 and rnd <= 2 and okp[3]:
        tes = np.where(avail & (posi == 3))[0]
        top4 = set(np.argsort(-ev[posi == 3])[:4]); allte = np.where(posi == 3)[0]
        top4 = {int(allte[i]) for i in top4}
        live = [i for i in tes if i in top4]
        if live:
            b = max(live, key=lambda i: ev[i])
            if rnd == 2 or surv_full[b] < 0.5: return int(b)
    nstart = [st["QB"], st["RB"], st["WR"], st["TE"]]; base = [counts[q] for q in POS]
    flex0 = any(base[q] > nstart[q] for q in (1, 2, 3)); n = len(mine); bench_w = 0.25
    from functools import lru_cache
    @lru_cache(None)
    def V(j, c0, c1, c2, c3, flex, lastp):
        if j == n: return 0.0
        c = [c0, c1, c2, c3]; out = -1e18
        for p in range(4):
            if c[p] >= CAPI[p] or (p in (0, 3) and c[p] >= 1 and rnd + j < 10): continue
            val = second[p, j] if lastp == p and mine[j] - mine[j - 1] <= 2 else best[p, j]
            if c[p] < nstart[p]: w, nf = 1.0, flex
            elif p != 0 and not flex: w, nf = 1.0, True
            else: w, nf = bench_w, flex
            cc = list(c); cc[p] += 1; out = max(out, w * val + V(j + 1, *cc, nf, p))
        return out if out > -1e17 else V(j + 1, c0, c1, c2, c3, flex, lastp)
    bp, bv = None, -1e18
    for p in range(4):
        if not okp[p] or base[p] >= CAPI[p] or (p in (0, 3) and base[p] >= 1 and rnd < 10): continue
        if base[p] < nstart[p]: w, nf = 1.0, flex0
        elif p != 0 and not flex0: w, nf = 1.0, True
        else: w, nf = bench_w, flex0
        c = list(base); c[p] += 1; v = w * best[p, 0] + V(1, *c, nf, p)
        if v > bv: bv, bp = v, p
    if bp is None:
        cand = np.where(avail & np.array([okp[q] for q in posi]))[0]; return int(cand[np.argmin(adp[cand])])
    m = np.where(avail & (posi == bp))[0]
    top = m[np.argmax(ev[m])]
    # timing: if the top player at this position will very likely still be there next pick, take the best one who won't
    if surv_full[top] >= 0.8:
        alt = [i for i in m if surv_full[i] < 0.5 and ev[i] >= 0.92 * ev[top]]
        if alt: return int(max(alt, key=lambda i: ev[i]))
    return int(top)

CFGS = {"c_adp": ("c_adp", None, None, False),
        "plan_market_view": ("plan", "ev_s", "market", False),     # planner that assumes generic opponents
        "plan_knows_league": ("plan", "ev_s", "league", False),    # planner that knows the league's habits
        "plan_league_blend": ("plan", "ev_b", "league", False),    # + ADP/ECR blended values (2021+)
        "plan_league_te2": ("plan", "ev_s", "league", True)}       # + top-4 TE by end of round 2

def run(lg, seasons, opp_model="league", sims=3, tag="v2"):
    rows = []; t0 = time.time()
    for y in seasons:
        D = augment(add_ros(T.season_data(y, lg)))
        for slot in range(12):
            for s in range(sims):
                rng = np.random.default_rng([y, slot, s, 7 if lg == "bdsm" else 11])
                z = rng.standard_normal(len(D["P"])); keepers = {}
                if lg == "league":
                    tm = rng.permutation(12); keepers = {int(t): int(r - 1) for t, r in zip(tm, T.KEEPER_RANKS)}
                sched = T.schedule(rng)
                for name, cfg in CFGS.items():
                    r, av = run_draft2(D, slot, cfg, z, keepers, opp_model)
                    w, pf, po = run_season_fixed(D, r, av, sched)
                    rows.append(dict(season=y, slot=slot, sim=s, cfg=name, pf=pf[slot] / pf.mean(), po=float(slot in po),
                                     wins=w[slot], has_ecr=D["has_ecr"],
                                     pos="".join(D["P"].pos.iat[i][0] for i in r[slot])))
        print(lg, opp_model, y, f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{__import__("os").path.dirname(__import__("os").path.abspath(__file__))}/results/{tag}_{lg}_{opp_model}.csv", index=False)

if __name__ == "__main__":
    lg, opp = sys.argv[1], sys.argv[2]
    print(HAB[lg])
    run(lg, [int(x) for x in sys.argv[3:]], opp)

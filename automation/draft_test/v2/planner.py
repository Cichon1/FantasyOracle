"""Rolling-horizon draft planner (preseason info only).
Each pick: (1) Monte Carlo the rest of the draft with opponents on ADP + noise (my own draws) -> expected best
available expected-season-points at each position at each of my remaining picks; (2) dynamic program over my remaining
picks choosing a position for each, maximizing expected roster value (starters x1, first extra RB/WR/TE = FLEX x1,
other bench x BENCH); (3) take the best available player at the first planned position. Re-plan every pick."""
import sys, time, numpy as np, pandas as pd
from functools import lru_cache
sys.path.insert(0, "" + __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))) + ""); sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import tournament as T
from sensitivity import add_ros, run_season_fixed

def make_plan(bench_w=0.15, M=200, value="exp_season"):
    def pick(cfg, D, avail, counts, roster, rnd, my_next, pick_no, rng_bad, z):
        P, lg = D["P"], D["lg"]; posi = D["pos"]; ev = P[value].values; adp = P["adp_ffc"].values
        sd = P["adp_ffc_sd"].values; st = T.START[lg]
        seat = T.pick_order()[pick_no - 1]
        mine = [k + 1 for k, t in enumerate(T.pick_order()) if t == seat and k + 1 >= pick_no]   # incl. now
        ai = np.where(avail)[0]
        rng = np.random.default_rng(pick_no * 101 + len(ai))
        ranks = np.argsort(np.argsort(adp[ai][None, :] + sd[ai][None, :] * rng.standard_normal((M, len(ai))), 1), 1)
        # opponent picks before each of my picks (from now): my j-th pick (j=0 now) has (mine[j]-pick_no) - j picks ahead
        opp_before = np.array([(m - pick_no) - j for j, m in enumerate(mine)])
        # E[best available ev at pos p at my j-th pick], and 2nd best (for back-to-back same-position picks)
        best = np.zeros((4, len(mine))); second = np.zeros((4, len(mine)))
        for p in range(4):
            cols = np.where(posi[ai] == p)[0]
            if not len(cols): continue
            e = ev[ai][cols]; R = ranks[:, cols]
            for j, c in enumerate(opp_before):
                v = np.where(R >= c, e[None, :], -1e9)
                top2 = -np.sort(-v, 1)[:, :2] if v.shape[1] > 1 else np.hstack([v, np.full((M,1),-1e9)])
                best[p, j] = np.mean(np.maximum(top2[:, 0], 0)); second[p, j] = np.mean(np.maximum(top2[:, 1], 0))
        # now: exact values
        okp = [T.need_ok(counts, q, rnd, T.ROUNDS - rnd + 1, lg) for q in T.POS]
        nstart = [st["QB"], st["RB"], st["WR"], st["TE"]]
        base = [counts[q] for q in T.POS]
        # flex already used if roster already has an extra RB/WR/TE beyond starters
        flex0 = any(base[q] > nstart[q] for q in (1, 2, 3))
        n = len(mine)

        @lru_cache(None)
        def V(j, c0, c1, c2, c3, flex, lastp):
            if j == n: return 0.0
            c = [c0, c1, c2, c3]; out = -1e18
            for p in range(4):
                if c[p] >= CAPI[p]: continue
                if p in (0, 3) and c[p] >= 1 and rnd + j < 10: continue
                if j == 0 and not okp[p]: continue
                val = second[p, j] if lastp == p and j > 0 and mine[j] - mine[j-1] <= 2 else best[p, j]
                if c[p] < nstart[p]: w, nf = 1.0, flex
                elif p != 0 and not flex: w, nf = 1.0, True
                else: w, nf = bench_w, flex
                c2_ = list(c); c2_[p] += 1
                out = max(out, w * val + V(j + 1, *c2_, nf, p))
            # if nothing legal (caps), skip pick
            return out if out > -1e17 else V(j + 1, c0, c1, c2, c3, flex, lastp)
        bestp, bv = None, -1e18
        for p in range(4):
            if not okp[p] or base[p] >= CAPI[p]: continue
            if p in (0, 3) and base[p] >= 1 and rnd < 10: continue
            if base[p] < nstart[p]: w, nf = 1.0, flex0
            elif p != 0 and not flex0: w, nf = 1.0, True
            else: w, nf = bench_w, flex0
            c = list(base); c[p] += 1
            v = w * best[p, 0] + V(1, *c, nf, p)
            if v > bv: bv, bestp = v, p
        if bestp is None:
            cand = np.where(avail & np.array([okp[q] for q in posi]))[0]
            return int(cand[np.argmin(adp[cand])])
        m = np.where(avail & (posi == bestp))[0]
        return int(m[np.argmax(ev[m])])
    return pick

CAPI = [T.CAP[q] for q in T.POS]
orig = T.seat_pick; PL = {}
T.seat_pick = lambda cfg, *a: PL[cfg[1]](cfg, *a) if cfg[0] == "plan" else orig(cfg, *a)

def run(lg, seasons, cfgs, sims=2, tag="plan"):
    rows = []; t0 = time.time()
    for y in seasons:
        D = add_ros(T.season_data(y, lg))
        for slot in range(12):
            for s in range(sims):
                rng = np.random.default_rng([y, slot, s, 7 if lg == "bdsm" else 11])
                z = rng.standard_normal(len(D["P"])); zb = rng.standard_normal(len(D["P"]))
                keepers = {}
                if lg == "league":
                    tm = rng.permutation(12); keepers = {int(t): int(r - 1) for t, r in zip(tm, T.KEEPER_RANKS)}
                sched = T.schedule(rng)
                for cfg in cfgs:
                    r, av = T.run_draft(D, slot, cfg, z, zb, keepers)
                    w, pf, po = run_season_fixed(D, r, av, sched)
                    rows.append(dict(season=y, slot=slot, sim=s, cfg=f"{cfg[0]}|{cfg[1]}", pf=pf[slot] / pf.mean(),
                                     po=float(slot in po), wins=w[slot],
                                     pos="".join(D["P"]["pos"].iat[i][0] for i in r[slot])))
        print(lg, y, f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{__import__("os").path.dirname(__import__("os").path.abspath(__file__))}/results/{tag}_{lg}.csv", index=False)

if __name__ == "__main__":
    lg = sys.argv[1]; seasons = [int(x) for x in sys.argv[2:]]
    for bw in (0.1, 0.25): PL[f"b{bw}"] = make_plan(bw)
    run(lg, seasons, [("c_adp", None), ("tiers", "adp")] + [("plan", k) for k in PL])

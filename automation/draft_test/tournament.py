"""Phase 4 (PREREG.md sections 4-6 + the Nov 24 engine addendum): draft strategy tournament.

Every (season, league, draft slot, simulated league) is one draft + one full season on that year's REAL weekly points:
  draft    12 teams, snake, 13 skill-position rounds (K / DEF: every team takes them in the last rounds - identical for
           all seats, left out). The League: plus one keeper per team (round-16 pick), ADP ranks of kept players
           1-4, 7, 9, 10, 12, 23, 26, 28, 38 assigned to random teams.
  opponents 11 teams draft by ADP + noise from each player's real ADP spread, with roster rules (2nd QB / 2nd TE not
           before round 10, caps QB 2, TE 2, RB 6, WR 7, starters filled by the end).
  test seat 1 team: one strategy x board (or a control).
  season   Weeks 1-14 head-to-head (random round-robin), Weeks 15-17 playoffs (BDSM 8 teams, The League 6 with two
           byes); seeds by wins then points. Every manager sets lineups each week from the same honest projection
           (season-to-date points per game shrunk toward the preseason expectation; inactive players known, as Yahoo
           shows them) and makes the same waiver move: in reverse standings order, one add per week of the best free
           agent who out-projects the team's worst player at that position.
  common random numbers: opponents' ADP noise, keepers, schedule are the same for every seat in a given league.

Outputs: model/lab/draft/tourney/<league>_<season>.parquet, one row per (seat config, slot, sim): title, playoffs,
wins, points for, points for vs league mean. Saved per (league, season) as each finishes; a rerun skips saved files.

  python automation/draft_test/tournament.py [seasons...]        (default 2013-2025)  SIMS=n (per slot, default 8)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board_data as B  # noqa: E402

OUT = os.path.join(B.OUT, "tourney")
SIMS = int(os.environ.get("SIMS", "8"))
TEAMS, ROUNDS = 12, 13
POS = ["QB", "RB", "WR", "TE"]
START = {"bdsm": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}, "league": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}}
CAP = {"QB": 2, "RB": 6, "WR": 7, "TE": 2}
REPL = {"bdsm": {"QB": 12, "RB": 29, "WR": 30, "TE": 13}, "league": {"QB": 12, "RB": 29, "WR": 41, "TE": 13}}
KEEPER_RANKS = [1, 2, 3, 4, 7, 9, 10, 12, 23, 26, 28, 38]
PLAYOFF = {"bdsm": 8, "league": 6}
SHRINK_K = 4.0

STRATEGIES = ["bpa", "zero_rb", "hero_rb", "robust_rb", "wr_heavy", "early_qb", "late_qb", "elite_te", "late_te",
              "tiers", "upside_late", "handcuff", "value_vs_adp"]
BOARDS = ["adp", "ours"]                      # + "ecr" from 2021 (FP's preseason ranks archive)
CONTROLS = ["c_adp", "c_ecr", "c_noisy"]


# ------------------------------------------------------------------ season data
def season_data(y, lg):
    pool = pd.read_parquet(os.path.join(B.OUT, f"pool_{y}.parquet"))
    pool = pool[pool["matched"]].drop_duplicates("gsis_id")
    wk = pd.read_parquet(os.path.join(B.OUT, "weekly_points.parquet"))
    prev_pools = pd.concat([pd.read_parquet(os.path.join(B.OUT, f"pool_{s}.parquet")) for s in B.SEASONS if s < y])
    cur = wk[wk["season"] == y]
    last = wk[wk["season"] == y - 1] if (wk["season"] == y - 1).any() else wk[wk["season"] == y - 2]
    # everyone who played this season or last joins as a waiver / late-round option, ranked after ADP players
    extra_ids = (set(cur["gsis_id"]) | set(last["gsis_id"])) - set(pool["gsis_id"])
    info = pd.concat([cur, last]).drop_duplicates("gsis_id", keep="first").set_index("gsis_id")
    lastpts = last.groupby("gsis_id")[f"pts_{lg}"].sum()
    ex = pd.DataFrame({"gsis_id": list(extra_ids)})
    ex["pos"] = ex["gsis_id"].map(info["pos"])
    ex = ex[ex["pos"].isin(POS)]
    ex["team"] = ex["gsis_id"].map(info["team"])
    ex["prev"] = ex["gsis_id"].map(lastpts).fillna(0)
    ex = ex.sort_values("prev", ascending=False)
    ex["adp_ffc"] = pool["adp_ffc"].max() + 1 + np.arange(len(ex)) * 0.5
    ex["adp_ffc_sd"] = 15.0
    P = pd.concat([pool[["gsis_id", "player", "pos", "team", "adp_ffc", "adp_ffc_sd"]].assign(in_pool=True),
                   ex[["gsis_id", "pos", "team", "adp_ffc", "adp_ffc_sd", "prev"]].assign(in_pool=False)],
                  ignore_index=True).sort_values("adp_ffc").reset_index(drop=True)
    P["adp_ffc_sd"] = P["adp_ffc_sd"].fillna(P["adp_ffc"] * 0.15).clip(lower=0.5)
    # weekly actuals and played mask (weeks 1-17)
    piv = cur.pivot_table(index="week", columns="gsis_id", values=f"pts_{lg}", aggfunc="sum")
    piv = piv.reindex(index=range(1, 18), columns=P["gsis_id"])
    played = piv.notna().values
    pts = np.nan_to_num(piv.values)
    # preseason expectation from ADP (isotonic by position, seasons < y): season points and points per game played
    from sklearn.isotonic import IsotonicRegression
    tr = prev_pools[prev_pools["matched"]]
    P["exp_season"] = 0.0
    P["exp_ppg"] = 0.0
    P["up80"] = 0.0
    repl = {}
    for p in POS:
        t = tr[tr["pos"] == p]
        m = (P["pos"] == p).values
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(t["adp_ffc"], t[f"pts_{lg}"])
        ppg = (t[f"pts_{lg}"] / t["games"].clip(lower=1)).where(t["games"] > 0)
        iso_g = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(t["adp_ffc"][ppg.notna()], ppg.dropna())
        P.loc[m, "exp_season"] = iso.predict(P.loc[m, "adp_ffc"])
        P.loc[m, "exp_ppg"] = iso_g.predict(P.loc[m, "adp_ffc"])
        # upside: 80th percentile of actual points among prior players with a similar positional ADP rank
        t = t.assign(r=t.groupby("season")["adp_ffc"].rank())
        rr = P.loc[m, "adp_ffc"].rank().values
        P.loc[m, "up80"] = [np.nanpercentile(t.loc[(t["r"] - r).abs() <= 3, f"pts_{lg}"], 80)
                            if ((t["r"] - r).abs() <= 3).sum() >= 5 else 0.0 for r in rr]
        repl[p] = float(np.mean([g.nlargest(REPL[lg][p], f"pts_{lg}")[f"pts_{lg}"].iloc[-1]
                                 for _, g in t.groupby("season") if len(g) >= REPL[lg][p]]))
    # players outside the ADP pool: expectation from last season, discounted (unknowns: low)
    np_ = ~P["in_pool"].values
    last_g = last.groupby("gsis_id")["week"].nunique()
    prev_ppg = (P["gsis_id"].map(lastpts) / P["gsis_id"].map(last_g)).fillna(0).values
    P.loc[np_, "exp_ppg"] = np.minimum(prev_ppg[np_] * 0.7, P.loc[P["in_pool"], "exp_ppg"].min())
    P.loc[np_, "exp_season"] = P.loc[np_, "exp_ppg"] * 8
    # boards: expected season points by each source
    bp = pd.read_parquet(os.path.join(B.OUT, "board_preds.parquet"))
    bp = bp[bp["season"] == y].set_index("gsis_id")
    P["b_adp"] = P["exp_season"]
    P["b_ours"] = P["gsis_id"].map(bp[f"pred_{lg}"]).fillna(P["exp_season"])
    ecr = P["gsis_id"].map(bp["fp_ecr"]) if "fp_ecr" in bp else pd.Series(np.nan, index=P.index)
    P["ecr"] = ecr
    if ecr.notna().sum() > 100:                 # ECR rank -> points with the same ADP curves (rank scale ~ ADP scale)
        P["b_ecr"] = 0.0
        for p in POS:
            t = tr[tr["pos"] == p]
            m = (P["pos"] == p).values
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(t["adp_ffc"], t[f"pts_{lg}"])
            P.loc[m, "b_ecr"] = iso.predict(P.loc[m, "ecr"].fillna(P["ecr"].max() + 50))
    # honest weekly projection: season-to-date ppg shrunk toward the preseason per-game expectation
    cum_pts = np.vstack([np.zeros(len(P)), np.cumsum(pts * played, axis=0)[:-1]])
    cum_g = np.vstack([np.zeros(len(P)), np.cumsum(played, axis=0)[:-1]])
    proj = (cum_pts + SHRINK_K * P["exp_ppg"].values) / (cum_g + SHRINK_K)
    proj = np.where(played, proj, -1.0)        # inactive: known before lock, never started
    # handcuff map: each RB's backup on his NFL team (next RB by ADP)
    back = {}
    for t, g in P[P["pos"] == "RB"].groupby("team"):
        idx = g.sort_values("adp_ffc").index.tolist()
        for a, b in zip(idx, idx[1:]):
            back.setdefault(a, b)
    return {"P": P, "pts": pts, "proj": proj, "repl": repl, "back": back, "lg": lg, "y": y,
            "pos": P["pos"].map({p: i for i, p in enumerate(POS)}).values}


# ------------------------------------------------------------------ draft
def pick_order(slot_team=None):
    order = []
    for r in range(ROUNDS):
        order += list(range(TEAMS)) if r % 2 == 0 else list(range(TEAMS))[::-1]
    return order


def need_ok(counts, p, rnd, picks_left, lg):
    """roster rules shared by every manager. counts: dict pos -> n."""
    if counts[p] >= CAP[p]:
        return False
    if p in ("QB", "TE") and counts[p] >= 1 and rnd < 10:
        return False
    st = START[lg]
    short = sum(max(0, st[q] - counts[q]) for q in POS)
    flex_short = max(0, st["FLEX"] - sum(max(0, counts[q] - st[q]) for q in ("RB", "WR", "TE")))
    need = short + flex_short
    if picks_left <= need:                      # must fill starters
        if counts[p] < st[p]:
            return True
        return p in ("RB", "WR", "TE") and short == 0 and flex_short > 0
    return True


def seat_pick(cfg, D, avail, counts, roster, rnd, my_next_pick, pick_no, rng_bad, z):
    P, lg = D["P"], D["lg"]
    kind, board = cfg
    adp = P["adp_ffc"].values
    picks_left = ROUNDS - rnd + 1
    ok_pos = {p: need_ok(counts, p, rnd, picks_left, lg) for p in POS}
    cand = avail & np.array([ok_pos[p] for p in P["pos"]])
    if kind == "c_adp":
        return int(np.where(cand)[0][np.argmin(adp[cand])])
    if kind == "c_ecr":
        e = P["ecr"].fillna(9999).values
        return int(np.where(cand)[0][np.argmin(np.where(e < 9999, e, 9999 + adp)[cand])])
    if kind == "c_noisy":
        s = adp + 2.5 * P["adp_ffc_sd"].values * rng_bad
        return int(np.where(cand)[0][np.argmin(s[cand])])
    val = P[f"b_{board}"].values - P["pos"].map(D["repl"]).values
    posv = P["pos"].values
    n = {p: counts[p] for p in POS}
    allow = {p: True for p in POS}
    if kind == "zero_rb" and rnd <= 5:
        allow["RB"] = False
    if kind == "hero_rb":
        if rnd <= 2 and n["RB"] >= 1:
            allow["RB"] = False
        if 3 <= rnd <= 5:
            allow["RB"] = False
    if kind == "robust_rb" and rnd <= 4 and n["RB"] < 3:
        allow = {p: p == "RB" for p in POS} if (4 - rnd + 1) <= (3 - n["RB"]) else allow
    if kind == "wr_heavy" and rnd <= 4 and n["WR"] < 3:
        allow = {p: p == "WR" for p in POS} if (4 - rnd + 1) <= (3 - n["WR"]) else allow
    if kind == "early_qb":
        if rnd <= 3 and n["QB"] == 0 and rnd == 3:
            allow = {p: p == "QB" for p in POS}
    if kind == "late_qb" and rnd < 8:
        allow["QB"] = False
    if kind == "elite_te" and rnd <= 3 and n["TE"] == 0 and rnd == 3:
        allow = {p: p == "TE" for p in POS}
    if kind == "late_te" and rnd < 10:
        allow["TE"] = False
    c2 = cand & np.array([allow[p] for p in posv])
    if not c2.any():
        c2 = cand
    if kind == "upside_late" and rnd >= 8:
        val = P["up80"].values - P["pos"].map(D["repl"]).values
    if kind == "handcuff" and rnd >= 10:
        rbs = [i for i in roster if posv[i] == "RB"]
        if rbs:
            b = D["back"].get(min(rbs, key=lambda i: adp[i]))
            if b is not None and c2[b]:
                return int(b)
    if kind == "tiers":
        # take the position whose value drops most before my next pick (by ADP, who will still be there)
        best, gain = None, -1e9
        for p in POS:
            m = c2 & (posv == p)
            if not m.any():
                continue
            top = val[m].max()
            later = m & (adp > my_next_pick)
            drop = top - (val[later].max() if later.any() else val[m].min())
            if drop > gain:
                best, gain = p, drop
        m = c2 & (posv == best)
        return int(np.where(m)[0][np.argmax(val[m])])
    if kind == "value_vs_adp":
        # best board value among players the market will take before my next pick; else best value overall
        soon = c2 & (adp <= my_next_pick + 2)
        if soon.any():
            i_soon = np.where(soon)[0][np.argmax(val[soon])]
            i_all = np.where(c2)[0][np.argmax(val[c2])]
            return int(i_soon if val[i_soon] >= 0.85 * val[i_all] else i_all)
    return int(np.where(c2)[0][np.argmax(val[c2])])


def run_draft(D, seat, seat_cfg, z, z_bad, keepers):
    P, lg = D["P"], D["lg"]
    N = len(P)
    avail = np.ones(N, bool)
    rosters = [[] for _ in range(TEAMS)]
    counts = [{p: 0 for p in POS} for _ in range(TEAMS)]
    for t, k in keepers.items():
        avail[k] = False
        rosters[t].append(k)
        counts[t][P["pos"].iat[k]] += 1
    sampled = P["adp_ffc"].values + P["adp_ffc_sd"].values * z
    posv = P["pos"].values
    order = pick_order()
    seat_picks = [i for i, t in enumerate(order) if t == seat]
    for k, t in enumerate(order):
        rnd = k // TEAMS + 1
        if t == seat:
            nxt = [q for q in seat_picks if q > k]
            i = seat_pick(seat_cfg, D, avail, counts[t], rosters[t], rnd, (nxt[0] + 1) if nxt else 999, k + 1, z_bad, z)
        else:
            picks_left = ROUNDS - rnd + 1
            okp = np.array([need_ok(counts[t], p, rnd, picks_left, lg) for p in POS])
            cand = avail & okp[D["pos"]]
            i = int(np.where(cand)[0][np.argmin(sampled[cand])])
        avail[i] = False
        rosters[t].append(i)
        counts[t][posv[i]] += 1
    return rosters, avail


# ------------------------------------------------------------------ season
def lineup_score(roster, w, D):
    proj, pts, posi = D["proj"][w], D["pts"][w], D["pos"]
    st = START[D["lg"]]
    r = np.array(roster)
    order = r[np.argsort(-proj[r])]
    used, total = set(), 0.0
    for p, n in [("QB", st["QB"]), ("RB", st["RB"]), ("WR", st["WR"]), ("TE", st["TE"])]:
        c = [i for i in order if posi[i] == POS.index(p) and proj[i] >= 0][:n]
        used.update(c)
        total += pts[c].sum()
    fl = [i for i in order if posi[i] in (1, 2, 3) and i not in used and proj[i] >= 0][:st["FLEX"]]
    return total + pts[fl].sum()


def schedule(rng):
    teams = list(rng.permutation(TEAMS))
    rounds = []
    for r in range(TEAMS - 1):
        pairs = [(teams[i], teams[TEAMS - 1 - i]) for i in range(TEAMS // 2)]
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return [rounds[w % (TEAMS - 1)] for w in range(14)]


def run_season(D, rosters, avail, sched):
    rosters = [list(r) for r in rosters]
    avail = avail.copy()
    posi, proj = D["pos"], D["proj"]
    wins, pf = np.zeros(TEAMS), np.zeros(TEAMS)
    for w in range(17):
        if w >= 1:                                # waivers before week w+1, reverse standings
            for t in np.lexsort((pf, wins)):
                fa = np.where(avail)[0]
                if not len(fa):
                    break
                best_gain, move = 1.0, None
                for p in range(4):
                    f = fa[posi[fa] == p]
                    mine = [i for i in rosters[t] if posi[i] == p]
                    if not len(f) or not mine:
                        continue
                    cand = f[np.argmax(proj[w][f])]
                    worst = min(mine, key=lambda i: proj[w][i])
                    g = proj[w][cand] - max(proj[w][worst], 0)
                    if g > best_gain:
                        best_gain, move = g, (cand, worst)
                if move:
                    a, d = move
                    rosters[t].remove(d)
                    rosters[t].append(a)
                    avail[a], avail[d] = False, True
        sc = np.array([lineup_score(rosters[t], w, D) for t in range(TEAMS)])
        if w < 14:
            for a, b in sched[w]:
                wins[a] += sc[a] > sc[b]
                wins[b] += sc[b] > sc[a]
                if sc[a] == sc[b]:
                    wins[a] += 0.5
                    wins[b] += 0.5
            pf += sc
        elif w == 14:
            seeds = list(np.lexsort((pf, wins))[::-1])
            n = PLAYOFF[D["lg"]]
            if n == 8:
                alive = [seeds[i] if sc[seeds[i]] >= sc[seeds[7 - i]] else seeds[7 - i] for i in range(4)]
                byes = []
            else:
                byes = seeds[:2]
                alive = [seeds[2] if sc[seeds[2]] >= sc[seeds[5]] else seeds[5],
                         seeds[3] if sc[seeds[3]] >= sc[seeds[4]] else seeds[4]]
            playoff = set(seeds[:n])
        elif w == 15:
            if byes:
                lo = sorted(alive, key=lambda t: seeds.index(t))
                pairs = [(byes[0], lo[1]), (byes[1], lo[0])]
            else:
                pairs = [(alive[0], alive[3]), (alive[1], alive[2])]
            alive = [a if sc[a] >= sc[b] else b for a, b in pairs]
        elif w == 16:
            champ = alive[0] if sc[alive[0]] >= sc[alive[1]] else alive[1]
    return wins, pf, playoff, champ


# ------------------------------------------------------------------ tournament
def configs(D):
    cfgs = [(c, None) for c in CONTROLS if c != "c_ecr" or "b_ecr" in D["P"]]
    boards = BOARDS + (["ecr"] if "b_ecr" in D["P"] else [])
    cfgs += [(s, b) for s in STRATEGIES for b in boards if not (s == "value_vs_adp" and b == "adp")]
    return cfgs


def run_season_league(y, lg):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{lg}_{y}.parquet")
    if os.path.exists(path):
        return
    D = season_data(y, lg)
    P = D["P"]
    rows = []
    t0 = time.time()
    for slot in range(TEAMS):
        for s in range(SIMS):
            rng = np.random.default_rng([y, slot, s, 7 if lg == "bdsm" else 11])
            z = rng.standard_normal(len(P))
            z_bad = rng.standard_normal(len(P))
            keepers = {}
            if lg == "league":
                teams = rng.permutation(TEAMS)
                keepers = {int(t): int(r - 1) for t, r in zip(teams, KEEPER_RANKS)}
            sched = schedule(rng)
            for cfg in configs(D):
                rosters, avail = run_draft(D, slot, cfg, z, z_bad, keepers)
                wins, pf, playoff, champ = run_season(D, rosters, avail, sched)
                rows.append({"season": y, "league": lg, "slot": slot + 1, "sim": s, "strategy": cfg[0],
                             "board": cfg[1] or "-", "title": float(champ == slot), "playoffs": float(slot in playoff),
                             "wins": wins[slot], "pf": pf[slot], "pf_rel": pf[slot] / pf.mean(),
                             "draft_pos": ",".join(P["pos"].iat[i][0] for i in rosters[slot])})
        print(f"  {lg} {y} slot {slot + 1} {time.time() - t0:.0f}s", flush=True)
    tmp = path + ".tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)                       # autosave per (league, season)


def calibration(y=2019, lg="bdsm", n=40):
    """opponent model check: simulated draft spots vs real ADP spread, and positions taken by round."""
    D = season_data(y, lg)
    P = D["P"]
    spots = {}
    pos_round = np.zeros((ROUNDS, 4))
    for s in range(n):
        rng = np.random.default_rng([y, 99, s])
        z = rng.standard_normal(len(P))
        rosters, _ = run_draft(D, 0, ("c_adp", None), z, z, {})
        order = pick_order()
        seen = [0] * TEAMS
        for k, t in enumerate(order):
            i = rosters[t][seen[t]]
            seen[t] += 1
            spots.setdefault(i, []).append(k + 1)
            pos_round[k // TEAMS, D["pos"][i]] += 1
    top = P.head(150)
    sim_sd = np.array([np.std(spots.get(i, [np.nan])) for i in top.index])
    by_r = []
    for r in range(0, 150, 12):
        m = slice(r, r + 12)
        by_r.append((r // 12 + 1, round(float(np.nanmean(sim_sd[m])), 1), round(float(top["adp_ffc_sd"].iloc[m].mean()), 1)))
    print("round: simulated sd vs real ADP sd", by_r)
    print("positions by round (avg per round, QB RB WR TE):", (pos_round / n).round(1).tolist())


if __name__ == "__main__":
    if sys.argv[1:2] == ["calib"]:
        calibration()
        sys.exit()
    seasons = [int(a) for a in sys.argv[1:]] or list(range(2013, 2026))
    for lg in os.environ.get("LEAGUES", "bdsm league").split():
        for y in seasons:
            run_season_league(y, lg)

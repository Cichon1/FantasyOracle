"""Nov 19, 2026: evolutionary tournament for fantasy strategies (draft, lineups, waivers, K / DEF streaming, trades).

Real seasons and real points (2023-2025 full, 2026 through Week 10), BDSM League format: 12 teams, 15-man rosters,
start QB, 2 RB, 2 WR, TE, W/R/T, K, DEF; 15-round snake draft; 14-week regular season, 8-team playoffs weeks 15-17;
trades through Week 12. Skill-player projections are the walk-forward ones from the earlier sims (lab11 / lab12: model,
FantasyPros, Sleeper, ESPN); K / DEF from lab17_kdef.py (Yahoo default scoring; Vegas-based projections, plus Sleeper
/ ESPN where pulled).

Every team runs a strategy - a set of numbers (GENES) for how it drafts, which projections it trusts for lineups and for
rest-of-season value, how aggressive it is on waivers, whether it streams K and DEF, and how it proposes and accepts
trades. Trades are 1-for-1 between teams, each side judging with its own valuation, so a team with better values can
win trades against teams that accept on worse ones - and a strategy that accepts too easily gets fleeced.

v2 (Nov 22, EVO_V2=1): trades can also be 2-for-1 / 1-for-2 (gene pkg_p); waivers are FAAB (100 budget, blind bids,
genes faab_aggr / faab_reserve; EVO_WAIVER=rolling keeps reverse-standings order); and the typical managers act more
like people: they answer only EVO_HUMAN_RESPONSE of offers, value their own players EVO_HUMAN_ENDOW higher, and turn
down deals whose market (FantasyPros) value in is under EVO_HUMAN_LOOKS of the value out.

Each league seats N_TYPICAL fixed "typical manager" teams (FantasyPros board and rankings, moderate waivers, rarely
proposes, accepts trades that look good by FantasyPros) plus strategies from the population. Fitness = wins / 14 +
playoffs + 2 x title, averaged over every league a strategy has played (survivors keep their history). Each
generation the bottom half is replaced by mutated / crossed-over copies of the top half.

Overfitting guard: evolve only on 2023-2024; every VALIDATE_EVERY generations the current top strategies (and the
fixed baselines) are scored on 2025 in fresh leagues against typical managers; the final report adds 2026 (Weeks 1-10:
wins and points only), which nothing was tuned on.

  python lab18_evo.py FRAME.parquet OUT_DIR  (EVO_GENERATIONS, EVO_POP, EVO_LEAGUES, EVO_SEED, EVO_MINUTES)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import lab5 as L
import lab11_sim as S11
import lab12_season as G

TEAMS, ROSTER, ROUNDS = 12, 15, 15
REG, LAST, TRADE_DEADLINE = 14, 17, 12
POSN = ["QB", "RB", "WR", "TE", "K", "DEF"]
MIN = [1, 2, 2, 1, 1, 1]
CAP = [3, 7, 7, 3, 2, 2]
SLOTS = [[0], [1], [1], [2], [2], [3], [1, 2, 3], [4], [5]]      # lineup order: fixed slots first, flex, K, DEF
SRC = ["model", "fp", "sleeper", "espn"]
N_TYPICAL = int(os.environ.get("EVO_TYPICAL", "4"))
# v2 (Nov 22): 2-for-1 / 1-for-2 trades, FAAB waivers, and "human" typical managers who ignore some offers, overvalue
# their own players and won't take a deal that looks bad by market value
V2 = os.environ.get("EVO_V2", "0") == "1"
WAIVER = os.environ.get("EVO_WAIVER", "faab" if V2 else "rolling")      # "faab" or "rolling" (reverse standings)
FAAB_BUDGET = 100.0
HUMAN_RESPONSE = float(os.environ.get("EVO_HUMAN_RESPONSE", "0.5"))     # chance a typical manager answers an offer
HUMAN_ENDOW = float(os.environ.get("EVO_HUMAN_ENDOW", "0.10"))         # they value their own players this much higher
HUMAN_LOOKS = float(os.environ.get("EVO_HUMAN_LOOKS", "0.9"))          # market value in / out they need at least

# gene: (low, high, kind)  kind: f = float, i = int
GENES = {
    "draft_model_w": (0.0, 1.0, "f"),    # draft board: 0 = FantasyPros consensus, 1 = the model's preseason board
    "bias_QB": (-0.4, 0.4, "f"), "bias_RB": (-0.4, 0.4, "f"), "bias_WR": (-0.4, 0.4, "f"), "bias_TE": (-0.4, 0.4, "f"),
    "qb_min_round": (1, 12, "i"), "te_min_round": (1, 12, "i"), "k_min_round": (6, 15, "i"), "def_min_round": (6, 15, "i"),
    "lw_model": (0.0, 1.0, "f"), "lw_fp": (0.0, 1.0, "f"), "lw_sleeper": (0.0, 1.0, "f"), "lw_espn": (0.0, 1.0, "f"),
    "rw_model": (0.0, 1.0, "f"), "rw_fp": (0.0, 1.0, "f"), "rw_sleeper": (0.0, 1.0, "f"), "rw_espn": (0.0, 1.0, "f"),
    "kd_lab": (0.0, 1.0, "f"), "kd_sleeper": (0.0, 1.0, "f"), "kd_espn": (0.0, 1.0, "f"),
    "waiver_margin": (0.0, 30.0, "f"),   # ROS points over the player dropped needed for a pickup
    "max_moves": (0, 3, "i"),
    "stream_k": (0.0, 1.0, "f"), "stream_def": (0.0, 1.0, "f"), "stream_margin": (0.0, 4.0, "f"),
    "trade_p": (0.0, 1.0, "f"),          # chance per week of proposing a trade (weeks 3-12)
    "trade_min_gain": (0.0, 3.0, "f"),   # own lineup gain (points per week, own values) required to propose
    "accept_thr": (-0.5, 3.0, "f"),      # own lineup gain required to accept an offer
    "pkg_p": (0.0, 1.0, "f"),            # v2: chance a proposal also looks at 2-for-1 and 1-for-2 packages
    "faab_aggr": (0.0, 2.0, "f"),        # v2: FAAB bid = faab_aggr x rest-of-season gain (points), capped by budget
    "faab_reserve": (0.0, 60.0, "f"),    # v2: budget kept back for later
}
TYPICAL = {"draft_model_w": 0.0, "bias_QB": 0, "bias_RB": 0, "bias_WR": 0, "bias_TE": 0, "qb_min_round": 4,
           "te_min_round": 4, "k_min_round": 14, "def_min_round": 13, "lw_model": 0, "lw_fp": 1, "lw_sleeper": 0,
           "lw_espn": 0, "rw_model": 0, "rw_fp": 1, "rw_sleeper": 0, "rw_espn": 0, "kd_lab": 1, "kd_sleeper": 0,
           "kd_espn": 0, "waiver_margin": 10, "max_moves": 1, "stream_k": 0.2, "stream_def": 0.3, "stream_margin": 2,
           "trade_p": 0.1, "trade_min_gain": 1.0, "accept_thr": 0.5, "pkg_p": 0.3, "faab_aggr": 0.5,
           "faab_reserve": 15}
BASELINES = {   # fixed reference strategies, scored alongside the evolved ones
    "typical (FantasyPros)": TYPICAL,
    "official (model+FP)": {**TYPICAL, "lw_model": 1, "lw_fp": 1, "rw_model": 1, "rw_fp": 1},
    "model+Sleeper+ESPN": {**TYPICAL, "lw_model": 1, "lw_fp": 0, "lw_sleeper": 1, "lw_espn": 1, "rw_model": 1,
                           "rw_fp": 0, "rw_sleeper": 1, "rw_espn": 1, "kd_sleeper": 1, "kd_espn": 1},
}


# --------------------------------------------------------------------------------------------- season data
class SeasonData:
    """Skill players (lab12 Season) + 32 kicker and 32 defense units, as week-indexed arrays."""

    def __init__(self, S, d, cv, games, ids, pre, kdef, other_kdef):
        board, _ = G.draft_board(S, games, ids)
        last_wk = int(d[d["season"] == S]["week"].max())
        se = G.Season(d, S, board, games, LAST)
        self.S, self.W = S, last_wk
        curve = cv[S - 1]
        P = len(se.pid)
        teams = sorted(set(games[(games["season"] == S) & (games["game_type"] == "REG")]["home_team"]))
        self.nk = len(teams)
        N = P + 2 * self.nk
        Wn = LAST + 1
        self.pos = np.r_[se.pos, np.full(self.nk, 4), np.full(self.nk, 5)]
        self.name = np.r_[se.name, [t + " K" for t in teams], [t + " DEF" for t in teams]]
        z = lambda: np.zeros((N, Wn))
        self.act, self.avail = z(), np.zeros((N, Wn), bool)
        self.act[:P] = se.act
        self.avail[:P] = se.has & ~se.sat
        # weekly projections by source (skill: lab11 rows; K / DEF: lab17 projection, Sleeper / ESPN if pulled)
        self.wk = {s: z() for s in SRC}
        self.wk["model"][:P], self.wk["fp"][:P] = se.w_model, se.w_fp
        self.wk["sleeper"][:P], self.wk["espn"][:P] = se.w_sleeper, se.w_espn
        self.kd = {s: z() for s in ["lab", "sleeper", "espn"]}
        u = kdef[kdef["season"] == S].set_index(["team", "week"])
        for j, t in enumerate(teams):
            for w in range(1, Wn):
                if (t, w) not in u.index:
                    continue
                r = u.loc[(t, w)]
                for unit, col, off in [(4, "k", P), (5, "def", P + self.nk)]:
                    i = off + j
                    if bool(r["played"]):
                        self.act[i, w] = r[f"{col}_pts"]
                        self.avail[i, w] = True
                    self.kd["lab"][i, w] = r[f"{col}_proj"] if pd.notna(r[f"{col}_proj"]) else 0.0
        for src, df in other_kdef.items():
            if df is None:
                continue
            x = df[df["season"] == S].copy()
            x["team"] = x["team"].astype(str).map(lambda t: G.FP_TEAM.get(t, t))
            x = x[x["week"] <= LAST].groupby(["week", "team", "pos"], as_index=False)["proj_ppr"].max()   # several kickers: the top one
            for _, r in x.iterrows():
                t = r["team"]
                if t not in teams:
                    continue
                j = teams.index(t)
                i = (P if r["pos"] == "K" else P + self.nk) + j
                self.kd[src][i, int(r["week"])] = float(r["proj_ppr"] or 0)
        for s in ["sleeper", "espn"]:      # fall back to the lab projection where the site had none
            self.kd[s] = np.where(self.kd[s] == 0, self.kd["lab"], self.kd[s])
        # rest-of-season per-game rates by source and expected games left
        self.rate = {s: z() for s in SRC}
        self.rate["model"][:P], self.rate["sleeper"][:P], self.rate["espn"][:P] = (se.rates["model"], se.rates["sleeper"],
                                                                                  se.rates["espn"])
        for w in range(1, Wn):
            rk = np.where(np.isnan(se.ros_rank[:, w]), se.wk_rank[:, w], se.ros_rank[:, w])
            rk = np.where(np.isnan(rk), se._last_rank(w) if w > 1 else rk, rk)
            rk = np.where(np.isnan(rk), se.pre_rank, rk)
            ok = ~np.isnan(rk) & (se.pos >= 0)
            v = np.zeros(P)
            v[ok] = S11.rank_pts(curve, np.array(G.POS)[se.pos[ok]], rk[ok])
            self.rate["fp"][:P, w] = v
        kd_rate = {}
        for s in ["lab", "sleeper", "espn"]:      # K / DEF ROS rate: average projection over the season so far + this week
            c = np.cumsum(self.kd[s], axis=1)
            n = np.maximum(1, np.cumsum(self.kd[s] != 0, axis=1))
            kd_rate[s] = c / n
        for s in SRC:
            m = {"model": "lab", "fp": "lab", "sleeper": "sleeper", "espn": "espn"}[s]
            self.rate[s][P:] = kd_rate[m][P:]
        self.games_left = np.zeros((N, Wn))
        self.games_left[:P] = se.games_left
        plays_kd = np.zeros((2 * self.nk, Wn), bool)
        g = games[(games["season"] == S) & (games["game_type"] == "REG")]
        tw = {(t, int(w)) for t, w in zip(g["home_team"], g["week"])} | {(t, int(w)) for t, w in zip(g["away_team"], g["week"])}
        for j, t in enumerate(teams):
            for w in range(1, Wn):
                plays_kd[j, w] = plays_kd[self.nk + j, w] = (t, w) in tw
        for w in range(1, Wn):
            self.games_left[P:, w] = plays_kd[:, w:LAST + 1].sum(1)
        out_now = np.zeros((N, Wn), bool)
        out_now[:P] = se.plays & ~(se.has & se.exp)
        missed = np.zeros((N, Wn))
        missed[:P] = se.missed
        self.exp_games = np.maximum(0, self.games_left - np.where(out_now, missed + 1, 0))
        # draft boards: FantasyPros consensus (overall rank) and the model's preseason projection (lab13)
        self.P = P
        ecr = board.set_index("player_id")["ecr"]
        self.fp_board = np.full(N, 999.0)
        for i, p in enumerate(se.pid):
            if p in ecr.index:
                self.fp_board[i] = ecr[p]
        mp = pre[pre["season"] == S].set_index("player_id")["model_total"]
        mt = np.array([mp.get(p, np.nan) for p in se.pid])
        model_board = np.full(N, 999.0)
        # model board in overall-rank units: value over the 12th QB / 30th RB / 36th WR / 12th TE, ranked
        vor = np.full(P, -1e9)
        for q, k in zip(range(4), [12, 30, 36, 12]):
            sel = (se.pos == q) & ~np.isnan(mt)
            if sel.sum() > k:
                vor[sel] = mt[sel] - np.sort(mt[sel])[::-1][k - 1]
        order = np.argsort(-vor)
        rk = np.empty(P)
        rk[order] = np.arange(1, P + 1)
        model_board[:P] = np.where(vor > -1e8, rk, 999.0)
        self.model_board = model_board
        # K / DEF draft order: last season's points (ranked), placed after skill players by the round rules
        prev = kdef[(kdef["season"] == S - 1) & kdef["played"]].groupby("team")[["k_pts", "def_pts"]].sum()
        for j, t in enumerate(teams):
            for unit, col, off in [(4, "k_pts", P), (5, "def_pts", P + self.nk)]:
                v = prev[col].get(t, np.nan) if col in prev else np.nan
                rank = (prev[col].rank(ascending=False).get(t, 32) if col in prev else 32)
                self.fp_board[off + j] = self.model_board[off + j] = 150 + rank
        self.ecr_sd = np.ones(N) * 3.0
        sd = board.set_index("player_id")["sd"]
        for i, p in enumerate(se.pid):
            if p in sd.index:
                self.ecr_sd[i] = max(1.0, sd[p])
        self.draft_pool = np.where((self.fp_board < 400) | (self.model_board < 400))[0]
        self.teams_nfl = teams


# ------------------------------------------------------------------------------------------------ strategies
def random_strategy(rng):
    s = {}
    for k, (lo, hi, kind) in GENES.items():
        s[k] = int(rng.integers(lo, hi + 1)) if kind == "i" else float(rng.uniform(lo, hi))
    return s


def mutate(s, rng, rate=0.25, scale=0.15):
    c = dict(s)
    for k, (lo, hi, kind) in GENES.items():
        if rng.random() < rate:
            if kind == "i":
                c[k] = int(np.clip(c[k] + rng.integers(-2, 3), lo, hi))
            else:
                c[k] = float(np.clip(c[k] + rng.normal(0, scale * (hi - lo)), lo, hi))
    return c


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENES}


def weights(s, prefix, names):
    w = np.array([max(0.0, s[f"{prefix}{n}"]) for n in names])
    return w / w.sum() if w.sum() > 0 else None


class Brain:
    """A strategy's precomputed views of the season: weekly start scores and rest-of-season values."""

    def __init__(self, s, sd):
        self.s = s
        lw = weights(s, "lw_", SRC)
        lw = lw if lw is not None else np.array([0, 1, 0, 0])
        kw = weights(s, "kd_", ["lab", "sleeper", "espn"])
        kw = kw if kw is not None else np.array([1, 0, 0])
        rw = weights(s, "rw_", SRC)
        rw = rw if rw is not None else np.array([0, 1, 0, 0])
        P = sd.P
        self.week = sum(w * sd.wk[n] for w, n in zip(lw, SRC))
        self.week[P:] = sum(w * sd.kd[n][P:] for w, n in zip(kw, ["lab", "sleeper", "espn"]))
        rate = sum(w * sd.rate[n] for w, n in zip(rw, SRC))
        self.rate = rate                                        # per game
        self.ros = rate * sd.exp_games                           # rest-of-season points
        board = (1 - s["draft_model_w"]) * sd.fp_board + s["draft_model_w"] * sd.model_board
        bias = np.array([s["bias_QB"], s["bias_RB"], s["bias_WR"], s["bias_TE"], 0, 0])
        self.board = board * np.exp(-bias[sd.pos])


# ------------------------------------------------------------------------------------------------ the league
def best_lineup(roster, pos, score, act=None):
    """Greedy best lineup (fixed slots, then flex) by score; returns (score total, actual total)."""
    order = sorted(roster, key=lambda j: -score[j])
    used, tot, real = set(), 0.0, 0.0
    for slot in SLOTS:
        for j in order:
            if j not in used and pos[j] in slot and score[j] > -1e8:
                used.add(j)
                tot += score[j]
                if act is not None:
                    real += act[j]
                break
    return tot, real


def counts(roster, pos):
    c = [0] * 6
    for j in roster:
        c[pos[j]] += 1
    return c


def legal_swap(roster, pos, add, drop):
    c = counts(roster, pos)
    c[pos[drop]] -= 1
    c[pos[add]] += 1
    return all(MIN[q] <= c[q] <= CAP[q] for q in range(6))


def legal_move(roster, pos, out, inn):
    """Roster after removing `out` and adding `inn` (lists) stays within the position minimums / caps."""
    c = counts(roster, pos)
    for j in out:
        c[pos[j]] -= 1
    for j in inn:
        c[pos[j]] += 1
    return all(MIN[q] <= c[q] <= CAP[q] for q in range(6))


def settle(roster, out, inn, ros, pos, rostered, taken=()):
    """Roster after a trade: remove `out`, add `inn`; if it's now too big, drop the lowest-value players not just
    received; if too small, add the best free agents (skill players). Returns (new roster, dropped to FA, added
    from FA) or None if the position limits can't be met."""
    n = len(roster)
    new = [j for j in roster if j not in out] + list(inn)
    dropped, added = [], []
    while len(new) > n:
        cut = None
        for j in sorted(new, key=lambda j: ros[j]):
            if j in inn:
                continue
            c = counts([x for x in new if x != j], pos)
            if all(MIN[q] <= c[q] for q in range(6)):
                cut = j
                break
        if cut is None:
            return None
        new.remove(cut); dropped.append(cut)
    if len(new) < n:
        fa = np.where(~rostered & (pos < 4))[0]
        fa = [j for j in fa[np.argsort(-ros[fa])] if j not in taken]
        for j in fa:
            if len(new) >= n:
                break
            c = counts(new + [j], pos)
            if c[pos[j]] <= CAP[pos[j]]:
                new.append(int(j)); added.append(int(j))
    c = counts(new, pos)
    if len(new) != n or not all(MIN[q] <= c[q] <= CAP[q] for q in range(6)):
        return None
    return new, dropped, added


def market(sd, w):
    """Market view of rest-of-season value (FantasyPros), for the typical managers' "does this look fair" check."""
    return sd.rate["fp"][:, w] * sd.exp_games[:, w]


def run_league(sd, strategies, rng, last_week=None):
    """One league-season. strategies: list of 12 gene dicts. Returns per-team wins, playoffs, title, points, trades."""
    last_week = last_week or sd.W
    brains = [Brain(s, sd) for s in strategies]
    pos = sd.pos
    N = len(pos)
    # ---- draft (snake; each team its own board + noise sized by the experts' spread)
    pool = sd.draft_pool
    taken = np.zeros(N, bool)
    rosters = [[] for _ in range(TEAMS)]
    noise = rng.normal(0, 1, (TEAMS, len(pool))) * sd.ecr_sd[pool] * 0.8
    for rd in range(ROUNDS):
        order = range(TEAMS) if rd % 2 == 0 else range(TEAMS - 1, -1, -1)
        for t in order:
            s = strategies[t]
            c = counts(rosters[t], pos)
            left = ROUNDS - rd
            need = [q for q in range(6) if c[q] < MIN[q]]
            must = sum(MIN[q] - c[q] for q in need) >= left
            okq = np.zeros(6, bool)
            for q in range(6):
                minr = {0: s["qb_min_round"], 3: s["te_min_round"], 4: s["k_min_round"], 5: s["def_min_round"]}.get(q, 1)
                okq[q] = c[q] < CAP[q] and (rd + 1 >= minr or must) and (not must or q in need)
                if q in (4, 5) and c[q] >= 1 and not must:
                    okq[q] = False                       # one K / DEF in the draft is enough
            cand = ~taken[pool] & okq[pos[pool]]
            if not cand.any():
                cand = ~taken[pool] & np.isin(pos[pool], need if need else [1, 2])
            sc = brains[t].board[pool] + noise[t]
            j = np.where(cand)[0][np.argmin(sc[cand])]
            taken[pool[j]] = True
            rosters[t].append(int(pool[j]))
    rostered = np.zeros(N, bool)
    for r in rosters:
        rostered[r] = True
    # ---- schedule
    tm = list(rng.permutation(TEAMS))
    rr = []
    for _ in range(TEAMS - 1):
        rr.append([(tm[i], tm[TEAMS - 1 - i]) for i in range(TEAMS // 2)])
        tm = [tm[0]] + [tm[-1]] + tm[1:-1]
    sched = rr + rr[: REG - (TEAMS - 1)]
    wins, pf = np.zeros(TEAMS), np.zeros(TEAMS)
    pts_all = np.zeros((TEAMS, LAST + 1))
    trades, moves = np.zeros(TEAMS), np.zeros(TEAMS)
    alive, seeds = None, None
    playoff = last_week >= LAST
    budget = np.full(TEAMS, FAAB_BUDGET)
    faab_spent = np.zeros(TEAMS)

    def best_claim(t, w):
        """Team t's best rest-of-season upgrade from free agents (skill players): (gain, add, drop) or None."""
        ros = brains[t].ros[:, w]
        fa = np.where(~rostered & (pos < 4))[0]
        best = None
        for q in range(4):
            fq = fa[pos[fa] == q]
            if not len(fq):
                continue
            add = fq[np.argmax(ros[fq])]
            for dj in sorted(rosters[t], key=lambda j: ros[j]):
                if pos[dj] < 4 and legal_swap(rosters[t], pos, add, dj):
                    g = ros[add] - ros[dj]
                    if g > strategies[t]["waiver_margin"] and (best is None or g > best[0]):
                        best = (g, int(add), int(dj))
                    break
        return best
    for w in range(1, min(last_week, LAST) + 1):
        if w >= 2:
            order = np.lexsort((pf, wins))
            fa = np.where(~rostered)[0]
            if WAIVER == "faab":
                # 1') FAAB: every team claims its best upgrade with a blind bid; the highest bid wins each player (ties
                # to the worse record); up to max_moves rounds
                n_done = np.zeros(TEAMS, int)
                rank_of = {int(t): i for i, t in enumerate(order)}
                for _rd in range(3):
                    claims = {}
                    for t in order:
                        s = strategies[t]
                        if n_done[t] >= s["max_moves"]:
                            continue
                        c = best_claim(t, w)
                        if not c:
                            continue
                        spare = max(0.0, budget[t] - s["faab_reserve"])
                        bid = float(min(spare, round(s["faab_aggr"] * c[0])))
                        claims.setdefault(c[1], []).append((bid, -rank_of[int(t)], int(t), c[2]))
                    if not claims:
                        break
                    for add, cl in claims.items():
                        bid, _, t, dj = max(cl)
                        if rostered[add] or dj not in rosters[t]:
                            continue
                        rosters[t].remove(dj); rosters[t].append(int(add))
                        rostered[dj], rostered[add] = False, True
                        budget[t] -= bid; faab_spent[t] += bid
                        n_done[t] += 1; moves[t] += 1
            for t in order:
                b, s = brains[t], strategies[t]
                ros = b.ros[:, w]
                n_mv = 0 if WAIVER != "faab" else s["max_moves"]       # FAAB handled the upgrades above
                # 1) rest-of-season upgrade (skill players)
                while n_mv < s["max_moves"]:
                    fa = np.where(~rostered & (pos < 4))[0]
                    if not len(fa):
                        break
                    best_gain, sw = s["waiver_margin"], None
                    for q in range(4):
                        fq = fa[pos[fa] == q]
                        if not len(fq):
                            continue
                        add = fq[np.argmax(ros[fq])]
                        for dj in sorted(rosters[t], key=lambda j: ros[j]):
                            if pos[dj] < 4 and legal_swap(rosters[t], pos, add, dj):
                                g = ros[add] - ros[dj]
                                if g > best_gain:
                                    best_gain, sw = g, (add, dj)
                                break
                    if not sw:
                        break
                    add, dj = sw
                    rosters[t].remove(dj); rosters[t].append(int(add))
                    rostered[dj], rostered[add] = False, True
                    n_mv += 1
                    moves[t] += 1
                # 2) K / DEF streaming by this week's projection
                for q, gene in [(4, "stream_k"), (5, "stream_def")]:
                    if rng.random() >= s[gene]:
                        continue
                    mine = [j for j in rosters[t] if pos[j] == q]
                    fq = np.where(~rostered & (pos == q) & sd.avail[:, w])[0]
                    if not mine or not len(fq):
                        continue
                    cur = max(mine, key=lambda j: b.week[j, w] if sd.avail[j, w] else -1e9)
                    cur_v = b.week[cur, w] if sd.avail[cur, w] else 0.0
                    add = fq[np.argmax(b.week[fq, w])]
                    if b.week[add, w] > cur_v + s["stream_margin"]:
                        dj = min(mine, key=lambda j: b.week[j, w] if sd.avail[j, w] else -1e9)
                        rosters[t].remove(dj); rosters[t].append(int(add))
                        rostered[dj], rostered[add] = False, True
                        moves[t] += 1
                # 3) fill a starting slot the roster can't fill this week (bye / out)
                c_av = [0] * 6
                for j in rosters[t]:
                    if sd.avail[j, w]:
                        c_av[pos[j]] += 1
                short = [q for q in range(6) if c_av[q] < [1, 2, 2, 1, 1, 1][q]]
                for q in short[:1]:
                    fq = np.where(~rostered & (pos == q) & sd.avail[:, w])[0]
                    if not len(fq):
                        continue
                    add = fq[np.argmax(b.week[fq, w])]
                    starters = set()
                    for slot in SLOTS:
                        for j in sorted(rosters[t], key=lambda j: -(b.week[j, w] if sd.avail[j, w] else -1e9)):
                            if j not in starters and pos[j] in slot and sd.avail[j, w]:
                                starters.add(j)
                                break
                    for dj in sorted(rosters[t], key=lambda j: ros[j]):
                        if dj in starters or not legal_swap(rosters[t], pos, add, dj):
                            continue
                        rosters[t].remove(dj); rosters[t].append(int(add))
                        rostered[dj], rostered[add] = False, True
                        moves[t] += 1
                        break
            # 4) trades (weeks 3 to the deadline), each side by its own rest-of-season values. v1: 1-for-1 only.
            # v2: also 2-for-1 / 1-for-2 (the side getting more drops its weakest player, the other adds its best free
            # agent), and typical managers answer only some offers, overvalue their own players and need the deal to
            # look fair by market value.
            if 3 <= w <= TRADE_DEADLINE:
                mk = market(sd, w) if V2 else None
                for t in rng.permutation(TEAMS):
                    s = strategies[t]
                    if rng.random() >= s["trade_p"]:
                        continue
                    u = int(rng.choice([x for x in range(TEAMS) if x != t]))
                    bt, bu = brains[t], brains[u]
                    rt, ru = bt.rate[:, w], bu.rate[:, w]
                    base_t, _ = best_lineup(rosters[t], pos, rt)
                    base_u, _ = best_lineup(rosters[u], pos, ru)
                    mine = sorted([j for j in rosters[t] if pos[j] < 4], key=lambda j: -rt[j])
                    theirs = sorted([j for j in rosters[u] if pos[j] < 4], key=lambda j: -rt[j])[:8]
                    cands = [([g_], [r_]) for g_ in mine for r_ in theirs]
                    if V2 and rng.random() < s.get("pkg_p", 0):
                        m7, t6 = mine[:7], theirs[:6]
                        cands += [([m7[i], m7[k]], [r_]) for i in range(len(m7)) for k in range(i + 1, len(m7)) for r_ in theirs]
                        cands += [([g_], [t6[i], t6[k]]) for g_ in mine for i in range(len(t6)) for k in range(i + 1, len(t6))]
                    typical_u = strategies[u] is TYPICAL
                    best = None
                    for give, get in cands:
                        nt = settle(rosters[t], give, get, bt.ros[:, w], pos, rostered)
                        if nt is None:
                            continue
                        gt = best_lineup(nt[0], pos, rt)[0] - base_t
                        if gt < s["trade_min_gain"] or (best and gt <= best[0]):
                            continue
                        nu = settle(rosters[u], get, give, bu.ros[:, w], pos, rostered, taken=nt[2])
                        if nu is None:
                            continue
                        gu = best_lineup(nu[0], pos, ru)[0] - base_u
                        need = strategies[u]["accept_thr"]
                        if V2 and typical_u:
                            need += HUMAN_ENDOW * sum(ru[j] for j in get)
                            if mk[give].sum() < HUMAN_LOOKS * mk[get].sum():
                                continue
                        if gu >= need:
                            best = (gt, give, get, nt, nu)
                    if best and V2 and typical_u and rng.random() >= HUMAN_RESPONSE:
                        best = None                               # they never answered
                    if best:
                        _, give, get, nt, nu = best
                        for team, (new, dropped, added) in [(t, nt), (u, nu)]:
                            rosters[team] = new
                            for j in dropped:
                                rostered[j] = False
                            for j in added:
                                rostered[j] = True
                        trades[t] += 1; trades[u] += 1
        # ---- lineups and scores
        pts = np.zeros(TEAMS)
        for t in range(TEAMS):
            sc = np.where(sd.avail[:, w], brains[t].week[:, w], -1e9)
            pts[t] = best_lineup(rosters[t], pos, sc, sd.act[:, w])[1]
        pts_all[:, w] = pts
        if w <= REG:
            for a, b2 in sched[w - 1]:
                pf[a] += pts[a]; pf[b2] += pts[b2]
                if pts[a] > pts[b2]:
                    wins[a] += 1
                elif pts[b2] > pts[a]:
                    wins[b2] += 1
                else:
                    wins[a] += .5; wins[b2] += .5
            if w == REG and playoff:
                seeds = list(np.lexsort((-pf, -wins))[:8])
                alive = list(seeds)
        elif playoff and alive:
            k = len(alive)
            alive = [alive[i] if pts[alive[i]] >= pts[alive[k - 1 - i]] else alive[k - 1 - i] for i in range(k // 2)]
    made = np.zeros(TEAMS)
    title = np.zeros(TEAMS)
    if seeds is not None:
        made[seeds] = 1
        title[alive[0]] = 1
    last = np.zeros(TEAMS)
    last[np.lexsort((pf, wins))[0]] = 1
    weeks = min(last_week, REG)
    return {"wins": wins, "win_rate": wins / weeks, "playoffs": made, "title": title, "last": last,
            "ppw": pts_all[:, 1:min(last_week, LAST) + 1].mean(1), "trades": trades, "moves": moves, "faab": faab_spent, "rosters": rosters}


def fitness(r, t, full):
    return r["win_rate"][t] + (r["playoffs"][t] + 2 * r["title"][t] if full else 0.0)


# ------------------------------------------------------------------------------------------------ evolution
def main():
    t0 = time.time()
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    gens = int(os.environ.get("EVO_GENERATIONS", "50"))
    pop_n = int(os.environ.get("EVO_POP", "40"))
    n_leagues = int(os.environ.get("EVO_LEAGUES", "60"))
    minutes = float(os.environ.get("EVO_MINUTES", "330"))
    val_every = int(os.environ.get("VALIDATE_EVERY", "10"))
    seed = int(os.environ.get("EVO_SEED", "1"))
    rng = np.random.default_rng(seed)
    frame = pd.read_parquet(sys.argv[1], columns=["season", "week", "season_type", "position", "expected",
                                                   "ecr_week_pos_rank", L.TARGET])
    cv = S11.curves(frame)
    d = S11.prepare(pd.read_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet")), cv)
    games = pd.read_parquet("data/games.parquet", columns=["season", "week", "game_type", "gameday", "home_team",
                                                           "away_team"])
    ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], low_memory=False)
    pre = pd.read_parquet(os.path.join(L.LAB_DIR, "preseason_proj.parquet"))
    kdef = pd.read_parquet(os.path.join(L.LAB_DIR, "kdef_weekly.parquet"))
    other = load_other_kdef()
    sds = {S: SeasonData(S, d, cv, games, ids, pre, kdef, other) for S in [2023, 2024, 2025, 2026]}
    print("data ready", round((time.time() - t0) / 60, 1), "min", flush=True)
    pop = [random_strategy(rng) for _ in range(pop_n - len(BASELINES))] + [dict(v) for v in BASELINES.values()]
    hist = [new_hist_rec() for _ in pop]
    log = []
    last_val = -1
    rank = np.arange(len(pop))
    for gen in range(gens):
        if (time.time() - t0) / 60 > minutes:
            break
        n_slots = TEAMS - N_TYPICAL
        for lg in range(n_leagues):
            S = 2023 if rng.random() < 0.5 else 2024
            idx = list(rng.choice(len(pop), n_slots, replace=len(pop) < n_slots))
            strategies = [pop[i] for i in idx] + [TYPICAL] * N_TYPICAL
            order = rng.permutation(TEAMS)                 # random draft slots
            strategies = [strategies[o] for o in order]
            seat_of = {int(np.where(order == k)[0][0]): idx[k] for k in range(n_slots)}
            r = run_league(sds[S], strategies, rng)
            for seat, pi in seat_of.items():
                h = hist[pi]
                h["n"] += 1
                h["ss"] += fitness(r, seat, True) ** 2
                for key, val in [("fit", fitness(r, seat, True)), ("wins", r["wins"][seat]), ("po", r["playoffs"][seat]),
                                 ("title", r["title"][seat]), ("trades", r["trades"][seat])]:
                    h[key] += val
        means = [h["fit"] / max(1, h["n"]) for h in hist]
        # rank by a lower confidence bound (mean - 1 standard error), so a strategy has to keep winning over many
        # leagues to stay on top - one lucky generation isn't enough
        lcb = [lcb_of(h) for h in hist]
        rank = np.argsort(lcb)[::-1]
        best = rank[0]
        log.append({"gen": gen, "minutes": round((time.time() - t0) / 60, 1), "best_fit": round(means[best], 4),
                    "best_lcb": round(lcb[best], 4),
                    "median_fit": round(float(np.median(means)), 4), "best_n": hist[best]["n"],
                    "baselines": {k: round(means[len(pop) - len(BASELINES) + i], 4) for i, k in enumerate(BASELINES)}})
        print(json.dumps(log[-1]), flush=True)
        if (gen + 1) % val_every == 0 or gen == gens - 1:
            validate(sds, pop, hist, rank, rng, out_dir, gen)
            last_val = gen
        # selection: keep the top half (baselines always stay, for reference); refill with children of the top half
        keep = set(rank[: pop_n // 2]) | set(range(len(pop) - len(BASELINES), len(pop)))
        parents = [i for i in rank[: pop_n // 2]]
        new_pop, new_hist = [], []
        for i in sorted(keep):
            new_pop.append(pop[i]); new_hist.append(hist[i])
        while len(new_pop) < pop_n:
            a, b = rng.choice(parents, 2, replace=False)
            child = mutate(crossover(pop[a], pop[b], rng), rng)
            new_pop.append(child)
            new_hist.append(new_hist_rec())
        pop, hist = new_pop, new_hist
        pop, hist = reorder_baselines(pop, hist)
        json.dump({"log": log, "population": [{"genes": p, **h} for p, h in zip(pop, hist)]},
                  open(os.path.join(out_dir, "evo_state.json"), "w"), indent=1, default=float)
    if log and last_val != log[-1]["gen"]:            # stopped on time: score the final population too
        lcb = [lcb_of(h) for h in hist]
        validate(sds, pop, hist, np.argsort(lcb)[::-1], rng, out_dir, log[-1]["gen"])
    print("done", round((time.time() - t0) / 60, 1), "min", flush=True)


def new_hist_rec():
    return {"n": 0, "fit": 0.0, "ss": 0.0, "wins": 0.0, "po": 0.0, "title": 0.0, "trades": 0.0}





def lcb_of(h):
    n = h["n"]
    if n < 2:
        return -1e9
    m = h["fit"] / n
    var = max(0.0, h["ss"] / n - m * m)
    return m - np.sqrt(var / n)


def load_other_kdef():
    """Sleeper / ESPN K and DEF projections pulled by automation/other_projections.py (None where missing)."""
    other = {}
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "automation", "history", "other_proj")
    for src in ["sleeper", "espn"]:
        fs = [os.path.join(od, f) for f in os.listdir(od) if f.startswith(f"{src}_kdef_")] if os.path.isdir(od) else []
        other[src] = pd.concat([pd.read_csv(f) for f in fs]) if fs else None
        if other[src] is not None and src == "espn":
            other[src] = espn_teams(other[src])     # ESPN gives the team as proTeamId
    return other


def reorder_baselines(pop, hist):
    bl = list(BASELINES.values())
    is_b = [any(p == v for v in bl) for p in pop]
    order = [i for i in range(len(pop)) if not is_b[i]] + [i for i in range(len(pop)) if is_b[i]]
    return [pop[i] for i in order], [hist[i] for i in order]


def validate(sds, pop, hist, rank, rng, out_dir, gen, n_val=int(os.environ.get("EVO_VAL_LEAGUES", "100"))):
    """Score the current top 8 strategies and the baselines on 2025 (full season) and 2026 (Weeks 1-10), each as two
    seats in leagues of typical managers, paired against the same leagues with those seats typical."""
    top = [int(i) for i in rank[:8]]
    cands = [("evolved #%d" % (k + 1), pop[i]) for k, i in enumerate(top)] + list(BASELINES.items())
    res = {}
    for S, full in [(2025, True), (2026, False)]:
        for name, strat in cands:
            diffs = {"fit": [], "wins": [], "playoffs": [], "title": [], "last": [], "ppw": []}
            for lg in range(n_val // (2 if not full else 1)):
                seed = int(rng.integers(1 << 31))
                seats = list(np.random.default_rng(seed).choice(TEAMS, 2, replace=False))
                base = [TYPICAL] * TEAMS
                test = [strat if i in seats else TYPICAL for i in range(TEAMS)]
                r0 = run_league(sds[S], base, np.random.default_rng(seed))
                r1 = run_league(sds[S], test, np.random.default_rng(seed))
                for st in seats:
                    diffs["fit"].append(fitness(r1, st, full) - fitness(r0, st, full))
                    for k in ["wins", "playoffs", "title", "last", "ppw"]:
                        diffs[k].append(r1[k][st] - r0[k][st])
            res[f"{S} {name}"] = {k: [round(float(np.mean(v)), 4), round(float(np.std(v) / np.sqrt(max(1, len(v) / 2))), 4)]
                                  for k, v in diffs.items()}
    rec = {"gen": gen, "validation": res, "top_genes": {f"evolved #{k + 1}": pop[i] for k, i in enumerate(top)},
           "top_train": {f"evolved #{k + 1}": {kk: round(hist[i][kk] / max(1, hist[i]["n"]), 4) for kk in
                                               ["fit", "wins", "po", "title", "trades"]} | {"n": hist[i]["n"]}
                         for k, i in enumerate(top)}}
    path = os.path.join(out_dir, "evo_validation.json")
    allv = json.load(open(path)) if os.path.exists(path) else []
    allv.append(rec)
    json.dump(allv, open(path, "w"), indent=1, default=float)
    print("validation gen", gen, json.dumps({k: v["fit"] for k, v in res.items()}), flush=True)


ESPN_TEAM = {1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
             11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
             21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
             33: "BAL", 34: "HOU"}


def espn_teams(df):
    df = df.copy()
    df["team"] = pd.to_numeric(df["team"], errors="coerce").map(ESPN_TEAM)
    return df.dropna(subset=["team"])


if __name__ == "__main__":
    main()

"""Nov 17, 2026 league sim, full-season stage: persistent teams, a real draft, weekly waivers, byes and injuries.

Real seasons (2023, 2024, 2025 weeks 1-17; 2026 weeks 1-10 so far), real points, and the projections each source had
before each week (walk-forward model rows from lab11_preds.py). Only the leagues are simulated.

League (BDSM League settings, K and DEF left out): 12 teams, start QB, RB, RB, WR, WR, TE, W/R/T; 13 QB/RB/WR/TE per
roster (at least 1 QB / 2 RB / 2 WR / 1 TE; at most 2 QB, 6 RB, 7 WR, 2 TE). Regular season weeks 1-14 (round robin,
then weeks 1-3 again), 8-team playoffs weeks 15-17 (seeded by wins, then points).

Draft (the same for every source, so the sim scores in-season management): FantasyPros' PPR draft consensus, the last
snapshot before the season's first game. Each team drafts from its own board: consensus rank + noise sized by the
experts' own spread on that player. Snake, 13 rounds.

Each week (2+), in reverse order of the standings, every team may make up to two waiver moves with its own source:
  upgrade  best free agent by rest-of-season value, if worth more than the team's least valuable droppable player
  hole     if a starting slot can't be filled (bye, out), the best free agent at that position by this week's
           projection, dropping the least valuable player the team can spare
then sets its lineup with its own source (players who didn't suit up are benched: inactives are on Yahoo before lock).

Sources (per team):
  fp        weekly: FantasyPros weekly rank -> points (last season's rank -> points curve)
            ROS:    FantasyPros rest-of-season rank -> points per game x team games left (weekly rank if no ROS rank)
  model     weekly: the model's projection (0 if not expected to play)
            ROS:    projected rate x expected games left. Rate = this week's projection if expected to play, else the
                    latest one. A player out now is assumed to miss as many more games as he already has in a row.
  official  weekly: website blend (model and FantasyPros ranks averaged within position, model points in that order)
            ROS:    the same rank blend applied to the two ROS values
Mixes: model lineups + FantasyPros waivers, FantasyPros lineups + model waivers (to split where any edge comes from).

Paired design: in each simulated league two random seats use the tested source and ten use FantasyPros; the same
league (same draft boards, schedule, seats) is replayed with all twelve on FantasyPros. The difference for the two
seats is the source's effect. Standard errors over leagues.

  python lab12_season.py FRAME.parquet            (SIM_LEAGUES, default 200)  -> lab/season_sim_*.{json,csv}
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import lab5 as L
import lab11_sim as S11

N_LEAGUES = int(os.environ.get("SIM_LEAGUES", "200"))
TEAMS, ROUNDS, LAST = 12, 13, 17
REG_WEEKS, PO_TEAMS = 14, 8
MIN = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
CAP = {"QB": 2, "RB": 6, "WR": 7, "TE": 2}
POS = ["QB", "RB", "WR", "TE"]
PC = {p: i for i, p in enumerate(POS)}
DRAFT_NOISE = 0.8          # x the experts' standard deviation on the player
MARGIN_PTS = 5.0           # an upgrade must be worth 5+ rest-of-season points more than the player dropped
REPL = {0: 13, 1: 30, 2: 36, 3: 13}   # replacement level: the Nth player at the position (12 starters + flex share)
FP_TEAM = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}
VARIANTS = {             # (lineup source, waiver source) for the two tested seats
    "model": ("model", "model"),
    "official": ("official", "official"),
    "model_lineups_only": ("model", "fp"),
    "model_waivers_only": ("fp", "model"),
    "control": ("fp", "fp"),
}


# ----------------------------------------------------------------------------------------------- season tables
def draft_board(season, games, ids):
    fp = pd.read_parquet("fpdata/db_fpecr.parquet", columns=["fp_page", "scrape_date", "id", "pos", "team", "ecr", "sd",
                                                                "player"])
    fp = fp[fp["fp_page"] == "/nfl/rankings/ppr-cheatsheets.php"]
    kick = games[(games["season"] == season) & (games["week"] == 1) & (games["game_type"] == "REG")]["gameday"].min()
    snap = fp[fp["scrape_date"] <= kick]["scrape_date"].max()
    b = fp[(fp["scrape_date"] == snap) & fp["pos"].isin(POS)].copy()
    m = ids.dropna(subset=["fantasypros_id", "gsis_id"]).drop_duplicates("fantasypros_id")
    m = dict(zip(m["fantasypros_id"].astype(int).astype(str), m["gsis_id"]))
    b["player_id"] = b["id"].astype(str).map(m)
    b = b.dropna(subset=["player_id"]).sort_values("ecr").drop_duplicates("player_id")
    b["team"] = b["team"].replace(FP_TEAM)
    return b[["player_id", "player", "pos", "team", "ecr", "sd"]].reset_index(drop=True), snap


class Season:
    """Player x week arrays for one season (weeks 1..LAST, index 0 unused)."""

    def __init__(self, d, season, board, games, last_week):
        rows = d[d["season"] == season]
        self.W = last_week
        pids = list(dict.fromkeys(list(rows["player_id"]) + list(board["player_id"].head(250))))
        self.pid = np.array(pids)
        ix = {p: i for i, p in enumerate(pids)}
        P, W = len(pids), LAST + 1
        info = rows.drop_duplicates("player_id", keep="last").set_index("player_id")
        bi = board.set_index("player_id")
        self.name = np.array([info["player_display_name"].get(p, bi["player"].get(p, p) if p in bi.index else p)
                              for p in pids])
        pos = [info["position"].get(p, bi["pos"].get(p) if p in bi.index else None) for p in pids]
        self.pos = np.array([PC.get(p, -1) for p in pos])
        z = lambda: np.zeros((P, W))
        self.has, self.act, self.sat, self.exp = z().astype(bool), z(), z().astype(bool), z().astype(bool)
        self.w_model, self.w_fp, self.w_off, self.ros_rank = z(), z(), z(), np.full((P, W), np.nan)
        self.wk_rank = np.full((P, W), np.nan)
        team = np.full((P, W), "", dtype=object)
        r = rows[rows["week"] <= LAST]
        pi, wi = r["player_id"].map(ix).values, r["week"].values.astype(int)
        self.has[pi, wi] = True
        self.act[pi, wi] = r[L.TARGET].values
        self.sat[pi, wi] = r["sat"].values
        self.exp[pi, wi] = r["expected"].values
        self.w_model[pi, wi] = r["model"].values
        self.w_fp[pi, wi] = r["fp"].values
        self.w_off[pi, wi] = r["official"].values
        self.ros_rank[pi, wi] = r["ecr_ros_pos_rank"].values
        self.wk_rank[pi, wi] = r["fp_rank"].values
        team[pi, wi] = r["team"].values
        # team for weeks without a row: last known, else the draft board's
        for i, p in enumerate(pids):
            last = bi["team"].get(p, "") if p in bi.index else ""
            first = next((t for t in team[i, 1:] if t), last)
            cur = first
            for w in range(1, W):
                cur = team[i, w] or cur
                team[i, w] = cur
        g = games[(games["season"] == season) & (games["game_type"] == "REG")]
        plays = {(t, int(w)) for t, w in zip(g["home_team"], g["week"])} | {(t, int(w)) for t, w in zip(g["away_team"], g["week"])}
        self.plays = np.array([[(team[i, w], w) in plays for w in range(W)] for i in range(P)])
        self.games_left = np.zeros((P, W))
        for w in range(1, W):
            self.games_left[:, w] = self.plays[:, w:LAST + 1].sum(1)
        # model rate: latest expected projection; games in a row missed so far (team games only)
        self.rate, self.missed = z(), z()
        last_rate, run = np.zeros(P), np.zeros(P)
        for w in range(1, W):
            avail = self.has[:, w] & self.exp[:, w]
            self.missed[:, w] = run
            self.rate[:, w] = np.where(avail, self.w_model[:, w], last_rate)
            last_rate = np.where(avail, self.w_model[:, w], last_rate)
            played = self.has[:, w] & ~self.sat[:, w]
            run = np.where(self.plays[:, w], np.where(played, 0, run + 1), run)
        self.board = board
        self.ix = ix
        pre = rows.groupby("player_id")["ecr_preseason_pos_rank"].first()
        bp = board.assign(pr=board.groupby("pos")["ecr"].rank(method="first")).set_index("player_id")["pr"]
        self.pre_rank = np.array([pre.get(p, np.nan) if not np.isnan(pre.get(p, np.nan)) else bp.get(p, np.nan)
                                  for p in pids], dtype=float)

    def ros(self, w, curve):
        """Rest-of-season values (points) for each source, before week w."""
        out_now = self.plays[:, w] & ~(self.has[:, w] & self.exp[:, w])
        k = np.where(out_now, self.missed[:, w] + 1, 0)
        model = self.rate[:, w] * np.maximum(0, self.games_left[:, w] - np.where(out_now, k, 0))
        rk = np.where(np.isnan(self.ros_rank[:, w]), self.wk_rank[:, w], self.ros_rank[:, w])
        rk = np.where(np.isnan(rk), self.pre_rank, rk)
        # players with no row this week (bye, IR): their latest ranks
        if w > 1:
            prev = self._last_rank(w)
            rk = np.where(np.isnan(rk), prev, rk)
        per = np.zeros(len(rk))
        ok = ~np.isnan(rk) & (self.pos >= 0)
        per[ok] = S11.rank_pts(curve, np.array(POS)[self.pos[ok]], rk[ok])
        fp = per * self.games_left[:, w]
        off = model.copy()
        for p in range(4):
            sel = np.where((self.pos == p) & ok & (model > 0))[0]
            if len(sel) < 2:
                continue
            mr = pd.Series(-model[sel]).rank(method="first").values
            er = pd.Series(rk[sel]).rank(method="first").values
            order = pd.Series(mr + er + mr * 1e-6).rank(method="first").values.astype(int) - 1
            off[sel] = np.sort(model[sel])[::-1][order]
        out = {}
        for k, v in {"model": model, "fp": fp, "official": off}.items():
            vor = v.copy()
            for p in range(4):
                sel = self.pos == p
                srt = np.sort(v[sel])[::-1]
                vor[sel] = v[sel] - (srt[REPL[p] - 1] if len(srt) >= REPL[p] else 0.0)
            out[k] = vor
        return out

    def _last_rank(self, w):
        if not hasattr(self, "_lr"):
            P = len(self.pid)
            self._lr = np.full((P, LAST + 1), np.nan)
            cur = np.full(P, np.nan)
            for ww in range(1, LAST + 1):
                rk = np.where(np.isnan(self.ros_rank[:, ww]), self.wk_rank[:, ww], self.ros_rank[:, ww])
                self._lr[:, ww] = cur
                cur = np.where(np.isnan(rk), cur, rk)
        return self._lr[:, w]

    def weekly(self, w, src):
        v = {"model": self.w_model, "fp": self.w_fp, "official": self.w_off}[src][:, w].copy()
        startable = self.has[:, w] & ~self.sat[:, w]       # inactives are known before lock
        return np.where(startable, v, -1e9)


# ------------------------------------------------------------------------------------------------------ league
def lineup(roster, pos, score, act):
    need = [1, 2, 2, 1]
    flex, pts, used = 1, 0.0, 0
    for i in sorted(roster, key=lambda j: -score[j]):
        if score[i] <= -1e8:
            break
        p = pos[i]
        if need[p] > 0:
            need[p] -= 1
        elif flex and p != 0:
            flex -= 1
        else:
            continue
        pts += act[i]
        used += 1
        if used == 7:
            break
    return pts


def unfilled(roster, pos, score):
    """Positions whose starting slots this roster can't fill this week (counting the flex for RB/WR/TE)."""
    have = [0, 0, 0, 0]
    for i in roster:
        if score[i] > -1e8:
            have[pos[i]] += 1
    short = [p for p in range(4) if have[p] < [1, 2, 2, 1][p]]
    if not short and sum(max(0, have[p] - [1, 2, 2, 1][p]) for p in (1, 2, 3)) < 1:
        short = [2]      # flex empty: a WR is the usual fill
    return short


def counts(roster, pos):
    c = [0, 0, 0, 0]
    for i in roster:
        c[pos[i]] += 1
    return c


def can_swap(roster, pos, add, drop):
    c = counts(roster, pos)
    c[pos[drop]] -= 1
    c[pos[add]] += 1
    return all(c[PC[p]] >= MIN[p] and c[PC[p]] <= CAP[p] for p in POS)


def draft(se, rng_boards):
    """Snake draft; each team picks the best player on its own noisy board that fits its roster rules."""
    b = se.board.head(260)
    cand = np.array([se.ix[p] for p in b["player_id"] if p in se.ix and se.pos[se.ix[p]] >= 0])
    ecr = b.set_index("player_id").loc[se.pid[cand], ["ecr", "sd"]].values
    boards = ecr[:, 0][None, :] + rng_boards.normal(0, 1, (TEAMS, len(cand))) * np.maximum(ecr[:, 1], 1.0)[None, :] * DRAFT_NOISE
    taken = np.zeros(len(cand), bool)
    rosters = [[] for _ in range(TEAMS)]
    for rd in range(ROUNDS):
        order = range(TEAMS) if rd % 2 == 0 else range(TEAMS - 1, -1, -1)
        for t in order:
            c = counts(rosters[t], se.pos)
            left = ROUNDS - rd
            need = [p for p in POS if c[PC[p]] < MIN[p]]
            must = sum(MIN[p] - c[PC[p]] for p in need) >= left
            okpos = np.array([(c[p] < CAP[POS[p]]) and (not must or POS[p] in need) for p in range(4)])
            ok = ~taken & okpos[se.pos[cand]]
            j = np.where(ok)[0][np.argmin(boards[t, ok])]
            taken[j] = True
            rosters[t].append(int(cand[j]))
    return rosters


def schedule(rng):
    teams = list(rng.permutation(TEAMS))
    rr = []
    for r in range(TEAMS - 1):
        rr.append([(teams[i], teams[TEAMS - 1 - i]) for i in range(TEAMS // 2)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rr + rr[: REG_WEEKS - (TEAMS - 1)]


def play_season(se, curves_prev, rosters0, sched, seats, variant):
    lsrc, wsrc = VARIANTS[variant]
    src_l = ["fp"] * TEAMS
    src_w = ["fp"] * TEAMS
    for s in seats:
        src_l[s], src_w[s] = lsrc, wsrc
    rosters = [list(r) for r in rosters0]
    rostered = np.zeros(len(se.pid), bool)
    for r in rosters:
        rostered[r] = True
    wins, pf = np.zeros(TEAMS), np.zeros(TEAMS)
    weekly_pts = np.zeros((TEAMS, LAST + 1))
    moves = np.zeros(TEAMS)
    last = min(se.W, LAST)
    playoff = last >= LAST
    alive, seeds = None, None
    for w in range(1, last + 1):
        if w >= 2:
            ros = se.ros(w, curves_prev)
            wk = {s: se.weekly(w, s) for s in set(src_l + src_w)}
            order = np.lexsort((pf, wins))           # worst record first
            fa_mask = ~rostered & (se.pos >= 0)
            for t in order:
                vr, vw = ros[src_w[t]], wk[src_w[t]]
                # 1) upgrade by rest-of-season value over replacement: the best gain over all positions
                best_gain, swap = 0.0, None
                for p in range(4):
                    fa = np.where(fa_mask & (se.pos == p))[0]
                    if not len(fa):
                        continue
                    add = fa[np.argmax(vr[fa])]
                    for dj in sorted(rosters[t], key=lambda j: vr[j]):
                        if can_swap(rosters[t], se.pos, add, dj):
                            gain = vr[add] - vr[dj] - MARGIN_PTS
                            if gain > best_gain:
                                best_gain, swap = gain, (add, dj)
                            break
                if swap:
                    add, dj = swap
                    rosters[t].remove(dj); rosters[t].append(add)
                    rostered[dj], rostered[add] = False, True
                    fa_mask[dj], fa_mask[add] = True, False
                    moves[t] += 1
                # 2) fill a starting slot the roster can't fill this week
                short = unfilled(rosters[t], se.pos, vw)
                if short:
                    p = short[0]
                    fa = np.where(fa_mask & (se.pos == p) & (vw > -1e8))[0]
                    if len(fa):
                        best = fa[np.argmax(vw[fa])]
                        if vw[best] > 0:
                            protect = set(sorted(rosters[t], key=lambda j: -vw[j])[:7])
                            for dj in sorted(rosters[t], key=lambda j: vr[j]):
                                if dj in protect or not can_swap(rosters[t], se.pos, best, dj):
                                    continue
                                rosters[t].remove(dj); rosters[t].append(best)
                                rostered[dj], rostered[best] = False, True
                                fa_mask[dj], fa_mask[best] = True, False
                                moves[t] += 1
                                break
        pts = np.array([lineup(rosters[t], se.pos, se.weekly(w, src_l[t]), se.act[:, w]) for t in range(TEAMS)])
        weekly_pts[:, w] = pts
        if w <= REG_WEEKS:
            for a, b in sched[w - 1]:
                pf[a] += pts[a]; pf[b] += pts[b]
                if pts[a] > pts[b]:
                    wins[a] += 1
                elif pts[b] > pts[a]:
                    wins[b] += 1
                else:
                    wins[a] += 0.5; wins[b] += 0.5
            if w == REG_WEEKS and playoff:
                seeds = list(np.lexsort((-pf, -wins))[:PO_TEAMS])
                alive = list(seeds)
        elif playoff and alive is not None:
            nxt = []
            k = len(alive)
            for i in range(k // 2):
                a, b = alive[i], alive[k - 1 - i]
                nxt.append(a if pts[a] >= pts[b] else b)
            alive = nxt
    champ = alive[0] if (playoff and alive and len(alive) == 1) else None
    made = np.zeros(TEAMS, bool)
    if seeds is not None:
        made[seeds] = True
    return {"wins": wins, "pf_week": weekly_pts[:, 1:min(last, REG_WEEKS) + 1].mean(1), "playoffs": made,
            "title": np.array([t == champ for t in range(TEAMS)]), "moves": moves,
            "pts_week_all": weekly_pts[:, 1:last + 1].mean(1)}


# --------------------------------------------------------------------------------------------------------- run
def main():
    t0 = time.time()
    frame = pd.read_parquet(sys.argv[1], columns=["season", "week", "season_type", "position", "expected",
                                                   "ecr_week_pos_rank", L.TARGET])
    rows = pd.read_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet"))
    cv = S11.curves(frame)
    d = S11.prepare(rows, cv)
    games = pd.read_parquet("data/games.parquet", columns=["season", "week", "game_type", "gameday", "home_team",
                                                           "away_team"])
    ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], low_memory=False)
    seasons = [int(s) for s in os.environ.get("SIM_SEASONS", "2023,2024,2025,2026").split(",")]
    recs, checks = [], {}
    for S in seasons:
        board, snap = draft_board(S, games, ids)
        last_wk = int(d[d["season"] == S]["week"].max())
        se = Season(d, S, board, games, last_wk)
        checks[S] = {"draft_snapshot": snap, "board": len(board), "players": len(se.pid), "last_week": last_wk}
        rng = np.random.default_rng(1000 + S)
        for lg in range(N_LEAGUES):
            rb = np.random.default_rng(rng.integers(1 << 31))
            rosters = draft(se, rb)
            sched = schedule(rb)
            seats = list(rb.choice(TEAMS, 2, replace=False))
            if lg == 0:
                checks[S]["example_first_round"] = [se.name[rosters[t][0]] for t in range(TEAMS)]
            res = {v: play_season(se, cv[S - 1], rosters, sched, seats, v) for v in VARIANTS}
            for v, r in res.items():
                for s in seats:
                    recs.append({"season": S, "league": lg, "seat": int(s), "variant": v,
                                 **{k: float(r[k][s]) for k in ["wins", "pf_week", "playoffs", "title", "moves",
                                                                 "pts_week_all"]}})
            if lg % 50 == 0:
                print(S, lg, round((time.time() - t0) / 60, 1), "min", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(L.LAB_DIR, "season_sim_seats.csv"), index=False)
    summary = summarize(df)
    json.dump({"leagues_per_season": N_LEAGUES, "checks": checks, "summary": summary},
              open(os.path.join(L.LAB_DIR, "season_sim_results.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1))


def summarize(df):
    ctrl = df[df["variant"] == "control"].set_index(["season", "league", "seat"])
    out = {}
    for scope, sub in [("2023-2025", df[df["season"] <= 2025]), ("2026 wk1-10", df[df["season"] == 2026])] + \
                      [(str(s), df[df["season"] == s]) for s in sorted(df["season"].unique())]:
        if sub.empty:
            continue
        o = {}
        for v in VARIANTS:
            if v == "control":
                continue
            a = sub[sub["variant"] == v].set_index(["season", "league", "seat"])
            c = ctrl.loc[a.index]
            r = {}
            for k in ["wins", "pf_week", "pts_week_all", "playoffs", "title", "moves"]:
                diff = (a[k] - c[k]).groupby(level=[0, 1]).mean()       # one observation per league
                r[k] = [round(diff.mean(), 4), round(diff.std(ddof=1) / np.sqrt(len(diff)), 4)]
            r["control_rates"] = {k: round(c[k].mean(), 4) for k in ["wins", "playoffs", "title", "pf_week", "moves"]}
            o[v] = r
        out[scope] = o
    return out


if __name__ == "__main__":
    main()

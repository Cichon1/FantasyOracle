"""Nov 17, 2026 league sim, draft stage: does drafting from the model's board win more?

Same full-season league as lab12_season.py (BDSM settings, persistent teams, weekly waivers, byes and injuries,
standings, 8-team playoffs), but the two tested seats now draft from their own source's board:

  fp        FantasyPros' PPR draft consensus (everyone else, and the control)
  model     the model's preseason projections (lab13_preseason.py), dropped into FantasyPros' positional slots:
            the k-th RB on the FantasyPros board is replaced by the model's k-th RB, and so on. Isolates "which
            player at each position" from "which position when".
  official  the same, ordering each position by the average of the model's and FantasyPros' ranks
  vor       the model's own board, ranked by projected points over replacement (QB12, RB30, WR36, TE12) - the
            model also decides which position to take when

Each seat's board gets the same per-player noise as everyone else's (sized by the experts' spread on the player).
Variants (tested seats): draft source x in-season source (lineups and waivers):
  model_draft / official_draft / vor_draft   with FantasyPros in-season (draft effect alone)
  model_full / official_full                 model / official draft and in-season
Control: the identical league with every seat on FantasyPros.

  python lab13_draft.py FRAME.parquet       (SIM_LEAGUES, default 1000)  -> lab/draft_sim_results.json
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

N_LEAGUES = int(os.environ.get("SIM_LEAGUES", "1000"))
REPL_TOTAL = {"QB": 12, "RB": 30, "WR": 36, "TE": 12}
VARIANTS = {                       # (draft board, in-season variant in lab12)
    "control": ("fp", "control"),
    "model_draft": ("model", "control"),
    "official_draft": ("official", "control"),
    "vor_draft": ("vor", "control"),
    "model_full": ("model", "model"),
    "official_full": ("official", "official"),
}
EXTRA_PER_POS = {"QB": 36, "RB": 80, "WR": 100, "TE": 36}


def boards(se, fp_board, pre):
    """Board scores (lower = earlier) per source for the candidate players (indexes into se)."""
    fb = fp_board[fp_board["player_id"].isin(se.ix)].head(300).copy()
    mp = pre.set_index("player_id")["model_total"]
    cand_ids = list(dict.fromkeys(list(fb["player_id"]) +
                                  [p for p in pre.sort_values("model_total", ascending=False)["player_id"] if p in se.ix]))
    cand = np.array([se.ix[p] for p in cand_ids])
    pos = np.array([G.POS[se.pos[i]] if se.pos[i] >= 0 else None for i in cand])
    keep = pos != None  # noqa: E711
    cand, pos, cand_ids = cand[keep], pos[keep], [c for c, k in zip(cand_ids, keep) if k]
    ecr = fb.set_index("player_id")["ecr"].reindex(cand_ids).values
    sd = np.maximum(fb.set_index("player_id")["sd"].reindex(cand_ids).fillna(3.0).values, 1.0)
    big = np.nanmax(ecr) + 1
    ecr_f = np.where(np.isnan(ecr), big + np.arange(len(ecr)), ecr)
    model = mp.reindex(cand_ids).fillna(0.0).values
    out = {"fp": ecr_f}
    for name in ["model", "official"]:
        sc = np.full(len(cand), np.inf)
        for p in G.POS:
            sel = np.where(pos == p)[0]
            slots = np.sort(ecr_f[sel])                          # FantasyPros' slots for this position
            mr = pd.Series(-model[sel]).rank(method="first").values
            if name == "model":
                key = mr
            else:
                er = pd.Series(ecr_f[sel]).rank(method="first").values
                key = mr + er + mr * 1e-6
            order = np.argsort(key, kind="stable")
            sc[sel[order]] = slots
        out[name] = sc
    vor = model.copy()
    for p in G.POS:
        sel = pos == p
        srt = np.sort(model[sel])[::-1]
        vor[sel] = model[sel] - srt[REPL_TOTAL[p] - 1]
    out["vor"] = pd.Series(-vor).rank(method="first").values
    return cand, sd, out


def draft(se, cand, sd, scores, src, z):
    """Snake draft; team t picks the best player on its board (score[src[t]] + its noise) that fits the roster rules."""
    taken = np.zeros(len(cand), bool)
    rosters = [[] for _ in range(G.TEAMS)]
    b = {t: scores[src[t]] + z[t] * sd * G.DRAFT_NOISE for t in range(G.TEAMS)}
    pos = se.pos[cand]
    for rd in range(G.ROUNDS):
        order = range(G.TEAMS) if rd % 2 == 0 else range(G.TEAMS - 1, -1, -1)
        for t in order:
            c = G.counts(rosters[t], se.pos)
            left = G.ROUNDS - rd
            need = [p for p in G.POS if c[G.PC[p]] < G.MIN[p]]
            must = sum(G.MIN[p] - c[G.PC[p]] for p in need) >= left
            okpos = np.array([(c[p] < G.CAP[G.POS[p]]) and (not must or G.POS[p] in need) for p in range(4)])
            ok = ~taken & okpos[pos]
            j = np.where(ok)[0][np.argmin(b[t][ok])]
            taken[j] = True
            rosters[t].append(int(cand[j]))
    return rosters


def main():
    t0 = time.time()
    frame = pd.read_parquet(sys.argv[1], columns=["season", "week", "season_type", "position", "expected",
                                                   "ecr_week_pos_rank", L.TARGET])
    rows = pd.read_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet"))
    cv = S11.curves(frame)
    d = S11.prepare(rows, cv)
    pre_all = pd.read_parquet(os.path.join(L.LAB_DIR, "preseason_proj.parquet"))
    games = pd.read_parquet("data/games.parquet", columns=["season", "week", "game_type", "gameday", "home_team",
                                                           "away_team"])
    ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], low_memory=False)
    recs, checks = [], {}
    for S in [int(s) for s in os.environ.get("SIM_SEASONS", "2023,2024,2025,2026").split(",")]:
        fpb, snap = G.draft_board(S, games, ids)
        pre = pre_all[pre_all["season"] == S]
        extra = pd.concat([g.nlargest(EXTRA_PER_POS[p], "model_total") for p, g in pre.groupby("pos")])
        einfo = extra.set_index("player_id")[["name", "pos"]].assign(team="")
        last_wk = int(d[d["season"] == S]["week"].max())
        se = G.Season(d, S, fpb, games, last_wk, extra=list(extra["player_id"]), extra_info=einfo)
        cand, sd, scores = boards(se, fpb, pre)
        checks[S] = {"draft_snapshot": snap, "candidates": int(len(cand)),
                     "model_top12": [se.name[cand[i]] for i in np.argsort(scores["vor"])[:12]]}
        rng = np.random.default_rng(3000 + S)
        for lg in range(N_LEAGUES):
            rb = np.random.default_rng(rng.integers(1 << 31))
            z = rb.normal(0, 1, (G.TEAMS, len(cand)))
            sched = G.schedule(rb)
            seats = list(rb.choice(G.TEAMS, 2, replace=False))
            for v, (dsrc, ins) in VARIANTS.items():
                src = ["fp"] * G.TEAMS
                for s in seats:
                    src[s] = dsrc
                rosters = draft(se, cand, sd, scores, src, z)
                r = G.play_season(se, cv[S - 1], rosters, sched, seats, ins)
                drafted_pts = [float(se.act[rosters[s]].sum()) for s in seats]
                for s, dp in zip(seats, drafted_pts):
                    recs.append({"season": S, "league": lg, "seat": int(s), "variant": v, "drafted_pts": dp,
                                 **{k: float(r[k][s]) for k in ["wins", "pf_week", "playoffs", "title", "moves",
                                                                 "pts_week_all"]}})
            if lg % 100 == 0:
                print(S, lg, round((time.time() - t0) / 60, 1), "min", flush=True)
    df = pd.DataFrame(recs)
    ctrl = df[df["variant"] == "control"].set_index(["season", "league", "seat"])
    summary = {}
    for scope, sub in [("2023-2025", df[df["season"] <= 2025])] + [(str(s), df[df["season"] == s])
                                                                   for s in sorted(df["season"].unique())]:
        o = {}
        for v in VARIANTS:
            if v == "control":
                continue
            a = sub[sub["variant"] == v].set_index(["season", "league", "seat"])
            c = ctrl.loc[a.index]
            r = {}
            for k in ["wins", "playoffs", "title", "pf_week", "pts_week_all", "drafted_pts", "moves"]:
                diff = (a[k] - c[k]).groupby(level=[0, 1]).mean()
                r[k] = [round(diff.mean(), 4), round(diff.std(ddof=1) / np.sqrt(len(diff)), 4)]
            r["by_season_wins"] = {int(s): round((a[k2] - c[k2]).xs(s, level=0).mean(), 3)
                                   for s in sorted(set(a.index.get_level_values(0))) for k2 in ["wins"]}
            o[v] = r
        summary[scope] = o
    json.dump({"leagues_per_season": N_LEAGUES, "checks": checks, "summary": summary},
              open(os.path.join(L.LAB_DIR, "draft_sim_results.json"), "w"), indent=1, default=str)
    print(json.dumps(summary["2023-2025"], indent=1))


if __name__ == "__main__":
    main()

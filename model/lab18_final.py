"""Nov 19, 2026: final scoring for the evolutionary tournament (lab18_evo.py), after all islands finish.

Collects every island's current top strategies (its last validation record) plus the fixed baselines, and scores each
on seasons it never trained on, with many more leagues than the islands' own checks:
  2025 (full season: wins, playoffs, title, last place) and 2026 Weeks 1-10 (wins and points)
Each candidate takes two seats in leagues of typical managers; the identical league with those seats typical is the
control (paired). Also: how each gene relates to the 2025 result across candidates, and the genes of the strategies
that held up on both 2025 and 2026.

  python lab18_final.py FRAME.parquet RESULTS_DIR ISLAND_DIR [ISLAND_DIR ...]   (FINAL_LEAGUES default 600)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import lab18_evo as E
import lab5 as L
import lab11_sim as S11

N_FINAL = int(os.environ.get("FINAL_LEAGUES", "400"))
N_ABLATE = int(os.environ.get("FINAL_ABLATE_LEAGUES", "300"))
# v3: where the consensus strategy's edge comes from - each group of genes set back to the typical manager's
ABLATE = {
    "draft": ["draft_model_w", "bias_QB", "bias_RB", "bias_WR", "bias_TE", "qb_min_round", "te_min_round", "k_min_round",
              "def_min_round"],
    "lineup sources": ["lw_model", "lw_fp", "lw_sleeper", "lw_espn"],
    "rest-of-season values": ["rw_model", "rw_fp", "rw_sleeper", "rw_espn"],
    "K/DEF projections": ["kd_lab", "kd_sleeper", "kd_espn"],
    "K/DEF streaming": ["stream_k", "stream_def", "stream_margin"],
    "waivers and FAAB": ["waiver_margin", "max_moves", "faab_aggr", "faab_reserve"],
    "trading (all)": ["trade_p", "trade_min_gain", "accept_thr", "pkg_p"],
}
TOP_PER_ISLAND = int(os.environ.get("FINAL_TOP", "3"))


BASE_CACHE = {}


def score(sd, strat, full, n, seed):
    """Every candidate plays the same leagues (same seeds), so the no-change control runs once per league."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["fit", "wins", "playoffs", "title", "last", "ppw", "trades", "moves"]}
    for _ in range(n):
        s = int(rng.integers(1 << 31))
        seats = list(np.random.default_rng(s).choice(E.TEAMS, 2, replace=False))
        if (sd.S, s) not in BASE_CACHE:
            BASE_CACHE[(sd.S, s)] = E.run_league(sd, [E.TYPICAL] * E.TEAMS, np.random.default_rng(s))
        r0 = BASE_CACHE[(sd.S, s)]
        r1 = E.run_league(sd, [strat if i in seats else E.TYPICAL for i in range(E.TEAMS)], np.random.default_rng(s))
        d = {k: [] for k in out}
        for st in seats:
            d["fit"].append(E.fitness(r1, st, full) - E.fitness(r0, st, full))
            for k in ["wins", "playoffs", "title", "last", "ppw", "trades", "moves"]:
                d[k].append(r1[k][st] - r0[k][st])
        for k in out:
            out[k].append(np.mean(d[k]))           # one observation per league (the two seats share it)
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) for k, v in out.items()}


def main():
    t0 = time.time()
    print("rules", json.dumps(E.rules()), flush=True)
    E.check_request()
    res_dir = sys.argv[2]
    os.makedirs(res_dir, exist_ok=True)
    islands = sys.argv[3:]
    cands = {}
    gens = {}
    mismatched = []
    for k, isl in enumerate(sorted(islands)):
        p = os.path.join(isl, "evo_validation.json")
        if not os.path.exists(p):
            continue
        v = json.load(open(p))[-1]
        if v.get("rules") != E.rules():       # the islands must have evolved under the rules the final scores with
            mismatched.append(os.path.basename(isl.rstrip("/")))
        gens[os.path.basename(isl.rstrip("/"))] = v["gen"]
        for name, genes in list(v["top_genes"].items())[:TOP_PER_ISLAND]:
            if genes in E.BASELINES.values() or genes == E.TYPICAL:
                continue
            cands[f"island {k + 1} {name}"] = genes
    if mismatched:      # don't spend hours scoring strategies evolved under other rules
        raise SystemExit(f"islands evolved under other rules than the final's {E.rules()}: {mismatched}")
    for name, genes in E.BASELINES.items():
        cands[name] = genes
    frame = pd.read_parquet(sys.argv[1], columns=["season", "week", "season_type", "position", "expected",
                                                   "ecr_week_pos_rank", L.TARGET])
    cv = S11.curves(frame)
    d = S11.prepare(pd.read_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet")), cv)
    games = pd.read_parquet("data/games.parquet", columns=["season", "week", "game_type", "gameday", "home_team",
                                                           "away_team"])
    ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], low_memory=False)
    pre = pd.read_parquet(os.path.join(L.LAB_DIR, "preseason_proj.parquet"))
    kdef = pd.read_parquet(os.path.join(L.LAB_DIR, "kdef_weekly.parquet"))
    other = E.load_other_kdef()
    sds = {S: E.SeasonData(S, d, cv, games, ids, pre, kdef, other) for S in [2025, 2026]}
    rows = []
    for i, (name, genes) in enumerate(cands.items()):
        r25 = score(sds[2025], genes, True, N_FINAL, 100)
        r26 = score(sds[2026], genes, False, N_FINAL // 2, 900)
        rows.append({"strategy": name, **{f"2025_{k}": v[0] for k, v in r25.items()},
                     **{f"2025_{k}_se": v[1] for k, v in r25.items()},
                     **{f"2026_{k}": v[0] for k, v in r26.items()}, **{f"2026_{k}_se": v[1] for k, v in r26.items()},
                     **{f"g_{k}": genes[k] for k in E.GENES}})
        print(name, round(r25["fit"][0], 3), round(r26["wins"][0], 3), round((time.time() - t0) / 60, 1), "min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(res_dir, "final_candidates.csv"), index=False)
    ev = df[~df["strategy"].isin(E.BASELINES)]
    gene_effect = {}
    for k in E.GENES:
        if ev[f"g_{k}"].nunique() > 2:
            gene_effect[k] = {"corr_with_2025_fit": round(float(ev[f"g_{k}"].corr(ev["2025_fit"], method="spearman")), 3),
                              "corr_with_2026_wins": round(float(ev[f"g_{k}"].corr(ev["2026_wins"], method="spearman")), 3)}
    both = ev[(ev["2025_fit"] > 2 * ev["2025_fit_se"]) & (ev["2026_wins"] > 0)].sort_values("2025_fit", ascending=False)
    # where the edge comes from: the consensus (median genes of every island's top strategies - not filtered on 2025 /
    # 2026, so it isn't picked on the seasons it's scored on) with one group of genes set back to the typical
    # manager's; the same leagues for every variant (2025 full season, 2026 Weeks 1-10)
    base = ev
    cons = {k: (int(round(float(base[f"g_{k}"].median()))) if E.GENES[k][2] == "i" else float(base[f"g_{k}"].median()))
            for k in E.GENES}
    variants = {"consensus": cons, "consensus, no trading": {**cons, "trade_p": 0.0}}
    for grp, keys in ABLATE.items():
        variants[f"consensus, {grp} -> typical"] = {**cons, **{k: E.TYPICAL[k] for k in keys}}
    ablation = {}
    for name, genes in variants.items():
        a25 = score(sds[2025], genes, True, N_ABLATE, 300)
        a26 = score(sds[2026], genes, False, N_ABLATE // 2, 1300)
        ablation[name] = {"2025_wins": round(a25["wins"][0], 3), "2025_wins_se": round(a25["wins"][1], 3),
                          "2025_playoffs": round(a25["playoffs"][0], 3), "2025_title": round(a25["title"][0], 3),
                          "2025_last": round(a25["last"][0], 3), "2025_trades": round(a25["trades"][0], 2),
                          "2026_wins_wk1_10": round(a26["wins"][0], 3), "2026_wins_se": round(a26["wins"][1], 3)}
        print("ablation", name, ablation[name], round((time.time() - t0) / 60, 1), "min", flush=True)
    top10 = ev.sort_values("2025_fit", ascending=False).head(10)
    summary = {"islands": gens, "rules": E.rules(), "islands_with_other_rules": mismatched,
               "note": "2026 = Weeks 1-10 only (wins out of 10); 2025 = full season (wins out of 14). Strategies picked by "
                       "2025 are in-sample on 2025: judge them by 2026. Gene correlations across candidates from few "
                       "islands are weak evidence; the ablation is the direct test.",
               "candidates": len(cands), "final_leagues": N_FINAL,
               "evolved_median": ev[["2025_wins", "2025_playoffs", "2025_title", "2025_last", "2025_trades", "2025_moves",
                                     "2026_wins", "2026_trades"]].median().round(4).to_dict(),
               "top10_by_2025_mean_2026_wins": round(float(top10["2026_wins"].mean()), 4),
               "consensus_genes": cons, "ablation": ablation,
               "baselines": df[df["strategy"].isin(E.BASELINES)].set_index("strategy")[
                   ["2025_fit", "2025_fit_se", "2025_wins", "2025_playoffs", "2025_title", "2025_last", "2026_wins",
                    "2026_wins_se", "2026_ppw"]].round(4).to_dict("index"),
               "top_by_2025": ev.sort_values("2025_fit", ascending=False).head(10)[
                   ["strategy", "2025_fit", "2025_fit_se", "2025_wins", "2025_playoffs", "2025_title", "2025_last",
                    "2026_wins", "2026_wins_se", "2026_ppw", "2025_trades", "2025_moves"]].round(4).to_dict("records"),
               "held_up_both": len(both),
               "median_genes_of_held_up": {k: round(float(both[f"g_{k}"].median()), 3) for k in E.GENES} if len(both) else {},
               "median_genes_all_evolved": {k: round(float(ev[f"g_{k}"].median()), 3) for k in E.GENES},
               "gene_effects": gene_effect}
    json.dump(summary, open(os.path.join(res_dir, "final_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()

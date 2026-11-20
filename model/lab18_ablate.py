"""Nov 19, 2026: where does the evolved strategy's edge come from? (ablation of the tournament winner)

The consensus strategy = the median genes of every island's top strategies (lab18_final.py, final_candidates.csv).
Each variant switches one group of genes back to the typical manager's and is scored the same way as the final step:
two seats in leagues of typical managers, paired with the identical league where those seats are typical; 2025 (full
season) and 2026 Weeks 1-10. Also the reverse: a typical manager with only one group switched to the consensus.

  python lab18_ablate.py FRAME.parquet final_candidates.csv OUT.json     (ABLATE_LEAGUES default 300)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import lab18_evo as E
import lab18_final as F
import lab5 as L
import lab11_sim as S11

N = int(os.environ.get("ABLATE_LEAGUES", "300"))
GROUPS = {
    "draft": ["draft_model_w", "bias_QB", "bias_RB", "bias_WR", "bias_TE", "qb_min_round", "te_min_round",
              "k_min_round", "def_min_round"],
    "lineup sources": ["lw_model", "lw_fp", "lw_sleeper", "lw_espn"],
    "rest-of-season values": ["rw_model", "rw_fp", "rw_sleeper", "rw_espn"],
    "K/DEF projections": ["kd_lab", "kd_sleeper", "kd_espn"],
    "K/DEF streaming": ["stream_k", "stream_def", "stream_margin"],
    "waivers": ["waiver_margin", "max_moves"],
    "trading (all)": ["trade_p", "trade_min_gain", "accept_thr"],
    "trade proposing": ["trade_p", "trade_min_gain"],
    "trade accepting": ["accept_thr"],
}


def main():
    cand = pd.read_csv(sys.argv[2])
    ev = cand[~cand["strategy"].isin(E.BASELINES)]
    cons = {}
    for k, (lo, hi, kind) in E.GENES.items():
        v = float(ev[f"g_{k}"].median())
        cons[k] = int(round(v)) if kind == "i" else v
    frame = pd.read_parquet(sys.argv[1], columns=["season", "week", "season_type", "position", "expected",
                                                   "ecr_week_pos_rank", L.TARGET])
    cv = S11.curves(frame)
    d = S11.prepare(pd.read_parquet(os.path.join(L.LAB_DIR, "sim_rows.parquet")), cv)
    games = pd.read_parquet("data/games.parquet", columns=["season", "week", "game_type", "gameday", "home_team",
                                                           "away_team"])
    ids = pd.read_csv("fpdata/db_playerids.csv", usecols=["fantasypros_id", "gsis_id"], low_memory=False)
    pre = pd.read_parquet(os.path.join(L.LAB_DIR, "preseason_proj.parquet"))
    kdef = pd.read_parquet(os.path.join(L.LAB_DIR, "kdef_weekly.parquet"))
    sds = {S: E.SeasonData(S, d, cv, games, ids, pre, kdef, E.load_other_kdef()) for S in [2025, 2026]}
    variants = {"consensus (all evolved genes)": cons}
    for g, keys in GROUPS.items():
        variants[f"consensus, {g} -> typical"] = {**cons, **{k: E.TYPICAL[k] for k in keys}}
    for g, keys in GROUPS.items():
        if g in ("trade proposing", "trade accepting"):
            continue
        variants[f"typical + consensus {g}"] = {**E.TYPICAL, **{k: cons[k] for k in keys}}
    out = {"consensus_genes": cons, "results": {}}
    for name, s in variants.items():
        r25 = F.score(sds[2025], s, True, N, 100)
        r26 = F.score(sds[2026], s, False, N // 2, 900)
        out["results"][name] = {"2025": {k: [round(v[0], 4), round(v[1], 4)] for k, v in r25.items()},
                                "2026": {k: [round(v[0], 4), round(v[1], 4)] for k, v in r26.items()}}
        print(f"{name:45s} 2025 wins {r25['wins'][0]:+.2f} ±{r25['wins'][1]:.2f}  playoffs {r25['playoffs'][0]:+.3f}  "
              f"title {r25['title'][0]:+.3f}  | 2026 wins {r26['wins'][0]:+.2f} ±{r26['wins'][1]:.2f}  "
              f"trades {r25['trades'][0]:.1f} moves {r25['moves'][0]:.0f}", flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

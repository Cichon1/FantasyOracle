"""
v8 candidate inputs (Oct 20, 2026): the "eye test" in numbers, from free in-season charting.

  eye_ftn   FTN charting (people watching every play; 2022+, published in-season):
            receivers  ey_created_rate       catches the throw didn't deserve, per target
                       ey_screen_tgt_rate    share of his targets that were screens
            QBs        ey_int_worthy_rate    interception-worthy throws per dropback
                       ey_qb_fault_sack_rate sacks charted as the QB's fault, per dropback
                       ey_throwaway_rate     throwaways per dropback
                       ey_out_of_pocket_rate dropbacks where he left the pocket
  eye_pfr   Pro Football Reference advanced stats (nflverse pfr_advstats, weekly, 2018+):
            runners    pf_ybc_att, pf_yac_att        yards before / after contact per carry
                       pf_brk_per_touch              broken tackles (rushing + receiving) per touch
            receivers  pf_drop_rate                  drops per target
            QBs        pf_bad_throw_rate             bad throws per attempt
                       pf_pressure_rate              times pressured per dropback

Every column is an exponential average (half-life 8 games) of the player's games strictly before the row's
game (af.roll + af.attach_prior), so nothing leaks.
"""
import glob
import os

import numpy as np
import pandas as pd

import advanced_features as af
import features_v4 as f4

HL = 8
FTN_COLS = ["ey_created_rate", "ey_screen_tgt_rate", "ey_int_worthy_rate", "ey_qb_fault_sack_rate",
            "ey_throwaway_rate", "ey_out_of_pocket_rate"]
PFR_COLS = ["pf_ybc_att", "pf_yac_att", "pf_brk_per_touch", "pf_drop_rate", "pf_bad_throw_rate", "pf_pressure_rate"]


def _ftn_plays(data_dir):
    p = f4.load_pbp(data_dir, ["game_id", "play_id", "season", "week", "season_type", "qb_dropback", "qb_spike",
                               "two_point_attempt", "receiver_player_id", "passer_player_id", "complete_pass", "sack"])
    ftn = pd.concat([pd.read_parquet(f, columns=["nflverse_game_id", "nflverse_play_id", "is_created_reception",
                                                  "is_screen_pass", "is_interception_worthy", "is_qb_fault_sack",
                                                  "is_throw_away", "is_qb_out_of_pocket"])
                     for f in sorted(glob.glob(os.path.join(data_dir, "ftn", "*.parquet")))], ignore_index=True)
    ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    ftn["play_id"] = pd.to_numeric(ftn["play_id"], errors="coerce")
    ftn = ftn.dropna(subset=["play_id"])
    ftn["play_id"] = ftn["play_id"].astype(int)
    d = p.merge(ftn.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="inner")
    return d[(d["qb_dropback"] == 1) & (d["qb_spike"] != 1) & (d["two_point_attempt"] != 1)]


def eye_ftn(data_dir, target):
    d = _ftn_plays(data_dir)
    b = lambda s: s.fillna(False).astype(bool).astype(float)
    # receivers
    r = d[d["receiver_player_id"].notna()].copy()
    r["tgt"] = 1.0
    r["created"] = b(r["is_created_reception"])
    r["screen"] = b(r["is_screen_pass"])
    rg = r.groupby(["receiver_player_id", "game_id", "season", "week"], as_index=False)[["tgt", "created", "screen"]].sum()
    rg = rg.rename(columns={"receiver_player_id": "player_id"})
    rg = af.roll(rg, "player_id", ["season", "week"], {"ey_created_rate": ("created", "tgt"),
                                                       "ey_screen_tgt_rate": ("screen", "tgt")}, HL)
    # QBs
    q = d[d["passer_player_id"].notna()].copy()
    q["db"] = 1.0
    for src, dst in [("is_interception_worthy", "intw"), ("is_qb_fault_sack", "qfs"), ("is_throw_away", "taw"),
                     ("is_qb_out_of_pocket", "oop")]:
        q[dst] = b(q[src])
    qg = q.groupby(["passer_player_id", "game_id", "season", "week"], as_index=False)[["db", "intw", "qfs", "taw", "oop"]].sum()
    qg = qg.rename(columns={"passer_player_id": "player_id"})
    qg = af.roll(qg, "player_id", ["season", "week"], {"ey_int_worthy_rate": ("intw", "db"),
                                                       "ey_qb_fault_sack_rate": ("qfs", "db"),
                                                       "ey_throwaway_rate": ("taw", "db"),
                                                       "ey_out_of_pocket_rate": ("oop", "db")}, HL)
    out = af.attach_prior(target, rg, "player_id", "player_id", ["ey_created_rate", "ey_screen_tgt_rate"])
    out = out.join(af.attach_prior(target, qg, "player_id", "player_id",
                                   ["ey_int_worthy_rate", "ey_qb_fault_sack_rate", "ey_throwaway_rate",
                                    "ey_out_of_pocket_rate"]))
    return out


def _pfr(data_dir, kind):
    fs = sorted(glob.glob(os.path.join(data_dir, "pfr", f"advstats_week_{kind}_*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def eye_pfr(data_dir, target, stats, players):
    ids = players[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    sack_col = next((c for c in ("sacks_suffered", "sacks") if c in stats.columns), None)
    st = stats[["player_id", "game_id", "season", "week", "carries", "receptions", "targets", "attempts"]
               + ([sack_col] if sack_col else [])].copy()
    st["sacks_suffered"] = st[sack_col] if sack_col else 0.0
    st = st.drop(columns=[c for c in ["sacks"] if c in st.columns and c != "sacks_suffered"])
    st = st.drop_duplicates(["player_id", "game_id"])
    out = pd.DataFrame(index=target.index)

    def prep(d, cols):
        d = d[d["game_type"].isin(["REG", "WC", "DIV", "CON", "SB", "POST"]) | d["game_type"].isna()].copy()
        d["player_id"] = d["pfr_player_id"].map(ids)
        d = d.dropna(subset=["player_id"])
        d = d.groupby(["player_id", "game_id", "season", "week"], as_index=False)[cols].sum(min_count=1)
        return d.merge(st, on=["player_id", "game_id", "season", "week"], how="left")

    ru = _pfr(data_dir, "rush")
    if ru is not None:
        g = prep(ru, ["rushing_yards_before_contact", "rushing_yards_after_contact", "rushing_broken_tackles",
                      "receiving_broken_tackles"])
        g["att"] = g["carries"].fillna(0)
        g["touch"] = g["att"] + g["receptions"].fillna(0)
        g["brk"] = g[["rushing_broken_tackles", "receiving_broken_tackles"]].fillna(0).sum(axis=1)
        g = af.roll(g, "player_id", ["season", "week"], {"pf_ybc_att": ("rushing_yards_before_contact", "att"),
                                                         "pf_yac_att": ("rushing_yards_after_contact", "att"),
                                                         "pf_brk_per_touch": ("brk", "touch")}, HL)
        out = out.join(af.attach_prior(target, g, "player_id", "player_id", ["pf_ybc_att", "pf_yac_att",
                                                                             "pf_brk_per_touch"]))
    rc = _pfr(data_dir, "rec")
    if rc is not None:
        g = prep(rc, ["receiving_drop"])
        g["tgt"] = g["targets"]
        g = af.roll(g, "player_id", ["season", "week"], {"pf_drop_rate": ("receiving_drop", "tgt")}, HL)
        out = out.join(af.attach_prior(target, g, "player_id", "player_id", ["pf_drop_rate"]))
    pa = _pfr(data_dir, "pass")
    if pa is not None:
        g = prep(pa, ["passing_bad_throws", "times_pressured"])
        g["att"] = g["attempts"]
        g["db"] = g["attempts"].fillna(0) + g["sacks_suffered"].fillna(0)
        g = af.roll(g, "player_id", ["season", "week"], {"pf_bad_throw_rate": ("passing_bad_throws", "att"),
                                                         "pf_pressure_rate": ("times_pressured", "db")}, HL)
        out = out.join(af.attach_prior(target, g, "player_id", "player_id", ["pf_bad_throw_rate", "pf_pressure_rate"]))
    for c in PFR_COLS:
        if c not in out:
            out[c] = np.nan
    return out[PFR_COLS]

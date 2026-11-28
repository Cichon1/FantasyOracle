"""Past draft results for BDSM League and The League, every season Yahoo has (follows each league's "renew" chain
back through earlier seasons). Used by the draft test's leaguemate model (automation/draft_test/).

Per season and league it saves automation/history/draft/league_drafts/<league>_<season>.csv:
  season, round, pick, team_key, manager_id (stable across seasons: hashed Yahoo manager guid), team_name,
  player_key, player, pos, nfl_team, cost (auction), is_keeper (when Yahoo marks it), final_rank
and a summary in league_drafts/summary.json. Manager names / guids are not stored.

  python automation/yahoo_drafts.py          (needs YAHOO_CLIENT_ID / YAHOO_CLIENT_SECRET, as yahoo_sync.py)
"""
import hashlib
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yahoo_sync as Y  # noqa: E402

OUT = os.path.join(Y.ROOT, "automation", "history", "draft", "league_drafts")
LEAGUES = {"126912": "bdsm", "200436": "league"}


def mid(guid):
    return "m" + hashlib.sha256(("fo-draft:" + guid).encode()).hexdigest()[:10]


def safe(tok, path):
    try:
        return Y.get(tok, path), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:300]


def one_season(tok, lk, name, summ):
    meta, err = safe(tok, f"league/{lk}/metadata")
    if meta is None:
        summ["errors"].append({"league": lk, "step": "metadata", "error": err})
        return None
    season = Y.t(meta, ".//season")
    renew = Y.t(meta, ".//renew")
    info = {"league_key": lk, "season": season, "name": Y.t(meta, ".//name"),
            "num_teams": Y.t(meta, ".//num_teams"), "draft_status": Y.t(meta, ".//draft_status"),
            "renew": renew}
    st, _ = safe(tok, f"league/{lk}/settings")
    if st is not None:
        info.update({k: Y.t(st, f".//{k}") for k in ("draft_type", "is_auction_draft", "uses_keeper",
                                                       "num_keepers", "draft_pick_time", "scoring_type")})
    # teams, managers, final rank
    teams = {}
    tm, err = safe(tok, f"league/{lk}/standings")
    if tm is None:
        tm, err = safe(tok, f"league/{lk}/teams")
    if tm is not None:
        for x in tm.findall(".//team"):
            g = Y.t(x, ".//managers/manager/guid") or Y.t(x, ".//managers/manager/manager_id")
            teams[Y.t(x, "team_key")] = {"manager_id": mid(g) if g else "", "team_name": Y.t(x, "name"),
                                         "final_rank": Y.t(x, ".//team_standings/rank")}
    else:
        summ["errors"].append({"league": lk, "step": "teams", "error": err})
    dr, err = safe(tok, f"league/{lk}/draftresults")
    if dr is None:
        summ["errors"].append({"league": lk, "step": "draftresults", "error": err})
        return info
    picks = []
    for d in dr.findall(".//draft_result"):
        picks.append({"round": Y.t(d, "round"), "pick": Y.t(d, "pick"), "team_key": Y.t(d, "team_key"),
                      "player_key": Y.t(d, "player_key"), "cost": Y.t(d, "cost")})
    keys = [p["player_key"] for p in picks if p["player_key"]]
    pl = {}
    for i in range(0, len(keys), 25):
        x, err = safe(tok, f"league/{lk}/players;player_keys={','.join(keys[i:i + 25])}")
        if x is None:
            summ["errors"].append({"league": lk, "step": f"players {i}", "error": err})
            continue
        for p in x.findall(".//player"):
            pl[Y.t(p, "player_key")] = {"player": Y.t(p, "name/full"), "pos": Y.t(p, "display_position"),
                                        "nfl_team": Y.t(p, "editorial_team_abbr"),
                                        "yahoo_is_keeper": Y.t(p, ".//is_keeper/kept") or Y.t(p, ".//is_keeper")}
        time.sleep(0.3)
    kept = set()
    k, _ = safe(tok, f"league/{lk}/players;status=K")
    if k is not None:
        kept = {Y.t(p, "player_key") for p in k.findall(".//player")}
    rows = []
    for p in picks:
        r = {"season": season, **p, **teams.get(p["team_key"], {}), **pl.get(p["player_key"], {})}
        r["is_keeper"] = int(p["player_key"] in kept or str(r.get("yahoo_is_keeper", "")).lower() in ("1", "true"))
        rows.append(r)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{name}_{season}.csv"), index=False)
    info["picks"] = len(rows)
    info["players_named"] = sum(1 for r in rows if r.get("player"))
    info["keepers_marked"] = sum(r["is_keeper"] for r in rows)
    return info


def main():
    if not (os.path.exists(Y.TOKEN) and os.environ.get("YAHOO_CLIENT_SECRET")):
        print("not connected")
        return
    tok = Y.access_token()
    gk = Y.t(Y.get(tok, "game/nfl"), ".//game_key")
    summ = {"seasons": [], "errors": []}
    for lid, name in LEAGUES.items():
        lk, seen = f"{gk}.l.{lid}", set()
        while lk and lk not in seen and len(seen) < 25:
            seen.add(lk)
            info = one_season(tok, lk, name, summ)
            if not info:
                break
            info["league"] = name
            summ["seasons"].append(info)
            print(json.dumps(info), flush=True)
            r = info.get("renew", "")
            lk = r.replace("_", ".l.") if r and "_" in r else None
    os.makedirs(OUT, exist_ok=True)
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

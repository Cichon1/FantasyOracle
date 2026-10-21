"""One-off check of the FantasyPros API with the repository secret FANTASYPROS_API_KEY.

Writes automation/state/fp_probe.json with the SHAPE of each response only (status, field names, counts, limits, one
sample player name) - no rankings - because this repository and its logs are public.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.fantasypros.com/public/v2/json/nfl/2026"
KEY = os.environ.get("FANTASYPROS_API_KEY", "")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "fp_probe.json")


def call(path, **q):
    url = f"{BASE}/{path}?{urllib.parse.urlencode(q)}"
    req = urllib.request.Request(url, headers={"x-api-key": KEY, "User-Agent": "FantasyOracle/1.0"})
    rec = {"request": f"{path}?{urllib.parse.urlencode(q)}"}
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            rec["status"] = r.status
            rec["limit_headers"] = {k: v for k, v in r.headers.items()
                                    if any(t in k.lower() for t in ("limit", "rate", "quota", "remaining", "tier"))}
            body = json.load(r)
    except urllib.error.HTTPError as e:
        rec["status"] = e.code
        rec["error"] = e.read().decode("utf-8", "ignore")[:300]
        return rec, None
    except Exception as e:
        rec["error"] = str(e)[:300]
        return rec, None
    rec["top_keys"] = sorted(body.keys()) if isinstance(body, dict) else type(body).__name__
    for k in ("count", "total_experts", "filters", "last_updated", "week", "type", "scoring", "position_id",
              "ranking_type_name", "tier", "limit", "truncated", "accuracy_weekly_season"):
        if isinstance(body, dict) and k in body:
            rec[k] = body[k]
    pl = body.get("players") if isinstance(body, dict) else None
    if isinstance(pl, list):
        rec["players_returned"] = len(pl)
        if pl:
            rec["player_fields"] = sorted(pl[0].keys())
            rec["first_player"] = [pl[0].get("player_name"), pl[0].get("rank_ecr"), pl[0].get("pos_rank")]
            rec["last_player"] = [pl[-1].get("player_name"), pl[-1].get("rank_ecr"), pl[-1].get("pos_rank")]
            ex = pl[0].get("experts")
            rec["player_experts_field"] = (f"{type(ex).__name__} with {len(ex)} entries" if ex is not None else None)
            rec["positions_seen"] = sorted({p.get("player_position_id") for p in pl if p.get("player_position_id")})
    exs = body.get("experts") if isinstance(body, dict) else None
    if isinstance(exs, list):
        rec["experts_returned"] = len(exs)
        if exs:
            rec["expert_fields"] = sorted(exs[0].keys()) if isinstance(exs[0], dict) else type(exs[0]).__name__
    for k in ("expert_pub", "expert_names", "experts_available"):
        v = body.get(k) if isinstance(body, dict) else None
        if v is not None:
            rec[k + "_shape"] = (f"dict with {len(v)} keys" if isinstance(v, dict) else
                                 f"{type(v).__name__} len {len(v) if hasattr(v, '__len__') else '?'}")
            if isinstance(v, dict) and v:
                k0 = next(iter(v)); rec[k + "_sample"] = {k0: v[k0] if not isinstance(v[k0], (list, dict)) else str(type(v[k0]))}
    return rec, body


def main():
    report = {"key_present": bool(KEY)}
    if not KEY:
        report["note"] = "no FANTASYPROS_API_KEY secret"
    else:
        r, ros = call("consensus-rankings", position="ALL", type="ROS", scoring="PPR", experts="show")
        report["ros_all_show"] = r
        report["ros_all_available"], _ = call("consensus-rankings", position="ALL", type="ROS", scoring="PPR",
                                              experts="available")
        report["weekly_rb"], _ = call("consensus-rankings", position="RB", type="WEEKLY", scoring="PPR", week=7,
                                      experts="show")
        report["weekly_all"], _ = call("consensus-rankings", position="ALL", type="WEEKLY", scoring="PPR", week=7)
        report["weekly_qb_std"], _ = call("consensus-rankings", position="QB", type="WEEKLY", scoring="STD", week=7)
        report["experts_ros"], _ = call("rankings/experts", type="ROS", scoring="PPR", include_overall="true")
        # does an expert filter work? use the first three experts named in the ROS consensus
        ids = []
        if ros:
            for k in ("expert_pub", "expert_names"):
                if isinstance(ros.get(k), dict):
                    ids = list(ros[k].keys())[:3]
                    break
            if not ids and isinstance(ros.get("experts_available"), dict):
                ids = [str(x) for x in ros["experts_available"].get("included", [])[:3]]
        report["filter_ids_tried"] = len(ids)
        if ids:
            report["ros_filtered_3"], _ = call("consensus-rankings", position="ALL", type="ROS", scoring="PPR",
                                               filters=":".join(ids), experts="show")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1, default=str)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v) for k, v in report.items()}))


if __name__ == "__main__":
    main()

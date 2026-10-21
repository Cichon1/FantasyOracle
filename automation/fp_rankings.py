"""FantasyPros expert rankings, pulled from their public rankings pages a few times a day (11 am, 2 pm, 5 pm, 8 pm Central).

Why: the model's expert-rank inputs used to come only from DynastyProcess's weekly copy (Fridays). This keeps them a few
hours old instead of up to a week old, and saves filtered versions so we can test whether a "recent experts only"
consensus predicts better than FantasyPros' default panel.

Each pull reads:
  - FantasyPros' consensus pages: weekly QB / RB / WR / TE and rest-of-season QB / RB / WR / TE (8 pages). Each page
    carries the full consensus (rank_ave, sd, best, worst for every ranked player) and the list of experts who rank that
    position, with when each last updated and their in-season accuracy rank.
  - Individual expert pages (public ones only), but only for experts who updated since we last read them - so most
    pulls fetch a handful. Each page holds that expert's full ranking (all positions).

Three versions per ranking type (weekly, rest of season) and position:
  default        FantasyPros' own consensus (their default expert panel)
  recent         average of the experts (with public pages) who updated after the cutoff below
  recent_acc     the same, limited to the 20 most accurate of them (FantasyPros in-season accuracy rank at the position)

Cutoffs (US Eastern):
  rest of season  updated after the last full slate ended (Monday night's game; on Monday, Sunday night's game).
                  Fewer than 8 experts -> last 7 days.
  weekly          Mon-Fri: same as rest of season. Sat-Sun: after Friday 4 pm ET (final injury report).
                  Fewer than 3 experts -> keep the last good version.

The model uses `default` for now (it is what it was trained on); all three are saved for scoring after a few weeks.

Rankings are FantasyPros' data and this repository is public, so everything with rankings in it is stored encrypted
(key derived from the FANTASYPROS_API_KEY secret): automation/state/fp_store.enc and automation/history/fp/*.enc.
automation/state/fp_status.json (counts and times only) is public and feeds the site's "experts used" note.

  python automation/fp_rankings.py           # pulls only at the scheduled Central hours
  python automation/fp_rankings.py --force   # pull now
"""
import argparse
import base64
import datetime as dt
import gzip
import hashlib
import html as H
import io
import json
import os
import re
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state")
HIST = os.path.join(HERE, "history", "fp")
STORE = os.path.join(STATE, "fp_store.enc")
STATUS = os.path.join(STATE, "fp_status.json")
SITE_STATUS = os.path.join(HERE, "..", "docs", "data", "fp_status.json")
BASE = "https://www.fantasypros.com"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.9"}
ET, CT = ZoneInfo("America/New_York"), ZoneInfo("America/Chicago")
HOURS_CT = {11, 14, 17, 20}
POS = ["QB", "RB", "WR", "TE"]
PAGES = {  # (type, pos) -> consensus page (same pages the DynastyProcess copy uses)
    ("weekly", "QB"): "/nfl/rankings/qb.php", ("weekly", "RB"): "/nfl/rankings/ppr-rb.php",
    ("weekly", "WR"): "/nfl/rankings/ppr-wr.php", ("weekly", "TE"): "/nfl/rankings/ppr-te.php",
    ("ros", "QB"): "/nfl/rankings/ros-qb.php", ("ros", "RB"): "/nfl/rankings/ros-ppr-rb.php",
    ("ros", "WR"): "/nfl/rankings/ros-ppr-wr.php", ("ros", "TE"): "/nfl/rankings/ros-ppr-te.php",
}
MIN_PANEL = {"ros": 8, "weekly": 3}
TOP_ACC = 20
MAX_EXPERT_PAGES = 45      # per pull; the rest wait for the next pull
PAUSE = 1.0                # seconds between page loads
KEEP_DAYS = 14


# ------------------------------------------------------------------------------------------- storage
def _fernet():
    from cryptography.fernet import Fernet
    k = os.environ.get("FANTASYPROS_API_KEY", "")
    if not k:
        raise SystemExit("FANTASYPROS_API_KEY is not set (needed to encrypt the saved rankings)")
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(("fantasy-oracle-fp:" + k).encode()).digest()))


def save_enc(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = gzip.compress(json.dumps(obj, separators=(",", ":")).encode())
    open(path, "wb").write(_fernet().encrypt(raw))


def load_enc(path, default=None):
    if not os.path.exists(path):
        return default
    return json.loads(gzip.decompress(_fernet().decrypt(open(path, "rb").read())))


# ------------------------------------------------------------------------------------------- fetching
def fetch(path):
    req = urllib.request.Request(BASE + path, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def js_var(page, name):
    m = re.search(r"\b" + name + r"\s*=\s*(\{.*?\});\s*\n", page, re.S) or re.search(r"\b" + name + r"\s*=\s*(\{.*?\});", page, re.S)
    return json.loads(m.group(1)) if m else None


def consensus_page(path):
    pg = fetch(path)
    e, g = js_var(pg, "ecrData"), js_var(pg, "expertGroupsData")
    if not e or not e.get("players"):
        raise RuntimeError(f"no rankings found on {path}")
    players = [[str(p["player_id"]), p.get("player_name"), p.get("player_team_id"), p.get("player_position_id"),
                _f(p.get("rank_ave")), _f(p.get("rank_std")), _f(p.get("rank_min")), _f(p.get("rank_max")),
                p.get("rank_ecr"), p.get("player_filename")] for p in e["players"]]
    experts = []
    for x in (g or {}).get("expert_data", []):
        experts.append({"id": int(x["id"]), "name": x.get("name"), "site": x.get("site"),
                        "updated": int(x.get("last_updated") or 0), "link": x.get("rankings_link") or "",
                        "acc": _i(x.get("in_season_pos_rank")) or _i(x.get("in_season_rank"))})
    return {"players": players, "experts": experts, "total_experts": e.get("total_experts"),
            "panel": (e.get("experts_available") or {}).get("included", [])}


def expert_page(link, kind):
    """One expert's full ranking (position=ALL) -> [[fp_id, pos, overall_rank], ...]."""
    p = link.replace(BASE, "")
    p = re.sub(r"position=[A-Za-z]+", "position=ALL", p) if "position=" in p else p + "&position=ALL"
    rows = _table(fetch(p))
    if kind == "weekly" and not any(r[1] == "QB" for r in rows):   # some weekly 'ALL' lists are flex only
        time.sleep(PAUSE)
        q = re.sub(r"position=ALL", "position=QB", p).replace("scoring=PPR", "scoring=STD")
        rows += [[i, "QB", 1000 + r] for i, _, r in _table(fetch(q)) if True]
    return rows


def _table(page):
    out = []
    for tr in re.findall(r"<tr>(.*?)</tr>", page, re.S):
        m = re.search(r"fp-id-(\d+)", tr)
        if not m:
            continue
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        pos = next((H.unescape(c) for c in cells[2:5] if re.fullmatch(r"QB|RB|WR|TE|K|DST|DEF", c)), None)
        if pos in POS:
            out.append([m.group(1), pos, int(cells[0])])
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------- cutoffs
def schedule(season):
    import pandas as pd
    url = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
    p = os.path.join(HERE, "..", "model", "data", "games.parquet")
    g = pd.read_parquet(p) if os.path.exists(p) else pd.read_parquet(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
    g = g[(g["season"] == season) & g["gametime"].notna()].copy()
    g["ko"] = pd.to_datetime(g["gameday"] + " " + g["gametime"]).dt.tz_localize(ET)
    g["end"] = g["ko"] + pd.Timedelta(hours=3, minutes=20)
    g["wd"] = g["ko"].dt.weekday
    return g


def cutoffs(now, season):
    """Rest-of-season and weekly cutoffs (US Eastern datetimes)."""
    import pandas as pd
    g = schedule(season)
    now_ts = pd.Timestamp(now)
    done = g[g["end"] <= now_ts]
    anchor = now_ts - pd.Timedelta(days=7)
    if len(done):
        complete = [w for w, x in g.groupby("week") if (x["end"] <= now_ts).all()]
        if complete:
            anchor = g[g["week"] == max(complete)]["end"].max()
        wk = done["week"].max()                 # week under way: Sunday's games all over -> Sunday night is the anchor
        cur = g[g["week"] == wk]
        sun = cur[cur["wd"] == 6]
        if len(sun) and (sun["end"] <= now_ts).all() and not (cur["end"] <= now_ts).all():
            anchor = max(anchor, sun["end"].max())
    ros = anchor
    weekly = anchor
    if now.weekday() in (5, 6):                 # Sat / Sun: after Friday's final injury report (4 pm ET)
        fri = (now - dt.timedelta(days=(now.weekday() - 4))).replace(hour=16, minute=0, second=0, microsecond=0)
        weekly = max(anchor, pd.Timestamp(fri))
    return ros.to_pydatetime(), weekly.to_pydatetime()


# ------------------------------------------------------------------------------------------- consensus
def build(experts, cache, kind, pos, cutoff, top=None):
    """Average position rank over the eligible experts' cached lists. -> (rows, panel ids)."""
    ts = cutoff.timestamp()
    elig = [x for x in experts if x["updated"] >= ts and str(x["id"]) in cache
            and cache[str(x["id"])].get("updated", 0) >= x["updated"]]
    if top:
        elig = sorted(elig, key=lambda x: (x["acc"] is None, x["acc"] or 9999))[:top]
    ranks = {}
    for x in elig:
        rows = [r for r in cache[str(x["id"])]["rows"] if r[1] == pos]
        rows.sort(key=lambda r: r[2])
        for k, r in enumerate(rows, 1):
            ranks.setdefault(r[0], []).append(k)
    n = len(elig)
    out = []
    for pid, rk in ranks.items():
        if n and len(rk) * 3 < n:             # ranked by fewer than a third of the panel: too thin to average
            continue
        m = sum(rk) / len(rk)
        sd = (sum((v - m) ** 2 for v in rk) / len(rk)) ** 0.5
        out.append([pid, round(m, 2), round(sd, 2), min(rk), max(rk), len(rk)])
    out.sort(key=lambda r: r[1])
    return out, [x["id"] for x in elig]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    now = dt.datetime.now(ET)
    if not a.force and now.astimezone(CT).hour not in HOURS_CT:
        print(f"not a pull hour ({now.astimezone(CT):%H:%M} CT)")
        return
    season = now.year if now.month >= 3 else now.year - 1
    store = load_enc(STORE, {}) or {}
    cache = store.setdefault("cache", {"weekly": {}, "ros": {}})
    status = {"pulled_at": now.isoformat(timespec="minutes"), "pages": 0, "expert_pages": 0, "errors": []}

    # 1) consensus pages
    pages = {}
    for (kind, pos), path in PAGES.items():
        try:
            pages[(kind, pos)] = consensus_page(path)
        except Exception as e:
            status["errors"].append(f"{path}: {str(e)[:120]}")
        status["pages"] += 1
        time.sleep(PAUSE)
    if not pages:
        raise SystemExit("no FantasyPros pages could be read: " + "; ".join(status["errors"]))

    # 2) expert pages that changed since we last read them
    want = {}
    for (kind, pos), pg in pages.items():
        for x in pg["experts"]:
            if not x["link"]:
                continue
            c = cache[kind].get(str(x["id"]))
            if c is None or c.get("updated", 0) < x["updated"]:
                w = want.setdefault((kind, x["id"]), dict(x, kind=kind))
                w["updated"] = max(w["updated"], x["updated"])
    todo = sorted(want.values(), key=lambda x: -x["updated"])[:MAX_EXPERT_PAGES]
    for x in todo:
        try:
            rows = expert_page(x["link"], x["kind"])
            if rows:
                cache[x["kind"]][str(x["id"])] = {"updated": x["updated"], "rows": rows}
        except Exception as e:
            status["errors"].append(f"expert {x['id']}: {str(e)[:80]}")
        status["expert_pages"] += 1
        time.sleep(PAUSE)
    status["expert_pages_waiting"] = max(0, len(want) - len(todo))

    # 3) the three versions
    ros_cut, wk_cut = cutoffs(now, season)
    status["cutoff_ros"], status["cutoff_weekly"] = ros_cut.isoformat(timespec="minutes"), wk_cut.isoformat(timespec="minutes")
    latest = {"pulled_at": status["pulled_at"], "default": {}, "recent": {}, "recent_acc": {}}
    last_good = store.setdefault("last_good", {})
    panels = {}
    for (kind, pos), pg in pages.items():
        key = f"{kind}|{pos}"
        latest["default"][key] = pg["players"]
        cut = ros_cut if kind == "ros" else wk_cut
        rows, ids = build(pg["experts"], cache[kind], kind, pos, cut)
        note = "since cutoff"
        if len(ids) < MIN_PANEL[kind]:
            if kind == "ros":
                rows, ids = build(pg["experts"], cache[kind], kind, pos, now - dt.timedelta(days=7))
                note = "fallback: last 7 days"
            elif key in last_good:
                rows, ids, note = last_good[key]["rows"], last_good[key]["ids"], "kept last good version"
        if kind == "weekly" and len(ids) >= MIN_PANEL[kind] and note == "since cutoff":
            last_good[key] = {"rows": rows, "ids": ids}
        latest["recent"][key] = rows
        rows_a, ids_a = build(pg["experts"], cache[kind], kind, pos, cut, top=TOP_ACC)
        if len(ids_a) < MIN_PANEL[kind]:
            rows_a, ids_a = rows, ids
        latest["recent_acc"][key] = rows_a
        panels[key] = {"default": pg["total_experts"], "recent": len(ids), "recent_acc": len(ids_a),
                       "listed": len(pg["experts"]), "public_pages": sum(1 for x in pg["experts"] if x["link"]),
                       "recent_rule": note}
    status["panels"] = panels

    # 4) save: latest, last pull of each day (what the model reads), history
    day = now.date().isoformat()
    daily = store.setdefault("daily", {})
    daily[day] = {"pulled_at": status["pulled_at"], "default": latest["default"]}
    for d in sorted(daily)[:-KEEP_DAYS]:
        daily.pop(d)
    store["latest"] = latest
    save_enc(STORE, store)
    save_enc(os.path.join(HIST, f"fp_{now:%Y-%m-%d_%H%M}.enc"), latest)
    status["stamp"] = hashlib.sha1(json.dumps(latest["default"], sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(STATE, exist_ok=True)
    json.dump(status, open(STATUS, "w"), indent=1)
    site = {"pulled_at": status["pulled_at"], "cutoff_ros": status["cutoff_ros"], "cutoff_weekly": status["cutoff_weekly"],
            "experts": {k: {"default": v["default"], "recent": v["recent"]} for k, v in panels.items()}}
    json.dump(site, open(SITE_STATUS, "w"), indent=1)
    print(json.dumps({k: status[k] for k in ("pulled_at", "pages", "expert_pages", "expert_pages_waiting", "stamp")}
                     | {"errors": len(status["errors"])}))


if __name__ == "__main__":
    main()

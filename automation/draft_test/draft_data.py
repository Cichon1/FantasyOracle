"""Phase 1 of the draft strategy test (see PREREG.md): pull the historical draft-season data, one unit at a time.

Every unit (source x year x format) is saved to its own file as soon as it's fetched (temp file + rename), and a unit
that's already saved is skipped, so a rerun resumes where a failed or timed-out run stopped. Licensed pages
(FantasyPros, Spotrac) are stored encrypted with the same key as the FantasyPros rankings (FANTASYPROS_API_KEY secret),
because the repository is public. FFC data (free with attribution: fantasyfootballcalculator.com) and nflverse data are
stored as is.

  python automation/draft_test/draft_data.py ffc          # Fantasy Football Calculator ADP API, 2010-2025
  python automation/draft_test/draft_data.py fp_adp       # FantasyPros ADP pages (ESPN / Yahoo / Sleeper / CBS ...), 2017-2025
  python automation/draft_test/draft_data.py nflverse     # draft picks, trades, PFR advanced stats
  python automation/draft_test/draft_data.py spotrac      # Spotrac incentive pages by year and team (2019-2026, whatever exists)
  python automation/draft_test/draft_data.py status       # what's saved

Output: automation/history/draft/<source>/...  and automation/history/draft/status.json
"""
import base64
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "automation", "history", "draft")
UA = {"User-Agent": "Mozilla/5.0 (FantasyOracle research; github.com/Cichon1/FantasyOracle)"}
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
TEAMS = ["ari", "atl", "bal", "buf", "car", "chi", "cin", "cle", "dal", "den", "det", "gb", "hou", "ind", "jax", "kc",
         "lv", "lac", "lar", "mia", "min", "ne", "no", "nyg", "nyj", "phi", "pit", "sf", "sea", "tb", "ten", "was"]


def _fernet():
    from cryptography.fernet import Fernet
    k = os.environ.get("FANTASYPROS_API_KEY", "")
    if not k:
        raise SystemExit("FANTASYPROS_API_KEY not set: licensed pages are only saved encrypted")
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(("fantasy-oracle-fp:" + k).encode()).digest()))


def save(path, data, enc=False):
    """Atomic write: a crash never leaves half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if enc:
        data = _fernet().encrypt(gzip.compress(data))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def load(path, enc=False):
    with open(path, "rb") as f:
        data = f.read()
    return gzip.decompress(_fernet().decrypt(data)).decode("utf-8") if enc else data.decode("utf-8")


def get(url, tries=5, binary=False):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                return b if binary else b.decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            if e.code == 429 and i < tries - 1:       # rate limited: wait as told (or a minute or more), then retry
                ra = e.headers.get("Retry-After")
                time.sleep(int(ra) if ra and ra.isdigit() else 60 * (i + 1))
                continue
            if i == tries - 1:
                raise
        except Exception:
            if i == tries - 1:
                raise
        time.sleep(3 * (i + 1))


def log(src, unit, status, extra=None):
    p = os.path.join(OUT, "status.json")
    st = json.load(open(p)) if os.path.exists(p) else {}
    st.setdefault(src, {})[unit] = {"status": status, "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                    **(extra or {})}
    save(p, json.dumps(st, indent=1, sort_keys=True))


# ---------------------------------------------------------------- Fantasy Football Calculator (free API, attribution)
def ffc():
    """ADP by format x league size x year. The response's meta carries the mock-draft window (start/end date), which the
    hindsight audit checks against each season's kickoff."""
    for y in range(2010, 2026):
        for fmt in ["ppr", "half-ppr", "standard"]:
            for n in [10, 12, 14]:
                path = os.path.join(OUT, "ffc", f"ffc_{fmt}_{n}_{y}.json")
                if os.path.exists(path):
                    continue
                txt = get(f"https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={n}&year={y}")
                if not txt or '"players"' not in txt:
                    log("ffc", f"{fmt}_{n}_{y}", "missing")
                    continue
                j = json.loads(txt)
                save(path, txt)
                log("ffc", f"{fmt}_{n}_{y}", "ok", {"players": len(j.get("players", [])), "meta": j.get("meta", {})})
                time.sleep(1.5)       # the API asks not to be called too often


# ---------------------------------------------------------------- FantasyPros ADP pages (encrypted)
FP_PAGES = {"ppr": "ppr-overall.php", "half": "half-point-ppr-overall.php", "std": "overall.php"}


def fp_adp():
    """Each page has a column per platform (ESPN, Yahoo, CBS, Sleeper, NFL, RTSports, Fantrax, FFC, ...) for that year.
    ESPN's ADP lives under PPR, Yahoo's under half-PPR. The raw page is saved; parsing happens in the audit / board."""
    for y in range(2017, 2026):
        for fmt, page in FP_PAGES.items():
            path = os.path.join(OUT, "fp_adp", f"fp_adp_{fmt}_{y}.enc")
            if os.path.exists(path):
                continue
            html = get(f"https://www.fantasypros.com/nfl/adp/{page}?year={y}")
            if not html or "<table" not in html:
                log("fp_adp", f"{fmt}_{y}", "missing")
                continue
            rows = html.count("<tr")
            dates = sorted(set(re.findall(r"(?:Updated|updated|as of|As of)[^<]{0,40}?(\w+ \d{1,2},? \d{4})", html)))
            save(path, html, enc=True)
            log("fp_adp", f"{fmt}_{y}", "ok", {"table_rows": rows, "dates_on_page": dates[:5]})
            time.sleep(2)


# ---------------------------------------------------------------- nflverse (open data)
NFLV = {
    "draft_picks.parquet": "draft_picks/draft_picks.parquet",
    "trades.parquet": "trades/trades.parquet",
    "advstats_season_pass.parquet": "pfr_advstats/advstats_season_pass.parquet",
    "advstats_season_rush.parquet": "pfr_advstats/advstats_season_rush.parquet",
    "advstats_season_rec.parquet": "pfr_advstats/advstats_season_rec.parquet",
}


def nflverse():
    for name, rel in NFLV.items():
        path = os.path.join(OUT, "nflverse", name)
        if os.path.exists(path):
            continue
        b = get(f"{NFLVERSE}/{rel}", binary=True)
        if not b:
            log("nflverse", name, "missing")
            continue
        save(path, b)
        log("nflverse", name, "ok", {"bytes": len(b)})


# ---------------------------------------------------------------- Spotrac incentives (encrypted)
def _body(html):
    """Page content without scripts / timestamps, to tell whether a year's page differs from another year's."""
    h = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    m = re.search(r"NFL Contract Incentives(.*)", h, re.S)
    return re.sub(r"\s+", " ", m.group(1) if m else h).encode("utf-8")


def spotrac():
    """Incentive clauses by year and team. Spotrac added this view in 2024; earlier years are tried and logged."""
    for y in range(2019, 2027):
        found = 0
        for t in TEAMS:
            path = os.path.join(OUT, "spotrac", str(y), f"incentives_{t}.enc")
            if os.path.exists(path):
                found += 1
                continue
            html = get(f"https://www.spotrac.com/nfl/contracts/incentives/_/year/{y}/team/{t}")
            n = len(re.findall(r"Incentive|Escalator", html or ""))
            if not html or n == 0:
                log("spotrac", f"{y}_{t}", "empty")
                if t == TEAMS[2] and found == 0:      # three teams with nothing: that year isn't there
                    break
                continue
            save(path, html, enc=True)
            found += 1
            log("spotrac", f"{y}_{t}", "ok", {"mentions": n, "sha": hashlib.sha256(_body(html)).hexdigest()[:12]})
            time.sleep(7)        # Spotrac rate-limits at ~2 s per page


def fp_inspect():
    """Structure of the saved FantasyPros ADP pages only (sizes, tags, notices) - never their data (public logs)."""
    d = os.path.join(OUT, "fp_adp")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        h = load(os.path.join(d, f), enc=True)
        t = re.search(r"<title>(.*?)</title>", h, re.S)
        print(f, {"bytes": len(h), "title": (t.group(1).strip()[:70] if t else None), "tr": h.count("<tr"),
                  "tables": re.findall(r"<table[^>]*id=\"([^\"]+)\"", h)[:5],
                  "not_available": "not available" in h.lower(), "captcha": any(x in h.lower() for x in ("captcha", "cf-chl", "access denied")),
                  "js_vars": sorted(set(re.findall(r"\bvar\s+(\w*(?:adp|ADP|ecr|data|Data)\w*)\s*=", h)))[:8],
                  "api_urls": sorted(set(re.findall(r"https?://[\w./-]*(?:adp|api)[\w./?=&-]*", h)))[:5],
                  "dates": re.findall(r"(?:ADP|adp)[^<]{0,60}?(\w{3,9} \d{1,2}, \d{4})", h)[:3]})


def spotrac_inspect():
    """Are the year pages different? Hash of each team's page body per year (no content printed)."""
    base = os.path.join(OUT, "spotrac")
    years = sorted(os.listdir(base)) if os.path.isdir(base) else []
    for t in TEAMS[:6]:
        hs = {}
        for y in years:
            f = os.path.join(base, y, f"incentives_{t}.enc")
            if os.path.exists(f):
                hs[y] = hashlib.sha256(_body(load(f, enc=True))).hexdigest()[:10]
        print(t, hs, "all years identical" if len(set(hs.values())) == 1 and len(hs) > 1 else "")


def status():
    p = os.path.join(OUT, "status.json")
    st = json.load(open(p)) if os.path.exists(p) else {}
    for src, units in st.items():
        c = {}
        for u in units.values():
            c[u["status"]] = c.get(u["status"], 0) + 1
        print(src, c)


if __name__ == "__main__":
    {"ffc": ffc, "fp_adp": fp_adp, "nflverse": nflverse, "spotrac": spotrac, "status": status, "fp_inspect": fp_inspect, "spotrac_inspect": spotrac_inspect}[sys.argv[1]]()

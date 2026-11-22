"""Trade-offer digest (Nov 22, 2026): the site's own trade finder, read for every league, as a short text summary.

Serves docs/ locally, opens the My team tab in headless Chrome, and for each Yahoo league reads the top rows of the
trade finder in "Win-win" mode (both sides gain by our values and the trade looks fair to them by market value) and,
where it applies, "Escape last place". Used by the scheduled Tue / Fri digests before the Nov 28 trade deadline.

  python automation/trade_digest.py [--top 5] [--json out.json]

Needs playwright + Google Chrome; d3 is served from node_modules (installed into /tmp/d3t if missing) because the
workspace can't reach the CDN.
"""
import argparse
import asyncio
import functools
import http.server
import json
import os
import shutil
import subprocess
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3 = "/tmp/d3t/node_modules/d3/dist/d3.min.js"


def ensure_d3():
    if not os.path.exists(D3):
        os.makedirs("/tmp/d3t", exist_ok=True)
        subprocess.run(["npm", "install", "--silent", "--prefix", "/tmp/d3t", "d3@7"], check=True)
    return open(D3).read()


def serve(port):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    h = functools.partial(Quiet, directory=os.path.join(ROOT, "docs"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


READ = """() => {
  const heads = [...document.querySelectorAll('#team-trade-table thead th')].map((t) => t.textContent.trim());
  const rows = [...document.querySelectorAll('#team-trade-table tbody tr')].map((tr) =>
    [...tr.children].map((td) => td.textContent.replace(/\\s+/g, ' ').trim()));
  return { heads, rows };
}"""


async def run(top, port):
    from playwright.async_api import async_playwright
    d3 = ensure_d3()
    chrome = shutil.which("google-chrome") or "/usr/bin/google-chrome"
    out = {}
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=chrome if os.path.exists(chrome) else None)
        pg = await b.new_page(viewport={"width": 1500, "height": 1000})
        await pg.route("**/cdn.jsdelivr.net/**", lambda r: r.fulfill(body=d3, content_type="application/javascript"))
        await pg.goto(f"http://127.0.0.1:{port}/index.html")
        await pg.wait_for_timeout(6000)
        await pg.evaluate("document.querySelector('button.nf-tab[data-tab=team]').click()")
        await pg.wait_for_timeout(1500)
        leagues = await pg.evaluate("[...document.querySelectorAll('#team-leagues button')].map(b => [b.dataset.id, b.textContent])")
        for lid, label in leagues:
            await pg.evaluate(f"document.querySelector('#team-leagues button[data-id=\"{lid}\"]').click()")
            await pg.wait_for_timeout(1500)
            res = {}
            for mode in ["Win-win", "Escape last place"]:
                await pg.evaluate(f"[...document.querySelectorAll('#team-trade-mode button')].find(b => b.textContent === '{mode}').click()")
                await pg.wait_for_timeout(5000 if mode == "Escape last place" else 2500)
                t = await pg.evaluate(READ)
                rows = [dict(zip(t["heads"], r)) for r in t["rows"] if len(r) >= 4][:top]
                res[mode] = rows if rows else [{"note": " ".join(" ".join(r) for r in t["rows"])[:200]}]
            out[label] = res
        await b.close()
    return out


def fmt(out):
    lines = []
    for lg, res in out.items():
        lines.append(f"## {lg}")
        for mode, rows in res.items():
            lines.append(f"**{mode}**")
            for r in rows:
                if "note" in r:
                    lines.append(f"- {r['note']}")
                    continue
                extra = []
                for k in ["You / wk", "Them / wk", "Looks like to them", "Your last place"]:
                    if r.get(k):
                        extra.append(f"{k} {r[k]}")
                lines.append(f"- Give {r.get('You give')} → get {r.get('You get')} (with {r.get('With')}) · " + " · ".join(extra))
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    srv = serve(a.port)
    try:
        out = asyncio.run(run(a.top, a.port))
    finally:
        srv.shutdown()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    print(fmt(out))


if __name__ == "__main__":
    main()

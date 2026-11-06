"""Builds docs/preview/index.html: the new Fantasy Oracle shell (loading screen, home, My Team, rankings, trade analyzer
and the "More" pages) around the existing dashboard sections and their scripts, which keep doing all the math.

  python docs/preview/build.py            # reads docs/index.html, writes docs/preview/index.html

The older sections are moved onto "More" pages unchanged (same element ids, same scripts); a handful of small,
listed patches let them follow the site-wide league dropdown and hand their results to the new pages (fo.js).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "index.html")
OUT = os.path.join(HERE, "index.html")
TEMPLATE = os.path.join(HERE, "shell.html")

src = open(SRC, encoding="utf-8").read()


def section(i):
    m = re.search(r'(<section class="nf-panel" id="%s".*?</section>)' % i, src, re.S)
    if not m:
        sys.exit("missing section " + i)
    return m.group(1)


def card(html, cid):
    """one <div class="nf-card" id=cid> ... </div> (balanced divs)"""
    start = html.index('id="%s"' % cid)
    start = html.rindex("<div", 0, start)
    depth, i = 0, start
    for m in re.finditer(r"<div\b|</div>", html[start:]):
        depth += 1 if m.group(0) != "</div>" else -1
        if depth == 0:
            return html[start:start + m.end()]
    sys.exit("unbalanced " + cid)


S = {k: section(k) for k in ["team", "projections", "ros", "trade", "matchups", "style", "weather", "experts", "model"]}

# --------------------------------------------------------------- scripts (everything after the page markup)
scripts = src[src.index("<script>", src.index('<div class="nf-page">')):src.rindex("</body>")]
scripts = scripts.replace("</div>\n</div>\n<script>", "<script>", 1)   # the dash-root / hub-wrap closers


def patch(old, new, count=1):
    global scripts
    if scripts.count(old) < 1:
        sys.exit("patch target not found: " + old[:80])
    scripts = scripts.replace(old, new, count)


# 1. the old tab bar is gone: the router in fo.js shows pages
patch("""  tabs.forEach((t) => t.addEventListener('click', () => show(t.dataset.tab, true)));
  const named = location.hash.slice(1);
  if (tabs.some((t) => t.dataset.tab === named)) show(named, false);
  window.nfShow = show;""", """  window.nfShow = (id) => window.foGo && window.foGo(id);""")
# 2. My team: follow the site-wide league; hand the lineup to the new My Team page
patch("""    if (!st.league || !leagues.find((l) => String(l.league_id) === st.league)) st.league = String(leagues[0].league_id);""",
      """    st.league = window.foLeague && leagues.find((l) => String(l.league_id) === window.foLeague) ? window.foLeague
      : (st.league && leagues.find((l) => String(l.league_id) === st.league) ? st.league : String(leagues[0].league_id));""")
patch("""    // ---------------------------------------------------------------- waiver wire
    const owned""", """    window.foTeamData = { lg, wk, lineup, bench, players, total, proj, ros, roster, adj, vegas, elig };
    if (window.foOnTeam) window.foOnTeam();
    // ---------------------------------------------------------------- waiver wire
    const owned""")
# 3. Trade analyzer: follow the site-wide league; hand its results to the new trade page
patch("""    if (!st.lid) st.lid = String(L.data[0].league_id);""",
      """    { const gl = window.foLeague && L.data.find((l) => String(l.league_id) === window.foLeague) ? window.foLeague : (st.lid || String(L.data[0].league_id));
      if (gl !== st.lid) { st.lid = gl; st.mine = null; st.theirs = null; st.give.clear(); st.get.clear(); } }""")
patch("""    const scB = scoreMap(res.before), scA = scoreMap(after);""",
      """    const scB = scoreMap(res.before), scA = scoreMap(after);
    window.foTradeData = { ctx, res, ss, ds, st, teams, name, scB, scA, rankOf, slotRank, ord, spB, spA,
      help: helpTrade(ctx, res, ds, ss), price: res.active ? priceTrade(ctx, res, ds) : null,
      headline: document.getElementById('ta-headline').textContent, sub: document.getElementById('ta-headline-sub').textContent,
      toggle(side, key) { const s = side === 'mine' ? st.give : st.get; if (s.has(key)) s.delete(key); else s.add(key); render(); },
      setTeam(side, id) { if (side === 'mine') { st.mine = id; st.give.clear(); if (st.theirs === id) st.theirs = null; } else { st.theirs = id; st.get.clear(); } render(); },
      clear() { st.give.clear(); st.get.clear(); render(); } };""")
patch("""  function render() {
    const R = dash.data('league_rosters'), L = dash.data('leagues'), P = dash.data('ros'), LP = dash.data('league_players'), TM = dash.data('trade_market');""",
      """  function render() { renderCore(); if (window.foOnTrade) window.foOnTrade(); }
  function renderCore() {
    const R = dash.data('league_rosters'), L = dash.data('leagues'), P = dash.data('ros'), LP = dash.data('league_players'), TM = dash.data('trade_market');""")
# 3b. the trade finder (now on the Trade Analyzer page) opens on Fleece
patch("""  const tst = { kind: 'All', mode: 'Win-win' }""", """  const tst = { kind: 'All', mode: 'Fleece' }""")
# 4. chart tooltips: the stage is zoomed, so pointer offsets are divided by the zoom
patch("""      const r = box.getBoundingClientRect();
      let x = ev.clientX - r.left + 12, y = ev.clientY - r.top + 12;""", """      const z = window.foScale || 1, r0 = box.getBoundingClientRect(), r = { left: r0.left, top: r0.top, width: r0.width / z, height: r0.height / z };
      let x = (ev.clientX - r.left) / z + 12, y = (ev.clientY - r.top) / z + 12;""")
patch("""      if (x + tw > r.width) x = Math.max(0, ev.clientX - r.left - tw - 12);
      if (y + th > r.height) y = Math.max(0, ev.clientY - r.top - th - 12);""", """      if (x + tw > r.width) x = Math.max(0, (ev.clientX - r.left) / z - tw - 12);
      if (y + th > r.height) y = Math.max(0, (ev.clientY - r.top) / z - th - 12);""")
# 5. data lives one folder up from the preview
patch("""fetch('data/datasets.json', {cache: 'no-store'})""", """fetch(window.FO_DATA + 'datasets.json', {cache: 'no-store'})""")

# --------------------------------------------------------------- the older sections, regrouped onto "More" pages
team, proj, ros, trade = S["team"], S["projections"], S["ros"], S["trade"]


def strip_section(html):
    """section contents without the outer <section>"""
    return re.sub(r'^<section[^>]*>|</section>$', "", html.strip(), flags=re.S)


more = {}
more["lineup"] = ('<div class="nf-controls hide-here" id="team-leagues" role="group" aria-label="League"></div>'
                  '<p class="mp-intro">The full start / sit and waiver tables behind the My Team page: the Vegas line, '
                  'notes where your Yahoo lineup differs from the best one, close calls, and each free agent\'s upgrade over your weakest player.</p>'
                  + card(team, "team-lineup-card") + card(team, "team-waiver-card"))
more["finder"] = card(team, "team-trades-card")
proj_inner = strip_section(proj)
more["weekly"] = ('<p class="mp-intro">Everything on the old weekly page: the full sortable projection table with search and game '
                  'status filters, this week\'s tiles and receiver usage. Weeks here follow the week buttons below.</p>' + proj_inner)
more["rosdetails"] = strip_section(ros)
trade_inner = strip_section(trade).replace('<div id="ta-leagues"', '<div class="hide-here" id="ta-leagues"')
more["tradedetails"] = ('<p class="mp-intro">The complete trade analysis for the trade checked on the Trade Analyzer page (same league, '
                        'same teams and players): the price in trade and actual value, the value check, draft picks (The League), '
                        'both lineups, the league power rankings and the playoff predictor.</p>' + trade_inner)
for k in ["matchups", "style", "weather", "experts", "model"]:
    more[k] = strip_section(S[k])
more["history"] = open(os.path.join(HERE, "history.html"), encoding="utf-8").read()

tpl = open(TEMPLATE, encoding="utf-8").read()
IDS = {"weekly": "projections", "rosdetails": "ros"}
for k, v in more.items():
    tpl = tpl.replace("<!--MORE:%s-->" % k, '<div class="nf-panel" id="%s">%s</div>' % (IDS.get(k, k if k in S else "mp-" + k), v))
tpl = tpl.replace("<!--SCRIPTS-->", scripts)
open(OUT, "w", encoding="utf-8").write(tpl)
print("wrote", OUT, len(tpl))

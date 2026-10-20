// Standings and regular-season matchups (weeks 1-14) for each of Jackson's Yahoo leagues, for the power rankings'
// playoff predictor and the trade analyzer's weeks-that-count. Run with Claude in Chrome on
// football.fantasysports.yahoo.com (signed in); writes text into the page (read it with get_page_text).
// Same parsing as yahoo_picks.js (The League's picks stay with that script).
(async () => {
  const LIDS = ['126912', '200436', '727092'], LAST_REG = 14;
  const t = (e) => (e ? e.textContent.replace(/\s+/g, ' ').trim() : '');
  const get = async (u) => new DOMParser().parseFromString(await fetch(u).then((r) => r.text()), 'text/html');
  const out = [];
  for (const LID of LIDS) {
    out.push('LEAGUE\t' + LID);
    const home = await get('/f1/' + LID);
    const st = [...home.querySelectorAll('table')].find((tb) => /W-L-T/.test(t(tb.querySelector('thead'))));
    out.push('STANDINGS');
    if (st) st.querySelectorAll('tbody tr').forEach((tr) => {
      const a = [...tr.querySelectorAll('a')].find((x) => new RegExp('/f1/' + LID + '/\\d+$').test(x.getAttribute('href') || ''));
      if (!a) return;
      const c = [...tr.querySelectorAll('td')].map(t);
      out.push([a.getAttribute('href').split('/').pop(), t(a), c[2], c[3], c[4]].join('\t'));   // team id, name, W-L-T, PF, PA
    });
    out.push('MATCHUPS');
    for (let w = 1; w <= LAST_REG; w++) {
      const d = await get('/f1/' + LID + '?matchup_week=' + w + '&module=matchups&lhst=matchups');
      const seen = new Set();
      d.querySelectorAll('a').forEach((a) => {
        const h = a.getAttribute('href') || '', m = h.match(/matchup\?week=(\d+)&mid1=(\d+)&mid2=(\d+)/);
        if (!m || +m[1] !== w || seen.has(m[2] + '-' + m[3])) return;
        seen.add(m[2] + '-' + m[3]);
        const li = a.closest('li'), nums = li ? (t(li).match(/-?\d+\.\d\d/g) || []) : [];
        out.push([w, m[2], m[3], nums.join(' ')].join('\t'));   // week, team1, team2, points / projected as shown
      });
    }
  }
  document.body.innerHTML = '';
  const pre = document.createElement('pre'); pre.textContent = out.join('\n'); document.body.appendChild(pre);
  return out.length + ' lines';
})();

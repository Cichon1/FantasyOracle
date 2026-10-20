// The League (Yahoo 200436): standings, weekly matchups for the rest of the regular season, and next season's draft
// pick ownership. Run with Claude in Chrome on football.fantasysports.yahoo.com (signed in); writes text into the page.
(async () => {
  const LID = '200436', t = (e) => (e ? e.textContent.replace(/\s+/g, ' ').trim() : '');
  const get = async (u) => new DOMParser().parseFromString(await fetch(u).then((r) => r.text()), 'text/html');
  const out = [];
  // standings
  const home = await get('/f1/' + LID);
  const st = [...home.querySelectorAll('table')].find((tb) => /W-L-T/.test(t(tb.querySelector('thead'))));
  const names = {};
  out.push('STANDINGS');
  st.querySelectorAll('tbody tr').forEach((tr) => {
    const a = [...tr.querySelectorAll('a')].find((x) => new RegExp('/f1/' + LID + '/\\d+$').test(x.getAttribute('href') || ''));
    const tid = a.getAttribute('href').split('/').pop(), c = [...tr.querySelectorAll('td')].map(t);
    names[t(a)] = tid;
    out.push([tid, t(a), c[2], c[3], c[4]].join('\t'));   // team id, name, W-L-T, PF, PA
  });
  // matchups, current week through week 14 (regular season), with current and projected points
  out.push('MATCHUPS');
  for (let w = 1; w <= 14; w++) {
    const d = await get('/f1/' + LID + '?matchup_week=' + w + '&module=matchups&lhst=matchups');
    const seen = new Set();
    d.querySelectorAll('a').forEach((a) => {
      const h = a.getAttribute('href') || '', m = h.match(/matchup\?week=(\d+)&mid1=(\d+)&mid2=(\d+)/);
      if (!m || +m[1] !== w || seen.has(m[2] + '-' + m[3])) return;
      seen.add(m[2] + '-' + m[3]);
      const li = a.closest('li'), nums = li ? (t(li).match(/-?\d+\.\d\d/g) || []) : [];
      out.push([w, m[2], m[3], nums.join(' ')].join('\t'));   // week, team1, team2, points / projected numbers as shown
    });
  }
  // next season's draft picks: owner -> original team, by round
  out.push('PICKS');
  const tp = await get('/f1/' + LID + '/showtradedpicks');
  let round = null;
  tp.querySelectorAll('h3, h4, h5, caption, tr').forEach((e) => {
    const x = t(e), m = x.match(/^Round (\d+)$/);
    if (m) { round = +m[1]; return; }
    if (e.tagName !== 'TR' || !round) return;
    const cells = [...e.querySelectorAll('td')];
    if (cells.length < 2) return;
    const owner = t(cells[0]), picks = [...cells[1].querySelectorAll('a, span, div, li')].map(t).filter((v) => names[v]);
    (picks.length ? picks : [t(cells[1])]).forEach((orig) => out.push([round, names[owner] || owner, names[orig] || orig].join('\t')));
  });
  document.body.innerHTML = '';
  const pre = document.createElement('pre'); pre.textContent = out.join('\n'); document.body.appendChild(pre);
  return out.length + ' lines';
})();

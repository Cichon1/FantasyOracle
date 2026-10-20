// Run in the page on football.fantasysports.yahoo.com (signed in to Yahoo). Reads each league's Rosters page
// (/f1/<id>/starters, every position incl. K / DEF / IDP and IR slots) and Settings page (/f1/<id>/settings), and writes
// TEAMS / ROSTERS / SETTINGS text into the page for yahoo_league.py (read it with get_page_text, then reload the page).
const LEAGUES = ['126912', '200436', '727092'];
const STATUS = /^(O|Q|D|IR|IR-R|PUP-R|PUP-P|NA|SUSP|NFI-R|COVID-19)$/;
const parseLeague = async (lid) => {
  const h = await fetch('/f1/' + lid + '/starters').then((r) => r.text());
  const d = new DOMParser().parseFromString(h, 'text/html');
  const out = [];
  d.querySelectorAll('table').forEach((t) => {
    const a = t.parentElement.querySelector('a[href^="/f1/' + lid + '/"]');
    const team = a ? a.textContent.trim() : '?';
    const tid = a ? a.getAttribute('href').split('/')[3] : '?';
    t.querySelectorAll('tbody tr').forEach((tr) => {
      const td = tr.querySelectorAll('td');
      const slot = td[0] ? td[0].textContent.trim() : '';
      const nm = tr.querySelector('a.name');
      let tp = '';
      tr.querySelectorAll('span').forEach((s) => { const x = s.textContent.trim(); if (!tp && /^[A-Za-z]{2,3} - [A-Z,\/]+$/.test(x)) tp = x; });
      const st = [...tr.querySelectorAll('span,abbr')].map((s) => s.textContent.trim()).find((x) => STATUS.test(x)) || '';
      if (nm) out.push([lid, tid, slot, nm.textContent.trim(), (nm.getAttribute('href') || '').split('/').pop() || '0',
        tp.split(' - ')[0] || '-', tp.split(' - ')[1] || '-', st]);
    });
    out.push(['#TEAM', lid, tid, team]);
  });
  return out;
};
const settings = async (lid) => {
  const h = await fetch('/f1/' + lid + '/settings').then((r) => r.text());
  const d = new DOMParser().parseFromString(h, 'text/html');
  const rows = [];
  d.querySelectorAll('tr').forEach((tr) => {
    const c = tr.querySelectorAll('td,th');
    if (c.length >= 2) {
      const k = c[0].textContent.trim().replace(/\s+/g, ' ').replace(/Yahoo Default$/, '').replace(/:$/, '');
      const v = c[1].textContent.trim().replace(/\s+/g, ' ');
      if (k && k.length < 70 && v.length < 300) rows.push([lid, k, v].join(' | '));
    }
  });
  return rows;
};
const teams = [], rows = [], sets = [];
for (const lid of LEAGUES) {
  for (const r of await parseLeague(lid)) {
    if (r[0] === '#TEAM') teams.push(r.slice(1).join(' ')); else rows.push(r.join(' ').trim());
  }
  sets.push(...await settings(lid));
}
document.body.innerHTML = '';
const art = document.createElement('article'), pre = document.createElement('pre');
pre.textContent = 'TEAMS\n' + [...new Set(teams)].join('\n') + '\nROSTERS\n' + rows.join('\n') + '\nSETTINGS\n' + sets.join('\n');
art.appendChild(pre); document.body.appendChild(art);
[teams.length, rows.length, sets.length];

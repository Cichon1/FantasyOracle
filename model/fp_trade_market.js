// FantasyPros Trade Market Values (redraft, PPR) -> CSV text, written into the page as <pre> for get_page_text.
// Run with Claude in Chrome on any www.fantasypros.com page (no sign-in needed). QB / RB / WR / TE with a market value.
(async () => {
  const r = await fetch('https://mpbnfl.fantasypros.com/api/tradeMarketValues?type=redraft&scoring=PPR&includeChartData=true', { credentials: 'omit' });
  const j = await r.json();
  const dates = [];
  for (let i = j.windowDays - 1; i >= 0; i--) { const d = new Date(j.anchorDate + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() - i); dates.push(d.toISOString().slice(0, 10)); }
  const key = (s) => s.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[.'’\-]/g, '')
    .replace(/\b(jr|sr|ii|iii|iv|v)\b/g, '').replace(/\s+/g, ' ').trim();
  const q = (s) => '"' + String(s).replace(/"/g, '""') + '"';
  const lines = ['as_of,window_start,fp_id,player,pos,team,market_value,market_rank_pos,trades_10d,value_change_10d,values_by_day,trades_by_day,name_key'];
  for (const p of j.players) {
    if (!['QB', 'RB', 'WR', 'TE'].includes(p.position) || !(p.marketValue > 0)) continue;
    const vm = Object.fromEntries((p.valueSeries || []).map((x) => [x.date, x.value]));
    const nm = Object.fromEntries((p.volumeSeries || []).map((x) => [x.date, x.volume]));
    const vals = dates.map((d) => vm[d] ?? ''), vols = dates.map((d) => nm[d] ?? '');
    const first = vals.find((v) => v !== '' && v != null);
    lines.push([j.anchorDate, dates[0], p.playerId, q(p.name), p.position, p.team || '', p.marketValue, p.marketRankPositional ?? '',
      vols.reduce((a, b) => a + (+b || 0), 0), first == null ? '' : p.marketValue - first, vals.join(' '), vols.join(' '), q(key(p.name))].join(','));
  }
  document.body.innerHTML = '';
  const pre = document.createElement('pre'); pre.textContent = lines.join('\n'); document.body.appendChild(pre);
  return (lines.length - 1) + ' players, as of ' + j.anchorDate;
})();

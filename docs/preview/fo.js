/* Fantasy Oracle — site shell and the redesigned pages (page design by Jackson Cichon).
   The model's numbers come from the same dashboard code as before (the scripts above this one); this file lays them
   out on the new pages: loading screen, home, My Team, Weekly / Rest of Season rankings, Trade Analyzer, Rank
   History, plus the zoom-to-fit stage, page transitions, the slide-down menus and the arrow scrollbars. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const num = (v, d = 0) => (v === '' || v == null || isNaN(+v) ? d : +v);
  const has = (v) => !(v === '' || v == null || (typeof v === 'number' && isNaN(v)) || v === 'NaN');
  const f1 = d3.format('.1f'), f0 = d3.format('.0f');
  const sg1 = (x) => (Math.abs(x) < 0.05 ? '0.0' : d3.format('+.1f')(x));
  const ord = (n) => n + (['th', 'st', 'nd', 'rd'][(n % 100 - 20) % 10] || ['th', 'st', 'nd', 'rd'][n % 100] || 'th');
  const h = (tag, attrs, ...kids) => {
    const e = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v; else if (k === 'style') e.style.cssText = v; else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? '' : v);
    }
    kids.flat().forEach((k) => { if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k))); });
    return e;
  };
  const POS = ['QB', 'RB', 'WR', 'TE'];
  const SLOTNAME = (s) => (s === 'W/R/T' || s === 'R/W/T' ? 'FLEX' : s === 'Q/W/R/T' ? 'SF' : s === 'W/R' || s === 'W/T' ? 'FLEX' : s);
  const chip = (txt, cls, size) => h('span', { class: 'chip ' + cls + (size ? ' ' + size : '') }, txt === 'SF' ? 'SFLX' : txt === 'FLEX' && size ? 'Flex' : txt);
  const confCls = (lab) => (lab === 'High' ? 'conf-hi' : lab === 'Medium' ? 'conf-md' : lab === 'Low' ? 'conf-lo' : 'muted');
  const D = (id) => dash.data(id);
  const ok = (...ids) => ids.every((i) => D(i).status === 'ok');

  // ================================================================ stage: zoom the 1920 x 1080 design to the window
  const vp = $('#viewport'), stage = $('#stage');
  function fit() {
    const W = window.innerWidth, H = window.innerHeight;
    const touch = matchMedia('(pointer: coarse)').matches;
    let s = Math.min(W / 1920, H / 1080), scroll = false;
    if (touch && W > H && W / H > 1.8) { s = W / 1920; scroll = 1080 * s > H + 2; }   // phones held sideways: fill the width, scroll down
    window.foScale = s;
    const left = Math.max(0, (W - 1920 * s) / 2), top = scroll ? 0 : Math.max(0, (H - 1080 * s) / 2);
    stage.style.transform = `translate(${left}px, ${top}px) scale(${s})`;
    vp.classList.toggle('scrolly', scroll);
    vp.style.setProperty('--stage-h', 1080 * s + 'px');
    if (scroll) { let pad = $('#stage-pad'); if (!pad) { pad = h('div', { id: 'stage-pad' }); vp.appendChild(pad); } pad.style.cssText = `height:${1080 * s}px;width:1px`; }
  }
  fit(); window.addEventListener('resize', fit);

  // ================================================================ league (one choice for the whole site)
  let leagues = [];
  const LS = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } } };
  window.foLeague = LS.get('fo.league') || null;
  const league = () => leagues.find((l) => String(l.league_id) === window.foLeague) || leagues[0];
  function setLeague(id) {
    if (String(id) === window.foLeague) return;
    window.foLeague = String(id); LS.set('fo.league', window.foLeague);
    leagueDD.set(window.foLeague);
    dash.setParams({});          // every module redraws for the new league
  }

  // ================================================================ slide-down dropdown (league, trade teams)
  function dropdown(root, { rowH, top, bottom, onPick, label }) {
    const box = root.querySelector('.box'), list = root.querySelector('.list');
    let items = [], value = null;
    const close = () => { root.classList.remove('open'); place(); };
    const place = () => {
      const i = Math.max(0, items.findIndex((x) => x.id === value));
      const open = root.classList.contains('open');
      list.style.transform = open ? 'none' : `translateY(${-i * rowH}px)`;
      list.style.transition = 'transform .42s cubic-bezier(.3,.85,.25,1)';
      root.style.setProperty('--ho', (top + items.length * rowH + bottom) + 'px');
      list.querySelectorAll('.opt').forEach((o) => o.setAttribute('aria-selected', String(o.dataset.id === value)));
    };
    root.addEventListener('click', (e) => {
      const o = e.target.closest('.opt');
      if (root.classList.contains('open') && o) { value = o.dataset.id; close(); onPick(value); return; }
      if (!root.classList.contains('open')) { closeMenus(root); root.classList.add('open'); place(); }
      else close();
      e.stopPropagation();
    });
    box.setAttribute('role', 'button'); box.setAttribute('aria-label', label); box.tabIndex = 0;
    box.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); box.click(); }
      if (e.key === 'Escape') close();
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault(); const i = items.findIndex((x) => x.id === value) + (e.key === 'ArrowDown' ? 1 : -1);
        if (items[i]) { value = items[i].id; place(); onPick(value); }
      }
    });
    return {
      setItems(arr, v) {
        const sig = arr.map((x) => x.id + x.text).join('|');
        if (sig !== root.dataset.sig) {
          root.dataset.sig = sig; items = arr;
          list.replaceChildren(...arr.map((x) => h('div', { class: 'opt', 'data-id': x.id, role: 'option' }, x.text)));
        }
        if (v != null) value = String(v); place();
      },
      set(v) { value = String(v); place(); }, close, el: root
    };
  }
  const openMenus = new Set();
  function closeMenus(except) {
    $$('.dd.open').forEach((d) => { if (d !== except) d.click && d.classList.remove('open'); });
    $$('.dd').forEach((d) => { if (d !== except) d.dispatchEvent(new Event('fo-close')); });
    if (except !== moreNav) setMore(false);
  }
  document.addEventListener('click', (e) => { if (!e.target.closest('.dd') && !e.target.closest('#more')) closeMenus(null); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenus(null); });
  const leagueDD = dropdown($('#league-dd'), { rowH: 52, top: 70, bottom: 40, label: 'League', onPick: (id) => setLeague(id) });
  $('#league-dd').addEventListener('fo-close', () => { $('#league-dd').classList.remove('open'); leagueDD.set(window.foLeague); });

  // ================================================================ More: the banner slides down
  const moreNav = $('#more'), moreBtn = $('#more-btn');
  function setMore(open) {
    moreNav.classList.toggle('open', open); moreBtn.setAttribute('aria-expanded', String(open));
    moreNav.style.setProperty('--hopen', ($('#more-list').offsetHeight + 150) + 'px');
  }
  moreBtn.addEventListener('click', (e) => { e.stopPropagation(); const o = !moreNav.classList.contains('open'); closeMenus(moreNav); setMore(o); });
  document.addEventListener('click', (e) => {
    const a = e.target.closest('[data-go]'); if (!a) return;
    e.preventDefault(); setMore(false); go(a.dataset.go);
    if (a.dataset.anchor) setTimeout(() => { const t = document.getElementById(a.dataset.anchor), sc = t && t.closest('.tp-scroll'); if (sc) sc.scrollTo({ top: t.offsetTop - 20, behavior: 'smooth' }); }, 450);
  });

  // ================================================================ the eye blinks like a lizard when clicked
  const eye = $('#eye');
  const hole = $('#eye-hole'), rim = $('#eye-rim'), ring = $('#eye-ring');
  const almond = (cy, hw, hh) => `M${50 - hw} ${cy} Q50 ${cy - 2 * hh} ${50 + hw} ${cy} Q50 ${cy + 2 * hh} ${50 - hw} ${cy}Z`;
  const OPEN = [50, 72, 78];                  // centre y, half width, half height: lids pulled back past the edge
  const setEye = (cy, hw, hh) => { const d = almond(cy, hw, hh); hole.setAttribute('d', d); rim.setAttribute('d', d); };
  setEye(...OPEN);
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2), easeIn = (t) => t * t * t, easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const lerp = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
  let blinking = false;
  function blink() {
    if (blinking) return; blinking = true; ring.setAttribute('opacity', '1');
    const SHUT = [57, 40, 0], SLIT = [60, 36, 1.5];
    const steps = [                            // [from, to, ms, easing]
      [OPEN, SHUT, 210, easeIn],               // lids snap shut
      [SHUT, SHUT, 170, (t) => t],             // closed
      [SHUT, SLIT, 120, ease],                 // a seam opens low on the eye
      [SLIT, [48, 50, 20], 230, ease],         // the almond widens
      [[48, 50, 20], OPEN, 330, easeOut]       // and the lids roll back
    ];
    let i = 0, t0 = performance.now();
    const tick = (now) => {
      const [a, b, ms, f] = steps[i], t = Math.min(1, (now - t0) / ms);
      setEye(...lerp(a, b, f(t)));
      if (t >= 1) { i++; t0 = now; if (i >= steps.length) { setEye(...OPEN); ring.setAttribute('opacity', '0'); blinking = false; return; } }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  eye.addEventListener('click', (e) => { e.stopPropagation(); closeMenus(null); blink(); });

  // ================================================================ long explanations fold away behind a "+"
  const KEEP = new Set(['ta-value-sum', 'ta-headline-sub', 'team-lineup-meta', 'ta-draft-rule', 'ta-draft-sum']);
  function plusBtn(label, targets, cls) {
    const b = h('button', { class: 'plus' + (cls ? ' ' + cls : ''), 'aria-expanded': 'false', title: 'Show the explanation' },
      h('span', { class: 'pm' }, '+'), label ? h('span', { class: 'pl' }, label) : null);
    targets.forEach((t) => t.classList.add('fold-hide'));
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      const open = b.getAttribute('aria-expanded') !== 'true';
      b.setAttribute('aria-expanded', String(open)); b.querySelector('.pm').textContent = open ? '–' : '+';
      targets.forEach((t) => t.classList.toggle('fold-hide', !open));
      BARS.forEach((u) => u());
    });
    return b;
  }
  function foldAll() {
    // notes that explain a card: the "+" sits at the end of the card's heading
    $$('.page .mp p.nf-note, .page .mp p.nf-take, .page .mp p.mp-intro').forEach((p) => {
      if (KEEP.has(p.id) || p.closest('.fold-done')) return;
      let prev = p.previousElementSibling;
      while (prev && prev.classList.contains('fold-hide')) prev = prev.previousElementSibling;
      const head = prev && (/^H[23]$/.test(prev.tagName) ? prev : prev.classList.contains('nf-ta-head') ? prev.querySelector('h2') : null);
      if (head) {
        // one button opens every note under the same heading
        const b = head.querySelector(':scope > .plus');
        if (b) { b._t.push(p); p.classList.add('fold-hide'); return; }
        const t = [p], nb = plusBtn('', t); nb._t = t; head.append(nb);
      } else {
        const label = p.classList.contains('mp-intro') ? 'About this page' : p.classList.contains('nf-take') ? 'Overview' : 'How to read this';
        p.before(plusBtn(label, [p], 'solo'));
      }
    });
    // "how it works" cards: everything after the heading
    $$('.page .mp .nf-method').forEach((c) => {
      const h2 = c.querySelector('h2'); if (!h2 || h2.querySelector('.plus')) return;
      h2.append(plusBtn('', [...c.children].filter((x) => x !== h2)));
    });
    // the explanation lines under the main tables
    [['wr-note', 'Key'], ['rr-note', 'Key'], ['tm-note', 'Key']].forEach(([id, label]) => {
      const n = document.getElementById(id); if (!n || n.dataset.folded) return;
      n.dataset.folded = '1';
      const b = plusBtn(label, [n], 'line');
      b.style.left = n.style.left; b.style.top = n.style.top;
      n.style.left = (parseFloat(n.style.left) + 96) + 'px'; n.style.width = (parseFloat(n.style.width || 1740) - 96) + 'px';
      n.before(b);
    });
  }
  foldAll();
  // card titles stay on one line: long ones get smaller (never under 24px)
  function fitTitles(root) {
    $$('.mp .nf-card > h2', root).forEach((t) => {
      t.style.fontSize = '';
      let fs = parseFloat(getComputedStyle(t).fontSize);
      while (t.scrollWidth > t.clientWidth + 1 && fs > 24) { fs -= 1; t.style.fontSize = fs + 'px'; }
    });
  }

  // ================================================================ arrow scrollbars
  const BARS = [];
  function arrowbar(bar, sc) {
    const thumb = h('div', { class: 'thumb' });
    const slim = bar.classList.contains('slim'), E = slim ? 30 : 36;   // room taken by each end button
    bar.append(h('div', { class: 'track' }), thumb,
      h('div', { class: 'end up', title: 'Scroll up' }, h('img', { src: 'assets/needle-black.png', alt: '' })),
      h('div', { class: 'end dn', title: 'Scroll down' }, h('img', { src: 'assets/needle-brown.png', alt: '' })));
    const draw = () => {};
    const upd = () => {
      const H = bar.clientHeight, sh = sc.scrollHeight, ch = sc.clientHeight;
      if (!H) return;
      if (sh <= ch + 2) { bar.classList.add('off'); return; }
      bar.classList.remove('off');
      const track = H - 2 * E - 4, th = Math.max(40, track * ch / sh), y = E + 2 + (track - th) * (sc.scrollTop / (sh - ch));
      thumb.style.height = th + 'px'; thumb.style.top = y + 'px';
    };
    sc.addEventListener('scroll', upd, { passive: true });
    let drag = null;
    thumb.addEventListener('pointerdown', (e) => { e.preventDefault(); e.stopPropagation(); thumb.setPointerCapture(e.pointerId); drag = { y: e.clientY, st: sc.scrollTop }; });
    thumb.addEventListener('pointermove', (e) => {
      if (!drag) return; const s = window.foScale || 1, H = bar.clientHeight, th = thumb.offsetHeight;
      const ratio = (sc.scrollHeight - sc.clientHeight) / Math.max(1, H - 2 * E - 4 - th);
      sc.scrollTop = drag.st + (e.clientY - drag.y) / s * ratio;
    });
    thumb.addEventListener('pointerup', () => { drag = null; });
    bar.addEventListener('pointerdown', (e) => {
      if (e.target === thumb) return;
      const s = window.foScale || 1, r = bar.getBoundingClientRect(), y = (e.clientY - r.top) / s;
      const H = bar.clientHeight;
      if (y < E) sc.scrollBy({ top: -sc.clientHeight * 0.4, behavior: 'smooth' });
      else if (y > H - E) sc.scrollBy({ top: sc.clientHeight * 0.4, behavior: 'smooth' });
      else sc.scrollBy({ top: (y < thumb.offsetTop ? -1 : 1) * sc.clientHeight * 0.85, behavior: 'smooth' });
    });
    const ro = new ResizeObserver(() => { draw(); upd(); });
    ro.observe(sc); if (sc.firstElementChild) ro.observe(sc.firstElementChild);
    new MutationObserver(() => requestAnimationFrame(upd)).observe(sc, { childList: true, subtree: true });
    draw(); upd();
    BARS.push(upd);
    return upd;
  }
  function hbar(sc) {
    const paper = sc.closest('.paper'); if (!paper) return;
    const bar = h('div', { class: 'hbar' }), thumb = h('div', { class: 'thumb' });
    bar.append(h('div', { class: 'track' }), thumb, h('div', { class: 'end l', title: 'Scroll left' }, h('img', { src: 'assets/needle-black.png', alt: '' })), h('div', { class: 'end r', title: 'Scroll right' }, h('img', { src: 'assets/needle-brown.png', alt: '' })));
    paper.after(bar);
    const place = () => { bar.style.left = paper.offsetLeft + 'px'; bar.style.width = paper.offsetWidth + 'px'; bar.style.top = (paper.offsetTop + paper.offsetHeight + 6) + 'px'; };
    const upd = () => {
      place();
      const W = bar.clientWidth, sw = sc.scrollWidth, cw = sc.clientWidth;
      if (!W || sw <= cw + 2) { bar.classList.add('off'); return; }
      bar.classList.remove('off');
      const track = W - 76, tw = Math.max(60, track * cw / sw), x = 38 + (track - tw) * (sc.scrollLeft / (sw - cw));
      thumb.style.width = tw + 'px'; thumb.style.left = x + 'px';
    };
    sc.addEventListener('scroll', upd, { passive: true });
    let drag = null;
    thumb.addEventListener('pointerdown', (e) => { e.preventDefault(); e.stopPropagation(); thumb.setPointerCapture(e.pointerId); drag = { x: e.clientX, sl: sc.scrollLeft }; });
    thumb.addEventListener('pointermove', (e) => { if (!drag) return; const s = window.foScale || 1; const ratio = (sc.scrollWidth - sc.clientWidth) / Math.max(1, bar.clientWidth - 76 - thumb.offsetWidth); sc.scrollLeft = drag.sl + (e.clientX - drag.x) / s * ratio; });
    thumb.addEventListener('pointerup', () => { drag = null; });
    bar.addEventListener('pointerdown', (e) => {
      if (e.target === thumb) return;
      const s = window.foScale || 1, r = bar.getBoundingClientRect(), x = (e.clientX - r.left) / s;
      sc.scrollBy({ left: (x < thumb.offsetLeft ? -1 : 1) * sc.clientWidth * 0.6, behavior: 'smooth' });
    });
    const ro = new ResizeObserver(upd); ro.observe(sc); if (sc.firstElementChild) ro.observe(sc.firstElementChild);
    new MutationObserver(() => requestAnimationFrame(upd)).observe(sc, { childList: true, subtree: true });
    upd(); BARS.push(upd);
  }
  $$('.sc.hx').forEach(hbar);
  $$('.arrowbar').forEach((bar) => {
    const sc = bar.dataset.for ? document.getElementById(bar.dataset.for) : bar.parentElement.querySelector('.sc');
    if (sc) arrowbar(bar, sc);
  });

  // ================================================================ pages and transitions
  const ORDER = ['home', 'team', 'projections', 'ros', 'trade', 'league', 'matchups', 'style', 'weather', 'experts', 'model', 'lineup', 'tradedetails', 'weekly', 'rosdetails', 'history'];
  const PAGES = {
    home: { title: '' }, team: { title: 'My Team', league: true }, projections: { title: 'Weekly Rankings', league: true },
    ros: { title: 'Rest of Season Rankings', league: true, long: true }, trade: { title: 'Trade Analyzer', league: true },
    league: { title: 'League View', league: true }
  };
  $$('.page.more').forEach((p) => { PAGES[p.dataset.page] = { title: p.dataset.title, league: p.classList.contains('league'), long: p.dataset.title.length > 15 }; });
  const RENDER = {};
  const dirty = new Set(Object.keys(PAGES));
  let cur = null, ready = false;
  const pageEl = (id) => $(`.page[data-page="${id}"]`);
  const sides = {};
  function setTitle(id) {
    const P = PAGES[id] || {};
    const t = $('#tb-title'); t.textContent = P.title || ''; t.classList.remove('long');
    const sub = sides[id] ? sides[id].slice().reverse().filter(Boolean).join('  ·  ') : '';
    $('#tb-sub').textContent = sub; t.classList.toggle('two', !!sub);
    // the ribbon grows with the title; it never gets narrower than the design's 523px
    const w = Math.max(523, Math.ceil(t.scrollWidth) + 70 + 70);
    $('#titlebar').style.setProperty('--tw', w + 'px');
    const W = w;
    $('#tb-rib').setAttribute('viewBox', `0 0 ${W} 139`);
    $('#tb-rib path').setAttribute('d', `M0 0H${W}L${W - 32.5} 64.9C${W - 33.9} 67.7 ${W - 33.9} 71 ${W - 32.5} 73.8L${W} 139H0Z`);
  }
  window.foSides = (id, l, r) => { sides[id] = [l, r]; if (cur === id && !$('#titlebar').classList.contains('swap')) setTitle(id); };
  function go(id, opts = {}) {
    if (!PAGES[id]) id = 'home';
    if (id === cur && !opts.force) return;
    const from = cur; cur = id;
    const hash = id === 'home' ? '' : '#' + id;
    if (location.hash !== hash && !opts.fromHash) history.pushState(null, '', location.pathname + location.search + hash);
    document.body.classList.toggle('on-home', id === 'home');
    document.body.classList.toggle('sub', id !== 'home');
    document.body.classList.toggle('no-league', !PAGES[id].league);
    $$('#more-list a').forEach((a) => a.toggleAttribute('aria-current', a.dataset.go === id));
    if (a11yMain) a11yMain.setAttribute('aria-label', PAGES[id].title || 'Home');
    const tb = $('#titlebar');
    if (from && from !== 'home' && id !== 'home') { tb.classList.add('swap'); setTimeout(() => { setTitle(id); tb.classList.remove('swap'); }, 230); }
    else setTitle(id);
    const back = from != null && ORDER.indexOf(id) < ORDER.indexOf(from);
    if (from) {
      const o = pageEl(from);
      o.classList.remove('enter', 'back'); o.classList.add('leave'); if (back) o.classList.add('back');
      o.classList.remove('on');
      setTimeout(() => { if (cur !== from) o.classList.remove('leave', 'back'); }, 360);
    }
    const n = pageEl(id);
    n.classList.remove('leave'); n.classList.add('on');
    if (opts.anim !== false) { n.classList.remove('enter', 'back'); void n.offsetWidth; n.classList.add('enter'); if (back) n.classList.add('back'); setTimeout(() => n.classList.remove('enter', 'back'), 1100); }
    renderPage(id);
    setTimeout(() => fitTitles(n), 30);
    const hook = { weather: 'weather', model: 'model', style: 'style', experts: 'experts', rosdetails: 'ros' }[id];
    if (hook) setTimeout(() => { const b = document.querySelector('#tabhooks [data-tab="' + hook + '"]'); if (b) b.click(); }, 80);
    setTimeout(() => { window.dispatchEvent(new Event('resize')); BARS.forEach((u) => u()); }, 60);
    const sc = n.querySelector('.mp-scroll'); if (sc && from) sc.scrollTop = 0;
  }
  const a11yMain = $('#pages');
  window.foGo = go;
  window.addEventListener('popstate', () => go(location.hash.slice(1) || 'home', { fromHash: true }));
  function renderPage(id) { if (RENDER[id] && ready) { try { RENDER[id](); dirty.delete(id); } catch (e) { console.error('render ' + id, e); } } }
  function refresh() {
    Object.keys(RENDER).forEach((k) => dirty.add(k));
    if (cur) renderPage(cur);
  }

  // ================================================================ loading screen -> first page
  const t0 = performance.now();
  const CORE = ['projections', 'ros', 'leagues', 'league_rosters'];
  function boot() {
    const done = CORE.filter((i) => D(i).status !== 'loading').length;
    $('#loadbar').style.setProperty('--p', done / CORE.length);
    const el = performance.now() - t0;
    if (el > 1200) document.body.classList.add('slow');
    if ((done < CORE.length && el < 9000) || el < 1000) { requestAnimationFrame(boot); return; }
    ready = true;
    const first = location.hash.slice(1);
    document.body.classList.remove('loading', 'slow');
    go(PAGES[first] ? first : 'home', { fromHash: true, force: true });
  }
  requestAnimationFrame(boot);

  // ================================================================ shared: league scoring and rest-of-season ranks
  const adj = (r, lg) => {
    if (!r || !lg) return 0;
    let a = (num(lg.fum_lost, -2) + 2) * num(r.fum_pg) + (num(lg.rec_pt, 1) - 1) * num(r.rec_pg)
      + num(lg.bonus_rush150) * num(r.p_rush150) + num(lg.bonus_rec175) * num(r.p_rec175);
    if (r.pos === 'QB') a += (num(lg.pass_td, 4) - 4) * num(r.pass_td_pg) + (num(lg.pass_int, -2) + 2) * num(r.int_pg)
      + (1 / num(lg.pass_yds_per_pt, 25) - 1 / 25) * num(r.pass_yds_pg) + num(lg.bonus_pass400) * num(r.p_pass400);
    return a;
  };
  const scoringText = (lg) => {
    const bits = [num(lg.pass_td, 4) + '-pt pass TDs'];
    if (num(lg.pass_yds_per_pt, 25) !== 25) bits.push('1 pt / ' + num(lg.pass_yds_per_pt) + ' pass yds');
    if (num(lg.pass_int, -2) !== -2) bits.push(num(lg.pass_int) ? num(lg.pass_int) + ' per INT' : 'no INT penalty');
    if (!num(lg.fum_lost)) bits.push('no fumble penalty');
    if (num(lg.bonus_pass400) || num(lg.bonus_rush150) || num(lg.bonus_rec175)) bits.push('yardage bonuses');
    return bits.join(', ');
  };
  const rosCache = new Map();
  // rows: ROS rows of one week (live ros_v5, or a history week); rates: the live ROS rows (per-game scoring rates)
  function leagueRos(rows, lg, tag) {
    const key = tag + '|' + lg.league_id + '|' + rows.length;
    if (rosCache.has(key)) return rosCache.get(key);
    const rates = new Map(D('ros').data.map((r) => [r.key, r]));
    const out = new Map(), by = {};
    rows.forEach((r) => {
      const rr = rates.get(r.key) || r;
      const d = adj(rr, lg) * num(rr.play_rate, num(rr.avail, 1)) * num(r.games_left);
      const o = { r, pts: Math.max(0, num(r.ros_ppr) + d), lo: Math.max(0, num(r.ros_low) + d), hi: Math.max(0, num(r.ros_high) + d), model: num(r.ros_model) + d };
      o.ppg = num(r.games_left) ? o.pts / num(r.games_left) : 0;
      out.set(r.key, o); (by[r.pos] = by[r.pos] || []).push(o);
    });
    Object.values(by).forEach((list) => {
      list.sort((a, b) => b.pts - a.pts || num(a.r.ros_rank) - num(b.r.ros_rank)).forEach((o, i) => { o.rank = i + 1; });
      list.slice().sort((a, b) => b.model - a.model).forEach((o, i) => { o.mrank = i + 1; });
    });
    rosCache.set(key, out);
    return out;
  }
  const mineKeys = (lg) => new Set(D('league_rosters').status === 'ok' ? D('league_rosters').data.filter((p) => String(p.league_id) === String(lg.league_id) && String(p.is_me) === '1' && p.ros_key).map((p) => p.ros_key) : []);
  const rosteredKeys = (lg) => new Set(D('league_rosters').status === 'ok' ? D('league_rosters').data.filter((p) => String(p.league_id) === String(lg.league_id) && p.ros_key).map((p) => p.ros_key) : []);
  const teamName = (lid, tid) => {
    const S = D('league_standings'); const r = S.status === 'ok' ? S.data.find((x) => String(x.league_id) === String(lid) && String(x.team_id) === String(tid)) : null;
    if (r) return r.team;
    const R = D('league_rosters'); const p = R.status === 'ok' ? R.data.find((x) => String(x.league_id) === String(lid) && String(x.team_id) === String(tid)) : null;
    return p ? p.team : 'team ' + tid;
  };

  // ================================================================ common: data events
  dash.onData(() => {
    const L = D('leagues');
    if (L.status === 'ok') {
      leagues = L.data;
      if (!window.foLeague || !league() || String(league().league_id) !== window.foLeague) window.foLeague = String(league().league_id);
      leagueDD.setItems(leagues.map((l) => ({ id: String(l.league_id), text: l.league })), window.foLeague);
    }
    rosCache.clear();
    refresh();
  });

  // ================================================================ MY TEAM
  let wpos = 'All';
  const wvChips = $('#wv-chips');
  ['All', ...POS].forEach((p) => wvChips.append(h('button', { class: 'chip ' + p, 'aria-pressed': String(p === wpos), onclick: () => { wpos = p; $$('button', wvChips).forEach((b) => b.setAttribute('aria-pressed', String(b.textContent === p))); RENDER.team(); } }, p)));
  window.foOnTeam = () => { if (cur === 'team') RENDER.team(); else dirty.add('team'); };
  const cell = (x, text, cls, w) => h('div', { class: 'c ' + (cls || ''), style: `left:${x}px` + (w ? `;width:${w}px` : '') }, text);
  const opp = (w) => (w ? (w.venue === 'Home' ? 'vs ' : '@') + w.opp : '—');
  const tmMoves = new Map();   // league + week -> starting keys you set (null = the best lineup)
  let tmSel = null;
  // click a player, then the one to swap with; or drag him onto that player (mouse or finger)
  function tmDragStart(e, row, key, swap, canSwap) {
    if (e.button > 0) return;
    const s0 = window.foScale || 1, x0 = e.clientX, y0 = e.clientY;
    let ghost = null, over = null;
    const move = (ev) => {
      const dx = (ev.clientX - x0) / s0, dy = (ev.clientY - y0) / s0;
      if (!ghost && Math.hypot(dx, dy) < 8) return;
      if (!ghost) {
        tmSel = key;
        ghost = row.cloneNode(true); ghost.classList.add('ghost');
        const r = row.getBoundingClientRect(), sr = stage.getBoundingClientRect();
        ghost.style.cssText += `;position:absolute;left:${(r.left - sr.left) / s0}px;top:${(r.top - sr.top) / s0}px;width:${r.width / s0}px;z-index:60`;
        stage.appendChild(ghost); row.classList.add('dragging');
      }
      ghost.style.transform = `translate(${dx}px, ${dy}px)`;
      ghost.style.display = 'none';
      const t = document.elementFromPoint(ev.clientX, ev.clientY); ghost.style.display = '';
      const tr = t && t.closest('.mv');
      if (over && over !== tr) over.classList.remove('over');
      if (over) over.classList.remove('over', 'nope');
      over = tr && tr.dataset.key !== key ? tr : null; if (over) over.classList.add(canSwap(key, over.dataset.key) ? 'over' : 'nope');
    };
    const up = () => {
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up);
      if (!ghost) {   // a click
        if (!tmSel) { tmSel = key; RENDER.team(); } else swap(tmSel, key);
        return;
      }
      ghost.remove(); row.classList.remove('dragging');
      if (over && canSwap(key, over.dataset.key)) swap(key, over.dataset.key); else { tmSel = null; RENDER.team(); }
    };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up); window.addEventListener('pointercancel', up);
  }
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && tmSel) { tmSel = null; RENDER.team(); } });
  RENDER.team = () => {
    const T = window.foTeamData, lg = league();
    const start = $('#tm-start'), bench = $('#tm-bench'), benchH = $('#tm-bench-h'), wv = $('#wv'), wvH = $('#wv-h');
    if (!T || !lg || String(T.lg.league_id) !== String(lg.league_id) || !ok('ros')) { start.replaceChildren(h('div', { class: 'empty' }, 'Loading your roster…')); return; }
    const ranks = leagueRos(D('ros').data, lg, 'live');
    const rk = (key) => { const o = ranks.get(key); return o ? o.r.pos + o.rank : '—'; };
    $('#tm-name').textContent = lg.my_team_name;
    // opponent this week
    const SC = D('league_schedule');
    let oppName = '';
    if (SC.status === 'ok') {
      const m = SC.data.find((x) => String(x.league_id) === String(lg.league_id) && num(x.week) === T.wk && (String(x.team1) === String(lg.my_team) || String(x.team2) === String(lg.my_team)));
      if (m) oppName = teamName(lg.league_id, String(m.team1) === String(lg.my_team) ? m.team2 : m.team1);
    }
    window.foSides('team', oppName ? 'Opp: ' + oppName : '', 'Week ' + T.wk);
    // ---- starting roster and bench: the best lineup, or the one you set by moving players around
    const SO = (sl) => { const k = POS.indexOf(sl); return k >= 0 ? k : 4 + (sl === 'SF' ? 1 : 0); };
    const best = T.lineup.map((l, i) => ({ l, i })).sort((a, b) => SO(SLOTNAME(a.l.slot)) - SO(SLOTNAME(b.l.slot)) || a.i - b.i).map((o) => o.l);
    const slots = best.map((l) => l.slot);
    const everyone = best.map((l) => l.x).filter(Boolean).concat(T.bench);
    const byKey = new Map(everyone.map((x) => [x.p.key, x]));
    const bestTotal = best.reduce((t, l) => t + (l.x ? l.x.v : 0), 0);
    const mkey = 'fo.lineup.' + lg.league_id + '.' + T.wk;
    let mine = tmMoves.get(mkey);
    if (mine === undefined) { try { mine = JSON.parse(LS.get(mkey) || 'null'); } catch (e) { mine = null; } }
    // a saved lineup still has to fit this roster: same slots, players still here, eligible for their slot, not on IR
    const valid = (arr) => Array.isArray(arr) && arr.length === slots.length && new Set(arr.filter(Boolean)).size === arr.filter(Boolean).length
      && arr.every((k, i) => !k || (byKey.has(k) && byKey.get(k).p.slot !== 'IR' && T.elig(slots[i]).includes(byKey.get(k).p.pos)));
    if (mine && !valid(mine)) mine = null;
    const keys = mine || best.map((l) => (l.x ? l.x.p.key : null));
    tmMoves.set(mkey, mine);
    const lineup = slots.map((slot, i) => ({ slot, x: keys[i] ? byKey.get(keys[i]) : null }));
    const used = new Set(keys.filter(Boolean));
    const benchList = everyone.filter((x) => !used.has(x.p.key)).sort((a, b) => (a.p.slot === 'IR') - (b.p.slot === 'IR') || b.v - a.v);
    const total = lineup.reduce((t, l) => t + (l.x ? l.x.v : 0), 0);
    const save = (arr) => { const same = arr.every((k, i) => k === (best[i].x ? best[i].x.p.key : null)); const v = same ? null : arr; tmMoves.set(mkey, v); if (v) LS.set(mkey, JSON.stringify(v)); else { try { localStorage.removeItem(mkey); } catch (e) { /* ignore */ } } tmSel = null; RENDER.team(); };
    // can player x go into starting slot i (and, for a starter swap, the other one into his slot)?
    const fits = (x, i) => !!x && x.p.slot !== 'IR' && T.elig(slots[i]).includes(x.p.pos);
    const where = (key) => { const i = keys.indexOf(key); return i >= 0 ? { start: true, i } : { start: false }; };
    function swap(aKey, bKey) {
      if (!aKey || !bKey || aKey === bKey) { tmSel = null; RENDER.team(); return; }
      const A = where(aKey), B = where(bKey), arr = keys.slice();
      if (A.start && B.start) { if (!fits(byKey.get(aKey), B.i) || !fits(byKey.get(bKey), A.i)) return flash(); arr[A.i] = bKey; arr[B.i] = aKey; }
      else if (A.start || B.start) { const S = A.start ? A : B, inK = A.start ? bKey : aKey; if (!fits(byKey.get(inK), S.i)) return flash(); arr[S.i] = inK; }
      else { tmSel = null; RENDER.team(); return; }   // two bench players: nothing to change
      save(arr);
    }
    const canSwap = (aKey, bKey) => {
      if (!aKey || !bKey || aKey === bKey) return false;
      const A = where(aKey), B = where(bKey);
      if (A.start && B.start) return fits(byKey.get(aKey), B.i) && fits(byKey.get(bKey), A.i);
      if (A.start) return fits(byKey.get(bKey), A.i);
      if (B.start) return fits(byKey.get(aKey), B.i);
      return false;
    };
    const flash = () => { const el = $('#tm-start').closest('.pn'); el.animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-8px)' }, { transform: 'translateX(8px)' }, { transform: 'translateX(0)' }], { duration: 260 }); tmSel = null; RENDER.team(); };
    const mover = (row, key) => {
      if (!key) return;
      row.dataset.key = key; row.classList.add('mv');
      if (tmSel === key) row.classList.add('sel');
      else if (tmSel && canSwap(tmSel, key)) row.classList.add('ok');
      else if (tmSel) row.classList.add('no');
      row.addEventListener('pointerdown', (e) => tmDragStart(e, row, key, swap, canSwap));
    };
    const n = lineup.length, pitch = Math.min(48.5, 336 / Math.max(1, n));
    const rows = [h('div', { class: 'r h', style: 'top:6px;height:34px' }, cell(15, 'Slot'), cell(118, 'Player'), cell(351, 'Opp'), cell(466, 'Proj'), cell(560, 'Start%'), cell(665, 'ROS'))];
    lineup.forEach((l, i) => {
      const y = 54 + i * pitch, x = l.x, sl = SLOTNAME(l.slot);
      const r = h('div', { class: 'r b', style: `top:${y}px;height:39px` });
      r.append(h('div', { class: 'c', style: 'left:15px' }, chip(sl === 'SF' ? 'SF' : sl, POS.includes(sl) ? sl : (sl === 'SF' ? 'SF' : 'FLEX'))));
      if (!x) { r.append(cell(118, 'Empty — nobody eligible', 'bad')); rows.push(r); return; }
      const nm = h('div', { class: 'c nm', style: 'left:118px;max-width:228px' }, x.p.player);
      if (!x.why && !x.yahooStart) nm.append(h('span', { class: 'flag start', title: 'On your bench in Yahoo' }));
      if (x.why) nm.append(h('span', { class: 'flag out', title: x.why }));
      r.append(nm, cell(351, x.why === 'Bye' ? 'Bye' : opp(x.w), x.why ? 'muted' : ''), cell(466, x.why ? '0.0' : f1(x.v), x.why ? 'muted' : ''));
      const sp = x.why || !x.w || !has(x.w.start_pct) ? null : num(x.w.start_pct);
      r.append(cell(560, sp == null ? 'N/a' : Math.round(sp) + '%', sp == null ? 'muted' : confCls(x.w.confidence)));
      r.append(cell(665, x.p.ros_key ? rk(x.p.ros_key) : '—'));
      const vl = T.vegas(x.p.player, x.p.pos); r.title = [x.why, vl ? 'Vegas: ' + vl : ''].filter(Boolean).join(' · ');
      mover(r, x.p.key);
      rows.push(r);
    });
    const ty = 54 + n * pitch + 4;
    const d = total - bestTotal;
    const totRow = h('div', { class: 'r b', style: `top:${Math.max(ty, 390)}px;height:40px` }, cell(15, 'Projected total'), cell(466, f1(total), '', null),
      mine ? cell(560, (Math.abs(d) < 0.05 ? 'same as best' : sg1(d) + ' vs best'), Math.abs(d) < 0.05 ? 'muted' : d < 0 ? 'bad' : 'good') : cell(560, 'best lineup', 'muted'));
    rows.push(totRow);
    start.replaceChildren(...rows);
    totRow.children[1].style.color = 'var(--navy)';
    totRow.children[2].style.fontSize = '22px';
    totRow.append(h('div', { style: 'position:absolute;left:15px;top:34px;width:199px;height:8px' }, svgUnder()));
    const ctl = $('#tm-ctl');
    ctl.replaceChildren(...[
      mine ? h('button', { class: 'tg', onclick: () => save(best.map((l) => (l.x ? l.x.p.key : null))) }, 'Reset to best (' + f1(bestTotal) + ')') : null,
      h('span', { class: 'hint' }, tmSel ? 'Now click who to swap him with' : 'Click or drag players to swap them')].filter(Boolean));
    // ---- bench
    benchH.replaceChildren(h('div', { class: 'r h', style: 'top:6px;height:34px' }, cell(12, 'Slot'), cell(90, 'Pos'), cell(179, 'Player'), cell(440, 'Opp'), cell(556, 'Proj'), cell(633, 'Start%'), cell(729, 'ROS')));
    const br = benchList.map((x, i) => {
      const isIR = x.p.slot === 'IR';
      const r = h('div', { class: 'r b', style: `top:${2 + i * 48.5}px;height:39px` });
      r.append(isIR ? h('div', { class: 'c ir', style: 'left:12px' }, 'IR', h('img', { src: 'assets/caduceus.png', alt: '' })) : cell(17, 'BN'));
      r.append(h('div', { class: 'c', style: 'left:90px' }, chip(x.p.pos, x.p.pos)));
      const nm = h('div', { class: 'c nm', style: 'left:179px;max-width:240px' }, x.p.player);
      if (!x.why && x.yahooStart) nm.append(h('span', { class: 'flag sit', title: 'Starting in Yahoo' }));
      r.append(nm, cell(430, isIR ? opp(x.w) : (x.why === 'Bye' ? 'Bye' : opp(x.w)), x.why ? 'muted' : ''));
      r.append(cell(559, x.why ? 'N/a' : f1(x.v), x.why ? 'muted' : ''));
      const sp = x.why || !x.w || !has(x.w.start_pct) ? null : num(x.w.start_pct);
      r.append(cell(633, sp == null ? 'N/a' : Math.round(sp) + '%', sp == null ? 'muted' : confCls(x.w.confidence)));
      r.append(cell(729, x.p.ros_key ? rk(x.p.ros_key) : '—'));
      const vl = T.vegas(x.p.player, x.p.pos);
      r.title = [x.why, vl ? 'Vegas: ' + vl : ''].filter(Boolean).join(' · ');
      if (!isIR) mover(r, x.p.key);
      return r;
    });
    bench.style.height = (benchList.length * 48.5 + 6) + 'px';
    bench.replaceChildren(...br);
    // ---- waiver wire (best unrostered players in this league's scoring)
    const owned = rosteredKeys(lg), lgv = (r) => Math.max(0, num(r.ros_ppg) + adj(r, lg) * num(r.play_rate, num(r.avail, 1)));
    const worst = {};
    POS.forEach((p) => { const a = T.players.filter((x) => x.p.pos === p && x.r && x.p.slot !== 'IR').map((x) => ({ x, v: lgv(x.r) })).sort((a, b) => a.v - b.v); worst[p] = a[0]; });
    const TR = D('trending'), adds = new Map(TR.status === 'ok' ? TR.data.map((t) => [t.key, num(t.adds_24h)]) : []);
    const gain = (o) => o.v - (worst[o.r.pos] ? worst[o.r.pos].v : 0);
    const fa = D('ros').data.filter((r) => POS.includes(r.pos) && !owned.has(r.key) && r.status !== 'IR / reserve' && (wpos === 'All' || r.pos === wpos))
      .map((r) => ({ r, v: lgv(r) })).sort((a, b) => (wpos === 'All' ? gain(b) - gain(a) : b.v - a.v)).slice(0, 15);
    wvH.replaceChildren(h('div', { class: 'r h', style: 'top:4px;height:34px' }, cell(14, 'Pos'), cell(75, 'Player'), cell(250, 'This wk'), cell(362, 'ROS')));
    wv.style.height = (fa.length * 49 + 4) + 'px';
    wv.replaceChildren(...fa.map(({ r, v }, i) => {
      const w = T.proj.get(r.key);
      const row = h('div', { class: 'r b', style: `top:${i * 49}px;height:39px` },
        h('div', { class: 'c', style: 'left:4px' }, chip(r.pos, r.pos)),
        h('div', { class: 'c', style: 'left:75px;max-width:178px;overflow:hidden;text-overflow:ellipsis;display:block;line-height:39px' }, r.player),
        cell(282, w && w.status !== 'Inactive' ? f1(num(w.proj_ppr) + adj(r, lg)) : (r.status === 'Bye this week' ? 'Bye' : '—'), w ? '' : 'muted'),
        cell(362, r.pos + ' ' + ((ranks.get(r.key) || {}).rank || '—')));
      const wo = worst[r.pos], g = wo ? v - wo.v : null, ad = adds.get(r.key);
      row.title = r.player + ' (' + r.team + ') · ' + f1(v) + ' ROS pts / game' + (wo ? ' · ' + (g > 0 ? sg1(g) + ' per game over ' : 'not better than ') + wo.x.p.player : '') + (ad ? ' · ' + d3.format(',')(ad) + ' Sleeper adds in 24h' : '');
      return row;
    }));
    if (!fa.length) wv.replaceChildren(h('div', { class: 'empty', style: 'font-size:20px;padding:10px' }, 'No unrostered players with a projection here.'));
    const pulled = d3.max(T.roster, (p) => p.pulled_at);
    $('#tm-note').innerHTML = '';
    $('#tm-note').append(lg.league + ' scoring · roster as of ' + (pulled ? new Date(pulled).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : '—') + '  ',
      h('span', { class: 'flag start' }), ' start (benched in Yahoo)  ', h('span', { class: 'flag sit' }), ' sit (starting in Yahoo)  ', h('span', { class: 'flag out' }), ' can’t play · hover a row for the Vegas line · ',
      h('a', { 'data-go': 'lineup', style: 'color:var(--gold);cursor:pointer' }, 'Lineup Details'));
  };
  function svgUnder() {
    const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    s.setAttribute('viewBox', '0 0 199 8'); s.setAttribute('width', '199'); s.setAttribute('height', '8');
    s.innerHTML = '<path d="M1 5 C 40 2, 90 6, 130 3.5 S 185 2.5, 198 4" fill="none" stroke="#171b33" stroke-width="2.6" stroke-linecap="round"/>';
    return s;
  }

  // ================================================================ rankings tables (weekly and ROS)
  function weekTabs(box, weeks, curW, reW, onPick) {
    const sig = weeks.join(',') + '|' + curW;
    if (box.dataset.sig === sig) return;
    box.dataset.sig = sig;
    box.replaceChildren(...weeks.map((w) => h('button', { class: 'wk' + (reW(w) ? ' re' : ''), 'aria-pressed': String(w === curW), title: reW(w) ? 'Re-created after the fact' : null, onclick: () => onPick(w) }, 'Week ' + w)));
  }
  function posTabs(box, curP, onPick) {
    if (!box.childElementCount) ['All', ...POS].forEach((p) => box.append(h('button', { class: 'chip ' + p, onclick: () => onPick(p) }, p)));
    $$('button', box).forEach((b) => b.setAttribute('aria-pressed', String(b.textContent === curP)));
  }
  function toggles(box, list, curV, onPick) {
    if (!box.childElementCount) list.forEach((t) => box.append(h('button', { class: 'tg', onclick: () => onPick(t) }, t)));
    $$('button', box).forEach((b) => b.setAttribute('aria-pressed', String(b.textContent === curV)));
  }
  function rkTable(table, cols, rows, st, rerender) {
    const thead = table.tHead, tbody = table.tBodies[0];
    const sig = cols.map((c) => c.k).join(',');
    if (thead.dataset.sig !== sig) {
      thead.dataset.sig = sig;
      thead.replaceChildren(h('tr', null, cols.map((c) => h('th', { class: (c.hc || '') + ' ' + (c.g || '') + (c.div ? ' div' : '') + (c.l ? ' l' : ''), 'data-k': c.k, style: c.w ? `min-width:${c.w}px` : null },
        c.sort === false ? c.label : h('button', { onclick: () => { if (st.sort === c.k) st.dir *= -1; else { st.sort = c.k; st.dir = c.asc ? 1 : -1; } rerender(); }, html: null }, c.label)))));
      $$('th', thead).forEach((th, i) => { if (cols[i].html) { const b = th.querySelector('button') || th; b.innerHTML = cols[i].html; } });
    }
    $$('th', thead).forEach((th) => { th.removeAttribute('aria-sort'); if (th.dataset.k === st.sort) th.setAttribute('aria-sort', st.dir === 1 ? 'ascending' : 'descending'); });
    const c = cols.find((x) => x.k === st.sort) || cols[0];
    const val = c.v || ((r) => r[c.k]);
    const sorted = rows.slice().sort((a, b) => {
      const x = val(a), y = val(b);
      if (typeof x === 'string' || typeof y === 'string') return st.dir * String(x ?? '').localeCompare(String(y ?? ''));
      const xn = x == null || isNaN(x), yn = y == null || isNaN(y);
      if (xn && yn) return 0; if (xn) return 1; if (yn) return -1;
      return st.dir * (x - y);
    });
    const frag = document.createDocumentFragment();
    sorted.forEach((r) => {
      const tr = h('tr', { class: r._mine ? 'mine' : null });
      cols.forEach((col) => {
        const td = h('td', { class: (col.g || '') + (col.div ? ' div' : '') + (col.l ? ' l' : '') });
        const out = col.f(r, td);
        if (out != null) { if (out.nodeType) td.append(out); else td.textContent = out; }
        tr.append(td);
      });
      frag.append(tr);
    });
    if (!sorted.length) frag.append(h('tr', null, h('td', { class: 'empty', colspan: cols.length }, st.empty || 'No players match.')));
    tbody.replaceChildren(frag);
  }
  const bar = (lo, mid, hi, act, max, pos) => {
    const b = h('span', { class: 'bar ' + (pos || '') + (has(act) && has(mid) ? (num(act) >= num(mid) ? ' up' : ' dn') : '') }), x = (v) => Math.max(0, Math.min(100, 100 * v / max));
    if (has(lo) && has(hi)) b.append(h('i', { style: `left:${x(lo)}%;width:${Math.max(1.5, x(hi) - x(lo))}%` }));
    if (has(mid)) b.append(h('b', { style: `left:${x(mid)}%` }));
    if (has(act)) b.append(h('u', { style: `left:${x(act)}%`, title: 'Actual' }));
    return b;
  };
  const injCell = (inj, td) => {
    const s = String(inj ?? '');
    if (!s || s === 'Not listed') { td.classList.add('inj-n'); return 'N/a'; }
    td.classList.add(/DNP|Out|Doubtful|Inactive|IR/.test(s) ? 'inj-d' : 'inj-q');
    return s;
  };
  const updn = (d, td) => { if (d == null || isNaN(d)) return '—'; if (Math.abs(d) < 0.5) { td.classList.add('muted'); return '0'; } td.classList.add(d > 0 ? 'good' : 'bad'); return h('span', null, h('span', { class: 'updn' }, d > 0 ? '▲' : '▼'), String(Math.abs(Math.round(d)))); };

  // ---------------------------------------------------------------- WEEKLY RANKINGS
  const wst = { week: null, pos: 'All', status: 'All', q: '', sort: 'pts', dir: -1 };
  $('#wr-q').addEventListener('input', (e) => { wst.q = e.target.value.toLowerCase(); RENDER.projections(); });
  const RE_WEEKS = 4;   // Weeks 1-4 of 2026 were re-created after the fact; from Week 5 the projections were made live
  RENDER.projections = () => {
    const lg = league(); const P = D('projections');
    if (!lg || !ok('projections', 'ros')) { $('#wr-table').tBodies[0].replaceChildren(h('tr', null, h('td', { class: 'empty' }, 'Loading…'))); return; }
    const weeks = [...new Set(P.data.map((r) => +r.week))].sort((a, b) => a - b);
    if (wst.week == null || !weeks.includes(wst.week)) wst.week = weeks.includes(nfWeek()) ? nfWeek() : weeks[weeks.length - 1];
    weekTabs($('#wr-weeks'), weeks, wst.week, (w) => w <= RE_WEEKS, (w) => { wst.week = w; RENDER.projections(); });
    posTabs($('#wr-pos'), wst.pos, (p) => { wst.pos = p; RENDER.projections(); });
    toggles($('#wr-status'), ['All', 'Upcoming', 'Final', 'Inactive'], wst.status, (t) => { wst.status = t; RENDER.projections(); });
    const ros = new Map(D('ros').data.map((r) => [r.key, r])), mine = mineKeys(lg);
    const all = P.data.filter((r) => +r.week === wst.week).map((r) => {
      const out = r.status === 'Inactive' || r.status === 'Did not play';
      const a = out ? 0 : adj(ros.get(r.key), lg);
      return { r, pts: Math.max(0, num(r.proj_ppr) + a), lo: has(r.ppr_low) ? Math.max(0, num(r.ppr_low) + a) : NaN, hi: has(r.ppr_high) ? num(r.ppr_high) + a : NaN, _mine: mine.has(r.key) };
    });
    const byPos = {}; all.forEach((o) => (byPos[o.r.pos] = byPos[o.r.pos] || []).push(o));
    Object.values(byPos).forEach((l) => l.sort((a, b) => b.pts - a.pts).forEach((o, i) => { o.prank = i + 1; }));
    const max = d3.max(all, (o) => Math.max(o.hi || 0, num(o.r.act_ppr))) || 1;
    const rows = all.filter((o) => (wst.pos === 'All' || o.r.pos === wst.pos)
      && (wst.status === 'All' || (wst.status === 'Upcoming' ? o.r.status === 'Upcoming' : wst.status === 'Inactive' ? o.r.status === 'Inactive' : (o.r.status !== 'Upcoming' && o.r.status !== 'Inactive')))
      && (!wst.q || (o.r.key + ' ' + o.r.opp).toLowerCase().includes(wst.q)));
    const n = (k, fmt = f0) => (o) => (has(o.r[k]) ? fmt(num(o.r[k])) : '—');
    const nv = (k) => (o) => (has(o.r[k]) ? num(o.r[k]) : NaN);
    const cols = [
      { k: 'prank', label: 'Pos Rank', html: 'Pos<br>Rank', hc: 'sm', v: (o) => (wst.pos === 'All' ? o.pts : -o.prank) * 1, f: (o) => o.r.pos + o.prank, w: 56 },
      { k: 'player', label: 'Player', l: true, asc: true, v: (o) => o.r.key, f: (o) => o.r.key, w: 250 },
      { k: 'pos', label: 'Pos', asc: true, v: (o) => o.r.pos, f: (o) => chip(o.r.pos, o.r.pos, 'sm') },
      { k: 'opp', label: 'Opp', asc: true, v: (o) => o.r.opp, f: (o) => (o.r.venue === 'Home' ? '' : '@') + o.r.opp },
      { k: 'injury', label: 'Injury', asc: true, v: (o) => o.r.injury || '', f: (o, td) => injCell(o.r.injury, td), w: 140 },
      { k: 'lo', label: 'Low', g: 'g1', v: (o) => o.lo, f: (o) => (isNaN(o.lo) ? '—' : f1(o.lo)) },
      { k: 'pts', label: 'Proj PPR', g: 'g1', v: (o) => o.pts, f: (o) => h('span', null, bar(o.lo, o.pts, o.hi, o.r.act_ppr, max, o.r.pos), h('span', { class: 'pv' }, f1(o.pts))), w: 230 },
      { k: 'hi', label: 'High', g: 'g1', v: (o) => o.hi, f: (o) => (isNaN(o.hi) ? '—' : f1(o.hi)) },
      { k: 'act', label: 'Actual', g: 'g2', v: nv('act_ppr'), f: n('act_ppr', f1) },
      { k: 'vs', label: '+/-', g: 'g2', v: nv('vs_proj'), f: (o, td) => { const v = num(o.r.vs_proj, NaN); if (isNaN(v)) return '—'; td.classList.add(v >= 0 ? 'good' : 'bad'); return sg1(v); } },
      { k: 'team', label: 'Team Pts', v: nv('implied_pts'), f: n('implied_pts', f1) },
      { k: 'py', label: 'Pass Yds', div: true, v: nv('pass_yds'), f: n('pass_yds') },
      { k: 'apy', label: 'Act', g: 'g2', v: nv('act_pass_yds'), f: n('act_pass_yds') },
      { k: 'ry', label: 'Rush Yds', v: nv('rush_yds'), f: n('rush_yds') },
      { k: 'ary', label: 'Act', g: 'g2', v: nv('act_rush_yds'), f: n('act_rush_yds') },
      { k: 'cy', label: 'Rec Yds', v: nv('rec_yds'), f: n('rec_yds') },
      { k: 'acy', label: 'Act', g: 'g2', v: nv('act_rec_yds'), f: n('act_rec_yds') },
      { k: 'rec', label: 'Rec', v: nv('receptions'), f: n('receptions', f1) },
      { k: 'arec', label: 'Act', g: 'g2', v: nv('act_receptions'), f: n('act_receptions') },
      { k: 'sp', label: 'Start %', div: true, v: nv('start_pct'), f: (o, td) => { if (!has(o.r.start_pct)) return '—'; td.classList.add(confCls(o.r.confidence)); return Math.round(num(o.r.start_pct)) + '%'; } },
      { k: 'model', label: 'Model', v: nv('proj_model'), f: n('proj_model', f1) },
      { k: 'exp', label: 'Expert Rk', v: (o) => -num(o.r.expert_rank, NaN), f: (o) => (has(o.r.expert_rank) ? o.r.pos + f0(num(o.r.expert_rank)) : '—') },
      { k: 'xfp', label: 'xFP', v: nv('xfp'), f: n('xfp', f1) },
      { k: 'game', label: 'Game', asc: true, v: (o) => o.r.status, f: (o, td) => { if (o.r.status === 'Inactive' || o.r.status === 'Did not play') td.classList.add('bad'); return o.r.status; } }
    ];
    wst.empty = 'No players match these filters this week. Inactive players appear once game-day inactives are posted.';
    rkTable($('#wr-table'), cols, rows, wst, RENDER.projections);
    $('#wr-note').textContent = 'Projections in ' + lg.league + ' scoring (' + scoringText(lg) + '); Actual and +/- are standard PPR. Shaded rows are your players. '
      + (wst.week <= RE_WEEKS ? 'Week ' + wst.week + ' was re-created after the fact by models trained only on earlier games.' : 'White tick on a bar = actual points.');
  };

  // ---------------------------------------------------------------- REST OF SEASON
  const rst = { week: null, pos: 'All', status: 'All', q: '', sort: 'pts', dir: -1 };
  $('#rr-q').addEventListener('input', (e) => { rst.q = e.target.value.toLowerCase(); RENDER.ros(); });
  function rosWeeks() {
    const live = D('ros').data, curW = num(live[0] && live[0].from_week);
    const H = D('ros_history');
    const hw = H.status === 'ok' ? [...new Set(H.data.map((r) => +r.week))] : [];
    const weeks = [...new Set(hw.concat([curW]))].filter((w) => w > 0).sort((a, b) => a - b);
    const re = new Set(H.status === 'ok' ? H.data.filter((r) => r.source === 're-created').map((r) => +r.week) : []);
    return { curW, weeks, re };
  }
  function rosRowsFor(w, curW) {
    if (w === curW) return D('ros').data;
    return D('ros_history').data.filter((r) => +r.week === w);
  }
  // season-to-date PPR points through the week before w, and the rank they give at each position
  function seasonRanks(w) {
    const P = D('projections').data, pts = new Map(), g = new Map(), pos = new Map();
    P.forEach((r) => { if (+r.week < w && has(r.act_ppr)) { pts.set(r.key, (pts.get(r.key) || 0) + num(r.act_ppr)); pos.set(r.key, r.pos); g.set(r.key, (g.get(r.key) || 0) + 1); } });
    const by = {}; pts.forEach((v, k) => (by[pos.get(k)] = by[pos.get(k)] || []).push([k, v]));
    const rank = new Map(); Object.values(by).forEach((l) => l.sort((a, b) => b[1] - a[1]).forEach(([k], i) => rank.set(k, i + 1)));
    return { rank, pts, g };
  }
  RENDER.ros = () => {
    const lg = league();
    if (!lg || !ok('ros', 'projections')) { $('#rr-table').tBodies[0].replaceChildren(h('tr', null, h('td', { class: 'empty' }, 'Loading…'))); return; }
    const { curW, weeks, re } = rosWeeks();
    if (rst.week == null || !weeks.includes(rst.week)) rst.week = curW;
    weekTabs($('#rr-weeks'), weeks, rst.week, (w) => re.has(w), (w) => { rst.week = w; RENDER.ros(); });
    posTabs($('#rr-pos'), rst.pos, (p) => { rst.pos = p; RENDER.ros(); });
    toggles($('#rr-status'), ['All', 'Healthy', 'Rostered', 'Free agents'], rst.status, (t) => { rst.status = t; RENDER.ros(); });
    const rows0 = rosRowsFor(rst.week, curW);
    const L = leagueRos(rows0, lg, 'w' + rst.week);
    const prevW = weeks.filter((w) => w < rst.week).pop();
    const Lp = prevW != null ? leagueRos(rosRowsFor(prevW, curW), lg, 'w' + prevW) : null;
    const SR = seasonRanks(rst.week), mine = mineKeys(lg), owned = rosteredKeys(lg);
    const live = new Map(D('ros').data.map((r) => [r.key, r]));
    const list = [...L.values()].map((o) => Object.assign(o, { _mine: mine.has(o.r.key), cr: SR.rank.get(o.r.key), prev: Lp && Lp.get(o.r.key) ? Lp.get(o.r.key).rank : null, lv: live.get(o.r.key) || o.r }));
    const max = d3.max(list, (o) => o.hi) || 1;
    const rows = list.filter((o) => (rst.pos === 'All' || o.r.pos === rst.pos)
      && (rst.status === 'All' || (rst.status === 'Healthy' ? !/IR|reserve|Out/i.test(String(o.r.status) + ' ' + String(o.r.injury)) : rst.status === 'Rostered' ? owned.has(o.r.key) : !owned.has(o.r.key)))
      && (!rst.q || (o.r.key).toLowerCase().includes(rst.q)));
    const spg = (o) => (SR.g.get(o.r.key) ? SR.pts.get(o.r.key) / SR.g.get(o.r.key) : (rst.week === curW ? num(o.r.season_ppg, NaN) : NaN));
    const cols = [
      { k: 'rank', label: 'Pos Rank', html: 'Pos<br>Rank', hc: 'sm', v: (o) => (rst.pos === 'All' ? o.pts : -o.rank), f: (o) => o.r.pos + o.rank, w: 56 },
      { k: 'player', label: 'Player', l: true, asc: true, v: (o) => o.r.key, f: (o) => o.r.key, w: 250 },
      { k: 'pos', label: 'Pos', asc: true, v: (o) => o.r.pos, f: (o) => chip(o.r.pos, o.r.pos, 'sm') },
      { k: 'injury', label: 'Injury', asc: true, v: (o) => o.r.injury || '', f: (o, td) => injCell(o.r.injury || (/IR|reserve/i.test(o.r.status) ? o.r.status : ''), td), w: 140 },
      { k: 'lo', label: 'Low', g: 'g1', v: (o) => o.lo, f: (o) => f0(o.lo) },
      { k: 'pts', label: 'ROS PPR', g: 'g1', v: (o) => o.pts, f: (o) => h('span', null, bar(o.lo, o.pts, o.hi, null, max, o.r.pos), h('span', { class: 'pv' }, f1(o.pts))), w: 230 },
      { k: 'hi', label: 'High', g: 'g1', v: (o) => o.hi, f: (o) => f0(o.hi) },
      { k: 'cr', label: 'Current rk', g: 'g2', v: (o) => (o.cr == null ? NaN : -o.cr), f: (o) => (o.cr == null ? '—' : o.r.pos + o.cr) },
      { k: 'pm', label: '+/-', g: 'g2', v: (o) => (o.cr == null ? NaN : o.cr - o.rank), f: (o, td) => (o.cr == null ? '—' : updn(o.cr - o.rank, td)) },
      { k: 'spg', label: 'Season PPG', v: (o) => spg(o), f: (o) => (isNaN(spg(o)) ? '—' : f1(spg(o))) },
      { k: 'mr', label: 'Model Rk', div: true, v: (o) => -o.mrank, f: (o) => o.r.pos + o.mrank },
      { k: 'er', label: 'Expert Rk', v: (o) => -num(o.r.expert_rank, NaN), f: (o) => (has(o.r.expert_rank) ? o.r.pos + f0(num(o.r.expert_rank)) : '—') },
      { k: 'bye', label: 'Bye', v: (o) => num(o.lv.bye_week, NaN), f: (o) => (has(o.lv.bye_week) && num(o.lv.bye_week) >= rst.week ? 'Wk ' + f0(num(o.lv.bye_week)) : '—') },
      { k: 'chg', label: 'Wk chg', div: true, v: (o) => (o.prev == null ? NaN : o.prev - o.rank), f: (o, td) => (o.prev == null ? '—' : updn(o.prev - o.rank, td)) },
      { k: 'sp', label: 'Start %', v: (o) => num(o.r.ros_start_pct, NaN), f: (o, td) => { if (!has(o.r.ros_start_pct)) return '—'; td.classList.add(confCls(o.r.ros_confidence)); return Math.round(num(o.r.ros_start_pct)) + '%'; } },
      { k: 'ppg', label: 'Per game', v: (o) => o.ppg, f: (o) => f1(o.ppg) },
      { k: 'gl', label: 'Games', v: (o) => num(o.r.games_left, NaN), f: (o) => f0(num(o.r.games_left)) },
      { k: 'tw', label: 'This wk', v: (o) => num(o.r.this_week, NaN), f: (o) => (has(o.r.this_week) ? f1(num(o.r.this_week) + adj(o.lv, lg)) : '—') },
      { k: 'oa', label: 'Opp. allowed', v: (o) => num(o.r.opp_allowed, NaN), f: (o) => (has(o.r.opp_allowed) ? d3.format('.2f')(num(o.r.opp_allowed)) : '—') }
    ];
    rkTable($('#rr-table'), cols, rows, rst, RENDER.ros);
    const H = D('ros_history');
    $('#rr-note').textContent = 'Weeks ' + rst.week + '–' + num(rows0[0] && rows0[0].through_week, 17) + ' in ' + lg.league + ' scoring. Current rk: his rank on points so far'
      + (rst.week > 1 ? ' (through Week ' + (rst.week - 1) + ')' : '') + '; +/- = places the ROS ranking expects him to climb. Wk chg: ROS rank vs the week before. '
      + (re.has(rst.week) ? 'Week ' + rst.week + ' was re-created after the fact.' : (H.status === 'ok' && weeks.length < 3 ? 'Earlier weeks are being re-created and will appear as tabs.' : ''));
  };

  // ---------------------------------------------------------------- RANK HISTORY
  const hst = { pos: 'RB', span: 'week', sel: [] };
  RENDER.history = () => {
    const lg = league();
    if (!lg || !ok('ros', 'projections')) return;
    const { curW, weeks, re } = rosWeeks();
    const byW = new Map(weeks.map((w) => [w, leagueRos(rosRowsFor(w, curW), lg, 'w' + w)]));
    const mine = mineKeys(lg), N = { QB: 36, RB: 60, WR: 80, TE: 36 }[hst.pos];
    const posBox = $('#rh-pos');
    if (!posBox.childElementCount) POS.forEach((p) => posBox.append(h('button', { class: 'nf-chip', onclick: () => { hst.pos = p; hst.sel = []; RENDER.history(); } }, p)));
    $$('button', posBox).forEach((b) => b.setAttribute('aria-pressed', String(b.textContent === hst.pos)));
    const spanBox = $('#rh-span');
    const first = weeks[0], prev = weeks[weeks.length - 2];
    spanBox.replaceChildren(...[['week', 'Since last week'], ['season', 'Since Week ' + first]].map(([k, t]) => h('button', { class: 'nf-chip', 'aria-pressed': String(hst.span === k), onclick: () => { hst.span = k; RENDER.history(); } }, t)));
    const now = byW.get(curW), thenW = hst.span === 'week' ? prev : first, then = thenW != null ? byW.get(thenW) : null;
    const players = [...now.values()].filter((o) => o.r.pos === hst.pos);
    const path = (key) => weeks.map((w) => { const o = byW.get(w).get(key); return o ? { w, rank: o.rank } : null; }).filter(Boolean);
    const moves = then && thenW !== curW ? players.map((o) => { const t = then.get(o.r.key); return t ? { o, t: t.rank, n: o.rank, d: t.rank - o.rank } : null; })
      .filter((x) => x && (x.t <= N || x.n <= N)) : [];
    const fill = (tb, list) => {
      tb.replaceChildren(...list.map((x) => {
        const tr = h('tr', { class: 'rh-pick' + (hst.sel.includes(x.o.r.key) ? ' on' : '') + (mine.has(x.o.r.key) ? ' nf-picked' : ''), onclick: () => { const i = hst.sel.indexOf(x.o.r.key); if (i >= 0) hst.sel.splice(i, 1); else { hst.sel.push(x.o.r.key); if (hst.sel.length > 6) hst.sel.shift(); } RENDER.history(); } },
          h('td', { class: 'nf-l' }, x.o.r.key), h('td', null, hst.pos + x.t), h('td', null, hst.pos + x.n), h('td', { class: x.d > 0 ? 'up' : 'dn' }, (x.d > 0 ? '▲ ' : '▼ ') + Math.abs(x.d)), h('td', null, f1(x.o.pts)));
        return tr;
      }));
      if (!list.length) tb.replaceChildren(h('tr', null, h('td', { class: 'nf-l nf-muted', colspan: 5 }, weeks.length < 2 ? 'Needs two weeks of history — earlier weeks are being re-created.' : 'No moves.')));
    };
    fill($('#rh-up tbody'), moves.filter((x) => x.d > 0).sort((a, b) => b.d - a.d || a.n - b.n).slice(0, 12));
    fill($('#rh-dn tbody'), moves.filter((x) => x.d < 0).sort((a, b) => a.d - b.d || a.n - b.n).slice(0, 12));
    $('#rh-up-h').textContent = 'Biggest risers · ' + hst.pos + (thenW != null ? ' since Week ' + thenW : '');
    $('#rh-dn-h').textContent = 'Biggest fallers · ' + hst.pos + (thenW != null ? ' since Week ' + thenW : '');
    if (!hst.sel.length) hst.sel = moves.filter((x) => x.d > 0).sort((a, b) => b.d - a.d).slice(0, 3).map((x) => x.o.r.key)
      .concat(moves.filter((x) => x.d < 0).sort((a, b) => a.d - b.d).slice(0, 2).map((x) => x.o.r.key));
    if (!hst.sel.length) hst.sel = players.filter((o) => mine.has(o.r.key)).slice(0, 5).map((o) => o.r.key);
    // chart
    const box = $('#rh-chart'), lgd = $('#rh-legend');
    box.replaceChildren(); lgd.replaceChildren();
    const W = box.clientWidth || 1600, Hh = 360, m = { l: 56, r: 150, t: 14, b: 34 };
    const series = hst.sel.map((k) => ({ k, p: path(k) })).filter((s) => s.p.length);
    const maxR = Math.max(12, d3.max(series.flatMap((s) => s.p), (d) => d.rank) || 12);
    const x = d3.scalePoint(weeks, [m.l, W - m.r]).padding(0.3), y = d3.scaleLinear([1, maxR + 1], [m.t, Hh - m.b]);
    const svg = d3.select(box).append('svg').attr('viewBox', `0 0 ${W} ${Hh}`);
    y.ticks(6).filter((t) => t >= 1).forEach((t) => { svg.append('line').attr('x1', m.l).attr('x2', W - m.r).attr('y1', y(t)).attr('y2', y(t)).attr('stroke', 'rgba(0,0,0,.12)'); svg.append('text').attr('x', m.l - 10).attr('y', y(t) + 5).attr('text-anchor', 'end').text(hst.pos + t); });
    weeks.forEach((w) => svg.append('text').attr('x', x(w)).attr('y', Hh - 8).attr('text-anchor', 'middle').text('Wk ' + w + (re.has(w) ? '*' : '')));
    const colors = ['#4a0600', '#13279c', '#154f02', '#8a7c10', '#4d00ad', '#2b6f7a'];
    series.forEach((s, i) => {
      const c = colors[i % colors.length];
      svg.append('path').attr('d', d3.line().x((d) => x(d.w)).y((d) => y(d.rank))(s.p)).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 3.5);
      s.p.forEach((d) => svg.append('circle').attr('cx', x(d.w)).attr('cy', y(d.rank)).attr('r', 5).attr('fill', c).append('title').text(s.k + ' · Week ' + d.w + ': ' + hst.pos + d.rank));
      s.c = c;
      lgd.append(h('span', null, h('span', { class: 'nf-sw', style: 'background:' + c }), s.k));
    });
    // end labels, nudged apart so they never overlap
    const labs = series.map((s) => { const l = s.p[s.p.length - 1]; return { s, x: x(l.w) + 10, y: y(l.rank) + 5 }; }).sort((a, b) => a.y - b.y);
    labs.forEach((l, i) => { if (i && l.y - labs[i - 1].y < 20) l.y = labs[i - 1].y + 20; });
    labs.forEach((l) => svg.append('text').attr('x', l.x).attr('y', l.y).style('fill', l.s.c).text(l.s.k.replace(/ \(.*\)/, '')));
    if (weeks.length < 2) svg.append('text').attr('x', W / 2).attr('y', Hh / 2).attr('text-anchor', 'middle').text('History starts this week — Weeks 1 to ' + (curW - 1) + ' are being re-created.');
    // every week table
    const thr = $('#rh-all thead tr'); thr.replaceChildren(h('th', { class: 'nf-l' }, 'Player'), ...weeks.map((w) => h('th', null, 'Wk ' + w + (re.has(w) ? '*' : ''))), h('th', null, 'Change'));
    const top = players.sort((a, b) => a.rank - b.rank).slice(0, N);
    $('#rh-all tbody').replaceChildren(...top.map((o) => {
      const p = path(o.r.key), firstR = p.length ? p[0].rank : null, d = firstR != null ? firstR - o.rank : 0;
      return h('tr', { class: 'rh-pick' + (mine.has(o.r.key) ? ' nf-picked' : ''), onclick: () => { if (!hst.sel.includes(o.r.key)) { hst.sel.push(o.r.key); if (hst.sel.length > 6) hst.sel.shift(); } RENDER.history(); $('#rh-chart').scrollIntoView({ behavior: 'smooth', block: 'nearest' }); } },
        h('td', { class: 'nf-l' }, o.r.key), ...weeks.map((w) => { const q = byW.get(w).get(o.r.key); return h('td', null, q ? String(q.rank) : '—'); }),
        h('td', { class: d > 0 ? 'nf-good' : d < 0 ? 'nf-bad' : 'nf-muted' }, d ? (d > 0 ? '▲ ' : '▼ ') + Math.abs(d) : '0'));
    }));
  };

  // ================================================================ TRADE ANALYZER
  window.foOnTrade = () => { if (cur === 'trade') RENDER.trade(); else if (cur === 'league') RENDER.league(); else dirty.add('trade'); };
  const teamDD = {};
  let lastX = null;   // the box just checked: only that X is drawn in
  function tradeSide(side, T) {
    const box = $('#ta-side-' + side), tid = side === 'mine' ? T.st.mine : T.st.theirs;
    const picked = side === 'mine' ? T.st.give : T.st.get;
    if (!box.dataset.built) {
      box.dataset.built = '1';
      box.append(
        h('div', { class: 'paper ta-head', style: 'position:absolute' }, h('div', { class: 'nm' }), h('div', { class: 'rec' }), h('div', { class: 'pw', style: 'left:600px' })),
        h('div', { class: 'dd ta-dd', style: 'left:23px;top:13px;width:765px;--hc:54px' }, h('div', { class: 'box', style: 'background:transparent;box-shadow:none' }, h('div', { class: 'list' }), h('span', { class: 'capbox', style: 'right:6px;top:4px' }), h('img', { class: 'caret', src: 'assets/caret.svg', alt: '', style: 'width:32px;height:37px' }))),
        h('div', { class: 'paper', style: 'left:23px;top:93px;width:765px;height:662px' },
          h('div', { class: 'gt th', style: 'left:0;right:0;top:0;height:64px' }),
          h('div', { class: 'sc', style: 'left:0;right:0;top:64px;bottom:0' }, h('div', { class: 'gt tv', style: 'left:0;right:0;top:0' }))),
        h('div', { class: 'arrowbar slim', style: 'left:790px;top:157px;height:598px' }));
      const dd = box.querySelector('.ta-dd');
      teamDD[side] = dropdown(dd, { rowH: 44, top: 4, bottom: 14, label: side === 'mine' ? 'Your team' : 'Trade partner', onPick: (id) => window.foTradeData.setTeam(side, id) });
      dd.addEventListener('fo-close', () => { dd.classList.remove('open'); });
      dd.querySelector('.box').style.height = 'var(--hc)';
      arrowbar(box.querySelector('.arrowbar'), box.querySelector('.sc'));
      box.append(h('style', null, `#ta-side-${side} .ta-dd.open .box{height:var(--ho)!important;background:#cfcfcf!important;box-shadow:var(--boxshadow)!important}#ta-side-${side} .ta-dd .list{opacity:0;transition:opacity .2s;font-size:32px;padding:4px 70px 10px 16px}#ta-side-${side} .ta-dd .opt{height:44px;line-height:44px}#ta-side-${side} .ta-dd.open .list{opacity:1}#ta-side-${side} .ta-dd .caret{right:10px;top:8px}#ta-side-${side} .ta-dd.open .caret{top:calc(var(--ho) - 46px)}#ta-side-${side} .ta-dd .capbox{width:44px;height:46px}#ta-side-${side} .ta-dd.open .capbox{top:calc(var(--ho) - 50px)!important}`));
    }
    const teams = T.teams.filter((t) => side === 'mine' || t.id !== T.st.mine);
    teamDD[side].setItems(teams.map((t) => ({ id: t.id, text: t.name + (t.me ? ' (you)' : '') })), tid);
    const S = D('league_standings'), rec = S.status === 'ok' ? S.data.find((r) => String(r.league_id) === T.st.lid && String(r.team_id) === tid) : null;
    const nm = box.querySelector('.nm'); nm.textContent = T.name(tid);
    const recEl = box.querySelector('.rec');
    recEl.textContent = rec ? 'Current Record: ' + num(rec.wins) + ' - ' + num(rec.losses) + (num(rec.ties) ? ' - ' + num(rec.ties) : '') : '';
    requestAnimationFrame(() => { recEl.style.left = Math.min(470, nm.offsetLeft + nm.offsetWidth + 14) + 'px'; });
    const sc = T.scB.get(tid), pw = box.querySelector('.pw');
    pw.textContent = sc == null ? '' : String(sc); pw.style.color = sc >= 50 ? '#8a7c10' : 'var(--red)'; pw.title = 'Power ranking score: strongest lineup in the league = 100, weakest = 0';
    // roster table (Site 8 adjustment: bigger rows, numbers in Times)
    const X = { name: 65, flag: 225, pos: 246, ros: 318, exp: 374, trd: 446, div: 517, tv: 540, act: 610, vm: 652 };
    box.querySelector('.th').replaceChildren(
      h('div', { style: 'position:absolute;left:65px;top:6px;font-size:40px' }, 'Player'), h('div', { style: 'position:absolute;left:235px;top:6px;font-size:40px' }, 'Pos'),
      h('div', { style: 'position:absolute;left:310px;width:200px;top:2px;text-align:center;font-size:21px;color:var(--red)' }, 'Position Rankings'),
      h('div', { style: 'position:absolute;left:530px;width:230px;top:2px;text-align:center;font-size:21px;color:var(--red)' }, 'Trade Value'),
      ...[[X.ros, 'ROS'], [X.exp - 4, 'Experts'], [X.trd - 4, 'Trades'], [531, 'Trade'], [594, 'Actual'], [663, 'vs Market']].map(([x, t]) => h('div', { style: `position:absolute;left:${x}px;top:30px;font-size:21px` }, t)),
      h('div', { style: `position:absolute;left:${X.div}px;top:0;width:2px;height:662px;background:var(--divider)` }));
    const order = { QB: 0, RB: 1, WR: 2, TE: 3 };
    const ps = T.ctx.players.filter((p) => p.team_id === tid && p.kind === 'skill').sort((a, b) => order[a.pos] - order[b.pos] || b.exp - a.exp || a.name.localeCompare(b.name));
    const tv = box.querySelector('.tv'), P = 39;
    tv.style.height = (ps.length * P + 8) + 'px';
    const vcls = { Undervalued: 'good', Overvalued: 'bad' };
    tv.replaceChildren(...ps.map((p, i) => {
      const on = picked.has(p.key), lab = p.value || p.valueT;
      const r = h('div', { class: 'r' + (on ? ' sel' : ''), style: `top:${i * P + 2}px;height:${P}px;font-size:21px;${i % 2 ? 'background:rgba(0,0,0,.05)' : ''}` });
      const cbx = h('button', { class: 'cb' + (on ? ' on' : '') + (on && lastX === p.key ? ' draw' : ''), style: 'left:10px;top:4.5px', 'aria-label': (on ? 'Remove ' : 'Add ') + p.name, 'aria-pressed': String(on), onclick: () => { lastX = on ? null : p.key; window.foTradeData.toggle(side, p.key); } });
      if (on) cbx.innerHTML = '<svg viewBox="0 0 30 30" aria-hidden="true"><path class="s1" pathLength="1" d="M5.5 6.5 C 10 11, 15 16.5, 24.5 24"/><path class="s2" pathLength="1" d="M23.5 5.5 C 18 10, 12.5 16, 6 24.5"/></svg>';
      r.append(cbx);
      r.append(h('div', { class: 'c nm', style: `left:${X.name}px;max-width:158px;line-height:${P}px` }, p.name));
      const ir = p.slot === 'IR' || /^IR|PUP|NFI/.test(p.status);
      if (ir) r.append(h('img', { src: 'assets/caduceus.png', alt: 'IR', title: p.status || 'IR', style: `position:absolute;left:${X.flag - 4}px;top:9px;width:22px` }));
      else if (p.status) r.append(h('div', { class: 'c', style: `left:${X.flag}px;color:var(--red)`, title: p.status }, p.status === 'O' ? 'O' : p.status === 'D' ? 'D' : 'Q'));
      r.append(h('div', { class: 'c', style: `left:${X.pos}px` }, chip(p.pos, p.pos, 'xs')));
      r.append(cell(X.ros, p.rosKey ? p.rank : '—'), cell(X.exp, p.market || '—', p.market ? '' : 'muted'), cell(X.trd, p.tradeRank || '—', p.tradeRank ? '' : 'muted'));
      r.append(cell(X.tv, p.rosKey ? f0(p.trade) : '—', p.tsrc !== 'trade market' ? 'muted' : ''), cell(X.act, p.rosKey ? f0(p.actual) : '—'));
      r.append(cell(X.vm, lab || '—', vcls[lab] || (lab ? '' : 'muted')));
      r.title = p.name + (p.rosKey ? ' · ' + f1(p.exp) + ' pts/wk · trade value ' + f0(p.trade) + ' (' + p.tsrc + ') · actual ' + f0(p.actual) : '');
      return r;
    }));
  }
  function verdictMine(T) {
    const { res, price, help } = T;
    if (!res.active) return 'Check players on both rosters to build a trade.';
    const it = (p) => p.name + ' (' + f0(p.trade) + ' / ' + f0(p.actual) + ')';
    const H = help[T.st.mine];
    return `<b>You give</b> ${res.giveP.map(it).join(', ') || 'nothing'}. <b>You get</b> ${res.getP.map(it).join(', ') || 'nothing'}. `
      + `(trade / actual value) By our projections you net ${sg1(price.A.netMine)} in value and ${sg1(H.d)} lineup pts a week; `
      + (price.T.fair ? 'the market calls it fair.' : price.T.netMine > 0 ? 'the market says you win it.' : 'the market says you overpay.');
  }
  function verdictTheirs(T) {
    const { res, price, help, ctx, st } = T;
    if (!res.active) return 'Their side fills in once players are checked.';
    const H = help[st.theirs], look = price.T.netTheirs, band = price.T.band;
    let s;
    if (look >= band) s = '<b>Likely to accept</b>: by trade-market value they come out ' + sg1(look) + ' ahead.';
    else if (look > -band) s = '<b>Could go either way</b>: about even by market value (' + sg1(look) + ').';
    else {
      const need = -look;
      const pool = ctx.players.filter((p) => p.team_id === st.mine && p.kind === 'skill' && p.rosKey && !st.give.has(p.key) && p.trade > 0).sort((a, b) => a.trade - b.trade);
      const add = pool.find((p) => p.trade >= need) || pool[pool.length - 1];
      s = '<b>Unlikely</b>: by market value they give up ' + f0(need) + ' more than they get.' + (add ? ' Adding <b>' + add.name + '</b> (trade value ' + f0(add.trade) + ') would ' + (add.trade >= need - band ? 'even it out.' : 'close part of the gap.') : '');
    }
    return s + ' Our model: their lineup ' + sg1(H.d) + ' pts a week.';
  }
  const pct = (p) => (p == null ? '—' : p >= 0.995 ? '>99%' : p > 0 && p < 0.005 ? '<1%' : Math.round(p * 100) + '%');
  function tradeCenter(T) {
    const c = $('#ta-center');
    // one block per team, side by side under its roster
    const block = (tid, x0) => {
      const H = T.help[tid], sb = H.sb, sa = H.sa, act = T.res.active;
      const state = !act ? '' : H.d > 0.05 ? 'Winning' : H.d < -0.05 ? 'Losing' : 'Even';
      const rb = T.rankOf(T.res.before, tid), ra = T.rankOf(T.res.after, tid);
      const recOf = (s) => (s ? Math.round(s.wins) + ' - ' + (Math.round(s.games) - Math.round(s.wins)) : '—');
      const tone = (d) => (d > 0.0001 ? 'good' : d < -0.0001 ? 'bad' : '');
      const vb = (x, y2, title, b, a, aCls) => h('div', { class: 'vbox', style: `left:${x}px;top:${y2}px;width:190px;height:118px` },
        h('div', { class: 't' }, title), h('div', { class: 'ba', style: 'top:40px' }, h('div', null, h('span', null, 'Before'), h('span', { class: 'v' }, b)), h('div', null, h('span', null, 'After'), h('span', { class: 'v ' + (aCls || '') }, a))));
      const meter = (x, v, lab) => {
        const col = `color-mix(in srgb, var(--gold) ${Math.max(0, Math.min(100, (v - 15) * 2))}%, var(--red))`;
        return h('div', { class: 'meter', style: `left:${x}px;top:52px;height:330px;width:48px` }, h('s', null, lab), h('i', { style: `height:${Math.max(3, v) / 100 * 330}px;background:${col}` }), h('b', null, String(v)));
      };
      const sA = T.scA.get(tid), sB = T.scB.get(tid), recA = recOf(sa), recB = recOf(sb);
      return [
        h('div', { style: `position:absolute;left:${x0 + 28}px;top:16px;font-size:40px;white-space:nowrap;max-width:520px;overflow:hidden;text-overflow:ellipsis` }, T.name(tid)),
        h('div', { class: 'state ' + (state === 'Winning' ? 'good' : state === 'Losing' ? 'bad' : 'muted'), style: `left:${x0 + 600}px;top:14px` }, state),
        h('div', { class: 'paper', style: `left:${x0 + 20}px;top:82px;width:820px;height:428px;--p:1` },
          vb(18, 18, 'Pts / week', f1(H.b), f1(H.a), tone(H.d)),
          vb(222, 18, 'Playoff Predictor', sb ? pct(sb.playoff) : '—', sa ? pct(sa.playoff) : '—', sb && sa ? tone(sa.playoff - sb.playoff) : ''),
          vb(18, 154, 'Team Power Ranking', ord(rb), ord(ra), tone(rb - ra)),
          vb(222, 154, 'Proj. Record', recB, recA, sa && sb ? tone(sa.wins - sb.wins) : ''),
          vb(18, 290, 'Title Odds', sb ? pct(sb.title) : '—', sa ? pct(sa.title) : '—', sb && sa ? tone(sa.title - sb.title) : ''),
          vb(222, 290, 'First-round Bye', sb ? pct(sb.bye) : '—', sa ? pct(sa.bye) : '—', sb && sa ? tone(sa.bye - sb.bye) : ''),
          h('div', { class: 'verdict', style: 'left:430px;top:16px;width:220px;height:396px' }, h('div', { style: 'font-size:22px;margin-bottom:6px' }, 'Verdict')),
          meter(672, sB, 'Before'), meter(750, sA, 'After'))
      ];
    };
    c.replaceChildren(...block(T.st.mine, 0), ...block(T.st.theirs, 878),
      h('div', { style: 'position:absolute;left:868px;top:20px;width:3px;height:500px;background:var(--divider)' }));
    const vs = $$('.verdict', c);
    const fillV = (el, html) => { const d = h('div'); d.innerHTML = html; el.append(d); shrink(el); };
    fillV(vs[0], verdictMine(T)); fillV(vs[1], verdictTheirs(T));
    c.append(h('div', { style: 'position:absolute;left:40px;right:40px;top:522px;display:flex;justify-content:space-between;font-size:21px' },
      h('a', { style: 'cursor:pointer;color:var(--red)', onclick: () => window.foTradeData.clear() }, T.res.active ? 'Clear trade' : ''),
      h('a', { 'data-go': 'tradedetails', style: 'cursor:pointer;color:var(--red)' }, 'Full analysis: Trade Details ›')));
  }
  // verdict text: starts at 19px and gives way to 16px (12 pt) at the least, never smaller
  function shrink(el) { let fs = 19; el.style.fontSize = fs + 'px'; while (el.scrollHeight > el.clientHeight + 1 && fs > 16) { fs -= 0.5; el.style.fontSize = fs + 'px'; } if (el.scrollHeight > el.clientHeight + 1) el.style.overflowY = 'auto'; }
  function tradeLineup(side, T) {
    const box = $('#ta-lu-' + side), tid = side === 'mine' ? T.st.mine : T.st.theirs;
    const B = T.res.before.get(tid).slots, A = T.res.after.get(tid).slots;
    const n = B.length, pitch = Math.min(28, 186 / Math.max(1, n));
    const rows = [h('div', { class: 'r', style: 'top:4px;height:30px;font-size:24px' }, cell(14, 'Slot'), cell(70, 'Player'), cell(240, 'Points'), cell(318, 'Rank'), cell(405, 'Player'), cell(590, 'Points'), cell(670, 'Rank'))];
    const SO = (l) => { const b = l.replace(/\d+$/, ''); const k = POS.indexOf(b); return k >= 0 ? k : b === 'FLEX' ? 4 : 5; };
    B.map((s, i) => ({ s, i })).sort((a, b) => SO(a.s.label) - SO(b.s.label) || a.i - b.i).forEach(({ s, i }, j) => {
      const pb = s.p, pa = A[i].p, changed = (pb?.key || '') !== (pa?.key || '');
      const r0 = T.slotRank(T.res.before, tid, i), r1 = T.slotRank(T.res.after, tid, i);
      const lab = s.label.replace(/\d+$/, ''), cls = POS.includes(lab) ? lab : 'FLEX';
      rows.push(h('div', { class: 'r', style: `top:${40 + j * pitch}px;height:${pitch}px;font-size:19px;${j % 2 ? 'background:rgba(0,0,0,.05)' : ''}` },
        h('div', { class: 'c', style: 'left:12px' }, chip(lab === 'SUPERFLEX' ? 'SF' : cls === 'FLEX' ? 'FLEX' : lab, lab === 'SUPERFLEX' ? 'SF' : cls, 'xs')),
        h('div', { class: 'c nm', style: `left:70px;max-width:165px;line-height:${pitch}px` }, pb ? pb.name : '—'), cell(250, pb ? f1(pb.exp) : '—'), cell(322, ord(r0)),
        h('div', { class: 'c nm' + (changed ? ' gold' : ''), style: `left:405px;max-width:180px;line-height:${pitch}px` }, pa ? pa.name : '—'),
        cell(600, pa ? f1(pa.exp) : '—', changed ? 'gold' : ''), cell(674, ord(r1), r1 < r0 ? 'good' : r1 > r0 ? 'bad' : '')));
    });
    box.replaceChildren(
      h('div', { class: 'paper', style: 'left:23px;top:13px;width:765px;height:54px' }, h('div', { style: 'position:absolute;left:12px;top:0;font-size:46px' }, 'Lineup'),
        h('div', { style: 'position:absolute;right:16px;top:16px;font-size:22px;color:#3b3430;max-width:520px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis' }, T.name(tid) + ' · before → after')),
      h('div', { class: 'paper', style: 'left:23px;top:85px;width:765px;height:228px' }, h('div', { class: 'gt', style: 'inset:0' }, rows),
        h('div', { style: 'position:absolute;left:388px;top:0;width:3px;height:228px;background:var(--red)' })));
  }
  RENDER.trade = () => {
    const T = window.foTradeData, lg = league();
    if (!T || !lg || T.st.lid !== String(lg.league_id)) { $('#ta-center').replaceChildren(h('div', { class: 'empty' }, 'Loading the league…')); return; }
    tradeSide('mine', T); tradeSide('theirs', T); tradeCenter(T); tradeLineup('mine', T); tradeLineup('theirs', T);
    // the page is as long as the trade finder needs
    const f = $('.tp-finder'); if (f) $('.tp-inner').style.height = (f.offsetTop + f.offsetHeight + 60) + 'px';
  };

  // ================================================================ LEAGUE VIEW: standings, playoff and title odds, weekly matchups
  const lst = { week: null, sort: 'seed', dir: 1 };
  RENDER.league = () => {
    const T = window.foTradeData, lg = league();
    const SC = D('league_schedule'), S = D('league_standings');
    if (!T || !lg || T.st.lid !== String(lg.league_id) || SC.status !== 'ok' || S.status !== 'ok') { $('#lv-table').tBodies[0].replaceChildren(h('tr', null, h('td', { class: 'empty' }, 'Loading the league…'))); return; }
    const lid = String(lg.league_id), ss = T.ss, sim = ss ? ss.simB : null;
    const st = S.data.filter((r) => String(r.league_id) === lid);
    const sched = SC.data.filter((r) => String(r.league_id) === lid);
    const rank = new Map(st.slice().sort((a, b) => num(b.wins) + num(b.ties) / 2 - num(a.wins) - num(a.ties) / 2 || num(b.points_for) - num(a.points_for)).map((r, i) => [String(r.team_id), i + 1]));
    const strength = new Map([...T.res.before].map(([t, x]) => [t, x.v]));
    $('#lv-sub').textContent = st.length + ' teams · ' + String(lg.playoffs || '').replace(/\s*\(.*\)/, '');
    $('#lv-sub').title = (lg.playoffs || '') + ' · playoff and title odds from the season simulation (our lineup projections, the real schedule and results so far)';
    // ---- standings
    const rows = st.map((r) => {
      const t = String(r.team_id), s = sim ? sim.get(t) : null;
      return { t, name: r.team || T.name(t), w: num(r.wins), l: num(r.losses), ti: num(r.ties), pf: num(r.points_for), pa: num(r.points_against), seed: rank.get(t),
        pw: T.scB.get(t), prk: T.rankOf(T.res.before, t), wk: strength.get(t), po: s ? s.playoff : null, ch: s ? s.title : null, bye: s ? s.bye : null,
        pw2: s ? s.wins : null, g: s ? s.games : null, me: t === String(lg.my_team) };
    });
    const sorters = { seed: (r) => r.seed, name: (r) => r.name, rec: (r) => -(r.w + r.ti / 2), pf: (r) => -r.pf, pa: (r) => -r.pa, pw: (r) => -(r.pw ?? 0), wk: (r) => -(r.wk ?? 0), proj: (r) => -(r.pw2 ?? 0), po: (r) => -(r.po ?? 0), bye: (r) => -(r.bye ?? 0), ch: (r) => -(r.ch ?? 0) };
    const vf = sorters[lst.sort] || sorters.seed;
    rows.sort((a, b) => { const x = vf(a), y = vf(b); return (typeof x === 'string' ? x.localeCompare(y) : x - y) * lst.dir || a.seed - b.seed; });
    const byes = rows.some((r) => (r.bye || 0) > 0.001);
    const cols = [['seed', '#', ''], ['name', 'Team', 'l'], ['rec', 'Record', 'div'], ['pf', 'PF', ''], ['pa', 'PA', ''], ['pw', 'Power', 'div'], ['wk', 'Proj / wk', ''], ['proj', 'Proj. Record', ''], ['po', 'Playoffs', 'div g2'], ...(byes ? [['bye', 'Bye', 'g2']] : []), ['ch', 'Champion', 'g2']];
    const thead = $('#lv-table').tHead;
    thead.replaceChildren(h('tr', null, cols.map(([k, t, c]) => h('th', { class: c, 'aria-sort': lst.sort === k ? (lst.dir === 1 ? 'ascending' : 'descending') : null },
      h('button', { style: 'all:unset;cursor:pointer', onclick: () => { if (lst.sort === k) lst.dir *= -1; else { lst.sort = k; lst.dir = 1; } RENDER.league(); } }, t)))));
    const oddsCell = (p, col) => h('span', { class: 'odds' }, h('span', { class: 'bar' }, h('i', { style: `width:${Math.max(1.5, (p || 0) * 100)}%;--oc:${col}` })), h('span', { style: 'min-width:52px;text-align:right' }, pct(p)));
    const recStr = (w, l, t) => w + ' - ' + l + (t ? ' - ' + t : '');
    $('#lv-table').tBodies[0].replaceChildren(...rows.map((r) => h('tr', { class: r.me ? 'mine' : null },
      h('td', null, String(r.seed)), h('td', { class: 'l nmc', title: r.name }, r.name + (r.me ? '  (you)' : '')), h('td', { class: 'div' }, recStr(r.w, r.l, r.ti)),
      h('td', null, f1(r.pf)), h('td', null, f1(r.pa)),
      h('td', { class: 'div', style: `color:${(r.pw ?? 0) >= 50 ? '#6b5f0b' : '#4a0600'}` }, r.pw == null ? '—' : h('span', null, String(r.pw), h('small', { style: 'font-size:17px;color:#444;margin-left:6px' }, ord(r.prk)))),
      h('td', null, r.wk == null ? '—' : f1(r.wk)),
      h('td', null, r.pw2 == null ? '—' : recStr(Math.round(r.pw2), Math.round(r.g) - Math.round(r.pw2), 0)),
      h('td', { class: 'div g2' }, oddsCell(r.po, '#154f02')), byes ? h('td', { class: 'g2' }, oddsCell(r.bye, '#445ec2')) : null, h('td', { class: 'g2' }, oddsCell(r.ch, '#ab9d1d')))));
    // ---- weekly matchups
    const weeks = [...new Set(sched.map((m) => num(m.week)))].sort((a, b) => a - b);
    const live = sched.find((m) => m.status === 'in progress'), next = sched.filter((m) => m.status !== 'final').map((m) => num(m.week)).sort((a, b) => a - b)[0];
    const curW = live ? num(live.week) : next || weeks[weeks.length - 1];
    if (lst.week == null || !weeks.includes(lst.week)) lst.week = curW;
    const wb = $('#lv-weeks');
    wb.replaceChildren(...weeks.map((w) => h('button', { class: 'wk', 'aria-pressed': String(w === lst.week), style: 'min-width:66px;margin-right:8px', onclick: () => { lst.week = w; RENDER.league(); } }, (w === curW ? '• ' : '') + 'Wk ' + w)));
    const spB = ss ? ss.spB : null;
    const winP = (a, b) => {   // chance team a outscores team b in one week: our lineup strength and weekly spread
      const ma = strength.get(a), mb = strength.get(b); if (ma == null || mb == null) return null;
      const sa = spB && spB.get(a) ? spB.get(a).sd : 25, sb = spB && spB.get(b) ? spB.get(b).sd : 25;
      const z = (ma - mb) / Math.sqrt(sa * sa + sb * sb); return 0.5 * (1 + erf(z / Math.SQRT2));
    };
    const ms = sched.filter((m) => num(m.week) === lst.week);
    const mine = String(lg.my_team);
    $('#lv-m').replaceChildren(...ms.sort((a, b) => ((String(b.team1) === mine || String(b.team2) === mine) - (String(a.team1) === mine || String(a.team2) === mine))).map((m) => {
      const a = String(m.team1), b = String(m.team2), fin = m.status === 'final', inp = m.status === 'in progress';
      const pa = num(m.points1), pb = num(m.points2), ja = num(m.proj1), jb = num(m.proj2);
      let p = winP(a, b);
      if (inp && has(m.proj1) && has(m.proj2)) { const z = (ja - jb) / 18; p = 0.5 * (1 + erf(z / Math.SQRT2)); }
      const side = (t, pts, proj, r, won) => h('div', { class: 'tm' + (r ? ' r' : '') },
        h('b', { class: fin ? (won ? 'win' : 'lose') : '' }, T.name(t)),
        h('span', { class: 'sc2' }, fin || inp ? f1(pts) : f1(strength.get(t) ?? proj)),
        h('small', null, fin ? (won ? 'Won' : 'Lost') : inp ? 'Yahoo proj ' + f1(proj) : 'our proj · Yahoo ' + f1(proj)));
      const el = h('div', { class: 'lv-m' + (a === mine || b === mine ? ' mine' : '') },
        side(a, pa, ja, false, pa > pb), h('div', { class: 'vs' }, fin ? 'Final' : inp ? 'Live' : 'vs'), side(b, pb, jb, true, pb > pa));
      if (!fin && p != null) el.append(h('div', { class: 'wp', title: 'Win chance' }, h('i', { style: `width:${p * 100}%` })), h('div', { class: 'wpl' }, h('span', null, pct(p) + ' win'), h('span', null, pct(1 - p) + ' win')));
      return el;
    }));
  };
  // error function for win chances
  function erf(x) { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; }
  window.foOnTradeLeague = () => { if (cur === 'league') RENDER.league(); };

  // ================================================================ More pages: redraw charts when shown
  ['matchups', 'style', 'weather', 'experts', 'model', 'lineup', 'tradedetails', 'weekly', 'rosdetails'].forEach((k) => { RENDER[k] = () => {}; });
})();

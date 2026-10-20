/* Fantasy hub - stand-alone runtime for the dashboard page.
 *
 * The page was written for claude.ai's Dashboard type, which gives it a `dash` object
 * (datasets, calculated values, filters). This file provides the same calls on a plain
 * website: datasets are CSV files in data/, listed in data/datasets.json, which the
 * GitHub job rewrites whenever the model updates.
 */
(function () {
  const V = Date.now();                 // cache-buster: always fetch the newest files
  const sets = {};                      // id -> {status, data, columns, message}
  const calcs = {};                     // id -> {inputs, fn, title, description}
  const calcMemo = {};                  // id -> {ver, res}
  const subs = [];
  let catalog = null, version = 0, params = {}, defaults = {};

  const fire = () => {
    version++;
    for (const fn of subs) {
      try { fn(); } catch (e) { console.error(e); }
    }
  };
  let pending = null;
  const soon = () => { if (!pending) pending = setTimeout(() => { pending = null; fire(); }, 30); };

  function load(id) {
    if (sets[id]) return sets[id];
    const s = sets[id] = { status: 'loading', data: [], columns: [], message: '' };
    const go = () => {
      const meta = catalog && catalog[id];
      if (!meta) { s.status = 'missing'; s.message = 'No such data'; soon(); return; }
      fetch('data/' + meta.file + '?v=' + V)
        .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
        .then((t) => {
          if (meta.file.endsWith('.json')) {
            const j = JSON.parse(t);
            s.data = Array.isArray(j) ? j : (j.rows || j.data || j.results || j);
            s.columns = Array.isArray(s.data) && s.data.length ? Object.keys(s.data[0]) : [];
          } else {
            const rows = d3.csvParse(t);
            s.columns = rows.columns;
            s.data = rows.slice();
          }
          s.status = 'ok';
        })
        .catch((e) => { s.status = 'error'; s.message = String(e.message || e); })
        .finally(soon);
    };
    catalogReady.then(go);
    return s;
  }

  function view(s) {
    return { status: s.status, data: s.data, columns: s.columns, message: s.message, refreshing: false, meta: [],
             rows: s.data, value: s.data };
  }

  function calcData(id) {
    const c = calcs[id];
    const memo = calcMemo[id];
    if (memo && memo.ver === version) return memo.res;
    const ins = c.inputs.map((i) => data(i));
    let res;
    const bad = ins.find((d) => d.status !== 'ok');
    if (bad) res = { status: bad.status === 'loading' ? 'loading' : bad.status, data: [], columns: [], message: bad.message || '' };
    else {
      try {
        const out = c.fn(...ins.map((d) => d.data));
        res = { status: 'ok', data: out, columns: Array.isArray(out) && out.length && typeof out[0] === 'object' ? Object.keys(out[0]) : [], message: '' };
      } catch (e) {
        console.error('calc ' + id, e);
        res = { status: 'error', data: [], columns: [], message: String(e.message || e) };
      }
    }
    res.refreshing = false; res.meta = []; res.rows = res.data; res.value = res.data;
    calcMemo[id] = { ver: version, res };
    return res;
  }

  function data(id) {
    if (calcs[id]) return calcData(id);
    return view(load(id));
  }

  const catalogReady = Promise.all([
    fetch('data/datasets.json?v=' + V).then((r) => r.json()),
    fetch('data/params.json?v=' + V).then((r) => (r.ok ? r.json() : {})).catch(() => ({}))
  ]).then(([cat, p]) => {
    catalog = cat.datasets || cat;
    defaults = p || {};
    const q = new URLSearchParams(location.search);
    params = Object.assign({}, defaults);
    for (const [k, v] of q) params[k] = isNaN(+v) || v === '' ? v : +v;
    showUpdated(cat);
    soon();
  }).catch((e) => console.error('datasets.json', e));

  function showUpdated(cat) {
    const el = document.getElementById('hub-updated');
    if (!el || !cat.updated) return;
    const d = new Date(cat.updated);
    el.textContent = 'Updated ' + d.toLocaleString([], { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
      + (cat.note ? ' · ' + cat.note : '');
  }

  const cssColor = (n) => `var(--hub-series-${n})`;

  window.dash = {
    data,
    onData(fn) { subs.push(fn); try { fn(); } catch (e) { console.error(e); } },
    calc(id, spec) { calcs[id] = spec; },
    loader() {},
    colors: [1, 2, 3, 4, 5, 6, 7, 8].map(cssColor),
    refresh(id) { delete sets[id]; load(id); },
    params() { return Object.assign({}, params); },
    setParams(p) {
      Object.assign(params, p);
      const q = new URLSearchParams(location.search);
      for (const [k, v] of Object.entries(p)) {
        if (v === null || v === undefined || defaults[k] === v) q.delete(k); else q.set(k, v);
      }
      const qs = q.toString();
      history.replaceState(null, '', location.pathname + (qs ? '?' + qs : '') + location.hash);
      fire();
    },
    resetParams() { params = Object.assign({}, defaults); history.replaceState(null, '', location.pathname + location.hash); fire(); },
    setLink(name) { history.replaceState(null, '', location.pathname + location.search + '#' + name); }
  };

  let rz = null;
  let lastW = window.innerWidth;
  window.addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => { if (window.innerWidth !== lastW || true) { lastW = window.innerWidth; fire(); } }, 120);
  });
})();

/* moe-cache-bench charts: interactive SVG charts without any library. Used by the GUI and by the exported HTML report.
   MoeCharts.render(hostElement, result) draws every chart that the result has data for.
   Hover a bar or a point for details, click a legend entry to hide or show a series. */
const MoeCharts = (() => {
  const COL = { stock: '#9aa0a6', plugin: '#2f81f7' };
  const PAL = ['#2f81f7', '#3fb950', '#d29922', '#f85149', '#a371f7', '#39c5cf', '#db61a2', '#9aa0a6'];
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const nz = v => v !== null && v !== undefined && !Number.isNaN(v);
  const fmt = (v, d) => !nz(v) ? '-' : (d !== undefined ? v.toFixed(d) : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
  function niceMax(v) {
    if (!(v > 0)) return 1;
    const e = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (v <= m * e) return m * e;
    return 10 * e;
  }
  function tipEl() {
    let t = document.getElementById('moe-tip');
    if (!t) {
      t = document.createElement('div'); t.id = 'moe-tip';
      t.style.cssText = 'position:fixed;pointer-events:none;z-index:9999;display:none;background:var(--card,#fff);color:var(--fg,#111);border:1px solid var(--line,#ccc);border-radius:8px;padding:8px 11px;font:12.5px/1.45 system-ui,sans-serif;box-shadow:0 6px 18px rgba(0,0,0,.28);max-width:360px';
      document.body.appendChild(t);
    }
    return t;
  }
  function showTip(ev, html) {
    const t = tipEl(); t.innerHTML = html; t.style.display = 'block';
    const w = t.offsetWidth, h = t.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > innerWidth - 6) x = ev.clientX - w - 14;
    if (y + h > innerHeight - 6) y = ev.clientY - h - 14;
    t.style.left = Math.max(4, x) + 'px'; t.style.top = Math.max(4, y) + 'px';
  }
  const hideTip = () => { tipEl().style.display = 'none'; };

  /* spec: {title, note, unit, type: 'bar'|'stack'|'line', cats:[...], series:[{name,color,values:[...],tips:[html per cat],dash}], dec, xlabel} */
  function chart(host, spec) {
    const box = document.createElement('div'); box.className = 'chart'; box.style.position = 'relative';
    host.appendChild(box);
    const hidden = new Set();
    const W = 860, H = spec.height || 330, L = 66, R = 18, T = 14, B = 58;
    function draw() {
      const vis = spec.series.filter(s => !hidden.has(s.name));
      const pw = W - L - R, ph = H - T - B, n = spec.cats.length;
      let top = 0;
      if (spec.type === 'stack') for (let i = 0; i < n; i++) top = Math.max(top, vis.reduce((a, s) => a + (nz(s.values[i]) ? s.values[i] : 0), 0));
      else for (const s of vis) for (const v of s.values) if (nz(v)) top = Math.max(top, v);
      const ymax = spec.unit === '%' ? 100 : niceMax(top * 1.12 || 1);
      const Y = v => T + ph - ph * v / ymax;
      let h = `<div style="font-weight:600;font-size:15px;margin:2px 4px">${esc(spec.title)}</div>` + (spec.note ? `<div class="muted" style="font-size:12px;margin:0 4px 4px">${esc(spec.note)}</div>` : '');
      h += '<div style="margin:2px 4px 0">' + spec.series.map(s => `<span data-leg="${esc(s.name)}" style="cursor:pointer;margin-right:14px;font-size:12px;opacity:${hidden.has(s.name) ? .35 : 1};white-space:nowrap"><span style="display:inline-block;width:11px;height:11px;border-radius:2px;background:${s.color};vertical-align:-1px;margin-right:4px"></span>${esc(s.name)}</span>`).join('') + '</div>';
      let g = '';
      for (let i = 0; i <= 5; i++) {
        const y = T + ph - ph * i / 5, v = ymax * i / 5;
        g += `<line x1="${L}" y1="${y.toFixed(1)}" x2="${W - R}" y2="${y.toFixed(1)}" stroke="var(--grid,#ddd)" stroke-width="1"/><text x="${L - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="var(--muted,#666)">${+v.toPrecision(3)}</text>`;
      }
      g += `<text x="14" y="${T + ph / 2}" transform="rotate(-90 14 ${T + ph / 2})" text-anchor="middle" font-size="11" fill="var(--muted,#666)">${esc(spec.unit || '')}</text>`;
      const gw = pw / Math.max(1, n);
      const catTip = i => `<b>${esc(spec.cats[i])}</b>` + vis.map(s => `<br><span style="color:${s.color}">●</span> ${esc(s.name)}: <b>${fmt(s.values[i], spec.dec)}</b> ${esc(spec.unit || '')}`).join('');
      const marks = [];
      if (spec.type === 'line') {
        const X = i => L + (n > 1 ? pw * i / (n - 1) : pw / 2);
        for (let i = 0; i < n; i++) g += `<text x="${X(i).toFixed(1)}" y="${H - 34}" text-anchor="middle" font-size="12" fill="var(--fg,#111)">${esc(spec.cats[i])}</text>`;
        for (const s of vis) {
          const pts = s.values.map((v, i) => nz(v) ? [X(i), Y(v), i, v] : null).filter(Boolean);
          if (pts.length > 1) g += `<polyline fill="none" stroke="${s.color}" stroke-width="2.6" ${s.dash ? 'stroke-dasharray="6 4"' : ''} points="${pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ')}"/>`;
          for (const p of pts) {
            g += `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="4.5" fill="${s.color}"/>`;
            marks.push(`<circle data-i="${p[2]}" data-s="${esc(s.name)}" cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="13" fill="transparent" style="cursor:crosshair"/>`);
          }
        }
      } else {
        const bwAll = spec.type === 'stack' ? Math.min(90, gw * 0.62) : Math.min(70, gw * 0.8 / Math.max(1, vis.length));
        for (let i = 0; i < n; i++) {
          const cx = L + gw * i + gw / 2;
          let acc = 0;
          vis.forEach((s, si) => {
            const v = s.values[i];
            if (!nz(v)) return;
            let x, y0, hh;
            if (spec.type === 'stack') { x = cx - bwAll / 2; y0 = Y(acc + v); hh = Y(acc) - y0; acc += v; }
            else { x = cx - bwAll * vis.length / 2 + si * bwAll; y0 = Y(v); hh = T + ph - y0; }
            const w = bwAll - (spec.type === 'stack' ? 0 : 4);
            g += `<rect data-mark x="${(x + (spec.type === 'stack' ? 0 : 2)).toFixed(1)}" y="${y0.toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(0, hh).toFixed(1)}" rx="3" fill="${s.color}" data-i="${i}" data-s="${esc(s.name)}" style="cursor:crosshair"/>`;
            if (spec.type !== 'stack' && w > 30) g += `<text x="${(x + 2 + w / 2).toFixed(1)}" y="${(y0 - 6).toFixed(1)}" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--fg,#111)" pointer-events="none">${fmt(v, spec.dec)}</text>`;
          });
          if (spec.type === 'stack') g += `<text x="${cx.toFixed(1)}" y="${(Y(acc) - 6).toFixed(1)}" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--fg,#111)" pointer-events="none">${fmt(acc, spec.dec)}</text>`;
          g += `<text x="${cx.toFixed(1)}" y="${H - 34}" text-anchor="middle" font-size="12" fill="var(--fg,#111)">${esc(spec.cats[i])}</text>`;
          if (spec.ratio && vis.length === 2 && nz(vis[0].values[i]) && nz(vis[1].values[i]) && vis[0].values[i] > 0)
            g += `<text x="${cx.toFixed(1)}" y="${H - 12}" text-anchor="middle" font-size="13.5" font-weight="700" fill="${COL.plugin}">${spec.ratio} ${(vis[1].values[i] / vis[0].values[i]).toFixed(1)}x</text>`;
        }
      }
      if (spec.xlabel) g += `<text x="${L + pw / 2}" y="${H - 12}" text-anchor="middle" font-size="11" fill="var(--muted,#666)">${esc(spec.xlabel)}</text>`;
      box.innerHTML = h + `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${esc(spec.title)}" style="display:block">${g}${marks.join('')}</svg>`;
      box.querySelectorAll('[data-leg]').forEach(el => el.onclick = () => { const k = el.dataset.leg; hidden.has(k) ? hidden.delete(k) : hidden.add(k); hideTip(); draw(); });
      box.querySelectorAll('[data-i]').forEach(el => {
        const i = +el.dataset.i, s = spec.series.find(x => x.name === el.dataset.s);
        el.onmousemove = ev => {
          showTip(ev, catTip(i) + (s && s.tips && s.tips[i] ? `<div style="margin-top:5px;padding-top:5px;border-top:1px solid var(--line,#ccc)"><b>${esc(s.name)}</b><br>${s.tips[i]}</div>` : ''));
          if (el.hasAttribute('data-mark')) el.style.opacity = .78;
        };
        el.onmouseleave = () => { hideTip(); if (el.hasAttribute('data-mark')) el.style.opacity = 1; };
      });
    }
    draw();
  }

  // ---- chart definitions from a result
  function build(res) {
    const S = res.summary, out = [];
    const keys = new Set([...Object.keys(S.speed), ...Object.keys(S.long || {}), ...Object.keys(S.sweep || {}), ...Object.keys(S.deep || {})]);
    const lims = [...new Set([...keys].map(k => k.split('|')[0]))].sort((a, b) => (+a || 0) - (+b || 0));
    const modes = ['stock', 'plugin'].filter(m => lims.some(l => keys.has(l + '|' + m)));
    const lab = l => l === 'None' ? 'no memory limit' : l + ' GB limit';
    const mname = m => m === 'stock' ? 'stock llama.cpp' : 'with moe-cache';
    const cats = lims.map(lab);
    const both = modes.length === 2;
    const perMode = (get, tipf) => modes.map(m => ({ name: mname(m), color: COL[m], values: lims.map(l => get(S, l + '|' + m, m)), tips: lims.map(l => tipf ? tipf(l + '|' + m, m, l) : '') }));
    const dp = (k, f) => S.deep && S.deep[k] && S.deep[k].steady ? S.deep[k].steady[f] : null;
    const runCols = [...keys].filter(k => S.speed[k]).sort((a, b) => ((+a.split('|')[0] || 0) - (+b.split('|')[0] || 0)) || (a.endsWith('stock') ? -1 : 1));
    const runName = k => { const [l, m] = k.split('|'); return `${m === 'stock' ? 'stock' : 'moe-cache'}, ${l === 'None' ? 'no limit' : l + ' GB'}`; };
    const runColor = k => k.endsWith('stock') ? COL.stock : COL.plugin;

    if (Object.keys(S.speed).length) {
      out.push({ title: 'Generation speed (steady state)', note: 'Mean of requests 2 and later. Hover for the individual requests.', unit: 'tokens/s', type: 'bar', cats, ratio: 'moe-cache',
        series: perMode((S, k) => S.speed[k] && S.speed[k].steady, k => { const e = S.speed[k]; if (!e) return ''; return `range ${fmt(e.min)} to ${fmt(e.max)} tok/s over ${e.n} requests<br>per request: ${e.per_req.map(v => fmt(v)).join(', ')}<br>first request ${fmt(e.first)}`; }) });
      out.push({ title: 'Speed of every request', note: 'Shows warm-up and slow-downs (heat, cache churn) inside a run.', unit: 'tokens/s', type: 'line', cats: Object.values(S.speed)[0].per_req.map((_, i) => '#' + (i + 1)),
        series: runCols.map(k => ({ name: runName(k), color: runColor(k), dash: k.endsWith('stock'), values: S.speed[k].per_req, tips: S.speed[k].per_req.map((_, i) => { const d = S.deep[k]; const p = d && d.per_req; return p ? `${p.temp ? 'CPU ' + fmt(p.temp[i], 0) + ' C<br>' : ''}${p.io_mb_tok ? 'SSD ' + fmt(p.io_mb_tok[i], 1) + ' MB/token<br>' : ''}${p.hit_dec_pct ? 'cache hit ' + fmt(p.hit_dec_pct[i], 1) + ' %' : ''}` : ''; }) })) });
      out.push({ title: 'Cold start: first request vs steady state', unit: 'tokens/s', type: 'bar', cats: runCols.map(runName),
        series: [{ name: 'first request', color: '#d29922', values: runCols.map(k => S.speed[k].first) }, { name: 'steady state', color: '#3fb950', values: runCols.map(k => S.speed[k].steady) }] });
    }
    if (S.long && Object.keys(S.long).length) out.push({ title: 'Prompt processing speed (long prompt)', unit: 'tokens/s', type: 'bar', cats, ratio: 'moe-cache', series: perMode((S, k) => S.long[k]) });

    const swKeys = Object.keys(S.sweep || {}).sort();
    if (swKeys.length) {
      const lens = [...new Set(swKeys.flatMap(k => Object.keys(S.sweep[k])))].sort((a, b) => a - b);
      const ser = (f) => swKeys.map(k => ({ name: runName(k), color: runColor(k), dash: k.endsWith('stock'), values: lens.map(n => S.sweep[k][n] ? f(S.sweep[k][n]) : null) }));
      const cl = lens.map(n => n + ' tokens in');
      out.push({ title: 'Time to first token vs input length', note: 'Artificial-Analysis-style latency: how long until the first token appears.', unit: 'seconds', type: 'line', cats: cl, series: ser(x => x.ttft_ms / 1000) });
      out.push({ title: 'Output speed vs input length', unit: 'tokens/s', type: 'line', cats: cl, series: ser(x => x.out_tps) });
      out.push({ title: 'End-to-end latency vs input length', note: 'Time from sending the request to the last output token.', unit: 'seconds', type: 'line', cats: cl, series: ser(x => x.e2e_s) });
    }

    if (modes.some(m => lims.some(l => dp(l + '|' + m, 'io_mb_tok') !== null && dp(l + '|' + m, 'io_mb_tok') !== undefined))) {
      out.push({ title: 'Data read from the SSD per generated token', note: 'Stock: page-cache misses (from /proc/PID/io). moe-cache: its own expert reads. Lower is better.', unit: 'MB / token', type: 'bar', cats,
        series: perMode((S, k) => dp(k, 'io_mb_tok'), (k) => `page faults: ${fmt(dp(k, 'majflt_tok'), 1)} per token<br>CPU time: ${fmt(dp(k, 'cpu_ms_tok'), 1)} ms per token`) });
      out.push({ title: 'CPU time per generated token', note: 'Total CPU time the server used per token (all threads). More is not better: it also shows waiting and spinning.', unit: 'ms / token', type: 'bar', cats, series: perMode((S, k) => dp(k, 'cpu_ms_tok')) });
    }
    const plug = lims.filter(l => S.deep && S.deep[l + '|plugin'] && S.deep[l + '|plugin'].steady.hit_dec_pct !== undefined);
    if (plug.length) {
      const nreq = S.deep[plug[0] + '|plugin'].per_req.hit_dec_pct.length;
      out.push({ title: 'Cache hit rate per request (decode steps)', note: 'Share of expert uses that found the expert already in RAM. It starts low and rises as the cache warms up.', unit: '%', type: 'line', cats: [...Array(nreq)].map((_, i) => '#' + (i + 1)),
        series: plug.map((l, i) => ({ name: lab(l), color: PAL[i % PAL.length], values: S.deep[l + '|plugin'].per_req.hit_dec_pct, tips: S.deep[l + '|plugin'].per_req.hit_dec_pct.map((_, j) => `misses ${fmt(S.deep[l + '|plugin'].per_req.misses_tok[j], 1)} per token<br>evictions ${fmt(S.deep[l + '|plugin'].per_req.evictions_tok[j], 2)} per token`) })) });
      out.push({ title: 'Evictions per generated token (steady state)', note: 'Experts pushed out of RAM to make room. Tooltip: how long the evicted experts had been idle (in layer-groups); a large number means the LRU choice was good.', unit: 'evictions / token', type: 'bar', cats: plug.map(lab),
        series: [{ name: 'with moe-cache', color: COL.plugin, values: plug.map(l => dp(l + '|plugin', 'evictions_tok')), tips: plug.map(l => `evicted expert idle for ${fmt(dp(l + '|plugin', 'evict_age'), 0)} groups on average<br>cache holds ${fmt(dp(l + '|plugin', 'resident_gb'), 1)} of ${fmt(dp(l + '|plugin', 'budget_gb'), 1)} GB<br>${fmt(dp(l + '|plugin', 'misses_tok'), 1)} misses per token`) }] });
      out.push({ title: 'Where the time goes per generated token (moe-cache)', note: 'Averaged over each request including its prompt phase, so the total is a little above 1000 / tok/s. Expert compute = the CPU matrix multiplications. SSD reads = waiting for experts. Everything else = GPU work, attention, sampling, server.', unit: 'ms / token', type: 'stack', cats: plug.map(lab), dec: 1,
        series: [['ms_tok_compute', 'expert compute (CPU)', '#2f81f7'], ['ms_tok_reads', 'SSD reads', '#f85149'], ['ms_tok_bookkeeping', 'cache bookkeeping', '#d29922'], ['ms_tok_other', 'everything else', '#9aa0a6']]
          .map(([f, n, c]) => ({ name: n, color: c, values: plug.map(l => dp(l + '|plugin', f)) })) });
    }
    for (const l of lims) {
      const L = S.layers && S.layers[l + '|plugin'];
      if (L && L.hits.length) out.push({ title: `Cache hit rate by layer (${lab(l)})`, note: 'Which layers keep missing. Includes prompt processing.', unit: '%', type: 'bar', cats: L.hits.map((_, i) => 'L' + i), height: 280,
        series: [{ name: 'hit rate', color: COL.plugin, values: L.hits.map((h, i) => h + L.misses[i] ? 100 * h / (h + L.misses[i]) : null), tips: L.hits.map((h, i) => `${h} hits, ${L.misses[i]} misses`) }] });
    }
    if (modes.some(m => lims.some(l => dp(l + '|' + m, 'cg_peak')))) out.push({ title: 'Peak memory of the server (cgroup)', note: 'Includes page cache that the kernel keeps for the model file. The limit is the cap you set.', unit: 'GB', type: 'bar', cats, series: perMode((S, k) => dp(k, 'cg_peak'), (k) => `anonymous ${fmt(dp(k, 'rssanon'), 1)} GB, file-backed ${fmt(dp(k, 'rssfile'), 1)} GB`) });
    if (modes.some(m => lims.some(l => dp(l + '|' + m, 'gpu_mb')))) out.push({ title: 'GPU memory in use', unit: 'GB', type: 'bar', cats, series: perMode((S, k) => dp(k, 'gpu_mb') ? dp(k, 'gpu_mb') / 1000 : null) });
    const withTemp = runCols.filter(k => S.deep[k] && S.deep[k].per_req.temp);
    if (withTemp.length) out.push({ title: 'CPU temperature after each request', note: 'Heat explains slow-downs inside a run.', unit: '°C', type: 'line', cats: S.deep[withTemp[0]].per_req.temp.map((_, i) => '#' + (i + 1)), series: withTemp.map(k => ({ name: runName(k), color: runColor(k), dash: k.endsWith('stock'), values: S.deep[k].per_req.temp })) });
    return out;
  }
  function render(host, res) { host.innerHTML = ''; build(res).forEach(spec => chart(host, spec)); }
  return { render, build, chart };
})();

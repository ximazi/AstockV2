let virtualLibrary = {positions:[], labels:{}};
let virtualTab = 'OPEN';
const virtualPending = new Set();

function paperAddMarkup(symbol, date, price, withLabel = false, strategy = '') {
  const held = virtualLibrary.positions.some(p => p.symbol === symbol && p.status === 'OPEN');
  const label = held ? '已在虚拟持仓' : (strategy === 'S4_LAD' ? '记入虚拟持仓（S4 分批止盈）' : '添加到虚拟持仓');
  return `<button type="button" class="paper-add ${withLabel ? 'paper-command' : ''}" data-paper-add="${esc(symbol)}" data-paper-date="${esc(date)}" data-paper-strategy="${esc(strategy)}" ${held ? 'disabled' : ''} aria-label="${label} ${esc(symbol)}" title="${label} · ${esc(date)} 收盘 ${num(price, 3)}"><img src="/static/vendor/lucide/plus.svg" alt="">${withLabel ? label : ''}</button>`;
}

async function refreshVirtualLibrary() {
  virtualLibrary = await getJson('/api/virtual-positions');
}

function paperObservation(p) {
  const o = p.observation || {status:'NEW', reason:'等待日常持仓扫描', events:[]};
  const expected = virtualLibrary.report_date || latestDailyDate;
  const stale = p.status === 'OPEN' && o.as_of && o.as_of < expected;
  return stale ? {...o, status:'DATA_WAIT', reason:`最新日报 ${expected}，持仓仅评估至 ${o.as_of}。上次：${o.reason}`} : o;
}

async function loadVirtualPositions() {
  await refreshVirtualLibrary();
  if (sourceMode !== 'portfolio') return;
  $('#candidateDate').textContent = `持仓 ${virtualLibrary.positions.filter(p=>p.status==='OPEN').length}`;
  const rows = virtualLibrary.positions.filter(p=>p.status===virtualTab);
  $('#candidateList').innerHTML = `<button class="paper-overview" data-paper-overview>持仓总览</button>` + rows.map(p=>{
    const o = paperObservation(p);
    return `<button class="candidate" data-paper-analyze="${esc(p.symbol)}"><div class="candidate-top"><span class="candidate-name">${esc(p.name)}</span><span class="paper-state state-${esc(o.status)}">${p.status === 'CLOSED' ? '已模拟卖出' : esc(virtualLibrary.labels[o.status])}</span></div><div class="candidate-meta"><span>${esc(p.symbol)}</span><span>${esc(p.entry_date)} · ${num(p.entry_price,3)}</span></div></button>`;
  }).join('');
}

function renderVirtualPortfolio() {
  const rows = virtualLibrary.positions.filter(p=>p.status===virtualTab);
  const open = virtualLibrary.positions.filter(p=>p.status==='OPEN');
  const alerts = open.filter(p=>['SELL','REDUCE'].includes(paperObservation(p).status));
  $('#workspace').innerHTML = `<section class="section paper-heading"><div class="section-head"><h1>虚拟持仓</h1><span>风险评估 · 模拟成本</span></div>
    <div class="paper-totals"><span>持仓中 <b>${open.length}</b></span><span>卖出 / 减仓提醒 <b>${alerts.length}</b></span><span>待核验 / 待扫描 <b>${open.filter(p=>['DATA_WAIT','NEW'].includes(paperObservation(p).status)).length}</b></span></div>
    <div class="paper-tabs" role="group" aria-label="持仓状态"><button data-paper-tab="OPEN" aria-pressed="${virtualTab==='OPEN'}">持仓中</button><button data-paper-tab="CLOSED" aria-pressed="${virtualTab==='CLOSED'}">已卖出</button></div></section>
    <section class="section"><div class="table-wrap paper-table"><table><thead><tr><th>股票 / 题材 / 路径</th><th>加入日期 / 收盘成本</th><th>${virtualTab==='OPEN'?'行情日期 / 收盘价':'卖出日期 / 收盘价'}</th><th>复权价差</th><th>当前阶段 / 持仓天数</th><th>风险建议 / 依据</th><th>操作</th></tr></thead><tbody>${rows.map(p=>{
      const o = paperObservation(p), closed = p.status==='CLOSED', gain = closed ? p.return_pct : o.return_pct;
      return `<tr><td><b>${esc(p.name)}</b><small>${esc(p.symbol)}</small><small>${esc(p.theme || '题材未记录')} ${esc(p.route_ids)}</small></td><td>${esc(p.entry_date)}<small>${num(p.entry_price,3)}</small></td>
        <td>${esc(closed ? p.exit_date : o.price_date || '待扫描')}<small>${num(closed ? p.exit_price : o.close,3)}</small></td><td class="${pctClass(gain)}">${num(gain,2)}%</td><td>${esc(o.lifecycle || '待评估')}<small>${o.held_sessions ?? '--'} 个活跃交易日</small></td>
        <td class="paper-reason"><b class="paper-state state-${esc(o.status)}">${closed ? '已模拟卖出' : esc(virtualLibrary.labels[o.status])}</b><small>${esc(o.reason)}</small>${o.strategy === 'S4_LAD' ? `<small>S4 分批止盈 · 剩余 ${Math.round((o.lad_remaining ?? 1) * 100)}% · 止盈 ${num(o.tp1,2)} / ${num(o.tp2,2)} · 规则模拟收益 ${num(o.lad_return_pct,1)}%</small>` : ''}<small>评估 ${esc(o.as_of || '未执行')} · 硬底线 ${num(o.hard_floor,3)} / 收盘保护 ${num(o.soft_floor,3)}</small>
        ${p.last_actionable_observation ? `<small>上次待处置风险 ${esc(p.last_actionable_observation.as_of)}：${esc(p.last_actionable_observation.reason)}</small>` : ''}
        ${(o.events || []).length ? `<details><summary>风险轨迹 ${(o.events || []).length}</summary>${o.events.map(e=>`<p>${esc(e.date)} · ${esc(e.label)}</p>`).join('')}</details>` : ''}</td>
        <td><div class="paper-actions"><button data-paper-analyze="${esc(p.symbol)}">查看分析</button>${closed ? '' : `<button data-paper-close="${esc(p.id)}">模拟卖出</button>`}</div></td></tr>`;
    }).join('')}</tbody></table></div>${rows.length ? '' : `<div class="empty">${virtualTab==='OPEN'?'暂无虚拟持仓':'暂无模拟卖出记录'}</div>`}<p class="subtle">${esc(virtualLibrary.notice)}</p></section>`;
}

function renderVirtualContext(symbol) {
  const p = virtualLibrary.positions.find(p=>p.symbol===symbol && p.status===virtualTab);
  if (!p) return '';
  const o = paperObservation(p);
  return `<section class="section paper-entry"><button data-paper-overview>返回持仓总览</button><b>本笔虚拟持仓</b><span>${esc(p.entry_date)} 收盘 ${num(p.entry_price,3)}</span><span>${esc(virtualLibrary.labels[o.status])} · ${esc(o.reason)}</span><p class="subtle">本笔风险按加入时成本计算。下方“正式卖点方案”为信号参考周期，不替代本笔持仓评估。</p></section>`;
}

async function paperPost(url, body) {
  const response = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

document.addEventListener('click', async event => {
  const add = event.target.closest('[data-paper-add]');
  const close = event.target.closest('[data-paper-close]');
  const analyze = event.target.closest('[data-paper-analyze]');
  const tab = event.target.closest('[data-paper-tab]');
  if (analyze) { $('#dateInput').value = virtualLibrary.report_date || latestDailyDate; return runAnalysis(analyze.dataset.paperAnalyze); }
  if (tab || event.target.closest('[data-paper-overview]')) {
    if (tab) virtualTab = tab.dataset.paperTab;
    try { await loadVirtualPositions(); if (sourceMode==='portfolio') renderVirtualPortfolio(); }
    catch (error) { $('#runStatus').textContent = error.message; }
    return;
  }
  const button = add || close;
  if (!button) return;
  const key = add ? add.dataset.paperAdd : close.dataset.paperClose;
  if (virtualPending.has(key)) return;
  const exitDate = virtualLibrary.report_date || latestDailyDate;
  if (close && !window.confirm(`按 ${exitDate} 本地收盘价模拟卖出全部，保留历史记录？`)) return;
  virtualPending.add(key);
  button.disabled = true;
  try {
    const result = add ? await paperPost('/api/virtual-positions', add.dataset.paperStrategy
        ? {symbol:key, date:add.dataset.paperDate, strategy:add.dataset.paperStrategy} : {symbol:key, date:add.dataset.paperDate})
      : await paperPost(`/api/virtual-positions/${encodeURIComponent(key)}/close`, {date:exitDate});
    await refreshVirtualLibrary();
    if (add) {
      document.querySelectorAll('[data-paper-add]').forEach(b=>{
        if (b.dataset.paperAdd===key) {
          b.disabled=true; b.title='已在虚拟持仓'; b.setAttribute('aria-label',`已在虚拟持仓 ${key}`);
          if (b.classList.contains('paper-command')) b.innerHTML='<img src="/static/vendor/lucide/plus.svg" alt="">已在虚拟持仓';
        }
      });
      $('#runStatus').textContent = `${result.created ? '已加入虚拟持仓' : '已存在，未重复加入'} · ${result.position.entry_date} · 收盘 ${num(result.position.entry_price,3)}`;
    } else {
      await loadVirtualPositions();
      if (sourceMode==='portfolio') renderVirtualPortfolio();
      $('#runStatus').textContent = `已模拟卖出 · ${result.exit_date} · ${num(result.exit_price,3)}`;
    }
  } catch (error) { button.disabled=false; $('#runStatus').textContent = `虚拟持仓：${error.message}`; }
  finally { virtualPending.delete(key); }
});

$('#portfolioSource').addEventListener('click', () => switchSource('portfolio'));

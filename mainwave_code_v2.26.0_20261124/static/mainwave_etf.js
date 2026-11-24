/* Uses the workbench's existing chart and lifecycle renderers; no network refresh on reads. */
let etfLibrary = null;
let etfOverviewOpen = true;

async function loadEtfs() {
  etfLibrary = await getJson(`/api/etfs?date=${encodeURIComponent($('#dateInput').value)}`);
  const mode = $('#etfFilter').value;
  const filtered = etfLibrary.records.filter(r => mode === 'all'
    || (mode === 'priority' && ['优先研究','趋势内候选'].includes(r.tier))
    || (mode === 'strong' && r.strong_trend)
    || (mode === 'hot' && r.overheated)
    || (mode === 'oversold' && r.oversold));
  $('#candidateDate').textContent = `${etfLibrary.as_of} · ${filtered.length}/10${etfLibrary.older_report ? ' · 较早报告' : ''}`;
  $('#candidateList').innerHTML = filtered.map(r => `<button class="candidate ${activeSymbol === r.code ? 'active' : ''}" data-symbol="${esc(r.code)}">
    <div class="candidate-top"><span class="candidate-name">${esc(r.theme)}</span><span class="candidate-tier ${r.overheated ? 'd' : ''}">${esc(r.tier)}</span></div>
    <div class="candidate-meta"><span>${esc(r.code)} · ${esc(r.name)}</span><span>${esc(r.stage || r.error)}</span><span>EMA20 ${num(r.bias20_pct,1)}%</span></div></button>`).join('') || '<div class="empty">当前没有符合该分类的ETF</div>';
  document.querySelectorAll('.candidate').forEach(b => b.addEventListener('click', () => runEtf(b.dataset.symbol)));
}

function etfOverview() {
  const c = etfLibrary.counts;
  return `<details class="section etf-overview" ${etfOverviewOpen ? 'open' : ''}><summary>10只ETF总览 · ${esc(etfLibrary.as_of)}</summary>
    <p>强势趋势 ${c.strong} · 正偏离过热 ${c.overheated} · 负偏离超调 ${c.oversold} · 优先研究 ${c.priority} · 当日行情 ${c.fresh}/10</p>
    <div class="table-wrap"><table><thead><tr><th>ETF / 题材</th><th>连续周期</th><th>参与价值</th><th>强趋势</th><th>EMA20偏离</th><th>近20日</th><th>日均成交额</th></tr></thead><tbody>
    ${etfLibrary.records.map(r => `<tr><td><button class="etf-link" data-etf="${esc(r.code)}">${esc(r.code)}</button><br>${esc(r.theme)}</td><td>${esc(r.stage || '数据不足')}</td><td>${esc(r.tier)}</td><td>${r.strong_trend ? '是' : '否'}</td><td class="${pctClass(r.bias20_pct)}">${num(r.bias20_pct,1)}%</td><td>${num(r.r20_pct,1)}%</td><td>${num(r.amount20_yi,2)}亿</td></tr>`).join('')}
    </tbody></table></div></details>`;
}

function renderEtf(data) {
  currentData = data;
  const m = data.metrics, d = data.decision, n = data.nav, s = data.snapshot;
  const history = data.lifecycle.history.slice().reverse();
  $('#workspace').innerHTML = `${etfOverview()}
    <section class="decision-band ${d.overheated || !d.eligible ? 'risk' : 'watch'}">
      <div><div class="identity-line"><h1>${esc(data.identity.name)}</h1><span class="code">${esc(data.identity.code)} · ${esc(data.identity.scope)}</span></div>
      <div class="decision-copy"><strong>${esc(d.tier)}</strong><span class="reason">${esc(d.action)}</span></div></div>
      <div class="permission"><b>${d.strong_trend ? '强势趋势' : '趋势待确认/修复'}</b><span>报告 ${esc(data.meta.report_date)} · 行情 ${esc(data.meta.price_date)}</span></div>
    </section>
    ${data.meta.requested_date !== data.meta.report_date ? `<section class="section"><strong>所选日期 ${esc(data.meta.requested_date)}，展示最近已生成报告 ${esc(data.meta.report_date)}</strong></section>` : ''}
    <section class="metrics">
      ${metric('收盘价', m.close, '', '', 3)}${metric('今年涨幅', m.ytd_pct, '%', '上年12月31日收盘起')}${metric('5日涨幅', m.r5_pct, '%')}${metric('20日涨幅', m.r20_pct, '%')}
      ${metric('EMA20偏离', m.bias20_pct, '%')}${metric('偏离 / ATR20', m.bias_atr, 'x')}${metric('自身偏离分位', m.bias_percentile, '%', `历史${m.bias_samples}日，不含当日`)}${metric('20日日均成交', m.amount20_yi, '亿')}${metric('距60日收盘高点', m.drawdown60_pct, '%')}
    </section>
    <section class="section"><div class="section-head"><h2>连续生命周期</h2><span>${esc(data.lifecycle.current_label)} · 本轮启动 ${esc(data.lifecycle.started_at || '尚未识别')}</span></div><div class="lifecycle">${renderLifecycle(data)}</div><p class="subtle">${esc(data.lifecycle.reason)}</p></section>
    <section class="section"><div class="section-head"><h2>ETF价格与阶段</h2><div class="stage-controls"><label><input id="stageBackgroundToggle" type="checkbox" ${showStageBackground ? 'checked' : ''}>阶段背景</label><label>背景浓度 <input id="stageOpacity" type="range" min="15" max="50" step="5" value="${Math.round(stageOpacity*100)}" aria-label="背景浓度"><output id="stageOpacityValue">${Math.round(stageOpacity*100)}%</output></label></div><span>${esc(data.chart.price_basis)}</span></div><div class="stage-legend">${renderStageLegend(data)}</div><div id="stageReadout" class="stage-readout"></div><div id="priceChart"></div></section>
    <section class="details">
      <div class="detail"><h2>参与条件与边界</h2><dl class="kv"><dt>正偏离过热</dt><dd>${d.overheated ? '是，等待消化' : '未触发'}</dd><dt>负偏离超调</dt><dd>${d.oversold ? '是，仅观察反转' : '未触发'}</dd><dt>初步企稳条件</dt><dd>${d.stabilizing ? '满足，不等于底部确认' : '尚未满足'}</dd><dt>EMA20 / EMA60</dt><dd>${num(m.ema20,4)} / ${num(m.ema60,4)}</dd><dt>前20日低 / 高</dt><dd>${num(m.support20,4)} / ${num(m.resistance20,4)}</dd><dt>权限</dt><dd>${esc(d.permission)}</dd></dl></div>
      <div class="detail"><h2>净值与跨境风险</h2><dl class="kv"><dt>投资敞口</dt><dd>${esc(data.identity.exposure)}</dd><dt>净值对应日</dt><dd>${esc(n.date || '未知')}</dd><dt>净值公告日</dt><dd>${esc(n.ann_date || '未知')}</dd><dt>配对折溢价</dt><dd>${num(n.premium_pct,2)}%</dd><dt>有效性</dt><dd>${esc(n.status)}</dd><dt>当前盘中溢价</dt><dd>未接入IOPV，需交易前核对</dd></dl></div>
      <div class="detail"><h2>选样与口径核对</h2><dl class="kv"><dt>固定选样日</dt><dd>${esc(s.as_of)}</dd><dt>原表今年涨幅</dt><dd>${num(s.user_ytd_pct,1)}%</dd><dt>复核选样日涨幅</dt><dd>${num(s.verified_ytd_pct,2)}%</dd><dt>收益口径</dt><dd>人民币场内收盘价，前复权；非基金净值回报</dd><dt>成交额门槛</dt><dd>近20日均额 ≥ 0.5亿元，每日重检</dd><dt>同类第一名</dt><dd>用户选样，未重跑全市场排名</dd></dl></div>
    </section>
    <section class="section"><ul class="etf-notes">${data.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></section>
    <section class="table-section"><div class="section-head"><h2>逐日连续记录</h2><span>${esc(data.meta.version)} · ETF观察阈值待校准</span></div><div class="table-wrap"><table><thead><tr><th>日期</th><th>周期阶段</th><th>本轮启动</th><th>变化原因</th><th>速度</th></tr></thead><tbody>${history.map(r=>`<tr><td>${esc(r.date)}</td><td>${esc(r.cycle_stage_label)}</td><td>${esc(r.cycle_started_at || '-')}</td><td>${esc(r.cycle_reason_label)}</td><td>${esc(r.speed_label || '-')}</td></tr>`).join('')}</tbody></table></div></section>`;
  renderChart(data);
  $('.etf-overview').addEventListener('toggle', event => { etfOverviewOpen = event.target.open; });
  document.querySelectorAll('[data-etf]').forEach(b => b.addEventListener('click', () => runEtf(b.dataset.etf)));
  $('#stageBackgroundToggle').addEventListener('change', event => { showStageBackground = event.target.checked; updateStageDisplay(); });
  $('#stageOpacity').addEventListener('input', event => { stageOpacity = Number(event.target.value)/100; $('#stageOpacityValue').textContent = `${event.target.value}%`; updateStageDisplay(); });
}

async function runEtf(code) {
  document.body.classList.add('loading');
  $('#runStatus').textContent = '正在读取ETF报告';
  try {
    const data = await getJson(`/api/etfs/${encodeURIComponent(code)}?date=${encodeURIComponent($('#dateInput').value)}`);
    activeSymbol = data.identity.code;
    $('#symbolInput').value = activeSymbol;
    renderEtf(data);
    $('#runStatus').textContent = `${data.meta.report_date} · ${activeSymbol} · 仅观察`;
    document.querySelectorAll('.candidate').forEach(b => b.classList.toggle('active', b.dataset.symbol === activeSymbol));
  } catch (error) {
    $('#workspace').innerHTML = `<div class="error">${esc(error.message)}</div>`;
    $('#runStatus').textContent = 'ETF报告读取失败';
  } finally { document.body.classList.remove('loading'); }
}

$('#etfSource').addEventListener('click', () => switchSource('etf'));
$('#etfFilter').addEventListener('change', () => loadEtfs().catch(error => { $('#candidateList').innerHTML = `<div class="error">${esc(error.message)}</div>`; }));

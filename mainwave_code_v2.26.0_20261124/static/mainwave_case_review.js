/* Historical route provenance and independently labelled structural paper trades. */
function renderCaseReview(data) {
  const review = data.case_review;
  if (!review) return '';
  const route = review.route, trade = review.trade;
  const statuses = {NO_ENTRY:'未入场', ENTRY_PENDING:'等待下一开盘', OPEN:'模拟持有中', EXIT_PENDING:'退出待成交', CLOSED:'模拟已平仓'};
  const price = point => point ? `${point.date} · ${num(point.raw_price, 3)}` : '未发生';
  const result = trade.gross_return_pct != null ? `${num(trade.gross_return_pct)}%（已实现毛价差）`
    : trade.unrealized_return_pct != null ? `${num(trade.unrealized_return_pct)}%（未实现，截至 ${trade.mark_at}）` : '不适用，未成交';
  return `<section class="section" id="caseReview">
    <div class="section-head"><h2>${esc(route.label)}</h2><span>目标 ${esc(route.target_symbol)} · ${esc(route.assignment)}</span></div>
    <ol class="case-route-steps">${route.steps.map(step=>`<li>${esc(step)}</li>`).join('')}</ol>
    <p class="analysis-next">目标股点火：<b>${esc(route.target_ignition_at || '未核验')}</b>${route.target_ignition_kind === 'STRUCTURAL_PROXY' ? '（结构启动代理，非涨停点火定义）' : '（审计窗口内首次涨停）'}；题材点火：<b>${esc(route.theme_ignition_at || '未核验')}</b>。题材点火不替代目标股的买点。</p>
    ${route.leader_symbol || route.link_evidence ? `<p>领头股 ${esc(route.leader_symbol || '缺证据')} · 传播关联 ${esc(route.link_evidence || '缺证据')}</p>` : ''}
    <details class="analysis-detail"><summary>线路证据与限制 · ${route.evidence_gaps.length} 项待核验</summary><p>记录来源：${esc(route.source)}；点火前资金代理首次记录：${esc(route.capital_first_seen || '未核验')}。资金代理不是主力账户持仓证明；有早期记录也不证明持续到点火。</p><ul>${route.evidence_gaps.map(g=>`<li>${esc(g)}</li>`).join('')}</ul></details>
    <div class="section-head"><h2>点火与模拟买卖</h2><span>${esc(statuses[trade.status] || trade.status)}</span></div>
    <dl class="analysis-state"><div><dt>买入日期 · 当时原始价</dt><dd>${esc(price(trade.entry))}</dd></div><div><dt>卖出日期 · 当时原始价</dt><dd>${esc(price(trade.exit))}</dd></div><div><dt>单笔复权收益</dt><dd>${esc(result)}</dd></div><div><dt>结论</dt><dd>${esc(trade.reason)}</dd></div></dl>
    <div class="table-wrap"><table><thead><tr><th>事件</th><th>发生日期</th><th>收盘信号日</th><th>当时原始价</th><th>图上复权价</th><th>依据</th></tr></thead><tbody>${review.events.map(e=>`<tr><td>${esc(e.label)}</td><td>${esc(e.date)}</td><td>${esc(e.signal_at || '--')}</td><td>${num(e.raw_price,3)}</td><td>${num(e.chart_price,3)}</td><td>${esc(e.reason || '独立事件，不自动成交')}</td></tr>`).join('') || '<tr><td colspan="6">窗口内没有可核验的点火或成交位置</td></tr>'}</tbody></table></div>
    ${trade.rejections.length ? `<details class="analysis-detail"><summary>未成交复核 · ${trade.rejections.length} 次</summary><ul>${trade.rejections.map(r=>`<li>${esc(r.signal_at)} 信号 / ${esc(r.date)} 复核：${esc(r.reason)}</li>`).join('')}</ul></details>` : ''}
    <p class="subtle">${esc(review.rules)}</p>
  </section>`;
}

function caseChartTraces(data) {
  const points = data.case_review?.events || [];
  const styles = {
    TARGET_IGNITION: ['目标股点火 / 结构启动', 'star', '#a06600'],
    THEME_IGNITION: ['题材点火', 'diamond-open', '#326a8b'],
    BUY: ['模拟买入', 'triangle-up', '#c52e3b'],
    SELL: ['模拟卖出', 'triangle-down', '#11734c'],
  };
  return Object.entries(styles).flatMap(([kind, [name, symbol, color]]) => {
    const rows = points.filter(e=>e.kind === kind);
    return rows.length ? [{type:'scatter', mode:'markers', name, x:rows.map(e=>e.date), y:rows.map(e=>e.chart_price),
      marker:{symbol, size:kind === 'THEME_IGNITION'?18:14, color, line:{width:2, color:kind === 'THEME_IGNITION'?color:'#fff'}},
      customdata:rows.map(e=>[e.label,e.raw_price,e.signal_at || '--',e.reason || '非买入信号']),
      hovertemplate:'%{x}<br>%{customdata[0]}<br>当时原始价 %{customdata[1]:.3f}<br>图上复权价 %{y:.3f}<br>信号日 %{customdata[2]}<br>%{customdata[3]}<extra></extra>'}] : [];
  });
}

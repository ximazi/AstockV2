function renderAnalysisCore(core) {
  if (!core) return '';
  const state = core.state_vector;
  const window = core.review_window;
  const evolution = core.evolution;
  const breakout = core.breakout || {};
  const route = core.route_evidence || {};
  const contract = core.entry_contract || {};
  const buy = core.buy_plan;
  const sell = core.sell_plan;
  const exit = sell?.observation || {};
  const addLabels = {NO_RECENT_ACCEPTANCE:'无近期可用信号', WAIT_INITIAL_FILL:'等待首仓实际成交', DATA_GAP:'日历或行情待核验', WINDOW_EXPIRED:'补仓窗口已到期', NO_STRUCTURE:'缺少结构边界', INITIAL_REFERENCE_REJECTED:'参考首仓价格不合格', WAIT_PULLBACK_RECLAIM:'等待回撤后再转强', STRUCTURE_INVALID:'本轮结构失效', SELL_REFERENCE_CANCELLED:'已出现参考退出，取消补仓', CONDITIONAL_ADD_REVIEW:'回撤转强已出现，待核实际持仓', EXECUTION_BAND_EMPTY:'保护位与价格上界冲突，暂不补仓'};
  const contractLabels = {PRICE_INDEPENDENT:'独立价格确认', THEME_REQUIRED:'题材确认必需', PROPAGATION_REQUIRED:'紧链传播证据必需', STRUCTURE_ONLY:'结构复核'};
  const originLabels = {PRE_ATTEMPT:'启动前准备', ATTEMPT_UNDERWAY:'突破尝试中形成准备', POST_ACCEPTANCE:'已启动后的整理', UNKNOWN:'准备前因未知'};
  const prepLabels = {OBSERVING:'准备观察', EVIDENCE_STRONG:'准备证据增强', ATTEMPT:'已尝试突破', ACCEPTED:'突破已接受', ADVANCING:'已推进', INVALID:'本轮准备失效', EXPIRED:'观察到期'};
  const labels = {PASS:'满足', FAIL:'不满足', WAIT:'待确认', UNKNOWN:'未知', OBSERVED:'已有观测', PARTIAL:'部分证据',
    STRUCTURAL_PROXY:'结构代理', HISTORICAL_PRICE_PROXY:'价格样本研究，未验证盈利', NOT_MATCHED:'未匹配', OBSERVE:'提前观察', RESEARCH_ONLY:'研究阶段', RESEARCH_RELEVANT:'适合研究'};
  const marketLabels = {BULL:'牛市结构', BEAR:'熊市结构', RANGE:'震荡结构', NORMAL:'常态', PANIC:'恐慌', EUPHORIA:'狂热', UNKNOWN:'未知'};
  const evidence = value => typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value ?? '');
  return `<section class="section analysis-core" aria-label="统一分析框架">
    <div class="section-head"><h2>机会分析</h2><span>${esc(core.as_of)} · ${esc(core.mechanism.label)}</span></div>
    <dl class="analysis-state">
      <div><dt>市场结构 / 温度</dt><dd>${esc(marketLabels[core.market.structure] || core.market.structure)} / ${esc(marketLabels[core.market.temperature] || core.market.temperature)}</dd></div>
      <div><dt>价格大阶段</dt><dd>${esc(state.phase_label)}</dd></div>
      <div><dt>连续阶段</dt><dd>${esc(state.stage_label)}</dd></div>
      <div><dt>发现线路 / 介入形态</dt><dd>${esc(core.route_evidence?.route_ids || '未登记线路')} / ${esc(core.active_strategy)} · ${esc(labels[core.strategies.find(s=>s.code===core.active_strategy)?.status] || '待形成')}</dd></div>
      <div><dt>最近准备起点（量价重建）</dt><dd>${esc(route.preparation_first_seen || '尚无有效准备')} · ${esc(originLabels[route.preparation_origin] || '前因待确认')}</dd></div>
      <div><dt>准备记录状态</dt><dd>${esc(prepLabels[route.preparation_state] || '未建立')} ${route.preparation_age != null ? `· 距起点 ${num(route.preparation_age, 0)} 个活跃交易日` : ''}${route.preparation_id && !route.preparation_valid ? '<br>旧记录已失效，不作当前准备依据' : ''}</dd></div>
      <div><dt>介入证据契约</dt><dd>${esc(contractLabels[contract.contract] || '待刷新分析')} · ${contract.requirements_met === true ? '必要背景已满足' : '必要背景待确认'}</dd></div>
    </dl>
    <div class="analysis-boundary"><span>冻结确认位 <b>${num(window.trigger ?? window.lower, 3)}</b></span><span>观察上界 <b>${num(window.upper, 3)}</b></span><span>${core.active_strategy === 'S4' ? '波动风险参考位' : '结构失效位'} <b>${num(window.invalid, 3)}</b></span><span>确认日 ${esc(window.confirmed_at || '未确认')}</span></div>
    <p class="analysis-next"><b>个股结构：</b>${esc(core.structural_advice_label)}<br><b>下一步条件：</b>${esc(core.next_condition)}</p>
    ${buy ? `<details class="analysis-detail" open><summary>正式买点方案 · 首仓50% · 回撤转强补仓</summary>
      <p><b>当前裁定：</b>${buy.initial_review ? '首仓50%可进入价格复核' : '尚无首仓介入许可'}；${buy.blocked ? '补仓被风险或证据门槛阻断' : esc(addLabels[buy.observation.status] || buy.observation.status)}</p>
      <p>${esc(buy.allocation_basis)} ${esc(buy.rules.initial)}</p><p>${esc(buy.rules.add)}</p><p>${esc(buy.rules.confirmation)}</p><p>${esc(buy.rules.opening)}</p>
      ${buy.observation.reference_entry_date ? `<p>假设首仓参考日 ${esc(buy.observation.reference_entry_date)} · 参考价 ${num(buy.observation.reference_entry_price, 3)} · 已过 ${num(buy.observation.age_sessions, 0)} 个交易日。此处不是成交记录。</p>` : ''}
      ${(buy.observation.confirmations || []).map(c=>`<p>第${num(c.tranche, 0)}档 ${num(c.budget_fraction * 100, 2)}% · 测试 ${esc(c.touch_date)} · 确认 ${esc(c.confirm_date)} · 下一交易日开盘须高于 ${num(c.min_price, 3)} 且不超过 ${num(c.max_price, 3)}</p>`).join('')}
      <p>${esc(buy.position_requirements)}</p><p>${esc(buy.rules.cancel)}</p>
      ${buy.block_reasons.length ? `<p>阻断：${esc(buy.block_reasons.join('；'))}</p>` : ''}
      <p class="subtle">${esc(buy.version)} · 仅分析建议，不生成成交；退出或减仓警报会取消未完成补仓。</p>
    </details>` : ''}
    ${sell ? `<details class="analysis-detail" open><summary>正式卖点方案 · 收盘确认与条件续持</summary>
      <p><b>当前参考：</b>${esc(sell.status_label)}${sell.reason_label ? ` · ${esc(sell.reason_label)}` : ''}</p>
      <p><b>市场与个股同强：</b>${exit.market_known ? (exit.strong ? '满足强势条件' : '按常态保护') : '指数证据不足，不启用强势宽容'}${exit.current_structure_alert ? '；当前存在结构破坏或失败反抽警报' : ''}</p>
      ${exit.reference?.reference_entry_date ? `<p>参考首仓 ${esc(exit.reference.reference_entry_date)} · 参考价 ${num(exit.reference.reference_entry_price, 3)} · ${num(exit.held_sessions, 0)} 个持有交易日${exit.closed_at ? ` · 参考结束 ${esc(exit.closed_at)}` : ''}</p>` : ''}
      ${exit.hard_floor != null ? `<div class="analysis-boundary"><span>固定硬底线 <b>${num(exit.hard_floor, 3)}</b></span><span>下次收盘保护位 <b>${num(exit.soft_floor, 3)}</b></span><span>连续失守 ${num(exit.soft_breaches, 0)} / ${num(exit.soft_required, 0)} 次</span></div>` : ''}
      ${exit.extension_review ? `<p>第20日复核 ${esc(exit.extension_review.date)}：${exit.extension_review.extend ? '满足条件，允许续持；风险规则继续有效' : '未满足条件，下一交易日退出参考'}</p>` : ''}
      ${exit.evidence && Object.values(exit.evidence).some(Boolean) ? `<p>风险证据：${esc(Object.entries(exit.evidence).filter(([,v])=>v).map(([k])=>({heavy_break:'放量弱收盘破位', failed_retest:'失败反抽', gap_break:'跳空破位', weak_peak_close:'高位冲高弱收盘', volume_stall:'放量滞涨', repeated_wicks:'反复上影'}[k] || k)).join('；'))}。仅为量价风险代理，不证明主力出货。</p>` : ''}
      <p>${esc(sell.basis)}</p><p>${esc(sell.price_basis)}</p>
      <details><summary>退出规则与执行边界</summary>${Object.values(sell.rules).map(rule=>`<p>${esc(rule)}</p>`).join('')}<p>${esc(sell.execution)}</p></details>
      <p class="subtle">${esc(sell.version)} · 用户批准的正式分析规则，尚未证明稳定收益优势；不生成成交。</p>
    </details>` : ''}
    ${core.entry_timing ? `<details class="analysis-detail"><summary>回踩介入观察 · ${esc(core.entry_timing.plan_label)}（研究）</summary><p>测试起点：${esc(core.entry_timing.plan_test_at || '未形成')} · 恢复确认：${esc(core.entry_timing.plan_confirmed_at || '未确认')} · ${esc(core.entry_timing.plan_kind === 'ACCEPTANCE_RETEST' ? '接受后早期回踩' : core.entry_timing.plan_kind === 'ADVANCED_RETEST' ? '推进后回踩' : '等待新形态')}</p><p>${esc(core.entry_timing.rules.entry)}</p><p>${esc(core.entry_timing.rules.opening)}</p><p class="subtle">${esc(core.entry_timing.rules.boundary)} 不改变原生产生命周期，不构成成交授权。</p></details>` : ''}
    <dl class="analysis-state" aria-label="突破与回踩事件链">
      <div><dt>突破事件</dt><dd>${esc(breakout.breakout_status_label || '尚未建立')}</dd></div>
      <div><dt>突破尝试 / 接受</dt><dd><time>${esc(breakout.breakout_attempt_at || '--')}</time> / <time>${esc(breakout.breakout_accepted_at || '--')}</time></dd></div>
      <div><dt>突破后推进</dt><dd>${esc(breakout.breakout_advanced_at || '尚未推进')}</dd></div>
      <div><dt>${core.active_strategy === 'S4' ? '接受前因' : '回踩前因'}</dt><dd>${core.chain_linked ? '已绑定同一突破事件' : '前因未完整绑定'}</dd></div>
    </dl>
    ${evolution ? `<details class="analysis-detail" open><summary>证据变化 · ${esc(evolution.previous_as_of || '首次留档')}</summary>
      <p class="analysis-next">${esc(evolution.summary)}</p>
      <p class="subtle">基线 ${esc(evolution.origin.as_of)} · 上次 ${esc(evolution.previous_as_of || '无同版前值')} · 当前 ${esc(evolution.latest.as_of)}</p>
      <div class="analysis-checks">${evolution.changes.map(c=>`<div class="analysis-check"><span>${esc(c.label)}<small class="evidence-comparison">${esc(labels[evolution.origin.checks[c.id]] || '--')} → ${esc(labels[c.before] || '--')} → ${esc(labels[c.now] || c.now)}</small></span><b class="check-${c.now.toLowerCase()}">${esc(c.change_label)}</b></div>`).join('')}</div>
      <p class="subtle">${esc(evolution.boundary)}</p>
    </details>` : ''}
    <details class="analysis-detail"><summary>策略条件与卡点</summary><div class="analysis-checks">${core.checks.map(c=>`<div class="analysis-check"><span>${esc(c.label)}</span><b class="check-${c.status.toLowerCase()}">${esc(labels[c.status] || c.status)}</b></div>`).join('')}</div></details>
    <details class="analysis-detail"><summary>六维证据与缺口</summary><div class="analysis-dimensions">${core.dimensions.map(d=>`<details class="dimension-row"><summary><b>${esc(d.code)} ${esc(d.label)}</b><span>${esc(labels[d.status] || d.status)}</span></summary><pre>${esc(evidence(d.observation))}</pre><p>来源：${esc(d.source)} · ${esc(d.as_of)}</p><p>缺口或边界：${esc(d.missing.join('；') || '无已记录缺口')}</p></details>`).join('')}</div></details>
    <details class="analysis-detail"><summary>策略适配与研究边界</summary><div class="analysis-checks">${core.strategies.map(s=>`<div class="analysis-check"><span>${esc(s.code)} ${esc(s.label)}</span><b>${esc(labels[s.status] || s.status)}</b></div>`).join('')}</div><ul>${core.limitations.map(s=>`<li>${esc(s)}</li>`).join('')}</ul><p class="subtle">版本 ${esc(core.version)}</p></details>
  </section>`;
}

function renderPriceVolume(observation) {
  if (!observation) return '';
  const r = observation.latest || {};
  const events = (observation.timeline || []).filter(row=>row.event || row.invalid_reason).slice(-15).reverse();
  const reasons = {NO_COMBINATION:'未满足观察组合', MISSING_OR_INVALID_BASELINE:'行情或历史基准不足', FLOAT_DENOMINATOR_CHANGE:'流通股本代理变化，不能直接比较换手', ILLIQUID_LOCKED_OR_GAP:'成交不足、零振幅或异常跳空', ACTIVITY_WITHOUT_NET_PROGRESS:'交易活动增加，净推进有限', ABOVE_FROZEN_LEVEL_WITH_LOW_ACTIVITY:'已知结构上方低量有效推进', LOW_ACTIVITY_DOWN_NOT_A_BOTTOM:'低量下跌，不能视为见底', LOCAL_ACTIVITY_JUMP_WEAK_RESULT:'局部活动激增，关键位置收盘偏弱'};
  const endings = {TTL_EXPIRED:'观察到期', STRUCTURE_FAILED:'结构失效', NEW_BREAKOUT:'新突破，不拼接旧序列', DATA_OR_LIQUIDITY_BREAK:'数据或流动性中断'};
  return `<section class="section" aria-label="量价观察">
    <div class="section-head"><h2>量价观察</h2><span>只读研究 · 参数待校准</span></div>
    <p><b>${esc(r.label || '缺少可计算行情')}</b> · ${esc(r.date || '')} · ${esc(reasons[r.reason] || r.reason || '')}</p>
    <dl class="analysis-state"><div><dt>收盘推进 / 实体 / 跳空</dt><dd>${num(r.progress_atr,2)} / ${num(r.body_atr,2)} / ${num(r.gap_atr,2)} ATR</dd></div>
      <div><dt>相对换手 / 成交量 / 成交额</dt><dd>${num(r.relative_turnover,2)} / ${num(r.relative_volume,2)} / ${num(r.relative_amount,2)}</dd></div>
      <div><dt>观察起点 / 最近转换</dt><dd>${esc(r.first_seen || '未建立')} / ${esc(r.latest_event_at || '无')} ${esc(observation.event_labels?.[r.latest_event] || '')}</dd></div>
      <div><dt>连续观察</dt><dd>${r.episode_id ? `${r.episode_valid?'有效':'已结束'} · ${num(r.age_sessions,0)} / ${num(r.expires_after_sessions,0)} 个活跃交易日` : '未形成连续序列'}</dd></div></dl>
    <div class="stage-legend">${Object.entries(observation.labels || {}).map(([key,label])=>`<span><i style="background:${observation.colors[key]}"></i>${esc(label)}</span>`).join('')}</div>
    <details class="analysis-detail"><summary>转换轨迹与计算口径</summary>
      <p>${esc(observation.notice)} ${esc(observation.source_notice || '')}</p>
      <p>基准：此前${num(observation.rules.baseline_sessions,0)}日中位数，最少${num(observation.rules.min_baseline,0)}日；截至 ${esc(r.baseline_end || '未知')}。活动依据：${esc({TURNOVER:'换手率', VOLUME:'成交量（换手缺失）', MISSING:'缺失'}[r.activity_source] || '未知')}。只有收盘后可知。</p>
      <p>父突破：${esc(r.parent_breakout_id || '尚未绑定')}；观察ID：${esc(r.episode_id || '无')}。缺口：${esc(r.missing || '无上述字段缺口')}。价格层级均为复权指数，不是人民币报价。</p>
      <div class="table-wrap"><table><thead><tr><th>可知日期</th><th>转换/结束</th><th>父突破</th><th>观察起点</th></tr></thead><tbody>${events.map(e=>`<tr><td>${esc(e.known_at)}</td><td>${esc(e.event_label || endings[e.invalid_reason] || e.invalid_reason)}</td><td>${esc(e.parent_breakout_id || '未绑定')}</td><td>${esc(e.first_seen)}</td></tr>`).join('') || '<tr><td colspan="4">未形成转换</td></tr>'}</tbody></table></div>
      <p class="subtle">${esc(observation.version)} · 不自动解除风险、不自动补仓。</p>
    </details></section>`;
}

function priceVolumeTrace(observation, dates) {
  if (!observation?.timeline?.length) return [];
  const byDate = Object.fromEntries(observation.timeline.map(r=>[r.date,r]));
  const keys = Object.keys(observation.labels), n = keys.length;
  const colors = keys.flatMap((key,i)=>[[i/n,observation.colors[key]],[(i+1)/n,observation.colors[key]]]);
  const traces = [{type:'heatmap', x:dates, y:[0], z:[dates.map(d=>Math.max(0,keys.indexOf(byDate[d]?.state)))],
    text:[dates.map(d=>{const r=byDate[d] || {}; return `${esc(r.label || '未明确')}<br>${esc(r.event_label || '')}<br>推进 ${num(r.progress_atr,2)}ATR · 相对换手 ${num(r.relative_turnover,2)}<br>起点 ${esc(r.first_seen || '--')} · 父突破 ${esc(r.parent_breakout_id || '--')}`;})],
    hovertemplate:'%{x}<br>%{text}<extra>只读量价观察</extra>', zmin:-.5, zmax:n-.5,
    colorscale:colors, showscale:false, showlegend:false, name:'量价观察', xaxis:'x', yaxis:'y3'}];
  const visibleDates = new Set(dates);
  const transitions = observation.timeline.filter(r=>r.event && visibleDates.has(r.date));
  if (transitions.length) traces.push({type:'scatter', mode:'markers', x:transitions.map(r=>r.date), y:transitions.map(()=>0),
    marker:{symbol:'diamond',size:7,color:'#263c40',line:{color:'#fff',width:1}},
    text:transitions.map(r=>esc(r.event_label)), showlegend:false, xaxis:'x', yaxis:'y3',
    hovertemplate:'%{x}<br>%{text}<extra>量价转换，不是买卖点</extra>'});
  return traces;
}

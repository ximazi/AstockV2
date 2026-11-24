/* Frozen evidence is read-only; human judgements are persisted as separate revisions. */
let reviewCase = null;
let reviewRecord = null;
let reviewExitSelection = null;
let reviewEntryExitStudy = null;
let reviewPriceVolume = null;
let reviewStages = {};
let reviewDirty = false;
let reviewSaving = false;
const reviewFields = {summary:'复盘总结', lifecycle_notes:'生命周期判断', entry_notes:'入场时机判断', exit_notes:'卖点与风险判断', ai_notes:'AI分析记录', improvements:'改进假设与待验证项'};
const reviewIcon = name => `<img src="/static/vendor/lucide/${name}.svg" width="16" height="16" alt="">`;
const REVIEW_PAGE_SIZE = 20;
let reviewLibrary = {cases:[]};
let reviewPage = 0;
let reviewQuery = '';
let reviewOutcome = 'ALL';
let reviewRoute = 'ALL';
const reviewRouteLabels = {R1:'R1 · 个股点火反查', R2:'R2 · 个股先行埋伏', R3:'R3 · 题材传播承接'};
const reviewRouteLabel = route => reviewRouteLabels[route] || route || '未标识路径';
const reviewSelected = new Set();
let reviewJob = null;
let reviewJobTimer = null;
let reviewBatchStarting = false;
let reviewBatchError = '';
let reviewRequest = 0;

function reviewOutcomeOf(row) {
  if (row.trade_status === 'CLOSED') return row.net_pct > 0 ? 'PROFIT' : row.net_pct < 0 ? 'LOSS' : 'FLAT';
  if (row.trade_status?.startsWith('ENTRY') || row.trade_status?.startsWith('INVALID')) return 'NO_FILL';
  return 'UNRESOLVED';
}
const reviewOutcomeLabels = {PROFIT:'盈利', LOSS:'亏损', FLAT:'持平', NO_FILL:'未成交', UNRESOLVED:'未结束/待核验'};

function reviewStatusLabel(row) {
  if (row.signal_review_status === 'REVIEWED') return row.revision && row.review_status === 'REVIEWED' ? '已复盘' : '已复盘（原方案）';
  if (row.revision) return row.review_status === 'REVIEWED' ? '已复盘' : '草稿';
  return row.signal_review_status === 'DRAFT' ? '草稿（原方案）' : '待复盘';
}

function filteredReviewCases() {
  const query = reviewQuery.trim().toLowerCase();
  return reviewLibrary.cases.filter(r =>
    (reviewRoute === 'ALL' || r.route === reviewRoute) &&
    (reviewOutcome === 'ALL' || (reviewOutcome === 'SAVED' ? r.revision > 0 || r.exit_selection?.related_reviews?.length > 0 : reviewOutcomeOf(r) === reviewOutcome)) &&
    (!query || `${r.symbol} ${r.name} ${r.signal_date}`.toLowerCase().includes(query)));
}

function confirmReviewLeave() {
  if (reviewSaving) return false;
  if (reviewDirty && !window.confirm('分析尚未保存，确定离开并放弃本页修改？')) return false;
  reviewDirty = false;
  return true;
}

async function loadReviewCases() {
  const data = await getJson('/api/reviews');
  if (sourceMode !== 'review') return;
  reviewLibrary = data;
  const known = new Set(data.cases.map(r=>r.id));
  for (const id of reviewSelected) if (!known.has(id)) reviewSelected.delete(id);
  $('#candidateDate').textContent = `${data.cases.length}案例 · 每信号一例`;
  renderReviewList();
  if (!reviewJob) {
    try {
      const jobs = await getJson('/api/review-exports');
      if (jobs.length) { reviewJob = jobs[0]; renderReviewJob(); pollReviewJob(); }
    } catch (error) { console.warn(error.message); }
  }
}

function renderReviewList() {
  if (sourceMode !== 'review') return;
  const filtered = filteredReviewCases(), pages = Math.max(1, Math.ceil(filtered.length / REVIEW_PAGE_SIZE));
  reviewPage = Math.min(reviewPage, pages - 1);
  const visible = filtered.slice(reviewPage * REVIEW_PAGE_SIZE, (reviewPage + 1) * REVIEW_PAGE_SIZE);
  $('#candidateList').innerHTML = `<div class="review-list-controls">
    <input id="reviewQuery" type="search" aria-label="搜索复盘案例" placeholder="代码、名称或信号日期" value="${esc(reviewQuery)}">
    <div class="review-list-filters">
    <select id="reviewRoute" aria-label="复盘路径">${Object.entries({ALL:'全部路径',...reviewRouteLabels}).map(([k,v])=>`<option value="${k}" ${reviewRoute===k?'selected':''}>${v}</option>`).join('')}</select>
    <select id="reviewOutcome" aria-label="案例结果">${Object.entries({ALL:'全部结果',...reviewOutcomeLabels,SAVED:'已有分析'}).map(([k,v])=>`<option value="${k}" ${reviewOutcome===k?'selected':''}>${v}</option>`).join('')}</select></div>
    <div class="review-selection"><label><input id="reviewSelectPage" type="checkbox">本页</label><label><input id="reviewSelectFiltered" type="checkbox">筛选结果(${filtered.length})</label><button id="reviewClear" type="button" title="清空所有勾选" aria-label="清空所有勾选">${reviewIcon('trash-2')}</button></div>
    <div id="reviewSelectionCount" role="status"></div>
    <div class="review-toolbar review-bulk-buttons"><button id="bulkCase" type="button">${reviewIcon('download')}批量案例</button><button id="bulkReview" type="button">${reviewIcon('download')}批量分析</button></div>
    <div id="reviewJobStatus" class="review-job" role="status"></div>
    <details><summary>案例范围</summary><p class="subtle">${esc(reviewLibrary.scope)}</p></details>
    <div class="review-pagination"><button id="reviewPrev" type="button" title="上一页" aria-label="上一页" ${reviewPage===0?'disabled':''}>&#8592;</button><span>${reviewPage+1} / ${pages}页 · 每页${REVIEW_PAGE_SIZE}例</span><button id="reviewNext" type="button" title="下一页" aria-label="下一页" ${reviewPage+1===pages?'disabled':''}>&#8594;</button></div>
  </div>` + visible.map(row => `<div class="review-list-row"><input type="checkbox" data-review-select="${esc(row.id)}" aria-label="选择 ${esc(row.name)} ${esc(row.signal_date)} ${esc(row.exit_policy)}" ${reviewSelected.has(row.id)?'checked':''}>
    <button class="candidate ${reviewCase?.summary.id === row.id ? 'active' : ''}" data-review="${esc(row.id)}">
      <div class="candidate-top"><b class="candidate-name">${esc(row.name)}</b><span class="candidate-tier">${reviewStatusLabel(row)}</span></div>
      <div class="candidate-meta" style="margin-left:0"><span>${esc(row.symbol)}</span><span>${esc(row.signal_date)}</span></div>
      <div class="candidate-meta" style="margin-left:0">${esc(reviewRouteLabel(row.route))}</div>
      <div class="candidate-meta" style="margin-left:0">${esc(row.exit_policy)} · ${reviewOutcomeLabels[reviewOutcomeOf(row)]} ${row.net_pct == null ? '' : num(row.net_pct,2)+'%'}</div>
      ${row.exit_selection ? `<div class="candidate-meta" style="margin-left:0">${esc(row.exit_selection.label)}${row.exit_selection.related_reviews.length ? ' · 另有原方案分析' : ''}</div>` : ''}
      ${row.sample_label ? `<div class="candidate-meta" style="margin-left:0">亏损诊断 · ${esc(row.sample_label)}</div>` : ''}
      ${row.input_error ? '<div class="candidate-meta">输入待核验</div>' : ''}
    </button></div>`).join('') + (!filtered.length ? '<p class="subtle">当前路径及筛选条件下无匹配案例</p>' : '');
  document.querySelectorAll('[data-review]').forEach(button => button.addEventListener('click', () => {
    if (confirmReviewLeave()) runReviewCase(button.dataset.review);
  }));
  const checked = rows => rows.length > 0 && rows.every(r=>reviewSelected.has(r.id));
  const refreshSelection = () => {
    $('#reviewSelectPage').checked = checked(visible);
    $('#reviewSelectFiltered').checked = checked(filtered);
    $('#reviewSelectPage').indeterminate = !checked(visible) && visible.some(r=>reviewSelected.has(r.id));
    $('#reviewSelectFiltered').indeterminate = !checked(filtered) && filtered.some(r=>reviewSelected.has(r.id));
    $('#reviewSelectionCount').textContent = `已勾选 ${reviewSelected.size} 例（跨页、跨筛选保留）`;
    for (const id of ['bulkCase','bulkReview']) $('#'+id).disabled = !reviewSelected.size || reviewBatchStarting || ['RUNNING','QUEUED'].includes(reviewJob?.status);
  };
  document.querySelectorAll('[data-review-select]').forEach(box=>box.addEventListener('change',()=>{
    if (box.checked) reviewSelected.add(box.dataset.reviewSelect); else reviewSelected.delete(box.dataset.reviewSelect);
    refreshSelection();
  }));
  for (const [id, rows] of [['reviewSelectPage',visible],['reviewSelectFiltered',filtered]]) $('#'+id).addEventListener('change', e=>{
    rows.forEach(r=> e.target.checked ? reviewSelected.add(r.id) : reviewSelected.delete(r.id)); renderReviewList();
  });
  $('#reviewClear').addEventListener('click',()=>{reviewSelected.clear();renderReviewList();});
  $('#reviewPrev').addEventListener('click',()=>{reviewPage--;renderReviewList();});
  $('#reviewNext').addEventListener('click',()=>{reviewPage++;renderReviewList();});
  $('#reviewQuery').addEventListener('input', e=>{
    const caret=e.target.selectionStart; reviewQuery=e.target.value; reviewPage=0; renderReviewList();
    $('#reviewQuery').focus(); $('#reviewQuery').setSelectionRange(caret,caret);
  });
  $('#reviewOutcome').addEventListener('change', e=>{reviewOutcome=e.target.value;reviewPage=0;renderReviewList();});
  $('#reviewRoute').addEventListener('change', e=>{reviewRoute=e.target.value;reviewPage=0;renderReviewList();});
  $('#bulkCase').addEventListener('click',()=>startReviewBatch('case'));
  $('#bulkReview').addEventListener('click',()=>startReviewBatch('review'));
  refreshSelection(); renderReviewJob();
}

function renderReviewJob() {
  const target=$('#reviewJobStatus'); if (!target) return;
  if (!reviewJob) { target.textContent=reviewBatchError; return; }
  const labels={QUEUED:'等待导出',RUNNING:'正在导出',COMPLETE:'导出完成',PARTIAL:'部分失败',FAILED:'导出失败',CANCELLED:'已取消',INTERRUPTED:'任务中断'};
  const running=['QUEUED','RUNNING'].includes(reviewJob.status);
  target.innerHTML=`${esc(reviewBatchError)}<br>${labels[reviewJob.status] || esc(reviewJob.status)} ${reviewJob.completed}/${reviewJob.total} · 失败${reviewJob.failed}<br>${esc(reviewJob.error || '')}`+
    (running?'<button id="cancelReviewBatch" type="button">取消导出</button>':'')+
    (reviewJob.download_url?`<a href="${esc(reviewJob.download_url)}" download>${reviewIcon('download')}下载ZIP${reviewJob.status==='COMPLETE'?'':'（含未完成清单）'}</a>`:'');
  if (running) $('#cancelReviewBatch').addEventListener('click',async()=>{
    await fetch(`/api/review-exports/${reviewJob.id}/cancel`,{method:'POST'}); $('#cancelReviewBatch').disabled=true; $('#cancelReviewBatch').textContent='正在取消';
  });
  for (const id of ['bulkCase','bulkReview']) if ($('#'+id)) $('#'+id).disabled = !reviewSelected.size || reviewBatchStarting || running;
}

async function pollReviewJob() {
  clearTimeout(reviewJobTimer);
  if (!reviewJob || !['QUEUED','RUNNING'].includes(reviewJob.status)) return;
  try { reviewJob=await getJson(`/api/review-exports/${reviewJob.id}`); renderReviewJob(); }
  catch(error) { if ($('#reviewJobStatus')) $('#reviewJobStatus').textContent=`状态读取失败，正在重试：${error.message}`; }
  if (['QUEUED','RUNNING'].includes(reviewJob.status)) reviewJobTimer=setTimeout(pollReviewJob,1500);
}

async function startReviewBatch(kind) {
  if (reviewBatchStarting || reviewSaving || !reviewSelected.size) return;
  reviewBatchStarting=true;
  reviewBatchError='';
  try {
    if (kind==='review' && reviewDirty && reviewSelected.has(reviewCase?.summary.id) && !await saveReviewRecord()) return;
    const ids=[...reviewSelected];
    if (kind==='case' && ids.length>100 && !window.confirm(`将导出${ids.length}个完整案例及K线图，可能耗时较长并占用较多磁盘空间，继续？`)) return;
    const response=await fetch('/api/review-exports',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case_ids:ids,kind})});
    const data=await response.json(); if (!response.ok) throw new Error(data.error || '批量导出失败');
    reviewJob=data; renderReviewJob(); pollReviewJob();
  } catch(error) { reviewBatchError=error.message; }
  finally { reviewBatchStarting=false; renderReviewJob(); }
}

async function runReviewCase(id) {
  const requestId=++reviewRequest;
  $('#runStatus').textContent = '正在载入复盘案例';
  try {
    const data = await getJson(`/api/reviews/${encodeURIComponent(id)}`);
    if (sourceMode !== 'review' || requestId!==reviewRequest) return;
    reviewCase = data.case;
    reviewRecord = data.review;
    reviewExitSelection = data.exit_selection;
    reviewEntryExitStudy = data.entry_exit_study;
    reviewPriceVolume = data.price_volume_context;
    reviewStages = data.stages;
    reviewDirty = false;
    renderReviewWorkbench();
    renderReviewList();
    $('#symbolInput').value = reviewCase.summary.symbol;
    $('#dateInput').value = reviewCase.summary.window_end;
    $('#runStatus').textContent = `案例复盘 · ${reviewCase.summary.name}`;
  } catch (error) {
    if (sourceMode !== 'review' || requestId!==reviewRequest) return;
    $('#workspace').innerHTML = `<div class="error">${esc(error.message)}</div>`;
    $('#runStatus').textContent = '复盘案例载入失败';
  }
}

function annotationRow(item = {}) {
  const s = reviewCase.summary;
  return `<div class="review-annotation">
    <label>起始日期<input type="date" name="start" min="${s.window_start}" max="${s.window_end}" value="${esc(item.start || s.signal_date)}" required></label>
    <label>结束日期<input type="date" name="end" min="${s.window_start}" max="${s.window_end}" value="${esc(item.end || s.signal_date)}" required></label>
    <label>人工阶段<select name="stage">${Object.entries(reviewStages).map(([key,label]) => `<option value="${esc(key)}" ${key === (item.stage || 'UNCERTAIN') ? 'selected' : ''}>${esc(label)}</option>`).join('')}</select></label>
    <label>识别依据<textarea name="note" maxlength="5000" rows="2">${esc(item.note || '')}</textarea></label>
    <button type="button" class="review-remove" title="删除这条人工标注" aria-label="删除这条人工标注">${reviewIcon('trash-2')}</button>
  </div>`;
}

function markReviewDirty() {
  reviewDirty = true;
  $('#reviewSaveStatus').classList.remove('failed');
  $('#reviewSaveStatus').textContent = '有未保存的修改';
  $('#exportReviewDraft').hidden = false;
}

function renderEntryExitStudy(study, caseId) {
  if (!study) return '';
  const statuses = {CLOSED:'已平仓', CENSORED:'未结束/待会话', WAIT_CENSORED:'等待中，窗口未满', WINDOW_EXPIRED:'回踩窗口到期', INVALID_BEFORE_ENTRY:'等待期间原结构失效', ENTRY_OPEN_BIAS:'开盘偏离过大', ENTRY_OPEN_GAP:'高开超限', ENTRY_OPEN_STRUCTURE:'开盘低于结构位', ENTRY_LIMIT_UP:'开盘涨停不可买', HOLDING_DATA_UNVERIFIED:'持有期数据待核验'};
  const exitLabels = {MATURE_WEAK_CLOSE_EXIT:'成熟后弱收盘，次日退出', STRUCTURE_CLOSE_EXIT:'收盘结构失效，次日退出', DEFERRED_STOP:'受T+1等约束延迟止损', INITIAL_STOP:'绝对风险止损', TRAIL_STOP:'跟踪保护', HORIZON:'持有窗口结束'};
  const table = rows => `<div class="table-wrap"><table><thead><tr><th>固定假设</th><th>测试 / 确认</th><th>买入</th><th>末次退出</th><th>净价差</th><th>状态 / 原因</th></tr></thead><tbody>${rows.map(([label,r])=>`<tr><td>${esc(label)}</td><td>${esc(r.test_date || '--')}<br>${esc(r.signal_date || '--')}</td><td>${esc(r.entry_date || '--')}<br>${num(r.entry_price,3)}</td><td>${esc(r.exit_date || '--')}<br>${num(r.fills.at(-1)?.raw_price,3)}</td><td>${r.net==null?(r.entry_date?'未结算':'未成交/等待'):num(r.net,2)+'%'}</td><td>${esc(statuses[r.status] || r.status)}<br>${esc(exitLabels[r.reason] || r.reason || '')}</td></tr>`).join('')}</tbody></table></div>`;
  return `<section class="section" aria-label="买卖点改进验证"><div class="section-head"><h2>买卖点改进验证</h2><a href="/api/reviews/${encodeURIComponent(caseId)}/entry-exit-study" download title="下载买卖点研究JSON">${reviewIcon('download')}研究附件</a></div>
    <p class="review-notice">研究观察，尚非验证通过的买入策略。${esc(study.notice)}</p>
    ${table(['baseline','profit_only','pullback'].map(k=>[study.labels[k],study.arms[k]]))}
    <details><summary>归因对照与独立再参与</summary>${table(['opening_only','pullback_entry_only','strong_reclaim'].map(k=>[study.labels[k],study.arms[k]]))}
    <p>${esc(study.strong_reclaim_rule)}</p><h3>原分批交易退出后的新机会</h3>${table(study.followups_after_baseline.map((r,i)=>['独立跟踪 '+(i+1),r]))}
    <p>退出当日或之后出现独立的新确认，最早下一会话重新参与；准备证据可在持有期间形成，同一个确认不重复使用，不把后续行情拼入原交易。</p></details>
    <details><summary>触发证据、价格拒绝与退出流水</summary>${Object.entries(study.arms).map(([k,r])=>`<h3>${esc(study.labels[k])}</h3><p>${(r.rejected || []).map(x=>esc(x.date)+'：'+esc(statuses[x.status] || x.status)).join('；') || '无已记录的开盘拒绝'}</p>${r.fills.map(x=>`<p>${esc(x.date)} · ${num(x.raw_price,3)} · 原始仓位${num(x.weight*100,0)}% · ${esc(exitLabels[x.reason] || x.reason)}</p>`).join('')}`).join('')}</details>
    <details><summary>固定规则与边界</summary>${Object.values(study.rules).map(x=>`<p>${esc(x)}</p>`).join('')}</details></section>`;
}

function renderReviewWorkbench() {
  const c = reviewCase, s = c.summary, r = reviewRecord, trade = c.source_row;
  const marks = c.view.case_review.events;
  const buyMark=marks.find(m=>m.kind==='BUY'), sellMarks=marks.filter(m=>m.kind==='SELL');
  const lastSell=sellMarks[sellMarks.length-1];
  currentData = {...c.view, price_volume_context:reviewPriceVolume};
  $('#workspace').innerHTML = `
    <section class="section" aria-label="案例概要">
      <div class="section-head"><div><h1 class="review-heading">${esc(s.name)} <small>${esc(s.symbol)}</small></h1><span>${esc(reviewRouteLabel(s.route))} / ${esc(s.signal_date)}</span></div>
        <div class="review-toolbar"><a href="/api/reviews/${encodeURIComponent(s.id)}/export/case" download title="导出案例ZIP">${reviewIcon('download')}导出案例</a><button type="button" id="exportReviewResult" title="导出已保存分析ZIP">${reviewIcon('download')}导出分析结果</button></div></div>
      <p class="review-notice">${esc(c.notice)}</p>
      ${reviewExitSelection ? `<p class="review-notice">${esc(reviewExitSelection.label)}：${esc(reviewExitSelection.notice)}${s.id!==reviewExitSelection.selected_id?' 当前查看原方案归档，不是列表保留方案。':''}</p>
      ${reviewExitSelection.related_reviews.length || s.id!==reviewExitSelection.selected_id ? `<details ${reviewExitSelection.related_reviews.some(item=>item.review_status==='REVIEWED')?'open':''}><summary>该信号已有分析与保留方案</summary><p>原方案的人工结论独立保留，不自动移用到新的买卖流水。${r.status!=='REVIEWED' && reviewExitSelection.related_reviews.some(item=>item.review_status==='REVIEWED')?'本信号已复盘，当前查看的退出方案尚未完成独立复盘。':''}</p><div class="review-toolbar"><button type="button" data-related-review="${esc(reviewExitSelection.selected_id)}">查看列表保留方案</button>${reviewExitSelection.related_reviews.map(item=>`<button type="button" data-related-review="${esc(item.id)}">${esc(item.exit_policy)} · ${item.review_status==='REVIEWED'?'已复盘':'草稿'} · 原分析版本${item.revision}</button>`).join('')}</div></details>` : ''}` : ''}
      <dl class="review-meta"><div><dt>原始信号 / 退出规则</dt><dd>accepted_bias2 / ${esc(s.exit_policy)}<br>${esc(trade.status)}</dd></div><div><dt>回测买入</dt><dd>${esc(buyMark?.date || '未成交/待入场')} · ${num(buyMark?.raw_price, 3)}</dd></div><div><dt>${sellMarks.length>1?'回测分批卖出（末笔）':'回测卖出'}</dt><dd>${esc(trade.exit_date || (lastSell?'部分退出，尚未平仓':'尚无卖出'))} · ${num(lastSell?.raw_price,3)}</dd></div><div><dt>单笔回测净价差</dt><dd>${trade.net == null ? '未结束，不计已实现净收益' : num(trade.net,2)+'%'} · 固定总成本假设0.30个百分点</dd></div></dl>
      ${sellMarks.length>1?`<details><summary>全部卖出流水 · ${sellMarks.length}笔</summary>${sellMarks.map(m=>`<p>${esc(m.date)} · 原价${num(m.raw_price,3)} · 原始仓位${num(m.weight*100,0)}% · ${esc(m.reason)}</p>`).join('')}</details>`:''}
      <details><summary>来源、观察区间与复现口径</summary><p class="review-summary-text">${esc(c.selection)}<br>${esc(s.window_start)} 至 ${esc(s.window_end)}，${c.daily.length}个活跃交易日。${esc(c.reconstruction.boundary_note)}<br>本地周期失效：${esc(c.reconstruction.target_cycle_failed_at || '截至窗口末尚未观测到')}。原交易流水已核对一致；并非完整选股能力验证。<br>引擎 ${esc(c.engine_version)} / ${esc(c.reconstruction.cycle_version)}<br>当前行业分类：${esc(c.view.identity.industry || '未知')}，不是历史题材证据。<br>案例ID：${esc(s.id)}<br>SHA256：${esc(c.case_hash)}</p></details>
    </section>
    ${renderEntryExitStudy(reviewEntryExitStudy, s.id)}
    ${renderPriceVolume(reviewPriceVolume)}
    <section class="section" aria-label="复盘生命周期图">
      <div class="section-head"><h2>完整观察窗口 · 生命周期与回测买卖</h2><div class="stage-controls"><label><input id="stageBackgroundToggle" type="checkbox" ${showStageBackground?'checked':''}> 阶段背景</label><label>背景浓度<input id="stageOpacity" type="range" min="15" max="50" step="5" value="${Math.round(stageOpacity*100)}" aria-label="背景浓度"><output id="stageOpacityValue">${Math.round(stageOpacity*100)}%</output></label></div></div>
      <div class="stage-legend">${renderStageLegend(c.view)}</div><div id="stageReadout" class="stage-readout"></div><div id="priceChart"></div>
      <p class="subtle">${esc(c.view.chart.price_basis)}；星标是价格接受代理，不等于题材点火。曲线包含退出后的行情，未标识为自动买卖建议。</p>
    </section>
    <section class="section" aria-label="人工复盘分析">
      <div class="section-head"><h2>人工复盘</h2><span>系统原始阶段保持不变</span></div>
      <form id="reviewForm"><fieldset id="reviewInputs" style="border:0;margin:0;padding:0;min-width:0">
        <div class="review-controls"><label>复盘人<input name="reviewer" maxlength="100" value="${esc(r.reviewer)}"></label>
        <label>复盘状态<select name="status"><option value="DRAFT" ${r.status === 'DRAFT'?'selected':''}>草稿</option><option value="REVIEWED" ${r.status === 'REVIEWED'?'selected':''}>已完成复盘</option></select></label>
        <label>对系统判断<select name="verdict">${Object.entries({UNCERTAIN:'尚不能判断', AGREE:'基本认可', PARTIAL:'部分认可', DISAGREE:'不认可'}).map(([key,label])=>`<option value="${key}" ${r.verdict===key?'selected':''}>${label}</option>`).join('')}</select></label></div>
        <div class="review-form-grid">${Object.entries(reviewFields).map(([key,label])=>`<label class="${key==='summary'?'review-wide':''}">${label}<textarea name="${key}" maxlength="20000" rows="4">${esc(r[key])}</textarea></label>`).join('')}</div>
        <div class="section-head" style="margin-top:20px"><h2>人工阶段标注</h2><div class="review-toolbar"><button type="button" id="addReviewAnnotation">${reviewIcon('plus')}添加阶段</button></div></div>
        <div id="reviewAnnotations">${r.annotations.map(annotationRow).join('')}</div>
      </fieldset><div class="review-toolbar" style="margin-top:18px"><button type="submit" class="review-save" id="saveReview">${reviewIcon('save')}保存分析</button><button type="button" id="reloadReview">重新载入已保存版本</button><button type="button" id="exportReviewDraft" hidden>${reviewIcon('download')}导出未保存草稿</button></div>
      <p id="reviewSaveStatus" class="review-save-status" role="status">${r.revision ? '已保存' : '尚未填写人工结论'}</p><p id="reviewRevision" class="review-versions">版本 ${r.revision} · ${esc(r.updated_at || '未保存')}</p></form>
    </section>
    <section class="section"><details><summary>逐日系统证据 · ${c.daily.length}条</summary><div class="table-wrap review-daily"><table><thead><tr><th>日期</th><th>系统阶段</th><th>变化原因</th><th>准备身份 / 状态</th><th>突破状态</th><th>回踩形态</th><th>换手率</th></tr></thead><tbody>${c.daily.map(day=>`<tr><td>${esc(day.date)}</td><td>${esc(day.cycle_stage_label)}</td><td>${esc(day.cycle_reason_label)}</td><td>${esc(day.preparation_id || '无')}<br>${esc(day.preparation_state || '')}</td><td>${esc(day.breakout_status)}</td><td>${esc(day.setup_status_label)}</td><td>${num(day.turnover_pct,2)}%</td></tr>`).join('')}</tbody></table></div></details></section>`;
  renderChart(currentData);
  document.querySelectorAll('[data-related-review]').forEach(button=>button.addEventListener('click',()=>{
    if (confirmReviewLeave()) runReviewCase(button.dataset.relatedReview);
  }));
  $('#stageBackgroundToggle').addEventListener('change', e => { showStageBackground = e.target.checked; updateStageDisplay(); });
  $('#stageOpacity').addEventListener('input', e => { stageOpacity = Number(e.target.value)/100; $('#stageOpacityValue').textContent = `${e.target.value}%`; updateStageDisplay(); });
  $('#reviewForm').addEventListener('input', markReviewDirty);
  $('#reviewForm').addEventListener('change', markReviewDirty);
  $('#reviewForm').addEventListener('submit', e => { e.preventDefault(); saveReviewRecord(); });
  $('#addReviewAnnotation').addEventListener('click', () => { $('#reviewAnnotations').insertAdjacentHTML('beforeend', annotationRow()); markReviewDirty(); });
  $('#reviewAnnotations').addEventListener('click', e => { const button = e.target.closest('.review-remove'); if (button) { button.closest('.review-annotation').remove(); markReviewDirty(); } });
  $('#reloadReview').addEventListener('click', () => { if (confirmReviewLeave()) runReviewCase(s.id); });
  $('#exportReviewDraft').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(reviewBody(), null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob); downloadReview(url, `${s.id}_unsaved_draft.json`); setTimeout(()=>URL.revokeObjectURL(url), 1000);
  });
  $('#exportReviewResult').addEventListener('click', async () => {
    if (reviewSaving || (reviewDirty && !await saveReviewRecord())) return;
    downloadReview(`/api/reviews/${encodeURIComponent(s.id)}/export/review`, `${s.id}_review.zip`);
  });
}

function downloadReview(url, filename) {
  const link = document.createElement('a'); link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
}

function reviewBody() {
  const form = new FormData($('#reviewForm'));
  const body = {case_id:reviewCase.summary.id, case_hash:reviewCase.case_hash, revision:reviewRecord.revision};
  for (const key of ['reviewer','status','verdict',...Object.keys(reviewFields)]) body[key] = form.get(key);
  body.annotations = [...document.querySelectorAll('.review-annotation')].map(row => Object.fromEntries(['start','end','stage','note'].map(key=>[key,row.querySelector(`[name="${key}"]`).value])));
  return body;
}

async function saveReviewRecord() {
  if (reviewSaving || !$('#reviewForm').reportValidity()) return false;
  const body = reviewBody();
  reviewSaving = true;
  $('#reviewInputs').disabled = true;
  $('#saveReview').disabled = true;
  $('#reviewSaveStatus').textContent = '正在保存';
  try {
    const response = await fetch(`/api/reviews/${encodeURIComponent(reviewCase.summary.id)}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    reviewRecord = data;
    reviewDirty = false;
    $('#reviewSaveStatus').classList.remove('failed');
    $('#reviewSaveStatus').textContent = '分析已保存';
    $('#reviewRevision').textContent = `版本 ${data.revision} · ${data.updated_at}`;
    $('#exportReviewDraft').hidden = true;
    try { await loadReviewCases(); } catch (error) { $('#candidateList').innerHTML = `<div class="error">列表刷新失败：${esc(error.message)}</div>`; }
    return true;
  } catch (error) {
    $('#reviewSaveStatus').classList.add('failed');
    $('#reviewSaveStatus').textContent = `保存失败：${error.message}`;
    $('#exportReviewDraft').hidden = false;
    return false;
  } finally {
    reviewSaving = false;
    $('#reviewInputs').disabled = false;
    $('#saveReview').disabled = false;
  }
}

window.addEventListener('beforeunload', event => { if (reviewDirty || reviewSaving) { event.preventDefault(); event.returnValue = ''; } });
$('#reviewSource').addEventListener('click', () => switchSource('review'));

(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const esc = v => String(v ?? "").replace(/[&<>"']/g, x => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
  const labels = {PLAN_REVIEW:"可复核模拟计划",MARKET_WAIT:"等待市场",WAIT_SETUP:"等待回踩确认",RISK_WATCH:"结构风险",DATA_WAIT:"数据待核",UNCONFIRMED:"角色待确认",DAILY_LEADER_CANDIDATE:"日级领头候选",RELATIVE_STRENGTH_FRONT:"相对强势前排",REGISTERED:"已登记",PENDING_REVIEW:"待评审",OPEN:"已提交",PARTIAL:"部分成交",FILLED:"已成交",CANCELLED:"已撤余单",EXIT_BLOCKED:"退出受阻",UNKNOWN:"状态未知",HOLD:"持有",EXIT_REQUIRED:"应退出",REDUCE_REQUIRED:"应减仓",BUY:"买入",SELL:"卖出",strength_top3:"强势Top3",participation_top3:"参与Top3",legacy_top3:"旧榜Top3",momentum_top3:"动量基线Top3",all_candidates:"全候选"};
  const label = v => labels[v] || v || "--";
  Object.assign(labels,{DATA_WAIT:"数据待核",STRUCTURE_RISK:"结构风险未解除",WAIT_PULLBACK_CONFIRMATION:"回踩确认未完成",TRACKING_ONLY:"仅续踪",BENCHMARK_MISSING:"市场基准缺失",PASS:"通过",FAIL:"未通过",WAIT:"等待",OBSERVATION_ONLY:"仅观察",FIRST_PULLBACK:"推进后首次回踩",BREAKOUT_ACCEPTANCE:"突破接受",BASE_BREAK:"平台突破",LATER_PULLBACK:"后续回踩",PULLBACK_UNKNOWN:"回踩归属待确认"});
  const fmt = (v, digits=2) => v === null || v === undefined ? "--" : Number(v).toFixed(digits);
  const money = v => v == null ? "--" : (v / 100).toLocaleString("zh-CN", {minimumFractionDigits:2, maximumFractionDigits:2});
  let data, page = 0, currentDetail, diagData, auditData, preflightData, runData, intradayData, perfData, relayTimer;
  function error(e) { $("error").hidden = false; $("error").textContent = e.message || e; }
  async function api(path, body) {
    const response = await fetch(path, body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "请求失败"); return result;
  }
  const totals = (id, pairs) => $(id).innerHTML = pairs.map(([k,v]) => `<div class="total"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
  function state(v) { return `<span class="pill ${v.includes("RISK") || v.includes("EXIT") ? "risk" : v === "PLAN_REVIEW" ? "ready" : "wait"}">${esc(label(v))}</span>`; }
  function empty(id, n, text="暂无记录") { $(id).innerHTML = `<tr><td colspan="${n}" class="empty">${esc(text)}</td></tr>`; }
  async function load(identity="") {
    $("error").hidden = true; $("refresh").disabled = true;
    try {
      data = await api("/api/trading" + (identity ? "?snapshot_id=" + encodeURIComponent(identity) : ""));
      const s = data.snapshot;
      $("snapshots").innerHTML = data.snapshots.map(r => `<option value="${esc(r.id)}">${esc(r.as_of)} · ${esc(r.created_at.slice(11,19))} · ${esc(r.id.slice(0,8))}</option>`).join("");
      if(s) $("snapshots").value = s.snapshot_id;
      $("runDate").textContent = s ? `${s.as_of} · ${s.version}` : "尚无闭环扫描记录";
      $("registryCount").textContent = data.registry ? `${data.registry.summary.knowledge_items}项知识 · ${data.registry.summary.pending_sources}份来源待评审` : "知识登记未生成";
      totals("totals", [["候选与续踪",s?.summary.total || 0],["可计算强势排名",s?.summary.rankable || 0],["可复核模拟计划",s?.summary.paper_plan_eligible || 0],["真实订单",0]]);
      $("scope").textContent = s ? s.scope + " 排名基线未校准；强势分不代表胜率。" : "暂无结果";
      const records = s?.records || [];
      $("theme").innerHTML = '<option value="">全部题材</option>' + [...new Set(records.map(r=>r.theme))].sort().map(v=>`<option>${esc(v)}</option>`).join("");
      $("state").innerHTML = '<option value="">全部状态</option>' + [...new Set(records.map(r=>r.state))].map(v=>`<option value="${esc(v)}">${esc(label(v))}</option>`).join("");
      page = 0; renderCandidates(); renderPaper(); renderKnowledge(); renderValidation();
      loadReview(); loadFupan(); loadBoardRelay(); loadBrief(); loadPreflight(); loadDiagnostics(); loadAudit(); loadRun(); loadIntraday(); loadLedger();
    } catch(e) {error(e);} finally {$("refresh").disabled=false;}
  }
  function renderCandidates() {
    const q = $("search").value.trim().toLowerCase(), key = $("order").value;
    const rows = (data.snapshot?.records || []).filter(r => (!q || (r.symbol+r.name).toLowerCase().includes(q)) && (!$("theme").value || r.theme === $("theme").value) && (!$("route").value || r.route_ids.split("|").includes($("route").value)) && (!$("state").value || r.state === $("state").value)).sort((a,b)=>(a[key]??1e9)-(b[key]??1e9));
    page = Math.min(page,Math.max(0,Math.ceil(rows.length/20)-1));
    $("candidates").innerHTML = rows.slice(page*20,page*20+20).map(r => `<tr><td>${r.participation_rank} / ${r.strength_rank??"--"}</td><td><b>${esc(r.name)}</b><small>${esc(r.symbol)}</small></td><td>${esc(r.theme)}<small>${esc(r.route_ids||"未归属")}</small></td><td>${esc(label(r.role))}<small>因果带动未确认</small></td><td>${state(r.state)}</td><td>${fmt(r.strength_score)}<small>${r.rank_change==null?"首次/无可比前日":(r.rank_change>0?"上升 ":r.rank_change<0?"下降 ":"持平 ")+Math.abs(r.rank_change)}</small></td><td>${fmt(r.features.return5)}% / ${fmt(r.features.return20)}%</td><td>${fmt(r.features.relative_turnover)}倍</td><td><button data-detail="${esc(r.opportunity_id)}">查看证据</button></td></tr>`).join("");
    if(!rows.length) empty("candidates",9);
    $("pageLabel").textContent = `${page+1} / ${Math.max(1,Math.ceil(rows.length/20))} 页 · ${rows.length}只`;
    $("previous").disabled=page===0; $("next").disabled=(page+1)*20>=rows.length;
    const top=rows.filter(r=>r.strength_score!==null).slice(0,5).reverse();
    Plotly.react("strengthChart",[{type:"bar",orientation:"h",y:top.map(r=>r.name),x:top.map(r=>r.strength_score),marker:{color:top.map(r=>r.state==="RISK_WATCH"?"#b54d60":"#338363")},hovertemplate:"%{y} 强势分 %{x}<extra></extra>"}],{height:260,margin:{l:90,r:20,t:20,b:40},paper_bgcolor:"white",plot_bgcolor:"white",font:{family:"Microsoft YaHei",size:12},xaxis:{range:[0,100],title:{text:"冻结基线强势分"},gridcolor:"#e6ece7"},yaxis:{automargin:true},showlegend:false},{responsive:true,displayModeBar:false});
  }
  function renderPaper() {
    const p=data.paper; $("accountNotice").textContent=p.notice + " 首期仅首仓与退出模拟；条件补仓、盘中撮合尚未启用。"; $("accountForm").hidden=p.configured;
    totals("accountTotals",[["账面现金",money(p.account?.cash_cents)],["未成交预留",money(p.account?.reserved_cents)],["可用现金",money(p.account?.available_cents)],["账本对账",p.configured?(p.reconciled?"一致":"异常"):"未初始化"]]);
    $("positions").innerHTML=p.positions.map(r=>`<tr><td>${esc(r.plan.name)}<small>${esc(r.plan.symbol)}</small></td><td>${r.remaining}</td><td>${esc(r.lots[0]?.entry_day)}</td><td>${state(r.valuation?.status||"DATA_WAIT")}<small>${esc(r.valuation?.reason||r.valuation?.reasons?.join("; ")||"")}</small></td><td>${esc(r.plan.exit_version)}</td></tr>`).join("");
    if(!p.positions.length) empty("positions",5,"暂无成交驱动持仓");
    $("orders").innerHTML=p.orders.map(o=>`<tr><td>${esc(o.id.slice(0,12))}<small>${esc(o.created_at)}</small></td><td>${esc(label(o.side))}</td><td>${o.quantity} / ${o.filled}</td><td>${state(o.status)}</td><td>${money(o.reserve_cents)}</td><td>${o.side==="BUY"&&["OPEN","PARTIAL"].includes(o.status)?`<button data-cancel="${esc(o.id)}">撤销余单</button>`:"--"}</td></tr>`).join("");
    if(!p.orders.length) empty("orders",6);
    $("ledgerEvents").textContent=JSON.stringify({fills:p.fills,events:p.events},null,2);
  }
  function renderKnowledge() {
    const reg=data.registry; $("sources").innerHTML=(reg?.sources||[]).map(r=>`<tr><td>${esc(r.source_id)}<small>${esc(r.name)}</small></td><td>${esc(label(r.status))}</td><td>${esc(r.path)}</td><td>${esc(r.sha256)}</td></tr>`).join("");
    const q=$("knowledgeSearch").value.toLowerCase();
    $("knowledgeRows").innerHTML=(reg?.items||[]).filter(r=>(r.knowledge_id+r.claim).toLowerCase().includes(q)).map(r=>`<tr><td>${esc(r.knowledge_id)}</td><td>${esc(r.claim)}</td><td>${esc(r.disposition)}</td><td>${r.start_line}-${r.end_line}</td><td>${esc(r.validation)}</td></tr>`).join("");
  }
  function renderValidation() {
    const v=data.validation; $("validationNotice").textContent=v?.notice||"暂无后续验证样本。";
    totals("validationTotals",[["成熟观察行",v?.summary.mature_rows||0],["未成熟观察行",v?.summary.pending_rows||0],["独立机会数",v?.summary.unique_opportunities||0],["自动调参","关闭"]]);
    $("clockNotice").textContent = v?.clock?.note || "";
    $("comparisons").innerHTML=(v?.comparisons||[]).map(r=>`<tr><td>${esc(r.signal_day)}</td><td>${r.horizon}日</td><td>${esc(label(r.group))}</td><td>${r.selected_n} / ${r.mature_n}</td><td>${r.independent_opportunities ?? "--"}</td><td>${fmt(r.mean_return_pct)}%</td><td>${fmt(r.positive_fraction==null?null:r.positive_fraction*100)}%</td><td>${fmt(r.worst)}%</td><td>${fmt(r.top_decile_share)}</td><td>${r.complete?"完整":"未完整，不作定论"}</td></tr>`).join("");
    if(!v?.comparisons?.length) empty("comparisons",10,"暂无可比较结果");
    renderStrata();
  }

  const tone = v => v === "PASS" || v === "READY" ? "ready" : v === "FAIL" || v === "FAILED" || v === "BLOCKED" ? "risk" : "wait";
  const pill = v => `<span class="pill ${tone(v)}">${esc(label(v))}</span>`;
  Object.assign(labels,{REQUIRED:"必要",OPTIONAL:"可降级",READY:"通过",DEGRADED:"降级运行",BLOCKED:"缺必要输入",
    NOT_RUN:"未运行",INSUFFICIENT_DATA:"证据不足",WAITING:"条件未到",FAILED:"未通过",NECESSARY:"必要",SCORING:"加减分",
    SUFFICIENT:"样本足够",INSUFFICIENT_SAMPLE_RETAINED:"样本不足（保留不足状态）",
    MATURE:"已成熟",IMMATURE:"未成熟",DATA_GAP:"窗口内缺日",PROTOCOL_DAY_MISSING:"当日无冻结协议",
    WINDOW_IMMATURE:"后验窗口未成熟",NO_ACTIVE_ROWS:"当日无活跃行",NO_SNAPSHOT:"暂无快照"});

  async function loadPreflight() {
    try {
      preflightData = await api("/api/trading/preflight" + (data.snapshot ? "?as_of=" + encodeURIComponent(data.snapshot.as_of) : ""));
      const p = preflightData;
      $("preflightNotice").textContent = p.notice || "启动自检尚未运行。";
      totals("preflightTotals", [["自检结论", label(p.status)], ["必要项缺失", (p.blocking || []).length],
        ["降级项", (p.degraded || []).length], ["行情可用", p.data_wait?.counts?.ready ?? "--"]]);
      $("preflightRows").innerHTML = (p.checks || []).map(c =>
        `<tr><td>${esc(c.label)}</td><td>${esc(label(c.tier))}</td><td>${pill(c.status)}</td><td>${esc(c.detail)}</td><td>${esc(c.impact || "--")}</td></tr>`).join("");
      if (!p.checks?.length) empty("preflightRows", 5, "自检未运行");
      $("contractRows").innerHTML = Object.entries(p.contract || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
      const w = p.data_wait || {counts: {}};
      $("dataWaitRows").innerHTML = [["缺失行情", "missing"], ["未更新到当日", "stale"], ["窗口内缺日", "gapped"]].map(([name, key]) =>
        `<tr><td>${esc(name)}</td><td>${w.counts?.[key] ?? 0}</td><td>${esc((w[key] || []).slice(0, 8).join(" ") || "--")}</td></tr>`).join("");
    } catch (e) { error(e); }
  }

  async function loadDiagnostics() {
    try {
      const q = new URLSearchParams();
      if ($("diagRoute").value) q.set("route", $("diagRoute").value);
      if ($("diagVerdict").value) q.set("verdict", $("diagVerdict").value);
      diagData = await api("/api/trading/diagnostics" + (q.toString() ? "?" + q : ""));
      const d = diagData;
      $("diagNotice").textContent = d.notice || "";
      totals("diagTotals", [["已分析", d.summary?.analyzed ?? 0], ["有声明路径", d.summary?.with_route ?? 0],
        ["无路径（结构观察）", d.summary?.without_route ?? 0], ["筛选后命中", d.matched ?? 0]]);
      $("routeSummary").innerHTML = (d.routes || []).map(r =>
        `<tr><td><b>${esc(r.route)}</b> ${esc(r.label)}</td><td>${esc(r.question)}</td><td>${r.total}</td>
         <td>${Object.entries(r.verdicts || {}).map(([k, v]) => `${esc(label(k))} ${v}`).join("　")}</td>
         <td>${Object.entries(r.first_failures || {}).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${esc(k)} ${v}`).join("<br>") || "--"}</td></tr>`).join("");
      if (!d.routes?.length) empty("routeSummary", 5, "当日没有声明路径的候选");
      $("stepRows").innerHTML = (d.routes || []).flatMap(r => (r.steps || []).map(s =>
        `<tr><td>${esc(r.route)}</td><td>${esc(s.label)}</td><td>${esc(label(s.kind))}</td>
         <td>${s.denominator}</td><td>${s.counts?.PASS ?? 0}</td><td>${s.counts?.FAIL ?? 0}</td><td>${s.counts?.WAIT ?? 0}</td>
         <td>${s.counts?.UNKNOWN ?? 0}</td><td>${fmt(s.pass_rate == null ? null : s.pass_rate * 100, 1)}%</td></tr>`)).join("");
      if (!d.routes?.length) empty("stepRows", 9, "暂无环节统计");
      $("caseRows").innerHTML = (d.routes || []).flatMap(r =>
        Object.entries(r.cases || {}).flatMap(([verdict, rows]) => rows.map(c =>
          `<tr><td>${esc(r.route)}</td><td>${pill(verdict)}</td><td>${esc(c.name || c.symbol)}<small>${esc(c.symbol)}</small></td>
           <td>${esc(c.first_blocking_step || "--")}</td><td>${esc(c.next_condition || "--")}</td></tr>`))).join("");
      if (!d.routes?.length) empty("caseRows", 5, "暂无案例");
    } catch (e) { error(e); }
  }

  async function loadAudit() {
    try {
      auditData = await api("/api/trading/validation/audit");
      const a = auditData.ablation || {}, m = auditData.missed || {}, ps = auditData.protocols || {};
      $("protocolRows").innerHTML = (ps.protocols || []).map(r =>
        `<tr><td>${esc(r.protocol)}<small>${esc(r.version || "")}</small></td><td>${esc(r.mode || "--")}</td>
         <td>${esc((r.window?.start || "--") + " ~ " + (r.window?.end || "--"))}</td><td>${r.frozen_days}</td>
         <td>${esc((r.horizons || []).join("/") || "--")}</td></tr>`).join("");
      if (!ps.protocols?.length) empty("protocolRows", 5, "尚未登记协议");
      $("ablationNotice").textContent = a.status === "READY"
        ? `${a.protocol} ${a.as_of}：活跃 ${a.active_rows} 行，带记忆加分 ${a.memory_rows} 行；Top${a.top_k} 重合 ${a.top_overlap}。${a.finding}`
        : `${label(a.status)}。${a.notice || ""}`;
      $("ablationRows").innerHTML = (a.rank_moves || []).map(r =>
        `<tr><td>${esc(r.name || r.symbol)}<small>${esc(r.symbol)}</small></td><td>${r.removed_points}</td><td>${r.baseline_rank}</td><td>${r.ablated_rank}</td></tr>`).join("");
      if (!a.rank_moves?.length) empty("ablationRows", 4, "当日没有受记忆影响的名次");
      $("missedNotice").textContent = m.status === "READY"
        ? `${m.rule?.definition}；扫描 ${m.scanned_symbols} 只，符合后验 ${m.summary?.movers} 只，其中在池 ${m.summary?.in_pool} 只。${m.notice}`
        : `${label(m.status)}。${m.notice || ""}`;
      $("missedRows").innerHTML = (m.cases || []).slice(0, 50).map(c =>
        `<tr><td>${esc(c.name || c.symbol)}<small>${esc(c.symbol)}</small></td><td>${c.in_pool ? "在池" : "未进入候选域"}</td>
         <td>${esc(c.stage || "--")}</td><td>${esc((c.blockers || []).map(label).join("；") || c.reason || "--")}</td>
         <td>${fmt(c.close_gain_pct)}%</td></tr>`).join("");
      if (!m.cases?.length) empty("missedRows", 5, m.status === "READY" ? "没有符合后验定义的漏选案例" : "后验窗口未成熟，暂不给结论");
    } catch (e) { error(e); }
  }

  function renderStrata() {
    const v = data.validation, key = $("stratum").value;
    const rows = v?.strata?.[key] || [];
    $("strataRows").innerHTML = rows.map(r =>
      `<tr><td>${esc(r.group)}</td><td>${r.selected_n}</td><td>${r.mature_n}</td><td>${pill(r.status)}</td>
       <td>${fmt(r.mean)}%</td><td>${fmt(r.median)}%</td><td>${fmt(r.positive_fraction == null ? null : r.positive_fraction * 100)}%</td></tr>`).join("");
    if (!rows.length) empty("strataRows", 7, "暂无分层样本");
  }


  Object.assign(labels,{FRESH:"最新",PARTIAL_STALE:"部分过期",CONSISTENT:"账本一致",MISMATCH:"账本不一致",
    UNCONFIGURED:"未配置模拟账户",PENDING_EXITS:"待退出",POSITION_RISK:"持仓风险",
    EXPIRING_TASKS:"到期任务",NEW_CANDIDATES:"新增机会",PLAN_APPROVED:"批准计划",
    ORDER_BUY:"买入委托",ORDER_SELL:"卖出委托",FILL:"模拟成交",EXIT_REQUIRED:"应退出",
    REDUCE_REQUIRED:"应减仓",DONE:"已完成",CURRENT:"规则未变",FROZEN_RULES_RESTORED:"按原参数运行",
    PENDING_REVIEW:"规则待核"});

  async function loadRun() {
    try {
      runData = await api("/api/trading/run" + (data.snapshot ? "?as_of=" + encodeURIComponent(data.snapshot.as_of) : ""));
      const r = runData, f = r.freshness || {}, p = r.pending || {}, rec = r.reconciliation || {};
      $("runNotice").textContent = p.note || "";
      totals("runTotals", [["数据新鲜度", label(f.status)], ["账本对账", label(rec.status)],
        ["待处理任务", (p.tasks || []).length], ["运行锁", (r.locks || []).length ? "占用中" : "空闲"]]);
      $("freshRows").innerHTML = Object.entries(f.sources || {}).map(([k, v]) =>
        `<tr><td>${esc(k)}</td><td>${pill(v.fresh ? "PASS" : "FAIL")}</td><td>${esc(v.impact || "")}</td></tr>`).join("");
      if (!Object.keys(f.sources || {}).length) empty("freshRows", 3, "暂无数据源记录");
      $("taskRows").innerHTML = (p.tasks || []).map(t =>
        `<tr><td>${esc(label(t.group))}</td><td>${esc(t.symbol || t.plan_id || "--")}</td>
         <td>${esc(label(t.status))}</td><td>${esc(t.action || "--")}</td></tr>`).join("");
      if (!(p.tasks || []).length) empty("taskRows", 4, "没有待处理任务");
      $("checkpointRows").innerHTML = (r.checkpoints || []).map(c =>
        `<tr><td>${esc(c.stage)}</td><td>${esc(c.step)}</td><td>${pill(c.status === "DONE" ? "PASS" : c.status)}</td>
         <td>${esc(c.updated_at)}</td></tr>`).join("");
      if (!(r.checkpoints || []).length) empty("checkpointRows", 4, "本日尚无断点记录");
      const plans = (data.paper?.positions || []).map(x => x.plan);
      $("planPicker").innerHTML = '<option value="">选择计划</option>' +
        plans.map(x => `<option value="${esc(x.id)}">${esc(x.name)} ${esc(x.symbol)}</option>`).join("");
      if (plans.length) { $("planPicker").value = plans[0].id; loadTimeline(); }
      else { empty("timelineRows", 6, "尚无模拟计划"); empty("lotRows", 5, "尚无持仓批次"); }
    } catch (e) { error(e); }
  }

  async function loadTimeline() {
    const id = $("planPicker").value;
    if (!id) { empty("timelineRows", 6, "尚无模拟计划"); empty("lotRows", 5, "尚无持仓批次"); return; }
    try {
      const t = await api(`/api/trading/plans/${encodeURIComponent(id)}/timeline`);
      $("timelineNotice").textContent = `${t.notice} 规则分派：${label(t.dispatch?.status)}。`;
      $("timelineRows").innerHTML = (t.events || []).map(e =>
        `<tr><td>${esc(e.day)}</td><td>${esc(label(e.kind))}${e.tranche ? " 第" + e.tranche + "档" : ""}</td>
         <td>${esc(e.detail || "")}${e.status ? "（" + esc(label(e.status)) + "）" : ""}</td>
         <td>${e.quantity ?? "--"}${e.filled_quantity != null ? " / 已成" + e.filled_quantity : ""}</td>
         <td>${e.price_cents == null ? "--" : fmt(e.price_cents / 100)}</td>
         <td>${e.filled ? "已成交" : "未成交"}</td></tr>`).join("");
      if (!(t.events || []).length) empty("timelineRows", 6, "暂无事件");
      $("lotRows").innerHTML = (t.lots || []).map(l =>
        `<tr><td>${esc(l.entry_day)}</td><td>${l.quantity}</td><td>${l.remaining}</td>
         <td>${fmt(l.price_cents / 100)}</td><td>${fmt(l.entry_fee_cents / 100)}</td></tr>`).join("");
      if (!(t.lots || []).length) empty("lotRows", 5, "尚无成交批次");
    } catch (e) { error(e); }
  }

  Object.assign(labels,{SUPPORT:"支持",AGAINST:"反对",WAIT:"等待",UNKNOWN:"信息不足",
    OBSERVED_INTRADAY:"盘中已观察",OBSERVATION_ONLY_DATA_INSUFFICIENT:"仅观察·数据不足",
    OBSERVATION_ONLY_NEEDS_INTRADAY_SNAPSHOT:"仅观察·需尾盘快照",NOT_THIS_SETUP:"非该形态",
    MAPPED_NOT_ENABLED:"已映射未启用",AFTER_CLOSE:"收盘",INTRADAY_1430:"盘中14:30",
    NO_INTRADAY_RUN:"当日未做盘中运行",NO_CLOSING_SNAPSHOT:"无收盘快照",
    VALID:"通过",INVALID:"未通过",EXPIRED:"已过期",INVALID_STRUCTURE:"结构非法",
    TIMEOUT:"调用超时",CLIENT_ERROR:"调用失败",OK:"正常"});

  async function loadIntraday() {
    try {
      intradayData = await api("/api/trading/intraday?limit=200");
      const d = intradayData, s = d.summary || {};
      $("intradayNotice").textContent = d.notice || label(d.status);
      totals("intradayTotals", [["机会数", s.opportunities ?? 0], ["有盘中报价", s.with_intraday_quote ?? 0],
        ["可比口径", s.comparable ?? 0], ["盘中与收盘差异", s.differences ?? 0]]);
      const rows = (d.records || []).filter(r => r.intraday_quality !== "NO_QUOTE").slice(0, 60);
      $("intradayRows").innerHTML = rows.map(r =>
        `<tr><td>${esc(r.name)}<small>${esc(r.symbol)}</small></td><td>${state(r.close_state || "DATA_WAIT")}</td>
         <td>${r.intraday_quote == null ? "--" : fmt(r.intraday_quote)}<small>${esc(r.intraday_quality)}</small></td>
         <td>${r.comparable ? "可比" : "不可比"}</td>
         <td>${r.difference ? esc(r.difference.intraday) + " ≠ " + esc(r.difference.close) : "--"}</td>
         <td>${esc(r.difference?.reading || r.difference_note || "")}</td></tr>`).join("");
      if (!rows.length) empty("intradayRows", 6, "当日没有带新鲜报价的盘中观察");
      const picks = (d.records || []).slice(0, 200);
      $("modePicker").innerHTML = picks.map(r =>
        `<option value="${esc(r.opportunity_id)}">${esc(r.name)} ${esc(r.symbol)}</option>`).join("");
      renderModes();
    } catch (e) { error(e); }
  }

  function renderModes() {
    const id = $("modePicker").value;
    const row = (intradayData?.records || []).find(r => r.opportunity_id === id);
    $("modeRows").innerHTML = (row?.modes || []).map(m =>
      `<tr><td>${esc(m.label || m.mode)}</td><td>${pill(m.state)}</td><td>${esc(label(m.source))}</td>
       <td>${esc(m.evidence || "--")}</td><td>${esc(m.confirm || "--")}</td><td>${esc(m.invalidate || "--")}</td>
       <td>${m.executable ? "可执行" : "仅观察"}</td></tr>`).join("");
    if (!row) empty("modeRows", 7, "请选择机会");
    loadAssessment();
  }

  async function loadAssessment() {
    const id = $("modePicker").value;
    if (!id) return;
    try {
      const a = await api(`/api/trading/assessment/${encodeURIComponent(id)}`);
      $("aiNotice").textContent = a.injection_policy + " 未接入模型时结论固定为信息不足，不猜测。";
      $("evidenceRows").innerHTML = (a.bundle?.evidence || []).map(e =>
        `<tr><td>${esc(e.id)}</td><td>${esc(e.kind)}</td><td>${esc(e.label)}</td>
         <td>${esc(JSON.stringify(e.value))}</td><td>${esc(e.as_of)}</td><td>${esc(e.source)}</td></tr>`).join("");
      if (!(a.bundle?.evidence || []).length) empty("evidenceRows", 6, "暂无证据包");
      const rows = a.history?.assessments || [];
      $("aiRows").innerHTML = rows.map(r =>
        `<tr><td>${esc(r.created_at)}</td><td>${pill(r.assessment?.verdict)}</td>
         <td>${esc(r.assessment?.model_version || r.model)}</td><td>${pill(r.validation?.status)}</td>
         <td>${esc(r.assessment?.expires_at || "--")}</td>
         <td>${esc(r.assessment?.reason || (r.assessment?.unknowns || []).join("；") || "")}</td></tr>`).join("");
      if (!rows.length) empty("aiRows", 6, "尚未做过评估");
      const shadow = await api("/api/trading/assessment/shadow");
      $("shadowRows").innerHTML = (shadow.pure_rules || []).slice(0, 15).map(r =>
        `<tr><td>${esc(r.name || r.symbol)}<small>${esc(r.symbol)}</small></td><td>${r.strength_rank}</td>
         <td>${pill(r.verdict)}</td></tr>`).join("");
      if (!(shadow.pure_rules || []).length) empty("shadowRows", 3, "暂无对照");
    } catch (e) { error(e); }
  }

  Object.assign(labels,{REGISTERED:"已登记",RUNNING:"运行中",CONCLUDED:"已结论",DECIDED:"已决定",
    PUBLISHED:"已发布",VETOED:"已否决",ROLLED_BACK:"已回退",PENDING_CONFIRMATION:"待确认",
    APPLIED:"已调账",DISMISSED:"已排除",SHARE_EVENT:"股数事件",CASH_EVENT:"现金事件",
    NO_VALUATION:"无估值",HOLD:"持有",MARKED:"已估值"});

  async function loadLedger() {
    try {
      perfData = await api("/api/trading/performance" + (data.snapshot ? "?as_of=" + encodeURIComponent(data.snapshot.as_of) : ""));
      const p = perfData, a = p.account || {};
      $("perfNotice").textContent = p.status === "READY" ? (p.separation?.note || "") : (p.note || label(p.status));
      totals("perfTotals", [["权益", a.equity_cents == null ? "未估值" : money(a.equity_cents)],
        ["净收益", a.net_return == null ? "--" : (a.net_return * 100).toFixed(2) + "%"],
        ["最大回撤", a.max_drawdown == null ? "--" : (a.max_drawdown * 100).toFixed(2) + "%"],
        ["资金占用", a.capital_usage == null ? "--" : (a.capital_usage * 100).toFixed(1) + "%"],
        ["已实现", money(a.realized_cents)], ["未实现", a.unrealized_cents == null ? "未估值" : money(a.unrealized_cents)]]);
      $("perfPlans").innerHTML = (p.plans || []).map(r =>
        `<tr><td>${esc(r.name || r.symbol)}<small>${esc(r.symbol)}</small></td><td>${r.remaining}</td><td>${r.sellable_today}<small>T+1锁定 ${r.t_plus_1_locked}</small></td>
         <td>${money(r.cost_remaining_cents)}</td><td>${money(r.realized_cents)}</td><td>${r.unrealized_cents == null ? "--" : money(r.unrealized_cents)}</td>
         <td>${pill(r.mark_status)}</td><td>${r.exposure_continues ? "是" : "否"}<small>${esc(r.exposure_note || "")}</small></td></tr>`).join("");
      if (!(p.plans || []).length) empty("perfPlans", 8, "尚无模拟计划");
      $("navRows").innerHTML = (p.nav || []).map(n =>
        `<tr><td>${esc(n.day)}</td><td>${money(n.cash_cents)}</td><td>${n.holdings_cents == null ? "--" : money(n.holdings_cents)}</td>
         <td>${n.equity_cents == null ? "--" : money(n.equity_cents)}</td><td>${n.drawdown == null ? "--" : (n.drawdown * 100).toFixed(2) + "%"}</td><td>${pill(n.status)}</td></tr>`).join("");
      if (!(p.nav || []).length) empty("navRows", 6, "尚无净值记录");
      const e = p.exposure || {}, caps = e.caps || {};
      const rows = [...Object.entries(e.by_symbol || {}).map(([k, v]) => ["个股 " + k, v, caps.symbol_cap_cents]),
                    ...Object.entries(e.by_theme || {}).map(([k, v]) => ["题材 " + k, v, caps.theme_cap_cents])];
      if (a.invested_cents != null) rows.push(["总敞口", a.invested_cents, caps.gross_cap_cents]);
      $("exposureRows").innerHTML = rows.map(([k, v, c]) => `<tr><td>${esc(k)}</td><td>${money(v)}</td><td>${money(c)}</td></tr>`).join("");
      if (!rows.length) empty("exposureRows", 3, "无持仓敞口");
      $("breachNotice").textContent = (e.breaches || []).length ? "超限：" + e.breaches.join("；") : (e.note || "");
      const x = await api("/api/trading/experiments");
      const c = x.cadence || {};
      $("cadenceNotice").textContent = `复盘节奏 ${(c.schedule || []).join(" / ")}；已完成 ${(c.done || []).join("、") || "无"}；下一次 ${c.next || "无"}${(c.overdue || []).length ? "；逾期 " + c.overdue.join("、") : ""}。${c.rule || ""}`;
      $("expRows").innerHTML = (x.experiments || []).map(r =>
        `<tr><td>${esc(r.id)}</td><td>${esc(r.hypothesis)}</td><td>${esc(r.control)}</td><td>${esc(r.metric)}</td>
         <td>${(r.attempts || []).length} / ${r.max_attempts}</td><td>${pill(r.status)}</td><td>${pill(r.publish_check?.status || "NOT_RUN")}</td>
         <td>${esc(label(r.decision?.decision || "--"))}</td>
         <td>${r.status === "CONCLUDED" ? `<button data-expcheck="${esc(r.id)}">发布前检查</button>` : ""}</td></tr>`).join("");
      if (!(x.experiments || []).length) empty("expRows", 9, "尚无登记实验；登记须含假设、对照、样本切分、发现口径指标与尝试次数");
      await loadOps();
    } catch (err) { error(err); }
  }

  async function loadOps() {
    const o = await api("/api/trading/ops");
    totals("opsTotals", [["schema 版本", `${o.schema_version} / ${o.latest_schema}`], ["待迁移", (o.pending_migrations || []).length],
      ["备份数", (o.backups || []).length], ["归档数", (o.archives || []).length]]);
    $("caRows").innerHTML = (o.corporate_actions_pending || []).map(r =>
      `<tr><td>${esc(r.plan_id.slice(0, 12))}</td><td>${esc(r.day)}</td><td>${r.factor}</td><td>${esc(label(r.kind))}<small>${esc(r.basis)}</small></td>
       <td><button data-ca="${esc(r.id)}" data-kind="SHARE_EVENT">股数</button> <button data-ca="${esc(r.id)}" data-kind="CASH_EVENT">现金</button> <button data-ca="${esc(r.id)}" data-kind="NOT_AN_EVENT">排除</button></td></tr>`).join("");
    if (!(o.corporate_actions_pending || []).length) empty("caRows", 5, "没有待确认的公司行动");
    const rows = [...(o.backups || []).map(b => ["备份", b.backup, b.sha256, Object.keys(b.counts || {}).length + " 表", "否"]),
                  ...(o.archives || []).map(a => ["归档", a.path, a.sha256, a.rows + " 行", a.pruned ? "是" : "否"])];
    $("archiveRows").innerHTML = rows.map(r => `<tr>${r.map(v => `<td>${esc(v)}</td>`).join("")}</tr>`).join("");
    if (!rows.length) empty("archiveRows", 5, "尚无备份或归档");
  }

  Object.assign(labels,{IMPLEMENTED:"软件已实现",MISSING_INPUT:"缺输入",NOT_DONE:"历史验证未做",
    PARTIAL:"历史验证部分完成",DONE:"历史验证完成",NOT_STARTED:"前向未开始",RECORDING:"前向记录中",
    STRONG_BUT_NOT_ACTIONABLE:"最强但当前不合适",ACTIONABLE_FOR_REVIEW:"可进入复核",OBSERVE:"观察"});
  const stampText = s => `${label(s.software)} ｜ ${label(s.historical)}${s.historical_note ? "（" + s.historical_note + "）" : ""} ｜ ${label(s.forward)}${s.forward_note ? "（" + s.forward_note + "）" : ""} ｜ 口径：${s.note}`;
  const stampClass = s => s.historical === "DONE" && s.forward === "MATURE" ? "ready" : s.software === "MISSING_INPUT" ? "risk" : "wait";

  const ratioText = r => (!r || r.value == null) ? (r && r.status || "无可比样本") : (r.denominator > 1e6
    ? `${(r.value * 100).toFixed(1)}%（${(r.numerator / 1e8).toFixed(1)} 亿 / ${(r.denominator / 1e8).toFixed(1)} 亿）`
    : `${(r.value * 100).toFixed(1)}%（${r.numerator}/${r.denominator}）`);
  const dist = d => (!d || !d.n) ? "无可比样本" : `n=${d.n} 均值 ${d.mean}% 中位 ${d.median}% 上涨 ${d.up} 最差 ${d.worst}%`;

  const RV = {
    kind: {U: "收盘涨停", Z: "炸板", D: "收盘跌停", DZ: "触跌停未封死"},
    board: {MAIN: "主板", CHINEXT: "创业板", STAR: "科创板"},
    pos: {HIGH: "高位", MID: "中位", LOW: "低位"},
    loss: {yesterday_pool_negative: "昨日涨停股今日收跌", limit_down: "收盘跌停", seal_failed: "炸板", gap_up_reversal: "高开低走", breakdown_20d: "跌破20日低点"},
    structure: {RANGE: "震荡", BULL: "上行", BEAR: "下行", TRANSITION: "过渡"},
    temperature: {NORMAL: "正常", HOT: "过热", COLD: "偏冷", OVERHEATED: "过热"},
    state: {GREEN: "绿灯（常态）", ORANGE: "橙灯（警戒）", RED: "红灯（避险）"},
    basis: {HISTORICAL_PREVIOUS_REVIEW: "取自前一日归档", FROZEN_PREVIOUS_REVIEW: "取自前一日冻结记录", RECONSTRUCTED_FROM_BARS_NOT_FROZEN: "由日线重建（非冻结）"},
    recon: {RECONCILED: "已对齐", RECONCILIATION_PENDING: "有差异，待逐股核对", REFERENCE_UNAVAILABLE: "无参考数据"},
    stage: {REVIEW_READY: "完整", PARTIAL_DATA: "数据部分缺失", PRICE_FINAL_REFERENCE_PENDING: "参考数据待补", DATA_WAIT: "数据等待"},
    section: {READY: "就绪", PARTIAL: "部分", NOT_IMPLEMENTED_P1: "未实施（P1）", NO_MAPPING: "无题材映射", DATA_WAIT: "数据等待"},
    excluded: {ST: "ST", NOT_TRADED_OR_MISSING: "缺当日行情", SUSPENDED: "停牌", INVALID_ROW: "无效行"},
    poolExcluded: {SUSPENDED_OR_MISSING: "停牌或缺数据", CORPORATE_ACTION_PENDING: "公司行动待核", MISSING: "缺数据"},
    bucket: {strong: "强收盘（≥0.7）", mid: "中性", weak: "弱收盘（≤0.3）"},
  };
  const rv = (map, k) => (RV[map] && RV[map][k]) || k;
  const signed = v => v == null ? "--" : (v > 0 ? "+" : "") + Number(v).toFixed(2) + "%";
  const pairs = (obj, map) => Object.entries(obj || {}).map(([k, v]) => `${rv(map, k)} ${v}`).join(" / ") || "无";
  const factText = f => Object.entries(f || {}).map(([k, v]) => k === "height" ? `${v} 板` : k === "return20_pct" ? `20日 ${signed(v)}` : k === "close_vs_pre" ? `今日 ${signed(v)}` : k === "theme" ? `题材 ${v}` : k === "limit_up" ? `涨停 ${v} 只` : k === "limit" ? rv("kind", v) : `${k} ${v}`).join("，");
  const distRow = (name, d) => !d || !d.n ? `<tr><td>${name}</td><td colspan="6" class="empty">无可比样本</td></tr>`
    : `<tr><td>${name}</td><td>${d.n}</td><td>${signed(d.mean)}</td><td>${signed(d.median)}</td><td>${d.up}</td><td>${signed(d.worst)}</td><td>${signed(d.best)}</td></tr>`;

  const sentVal = (name, v) => v == null ? "--" : ["seal_failure_rate", "pool_up_share", "promotion_1_to_2"].includes(name) ? (v * 100).toFixed(1) + "%"
    : ["pool_mean_pct", "open_premium_pct"].includes(name) ? signed(v) : String(v);

  const FP = {status: {DATA_GAP: "缺数据", WATCH: "观察", WAIT_SETUP: "等结构", WAIT_TRIGGER: "等触发", PLAN_READY: "预案齐备", SIGNAL_CONFIRMED: "已触发", REJECT: "否决", EXPIRED: "过期"},
              mode: {STRONG_CONFIRM: "强势确认", TREND_PULLBACK: "趋势回踩", OVERNIGHT: "尾盘隔夜"},
              rmode: {FROZEN_EOD: "前向冻结（收盘后）", HISTORICAL_DEMO: "事后重建（非前向样本）", INTRADAY_UPDATE: "盘中更新", TEMPLATE: "模板"}};
  async function loadBoardRelay() {
    if (!$("relayWatch")) return;
    clearTimeout(relayTimer);
    const id = data.snapshot?.as_of || "";
    $("relayRows").innerHTML = "";
    $("relayStatus").textContent = "";
    try {
      const result = await api("/api/trading/board-relay" + (id ? "?as_of=" + encodeURIComponent(id) : ""));
      if (id !== (data.snapshot?.as_of || "")) return;
      const w = result.watch, a = result.auction, st = result.status;
      if (!w) { $("relayStatus").textContent = st?.status || "暂无接力观察档案"; return; }
      $("relayStatus").textContent = [w.as_of, w.mode === "REPLAY" ? "历史重算，不用于推送" : "前向观察", w.status,
        `涨停母池 ${w.universe ?? "--"} / 冲板失败 ${w.failed_seals ?? "--"}`,
        w.gaps?.length ? "缺 " + w.gaps.join("、") : "", "竞价 " + (a?.status || "暂无归档"),
        a?.alerts?.length ? "弱转强观察：" + a.alerts.join("、") : "",
        st?.status === "FAILED" ? "最后一次生成失败，当前为旧版本" : ""].filter(Boolean).join(" · ");
      const sectors = {FIRST:"1进2观察", RELAY:"2进3观察", HEIGHT:"高位观察"};
      const rows = new Map(w.rows.map(r => [r.symbol, r]));
      const auctions = new Map((a?.rows || []).map(r => [r.symbol, r]));
      const hhmm = s => s == null ? "--" : [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map(n=>String(n).padStart(2,"0")).join(":");
      $("relayRows").innerHTML = Object.entries(w.focus).flatMap(([lane, picks]) => picks.map(s => {
        const r = rows.get(s), x = auctions.get(s);
        if (!r) return "";
        const ladder = Object.entries(r.peer_ladder || {}).map(([h,n]) => `${h}板×${n}`).join("、");
        const seat = r.seat_evidence?.disclosures?.map(d => `拉萨席位 ${d.seat_count}，买额占前五 ${fmt(d.buy_share*100)}%`).join("；") || r.seat_evidence?.status;
        const notice = r.notices.map(n => n.title).join("；");
        return `<tr><td>${esc(sectors[lane])}</td><td><b>${esc(r.name)}</b><small>${esc(s)}</small></td><td>${esc(r.theme || "待核")}<small>${esc(ladder || "无已证同链")}</small></td><td>${fmt(r.price.position60 == null ? null : r.price.position60*100)}%<small>EMA20 ${fmt(r.price.ema20_bias_pct)}%</small></td><td>${hhmm(r.first_seal_s)} / ${hhmm(r.last_seal_s)}</td><td>${esc(r.state)}</td><td>${esc([seat,notice].filter(Boolean).join("；") || "--")}</td><td>${x ? esc(x.state + (x.gap_pct == null ? "" : `｜高开 ${fmt(x.gap_pct)}%｜竞价额 ${fmt(x.amount_share_pct)}%`)) : "--"}</td></tr>`;
      })).join("");
      if (!$("relayRows").innerHTML) empty("relayRows", 8, "没有条件齐备的提名");
      const cst = new Date(Date.now() + 8*3600*1000).toISOString();
      const hm = cst.slice(11,16);
      if (w.mode === "FORWARD" && cst.slice(0,10) === w.target && hm >= "09:15" && hm < "09:30"
          && !["READY","PERMISSION_DENIED","EMPTY_WATCH"].includes(a?.status)) relayTimer = setTimeout(loadBoardRelay, 15000);
    } catch (e) { $("relayStatus").textContent = e.message; }
  }
  async function loadFupan() {
    const ids = ["fupanVerdict", "fupanBanner", "fupanAnalyzer", "fupanAnalyze", "fupanAnalyzeMsg", "fupanTotals", "fupanStocks", "fupanIssues", "fupanAudit", "fupanOpen", "fupanPacket", "fupanPrompt", "fupanJson", "fupanMerge", "fupanMergeMsg"];
    if (ids.some(id => !$(id))) return;
    const clear = () => { $("fupanVerdict").textContent = ""; $("fupanTotals").innerHTML = ""; $("fupanStocks").innerHTML = ""; $("fupanIssues").textContent = ""; $("fupanAudit").textContent = ""; for (const id of ["fupanOpen", "fupanPacket", "fupanPrompt", "fupanJson"]) $(id).removeAttribute("href"); };
    try {
      const day = data.snapshot ? "?as_of=" + encodeURIComponent(data.snapshot.as_of) : "";
      const r = await api("/api/trading/fupan" + day);
      if (r.status !== "READY") { clear(); $("fupanBanner").textContent = r.note || "该日没有双层复盘报告"; return; }
      const m = r.meta, ps = r.packet_stats || {};
      $("fupanVerdict").innerHTML = `<b>${esc(r.summary.decision)}</b>`;
      $("fupanBanner").textContent = `${m.title}（${r.pointer.version}）｜ ${FP.rmode[m.mode] || m.mode} ｜ 目标日 ${m.target_date} ｜ 生成 ${m.rendered_at} ｜ ${m.banner}`
        + (r.pointer.analysis_merged ? " ｜ 已合并 AI 解读" : " ｜ 未合并 AI 解读（结论为本地计算）");
      $("fupanOpen").href = r.links.html; $("fupanPacket").href = r.links.packet; $("fupanPrompt").href = r.links.prompt; $("fupanJson").href = r.links.json;
      totals("fupanTotals", [["接力情绪", (m.banner.match(/接力情绪(..)/) || ["", "--"])[1]], ["必保留股票", m.required_symbols.join("、") || "无"], ["股票卡", String(r.stocks.length)],
        ["缺口/问题", String(r.issues.length)], ["AI 事实包", ps.ai_packet_bytes ? `${(ps.ai_packet_bytes / 1024).toFixed(0)} KB，${ps.stock_cards} 卡，必保留${ps.covered_required ? "已覆盖" : "未覆盖"}` : "--"],
        ["风险", r.summary.risk]]);
      $("fupanStocks").innerHTML = r.stocks.map(s => `<tr><td><b>${esc(s.opportunity_label)}</b></td><td>${esc(s.name)}<small>${esc(s.symbol)}</small>${s.pinned ? ' <span class="pill wait">指定</span>' : ""}</td><td>${esc(s.opportunity_reason)}</td>`
        + `<td><span class="pill ${s.status === "PLAN_READY" ? "ready" : s.status === "DATA_GAP" || s.status === "REJECT" ? "risk" : "wait"}">${esc(FP.status[s.status] || s.status)}</span>${s.role ? "<br><small>" + esc(s.role) + "</small>" : ""}</td>`
        + `<td>${esc(FP.mode[s.mode] || s.mode)}</td><td>${signed(s.metrics.change_pct)}</td><td>${s.metrics.return20_pct == null ? "--" : signed(s.metrics.return20_pct)}</td>`
        + `<td>${s.plan ? esc(`${s.plan.entry_min.toFixed(2)}–${s.plan.entry_max.toFixed(2)} / 失效 ${s.plan.invalidation_price.toFixed(2)}`) : "--"}</td>`
        + `<td>${esc((s.missing || []).join("；") || "无")}</td><td>${esc(s.next_check)}</td></tr>`).join("") || `<tr><td colspan="10" class="empty">无股票卡</td></tr>`;
      $("fupanIssues").textContent = r.issues.length ? "问题与缺口：" + r.issues.map(q => `[${q.severity}] ${q.title}：${q.detail}`).join("　") : "";
      const au = r.audit;
      $("fupanAudit").textContent = `${au.title}｜${au.kind}｜${au.note}（冻结计划 ${au.frozen_plan_available ? "有" : "无"}，实际成交 ${au.actual_fills_available ? "有" : "无"}，样本 ${au.review_sample_count}）`;
      const am = r.analysis_meta || {}, az = r.analyzer || {};
      $("fupanAnalyzer").textContent = am.model ? `AI 解读：${am.provider} / ${am.model}，${am.called_at}${am.cached ? "（缓存）" : ""}，耗时 ${am.latency_s}s，tokens 入 ${am.usage && am.usage.input_tokens != null ? am.usage.input_tokens : "--"} / 出 ${am.usage && am.usage.output_tokens != null ? am.usage.output_tokens : "--"}${(am.problems || []).length ? "；校验提示：" + am.problems.join("；") : ""}`
        : az.status ? `AI 解读：${az.status}${az.detail ? "，" + az.detail : ""}${az.error ? "，" + az.error : ""}` : "AI 解读：尚未请求";
      $("fupanAnalyze").onclick = async () => {
        $("fupanAnalyze").disabled = true; $("fupanAnalyzeMsg").textContent = "已发送给模型，通常 1–3 分钟…";
        try {
          const res = await api("/api/trading/fupan/analyze", {as_of: r.as_of});
          $("fupanAnalyzeMsg").textContent = `${res.status}${res.pointer ? "：已生成 " + res.pointer.version : ""}${res.error ? "：" + res.error : ""}${res.detail ? "：" + res.detail : ""}`;
          loadFupan();
        } catch (e) { $("fupanAnalyzeMsg").textContent = "请求失败：" + e.message; }
        finally { $("fupanAnalyze").disabled = false; }
      };
      $("fupanMerge").onclick = async () => {
        $("fupanMergeMsg").textContent = "合并中…";
        try {
          const text = $("fupanAnalysis").value.trim();
          const analysis = JSON.parse(text);
          const res = await api("/api/trading/fupan/analysis", {as_of: r.as_of, analysis});
          $("fupanMergeMsg").textContent = `已合并为 ${res.pointer.version}`; loadFupan();
        } catch (e) { $("fupanMergeMsg").textContent = "合并失败：" + e.message; }
      };
    } catch (e) { clear(); $("fupanBanner").textContent = "双层复盘读取失败：" + e.message; }
  }

  async function loadReview() {
    const ids = ["reviewTotals", "reviewGaps", "reviewMarket", "reviewBoards", "reviewRecon", "reviewPool", "reviewPoolTable", "reviewSentimentNote", "reviewSentiment", "reviewSeries", "reviewLadderLine", "reviewLadder", "reviewThemeNote", "reviewThemes", "reviewActivityLine", "reviewActivity", "reviewLoss", "reviewBellwethers", "reviewCandidateSource", "reviewCandidates", "reviewEligible", "reviewP1"];
    const clearReview = () => { for (const id of ids) if ($(id)) $(id).textContent = ""; if ($("reviewExport")) $("reviewExport").removeAttribute("href"); };
    if (ids.some(id => !$(id))) {
      $("reviewBanner").textContent = "页面模板已更新但服务仍在用旧版：请重启可视化服务（python mainwave_visual_app.py）后强制刷新（Ctrl+F5）";
      return;
    }
    try {
      const day = data.snapshot ? "?as_of=" + encodeURIComponent(data.snapshot.as_of) : "";
      const r = await api("/api/trading/eod_review" + day);
      if (!["READY", "PARTIAL", "STALE_REVIEW"].includes(r.status)) { clearReview(); $("reviewBanner").textContent = r.note || "该日没有盘后复盘产物"; return; }
      $("reviewBanner").textContent = `${r.note ? r.note + " ｜ " : ""}${r.as_of} 盘后复盘（${r.review_version || ""}）｜ 数据完整度：${rv("stage", r.stage)} ｜ 生成时间 ${r.known_at} ｜ 本页只有事实，解释、假设与次日核对（P1）尚未实施`;
      const c = r.boards.counts;
      totals("reviewTotals", [["可核验股票", `${r.coverage.in_domain}（排除：${pairs(r.coverage.excluded, "excluded")}）`],
        ["收盘涨停 / 炸板 / 收盘跌停", `${c.U} / ${c.Z} / ${c.D}`], ["封板失败率（炸板 ÷ 触及涨停）", ratioText(r.boards.seal_failure_rate)],
        ["涨停股曾开板比例", ratioText(r.boards.open_rate)], ["最高连板", `${r.ladder.max_height} 板`]]);
      $("reviewGaps").textContent = (r.gaps || []).length ? "本页缺口：" + r.gaps.join("；") : "";
      const m = r.market, ctx = m.context || {};
      $("reviewMarket").textContent = `沪深300 今日 ${signed(m.index && m.index.return_pct)}，5 日 ${signed(m.index && m.index.return5_pct)}，20 日 ${signed(m.index && m.index.return20_pct)}。`
        + ` 市场状态：${rv("state", ctx.state)}，结构 ${rv("structure", ctx.structure)}，温度 ${rv("temperature", ctx.temperature)}，趋势类复核 ${ctx.trend_review_allowed ? "允许" : "不允许"}，数据 ${ctx.data_status === "FRESH" ? "新鲜" : ctx.data_status || "--"}。`
        + ` 宽度：上涨 ${m.breadth.up} / 下跌 ${m.breadth.down} / 平 ${m.breadth.flat}（共 ${m.breadth.domain} 只）。`;
      $("reviewBoards").innerHTML = Object.entries(r.boards.by_board || {}).map(([b, v]) => `<tr><td>${rv("board", b)}</td><td>${v.domain ?? 0}</td><td>${v.U ?? 0}</td><td>${v.Z ?? 0}</td><td>${v.D ?? 0}</td><td>${v.DZ ?? 0}</td></tr>`).join("") || `<tr><td colspan="6" class="empty">无</td></tr>`;
      $("reviewRecon").innerHTML = Object.entries(r.boards.reconciliation || {}).map(([k, v]) => v.status === "REFERENCE_UNAVAILABLE" || v.local == null
        ? `<tr><td>${rv("kind", k)}</td><td colspan="6" class="empty">${rv("recon", v.status)}${(v.issues || []).length ? "：" + v.issues.join("；") : ""}</td></tr>`
        : `<tr><td>${rv("kind", k)}</td><td>${v.local}</td><td>${v.reference}</td><td>${v.both}</td><td>${v.local_only_n ?? 0}</td><td>${v.reference_only_n ?? 0}</td><td>${rv("recon", v.status)}</td></tr>`).join("") || `<tr><td colspan="7" class="empty">未请求参考数据</td></tr>`;
      const y = r.yesterday_pool;
      $("reviewPool").textContent = `昨日收盘涨停 ${y.n_pool} 只（母池${rv("basis", y.basis)}），今日可核验 ${y.n_valid} 只${Object.keys(y.excluded || {}).length ? "，未核验：" + pairs(y.excluded, "poolExcluded") : ""}；今日继续封板 ${y.continued_U} 只。日内最高对前收是事后幅度，不是可实现收益。`;
      $("reviewPoolTable").innerHTML = distRow("开盘价对昨收", y.open_vs_pre) + distRow("收盘价对昨收", y.close_vs_pre) + distRow("收盘价对开盘价", y.close_vs_open) + distRow("日内最高对昨收（事后）", y.high_vs_pre);
      const se = r.sentiment || {};
      $("reviewSentimentNote").textContent = `${se.note || ""} ${se.readings || ""}`;
      $("reviewSentiment").innerHTML = Object.entries(se.metrics || {}).map(([name, mt]) => mt.status !== "OK"
        ? `<tr><td>${esc(mt.label)}</td><td>${sentVal(name, mt.today)}</td><td colspan="4" class="empty">无可比样本</td></tr>`
        : `<tr><td>${esc(mt.label)}</td><td><b>${sentVal(name, mt.today)}</b></td><td>${sentVal(name, mt.median)}</td><td>${sentVal(name, mt.min)}–${sentVal(name, mt.max)}</td><td>${(mt.percentile * 100).toFixed(0)}%</td><td><span class="pill ${mt.reading === "热" || mt.reading === "偏热" ? "risk" : mt.reading === "中性" ? "wait" : "ready"}">${mt.reading}</span></td></tr>`).join("") || `<tr><td colspan="6" class="empty">无</td></tr>`;
      $("reviewSeries").innerHTML = (m.series_20d || []).map(s => `<tr><td>${s.day}</td><td>${s.U}</td><td>${s.Z}</td><td>${s.D}</td><td>${s.domain}</td></tr>`).join("") || `<tr><td colspan="5" class="empty">无</td></tr>`;
      const l = r.ladder;
      $("reviewLadderLine").textContent = `今日高度分布：${Object.entries(l.by_height || {}).map(([h, n]) => `${h} 板 ${n} 只`).join("，") || "无涨停"}。最高板：${Object.entries(l.heights || {}).slice(0, 8).map(([s, h]) => `${s}（${h} 板）`).join("、") || "无"}。连板按市场日历连续计数，停牌或缺日打断。`;
      $("reviewLadder").innerHTML = Object.entries(l.promotion_from_previous || {}).map(([h, p]) => `<tr><td>${h} 板</td><td>${p.pool}</td><td>${p.denominator}</td><td>${p.numerator}</td><td>${p.broke_n}</td><td>${ratioText(p)}</td></tr>`).join("") || `<tr><td colspan="6" class="empty">昨日无梯队</td></tr>`;
      const ov = r.themes.overlap_of_limit_ups || {};
      $("reviewThemeNote").textContent = `前 10 行按涨停只数排（大概念占优），标"占比榜"的按涨停占成员比例排（≥3 只涨停且成员 ≤300，只是展示过滤），末尾为登记的紧链题材。涨停股所属题材数：${Object.entries(ov).map(([k, v]) => `${k} 个题材 ${v} 只`).join("，") || "无"}。跨题材重叠使各题材占比不能相加；成员表是当前版本，没有历史时点版本。`;
      $("reviewThemes").innerHTML = (r.themes.top_by_limit_up || []).slice(0, 10).concat((r.themes.top_by_limit_share || []).slice(0, 8).map(t => ({...t, theme: t.theme + "（占比榜）"})), r.themes.chains || []).map(t => `<tr><td>${esc(t.theme)}</td><td>${t.members}</td><td>${t.up}</td><td>${t.limit_up}</td><td>${t.above_ma20}</td><td>${ratioText(t.breadth)}</td><td>${ratioText(t.amount_share)}</td></tr>`).join("") || `<tr><td colspan="7" class="empty">无题材统计</td></tr>`;
      const a = r.activity;
      $("reviewActivityLine").textContent = `成交额前 20 只占全市场 ${ratioText(a.top20_share)}；收盘位置分布：${pairs(a.close_location_buckets, "bucket")}。成交额是交易活跃度，不是净买入；相对换手 = 当日换手 ÷ 前 20 日换手中位数。`;
      $("reviewActivity").innerHTML = (a.top20 || []).map(x => `<tr><td>${esc(x.symbol)}</td><td>${rv("board", x.board)}</td><td>${(x.amount / 1e8).toFixed(1)}</td><td>${signed(x.pct_chg)}</td><td>${x.close_location ?? "--"}</td><td>${x.relative_turnover ?? "--"}</td><td>${x.position20 ?? "--"}</td><td>${x.limit ? rv("kind", x.limit) : ""}</td></tr>`).join("");
      $("reviewLoss").innerHTML = Object.entries(r.loss_effects.events || {}).map(([k, v]) => `<tr><td>${rv("loss", k)}</td><td>${v.count}</td><td>${pairs(v.by_board, "board")}</td><td>${pairs(v.by_position, "pos")}</td><td>${(v.symbols || []).slice(0, 6).join("、")}</td></tr>`).join("");
      const li = r.lists;
      $("reviewBellwethers").innerHTML = (li.bellwethers || []).map(p => `<tr><td>${esc(p.symbol)} ${esc(p.name || "")}</td><td>${esc(p.role_label)}</td><td>${esc(p.question)}</td><td>${esc(factText(p.facts))}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">无提名</td></tr>`;
      $("reviewCandidateSource").textContent = li.source || "";
      $("reviewCandidates").innerHTML = (li.continuing_candidates || []).map(c => `<tr><td>${esc(c.name || "")}<small>${esc(c.symbol)}</small></td><td>${esc(c.theme || "")}</td><td>${esc(c.route_ids || "")}</td><td>${esc(label(c.state || ""))}</td><td>${esc((c.blockers || []).map(b => label(b)).join("；"))}</td><td>${esc(c.verdict_reading || "")}</td><td>${c.strength_rank ?? "--"}</td><td>${c.participation_rank ?? "--"}</td><td>${esc(c.continuity_event || "")}</td></tr>`).join("") || `<tr><td colspan="9" class="empty">无候选</td></tr>`;
      $("reviewEligible").textContent = li.review_note + ((li.review_eligible || []).length ? "：" + li.review_eligible.map(c => `${c.symbol} 参与#${c.participation_rank}`).join("、") : "");
      const sec = r.sections || {};
      $("reviewP1").textContent = `未实施（P1）：结论、昨日核对、明日计划本页不生成。各节完成度——市场 ${rv("section", sec.market)}，涨跌停 ${rv("section", sec.boards)}，梯队 ${rv("section", sec.ladder)}，题材 ${rv("section", sec.themes)}，活跃 ${rv("section", sec.activity)}，亏钱 ${rv("section", sec.loss_effects)}，清单 ${rv("section", sec.lists)}。`;
      $("reviewExport").href = "/api/trading/eod_review/report.md" + day;
    } catch (e) { clearReview(); $("reviewBanner").textContent = "复盘读取失败：" + e.message; error(e); }
  }

  async function loadBrief() {
    try {
      const b = await api("/api/trading/brief" + (data.snapshot ? "?as_of=" + encodeURIComponent(data.snapshot.as_of) : ""));
      if (b.status === "NO_SNAPSHOT") { $("briefBanner").textContent = "尚无收盘快照"; return; }
      $("briefBanner").textContent = b.banner;
      totals("briefTotals", [["筛选能力已证明", b.screening_ability_proven ? "是" : "否"], ["原因", b.proof_reason],
        ["缺输入", (b.missing_inputs || []).length ? b.missing_inputs.join("、") : "无"], ["交付日", b.as_of]]);
      const L = b.status_legend || {};
      $("briefLegend").textContent = "三枚印章各自独立：" + Object.entries(L).map(([k, m]) => `${k}=${Object.values(m).join("/")}`).join("；");
      const S = b.sections;
      for (const [id, key] of [["stampTop", "top"], ["stampDiag", "diagnosis"], ["stampChange", "cross_day"], ["stampOutcome", "outcomes"], ["stampBooks", "books"]]) {
        $(id).innerHTML = `<span class="pill ${stampClass(S[key].stamp)}">${esc(stampText(S[key].stamp))}</span>`;
      }
      const rows = [...S.top.strength.map(r => ({...r, _list: "强势"})), ...S.top.participation.filter(p => !S.top.strength.some(s => s.symbol === p.symbol)).map(r => ({...r, _list: "参与"}))];
      $("briefTop").innerHTML = rows.map(r =>
        `<tr><td>${r.strength_rank ?? "--"} / ${r.participation_rank ?? "--"}<small>${r._list}榜</small></td><td><b>${esc(r.name)}</b><small>${esc(r.symbol)}</small></td>
         <td>${esc(r.theme)}<small>${esc(r.route_ids || "无路径")}</small></td><td>${state(r.state || "DATA_WAIT")}<small>${esc((r.blockers || []).join("；") || "无阻断")}</small></td>
         <td>${esc(label(r.verdict))}<small>${esc(r.verdict_reading || "")}</small></td>
         <td>${r.rank_change == null ? "首次" : (r.rank_change > 0 ? "+" : "") + r.rank_change}<small>${esc(r.change_reading || "")}</small></td>
         <td>${(r.diagnosis || []).map(c => `${esc(c.route)}:${esc(label(c.verdict))}${c.stopped_at ? "@" + esc(c.stopped_at) : ""}`).join("<br>") || "无声明路径"}</td></tr>`).join("");
      if (!rows.length) empty("briefTop", 7, "无榜单");
      const diag = Object.entries(S.diagnosis.per_stock || {}).flatMap(([sym, cards]) => cards.length ? cards.map(c =>
        `<tr><td>${esc(sym)}</td><td>${esc(c.route)}</td><td>${pill(c.verdict)}</td><td>${c.stopped_at ? esc(c.stopped_at) + "：" + esc(label(c.stopped_status)) : "--"}</td><td>${esc(c.next_condition || "--")}</td></tr>`)
        : [`<tr><td>${esc(sym)}</td><td>无声明路径</td><td>按结构观察</td><td>--</td><td>--</td></tr>`]);
      $("briefDiag").innerHTML = diag.join(""); if (!diag.length) empty("briefDiag", 5, "无诊断");
      const x = S.cross_day;
      $("briefChangeLine").textContent = `对照日 ${x.previous_day || "无"}（${x.comparison_basis}）；进入 Top：${(x.entered_top || []).join("、") || "无"}；离开：${(x.left_top || []).join("、") || "无"}；留任 ${(x.stayed || []).length}；连续性事件 ${JSON.stringify(x.continuity?.by_event || {})}`;
      $("briefChange").innerHTML = (x.rows || []).map(r =>
        `<tr><td>${esc(r.name)}<small>${esc(r.symbol)}</small></td><td>${r.rank_change == null ? "首次" : (r.rank_change > 0 ? "+" : "") + r.rank_change}</td><td>${esc(r.attribution || "--")}</td><td>${esc(r.reading || "--")}</td><td>${r.event ? esc(r.event) + "：" + esc(r.reason || "") : "--"}</td></tr>`).join("");
      if (!(x.rows || []).length) empty("briefChange", 5, "无跨日记录");
      const o = S.outcomes;
      $("briefCohorts").innerHTML = (o.cohorts || []).map(c =>
        `<tr><td>${esc(c.signal_day || "--")}</td><td>${c.basis === "FORWARD_RECORDED_AFTER_CLOSE" ? "前向" : "回建"}</td><td>${esc(label(c.group))}${c.alpha_scope === "R3_RESEARCH" ? "（仅研究）" : ""}</td><td>${c.selected}</td><td>${c.mature}</td><td>${c.independent_opportunities ?? "--"}</td><td>${c.theme_day_clusters ?? "--"}</td><td>${fmt(c.mean)}</td><td>${c.positive_fraction == null ? "--" : (c.positive_fraction * 100).toFixed(0) + "%"}</td><td>${fmt(c.worst)}</td><td>${esc(c.reading)}</td></tr>`).join("");
      if (!(o.cohorts || []).length) empty("briefCohorts", 11, "无对照组");
      $("briefMissed").textContent = `漏选审计：${label(o.missed?.status)}；定义：${o.missed?.rule || "--"}；${o.missed?.notice || ""}${o.missed?.summary ? " 摘要：" + JSON.stringify(o.missed.summary) : ""}`;
      $("briefProcess").innerHTML = (o.process || []).map(p =>
        `<tr><td>${esc(p.route)} ${esc(p.label)}</td><td>${p.total}</td><td>${esc(Object.entries(p.verdicts || {}).map(([k, v]) => label(k) + " " + v).join("　"))}</td><td>${esc(Object.entries(p.top_stops || {}).map(([k, v]) => k + " " + v).join("；") || "--")}</td></tr>`).join("");
      if (!(o.process || []).length) empty("briefProcess", 4, "无过程统计");
      $("briefProcessNote").textContent = o.process_note || "";
      const bk = S.books || {}, sig = bk.signal || {}, au = bk.auto || {}, ma = bk.manual || {};
      const chosen = (sig.trade_set || []).map(r => `${r.symbol}#${r.participation_rank}`).join("、") || "空集";
      $("briefBooks").textContent = `${bk.separation || ""} 信号账交易集：${chosen}（可复核 ${sig.eligible ?? 0}，R3-only 排除 ${sig.excluded_r3_research ?? 0}）。` +
        ` 自动账：${au.live ? "当日实跑" : "非当日不下单"}，计划 ${au.plans ?? 0}，持仓 ${au.open_positions ?? 0}，成交 ${au.fills ?? 0}，对账 ${au.reconciliation || "--"}，本次 ${(au.placed || []).map(p => p.symbol + ":" + p.status).join("、") || "无下单"}。` +
        ` 人工账：${ma.status || "--"}，绩效 ${ma.performance || "--"}。`;
    } catch (e) { error(e); }
  }
  async function detail(id) {
    try {
      const d=await api(`/api/trading/opportunities/${encodeURIComponent(id)}?snapshot_id=${encodeURIComponent(data.snapshot.snapshot_id)}`);currentDetail=d;
      const r=d.record; $("detailTitle").textContent=`${r.name} ${r.symbol} · ${d.as_of}`;
      $("detailContent").innerHTML=`<dl><dt>路径 / 模式</dt><dd>${esc(r.route_ids||"未归属")} / 趋势回踩</dd><dt>结构场景</dt><dd>${esc(r.setup_kind)}</dd><dt>当前判断</dt><dd>${esc(label(r.state))}</dd><dt>待满足条件</dt><dd>${esc(r.next_condition)}</dd><dt>阻断原因</dt><dd>${esc(r.blockers.join(" · ")||"无；仍需账户预算复核")}</dd><dt>冻结价格范围</dt><dd>${fmt(r.window.lower)} 至 ${fmt(r.window.upper)}；失效 ${fmt(r.window.invalid)}</dd><dt>排名版本</dt><dd>${esc(r.ranking_version)}</dd></dl><button id="approvePlan" ${!r.paper_plan_eligible||!data.paper.configured?"disabled":""}>批准50%预算模拟首仓计划</button><h3>分项依据与角色证据</h3><pre>${esc(JSON.stringify({features:r.features,role_evidence:r.role_evidence,rank_reason:r.rank_reason},null,2))}</pre><h3>跨日记录</h3><pre>${esc(JSON.stringify(d.events,null,2))}</pre><h3>历史复盘</h3><pre>${esc(JSON.stringify(d.assessments,null,2))}</pre>`;
      $("detailContent").insertAdjacentHTML("afterbegin", `<h3>条件复核</h3><div class="table-wrap"><table><thead><tr><th>检查项</th><th>结果</th></tr></thead><tbody>${(r.evidence.checks||[]).map(c=>`<tr><td>${esc(c.label)}</td><td>${esc(label(c.status))}</td></tr>`).join("")}</tbody></table></div><p class="warning">${esc(r.blockers.map(label).join("；")||"结构条件已满足，仍需账户与可成交性复核")}</p>`);
      const phase=r.evidence.state_vector||{};
      $("detailContent").insertAdjacentHTML("afterbegin",`<dl><dt>生命周期</dt><dd>${esc(phase.phase_label||"待核")} / ${esc(phase.stage_label||"待核")}</dd><dt>三模式状态</dt><dd>趋势回踩：${esc(label(r.modes.TREND_PULLBACK))}<br>强势确认：${esc(label(r.modes.STRENGTH_CONFIRM))}<br>尾盘隔夜：${esc(label(r.modes.OVERNIGHT))}</dd><dt>知识版本</dt><dd>${esc(data.snapshot.knowledge_registry_id.slice(0,16))}</dd></dl>`);
      $("exportOpportunity").href=`/api/trading/opportunities/${encodeURIComponent(id)}?snapshot_id=${encodeURIComponent(d.snapshot_id)}&export=1`;
      $("reviewForm").reset();$("reviewSaved").textContent="";
      if(!$("detail").open) $("detail").showModal();
      $("approvePlan").onclick=async()=>{if(!window.confirm("批准独立模拟账户的首仓计划？这不会生成真实订单。"))return;try{const out=await api("/api/trading/paper/plans",{snapshot_id:d.snapshot_id,symbol:r.symbol});$("reviewSaved").textContent=out.reason||"模拟委托已登记，等待后续交易日撮合";await load(d.snapshot_id);}catch(e){error(e);$("reviewSaved").textContent=e.message;}};
    } catch(e){error(e);}
  }
  document.querySelectorAll("[data-tab]").forEach(b=>b.onclick=()=>{document.querySelectorAll("[data-tab]").forEach(x=>{x.setAttribute("aria-selected",String(x===b));$(x.dataset.tab).hidden=x!==b;});if(b.dataset.tab==="opportunities")Plotly.Plots.resize($("strengthChart"));});
  for(const id of ["search","theme","route","state","order"])$(id).addEventListener("input",()=>{page=0;renderCandidates();});
  $("knowledgeSearch").oninput=renderKnowledge;
  $("diagRoute").onchange=loadDiagnostics; $("diagVerdict").onchange=loadDiagnostics;
  $("stratum").onchange=renderStrata;
  $("planPicker").onchange=loadTimeline;
  $("modePicker").onchange=renderModes;
  $("runBackup").onclick=async()=>{$("opsStatus").textContent="备份中";try{const r=await api("/api/trading/ops/backup",{});$("opsStatus").textContent=`备份完成，校验：${label(r.verification?.status)}`;await loadOps();}catch(e){$("opsStatus").textContent=e.message;}};
  $("caRows").onclick=async e=>{const b=e.target.closest("[data-ca]");if(!b)return;if(!window.confirm("按所选类型调整模拟账本？"))return;try{await api(`/api/trading/corporate_actions/${encodeURIComponent(b.dataset.ca)}/confirm`,{kind:b.dataset.kind});await loadLedger();}catch(err){error(err);}};
  $("expRows").onclick=async e=>{const b=e.target.closest("[data-expcheck]");if(!b)return;try{const r=await api(`/api/trading/experiments/${encodeURIComponent(b.dataset.expcheck)}/check`,{});window.alert(`发布前检查：${label(r.check?.status)}`);await loadLedger();}catch(err){error(err);}};
  $("runAssessment").onclick=async()=>{const id=$("modePicker").value;if(!id)return;$("aiStatus").textContent="评估中";try{const r=await api(`/api/trading/assessment/${encodeURIComponent(id)}`,{});$("aiStatus").textContent=`结论：${label(r.assessment?.verdict)}（${label(r.validation?.status)}）`;await loadAssessment();}catch(e){$("aiStatus").textContent=e.message;}};
  $("previous").onclick=()=>{page--;renderCandidates();};$("next").onclick=()=>{page++;renderCandidates();};
  $("refresh").onclick=()=>load();$("snapshots").onchange=()=>load($("snapshots").value);
  $("candidates").onclick=e=>{const b=e.target.closest("[data-detail]");if(b)detail(b.dataset.detail);};
  $("orders").onclick=async e=>{const b=e.target.closest("[data-cancel]");if(!b)return;try{await api(`/api/trading/paper/orders/${encodeURIComponent(b.dataset.cancel)}/cancel`,{});await load();}catch(err){error(err);}};
  $("accountForm").onsubmit=async e=>{e.preventDefault();const body=Object.fromEntries([...new FormData(e.target)].map(([k,v])=>[k,Number(v)]));if(!window.confirm("创建独立模拟账户？本金和配置将冻结，旧观察持仓不会迁入。"))return;try{await api("/api/trading/paper/account",body);await load();}catch(err){error(err);}};
  $("closeDetail").onclick=()=>$("detail").close();
  $("reviewForm").onsubmit=async e=>{e.preventDefault();const body=Object.fromEntries(new FormData(e.target));body.snapshot_id=currentDetail.snapshot_id;try{await api(`/api/trading/opportunities/${currentDetail.record.opportunity_id}/review`,body);await detail(currentDetail.record.opportunity_id);$("reviewSaved").textContent="已保存；旧复盘保留";}catch(err){$("reviewSaved").textContent=err.message;}};
  load();
})();

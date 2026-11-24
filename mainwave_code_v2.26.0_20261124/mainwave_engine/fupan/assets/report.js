'use strict';
(() => {
const $ = s => document.querySelector(s);
const esc = x => String(x == null ? '' : x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (v,d=2) => typeof v==='number' && Number.isFinite(v) ? new Intl.NumberFormat('zh-CN',{maximumFractionDigits:d,minimumFractionDigits:d}).format(v) : '—';
const pct = (v,d=2) => typeof v==='number' && Number.isFinite(v) ? `${v>0?'+':''}${number(v,d)}%` : '—';
const statusMap={DATA_GAP:['数据待补','warn'],WATCH:['仅观察','info'],WAIT_SETUP:['等结构','info'],WAIT_TRIGGER:['等触发','info'],PLAN_READY:['预案齐备','good'],SIGNAL_CONFIRMED:['快照时已触发','good'],REJECT:['本模式不参与','block'],EXPIRED:['已过期','dark']};
const modeMap={STRONG_CONFIRM:'强势确认',TREND_PULLBACK:'趋势回踩',OVERNIGHT:'尾盘隔夜'};
const originMap={USER_DATA:'用户原始数据',LOCAL_CALCULATION:'本地计算',USER_MATERIAL:'用户原材料',DESIGN:'新增设计',EXTERNAL_DOCUMENT:'外部技术文档'};
let data, activeView='brief', currentQuery='',currentMode='',onlyPinned=false,hotspotQuery='',onlyHotspotFocus=true;
const state={lastFocus:null,printDetails:null};

function schemaCheck(v,s,path='$'){
 let out=[];if(s.anyOf){if(s.anyOf.some(a=>schemaCheck(v,a,path).length===0))return [];return [path+': 不符合可选结构'];}
 if('const' in s&&v!==s.const)out.push(path+': 常量不符');
 if(s.enum&&!s.enum.includes(v))out.push(path+': 枚举不符');
 const types=Array.isArray(s.type)?s.type:s.type?[s.type]:[];
 const match=t=>t==='null'?v===null:t==='array'?Array.isArray(v):t==='object'?v!==null&&typeof v==='object'&&!Array.isArray(v):t==='integer'?typeof v==='number'&&Number.isInteger(v):t==='number'?typeof v==='number'&&Number.isFinite(v):typeof v===t;
 if(types.length&&!types.some(match))return [path+': 类型错误'];
 if(typeof v==='string'){if(s.minLength!==undefined&&v.length<s.minLength)out.push(path+': 太短');if(s.pattern&&!new RegExp(s.pattern).test(v))out.push(path+': 格式错误');}
 if(typeof v==='number'){if(!Number.isFinite(v))out.push(path+': 非有限数');if(s.minimum!==undefined&&v<s.minimum)out.push(path+': 低于下限');if(s.maximum!==undefined&&v>s.maximum)out.push(path+': 超过上限');}
 if(Array.isArray(v)){if(s.minItems!==undefined&&v.length<s.minItems)out.push(path+': 数量不足');if(s.maxItems!==undefined&&v.length>s.maxItems)out.push(path+': 数量过多');if(s.uniqueItems&&new Set(v.map(x=>JSON.stringify(x))).size!==v.length)out.push(path+': 重复项');if(s.items)v.forEach((x,i)=>out.push(...schemaCheck(x,s.items,path+'['+i+']')));}
 if(v!==null&&typeof v==='object'&&!Array.isArray(v)){for(const k of s.required||[])if(!(k in v))out.push(path+': 缺少'+k);for(const[k,x]of Object.entries(v)){if(s.properties&&k in s.properties)out.push(...schemaCheck(x,s.properties[k],path+'.'+k));else if(s.additionalProperties===false)out.push(path+': 未定义字段'+k);}}
 return out;
}
function validate(d){
 const errors=schemaCheck(d,JSON.parse($('#report-schema').textContent));
 if(errors.length)return errors;
 const req=['meta','summary','metrics','history','claims','sections','stocks','modes','issues','sources','spec','comparisons','scenarios','audit','requests'];
 if(!d||typeof d!=='object'||Array.isArray(d))return ['根对象必须为JSON对象'];
 if(d.schema_version!=='review.report.v2')errors.push('schema_version必须为review.report.v2');
 req.forEach(k=>{if(!(k in d))errors.push(`缺少${k}`)});
 if(errors.length)return errors;
 ['metrics','history','claims','sections','stocks','modes','issues','sources','comparisons','scenarios','requests'].forEach(k=>{if(!Array.isArray(d[k]))errors.push(`${k}必须为数组`)});
 if(errors.length)return errors;
 if(!d.meta||d.meta.order_enabled!==false)errors.push('报告必须保持order_enabled=false');
 if(!['TEMPLATE','HISTORICAL_DEMO','FROZEN_EOD','INTRADAY_UPDATE'].includes(d.meta.mode))errors.push('未知报告模式');
 if(!d.summary||!Array.isArray(d.summary.focus_ids)||!Array.isArray(d.summary.claim_ids))errors.push('summary缺少focus_ids/claim_ids数组');
 if(errors.length)return errors;
 const sets={};for(const key of ['sources','claims','stocks']){sets[key]=new Set();for(const x of d[key]){if(typeof x.id!=='string'||!x.id)errors.push(key+'的id无效');if(sets[key].has(x.id))errors.push('重复ID: '+x.id);sets[key].add(x.id);}}
 d.summary.claim_ids.forEach(id=>{if(!sets.claims.has(id))errors.push('摘要引用不存在的结论'+id)});
 d.summary.focus_ids.forEach(id=>{if(!sets.stocks.has(id))errors.push('精华卡不存在: '+id)});
 if(d.summary.focus_ids.length>2)errors.push('精华任务最多2项；其余指定股仍必须在持续池');
 if(d.sections.length!==7||d.sections.map(x=>x.id).join(',')!=='01,02,03,04,05,06,07')errors.push('必须保留01–07七步顺序');
 d.sections.forEach(s=>{if(!sets.claims.has(s.claim_id))errors.push('段落引用不存在结论'+s.claim_id);if(!s.table||!Array.isArray(s.table.rows)||!Array.isArray(s.table.columns))errors.push('段落表结构无效');});
 const walk=x=>{if(!x||typeof x!=='object')return;Object.entries(x).forEach(([k,v])=>{if(['evidence_ids','support_ids','source_ids'].includes(k)&&Array.isArray(v))v.forEach(id=>{if(!sets.sources.has(id))errors.push('证据不存在: '+id)});else if(k==='evidence_id'&&typeof v==='string'&&!sets.sources.has(v))errors.push('证据不存在: '+v);else walk(v);})};walk(d);
 for(const sym of (d.meta.required_symbols||[])){if(!d.stocks.some(s=>s.symbol===sym))errors.push('漏掉必保留股票: '+sym);}
 const ktime=d.meta.knowledge_as_of?Date.parse(d.meta.knowledge_as_of):NaN;
 for(const s of d.stocks){
  if(!(s.status in statusMap))errors.push('未知股票状态: '+s.status);
  if(!(s.mode in modeMap))errors.push('未知模式: '+s.mode);
  if(!(d.meta.allowed_modes||[]).includes(s.mode))errors.push('模式超出允许范围: '+s.mode);
  if(s.plan && s.plan.source_ids)for(const id of s.plan.source_ids){const src=d.sources.find(e=>e.id===id);if(!src||!src.available_at||!Number.isFinite(Date.parse(src.available_at))||Date.parse(src.available_at)>ktime)errors.push('计划证据时点无效: '+id)}
  if(['PLAN_READY','SIGNAL_CONFIRMED'].includes(s.status)){
   const p=s.plan;
   if(d.meta.mode==='HISTORICAL_DEMO'||d.meta.mode==='TEMPLATE')errors.push('演示/空白页不能声明批准预案或已触发');
   if(!p){errors.push('预案齐备却没有plan: '+s.id);continue;}
   for(const k of ['id','strategy_version','signal_condition','exit_rule','valid_from','expires_at'])if(typeof p[k]!=='string'||!p[k].trim())errors.push(s.id+'计划缺少'+k);
   for(const k of ['entry_min','entry_max','invalidation_price','max_loss_cash'])if(typeof p[k]!=='number'||!Number.isFinite(p[k])||p[k]<=0)errors.push(s.id+'价格/风险必须是正数: '+k);
   if(p.entry_min>p.entry_max||p.invalidation_price>=p.entry_min)errors.push(s.id+'做多价格关系不成立');
   if(p.approval_status!=='APPROVED'||p.budget_status!=='PASS')errors.push(s.id+'审批或预算未通过');
   if(!s.model_version||s.missing.length||p.hard_blocks?.length)errors.push(s.id+'仍有缺失、阻断或模式版本');
   if(!Array.isArray(p.source_ids)||!p.source_ids.length)errors.push(s.id+'缺计划证据');
   if(!Number.isFinite(ktime)||!Number.isFinite(Date.parse(p.expires_at))||Date.parse(p.expires_at)<=ktime||!Number.isFinite(Date.parse(p.valid_from))||Date.parse(p.valid_from)>=Date.parse(p.expires_at))errors.push(s.id+'计划时效无效');
   if(s.status==='SIGNAL_CONFIRMED'&&(!p.quote_as_of||!Number.isFinite(Date.parse(p.quote_as_of))||Date.parse(p.quote_as_of)>ktime||Date.parse(p.quote_as_of)>Date.parse(d.meta.market_as_of)))errors.push(s.id+'报价时点无效');
   if(s.status==='SIGNAL_CONFIRMED'&&(d.meta.mode!=='INTRADAY_UPDATE'||p.trigger_state!=='CONFIRMED'||p.quote_fresh!==true||d.meta.quote_status!=='LIVE_SNAPSHOT'||Date.parse(p.valid_from)>ktime))errors.push(s.id+'已触发状态缺实时确认');
  }
 }
 if(d.meta.forward_frozen && d.meta.mode==='HISTORICAL_DEMO')errors.push('历史演示不能标记前向冻结');
 if(d.meta.forward_frozen){
  if(!d.meta.frozen_at||!Number.isFinite(Date.parse(d.meta.frozen_at)))errors.push('前向报告必须有frozen_at');
  if(Date.parse(d.meta.frozen_at)<ktime)errors.push('冻结时刻早于知识截止');
  for(const src of d.sources.filter(s=>s.origin!=='USER_MATERIAL'&&s.origin!=='DESIGN')) if(!src.available_at||Date.parse(src.available_at)>ktime)errors.push('前向报告证据可知时刻未知/越界: '+src.id);
 }
 for(const pair of d.comparisons){if(pair.matched&&pair.missing.length)errors.push('比较齐备却有缺失: '+pair.id);}
 for(const r of d.history)if(d.meta.trade_date&&r.date>d.meta.trade_date)errors.push('图表包含复盘日之后的数据');
 return [...new Set(errors)];
}
window.ReviewValidate=validate;
function badge(text,kind='info'){return `<span class="badge ${esc(kind)}">${esc(text)}</span>`}
function status(s){const m=statusMap[s.status]||['未知','warn'];return badge(m[0],m[1])}
function ref(id){return `<button class="ref" data-source="${esc(id)}" aria-label="查看证据 ${esc(id)}">证据 · ${esc(id)}</button>`}
function refs(ids=[]){return `<div class="refs">${ids.map(ref).join('')}</div>`}
function pList(arr){return arr.length?arr.map(x=>`<p>${esc(x)}</p>`).join(''):'<p class="muted">待提供。</p>'}
function claim(id){return data.claims.find(c=>c.id===id)}
function dateDisplay(x){return x?String(x).replace('T',' ').replace('+08:00',' CST'):'待提供'}
function heading(title,subtitle='',right=''){return `<div class="section-head"><div><h2>${esc(title)}</h2>${subtitle?`<p>${esc(subtitle)}</p>`:''}</div>${right}</div>`}
function table(t){const hotspot=t.columns[0]==='观察层';return `<div class="table-title">${esc(t.title)}</div>${hotspot?`<div class="filterbar"><input type="search" id="hotspotSearch" aria-label="搜索热点全池" placeholder="股票 / 题材 / 场景" value="${esc(hotspotQuery)}"><label><input type="checkbox" id="hotspotFocus" ${onlyHotspotFocus?'checked':''}>只看角色焦点与早期结构</label><span class="small muted" id="hotspotCount"></span></div>`:''}<div class="table-wrap"><table><thead><tr>${t.columns.map(c=>`<th scope="col">${esc(c)}</th>`).join('')}</tr></thead><tbody>${t.rows.length?t.rows.map((r,i)=>`<tr${hotspot?` data-hotspot-row="${i}" data-hotspot-focus="${r[0]==='全池保留'?'0':'1'}"`:''}>${r.map(c=>`<td>${esc(c)}</td>`).join('')}</tr>`).join(''):`<tr><td colspan="${t.columns.length}" class="muted">未加载数据。</td></tr>`}</tbody></table></div>${t.note?`<p class="table-note">${esc(t.note)}</p>`:''}${refs(t.evidence_ids)}`}
function miniMetric(label,value,cl=''){return `<div><small>${esc(label)}</small><b class="mono ${cl}">${esc(value)}</b></div>`}
function planCount(){return data.stocks.filter(s=>['PLAN_READY','SIGNAL_CONFIRMED'].includes(s.status)).length}
function triggerCount(){return data.stocks.filter(s=>s.status==='SIGNAL_CONFIRMED').length}
function focusCard(s){
 const mode=modeMap[s.mode];return `<article class="panel stock-card"><div class="stock-head"><div><h3 class="stock-name">${esc(s.name)}</h3><div class="stock-code mono">${esc(s.symbol)} · ${esc(s.role)}</div></div>${status(s)}</div><div class="badges mode">${badge(s.opportunity_label,'good')}${badge(mode,'info')}${s.pinned?badge('指定跟踪','dark'):''}</div><p class="reason">${esc(s.opportunity_reason)}</p><div class="stock-mini">${miniMetric('当日涨幅',pct(s.metrics.change_pct),s.metrics.change_pct>0?'up':s.metrics.change_pct<0?'down':'')}${miniMetric('成交额 / 亿',number(s.metrics.amount_yi))}${miniMetric('收盘位置',number(s.metrics.close_location,3))}</div><dl class="stock-row"><dt>下一步</dt><dd>${esc(s.next_check)}</dd></dl><dl class="stock-row"><dt>入场条件</dt><dd>${esc(s.plan?s.plan.signal_condition:'尚未形成完整合约；不把观察结论写成买入指令。')}</dd></dl><dl class="stock-row"><dt>主要缺口</dt><dd>${esc(s.missing.join('；')||'按已批准计划复核。')}</dd></dl><div class="card-bottom"><span>机会判断 ≠ 交易许可</span><button class="btn text" data-jump="stock-${esc(s.id)}">展开证据与条件 →</button></div></article>`;
}
function stockTable(){
 return `<div class="panel table-panel"><div class="filterbar"><input type="search" id="stockSearch" placeholder="搜索股票名称 / 代码" value="${esc(currentQuery)}" aria-label="搜索股票"><select id="modeFilter" aria-label="按交易模式筛选"><option value="">全部模式</option>${Object.entries(modeMap).map(([k,v])=>`<option value="${k}" ${currentMode===k?'selected':''}>${v}</option>`).join('')}</select><label><input type="checkbox" id="pinnedFilter" ${onlyPinned?'checked':''}>只看指定跟踪</label><span class="small muted" id="filterCount"></span></div><div class="table-wrap"><table><thead><tr><th>标的 / 角色</th><th>机会判断</th><th>对应模式</th><th>计划状态</th><th>下一次核验</th></tr></thead><tbody>${data.stocks.length?data.stocks.map(s=>`<tr data-stock-row="${esc(s.id)}"><td><button class="name-link" data-jump="stock-${esc(s.id)}">${esc(s.name)}</button><div class="small muted">${esc(s.role)}${s.pinned?' · 指定':''}</div></td><td>${esc(s.opportunity_label)}</td><td>${esc(modeMap[s.mode])}</td><td>${status(s)}</td><td>${esc(s.next_check)}</td></tr>`).join(''):'<tr><td colspan="5">尚未加载股票；持仓、计划与指定跟踪股不得被展示限额剔除。</td></tr>'}</tbody></table></div><p class="table-note" style="padding:0 18px 13px">此表不使用“容量大即优先”的总排名。排序顺序为展示顺序，不是预计收益高低。</p></div>`;
}
function scenarios(){return `<div class="scenario-grid">${data.scenarios.map((s,i)=>`<article class="panel scenario"><div class="number">SCENARIO 0${i+1}</div><h3>${esc(s.name)}</h3><p>${esc(s.evidence)}</p><p class="action">${esc(s.action)}</p><p class="veto">约束：${esc(s.veto)}</p></article>`).join('')}</div>`}
function risks(max=3){const ii=data.issues.slice(0,max);return ii.length?`<div class="panel risk-list">${ii.map(i=>`<div class="risk-item">${badge(i.severity==='BLOCK'?'阻断':'待核',i.severity==='BLOCK'?'block':'warn')}<div><strong>${esc(i.title)}</strong><p>${esc(i.detail)}</p><p>${esc(i.impact)}</p></div></div>`).join('')}</div>`:'<div class="empty">未加载数据质量记录。空列表不等于不存在风险。</div>'}
function brief(){
 const count=data.meta.mode==='TEMPLATE'?'—':planCount(),confirmed=data.meta.mode==='TEMPLATE'?'—':triggerCount();
 return `<section role="tabpanel" id="brief" aria-labelledby="tab-brief"><div class="print-heading">第一部分 · 精华版</div><div class="panel hero"><div><div class="eyebrow">THE TAKEAWAY / 今日判断</div><h1>${esc(data.summary.claim_ids.length?claim(data.summary.claim_ids[0]).title:'先给结论，再展开证据。')}</h1><p>${esc(data.summary.claim_ids.length?data.summary.claim_ids.map(id=>claim(id).conclusion).join(' '):'导入经本地校验的日报JSON，生成精华版与细化版。两层共用同一组结论、事实与计划状态。')}</p><p class="hero-objective">目标与期限：${esc(data.meta.horizon)}</p><div class="refs">${data.summary.claim_ids.map(id=>`<button class="ref" data-jump="section-${esc(data.sections.find(s=>s.claim_id===id)?.id||'01')}">结论依据 · ${esc(id)}</button>`).join('')}</div></div><div class="hero-right"><div class="readiness"><div><strong class="mono">${count}</strong><span>预案齐备 · 非下单数</span></div><div><strong class="mono">${confirmed}</strong><span>快照时已触发</span></div></div><div class="divider"></div><p>${esc(data.summary.decision)}</p></div></div>
 <div class="metric-grid">${data.metrics.length?data.metrics.map(m=>`<div class="panel metric"><div class="label">${esc(m.label)}</div><div class="value mono">${number(m.value,Number.isInteger(m.value)?0:1)}<span class="unit">${esc(m.unit)}</span></div><div class="foot"><span>${esc(m.note)}</span><button class="ref" data-source="${esc(m.evidence_id)}">来源</button></div></div>`).join(''):['市场广度','接力溢价','梯队与失败成本','数据覆盖'].map(x=>`<div class="panel metric"><div class="label">${x}</div><div class="value">—</div><div class="foot">等待本地事实包</div></div>`).join('')}</div>
 ${heading(data.summary.focus_title,data.summary.focus_note,`<button class="btn text" data-jump="section-06">完整分析 →</button>`)}
 <div class="cards2">${data.summary.focus_ids.length?data.summary.focus_ids.map(id=>focusCard(data.stocks.find(s=>s.id===id))).join(''):'<div class="empty"><strong>重点卡 01 · 待生成</strong>有证据时显示最重要的机会或参与计划；不能为了填满版面编出标的。</div><div class="empty"><strong>重点卡 02 · 可为空</strong>指定跟踪股仍全部保留在持续池；“可观察”不会伪装成“可立即买”。</div>'}</div>
 ${data.stocks.some(s=>s.status==='PLAN_READY'||s.status==='SIGNAL_CONFIRMED')?`${heading('参与计划｜仍需最新行情与账户核验','只展示经契约检查的计划，最多两项。')}<div class="cards2">${data.stocks.filter(s=>['PLAN_READY','SIGNAL_CONFIRMED'].includes(s.status)).slice(0,2).map(focusCard).join('')}</div>`:''}
 ${heading('持续跟踪与替代标的','保留原理由；对替代标的使用相同证据标准。')} ${stockTable()}
 ${heading('下一交易日：三种情形，三种响应','没有未经校准的概率，也没有无条件买入。')} ${scenarios()}
 ${heading('会改变结论的关键问题','风险和数据缺口不能折叠到报告最后。',`<button class="btn text" data-jump="quality">全部问题 →</button>`)} ${risks(3)}
 <div class="callout warning" style="margin-top:18px"><strong>本版修正：</strong>${esc(data.summary.risk)}</div>
 </section>`;
}
function renderBarChart(){
 if(!data.history.length)return '<div class="empty">没有历史序列，不绘制示意行情。</div>';
 const max=Math.max(...data.history.map(r=>r.limit_up),1);return `<div class="chart-box"><div class="table-title">收盘涨停家数｜5个交易日</div><div class="bar-chart" role="img" aria-label="${esc(data.history.map(r=>r.date+' '+r.limit_up+'只').join('；'))}">${data.history.map(r=>`<div class="bar-col"><span class="bar-label">${r.limit_up}</span><div class="bar" style="height:${Math.round(r.limit_up/max*102)}px"></div><span class="bar-date">${esc(r.date.slice(5).replace('-','/'))}</span></div>`).join('')}</div><div class="legend">纵向从0计数；当日统计域略有变化。${ref('H:5D')}</div></div>`;
}
function renderLineChart(){
 if(!data.history.length)return '<div class="empty">没有历史序列，不绘制示意收益。</div>';
 const w=480,h=185,x0=40,x1=460,y0=23,y1=140;const vals=data.history.map(r=>r.pool_close_median);let lo=Math.min(...vals,0),hi=Math.max(...vals,0);const pad=Math.max((hi-lo)*.2,.4);lo-=pad;hi+=pad;
 const xy=(v,i)=>[x0+(x1-x0)*i/Math.max(vals.length-1,1),y1-(v-lo)/(hi-lo)*(y1-y0)];const points=vals.map((v,i)=>xy(v,i).join(',')).join(' ');const zy=xy(0,0)[1];
 return `<div class="chart-box"><div class="table-title">昨日涨停池｜收盘溢价中位数</div><svg class="line-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(data.history.map(r=>r.date+' '+pct(r.pool_close_median)).join('；'))}"><line x1="${x0}" y1="${zy}" x2="${x1}" y2="${zy}" stroke="#bbc9d2" stroke-dasharray="4 4"/><text x="7" y="${zy+4}" fill="#748897" font-size="11">0%</text><polyline points="${points}" fill="none" stroke="#147a89" stroke-width="2.5"/>${vals.map((v,i)=>{const[x,y]=xy(v,i);return `<circle cx="${x}" cy="${y}" r="4" fill="#147a89"/><text x="${x}" y="${y-10}" text-anchor="middle" fill="#344f61" font-size="11">${v>0?'+':''}${number(v,2)}%</text><text x="${x}" y="166" text-anchor="middle" fill="#748897" font-size="11">${esc(data.history[i].date.slice(5))}</text>`}).join('')}</svg><div class="small muted">同一类母池按日变化；不是任何策略的实际收益。${ref('H:5D')}</div></div>`;
}
function chain(c){return `<div class="reason-grid"><div class="reason-cell"><div class="kicker">01 / 可观察事实</div>${pList(c.facts)}</div><div class="reason-cell"><div class="kicker">02 / 时间或同类比较</div><p>${esc(c.comparison)}</p></div><div class="reason-cell"><div class="kicker">03 / 解释与判断 · 非原始事实</div><p>${esc(c.interpretation)}</p></div><div class="reason-cell contrary"><div class="kicker">04 / 反证与尚未知</div>${pList(c.contrary)}</div></div><div class="next-check"><b>下一步核验：</b>${esc(c.next_check)}</div>${refs(c.evidence_ids)}`}
function modelCards(){return `<div class="mode-grid">${data.modes.map(m=>`<div class="mode-card"><h3>${esc(m.name)}</h3>${badge(m.status,'info')}<p><b>范围：</b>${esc(m.scope)}</p><p><b>最少数据：</b>${esc(m.minimum)}</p><p class="rule">${esc(m.rule)}</p></div>`).join('')}</div>`}
function comparisonCards(){return data.comparisons.map(c=>`<article class="comparison"><div class="badges">${badge(c.matched?'比较条件齐备':'比较条件未齐','warn')}</div><h3 style="margin-top:9px">${esc(c.title)}</h3><p><b>比较目标：</b>${esc(c.objective)}</p><p class="verdict">${esc(c.conclusion)}</p><p><b>缺口：</b>${esc(c.missing.join('；'))}</p>${refs(c.support_ids)}</article>`).join('')}
function stockDetails(){return data.stocks.map(s=>{const p=s.plan;return `<article class="stock-details" id="stock-${esc(s.id)}"><div class="stock-head"><h3>${esc(s.name)} <span class="small mono muted">${esc(s.symbol)}</span></h3><div class="badges" style="margin-top:8px">${status(s)}${badge(modeMap[s.mode])}${badge(s.role,'dark')}${s.pinned?badge('指定跟踪','info'):''}</div></div><div class="details-grid"><div class="kv"><small>机会判断 · 不等于参与许可</small><p>${esc(s.opportunity_reason)}</p></div><div class="kv"><small>最强反证 / 未知</small><p>${esc(s.contrary)}</p></div><div class="kv"><small>原计划与期限</small><p>${esc(s.prior_plan_id||'未提供原计划，不假定原等待条件。')} / ${esc(s.horizon)}</p></div><div class="kv"><small>下一步与失效</small><p>${esc(s.next_check)}</p><p>${esc(s.invalidating)}</p></div></div><div class="contract-grid"><div class="kv"><small>触发 / 许可区间</small><p>${p?esc(number(p.entry_min)+'–'+number(p.entry_max)):'未定义 · 不填猜测价'}</p></div><div class="kv"><small>结构失效 / 退出</small><p>${p?esc(number(p.invalidation_price)+' / '+p.exit_rule):'未提供完整合约'}</p></div><div class="kv"><small>风险预算</small><p>${p?esc(number(p.max_loss_cash)+'元 / '+p.budget_status):'未知 · 不计算仓位'}</p></div><div class="kv"><small>有效期 / 版本</small><p>${p?esc(dateDisplay(p.expires_at)+' / '+p.strategy_version):'补证后冻结 · 禁止追认'}</p></div></div><p class="table-note">${esc(s.notes)}</p>${refs(s.support_ids)}</article>`}).join('')}
function auditBlock(){const a=data.audit;return `<section class="panel detail-section" id="audit"><div class="detail-head"><span class="step">A</span><div><h2>预案兑现与误判复盘</h2><p>保留原记录，不用结果重写过程。</p></div></div><div class="callout warning"><strong>${esc(a.title)}</strong><p>${esc(a.note)}</p></div><div class="audit-grid"><div class="audit-cell"><small>原判断的问题</small><p>${esc(a.previous)}</p></div><div class="audit-cell"><small>本版修订</small><p>${esc(a.correction)}</p></div><div class="audit-cell"><small>尚未证明</small><p>${esc(a.not_proven)}</p></div></div><div class="badges" style="margin-top:15px">${badge('原冻结计划：'+(a.frozen_plan_available?'有':'未提供'),a.frozen_plan_available?'good':'warn')}${badge('同时间报价：'+(a.matched_quote_available?'有':'未提供'),'warn')}${badge('实际成交：'+(a.actual_fills_available?'有':'未提供'),'warn')}</div><details><summary>错误归因分类与验证要求</summary><div class="detail-content">${table({title:'分类不等于已经归因；先取得各环节证据',columns:['编码','类型','判定含义'],rows:data.spec.errors,evidence_ids:[],note:'优化需冻结样本与评价期限，比较修改前后；一次结果不能验证策略收益。'})}</div></details><div class="next-check"><b>持仓与执行：</b>${data.meta.portfolio_status==='NOT_PROVIDED'?'账户完整记录未提供；不能把“没有导入记录”显示为账户空仓。':'只按已核验账户快照展示；报告不具备交易权限。'}</div></section>`}
function specification(){return `<section class="panel detail-section" id="method"><div class="detail-head"><span class="step">B</span><div><h2>本版方法与AI数据契约</h2><p>本地先计算，AI做对比解释；不把全量表格当提示词。</p></div></div><div class="callout">${esc(data.spec.new_vs_source)}</div><div class="principles" style="margin-top:20px">${data.spec.principles.map(([t,b])=>`<div class="principle"><h4>${esc(t)}</h4><p>${esc(b)}</p></div>`).join('')}</div><div class="workflow">${data.spec.workflow.map(([n,t,b])=>`<div class="flow-step"><span>${esc(n)}</span><b>${esc(t)}</b><p>${esc(b)}</p></div>`).join('')}</div><div class="next-check"><b>上下文预算：</b>${esc(data.spec.budget)}</div><details open><summary>AI前置数据清单 · 按模式决定最少证据</summary><div class="detail-content">${table({title:'P0必须随包，P1按模式必需，P2用于事后评价',columns:['优先级 / 数据组','本地需要的原始数据','本地处理','发送AI的内容'],rows:data.spec.data_matrix,evidence_ids:[],note:'报告展示包可携带全文证据；AI精简包只发必要摘要与来源索引。两者不能混用。'})}</div></details><details><summary>本次有界补证任务</summary><div class="detail-content">${data.requests.length?table({title:'按问题补数据，不索取全市场分钟线',columns:['任务','对象','要验证什么','字段','最多行数','时点边界'],rows:data.requests.map(r=>[r.id+' / '+r.priority,r.symbols.join('、'),r.purpose,r.fields.join('、'),r.max_rows,r.cutoff]),evidence_ids:[],note:'返回MISSING是合法结果；不能在缺数据时补写支撑、成交或资金身份。'}):'<div class="empty">尚无补证任务；由分析结论明确指出缺口后生成。</div>'}</div></details><details><summary>原材料覆盖与待验证登记</summary><div class="detail-content">${table({title:'按来源保留知识，不把新设计冒充原规则',columns:['来源','知识点','系统位置','待验证/边界'],rows:data.spec.coverage.map(r=>r.slice(0,4)),evidence_ids:data.spec.coverage.map(r=>r[4]),note:'本表是来源级映射，不声称重新验证了原系统155条登记项。未纳入本日事实判断的知识仍可在对应模式使用或待验证。'})}</div></details><p class="table-note">${esc(data.spec.security)}</p></section>`}
function qualityBlock(){return `<section class="panel detail-section" id="quality"><div class="detail-head"><span class="step">C</span><div><h2>数据质量、来源与版本</h2><p>同源数据只存一份，所有判断可回到证据。</p></div></div>${risks(99)}<details><summary>证据目录 · ${data.sources.length}个对象</summary><div class="detail-content sources-table"><div class="table-wrap"><table><thead><tr><th>证据</th><th>类型 / 来源</th><th>快照哈希</th><th>操作</th></tr></thead><tbody>${data.sources.map(s=>`<tr><td>${esc(s.id)}<div class="small muted">${esc(s.title)}</div></td><td>${esc(originMap[s.origin]||s.origin)}<div class="small muted">${esc(s.source_path)}</div></td><td class="mono">${esc(s.snapshot_sha256.slice(0,16))}…</td><td><button class="ref" data-source="${esc(s.id)}">查看原文</button></td></tr>`).join('')}</tbody></table></div></div></details><div class="version-grid"><div>市场截止<strong>${esc(dateDisplay(data.meta.market_as_of))}</strong></div><div>知识截止<strong>${esc(dateDisplay(data.meta.knowledge_as_of))}</strong></div><div>本版渲染<strong>${esc(dateDisplay(data.meta.rendered_at))}</strong></div><div>原始前向冻结<strong>${data.meta.forward_frozen?'是':'否 / 非前向证明'}</strong></div><div>规则版本<strong>${esc(data.meta.rule_version)}</strong></div><div>题材映射<strong>${esc(data.meta.mapping_version)}</strong></div></div></section>`}
function detail(){return `<section role="tabpanel" id="detail" aria-labelledby="tab-detail" hidden><div class="print-heading">第二部分 · 细化版</div><div class="panel detail-intro"><div class="eyebrow">DATA → COMPARISON → INTERPRETATION → CONDITION</div><h2 style="margin-top:7px">让每一个结论都有可检查的依据。</h2><p>七步顺序保持不变。正文依次列出事实、比较、解释、反证与下一步条件；附录处理模式遗漏、证据不对称及压缩漏失。</p><div class="hint">开启“段落规范”后，各小节会显示写作任务。精华摘要必须引用这里的同一个claim_id，不单独重写一套结论。</div></div>
 ${data.sections.map(s=>{const c=claim(s.claim_id);return `<section class="panel detail-section" id="section-${s.id}"><div class="detail-head"><span class="step">${s.id}</span><div><h2>${esc(s.title)}</h2><p>${esc(s.subtitle)}</p></div></div><div class="hint"><b>本段输出：</b>${esc(s.hint)}</div><div class="takeaway">${esc(c.conclusion)}</div>${table(s.table)}${s.id==='01'&&data.history.length?`<div class="charts-row">${renderBarChart()}${renderLineChart()}</div>`:''}${chain(c)}${s.id==='06'?`<div class="section-head"><div><h2>三种模式并行评估</h2><p>不是只允许回踩，也不是强势就能跳过风险。</p></div></div>${modelCards()}${comparisonCards()}${stockDetails()}`:''}${s.id==='07'?`<div style="margin-top:18px">${scenarios()}</div>`:''}</section>`}).join('')}
 ${auditBlock()}${specification()}${qualityBlock()}</section>`}
function render(){
 const m=data.meta;
 $('#crumb-date').textContent=m.trade_date||'未加载数据';
 document.title=`每日交易复盘 ${m.trade_date||'空白模板'} · V2.0`;
 $('#app').innerHTML=`<div class="report-meta"><div><div class="eyebrow">DAILY REVIEW / ${esc(m.trade_date||'TEMPLATE')}</div><p class="small muted">观察日：${esc(m.target_date||'待交易日历确认')} · ${esc(m.scope)}</p></div><div class="badges">${badge(m.source_status==='PARTIAL_DATA'?'数据部分覆盖':m.source_status==='NOT_LOADED'?'未加载数据':m.source_status,'warn')}${badge(m.mode==='HISTORICAL_DEMO'?'历史重建':m.mode==='TEMPLATE'?'可复用空白模板':m.mode,'dark')}</div></div><div class="context"><b>边界</b><span>${esc(m.banner)}</span></div><div class="view-tabs" role="tablist" aria-label="选择报告层级"><button id="tab-brief" class="view-tab" role="tab" data-view="brief" aria-controls="brief" aria-selected="true">精华版<small>结论先行</small></button><button id="tab-detail" class="view-tab" role="tab" data-view="detail" aria-controls="detail" aria-selected="false">细化版<small>数据与推导</small></button></div>${brief()}${detail()}<footer class="report-foot"><span>输入边界：${esc(m.objective)}<br>分析性质：报告与模板，不承诺收益，不替代独立交易风控。</span><span class="mono">${esc(m.report_id)}<br>schema: ${esc(data.schema_version)} · 离线 / 无交易接口</span></footer>`;
 switchView(activeView,false);applyFilters();
 $('#subnav').innerHTML=data.sections.map(s=>`<a href="#section-${s.id}" data-jump="section-${s.id}">${s.id} · ${esc(s.title)}</a>`).join('')+'<a href="#audit" data-jump="audit">A · 误判与兑现</a><a href="#method" data-jump="method">B · AI与数据规范</a><a href="#quality" data-jump="quality">C · 证据与版本</a>';
}
function switchView(view,scroll=true){
 activeView=view==='detail'?'detail':'brief';
 ['brief','detail'].forEach(v=>{const p=$('#'+v);if(p)p.hidden=v!==activeView;const tab=$('#tab-'+v);if(tab){tab.setAttribute('aria-selected',String(v===activeView));tab.setAttribute('tabindex',v===activeView?'0':'-1')}});
 document.querySelectorAll('.rail-link').forEach(b=>b.classList.toggle('active',b.dataset.view===activeView));
 if(scroll)window.scrollTo({top:0,behavior:'smooth'});
}
function jump(id){switchView('detail',false);const el=document.getElementById(id);if(el){el.scrollIntoView({behavior:'smooth',block:'start'});el.setAttribute('tabindex','-1');el.focus({preventScroll:true});history.replaceState(null,'','#'+id);}}
function applyFilters(){let n=0;document.querySelectorAll('[data-stock-row]').forEach(row=>{const s=data.stocks.find(s=>s.id===row.dataset.stockRow);const match=(!currentQuery||(s.name+s.symbol).toLowerCase().includes(currentQuery.toLowerCase()))&&(!currentMode||s.mode===currentMode)&&(!onlyPinned||s.pinned);row.hidden=!match;row.classList.toggle('table-filter-hidden',!match);if(match)n++;});if($('#filterCount'))$('#filterCount').textContent=`${n} / ${data.stocks.length}只`;applyHotspotFilter();}
function applyHotspotFilter(){const rows=[...document.querySelectorAll('[data-hotspot-row]')],query=hotspotQuery.trim().toLowerCase();let shown=0;rows.forEach(row=>{const match=query?row.textContent.toLowerCase().includes(query):!onlyHotspotFocus||row.dataset.hotspotFocus==='1';row.hidden=!match;row.classList.toggle('table-filter-hidden',!match);if(match)shown++;});if($('#hotspotCount'))$('#hotspotCount').textContent=`${shown} / ${rows.length}只${query?' · 全池搜索':''}`;}
function openSource(id){const s=data.sources.find(x=>x.id===id);if(!s)return;state.lastFocus=document.activeElement;$('#sourceTitle').textContent=s.title;$('#sourceMeta').textContent=`${s.id} · ${originMap[s.origin]||s.origin}\n来源：${s.source_path}\n定位：${s.locator}\n可知时刻：${s.available_at||'原始到达时间未提供；不能作为前向可知证明'}\nSHA-256：${s.snapshot_sha256}`;$('#sourceMeta').style.whiteSpace='pre-wrap';$('#sourceContent').textContent=s.content||'未携带全文；请按来源目录到本地证据层读取。';$('#sourceDialog').showModal();}
function notify(msg){$('#toast').textContent=msg;$('#toast').hidden=false;setTimeout(()=>$('#toast').hidden=true,4200)}
function downloadJSON(){const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`review_${data.meta.trade_date||'template'}_v2.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function errorDisplay(err){const box=document.createElement('div');box.className='error-banner';const title=document.createElement('b');title.textContent='数据校验未通过，报告未替换';const p=document.createElement('p');p.textContent=err.slice(0,12).join('；');box.append(title,p);$('#main').prepend(box);setTimeout(()=>box.remove(),18000);}
document.addEventListener('click',e=>{const b=e.target.closest('[data-source],[data-jump],[data-view]');if(!b)return;if(b.dataset.source){e.preventDefault();openSource(b.dataset.source)}else if(b.dataset.jump){e.preventDefault();jump(b.dataset.jump)}else if(b.dataset.view){e.preventDefault();switchView(b.dataset.view);history.replaceState(null,'','#'+activeView);}});
document.addEventListener('input',e=>{if(e.target.id==='stockSearch'){currentQuery=e.target.value;applyFilters()}if(e.target.id==='hotspotSearch'){hotspotQuery=e.target.value;applyHotspotFilter()}});
document.addEventListener('change',e=>{if(e.target.id==='modeFilter'){currentMode=e.target.value;applyFilters()}if(e.target.id==='pinnedFilter'){onlyPinned=e.target.checked;applyFilters()}if(e.target.id==='hotspotFocus'){onlyHotspotFocus=e.target.checked;applyHotspotFilter()}});
document.addEventListener('keydown',e=>{if(e.target.matches('[role=tab]')&&['ArrowRight','ArrowLeft','Home','End'].includes(e.key)){e.preventDefault();const v=e.key==='Home'?'brief':e.key==='End'?'detail':activeView==='brief'?'detail':'brief';switchView(v,false);$('#tab-'+v).focus();}});
$('#showHints').addEventListener('change',e=>document.body.classList.toggle('show-hints',e.target.checked));
$('#importBtn').addEventListener('click',()=>$('#jsonFile').click());
$('#jsonFile').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;try{if(f.size>8*1024*1024)throw new Error('报告展示JSON大于8MB，请压缩展示包；不是删除风险数据。');const incoming=JSON.parse(await f.text()),errors=validate(incoming);if(errors.length){errorDisplay(errors);return;}const oldData=data;try{data=incoming;currentQuery='';currentMode='';onlyPinned=false;activeView='brief';render();}catch(renderError){data=oldData;render();throw renderError;}notify('已在本机加载；未发送网络请求。');}catch(err){errorDisplay([err.message||String(err)]);}finally{e.target.value='';}});
$('#exportBtn').addEventListener('click',downloadJSON);
$('#printBtn').addEventListener('click',()=>{document.body.classList.remove('print-all');window.print()});
$('#printAllBtn').addEventListener('click',()=>{state.printDetails=[...document.querySelectorAll('details')].map(el=>[el,el.open]);document.querySelectorAll('details').forEach(el=>el.open=true);document.body.classList.add('print-all');document.querySelectorAll('.table-filter-hidden').forEach(r=>r.hidden=false);window.print()});
window.addEventListener('afterprint',()=>{document.body.classList.remove('print-all');if(state.printDetails){state.printDetails.forEach(([el,open])=>el.open=open);state.printDetails=null;}applyFilters()});
$('#closeDialog').addEventListener('click',()=>$('#sourceDialog').close());
$('#sourceDialog').addEventListener('click',e=>{if(e.target===$('#sourceDialog'))$('#sourceDialog').close()});
$('#sourceDialog').addEventListener('close',()=>state.lastFocus?.focus());
try{data=JSON.parse($('#report-data').textContent);const errors=validate(data);if(errors.length)throw new Error(errors.join('；'));render();const hash=location.hash.slice(1);if(hash==='detail'||hash==='brief')switchView(hash,false);else if(hash)requestAnimationFrame(()=>jump(hash));window.__REPORT_READY__=true;}catch(err){$('#app').innerHTML='<div class="error-banner"><b>报告无法加载</b><p>'+esc(err.message)+'</p></div>';window.__REPORT_READY__=false;}
window.__getReportData=()=>JSON.parse(JSON.stringify(data));
})();

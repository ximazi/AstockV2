"""Build the day's review.report.v2 from artefacts the pipeline already froze.

Inputs (all read-only): the eod_review fact package, trading_closure.json, books_daily.json, the auto book's
paper plans, jev_shadow.json, the bars behind every named stock, concept_mapping.json and fupan_tracking.json
(the user's pinned list). Nothing here recomputes a ranking or touches a ledger; the report can only describe,
never order. Opportunity comes first: a transparent mechanical priority (score with its parts) is kept apart
from eligibility (engine state, plan) and from gaps, so a strong stock without an executable plan stays visible."""
import hashlib
import json
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import eod_review as R
from ..trading import books as engine_books, modes as engine_modes, store
from .. import entry_contracts
from .. import hotspot_core as HC
from .. import hotspot_relay as HR
from .. import hotspot_funnel as HF
from .. import first_board_watch as FW
from .. import entry_review as ER, theme_risk as TR
from .. import theme_semantics as TS
from . import ASSETS, VERSION, SCHEMA_VERSION
from .ai_packet import merge_analysis, prepare, prompt_text, stats
from .contract import validate
from .render import render

TZ = timezone(timedelta(hours=8))
STATIC = json.loads((ASSETS / "static_texts.json").read_text(encoding="utf-8"))
MODE_OF = {"TREND_PULLBACK": "TREND_PULLBACK", "STRENGTH_CONFIRM": "STRONG_CONFIRM", "OVERNIGHT": "OVERNIGHT"}
MODE_LABEL = {"STRONG_CONFIRM": "强势确认", "TREND_PULLBACK": "趋势回踩", "OVERNIGHT": "尾盘隔夜"}
ROLE_MODE = {"HEIGHT_LEADER": "STRONG_CONFIRM", "THEME_LEADER": "STRONG_CONFIRM", "TREND_CORE": "TREND_PULLBACK",
             "RELAY_FAILURE": "STRONG_CONFIRM", "LIMIT_DOWN": "STRONG_CONFIRM"}
STATUS_LABEL = {"DATA_GAP": "缺数据", "WATCH": "观察", "WAIT_SETUP": "等结构", "WAIT_TRIGGER": "等触发",
                "PLAN_READY": "预案齐备", "SIGNAL_CONFIRMED": "已触发", "REJECT": "否决", "EXPIRED": "过期"}
STATE_STATUS = {"PLAN_REVIEW": "WAIT_TRIGGER", "WAIT_SETUP": "WAIT_SETUP", "RISK_WATCH": "WATCH"}
MODE_MISSING = {"STRONG_CONFIRM": "分钟或盘口过程（封板、开板、回封先后）", "OVERNIGHT": "14:30 后快照与次日早盘退出窗口"}
BLOCKER_LABEL = {"STRUCTURE_RISK": "结构风险", "WAIT_PULLBACK_CONFIRMATION": "回踩未确认", "TRACKING_ONLY": "仅续踪", "MARKET_WAIT": "等待市场",
                 "DATA_WAIT": "数据待核", "结构风险": "结构风险", "回踩未确认": "回踩未确认", "仅续踪": "仅续踪"}
TRACKING = Path("fupan_tracking.json")
MAPPING = Path("concept_mapping.json")
MAX_WATCH = 8
# Opportunity pool: how many ranked names the report carries, how many become focus cards, and the two verdict bars.
POOL = {"top": 10, "focus": 2, "strong": 60, "some": 40, "strength_rank": 20, "ladder_min_height": 2, "limit_amount_top": 10}
HOT, COLD = ("热", "偏热"), ("冷", "偏冷")
SENT_KEYS = ("limit_up", "seal_failure_rate", "limit_down", "pool_mean_pct", "pool_up_share", "open_premium_pct", "promotion_1_to_2", "max_height")


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _mtime(path):
    path = Path(path)
    return datetime.fromtimestamp(path.stat().st_mtime, TZ).isoformat(timespec="seconds") if path.exists() else None


def _pct(x, digits=2):
    return "--" if x is None else f"{x:+.{digits}f}%"


def _num(x, digits=2):
    return "--" if x is None else f"{x:.{digits}f}"


def _ratio(r):
    if not r or r.get("value") is None:
        return "--"
    return f"{r['value'] * 100:.1f}%（{r.get('numerator')}/{r.get('denominator')}）"


def next_session(sessions, day):
    later = [d for d in sessions if d > day]
    if later:
        return later[0]
    d = datetime.fromisoformat(day) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.date().isoformat()


# ---------------------------------------------------------------- per-stock facts from bars
def stock_metrics(row):
    """change / amount / close location / 20-day return from the stock's own bars; None where the bar is unusable."""
    f = row["frame"] if row else None
    if f is None or f.empty:
        return {"change_pct": None, "amount_yi": None, "close_location": None, "return20_pct": None}
    b = f.iloc[-1]
    if not bool(b.valid):
        return {"change_pct": None, "amount_yi": None, "close_location": None, "return20_pct": None}
    spread = float(b.high - b.low)
    return {"change_pct": round(float(b.close / b.preclose - 1) * 100, 2),
            "amount_yi": round(float(b.amount) / 1e8, 2),
            "close_location": round(float((b.close - b.low) / spread), 2) if spread > 0 else None,
            "return20_pct": round(float(b.close / f.close.iloc[-21] - 1) * 100, 2) if len(f) >= 21 and f.close.iloc[-21] > 0 else None}


def opportunity(symbol, row, facts, mapping, leads):
    """Mechanical review priority from facts already on the table; every point is named so the reader can disagree.

    It ranks what deserves a look first. It is not a return forecast and never decides eligibility."""
    m = stock_metrics(row)
    if m["change_pct"] is None:
        return {"score": None, "parts": [], "text": "无当日可核验行情"}
    f = row["frame"]
    b = f.iloc[-1]
    parts = [(f"今日 {m['change_pct']:+.1f}%", round(min(max(m["change_pct"], 0.0), 10.0) * 2.5))]
    if m["return20_pct"] is not None:
        parts.append((f"20日 {m['return20_pct']:+.0f}%", round(min(max(m["return20_pct"], 0.0), 60.0) / 3)))
    if m["close_location"] is not None:
        parts.append((f"收盘位置 {m['close_location']:.2f}", round(m["close_location"] * 15)))
    previous = f.turnover.iloc[-21:-1]
    previous = previous[previous > 0]
    if len(previous) >= 5 and b.turnover > 0:
        rel = float(b.turnover) / float(previous.median())
        parts.append((f"相对换手 {rel:.1f}×", round(min(rel, 3.0) * 5)))
    height = int(facts["ladder"]["heights"].get(symbol, 0) or 0)
    if height:
        parts.append((f"{height} 板", min(height, 3) * 5))
    hit = sorted({c for c in (mapping.get(symbol) or {}).get("concepts", []) if c in leads}, key=lambda c: (-leads[c][1], c))
    if hit:
        parts.append((f"题材 {' / '.join(hit)}（{leads[hit[0]][0]}，最多涨停 {leads[hit[0]][1]} 只，多题材只加一次）", 10))
    events = facts["loss_effects"]["events"]
    for key, label in (("gap_up_reversal", "高开低走"), ("breakdown_20d", "跌破20日低点")):
        if symbol in events.get(key, {}).get("symbols", []):
            parts.append((label, -15))
    if m["amount_yi"] is not None and m["amount_yi"] < 1:
        parts.append(("成交额不足 1 亿", -10))
    score = max(0, min(100, sum(p for _, p in parts)))
    return {"score": score, "parts": parts, "text": "；".join(f"{label} {pts:+d}" for label, pts in parts)}


def theme_leads(facts):
    """Themes whose members earn the theme point: the five with most limit-ups and the five most concentrated."""
    t = facts["themes"]
    leads = {x["theme"]: ("涨停榜", x["limit_up"]) for x in t.get("top_by_limit_up", [])[:5]}
    leads.update({x["theme"]: ("占比榜", x["limit_up"]) for x in t.get("top_by_limit_share", [])[:5]})
    return leads


def pool_history(rows, sessions, day, window=20):
    """Per session: limit counts and how yesterday's limit-up pool traded; the same arithmetic as eod_review's series."""
    position = {d: i for i, d in enumerate(sessions)}
    days = [d for d in sessions if d <= day][-window:]
    per = {d: {"U": 0, "D": 0, "close": [], "close_open": []} for d in days}
    for row in rows.values():
        f = row["frame"]
        dates = f.date.tolist()
        for i, d in enumerate(dates):
            if d not in per:
                continue
            b = f.iloc[i]
            if bool(b.st) or not bool(b.traded) or not bool(b.valid):
                continue
            p = per[d]
            p["U"] += bool(b.is_U)
            p["D"] += bool(b.is_D)
            if i == 0 or position.get(d, -9) - position.get(dates[i - 1], -9) != 1:
                continue
            prev = f.iloc[i - 1]
            if bool(prev.is_U) and bool(prev.traded) and not bool(prev.st) and abs(b.preclose / prev.close - 1) <= .001:
                p["close"].append(float(b.close / b.preclose - 1) * 100)
                if b.open > 0:
                    p["close_open"].append(float(b.close / b.open - 1) * 100)
    return [{"date": d, "limit_up": int(p["U"]), "limit_down": int(p["D"]),
             "pool_close_median": round(statistics.median(p["close"]), 3) if p["close"] else 0.0,
             "pool_close_open_mean": round(statistics.fmean(p["close_open"]), 3) if p["close_open"] else 0.0}
            for d, p in per.items()]


# ---------------------------------------------------------------- readings (deterministic, from percentiles)
def _read(s, key):
    mt = s.get(key) or {}
    return mt.get("reading") if mt.get("status") == "OK" else None


def _reading(s, key):
    mt = s.get(key) or {}
    if mt.get("status") != "OK":
        return "近20日无可比样本"
    return f"近20日分位 {mt['percentile'] * 100:.0f}%（{mt['reading']}）"


def _line(s, key, fmt):
    mt = s.get(key) or {}
    if mt.get("status") != "OK":
        return f"{mt.get('label', key)} 无可比样本"
    return f"{mt['label']} {fmt(mt['today'])}（{mt['reading']}，近20日中位 {fmt(mt['median'])}）"


def readings(facts, names=None):
    """Environment and risk readings written by rule from the 20-day percentiles; relative, not predictive."""
    names = names or {}
    s = facts.get("sentiment", {}).get("metrics", {})
    y, l, ev = facts["yesterday_pool"], facts["ladder"], facts["loss_effects"]["events"]
    hot = sum(1 for k in SENT_KEYS if _read(s, k) in HOT)
    cold = sum(1 for k in SENT_KEYS if _read(s, k) in COLD)
    known = sum(1 for k in SENT_KEYS if _read(s, k))
    env = "偏热" if hot >= 5 else "偏冷" if cold >= 5 else "中性"
    pc = lambda v: f"{v * 100:.0f}%"
    sg = lambda v: f"{v:+.2f}%"
    n = lambda v: f"{v:g}"
    env_text = (f"接力情绪{env}：{known} 项可比指标中 {hot} 项偏热以上、{cold} 项偏冷以下。"
                f"{_line(s, 'pool_mean_pct', sg)}；{_line(s, 'open_premium_pct', sg)}；{_line(s, 'pool_up_share', pc)}。")
    env_conclusion = {"偏热": "环境支持按强势确认与趋势回踩两条线找机会；热度高时炸板与高位失败成本同步上升，机会要配缺口核对。",
                      "中性": "环境中性：机会看个股结构与题材一致性，不靠情绪推动。",
                      "偏冷": "环境偏冷：只按趋势回踩与持仓合约行事，不追强势。"}[env]
    promo, height = _read(s, "promotion_1_to_2"), _read(s, "max_height")
    ladder_text = (f"{_line(s, 'promotion_1_to_2', pc)}；{_line(s, 'max_height', n)}；今日 " + "，".join(f"{h} 板 {c} 只" for h, c in l["by_height"].items()) + "。")
    ladder_read = ("梯队有延续且在拔高" if promo in HOT and height in HOT else "梯队有延续但没有拔高" if promo in HOT
                   else "梯队断层，接力成本高" if promo in COLD else "梯队中性")
    leaders = [f"{names.get(sym) or sym}（{h} 板）" for sym, h in l["heights"].items() if h >= 3][:6]
    t = facts["themes"]
    top_u = t["top_by_limit_up"][:1]
    top_s = t["top_by_limit_share"][:1]
    theme_text = ((f"涨停最集中：{top_u[0]['theme']}（涨停 {top_u[0]['limit_up']} 只，广度 {_ratio(top_u[0].get('breadth'))}）" if top_u else "无涨停题材统计")
                  + (f"；小范围一致性最高：{top_s[0]['theme']}（涨停 {top_s[0]['limit_up']}/{top_s[0]['members']}）" if top_s else "") + "。")
    seal, ld = _read(s, "seal_failure_rate"), _read(s, "limit_down")
    neg = ev.get("yesterday_pool_negative", {}).get("count", 0)
    risk_text = (f"{_line(s, 'seal_failure_rate', pc)}；{_line(s, 'limit_down', n)}；昨日池收绿 {neg}/{y.get('n_valid')}，"
                 f"高开低走 {ev.get('gap_up_reversal', {}).get('count', 0)} 只，跌破20日低点 {ev.get('breakdown_20d', {}).get('count', 0)} 只。")
    risk_read = ("亏钱效应未扩散：炸板与跌停都少，接力可以继续" if seal in HOT and ld in HOT
                 else "亏钱效应扩散：炸板与跌停同时偏多，先看持仓合约，不追高" if seal in COLD and ld in COLD
                 else "亏钱效应局部：集中在断板与高位股，避开断板修复之前的接力")
    return {"env": env, "hot": hot, "cold": cold, "env_text": env_text, "env_conclusion": env_conclusion,
            "ladder_text": ladder_text, "ladder_read": ladder_read, "leaders": leaders, "theme_text": theme_text,
            "risk_text": risk_text, "risk_read": risk_read, "neg": neg}


# ---------------------------------------------------------------- verdict
PLAN_WORDS = {"PLAN_READY": "已批准待触发（今日生效）", "WAIT_TRIGGER": "已批准待触发（重建或未到生效日）", "EXPIRED": "过期未成交", "WATCH": "已成交持仓"}


def verdict(stocks):
    """The standing daily rule: ✅ an executable plan exists / 🟡 a strong line without a buy point / ❌ nothing.

    Executable means an approved plan of the named account whose own window is the target session (PLAN_READY);
    an expired or reconstructed plan is named as such and is never a buy point. Words, not only marks."""
    ranked = [s for s in stocks if s["_score"] is not None and s["status"] != "DATA_GAP"]
    executable = [s for s in stocks if s["plan"] and s["status"] == "PLAN_READY"]
    historical = [s for s in stocks if s["plan"] and s["status"] == "WAIT_TRIGGER"]
    expired = [s for s in stocks if s["plan"] and s["status"] == "EXPIRED"]
    held = [s for s in stocks if s["plan"] and s["status"] == "WATCH" and "已成交持仓" in s["contrary"]]
    strong = [s for s in ranked if s["_score"] >= POOL["strong"] and s not in executable and s not in historical and s not in expired]
    parts = []
    if executable:
        parts.append(f"✅ 有买点 {len(executable)} 个（{PLAN_WORDS['PLAN_READY']}）：" + "、".join(
            f"{s['name']}（{MODE_LABEL[s['mode']]}，计划带 {s['plan']['entry_min']:.2f}–{s['plan']['entry_max']:.2f}，失效 {s['plan']['invalidation_price']:.2f}）" for s in executable))
    if historical:
        parts.append(f"🟡 {PLAN_WORDS['WAIT_TRIGGER']}，不是当前买入许可：" + "、".join(
            f"{s['name']}（计划带 {s['plan']['entry_min']:.2f}–{s['plan']['entry_max']:.2f}）" for s in historical))
    if expired:
        parts.append(f"⚪ {PLAN_WORDS['EXPIRED']}（延期须新批准）：" + "、".join(s["name"] for s in expired))
    if held:
        parts.append(f"⚪ {PLAN_WORDS['WATCH']}：" + "、".join(s["name"] for s in held))
    if strong:
        blockers = sorted({x for s in strong[:3] for m in s["missing"] if m.startswith("引擎阻断：") for x in m[5:].split("、")})
        gaps = sorted({m for s in strong[:3] for m in s["missing"] if not m.startswith("引擎阻断：")}) + (["引擎阻断：" + "、".join(blockers)] if blockers else [])
        gaps = gaps or ["可执行模式的触发与失效价"]
        parts.append("🟡 强势主线无买点（有观察价值，待确认）：" + "、".join(f"{s['name']}（{s['_score']} 分）" for s in strong[:3]) + "，差：" + "；".join(gaps))
    if not parts:
        top = ranked[0] if ranked else None
        parts.append(f"❌ 无：机会池最高 {top['_score']} 分（{top['name']}），低于 {POOL['strong']} 分且无可执行计划" if top else "❌ 无：没有可评分的股票")
    code = "✅" if executable else "🟡" if strong or historical else "❌"
    return code, "；".join(parts)


# ---------------------------------------------------------------- sources
class Sources:
    def __init__(self):
        self.items = []
        self.ids = set()

    def add(self, sid, title, origin, path, locator, available_at, content):
        if sid in self.ids:
            return sid
        path = str(path) if path else ""
        sha = R.file_sha(path) if path and Path(path).exists() else _digest(content)
        self.items.append({"id": sid, "title": title, "origin": origin, "source_path": path, "locator": locator,
                           "snapshot_sha256": sha, "available_at": available_at, "content": str(content)[:6000]})
        self.ids.add(sid)
        return sid


# ---------------------------------------------------------------- plans
def plan_from_paper(payload, order, fills, day, effective_day, source_ids, blockers, account_id="AUTO100W"):
    """The auto book's frozen plan in the report's plan shape. Prices are the plan's own cents; the validity window
    is the plan's own effective session (the session after its signal day), never the report's target day. Approval
    and budget are ledger facts of the named account; trigger state comes from the order, fills are separate."""
    entry_min, entry_max, stop = payload["lower_cents"] / 100, payload["limit_cents"] / 100, payload["stop_cents"] / 100
    risk = (order or {}).get("risk_cents")
    rules = payload.get("exit_rules", {})
    status = (order or {}).get("status", "")
    trigger = "CONFIRMED" if status in ("FILLED", "PARTIAL") or fills else "PENDING" if status == "OPEN" else "FAILED"
    return {"id": payload["id"], "strategy_version": f"{account_id} / {payload.get('ranking_version', '')} / {payload.get('exit_version', '')}",
            "approval_status": "APPROVED" if order else "DRAFT",
            "signal_condition": f"账户 {account_id}（旧自动账，不是新 10 万账）；信号日 {payload.get('entry_day', '')}，生效日 {effective_day}："
                                f"{payload.get('model', '')} 开盘按固定上限模型买入 {payload.get('quantity')} 股，参考带 {entry_min:.2f}–{entry_max:.2f}；"
                                f"机械规则批准，非人工推荐；本报告只读取计划，不批准、不延期。",
            "trigger_state": trigger,
            "entry_min": entry_min, "entry_max": entry_max, "invalidation_price": stop,
            "exit_rule": f"硬底线 {stop:.2f}（首仓价-2ATR0）；{rules.get('soft', '')} {rules.get('structure', '')}".strip(),
            "max_loss_cash": round(risk / 100, 2) if risk else round(payload.get("quantity", 0) * (entry_max - stop), 2),
            "budget_status": "PASS" if order else "UNKNOWN",
            "valid_from": f"{effective_day}T09:30:00+08:00", "expires_at": f"{effective_day}T15:00:00+08:00",
            "quote_as_of": None, "quote_fresh": False, "source_ids": list(source_ids), "hard_blocks": list(blockers)}


def auto_plans(books, root):
    """Plan payloads and open orders of the auto book, keyed by symbol; the ledger is only read."""
    auto = (books or {}).get("auto") or {}
    db = Path(auto["db"]) if auto.get("db") and Path(auto["db"]).exists() else Path(root) / engine_books.AUTO_DB
    out = {}
    if not db.exists():
        return out
    import sqlite3
    with sqlite3.connect(db) as conn:
        plans = {json.loads(p)["symbol"]: json.loads(p) for (p,) in conn.execute("SELECT payload FROM paper_plans")}
        orders = {}
        for pid, status, risk, filled, qty in conn.execute("SELECT plan_id, status, risk_cents, filled, quantity FROM paper_orders ORDER BY created_at"):
            orders.setdefault(pid, []).append({"status": status, "risk_cents": risk, "filled": filled, "quantity": qty})
        lots = {}
        for symbol, remaining in conn.execute("SELECT symbol, remaining FROM paper_lots"):
            lots[symbol] = lots.get(symbol, 0) + remaining
    for symbol, payload in plans.items():
        order_rows = orders.get(payload["id"], [])
        out[symbol] = {"payload": payload, "order": next((o for o in order_rows if o["status"] in ("OPEN", "FILLED", "PARTIAL")), None) or (order_rows[-1] if order_rows else None),
                       "fills": sum(o["filled"] for o in order_rows), "open_position": lots.get(symbol, 0), "entry_day": payload.get("entry_day"),
                       "account_id": "AUTO100W"}
    return out


# ---------------------------------------------------------------- report
def s4_verdict(s4, code):
    """S4 is a pattern candidate, not an entry permit; a market trend-exit alert still takes precedence."""
    if not s4:
        return code, "", ""
    m = s4.get("market_risk") or {}
    warn = [r["index"] for r in (s4.get("market_warning") or {}).get("indices", []) if r.get("level") == "WARNING"]
    hold = ("大盘趋势清仓中" if m.get("level") == "TREND_EXIT" else "大盘趋势临界（明日平收即触发清仓）" if m.get("level") == "CRITICAL"
            else "大盘状态未确认" if not s4.get("buy_advised") else f"短期回调预警（{'、'.join(warn)}）" if warn else "")
    rec = s4.get("recommended", [])
    names = "、".join(f"{r['name']}（{r['symbol'][3:]}）" for r in rec) or "无"
    capped = "🔴" if m.get("level") == "TREND_EXIT" else "🟡"
    if rec and not hold:
        return "🟡", f"🟡 S4 形态候选 {len(rec)} 只（尚须题材、承接与入场价格复核）：{names}", hold
    if rec:
        return capped, f"{capped} S4 形态候选 {len(rec)} 只（{names}），但{hold}，不建议新开仓", hold
    return (capped if hold and code == "✅" else code), "S4 今日无推荐" + (f"；{hold}，不建议新开仓" if hold else ""), hold


def build(day, out_root=R.OUT_ROOT, data_root="data", tracking_path=TRACKING, now=None, analysis=None, rows=None, mapping_path=MAPPING):
    out_root, data_root = Path(out_root), Path(data_root)
    folder = out_root / day.replace("-", "")
    pointer = R.read_json(folder / "eod_review/latest.json")
    if not pointer or not Path(pointer.get("path", "")).exists():
        raise ValueError(f"DATA_WAIT: {day} 没有盘后复盘事实包（eod_review），先运行 eod_review")
    facts = R.read_json(Path(pointer["path"]) / "facts.json")
    manifest = R.read_json(Path(pointer["path"]) / "manifest.json")
    closure = R.read_json(folder / "trading_closure.json")
    books = R.read_json(folder / "books_daily.json")
    jev = R.read_json(folder / "jev_shadow.json")
    shadow_rows = {r["symbol"]: r for r in R.read_json(folder / "ranking_shadow.json").get("rows", [])}
    register = R.read_json(folder / "discovery_events.json")
    drafts_by = {}
    for d in R.read_json(folder / "entry_contract_drafts.json").get("drafts", []):
        drafts_by.setdefault(d["symbol"], []).append(d)
    market_policy = R.read_json(folder / "market_policy.json")
    # Previous session's B1/B3 shadow drafts, checked against today's bars (contract reading, not a return).
    prior_session = next((d for d in reversed(R.calendar(data_root)) if d < day), "")
    prior_drafts = R.read_json(out_root / prior_session.replace("-", "") / "entry_contract_drafts.json").get("drafts", []) if prior_session else []
    tracking = R.read_json(tracking_path) if Path(tracking_path).exists() else {}
    mapping = R.read_json(mapping_path).get("mapping", {}) if Path(mapping_path).exists() else {}
    sessions = R.calendar(data_root)
    if rows is None:
        panel, _ = R.load_panel(data_root, day)
        rows, _ = R.day_rows(panel, day)
    now = now or datetime.now(TZ)
    now_text = now.isoformat(timespec="seconds")
    mapping, membership = TS.apply_membership_evidence(mapping, day, now_text)
    target = next_session(sessions, day)
    live = now < datetime.fromisoformat(f"{target}T09:15:00+08:00")
    mode = "FROZEN_EOD" if live else "HISTORICAL_DEMO"
    previous_day = facts.get("previous_day", "")
    previous = latest_report(out_root, previous_day) if previous_day else None

    src = Sources()
    M = src.add(f"M:{day}", f"{day} 盘后复盘事实包（{manifest.get('review_version')}）", "LOCAL_CALCULATION",
                Path(pointer["path"]) / "facts.json", f"eod_review/{manifest.get('review_version')}/facts.json",
                manifest.get("known_at"), _facts_summary(facts, manifest))
    H = src.add("H:20D", "近 20 个交易日涨跌停与昨日涨停池表现", "LOCAL_CALCULATION", "", "本地由日线重算", now_text,
                "昨日涨停池=前一交易日收盘涨停且今日连续交易的股票；收盘/开盘为对昨收、对开盘的简单收益。")
    K = src.add("K:design", "复盘方法与报告规范 v2（设计文档）", "DESIGN", ASSETS / "method_notes_v2.md", "assets/method_notes_v2.md", None,
                (ASSETS / "method_notes_v2.md").read_text(encoding="utf-8")[:4000])
    O = src.add("O:pool", "机会池机械打分规则", "LOCAL_CALCULATION", "", f"fupan/build.py opportunity() {VERSION}", now_text,
                "今日涨幅 ≤25 分（每 1% 2.5 分，上限 10%）；20 日收益 ≤20 分（每 3% 1 分，上限 60%）；收盘位置 ≤15 分；相对换手 ≤15 分（每 1× 5 分，上限 3×）；"
                "连板 5 分/板 ≤15 分；题材涨停榜/占比榜前五成员 +10；高开低走 −15；跌破 20 日低点 −15；成交额不足 1 亿 −10。"
                "分数是复核优先级，不是收益预测；资格与缺口另列。")
    observation_inputs, observation_gaps = HC.load_inputs(folder, day, cutoff=now_text)
    s4 = observation_inputs.get("s4", {})
    S4 = src.add("S:s4", "S4 机会扫描（s4_opportunities.json）", "LOCAL_CALCULATION", folder / "s4_opportunities.json",
                 s4.get("version", ""), _mtime(folder / "s4_opportunities.json") if s4 else None,
                 s4.get("notice", "")) if s4 else None
    C = src.add("C:closure", "收盘冻结快照（trading_closure.json）", "LOCAL_CALCULATION", folder / "trading_closure.json",
                f"snapshot {closure.get('snapshot_id', '')[:12]}", closure.get("recorded_at") or _mtime(folder / "trading_closure.json"),
                json.dumps({k: closure.get(k) for k in ("as_of", "version", "observation_basis", "summary", "scope")}, ensure_ascii=False)) if closure else None
    BK = src.add("BK:books", "三账日报（books_daily.json）", "LOCAL_CALCULATION", folder / "books_daily.json", "auto/signal/manual",
                 _mtime(folder / "books_daily.json"), json.dumps({"auto": {k: (books.get("auto") or {}).get(k) for k in ("live", "plans", "fills", "open_positions", "placed", "executions", "reconciliation")},
                                                                    "signal": {k: (books.get("signal") or {}).get(k) for k in ("trade_set", "eligible", "rule")}}, ensure_ascii=False)) if books else None
    J = src.add("J:jev", "Jev 影子评估（不改状态）", "LOCAL_CALCULATION", folder / "jev_shadow.json", jev.get("question_set", ""),
                _mtime(folder / "jev_shadow.json"), json.dumps(jev.get("summary"), ensure_ascii=False)) if jev else None
    T = src.add("T:tracking", "指定跟踪名单（fupan_tracking.json，用户维护）", "USER_DATA", tracking_path, "pinned",
                _mtime(tracking_path), json.dumps(tracking.get("pinned", []), ensure_ascii=False)) if tracking else None
    P = src.add(f"P:{previous_day}", f"{previous_day} 复盘报告（前一份）", "LOCAL_CALCULATION", previous["_path"], previous["meta"]["report_id"],
                previous["meta"].get("rendered_at"), json.dumps({"focus": previous["summary"]["focus_ids"], "plans": [s["symbol"] for s in previous["stocks"] if s["plan"]]}, ensure_ascii=False)) if previous else None

    # --- market facts
    b, y, l, t, m, s = facts["boards"], facts["yesterday_pool"], facts["ladder"], facts["themes"], facts["market"], facts.get("sentiment", {}).get("metrics", {})
    hist = pool_history(rows, sessions, day)
    metrics = [
        {"label": "收盘涨停", "value": b["counts"]["U"], "unit": "只", "note": _reading(s, "limit_up"), "evidence_id": M},
        {"label": "炸板", "value": b["counts"]["Z"], "unit": "只", "note": "触及涨停但收盘未封住", "evidence_id": M},
        {"label": "收盘跌停", "value": b["counts"]["D"], "unit": "只", "note": _reading(s, "limit_down"), "evidence_id": M},
        {"label": "封板失败率", "value": _v(b["seal_failure_rate"], 100), "unit": "%", "note": _reading(s, "seal_failure_rate"), "evidence_id": M},
        {"label": "昨日涨停池收盘对昨收均值", "value": y["close_vs_pre"].get("mean"), "unit": "%", "note": _reading(s, "pool_mean_pct"), "evidence_id": M},
        {"label": "昨日涨停池开盘溢价均值", "value": y["open_vs_pre"].get("mean"), "unit": "%", "note": _reading(s, "open_premium_pct"), "evidence_id": M},
        {"label": "一进二晋级率", "value": _v(l["promotion_from_previous"].get("1"), 100), "unit": "%", "note": _reading(s, "promotion_1_to_2"), "evidence_id": M},
        {"label": "最高连板", "value": l["max_height"], "unit": "板", "note": _reading(s, "max_height"), "evidence_id": M},
        {"label": "沪深300", "value": (m.get("index") or {}).get("return_pct"), "unit": "%", "note": f"20日 {_pct((m.get('index') or {}).get('return20_pct'))}；状态 {(m.get('context') or {}).get('state', '--')}", "evidence_id": M},
    ]

    # --- universe: required names, then everything strong enough to deserve a look (C)
    records = {r["symbol"]: r for r in closure.get("records", [])}
    core = HC.build(day, rows, facts, mapping, records, observation_inputs)
    core_by = {r["symbol"]: r for r in core["rows"]}
    HC_ID = src.add("HC:observations", "热点核心与当日结构观察（无交易权限）", "LOCAL_CALCULATION", "", HC.VERSION, now_text,
                    json.dumps({"scope": core["scope"], "ranking": core["ranking"], "mapping_basis": core["mapping_basis"],
                                "membership_evidence": membership, "mapping_review": core["mapping_review"],
                                "sources": {k: v.get("version") for k, v in observation_inputs.items()}, "gaps": observation_gaps}, ensure_ascii=False))
    jev_rows = {r["symbol"]: r for r in jev.get("rows", [])} if jev else {}
    plans = auto_plans(books, store.ROOT)
    auto = books.get("auto") or {}
    placed = {p["symbol"]: p for p in auto.get("placed", [])}
    fills_by = {e["symbol"]: e for e in auto.get("executions", [])}
    bell = {x["symbol"]: x for x in facts["lists"].get("bellwethers", [])[:MAX_WATCH]}
    cands = {x["symbol"]: x for x in facts["lists"].get("continuing_candidates", [])[:MAX_WATCH]}
    names = {**{sym: v.get("name") for sym, v in mapping.items()}, **{r["symbol"]: r.get("name") for r in closure.get("records", [])},
             **{sym: x.get("name") for sym, x in {**bell, **cands}.items()}, **{x["symbol"]: x.get("name") for x in tracking.get("pinned", [])}}
    rd = readings(facts, names)
    today_bars = {}
    for d in prior_drafts:
        f = (rows.get(d["symbol"]) or {}).get("frame")
        if f is not None and not f.empty and f.iloc[-1].date == day and bool(f.iloc[-1].valid):
            today_bars[d["symbol"]] = {"open": float(f.iloc[-1].open), "close": float(f.iloc[-1].close)}
    draft_check = entry_contracts.forward_check(prior_drafts, today_bars, day)
    pinned = [x["symbol"] for x in tracking.get("pinned", [])]
    previous_plans = [x["symbol"] for x in (previous or {}).get("stocks", []) if x.get("plan")] if previous else []
    required = list(dict.fromkeys(pinned + list(placed) + [r["symbol"] for r in facts["lists"].get("review_eligible", [])]
                                  + [sym for sym, p in plans.items() if p["open_position"] > 0] + previous_plans))
    leads = theme_leads(facts)
    amount_of = lambda sym: float(rows[sym]["frame"].iloc[-1].amount) if sym in rows and not rows[sym]["frame"].empty else 0.0
    universe = list(dict.fromkeys(
        required + list(bell) + list(cands) + list(core_by)
        + [r["symbol"] for r in records.values() if r.get("strength_rank") and r["strength_rank"] <= POOL["strength_rank"]]
        + [sym for sym, h in sorted(l["heights"].items(), key=lambda kv: -kv[1]) if h >= POOL["ladder_min_height"]]
        + sorted(b["U"], key=amount_of, reverse=True)[:POOL["limit_amount_top"]]))
    scored = {sym: opportunity(sym, rows.get(sym), facts, mapping, leads) if sym in rows and rows[sym].get("in_domain", True) else
              {"score": None, "parts": [], "text": "不在可核验域或无当日行情"} for sym in universe}
    pool = [sym for sym, o in sorted(scored.items(), key=lambda kv: (-(kv[1]["score"] if kv[1]["score"] is not None else -1), kv[0])) if o["score"] is not None][:POOL["top"]]
    rank = {sym: i + 1 for i, sym in enumerate(pool)}
    # Shadow preparation watch (low-extension names outside the engine's candidate domain): a few enter as cards
    # so the reader sees what is coiling before it moves; they carry no plan and no permission.
    watch_path = folder / "preparation_watch.csv"
    prep_watch = []
    if watch_path.exists():
        import csv
        with open(watch_path, encoding="utf-8-sig", newline="") as fh:
            prep_watch = [w["symbol"] for w in csv.DictReader(fh) if w.get("symbol") in rows][:5]
    for sym in prep_watch:
        scored.setdefault(sym, opportunity(sym, rows.get(sym), facts, mapping, leads))
    chosen = list(dict.fromkeys(required + pool + list(bell) + list(cands)[:4] + prep_watch + core["focus"] + core["early_setups"]))
    admission_symbols = list(dict.fromkeys(chosen + [sym for sym, r in records.items() if r.get("paper_plan_eligible")]))
    admission = ER.build(day, closure, mapping, admission_symbols, now_text)
    admission["theme_history"] = ER.cooling(facts, manifest, out_root, now_text)
    relay = core["relay"] = HR.build(day, out_root, facts, core, admission, now_text)
    funnel = core["funnel"] = HF.build(day, now_text, facts, core, observation_inputs.get("attribution", {}), admission, relay, records)
    prior_days = [d for d in sessions if d < day][-FW.LOOKBACK:]
    preparation = FW.load_pools(out_root, prior_days + [day], now_text) if live else {}
    watch = funnel["first_board_watch"] = FW.build(day, target, prior_days, preparation, core, funnel)
    p5 = funnel["p5_limit_up"] = facts.get("p5_limit_up", {})
    p5_feedback = funnel["p5_limit_up_feedback"] = facts.get("p5_limit_up_feedback", [])
    previous_core = R.read_json(Path(relay["prior_source"])) if relay["prior_source"] else {}
    feedback = funnel["first_board_feedback"] = FW.followup(previous_core, facts, rows) if live else []
    p5_note = (f"P5先入池后首板重点 {p5['focus_count']} 只，封板全量对账，无题材或名次筛选。" if p5 else "此版事实包无P5与收盘封板交集。")
    if p5.get("unverified_dates"):
        p5_note += " 事实层P5档案缺口：" + "、".join(p5["unverified_dates"]) + "。"
    watch_note = p5_note + f"严格首板短线观察 {len(watch['picks'])} 只，只跟踪下一交易日。不追高或一字，不改变市场门与交易许可。"
    if watch["unverified_dates"]:
        watch_note += " 准备档案未通过时点核验：" + "、".join(watch["unverified_dates"]) + "。"
    if not funnel["attribution_available_at"]:
        watch_note += " 同题材封板归因未核实。"
    relay_id = src.add("HR:relay", "埋伏与热点接力核验（观察）", "LOCAL_CALCULATION",
                       relay["prior_source"] or "", HR.VERSION, now_text,
                       json.dumps({k: relay[k] for k in ("previous_day", "prior_themes_comparable", "prior_watch_count",
                                                         "watch_hits_today", "attributed", "limit_count", "chains")}, ensure_ascii=False))
    admission_by = {r["symbol"]: r for r in admission["rows"]}
    ER_ID = src.add("ER:admission", "入场复核与题材风险（不改冻结排序）", "LOCAL_CALCULATION", TR.CONFIG, ER.VERSION, now_text,
                    json.dumps(admission, ensure_ascii=False))
    chosen = list(dict.fromkeys(chosen + admission["top"]))
    stocks = []
    for sym in chosen:
        role = bell.get(sym)
        record = records.get(sym) or (cands.get(sym) if sym in cands else None)
        stocks.append(stock_card(sym, names.get(sym), rows.get(sym), record, placed.get(sym), plans.get(sym), fills_by.get(sym), jev_rows.get(sym),
                                 tracking, day, target, live, src, [x for x in (C, BK, M, O) if x], pinned=sym in pinned, role=role,
                                 opp=scored.get(sym), rank=rank.get(sym), shadow=shadow_rows.get(sym), sessions=sessions, drafts=drafts_by.get(sym), observation=core_by.get(sym)))
    for st in stocks:
        risk = admission_by[st["symbol"]]["theme_risk"]
        if risk["theme_risk_action"] != "ALLOW":
            st["missing"].append(risk["theme_risk_reason"])
            st["support_ids"].append(ER_ID)
            st["next_check"] = "先核实题材风险与修复：" + risk["theme_risk_reason"]
            if st["plan"]:
                st["plan"], st["status"] = None, "WATCH"
        if st["symbol"] in prep_watch:
            st["role"] = (st["role"] + "｜" if st["role"] else "") + "准备候选（影子）"
    stocks.sort(key=lambda st: (-(st["_score"] if st["_score"] is not None else -1), st["symbol"]))
    code, decision = verdict(stocks)
    code, head, hold = s4_verdict(s4, code)
    if s4:
        decision = head + "；以下为观察榜（排序未验证）：" + decision
    decision = f"入场复核 {admission['eligible_count']} 只（最多展示2只，不强制交易）；" + decision
    if not admission["eligible_count"] and code == "✅":
        code = "🟡"
    issues, requests = [], []
    if observation_gaps or core["unavailable"]:
        issues.append({"id": "Q_OBSERVATIONS", "severity": "WARN", "title": "当日观察输入缺口",
                       "detail": "；".join(observation_gaps + (["无可核验行情：" + "、".join(core["unavailable"])] if core["unavailable"] else [])),
                       "impact": "未读取旧日期或本轮失败的扫描；热点观察不完整", "evidence_ids": [HC_ID]})
    if membership["errors"]:
        issues.append({"id": "Q_MEMBERSHIP", "severity": "WARN", "title": "题材关联证据缺口",
                       "detail": "；".join(membership["errors"]), "impact": "保留原映射，不推断新题材", "evidence_ids": [HC_ID]})
    if core["mapping_review"]:
        requests.append({"id": "REQ_MEMBERSHIP", "priority": "P1", "symbols": [r["symbol"] for r in core["mapping_review"][:8]],
                         "purpose": "核对连板核心的当日主炒题材与成员标签，不能只按行业归类；叙事与经营证据分开",
                         "fields": ["涨停归因原文与来源", "公司公告", "题材关联获知/生效时间"], "max_rows": 8, "cutoff": now_text})
    for st in stocks:
        if st["status"] == "DATA_GAP":
            issues.append({"id": f"Q_GAP_{st['symbol']}", "severity": "WARN", "title": f"{st['name']} 缺当日可核验行情",
                           "detail": "本地日线中没有当日有效 K 线；不能补造数字。", "impact": "该股只列缺口，不做任何判断", "evidence_ids": st["support_ids"]})
    no_engine = [st["symbol"] for st in stocks if st["pinned"] and not records.get(st["symbol"])]
    if no_engine:
        requests.append({"id": "REQ1", "priority": "P0", "symbols": no_engine,
                         "purpose": "指定跟踪股不在引擎候选域：需要原计划、模式与关键价，才能与自动账计划同标准比较",
                         "fields": ["60日OHLCV", "原计划/触发/失效/有效期", "结构事件与关键价算法"], "max_rows": 120, "cutoff": now_text})
    strong = [st["symbol"] for st in stocks if st["mode"] == "STRONG_CONFIRM" and st["status"] != "DATA_GAP" and st["_score"] is not None and st["_score"] >= POOL["some"]]
    if strong:
        requests.append({"id": "REQ2", "priority": "P1", "symbols": strong[:8],
                         "purpose": "强势确认模式只有日线无法核验封板/开板/回封先后与可成交性；这些是机会池里分数最高的强势股",
                         "fields": ["分钟K线", "盘口封单快照", "事件前后窗口", "缺失与精度声明"], "max_rows": 120, "cutoff": now_text})

    # --- sections 01–07
    sec = {x["id"]: x for x in STATIC["sections"]}
    s4_rows = [[f"S4 #{r['rank']}", f"{r['name']}（{r['symbol']}）",
                f"S4 回踩再启动｜涨 {r['pct']:+.1f}%，偏离 EMA20 {r['bias_ema20']:+.1f}%，距 60 日高 {r['dd60']:.1f}%，行业内强度 "
                + ("—" if r.get("lead") is None else f"{r['lead']:.0%}")
                + f"；收盘位置 {r.get('clv', '未知')}" + ("（收在日内中点下方，承接反证，非硬否决）" if r.get("clv") is not None and r["clv"] < .5 else ""),
                f"形态候选，但{hold}，不建议新开" if hold else "形态候选（非入场许可）", "S4 · 分批止盈",
                f"止损 {r['plan']['stop']}；止盈 {r['plan']['tp1']} / {r['plan']['tp2']}" + (f"；{r['plan']['denial']}" if r["plan"].get("denial") else ""),
                 "等待市场修复" if hold else f"先复核题材与价格，再手工选择 {r['symbol'][3:]}"] for r in s4.get("recommended", [])]
    entry_rows = [[f"入场复核 #{i}", sym, "沿用冻结参与排序", "条件复核通过，非成交许可", "趋势回踩",
                   "仍须成交价与账户风险复核", "按原计划检查，不追高"] for i, sym in enumerate(admission["top"], 1)]
    if not entry_rows:
        entry_rows = [["入场复核", "暂无", "市场、结构、时效与题材风险分开核验", "等待", "—", "不以强度分替代资格", "不补位凑数"]]
    pool_rows = [[f"观察 #{st['_rank']}" if st["_rank"] else "—", f"{st['name']}（{st['symbol']}）", f"{st['_score'] if st['_score'] is not None else '--'} 分：{st['opportunity_reason']}",
                  STATUS_LABEL[st["status"]] + (f"｜{st['role']}" if st["role"] else ""), MODE_LABEL[st["mode"]], "；".join(st["missing"]) or "无", st["next_check"]] for st in stocks]
    core_rows = sorted(core["rows"], key=lambda r: (r["symbol"] not in core["focus"], r["symbol"] not in core["early_setups"]))
    executable = [st for st in stocks if st["plan"]]
    sections = [
        _section(sec["01"], f"昨日涨停池｜{y.get('n_valid')} 只可核验（母池 {y.get('n_pool')}，{y.get('basis')}）", ["观察口径", "均值", "中位数", "正值/样本"],
                 [[lab, _pct(d.get("mean")), _pct(d.get("median")), f"{d.get('up')}/{d.get('n')}"] for lab, d in
                  (("开盘对昨收", y["open_vs_pre"]), ("收盘对昨收", y["close_vs_pre"]), ("收盘对开盘", y["close_vs_open"]), ("日内最高对昨收（事后）", y["high_vs_pre"]))],
                 [M, H], "开盘溢价是持有者体验；日内最高是事后幅度，都不是新开仓可实现收益。", "C01"),
        _section(sec["02"], "梯队晋级与算术对账", ["昨日高度", "原母池", "继续封板", "断板", "晋级率"],
                 [[f"{h} 板", p.get("pool"), p.get("numerator"), len(p.get("broke", [])) if isinstance(p.get("broke"), list) else p.get("broke_n"), _ratio(p)]
                  for h, p in sorted(l["promotion_from_previous"].items(), key=lambda kv: int(kv[0]))],
                 [M], f"今日高度分布：{'，'.join(f'{h} 板 {n} 只' for h, n in l['by_height'].items())}；最高 {l['max_height']} 板。连板按市场日历连续计数。", "C02"),
        _section(sec["03"], "题材｜按完整成员表统计（跨题材重叠不可相加）", ["题材", "去重成员", "上涨", "收盘涨停", "广度", "成交额占比"],
                 [[x["theme"], x["members"], x["up"], x["limit_up"], _ratio(x["breadth"]), _ratio(x["amount_share"])] for x in t["top_by_limit_up"][:8] + t.get("chains", [])],
                 [M], "成员表为当前映射版本，没有历史时点版本；成交额占比是活跃度，不是净买入。机会池题材加分只给涨停榜与占比榜前五的成员。", "C03"),
        _section({**sec["04"], "title": "热点核心与结构观察", "subtitle": "核心强度与买入许可分离；首板不必已创20日新高"},
                 f"完整观察并集 {len(core['rows'])} 只｜角色焦点 {len(core['focus'])} 只（非推荐）",
                 ["观察层", "股票", "热点归属（可重叠）", "角色 / 场景", "涨幅", "成交额 / 亿", "生命周期｜原引擎状态", "风险 / 许可"],
                 list(FW.limit_up_table_rows(p5, p5_feedback)) + list(FW.table_rows(watch, feedback))
                 + list(HF.table_rows(funnel)) + [["角色焦点" if r["symbol"] in core["focus"] else "早期S5" if r["symbol"] in core["early_setups"] else "全池保留",
                   f"{r['name']}（{r['symbol']}）", "、".join(r["themes"]) or "未匹配前五热点", "、".join(r["roles"]),
                    _pct(r["change_pct"]), r["amount_yi"], r["lifecycle"] + "｜" + r["engine_state"],
                    (f"EMA20偏离 {r['ema20_bias_pct']:+.2f}%；" if r.get("ema20_bias_pct") is not None else "")
                    + ("；".join(r["risks"]) or r["permission"])] for r in core_rows]
                 + [["原风向标", f"{st['name']}（{st['symbol']}）", "不在热点并集", st["role"], _pct(st["metrics"]["change_pct"]),
                     _num(st["metrics"]["amount_yi"], 1), STATUS_LABEL[st["status"]], "；".join(st["missing"]) or "仅观察"]
                     for st in stocks if st.get("_bellwether") and st["symbol"] not in core_by]
                  + [["热点接力", members, theme, state, "--", "--", feedback, "只观察，不改变许可"]
                     for theme, state, members, feedback in HR.lines(relay)],
                  [M, HC_ID, relay_id], watch_note + core["ranking"] + "。" + core["mapping_basis"]
                  + f"。交叉归因 {relay['attributed']}/{relay['limit_count']} 只涨停；昨收合格埋伏池与今日封板交集："
                  + ("不可比" if relay["watch_hits_today"] is None else "、".join(relay["watch_hits_today"]) or "无")
                  + f"。紧链角色榜 {len(funnel['role_watch'])} 只；正式参与榜 {len(funnel['participation_top'])} 只（不补位）。"
                  + "逐股漏斗见 hotspot_funnel.json；接力只比较已核实紧链，无归因不等于退潮。", "C04"),
        _section(sec["05"], "风险事件｜同一股票可重复命中不同事件", ["事件", "数量", "板块分布", "位置分布", "示例"],
                 [[R.LABELS["loss"].get(k, k), v["count"], _pairs(v.get("by_board")), _pairs(v.get("by_position")), "、".join(v.get("symbols", [])[:5])]
                  for k, v in facts["loss_effects"]["events"].items()],
                 [M], "价格大跌不能单独证明主力出货或业绩原因。高开低走与跌破 20 日低点在机会池各扣 15 分。", "C05"),
        _section({**sec["06"], "title": "入场复核、S4形态与强度观察", "subtitle": "先检查许可；强度排名不是买入价值排名"},
                 f"入场复核 {admission['eligible_count']} 只｜S4形态候选 {len(s4_rows)} 只｜强度观察前 {len(pool)}，排序未验证；必保留 {len(required)} 只跟踪对象",
                 ["优先级", "股票", "机会依据", "状态｜角色", "模式", "缺口 / 关键价", "下一步"],
                 entry_rows + s4_rows + pool_rows, [x for x in (ER_ID, S4, M, O, C, BK, T) if x],
                 "S4按研究口径筛出形态并带原离场合约，不等于可以买入；强度分不是买入价值、收益预测或交易许可。", "C06"),
        _section(sec["07"], "三分支计划｜不赋予未经校准的概率", ["分支", "观察证据", "允许的分析动作", "不可做"], STATIC["section07_rows"],
                 [K] + ([BK] if BK else []), f"具体条件计划见各股卡片；今日自动账计划 {len(executable)} 条，人工不加单。", "C07"),
    ]

    cooling_rows = [r for r in admission["theme_history"]["rows"] if r["status"] == "COOLING"]
    if cooling_rows:
        issues.append({"id": "Q_THEME_COOLING", "severity": "WARN", "title": "题材绝对热度连续降温提示",
                       "detail": "；".join(r["theme"] + "：" + " → ".join(f"{p['date']} 涨停{p['limit_up']} / 上涨{p['breadth']:.1%}" for p in r["points"]) for r in cooling_rows),
                       "impact": admission["theme_history"]["note"], "evidence_ids": [ER_ID]})

    # --- claims: readings by rule, opportunity first (B)
    top_lines = [f"#{st['_rank']} {st['name']} {st['_score']} 分（{STATUS_LABEL[st['status']]}，{MODE_LABEL[st['mode']]}）：{st['opportunity_reason']}" for st in stocks if st["_rank"]][:5]
    claims = [
        _claim("C06", decision, (top_lines or ["机会池为空"]) + ([f"发现完整性：候选 {register['summary']['candidates']}、准备观察 {register['summary']['watch']}、重叠 {register['summary']['overlap']}、域外 {register['summary']['outside']}、并集 {register['summary']['union']}；今日新增 {register['summary']['new_today']}、重激活 {register['summary']['reactivated']}、失活 {register['summary']['deactivated']}；状态 {register['summary']['by_state']}"] if register else ["发现完整性：本日无发现事件登记"]),
               f"机会池候选 {len(universe)} 只：必保留、风向标、续踪候选、强度榜、连板、涨停成交额榜，以及热点核心/强势/S1至S5当日并集；取前 {POOL['top']}。",
               f"可执行计划 {len(executable)} 条（只有趋势回踩模式能出计划）；强势股缺分钟/盘口过程，机会保留、执行不过门槛。",
               [f"机会分是当日强度与承接的机械加总，不是明日收益；{rd['risk_read']}。"] + [f"{st['name']}：{st['contrary']}" for st in stocks[:3] if st["contrary"] != "无额外反证记录"][:2],
               f"优先复核机会池前 {POOL['focus']}：{'、'.join(st['name'] for st in stocks if st['_rank'] and st['_rank'] <= POOL['focus']) or '无'}；可执行计划 {len(executable)} 条，其余机会等缺口补齐后再谈执行。",
               f"{target} 收盘后按机会池前 {POOL['focus']} 逐一核对：延续（新高或再封板）、分化（回落但守住 20 日线）、转弱（破位）。",
               [x for x in (M, O, C, BK, T) if x], "INTERPRETATION"),
        _claim("C01", f"接力情绪{rd['env']}", [f"母池 {y.get('n_pool')} 只，可核验 {y.get('n_valid')} 只；收盘对昨收均值 {_pct(y['close_vs_pre'].get('mean'))}，中位 {_pct(y['close_vs_pre'].get('median'))}，收红 {y['close_vs_pre'].get('up')}/{y['close_vs_pre'].get('n')}。",
                                                f"开盘对昨收均值 {_pct(y['open_vs_pre'].get('mean'))}；收盘对开盘均值 {_pct(y['close_vs_open'].get('mean'))}。"],
               rd["env_text"], "这是相对近 20 日的读法，不是绝对阈值，也不是预测。", [f"昨日池收绿 {rd['neg']} 只。", "日内最高是事后幅度，不是可实现收益。"] + _gap_lines(manifest, "母池"),
               rd["env_conclusion"], "按新冻结母池，明日分别核对开盘溢价与开盘后保留程度。", [M, H], "INTERPRETATION"),
        _claim("C05", rd["risk_read"].split("：")[0], [rd["risk_text"]], _reading(s, "seal_failure_rate") + "；" + _reading(s, "limit_down"),
               "炸板率与跌停数方向取反后读热冷；亏钱效应看断板、高开低走与破位是否向多个方向扩散。", ["价格大跌不能单独证明原因。"],
               rd["risk_read"] + "。", "明日核对亏钱效应是否扩散到多个方向。", [M], "INTERPRETATION"),
        _claim("C02", rd["ladder_read"], [rd["ladder_text"]], _reading(s, "promotion_1_to_2") + "；" + _reading(s, "max_height"),
               "晋级率看接力意愿，高度看空间；两者同热才是拔高。", ["晋级率高不等于明日仍高；断板股的亏损效应另看。"],
               (f"最高板 {'、'.join(rd['leaders'])} 为明日核对对象。" if rd["leaders"] else "无 3 板以上股票。"), "明日核对最高板是否晋级与断板修复。", [M], "INTERPRETATION"),
        _claim("C03", "题材集中度", [rd["theme_text"]], "题材成员表无历史版本，不做跨日比较。", "涨停榜看资金集中在哪，占比榜看小范围一致性；两榜前五成员在机会池加 10 分。",
               ["跨概念重叠使占比不可相加；大概念天然占优，不能单独作为方向证据。"], "方向证据看题材内是否有 2 板以上与趋势核心同时出现。",
               "明日核对前列题材是否仍有新涨停与广度。", [M], "INTERPRETATION"),
        _claim("C04", "热点核心与买入资格分开看", [f"当日观察并集 {len(core['rows'])} 只；热点U/Z、强势扫描、S1至S5互相去重；角色焦点 {len(core['focus'])} 只；S5早期结构 {len(core['early_setups'])} 只。"],
               core["ranking"], "不再以是否进入三路径或是否突破20日高点，决定热点首板能否被看见。", [core["mapping_basis"], "强势与过热可以同时成立；日线不能证明资金身份或谁带动谁。"],
               "先复核热点角色，再核验当下结构、价格和原风控；观察榜不创建任何交易许可。", "明日按角色核对延续、分化、转弱及成交可达性。", [M, HC_ID], "INTERPRETATION"),
        _claim("C07", f"{target} 计划", [f"自动账计划 {len(executable)} 条按下一开盘固定上限模型执行；人工账不加单。",
                                       "未执行原因汇总：" + ("；".join(f"{k} {v}" for k, v in _not_executed(stocks).items()) or "无"),
                                       (f"上一交易日 B1/B3 影子草案次日核对：{draft_check['checked']} 条，" + "，".join(f"{k} {v}" for k, v in sorted(draft_check["counts"].items()))
                                        + f"；带内开盘 {draft_check['in_band']} 条中收盘守住失效位 {draft_check['in_band_close_held']} 条（合同读法，不是收益）") if draft_check["checked"] else "上一交易日无 B1/B3 影子草案可核对",
                                       ("影子市场政策：" + "，".join(f"{e} {market_policy['policy'][e]['shadow']}" for e in ("E1", "E2", "E3") if e in market_policy.get("policy", {}))
                                        + f"；冻结闸门 {market_policy.get('policy', {}).get('frozen_gate', '--')}（仍决定一切纸面许可）") if market_policy else "影子市场政策：无"],
               "三分支只列证据与允许动作，不赋予概率。",
               "延续：核验已批准模式触发；分化：更新同目标比较；转弱：撤销失效预案，按持仓合约处理。", ["计划完整与盘中触发是两件事；无实时报价不能标已触发。"],
               f"{code} 结论沿用 C06；{target} 收盘后按冻结计划核对兑现与误判。", f"{target} 收盘后核对。", [K] + ([BK] if BK else []), "INTERPRETATION"),
    ]

    for i, g in enumerate(manifest.get("gaps", []), 1):
        issues.append({"id": f"Q{i:02d}", "severity": "WARN", "title": "事实包缺口", "detail": g, "impact": "相关数字按缺口读", "evidence_ids": [M]})
    for k, v in b.get("reconciliation", {}).items():
        if v.get("status") not in ("RECONCILED", None):
            issues.append({"id": f"Q_RECON_{k}", "severity": "WARN", "title": f"{R.LABELS['kind'].get(k, k)} 与参考名单对账未完成",
                           "detail": f"本地 {v.get('local')} / 参考 {v.get('reference')} / 两边都有 {v.get('both')}", "impact": "差异逐股核对，不自动改数", "evidence_ids": [M]})
    issues.append({"id": "Q_NO_AI", "severity": "WARN", "title": "AI 解读未合并", "detail": "结论为本地规则读法；模型解读可再合并为新版本。", "impact": "解释层为规则生成", "evidence_ids": [M]})

    audit = _audit(previous, stocks, rows, auto, day)
    comparisons = []
    if len(required) >= 2:
        missing = sorted({mm for st in stocks if st["symbol"] in required for mm in st["missing"]})
        comparisons.append({"id": "CMP1", "title": "必保留股同标准比较", "symbols": required[:4], "objective": tracking.get("objective", "同目标、同期限、同预算比较"),
                            "matched": False, "conclusion": "机会分可比，执行证据未齐；容量差异不替代短线机会判断。", "support_ids": [x for x in (M, O, C, T) if x], "missing": missing or ["同期限持有合约"]})
    focus = [st["id"] for st in stocks if st["status"] != "DATA_GAP" and st["_score"] is not None][:POOL["focus"]] or [st["id"] for st in stocks[:POOL["focus"]]]
    engine_status = {mm: spec.get("status", "") for mm, spec in engine_modes.OBSERVATION_ONLY.items()}
    modes_out = [{**md, "status": "纸面可执行（趋势回踩）" if md["id"] == "TREND_PULLBACK" else f"仅观察：{engine_status.get('STRENGTH_CONFIRM' if md['id'] == 'STRONG_CONFIRM' else md['id'], md['status'])}"}
                 for md in STATIC["modes"]]
    report = {
        "schema_version": SCHEMA_VERSION,
        "meta": {"report_id": f"fupan-{day}", "title": f"每日交易复盘 · {day}", "trade_date": day, "target_date": target,
                 "market_as_of": f"{day}T15:00:00+08:00", "knowledge_as_of": now_text, "rendered_at": now_text,
                 "frozen_at": now_text if live else None, "timezone": "Asia/Shanghai", "mode": mode, "forward_frozen": live,
                 "source_status": f"eod_review {manifest.get('stage')} / closure {closure.get('observation_basis', 'MISSING')}",
                 "quote_status": "EOD_BARS_ONLY", "order_enabled": False,
                 "objective": tracking.get("objective", "捕捉与既定模式匹配的短线机会；先识别机会，再核验可执行性。"),
                 "horizon": tracking.get("horizon", "各候选使用原策略合约；未提供的持有期限不补造。"),
                 "allowed_modes": ["STRONG_CONFIRM", "TREND_PULLBACK", "OVERNIGHT"],
                 "rule_version": f"{closure.get('version', '')} / {(closure.get('rules') or {}).get('validation', '')} / {VERSION}",
                 "mapping_version": (manifest.get("sources", {}).get("concept_mapping") or {}).get("generated", "未知"),
                 "scope": f"{facts['domain']['label']}；{closure.get('scope', '')}",
                 "banner": f"{code} {'前向冻结' if live else '事后重建（非前向样本）'} · {day} 收盘后 · 接力情绪{rd['env']} · 不下单 · 三账分离",
                 "portfolio_status": "VERIFIED_SNAPSHOT" if auto.get("ledger_touched") else "NOT_PROVIDED", "required_symbols": required},
        "summary": {"claim_ids": ["C06", "C01", "C05", "C02", "C03", "C04", "C07"], "focus_ids": focus, "focus_title": f"强度观察 · 前 {len(focus)}（不是买入价值榜）",
                    "focus_note": f"入场复核 {admission['eligible_count']} 只，单列在第六节。此处沿用原机械分，仅表示优先观察，不是收益概率或交易许可。",
                    "risk": f"{rd['risk_read']}。炸板 {b['counts']['Z']}、跌停 {b['counts']['D']}、昨日池收绿 {rd['neg']} 只；缺口 {len(manifest.get('gaps', []))} 项。",
                    "decision": decision + "。本报告不下单；自动账按冻结规则执行，人工账不加单。"},
        "metrics": metrics, "history": hist, "claims": claims, "sections": sections,
        "stocks": [{k: v for k, v in st.items() if not k.startswith("_")} for st in stocks],
        "modes": modes_out, "issues": issues, "sources": src.items, "spec": STATIC["spec"], "comparisons": comparisons,
        "scenarios": STATIC["scenarios"], "audit": audit, "requests": requests,
    }
    merged = merge_analysis(report, analysis) if analysis else None
    problems = validate(report)
    if problems:
        raise ValueError("报告不满足契约: " + "; ".join(problems[:8]))
    inputs = {"fupan": VERSION, "eod_review": pointer.get("input_hash"), "closure": closure.get("snapshot_id"),
              "books": R.file_sha(folder / "books_daily.json") if (folder / "books_daily.json").exists() else "",
              "jev": R.file_sha(folder / "jev_shadow.json") if (folder / "jev_shadow.json").exists() else "",
              "tracking": R.file_sha(tracking_path) if Path(tracking_path).exists() else "", "analysis": _digest(analysis) if analysis else "",
              "mode": mode, "code": R.file_sha(__file__), "hotspot_code": R.file_sha(HC.__file__),
              "relay_code": R.file_sha(HR.__file__), "funnel_code": R.file_sha(HF.__file__),
              "first_board_watch_code": R.file_sha(FW.__file__),
              "renderer": {name: R.file_sha(ASSETS / name) for name in ("report_template.html", "report.js", "report.css", "report.schema.json")},
              "observations": _digest(observation_inputs), "observation_gaps": observation_gaps, "hotspot_core": _digest(core),
              "membership_evidence": _digest(membership), "mapping_resolved": _digest(mapping), "membership_code": R.file_sha(TS.__file__),
              "entry_review": _digest({k: v for k, v in admission.items() if k != "known_at"}), "entry_code": R.file_sha(ER.__file__), "theme_risk_code": R.file_sha(TR.__file__),
              "theme_risk_config": R.file_sha(TR.CONFIG) if TR.CONFIG.exists() else ""}
    return report, {"input_hash": _digest(inputs), "inputs": inputs, "merged": merged, "live": live, "verdict": code, "draft_check": draft_check, "hotspot_core": core, "entry_review": admission}


def stock_card(symbol, name, row, record, placed, plan, execution, jev_row, tracking, day, target, live, src, base_ids, pinned=False, role=None, opp=None, rank=None, shadow=None, sessions=None, drafts=None, observation=None):
    metrics = stock_metrics(row)
    opp = opp or {"score": None, "parts": [], "text": "未评分"}
    path = Path("data") / f"{symbol}_daily.csv"
    tail = row["frame"].tail(5)[["date", "open", "high", "low", "close", "amount"]].to_string(index=False) if row is not None and not row["frame"].empty else "无当日行情"
    sid = src.add(f"S:{symbol}", f"{name or symbol} 日线（最近 5 根）", "USER_DATA", path, "date/open/high/low/close/amount", _mtime(path), tail)
    support = [x for x in base_ids if x] + [sid]
    pin = next((x for x in tracking.get("pinned", []) if x["symbol"] == symbol), None)
    height = any(label.endswith(" 板") for label, _ in opp["parts"])
    record_mode = MODE_OF.get((record or {}).get("mode") or "", None)
    # A real plan keeps its contracted mode. Without one, do not mislabel a board leader as a pullback entry.
    mode = ((record_mode if plan else None) or (pin or {}).get("mode")
            or ("STRONG_CONFIRM" if height else None) or ROLE_MODE.get((role or {}).get("role", ""))
            or record_mode or ("STRONG_CONFIRM" if (metrics["change_pct"] or 0) >= 9 else "TREND_PULLBACK"))
    missing, notes = [], []
    if observation:
        support.append("HC:observations")
        notes.append("热点归属：" + ("、".join(observation["themes"]) or "未匹配前五热点") + "（当前映射，非上涨归因）")
        notes.extend(observation["risks"])
    if MODE_MISSING.get(mode):
        missing.append(MODE_MISSING[mode])
    if record and record.get("blockers"):
        blockers = "、".join(BLOCKER_LABEL.get(str(x), str(x)) for x in record["blockers"])
        notes.append("阻断：" + blockers)
        missing.append("引擎阻断：" + blockers)
    if jev_row:
        fl = jev_row.get("flags", {})
        notes.append(f"Jev 影子：质量 {fl.get('setup_tier')}，阶段 {fl.get('entry_stage')}（不触发状态变化）")
    if shadow:
        # The layered shadow order answers "worth the next unit of review?", which the attention score above does not;
        # both stay visible, neither is a permission.
        notes.append(f"分层影子 #{shadow.get('shadow_rank')}：{shadow.get('reading', '')}")
    for d in drafts or []:
        notes.append(f"{d['contract']} {d.get('note', '')[:12]}影子草案：" + ("齐备（无批准）" if d.get("ready") else "缺 " + "、".join(d.get("missing") or [])))
    if metrics["change_pct"] is None:
        status = "DATA_GAP"
        missing.append("当日可核验行情")
    elif plan and plan.get("order"):
        effective = next_session(sessions or [], plan.get("entry_day") or plan["payload"].get("entry_day") or day)
        if plan.get("open_position"):
            status = "WATCH"
            notes.append(f"已成交持仓（{plan.get('account_id', 'AUTO100W')}）：按持仓合约处理，不是新买点")
        elif effective < target:
            status = "EXPIRED"
            notes.append(f"计划生效日 {effective} 已过且未成交：过期，不因新报告续命")
        elif effective == target and live:
            status = "PLAN_READY"
        else:
            status = "WAIT_TRIGGER"
            notes.append("事后重建或未到生效日：已批准待触发，不是当前买入许可" if not live else f"计划生效日 {effective} 晚于目标日")
    elif record and record.get("state") in STATE_STATUS:
        status = STATE_STATUS[record["state"]]
    else:
        status = "WATCH"
        if pin and not record:
            missing.append("原计划、触发价与失效价未定义")
    if status in ("WATCH", "WAIT_SETUP") and not record and not pin:
        missing.append("不在引擎候选域：无冻结的触发价与失效价")
    plan_obj = None
    if plan and plan.get("order") and status in ("PLAN_READY", "WAIT_TRIGGER", "EXPIRED", "WATCH") and metrics["change_pct"] is not None:
        effective = next_session(sessions or [], plan.get("entry_day") or plan["payload"].get("entry_day") or day)
        plan_obj = plan_from_paper(plan["payload"], plan["order"], plan["fills"], day, effective, support, (record or {}).get("blockers", []),
                                   plan.get("account_id", "AUTO100W"))
        if status == "PLAN_READY" and missing:
            status = "WAIT_TRIGGER"  # a plan with an open evidence gap is not "ready" under the contract
        if status == "PLAN_READY" and plan_obj["trigger_state"] == "FAILED":
            status = "EXPIRED"
    role_text = (role or {}).get("role_label") or ("指定跟踪" if pin else f"候选（参与榜 #{record.get('participation_rank')}）" if record and record.get("participation_rank") else "")
    if observation:
        role_text = "｜".join(filter(None, [role_text, "、".join(observation["roles"])]))
    if placed:
        role_text = (role_text + "｜" if role_text else "") + "自动账 " + placed.get("status", "")
    score = opp["score"]
    label = (f"机会 #{rank} · {score} 分" if rank else f"未入前 {POOL['top']} · {score} 分" if score is not None else "未评分")
    return {"id": symbol, "symbol": symbol, "name": name or symbol, "pinned": bool(pin) or pinned, "role": role_text, "role_verified": False, "mode": mode,
            "model_version": (plan or {}).get("payload", {}).get("ranking_version") if plan else None,
            "opportunity_label": label, "opportunity_reason": opp["text"],
            "status": status, "horizon": "首仓后最长 20 个持有交易日，条件满足可续至 60 日" if plan_obj else tracking.get("horizon", "待原策略合约提供"),
            "metrics": metrics, "support_ids": support, "contrary": "；".join(notes) if notes else "无额外反证记录",
            "missing": missing, "next_check": (record or {}).get("next_condition") or (role or {}).get("question") or "明日核对开盘承接与结构是否保持",
            "invalidating": f"收盘跌破 {plan_obj['invalidation_price']:.2f}" if plan_obj else "结构失效位未定义" if not record else f"失效 {_num((record.get('window') or {}).get('invalid'))}",
            "prior_plan_id": (pin or {}).get("plan_id"), "plan": plan_obj, "review_deadline": f"{target} 收盘后",
            "notes": ("；".join(notes) + ("；" if notes else "") + ("成交 " + str(execution.get("filled")) + "/" + str(execution.get("ordered")) if execution else "")).strip("；"),
            "_bellwether": role is not None, "_score": score, "_rank": rank}


# ---------------------------------------------------------------- helpers
def _not_executed(stocks):
    """Why each non-executable card did not become a plan: one bucket per card, counted."""
    from collections import Counter
    reasons = Counter()
    for st in stocks:
        if st["status"] == "PLAN_READY":
            continue
        if st["status"] == "DATA_GAP":
            reasons["数据缺失"] += 1
        elif st["status"] == "EXPIRED":
            reasons["计划过期未成交"] += 1
        elif any(m.startswith("引擎阻断") for m in st["missing"]):
            reasons["结构/回踩未确认"] += 1
        elif any("分钟" in m or "盘口" in m for m in st["missing"]):
            reasons["模式缺分钟盘口数据"] += 1
        elif any("不在引擎候选域" in m or "原计划" in m for m in st["missing"]):
            reasons["无冻结触发价与失效价"] += 1
        else:
            reasons["观察中"] += 1
    return dict(reasons)


def _v(r, scale=1):
    return None if not r or r.get("value") is None else round(r["value"] * scale, 2)


def _gap_lines(manifest, word):
    return [g for g in manifest.get("gaps", []) if word in g]


def _pairs(d):
    return "，".join(f"{k} {v}" for k, v in (d or {}).items()) or "--"


def _section(static, table_title, columns, rows, evidence, note, claim_id):
    return {**static, "claim_id": claim_id, "table": {"title": table_title, "columns": columns, "rows": [[("--" if v is None else v) for v in r] for r in rows],
                                                      "evidence_ids": evidence, "note": note}}


def _claim(cid, title, facts, comparison, interpretation, contrary, conclusion, next_check, evidence, kind="LOCAL_CALCULATION"):
    return {"id": cid, "title": title, "type": kind, "facts": facts, "comparison": comparison, "interpretation": interpretation,
            "contrary": contrary, "conclusion": conclusion, "next_check": next_check, "evidence_ids": evidence}


def _facts_summary(facts, manifest):
    b, y = facts["boards"]["counts"], facts["yesterday_pool"]
    return (f"域 {facts['coverage']['in_domain']}；U/Z/D {b['U']}/{b['Z']}/{b['D']}；昨日池 {y.get('n_valid')}/{y.get('n_pool')}，收盘对昨收均值 {y['close_vs_pre'].get('mean')}；"
            f"最高 {facts['ladder']['max_height']} 板；stage {manifest.get('stage')}；gaps {len(manifest.get('gaps', []))}")


def _audit(previous, stocks, rows, auto, day):
    if not previous:
        return {"title": "昨日预案兑现与误判归因", "kind": "EMPTY", "note": "没有前一份复盘报告；只对真实冻结计划复盘，不事后补写。",
                "previous": "待提供", "correction": "待形成可检验的修订", "not_proven": "缺数据时不计算漏赚、命中率或策略改进幅度。",
                "frozen_plan_available": False, "matched_quote_available": False, "actual_fills_available": False, "review_sample_count": 0}
    lines = []
    for st in previous["stocks"]:
        p = st.get("plan")
        if not p:
            continue
        row = rows.get(st["symbol"])
        bar = row["frame"].iloc[-1] if row is not None and not row["frame"].empty and row["frame"].iloc[-1].date == day else None
        if bar is None:
            lines.append(f"{st['name']}：计划带 {p['entry_min']:.2f}–{p['entry_max']:.2f}，今日无可核验行情")
            continue
        inside = p["entry_min"] <= float(bar.open) <= p["entry_max"]
        lines.append(f"{st['name']}：计划带 {p['entry_min']:.2f}–{p['entry_max']:.2f}，今日开盘 {float(bar.open):.2f}（{'带内' if inside else '带外'}），收盘 {float(bar.close):.2f}，"
                     f"{'跌破' if float(bar.close) < p['invalidation_price'] else '未破'}失效价 {p['invalidation_price']:.2f}")
    focus_lines = []
    for fid in previous["summary"].get("focus_ids", []):
        prev = next((x for x in previous["stocks"] if x["id"] == fid), None)
        row = rows.get(fid)
        bar = row["frame"].iloc[-1] if row is not None and not row["frame"].empty and row["frame"].iloc[-1].date == day else None
        if prev and bar is not None and bar.preclose > 0:
            focus_lines.append(f"{prev['name']}（{prev.get('opportunity_label', '')}）今日 {float(bar.close / bar.preclose - 1) * 100:+.2f}%")
    frozen = bool(previous["meta"].get("forward_frozen"))
    return {"title": f"{previous['meta']['trade_date']} 预案与机会池兑现核对", "kind": "FORWARD_EVALUATION" if frozen and (lines or focus_lines) else "EMPTY",
            "note": "；".join(lines + focus_lines) or "前一份报告没有计划与机会池，无兑现可核对。",
            "previous": f"前一份报告 {previous['meta']['report_id']}（{'前向冻结' if frozen else '事后重建'}），焦点 {'、'.join(previous['summary']['focus_ids'])}",
            "correction": "只记录计划带、机会池与实际开收盘的关系；策略参数不在此变更。",
            "not_proven": "一天的兑现不证明策略或打分有效；无逐笔成交与同时报价时不计算可得收益。",
            "frozen_plan_available": frozen, "matched_quote_available": False,
            "actual_fills_available": bool(auto.get("fills")), "review_sample_count": len(lines) + len(focus_lines)}


def latest_report(out_root, day):
    pointer = R.read_json(Path(out_root) / day.replace("-", "") / "fupan/latest.json")
    if not pointer or not (Path(pointer.get("path", "")) / "report.json").exists():
        return None
    report = R.read_json(Path(pointer["path"]) / "report.json")
    report["_path"] = str(Path(pointer["path"]) / "report.json")
    return report


# ---------------------------------------------------------------- persistence
def persist(day, report, meta, out_root=R.OUT_ROOT):
    """Immutable versions like eod_review: the same inputs reuse a version, changed inputs get the next one."""
    folder = Path(out_root) / day.replace("-", "") / "fupan"
    latest = R.read_json(folder / "latest.json")
    base_hash = _digest({k: v for k, v in meta.get("inputs", {}).items() if k not in ("analysis", "merge_base")})
    same = latest.get("input_hash") == meta["input_hash"]
    # A rebuild with unchanged facts keeps the version that already carries the model's answer.
    superset = not meta.get("inputs", {}).get("analysis") and latest.get("base_hash") == base_hash and latest.get("analysis_merged")
    if (same or superset) and latest.get("path") and (Path(latest["path"]) / "report.html").exists():
        return {**latest, "reused": True}
    existing = [int(p.name[1:]) for p in folder.glob("v???") if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
    version = f"v{max(existing, default=0) + 1:03d}"
    target = folder / version
    target.mkdir(parents=True, exist_ok=False)
    report["meta"]["report_id"] = f"fupan-{day}-{version}"
    packet = prepare(report)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    render(report, target / "report.html")
    (target / "ai_packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
    (target / "ai_prompt.md").write_text(prompt_text(packet), encoding="utf-8")
    (target / "packet_stats.json").write_text(json.dumps(stats(report, packet), ensure_ascii=False, indent=1), encoding="utf-8")
    if meta.get("draft_check"):
        (target / "draft_check.json").write_text(json.dumps(meta["draft_check"], ensure_ascii=False, indent=1), encoding="utf-8")
    if meta.get("hotspot_core"):
        (target / "hotspot_core.json").write_text(json.dumps(meta["hotspot_core"], ensure_ascii=False, indent=1), encoding="utf-8")
        if "funnel" in meta["hotspot_core"]:
            (target / "hotspot_funnel.json").write_text(json.dumps(meta["hotspot_core"]["funnel"], ensure_ascii=False, indent=1), encoding="utf-8")
    if meta.get("entry_review"):
        (target / "entry_review.json").write_text(json.dumps(meta["entry_review"], ensure_ascii=False, indent=1), encoding="utf-8")
    (target / "inputs.json").write_text(json.dumps({**meta, "version": version}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # The rule version (no model text) is what every later merge starts from, so answers never stack.
    rule_path = str(target) if not meta.get("inputs", {}).get("analysis") else meta.get("rule_path") or latest.get("rule_path", "")
    pointer = {"version": version, "path": str(target), "input_hash": meta["input_hash"], "base_hash": base_hash, "rule_path": rule_path,
               "as_of": day, "mode": report["meta"]["mode"],
               "forward_frozen": report["meta"]["forward_frozen"], "rendered_at": report["meta"]["rendered_at"],
               "required_symbols": report["meta"]["required_symbols"], "stocks": len(report["stocks"]), "issues": len(report["issues"]),
               "verdict": meta.get("verdict", ""), "analysis_merged": bool(meta.get("merged")), "previous_version": latest.get("version", "")}
    (folder / "latest.json").write_text(json.dumps(pointer, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**pointer, "reused": False}


def apply_analysis(day, analysis, out_root=R.OUT_ROOT):
    """Merge an analyzer's JSON into the latest report as a new version; no panel reload, no fact change."""
    current = latest_report(out_root, day)
    if not current:
        raise ValueError(f"{day} 没有复盘报告可合并")
    pointer = R.read_json(Path(out_root) / day.replace("-", "") / "fupan/latest.json")
    rule_path = pointer.get("rule_path", "")
    if rule_path and (Path(rule_path) / "report.json").exists():
        current = R.read_json(Path(rule_path) / "report.json")
    current.pop("_path", None)
    merged = merge_analysis(current, analysis)
    current["meta"]["rendered_at"] = datetime.now(TZ).isoformat(timespec="seconds")
    problems = validate(current)
    if problems:
        raise ValueError("合并后不满足契约: " + "; ".join(problems[:8]))
    previous = R.read_json(Path(R.read_json(Path(out_root) / day.replace("-", "") / "fupan/latest.json")["path"]) / "inputs.json")
    inputs = {**previous.get("inputs", {}), "analysis": _digest(analysis), "merge_base": Path(rule_path).name if rule_path else ""}
    return persist(day, current, {"input_hash": _digest(inputs), "inputs": inputs, "merged": merged, "live": current["meta"]["forward_frozen"],
                                  "verdict": previous.get("verdict", ""), "rule_path": rule_path, "hotspot_core": previous.get("hotspot_core"),
                                  "entry_review": previous.get("entry_review")}, out_root)

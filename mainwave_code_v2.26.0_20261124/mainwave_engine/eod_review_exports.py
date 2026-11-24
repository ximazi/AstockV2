"""Supplement exports for the after-close review: raw rows behind every number, so a reader can recompute.

A. definitions and stock-by-stock reconciliation   B. last-N-session layered history
C. theme members and key-stock bars                 D. the sole candidate's plan and execution record
E. dragon-tiger lists (top_list / top_inst) as collected, with "not collected" stated as such.
Nothing here recomputes a ranking or touches a ledger."""
import argparse
import json
from collections import Counter
from pathlib import Path
import statistics

import pandas as pd

from . import eod_review as R
from .pipeline import _write

VERSION = "eod_review_exports_v1_20260922"
KEY_THEMES = ("流感", "创新药", "医疗器械概念", "租售同权", "全国地产", "特色小镇", "百货", "出版业", "环境保护")


def _names(mapping):
    return {s: (v.get("name") or "") for s, v in mapping.items()}


def _symbol_of(ts_code):
    from . import intraday_quotes as Q
    return Q.symbol_of(ts_code)


def _reference_lists(day, reference_root=None):
    root = Path(reference_root or R.REFERENCE_ROOT) / "reference"
    compact = day.replace("-", "")
    out = {}
    for kind, name in (("U", f"limit_list_d_{compact}.json"), ("Z", f"limit_list_d_Z_{compact}.json"), ("D", f"limit_list_d_D_{compact}.json")):
        payload = R.read_json(root / name)
        out[kind] = {_symbol_of(r["ts_code"]): r for r in payload.get("records", [])} if payload else None
    limits = R.read_json(root / f"stk_limit_{compact}.json")
    out["limits"] = {_symbol_of(r["ts_code"]): {"up": R.number(r.get("up_limit")), "down": R.number(r.get("down_limit")), "pre_close": R.number(r.get("pre_close"))}
                     for r in limits.get("records", [])} if limits else {}
    return out


def _bar(f, i):
    b = f.iloc[i]
    return {"date": b.date, "open": float(b.open), "high": float(b.high), "low": float(b.low), "close": float(b.close),
            "preclose": float(b.preclose) if b.preclose == b.preclose else None, "volume": float(b.volume), "amount": float(b.amount),
            "turnover": float(b.turnover) if b.turnover == b.turnover else None, "up_limit": float(b.up_limit) if b.up_limit == b.up_limit else None,
            "down_limit": float(b.down_limit) if b.down_limit == b.down_limit else None,
            "kind": "U" if bool(b.is_U) else "Z" if bool(b.is_Z) else "D" if bool(b.is_D) else "DZ" if bool(b.is_DZ) else "",
            "st": bool(b.st), "traded": bool(b.traded)}


def heights_of(f, position):
    """Consecutive limit-up height per bar, broken by any gap in the market calendar (same rule as the review)."""
    dates, is_u, st, traded = f.date.tolist(), f.is_U.tolist(), f.st.tolist(), f.traded.tolist()
    heights, height = [], 0
    for i, d in enumerate(dates):
        consecutive = i > 0 and position.get(d, -9) - position.get(dates[i - 1], -9) == 1
        height = (height + 1 if (consecutive and height > 0) else 1) if (is_u[i] and traded[i] and not st[i]) else 0
        heights.append(height)
    return heights


# ---------------------------------------------------------------- A
def export_a(day, facts, rows, refs, names, out, prev_facts=None):
    prev = facts["previous_day"]
    perf = {m["symbol"]: m for m in facts["yesterday_pool"]["performance"]}
    pool_rows = []
    for s in facts["yesterday_pool"]["members"]:
        m = perf.get(s, {})
        f = rows.get(s, {}).get("frame")
        prev_bar = _bar(f, -2) if f is not None and len(f) >= 2 and f.date.iloc[-2] == prev else {}
        pool_rows.append({"symbol": s, "name": names.get(s, ""), "board": R.board_of(s), "status_today": m.get("status", "MISSING"),
                          "prev_close": prev_bar.get("close"), "prev_up_limit": prev_bar.get("up_limit"),
                          **{k: m.get(k) for k in ("open_vs_pre", "close_vs_pre", "close_vs_open", "high_vs_pre", "limit_today")}})
    pd.DataFrame(pool_rows).to_csv(out / "A_pool_yesterday.csv", index=False, encoding="utf-8-sig")

    lists = []
    for kind in ("U", "Z", "D", "DZ"):
        for s in facts["boards"][kind]:
            f = rows[s]["frame"]
            b = _bar(f, -1)
            ref = (refs.get("U") or {}).get(s) or (refs.get("Z") or {}).get(s) or (refs.get("D") or {}).get(s) or {}
            lists.append({"symbol": s, "name": names.get(s, ""), "kind": kind, "kind_label": R.LABELS["kind"][kind], "board": R.board_of(s),
                          "preclose": b["preclose"], "up_limit": b["up_limit"], "down_limit": b["down_limit"],
                          "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
                          "price_basis": rows[s].get("price_basis", ""),
                          "ref_first_time": ref.get("first_time"), "ref_last_time": ref.get("last_time"),
                          "ref_open_times": ref.get("open_times"), "ref_limit_times": ref.get("limit_times"), "ref_limit_type": ref.get("limit") or ("" if not ref else kind)})
    pd.DataFrame(lists).to_csv(out / "A_limit_lists_today.csv", index=False, encoding="utf-8-sig")

    diffs = []
    for kind in ("U", "Z", "D"):
        recon = facts["boards"]["reconciliation"].get(kind, {})
        for side, key in (("LOCAL_ONLY", "local_only"), ("REFERENCE_ONLY", "reference_only")):
            for s in recon.get(key, []):
                f = rows.get(s, {}).get("frame")
                b = _bar(f, -1) if f is not None else {}
                ref = (refs.get(kind) or {}).get(s, {})
                lim = refs["limits"].get(s, {})
                diffs.append({"kind": kind, "kind_label": R.LABELS["kind"][kind], "side": side, "symbol": s, "name": names.get(s, ""),
                              "in_domain": rows.get(s, {}).get("in_domain"), "exclusion": rows.get(s, {}).get("reason", ""),
                              "local_close": b.get("close"), "local_high": b.get("high"), "local_low": b.get("low"),
                              "local_up_limit": b.get("up_limit"), "local_down_limit": b.get("down_limit"), "local_kind": b.get("kind"),
                              "reference_up_limit": lim.get("up"), "reference_down_limit": lim.get("down"),
                              "reference_record": json.dumps(ref, ensure_ascii=False) if ref else "",
                              "local_rule": "收盘价==涨停价→U；最高≥涨停价且收盘<涨停价→Z；收盘价==跌停价→D（容差0.005）"})
    pd.DataFrame(diffs).to_csv(out / "A_reconciliation_differences.csv", index=False, encoding="utf-8-sig")

    # Two ways the review counts first-board promotion; both listed with their members.
    sessions = R.calendar(DATA_ROOT["path"])
    position = {d: i for i, d in enumerate(sessions)}
    sentiment_pool, sentiment_cont = [], []
    for s, row in rows.items():
        f = row["frame"]
        if len(f) < 2 or f.date.iloc[-1] != day or f.date.iloc[-2] != prev:
            continue
        h = heights_of(f, position)
        last, before = f.iloc[-1], f.iloc[-2]
        ok = (not bool(last.st) and bool(last.traded) and bool(before.is_U) and bool(before.traded) and not bool(before.st)
              and last.preclose and before.close and abs(last.preclose / before.close - 1) <= .001)
        if ok and h[-2] == 1:
            sentiment_pool.append(s)
            if bool(last.is_U):
                sentiment_cont.append(s)
    ladder_1 = facts["ladder"]["promotion_from_previous"].get("1", {})
    # The frozen previous-day heights are the ladder's real pool; the facts' broke list is truncated to 50.
    ladder_pool = {s for s, h in ((prev_facts or {}).get("ladder", {}).get("heights") or {}).items() if int(h) == 1}
    if not ladder_pool:
        ladder_pool = set(ladder_1.get("continued", [])) | set(ladder_1.get("broke", []))
    _write(out / "A_promotion_1_to_2_reconcile.json", {
        "ladder_table": {"definition": "昨日冻结梯队中高度=1者 → 今日可核验（有当日行情且成交）中继续封板者", "pool": ladder_1.get("pool"),
                         "checkable": ladder_1.get("denominator"), "continued": ladder_1.get("continued", []),
                         "broke": sorted(ladder_pool - set(ladder_1.get("continued", []))), "broke_list_in_facts_truncated_to": 50,
                         "members": sorted(ladder_pool), "uncheckable": ladder_1.get("uncheckable"), "value": ladder_1.get("value")},
        "sentiment_table": {"definition": "两日相邻且今日非ST、有成交、前收与昨收一致（无公司行动）的昨日一板股 → 今日继续封板者",
                            "pool": sorted(sentiment_pool), "continued": sorted(sentiment_cont),
                            "value": round(len(sentiment_cont) / len(sentiment_pool), 4) if sentiment_pool else None},
        "difference": {"in_ladder_not_sentiment": sorted(ladder_pool - set(sentiment_pool)),
                       "in_sentiment_not_ladder": sorted(set(sentiment_pool) - ladder_pool),
                       "reading": "口径差异只来自成员资格过滤（公司行动、ST、停牌），不是计算错误；两处都保留，报告应注明口径"}})
    pending = [m for m in facts["yesterday_pool"]["performance"] if m["status"] != "OK"]
    pd.DataFrame([{**m, "name": names.get(m["symbol"], "")} for m in pending]).to_csv(out / "A_corporate_action_pending.csv", index=False, encoding="utf-8-sig")
    (out / "A_field_definitions.md").write_text(FIELD_DEFINITIONS, encoding="utf-8")


FIELD_DEFINITIONS = """# 字段定义（复盘产物中派生字段的口径与出处）

| 字段 | 定义 | 出处 |
|---|---|---|
| 收盘涨停 U | 收盘价 == 涨停价（容差 0.005 元） | eod_review.annotate / apply_reference_prices |
| 炸板 Z | 最高价 ≥ 涨停价 且 收盘价 < 涨停价 | 同上 |
| 收盘跌停 D / 触跌停未封死 DZ | 收盘价 == 跌停价 / 最低价 ≤ 跌停价 且收盘 > 跌停价 | 同上 |
| 涨跌停价 | 有交易所参考价（Tushare stk_limit）用参考价；否则前收 ×(1±10%/20%) 四舍五入到分；参考价与推算不符者列入 limit_price_disagreements | 同上 |
| 统计域 | 沪深 60/00/30/68 前缀，当日 isST=0、有成交、数据有效；ST、停牌、缺行、指数代码不入分母 | eod_review.day_rows |
| 封板失败率 | Z ÷ (U+Z)，按股票去重 | boards_fact |
| 涨停股曾开板比例 | 参考 U 名单（按本地域过滤去重）中 open_times>0 者 ÷ 有 open_times 字段者；open_times 是次数 | boards_fact |
| 昨日涨停母池 | 上一交易日（按市场日历）收盘涨停股；有前一日冻结/归档版则取其名单（basis 标明），否则由日线重建 | build / pool_performance |
| 开盘/收盘/日内最高 对昨收 | (open/close/high ÷ preclose − 1)×100；收盘对开盘 = (close ÷ open − 1)×100；停牌、缺数据、preclose≠昨收（公司行动）单列不计入 | pool_performance |
| 连板高度 | 连续市场交易日收盘涨停天数；停牌或缺日打断 | ladder_fact / heights |
| 晋级率 | 昨日 n 板母池中今日继续封板者 ÷ 今日可核验者（有当日行情且成交）；母池数与不可核验数并列 | ladder_fact |
| 收盘位置 | (close − low) ÷ (high − low)；high==low 时无定义（不计入分布，故分布合计可少于域） | activity_fact |
| 相对换手 | 当日换手 ÷ 前 20 日换手中位数（只用 >0 的值，至少 10 个） | activity_fact |
| 20 日位置 position20 | (close − 20 日最低) ÷ (20 日最高 − 20 日最低)，含当日；高位 ≥0.8、低位 ≤0.2、其余中位——按**当日收盘后**位置，不是昨日位置 | activity_fact / loss_fact |
| 亏钱事件 | 昨日涨停股今日收跌；收盘跌停；炸板；高开低走 = 开盘 ≥ 昨收×1.02 且收盘 < 开盘 且收盘 < 昨收；跌破 20 日低点 = 收盘 < 前 20 日最低价；一只股票可同时属多类 | loss_fact |
| 题材广度 / 涨停占比 / 成交额占比 | 上涨成员 ÷ 成员；涨停成员 ÷ 成员；去重成员成交额 ÷ 域内成交额；成员表为 concept_mapping.json 当前版本 | theme_fact |
| 市场状态 绿灯/橙灯/红灯 | 数据新鲜且：温度 PANIC→红；结构 BEAR 或温度 EUPHORIA 或上下文 POST_PANIC→橙；否则绿。趋势类复核仅绿灯允许 | market_context.py |
| 结构风险 RISK_WATCH | 价值分层为 RISK_WATCH、或卖出结构告警、或生命周期风险覆盖（非市场类）任一成立 | trading/ranking.py |
| MARKET_WAIT / WAIT_SETUP / PLAN_REVIEW | 市场未放行→MARKET_WAIT；不是 FIRST_PULLBACK 或确认窗口不活跃或未通过初审→WAIT_SETUP（阻断 WAIT_PULLBACK_CONFIRMATION）；无阻断→PLAN_REVIEW | trading/ranking.py |
| R1 / R2 / R3 | R1 目标股点火后反查（要求突破接受且 demand、bias_atr ≤ 2 等价格确认）；R2 个股先行埋伏（准备→保留→收敛→结构确认，可价格独立）；R3 紧链扩散（研究口径，需 strict_chain_verified，不进策略统计） | route_contracts.py / route_evidence.py |
| 接力情绪读法 | 每项指标今日值在自身近 20 日的分位五等分：冷/偏冷/中性/偏热/热；炸板率与跌停数方向取反；不合成总分 | sentiment_fact |

未采集：分钟级封板/开板过程（只有 Tushare 的首次/最后封板时间与开板次数）、龙虎榜以外的资金身份、题材成员的历史时点版本。
"""


# ---------------------------------------------------------------- B
def export_b(day, rows, out, n_days=5):
    sessions = R.calendar(DATA_ROOT["path"])
    position = {d: i for i, d in enumerate(sessions)}
    days = [d for d in sessions if d <= day][-n_days:]
    per = {d: {"pool": [], "open": [], "close_open": [], "cont": 0, "promo": {}, "broke": [], "gap_rev": 0, "breakdown": 0, "D": 0, "U": 0, "Z": 0,
               "amount": 0.0, "amounts": [], "domain": 0} for d in days}
    for s, row in rows.items():
        f = row["frame"]
        h = heights_of(f, position)
        dates = f.date.tolist()
        for i, d in enumerate(dates):
            if d not in per:
                continue
            b = f.iloc[i]
            if bool(b.st) or not bool(b.traded):
                continue
            p = per[d]
            p["domain"] += 1; p["U"] += bool(b.is_U); p["Z"] += bool(b.is_Z); p["D"] += bool(b.is_D)
            p["amount"] += float(b.amount or 0); p["amounts"].append(float(b.amount or 0))
            if b.preclose and b.open >= b.preclose * 1.02 and b.close < b.open and b.close < b.preclose:
                p["gap_rev"] += 1
            if i >= 20 and b.close < f.low.iloc[i - 20:i].min():
                p["breakdown"] += 1
            consecutive = i > 0 and position.get(d, -9) - position.get(dates[i - 1], -9) == 1
            prev = f.iloc[i - 1] if i > 0 else None
            if consecutive and prev is not None and bool(prev.is_U) and bool(prev.traded) and not bool(prev.st) and b.preclose and prev.close and abs(b.preclose / prev.close - 1) <= .001:
                ret = (b.close / b.preclose - 1) * 100
                p["pool"].append(ret); p["open"].append((b.open / b.preclose - 1) * 100); p["close_open"].append((b.close / b.open - 1) * 100 if b.open else None)
                p["cont"] += bool(b.is_U)
                hp = h[i - 1]
                bucket = p["promo"].setdefault(str(hp), {"pool": 0, "continued": [], "broke": []})
                bucket["pool"] += 1
                (bucket["continued"] if bool(b.is_U) else bucket["broke"]).append(s)
                if not bool(b.is_U):
                    p["broke"].append({"day": d, "symbol": s, "height_yesterday": hp, "close_vs_pre": round(ret, 3),
                                       "low_vs_pre": round((b.low / b.preclose - 1) * 100, 3), "open_vs_pre": round((b.open / b.preclose - 1) * 100, 3),
                                       "kind_today": "Z" if bool(b.is_Z) else "D" if bool(b.is_D) else ""})
    hist, promo_hist, broke_rows = [], {}, []
    for d in days:
        p = per[d]
        top20 = sum(sorted(p["amounts"], reverse=True)[:20])
        hist.append({"day": d, "domain": p["domain"], "limit_up": p["U"], "seal_failed": p["Z"], "limit_down": p["D"],
                     "seal_failure_rate": round(p["Z"] / (p["U"] + p["Z"]), 4) if p["U"] + p["Z"] else None,
                     "pool_n": len(p["pool"]), "pool_open_mean": round(statistics.fmean(p["open"]), 3) if p["open"] else None,
                     "pool_close_mean": round(statistics.fmean(p["pool"]), 3) if p["pool"] else None,
                     "pool_close_median": round(statistics.median(p["pool"]), 3) if p["pool"] else None,
                     "pool_up_share": round(sum(v > 0 for v in p["pool"]) / len(p["pool"]), 4) if p["pool"] else None,
                     "pool_close_open_mean": round(statistics.fmean([v for v in p["close_open"] if v is not None]), 3) if p["close_open"] else None,
                     "pool_continued": p["cont"], "gap_up_reversal": p["gap_rev"], "breakdown_20d": p["breakdown"],
                     "amount_total_yi": round(p["amount"] / 1e8, 1), "top20_amount_share": round(top20 / p["amount"], 4) if p["amount"] else None})
        promo_hist[d] = {h: {"pool": v["pool"], "continued_n": len(v["continued"]), "broke_n": len(v["broke"]),
                             "rate": round(len(v["continued"]) / v["pool"], 4) if v["pool"] else None,
                             "continued": sorted(v["continued"]), "broke": sorted(v["broke"])} for h, v in sorted(p["promo"].items(), key=lambda kv: int(kv[0]))}
        broke_rows += p["broke"]
    pd.DataFrame(hist).to_csv(out / "B_history_5d.csv", index=False, encoding="utf-8-sig")
    _write(out / "B_promotion_history.json", {"definition": "同一口径：两日相邻、今日非ST有成交、无公司行动的昨日 n 板股；今日继续封板为晋级，否则断板",
                                              "days": promo_hist})
    pd.DataFrame(sorted(broke_rows, key=lambda r: (r["day"], -r["height_yesterday"], r["close_vs_pre"]))).to_csv(out / "B_broke_performance.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- C
def export_c(day, facts, rows, refs, mapping, names, out, key_symbols):
    members = []
    concept_of = {s: set(v.get("concepts", [])) for s, v in mapping.items()}
    sessions = R.calendar(DATA_ROOT["path"])
    position = {d: i for i, d in enumerate(sessions)}
    for theme in KEY_THEMES:
        for s, v in mapping.items():
            if theme not in v.get("concepts", []) or s not in rows or not rows[s]["in_domain"]:
                continue
            f = rows[s]["frame"]
            b = _bar(f, -1)
            h = heights_of(f, position)[-1]
            others = sorted((concept_of[s] & set(KEY_THEMES)) - {theme})
            members.append({"theme": theme, "symbol": s, "name": names.get(s, ""), "board": R.board_of(s), "pct_chg": round((b["close"] / b["preclose"] - 1) * 100, 2) if b["preclose"] else None,
                            "kind_today": b["kind"], "height": h, "amount_yi": round(b["amount"] / 1e8, 2),
                            "close_location": round((b["close"] - b["low"]) / (b["high"] - b["low"]), 3) if b["high"] > b["low"] else None,
                            "return20_pct": round((f.close.iloc[-1] / f.close.iloc[-21] - 1) * 100, 2) if len(f) >= 21 else None,
                            "other_key_themes": "|".join(others), "all_concepts_n": len(concept_of[s])})
    frame = pd.DataFrame(members, columns=["theme", "symbol", "name", "board", "pct_chg", "kind_today", "height", "amount_yi", "close_location", "return20_pct", "other_key_themes", "all_concepts_n"])
    frame.sort_values(["theme", "kind_today", "pct_chg"], ascending=[True, True, False]).to_csv(out / "C_theme_members.csv", index=False, encoding="utf-8-sig")

    bars = []
    for s in key_symbols:
        path = Path(DATA_ROOT["path"]) / f"{s}_daily.csv"
        if not path.exists():
            bars.append({"symbol": s, "note": "NOT_COLLECTED"}); continue
        f = pd.read_csv(path)
        f["date"] = f["date"].astype(str).str[:10]
        f = f[f.date.le(day)].tail(60)
        for r in f.to_dict("records"):
            bars.append({"symbol": s, "name": names.get(s, ""), **{k: r.get(k) for k in ("date", "open", "high", "low", "close", "preclose", "volume", "amount", "turnover", "pct_chg", "isST", "tradestatus")}})
    pd.DataFrame(bars).to_csv(out / "C_key_stocks_daily_60d.csv", index=False, encoding="utf-8-sig")

    process = []
    for s in key_symbols:
        rec = (refs.get("U") or {}).get(s) or (refs.get("Z") or {}).get(s) or (refs.get("D") or {}).get(s)
        process.append({"symbol": s, "name": names.get(s, ""), "in_reference_list": bool(rec),
                        "first_seal_time": rec.get("first_time") if rec else "NOT_IN_LIST", "last_seal_time": rec.get("last_time") if rec else "NOT_IN_LIST",
                        "open_times": rec.get("open_times") if rec else "NOT_IN_LIST", "limit_times": rec.get("limit_times") if rec else "NOT_IN_LIST",
                        "minute_bars": "NOT_COLLECTED", "note": "封板/开板时点来自 Tushare limit_list_d；分钟级过程未采集"})
    pd.DataFrame(process).to_csv(out / "C_limit_process_today.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- D
def export_d(day, out, out_root=None):
    folder = Path(out_root or R.OUT_ROOT) / day.replace("-", "")
    closure = R.read_json(folder / "trading_closure.json")
    books = R.read_json(folder / "books_daily.json")
    status = R.read_json(folder / "trading_closure_status.json")
    eligible = [r for r in closure.get("records", []) if r.get("paper_plan_eligible")]
    _write(out / "D_candidates_plan_review.json", {
        "as_of": day, "snapshot_id": closure.get("snapshot_id"), "recorded_at": closure.get("recorded_at"), "observation_basis": closure.get("observation_basis"),
        "ranking_version": closure.get("version"), "rules": status.get("rules"),
        "records": [{k: v for k, v in r.items() if k != "source_record"} for r in eligible],
        "auto_book": {k: books.get("auto", {}).get(k) for k in ("live", "ledger_touched", "placed", "executions", "plans", "fills", "open_positions", "effective_account", "reconciliation", "snapshot_id")},
        "signal_book": books.get("signal"),
        "observation_plan_previous_day": "NOT_IMPLEMENTED_P1：上一交易日没有保存的观察计划与兑现记录（设计文档 P1-A 未实施）",
        "sources": {"daily_brief": str(folder / "daily_brief.json"), "trading_closure": str(folder / "trading_closure.json"), "books_daily": str(folder / "books_daily.json")}})


# ---------------------------------------------------------------- E
def export_e(days, key_symbols, names, out):
    from . import intraday_quotes as Q
    try:
        client = Q.Client()
    except Exception as exc:
        _write(out / "E_top_list_status.json", {"status": "NOT_COLLECTED", "reason": str(exc)})
        return
    summary = {"days": {}, "key_symbols": {}, "note": "top_list=每日龙虎榜明细（上榜原因/买卖额），top_inst=席位明细；NOT_ON_LIST 是有采集但未上榜，NOT_COLLECTED 是当日未采集"}
    frames = {"top_list": [], "top_inst": []}
    for d in days:
        compact = d.replace("-", "")
        for api in ("top_list", "top_inst"):
            try:
                frame = client.query(api, trade_date=compact)
                frame["trade_date"] = compact
                frames[api].append(frame)
                summary["days"].setdefault(d, {})[api] = {"status": "COLLECTED", "rows": int(len(frame))}
            except Exception as exc:
                summary["days"].setdefault(d, {})[api] = {"status": "NOT_COLLECTED", "reason": str(exc)[:160]}
    for api, parts in frames.items():
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / f"E_{api}.csv", index=False, encoding="utf-8-sig")
    listed = pd.concat(frames["top_list"], ignore_index=True) if frames["top_list"] else pd.DataFrame(columns=["ts_code", "trade_date"])
    for s in key_symbols:
        code = Q.ts_code(s)
        hits = listed[listed.ts_code.eq(code)] if len(listed) else listed
        summary["key_symbols"][s] = {"name": names.get(s, ""), "on_list_days": sorted(set(hits.trade_date.astype(str))) if len(hits) else [],
                                     "status": "ON_LIST" if len(hits) else ("NOT_ON_LIST" if frames["top_list"] else "NOT_COLLECTED")}
    _write(out / "E_top_list_status.json", summary)


# ---------------------------------------------------------------- README (numbers pulled from the files just written)
def _csv(path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 5 else pd.DataFrame()


def export_readme(day, facts, out, key_symbols, prev_top):
    recon = R.read_json(out / "A_promotion_1_to_2_reconcile.json")
    diffs, pending, hist = _csv(out / "A_reconciliation_differences.csv"), _csv(out / "A_corporate_action_pending.csv"), _csv(out / "B_history_5d.csv")
    e = R.read_json(out / "E_top_list_status.json")
    y = facts["yesterday_pool"]
    lad, sen, diff = recon["ladder_table"], recon["sentiment_table"], recon["difference"]
    kind = R.LABELS["kind"]
    rows = []
    for r in diffs.itertuples():
        ref = r.reference_record if isinstance(r.reference_record, str) else "参考名单无此股"
        rows.append(f"| {kind.get(r.kind, r.kind)} | {'仅本地' if r.side == 'LOCAL_ONLY' else '仅参考'} | {r.symbol} {r.name} | 收盘 {r.local_close}，最高 {r.local_high}，涨停价 {r.local_up_limit}，本地判 {r.local_kind if isinstance(r.local_kind, str) else '-'} | {ref} |")
    lines = [f"# 复盘补充数据包 {day}（{VERSION}）", "",
             "原始导出，不含主观结论；每个文件对应解读中要求的补充包。缺失的一律写明 NOT_COLLECTED / NOT_IMPLEMENTED，不补造。", "",
             "## 文件索引", "", "| 包 | 文件 | 内容 |", "|---|---|---|",
             "| A | A_pool_yesterday.csv | 昨日涨停母池逐股：昨日收盘/涨停价、今日开盘/收盘/日内最高对昨收、状态 |",
             "| A | A_limit_lists_today.csv | 今日收盘涨停/炸板/收盘跌停/触跌停未封死名单，含本地价格判定与 Tushare 首次/最后封板时间、开板次数、连板数 |",
             "| A | A_reconciliation_differences.csv | 本地与参考名单差异逐股（价格、涨停价、判定依据、参考记录原文） |",
             "| A | A_promotion_1_to_2_reconcile.json | 两处一进二的分子分母与成员清单及差异 |",
             "| A | A_corporate_action_pending.csv | 公司行动待核股票 |",
             "| A | A_field_definitions.md | 收盘位置、相对换手、20日位置、绿灯、结构风险、R1/R2/R3 等定义与代码出处 |",
             "| B | B_history_5d.csv | 近 5 交易日：涨停/炸板/跌停、封板失败率、昨日涨停股开盘/收盘溢价与收红比例、高开低走、破20日低、成交额与前20集中度 |",
             "| B | B_promotion_history.json | 近 5 交易日各高度晋级/断板成员清单 |",
             "| B | B_broke_performance.csv | 近 5 交易日断板股逐只：昨日高度、今日开盘/最低/收盘对昨收 |",
             "| C | C_theme_members.csv | 重点题材（医药三项、地产三项、百货、出版、环境保护）成员逐股：涨跌、涨停/炸板、连板高度、成交额、重叠题材 |",
             "| C | C_key_stocks_daily_60d.csv | 风向标、参与复核标的、昨日≥4板股的 60 日日线 |",
             "| C | C_limit_process_today.csv | 上述股票的封板时点/开板次数（Tushare）；分钟级 NOT_COLLECTED |",
             "| D | D_candidates_plan_review.json | PLAN_REVIEW 记录全量（窗口价、失效位、路径证据、契约、买入计划）、自动账下单与执行、信号账；上一交易日观察计划 NOT_IMPLEMENTED_P1 |",
             "| E | E_top_list.csv / E_top_inst.csv / E_top_list_status.json | 近 5 交易日龙虎榜明细与席位；关键股上榜日；未采集明确标注 |", "",
             "## 解读中列出的口径问题，用数据回答", ""]
    if lad.get("value") is not None and sen.get("value") is not None:
        lines.append(f"1. **一进二 {lad['value']:.1%}（{len(lad.get('continued', []))}/{lad.get('checkable')}） vs {sen['value']:.1%}（{len(sen.get('continued', []))}/{len(sen.get('pool', []))}）**：分子相同，分母差在成员资格——梯队表用昨日冻结梯队（{lad.get('pool')} 只），情绪表要求两日相邻、无公司行动、今日非ST有成交。"
                     f" 只在梯队表：{'、'.join(diff['in_ladder_not_sentiment']) or '无'}；只在情绪表：{'、'.join(diff['in_sentiment_not_ladder']) or '无'}。两处都保留，报告注明口径。")
    else:
        lines.append("1. 一进二：当日无可比样本。")
    pend = "、".join(f"{r.symbol} {r.name}" for r in pending.itertuples()) or "无"
    lines.append(f"2. **昨日母池 {y['n_valid']} vs {y['n_pool']}**：{y['n_pool']} 只母池中 {pend} 为公司行动待核（前收≠昨收），溢价统计不计入（{y['n_valid']}），梯队“今日可核验”按有成交计入。")
    lines += ["3. **本地与参考名单差异**（详见 A_reconciliation_differences.csv）：", "", "| 类型 | 方向 | 股票 | 本地事实 | 参考记录 |", "|---|---|---|---|---|"] + (rows or ["| 无 | | | | |"])
    lines += ["", "   读法：收盘价等于涨停价但参考名单判炸板，是“收盘无封单”的情形——本地只看价格，参考看封单；最高价触及涨停价而参考未列炸板，是“触及但未封过”——本地 Z 是触及口径，参考 Z 是封过再开。两者都不是错，是定义不同，报告应写明采用哪一个。",
              "4. **收盘位置分布合计少于域**：high==low 的股票无定义，未计入分布（见 A_field_definitions.md）。",
              "5. **高位/低位按当日收盘后的 20 日位置**，不是昨日位置；高位补跌后可能落入中位，解读时注意（见定义）。", "",
              "## 近 5 交易日一览（B_history_5d.csv）", "",
              "| 日期 | 涨停 | 炸板 | 跌停 | 封板失败率 | 昨日涨停股开盘溢价 | 收盘溢价 | 收红比例 | 高开低走 | 破20日低 | 成交额(亿) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in hist.itertuples():
        lines.append(f"| {h.day} | {h.limit_up} | {h.seal_failed} | {h.limit_down} | {h.seal_failure_rate:.1%} | {h.pool_open_mean:+.2f}% | {h.pool_close_mean:+.2f}% | {h.pool_up_share:.0%} | {h.gap_up_reversal} | {h.breakdown_20d} | {h.amount_total_yi:.0f} |")
    ks = e.get("key_symbols") or {}
    lines += ["", f"## 关键股票（{len(key_symbols)} 只）", "", f"- 风向标与参与复核：{', '.join(key_symbols)}；昨日 ≥4 板：{', '.join(prev_top) or '无'}",
              "- 龙虎榜：" + ("; ".join(f"{s} {v.get('name', '')} {v.get('status')} {v.get('on_list_days') or ''}" for s, v in ks.items()) if ks else str(e.get("status", ""))), "",
              "## 未采集 / 未实施", "", "- 分钟级封板/开板过程：未采集（只有 Tushare 的时点与次数）。", "- 上一交易日的观察计划与兑现记录：未实施（P1）。",
              "- 题材成员的历史时点版本：无；成员表为当前版本。", "- 资金身份只到龙虎榜席位为止，不推断机构/游资。", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


DATA_ROOT = {"path": "data"}


# ---------------------------------------------------------------- driver
def run(day, data_root="data", out_root=None, reference_root=None, mapping_path="concept_mapping.json", days=5, use_tushare=True):
    DATA_ROOT["path"] = str(data_root)
    out_root = Path(out_root or R.OUT_ROOT)
    latest = R.read_json(out_root / day.replace("-", "") / "eod_review" / "latest.json")
    if not latest:
        raise SystemExit(f"{day} 没有复盘产物")
    facts = R.read_json(Path(latest["path"]) / "facts.json")
    out = Path(latest["path"]) / "exports"
    out.mkdir(exist_ok=True)
    mapping = R.read_json(mapping_path).get("mapping", {})
    names = _names(mapping)
    panel, _ = R.load_panel(data_root, day)
    rows, _ = R.day_rows(panel, day)
    refs = _reference_lists(day, reference_root)
    R.apply_reference_prices(rows, refs["limits"], reference_complete=bool(refs["limits"]))
    prev_latest = R.read_json(out_root / facts["previous_day"].replace("-", "") / "eod_review" / "latest.json") if facts.get("previous_day") else {}
    prev_facts = R.read_json(Path(prev_latest["path"]) / "facts.json") if prev_latest.get("path") else {}
    prev_top = [s for s, h in (prev_facts.get("ladder", {}).get("heights") or {}).items() if h >= 4]
    key_symbols = list(dict.fromkeys([p["symbol"] for p in facts["lists"]["bellwethers"]] + [c["symbol"] for c in facts["lists"].get("review_eligible", [])] + prev_top))
    export_a(day, facts, rows, refs, names, out, prev_facts)
    export_b(day, rows, out, days)
    export_c(day, facts, rows, refs, mapping, names, out, key_symbols)
    export_d(day, out, out_root)
    sessions = R.calendar(data_root)
    window = [d for d in sessions if d <= day][-days:]
    if use_tushare:
        export_e(window, key_symbols, names, out)
    else:
        _write(out / "E_top_list_status.json", {"status": "NOT_COLLECTED", "reason": "--no-tushare"})
    export_readme(day, facts, out, key_symbols, prev_top)
    _write(out / "manifest.json", {"version": VERSION, "as_of": day, "review_version": latest.get("version"), "key_symbols": key_symbols,
                                   "previous_day_boards_4plus": prev_top, "files": sorted(p.name for p in out.iterdir())})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-root", default=str(R.OUT_ROOT))
    parser.add_argument("--no-tushare", action="store_true")
    args = parser.parse_args()
    out = run(args.date, args.data_root, args.out_root, days=args.days, use_tushare=not args.no_tushare)
    print(f"[eod_review_exports] {args.date} -> {out} files={len(list(out.iterdir()))}")


if __name__ == "__main__":
    main()

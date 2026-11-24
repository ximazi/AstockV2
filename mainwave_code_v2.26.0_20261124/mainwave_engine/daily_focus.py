"""Summarize completed daily evidence; does not read or change any order book."""
import argparse
from pathlib import Path

from .eod_review import read_json
from .pipeline import _write


VERSION = "daily_focus_v1_20261120"
MAX_FOCUS = 3
WAIT_R1 = {"WAIT_ACCEPTANCE", "TREND_WATCH", "CONFIRMATION_REVIEW"}


def build(day, closure, facts, hotspot, report, s4):
    market = closure.get("market") or {}
    allowed = (market.get("data_status") == "FRESH" and market.get("state") == "GREEN"
               and market.get("trend_review_allowed") is True)
    records = {r["symbol"]: r for r in closure.get("records", []) if r.get("price_date") == day}
    ranked = sorted((r for r in records.values() if r.get("participation_rank") is not None),
                    key=lambda r: (r["participation_rank"], r["symbol"]))
    hot_stocks = {r["symbol"]: r for r in hotspot.get("rows", [])}
    hot_themes = [x for x in facts.get("themes", {}).get("top_by_limit_share", []) if not x.get("generic")][:3]
    hot_names = [t["theme"] for t in hot_themes]
    output, seen = [], set()

    def themes(symbol):
        stock = hot_stocks.get(symbol, {})
        record = records.get(symbol, {})
        tags = set(stock.get("themes", [])) | {record.get("theme")}
        return [t for t in hot_names if t in tags] or stock.get("themes", [])[:1] or [record.get("theme") or "未核实"]

    def add(r, role, evidence, watch, limit=2, close=None):
        symbol = r["symbol"]
        if symbol in seen or len(output) >= limit:
            return
        output.append({"symbol": symbol, "name": r.get("name") or symbol, "role": role,
                       "themes": themes(symbol), "close": close if close is not None else (r.get("features") or {}).get("close"),
                       "evidence": evidence, "next_check": watch})
        seen.add(symbol)

    formal = [r for r in ranked if allowed and r.get("paper_plan_eligible") is True]
    for r in formal[:2]:
        add(r, "正式计划复核", f"参与榜#{r['participation_rank']}，原规则已给出复核资格。",
            r.get("next_condition") or "核对原计划价格、有效期及账户风控。")

    strict = {r["symbol"] for r in hotspot.get("funnel", {}).get("first_board_watch", {}).get("picks", [])}
    p5 = facts.get("p5_limit_up") or {}
    first_boards = [r for r in p5.get("rows", []) if r["priority"] == "PRIOR_P5_FIRST_BOARD"
                    and not r.get("one_price") and records.get(r["symbol"], {}).get("state") not in {"RISK_WATCH", "DATA_WAIT"}]
    first_boards.sort(key=lambda r: (r["symbol"] not in strict, not set(themes(r["symbol"])).intersection(hot_names),
                                     records.get(r["symbol"], {}).get("participation_rank") or float("inf"), r["symbol"]))
    for r in first_boards[:1]:
        dates = "、".join(x["date"] for x in r["evidence"] if x["date"] < day)
        add(r, "P5首板关注", f"先入P5（{dates}），今日收盘首板。",
            f"等次日回踩承接；失守首板低点 {r['first_board_low']:.2f} 则取消本轮观察。", close=r["first_board_close"])

    r1 = []
    for r in ranked[:15]:
        source = r.get("source_record") or {}
        bias = source.get("route_bias_atr")
        if ("R1" in (r.get("route_ids") or "").split("|") and source.get("route_status") in WAIT_R1
                and isinstance(bias, (int, float)) and -1 <= bias <= 2 and r.get("state") not in {"RISK_WATCH", "DATA_WAIT"}):
            r1.append(r)
    def r1_order(r, used):
        main_theme = themes(r["symbol"])[0]
        heat = hot_names.index(main_theme) if main_theme in hot_names else len(hot_names)
        return heat == len(hot_names), main_theme in used, heat, r["participation_rank"]

    for _ in range(2 - len(output)):
        used = {row["themes"][0] for row in output}
        r = min((r for r in r1 if r["symbol"] not in seen), key=lambda r: r1_order(r, used), default=None)
        if r is None:
            break
        source = r["source_record"]
        detail = f"参与榜#{r['participation_rank']}；R1相对20日均线偏离 {source['route_bias_atr']:+.2f} ATR。"
        bias_pct = source.get("ema20_bias_pct")
        if isinstance(bias_pct, (int, float)):
            detail += f"距EMA20 {bias_pct:+.2f}%。"
        watch = "等突破站稳，再看回踩承接。" if source["route_status"] == "WAIT_ACCEPTANCE" else r.get("next_condition")
        add(r, "R1结构关注", detail, watch or "等同轮结构确认，不追高。")

    for r in s4.get("recommended", [])[:1]:
        add(r, "S4形态关注", f"S4回踩再启动；原S4市场规则{'允许复核' if s4.get('buy_advised') else '不允许新开仓'}。",
            "核对本轮回踩低点是否守住和再转强是否延续。", close=r.get("close"))

    height = facts.get("ladder", {}).get("max_height") or 0
    focus = report.get("summary", {}).get("focus_ids", [])
    leaders = [*focus, *hotspot.get("market_leaders", [])]
    for symbol in leaders:
        stock = hot_stocks.get(symbol)
        if not stock or height < 2 or stock.get("height") != height or symbol in seen:
            continue
        add(stock, "热点风向标", f"市场最高梯队 {height} 板；用于判断题材延续，不作为买点。",
            "看分歧时能否守住、同题材独立成员是否有承接；不接加速。", limit=MAX_FOCUS, close=stock.get("close"))
        break
    return {"version": VERSION, "as_of": day, "market_gate": market.get("state") if market.get("data_status") == "FRESH" else "UNKNOWN",
            "formal_review_count": len(formal) if closure else None, "hot_themes": hot_themes, "p5_focus_count": p5.get("focus_count"),
            "p5_unverified_dates": p5.get("unverified_dates", []), "rows": output, "orders_enabled": False}


def render(payload):
    formal = payload["formal_review_count"]
    review = "待核" if formal is None else f"{formal} 只"
    lines = [f"# 每日核心关注 | {payload['as_of']}", "",
             f"市场门：{payload['market_gate']}；正式计划复核：{review}。以下最多 3 只，不补位、不下单。"]
    if payload["hot_themes"]:
        lines.append("热点（封板占比榜）：" + "；".join(f"{t['theme']} {t['limit_up']}/{t['members']}" for t in payload["hot_themes"]) + "。")
        lines.append("个股题材为当日榜单关联，不代表主营核验。")
    if payload["p5_focus_count"] is not None:
        lines.append(f"P5首板重点全表：{payload['p5_focus_count']} 只，详见盘后事实层。")
    if payload["p5_unverified_dates"]:
        lines.append("P5时点缺口：" + "、".join(payload["p5_unverified_dates"]) + "。")
    if payload.get("missing"):
        lines.append("未使用的输入：" + "、".join(payload["missing"]) + "。")
    lines += [""]
    for i, r in enumerate(payload["rows"], 1):
        price = "" if r["close"] is None else f" · 收盘 {r['close']:.2f}"
        lines += [f"{i}. **{r['name']} {r['symbol'][3:]}** · {r['role']}{price} · {' / '.join(r['themes'])}",
                  f"   {r['evidence']} 下一步：{r['next_check']}"]
    if not payload["rows"]:
        lines.append("没有证据齐备的核心关注标的，继续等待。")
    return "\n".join(lines) + "\n"


def load(day, folder, allow_s4=True):
    def dated(name):
        obj = read_json(folder / name)
        return obj if obj.get("as_of") == day else {}

    run = dated("mainwave_run_status.json")
    if run.get("status") != "SUCCESS":
        raise ValueError("本日主流程未成功，不读取已有榜单。")
    steps = {s["name"]: s for s in run.get("steps", [])}

    def archive(step, name, acceptable):
        status = dated(f"{step}/status.json")
        if steps.get(step, {}).get("exit_code") != 0 or status.get("status") not in acceptable:
            return {}
        version = status.get("version")
        if not version:
            return {}
        return read_json(folder / step / version / name)

    closure = dated("trading_closure.json") if steps.get("trading_closure", {}).get("exit_code") == 0 else {}
    facts = archive("eod_review", "facts.json", {"SUCCESS", "PARTIAL"})
    facts = facts if facts.get("as_of") == day else {}
    hotspot = archive("fupan", "hotspot_core.json", {"SUCCESS", "PARTIAL"})
    hotspot = hotspot if hotspot.get("as_of") == day else {}
    report = archive("fupan", "report.json", {"SUCCESS", "PARTIAL"})
    report = report if report.get("meta", {}).get("trade_date") == day else {}
    s4 = dated("s4_opportunities.json") if allow_s4 else {}
    payload = build(day, closure, facts, hotspot, report, s4)
    payload["missing"] = [label for value, label in ((closure, "交易闭环"), (facts, "盘后事实"), (hotspot, "热点榜"), (s4, "S4")) if not value]
    if facts and "p5_limit_up" not in facts:
        payload["missing"].append("P5首板归档（旧版事实包）")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--skip-s4", action="store_true")
    args = parser.parse_args()
    folder = Path("results/mainwave_engine") / args.date.replace("-", "")
    payload = load(args.date, folder, not args.skip_s4)
    report = render(payload)
    _write(folder / "daily_focus.json", payload)
    (folder / "daily_focus.md").write_text(report, encoding="utf-8")
    print(report.replace("**", ""), end="", flush=True)
    print(f"核心关注报告：{folder / 'daily_focus.md'}", flush=True)


if __name__ == "__main__":
    main()

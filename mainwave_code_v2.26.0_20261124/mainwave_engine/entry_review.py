"""Report-only admission checks; keep frozen rankings and discovery evidence unchanged."""
from pathlib import Path

from . import eod_review as R, theme_risk as TR

VERSION = "entry_review_v1_20261027"


def build(day, snapshot, mapping, symbols, known_at):
    records = {r["symbol"]: r for r in snapshot.get("records", [])}
    market = snapshot.get("market") or {}
    market_ok = (snapshot.get("as_of") == day and market.get("as_of") == day
                 and market.get("data_status") == "FRESH" and market.get("trend_review_allowed") is True)
    rows = []
    for symbol in symbols:
        record = records.get(symbol) or {}
        themes = list(mapping.get(symbol, {}).get("concepts", [])) + [record.get("theme", "")]
        risk = TR.evaluate_many(themes, day, known_at)
        reasons = list(record.get("blockers") or [])
        if not market_ok:
            reasons.append("MARKET_WAIT")
        if not record or record.get("price_date") != day:
            reasons.append("NO_FRESH_ENGINE_RECORD")
        if record.get("paper_plan_eligible") is not True or record.get("state") != "PLAN_REVIEW":
            reasons.append("NO_ENTRY_PERMISSION")
        if (record.get("window") or {}).get("active") is not True:
            reasons.append("NO_ACTIVE_WINDOW")
        if risk["theme_risk_action"] != "ALLOW":
            reasons.append("THEME_RISK_REVIEW")
        rows.append({"symbol": symbol, "eligible": not reasons, "reasons": list(dict.fromkeys(reasons)),
                     "rank": record.get("participation_rank"), "theme_risk": risk})
    queue = sorted([r for r in rows if r["eligible"]], key=lambda r: (r["rank"] or 10**9, r["symbol"]))
    return {"as_of": day, "known_at": known_at, "version": VERSION, "orders_enabled": False,
            "market_allowed": market_ok, "rows": rows, "top": [r["symbol"] for r in queue[:2]],
            "eligible_count": len(queue), "note": "沿用原参与排序，仅增加报告复核；不是收益预测或买入指令，不补位凑数。"}


def cooling(facts, manifest, out_root, known_at):
    """Three consecutive archived sessions, same mapping; absence from a top list is unknown, not zero."""
    series = [(facts, manifest)]
    sources = []
    while len(series) < 3:
        prior, source = R.previous_review(out_root, series[-1][0].get("previous_day"), known_at)
        if not prior:
            break
        previous_manifest = R.read_json(Path(source["path"]).parent / "manifest.json")
        series.append((prior, previous_manifest))
        sources.append(source)
    series.reverse()
    mapping_ids = [m.get("sources", {}).get("concept_mapping", {}).get("sha256") for _, m in series]
    comparable = len(series) == 3 and all(mapping_ids) and len(set(mapping_ids)) == 1
    tables = [{r["theme"]: r for key in ("top_by_limit_up", "top_by_limit_share", "top_by_amount_share")
               for r in f.get("themes", {}).get(key, [])} for f, _ in series]
    names = sorted({name for table in tables for name in table})
    result = []
    for name in names:
        points = [{"date": f["as_of"], "limit_up": table.get(name, {}).get("limit_up"),
                   "breadth": table.get(name, {}).get("breadth", {}).get("value")} for (f, _), table in zip(series, tables)]
        complete = comparable and all(p["limit_up"] is not None and p["breadth"] is not None for p in points)
        declines = complete and all(points[i]["limit_up"] < points[i-1]["limit_up"]
                                    and points[i]["breadth"] < points[i-1]["breadth"] for i in (1, 2))
        result.append({"theme": name, "status": "COOLING" if declines else "NO_CONTINUOUS_COOLING" if complete else "UNKNOWN",
                       "points": points})
    return {"rows": result, "sources": sources, "mapping_comparable": comparable,
            "note": "连续三交易日涨停数与上涨广度均回落只提示绝对热度降温；未剔除大盘共同下跌，不等于独立题材退潮，未校准为硬门槛。缺档/映射变化/跌出榜单记未知，不证明题材健康。"}

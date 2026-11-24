"""Report-only discovery union. Market leadership never grants trading permission."""
from pathlib import Path

from . import eod_review as R
from .review_clock import instant

VERSION = "hotspot_core_observation_v3_20261029"
FILES = {"strong": ("strong_trend_shadow.json", "strong_trend_shadow"),
         "attribution": ("theme_attribution.json", "limit_theme"),
         **{f"s{i}": (f"s{i}_observations.json", "s3_observations" if i == 3 else "scenario_observations") for i in (1, 2, 3, 5)},
         "s4": ("s4_opportunities.json", "s4_opportunities")}


def load_inputs(folder, day, cutoff=None):
    """Reject old dates and failed current-run outputs, including old files left by a failed rerun."""
    folder = Path(folder)
    run = R.read_json(folder / "mainwave_run_status.json")
    outer = R.read_json(folder / "daily_full_status.json")
    steps = {s["name"]: s["status"] for s in outer.get("steps", [])
             if s["name"] in ("s3_observations", "s4_opportunities")} if outer.get("as_of") == day else {}
    if run.get("as_of") == day:
        steps.update({s["name"]: s["status"] for s in run.get("steps", [])})
    inputs, gaps = {}, []
    for key, (name, step) in FILES.items():
        data = R.read_json(folder / name)
        if key == "attribution":
            health = R.read_json(folder / "limit_theme/status.json")
            if (health.get("as_of") != day or health.get("status") not in {"SUCCESS", "PARTIAL"}
                    or not health.get("scope", {}).get("full_limit_up_coverage") or health.get("path") != data.get("report_path")):
                gaps.append(f"{name}: no matching completed full-coverage attribution run")
                continue
            if cutoff is not None:
                published = instant(health.get("available_at"))
                artifact = Path(health["path"]) / "report.json"
                if (not published or not instant(cutoff) or published > instant(cutoff)
                        or not health.get("sha256") or R.file_sha(artifact) != health["sha256"]):
                    gaps.append(f"{name}: publication time/hash not proven at review cutoff")
                    continue
                data = {**R.read_json(artifact), "report_path": health["path"], "available_at": health["available_at"],
                        "sha256": health["sha256"]}
        allowed = (None, "SUCCESS", "WARNING") if key == "attribution" else (None, "SUCCESS")
        if steps.get(step) not in allowed or data.get("as_of") != day or data.get("missing_source") or data.get("status") == "FAILED":
            gaps.append(f"{name}：当日扫描缺失、失败或日期不符")
        else:
            inputs[key] = data
    # S1/S2 are projections of the strong scan, not independent confirmation.
    if "strong" not in inputs:
        for key in ("s1", "s2"):
            inputs.pop(key, None)
    return inputs, gaps


def hot_themes(facts):
    themes = {}
    for key in ("top_by_limit_up", "top_by_limit_share"):
        for item in facts.get("themes", {}).get(key, [])[:5]:
            if not item.get("generic"):
                themes[item["theme"]] = item
    return sorted(themes.values(), key=lambda t: (-t["limit_up"], t.get("members", 0), t["theme"]))


def build(day, rows, facts, mapping, records, inputs):
    themes = hot_themes(facts)
    strong = {r["symbol"]: r for r in inputs.get("strong", {}).get("boards", {}).get("attention", [])}
    scenarios = {}
    for key, data in inputs.items():
        if key in ("strong", "attribution"):
            continue
        for r in data.get("recommended" if key == "s4" else "observations", []):
            scenarios.setdefault(r["symbol"], []).append(key.upper())
    up, failed = set(facts["boards"]["U"]), set(facts["boards"]["Z"])
    heights = facts["ladder"]["heights"]
    attribution = inputs.get("attribution", {})
    attributed = {r["symbol"]: r for r in attribution.get("rows", []) if r.get("status") == "CORROBORATED"}
    events = facts.get("loss_effects", {}).get("events", {})
    result, unavailable = [], []
    for symbol in sorted(up | failed | set(strong) | set(scenarios)):
        matched = [t for t in themes if t["theme"] in mapping.get(symbol, {}).get("concepts", [])]
        height = int(heights.get(symbol, 0) or 0)
        if not matched and height < 2 and symbol not in strong and symbol not in scenarios and symbol not in attributed:
            continue
        row = rows.get(symbol)
        f = row.get("frame") if row else None
        if row is None or not row.get("in_domain", True) or f is None or f.empty or str(f.iloc[-1].date) != day or not bool(f.iloc[-1].valid):
            unavailable.append(symbol)
            continue
        b = f.iloc[-1]
        rec, core = records.get(symbol, {}), strong.get(symbol, {})
        state = (rec.get("evidence") or {}).get("state_vector") or {}
        roles = ([f"连板核心观察（{height}板）"] if height >= 2 else ["首板观察（不要求20日新高）"] if symbol in up else [])
        if symbol in failed:
            roles.append("触板未封住")
        if core:
            roles.append("趋势再转强" if core.get("entry") == "REACCELERATION" else "趋势突破")
        signals = sorted(scenarios.get(symbol, []))
        roles.extend(signals)
        risks = [label for key, label in (("gap_up_reversal", "高开低走"), ("breakdown_20d", "跌破20日低点"))
                 if symbol in events.get(key, {}).get("symbols", [])]
        if float(b.high) == float(b.low):
            risks.append("一字价：不能假设可成交，也无日内承接证据")
        if symbol in up:
            risks.append("收盘涨停：次日成交与追价风险待核验")
        if height >= 2:
            risks.append("连续涨停已扩张，不等于低位埋伏或可成交买点")
            if not matched:
                risks.append("连板事实已识别，但未匹配热门题材；关联归因待核验")
        links = mapping.get(symbol, {}).get("_membership_evidence", [])
        risks.extend(link["risk_note"] for link in links if link.get("risk_note"))
        if any(link.get("review_on") and day >= link["review_on"] for link in links):
            risks.append("题材关联证据已到复核日期，需要重新核验")
        inferred = attributed.get(symbol)
        if inferred:
            risks.append("模型归因已获交叉支持，但不代表主营、唯一因果或买入许可")
        # A risk overlay and a leadership role can coexist; neither erases the other.
        risks.extend(str(x) for x in rec.get("blockers", []))
        risks.extend(x.get("label") or x["code"] for x in state.get("risk_overlays", []))
        result.append({"symbol": symbol, "name": mapping.get(symbol, {}).get("name") or core.get("name") or rec.get("name") or symbol,
                        "themes": list(dict.fromkeys([t["theme"] for t in matched] + ([inferred["primary_theme"]] if inferred else []))),
                        "attribution": {"theme": inferred["primary_theme"], "status": inferred["status"], "known_at": attribution.get("available_at") or attribution.get("known_at"),
                                        "trade_date": day,
                                        "source": attribution.get("report_path")} if inferred else None,
                        "roles": roles, "scenarios": signals,
                        "theme_evidence": links,
                        "ema20_bias_pct": (rec.get("source_record") or {}).get("ema20_bias_pct"),
                       "height": height, "close": float(b.close), "high": float(b.high), "low": float(b.low),
                       "change_pct": round((float(b.close) / float(b.preclose) - 1) * 100, 2) if b.preclose > 0 else None,
                       "amount_yi": round(float(b.amount) / 1e8, 2),
                       "lifecycle": " / ".join(filter(None, [state.get("phase_label"), state.get("stage_label")])) or "未进入三路径",
                       "engine_state": rec.get("state") or "NOT_IN_CANDIDATES", "risks": list(dict.fromkeys(risks)),
                       "permission": "仅观察；正式许可仍由原冻结合约和风控决定", "orders_enabled": False})
    # Separate role lanes avoid making first boards / early setups compete solely with mature winners.
    order = lambda r: (-r["height"], -(r["change_pct"] or 0), -r["amount_yi"], r["symbol"])
    result.sort(key=order)
    leaders = [r for r in result if r["height"] >= 2]
    # Market-space leaders must survive missing or incorrect theme membership.
    focus = [r["symbol"] for r in leaders if r["height"] == leaders[0]["height"]][:15] if leaders else []
    for theme in themes:
        members = [r for r in result if theme["theme"] in r["themes"]]
        for lane in (members, [r for r in members if r["height"] == 1], [r for r in members if r["scenarios"] and not r["height"]]):
            pick = next((r["symbol"] for r in lane if r["symbol"] not in focus), None)
            if pick and len(focus) < 15:
                focus.append(pick)
    early = [r["symbol"] for r in result if "S5" in r["scenarios"]]
    return {"as_of": day, "version": VERSION, "orders_enabled": False, "themes": themes, "rows": result,
            "market_leaders": [r["symbol"] for r in leaders],
            "mapping_review": [{"symbol": r["symbol"], "height": r["height"], "hot_matches": r["themes"],
                                "question": "当日主炒题材是否与成员标签一致？叙事与经营证据是否区分？"} for r in leaders],
            "focus": focus, "early_setups": early, "unavailable": unavailable,
            "scope": "热点U/Z+独立连板核心（不依赖题材标签）+当日强势观察+S1至S5合格输出；不包含S4排除项",
            "ranking": "先保留市场最高连板，再按题材高度核心、首板、非连板结构分道提名；全局去重最多15只；仅观察，未经收益验证",
            "mapping_basis": "基础概念映射非历史PIT；证据关联按获知/生效时间叠加，保留来源与关联类型；市场叙事不等于主营，标签并列不证明传播因果"}

"""EOD-only observation table. Entry permission is copied, never inferred from heat."""
from collections import Counter, defaultdict

from . import theme_registry
from .eod_review import digest
from .review_clock import instant

VERSION = "hotspot_funnel_shadow_v1_20261118"
ALIASES = {"华为昇腾具身智能算力": "华为算力"}
LABELS = {"CORROBORATED": "交叉支持", "DISPUTED": "叙事分歧", "MODEL_FAILED": "模型未完成",
          "VALIDATION_FAILED": "引文或契约失败", "REPRINT": "同源转载", "ONE_SOURCE": "单源待核",
          "INSUFFICIENT_MATERIAL": "原文不足", "DATA_WAIT": "归因档缺失或未发布"}


def reason(row):
    if not row:
        return "DATA_WAIT"
    status = row.get("processing")
    if status in {"MODEL_FAILED", "VALIDATION_FAILED", "INSUFFICIENT_MATERIAL"}:
        return status
    # Read older frozen reports without turning their processing errors into market evidence.
    errors = row.get("unknowns") or []
    for prefix, code in (("模型未完成", "MODEL_FAILED"), ("模型引用或格式未通过", "VALIDATION_FAILED")):
        if any(err.startswith(prefix) for err in errors):
            return code
    if row.get("status") in {"CORROBORATED", "DISPUTED"}:
        return row["status"]
    if row.get("primary_has_reprint"):
        return "REPRINT"
    if len((row.get("verification") or {}).get("media_origin_groups", [])) == 1:
        return "ONE_SOURCE"
    return "INSUFFICIENT_MATERIAL"


def build(day, cutoff, facts, core, attribution, admission, relay, records=None):
    if not instant(cutoff) or instant(cutoff) < instant(day + "T15:00:00+08:00"):
        raise ValueError("EOD funnel requires a post-close decision cutoff")
    published = instant(attribution.get("available_at"))
    valid = (attribution.get("as_of") == day and published and instant(cutoff)
             and published <= instant(cutoff) and attribution.get("scope", {}).get("full_limit_up_coverage"))
    attributed = {r["symbol"]: r for r in attribution.get("rows", [])} if valid else {}
    stocks = {r["symbol"]: r for r in core["rows"]}
    reviewed = {r["symbol"]: r for r in admission.get("rows", [])}
    registered = {r["label"]: r for r in theme_registry.themes_as_of(day)}
    groups, rows = defaultdict(list), []
    for s in sorted(set(facts["boards"]["U"])):
        r, a, e = stocks.get(s, {}), attributed.get(s, {}), reviewed.get(s, {})
        grade = reason(a)
        theme = a.get("primary_theme")
        canonical = ALIASES.get(theme, theme) if grade == "CORROBORATED" else None
        chain_id = (registered.get(canonical, {}).get("id") or "narrative:" + digest(canonical)[:16]) if canonical else None
        track = relay["followup"].get(s, {})
        route_states = track.get("routes", [])
        record = (records or {}).get(s) or {}
        source = record.get("source_record") or {}
        if not route_states and record.get("price_date") == day and source.get("route_ids") and source.get("route_status"):
            route_states = [{"route": source["route_ids"], "state": source["route_status"], "reason": source["route_status"],
                             "structure": source.get("breakout_status"), "basis": "FROZEN_CLOSURE"}]
        support = [c["evidence_id"] for h in a.get("hypotheses", []) if h["theme"] == theme for c in h["support"]]
        reasons = list(e.get("reasons") or [])
        if not e:
            reasons.append("NOT_IN_FORMAL_REVIEW")
        if admission.get("market_allowed") is not True and "MARKET_WAIT" not in reasons:
            reasons.append("MARKET_WAIT")
        item = {"symbol": s, "name": a.get("name") or r.get("name") or s,
                "in_hotspot_union": s in stocks, "original_focus": s in core.get("focus", []),
                "materials": len(a.get("material_ids", [])) if a else None, "grade": grade,
                "reported_theme": theme, "chain_id": chain_id, "chain": canonical, "evidence_ids": support,
                "alternative_themes": [h["theme"] for h in a.get("hypotheses", []) if h["theme"] != theme],
                "business": (a.get("business_reality") or {}).get("status"),
                "height": facts.get("ladder", {}).get("heights", {}).get(s, 1),
                "amount_yi": r.get("amount_yi"), "role": None,
                "route_states": route_states, "engine_state": r.get("engine_state"),
                "risks": r.get("risks", []), "position_bias_pct": r.get("ema20_bias_pct"),
                "review_eligible": e.get("eligible"), "blockers": reasons}
        rows.append(item)
        if canonical:
            groups[canonical].append(item)
    roles, diffusion = [], []
    for theme, members in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        ranked = sorted(members, key=lambda r: (-r["height"], -(r["amount_yi"] or 0), r["symbol"]))
        feedback = next((c["state"] for c in relay.get("chains", []) if c["theme"] == theme), None)
        active = len(members) >= 2 or feedback in {"仅原班封板", "接力扩散观察"}
        for label, choices in (("连板高度", [r for r in ranked if r["height"] >= 2]),
                               ("成交核心观察", sorted((r for r in ranked if r["amount_yi"]), key=lambda r: -r["amount_yi"])),
                               ("首板承接观察", [r for r in ranked if r["height"] == 1])):
            pick = next((r for r in choices if not r["role"]), None)
            if pick and (active or label == "连板高度"):
                pick["role"] = label
                roles.append(pick["symbol"])
        reg = registered.get(theme)
        # Source-linked product/business members can seed a new chain; ordinary concept tags cannot.
        linked = {s for s, stock in stocks.items() for link in stock.get("theme_evidence", [])
                  if link.get("relation_kind") in {"PRODUCT_APPLICATION", "CONFIRMED_BUSINESS"}
                  and theme in {ALIASES.get(t, t) for t in link.get("themes", [])}
                  and link.get("source_urls") and instant(link.get("known_at"))
                  and instant(link["known_at"]) <= instant(cutoff)
                  and link.get("effective_from", "9999") <= day <= link.get("expires_after", "9999")}
        nonleaders = sorted((linked | set((reg or {}).get("members", {}))) - {r["symbol"] for r in members})
        diffusion.append({"chain_id": ranked[0]["chain_id"], "theme": theme, "anchors": [r["symbol"] for r in ranked],
                          "independent_members": nonleaders, "feedback": feedback,
                          "stage": ("NO_VERIFIED_INDEPENDENT_MEMBERS" if not nonleaders else
                                    "WAIT_CHAIN_FOLLOWTHROUGH" if feedback != "接力扩散观察" else "CHECK_CARRIER_RESPONSE"),
                          "eligible": False})
    height = max((r["height"] for r in rows), default=0)
    emotion = [r["symbol"] for r in rows if height >= 2 and r["height"] == height and not r["chain"]]
    for r in rows:
        if r["symbol"] in emotion:
            r["role"] = "情绪高度观察（不认定产业链）"
    permission = (admission.get("market_allowed") is True and admission.get("as_of") == day)
    return {"version": VERSION, "as_of": day, "attribution_available_at": attribution.get("available_at") if valid else None,
            "source": attribution.get("report_path") if valid else None, "source_sha256": attribution.get("sha256") if valid else None,
            "universe": len(rows), "grades": dict(Counter(r["grade"] for r in rows)), "rows": rows,
            "role_watch": roles, "emotion_watch": emotion, "r3_shadow": diffusion, "orders_enabled": False,
            "participation_top": [s for s in admission.get("top", []) if reviewed.get(s, {}).get("eligible") is True][:2] if permission else []}


def table_rows(funnel):
    for r in sorted(funnel["rows"], key=lambda r: (not r["role"], r["chain"] or "", r["symbol"])):
        state = "、".join(x["reason"] for x in r["route_states"]) or r["engine_state"] or "无已核路线"
        blockers = "、".join(r["blockers"]) or "仅依原合约复核"
        yield ["逐股对账", f"{r['name']}（{r['symbol']}）", r["chain"] or r["reported_theme"] or "待核",
               (r["role"] or "未列角色") + " / " + LABELS[r["grade"]], "--", r["amount_yi"], state, blockers]


def forward_check(previous, facts):
    """Compare only actual previous publications within the frozen prior limit-up cohort."""
    funnel, pool = previous.get("funnel") or {}, facts.get("yesterday_pool") or {}
    if (funnel.get("version") != VERSION or funnel.get("as_of") != facts.get("previous_day")
            or pool.get("basis") != "FROZEN_PREVIOUS_REVIEW"):
        return {"status": "NO_COMPARABLE_FROZEN_PAIR"}
    domain = set(pool["members"])
    old = [s for s in previous.get("focus", []) if s in domain]
    new = [s for s in funnel["role_watch"] if s in domain]
    k = min(15, len(old), len(new))
    if not k:
        return {"status": "EMPTY_COMPARISON", "old_count": len(old), "shadow_count": len(new)}
    by = {r["symbol"]: r["close_vs_pre"] for r in pool.get("performance", [])
          if r.get("status") == "OK" and r.get("close_vs_pre") is not None}
    def observed(picks):
        values = [by[s] for s in picks if s in by]
        return {"symbols": picks, "observed": len(values), "missing": len(picks)-len(values),
                "positive": sum(v > 0 for v in values), "mean_pct": sum(values)/len(values) if values else None}
    return {"status": "OBSERVED", "signal_day": funnel["as_of"], "equal_budget": k,
            "basis": "昨日封板子池的收盘表现，不是可成交利润或全市场核心召回",
            "old_focus": observed(old[:k]), "shadow_roles": observed(new[:k])}

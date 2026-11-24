"""Market policy in dimensions, beside the frozen colour gate.

The frozen gate (market_context.state GREEN/ORANGE/RED → trend_review_allowed) stays the baseline and still
decides every paper permission. This module only reads the environment in separate dimensions and says,
per return engine, what a shadow policy would allow so the two can be compared day by day. Nothing here
lifts a gate. Readings are relative (20-day percentiles from eod_review) and dated by their source."""
VERSION = "market_policy_shadow_v1_20260923"
HOT, COLD = ("热", "偏热"), ("冷", "偏冷")


def _reading(metrics, key):
    m = (metrics or {}).get(key) or {}
    return m.get("reading") if m.get("status") == "OK" else None


def dimensions(market_context, sentiment_metrics=None, loss_events=None, sentiment_as_of=None):
    """Index structure, relay sentiment, theme persistence, loss effect, system health; each with its inputs."""
    ctx = market_context or {}
    s = sentiment_metrics or {}
    hot = sum(1 for k in ("limit_up", "pool_mean_pct", "pool_up_share", "open_premium_pct", "promotion_1_to_2") if _reading(s, k) in HOT)
    cold = sum(1 for k in ("limit_up", "pool_mean_pct", "pool_up_share", "open_premium_pct", "promotion_1_to_2") if _reading(s, k) in COLD)
    seal, ld = _reading(s, "seal_failure_rate"), _reading(s, "limit_down")
    neg = ((loss_events or {}).get("yesterday_pool_negative") or {}).get("count")
    return {"index_structure": {"reading": ctx.get("structure") or "UNKNOWN", "temperature": ctx.get("temperature"), "frozen_state": ctx.get("state"),
                                "fresh": ctx.get("data_status") == "FRESH"},
            "relay_sentiment": {"reading": "HOT" if hot >= 3 else "COLD" if cold >= 3 else "NEUTRAL" if s else "UNKNOWN", "hot": hot, "cold": cold, "as_of": sentiment_as_of},
            "loss_effect": {"reading": ("CONTAINED" if seal in HOT and ld in HOT else "SPREADING" if seal in COLD and ld in COLD else "LOCAL" if s else "UNKNOWN"),
                            "yesterday_pool_negative": neg, "as_of": sentiment_as_of},
            "system_health": {"reading": "OK" if ctx.get("data_status") == "FRESH" else "DEGRADED", "issues": ctx.get("issues") or []}}


def engine_policy(dims):
    """Per-engine shadow verdict. The frozen colour gate is reported beside it and remains authoritative."""
    idx, relay, loss, health = dims["index_structure"], dims["relay_sentiment"], dims["loss_effect"], dims["system_health"]
    hard_block = health["reading"] != "OK" or not idx.get("fresh")
    frozen_allows = idx.get("frozen_state") == "GREEN"
    out = {"hard_gate": "BLOCK" if hard_block else "PASS", "frozen_gate": idx.get("frozen_state") or "UNKNOWN", "frozen_allows_new_plans": frozen_allows}
    if hard_block:
        for e in ("E1", "E2", "E3"):
            out[e] = {"shadow": "BLOCK", "reason": "系统健康或数据不新鲜；硬门优先"}
        return out
    e1 = "ALLOW_SHADOW" if idx["reading"] in ("BULL", "RANGE") and loss["reading"] != "SPREADING" else "WAIT"
    e3 = "ALLOW_SHADOW" if relay["reading"] == "HOT" and loss["reading"] in ("CONTAINED", "LOCAL") else "WAIT"
    e2 = "RESEARCH_ONLY"
    out["E1"] = {"shadow": e1, "reason": f"指数结构 {idx['reading']}，亏钱效应 {loss['reading']}"}
    out["E3"] = {"shadow": e3, "reason": f"接力情绪 {relay['reading']}，亏钱效应 {loss['reading']}"}
    out["E2"] = {"shadow": e2, "reason": "缺 14:30 后快照，只研究"}
    out["divergence_from_frozen"] = [e for e in ("E1", "E3") if (out[e]["shadow"] == "ALLOW_SHADOW") != frozen_allows]
    return out


def evaluate(market_context, sentiment_metrics=None, loss_events=None, sentiment_as_of=None):
    dims = dimensions(market_context, sentiment_metrics, loss_events, sentiment_as_of)
    return {"version": VERSION, "dimensions": dims, "policy": engine_policy(dims),
            "note": "影子政策：冻结颜色闸门仍决定一切纸面许可；本表只记录分维度读数与各引擎的影子结论，供逐日对照。"}

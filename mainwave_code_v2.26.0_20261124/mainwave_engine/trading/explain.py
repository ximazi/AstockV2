"""Rank explanation: components, comparison domain and blockers for any position."""
from collections import Counter
import math
from . import ranking, store

VERSION = "rank_explain_v1_20260920"
WEIGHTS = ranking.RULES["strength"]
COMPONENTS = {"relative_momentum": "相对动量", "drawdown20": "20日回撤", "efficiency10": "10日推进效率"}
# Saved but deliberately outside the score until D05 proves an increment beyond the baseline.
OBSERVED_ONLY = {"close_location": "收盘位置", "relative_turnover": "相对换手", "turnover": "换手率"}
BLOCKER_LABELS = {
    "DATA_WAIT": "数据待核", "STRUCTURE_RISK": "结构风险", "MARKET_WAIT": "市场闸门未放行",
    "WAIT_PULLBACK_CONFIRMATION": "回踩确认未成立", "TRACKING_ONLY": "仅续踪观察",
    "BENCHMARK_MISSING": "基准缺失",
}
ROLE_LABELS = {"DAILY_LEADER_CANDIDATE": "趋势领头候选", "RELATIVE_STRENGTH_FRONT": "情绪前排",
               "UNCONFIRMED": "角色未确认"}


def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if isinstance(x, (int, float)) and isinstance(y, (int, float))
             and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 10:
        return None, len(pairs)
    n = len(pairs)
    mx, my = sum(p[0] for p in pairs) / n, sum(p[1] for p in pairs) / n
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if not sx or not sy:
        return None, n
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    return round(cov / (sx * sy), 4), n


def collinearity(payload):
    """Disclose overlap before any future version adds both volume and turnover."""
    rows = [r["features"] for r in payload["records"] if r["features"].get("status") == "READY"]
    value, n = _pearson([f.get("turnover") for f in rows], [f.get("relative_turnover") for f in rows])
    momentum, m = _pearson([f.get("relative_momentum") for f in rows], [f.get("return20") for f in rows])
    return {"turnover_vs_relative_turnover": {"pearson": value, "sample": n},
            "relative_momentum_vs_return20": {"pearson": momentum, "sample": m},
            "policy": "当前分数只用相对动量、回撤与推进效率；换手与收盘位置仅观察，"
                      "避免量能与换手重复加分。新增特征须经 D05 对照后才考虑进入权重。"}


def explain(record, payload, previous=None):
    """Everything needed to reconstruct one row's rank, without reading the source code."""
    features = record.get("features") or {}
    group = record.get("comparison_group") or {}
    peers = [r for r in payload["records"]
             if r["theme"] == record["theme"] and r["symbol"] != record["symbol"]
             and (r.get("features") or {}).get("status") == "READY"]
    contributions = []
    if record.get("strength_score") is not None:
        cohort = [r for r in payload["records"]
                  if (r.get("comparison_group") or {}).get("board") == group.get("board")
                  and r.get("strength_score") is not None]
        stored = record.get("score_percentiles") or {}
        for key, label in COMPONENTS.items():
            mine = features.get(key)
            if key in stored:
                percentile, basis = stored[key], "SCORED_PERCENTILE"
            else:
                # Older snapshots: the score's own tie rule (average rank), so ties explain the way they scored.
                values = [r["features"][key] for r in cohort if r["features"].get(key) is not None]
                below = sum(1 for v in values if v < mine) if mine is not None else 0
                at_or_below = sum(1 for v in values if v <= mine) if mine is not None else 0
                percentile = ((below + 1 + at_or_below) / 2 / len(values)) if values and mine is not None else None
                basis = "RECOMPUTED_AVERAGE_TIE"
            weight = WEIGHTS["relative_momentum" if key == "relative_momentum" else
                             "drawdown" if key == "drawdown20" else "efficiency"]
            contributions.append({
                "component": key, "label": label, "value": mine, "weight": weight, "basis": basis,
                "percentile": percentile,
                "points": round(100 * percentile * weight, 2) if percentile is not None else None})
    # Summed before per-row rounding, so three half-cents never read as a mismatch.
    points = float(sum(100 * c["percentile"] * c["weight"] for c in contributions if c["percentile"] is not None))
    consistency = ({"points_sum": round(points, 2), "score": record["strength_score"],
                    "consistent": bool(abs(points - float(record["strength_score"])) <= .05),
                    "note": "贡献取自评分用的同一分位；合计与总分不一致即解释失效"}
                   if contributions else None)
    observed = {k: {"label": v, "value": features.get(k), "in_score": False} for k, v in OBSERVED_ONLY.items()}

    previous_record = None
    if previous:
        previous_record = next((r for r in previous.get("records", [])
                                if r["opportunity_id"] == record["opportunity_id"]), None)
    change = {"previous_date": (previous or {}).get("as_of"), "previous_rank": record.get("previous_rank"),
              "current_rank": record.get("strength_rank"), "rank_change": record.get("rank_change"),
              "previous_score": (previous_record or {}).get("strength_score"),
              "score_change": None, "attribution": "NO_PRIOR_ROUND"}
    if previous_record and previous_record.get("strength_score") is not None and record.get("strength_score") is not None:
        change["score_change"] = round(record["strength_score"] - previous_record["strength_score"], 2)
        prior_cohort = sum(1 for r in previous.get("records", []) if r.get("strength_score") is not None)
        now_cohort = sum(1 for r in payload["records"] if r.get("strength_score") is not None)
        change["cohort"] = {"previous": prior_cohort, "current": now_cohort, "delta": now_cohort - prior_cohort}
        rank_down = (record.get("rank_change") or 0) < 0
        if change["score_change"] > 0 and rank_down:
            change["attribution"] = "COHORT_STRENGTHENED"
            change["reading"] = (f"自身分数上升 {change['score_change']}，名次仍下降 "
                                 f"{abs(record['rank_change'])} 位：比较域内其他候选改善更快")
        elif change["score_change"] < 0 and not rank_down:
            change["attribution"] = "COHORT_WEAKENED"
            change["reading"] = "自身分数下降但名次上升：比较域整体走弱，不是自身改善"
        else:
            change["attribution"] = "OWN_MOVE"
            change["reading"] = "名次变化方向与自身分数一致"

    blockers = [{"code": b, "label": BLOCKER_LABELS.get(b, b)} for b in record.get("blockers") or []]
    strong_but_unavailable = (record.get("strength_rank") is not None and record["strength_rank"] <= 15
                              and record.get("state") != "PLAN_REVIEW")
    return {"version": VERSION, "opportunity_id": record["opportunity_id"], "symbol": record["symbol"],
            "name": record.get("name"), "as_of": payload["as_of"],
            "strength": {"rank": record.get("strength_rank"), "score": record.get("strength_score"),
                         "question": "谁在走强", "contributions": contributions, "consistency": consistency,
                         "observed_not_scored": observed},
            "participation": {"rank": record.get("participation_rank"), "state": record.get("state"),
                              "question": "谁现在有可复核位置", "blockers": blockers,
                              "eligible": bool(record.get("paper_plan_eligible")),
                              "tracking_only": bool(record.get("tracking_only")),
                              "bias_atr": (record.get("source_record") or {}).get("route_bias_atr"),
                              "next_condition": record.get("next_condition")},
            "comparison_domain": {"board": group.get("board"), "board_count": group.get("count"),
                                  "limited": group.get("limited"),
                                  "theme": record["theme"], "theme_peer_count": len(peers),
                                  "theme_peers_used": len(peers) >= ranking.RULES["min_peer_count"],
                                  "self_excluded": True,
                                  "note": "分数按板块分组的分位计算；跨组最高分不可直接等量比较。"},
            "role": {"code": record.get("role"), "label": ROLE_LABELS.get(record.get("role"), record.get("role")),
                     "confirmed": bool(record.get("leadership_confirmed")),
                     "evidence_rows": len(record.get("role_evidence") or []),
                     "note": "分钟数据不足时只输出日级领先线索，不作因果带动结论。"},
            "change": change,
            "verdict": ("STRONG_BUT_NOT_ACTIONABLE" if strong_but_unavailable else
                        "ACTIONABLE_FOR_REVIEW" if record.get("paper_plan_eligible") else "OBSERVE"),
            "verdict_reading": ("排名靠前但当前不适合参与：" + "、".join(b["label"] for b in blockers)
                                if strong_but_unavailable and blockers else
                                "条件齐备，可进入人工复核" if record.get("paper_plan_eligible") else
                                "继续观察，等待下一条件")}


def board(payload, previous=None, limit=15):
    """Top lists are deduplicated by symbol while every theme link is kept."""
    ranked = [r for r in payload["records"] if r.get("strength_rank") is not None]
    strength = sorted(ranked, key=lambda r: r["strength_rank"])
    participation = sorted(payload["records"], key=lambda r: r["participation_rank"])
    themes = {}
    for record in payload["records"]:
        themes.setdefault(record["theme"], []).append(record["symbol"])

    def rows(source):
        seen, out = set(), []
        for record in source:
            if record["symbol"] in seen:
                continue
            seen.add(record["symbol"])
            out.append({"symbol": record["symbol"], "name": record.get("name"),
                        "opportunity_id": record["opportunity_id"], "theme": record["theme"],
                        "themes": sorted({t for t, syms in themes.items() if record["symbol"] in syms}),
                        "route_ids": record.get("route_ids", ""), "state": record.get("state"),
                        "role": record.get("role"), "strength_rank": record.get("strength_rank"),
                        "participation_rank": record.get("participation_rank"),
                        "strength_score": record.get("strength_score"),
                        "rank_change": record.get("rank_change"),
                        "blockers": record.get("blockers") or []})
            if len(out) >= limit:
                break
        return out

    explained = {r["opportunity_id"]: explain(r, payload, previous)
                 for r in strength[:limit] + participation[:limit]}
    strong_blocked = [e for e in explained.values() if e["verdict"] == "STRONG_BUT_NOT_ACTIONABLE"]
    return {"version": VERSION, "as_of": payload["as_of"], "snapshot_id": payload.get("snapshot_id", ""),
            "strength_top": rows(strength), "participation_top": rows(participation),
            "explained": explained, "collinearity": collinearity(payload),
            "summary": {"ranked": len(ranked), "strong_but_not_actionable": len(strong_blocked),
                        "explained_inconsistent": sum(1 for e in explained.values()
                                                      if e["strength"]["consistency"] and not e["strength"]["consistency"]["consistent"]),
                        "states": dict(Counter(r["state"] for r in payload["records"])),
                        "roles": dict(Counter(r.get("role") for r in payload["records"]))},
            "notice": "强势榜回答谁在走强，参与榜回答谁现在有可复核位置；高分不能抵消失效、过期或风险否决。"}

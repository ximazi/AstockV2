"""Shadow screening: the same closure records read in layers instead of one blended score.

The frozen rule (ranking.py) averages relative momentum, drawdown and efficiency, all of which reward a
stock that has already run. This module keeps the readings apart and orders candidates by a stated
lexicographic policy rather than by weights. A lexicographic order is still a policy: each earlier layer
overrides every later one, and that choice is written down here, not hidden in a weight.

v2 (2026-09-23, after the independent audit): route order only with event dates; change read from
structured fields, never from prose; re-eligibility tied to the batch date; REBUILDING kept apart from
FAILED; the cohort theme reading named for what it is (candidate cohort, not the market); a separate
market theme context from the full-membership theme facts; every count taken before any truncation;
the shortlist renamed for what it is. Shadow only: it never changes ranking.py's output, states or
permissions, and it holds no order authority."""
from collections import Counter
import statistics

from . import ranking

VERSION = "screening_shadow_v2_20260923"
EARLY = {"PROBED", "ABSORBING", "COILED", "ARMED", "READY", "IGNITION"}
CONTINUATION = {"MARKUP", "PULLBACK"}
LATE = {"OVERHEATED", "DISTRIBUTION_RISK"}
REPAIRING = {"REBUILDING"}
FAILED = {"FAILED"}
LIVE_PREPARATION = {"EVIDENCE_STRONG", "ADVANCING", "ACCEPTED", "OBSERVING", "ATTEMPT"}
STATE_ORDER = {"PLAN_REVIEW": 0, "MARKET_WAIT": 1, "WAIT_SETUP": 2, "RISK_WATCH": 3, "DATA_WAIT": 4}
STAGE_ORDER = {"EARLY": 0, "CONTINUATION": 1, "UNKNOWN": 2, "REPAIRING": 3, "LATE": 4, "FAILED": 5}
MAINLINE_ORDER = {"STRONG": 0, "NEUTRAL": 1, "UNKNOWN": 1, "WEAK": 2}
ROUTE_ORDER = {"R2_THEN_R1": 0, "R1": 1, "R1_AND_R2_UNORDERED": 1, "R2": 2, "R3": 3, "NONE": 4}
POSITION_ORDER = {"IN_BAND": 0, "UNKNOWN": 1, "EXTENDED": 2}
TRIGGER_ORDER = {"ABSORBED": 0, "NEUTRAL": 1, "WEAK_ACTIVE": 2}
CHANGE_ORDER = {"IMPROVING": 0, "PROGRESSING": 0, "UNCHANGED": 1, "NEW": 1, "UNKNOWN": 1, "WEAKENING": 2}
THRESHOLDS = {"extended_bias_atr": 2.0, "in_band_bias_low": -1.0, "strong_close": 0.6, "weak_close": 0.3,
              "absorb_turnover": 1.0, "weak_active_turnover": 1.5, "cohort_min_members": 3, "cohort_top": 3,
              "market_theme_top": 5, "market_breadth_strong": 0.6}
VARIANTS = ("V0_frozen", "V1_tracking", "V2_position", "V3_stage_route", "V4_multiday")
LABELS = {"stage": {"EARLY": "早期/刚启动", "CONTINUATION": "延续中", "UNKNOWN": "阶段未知", "REPAIRING": "失效后修复观察",
                    "LATE": "过热/派发风险", "FAILED": "本轮结构失效"},
          "mainline": {"STRONG": "候选群体较强", "NEUTRAL": "候选群体中性", "UNKNOWN": "候选群体样本不足", "WEAK": "候选群体较弱"},
          "market_theme": {"STRONG": "全市场题材活跃", "NEUTRAL": "全市场题材中性", "UNKNOWN": "全市场题材无当日表", "WEAK": "全市场题材偏弱"},
          "position": {"IN_BAND": "结构带内", "UNKNOWN": "位置未知", "EXTENDED": "偏离过大"},
          "trigger": {"ABSORBED": "强收盘且有承接", "NEUTRAL": "无明确触发", "WEAK_ACTIVE": "放量弱收盘"},
          "route": {"R2_THEN_R1": "先准备后点火（有日期）", "R1": "点火确认", "R1_AND_R2_UNORDERED": "点火与准备并存（先后未证）",
                    "R2": "先行准备", "R3": "题材传播", "NONE": "无当前路径"},
          "change": {"IMPROVING": "证据改善", "PROGRESSING": "阶段推进", "UNCHANGED": "证据延续", "NEW": "首份证据",
                     "UNKNOWN": "变化未知", "WEAKENING": "证据减弱"}}

number = ranking.number


# ---------------------------------------------------------------- theme readings
def theme_cohorts(records):
    """Reading of the candidate cohort itself: are the candidates in this theme still being pushed today?

    The cohort is already a filtered, self-referential sample; the result is named STRONG/WEAK for the cohort
    and never presented as the market's main line. Conditions are checked first, the top-N cut comes last."""
    groups = {}
    for r in records:
        f = r.get("features") or {}
        if f.get("status") != "READY" or r.get("theme") in (None, "", "未归属"):
            continue
        groups.setdefault(r["theme"], []).append(r)
    out = {}
    for theme, rows in groups.items():
        ex5 = [v for v in (number(x["features"].get("market_excess5")) for x in rows) if v is not None]
        strong = sum(1 for x in rows if trigger_of(x["features"]) == "ABSORBED")
        weak = sum(1 for x in rows if trigger_of(x["features"]) == "WEAK_ACTIVE")
        pushing = sum(1 for x in rows if stage_of(x)[0] in ("EARLY", "CONTINUATION"))
        n = len(rows)
        known = n >= THRESHOLDS["cohort_min_members"] and bool(ex5)
        med = round(statistics.median(ex5), 2) if ex5 else None
        out[theme] = {"state": "UNKNOWN" if not known else "NEUTRAL", "members": n, "median_excess5": med,
                      "strong_close": strong, "weak_active": weak, "pushing": pushing,
                      "qualifies_strong": known and med > 0 and strong >= 2 and pushing >= 2,
                      "qualifies_weak": known and med < 0 and weak * 2 >= n,
                      "basis": "CANDIDATE_COHORT_ONLY_NOT_MARKET_BREADTH"}
    strong_pool = sorted((t for t, c in out.items() if c["qualifies_strong"]), key=lambda t: -out[t]["median_excess5"])
    for theme in strong_pool[:THRESHOLDS["cohort_top"]]:
        out[theme]["state"] = "STRONG"
    weak_pool = sorted((t for t, c in out.items() if c["qualifies_weak"] and c["state"] != "STRONG"), key=lambda t: out[t]["median_excess5"])
    for theme in weak_pool[:THRESHOLDS["cohort_top"]]:
        out[theme]["state"] = "WEAK"
    return out


def market_theme_context(theme_facts):
    """Full-membership theme facts (eod_review) by theme name: breadth, limit-ups, table ranks. Dated by the facts."""
    if not theme_facts:
        return {}
    out = {}
    for key in ("top_by_limit_up", "top_by_limit_share", "top_by_amount_share", "chains"):
        for n, t in enumerate(theme_facts.get(key) or [], 1):
            name = t.get("theme")
            if not name:
                continue
            entry = out.setdefault(name, {"members": t.get("members"), "limit_up": t.get("limit_up"), "up": t.get("up"),
                                          "breadth": (t.get("breadth") or {}).get("value"), "ranks": {}, "as_of": theme_facts.get("as_of")})
            entry["ranks"][key] = n
    top = THRESHOLDS["market_theme_top"]
    for name, e in out.items():
        breadth = e["breadth"] if e["breadth"] is not None else 0.0
        listed_top = any(rank <= top for k, rank in e["ranks"].items() if k in ("top_by_limit_up", "top_by_limit_share"))
        e["state"] = ("STRONG" if listed_top and breadth >= THRESHOLDS["market_breadth_strong"]
                      else "WEAK" if breadth < 0.4 and (e["limit_up"] or 0) == 0 else "NEUTRAL")
    return out


# ---------------------------------------------------------------- per-record readings
def trigger_of(f):
    close_location, rel = number(f.get("close_location")), number(f.get("relative_turnover"))
    if close_location is None or rel is None:
        return "NEUTRAL"
    if close_location >= THRESHOLDS["strong_close"] and rel >= THRESHOLDS["absorb_turnover"]:
        return "ABSORBED"
    if close_location <= THRESHOLDS["weak_close"] and rel >= THRESHOLDS["weak_active_turnover"]:
        return "WEAK_ACTIVE"
    return "NEUTRAL"


def position_of(record):
    src = record.get("source_record") or {}
    sv = (record.get("evidence") or {}).get("state_vector") or {}
    bias = number(src.get("route_bias_atr"))
    if bias is None:
        bias = number(((record.get("evidence") or {}).get("route_evidence") or {}).get("bias_atr"))
    if sv.get("location") == "EXTENDED" or (bias is not None and bias > THRESHOLDS["extended_bias_atr"]):
        return "EXTENDED", bias
    if sv.get("location") == "IN_BAND" or (bias is not None and THRESHOLDS["in_band_bias_low"] <= bias <= THRESHOLDS["extended_bias_atr"]):
        return "IN_BAND", bias
    return "UNKNOWN", bias


def stage_of(record):
    sv = (record.get("evidence") or {}).get("state_vector") or {}
    stage = sv.get("stage")
    if stage in EARLY:
        return "EARLY", stage
    if stage in CONTINUATION:
        # A confirmed or forming first pullback right after the start still counts as early participation.
        if sv.get("setup_status") in ("CONFIRMED", "FORMING") and (record.get("source_record") or {}).get("local_role") in ("FIRST_RETEST", "ORIGIN_BREAKOUT"):
            return "EARLY", stage
        return "CONTINUATION", stage
    if stage in LATE:
        return "LATE", stage
    if stage in REPAIRING:
        return "REPAIRING", stage
    if stage in FAILED:
        return "FAILED", stage
    return "UNKNOWN", stage


def route_of(record):
    """Ordered 'prepared then ignited' only when the preparation date is known to precede the attempt date."""
    routes = {x for x in str(record.get("route_ids") or "").split("|") if x}
    re_ = (record.get("evidence") or {}).get("route_evidence") or {}
    if "R1" in routes and ("R2" in routes or re_.get("prepared_before_attempt")):
        prepared_at = re_.get("prepared_at") or re_.get("preparation_first_seen") or ""
        attempt_at = re_.get("attempt_at") or ""
        if re_.get("prepared_before_attempt") is True or (prepared_at and attempt_at and prepared_at < attempt_at):
            return "R2_THEN_R1"
        return "R1_AND_R2_UNORDERED"
    for r in ("R1", "R2", "R3"):
        if r in routes:
            return r
    return "NONE"


def preparation_of(record):
    re_ = (record.get("evidence") or {}).get("route_evidence") or {}
    state = re_.get("preparation_state") or (record.get("source_record") or {}).get("preparation_state") or ""
    return {"state": state, "live": state in LIVE_PREPARATION, "demand": bool(re_.get("demand")),
            "activity_contraction": bool(re_.get("activity_contraction")), "age": number(re_.get("preparation_age")),
            "prepared_at": re_.get("prepared_at") or re_.get("preparation_first_seen") or "", "attempt_at": re_.get("attempt_at") or ""}


def stage_transition(before, now):
    """Lifecycle movement is not a strength order: EARLY→CONTINUATION is progress, anything into LATE/FAILED is weakening,
    out of FAILED/REPAIRING into EARLY is a new event."""
    if before == now:
        return "UNCHANGED"
    if now in ("LATE", "FAILED"):
        return "WEAKENING"
    if before in ("FAILED", "REPAIRING") and now in ("EARLY", "CONTINUATION"):
        return "NEW"
    if before == "EARLY" and now == "CONTINUATION":
        return "PROGRESSING"
    if before in ("LATE", "REPAIRING") and now in ("EARLY", "CONTINUATION"):
        return "IMPROVING"
    if before == "CONTINUATION" and now == "EARLY":
        return "IMPROVING"
    return "UNCHANGED"


def change_of(current, previous):
    """Change from structured fields of the previous shadow row; without one, from the engine's evidence_change only
    when the sentence is unambiguous (a negated '改善' is not an improvement)."""
    if previous and previous.get("stage_bucket"):
        moves = [stage_transition(previous["stage_bucket"], current["stage_bucket"])]
        if previous.get("position") != current["position"]:
            moves.append("WEAKENING" if current["position"] == "EXTENDED" else "IMPROVING" if current["position"] == "IN_BAND" else "UNCHANGED")
        if previous.get("trigger") != current["trigger"]:
            moves.append("IMPROVING" if current["trigger"] == "ABSORBED" else "WEAKENING" if current["trigger"] == "WEAK_ACTIVE" else "UNCHANGED")
        prev_live = (previous.get("preparation") or {}).get("live")
        if prev_live is not None and prev_live != current["preparation"]["live"]:
            moves.append("IMPROVING" if current["preparation"]["live"] else "WEAKENING")
        if "NEW" in moves:
            return "NEW"
        if "WEAKENING" in moves:
            return "WEAKENING"
        if "IMPROVING" in moves:
            return "IMPROVING"
        if "PROGRESSING" in moves:
            return "PROGRESSING"
        return "UNCHANGED"
    text = str((current.get("_source") or {}).get("evidence_change") or "")
    if not text:
        return "UNKNOWN"
    if "新周期" in text or "首份" in text:
        return "NEW"
    negated = any(w in text for w in ("未改善", "无改善", "没有改善", "不改善"))
    weak = any(w in text for w in ("减弱", "消退", "退化", "恶化", "失效"))
    better = "改善" in text and not negated
    if better and not weak:
        return "IMPROVING"
    if weak and not better:
        return "WEAKENING"
    return "UNCHANGED" if ("延续" in text or "无新增" in text) else "UNKNOWN"


def re_eligible(record, as_of):
    """A continued-tracking name competes again only when its price and evidence carry the batch date and a structure
    is live. Re-entry to observation is not a trading permission; the frozen blockers except TRACKING_ONLY stay."""
    if not record.get("tracking_only") or not as_of:
        return False
    f = record.get("features") or {}
    sv = (record.get("evidence") or {}).get("state_vector") or {}
    src = record.get("source_record") or {}
    fresh = f.get("status") == "READY" and record.get("price_date") == as_of and (record.get("evidence") or {}).get("as_of") == as_of
    live = sv.get("setup_status") in ("CONFIRMED", "FORMING") or src.get("route_status") in ("CONFIRMATION_REVIEW", "PREPARATION_WATCH")
    return bool(fresh and live and "STRUCTURE_RISK" not in record.get("blockers", []))


def state_without(record, dropped):
    blockers = [b for b in record.get("blockers", []) if b not in dropped]
    if "DATA_WAIT" in blockers or "BENCHMARK_MISSING" in blockers:
        return "DATA_WAIT", blockers
    if "STRUCTURE_RISK" in blockers:
        return "RISK_WATCH", blockers
    if "MARKET_WAIT" in blockers:
        return "MARKET_WAIT", blockers
    return ("WAIT_SETUP" if blockers else "PLAN_REVIEW"), blockers


def read(record, cohorts, market_themes, previous=None, as_of=None):
    f = record.get("features") or {}
    stage_bucket, stage = stage_of(record)
    position, bias = position_of(record)
    prep = preparation_of(record)
    cohort = cohorts.get(record.get("theme"), {"state": "UNKNOWN", "members": 0})
    market = market_themes.get(record.get("theme"), {"state": "UNKNOWN"})
    trigger = trigger_of(f)
    eligible_again = re_eligible(record, as_of)
    state_v1, blockers_v1 = state_without(record, {"TRACKING_ONLY"} if eligible_again else set())
    row = {"opportunity_id": record["opportunity_id"], "symbol": record["symbol"], "name": record.get("name"), "theme": record.get("theme"),
           "frozen_state": record.get("state"), "frozen_strength_rank": record.get("strength_rank"), "frozen_participation_rank": record.get("participation_rank"),
           "state": state_v1, "blockers": blockers_v1, "re_eligible": eligible_again, "tracking_only": bool(record.get("tracking_only")),
           "stage_bucket": stage_bucket, "stage": stage, "position": position, "bias_atr": round(bias, 3) if bias is not None else None,
           "route": route_of(record), "route_ids": record.get("route_ids", ""), "preparation": prep, "trigger": trigger,
           "close_location": number(f.get("close_location")), "relative_turnover": number(f.get("relative_turnover")),
           "trend": {"market_excess5": number(f.get("market_excess5")), "market_excess20": number(f.get("market_excess20")),
                     "peer_excess20": number(f.get("peer_excess20")), "peer_count": f.get("peer_count"), "return20": number(f.get("return20")),
                     "drawdown20": number(f.get("drawdown20"))},
           "mainline": cohort["state"], "theme_cohort": cohort, "market_theme": market.get("state", "UNKNOWN"), "market_theme_context": market,
           "previous_shadow_rank": (previous or {}).get("shadow_rank"), "alpha_scope": record.get("alpha_scope"),
           "_source": record.get("source_record") or {}}
    row["change"] = change_of(row, previous)
    row.pop("_source")
    return row


def mainline_key(row):
    """Market breadth decides when a same-day table exists for the theme; otherwise the cohort reading."""
    if row.get("market_theme") in ("STRONG", "WEAK"):
        return MAINLINE_ORDER[row["market_theme"]]
    return MAINLINE_ORDER.get(row["mainline"], 1)


def keys(row):
    """Lexicographic keys for each ablation step; lower sorts first. No weights, no blended score. The order is a
    policy: state gate → position (extension is never outranked) → stage → theme → route → preparation → change →
    trigger → trend as the tie-break."""
    trend = -(row["trend"]["market_excess20"] if row["trend"]["market_excess20"] is not None else -999)
    frozen = (row["frozen_participation_rank"] if row["frozen_participation_rank"] is not None else 10 ** 6)
    base = (STATE_ORDER.get(row["state"], 9), row["alpha_scope"] == "R3_RESEARCH")
    position = (POSITION_ORDER[row["position"]],)
    trigger = (TRIGGER_ORDER[row["trigger"]],)
    stage_route = (STAGE_ORDER[row["stage_bucket"]], mainline_key(row), ROUTE_ORDER[row["route"]], 0 if row["preparation"]["live"] else 1)
    return {"V0_frozen": (frozen, row["symbol"]),
            "V1_tracking": (*base, frozen, row["symbol"]),
            "V2_position": (*base, *position, *trigger, trend, row["symbol"]),
            "V3_stage_route": (*base, *position, *stage_route, *trigger, trend, row["symbol"]),
            "V4_multiday": (*base, *position, *stage_route, CHANGE_ORDER[row["change"]], *trigger, trend, row["symbol"])}


def select_review_shortlist(rows, limit=2):
    """Names worth the next unit of review attention: at most `limit`, one per security, distinct themes,
    never extended, late or failed, never R3-only. It is not an allocation: it checks no holdings, orders,
    cash, ETF seat or account, and it grants nothing (see trading.allocation.allocate_ready_plans)."""
    chosen, seen = [], set()
    for row in rows:
        if len(chosen) >= limit:
            break
        if row["state"] not in ("PLAN_REVIEW", "WAIT_SETUP") or row["symbol"] in seen:
            continue
        if row["position"] != "IN_BAND" or row["stage_bucket"] in ("LATE", "FAILED", "REPAIRING") or row["alpha_scope"] == "R3_RESEARCH":
            continue
        if any(c["theme"] == row["theme"] and row["theme"] not in ("", "未归属") for c in chosen):
            continue
        chosen.append(row)
        seen.add(row["symbol"])
    return chosen


def watch_reconciliation(rows, preparation_watch):
    """Every count before any truncation: candidates, watch, overlap, outside, union."""
    known = {r["symbol"] for r in rows}
    watch_symbols = [w.get("symbol") for w in (preparation_watch or []) if w.get("symbol")]
    outside = [w for w in (preparation_watch or []) if w.get("symbol") and w["symbol"] not in known]
    return {"candidates": len(known), "watch": len(set(watch_symbols)), "overlap": len(known & set(watch_symbols)),
            "outside": len({w["symbol"] for w in outside}), "union": len(known | set(watch_symbols)),
            "note": "低偏离准备观察（影子）：域外名单进入待取证机会层，不进候选状态，无冻结触发价与失效价"}, outside


def evaluate(payload, previous=None, preparation_watch=None, theme_facts=None):
    as_of = payload.get("as_of")
    records = payload.get("records", [])
    cohorts = theme_cohorts(records)
    market_themes = market_theme_context(theme_facts)
    before = {r["opportunity_id"]: r for r in (previous or {}).get("rows", [])} if previous and previous.get("as_of", "") < (as_of or "") else {}
    rows = [read(r, cohorts, market_themes, before.get(r["opportunity_id"]), as_of) for r in records]
    for row in rows:
        row["keys"] = keys(row)
    for variant in VARIANTS:
        for n, row in enumerate(sorted(rows, key=lambda r: r["keys"][variant]), 1):
            row.setdefault("variant_ranks", {})[variant] = n
    for row in rows:
        row["shadow_rank"] = row["variant_ranks"]["V4_multiday"]
        theme_text = LABELS["mainline"][row["mainline"]] + ("／" + LABELS["market_theme"][row["market_theme"]] if row["market_theme"] != "UNKNOWN" else "")
        row["reading"] = "；".join([LABELS["stage"][row["stage_bucket"]] + (f"（{row['stage']}）" if row["stage"] else ""), theme_text,
                                   LABELS["route"][row["route"]], LABELS["position"][row["position"]] + (f"（{row['bias_atr']:+.1f}ATR）" if row["bias_atr"] is not None else ""),
                                   LABELS["trigger"][row["trigger"]], LABELS["change"][row["change"]]])
        row.pop("keys")
    rows.sort(key=lambda r: r["shadow_rank"])
    shortlist = select_review_shortlist(rows)
    top_frozen = {r["symbol"] for r in rows if r["frozen_participation_rank"] is not None and r["frozen_participation_rank"] <= 10}
    top_shadow = {r["symbol"] for r in rows[:10]}
    reconciliation, outside = watch_reconciliation(rows, preparation_watch)
    summary = {"total": len(rows), "re_eligible": sum(r["re_eligible"] for r in rows),
               "states": dict(Counter(r["state"] for r in rows)), "frozen_states": dict(Counter(r["frozen_state"] for r in rows)),
               "stage_buckets": dict(Counter(r["stage_bucket"] for r in rows)), "positions": dict(Counter(r["position"] for r in rows)),
               "cohort_states": dict(Counter(r["mainline"] for r in rows)), "market_theme_states": dict(Counter(r["market_theme"] for r in rows)),
               "cohort_strong": sorted(t for t, c in cohorts.items() if c["state"] == "STRONG"),
               "cohort_weak": sorted(t for t, c in cohorts.items() if c["state"] == "WEAK"),
               "market_themes_strong": sorted(t for t, c in market_themes.items() if c.get("state") == "STRONG"),
               "top10_overlap_with_frozen": len(top_frozen & top_shadow),
               "review_shortlist": [{k: r[k] for k in ("symbol", "name", "theme", "state", "reading")} for r in shortlist],
               "top10": [{k: r[k] for k in ("symbol", "name", "theme", "state", "frozen_participation_rank", "reading")} for r in rows[:10]],
               "preparation_watch": reconciliation}
    return {"version": VERSION, "as_of": as_of, "snapshot_id": payload.get("snapshot_id"), "thresholds": THRESHOLDS,
            "preparation_watch_outside": outside[:20], "preparation_watch_outside_total": reconciliation["outside"],
            "order": "state → position(不追高，作为本政策的偏好写明) → stage → theme(市场表优先，否则候选群体) → route → preparation → change → trigger → trend(tie)；无加权总分",
            "basis": "SHADOW_ONLY_NOT_CALIBRATED_NO_ORDERS", "cohorts": cohorts, "market_themes": market_themes,
            "summary": summary, "rows": rows}

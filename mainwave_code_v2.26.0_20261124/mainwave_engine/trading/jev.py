"""Jev (TypeSafe System One) as a shadow semantic-judgment layer.

Computed evidence goes in as a compact state; typed, calibrated probabilities come out. Nothing here
reads a price series, changes a rank, opens a gate or places an order. A probability is Jev's
confidence that the state matches a definition we wrote; it is never a probability that a stock rises.
The questions are code: versioned, pinned to the model version, replayable against the same state.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from threading import Lock

from . import store

VERSION = "jev_shadow_v1_1_20261027"
QUESTION_SET_VERSION = "jev_questions_v1_20260921"
MODEL = "jev-latest"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
TIMEOUT = (5, 20)
PRICE_PER_MILLION_INPUT_USD = 0.042
OFFLINE = "JEV_UNAVAILABLE"
# Shadow thresholds: computed and reported, never acted on in this version.
VETO_THRESHOLD = .85
LOW_CONFIDENCE = .30
NOTICE = ("影子运行：不改排名、不改闸门、不下单。概率是 Jev 对'状态符合某个定义'的信心，不是上涨概率；"
          "Jev 校准与交易校准是两层，不能混。")

# ---------------------------------------------------------------- questions (part of the program: version them)
STATE_NOTES = ("All facts were computed by code from daily bars and the discovery engine. Distances are in ATR "
               "multiples, returns in percent, percentiles within the day's candidate cohort. 'market_context' BEAR/BULL "
               "is the index regime gate. 'retained' means the earlier impulse's gains were held; 'demand'/'supply' are "
               "volume-effort readings; 'preparation_valid' means a pre-ignition base was found before the attempt.")

GLOBAL_QUESTIONS = {
    "Q01_setup_quality": {"type": "score", "instructions": (
        "Rate the overall quality of this trend-pullback setup as a candidate for a first entry, judging only the "
        "evidence given. Ignore how strong the recent return is; judge structure, retention, supply/demand and risk flags."),
        "criteria": ["LOW", "MEDIUM", "HIGH", "EXCEPTIONAL"]},
    "Q02_entry_stage": {"type": "choice", "instructions": (
        "Which stage best describes where an entry now would sit in this move?"),
        "criteria": {"EARLY": "confirmation just completed; price still near the boundary; no visible expansion yet",
                     "OPTIMAL": "trend confirmed and retained; price within a reasonable ATR distance of structure; risk/reward still sound",
                     "LATE": "trend still strong but price clearly extended (several ATR above MA20/structure); odds have fallen",
                     "EXHAUSTED": "evidence of maturity or exhaustion: overheated stage, distribution risk, failed retention or weak effort"}},
    "Q03_evidence_consistent": {"type": "noul", "instructions": (
        "Taken together, do these facts form a coherent early-stage continuation setup rather than a mix of "
        "supporting and contradicting signals?")},
    "Q04_adverse_evidence_material": {"type": "noul", "instructions": (
        "Is there material evidence against entering now (overextension, risk overlays, failed setup, missing "
        "confirmation, market gate closed), even if no single item is decisive?")},
    "Q05_requires_deep_review": {"type": "noul", "instructions": (
        "Does this state need a slower, reasoned review by a person or a reasoning model before any decision, "
        "because the evidence is ambiguous or conflicting?")},
}
R1_QUESTIONS = {
    "Q11_breakout_quality": {"type": "score", "instructions": (
        "Judge the quality of the breakout that this R1 candidate is continuing from: acceptance, retention, "
        "effort behind it."), "criteria": ["WEAK", "NORMAL", "STRONG"]},
    "Q12_continuation_readiness": {"type": "score", "instructions": (
        "How ready is this candidate to continue after the breakout: pullback depth, contraction, reclaim, "
        "distance from structure."), "criteria": ["LOW", "MEDIUM", "HIGH"]},
    "Q13_false_breakout_risk": {"type": "noul", "instructions": (
        "Is there a meaningful risk that the breakout fails (not retained, no demand, extended without base)?")},
}
R2_QUESTIONS = {
    "Q21_accumulation_quality": {"type": "score", "instructions": (
        "Judge the quality of the pre-ignition preparation for this R2 candidate: a valid base found before the "
        "attempt, its age and origin."), "criteria": ["LOW", "MEDIUM", "HIGH"]},
    "Q22_supply_absorption_quality": {"type": "score", "instructions": (
        "Judge how well supply has been absorbed: activity contraction, absence of contraction disagreement, "
        "retention of gains."), "criteria": ["LOW", "MEDIUM", "HIGH"]},
    "Q23_reacceleration_readiness": {"type": "score", "instructions": (
        "How ready is the candidate to re-accelerate out of the base given demand, contraction and position."),
        "criteria": ["LOW", "MEDIUM", "HIGH"]},
}


def question_set(route_ids):
    """Global questions for every candidate; route questions only for the routes the engine declared."""
    routes = {r for r in str(route_ids or "").split("|") if r}
    questions = dict(GLOBAL_QUESTIONS)
    if "R1" in routes:
        questions.update(R1_QUESTIONS)
    if "R2" in routes:
        questions.update(R2_QUESTIONS)
    return questions


def validate_questions(questions):
    """A question is a typed contract; a malformed one is refused before any call is made."""
    problems = []
    for name, q in questions.items():
        kind = q.get("type")
        if kind not in {"choice", "score", "noul"}:
            problems.append(f"{name}: type 非法")
        if not q.get("instructions"):
            problems.append(f"{name}: 缺 instructions")
        if kind == "choice" and (not isinstance(q.get("criteria"), dict) or len(q["criteria"]) < 2):
            problems.append(f"{name}: choice 需要 ≥2 个带说明的选项")
        if kind == "score" and (not isinstance(q.get("criteria"), list) or len(q["criteria"]) < 2):
            problems.append(f"{name}: score 的 criteria 必须是有序列表")
    return problems


# ---------------------------------------------------------------- state (only what code already computed)
def _num(value, digits=3):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if v == v else None


def state_of(record, snapshot):
    """Compact JSON of computed evidence. No identifiers, no prose, no raw bars: only what the decision needs."""
    f = record.get("features") or {}
    src = record.get("source_record") or {}
    ev = record.get("evidence") or {}
    route = ev.get("route_evidence") or {}
    vector = ev.get("state_vector") or {}
    breakout = ev.get("breakout") or {}
    checks = {c.get("id"): c.get("status") for c in ev.get("checks") or [] if isinstance(c, dict)}
    as_of = snapshot.get("as_of", "")

    def days_between(earlier, later):
        try:
            from datetime import date
            a, b = date.fromisoformat(str(earlier)[:10]), date.fromisoformat(str(later)[:10])
            return (b - a).days
        except ValueError:
            return None
    state = {
        "_notes": STATE_NOTES,
        "market_context": (snapshot.get("market") or {}).get("context") or ev.get("market", {}).get("context"),
        "market_gate_open": bool((snapshot.get("market") or {}).get("trend_review_allowed")),
        "routes": record.get("route_ids", ""),
        "engine_state": record.get("state"),
        "blockers": list(record.get("blockers") or []),
        "lifecycle_phase": vector.get("phase"), "lifecycle_stage": vector.get("stage"),
        "local_role": vector.get("local_role") or src.get("local_role"),
        "risk_overlays": [o.get("code") for o in vector.get("risk_overlays") or [] if isinstance(o, dict)],
        "setup_status": vector.get("setup_status") or src.get("setup_status"), "setup_kind": record.get("setup_kind") or "",
        "structural_advice": src.get("structural_advice"), "price_volume_label": src.get("price_volume_label"),
        "breakout_status": src.get("breakout_status") or breakout.get("breakout_event"),
        "days_since_breakout_accepted": days_between(breakout.get("breakout_accepted_at") or route.get("accepted_at"), as_of)
        if (breakout.get("breakout_accepted_at") or route.get("accepted_at")) else None,
        "retained": route.get("retained"), "demand": route.get("demand"), "supply": route.get("supply"),
        "activity_contraction": route.get("activity_contraction"),
        "contraction_disagreement": route.get("contraction_disagreement"),
        "confirmation_pass": route.get("confirmation_pass"),
        "preparation_valid": route.get("preparation_valid"), "preparation_state": route.get("preparation_state"),
        "preparation_age_sessions": _num(route.get("preparation_age"), 1),
        "prepared_before_attempt": route.get("prepared_before_attempt"),
        "route_risks": list(route.get("risks") or []), "route_status": route.get("status"),
        "distance_ma20_atr": _num(route.get("bias_atr") if route.get("bias_atr") is not None else src.get("route_bias_atr"), 2),
        "ema20_bias_pct": _num(src.get("ema20_bias_pct"), 2),
        "return5_pct": _num(f.get("return5"), 2), "return20_pct": _num(f.get("return20"), 2),
        "market_excess5_pct": _num(f.get("market_excess5"), 2), "market_excess20_pct": _num(f.get("market_excess20"), 2),
        "peer_excess20_pct": _num(f.get("peer_excess20"), 2),
        "drawdown20_pct": _num(f.get("drawdown20"), 2), "efficiency10": _num(f.get("efficiency10"), 3),
        "close_location": _num(f.get("close_location"), 3), "relative_turnover": _num(f.get("relative_turnover"), 2),
        "turnover_pct": _num(f.get("turnover"), 2),
        "strength_percentiles": {k: _num(v, 3) for k, v in (record.get("score_percentiles") or {}).items()},
        "checks": {k: v for k, v in checks.items() if k in ("setup", "breakout", "confirmation", "boundary", "price", "risk", "route_context")},
        "window_active": bool((record.get("window") or {}).get("active")),
        "confirmation_age_sessions": (record.get("window") or {}).get("confirmation_age_sessions"),
    }
    return {k: v for k, v in state.items() if v is not None}


# ---------------------------------------------------------------- client
class JevError(RuntimeError):
    pass


class Client:
    def __init__(self, key=None, endpoint=ENDPOINT, timeout=TIMEOUT, proxy=None):
        from dotenv import load_dotenv
        load_dotenv()
        self.key = key or os.environ.get("JEV_API_KEY", "")
        if not self.key:
            raise JevError("JEV_API_KEY 未配置")
        self.endpoint, self.timeout = endpoint, timeout
        self.proxy = proxy if proxy is not None else os.environ.get("JEV_PROXY") or os.environ.get("FUPAN_PROXY") or os.environ.get("HTTPS_PROXY") or ""
        self._lock, self._blocked = Lock(), ""

    def ask(self, state, questions, model=MODEL):
        import requests
        with self._lock:
            if self._blocked:
                raise JevError(self._blocked)
        started = time.perf_counter()
        try:
            response = requests.post(self.endpoint, json={"model": model, "state": state, "questions": questions},
                                     headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout,
                                     proxies={"http": self.proxy, "https": self.proxy} if self.proxy else None)
        except requests.RequestException as exc:
            raise JevError(f"网络错误：{type(exc).__name__}（请检查 JEV_PROXY 与服务连接）") from exc
        if response.status_code != 200:
            message = f"HTTP {response.status_code}: " + response.text[:300].replace(self.key, "[REDACTED]")
            if response.status_code in (401, 403, 451):
                with self._lock:
                    self._blocked = f"HTTP {response.status_code}: 服务鉴权或可用性失败，本轮后续请求已熔断"
            raise JevError(message)
        payload = response.json()
        payload["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return payload


def normalize(answers, questions):
    """One shape per question: label, confidence, probabilities keyed by label. Score keeps its expected index."""
    out = {}
    for name, answer in (answers or {}).items():
        kind = answer.get("type") or questions.get(name, {}).get("type")
        if kind == "choice":
            probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
            out[name] = {"type": kind, "label": answer.get("choice"), "confidence": _num(answer.get("confidence")), "probabilities": probs}
        elif kind == "score":
            legend = answer.get("legend") or {}
            labels = {k: (v.get("label") if isinstance(v, dict) else v) for k, v in legend.items()}
            probs = {labels.get(k, k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
            label = max(probs, key=probs.get) if probs else None
            out[name] = {"type": kind, "label": label, "score": _num(answer.get("score")), "confidence": _num(answer.get("confidence")),
                         "probabilities": probs}
        elif kind == "noul":
            p = float(answer.get("noul", 0))
            out[name] = {"type": kind, "label": "YES" if p >= .5 else "NO", "probability": round(p, 4),
                         "confidence": round(abs(2 * p - 1), 4), "probabilities": {"YES": round(p, 4), "NO": round(1 - p, 4)}}
    return out


def shadow_flags(answers):
    """What a later phase might do with these answers. Reported only; nothing downstream reads them yet."""
    stage = answers.get("Q02_entry_stage", {})
    adverse = answers.get("Q04_adverse_evidence_material", {}).get("probability", 0)
    review = answers.get("Q05_requires_deep_review", {}).get("probability", 0)
    late = (stage.get("probabilities") or {}).get("LATE", 0) + (stage.get("probabilities") or {}).get("EXHAUSTED", 0)
    return {"would_review": adverse >= VETO_THRESHOLD or review >= VETO_THRESHOLD,
            "would_wait_late": late >= VETO_THRESHOLD,
            "low_confidence": any((a.get("confidence") or 0) < LOW_CONFIDENCE for a in answers.values() if a.get("type") != "noul"),
            "setup_tier": answers.get("Q01_setup_quality", {}).get("label"),
            "entry_stage": stage.get("label"),
            "note": "影子标记，不触发任何状态变化；阈值待前向样本校准"}


# ---------------------------------------------------------------- storage
def _table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS jev_assessments (id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, "
                 "opportunity_id TEXT NOT NULL, symbol TEXT NOT NULL, as_of TEXT NOT NULL, state_hash TEXT NOT NULL, "
                 "question_set TEXT NOT NULL, model TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)")


def assess(record, snapshot, client, db=store.DB_PATH, at=None, model=MODEL):
    """One candidate: cached by (state hash, question set, model); a failure is a named row, not a guess."""
    state = state_of(record, snapshot)
    questions = question_set(record.get("route_ids"))
    state_hash = store.digest(state)
    identity = store.digest([state_hash, QUESTION_SET_VERSION, model])
    with store.connect(db) as conn:
        _table(conn)
        row = conn.execute("SELECT payload FROM jev_assessments WHERE id=?", (identity,)).fetchone()
    if row:
        return {**json.loads(row[0]), "cached": True}
    base = {"version": VERSION, "id": identity, "snapshot_id": snapshot.get("snapshot_id", ""),
            "opportunity_id": record["opportunity_id"], "symbol": record["symbol"], "as_of": snapshot.get("as_of"),
            "basis": snapshot.get("observation_basis"), "route_ids": record.get("route_ids", ""),
            "alpha_scope": record.get("alpha_scope"), "strength_rank": record.get("strength_rank"),
            "participation_rank": record.get("participation_rank"), "engine_state": record.get("state"),
            "state": state, "state_hash": state_hash, "question_set": QUESTION_SET_VERSION,
            "questions": sorted(questions), "model_requested": model, "created_at": at or store.now(), "cached": False}
    problems = validate_questions(questions)
    if problems:
        return {**base, "status": "INVALID_QUESTIONS", "problems": problems, "answers": {}}
    try:
        raw = client.ask(state, questions, model)
    except JevError as exc:
        return {**base, "status": "ERROR", "error": str(exc)[:300], "answers": {}}
    answers = normalize(raw.get("answers"), questions)
    result = {**base, "status": "RECORDED", "model": raw.get("model"), "answers": answers,
              "flags": shadow_flags(answers), "usage": raw.get("usage"), "latency_ms": raw.get("latency_ms"),
              "raw": raw.get("answers")}
    with store.connect(db) as conn:
        _table(conn)
        conn.execute("INSERT OR IGNORE INTO jev_assessments VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (identity, base["snapshot_id"], record["opportunity_id"], record["symbol"], base["as_of"] or "",
                      state_hash, QUESTION_SET_VERSION, raw.get("model", model), base["created_at"], store.encoded(result)))
    return result


def selection(snapshot, limit=30):
    """Who gets judged: the strength front, the participation front and every review-eligible name; R3-only stays research."""
    chosen, skipped = {}, {"R3_RESEARCH": 0}
    for r in snapshot.get("records", []):
        wanted = ((r.get("strength_rank") or 10 ** 6) <= limit or (r.get("participation_rank") or 10 ** 6) <= 15
                  or r.get("paper_plan_eligible"))
        if not wanted:
            continue
        if r.get("alpha_scope") == "R3_RESEARCH":
            skipped["R3_RESEARCH"] += 1
            continue
        chosen[r["opportunity_id"]] = r
    return list(chosen.values()), skipped


def run_shadow(snapshot, db=store.DB_PATH, client=None, limit=30, at=None, workers=6):
    """Judge the selected candidates beside the ranking. Unavailable means unavailable, never a made-up row."""
    records, skipped = selection(snapshot, limit)
    base = {"version": VERSION, "as_of": snapshot.get("as_of"), "snapshot_id": snapshot.get("snapshot_id", ""),
            "basis": snapshot.get("observation_basis"), "question_set": QUESTION_SET_VERSION, "model_requested": MODEL,
            "selected": len(records), "skipped": skipped, "notice": NOTICE}
    if client is None:
        try:
            client = Client()
        except JevError as exc:
            return {**base, "status": "UNAVAILABLE", "reason": str(exc), "rows": [], "summary": {}}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda r: assess(r, snapshot, client, db, at), records))
    rows.sort(key=lambda r: (r.get("strength_rank") or 10 ** 6, r["symbol"]))
    recorded = [r for r in rows if r.get("status") == "RECORDED"]
    errors = [r for r in rows if r.get("status") != "RECORDED"]
    tokens = sum((r.get("usage") or {}).get("input_tokens", 0) for r in recorded if not r.get("cached"))
    latencies = sorted(r.get("latency_ms") for r in recorded if r.get("latency_ms") is not None)
    from collections import Counter
    summary = {"recorded": len(recorded), "cached": sum(1 for r in recorded if r.get("cached")), "errors": len(errors),
               "model": next((r.get("model") for r in recorded), None),
               "setup_quality": dict(Counter(r["flags"]["setup_tier"] for r in recorded)),
               "entry_stage": dict(Counter(r["flags"]["entry_stage"] for r in recorded)),
               "would_review": sum(r["flags"]["would_review"] for r in recorded),
               "would_wait_late": sum(r["flags"]["would_wait_late"] for r in recorded),
               "low_confidence": sum(r["flags"]["low_confidence"] for r in recorded),
               "input_tokens": tokens, "cost_usd": round(tokens / 1e6 * PRICE_PER_MILLION_INPUT_USD, 5),
               "latency_ms_median": latencies[len(latencies) // 2] if latencies else None,
               "latency_ms_max": latencies[-1] if latencies else None}
    status = "RECORDED" if recorded and not errors else "PARTIAL" if recorded else "UNAVAILABLE"
    return {**base, "status": status, "rows": rows, "summary": summary,
            "errors": [{"symbol": r["symbol"], "error": r.get("error") or r.get("status")} for r in errors][:20]}


# ---------------------------------------------------------------- validation: does the judgment layer the outcomes?
ORDERED = {"Q01_setup_quality": ["LOW", "MEDIUM", "HIGH", "EXCEPTIONAL"],
           "Q02_entry_stage": ["EARLY", "OPTIMAL", "LATE", "EXHAUSTED"],
           "Q12_continuation_readiness": ["LOW", "MEDIUM", "HIGH"], "Q23_reacceleration_readiness": ["LOW", "MEDIUM", "HIGH"]}
NOULS = ("Q03_evidence_consistent", "Q04_adverse_evidence_material", "Q13_false_breakout_risk")
MIN_BUCKET = 20


def evaluate(day, db=store.DB_PATH, data_root=None):
    """Join recorded judgments with the frozen price-observation rows; report layering, never a score change."""
    from . import validation
    review = validation.evaluate(day, db, data_root)
    with store.connect(db) as conn:
        _table(conn)
        judged = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM jev_assessments WHERE as_of<? ORDER BY as_of", (day,))]
    by_key = {}
    for j in judged:
        by_key.setdefault((j["opportunity_id"], j["as_of"]), j)  # first judgment of a day stays the frozen one
    rows = []
    for r in review.get("records", []):
        j = by_key.get((r["opportunity_id"], r["signal_day"]))
        if not j or j.get("status") != "RECORDED":
            continue
        rows.append({**r, "jev": j["answers"], "flags": j.get("flags", {})})
    buckets = {}
    for question, order in ORDERED.items():
        buckets[question] = {}
        for horizon in review.get("horizons", []):
            per = {}
            for label in order:
                chosen = [r for r in rows if r["horizon"] == horizon and r["jev"].get(question, {}).get("label") == label]
                per[label] = validation._cohort(f"{question}={label}", chosen, chosen[0] if chosen else {"signal_day": "", "version": "", "basis": ""}, horizon)
            means = [per[l]["mean"] if per[l].get("mature_n", 0) >= MIN_BUCKET else None for l in order]
            usable = [m for m in means if m is not None]
            per["_layering"] = ("INSUFFICIENT_SAMPLE" if len(usable) < 2 else
                                "MONOTONIC" if usable == sorted(usable, reverse=(question == "Q02_entry_stage")) else "NOT_MONOTONIC")
            buckets[question][str(horizon)] = per
    for question in NOULS:
        buckets[question] = {}
        for horizon in review.get("horizons", []):
            per = {}
            for label, test in (("YES_HIGH", lambda p: p >= VETO_THRESHOLD), ("MID", lambda p: .5 <= p < VETO_THRESHOLD), ("NO", lambda p: p < .5)):
                chosen = [r for r in rows if r["horizon"] == horizon and question in r["jev"] and test(r["jev"][question]["probability"])]
                per[label] = validation._cohort(f"{question}={label}", chosen, chosen[0] if chosen else {"signal_day": "", "version": "", "basis": ""}, horizon)
            buckets[question][str(horizon)] = per
    mature = sum(1 for r in rows if r["status"] == "MATURE")
    return {"version": VERSION, "as_of": day, "question_set": QUESTION_SET_VERSION,
            "judged_rows": len(rows), "judged_opportunities": len({r["opportunity_id"] for r in rows}), "mature_rows": mature,
            "clusters": validation._clusters(rows) if rows else {"signal_days": 0, "theme_days": 0},
            "status": "READY" if mature >= MIN_BUCKET else "IMMATURE", "by_question": buckets,
            "reading": ("分层要求同一问题各标签的成熟样本 ≥ %d 且均值单调（Q02 应为 EARLY/OPTIMAL 高于 LATE/EXHAUSTED）；"
                        "达不到就是'尚未证明'，不是'无效'" % MIN_BUCKET),
            "notice": "价格观察不是成交收益；Jev 概率不是上涨概率；证明分层后才谈否决权，再之后才谈重排。"}


# ---------------------------------------------------------------- report
def render(shadow, review=None):
    s = shadow.get("summary") or {}
    lines = [f"# Jev 影子判断 {shadow.get('as_of')}（{shadow.get('status')}）", "", f"> {NOTICE}", "",
             f"- 模型：请求 {shadow.get('model_requested')} → 实际 {s.get('model') or '--'}；问题集 {shadow.get('question_set')}；"
             f"选中 {shadow.get('selected')}，跳过 {shadow.get('skipped')}",
             f"- 记录 {s.get('recorded', 0)}（缓存 {s.get('cached', 0)}），错误 {s.get('errors', 0)}；输入 token {s.get('input_tokens', 0)}，"
             f"估算成本 ${s.get('cost_usd', 0)}；延迟中位 {s.get('latency_ms_median')} ms / 最大 {s.get('latency_ms_max')} ms",
             f"- 分布：setup_quality {s.get('setup_quality')}；entry_stage {s.get('entry_stage')}；"
             f"影子标记 would_review {s.get('would_review', 0)}，would_wait_late {s.get('would_wait_late', 0)}，低置信 {s.get('low_confidence', 0)}", ""]
    if shadow.get("reason"):
        lines.append(f"- 不可用原因：{shadow['reason']}")
    lines += ["| 强势# | 股票 | 路径 | 引擎状态 | 质量 | 阶段(p) | 自洽 | 反证 | 需深审 | 路径问题 | 影子标记 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in shadow.get("rows", []):
        a = r.get("answers") or {}
        stage = a.get("Q02_entry_stage", {})
        route_q = "；".join(f"{k[:3]}={v.get('label')}" for k, v in a.items() if k.startswith(("Q1", "Q2")))
        flags = r.get("flags") or {}
        marks = "、".join(k for k in ("would_review", "would_wait_late", "low_confidence") if flags.get(k)) or "-"
        lines.append(f"| {r.get('strength_rank')} | {r['symbol']} | {r.get('route_ids')} | {r.get('engine_state')} | "
                     f"{a.get('Q01_setup_quality', {}).get('label', r.get('status'))} | {stage.get('label')}({(stage.get('probabilities') or {}).get(stage.get('label'), '')}) | "
                     f"{a.get('Q03_evidence_consistent', {}).get('probability', '')} | {a.get('Q04_adverse_evidence_material', {}).get('probability', '')} | "
                     f"{a.get('Q05_requires_deep_review', {}).get('probability', '')} | {route_q} | {marks} |")
    if review:
        lines += ["", f"## 前向分层（{review.get('status')}）", "",
                  f"- 已判断且有价格观察的行 {review.get('judged_rows')}，机会 {review.get('judged_opportunities')}，成熟 {review.get('mature_rows')}，簇 {review.get('clusters')}",
                  f"- {review.get('reading')}"]
        for question, per_h in (review.get("by_question") or {}).items():
            for horizon, per in per_h.items():
                cells = "，".join(f"{l}: n={c.get('selected_n', 0)}/成熟{c.get('mature_n', 0)}/均值{c.get('mean')}" for l, c in per.items() if not l.startswith("_"))
                lines.append(f"  - {question} @{horizon}日：{per.get('_layering', '')}；{cells}")
    return "\n".join(lines) + "\n"
